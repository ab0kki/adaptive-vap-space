"""Robust event-extraction runner used by scripts/30_extract_events.py.

The event-definition logic lives in ``events.py``. This module owns the manifest
and path-handling layer around those definitions.

Why this exists:
    VAP manifests can come from different runs/notebooks. Some manifests use
    ``status=done`` or ``status=exists``; older attached Kaggle outputs may use a
    different status label even though the prediction file is valid. Extraction
    should be driven by whether the prediction file can be resolved and loaded,
    not by a tiny whitelist of status strings.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import get
from .events import FRAME_HZ, collect_rows_for_interaction
from .paths import resolve_under
from .reports import write_csv, write_json
from .vad import read_vad_segments, vad_segments_to_frames
from .vap_outputs import load_prediction_arrays
from .vap_scores import describe_score_subsets


EVENT_COLUMNS = [
    "interaction_key",
    "project_split",
    "task",
    "task_family",
    "label",
    "score",
    "frame",
    "time_s",
    "speaker",
    "score_speaker",
    "source_event",
    "event_id",
    "positive_class",
    "score_variant",
    "paper_comparable",
]

SUMMARY_COLUMNS = [
    "interaction_key",
    "project_split",
    "duration_s",
    "shift_hold_events_total",
    "shift_events",
    "hold_events",
    "overlap_shift_events",
    "bc_events",
    "s_pred_clear_pos_events",
    "s_pred_clear_neg_events",
    "s_pred_overlap_pos_events",
    "s_pred_overlap_neg_events",
    "bc_pred_pos_events",
    "bc_pred_neg_events",
    "prediction_rows_total",
]

ATTRITION_COLUMNS = [
    "interaction_key",
    "project_split",
    "duration_s",
    "raw_single_speaker_alternations",
    "gap_shift_triads",
    "gap_shift_min_silence",
    "gap_shift_pre_ok",
    "gap_shift_post_ok",
    "gap_shift_pre_post_ok",
    "gap_shift_metric_context_ok",
    "rapid_gap_shift_triads",
    "overlap_shift_triads",
    "overlap_shift_short_enough",
    "overlap_shift_pre_post_ok",
    "overlap_shift_context_ok",
    "hold_triads",
    "final_shift_events",
    "final_hold_events",
    "final_overlap_shift_events",
    "final_bc_events",
]

SKIPPED_COLUMNS = ["interaction_key", "project_split", "status", "reason", "detail"]


def _clean_string(value: object) -> str:
    """Return a safe string without treating pandas NaN as a useful path."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def resolve_prediction_path(row: pd.Series, output_root: Path) -> Path:
    """Resolve one VAP prediction path from a manifest row.

    Resolution order is intentionally tolerant:

    1. ``prediction_relpath`` under the configured VAP output root.
    2. basename of ``prediction_relpath`` under ``output_root/predictions``.
    3. existing absolute/local ``prediction_path``.
    4. basename of ``prediction_path`` under ``output_root/predictions``.

    This mirrors how Kaggle audit notebooks often attach prior output datasets:
    absolute paths from the old run may be stale, but the prediction file names
    under the attached ``predictions/`` directory are still valid.
    """
    candidates: list[Path] = []

    rel = _clean_string(row.get("prediction_relpath", ""))
    if rel:
        candidates.append(output_root / rel)
        candidates.append(output_root / "predictions" / Path(rel).name)

    raw = _clean_string(row.get("prediction_path", ""))
    if raw:
        raw_path = Path(raw)
        candidates.append(raw_path)
        candidates.append(output_root / "predictions" / raw_path.name)

    # Last-resort convention used by this repo's VAP runner.
    key = _clean_string(row.get("interaction_key", ""))
    if key:
        candidates.append(output_root / "predictions" / f"{key}.json.gz")
        candidates.append(output_root / "predictions" / f"{key}.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "; ".join(str(c) for c in candidates[:8])
    raise FileNotFoundError(f"No prediction file found. Tried: {tried}")


def _output_frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Return a DataFrame with stable columns even when rows are empty."""
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    extra = [c for c in df.columns if c not in columns]
    return df[columns + extra]


def extract_events(cfg: dict) -> None:
    """Read datastore/VAP manifests and write scored event prediction rows."""
    dataset_root = Path(get(cfg, "dataset.root", "data/datastore"))
    output_root = Path(get(cfg, "vap.output_root", "outputs/vap"))
    out_events = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv")).parent
    out_events.mkdir(parents=True, exist_ok=True)

    stereo_path = dataset_root / "manifests" / "stereo_manifest.csv"
    vap_manifest_path = output_root / "vap_manifest.csv"
    if not stereo_path.exists():
        raise FileNotFoundError(f"Missing stereo manifest: {stereo_path}")
    if not vap_manifest_path.exists():
        raise FileNotFoundError(f"Missing VAP manifest: {vap_manifest_path}")

    stereo = pd.read_csv(stereo_path)
    vap_manifest = pd.read_csv(vap_manifest_path)
    merged = stereo.merge(vap_manifest, on="interaction_key", suffixes=("", "_vap"))

    params = {
        "min_context_s": 3.0,
        "pre_offset_shift_s": 1.0,
        "post_onset_shift_s": 1.0,
        "pre_offset_hold_s": 1.0,
        "post_onset_hold_s": 1.0,
        "shift_hold_eval_offset_s": 0.05,
        "shift_hold_eval_len_s": 0.10,
        "min_silence_shift_hold_s": 0.15,
        "max_overlap_shift_s": 0.15,
        "pred_window_s": 0.50,
        "negative_far_s": 2.0,
        "bc_max_duration_s": 1.0,
        "bc_min_duration_s": 0.20,
        "bc_pre_silence_s": 1.0,
        "bc_post_silence_s": 2.0,
        "short_long_eval_len_s": 0.20,
        "negative_sample_seed": int(get(cfg, "project.seed", 17)),
    }

    all_rows: list[dict] = []
    summaries: list[dict] = []
    attrition_rows: list[dict] = []
    skipped: list[dict] = []

    for _, row in merged.iterrows():
        key = _clean_string(row.get("interaction_key", ""))
        project_split = _clean_string(row.get("project_split", row.get("project_split_vap", "")))
        status = _clean_string(row.get("status", ""))

        try:
            pred_path = resolve_prediction_path(row, output_root)
            probs, p_now, p_future = load_prediction_arrays(pred_path)

            vad_a, _ = read_vad_segments(resolve_under(dataset_root, row["vad_a_relpath"]))
            vad_b, _ = read_vad_segments(resolve_under(dataset_root, row["vad_b_relpath"]))
            va = vad_segments_to_frames(vad_a, vad_b, len(p_now), frame_hz=FRAME_HZ)

            rows, summary, attrition = collect_rows_for_interaction(
                interaction_key=key,
                project_split=project_split,
                va=va,
                probs=probs,
                p_now=p_now,
                p_future=p_future,
                params=params,
            )

            all_rows.extend(rows)
            summaries.append(summary)
            attrition_rows.append(attrition)
        except Exception as exc:
            skipped.append({
                "interaction_key": key,
                "project_split": project_split,
                "status": status,
                "reason": type(exc).__name__,
                "detail": str(exc),
            })

    write_csv(out_events / "event_predictions.csv", _output_frame(all_rows, EVENT_COLUMNS))
    write_csv(out_events / "event_summary.csv", _output_frame(summaries, SUMMARY_COLUMNS))
    write_csv(out_events / "event_attrition.csv", _output_frame(attrition_rows, ATTRITION_COLUMNS))
    write_json(out_events / "score_subsets.json", describe_score_subsets())
    write_csv(out_events / "skipped_interactions.csv", _output_frame(skipped, SKIPPED_COLUMNS))

    print(f"Merged interactions={len(merged)}")
    print(f"Processed interactions={len(summaries)}")
    print(f"Skipped interactions={len(skipped)}")
    print(f"Wrote {out_events / 'event_predictions.csv'} rows={len(all_rows)}")
    print(f"Wrote {out_events / 'event_summary.csv'} interactions={len(summaries)}")
    print(f"Wrote {out_events / 'event_attrition.csv'} interactions={len(attrition_rows)}")

    if skipped:
        print(f"Skipped details: {out_events / 'skipped_interactions.csv'}")
