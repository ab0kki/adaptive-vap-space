"""VAP-paper-style event extraction.

This module ports the working event logic from the prototype repo into reusable
functions. It is intentionally transparent about reconstructed choices.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .config import get
from .paths import resolve_under
from .vad import read_vad_segments, vad_segments_to_frames, frame_segments, no_activity, single_speaker_region
from .vap_outputs import load_prediction_arrays, backchannel_prob_from_probs
from .reports import write_csv

FRAME_HZ = 50
HORIZON_S = 2.0


def sec_to_frame(t: float, frame_hz: int = FRAME_HZ) -> int:
    """Convert seconds to nearest frame."""
    return int(round(float(t) * frame_hz))


def valid_future_frame(frame: int, duration_s: float, min_context_s: float = 3.0) -> bool:
    """Avoid evaluating frames before enough context or beyond VAP future horizon."""
    t = frame / FRAME_HZ
    return t >= min_context_s and (t + HORIZON_S) <= duration_s


def extract_shift_hold_events(va: np.ndarray, params: dict) -> list[dict]:
    """Extract SHIFT/HOLD events from frame-level VAD."""
    n_frames = len(va)
    all_segments = []
    for speaker in [0, 1]:
        for s, e in frame_segments(va[:, speaker]):
            all_segments.append((s, e, speaker))
    all_segments = sorted(all_segments, key=lambda x: (x[0], x[1]))

    pre = sec_to_frame(params.get("shift_pre_offset_s", 1.0))
    post = sec_to_frame(params.get("shift_post_onset_s", 1.0))
    eval_offset = sec_to_frame(params.get("shift_hold_eval_offset_s", 0.05))
    eval_len = sec_to_frame(params.get("shift_hold_eval_len_s", 0.10))
    events = []
    for i in range(len(all_segments) - 1):
        s1, e1, spk1 = all_segments[i]
        s2, e2, spk2 = all_segments[i + 1]
        if s2 <= e1:
            continue
        if not no_activity(va, e1, s2):
            continue
        if not single_speaker_region(va, max(0, e1 - pre), e1, spk1):
            continue
        if not single_speaker_region(va, s2, min(n_frames, s2 + post), spk2):
            continue
        eval_s = e1 + eval_offset
        eval_e = eval_s + eval_len
        if eval_e > s2 or eval_e > n_frames:
            continue
        label = int(spk1 != spk2)
        events.append({
            "task_source": "shift" if label else "hold",
            "label": label,
            "prev_speaker": spk1,
            "next_speaker": spk2,
            "silence_start": e1,
            "silence_end": s2,
            "onset": s2,
            "eval_start": eval_s,
            "eval_end": eval_e,
        })
    return events


def extract_backchannel_events(va: np.ndarray, params: dict) -> list[dict]:
    """Detect short isolated listener activity used for BC-pred and S/L."""
    n_frames = len(va)
    max_dur = sec_to_frame(params.get("bc_max_duration_s", 1.0))
    pre = sec_to_frame(params.get("bc_pre_silence_s", 1.0))
    post = sec_to_frame(params.get("bc_post_silence_s", 2.0))
    main_min = params.get("bc_main_min_active_ratio", 0.10)
    events = []
    for bc_speaker in [0, 1]:
        main = 1 - bc_speaker
        for s, e in frame_segments(va[:, bc_speaker]):
            dur = e - s
            if dur <= 0 or dur > max_dur:
                continue
            pre_s = max(0, s - pre)
            post_e = min(n_frames, e + post)
            if va[pre_s:s, bc_speaker].any():
                continue
            if va[e:post_e, bc_speaker].any():
                continue
            # Guard against empty pre-window at the start of an interaction.
            # Without this, numpy warns on mean(empty). An empty pre-window cannot
            # satisfy the "main speaker active before BC" condition.
            if pre_s >= s:
                continue
            if va[pre_s:s, main].mean() < main_min:
                continue
            events.append({
                "task_source": "bc",
                "label": 1,
                "bc_speaker": bc_speaker,
                "main_speaker": main,
                "onset": s,
                "end": e,
            })
    return events


def add_row(rows: list[dict], **kwargs) -> None:
    """Append an event prediction row."""
    rows.append(kwargs)


def collect_rows_for_interaction(interaction_key: str, project_split: str, va: np.ndarray, probs, p_now, p_future, params: dict) -> tuple[list[dict], dict]:
    """Collect all task rows for one interaction."""
    duration_s = len(p_now) / FRAME_HZ
    min_context = params.get("min_context_s", 3.0)
    p_bc = backchannel_prob_from_probs(probs)
    sh_events = extract_shift_hold_events(va, params)
    bc_events = extract_backchannel_events(va, params)
    rows: list[dict] = []
    event_id = 0

    # S/H
    for ev in sh_events:
        shift_speaker = 1 - ev["prev_speaker"]
        event_id += 1
        for f in range(ev["eval_start"], ev["eval_end"]):
            if valid_future_frame(f, duration_s, min_context):
                add_row(rows, interaction_key=interaction_key, project_split=project_split, task="S/H",
                        label=ev["label"], score=float(p_now[f, shift_speaker]), frame=f, time_s=f/FRAME_HZ,
                        speaker=shift_speaker, source_event=ev["task_source"], event_id=f"SH_{event_id}", positive_class="SHIFT")

    # S-pred
    pred_len = sec_to_frame(params.get("pred_window_s", 0.50))
    far = sec_to_frame(params.get("negative_far_s", 2.0))
    pos_count = 0
    for ev in sh_events:
        if ev["label"] != 1:
            continue
        event_id += 1
        next_spk = ev["next_speaker"]
        for f in range(max(0, ev["silence_start"] - pred_len), ev["silence_start"]):
            if valid_future_frame(f, duration_s, min_context):
                pos_count += 1
                add_row(rows, interaction_key=interaction_key, project_split=project_split, task="S-pred",
                        label=1, score=float(p_future[f, next_spk]), frame=f, time_s=f/FRAME_HZ,
                        speaker=next_spk, source_event="shift_pred_pos", event_id=f"SP_{event_id}", positive_class="SHIFT_SOON")
    neg_count = 0
    for current in [0, 1]:
        other = 1 - current
        for seg_s, seg_e in frame_segments(va[:, current]):
            if seg_e - seg_s < pred_len + far:
                continue
            s, e = seg_s, seg_s + pred_len
            if va[s:e, other].any() or va[e:min(len(va), e+far), other].any():
                continue
            event_id += 1
            for f in range(s, e):
                if neg_count >= pos_count:
                    break
                if valid_future_frame(f, duration_s, min_context):
                    neg_count += 1
                    add_row(rows, interaction_key=interaction_key, project_split=project_split, task="S-pred",
                            label=0, score=float(p_future[f, other]), frame=f, time_s=f/FRAME_HZ,
                            speaker=other, source_event="shift_pred_neg", event_id=f"SPN_{event_id}", positive_class="SHIFT_SOON")
            if neg_count >= pos_count:
                break
        if neg_count >= pos_count:
            break

    # BC-pred
    bc_pos = 0
    for ev in bc_events:
        event_id += 1
        spk = ev["bc_speaker"]
        for f in range(max(0, ev["onset"] - pred_len), ev["onset"]):
            if valid_future_frame(f, duration_s, min_context):
                bc_pos += 1
                add_row(rows, interaction_key=interaction_key, project_split=project_split, task="BC-pred",
                        label=1, score=float(p_bc[f, spk]), frame=f, time_s=f/FRAME_HZ,
                        speaker=spk, source_event="bc_pred_pos", event_id=f"BC_{event_id}", positive_class="BC_SOON")
    bc_neg = 0
    for listener in [0, 1]:
        main = 1 - listener
        candidate = va[:, main] & (~va[:, listener])
        for seg_s, seg_e in frame_segments(candidate):
            if seg_e - seg_s < pred_len + far:
                continue
            s, e = seg_s, seg_s + pred_len
            if va[s:min(len(va), e+far), listener].any():
                continue
            event_id += 1
            for f in range(s, e):
                if bc_neg >= bc_pos:
                    break
                if valid_future_frame(f, duration_s, min_context):
                    bc_neg += 1
                    add_row(rows, interaction_key=interaction_key, project_split=project_split, task="BC-pred",
                            label=0, score=float(p_bc[f, listener]), frame=f, time_s=f/FRAME_HZ,
                            speaker=listener, source_event="bc_pred_neg", event_id=f"BCN_{event_id}", positive_class="BC_SOON")
            if bc_neg >= bc_pos:
                break
        if bc_neg >= bc_pos:
            break

    # S/L: LONG positive, SHORT negative.
    eval_len = sec_to_frame(params.get("short_long_eval_len_s", 0.20))
    for ev in sh_events:
        if ev["label"] == 1:
            event_id += 1
            spk = ev["next_speaker"]
            for f in range(ev["onset"], min(len(p_future), ev["onset"] + eval_len)):
                if valid_future_frame(f, duration_s, min_context):
                    add_row(rows, interaction_key=interaction_key, project_split=project_split, task="S/L",
                            label=1, score=float(p_future[f, spk]), frame=f, time_s=f/FRAME_HZ,
                            speaker=spk, source_event="long_shift", event_id=f"SL_{event_id}", positive_class="LONG")
    for ev in bc_events:
        event_id += 1
        spk = ev["bc_speaker"]
        for f in range(ev["onset"], min(len(p_future), ev["onset"] + eval_len)):
            if valid_future_frame(f, duration_s, min_context):
                add_row(rows, interaction_key=interaction_key, project_split=project_split, task="S/L",
                        label=0, score=float(p_future[f, spk]), frame=f, time_s=f/FRAME_HZ,
                        speaker=spk, source_event="short_bc", event_id=f"SL_{event_id}", positive_class="LONG")

    summary = {
        "interaction_key": interaction_key,
        "project_split": project_split,
        "duration_s": duration_s,
        "shift_hold_events_total": len(sh_events),
        "shift_events": sum(ev["label"] == 1 for ev in sh_events),
        "hold_events": sum(ev["label"] == 0 for ev in sh_events),
        "bc_events": len(bc_events),
        "prediction_rows_total": len(rows),
    }
    return rows, summary


def extract_events(cfg: dict) -> None:
    """Read datastore and VAP manifests, then write event prediction rows."""
    dataset_root = Path(get(cfg, "dataset.root", "data/datastore"))
    stereo = pd.read_csv(dataset_root / "manifests" / "stereo_manifest.csv")
    vap_manifest = pd.read_csv(Path(get(cfg, "vap.output_root", "outputs/vap")) / "vap_manifest.csv")
    merged = stereo.merge(vap_manifest, on="interaction_key", suffixes=("", "_vap"))
    params = {
        "min_context_s": 3.0,
        "shift_pre_offset_s": 1.0,
        "shift_post_onset_s": 1.0,
        "shift_hold_eval_offset_s": 0.05,
        "shift_hold_eval_len_s": 0.10,
        "pred_window_s": 0.50,
        "negative_far_s": 2.0,
        "bc_max_duration_s": 1.0,
        "bc_pre_silence_s": 1.0,
        "bc_post_silence_s": 2.0,
        "bc_main_min_active_ratio": 0.10,
        "short_long_eval_len_s": 0.20,
    }
    all_rows = []
    summaries = []
    for _, row in merged.iterrows():
        if str(row.get("status", "")).startswith("failed"):
            continue
        vap_root = Path(get(cfg, "vap.output_root", "outputs/vap"))
        pred_path = Path(str(row.get("prediction_path", "")))
        pred_rel = str(row.get("prediction_relpath", ""))
        if (not pred_path.exists()) and pred_rel and pred_rel != "nan":
            pred_path = vap_root / pred_rel
        probs, p_now, p_future = load_prediction_arrays(pred_path)
        n_frames = len(p_now)
        vad_a, _ = read_vad_segments(resolve_under(dataset_root, row["vad_a_relpath"]))
        vad_b, _ = read_vad_segments(resolve_under(dataset_root, row["vad_b_relpath"]))
        va = vad_segments_to_frames(vad_a, vad_b, n_frames, frame_hz=FRAME_HZ)
        rows, summary = collect_rows_for_interaction(row["interaction_key"], row.get("project_split", ""), va, probs, p_now, p_future, params)
        all_rows.extend(rows)
        summaries.append(summary)
        print(row["interaction_key"], summary)
    out_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    write_csv(out_csv, all_rows)
    write_csv(out_csv.parent / "event_summary.csv", summaries)
    print(f"Wrote {out_csv}")
