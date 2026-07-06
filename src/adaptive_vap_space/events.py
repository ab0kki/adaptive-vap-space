"""VAP-paper-style event extraction.

This module extracts VAP zero-shot evaluation events from two-speaker VAD.

Design goals:
- Use dialog-state templates, matching the VAP event repository.
- Keep paper timings where the paper is explicit.
- Avoid the old sorted-segment shortcut that produced suspiciously few/imbalanced events.
- Keep probability-score choices explicit and auditable.

Known limitation:
This is high-fidelity event extraction, but not a perfect reproduction of the
Discrete VAP 256-state task-specific subset scorers. S/H and S-pred use exported
p_future; BC-pred and S/L use the repository's documented approximate p_bc helper.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import get
from .paths import resolve_under
from .vad import read_vad_segments, vad_segments_to_frames, frame_segments
from .vap_outputs import load_prediction_arrays, backchannel_prob_from_probs
from .reports import write_csv


FRAME_HZ = 50
HORIZON_S = 2.0
EPS = 1e-8


def sec_to_frame(t: float, frame_hz: int = FRAME_HZ) -> int:
    """Convert seconds to nearest frame."""
    return int(round(float(t) * frame_hz))


def valid_future_frame(frame: int, duration_s: float, min_context_s: float = 3.0) -> bool:
    """Avoid evaluating frames before enough context or beyond VAP future horizon."""
    t = frame / FRAME_HZ
    return t >= min_context_s and (t + HORIZON_S) <= duration_s


def dialog_states(va: np.ndarray) -> np.ndarray:
    """Map two-speaker VA frames to VAP dialog states.

    State convention follows the VAP event repository:
    0 = only A
    1 = silence
    2 = overlap
    3 = only B
    """
    a = va[:, 0].astype(bool)
    b = va[:, 1].astype(bool)
    ds = np.ones(len(va), dtype=np.int64)
    ds[a & ~b] = 0
    ds[a & b] = 2
    ds[~a & b] = 3
    return ds


def state_runs(ds: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous dialog-state runs as half-open (start, end, state)."""
    ds = np.asarray(ds, dtype=np.int64)
    if len(ds) == 0:
        return []

    padded = np.concatenate([[ds[0] - 999], ds, [ds[-1] - 999]])
    changes = np.where(np.diff(padded) != 0)[0]
    runs: list[tuple[int, int, int]] = []
    for s, e in zip(changes[:-1], changes[1:]):
        runs.append((int(s), int(e), int(ds[s])))
    return runs


def single_speaker_from_state(state: int) -> int | None:
    """Return speaker id for single-speaker states, else None."""
    if state == 0:
        return 0
    if state == 3:
        return 1
    return None


def fill_hold_silences(va: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Fill A-silence-A and B-silence-B gaps for template pre/post checks.

    The VAP repository fills HOLD templates before matching SHIFT/HOLD events.
    This lets same-speaker IPUs separated by hold pauses count as one
    speaker-control region for the 1 s pre/post conditions.
    """
    filled = va.copy()
    runs = state_runs(ds)

    for i in range(len(runs) - 2):
        _, _, v0 = runs[i]
        s1, e1, v1 = runs[i + 1]
        _, _, v2 = runs[i + 2]

        if v1 != 1:
            continue
        if v0 == 0 and v2 == 0:
            filled[s1:e1, 0] = True
        elif v0 == 3 and v2 == 3:
            filled[s1:e1, 1] = True

    return filled


def last_speaker_array(va: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Return the most recent single speaker at each frame; -1 if unknown."""
    last = np.full(len(va), -1, dtype=np.int64)
    cur = -1

    for i, st in enumerate(ds):
        sp = single_speaker_from_state(int(st))
        if sp is not None:
            cur = sp
        last[i] = cur

    return last


def single_speaker_region(va: np.ndarray, start: int, end: int, speaker: int) -> bool:
    """True when only `speaker` is active for every frame in [start, end)."""
    start = max(0, int(start))
    end = min(len(va), int(end))
    if end <= start:
        return False

    other = 1 - speaker
    return bool(va[start:end, speaker].all() and not va[start:end, other].any())


def prob2(arr: np.ndarray, frame: int, speaker: int) -> float:
    """Read a two-speaker probability array robustly."""
    arr = np.asarray(arr, dtype=float)
    return float(arr[frame, speaker])


def normalized_shift_score(p_future: np.ndarray, frame: int, prev_speaker: int) -> float:
    """Score that the next speaker is the other speaker.

    The old baseline used p_now for the shift speaker during silence, which is
    not a future turn-taking score. This score uses p_future and normalizes the
    two speaker scores into a relative next-speaker probability.

    This is still a diagnostic score, not the exact paper Discrete 256-state
    subset comparison.
    """
    shift_speaker = 1 - prev_speaker
    p_shift = prob2(p_future, frame, shift_speaker)
    p_hold = prob2(p_future, frame, prev_speaker)
    return p_shift / (p_shift + p_hold + EPS)


def extract_shift_hold_events(va: np.ndarray, params: dict) -> list[dict]:
    """Extract SHIFT/HOLD silence events from dialog-state templates.

    Paper-faithful templates:
    - SHIFT: A -> silence -> B, or B -> silence -> A
    - HOLD:  A -> silence -> A, or B -> silence -> B

    This function intentionally excludes overlap-shift events from S/H because
    the paper S/H evaluation is defined over mutual silence.
    """
    n_frames = len(va)
    ds = dialog_states(va)
    filled = fill_hold_silences(va, ds)
    runs = state_runs(ds)

    pre_shift = sec_to_frame(params.get("pre_offset_shift_s", 1.0))
    post_shift = sec_to_frame(params.get("post_onset_shift_s", 1.0))
    pre_hold = sec_to_frame(params.get("pre_offset_hold_s", 1.0))
    post_hold = sec_to_frame(params.get("post_onset_hold_s", 1.0))

    metric_pad = sec_to_frame(params.get("shift_hold_eval_offset_s", 0.05))
    metric_dur = sec_to_frame(params.get("shift_hold_eval_len_s", 0.10))
    min_silence = metric_pad + metric_dur

    events: list[dict] = []

    for i in range(len(runs) - 2):
        _, _, v0 = runs[i]
        s1, e1, v1 = runs[i + 1]
        s2, _, v2 = runs[i + 2]

        if v1 != 1:
            continue
        if (e1 - s1) < min_silence:
            continue

        prev_speaker = single_speaker_from_state(v0)
        next_speaker = single_speaker_from_state(v2)
        if prev_speaker is None or next_speaker is None:
            continue

        is_shift = prev_speaker != next_speaker
        pre = pre_shift if is_shift else pre_hold
        post = post_shift if is_shift else post_hold

        pre_start = s1 - pre
        post_start = s2
        post_end = s2 + post

        if pre_start < 0 or post_end > n_frames:
            continue

        if not single_speaker_region(filled, pre_start, s1, prev_speaker):
            continue
        if not single_speaker_region(filled, post_start, post_end, next_speaker):
            continue

        eval_start = s1 + metric_pad
        eval_end = eval_start + metric_dur
        if eval_end > e1 or eval_end > n_frames:
            continue

        events.append({
            "task_source": "shift" if is_shift else "hold",
            "label": int(is_shift),
            "prev_speaker": int(prev_speaker),
            "next_speaker": int(next_speaker),
            "silence_start": int(s1),
            "silence_end": int(e1),
            "onset": int(s2),
            "eval_start": int(eval_start),
            "eval_end": int(eval_end),
        })

    return events


def extract_backchannel_events(va: np.ndarray, params: dict) -> list[dict]:
    """Detect short isolated listener activity used for BC-pred and S/L."""
    n_frames = len(va)
    ds = dialog_states(va)
    last = last_speaker_array(va, ds)

    max_dur = sec_to_frame(params.get("bc_max_duration_s", 1.0))
    min_dur = sec_to_frame(params.get("bc_min_duration_s", 0.20))
    pre = sec_to_frame(params.get("bc_pre_silence_s", 1.0))
    post = sec_to_frame(params.get("bc_post_silence_s", 2.0))
    onset_eval = sec_to_frame(params.get("short_long_eval_len_s", 0.20))

    events: list[dict] = []

    for bc_speaker in [0, 1]:
        main = 1 - bc_speaker
        for s, e in frame_segments(va[:, bc_speaker]):
            dur = e - s

            if dur < min_dur or dur > max_dur:
                continue
            if s - pre < 0 or e + post > n_frames:
                continue
            if va[s - pre:s, bc_speaker].any():
                continue
            if va[e:e + post, bc_speaker].any():
                continue
            if s <= 0 or last[s - 1] != main:
                continue
            if s + onset_eval > n_frames:
                continue

            events.append({
                "task_source": "bc",
                "label": 1,
                "bc_speaker": int(bc_speaker),
                "main_speaker": int(main),
                "onset": int(s),
                "end": int(e),
                "eval_start": int(s),
                "eval_end": int(s + onset_eval),
            })

    return events


def non_shift_mask(va: np.ndarray, horizon_frames: int, majority_ratio: float = 1.0) -> np.ndarray:
    """Frames where future horizon belongs to the last/current speaker.

    This follows the VAP repository idea of non-shift regions: the future horizon
    overwhelmingly belongs to the last speaker, so predicting a shift is wrong.
    """
    n = len(va)
    ds = dialog_states(va)
    last = last_speaker_array(va, ds)
    out = np.zeros((n, 2), dtype=bool)

    for f in range(0, max(0, n - horizon_frames - 1)):
        cur = last[f]
        if cur not in (0, 1):
            continue

        future = va[f + 1:f + 1 + horizon_frames]
        speech_total = float(future.sum())
        if speech_total <= 0:
            continue

        ratio = float(future[:, cur].sum()) / speech_total
        if ratio + EPS >= majority_ratio:
            out[f, cur] = True

    return out


def candidate_windows(mask: np.ndarray, dur_frames: int) -> list[dict]:
    """Return candidate fixed-length windows from a two-channel mask."""
    wins: list[dict] = []

    for speaker in [0, 1]:
        for s, e in frame_segments(mask[:, speaker]):
            if e - s < dur_frames:
                continue
            for start in range(s, e - dur_frames + 1, dur_frames):
                wins.append({
                    "start": int(start),
                    "end": int(start + dur_frames),
                    "speaker": int(speaker),
                })

    return wins


def choose_balanced_windows(windows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Choose up to n negative windows with seeded sampling.

    The VAP repository samples negatives from candidate non-shift regions. This
    implementation keeps the sampling reproducible.
    """
    if n <= 0 or not windows:
        return []
    if len(windows) <= n:
        return windows

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=n, replace=False)
    return [windows[int(i)] for i in sorted(idx.tolist())]


def add_row(rows: list[dict], **kwargs) -> None:
    rows.append(kwargs)


def add_window_rows(
    rows: list[dict],
    *,
    interaction_key: str,
    project_split: str,
    task: str,
    label: int,
    score_arr: np.ndarray,
    speaker: int,
    source_event: str,
    event_id: str,
    positive_class: str,
    start: int,
    end: int,
    duration_s: float,
    min_context: float,
    score_variant: str,
) -> int:
    """Append one row per valid frame inside a task window."""
    n = 0

    for f in range(start, end):
        if valid_future_frame(f, duration_s, min_context):
            add_row(
                rows,
                interaction_key=interaction_key,
                project_split=project_split,
                task=task,
                label=int(label),
                score=prob2(score_arr, f, speaker),
                frame=int(f),
                time_s=float(f / FRAME_HZ),
                speaker=int(speaker),
                source_event=source_event,
                event_id=event_id,
                positive_class=positive_class,
                score_variant=score_variant,
            )
            n += 1

    return n


def collect_rows_for_interaction(
    interaction_key: str,
    project_split: str,
    va: np.ndarray,
    probs,
    p_now,
    p_future,
    params: dict,
) -> tuple[list[dict], dict]:
    """Collect all VAP task rows for one interaction."""
    duration_s = len(p_now) / FRAME_HZ
    min_context = params.get("min_context_s", 3.0)

    p_bc = backchannel_prob_from_probs(probs)

    sh_events = extract_shift_hold_events(va, params)
    bc_events = extract_backchannel_events(va, params)

    pred_len = sec_to_frame(params.get("pred_window_s", 0.50))
    far = sec_to_frame(params.get("negative_far_s", 2.0))
    onset_len = sec_to_frame(params.get("short_long_eval_len_s", 0.20))
    negative_seed = int(params.get("negative_sample_seed", 42))

    rows: list[dict] = []
    event_id = 0

    # S/H: SHIFT positive, HOLD negative.
    for ev in sh_events:
        event_id += 1
        prev = ev["prev_speaker"]
        shift_speaker = 1 - prev

        for f in range(ev["eval_start"], ev["eval_end"]):
            if valid_future_frame(f, duration_s, min_context):
                add_row(
                    rows,
                    interaction_key=interaction_key,
                    project_split=project_split,
                    task="S/H",
                    label=ev["label"],
                    score=normalized_shift_score(p_future, f, prev),
                    frame=int(f),
                    time_s=float(f / FRAME_HZ),
                    speaker=int(shift_speaker),
                    source_event=ev["task_source"],
                    event_id=f"SH_{event_id}",
                    positive_class="SHIFT",
                    score_variant="p_future_shift_vs_hold_ratio",
                )

    # S-pred positives: 500 ms before valid SHIFT silence.
    sp_pos_events = 0
    for ev in sh_events:
        if ev["label"] != 1:
            continue

        event_id += 1
        start = ev["silence_start"] - pred_len
        end = ev["silence_start"]
        if start < 0:
            continue

        n = add_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S-pred",
            label=1,
            score_arr=p_future,
            speaker=ev["next_speaker"],
            source_event="shift_pred_pos",
            event_id=f"SP_{event_id}",
            positive_class="SHIFT_SOON",
            start=start,
            end=end,
            duration_s=duration_s,
            min_context=min_context,
            score_variant="p_future_next_speaker",
        )
        if n > 0:
            sp_pos_events += 1

    # S-pred negatives: sampled non-shift windows during single-speaker activity.
    ns = non_shift_mask(va, horizon_frames=far, majority_ratio=1.0)
    active_single = np.zeros_like(ns)
    active_single[:, 0] = va[:, 0] & ~va[:, 1]
    active_single[:, 1] = va[:, 1] & ~va[:, 0]

    sp_neg_candidates = candidate_windows(ns & active_single, pred_len)
    sp_neg_windows = choose_balanced_windows(sp_neg_candidates, sp_pos_events, seed=negative_seed)

    for win in sp_neg_windows:
        event_id += 1
        current = win["speaker"]
        target = 1 - current

        add_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S-pred",
            label=0,
            score_arr=p_future,
            speaker=target,
            source_event="shift_pred_neg",
            event_id=f"SPN_{event_id}",
            positive_class="SHIFT_SOON",
            start=win["start"],
            end=win["end"],
            duration_s=duration_s,
            min_context=min_context,
            score_variant="p_future_other_speaker",
        )

    # BC-pred positives: 500 ms before valid BC onset.
    bc_pos_events = 0
    for ev in bc_events:
        event_id += 1
        start = ev["onset"] - pred_len
        end = ev["onset"]
        if start < 0:
            continue

        n = add_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="BC-pred",
            label=1,
            score_arr=p_bc,
            speaker=ev["bc_speaker"],
            source_event="bc_pred_pos",
            event_id=f"BC_{event_id}",
            positive_class="BC_SOON",
            start=start,
            end=end,
            duration_s=duration_s,
            min_context=min_context,
            score_variant="p_bc_target_speaker",
        )
        if n > 0:
            bc_pos_events += 1

    # BC-pred negatives: sampled non-shift windows; may include non-active regions.
    bc_neg_candidates = candidate_windows(ns, pred_len)
    bc_neg_windows = choose_balanced_windows(bc_neg_candidates, bc_pos_events, seed=negative_seed + 1)

    for win in bc_neg_windows:
        event_id += 1
        current = win["speaker"]
        listener = 1 - current

        add_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="BC-pred",
            label=0,
            score_arr=p_bc,
            speaker=listener,
            source_event="bc_pred_neg",
            event_id=f"BCN_{event_id}",
            positive_class="BC_SOON",
            start=win["start"],
            end=win["end"],
            duration_s=duration_s,
            min_context=min_context,
            score_variant="p_bc_listener",
        )

    # S/L: SHORT positive = BC onset; LONG negative = SHIFT onset.
    for ev in bc_events:
        event_id += 1

        add_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S/L",
            label=1,
            score_arr=p_bc,
            speaker=ev["bc_speaker"],
            source_event="short_bc",
            event_id=f"SL_SHORT_{event_id}",
            positive_class="SHORT",
            start=ev["onset"],
            end=ev["onset"] + onset_len,
            duration_s=duration_s,
            min_context=min_context,
            score_variant="p_bc_short_speaker",
        )

    for ev in sh_events:
        if ev["label"] != 1:
            continue

        event_id += 1

        add_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S/L",
            label=0,
            score_arr=p_bc,
            speaker=ev["next_speaker"],
            source_event="long_shift",
            event_id=f"SL_LONG_{event_id}",
            positive_class="SHORT",
            start=ev["onset"],
            end=ev["onset"] + onset_len,
            duration_s=duration_s,
            min_context=min_context,
            score_variant="p_bc_long_speaker_should_be_low",
        )

    summary = {
        "interaction_key": interaction_key,
        "project_split": project_split,
        "duration_s": duration_s,
        "shift_hold_events_total": len(sh_events),
        "shift_events": int(sum(ev["label"] == 1 for ev in sh_events)),
        "hold_events": int(sum(ev["label"] == 0 for ev in sh_events)),
        "bc_events": int(len(bc_events)),
        "s_pred_pos_events": int(sp_pos_events),
        "s_pred_neg_events": int(len(sp_neg_windows)),
        "bc_pred_pos_events": int(bc_pos_events),
        "bc_pred_neg_events": int(len(bc_neg_windows)),
        "prediction_rows_total": int(len(rows)),
    }

    return rows, summary


def extract_events(cfg: dict) -> None:
    """Read datastore/VAP manifests and write event prediction rows."""
    dataset_root = Path(get(cfg, "dataset.root", "data/datastore"))
    stereo = pd.read_csv(dataset_root / "manifests" / "stereo_manifest.csv")
    vap_manifest = pd.read_csv(Path(get(cfg, "vap.output_root", "outputs/vap")) / "vap_manifest.csv")
    merged = stereo.merge(vap_manifest, on="interaction_key", suffixes=("", "_vap"))

    params = {
        "min_context_s": 3.0,

        # VAP paper S/H definition.
        "pre_offset_shift_s": 1.0,
        "post_onset_shift_s": 1.0,
        "pre_offset_hold_s": 1.0,
        "post_onset_hold_s": 1.0,
        "shift_hold_eval_offset_s": 0.05,
        "shift_hold_eval_len_s": 0.10,

        # VAP paper S-pred / BC-pred positive window and non-shift horizon.
        "pred_window_s": 0.50,
        "negative_far_s": 2.0,

        # VAP paper BC isolation, plus repo-style 200 ms minimum duration.
        "bc_max_duration_s": 1.0,
        "bc_min_duration_s": 0.20,
        "bc_pre_silence_s": 1.0,
        "bc_post_silence_s": 2.0,

        # VAP paper S/L onset region.
        "short_long_eval_len_s": 0.20,

        # Reproducible negative sampling.
        "negative_sample_seed": 42,
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

        rows, summary = collect_rows_for_interaction(
            row["interaction_key"],
            row["project_split"],
            va,
            probs,
            p_now,
            p_future,
            params,
        )

        all_rows.extend(rows)
        summaries.append(summary)

    out_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_csv(out_csv, pd.DataFrame(all_rows))
    write_csv(out_csv.parent / "event_summary.csv", pd.DataFrame(summaries))

    print(f"Wrote {len(all_rows)} event prediction rows to {out_csv}")
