"""Audited VAP event extraction and zero-shot score-row construction.

This module separates three concepts:

1. Dialogue-state extraction from Seamless-provided two-speaker VAD.
2. Event definitions for paper-comparable VAP tasks and exploratory overlap tasks.
3. Score-row construction from VAP outputs.

Paper-comparable event tasks:
- S/H-paper: SHIFT vs HOLD during mutual silence.
- S-pred-clear: 500 ms before paper-valid gap SHIFT events.
- BC-pred: 500 ms before isolated backchannels.
- S/L-paper: SHORT=BC onset, LONG=paper-valid SHIFT onset.

Exploratory event task:
- S-pred-overlap: 500 ms before rapid gap or overlap speaker-change events.

The primary score variant is ``paper_256``, reconstructed from the discrete VAP
256-class probability vector. Diagnostic score variants from exported VAP arrays
are also written, but should not be used as paper-comparable baseline claims.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import get
from .paths import resolve_under
from .vad import read_vad_segments, vad_segments_to_frames, frame_segments
from .vap_outputs import load_prediction_arrays, backchannel_prob_from_probs
from .vap_scores import compute_vap_scores, describe_score_subsets
from .reports import write_csv, write_json


FRAME_HZ = 50
HORIZON_S = 2.0
EPS = 1e-8


def sec_to_frame(t: float, frame_hz: int = FRAME_HZ) -> int:
    """Convert seconds to nearest frame."""
    return int(round(float(t) * frame_hz))


def valid_future_frame(frame: int, duration_s: float, min_context_s: float = 3.0) -> bool:
    """True if a frame has enough past context and a full future VAP horizon."""
    t = frame / FRAME_HZ
    return t >= min_context_s and (t + HORIZON_S) <= duration_s


def dialog_states(va: np.ndarray) -> np.ndarray:
    """Map two-speaker activity frames to VAP dialogue states.

    State convention follows the VAP event code:
        0 = only speaker A
        1 = silence
        2 = overlap
        3 = only speaker B
    """
    a = va[:, 0].astype(bool)
    b = va[:, 1].astype(bool)
    ds = np.ones(len(va), dtype=np.int64)
    ds[a & ~b] = 0
    ds[a & b] = 2
    ds[~a & b] = 3
    return ds


def state_runs(ds: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous dialogue-state runs as half-open (start, end, state)."""
    ds = np.asarray(ds, dtype=np.int64)
    if len(ds) == 0:
        return []

    padded = np.concatenate([[ds[0] - 999], ds, [ds[-1] - 999]])
    changes = np.where(np.diff(padded) != 0)[0]
    return [(int(s), int(e), int(ds[s])) for s, e in zip(changes[:-1], changes[1:])]


def single_speaker_from_state(state: int) -> int | None:
    """Return speaker id for single-speaker states, else None."""
    if state == 0:
        return 0
    if state == 3:
        return 1
    return None


def fill_hold_silences(va: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Fill A-silence-A and B-silence-B gaps for strict pre/post checks.

    The VAP event code fills HOLD templates before matching, allowing same-speaker
    IPUs separated by a hold pause to count as one speaker-control region.
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
    """True when only ``speaker`` is active for every frame in [start, end)."""
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


def extract_shift_hold_paper(va: np.ndarray, params: dict) -> list[dict]:
    """Extract paper-comparable S/H events from mutual-silence templates.

    SHIFT templates:
        A -> silence -> B
        B -> silence -> A

    HOLD templates:
        A -> silence -> A
        B -> silence -> B
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
    min_silence = sec_to_frame(params.get("min_silence_shift_hold_s", 0.15))

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


def extract_shift_overlap_exploratory(va: np.ndarray, params: dict) -> list[dict]:
    """Extract exploratory rapid-gap/overlap speaker-change events.

    These are not paper-comparable S/H events. They are kept separate as
    S-pred-overlap for future HRI/adaptation analyses where interruptions and
    near-zero-gap transitions matter.
    """
    n_frames = len(va)
    ds = dialog_states(va)
    filled = fill_hold_silences(va, ds)
    runs = state_runs(ds)

    pre = sec_to_frame(params.get("pre_offset_shift_s", 1.0))
    post = sec_to_frame(params.get("post_onset_shift_s", 1.0))
    max_gap = sec_to_frame(params.get("min_silence_shift_hold_s", 0.15))
    max_overlap = sec_to_frame(params.get("max_overlap_shift_s", 0.15))

    events: list[dict] = []

    for i in range(len(runs) - 2):
        _, _, v0 = runs[i]
        s1, e1, v1 = runs[i + 1]
        s2, _, v2 = runs[i + 2]

        prev_speaker = single_speaker_from_state(v0)
        next_speaker = single_speaker_from_state(v2)
        if prev_speaker is None or next_speaker is None or prev_speaker == next_speaker:
            continue

        source = None
        event_frame = None

        if v1 == 1 and (e1 - s1) < max_gap:
            source = "rapid_gap_shift"
            event_frame = s2
        elif v1 == 2 and (e1 - s1) <= max_overlap:
            source = "overlap_shift"
            event_frame = s1

        if source is None or event_frame is None:
            continue

        pre_start = s1 - pre
        post_start = s2
        post_end = s2 + post
        if pre_start < 0 or post_end > n_frames:
            continue

        if not single_speaker_region(filled, pre_start, s1, prev_speaker):
            continue
        if not single_speaker_region(filled, post_start, post_end, next_speaker):
            continue

        events.append({
            "task_source": source,
            "label": 1,
            "prev_speaker": int(prev_speaker),
            "next_speaker": int(next_speaker),
            "event_frame": int(event_frame),
            "middle_start": int(s1),
            "middle_end": int(e1),
            "onset": int(s2),
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
    """Frames where the future horizon belongs to the last/current speaker."""
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
    """Return non-overlapping fixed-length candidate windows from a two-channel mask."""
    wins: list[dict] = []
    for speaker in [0, 1]:
        for s, e in frame_segments(mask[:, speaker]):
            if e - s < dur_frames:
                continue
            for start in range(s, e - dur_frames + 1, dur_frames):
                wins.append({"start": int(start), "end": int(start + dur_frames), "speaker": int(speaker)})
    return wins


def choose_balanced_windows(windows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Choose up to ``n`` negative windows with seeded sampling."""
    if n <= 0 or not windows:
        return []
    if len(windows) <= n:
        return windows

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=n, replace=False)
    return [windows[int(i)] for i in sorted(idx.tolist())]


def add_score_window_rows(
    rows: list[dict],
    *,
    interaction_key: str,
    project_split: str,
    task: str,
    task_family: str,
    label: int,
    score_specs: list[tuple[str, np.ndarray, int]],
    speaker: int,
    source_event: str,
    event_id: str,
    positive_class: str,
    start: int,
    end: int,
    duration_s: float,
    min_context: float,
    paper_event: bool,
) -> int:
    """Append one row per frame and score variant for a task window.

    Returns the number of valid frames, not multiplied by score variants.
    """
    n_valid_frames = 0

    for f in range(start, end):
        if not valid_future_frame(f, duration_s, min_context):
            continue

        n_valid_frames += 1
        for score_variant, score_arr, score_speaker in score_specs:
            rows.append({
                "interaction_key": interaction_key,
                "project_split": project_split,
                "task": task,
                "task_family": task_family,
                "label": int(label),
                "score": prob2(score_arr, f, score_speaker),
                "frame": int(f),
                "time_s": float(f / FRAME_HZ),
                "speaker": int(speaker),
                "score_speaker": int(score_speaker),
                "source_event": source_event,
                "event_id": event_id,
                "positive_class": positive_class,
                "score_variant": score_variant,
                "paper_comparable": bool(paper_event and score_variant == "paper_256"),
            })

    return n_valid_frames


def transition_attrition(va: np.ndarray, duration_s: float, params: dict) -> dict:
    """Count how raw dialogue-state transitions are filtered into final events."""
    ds = dialog_states(va)
    filled = fill_hold_silences(va, ds)
    runs = state_runs(ds)

    metric_pad = sec_to_frame(params.get("shift_hold_eval_offset_s", 0.05))
    metric_dur = sec_to_frame(params.get("shift_hold_eval_len_s", 0.10))
    min_silence = sec_to_frame(params.get("min_silence_shift_hold_s", 0.15))
    pre = sec_to_frame(params.get("pre_offset_shift_s", 1.0))
    post = sec_to_frame(params.get("post_onset_shift_s", 1.0))
    max_overlap = sec_to_frame(params.get("max_overlap_shift_s", 0.15))

    out = {
        "raw_single_speaker_alternations": 0,
        "gap_shift_triads": 0,
        "gap_shift_min_silence": 0,
        "gap_shift_pre_ok": 0,
        "gap_shift_post_ok": 0,
        "gap_shift_pre_post_ok": 0,
        "gap_shift_metric_context_ok": 0,
        "rapid_gap_shift_triads": 0,
        "overlap_shift_triads": 0,
        "overlap_shift_short_enough": 0,
        "overlap_shift_pre_post_ok": 0,
        "overlap_shift_context_ok": 0,
        "hold_triads": 0,
    }

    single_runs = [(s, e, v) for (s, e, v) in runs if v in (0, 3)]
    for a, b in zip(single_runs[:-1], single_runs[1:]):
        if a[2] != b[2]:
            out["raw_single_speaker_alternations"] += 1

    for i in range(len(runs) - 2):
        _, _, v0 = runs[i]
        s1, e1, v1 = runs[i + 1]
        s2, _, v2 = runs[i + 2]

        prev_sp = single_speaker_from_state(v0)
        next_sp = single_speaker_from_state(v2)
        if prev_sp is None or next_sp is None:
            continue

        if v1 == 1 and prev_sp == next_sp:
            out["hold_triads"] += 1

        if v1 == 2 and prev_sp != next_sp:
            out["overlap_shift_triads"] += 1
            if (e1 - s1) <= max_overlap:
                out["overlap_shift_short_enough"] += 1
                pre_ok = s1 - pre >= 0 and single_speaker_region(filled, s1 - pre, s1, prev_sp)
                post_ok = s2 + post <= len(va) and single_speaker_region(filled, s2, s2 + post, next_sp)
                if pre_ok and post_ok:
                    out["overlap_shift_pre_post_ok"] += 1
                    if valid_future_frame(s1, duration_s, params.get("min_context_s", 3.0)):
                        out["overlap_shift_context_ok"] += 1

        if v1 != 1 or prev_sp == next_sp:
            continue

        out["gap_shift_triads"] += 1
        if (e1 - s1) < min_silence:
            out["rapid_gap_shift_triads"] += 1
            continue

        out["gap_shift_min_silence"] += 1
        pre_ok = s1 - pre >= 0 and single_speaker_region(filled, s1 - pre, s1, prev_sp)
        post_ok = s2 + post <= len(va) and single_speaker_region(filled, s2, s2 + post, next_sp)

        if pre_ok:
            out["gap_shift_pre_ok"] += 1
        if post_ok:
            out["gap_shift_post_ok"] += 1
        if pre_ok and post_ok:
            out["gap_shift_pre_post_ok"] += 1
            eval_start = s1 + metric_pad
            eval_end = eval_start + metric_dur
            if eval_end <= e1 and valid_future_frame(eval_start, duration_s, params.get("min_context_s", 3.0)):
                out["gap_shift_metric_context_ok"] += 1

    return out


def collect_rows_for_interaction(
    interaction_key: str,
    project_split: str,
    va: np.ndarray,
    probs: np.ndarray,
    p_now: np.ndarray,
    p_future: np.ndarray,
    params: dict,
) -> tuple[list[dict], dict, dict]:
    """Collect all scored event rows plus summary and attrition for one interaction."""
    duration_s = len(p_now) / FRAME_HZ
    min_context = params.get("min_context_s", 3.0)

    scores = compute_vap_scores(probs)
    p_bc_approx = backchannel_prob_from_probs(probs)

    sh_events = extract_shift_hold_paper(va, params)
    overlap_events = extract_shift_overlap_exploratory(va, params)
    bc_events = extract_backchannel_events(va, params)

    pred_len = sec_to_frame(params.get("pred_window_s", 0.50))
    far = sec_to_frame(params.get("negative_far_s", 2.0))
    onset_len = sec_to_frame(params.get("short_long_eval_len_s", 0.20))
    negative_seed = int(params.get("negative_sample_seed", 42))

    rows: list[dict] = []
    event_id = 0

    def eid(prefix: str) -> str:
        nonlocal event_id
        event_id += 1
        return f"{prefix}_{event_id}"

    # S/H-paper: positive score means SHIFT.
    for ev in sh_events:
        prev = ev["prev_speaker"]
        shift_speaker = 1 - prev
        add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S/H-paper",
            task_family="paper",
            label=ev["label"],
            score_specs=[
                ("paper_256", scores["p_silence"], shift_speaker),
                ("diag_p_now", p_now, shift_speaker),
                ("diag_p_future", p_future, shift_speaker),
            ],
            speaker=shift_speaker,
            source_event=ev["task_source"],
            event_id=eid("SH"),
            positive_class="SHIFT",
            start=ev["eval_start"],
            end=ev["eval_end"],
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )

    # S-pred-clear positives: 500 ms before paper-valid SHIFT silence.
    sp_pos_events = 0
    for ev in sh_events:
        if ev["label"] != 1:
            continue
        start = ev["silence_start"] - pred_len
        end = ev["silence_start"]
        if start < 0:
            continue
        n = add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S-pred-clear",
            task_family="paper",
            label=1,
            score_specs=[
                ("paper_256", scores["p_active"], ev["next_speaker"]),
                ("diag_p_future", p_future, ev["next_speaker"]),
            ],
            speaker=ev["next_speaker"],
            source_event="shift_pred_clear_pos",
            event_id=eid("SPC"),
            positive_class="SHIFT_SOON",
            start=start,
            end=end,
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )
        if n > 0:
            sp_pos_events += 1

    # S-pred-overlap positives: exploratory rapid-gap/overlap transition events.
    spo_pos_events = 0
    for ev in overlap_events:
        start = ev["event_frame"] - pred_len
        end = ev["event_frame"]
        if start < 0:
            continue
        n = add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S-pred-overlap",
            task_family="exploratory",
            label=1,
            score_specs=[
                ("paper_256", scores["p_active"], ev["next_speaker"]),
                ("diag_p_future", p_future, ev["next_speaker"]),
            ],
            speaker=ev["next_speaker"],
            source_event=ev["task_source"],
            event_id=eid("SPO"),
            positive_class="SHIFT_SOON",
            start=start,
            end=end,
            duration_s=duration_s,
            min_context=min_context,
            paper_event=False,
        )
        if n > 0:
            spo_pos_events += 1

    # Shared non-shift candidates for negative prediction windows.
    ns = non_shift_mask(va, horizon_frames=far, majority_ratio=1.0)
    active_single = np.zeros_like(ns)
    active_single[:, 0] = va[:, 0] & ~va[:, 1]
    active_single[:, 1] = va[:, 1] & ~va[:, 0]

    sp_neg_candidates = candidate_windows(ns & active_single, pred_len)
    sp_neg_windows = choose_balanced_windows(sp_neg_candidates, sp_pos_events, seed=negative_seed)

    for win in sp_neg_windows:
        current = win["speaker"]
        target = 1 - current
        add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S-pred-clear",
            task_family="paper",
            label=0,
            score_specs=[
                ("paper_256", scores["p_active"], target),
                ("diag_p_future", p_future, target),
            ],
            speaker=target,
            source_event="shift_pred_clear_neg",
            event_id=eid("SPCN"),
            positive_class="SHIFT_SOON",
            start=win["start"],
            end=win["end"],
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )

    spo_neg_windows = choose_balanced_windows(sp_neg_candidates, spo_pos_events, seed=negative_seed + 7)
    for win in spo_neg_windows:
        current = win["speaker"]
        target = 1 - current
        add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S-pred-overlap",
            task_family="exploratory",
            label=0,
            score_specs=[
                ("paper_256", scores["p_active"], target),
                ("diag_p_future", p_future, target),
            ],
            speaker=target,
            source_event="shift_pred_overlap_neg",
            event_id=eid("SPON"),
            positive_class="SHIFT_SOON",
            start=win["start"],
            end=win["end"],
            duration_s=duration_s,
            min_context=min_context,
            paper_event=False,
        )

    # BC-pred positives.
    bc_pos_events = 0
    for ev in bc_events:
        start = ev["onset"] - pred_len
        end = ev["onset"]
        if start < 0:
            continue
        n = add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="BC-pred",
            task_family="paper",
            label=1,
            score_specs=[
                ("paper_256", scores["p_bc"], ev["bc_speaker"]),
                ("diag_approx_bc", p_bc_approx, ev["bc_speaker"]),
            ],
            speaker=ev["bc_speaker"],
            source_event="bc_pred_pos",
            event_id=eid("BC"),
            positive_class="BC_SOON",
            start=start,
            end=end,
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )
        if n > 0:
            bc_pos_events += 1

    bc_neg_candidates = candidate_windows(ns, pred_len)
    bc_neg_windows = choose_balanced_windows(bc_neg_candidates, bc_pos_events, seed=negative_seed + 1)

    for win in bc_neg_windows:
        current = win["speaker"]
        listener = 1 - current
        add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="BC-pred",
            task_family="paper",
            label=0,
            score_specs=[
                ("paper_256", scores["p_bc"], listener),
                ("diag_approx_bc", p_bc_approx, listener),
            ],
            speaker=listener,
            source_event="bc_pred_neg",
            event_id=eid("BCN"),
            positive_class="BC_SOON",
            start=win["start"],
            end=win["end"],
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )

    # S/L-paper: SHORT positive = BC onset; LONG negative = paper-valid SHIFT onset.
    for ev in bc_events:
        add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S/L-paper",
            task_family="paper",
            label=1,
            score_specs=[
                ("paper_256", scores["p_bc"], ev["bc_speaker"]),
                ("diag_approx_bc", p_bc_approx, ev["bc_speaker"]),
            ],
            speaker=ev["bc_speaker"],
            source_event="short_bc",
            event_id=eid("SL_SHORT"),
            positive_class="SHORT",
            start=ev["onset"],
            end=ev["onset"] + onset_len,
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )

    for ev in sh_events:
        if ev["label"] != 1:
            continue
        add_score_window_rows(
            rows,
            interaction_key=interaction_key,
            project_split=project_split,
            task="S/L-paper",
            task_family="paper",
            label=0,
            score_specs=[
                ("paper_256", scores["p_bc"], ev["next_speaker"]),
                ("diag_approx_bc", p_bc_approx, ev["next_speaker"]),
            ],
            speaker=ev["next_speaker"],
            source_event="long_shift",
            event_id=eid("SL_LONG"),
            positive_class="SHORT",
            start=ev["onset"],
            end=ev["onset"] + onset_len,
            duration_s=duration_s,
            min_context=min_context,
            paper_event=True,
        )

    summary = {
        "interaction_key": interaction_key,
        "project_split": project_split,
        "duration_s": duration_s,
        "shift_hold_events_total": len(sh_events),
        "shift_events": int(sum(ev["label"] == 1 for ev in sh_events)),
        "hold_events": int(sum(ev["label"] == 0 for ev in sh_events)),
        "overlap_shift_events": int(len(overlap_events)),
        "bc_events": int(len(bc_events)),
        "s_pred_clear_pos_events": int(sp_pos_events),
        "s_pred_clear_neg_events": int(len(sp_neg_windows)),
        "s_pred_overlap_pos_events": int(spo_pos_events),
        "s_pred_overlap_neg_events": int(len(spo_neg_windows)),
        "bc_pred_pos_events": int(bc_pos_events),
        "bc_pred_neg_events": int(len(bc_neg_windows)),
        "prediction_rows_total": int(len(rows)),
    }

    attrition = {
        "interaction_key": interaction_key,
        "project_split": project_split,
        "duration_s": duration_s,
        **transition_attrition(va, duration_s, params),
        "final_shift_events": summary["shift_events"],
        "final_hold_events": summary["hold_events"],
        "final_overlap_shift_events": summary["overlap_shift_events"],
        "final_bc_events": summary["bc_events"],
    }

    return rows, summary, attrition


def _prediction_path(row: pd.Series, output_root: Path) -> Path:
    """Resolve a VAP prediction path from a vap_manifest row."""
    raw = str(row.get("prediction_path", ""))
    if raw and raw != "nan":
        p = Path(raw)
        if p.exists():
            return p

    rel = str(row.get("prediction_relpath", ""))
    if rel and rel != "nan":
        return output_root / rel

    raise FileNotFoundError(f"No prediction path for interaction {row.get('interaction_key')}")


def extract_events(cfg: dict) -> None:
    """Read datastore/VAP manifests and write scored event prediction rows."""
    dataset_root = Path(get(cfg, "dataset.root", "data/datastore"))
    output_root = Path(get(cfg, "vap.output_root", "outputs/vap"))
    out_events = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv")).parent
    out_events.mkdir(parents=True, exist_ok=True)

    stereo = pd.read_csv(dataset_root / "manifests" / "stereo_manifest.csv")
    vap_manifest = pd.read_csv(output_root / "vap_manifest.csv")
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
        key = str(row["interaction_key"])
        status = str(row.get("status", "done"))
        if status not in {"done", "exists", "ok"}:
            skipped.append({"interaction_key": key, "reason": f"vap_status={status}"})
            continue

        pred_path = _prediction_path(row, output_root)
        probs, p_now, p_future = load_prediction_arrays(pred_path)

        vad_a, _ = read_vad_segments(resolve_under(dataset_root, row["vad_a_relpath"]))
        vad_b, _ = read_vad_segments(resolve_under(dataset_root, row["vad_b_relpath"]))
        va = vad_segments_to_frames(vad_a, vad_b, len(p_now), frame_hz=FRAME_HZ)

        project_split = str(row.get("project_split", row.get("project_split_vap", "")))
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

    write_csv(out_events / "event_predictions.csv", all_rows)
    write_csv(out_events / "event_summary.csv", summaries)
    write_csv(out_events / "event_attrition.csv", attrition_rows)
    write_json(out_events / "score_subsets.json", describe_score_subsets())
    if skipped:
        write_csv(out_events / "skipped_interactions.csv", skipped)

    print(f"Wrote {out_events / 'event_predictions.csv'} rows={len(all_rows)}")
    print(f"Wrote {out_events / 'event_summary.csv'} interactions={len(summaries)}")
    print(f"Wrote {out_events / 'event_attrition.csv'} interactions={len(attrition_rows)}")
