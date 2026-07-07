"""VAP 256-state zero-shot score helpers.

This module is intentionally explicit because VAP event scoring was the main
ambiguous part of the audit.

The public VAP runner writes a 256-class probability distribution, ``probs``,
for each frame. Each class is one possible future voice-activity pattern over a
2-second projection window with four bins per speaker:

    speaker 0 bins: b0, b1, b2, b3
    speaker 1 bins: b0, b1, b2, b3

The codebook index convention follows the public VAP codebook: speaker 0 bins
come first, speaker 1 bins come second, and the first flattened element is the
least significant bit.

This module reconstructs three task-level VAP scores by summing explicit subsets
of those 256 classes:

    p_silence: next-speaker score during mutual silence, used by S/H-paper.
    p_active:  shift-to-speaker score during active speech, used by S-pred.
    p_bc:      backchannel score, used by BC-pred and S/L-paper.

These arrays are written to event rows as ``score_variant = paper_256``. Exported
``p_now`` and ``p_future`` remain useful diagnostics, but paper-style event
evaluation should prioritize explicit 256-state subsets.
"""
from __future__ import annotations

import numpy as np


N_BINS = 4
N_CLASSES = 256


def all_permutations_mono(n: int, start: int = 0) -> np.ndarray:
    """All binary vectors of length ``n`` from integer ``start`` to ``2**n - 1``."""
    rows = []
    for i in range(start, 2**n):
        rows.append([(i >> j) & 1 for j in range(n)])
    return np.asarray(rows, dtype=np.int64)


def end_of_segment_mono(n: int = N_BINS, max_active_bins: int = 2) -> np.ndarray:
    """Current-speaker ending patterns.

    For n=4 and max_active_bins=2, this returns 0000, 1000, and 1100. This
    mirrors the VAP turn-taking code's end-of-segment subset.
    """
    out = np.zeros((max_active_bins + 1, n), dtype=np.int64)
    for i in range(max_active_bins):
        out[i + 1, : i + 1] = 1
    return out


def on_activity_change_mono(n: int = N_BINS, min_active: int = 2) -> np.ndarray:
    """Patterns where the last ``min_active`` bins are explicitly active.

    For S/H and S-pred, this corresponds to the paper's idea that the candidate
    next speaker is active in the later part of the future projection window,
    while earlier bins may be optional.
    """
    base = np.zeros(n, dtype=np.int64)
    if min_active > 0:
        base[-min_active:] = 1

    permutable = n - min_active
    if permutable <= 0:
        return base[None, :]

    perms = all_permutations_mono(permutable, start=0)
    out = np.repeat(base[None, :], len(perms), axis=0)
    out[:, :permutable] = perms
    return out


def state_index(speaker0_bins: np.ndarray, speaker1_bins: np.ndarray) -> int:
    """Convert two 4-bin speaker vectors to the VAP 8-bit class index.

    The public VAP codebook flattens as speaker 0 bins followed by speaker 1
    bins. The least significant bit is the first element.
    """
    bits = np.concatenate([speaker0_bins, speaker1_bins]).astype(np.int64)
    return int(sum(int(bit) << i for i, bit in enumerate(bits.tolist())))


def combine_speakers(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Pair each speaker-0 pattern in ``a`` with each speaker-1 pattern in ``b``."""
    pairs = []
    for va in a:
        for vb in b:
            pairs.append(np.stack([va, vb], axis=0))
    return pairs


def pair_indices(pairs: list[np.ndarray]) -> np.ndarray:
    """Convert two-speaker bin patterns to sorted unique VAP class indices."""
    idx = [state_index(p[0], p[1]) for p in pairs]
    return np.asarray(sorted(set(idx)), dtype=np.int64)


def silence_next_speaker_indices() -> list[np.ndarray]:
    """Subset indices for P(next speaker = s) during mutual silence."""
    active = on_activity_change_mono(N_BINS, min_active=2)
    silent = np.zeros((1, N_BINS), dtype=np.int64)

    speaker0_next = pair_indices(combine_speakers(active, silent))
    speaker1_next = pair_indices(combine_speakers(silent, active))
    return [speaker0_next, speaker1_next]


def active_shift_indices() -> list[np.ndarray]:
    """Subset indices for P(shift to speaker = s) while the other speaker is active."""
    next_active = on_activity_change_mono(N_BINS, min_active=2)
    current_ending = end_of_segment_mono(N_BINS, max_active_bins=2)

    shift_to_0 = pair_indices(combine_speakers(next_active, current_ending))
    shift_to_1 = pair_indices(combine_speakers(current_ending, next_active))
    return [shift_to_0, shift_to_1]


def active_hold_indices() -> list[np.ndarray]:
    """Subset indices for no-shift/hold alternatives during active speech."""
    continuing = on_activity_change_mono(N_BINS, min_active=2)
    zero = np.zeros((1, N_BINS), dtype=np.int64)

    not_shift_to_0 = pair_indices(combine_speakers(zero, continuing))
    not_shift_to_1 = pair_indices(combine_speakers(continuing, zero))
    return [not_shift_to_0, not_shift_to_1]


def backchannel_indices() -> list[np.ndarray]:
    """Subset indices for P(backchannel by speaker = s).

    The BC speaker has some activity in the first three bins and no activity in
    the final bin. The current/main speaker has final-bin activity.
    """
    bc_early = all_permutations_mono(n=3, start=1)
    bc = np.concatenate(
        [bc_early, np.zeros((len(bc_early), 1), dtype=np.int64)],
        axis=1,
    )

    current_early = all_permutations_mono(n=3, start=0)
    current = np.concatenate(
        [current_early, np.ones((len(current_early), 1), dtype=np.int64)],
        axis=1,
    )

    bc_by_0 = pair_indices(combine_speakers(bc, current))
    bc_by_1 = pair_indices(combine_speakers(current, bc))
    return [bc_by_0, bc_by_1]


SILENCE_NEXT = silence_next_speaker_indices()
ACTIVE_SHIFT = active_shift_indices()
ACTIVE_HOLD = active_hold_indices()
BACKCHANNEL = backchannel_indices()


def _conditional_prob(probs: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """P(pos subset | pos subset or neg subset)."""
    p_pos = probs[:, pos].sum(axis=1)
    p_neg = probs[:, neg].sum(axis=1)
    return p_pos / (p_pos + p_neg + 1e-8)


def compute_vap_scores(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Compute VAP zero-shot task scores from 256-class probabilities.

    Returns:
        ``p_silence``: shape (T, 2), paper-style next-speaker score for S/H.
        ``p_active``: shape (T, 2), paper-style shift-to-speaker score for S-pred.
        ``p_bc``: shape (T, 2), paper-style backchannel score for BC-pred and S/L.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError(f"Expected probs shape (T, 256), got {probs.shape}")

    p_silence = np.zeros((len(probs), 2), dtype=float)
    p_active = np.zeros((len(probs), 2), dtype=float)
    p_bc = np.zeros((len(probs), 2), dtype=float)

    for speaker in [0, 1]:
        other = 1 - speaker
        p_silence[:, speaker] = _conditional_prob(
            probs,
            SILENCE_NEXT[speaker],
            SILENCE_NEXT[other],
        )
        p_active[:, speaker] = _conditional_prob(
            probs,
            ACTIVE_SHIFT[speaker],
            ACTIVE_HOLD[speaker],
        )
        p_bc[:, speaker] = probs[:, BACKCHANNEL[speaker]].sum(axis=1)

    return {"p_silence": p_silence, "p_active": p_active, "p_bc": p_bc}


def describe_score_subsets() -> dict[str, object]:
    """Return subset sizes for documentation/debugging."""
    return {
        "n_classes": N_CLASSES,
        "silence_next_sizes": [int(len(x)) for x in SILENCE_NEXT],
        "active_shift_sizes": [int(len(x)) for x in ACTIVE_SHIFT],
        "active_hold_sizes": [int(len(x)) for x in ACTIVE_HOLD],
        "backchannel_sizes": [int(len(x)) for x in BACKCHANNEL],
    }
