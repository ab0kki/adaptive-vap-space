"""Voice activity JSONL utilities."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

Segment = tuple[float, float]


def read_vad_segments(path: str | Path) -> tuple[list[Segment], int]:
    """Read Seamless VAD JSONL rows into `(start, end)` segments.

    Returns a tuple `(segments, bad_line_count)`. Invalid rows are counted rather
    than crashing the full dataset build.
    """
    path = Path(path)
    segments: list[Segment] = []
    bad = 0
    if not path.exists() or path.stat().st_size == 0:
        return segments, bad
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                s = float(obj.get("start"))
                e = float(obj.get("end"))
            except Exception:
                bad += 1
                continue
            if e > s:
                segments.append((s, e))
            else:
                bad += 1
    return segments, bad


def speech_seconds(segments: list[Segment]) -> float:
    """Return total speech duration from VAD segments."""
    return float(sum(max(0.0, e - s) for s, e in segments))


def oob_fraction(segments: list[Segment], duration_sec: float, tolerance_sec: float = 0.0) -> float:
    """Fraction of VAD segments outside the audio duration."""
    if not segments:
        return 0.0
    bad = [(s, e) for s, e in segments if s < -tolerance_sec or e > duration_sec + tolerance_sec]
    return len(bad) / max(1, len(segments))


def vad_segments_to_frames(
    vad_a: list[Segment],
    vad_b: list[Segment],
    n_frames: int,
    frame_hz: int = 50,
) -> np.ndarray:
    """Convert two speakers' VAD segments to a boolean frame array `(T, 2)`."""
    va = np.zeros((n_frames, 2), dtype=bool)
    for speaker, segments in enumerate([vad_a, vad_b]):
        for s, e in segments:
            fs = max(0, int(round(s * frame_hz)))
            fe = min(n_frames, int(round(e * frame_hz)))
            if fe > fs:
                va[fs:fe, speaker] = True
    return va


def frame_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous true regions as half-open frame intervals."""
    mask = np.asarray(mask).astype(bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def no_activity(va: np.ndarray, start: int, end: int) -> bool:
    """Return True if neither speaker is active in `[start, end)`."""
    start = max(0, start)
    end = min(len(va), end)
    return end <= start or not va[start:end].any()


def single_speaker_region(va: np.ndarray, start: int, end: int, speaker: int) -> bool:
    """Return True if exactly the requested speaker is active throughout most frames.

    This mirrors the old working script's strict check: target speaker active and
    the other speaker silent for the full region.
    """
    start = max(0, start)
    end = min(len(va), end)
    if end <= start:
        return False
    other = 1 - speaker
    return bool(va[start:end, speaker].all() and not va[start:end, other].any())


def interaction_vad_stats(vad_a: list[Segment], vad_b: list[Segment], duration_sec: float, frame_hz: int = 50) -> dict[str, float]:
    """Compute speech, overlap, and silence ratios for an interaction."""
    n_frames = max(1, int(round(duration_sec * frame_hz)))
    va = vad_segments_to_frames(vad_a, vad_b, n_frames, frame_hz)
    a = va[:, 0]
    b = va[:, 1]
    active_any = a | b
    overlap = a & b
    a_ratio = float(a.mean())
    b_ratio = float(b.mean())
    total_speech_ratio = float(active_any.mean())
    overlap_ratio = float(overlap.mean())
    silence_ratio = float((~active_any).mean())
    denom = max(a_ratio + b_ratio, 1e-8)
    return {
        "speaker_a_fraction": a_ratio / denom,
        "speaker_b_fraction": b_ratio / denom,
        "vad_ratio_a": a_ratio,
        "vad_ratio_b": b_ratio,
        "total_speech_ratio": total_speech_ratio,
        "overlap_ratio": overlap_ratio,
        "silence_ratio": silence_ratio,
    }
