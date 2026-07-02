"""Speaker and interaction validation logic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import audio_info
from .vad import read_vad_segments, speech_seconds, oob_fraction, interaction_vad_stats


@dataclass
class SpeakerValidation:
    """Validation result for one participant stream."""

    ok: bool
    reasons: list[str]
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    vad_segments: int
    vad_speech_sec: float
    vad_ratio: float
    vad_bad_lines: int
    transcript_rows: int


def count_jsonl_rows(path: str | Path) -> int:
    """Count non-empty rows in a JSONL/text file; missing files count as zero."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_speaker(audio_path: str | Path, vad_path: str | Path, transcript_path: str | Path | None, cfg: dict) -> SpeakerValidation:
    """Validate one speaker's audio/VAD/transcript files."""
    filters = cfg.get("filters", {})
    modalities = cfg.get("modalities", {})
    reasons: list[str] = []

    info = audio_info(audio_path)
    segments, bad = read_vad_segments(vad_path)
    vad_speech = speech_seconds(segments)
    duration = info.get("duration_sec")
    vad_ratio = vad_speech / duration if duration else 0.0
    transcript_rows = count_jsonl_rows(transcript_path) if transcript_path else 0

    if not info.get("audio_readable"):
        reasons.append("audio_error")
    if not Path(vad_path).exists() or Path(vad_path).stat().st_size == 0:
        reasons.append("missing_vad")
    if modalities.get("require_transcript", False) and transcript_rows == 0:
        reasons.append("missing_transcript")
    if duration is not None:
        if duration < filters.get("min_duration_sec", 0):
            reasons.append("too_short")
        if duration > filters.get("max_duration_sec", float("inf")):
            reasons.append("too_long")
        frac = oob_fraction(segments, duration, tolerance_sec=1.0)
        if frac > filters.get("max_vad_timestamp_oob_fraction", 0.05):
            reasons.append("vad_timestamp_oob")
    if len(segments) < filters.get("min_vad_segments", 0):
        reasons.append("too_few_vad_segments")
    if vad_ratio < filters.get("min_speaker_vad_ratio", 0.0):
        reasons.append("too_little_speech")
    if bad > 0:
        reasons.append("vad_bad_lines")

    return SpeakerValidation(
        ok=len([r for r in reasons if r != "vad_bad_lines"]) == 0,
        reasons=reasons,
        duration_sec=duration,
        sample_rate=info.get("sample_rate"),
        channels=info.get("channels"),
        vad_segments=len(segments),
        vad_speech_sec=vad_speech,
        vad_ratio=vad_ratio,
        vad_bad_lines=bad,
        transcript_rows=transcript_rows,
    )


def validate_interaction(vad_a, vad_b, speaker_a: SpeakerValidation, speaker_b: SpeakerValidation, cfg: dict) -> tuple[bool, list[str], dict]:
    """Validate a dyadic interaction and return `(ok, reasons, stats)`."""
    filters = cfg.get("filters", {})
    reasons: list[str] = []
    if not speaker_a.ok:
        reasons.extend([f"speaker_a:{r}" for r in speaker_a.reasons])
    if not speaker_b.ok:
        reasons.extend([f"speaker_b:{r}" for r in speaker_b.reasons])

    da = speaker_a.duration_sec
    db = speaker_b.duration_sec
    duration = min(da, db) if da and db else None
    if da is None or db is None:
        reasons.append("missing_audio_duration")
    elif abs(da - db) > filters.get("max_audio_duration_diff_sec", 2.0):
        reasons.append("duration_mismatch")
    if duration is not None and duration < filters.get("min_duration_sec", 0):
        reasons.append("too_short")

    stats = {"duration_sec": duration or 0.0}
    if duration:
        stats.update(interaction_vad_stats(vad_a, vad_b, duration))
        a_frac = stats["speaker_a_fraction"]
        b_frac = stats["speaker_b_fraction"]
        if a_frac < filters.get("min_speaker_fraction", 0.0) or a_frac > filters.get("max_speaker_fraction", 1.0):
            reasons.append("speaker_imbalance")
        if b_frac < filters.get("min_speaker_fraction", 0.0) or b_frac > filters.get("max_speaker_fraction", 1.0):
            reasons.append("speaker_imbalance")
        if stats["total_speech_ratio"] < filters.get("min_total_speech_ratio", 0.0):
            reasons.append("too_little_speech")
        if stats["overlap_ratio"] > filters.get("max_overlap_ratio", 1.0):
            reasons.append("too_much_overlap")
        if stats["silence_ratio"] > filters.get("max_silence_ratio", 1.0):
            reasons.append("too_much_silence")

    # Preserve order while removing duplicates.
    deduped = list(dict.fromkeys(reasons))
    return len(deduped) == 0, deduped, stats
