"""Stereo manifest helpers."""
from __future__ import annotations

from pathlib import Path
from .audio import write_stereo_wav
from .paths import relpath


def build_stereo_row(
    dataset_root: str | Path,
    interaction_key: str,
    project_split: str,
    audio_a: str | Path,
    audio_b: str | Path,
    vad_a: str | Path,
    vad_b: str | Path,
    transcript_a: str | Path | None,
    transcript_b: str | Path | None,
    speaker_a: str,
    speaker_b: str,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> dict:
    """Build a stereo WAV and return a manifest row."""
    dataset_root = Path(dataset_root)
    out_path = dataset_root / "processed" / "stereo_wav" / f"{interaction_key}.wav"
    info = write_stereo_wav(audio_a, audio_b, out_path, sample_rate=sample_rate, overwrite=overwrite)
    return {
        "interaction_key": interaction_key,
        "project_split": project_split,
        "stereo_relpath": relpath(out_path, dataset_root),
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "duration_sec": info["duration_sec"],
        "sample_rate": sample_rate,
        "audio_a_relpath": relpath(audio_a, dataset_root),
        "audio_b_relpath": relpath(audio_b, dataset_root),
        "vad_a_relpath": relpath(vad_a, dataset_root),
        "vad_b_relpath": relpath(vad_b, dataset_root),
        "transcript_a_relpath": relpath(transcript_a, dataset_root) if transcript_a else "",
        "transcript_b_relpath": relpath(transcript_b, dataset_root) if transcript_b else "",
        "status": info["status"],
    }
