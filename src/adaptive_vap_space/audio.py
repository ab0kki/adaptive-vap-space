"""Audio inspection and stereo WAV creation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


  # noqa: D401
def audio_info(path: str | Path) -> dict:
    """Return duration/sample-rate/channel info, marking unreadable files."""
    path = Path(path)
    try:
        info = sf.info(str(path))
        return {
            "audio_readable": True,
            "duration_sec": float(info.frames / info.samplerate),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
        }
    except Exception as e:
        return {
            "audio_readable": False,
            "duration_sec": None,
            "sample_rate": None,
            "channels": None,
            "audio_error": repr(e),
        }


def write_stereo_wav(
    audio_a: str | Path,
    audio_b: str | Path,
    out_path: str | Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> dict:
    """Create a stereo WAV with channel 0=speaker A and channel 1=speaker B.

    Both input signals are loaded as mono and resampled. The output is truncated
    to the common duration to preserve alignment.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        info = audio_info(out_path)
        return {"status": "exists", "stereo_path": str(out_path), **info}

    y_a, _ = librosa.load(audio_a, sr=sample_rate, mono=True)
    y_b, _ = librosa.load(audio_b, sr=sample_rate, mono=True)
    n = min(len(y_a), len(y_b))
    if n <= 0:
        raise ValueError("Cannot create stereo WAV from empty audio.")
    stereo = np.stack([y_a[:n], y_b[:n]], axis=1)
    tmp = out_path.with_name(out_path.name + ".part.wav")
    sf.write(tmp, stereo, sample_rate, format="WAV")
    tmp.replace(out_path)
    return {
        "status": "written",
        "stereo_path": str(out_path),
        "audio_readable": True,
        "duration_sec": n / sample_rate,
        "sample_rate": sample_rate,
        "channels": 2,
    }
