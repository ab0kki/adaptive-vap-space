"""Path helpers for portable datastore manifests."""
from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relpath(path: str | Path, root: str | Path) -> str:
    """Return `path` relative to `root` when possible.

    Manifest files should prefer relative paths so a dataset can move from
    `/kaggle/working` to `/kaggle/input` without rewriting every row.
    """
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_under(root: str | Path, maybe_relative: str | Path) -> Path:
    """Resolve a path that may be relative to `root` or already absolute."""
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return Path(root) / p


def datastore_dirs(dataset_root: str | Path) -> dict[str, Path]:
    """Return the standard datastore directory map and create directories."""
    root = Path(dataset_root)
    dirs = {
        "root": root,
        "raw_audio": root / "raw" / "audio",
        "raw_vad": root / "raw" / "vad",
        "raw_transcript": root / "raw" / "transcript",
        "processed_stereo": root / "processed" / "stereo_wav",
        "manifests": root / "manifests",
        "reports": root / "reports",
    }
    for key, path in dirs.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return dirs
