"""Robust streaming download helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import requests
from tqdm import tqdm


@dataclass
class DownloadResult:
    """Result of a download attempt."""

    ok: bool
    status: str
    path: str
    url: str | None = None
    error: str | None = None
    size_mb: float = 0.0


def file_size_mb(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0


def download_first_available(
    urls: list[str],
    target: str | Path,
    overwrite: bool = False,
    retries: int = 3,
    timeout_sec: int = 60,
    chunk_size_mb: int = 4,
) -> DownloadResult:
    """Download the first URL that returns HTTP 200.

    Existing non-empty files are treated as valid unless `overwrite=True`.
    Downloads go to `.part` first and are then atomically renamed.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return DownloadResult(True, "exists", str(target), size_mb=file_size_mb(target))

    last_error = None
    for url in urls:
        for attempt in range(1, retries + 1):
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                if tmp.exists():
                    tmp.unlink()
                with requests.get(url, stream=True, timeout=timeout_sec) as r:
                    if r.status_code != 200:
                        last_error = f"HTTP {r.status_code}"
                        break
                    total = int(r.headers.get("content-length", 0))
                    with tmp.open("wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=target.name,
                        leave=False,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size_mb * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                tmp.replace(target)
                return DownloadResult(True, "downloaded", str(target), url=url, size_mb=file_size_mb(target))
            except Exception as e:  # network failures vary widely
                last_error = repr(e)
                time.sleep(min(2 * attempt, 10))
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
    return DownloadResult(False, "failed", str(target), url=urls[-1] if urls else None, error=last_error)
