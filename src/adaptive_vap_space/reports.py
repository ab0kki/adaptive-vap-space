"""Small report-writing helpers."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd


def write_json(path: str | Path, data: dict) -> None:
    """Write a JSON file with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_csv(path: str | Path, rows: list[dict] | pd.DataFrame) -> None:
    """Write rows/DataFrame to CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)


def utc_now() -> str:
    """Return ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
