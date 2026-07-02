"""Configuration loading utilities.

All public scripts accept `--config path/to/config.yaml`. This module keeps config
handling small and explicit so the same commands work locally and on Kaggle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a nested config value using dot notation.

    Example
    -------
    `get(cfg, "dataset.root")` reads `cfg["dataset"]["root"]`.
    """
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require(cfg: dict[str, Any], dotted: str) -> Any:
    """Read a required nested config value and raise a helpful error if missing."""
    value = get(cfg, dotted, None)
    if value is None:
        raise KeyError(f"Missing required config value: {dotted}")
    return value
