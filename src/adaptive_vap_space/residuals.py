"""Residual utilities for EDA."""
from __future__ import annotations
import pandas as pd


def add_residual_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed/absolute/squared residual columns to event rows."""
    out = df.copy()
    out["signed_residual"] = out["label"].astype(float) - out["score"].astype(float)
    out["abs_residual"] = out["signed_residual"].abs()
    out["squared_residual"] = out["signed_residual"] ** 2
    return out
