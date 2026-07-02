"""EDA routines that save CSVs and PNGs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .config import get
from .residuals import add_residual_columns
from .reports import write_csv, write_json


def _hist(df: pd.DataFrame, col: str, out_path: Path, title: str) -> None:
    if col not in df.columns or df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    df[col].dropna().astype(float).hist(bins=30)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def eda_dataset(cfg: dict) -> None:
    """Analyze datastore quality and save simple reports/plots."""
    root = Path(get(cfg, "dataset.root", "data/datastore"))
    out = Path(get(cfg, "outputs.eda_root", "outputs/eda")) / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_csv(root / "manifests" / "interaction_manifest.csv")
    kept = interactions[interactions["keep"] == True].copy() if "keep" in interactions.columns else interactions
    write_csv(out / "interaction_quality.csv", interactions)
    summary = {
        "n_interactions": int(len(interactions)),
        "n_kept": int(len(kept)),
        "n_rejected": int(len(interactions) - len(kept)),
    }
    write_json(out / "dataset_summary.json", summary)
    for col in ["duration_sec", "speaker_a_fraction", "total_speech_ratio", "overlap_ratio", "silence_ratio"]:
        _hist(kept, col, out / "plots" / f"{col}.png", col)
    print(f"Wrote dataset EDA to {out}")


def eda_vap_errors(cfg: dict) -> None:
    """Analyze residuals over interaction time."""
    events_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    out = Path(get(cfg, "outputs.eda_root", "outputs/eda")) / "vap_errors"
    out.mkdir(parents=True, exist_ok=True)
    df = add_residual_columns(pd.read_csv(events_csv))
    write_csv(out / "residual_rows.csv", df)
    summary = df.groupby(["interaction_key", "task"]).agg(
        mean_abs_residual=("abs_residual", "mean"),
        mean_squared_residual=("squared_residual", "mean"),
        n_rows=("label", "size"),
    ).reset_index()
    write_csv(out / "interaction_residual_summary.csv", summary)
    for task, g in df.groupby("task"):
        plt.figure()
        bins = pd.cut(g["time_s"], bins=20)
        binned = g.groupby(bins, observed=False)["abs_residual"].mean()
        binned.plot(kind="line")
        plt.title(f"Mean abs residual over time: {task}")
        plt.xlabel("time bin")
        plt.ylabel("mean abs residual")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        (out / "plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "plots" / f"time_residual_{task.replace('/', '_')}.png")
        plt.close()
    print(f"Wrote VAP error EDA to {out}")


def eda_event_errors(cfg: dict) -> None:
    """Summarize residuals by event source/task."""
    events_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    out = Path(get(cfg, "outputs.eda_root", "outputs/eda")) / "event_errors"
    out.mkdir(parents=True, exist_ok=True)
    df = add_residual_columns(pd.read_csv(events_csv))
    summary = df.groupby(["task", "source_event"]).agg(
        mean_abs_residual=("abs_residual", "mean"),
        mean_squared_residual=("squared_residual", "mean"),
        n_rows=("label", "size"),
    ).reset_index()
    write_csv(out / "event_window_summary.csv", summary)
    print(f"Wrote event error EDA to {out}")
