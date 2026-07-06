"""EDA routines that save CSVs and PNGs.

These routines are diagnostic. They do not define training or evaluation labels.
"""
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


def _load_canonical_splits(root: Path) -> pd.DataFrame:
    """Load canonical interaction split assignments.

    project_splits.csv is preferred because it is the final stable split file.
    stereo_manifest.csv is the fallback because event extraction uses it.
    """
    split_path = root / "manifests" / "project_splits.csv"
    stereo_path = root / "manifests" / "stereo_manifest.csv"

    if split_path.exists():
        splits = pd.read_csv(split_path)[["interaction_key", "project_split"]].drop_duplicates("interaction_key")
        splits = splits.rename(columns={"project_split": "canonical_project_split"})
        return splits

    if stereo_path.exists():
        splits = pd.read_csv(stereo_path)[["interaction_key", "project_split"]].drop_duplicates("interaction_key")
        splits = splits.rename(columns={"project_split": "canonical_project_split"})
        return splits

    return pd.DataFrame(columns=["interaction_key", "canonical_project_split"])


def eda_dataset(cfg: dict) -> None:
    """Analyze datastore quality and save simple reports/plots.

    The interaction manifest can contain stale split labels from incremental
    building. For split-aware EDA, use the canonical final split from
    project_splits.csv or stereo_manifest.csv.
    """
    root = Path(get(cfg, "dataset.root", "data/datastore"))
    out = Path(get(cfg, "outputs.eda_root", "outputs/eda")) / "dataset"
    out.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_csv(root / "manifests" / "interaction_manifest.csv")
    splits = _load_canonical_splits(root)

    if not splits.empty:
        interactions = interactions.merge(splits, on="interaction_key", how="left")
        if "project_split" in interactions.columns:
            interactions["project_split_raw"] = interactions["project_split"]
        interactions["project_split"] = interactions["canonical_project_split"].combine_first(
            interactions.get("project_split_raw", pd.Series(index=interactions.index, dtype=object))
        )

        mismatch_cols = ["interaction_key", "project_split_raw", "canonical_project_split"]
        if "project_split_raw" in interactions.columns:
            mismatches = interactions[
                interactions["project_split_raw"].notna()
                & interactions["canonical_project_split"].notna()
                & (interactions["project_split_raw"] != interactions["canonical_project_split"])
            ][mismatch_cols]
            write_csv(out / "split_mismatches.csv", mismatches)

    kept = interactions[interactions["keep"] == True].copy() if "keep" in interactions.columns else interactions
    write_csv(out / "interaction_quality.csv", interactions)

    summary = {
        "n_interactions": int(len(interactions)),
        "n_kept": int(len(kept)),
        "n_rejected": int(len(interactions) - len(kept)),
    }
    if "project_split" in kept.columns:
        summary["kept_by_project_split"] = kept["project_split"].value_counts(dropna=False).to_dict()
    write_json(out / "dataset_summary.json", summary)

    for col in ["duration_sec", "speaker_a_fraction", "total_speech_ratio", "overlap_ratio", "silence_ratio"]:
        _hist(kept, col, out / "plots" / f"{col}.png", col)

    if "project_split" in kept.columns:
        split_summary = (
            kept.groupby("project_split", dropna=False)
            .agg(
                n_interactions=("interaction_key", "nunique"),
                mean_duration_sec=("duration_sec", "mean"),
                mean_speaker_a_fraction=("speaker_a_fraction", "mean"),
                mean_total_speech_ratio=("total_speech_ratio", "mean"),
                mean_overlap_ratio=("overlap_ratio", "mean"),
                mean_silence_ratio=("silence_ratio", "mean"),
            )
            .reset_index()
        )
        write_csv(out / "split_quality_summary.csv", split_summary)

    print(f"Wrote dataset EDA to {out}")


def _merge_event_durations(df: pd.DataFrame, events_csv: Path) -> pd.DataFrame:
    summary_path = events_csv.parent / "event_summary.csv"
    if not summary_path.exists() or "duration_s" in df.columns:
        return df
    summary = pd.read_csv(summary_path)[["interaction_key", "duration_s"]].drop_duplicates("interaction_key")
    return df.merge(summary, on="interaction_key", how="left")


def eda_vap_errors(cfg: dict) -> None:
    """Analyze residuals over interaction time."""
    events_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    out = Path(get(cfg, "outputs.eda_root", "outputs/eda")) / "vap_errors"
    out.mkdir(parents=True, exist_ok=True)

    df = add_residual_columns(pd.read_csv(events_csv))
    df = _merge_event_durations(df, events_csv)

    if "duration_s" in df.columns:
        df["relative_time"] = df["time_s"] / df["duration_s"]
        df["relative_time"] = df["relative_time"].clip(lower=0, upper=1)
        df["relative_time_bin"] = pd.cut(df["relative_time"], bins=[i / 10 for i in range(11)], include_lowest=True)

    write_csv(out / "residual_rows.csv", df)

    summary = df.groupby(["interaction_key", "task"]).agg(
        mean_abs_residual=("abs_residual", "mean"),
        mean_squared_residual=("squared_residual", "mean"),
        n_rows=("label", "size"),
    ).reset_index()
    write_csv(out / "interaction_residual_summary.csv", summary)

    if "relative_time_bin" in df.columns:
        rel_summary = (
            df.groupby(["task", "relative_time_bin"], observed=False)
            .agg(
                mean_abs_residual=("abs_residual", "mean"),
                mean_squared_residual=("squared_residual", "mean"),
                n_rows=("label", "size"),
            )
            .reset_index()
        )
        write_csv(out / "relative_time_residual_summary.csv", rel_summary)

    for task, g in df.groupby("task"):
        plt.figure()
        if "relative_time" in g.columns:
            bins = pd.cut(g["relative_time"], bins=20, include_lowest=True)
            xlabel = "relative interaction time bin"
        else:
            bins = pd.cut(g["time_s"], bins=20)
            xlabel = "time bin"
        binned = g.groupby(bins, observed=False)["abs_residual"].mean()
        binned.plot(kind="line")
        plt.title(f"Mean abs residual over time: {task}")
        plt.xlabel(xlabel)
        plt.ylabel("mean abs residual")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        (out / "plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "plots" / f"time_residual_{task.replace('/', '_')}.png")
        plt.close()

    print(f"Wrote VAP error EDA to {out}")


def eda_event_errors(cfg: dict) -> None:
    """Summarize residuals and classification errors by event source/task."""
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

    metrics_root = Path(get(cfg, "outputs.metrics_root", "outputs/metrics"))
    for protocol in ["val_5fold", "dev_to_test"]:
        eval_csv = metrics_root / protocol / "event_predictions_eval.csv"
        if not eval_csv.exists():
            continue

        ev = pd.read_csv(eval_csv)
        if "pred" not in ev.columns:
            continue
        ev["is_error"] = ev["pred"].astype(int) != ev["label"].astype(int)

        cls = (
            ev.groupby(["task", "source_event"], dropna=False)
            .agg(
                n_rows=("label", "size"),
                n_interactions=("interaction_key", "nunique"),
                n_errors=("is_error", "sum"),
                error_rate=("is_error", "mean"),
                positive_rate=("label", "mean"),
                mean_score=("score", "mean"),
            )
            .reset_index()
        )
        write_csv(out / f"classification_error_by_source_{protocol}.csv", cls)

        cm = (
            ev.groupby(["task", "source_event", "label", "pred"], dropna=False)
            .size()
            .reset_index(name="n_rows")
        )
        write_csv(out / f"classification_confusion_by_source_{protocol}.csv", cm)

    print(f"Wrote event error EDA to {out}")
