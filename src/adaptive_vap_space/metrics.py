"""Threshold calibration and VAP-style metric tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .config import get
from .reports import write_csv, write_json


DEFAULT_OBJECTIVES = ["weighted_f1", "macro_f1", "positive_f1", "balanced_accuracy"]


def threshold_candidates(y_score) -> np.ndarray:
    """Create deterministic candidate thresholds using the original quantile grid."""
    y_score = np.asarray(y_score, dtype=float)
    if len(y_score) == 0:
        return np.asarray([0.5], dtype=float)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    if len(candidates) == 0:
        candidates = np.asarray([0.5], dtype=float)
    return candidates.astype(float)


def choose_threshold(y_true, y_score, objective: str = "weighted_f1") -> tuple[float, float]:
    """Choose threshold maximizing an objective on calibration data."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    best_t, best_value = 0.5, -1.0
    df = pd.DataFrame({"label": y_true, "score": y_score})
    for t in threshold_candidates(y_score):
        m = metric_dict(df, float(t))
        value = float(m.get(objective, np.nan))
        if np.isfinite(value) and value > best_value:
            best_t, best_value = float(t), value
    return best_t, best_value


def metric_dict(df: pd.DataFrame, threshold: float) -> dict:
    """Compute standard binary metrics for a scored event-row DataFrame."""
    if df.empty:
        return {
            "n_frames": 0,
            "n_pos": 0,
            "n_neg": 0,
            "threshold": float(threshold),
            "weighted_f1": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "positive_precision": np.nan,
            "positive_recall": np.nan,
            "positive_f1": np.nan,
            "accuracy": np.nan,
            "precision_weighted": np.nan,
            "recall_weighted": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_true = df["label"].astype(int).to_numpy()
    y_score = df["score"].astype(float).to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    _, _, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    pcls, rcls, fcls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    balanced_accuracy = float(np.nanmean([tnr, tpr]))

    return {
        "n_frames": int(len(df)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "weighted_f1": float(fw),
        "macro_f1": float(fm),
        "balanced_accuracy": balanced_accuracy,
        "positive_precision": float(pcls[1]),
        "positive_recall": float(rcls[1]),
        "positive_f1": float(fcls[1]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_sweep_rows(calib: pd.DataFrame, test: pd.DataFrame, fold, task) -> list[dict]:
    """Return calibration and test metrics over a shared threshold grid."""
    rows = []
    for t in threshold_candidates(calib["score"]):
        for split_name, split_df in [("calib", calib), ("test", test)]:
            m = metric_dict(split_df, float(t))
            m.update({"fold": fold, "task": task, "split": split_name})
            rows.append(m)
    return rows


def objective_threshold_rows(calib: pd.DataFrame, test: pd.DataFrame, fold, task) -> list[dict]:
    """Evaluate test metrics using thresholds selected by several calibration objectives."""
    rows = []
    for objective in DEFAULT_OBJECTIVES:
        t, calib_value = choose_threshold(calib["label"], calib["score"], objective=objective)
        m = metric_dict(test, t)
        m.update({
            "fold": fold,
            "task": task,
            "threshold_objective": objective,
            "calib_objective_value": calib_value,
        })
        rows.append(m)
    return rows


def make_val_folds(keys: list[str], k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create k-fold calibration/eval rows by interaction key."""
    keys = np.array(sorted(set(keys)))
    if len(keys) < k:
        k = max(1, len(keys))

    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    folds = np.array_split(keys, k)

    rows = []
    for fold_id, test_keys in enumerate(folds):
        test = set(test_keys.tolist())
        for key in keys:
            rows.append({
                "interaction_key": key,
                "fold": fold_id,
                "split": "test" if key in test else "calib",
            })
    return pd.DataFrame(rows)


def build_protocol_rows(events: pd.DataFrame, cfg: dict, protocol: str) -> tuple[pd.DataFrame, str, str]:
    """Attach calibration/evaluation labels for the requested protocol.

    All protocols split by interaction_key, never by frame/event row.
    """
    if protocol == "val_5fold":
        rows = events[events["project_split"] == "val"].copy()
        if rows.empty:
            raise ValueError("No val rows found for val_5fold. Build project splits before evaluation.")
        folds = make_val_folds(
            rows["interaction_key"].unique().tolist(),
            k=5,
            seed=int(get(cfg, "splits.seed", 42)),
        )
        rows = rows.merge(folds, on="interaction_key", how="left")
        return rows, "calib", "test"

    if protocol == "dev_to_test":
        rows = events[events["project_split"].isin(["val", "test"])].copy()
        rows["fold"] = 0
        rows["split"] = rows["project_split"].map({"val": "calib", "test": "test"})

        n_calib = rows.loc[rows["split"] == "calib", "interaction_key"].nunique()
        n_test = rows.loc[rows["split"] == "test", "interaction_key"].nunique()
        if n_calib == 0 or n_test == 0:
            raise ValueError(
                "dev_to_test requires both val and test event rows. "
                f"Found val/calib interactions={n_calib}, test interactions={n_test}."
            )

        return rows, "calib", "test"

    raise ValueError(f"Unknown protocol: {protocol}")


def protocol_split_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize how many rows/interactions each protocol split uses."""
    if rows.empty:
        return pd.DataFrame()

    return (
        rows.groupby(["task", "split"], dropna=False)
        .agg(
            n_rows=("label", "size"),
            n_interactions=("interaction_key", "nunique"),
            n_pos=("label", lambda s: int((s.astype(int) == 1).sum())),
            n_neg=("label", lambda s: int((s.astype(int) == 0).sum())),
        )
        .reset_index()
    )


def event_level_metric_rows(eval_rows: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Compute event-level metrics using already-calibrated frame-level thresholds."""
    if eval_rows.empty:
        return pd.DataFrame()

    group_cols = [
        "fold",
        "task",
        "interaction_key",
        "project_split",
        "source_event",
        "event_id",
        "speaker",
        "positive_class",
    ]
    group_cols = [c for c in group_cols if c in eval_rows.columns]

    ev = (
        eval_rows.groupby(group_cols, dropna=False)
        .agg(
            label=("label", "first"),
            score=("score", "mean"),
            n_frames=("score", "size"),
            time_s=("time_s", "mean"),
        )
        .reset_index()
    )

    rows = []
    for _, th in thresholds.iterrows():
        fold = th["fold"]
        task = th["task"]
        threshold = float(th["threshold"])
        g = ev[(ev["fold"] == fold) & (ev["task"] == task)].copy()
        m = metric_dict(g, threshold)
        m.update({"fold": fold, "task": task, "unit": "event_mean_score"})
        rows.append(m)

    return pd.DataFrame(rows)


def evaluate(cfg: dict, protocol: str = "dev_to_test") -> None:
    """Evaluate event rows using calibrated thresholds."""
    events_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    events = pd.read_csv(events_csv)

    out_root = Path(get(cfg, "outputs.metrics_root", "outputs/metrics")) / protocol
    out_root.mkdir(parents=True, exist_ok=True)

    rows, calib_split, eval_split = build_protocol_rows(events, cfg, protocol)

    threshold_rows = []
    metric_rows = []
    eval_rows_out = []
    sweep_rows = []
    objective_rows = []

    for (fold, task), g in rows.groupby(["fold", "task"]):
        calib = g[g["split"] == calib_split]
        test = g[g["split"] == eval_split]

        t, calib_f1 = choose_threshold(calib["label"], calib["score"], objective="weighted_f1")
        threshold_rows.append({
            "fold": fold,
            "task": task,
            "threshold": t,
            "threshold_objective": "weighted_f1",
            "n_calib_frames": len(calib),
            "best_calib_weighted_f1": calib_f1,
        })

        m = metric_dict(test, t)
        m.update({"fold": fold, "task": task})
        metric_rows.append(m)

        tmp = test.copy()
        tmp["threshold"] = t
        tmp["pred"] = (tmp["score"] >= t).astype(int)
        eval_rows_out.append(tmp)

        sweep_rows.extend(threshold_sweep_rows(calib, test, fold, task))
        objective_rows.extend(objective_threshold_rows(calib, test, fold, task))

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)
    eval_rows = pd.concat(eval_rows_out, ignore_index=True) if eval_rows_out else pd.DataFrame()
    threshold_sweep = pd.DataFrame(sweep_rows)
    objective_metrics = pd.DataFrame(objective_rows)

    table = metrics.pivot_table(
        index="fold",
        columns="task",
        values="weighted_f1",
        aggfunc="first",
    ).reset_index()

    if not metrics.empty and "S/H" in set(metrics["task"]):
        sh = metrics[metrics["task"] == "S/H"][["fold", "positive_f1"]].rename(
            columns={"positive_f1": "S/H_SHIFT"}
        )
        table = table.merge(sh, on="fold", how="left")

    summary = (
        metrics.groupby("task")
        .agg({
            "weighted_f1": ["mean", "std"],
            "macro_f1": ["mean", "std"],
            "balanced_accuracy": ["mean", "std"],
            "positive_precision": ["mean", "std"],
            "positive_recall": ["mean", "std"],
            "positive_f1": ["mean", "std"],
            "n_frames": "sum",
            "n_pos": "sum",
            "n_neg": "sum",
        })
        .reset_index()
        if not metrics.empty
        else pd.DataFrame()
    )

    split_counts = protocol_split_counts(rows)
    event_metrics = event_level_metric_rows(eval_rows, thresholds)

    write_csv(out_root / "protocol_split_counts.csv", split_counts)
    write_csv(out_root / "thresholds.csv", thresholds)
    write_csv(out_root / "fold_metrics.csv", metrics)
    write_csv(out_root / "event_level_metrics.csv", event_metrics)
    write_csv(out_root / "threshold_sweep.csv", threshold_sweep)
    write_csv(out_root / "objective_threshold_metrics.csv", objective_metrics)
    write_csv(out_root / "event_predictions_eval.csv", eval_rows)
    write_csv(out_root / "table1_style_by_fold.csv", table)
    write_csv(out_root / "task_metrics.csv", summary)
    write_json(out_root / "summary.json", {
        "protocol": protocol,
        "n_event_rows": int(len(events)),
        "n_protocol_rows": int(len(rows)),
        "n_eval_rows": int(len(eval_rows)),
        "threshold_objectives": DEFAULT_OBJECTIVES,
    })

    print(f"Wrote metrics to {out_root}")
