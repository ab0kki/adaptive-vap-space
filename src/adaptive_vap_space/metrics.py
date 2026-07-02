"""Threshold calibration and VAP-style metric tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .config import get
from .reports import write_csv, write_json


def choose_threshold(y_true, y_score) -> tuple[float, float]:
    """Choose threshold maximizing weighted F1 on calibration data."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    candidates = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def metric_dict(df: pd.DataFrame, threshold: float) -> dict:
    """Compute standard binary metrics for a scored event-row DataFrame."""
    if df.empty:
        return {"n_frames": 0, "weighted_f1": np.nan, "macro_f1": np.nan, "positive_f1": np.nan, "accuracy": np.nan}
    y_true = df["label"].astype(int).to_numpy()
    y_score = df["score"].astype(float).to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pcls, rcls, fcls, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {
        "n_frames": int(len(df)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "weighted_f1": float(fw),
        "macro_f1": float(fm),
        "positive_precision": float(pcls[1]),
        "positive_recall": float(rcls[1]),
        "positive_f1": float(fcls[1]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    }


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
            rows.append({"interaction_key": key, "fold": fold_id, "split": "test" if key in test else "calib"})
    return pd.DataFrame(rows)


def evaluate(cfg: dict, protocol: str = "val_5fold") -> None:
    """Evaluate event rows using calibrated thresholds."""
    events_csv = Path(get(cfg, "outputs.events_csv", "outputs/events/event_predictions.csv"))
    events = pd.read_csv(events_csv)
    out_root = Path(get(cfg, "outputs.metrics_root", "outputs/metrics")) / protocol
    out_root.mkdir(parents=True, exist_ok=True)

    if protocol == "val_5fold":
        rows = events[events["project_split"] == "val"].copy()
        if rows.empty:
            print("No val rows found; falling back to all rows for smoke testing.")
            rows = events.copy()
        folds = make_val_folds(rows["interaction_key"].unique().tolist(), k=5, seed=int(get(cfg, "splits.seed", 42)))
        rows = rows.merge(folds, on="interaction_key", how="left")
        calib_split, eval_split = "calib", "test"
    elif protocol == "dev_to_test":
        rows = events[events["project_split"].isin(["val", "test"])].copy()
        rows["fold"] = 0
        rows["split"] = rows["project_split"].map({"val": "calib", "test": "test"})
        calib_split, eval_split = "calib", "test"
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    threshold_rows = []
    metric_rows = []
    eval_rows_out = []

    for (fold, task), g in rows.groupby(["fold", "task"]):
        calib = g[g["split"] == calib_split]
        test = g[g["split"] == eval_split]
        t, calib_f1 = choose_threshold(calib["label"], calib["score"])
        threshold_rows.append({"fold": fold, "task": task, "threshold": t, "n_calib_frames": len(calib), "best_calib_weighted_f1": calib_f1})
        m = metric_dict(test, t)
        m.update({"fold": fold, "task": task})
        metric_rows.append(m)
        tmp = test.copy()
        tmp["threshold"] = t
        tmp["pred"] = (tmp["score"] >= t).astype(int)
        eval_rows_out.append(tmp)

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)
    eval_rows = pd.concat(eval_rows_out, ignore_index=True) if eval_rows_out else pd.DataFrame()

    table = metrics.pivot_table(index="fold", columns="task", values="weighted_f1", aggfunc="first").reset_index()
    if not metrics.empty and "S/H" in set(metrics["task"]):
        sh = metrics[metrics["task"] == "S/H"][["fold", "positive_f1"]].rename(columns={"positive_f1": "S/H_SHIFT"})
        table = table.merge(sh, on="fold", how="left")
    summary = metrics.groupby("task").agg({"weighted_f1": ["mean", "std"], "positive_f1": ["mean", "std"], "n_frames": "sum"}).reset_index() if not metrics.empty else pd.DataFrame()

    write_csv(out_root / "thresholds.csv", thresholds)
    write_csv(out_root / "fold_metrics.csv", metrics)
    write_csv(out_root / "event_predictions_eval.csv", eval_rows)
    write_csv(out_root / "table1_style_by_fold.csv", table)
    write_csv(out_root / "task_metrics.csv", summary)
    write_json(out_root / "summary.json", {"protocol": protocol, "n_event_rows": int(len(events)), "n_eval_rows": int(len(eval_rows))})
    print(f"Wrote metrics to {out_root}")
