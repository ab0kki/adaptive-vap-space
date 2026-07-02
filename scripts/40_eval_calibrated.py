#!/usr/bin/env python

"""Evaluate VAP event rows with calibrated thresholds.

Usage:
    python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
    python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test

The `val_5fold` protocol creates folds by interaction_key only inside the val
split. Thresholds are fit on calibration folds and evaluated on held-out folds.
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.metrics import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", choices=["val_5fold", "dev_to_test"], default="val_5fold")
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, protocol=args.protocol)


if __name__ == "__main__":
    main()
