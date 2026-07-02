#!/usr/bin/env python

"""Evaluate VAP event rows with calibrated thresholds.

Recommended workflow:
    # Development/audit only: estimate threshold sensitivity inside the val split.
    python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold

    # Final baseline/model evaluation: fit thresholds on val and evaluate once on test.
    python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test

Protocols:
    val_5fold:
        Creates folds by interaction_key only inside the val split. Thresholds are
        fit on calibration folds and evaluated on held-out val folds. Use this
        for development diagnostics, not as the final reported test score.

    dev_to_test:
        Uses val as the calibration/dev split and test as the held-out evaluation
        split. This is the default because it matches the usual ML discipline:
        choose thresholds/hyperparameters on dev, then evaluate once on test.
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.metrics import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", choices=["val_5fold", "dev_to_test"], default="dev_to_test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, protocol=args.protocol)


if __name__ == "__main__":
    main()
