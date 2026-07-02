#!/usr/bin/env python

"""Run residual-over-time EDA for VAP event predictions.

Usage:
    python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.eda import eda_vap_errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    eda_vap_errors(load_config(args.config))


if __name__ == "__main__":
    main()
