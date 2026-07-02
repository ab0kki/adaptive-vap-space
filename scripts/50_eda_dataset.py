#!/usr/bin/env python

"""Run dataset-quality EDA and save CSV/PNG outputs.

Usage:
    python scripts/50_eda_dataset.py --config configs/datastore.yaml
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.eda import eda_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    eda_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
