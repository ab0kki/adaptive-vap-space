#!/usr/bin/env python

"""Run public pretrained VAP on full stereo WAVs.

Usage:
    python scripts/20_run_vap.py --config configs/datastore.yaml

Input:
    data/datastore/manifests/stereo_manifest.csv

Output:
    outputs/vap/predictions/*.json.gz
    outputs/vap/vap_manifest.csv
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.vap_runner import run_vap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_vap(cfg)


if __name__ == "__main__":
    main()
