#!/usr/bin/env python

"""Extract VAP-paper-style event prediction rows from VAD and VAP JSON.GZ.

Usage:
    python scripts/30_extract_events.py --config configs/datastore.yaml

Input:
    data/datastore/manifests/stereo_manifest.csv
    outputs/vap/vap_manifest.csv

Output:
    outputs/events/event_predictions.csv
    outputs/events/event_summary.csv
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.events import extract_events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    extract_events(cfg)


if __name__ == "__main__":
    main()
