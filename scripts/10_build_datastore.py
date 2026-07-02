#!/usr/bin/env python

"""Build a storage-safe Seamless audio/VAD/transcript datastore.

Usage:
    python scripts/10_build_datastore.py --config configs/datastore.yaml

This stage reads the Seamless filelist, downloads one candidate interaction at a
time into staging, validates it, keeps or rejects it, builds stereo WAVs for kept
interactions, and writes manifests/reports. It does not run VAP.
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.datastore import build_datastore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_datastore(cfg)


if __name__ == "__main__":
    main()
