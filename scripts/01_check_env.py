#!/usr/bin/env python

"""Check local/Kaggle environment for the Adaptive VAP Space pipeline.

Usage:
    python scripts/01_check_env.py --config configs/local_debug.yaml

This script does not download data. It verifies that Python dependencies are
importable and that configured external repository paths exist when needed.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from adaptive_vap_space.config import load_config, get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print("Config:", args.config)
    for module in ["numpy", "pandas", "soundfile", "librosa", "requests", "sklearn"]:
        __import__(module)
        print("import ok:", module)

    seamless_repo = Path(get(cfg, "external.seamless_repo", "external/seamless_interaction"))
    vap_repo = Path(get(cfg, "external.vap_repo", "external/VoiceActivityProjection"))
    print("seamless_repo:", seamless_repo, "exists=", seamless_repo.exists())
    print("vap_repo:", vap_repo, "exists=", vap_repo.exists())
    print("dataset_root:", get(cfg, "dataset.root"))
    print("OK")


if __name__ == "__main__":
    main()
