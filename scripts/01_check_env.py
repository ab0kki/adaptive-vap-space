#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from adaptive_vap_space.config import load_config, get

def check_import(name: str) -> None:
    __import__(name)
    print(f"import ok: {name}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    print("Config:", args.config)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    for module in ["numpy", "pandas", "soundfile", "librosa", "requests", "sklearn", "torch", "torchaudio", "torchcodec"]:
        check_import(module)
    seamless_repo = Path(get(cfg, "external.seamless_repo", "external/seamless_interaction"))
    vap_repo = Path(get(cfg, "external.vap_repo", "external/VoiceActivityProjection"))
    checkpoint = Path(get(cfg, "external.vap_checkpoint"))
    print("seamless_repo:", seamless_repo, "exists=", seamless_repo.exists())
    print("seamless_filelist:", seamless_repo / "assets/filelist.csv", "exists=", (seamless_repo / "assets/filelist.csv").exists())
    print("vap_repo:", vap_repo, "exists=", vap_repo.exists())
    print("vap_run_py:", vap_repo / "run.py", "exists=", (vap_repo / "run.py").exists())
    print("vap_checkpoint:", checkpoint, "exists=", checkpoint.exists())
    print("dataset_root:", get(cfg, "dataset.root"))
    missing = []
    if not (seamless_repo / "assets/filelist.csv").exists():
        missing.append(str(seamless_repo / "assets/filelist.csv"))
    if not (vap_repo / "run.py").exists():
        missing.append(str(vap_repo / "run.py"))
    if not checkpoint.exists():
        missing.append(str(checkpoint))
    if missing:
        raise SystemExit("Missing required paths:\n- " + "\n- ".join(missing))
    print("OK")

if __name__ == "__main__":
    main()
