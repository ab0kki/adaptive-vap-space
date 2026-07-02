#!/usr/bin/env python

"""Delete optional large datastore files after validated artifacts exist.

Usage examples:
    python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-raw-audio
    python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-transcripts
    python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-staging
    python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-stereo-wav

Raw mono audio can be deleted after stereo WAVs are built. Stereo WAVs can be deleted after VAP JSON.GZ outputs are complete, but then VAP cannot be rerun without rebuilding/redownloading audio. VAD is kept by default
because event extraction depends on it.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from adaptive_vap_space.config import load_config, get


def delete_glob(folder: Path, pattern: str) -> int:
    count = 0
    for p in folder.glob(pattern):
        if p.is_file():
            p.unlink()
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--delete-raw-audio", action="store_true")
    ap.add_argument("--delete-transcripts", action="store_true")
    ap.add_argument("--delete-staging", action="store_true")
    ap.add_argument("--delete-stereo-wav", action="store_true")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "dataset.root", "data/datastore"))

    actions = []
    if args.delete_raw_audio:
        actions.append(f"delete WAVs under {root / 'raw/audio'}")
    if args.delete_transcripts:
        actions.append(f"delete JSONL transcripts under {root / 'raw/transcript'}")
    if args.delete_staging:
        actions.append(f"delete staging directory {root / get(cfg, 'storage.staging_dirname', '_staging')}")
    if args.delete_stereo_wav:
        actions.append(f"delete stereo WAVs under {root / 'processed/stereo_wav'}")
    if not actions:
        print("No cleanup action requested.")
        return
    print("Planned cleanup:")
    for a in actions:
        print(" -", a)
    if not args.yes:
        answer = input("Proceed? Type YES: ")
        if answer != "YES":
            print("Cancelled.")
            return

    if args.delete_raw_audio:
        print("Deleted raw audio files:", delete_glob(root / "raw" / "audio", "*.wav"))
    if args.delete_transcripts:
        print("Deleted transcript files:", delete_glob(root / "raw" / "transcript", "*.jsonl"))
    if args.delete_staging:
        staging = root / get(cfg, "storage.staging_dirname", "_staging")
        shutil.rmtree(staging, ignore_errors=True)
        print("Deleted staging:", staging)
    if args.delete_stereo_wav:
        print("Deleted stereo WAV files:", delete_glob(root / "processed" / "stereo_wav", "*.wav"))


if __name__ == "__main__":
    main()
