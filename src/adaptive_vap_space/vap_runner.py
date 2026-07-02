"""Public VAP `run.py` wrapper.

This module does not reimplement VAP. It runs the public
VoiceActivityProjection `run.py` on each full stereo WAV and records a manifest.

The only environment-specific behavior is that upstream VAP may be run with a
separate Python executable configured by `external.vap_python`, so the main repo
can stay lightweight while VAP uses a PyTorch environment.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import os
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd

from .config import get
from .paths import ensure_dir, resolve_under, relpath
from .reports import write_csv
from .vap_outputs import load_json_maybe_gz, validate_vap_json


def configured_vap_python(cfg: dict[str, Any]) -> Path:
    """Return the Python executable used to run upstream VAP."""
    raw = get(cfg, "external.vap_python", None) or get(cfg, "vap.python", None)
    return Path(raw).expanduser() if raw else Path(sys.executable)


def preflight_vap_environment(vap_python: Path, vap_repo: Path, checkpoint: Path) -> None:
    """Fail fast if upstream VAP cannot run in the configured Python env."""
    problems = []
    if not vap_python.exists():
        problems.append(f"configured VAP Python does not exist: {vap_python}")
    if not (vap_repo / "run.py").exists():
        problems.append(f"missing public VAP run.py: {vap_repo / 'run.py'}")
    if not checkpoint.exists():
        problems.append(f"missing VAP checkpoint: {checkpoint}")
    if problems:
        raise RuntimeError("VAP preflight failed:\n- " + "\n- ".join(problems))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(vap_repo.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    checks = [
        ("torch import", [str(vap_python), "-c", "import torch; print(torch.__version__)"]),
        ("vap import", [str(vap_python), "-c", "import vap; print('vap import ok')"]),
    ]

    failures = []
    for label, cmd in checks:
        proc = subprocess.run(
            cmd,
            cwd=str(vap_repo),
            env=env,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            failures.append(
                f"{label} failed\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout.strip()}\n"
                f"stderr:\n{proc.stderr.strip()}"
            )

    if failures:
        raise RuntimeError(
            "VAP preflight failed before running any interactions.\n\n"
            + "\n\n".join(failures)
        )


def atomic_gzip_json(json_path: Path, gz_path: Path) -> Path:
    """Compress JSON to JSON.GZ using a temporary file then atomic rename."""
    tmp_path = gz_path.with_suffix(gz_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    with json_path.open("rb") as f_in, gzip.open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    tmp_path.replace(gz_path)
    json_path.unlink(missing_ok=True)
    return gz_path


def read_meta_if_valid(path: Path) -> tuple[bool, str, dict[str, Any]]:
    """Return validation state for an existing VAP JSON/JSON.GZ output."""
    if not path.exists():
        return False, "missing", {}
    return validate_vap_json(path)


def run_vap(cfg: dict[str, Any]) -> None:
    """Run public pretrained VAP on every stereo WAV in the datastore manifest."""
    dataset_root = Path(get(cfg, "dataset.root", "data/datastore"))
    output_root = Path(get(cfg, "vap.output_root", "outputs/vap"))
    pred_dir = ensure_dir(output_root / "predictions")

    stereo_manifest_path = dataset_root / "manifests" / "stereo_manifest.csv"
    if not stereo_manifest_path.exists():
        raise FileNotFoundError(f"Missing stereo manifest: {stereo_manifest_path}")

    stereo_manifest = pd.read_csv(stereo_manifest_path)

    vap_repo = Path(get(cfg, "external.vap_repo"))
    checkpoint = Path(get(cfg, "external.vap_checkpoint"))
    vap_python = configured_vap_python(cfg)

    # New behavior: fail once before looping if the VAP Python is wrong.
    preflight_vap_environment(vap_python, vap_repo, checkpoint)

    rows = []
    overwrite = bool(get(cfg, "vap.overwrite", False))
    use_gzip = bool(get(cfg, "vap.gzip", True))
    frame_hz = float(get(cfg, "vap.frame_hz", 50))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(vap_repo.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    for _, row in stereo_manifest.iterrows():
        key = row["interaction_key"]
        stereo_path = resolve_under(dataset_root, row["stereo_relpath"])

        out_json = pred_dir / f"{key}.json"
        out_gz = pred_dir / f"{key}.json.gz"
        final_path = out_gz if use_gzip else out_json

        ok, msg, meta = read_meta_if_valid(final_path) if not overwrite else (False, "overwrite requested", {})
        status = "exists" if ok else "pending"
        error = ""

        if not ok:
            # Remove only failed/corrupt outputs for this interaction.
            out_json.unlink(missing_ok=True)
            out_gz.unlink(missing_ok=True)
            out_gz.with_suffix(out_gz.suffix + ".part").unlink(missing_ok=True)

            cmd = [
                str(vap_python),
                str(vap_repo / "run.py"),
                "-a", str(stereo_path),
                "-sd", str(checkpoint),
                "-f", str(out_json),
            ]

            try:
                subprocess.run(cmd, check=True, env=env)
                final_path = atomic_gzip_json(out_json, out_gz) if use_gzip else out_json
                ok, msg, meta = validate_vap_json(final_path)
                status = "done" if ok else "failed_validation"
                error = "" if ok else msg
            except Exception as e:
                status = "failed"
                error = repr(e)

        if ok and not meta:
            d = load_json_maybe_gz(final_path)
            n_frames = len(d["p_now"][0])  # public VAP JSON p_now is [1, T, 2]
            meta = {"n_frames": n_frames, "probs_dim": 256}

        rows.append({
            "interaction_key": key,
            "project_split": row.get("project_split", ""),
            "stereo_relpath": row["stereo_relpath"],
            "prediction_relpath": relpath(final_path, output_root),
            "prediction_path": str(final_path),
            "n_frames": meta.get("n_frames", ""),
            "probs_dim": meta.get("probs_dim", ""),
            "duration_sec_from_vap": (meta.get("n_frames", 0) / frame_hz) if meta.get("n_frames") else "",
            "status": status,
            "error": error,
        })

        print(f"{key}: {status}" + (f" {error}" if error else ""))

    write_csv(output_root / "vap_manifest.csv", rows)
    print(f"Wrote {output_root / 'vap_manifest.csv'}")
