"""Storage-safe Seamless datastore builder.

This module ports the working ideas from the old prototype downloader but changes
storage behavior: each candidate interaction is downloaded into staging, validated,
and then either kept or deleted.
"""
from __future__ import annotations

from pathlib import Path
import random
import shutil
import pandas as pd

from .config import get
from .download import download_first_available, file_size_mb
from .filtering import validate_speaker, validate_interaction
from .paths import datastore_dirs, relpath
from .reports import write_csv, write_json, utc_now
from .seamless import load_filelist, build_candidates, urls_for, InteractionCandidate
from .stereo import build_stereo_row
from .vad import read_vad_segments


def assign_project_splits(keys: list[str], cfg: dict) -> dict[str, str]:
    """Assign deterministic train/val/test splits by interaction key."""
    seed = int(get(cfg, "splits.seed", get(cfg, "project.seed", 42)))
    ratios = {
        "train": float(get(cfg, "splits.train", 0.70)),
        "val": float(get(cfg, "splits.val", 0.15)),
        "test": float(get(cfg, "splits.test", 0.15)),
    }
    keys = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    split_map = {}
    for i, key in enumerate(keys):
        if i < n_train:
            split_map[key] = "train"
        elif i < n_train + n_val:
            split_map[key] = "val"
        else:
            split_map[key] = "test"
    return split_map


def _safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        src.unlink(missing_ok=True)
        return dst
    shutil.move(str(src), str(dst))
    return dst


def build_datastore(cfg: dict) -> None:
    """Build/update the datastore according to config."""
    dataset_root = Path(get(cfg, "dataset.root", "data/datastore"))
    dirs = datastore_dirs(dataset_root)
    staging_root = dataset_root / str(get(cfg, "storage.staging_dirname", "_staging"))
    staging_root.mkdir(parents=True, exist_ok=True)

    kept_path = dirs["manifests"] / "kept_interactions.csv"
    already_kept = set()
    if kept_path.exists():
        kept_df_existing = pd.read_csv(kept_path)
        if "interaction_key" in kept_df_existing.columns:
            already_kept = set(kept_df_existing["interaction_key"].astype(str))

    filelist = load_filelist(get(cfg, "external.seamless_repo"))
    labels = (
        get(cfg, "dataset.labels", None)
        or get(cfg, "selection.allowed_labels", None)
        or [get(cfg, "dataset.label", "naturalistic")]
    )
    if isinstance(labels, str):
        labels = [labels]
    labels = list(dict.fromkeys(str(x) for x in labels))

    valid_labels = set(filelist["label"].dropna().astype(str).unique())
    bad_labels = sorted(set(labels) - valid_labels)
    if bad_labels:
        raise ValueError(
            f"Invalid Seamless label(s): {bad_labels}. "
            f"Available labels are: {sorted(valid_labels)}"
        )

    candidates = []
    for label in labels:
        candidates.extend(build_candidates(
            filelist,
            label=label,
            source_splits=get(cfg, "dataset.source_splits", ["dev"]),
            preferred_vendors_only=bool(get(cfg, "dataset.preferred_vendors_only", False)),
        ))

    # Project-level vendor quality filter before shuffle/limit/download.
    # This keeps the research dataset V00-only and prevents wasting storage.
    allowed_vendors_pre = set(
        get(cfg, "selection.allowed_vendors", None)
        or get(cfg, "dataset.allowed_vendors", None)
        or []
    )
    if allowed_vendors_pre:
        candidates = [
            cand for cand in candidates
            if str(cand.interaction_key).split("_", 1)[0] in allowed_vendors_pre
        ]
    seed = int(get(cfg, "dataset.seed", get(cfg, "project.seed", 42)))
    if bool(get(cfg, "dataset.shuffle_candidates", True)):
        rng = random.Random(seed)
        rng.shuffle(candidates)
    max_candidates = int(get(cfg, "dataset.max_candidate_interactions", len(candidates)))
    candidates = candidates[:max_candidates]
    target = int(get(cfg, "dataset.target_kept_interactions", 300))

    candidate_rows = []
    download_rows = []
    speaker_rows = []
    interaction_rows = []
    stereo_rows = []
    rejected_rows = []

    # Load old manifest rows so an interrupted run can resume.
    def load_rows(path):
        return pd.read_csv(path).to_dict("records") if path.exists() else []
    download_rows.extend(load_rows(dirs["manifests"] / "download_attempts.csv"))
    speaker_rows.extend(load_rows(dirs["manifests"] / "speaker_manifest.csv"))
    interaction_rows.extend(load_rows(dirs["manifests"] / "interaction_manifest.csv"))
    stereo_rows.extend(load_rows(dirs["manifests"] / "stereo_manifest.csv"))
    rejected_rows.extend(load_rows(dirs["manifests"] / "rejected_interactions.csv"))

    kept_keys = set(already_kept)
    allowed_vendors = set(
        get(cfg, "selection.allowed_vendors", None)
        or get(cfg, "dataset.allowed_vendors", None)
        or []
    )

    for cand in candidates:
        if len(kept_keys) >= target:
            break
        candidate_rows.append(cand.__dict__)

        vendor = str(cand.interaction_key).split("_", 1)[0]
        if allowed_vendors and vendor not in allowed_vendors:
            rejected_rows.append({
                "interaction_key": cand.interaction_key,
                "rejection_reasons": "vendor_not_allowed",
                "vendor": vendor,
            })
            continue

        if cand.interaction_key in kept_keys:
            continue

        stage = staging_root / cand.interaction_key
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)

        paths = {}
        failed = False
        for tag, file_id in [("a", cand.file_id_a), ("b", cand.file_id_b)]:
            for modality in ["audio", "vad"] + (["transcript"] if get(cfg, "modalities.download_transcript", True) else []):
                ext = ".wav" if modality == "audio" else ".jsonl"
                target_path = stage / modality / f"{file_id}{ext}"
                result = download_first_available(
                    urls_for(cand.label, cand.split, modality, file_id),
                    target_path,
                    overwrite=bool(get(cfg, "download.overwrite", False)),
                    retries=int(get(cfg, "download.retries", 3)),
                    timeout_sec=int(get(cfg, "download.timeout_sec", 60)),
                    chunk_size_mb=int(get(cfg, "download.chunk_size_mb", 4)),
                )
                download_rows.append({
                    "interaction_key": cand.interaction_key,
                    "file_id": file_id,
                    "speaker_tag": tag.upper(),
                    "source_split": cand.split,
                    "label": cand.label,
                    "modality": modality,
                    "status": result.status,
                    "ok": result.ok,
                    "url": result.url or "",
                    "relpath": relpath(result.path, dataset_root),
                    "size_mb": result.size_mb,
                    "error": result.error or "",
                })
                if not result.ok and (modality in ["audio", "vad"] or get(cfg, f"modalities.require_{modality}", False)):
                    failed = True
                paths[(tag, modality)] = Path(result.path)

        if failed:
            reasons = ["missing_audio_or_vad"]
            rejected_rows.append({"interaction_key": cand.interaction_key, "rejection_reasons": ";".join(reasons)})
            interaction_rows.append({"interaction_key": cand.interaction_key, "keep": False, "rejection_reasons": ";".join(reasons)})
            if not get(cfg, "storage.keep_rejected_downloads", False):
                shutil.rmtree(stage, ignore_errors=True)
            continue

        sp_a = validate_speaker(paths[("a", "audio")], paths[("a", "vad")], paths.get(("a", "transcript")), cfg)
        sp_b = validate_speaker(paths[("b", "audio")], paths[("b", "vad")], paths.get(("b", "transcript")), cfg)
        vad_a, _ = read_vad_segments(paths[("a", "vad")])
        vad_b, _ = read_vad_segments(paths[("b", "vad")])
        ok, reasons, stats = validate_interaction(vad_a, vad_b, sp_a, sp_b, cfg)

        for tag, file_id, sp in [("A", cand.file_id_a, sp_a), ("B", cand.file_id_b, sp_b)]:
            speaker_rows.append({
                "interaction_key": cand.interaction_key,
                "file_id": file_id,
                "participant_id": cand.participant_a if tag == "A" else cand.participant_b,
                "speaker_tag": tag,
                "duration_sec": sp.duration_sec,
                "sample_rate": sp.sample_rate,
                "channels": sp.channels,
                "vad_segments": sp.vad_segments,
                "vad_speech_sec": sp.vad_speech_sec,
                "vad_ratio": sp.vad_ratio,
                "transcript_rows": sp.transcript_rows,
                "ok": sp.ok,
                "rejection_reasons": ";".join(sp.reasons),
            })

        base_row = {
            "interaction_key": cand.interaction_key,
            "source_label": cand.label,
            "source_split": cand.split,
            "file_id_a": cand.file_id_a,
            "file_id_b": cand.file_id_b,
            "participant_a": cand.participant_a,
            "participant_b": cand.participant_b,
            "keep": ok,
            "rejection_reasons": ";".join(reasons),
            **stats,
        }

        if ok:
            # Move kept files into canonical raw storage.
            final = {}
            for tag, file_id in [("a", cand.file_id_a), ("b", cand.file_id_b)]:
                for modality in ["audio", "vad"] + (["transcript"] if get(cfg, "modalities.download_transcript", True) else []):
                    ext = ".wav" if modality == "audio" else ".jsonl"
                    dst_dir = dirs[f"raw_{modality}"]
                    final[(tag, modality)] = _safe_move(paths[(tag, modality)], dst_dir / f"{file_id}{ext}")
            split_map = assign_project_splits(list(kept_keys | {cand.interaction_key}), cfg)
            project_split = split_map[cand.interaction_key]
            stereo_row = build_stereo_row(
                dataset_root=dataset_root,
                interaction_key=cand.interaction_key,
                project_split=project_split,
                audio_a=final[("a", "audio")],
                audio_b=final[("b", "audio")],
                vad_a=final[("a", "vad")],
                vad_b=final[("b", "vad")],
                transcript_a=final.get(("a", "transcript")),
                transcript_b=final.get(("b", "transcript")),
                speaker_a=cand.participant_a,
                speaker_b=cand.participant_b,
                sample_rate=int(get(cfg, "audio.sample_rate", 16000)),
                overwrite=False,
            )
            if bool(get(cfg, "storage.delete_raw_audio_after_stereo", False)):
                stereo_path = dataset_root / stereo_row["stereo_relpath"]
                raw_audio_deleted = []
                raw_audio_delete_errors = []
                if stereo_path.exists() and stereo_path.stat().st_size > 0:
                    for raw_audio_path in [final[("a", "audio")], final[("b", "audio")]]:
                        try:
                            raw_audio_path.unlink(missing_ok=True)
                            raw_audio_deleted.append(relpath(raw_audio_path, dataset_root))
                        except Exception as e:
                            raw_audio_delete_errors.append(f"{raw_audio_path}: {repr(e)}")
                else:
                    raw_audio_delete_errors.append(f"stereo_missing_or_empty: {stereo_path}")
                stereo_row["raw_audio_deleted_after_stereo"] = len(raw_audio_deleted)
                stereo_row["raw_audio_delete_errors"] = ";".join(raw_audio_delete_errors)
            else:
                stereo_row["raw_audio_deleted_after_stereo"] = 0
                stereo_row["raw_audio_delete_errors"] = ""

            stereo_rows.append(stereo_row)
            kept_keys.add(cand.interaction_key)
            base_row["project_split"] = project_split
        else:
            rejected_rows.append({"interaction_key": cand.interaction_key, "rejection_reasons": ";".join(reasons), **stats})

        interaction_rows.append(base_row)
        if not get(cfg, "storage.keep_rejected_downloads", False):
            shutil.rmtree(stage, ignore_errors=True)

        # Write progress manifests after each candidate so interrupted runs keep metadata.
        write_csv(dirs["manifests"] / "download_attempts.csv", download_rows)
        write_csv(dirs["manifests"] / "speaker_manifest.csv", speaker_rows)
        write_csv(dirs["manifests"] / "interaction_manifest.csv", interaction_rows)
        write_csv(dirs["manifests"] / "rejected_interactions.csv", rejected_rows)
        write_csv(dirs["manifests"] / "stereo_manifest.csv", stereo_rows)
        kept_df = pd.DataFrame([r for r in interaction_rows if r.get("keep") is True]).drop_duplicates("interaction_key", keep="last") if interaction_rows else pd.DataFrame()
        write_csv(dirs["manifests"] / "kept_interactions.csv", kept_df)

    # Recompute stable project splits for all kept interactions at the end.
    kept_df = pd.DataFrame([r for r in interaction_rows if r.get("keep") is True]).drop_duplicates("interaction_key", keep="last") if interaction_rows else pd.DataFrame()
    if not kept_df.empty:
        split_map = assign_project_splits(kept_df["interaction_key"].astype(str).tolist(), cfg)
        kept_df["project_split"] = kept_df["interaction_key"].map(split_map)
        split_rows = [{"interaction_key": k, "project_split": v} for k, v in sorted(split_map.items())]
        stereo_df = pd.DataFrame(stereo_rows).drop_duplicates("interaction_key", keep="last")
        if not stereo_df.empty:
            stereo_df["project_split"] = stereo_df["interaction_key"].map(split_map)
            write_csv(dirs["manifests"] / "stereo_manifest.csv", stereo_df)
        write_csv(dirs["manifests"] / "project_splits.csv", split_rows)
    write_csv(dirs["manifests"] / "candidates_seen.csv", candidate_rows)
    write_csv(dirs["manifests"] / "kept_interactions.csv", kept_df)

    report = {
        "timestamp_utc": utc_now(),
        "config_path": cfg.get("_config_path", ""),
        "candidate_interactions_considered": len(candidate_rows),
        "kept_interactions": int(len(kept_df)),
        "target_kept_interactions": target,
        "download_attempt_rows": len(download_rows),
        "rejected_rows": len(rejected_rows),
        "dataset_root": str(dataset_root),
        "raw_audio_mb": sum(file_size_mb(p) for p in dirs["raw_audio"].glob("*.wav")),
        "stereo_wav_mb": sum(file_size_mb(p) for p in dirs["processed_stereo"].glob("*.wav")),
    }
    write_json(dirs["reports"] / "datastore_report.json", report)
    (dirs["reports"] / "dataset_summary.md").write_text(
        f"# Dataset Summary\n\nKept interactions: {report['kept_interactions']} / {target}\n\n"
        f"Raw audio MB: {report['raw_audio_mb']:.1f}\n\nStereo WAV MB: {report['stereo_wav_mb']:.1f}\n",
        encoding="utf-8",
    )
