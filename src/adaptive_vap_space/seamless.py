"""Seamless filelist loading, candidate grouping, and URL construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd

from .ids import parse_file_id

BASE_URL = "https://dl.fbaipublicfiles.com/seamless_interaction"


@dataclass(frozen=True)
class InteractionCandidate:
    """A candidate dyadic interaction from the Seamless filelist."""

    interaction_key: str
    label: str
    split: str
    file_id_a: str
    file_id_b: str
    participant_a: str
    participant_b: str
    n_participants_available: int


def filelist_path(seamless_repo: str | Path) -> Path:
    """Return the expected Seamless filelist path."""
    return Path(seamless_repo) / "assets" / "filelist.csv"


def load_filelist(seamless_repo: str | Path) -> pd.DataFrame:
    """Load Seamless `assets/filelist.csv` and add parsed ID columns."""
    path = filelist_path(seamless_repo)
    if not path.exists():
        raise FileNotFoundError(
            f"Seamless filelist not found: {path}\n"
            "Clone https://github.com/facebookresearch/seamless_interaction.git into external/."
        )
    df = pd.read_csv(path)
    if "file_id" not in df.columns:
        raise ValueError(f"Expected filelist column 'file_id' in {path}")

    parsed_rows = []
    for file_id in df["file_id"].astype(str):
        try:
            p = parse_file_id(file_id)
            parsed_rows.append({
                "parsed_ok": True,
                "interaction_key": p.interaction_key,
                "participant_id": p.participant_id,
                "parsed_file_id": p.file_id,
            })
        except ValueError:
            parsed_rows.append({
                "parsed_ok": False,
                "interaction_key": None,
                "participant_id": None,
                "parsed_file_id": None,
            })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(parsed_rows)], axis=1)


def build_candidates(
    filelist: pd.DataFrame,
    label: str,
    source_splits: Iterable[str],
    preferred_vendors_only: bool = False,
) -> list[InteractionCandidate]:
    """Group filelist rows into deterministic dyadic candidates."""
    df = filelist.copy()
    df = df[df["parsed_ok"] == True]
    if "label" in df.columns:
        df = df[df["label"] == label]
    if "split" in df.columns:
        df = df[df["split"].isin(list(source_splits))]
    if preferred_vendors_only:
        df = df[df["file_id"].astype(str).str.startswith(("V00", "V01"))]

    candidates: list[InteractionCandidate] = []
    group_cols = ["interaction_key"]
    for key, g in df.groupby(group_cols):
        if isinstance(key, tuple):
            key = key[0]
        ids = sorted(g["file_id"].astype(str).unique().tolist())
        if len(ids) < 2:
            continue
        # Deterministic pair selection. If >2 participants exist, choose the first
        # two but record n_participants_available so the decision is auditable.
        a, b = ids[:2]
        pa = parse_file_id(a).participant_id
        pb = parse_file_id(b).participant_id
        row0 = g.iloc[0]
        candidates.append(InteractionCandidate(
            interaction_key=str(key),
            label=str(row0.get("label", label)),
            split=str(row0.get("split", "unknown")),
            file_id_a=a,
            file_id_b=b,
            participant_a=pa,
            participant_b=pb,
            n_participants_available=len(ids),
        ))
    return candidates


def urls_for(label: str, split: str, modality: str, file_id: str) -> list[str]:
    """Return public Seamless URLs to try for a modality/file_id."""
    if modality == "audio":
        return [f"{BASE_URL}/{label}/{split}/audio/{file_id}.wav"]
    if modality == "vad":
        return [
            f"{BASE_URL}/{label}/{split}/vad/{file_id}.jsonl",
            f"{BASE_URL}/{label}/{split}/metadata/vad/{file_id}.jsonl",
        ]
    if modality == "transcript":
        return [
            f"{BASE_URL}/{label}/{split}/transcript/{file_id}.jsonl",
            f"{BASE_URL}/{label}/{split}/metadata/transcript/{file_id}.jsonl",
        ]
    raise ValueError(f"Unsupported modality: {modality}")
