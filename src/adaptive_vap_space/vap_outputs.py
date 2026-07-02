"""VAP JSON loading and probability helpers."""
from __future__ import annotations

from pathlib import Path
import gzip
import json
import numpy as np


def load_json_maybe_gz(path: str | Path) -> dict:
    """Load a JSON or JSON.GZ file."""
    path = Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_vap_json(path: str | Path) -> tuple[bool, str, dict]:
    """Check that a VAP output is readable and has expected arrays."""
    try:
        d = load_json_maybe_gz(path)
        if "probs" not in d or "p_now" not in d or "p_future" not in d:
            return False, "missing required keys", {}
        probs = np.asarray(d["probs"])
        p_now = np.asarray(d["p_now"])
        p_future = np.asarray(d["p_future"])
        if p_now.ndim != 3:
            return False, f"bad p_now ndim={p_now.ndim}; expected [1,T,2]", {}
        if p_now.shape[0] != 1:
            return False, f"bad p_now batch={p_now.shape[0]}; expected 1", {}
        if p_now.shape[-1] != 2:
            return False, f"bad p_now speaker_dim={p_now.shape[-1]}; expected 2", {}
        if probs.ndim != 3:
            return False, f"bad probs ndim={probs.ndim}; expected [1,T,256]", {}
        if probs.shape[0] != 1:
            return False, f"bad probs batch={probs.shape[0]}; expected 1", {}
        if probs.shape[1] != p_now.shape[1]:
            return False, f"time mismatch probs={probs.shape[1]} p_now={p_now.shape[1]}", {}
        if probs.shape[-1] != 256:
            return False, f"bad probs_dim={probs.shape[-1]}", {}
        return True, "ok", {"n_frames": int(p_now.shape[1]), "probs_dim": 256}
    except Exception as e:
        return False, repr(e), {}


def load_prediction_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `probs`, `p_now`, and `p_future` arrays as `(T, ...)`.

    Public VAP JSONs may serialize arrays as `(2,T)` for p_now/p_future or `(T,2)`
    depending on version. This function normalizes to `(T,2)`.
    """
    d = load_json_maybe_gz(path)
    probs = np.asarray(d["probs"], dtype=float)
    p_now = np.asarray(d["p_now"], dtype=float)
    p_future = np.asarray(d["p_future"], dtype=float)

    # Public VAP run.py writes [1,T,*]. Remove only this known batch dimension.
    if probs.ndim == 3 and probs.shape[0] == 1:
        probs = probs[0]
    if p_now.ndim == 3 and p_now.shape[0] == 1:
        p_now = p_now[0]
    if p_future.ndim >= 3 and p_future.shape[0] == 1:
        p_future = p_future[0]

    if p_now.ndim == 2 and p_now.shape[0] == 2 and p_now.shape[1] != 2:
        p_now = p_now.T
    if p_future.ndim == 2 and p_future.shape[0] == 2 and p_future.shape[1] != 2:
        p_future = p_future.T
    if probs.ndim == 3 and probs.shape[0] == 2 and probs.shape[1] == len(p_now):
        # Some outputs may be (2,T,256); average channels if present.
        probs = probs.mean(axis=0)
    if probs.ndim == 3 and probs.shape[1] == 1:
        probs = probs[:, 0, :]
    if probs.ndim != 2:
        probs = probs.reshape((len(p_now), -1))
    return probs, p_now, p_future


def backchannel_prob_from_probs(probs: np.ndarray) -> np.ndarray:
    """Reconstruct a simple BC probability from 256-state VAP probabilities.

    This is a documented approximation: a future state is counted as a possible
    backchannel for speaker `s` when the earliest future bin for `s` is active but
    later bins for `s` are inactive. This captures short future activity but should
    be audited before publication.
    """
    probs = np.asarray(probs, dtype=float)
    n = probs.shape[0]
    p_bc = np.zeros((n, 2), dtype=float)
    states = np.arange(256)
    bits = ((states[:, None] >> np.arange(8)) & 1).astype(bool)
    # Convention used by VAP codebook is implementation-sensitive. This mask is
    # intentionally conservative and documented rather than claimed exact.
    for spk in [0, 1]:
        # speaker bins: spk*4 ... spk*4+3
        b0 = bits[:, spk * 4 + 0]
        b1 = bits[:, spk * 4 + 1]
        b2 = bits[:, spk * 4 + 2]
        b3 = bits[:, spk * 4 + 3]
        short_mask = (b0 | b1) & ~(b2 | b3)
        p_bc[:, spk] = probs[:, short_mask].sum(axis=1)
    return p_bc
