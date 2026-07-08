#!/usr/bin/env python

"""Extract audited VAP event prediction rows from VAD and VAP JSON.GZ.

Usage:
    python scripts/30_extract_events.py --config configs/datastore.yaml

Inputs:
    data/datastore/manifests/stereo_manifest.csv
        Interaction metadata, split labels, and relative paths to Seamless VAD.

    outputs/vap/vap_manifest.csv
        One row per VAP prediction file. The extraction runner resolves paths
        robustly because attached Kaggle output datasets can contain stale
        absolute paths from a previous notebook run.

    outputs/vap/predictions/*.json.gz
        VAP outputs containing ``probs``, ``p_now``, and ``p_future``.

Core logic:
    See ``src/adaptive_vap_space/events.py`` for dialogue-state extraction,
    event definitions, and score-row construction.

Manifest/path handling:
    See ``src/adaptive_vap_space/event_extraction.py``. Extraction is based on
    whether prediction files resolve and load, not on a narrow whitelist of
    manifest status strings.

Outputs:
    outputs/events/event_predictions.csv
        One row per evaluation frame per score variant.

    outputs/events/event_summary.csv
        Per-interaction final event counts.

    outputs/events/event_attrition.csv
        Per-interaction counts from raw dialogue-state alternations to final
        paper-valid and exploratory events.

    outputs/events/score_subsets.json
        Sizes of the 256-state VAP score subsets used by ``paper_256``.

    outputs/events/skipped_interactions.csv
        Rows skipped because prediction/VAD files could not be resolved or loaded.
"""
from __future__ import annotations
import argparse
from adaptive_vap_space.config import load_config
from adaptive_vap_space.event_extraction import extract_events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    extract_events(cfg)


if __name__ == "__main__":
    main()
