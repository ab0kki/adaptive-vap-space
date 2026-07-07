#!/usr/bin/env python

"""Extract audited VAP event prediction rows from VAD and VAP JSON.GZ.

Usage:
    python scripts/30_extract_events.py --config configs/datastore.yaml

Inputs:
    data/datastore/manifests/stereo_manifest.csv
        Interaction metadata, split labels, and relative paths to Seamless VAD.

    outputs/vap/vap_manifest.csv
        One row per VAP prediction file.

    outputs/vap/predictions/*.json.gz
        VAP outputs containing ``probs``, ``p_now``, and ``p_future``.

Core logic:
    See ``src/adaptive_vap_space/events.py`` for dialogue-state extraction,
    event definitions, score-row construction, and attrition reporting.

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
