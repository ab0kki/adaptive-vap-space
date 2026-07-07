# Kaggle Guide

This guide has two modes:

1. Full build mode: build datastore and run VAP.
2. Audit rerun mode: reuse an existing 300-interaction datastore and existing VAP outputs, then rerun only event extraction, metrics, and EDA.

For the current `event-fidelity-v1` audit, use audit rerun mode unless the VAP outputs are missing or corrupt.

## Audit rerun notebook, preferred current workflow

Attach Kaggle datasets containing:

```text
data/datastore/
outputs/vap/
```

Then clone this branch and install:

```bash
git clone --branch event-fidelity-v1 https://github.com/ab0kki/adaptive-vap-space.git
cd adaptive-vap-space
python -m pip install -e ".[dev]"
```

In the notebook, symlink or copy the attached datastore and VAP outputs so the repo sees:

```text
data/datastore/manifests/stereo_manifest.csv
outputs/vap/vap_manifest.csv
outputs/vap/predictions/*.json.gz
```

Then run:

```bash
python scripts/30_extract_events.py --config configs/datastore.yaml
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test
python scripts/50_eda_dataset.py --config configs/datastore.yaml
python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
python scripts/52_eda_event_errors.py --config configs/datastore.yaml
```

Key files to inspect first:

```text
outputs/events/event_attrition.csv
outputs/events/event_summary.csv
outputs/metrics/dev_to_test/fold_metrics.csv
outputs/metrics/dev_to_test/objective_threshold_metrics.csv
outputs/eda/event_errors/classification_error_by_source_dev_to_test.csv
```

For the main baseline view, filter metrics to:

```text
score_variant == paper_256
```

For diagnostic comparisons, inspect:

```text
diag_p_now
diag_p_future
diag_approx_bc
```

## Packaging audit results

When packaging results, include:

```text
outputs/events/
outputs/metrics/
outputs/eda/
outputs/logs/ if present
configs/datastore.yaml
docs/EVENT_EXTRACTION_AUDIT.md
src/adaptive_vap_space/events.py
src/adaptive_vap_space/metrics.py
src/adaptive_vap_space/vap_scores.py
src/adaptive_vap_space/vap_outputs.py
```

Do not include:

```text
outputs/vap/
outputs/vap/predictions/*.json.gz
raw audio
stereo wavs
data/datastore/
```

The VAP outputs can be very large and should remain in the Kaggle input dataset, not the packaged audit artifact.

## Full dataset build notebook, CPU

Use this only when rebuilding the datastore.

```bash
git clone --branch event-fidelity-v1 https://github.com/ab0kki/adaptive-vap-space.git
cd adaptive-vap-space
python -m pip install -e ".[dev]"
git clone https://github.com/facebookresearch/seamless_interaction.git external/seamless_interaction
python scripts/10_build_datastore.py --config configs/datastore.yaml
```

Save `data/datastore` as a private Kaggle Dataset.

## Full VAP baseline notebook, GPU

Use this only when VAP predictions do not already exist.

Attach your saved datastore dataset. Then:

```bash
git clone --branch event-fidelity-v1 https://github.com/ab0kki/adaptive-vap-space.git
cd adaptive-vap-space
python -m pip install -e ".[dev]"
git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git external/VoiceActivityProjection
python scripts/20_run_vap.py --config configs/datastore.yaml
```

After VAP finishes, save `outputs/vap` as a separate private Kaggle Dataset. Future audit runs can attach that output dataset and skip VAP inference.
