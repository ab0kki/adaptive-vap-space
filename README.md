# Adaptive VAP Space

`adaptive-vap-space` is a research repository for building a reproducible Seamless Interaction → pretrained Voice Activity Projection (VAP) baseline pipeline.

The repository is designed for a later adaptive-VAP project, but it does **not** implement adaptive models yet. First, it creates a trustworthy baseline and diagnostic dataset.

## Current audit branch

The current audit branch is `event-fidelity-v1`.

This branch assumes the 300-interaction datastore and pretrained VAP outputs may already exist from Kaggle. In that case, do **not** rerun VAP. Reuse:

```text
outputs/vap/vap_manifest.csv
outputs/vap/predictions/*.json.gz
```

Then rerun event extraction, metrics, and EDA.

The source-of-truth explanation for the current event/scoring/evaluation logic is:

```text
docs/EVENT_EXTRACTION_AUDIT.md
```

## What this repo does

Pipeline:

1. Read the Seamless Interaction filelist.
2. Group two participant files by `interaction_key`.
3. Download audio + Seamless-provided VAD + optional transcript for one interaction at a time.
4. Validate duration, VAD quality, speaker balance, speech ratio, overlap, and silence.
5. Build one stereo WAV per kept dyadic interaction.
6. Assign deterministic train/val/test splits by `interaction_key`.
7. Run the public pretrained VAP model on each full stereo WAV.
8. Save VAP `.json.gz` predictions with corruption checks and atomic writes.
9. Derive audited event rows from VAD and VAP outputs.
10. Compute VAP zero-shot score variants, including `paper_256` from the 256-class VAP probabilities.
11. Evaluate calibrated thresholds per `task × score_variant`.
12. Run EDA on dataset quality, VAP residuals, event errors, and event attrition.

## What this repo does not do yet

- It does not download video.
- It does not train or fine-tune full VAP.
- It does not implement adaptive VAP models yet.
- It does not claim exact paper reproduction on the paper's original datasets.
- It separates paper-comparable tasks from exploratory adaptation-motivated tasks.

## Quick local setup

```bash
cd ~/Desktop
mkdir -p adaptive-vap-space
cd adaptive-vap-space
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Clone external repos into `external/`:

```bash
mkdir -p external
git clone https://github.com/facebookresearch/seamless_interaction.git external/seamless_interaction
git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git external/VoiceActivityProjection
```

Check environment:

```bash
python scripts/01_check_env.py --config configs/local_debug.yaml
```

## Local smoke-test pipeline

Start with a tiny target so you can verify the code without waiting forever:

```bash
python scripts/10_build_datastore.py --config configs/local_debug.yaml
python scripts/20_run_vap.py --config configs/local_debug.yaml
python scripts/30_extract_events.py --config configs/local_debug.yaml
python scripts/40_eval_calibrated.py --config configs/local_debug.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/local_debug.yaml --protocol dev_to_test
python scripts/50_eda_dataset.py --config configs/local_debug.yaml
python scripts/51_eda_vap_errors.py --config configs/local_debug.yaml
python scripts/52_eda_event_errors.py --config configs/local_debug.yaml
```

## Main dataset build

Edit `configs/datastore.yaml`, especially:

```yaml
dataset:
  target_kept_interactions: 300
  max_candidate_interactions: 2500
filters:
  min_duration_sec: 180
```

Then run:

```bash
python scripts/10_build_datastore.py --config configs/datastore.yaml
```

The builder is storage-safe: rejected candidate files are downloaded into `_staging/`, logged, and deleted unless `storage.keep_rejected_downloads: true`.

## Reusing existing Kaggle VAP outputs

If the datastore and `outputs/vap` already exist from Kaggle, the audit rerun starts here:

```bash
python scripts/30_extract_events.py --config configs/datastore.yaml
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test
python scripts/50_eda_dataset.py --config configs/datastore.yaml
python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
python scripts/52_eda_event_errors.py --config configs/datastore.yaml
```

## Saving space

Raw mono audio can be large. After stereo WAVs have been built and validated, the VAP baseline only needs:

- `processed/stereo_wav/`
- VAD JSONL files
- manifests/reports

You can delete raw mono audio with:

```bash
python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-raw-audio
```

You can also delete transcripts if they are not needed:

```bash
python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-transcripts
```

After VAP `.json.gz` outputs are complete, you may also delete processed stereo WAVs to save space, but then you cannot re-run VAP without rebuilding/redownloading audio:

```bash
python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-stereo-wav
```

Do **not** delete VAD unless you have already extracted event rows and are sure you will not need to re-run event extraction.

## Output layout

```text
data/datastore/
  raw/audio/
  raw/vad/
  raw/transcript/
  processed/stereo_wav/
  manifests/
  reports/

outputs/
  vap/predictions/
  vap/vap_manifest.csv
  events/event_predictions.csv
  events/event_summary.csv
  events/event_attrition.csv
  events/score_subsets.json
  metrics/
  eda/
```

## Kaggle workflow

Use Kaggle CPU for dataset building, then save `data/datastore` as a private Kaggle Dataset. Use Kaggle GPU for VAP inference.

For the current audit, the preferred workflow is to attach the existing datastore and VAP-output datasets, clone this branch, and rerun only extraction/metrics/EDA.

See `docs/KAGGLE.md`.

## Research integrity rules

- `interaction_key` is the independent unit.
- Splits/folds are by interaction, never by frame/event/chunk.
- Every download attempt is logged.
- Every rejection has a reason.
- Paths in manifests are relative where possible.
- Thresholds are chosen on calibration data, never held-out rows.
- Metrics are reported per `task × score_variant` so paper-style and diagnostic scores are not mixed.
- `S-pred-overlap` is exploratory and should not be reported as paper-comparable.

## Local setup

See `docs/LOCAL_SETUP.md` for exactly how to activate the repo environment and run the smoke pipeline.
