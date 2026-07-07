# Event Extraction and VAP Baseline Audit

## Read this first

This is the single audit document for the `event-fidelity-v1` branch.

The branch is for **baseline reconstruction and audit**, not adaptation training. It assumes that a 300-interaction Seamless datastore already exists and that pretrained VAP has already been run in Kaggle, producing:

```text
outputs/vap/vap_manifest.csv
outputs/vap/predictions/*.json.gz
```

The audit notebook should **reuse those VAP outputs** and rerun only:

```bash
python scripts/30_extract_events.py --config configs/datastore.yaml
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test
python scripts/50_eda_dataset.py --config configs/datastore.yaml
python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
python scripts/52_eda_event_errors.py --config configs/datastore.yaml
```

Do not rerun VAP unless the VAP prediction files are missing or corrupt.

## What changed in this audit patch

This patch changed five things:

1. Added `src/adaptive_vap_space/vap_scores.py`.
   - This reconstructs VAP zero-shot task scores from the 256-class `probs` array.
   - These are labeled `score_variant = paper_256`.

2. Rewrote `src/adaptive_vap_space/events.py`.
   - It now separates dialogue states, event definitions, score-row construction, and attrition reporting.
   - It writes paper-comparable tasks and one exploratory overlap task.

3. Rewrote `src/adaptive_vap_space/metrics.py`.
   - Metrics are now computed per `task × score_variant`.
   - The threshold-objective audit still computes weighted F1, macro F1, positive F1, and balanced accuracy.

4. Updated `src/adaptive_vap_space/eda.py`.
   - EDA summaries preserve `score_variant` when it exists, so paper scores and diagnostic scores are not mixed.

5. Updated this document.
   - It is the source-of-truth explanation of the current audit pipeline.

## Important vocabulary

### Calibration objective / metric variant

A calibration objective answers:

> Which threshold should be chosen on calibration data?

The current code still audits these calibration objectives:

```text
weighted_f1
macro_f1
positive_f1
balanced_accuracy
```

The primary threshold in `thresholds.csv` is selected using:

```text
macro_f1
```

The other objectives are still saved in:

```text
outputs/metrics/<protocol>/objective_threshold_metrics.csv
```

### Score variant

A score variant answers:

> Which numeric VAP score is being thresholded?

This is different from the calibration objective.

For example, the same `S/H-paper` frame can be evaluated using:

```text
score_variant = paper_256
score_variant = diag_p_now
score_variant = diag_p_future
```

Each score variant produces its own score column values and its own calibrated metrics. Metrics must be grouped by `task × score_variant`; otherwise the evaluation would mix different definitions of the model score.

## End-to-end repository pipeline

### 1. Datastore build

Script:

```bash
python scripts/10_build_datastore.py --config configs/datastore.yaml
```

Main behavior:

- Reads the Seamless Interaction filelist.
- Groups two participant files by `interaction_key`.
- Downloads audio, VAD, and optional transcript for one candidate interaction at a time.
- Validates duration, VAD quality, speaker balance, speech ratio, overlap, and silence.
- Builds one stereo WAV per kept dyadic interaction.
- Assigns deterministic train/val/test splits by `interaction_key`.

The repository uses **Seamless-provided VAD metadata**. It does not generate VAD locally.

### 2. Pretrained VAP run

Script:

```bash
python scripts/20_run_vap.py --config configs/datastore.yaml
```

Main behavior:

- Calls the public `VoiceActivityProjection/run.py` script on each stereo WAV.
- Saves VAP `.json.gz` predictions to `outputs/vap/predictions/`.
- Writes `outputs/vap/vap_manifest.csv`.

Each prediction file contains:

```text
probs       frame-level 256-state VAP probability distribution
p_now       exported near/current next-speaker helper probability
p_future    exported later-future next-speaker helper probability
```

For the current audit, this stage was already run in Kaggle. The final audit notebook should reuse these files.

### 3. Event extraction and scoring

Script:

```bash
python scripts/30_extract_events.py --config configs/datastore.yaml
```

Main behavior:

- Loads `stereo_manifest.csv`.
- Loads `vap_manifest.csv`.
- Loads Seamless VAD for speaker A and speaker B.
- Converts VAD to 50 Hz two-speaker activity frames.
- Converts activity frames to dialogue states.
- Extracts paper-comparable events and exploratory overlap events.
- Computes multiple VAP score variants.
- Writes event rows, summary rows, and attrition rows.

Outputs:

```text
outputs/events/event_predictions.csv
outputs/events/event_summary.csv
outputs/events/event_attrition.csv
outputs/events/score_subsets.json
```

## Dialogue states

Input frame activity:

```text
va[t, 0] = speaker A active at frame t
va[t, 1] = speaker B active at frame t
```

Dialogue states:

```text
0 = only speaker A
1 = silence
2 = overlap
3 = only speaker B
```

The code first converts VAD to these states, then uses contiguous state runs to find event templates.

## Paper-comparable event tasks

### S/H-paper

Meaning:

> SHIFT versus HOLD during mutual silence.

Templates:

```text
A -> silence -> B = SHIFT
B -> silence -> A = SHIFT
A -> silence -> A = HOLD
B -> silence -> B = HOLD
```

Filters:

```text
minimum mutual silence = 150 ms
pre-offset = 1 s of only previous speaker
post-onset = 1 s of only next speaker
evaluation starts = 50 ms into silence
evaluation duration = 100 ms
minimum past context = 3 s
future horizon required = 2 s
```

Labels:

```text
SHIFT = 1
HOLD = 0
```

Scores written:

```text
paper_256      VAP 256-state silence next-speaker subset
diag_p_now     exported VAP p_now for the shift speaker
diag_p_future  exported VAP p_future for the shift speaker
```

### S-pred-clear

Meaning:

> Predict an upcoming clean paper-valid SHIFT while a speaker is still active.

Positive windows:

```text
500 ms before an S/H-paper SHIFT silence
```

Negative windows:

```text
500 ms sampled from non-shift single-speaker regions
```

Labels:

```text
SHIFT_SOON = 1
sampled non-shift = 0
```

Scores written:

```text
paper_256      VAP 256-state active-region shift subset
diag_p_future  exported VAP p_future for the target next speaker
```

### BC-pred

Meaning:

> Predict an upcoming backchannel.

Backchannel definition:

```text
short isolated listener activity
minimum duration = 200 ms
maximum duration = 1 s
same-speaker pre-silence = 1 s
same-speaker post-silence = 2 s
last speaker before BC = other speaker
```

Positive windows:

```text
500 ms before BC onset
```

Negative windows:

```text
sampled non-shift windows; silence/listening regions are allowed
```

Scores written:

```text
paper_256       VAP 256-state backchannel subset
diag_approx_bc  previous approximate BC reconstruction, retained only for comparison
```

### S/L-paper

Meaning:

> Decide whether a new speaker onset is SHORT/backchannel or LONG/proper shift.

Examples:

```text
SHORT positive = BC onset
LONG negative = S/H-paper SHIFT onset
```

Evaluation window:

```text
first 200 ms after onset
```

Labels:

```text
SHORT = 1
LONG = 0
```

Scores written:

```text
paper_256       VAP 256-state backchannel subset
diag_approx_bc  previous approximate BC reconstruction, retained only for comparison
```

## Exploratory task

### S-pred-overlap

Meaning:

> Predict upcoming rapid-gap or overlap speaker changes that are not part of the paper's clean S/H mutual-silence definition.

Events:

```text
rapid gap speaker change: less than 150 ms silence
overlap speaker change: up to 150 ms overlap
```

Positive windows:

```text
500 ms before the rapid-gap or overlap event
```

Negative windows:

```text
sampled non-shift single-speaker regions
```

Scores written:

```text
paper_256      same active-region shift subset as S-pred-clear
diag_p_future  exported VAP p_future for the target next speaker
```

This task is useful for adaptation/HRI, but it is not paper-comparable.

## Score variants in detail

### `paper_256`

This is the primary audit score. It is reconstructed from `probs`, the 256-class VAP probability vector.

The helper module writes three arrays:

```text
p_silence  used for S/H-paper
p_active   used for S-pred-clear and S-pred-overlap
p_bc       used for BC-pred and S/L-paper
```

### `diag_p_now`

Diagnostic only. Uses the exported VAP `p_now` helper array.

Currently written for:

```text
S/H-paper
```

### `diag_p_future`

Diagnostic only. Uses the exported VAP `p_future` helper array.

Currently written for:

```text
S/H-paper
S-pred-clear
S-pred-overlap
```

### `diag_approx_bc`

Diagnostic only. Uses the previous repository's approximate BC reconstruction.

Currently written for:

```text
BC-pred
S/L-paper
```

## Event row schema

`outputs/events/event_predictions.csv` contains one row per evaluation frame per score variant.

Important columns:

```text
interaction_key      independent interaction id
project_split        train/val/test split
task                 S/H-paper, S-pred-clear, S-pred-overlap, BC-pred, S/L-paper
task_family          paper or exploratory
score_variant        paper_256 or diagnostic variant
paper_comparable     true only for paper task + paper_256 score
label                binary target label
score                numeric VAP score for this row
frame                frame index at 50 Hz
time_s               time in seconds
speaker              event speaker used for event interpretation
score_speaker        speaker index used to read score array
source_event         shift, hold, bc_pred_pos, etc.
event_id             event identifier within interaction
positive_class       SHIFT, SHIFT_SOON, BC_SOON, or SHORT
```

## Attrition audit

`outputs/events/event_attrition.csv` is the key sanity audit for the concern that the code finds fewer shifts than a human hears.

It reports counts such as:

```text
raw_single_speaker_alternations
gap_shift_triads
rapid_gap_shift_triads
overlap_shift_triads
gap_shift_min_silence
gap_shift_pre_ok
gap_shift_post_ok
gap_shift_pre_post_ok
gap_shift_metric_context_ok
final_shift_events
final_hold_events
final_overlap_shift_events
final_bc_events
```

Interpretation:

- If raw alternations are high but final paper shifts are low, the paper's strict event definition is filtering many conversational turn trades.
- If raw alternations are also low, the Seamless VAD representation may not reflect the turns you heard.
- If `gap_shift_pre_post_ok` is high but `final_shift_events` is low, that points to a possible implementation bug or context/windowing issue.

## Evaluation pipeline

Scripts:

```bash
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test
```

Protocols:

```text
val_5fold    five folds inside the val split; split by interaction_key
dev_to_test  calibrate on val, evaluate on test
```

Metrics are grouped by:

```text
task × score_variant
```

That means `S/H-paper / paper_256` and `S/H-paper / diag_p_future` are evaluated separately.

Primary threshold-selection objective:

```text
macro_f1
```

Additional threshold-objective audit:

```text
weighted_f1
macro_f1
positive_f1
balanced_accuracy
```

Output files:

```text
outputs/metrics/<protocol>/thresholds.csv
outputs/metrics/<protocol>/fold_metrics.csv
outputs/metrics/<protocol>/event_level_metrics.csv
outputs/metrics/<protocol>/threshold_sweep.csv
outputs/metrics/<protocol>/objective_threshold_metrics.csv
outputs/metrics/<protocol>/event_predictions_eval.csv
outputs/metrics/<protocol>/task_metrics.csv
outputs/metrics/<protocol>/protocol_split_counts.csv
outputs/metrics/<protocol>/metrics_report.json
```

## EDA pipeline

Scripts:

```bash
python scripts/50_eda_dataset.py --config configs/datastore.yaml
python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
python scripts/52_eda_event_errors.py --config configs/datastore.yaml
```

EDA outputs preserve `score_variant` where present. This avoids mixing paper scores and diagnostic scores.

Output roots:

```text
outputs/eda/dataset
outputs/eda/vap_errors
outputs/eda/event_errors
```

## What to inspect first after a Kaggle run

1. `outputs/events/event_attrition.csv`
   - Check whether raw alternations are much larger than final paper shifts.

2. `outputs/events/event_summary.csv`
   - Check final event counts per interaction.

3. `outputs/metrics/dev_to_test/fold_metrics.csv`
   - Filter to `score_variant == paper_256` for main baseline inspection.

4. `outputs/metrics/dev_to_test/objective_threshold_metrics.csv`
   - Compare weighted-F1, macro-F1, positive-F1, and balanced-accuracy threshold choices.

5. `outputs/eda/event_errors/classification_error_by_source_dev_to_test.csv`
   - Check which event sources fail most.

## Known limitations

1. This is a proxy VAP baseline on Seamless Interaction, not an exact re-run of the paper's datasets.
2. Event counts are VAD-dependent.
3. `S-pred-overlap` is exploratory and must not be reported as paper-comparable.
4. The primary threshold still uses macro F1 to match the previous audit notebook behavior; paper-style weighted-F1 threshold rows are preserved in the objective audit.
5. The next qualitative audit should inspect a sample of high-attrition clips against audio.
