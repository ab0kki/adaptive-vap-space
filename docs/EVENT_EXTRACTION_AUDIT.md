# Event Extraction and VAP Baseline Audit

## 0. Scope and reproducibility contract

This is the single source-of-truth audit document for the `event-fidelity-v1` branch.

This branch supports **two valid workflows**:

1. **Full reproducible pipeline from scratch**
   - build the Seamless datastore;
   - run pretrained VAP;
   - extract events;
   - evaluate metrics;
   - run EDA.

2. **Kaggle audit rerun from existing artifacts**
   - reuse the already-built 300-interaction datastore;
   - reuse the already-computed pretrained VAP `.json.gz` outputs;
   - rerun only event extraction, metrics, and EDA.

The second workflow is a storage/time optimization, not a hidden dependency. The repository is still intended to work as a full pipeline from scratch as long as the external Seamless and VAP repositories, data URLs, and checkpoint path are available.

The current audit goal is not to train an adaptive model yet. The goal is to create a transparent VAP baseline evaluation that can later support intra-interaction or speaker-level calibration/adaptation experiments.

## 1. Full pipeline commands

### 1.1 Full pipeline from scratch

```bash
python scripts/10_build_datastore.py --config configs/datastore.yaml
python scripts/20_run_vap.py --config configs/datastore.yaml
python scripts/30_extract_events.py --config configs/datastore.yaml
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test
python scripts/50_eda_dataset.py --config configs/datastore.yaml
python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
python scripts/52_eda_event_errors.py --config configs/datastore.yaml
```

### 1.2 Audit rerun when datastore and VAP outputs already exist

If these already exist:

```text
data/datastore/manifests/stereo_manifest.csv
outputs/vap/vap_manifest.csv
outputs/vap/predictions/*.json.gz
```

then run:

```bash
python scripts/30_extract_events.py --config configs/datastore.yaml
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol val_5fold
python scripts/40_eval_calibrated.py --config configs/datastore.yaml --protocol dev_to_test
python scripts/50_eda_dataset.py --config configs/datastore.yaml
python scripts/51_eda_vap_errors.py --config configs/datastore.yaml
python scripts/52_eda_event_errors.py --config configs/datastore.yaml
```

This is the expected Kaggle audit workflow because VAP was already run for the 300 processed interactions.

## 2. Repository stages and files

### Stage A: datastore build

Script:

```text
scripts/10_build_datastore.py
```

Config:

```text
configs/datastore.yaml
```

Important config fields:

```text
dataset.root = data/datastore
dataset.target_kept_interactions = 300
modalities.require_audio = true
modalities.require_vad = true
modalities.download_video = false
filters.min_duration_sec = 180
splits.train/val/test = 0.70/0.15/0.15
audio.sample_rate = 16000
vap.frame_hz = 50
```

What this stage does:

1. reads the Seamless Interaction filelist;
2. groups candidate participant files into dyadic `interaction_key`s;
3. downloads audio and Seamless-provided VAD metadata;
4. optionally downloads transcript metadata;
5. validates duration, VAD, speech ratio, speaker balance, overlap, and silence;
6. builds one stereo WAV per kept interaction;
7. writes manifests and deterministic train/val/test splits by `interaction_key`.

The repository does **not** compute its own VAD. It uses Seamless-provided VAD JSONL metadata.

### Stage B: pretrained VAP inference

Script:

```text
scripts/20_run_vap.py
```

What this stage does:

1. reads `data/datastore/manifests/stereo_manifest.csv`;
2. runs public `VoiceActivityProjection/run.py` on each stereo WAV;
3. writes one compressed VAP prediction JSON per interaction;
4. writes `outputs/vap/vap_manifest.csv`.

Each VAP prediction file is expected to contain:

```text
probs       shape (T, 256), after removing batch dimension
p_now       shape (T, 2), exported VAP helper score
p_future    shape (T, 2), exported VAP helper score
```

`probs` is the primary source for paper-style score reconstruction in this audit.

### Stage C: event extraction and score-row construction

Script:

```text
scripts/30_extract_events.py
```

Core implementation:

```text
src/adaptive_vap_space/events.py
src/adaptive_vap_space/vap_scores.py
src/adaptive_vap_space/vap_outputs.py
```

What this stage does:

1. loads interaction metadata from the datastore;
2. loads VAP output paths from `vap_manifest.csv`;
3. loads `probs`, `p_now`, and `p_future` for each interaction;
4. loads Seamless VAD for speaker A and B;
5. converts VAD segments to 50 Hz frame activity;
6. converts frame activity to dialogue states;
7. extracts event candidates;
8. computes VAP score variants;
9. writes one row per evaluation frame per score variant;
10. writes event count and attrition audits.

Outputs:

```text
outputs/events/event_predictions.csv
outputs/events/event_summary.csv
outputs/events/event_attrition.csv
outputs/events/score_subsets.json
```

### Stage D: calibrated evaluation

Script:

```text
scripts/40_eval_calibrated.py
```

Core implementation:

```text
src/adaptive_vap_space/metrics.py
```

What this stage does:

1. reads `outputs/events/event_predictions.csv`;
2. splits rows by `interaction_key`, never by frame;
3. groups rows by `task × score_variant`;
4. selects thresholds on calibration rows;
5. evaluates on held-out rows;
6. writes frame-level and event-level metrics;
7. writes threshold-objective audits.

Protocols:

```text
val_5fold    folds inside val only; development/audit diagnostic
dev_to_test  calibrate on val, evaluate once on test
```

Outputs:

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

### Stage E: EDA

Scripts:

```text
scripts/50_eda_dataset.py
scripts/51_eda_vap_errors.py
scripts/52_eda_event_errors.py
```

Core implementation:

```text
src/adaptive_vap_space/eda.py
```

What this stage does:

1. summarizes dataset quality and split balance;
2. computes residual summaries over time and speaker ids;
3. computes classification-error summaries by task, event source, score variant, interaction, speaker, and relative time;
4. preserves `score_variant` so paper-style and diagnostic scores are not mixed.

## 3. Dialogue-state extraction

The code starts from a two-speaker frame activity matrix:

```text
va[t, 0] = speaker A active at frame t
va[t, 1] = speaker B active at frame t
```

At 50 Hz, one frame is 20 ms.

The code maps this to dialogue states:

```text
0 = only speaker A        va[t,0] == true  and va[t,1] == false
1 = silence               va[t,0] == false and va[t,1] == false
2 = overlap               va[t,0] == true  and va[t,1] == true
3 = only speaker B        va[t,0] == false and va[t,1] == true
```

Contiguous runs of the same state are then used for event templates. A run is stored as:

```text
(start_frame, end_frame, state)
```

where `end_frame` is exclusive.

The code also fills same-speaker hold silences for pre/post region checks. This means:

```text
A -> silence -> A
B -> silence -> B
```

can be treated as a continuous speaker-control region when checking whether the surrounding 1 second is controlled by only one speaker.

## 4. Event definitions

This section is the precise event-definition contract for the branch.

### 4.1 S/H-paper

Purpose:

> Evaluate whether VAP predicts SHIFT versus HOLD during mutual silence.

This task is paper-comparable within the limits of using Seamless data instead of the original paper datasets.

Dialogue-state templates:

```text
A -> silence -> B = SHIFT
B -> silence -> A = SHIFT
A -> silence -> A = HOLD
B -> silence -> B = HOLD
```

Frame filters:

```text
minimum silence duration       = 150 ms = 8 frames at 50 Hz after rounding
pre-offset region              = 1.0 s before silence start
post-onset region              = 1.0 s after next onset
evaluation offset              = 50 ms into the silence
evaluation duration            = 100 ms
minimum past context           = 3.0 s
required future VAP horizon    = 2.0 s
```

The pre-offset region must contain only the previous speaker. The post-onset region must contain only the next speaker.

Evaluation window:

```text
eval_start = silence_start + 50 ms
eval_end   = eval_start + 100 ms
```

Labels:

```text
SHIFT = 1
HOLD  = 0
```

Scored speaker:

```text
shift_speaker = 1 - previous_speaker
```

For SHIFT events, this is the true next speaker. For HOLD events, this is the other speaker; a high score would incorrectly predict a shift.

Score variants written:

```text
paper_256      = p_silence[shift_speaker]
diag_p_now     = p_now[shift_speaker]
diag_p_future  = p_future[shift_speaker]
```

Only `paper_256` is the primary VAP-subset baseline score.

### 4.2 S-pred-clear

Purpose:

> Predict an upcoming clean paper-valid SHIFT while the current speaker is still active.

Positive examples:

```text
window_start = S/H-paper SHIFT silence_start - 500 ms
window_end   = S/H-paper SHIFT silence_start
```

The parent SHIFT must satisfy the full `S/H-paper` definition. This is intentional: `S-pred-clear` is not a count of all human turn changes; it is prediction before paper-valid clean gap shifts.

Negative examples:

```text
500 ms sampled windows from non-shift single-speaker regions
```

The code defines non-shift candidate frames by checking that the future 2-second horizon remains with the current/last speaker. Negative sampling is seeded by `project.seed` for reproducibility.

Labels:

```text
SHIFT_SOON = 1
sampled non-shift = 0
```

Scored speaker:

```text
target = next speaker for positives
target = other speaker for negatives
```

Score variants written:

```text
paper_256      = p_active[target]
diag_p_future  = p_future[target]
```

### 4.3 S-pred-overlap

Purpose:

> Audit rapid-gap and overlap speaker changes excluded by `S/H-paper`.

This task is exploratory and not paper-comparable.

Positive event templates:

```text
rapid gap speaker change: A -> silence(<150 ms) -> B, or B -> silence(<150 ms) -> A
overlap speaker change:   A -> overlap(<=150 ms) -> B, or B -> overlap(<=150 ms) -> A
```

The same strict 1-second pre/post speaker-control checks are applied where possible.

Positive window:

```text
500 ms before the rapid-gap next-speaker onset or overlap onset
```

Negative windows:

```text
sampled from the same non-shift single-speaker candidate pool as S-pred-clear
```

Labels:

```text
SHIFT_SOON = 1
sampled non-shift = 0
```

Score variants written:

```text
paper_256      = p_active[target]
diag_p_future  = p_future[target]
```

Important interpretation:

`paper_256` here means the same VAP active-region subset score used for shift prediction. The task itself is exploratory because the event definition is not the paper's clean shift definition.

### 4.4 BC-pred

Purpose:

> Predict an upcoming backchannel.

Backchannel event definition:

```text
candidate speaker has activity segment with duration >= 200 ms
candidate speaker has activity segment with duration <= 1.0 s
candidate speaker has no activity in previous 1.0 s
candidate speaker has no activity in following 2.0 s
last single speaker before candidate segment is the other speaker
candidate onset has enough past context and future horizon
```

Positive window:

```text
window_start = BC onset - 500 ms
window_end   = BC onset
```

Negative examples:

```text
500 ms sampled windows from non-shift regions
```

Unlike S-pred negatives, these are allowed to include listening/silence-type regions, because backchannels can be predicted while the future backchanneling speaker is not currently active.

Labels:

```text
BC_SOON = 1
sampled negative = 0
```

Scored speaker:

```text
positive = backchannel speaker
negative = listener/other speaker relative to current non-shift speaker
```

Score variants written:

```text
paper_256       = p_bc[backchannel_or_listener_speaker]
diag_approx_bc  = approximate previous BC score for the same speaker
```

### 4.5 S/L-paper

Purpose:

> At the onset of a speaker segment, decide whether it is SHORT/backchannel or LONG/proper shift.

Positive examples:

```text
SHORT = BC onset
```

Negative examples:

```text
LONG = S/H-paper SHIFT onset
```

Evaluation window:

```text
first 200 ms after onset
```

Labels:

```text
SHORT = 1
LONG  = 0
```

Scored speaker:

```text
speaker who just started speaking
```

Score variants written:

```text
paper_256       = p_bc[starting_speaker]
diag_approx_bc  = approximate previous BC score for the starting speaker
```

## 5. VAP score reconstruction

This is the part that was ambiguous in the notes. The audit now makes it explicit.

### 5.1 What VAP outputs

The discrete VAP model predicts a distribution over 256 possible future voice-activity states at each frame.

Each class corresponds to 8 binary bins:

```text
speaker A bin 0, bin 1, bin 2, bin 3
speaker B bin 0, bin 1, bin 2, bin 3
```

The bins represent a 2-second future projection window with four bins per speaker.

### 5.2 What `paper_256` means

`paper_256` is the branch's primary zero-shot score variant. It is computed by summing selected subsets of the 256 VAP classes.

The helper function returns:

```text
p_silence  -> used for S/H-paper
p_active   -> used for S-pred-clear and S-pred-overlap
p_bc       -> used for BC-pred and S/L-paper
```

### 5.3 S/H-paper score from 256 classes

For each candidate next speaker, the subset requires:

```text
candidate next speaker: active in the last two future bins; first two bins optional
other speaker: silent in all four bins
```

The score is conditional between the two candidate next-speaker subsets:

```text
p_silence[speaker] = P(subset_next_speaker=speaker)
                     / (P(subset_next_speaker=A) + P(subset_next_speaker=B))
```

S/H-paper writes:

```text
score = p_silence[shift_speaker]
```

### 5.4 S-pred score from 256 classes

For a candidate target next speaker, the shift subset requires:

```text
target speaker: active in the last two future bins; first two bins optional
current speaker: ending pattern allowed in early bins
```

The hold/no-shift alternative requires the current speaker to continue.

The score is:

```text
p_active[target] = P(active shift-to-target subset)
                   / (P(active shift-to-target subset) + P(active hold alternative))
```

This is used for:

```text
S-pred-clear
S-pred-overlap
```

### 5.5 BC score from 256 classes

For a candidate backchannel speaker:

```text
BC speaker: some activity in the first three future bins, no activity in final bin
main speaker: activity in final bin
```

The score is:

```text
p_bc[bc_speaker] = sum probability of BC-compatible 256-state classes
```

This is used for:

```text
BC-pred
S/L-paper
```

### 5.6 Diagnostic score variants

Diagnostic variants are retained to compare against previous logic:

```text
diag_p_now      exported VAP p_now helper
diag_p_future   exported VAP p_future helper
diag_approx_bc  previous approximate BC reconstruction
```

They are not the main paper-style baseline score.

## 6. Event row schema

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
score_speaker        speaker index used to read the score array
source_event         shift, hold, bc_pred_pos, etc.
event_id             event identifier within interaction
positive_class       SHIFT, SHIFT_SOON, BC_SOON, or SHORT
```

## 7. Score variants versus metric/calibration variants

These are different.

### Score variant

Question:

> Which numeric VAP score is being evaluated?

Examples:

```text
paper_256
diag_p_now
diag_p_future
diag_approx_bc
```

### Calibration objective / metric variant

Question:

> Which metric is used to choose a threshold on calibration data?

Examples:

```text
weighted_f1
macro_f1
positive_f1
balanced_accuracy
```

The code evaluates each `task × score_variant` separately. For each pair, it selects a primary threshold using `macro_f1`, then separately writes objective-audit rows showing what would happen if the threshold were selected by weighted F1, macro F1, positive F1, or balanced accuracy.

## 8. Metric outputs

Metrics are computed per:

```text
task × score_variant
```

This prevents `paper_256` and diagnostic scores from being mixed.

Primary threshold-selection objective:

```text
macro_f1
```

Additional objective audit:

```text
weighted_f1
macro_f1
positive_f1
balanced_accuracy
```

Important files:

```text
thresholds.csv                    primary threshold chosen by macro_f1
fold_metrics.csv                  metrics using that primary threshold
objective_threshold_metrics.csv   metrics using thresholds chosen by each objective
threshold_sweep.csv               full threshold sweep
event_level_metrics.csv           event-averaged score metrics
event_predictions_eval.csv        held-out rows with pred column
```

For paper-style baseline inspection, filter:

```text
score_variant == paper_256
```

For paper-comparable rows only, filter:

```text
paper_comparable == true
```

For exploratory overlap analysis, inspect:

```text
task == S-pred-overlap
```

but do not compare it to the VAP paper.

## 9. Attrition audit

`outputs/events/event_attrition.csv` is the sanity check for whether event definitions are too strict.

Important columns:

```text
raw_single_speaker_alternations   all A/B single-speaker changes after reducing VAD to dialogue-state runs
gap_shift_triads                  A-silence-B or B-silence-A templates before filters
rapid_gap_shift_triads            gap shift triads with silence < 150 ms
overlap_shift_triads              A-overlap-B or B-overlap-A templates
gap_shift_min_silence             gap shifts with silence >= 150 ms
gap_shift_pre_ok                  gap shifts passing 1 s pre-offset check
gap_shift_post_ok                 gap shifts passing 1 s post-onset check
gap_shift_pre_post_ok             gap shifts passing both pre and post checks
gap_shift_metric_context_ok       gap shifts passing silence metric window, past context, and future horizon
final_shift_events                final S/H-paper SHIFT events
final_hold_events                 final S/H-paper HOLD events
final_overlap_shift_events        final S-pred-overlap positive events before prediction-window validity
final_bc_events                   final BC events
```

How to interpret:

- If `raw_single_speaker_alternations` is high but `final_shift_events` is low, the paper-valid clean S/H definition is filtering many human-perceived turn trades.
- If `raw_single_speaker_alternations` is low, the VAD/dialogue-state representation may not reflect the turns heard in the audio.
- If `gap_shift_pre_post_ok` is high but `final_shift_events` is low, inspect metric-window/context filters or possible implementation issues.
- If `overlap_shift_triads` is high, then many turn changes are overlap/interruptive and should be analyzed under `S-pred-overlap`, not `S/H-paper`.

## 10. What to inspect first after a run

1. `outputs/events/event_attrition.csv`
   - Does the code see raw turn alternations?
   - Which filters remove them?

2. `outputs/events/event_summary.csv`
   - Are final paper shift/hold/backchannel counts plausible?

3. `outputs/events/score_subsets.json`
   - Are 256-state subset sizes as expected?

4. `outputs/metrics/dev_to_test/fold_metrics.csv`
   - Start with `score_variant == paper_256`.

5. `outputs/metrics/dev_to_test/objective_threshold_metrics.csv`
   - Compare threshold choices for weighted F1, macro F1, positive F1, and balanced accuracy.

6. `outputs/eda/event_errors/classification_error_by_source_dev_to_test.csv`
   - Which task/source has the highest error rate?

## 11. Known limitations and integrity cautions

1. This is a proxy VAP baseline on Seamless Interaction, not an exact rerun of the paper's original datasets.
2. Event counts are VAD-dependent because event definitions are computed from Seamless-provided VAD.
3. `S-pred-overlap` is exploratory and must not be reported as paper-comparable.
4. The primary threshold still uses macro F1 to match the previous audit notebook behavior; paper-style weighted-F1 threshold rows are preserved in the objective audit.
5. The 256-state subset scoring is an explicit, documented implementation of the VAP subset idea, but it still requires empirical sanity checks against audio and against the official repository behavior.
6. Before making research claims, inspect event attrition and manually audit a sample of event boundaries.
