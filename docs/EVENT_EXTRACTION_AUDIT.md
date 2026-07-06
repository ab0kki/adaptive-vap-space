# Event Extraction Audit

## Purpose

This document describes how this repository extracts VAP-style zero-shot turn-taking events from two-speaker VAD and how those event rows are scored and evaluated.

The goal of this branch is to replace the earlier ad hoc event logic with a higher-fidelity implementation based on the VAP paper and the VAP event repository.

## Source grounding

The VAP paper evaluates future voice activity predictions using four zero-shot tasks:

1. S/H: SHIFT versus HOLD during mutual silence.
2. S-pred: prediction of an upcoming SHIFT before the end of the current speaker turn.
3. BC-pred: prediction of an upcoming backchannel.
4. S/L: SHORT versus LONG at speaker onset.

The VAP event repository implements event extraction through dialog-state masks, including shift, hold, short, long, predict-shift positive/negative, and predict-backchannel positive/negative masks.

## Input representation

For each interaction, the pipeline reads:

- Speaker A VAD segments.
- Speaker B VAD segments.
- VAP prediction arrays:
  - `probs`: 256-state VAP distribution.
  - `p_now`: exported current-speaker probability.
  - `p_future`: exported future-speaker probability.

The VAD segments are converted into a frame-level two-speaker activity array:

    va[t, 0] = speaker A active at frame t
    va[t, 1] = speaker B active at frame t

The repository uses 50 Hz frame rate, so one frame is 20 ms.

## Dialog states

The frame-level VAD is converted into dialog states:

    0 = only speaker A
    1 = silence
    2 = overlap
    3 = only speaker B

Contiguous dialog-state runs are then used to find turn-taking event templates.

## HOLD filling

The VAP event repository fills same-speaker hold pauses before matching event templates. This repository mirrors that idea for pre/post checks.

The relevant hold templates are:

    A -> silence -> A
    B -> silence -> B

For these templates, the silence is treated as belonging to the same speaker when checking whether the surrounding region is controlled by one speaker.

## S/H extraction

S/H events are extracted from mutual-silence templates:

    A -> silence -> B = SHIFT
    B -> silence -> A = SHIFT
    A -> silence -> A = HOLD
    B -> silence -> B = HOLD

The event must satisfy:

- 1 second of only the previous speaker before the silence.
- 1 second of only the next speaker after the silence.
- Enough silence to support the metric window.
- At least 3 seconds of past context.
- Enough future context for the 2 second VAP horizon.

The S/H evaluation window follows the VAP paper description:

    start = silence_start + 50 ms
    duration = 100 ms

S/H labels:

    SHIFT = 1
    HOLD = 0

S/H score:

    score = p_future[other speaker] / (p_future[previous speaker] + p_future[other speaker])

This replaces the earlier diagnostic baseline that used `p_now` for the candidate shift speaker during silence.

Remaining limitation: this is not yet the exact Discrete VAP 256-state subset score from the paper. It is a future-speaker diagnostic score using the exported VAP arrays.

## S-pred extraction

S-pred positives are extracted from valid SHIFT events.

The positive window is:

    500 ms before the SHIFT silence

S-pred negatives are sampled from non-shift regions where the current speaker remains dominant in the future horizon.

S-pred labels:

    SHIFT_SOON = 1
    sampled non-shift window = 0

S-pred score:

    score = p_future[target next speaker]

S-pred is not the full turn-shift event. It is the pre-shift prediction window before a valid SHIFT.

## Backchannel extraction

A backchannel candidate is a short isolated listener activity segment.

Conditions:

- Segment duration is at least 200 ms.
- Segment duration is at most 1 second.
- The same speaker has no activity in the 1 second before the segment.
- The same speaker has no activity in the 2 seconds after the segment.
- The last speaker before the segment was the other speaker.
- The event has enough past context and future horizon for scoring.

## BC-pred extraction

BC-pred positives are extracted from valid backchannel events.

The positive window is:

    500 ms before BC onset

BC-pred negatives are sampled from non-shift regions. Unlike S-pred negatives, BC-pred negatives may include non-active or silence regions because a backchannel can be predicted while listening.

BC-pred labels:

    BC_SOON = 1
    sampled negative window = 0

BC-pred score:

    score = p_bc[target backchannel speaker]

Remaining limitation: `p_bc` is reconstructed from the 256-state VAP probability vector using the repository's approximate helper. It is not yet an exact audited VAP class-subset score.

## S/L extraction

S/L evaluates the first 200 ms after a speaker starts speaking.

Positive SHORT examples:

    BC onset = 1

Negative LONG examples:

    SHIFT onset = 0

This fixes the earlier polarity issue where LONG had been treated as the positive class.

S/L score:

    score = p_bc[speaker who just started]

The score should be high for SHORT backchannel onsets and low for LONG shift onsets.

## Negative sampling

The VAP repository samples negatives from non-shift candidate regions. This branch also samples negatives from candidate non-shift regions, but uses a fixed random seed so the audit is reproducible.

## Calibration and metrics

All tasks use the same primary threshold calibration objective:

    macro_f1

Thresholds are selected on calibration data only. The held-out evaluation split is then evaluated with that selected threshold.

Macro F1 is used instead of weighted F1 because weighted F1 can hide majority-class dominance, especially for S/H.

The repository also writes an objective-threshold audit for:

- weighted_f1
- macro_f1
- positive_f1
- balanced_accuracy

This makes threshold effects visible and auditable.

## Main outputs

Event extraction writes:

- `outputs/events/event_predictions.csv`
- `outputs/events/event_summary.csv`

Evaluation writes:

- `outputs/metrics/<protocol>/thresholds.csv`
- `outputs/metrics/<protocol>/fold_metrics.csv`
- `outputs/metrics/<protocol>/event_level_metrics.csv`
- `outputs/metrics/<protocol>/threshold_sweep.csv`
- `outputs/metrics/<protocol>/objective_threshold_metrics.csv`
- `outputs/metrics/<protocol>/event_predictions_eval.csv`
- `outputs/metrics/<protocol>/task_metrics.csv`

EDA writes dataset-level and error-analysis outputs under:

- `outputs/eda/dataset`
- `outputs/eda/vap_errors`
- `outputs/eda/event_errors`

## Known limitations

1. Exact Discrete VAP 256-state subset scorers are not yet implemented.
2. `p_bc` remains approximate.
3. Event extraction is VAD-dependent, so strict VAD segmentation can still reduce the number of valid SHIFT or BC events.
4. This branch should be evaluated before merging to main.