# Event Extraction and VAP Baseline Audit

## Purpose

This branch resets the event-extraction and evaluation pipeline for the pretrained discrete VAP baseline. The goal is not to train adaptation yet. The goal is to create a transparent, auditable baseline that can later support intra-interaction or speaker-level adaptation experiments.

The repository already supports a 300-interaction Seamless datastore and already stores VAP outputs from the pretrained model under `outputs/vap`. The intended audit workflow is to reuse those existing VAP `.json.gz` files, rerun event extraction, rerun calibrated metrics, and rerun EDA in a new Kaggle notebook.

## Source grounding

The implementation is grounded in:

1. The VAP paper, which evaluates zero-shot turn-taking projections for S/H, S-pred, BC-pred, and S/L using the discrete 256-state VAP model.
2. The `VoiceActivityProjection` repository, which runs the pretrained VAP model and exports `probs`, `p_now`, and `p_future`.
3. The `vap_turn_taking` event and metric code, which defines dialog states, shift/hold/backchannel masks, non-shift negatives, and VAP score subsets.

## Input representation

For each interaction, the pipeline reads:

- Seamless-provided speaker A VAD JSONL.
- Seamless-provided speaker B VAD JSONL.
- VAP prediction arrays:
  - `probs`: frame-level 256-state VAP distribution.
  - `p_now`: exported near-future/current next-speaker diagnostic.
  - `p_future`: exported later-future next-speaker diagnostic.

The repository does not generate its own VAD. It uses the VAD metadata provided by Seamless Interaction.

The VAD segments are converted to a frame-level two-speaker activity matrix at 50 Hz:

    va[t, 0] = speaker A active at frame t
    va[t, 1] = speaker B active at frame t

One frame is 20 ms.

## Dialog states

The frame-level VAD is converted into VAP-style dialog states:

    0 = only speaker A
    1 = silence
    2 = overlap
    3 = only speaker B

Contiguous dialog-state runs are used to extract event candidates.

## Paper-comparable versus exploratory tasks

The branch separates paper-comparable tasks from exploratory adaptation-motivated tasks.

Paper-comparable tasks:

- `S/H-paper`
- `S-pred-clear`
- `BC-pred`
- `S/L-paper`

Exploratory task:

- `S-pred-overlap`

`S-pred-overlap` is intentionally not paper-comparable. It exists to audit rapid-gap and overlap speaker changes that matter for HRI/adaptation but are excluded from the paper's clean mutual-silence S/H definition.

## S/H-paper

`S/H-paper` evaluates SHIFT versus HOLD during mutual silence.

Templates:

    A -> silence -> B = SHIFT
    B -> silence -> A = SHIFT
    A -> silence -> A = HOLD
    B -> silence -> B = HOLD

Filters:

- Minimum mutual silence: 150 ms.
- Pre-offset: 1 second of only the previous speaker before the silence.
- Post-onset: 1 second of only the next speaker after onset.
- Evaluation starts 50 ms into the silence.
- Evaluation lasts 100 ms.
- The frame must have at least 3 seconds of past context.
- The frame must have a full 2-second VAP future horizon.

Label convention:

    SHIFT = 1
    HOLD = 0

The score is the probability of SHIFT, meaning the probability that the other speaker will be the next speaker. The primary score variant is `paper_256`.

## S-pred-clear

`S-pred-clear` evaluates upcoming SHIFT prediction while a speaker is still active.

Positive examples:

    500 ms before an `S/H-paper` SHIFT silence.

Negative examples:

    500 ms windows sampled from non-shift single-speaker regions where the future 2-second horizon remains with the current speaker and is far from future activity of the other speaker.

Label convention:

    SHIFT_SOON = 1
    sampled non-shift window = 0

Primary score:

    `paper_256`, using the 256-state active-region VAP subset score.

## S-pred-overlap

`S-pred-overlap` is exploratory.

It captures speaker-change events not included in `S/H-paper`:

- rapid gap speaker changes with less than 150 ms silence;
- overlap speaker changes with up to 150 ms overlap.

It uses the same 500 ms prediction window before the new-speaker/overlap event. It uses strict 1 second pre/post single-speaker checks when possible.

This task is useful for adaptation/HRI analysis but should not be compared to the VAP paper's S-pred numbers.

## BC-pred

`BC-pred` evaluates prediction of upcoming backchannels.

A backchannel event is a short isolated listener activity segment:

- minimum duration: 200 ms;
- maximum duration: 1 second;
- no activity by the same speaker in the previous 1 second;
- no activity by the same speaker in the following 2 seconds;
- the last speaker before the segment was the other speaker.

Positive examples:

    500 ms before BC onset.

Negative examples:

    sampled non-shift windows. These may include non-active regions because a backchannel can be predicted during listening/silence.

Primary score:

    `paper_256`, using the 256-state backchannel subset score.

Diagnostic score:

    `diag_approx_bc`, retained only for comparison with the previous branch.

## S/L-paper

`S/L-paper` evaluates SHORT versus LONG at onset.

Positive examples:

    SHORT = BC onset.

Negative examples:

    LONG = `S/H-paper` SHIFT onset.

Evaluation region:

    first 200 ms after onset.

Label convention:

    SHORT = 1
    LONG = 0

Primary score:

    `paper_256`, using the 256-state backchannel subset score.

## Zero-shot score variants

The branch writes multiple score variants to make the audit explicit.

Primary paper-style score:

- `paper_256`: reconstructed from the 256-class VAP probability vector using task-specific class subsets.

Diagnostic scores:

- `diag_p_now`
- `diag_p_future`
- `diag_approx_bc`

Only `paper_256` should be used for paper-comparable baseline claims. Diagnostic scores are included to understand how exported VAP helper arrays differ from the explicit 256-state subsets.

Rows include:

    task
    task_family
    score_variant
    paper_comparable
    source_event
    positive_class
    event_id
    speaker
    score_speaker
    frame
    time_s
    label
    score

## Attrition audit

Every extraction run writes:

    outputs/events/event_attrition.csv

This file explains how raw dialogue transitions are filtered into final event definitions. It includes counts for:

- raw single-speaker alternations;
- gap shift triads;
- rapid gap shift triads;
- overlap shift triads;
- gap shifts passing minimum silence;
- gap shifts passing pre/post checks;
- gap shifts passing metric/context checks;
- final paper SHIFT/HOLD counts;
- final exploratory overlap-shift counts;
- final BC counts.

This audit is required because paper-valid S/H shifts are not equivalent to all conversational turn trades heard in the audio.

## Metrics

Metrics are computed per:

    task × score_variant

This prevents paper-style scores and diagnostic scores from being mixed.

The primary threshold-selection objective remains:

    macro_f1

The repository also writes objective-threshold audits for:

- weighted_f1
- macro_f1
- positive_f1
- balanced_accuracy

This preserves the previous audit notebook behavior while allowing paper-style weighted-F1 thresholds to be inspected from `objective_threshold_metrics.csv`.

Evaluation outputs include:

- `outputs/metrics/<protocol>/thresholds.csv`
- `outputs/metrics/<protocol>/fold_metrics.csv`
- `outputs/metrics/<protocol>/event_level_metrics.csv`
- `outputs/metrics/<protocol>/threshold_sweep.csv`
- `outputs/metrics/<protocol>/objective_threshold_metrics.csv`
- `outputs/metrics/<protocol>/event_predictions_eval.csv`
- `outputs/metrics/<protocol>/task_metrics.csv`

## EDA

EDA preserves `score_variant` where present, so paper-style and diagnostic scores are not silently mixed.

Outputs include:

- `outputs/eda/dataset`
- `outputs/eda/vap_errors`
- `outputs/eda/event_errors`

## Known limitations

1. The repository is still a proxy evaluation on Seamless Interaction rather than the exact datasets used in the VAP paper.
2. Seamless VAD quality directly affects event counts.
3. `S-pred-overlap` is exploratory and must not be reported as paper-comparable.
4. Threshold-calibrated metrics are diagnostic; paper-comparison should prioritize `paper_256` and inspect weighted-F1 objective rows.
5. The next audit should visually inspect high-attrition clips and verify a sample of event boundaries against audio.
