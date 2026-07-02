# Evaluation Spec

Evaluation follows VAP-paper-style tasks as closely as possible while using a frozen pretrained public VAP baseline.

## Tasks

- `S/H`: shift vs hold during mutual silence. Positive class is SHIFT.
- `S-pred`: prediction of an upcoming shift. Positive class is SHIFT_SOON.
- `BC-pred`: prediction of an upcoming backchannel. Positive class is BC_SOON.
- `S/L`: short vs long next activity. Current implementation uses LONG as positive class by default.

## Threshold calibration

For `val_5fold`, folds are created by `interaction_key` only. For each task/fold, thresholds are selected on calibration folds and evaluated on the held-out fold.

For `dev_to_test`, thresholds are selected on `val` and evaluated on `test`.

## Audit note

BC-pred may require reconstructing a probability from the 256 VAP codebook states. This is documented in code and should be audited before publication.
