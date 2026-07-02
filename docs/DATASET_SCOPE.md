# Dataset Scope for 300-Interaction Build

This build uses the Seamless Interaction labels `naturalistic` and `improvised`.

The build restricts interactions to vendor `V00` only. This is a stricter project-level audio-quality filter.

The independent data unit is `interaction_key`. Project splits, calibration folds, and future adaptation episodes must never split events/chunks from the same interaction across folds.

Raw audio may be deleted after a valid stereo WAV is constructed. VAD and transcript files are kept because VAD is needed for event extraction and evaluation.
