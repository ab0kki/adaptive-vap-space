# Decisions

## 2026-07-01: Baseline uses full stereo WAVs

Decision: use one full stereo WAV per dyadic interaction as input to public VAP `run.py`.

Reason: the public VAP pretrained inference path accepts stereo waveform input. Pre-chunking into training windows is unnecessary for the frozen baseline.

Risk: long files may take longer to run.

Verification: `outputs/vap/vap_manifest.csv` should contain one row per stereo WAV and `probs_dim = 256`.

## 2026-07-01: Storage-safe staging

Decision: download candidate interactions into `_staging/`, keep only validated interactions, and delete rejected staging files by default.

Reason: avoids filling local/Kaggle storage with rejected audio.

Risk: rejected files must be re-downloaded if filters are changed.

Verification: rejected interactions appear in `manifests/rejected_interactions.csv`; staging is empty or absent after build.
