# AI Collaboration Protocol

This repository is designed so AI assistance can be grounded in inspectable code and artifacts.

## Rules

1. Inspect the current file before claiming what it does.
2. Preserve `interaction_key` as the independent data unit.
3. Never split frames/events/chunks from the same interaction across train/val/test or calibration/evaluation folds.
4. Keep dataset paths portable and relative where possible.
5. Record every download attempt and every rejection reason.
6. Use public VAP `run.py` on full stereo WAVs for the baseline.
7. Save VAP outputs as `.json.gz` with corruption checks.
8. Choose thresholds only on calibration rows.
9. Do not add adaptive models until the baseline and EDA are stable.

## Debugging response format

When debugging, answer with:

1. What failed.
2. Why it failed, grounded in traceback/logs.
3. Whether data/artifacts are at risk.
4. Smallest safe fix.
5. Exact rerun command.
6. Verification command.
