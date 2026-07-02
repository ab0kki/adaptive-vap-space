# Local Setup

This repo uses a local `.venv` environment inside the repository.

## Folder meanings

- `.venv/`: the Python environment to activate for this repo.
- `python310/`: local Python 3.10 used to create `.venv`; keep it if `.venv/pyvenv.cfg` references it.
- `.vscode/`: VS Code settings telling VS Code to use `.venv/bin/python`.
- `data/`: downloaded and processed dataset files.
- `outputs/`: VAP predictions, metrics, and EDA outputs.
- `external/`: cloned external repositories such as VoiceActivityProjection and Seamless Interaction.

## Start the repo in a new terminal

```bash
cd ~/Desktop/adaptive-vap-space
source .venv/bin/activate
which python
python --version
```

Expected Python path:

```text
/Users/abbeybrown/Desktop/adaptive-vap-space/.venv/bin/python
Python 3.10.x
```

Do not run `conda activate adaptive-vap-space` for normal repo use.
Do not activate `python310/` directly.

## Check environment

```bash
python scripts/01_check_env.py --config configs/smoke20.yaml
pytest -q
```

## Run the 20-interaction smoke pipeline

```bash
python scripts/20_run_vap.py --config configs/smoke20.yaml
python scripts/30_extract_events.py --config configs/smoke20.yaml
python scripts/40_eval_calibrated.py --config configs/smoke20.yaml --protocol val_5fold
python scripts/50_eda_dataset.py --config configs/smoke20.yaml
python scripts/51_eda_vap_errors.py --config configs/smoke20.yaml
python scripts/52_eda_event_errors.py --config configs/smoke20.yaml
```
## VAP audio dependency note

The public VAP runner uses `torchaudio.load`. On current torchaudio versions this may require `torchcodec`, so `torchcodec` is included in `pyproject.toml`.

