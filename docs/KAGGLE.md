# Kaggle Guide

## Dataset build notebook, CPU

```bash
git clone https://github.com/<your-user>/adaptive-vap-space.git
cd adaptive-vap-space
python -m pip install -e ".[dev]"
git clone https://github.com/facebookresearch/seamless_interaction.git /kaggle/working/seamless_interaction
python scripts/10_build_datastore.py --config configs/kaggle.yaml
```

Save `/kaggle/working/datastore` as a private Kaggle Dataset.

## VAP baseline notebook, GPU

Attach your saved datastore dataset. Then:

```bash
git clone https://github.com/<your-user>/adaptive-vap-space.git
cd adaptive-vap-space
python -m pip install -e ".[dev]"
git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git /kaggle/working/VoiceActivityProjection
python scripts/20_run_vap.py --config configs/kaggle.yaml
python scripts/30_extract_events.py --config configs/kaggle.yaml
python scripts/40_eval_calibrated.py --config configs/kaggle.yaml --protocol val_5fold
python scripts/50_eda_dataset.py --config configs/kaggle.yaml
```

If your datastore is attached under `/kaggle/input/...`, edit `dataset.root` in a copied config or pass a config with that path.
