# Adding Prototype Models Later

Future adaptive models should read stable artifacts instead of redoing data processing:

- `data/datastore/manifests/project_splits.csv`
- `outputs/events/event_predictions.csv`
- `outputs/vap/vap_manifest.csv`

Recommended future pattern:

```text
scripts/80_train_adapter.py
src/adaptive_vap_space/models/<model>.py
outputs/models/<experiment_name>/
```

Keep interaction boundaries intact. For within-interaction adaptation, define an early context region and evaluate later frames from the same interaction without leaking future labels into calibration.
