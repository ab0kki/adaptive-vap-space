# Datastore Design

The datastore is the canonical local/Kaggle dataset artifact.

```text
data/datastore/
  raw/audio/          # optional after stereo is built
  raw/vad/            # required for event extraction
  raw/transcript/     # optional
  processed/stereo_wav/
  manifests/
  reports/
```

## Storage safety

The builder downloads one interaction into `_staging/<interaction_key>/`, validates it, and then either:

- moves kept files into `raw/` and writes a stereo WAV, or
- logs rejection reasons and deletes staging files.

This avoids retaining many rejected candidate audio files.

## Can raw audio be deleted?

Yes, after `processed/stereo_wav/*.wav` has been built and checked, the raw mono audio is not needed for VAP inference. It is needed only if you want to rebuild stereo without redownloading.

Run:

```bash
python scripts/70_cleanup_storage.py --config configs/datastore.yaml --delete-raw-audio
```

Do not delete VAD unless you do not need to re-run event extraction.
