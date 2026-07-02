import gzip
import json

from adaptive_vap_space.vap_outputs import validate_vap_json, load_prediction_arrays


def test_validate_vap_json_gz_public_vap_shape(tmp_path):
    p = tmp_path / "x.json.gz"
    d = {
        "probs": [[[0.0] * 256]],
        "p_now": [[[0.1, 0.9]]],
        "p_future": [[[0.2, 0.8]]],
    }
    with gzip.open(p, "wt") as f:
        json.dump(d, f)

    ok, msg, meta = validate_vap_json(p)
    assert ok, msg
    assert meta["n_frames"] == 1
    assert meta["probs_dim"] == 256


def test_load_prediction_arrays_removes_public_vap_batch_dim(tmp_path):
    p = tmp_path / "x.json.gz"
    d = {
        "probs": [[[0.0] * 256, [1.0] * 256]],
        "p_now": [[[0.1, 0.9], [0.8, 0.2]]],
        "p_future": [[[0.2, 0.8], [0.7, 0.3]]],
    }
    with gzip.open(p, "wt") as f:
        json.dump(d, f)

    probs, p_now, p_future = load_prediction_arrays(p)
    assert probs.shape == (2, 256)
    assert p_now.shape == (2, 2)
    assert p_future.shape == (2, 2)
