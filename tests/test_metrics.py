import pandas as pd
from adaptive_vap_space.metrics import choose_threshold, make_val_folds, metric_dict


def test_choose_threshold():
    t, f1 = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.1 <= t <= 0.9
    assert f1 >= 0.9


def test_make_val_folds_by_interaction():
    folds = make_val_folds(["a", "b", "c", "d", "e"], k=5, seed=1)
    assert set(folds.columns) == {"interaction_key", "fold", "split"}
    assert folds.groupby(["fold", "interaction_key"]).size().max() == 1


def test_metric_dict():
    df = pd.DataFrame({"label": [0, 1], "score": [0.1, 0.9]})
    m = metric_dict(df, 0.5)
    assert m["accuracy"] == 1.0
