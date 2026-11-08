import numpy as np
import pandas as pd

from dews import resample_benchmark as rb
from dews import v6_config as cfg
from dews.v6_signals import extract_episode


def _episode():
    rng = np.random.default_rng(1)
    n = cfg.REFERENCE_ROWS + cfg.POOL_ROWS + 100
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = ((X["a"] + 0.3 * rng.normal(size=n)) > 0).astype(int).to_numpy()
    return rb.build_episode(X, y, dataset="toy", kind="gradual_shift", seed=5)


def test_monitor_signals_do_not_depend_on_stream_labels():
    ep = _episode()
    a = extract_episode(ep)
    ep.labels = [1 - lab for lab in ep.labels]  # change every streamed label
    b = extract_episode(ep)
    for key in ("signals", "aux", "rws", "mean_ks_raw"):
        assert np.allclose(a[key], b[key]), key
    assert not np.allclose(a["accuracy"], b["accuracy"])


def test_extraction_shapes_and_labeled_subsample():
    r = extract_episode(_episode())
    n = cfg.N_BATCHES
    assert r["signals"].shape[0] == r["aux"].shape[0] == len(r["rws"]) == len(r["accuracy"]) == n
    assert np.all((r["accuracy_obs20"] >= 0) & (r["accuracy_obs20"] <= 1))
    assert abs(r["accuracy_obs20"].mean() - r["accuracy"].mean()) < 0.05
