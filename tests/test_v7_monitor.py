import numpy as np

from dews import v7_config as cfg
from dews import v7_monitor as mon
from tests.test_v6_eval import _episode


def test_robust_z_uses_only_burn_in():
    x = np.r_[np.full(cfg.BURN_IN_BATCHES, 1.0), np.full(10, 5.0)]
    z = mon.robust_z(x)
    assert np.allclose(z[: cfg.BURN_IN_BATCHES], 0.0) and np.allclose(z[cfg.BURN_IN_BATCHES:], 4.0 / cfg.SCALE_FLOOR)


def test_scores_are_masked_during_burn_in():
    a = mon.watch_score(_episode(), 5, 1.0)
    assert np.all(np.isnan(a[: cfg.BURN_IN_BATCHES]))
    assert np.all(np.isfinite(a[cfg.BURN_IN_BATCHES:]))


def test_warning_features_ignore_labels_that_have_not_arrived():
    ep = _episode()
    t = 25
    for delay in (0, 2, 5, 10):
        changed = dict(ep, accuracy=ep["accuracy"].copy())
        changed["accuracy"][t - max(delay, 1) + 1:] = 0.0
        assert np.allclose(mon.warning_features(ep, delay, 1.0)[t], mon.warning_features(changed, delay, 1.0)[t])
