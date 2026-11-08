import numpy as np

from dews import v6_config as cfg
from dews import v6_eval as ev


def _episode(kind="gradual_shift", category="harmful", onset=12, event=20, seed=0):
    rng = np.random.default_rng(seed)
    n = cfg.N_BATCHES
    acc = np.full(n, 0.8) + 0.01 * rng.normal(size=n)
    if event is not None:
        acc[event:] = 0.6
    return {"kind": kind, "category": category, "onset": onset, "event_onset": event,
            "signals": rng.normal(size=(n, 8)), "aux": rng.random(size=(n, 4)), "rws": rng.random(n),
            "mean_ks_raw": rng.random(n), "accuracy": acc, "accuracy_obs20": acc.copy(), "dataset": "toy", "seed": seed}


def test_system_features_ignore_labels_that_have_not_arrived():
    ep = _episode()
    t = 25
    for delay in cfg.LABEL_DELAYS:
        changed = dict(ep, accuracy=ep["accuracy"].copy())
        changed["accuracy"][t - max(delay, 1) + 1:] = 0.0
        a = ev.system_features(ep, delay, 1.0)[t]
        b = ev.system_features(changed, delay, 1.0)[t]
        assert np.allclose(a, b)


def test_tier_windows_and_metrics():
    eps = [_episode(), _episode("clean", "clean", None, None, 1), _episode("abrupt_shift", "harmless", 12, None, 2)]
    m = ev.build_masks(eps, delay=5)
    assert m.t1[0, 12] and m.t1[0, 19] and not m.t1[0, 20]
    assert m.t2[0, cfg.confirmation_batch(20, 5) - 1] and not m.t2[0, cfg.confirmation_batch(20, 5)]
    assert m.clean[1, cfg.WARMUP_BATCHES:].all() and not m.clean[0, 12]
    scores = np.zeros((3, cfg.N_BATCHES))
    scores[0, 17] = 1.0      # T1 warning, lead 3
    scores[2, 30] = 1.0      # alert after a harmless shift
    out = ev.metrics(scores, 0.5, m)
    assert out["gradual"]["t1_share"] == 1.0 and out["gradual"]["t1_median_lead"] == 3
    assert out["harmless_alert_share"] == 1.0 and out["false_alerts_per_100_clean"] == 0.0


def test_spurious_episodes_are_excluded():
    m = ev.build_masks([_episode(category="spurious")], delay=2)
    assert not m.clean.any() and not m.harmful_gradual.any()


def test_threshold_respects_false_alert_budget():
    eps = [_episode(seed=i) for i in range(3)] + [_episode("clean", "clean", None, None, 10 + i) for i in range(3)]
    m = ev.build_masks(eps, delay=2)
    rng = np.random.default_rng(0)
    scores = rng.random((len(eps), cfg.N_BATCHES))
    thr = ev.select_threshold(scores, m)
    assert ev.metrics(scores, thr, m)["false_alerts_per_100_clean"] <= cfg.FALSE_ALERT_BUDGET
