import numpy as np

from dews import v2


def test_state_target_marks_post_onset_batches_positive():
    acc = np.array([0.8] * 10 + [0.7] * 15)
    y = v2.state_targets(acc)
    assert y[4] == 0.0 or y[4] == 1.0
    assert y[5:10].tolist() == [1.0] * 5          # event starts inside horizon
    assert y[12] == 1.0                           # degradation continues: positive
    assert np.isnan(y[-1])                        # horizon not observable


def test_target_matures_after_horizon_confirmation_and_delay():
    assert v2.target_mature_batch(3, 0) == 3 + v2.H + 1 + 1
    assert v2.target_mature_batch(3, 5) == 3 + v2.H + 1 + 5


def test_label_features_use_only_matured_labels():
    acc = np.array([0.8] * 10 + [0.5] * 10)
    f2 = v2.label_features(acc, delay=2)
    assert f2[11, 0] == 0.0                       # batch 10 labels not yet usable
    assert f2[12, 0] > 0.2                        # usable once delay has passed
    f5 = v2.label_features(acc, delay=5)
    assert f5[14, 0] == 0.0 and f5[15, 0] > 0.2


def test_episode_scenarios_are_deterministic_and_event_free_when_clean():
    a = v2.build_episode("harmful_gradual", 7)
    b = v2.build_episode("harmful_gradual", 7)
    assert np.allclose(a.batches[20].to_numpy(), b.batches[20].to_numpy())
    assert a.onset is not None and v2.build_episode("clean", 7).onset is None


def test_cbpe_scores_use_only_matured_labels():
    rng = np.random.default_rng(0)
    n = 30
    signals = np.zeros((n, len(v2.BASE_SIGNALS)))
    signals[:, v2.BASE_SIGNALS.index("mean_confidence")] = 0.8 + 0.01 * rng.normal(size=n)
    acc = 0.8 + 0.01 * rng.normal(size=n)
    base = {"signals": signals, "accuracy": acc}
    for delay in (0, 2, 5):
        t, lag = 20, max(delay, 1)
        changed = acc.copy()
        changed[t - lag + 1:] = 0.1  # labels that have not arrived by time t
        for name in ("da_cbpe", "da_cbpe_forecast"):
            a = v2.cbpe_scores(base, delay)[name][t]
            b = v2.cbpe_scores({"signals": signals, "accuracy": changed}, delay)[name][t]
            assert a == b


def test_anchored_estimator_uses_only_matured_labels():
    from dews.anchored import AnchoredEstimator

    rng = np.random.default_rng(1)
    n = 30

    def make(seed_shift=0.0):
        r = np.random.default_rng(int(seed_shift * 100) + 5)
        return {"signals": r.normal(size=(n, len(v2.BASE_SIGNALS))), "aux": r.normal(size=(n, 4)),
                "accuracy": 0.8 + 0.02 * rng.normal(size=n)}

    train = [make(i) for i in range(4)]
    sig = make(9)
    for delay in (0, 2, 5):
        est = AnchoredEstimator(delay).fit(train)
        t, lag = 20, max(delay, 1)
        changed = dict(sig, accuracy=sig["accuracy"].copy())
        changed["accuracy"][t - lag + 1:] = 0.1
        assert est.estimate(sig)[t] == est.estimate(changed)[t]
