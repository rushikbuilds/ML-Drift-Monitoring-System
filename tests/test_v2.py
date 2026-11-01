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
