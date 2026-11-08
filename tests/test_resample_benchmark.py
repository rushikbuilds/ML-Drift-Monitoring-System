import numpy as np
import pandas as pd
import pytest

from dews import resample_benchmark as rb
from dews import v6_config as cfg


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = cfg.REFERENCE_ROWS + cfg.POOL_ROWS + 500
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "noise"])
    y = ((X["a"] + 0.5 * X["b"] + 0.5 * rng.normal(size=n)) > 0).astype(int).to_numpy()
    return X, y


def build(data, kind, seed=7):
    X, y = data
    return rb.build_episode(X, y, dataset="toy", kind=kind, seed=seed)


def test_shift_episodes_keep_true_labels(data):
    for kind in ("clean", "gradual_shift", "abrupt_shift"):
        ep = build(data, kind)
        for idx, lab in zip(ep.meta["sampled_pool_indices"], ep.labels):
            assert np.array_equal(ep.meta["pool_labels"][idx], lab)


def test_concept_change_flips_only_after_onset(data):
    ep = build(data, "concept_change")
    pool_labels = ep.meta["pool_labels"]
    for t, (idx, lab) in enumerate(zip(ep.meta["sampled_pool_indices"], ep.labels)):
        flipped = np.sum(pool_labels[idx] != lab)
        if t < ep.onset:
            assert flipped == 0
    late = ep.meta["sampled_pool_indices"][-1]
    assert np.sum(pool_labels[late] != ep.labels[-1]) > 0


def test_onset_and_shape(data):
    ep = build(data, "gradual_shift")
    assert cfg.ONSET_RANGE[0] <= ep.onset <= cfg.ONSET_RANGE[1]
    assert len(ep.batches) == cfg.N_BATCHES and all(len(b) == cfg.BATCH_SIZE for b in ep.batches)
    assert build(data, "clean").onset is None


def test_gradual_shift_moves_the_chosen_feature_only_after_onset(data):
    ep = build(data, "gradual_shift", seed=11)
    f, s = ep.meta["feature"], ep.meta["sign"]
    before = np.mean([b[f].mean() for b in ep.batches[:ep.onset]])
    after = np.mean([b[f].mean() for b in ep.batches[-5:]])
    assert s * (after - before) > 0.5


def test_episodes_are_deterministic(data):
    a, b = build(data, "abrupt_shift", seed=3), build(data, "abrupt_shift", seed=3)
    assert np.array_equal(a.batches[30].to_numpy(), b.batches[30].to_numpy())
    assert np.allclose(a.importance, b.importance)


def test_permutation_importance_ranks_the_noise_feature_low(data):
    ep = build(data, "clean")
    assert np.argmax(ep.importance) == 0 and ep.importance[3] < ep.importance[0]


def test_episode_plan_uses_disjoint_seeds():
    dev = {s for _, _, s in rb.episode_plan(cfg.DEVELOPMENT_DATASETS[0])}
    held = {s for _, _, s in rb.episode_plan(cfg.HELD_OUT_DATASETS[0])}
    assert len(dev) == sum(cfg.EPISODES_PER_DATASET.values()) and dev.isdisjoint(held)
