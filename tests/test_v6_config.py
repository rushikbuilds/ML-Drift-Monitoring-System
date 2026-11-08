import numpy as np

from dews import v6_config as c


def test_held_out_split_matches_recorded_draw():
    drawn = sorted(np.random.default_rng(c.SPLIT_SEED).choice(sorted(c.ALL_DATASETS), 5, replace=False).tolist())
    assert tuple(drawn) == c.HELD_OUT_DATASETS
    assert set(c.DEVELOPMENT_DATASETS).isdisjoint(c.HELD_OUT_DATASETS)
    assert set(c.DEVELOPMENT_DATASETS) | set(c.HELD_OUT_DATASETS) == set(c.ALL_DATASETS)


def test_episode_design_fits_the_protocol():
    assert c.REFERENCE_ROWS > c.IMPORTANCE_HOLDOUT_ROWS
    assert c.POOL_ROWS >= c.BATCH_SIZE * c.N_BATCHES
    latest_confirmation = c.confirmation_batch(c.ONSET_RANGE[1] + c.RAMP_BATCHES, max(c.LABEL_DELAYS))
    assert latest_confirmation < c.N_BATCHES
    assert c.ONSET_RANGE[0] >= c.WARMUP_BATCHES


def test_seed_ranges_do_not_overlap():
    per_dataset = sum(c.EPISODES_PER_DATASET.values())
    dev_max = c.DEV_SEED_BASE + 100 * (len(c.ALL_DATASETS) - 1) + per_dataset
    assert per_dataset < 100 and dev_max < c.HELD_OUT_SEED_BASE


def test_confirmation_uses_causal_label_rule():
    assert c.confirmation_batch(20, 0) == 22
    assert c.confirmation_batch(20, 5) == 26


def test_hypotheses_are_complete():
    assert [h.id for h in c.HYPOTHESES] == ["H1", "H2", "H3", "H4", "H5"]
    assert c.PRIMARY["label_delay"] in c.LABEL_DELAYS
