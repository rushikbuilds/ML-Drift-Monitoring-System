"""Tests for the adaptive threshold engine."""
from __future__ import annotations

from dews.adaptive_threshold import AdaptiveThreshold


def test_warmup_returns_initial_threshold() -> None:
    at = AdaptiveThreshold(warmup_batches=4, initial_threshold=0.30)
    # During warmup the fixed initial threshold is returned
    for i in range(3):
        assert at.update(0.10 + i * 0.01, batch_index=i) == 0.30
    assert not at.is_warmed_up


def test_finalization_computes_learned_threshold() -> None:
    at = AdaptiveThreshold(warmup_batches=4, sensitivity=2.0, initial_threshold=0.30, floor=0.0, ceiling=1.0)
    scores = [0.10, 0.12, 0.11, 0.13]
    for i, s in enumerate(scores):
        threshold = at.update(s, batch_index=i)
    # After 4 scores the warmup should be done
    assert at.is_warmed_up
    # Learned threshold = mean + 2*std, should be > mean(scores)
    assert threshold > 0.10


def test_post_warmup_ema_updates_threshold() -> None:
    at = AdaptiveThreshold(warmup_batches=3, sensitivity=2.0, ema_alpha=0.3, floor=0.0, ceiling=1.0)
    warmup_scores = [0.10, 0.12, 0.11]
    for i, s in enumerate(warmup_scores):
        at.update(s, batch_index=i)
    t1 = at.current_threshold
    # Feed a much higher score — threshold should increase
    at.update(0.50, batch_index=3)
    t2 = at.current_threshold
    assert t2 > t1


def test_floor_clamping() -> None:
    at = AdaptiveThreshold(warmup_batches=3, sensitivity=0.0, floor=0.20, ceiling=1.0)
    for i in range(3):
        at.update(0.01, batch_index=i)
    # sensitivity=0 means threshold = mean, which is 0.01 — but floor=0.20 should clamp it
    assert at.current_threshold >= 0.20


def test_ceiling_clamping() -> None:
    at = AdaptiveThreshold(warmup_batches=3, sensitivity=100.0, floor=0.0, ceiling=0.60)
    for i in range(3):
        at.update(0.50, batch_index=i)
    # Huge sensitivity would push threshold very high, but ceiling=0.60 clamps it
    assert at.current_threshold <= 0.60


def test_to_dict_serialization() -> None:
    at = AdaptiveThreshold(warmup_batches=2, sensitivity=2.5)
    at.update(0.10, batch_index=0)
    at.update(0.15, batch_index=1)
    state = at.to_dict()
    assert "current_threshold" in state
    assert "is_warmed_up" in state
    assert "scores_seen" in state
    assert state["scores_seen"] == 2


def test_history_snapshots_recorded() -> None:
    at = AdaptiveThreshold(warmup_batches=2)
    at.update(0.10, batch_index=0)
    at.update(0.15, batch_index=1)
    at.update(0.20, batch_index=2)
    assert len(at.history) == 3
    assert at.history[0].is_warmup is True
    assert at.history[2].is_warmup is False
