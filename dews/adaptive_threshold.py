"""Adaptive threshold engine that dynamically adjusts ds_crit.

Instead of a fixed critical drift threshold, the adaptive engine learns what
"normal" drift looks like during an initial warm-up period, then sets the
threshold as:

    ds_crit_adaptive = mean(normal_scores) + sensitivity * std(normal_scores)

This significantly reduces false positives while preserving recall on real
drift events.  After the warm-up, the threshold can optionally continue to
adapt using an exponential moving average (EMA).

Usage:
    adaptive = AdaptiveThreshold(warmup_batches=6, sensitivity=2.5)
    for batch_idx, ds in enumerate(scores):
        threshold = adaptive.update(ds)
        if ds >= threshold:
            print("Drift detected!")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSnapshot:
    """Point-in-time snapshot of the adaptive threshold state."""
    batch_index: int
    threshold: float
    mean: float
    std: float
    is_warmup: bool


@dataclass
class AdaptiveThreshold:
    """Dynamically adjusts the drift score critical threshold.

    Parameters
    ----------
    warmup_batches : int
        Number of initial batches used to learn the "normal" drift
        score distribution.  During warm-up the fixed ``initial_threshold``
        is used.
    sensitivity : float
        Number of standard deviations above the mean to set the
        threshold.  Higher = fewer false positives but more missed drift.
    initial_threshold : float
        Threshold used during the warm-up phase and as a floor.
    ema_alpha : float
        Exponential moving average smoothing factor (0-1) for post-warmup
        threshold updates.  0 = no adaptation after warm-up, 1 = fully
        reactive to latest scores.
    floor : float
        Absolute minimum threshold.  Prevents the adaptive threshold from
        dropping so low that noise triggers alerts.
    ceiling : float
        Absolute maximum threshold.  Prevents runaway thresholds from
        masking real drift.
    """
    warmup_batches: int = 6
    sensitivity: float = 2.5
    initial_threshold: float = 0.30
    ema_alpha: float = 0.15
    floor: float = 0.15
    ceiling: float = 0.70
    # --- internal state ---
    _scores: list[float] = field(default_factory=list, repr=False)
    _warmup_mean: float = 0.0
    _warmup_std: float = 0.0
    _current_threshold: float = 0.0
    _ema_mean: float = 0.0
    _ema_var: float = 0.0
    _warmed_up: bool = False
    _history: list[ThresholdSnapshot] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._current_threshold = self.initial_threshold

    @property
    def current_threshold(self) -> float:
        return self._current_threshold

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def history(self) -> list[ThresholdSnapshot]:
        return list(self._history)

    def update(self, drift_score: float, batch_index: int = -1) -> float:
        """Ingest a new drift score and return the current adaptive threshold.

        During the warm-up phase the ``initial_threshold`` is returned.
        Once enough scores are collected, the threshold switches to the
        learned value and is continuously refined via EMA.
        """
        self._scores.append(drift_score)

        if not self._warmed_up:
            if len(self._scores) >= self.warmup_batches:
                self._finalize_warmup()
            else:
                self._record(batch_index, is_warmup=True)
                return self._current_threshold

        # Post-warmup: update EMA of mean and variance
        self._ema_mean = (1 - self.ema_alpha) * self._ema_mean + self.ema_alpha * drift_score
        diff_sq = (drift_score - self._ema_mean) ** 2
        self._ema_var = (1 - self.ema_alpha) * self._ema_var + self.ema_alpha * diff_sq

        ema_std = math.sqrt(max(self._ema_var, 1e-8))
        raw_threshold = self._ema_mean + self.sensitivity * ema_std
        self._current_threshold = max(self.floor, min(raw_threshold, self.ceiling))

        self._record(batch_index, is_warmup=False)
        return self._current_threshold

    def _finalize_warmup(self) -> None:
        """Compute baseline statistics from the warm-up window."""
        warmup_scores = self._scores[:self.warmup_batches]
        n = len(warmup_scores)
        self._warmup_mean = sum(warmup_scores) / n
        variance = sum((s - self._warmup_mean) ** 2 for s in warmup_scores) / max(n - 1, 1)
        self._warmup_std = math.sqrt(max(variance, 1e-8))
        self._ema_mean = self._warmup_mean
        self._ema_var = variance
        raw = self._warmup_mean + self.sensitivity * self._warmup_std
        self._current_threshold = max(self.floor, min(raw, self.ceiling))
        self._warmed_up = True
        logger.info(
            "Adaptive threshold warm-up complete: mean=%.4f std=%.4f threshold=%.4f",
            self._warmup_mean, self._warmup_std, self._current_threshold,
        )

    def _record(self, batch_index: int, is_warmup: bool) -> None:
        ema_std = math.sqrt(max(self._ema_var, 1e-8)) if self._warmed_up else 0.0
        self._history.append(ThresholdSnapshot(
            batch_index=batch_index,
            threshold=self._current_threshold,
            mean=self._ema_mean if self._warmed_up else 0.0,
            std=ema_std,
            is_warmup=is_warmup,
        ))

    def to_dict(self) -> dict:
        """Serialize the current state for API responses."""
        return {
            "current_threshold": round(self._current_threshold, 6),
            "is_warmed_up": self._warmed_up,
            "warmup_mean": round(self._warmup_mean, 6),
            "warmup_std": round(self._warmup_std, 6),
            "ema_mean": round(self._ema_mean, 6),
            "ema_std": round(math.sqrt(max(self._ema_var, 1e-8)), 6) if self._warmed_up else 0.0,
            "scores_seen": len(self._scores),
            "sensitivity": self.sensitivity,
            "floor": self.floor,
            "ceiling": self.ceiling,
        }
