"""Delay-anchored accuracy estimation (final v4 method).

At time t, labels have arrived only for batches b <= t - max(L, 1). The estimator
anchors on the mean realized accuracy of the latest ``anchor_len`` such batches
and adds a linear correction predicted from the change, since the anchor, of
label-free signals: confidence statistics, drift statistics, and disagreement
with auxiliary models trained on the reference rows (RandomForest and
HistGradientBoosting; monitor-side only, the monitored classifier is untouched).
The correction model is a ridge regression in accuracy units, so it can be fit on
other streams and transferred.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG
from .v2 import BASE_SIGNALS, BASELINE_BATCHES, Episode

_IDX = {n: i for i, n in enumerate(BASE_SIGNALS)}
RIDGE_ALPHA = 50.0
ANCHOR_LEN = 3


def aux_signals(ep: Episode) -> np.ndarray:
    """Per batch: disagreement of the frozen model with RF and HGB, RF-vs-HGB, and their mean confidence."""
    lr = DEFAULT_EARLY_WARNING_CONFIG.classifier.build_pipeline().fit(ep.reference, ep.reference_labels)
    rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=0,
                                n_jobs=1).fit(ep.reference, ep.reference_labels)
    hgb = HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=0).fit(ep.reference, ep.reference_labels)
    rows = []
    for b in ep.batches:
        p0, p1, p2 = lr.predict(b), rf.predict(b), hgb.predict(b)
        conf = (rf.predict_proba(b).max(axis=1).mean() + hgb.predict_proba(b).max(axis=1).mean()) / 2
        rows.append([np.mean(p0 != p1), np.mean(p0 != p2), np.mean(p1 != p2), conf])
    return np.asarray(rows)


def signal_matrix(sig: Mapping[str, Any]) -> np.ndarray:
    s, a = np.asarray(sig["signals"]), np.asarray(sig["aux"])
    return np.column_stack([
        s[:, _IDX["mean_confidence"]], s[:, _IDX["mean_entropy"]], s[:, _IDX["low_confidence_rate"]],
        s[:, _IDX["positive_prediction_rate"]], s[:, _IDX["s_stat"]], s[:, _IDX["mean_ks"]],
        s[:, _IDX["drift_score"]], a[:, 0], a[:, 1], a[:, 2], a[:, 3]])


def delta_dataset(sig: Mapping[str, Any], delay: int, anchor_len: int = ANCHOR_LEN):
    """Signal changes since the anchor and the anchor accuracy (NaN until labels exist). Causal."""
    S, acc = signal_matrix(sig), np.asarray(sig["accuracy"])
    n, lag = len(acc), max(delay, 1)
    X = np.zeros((n, S.shape[1]))
    anchor = np.full(n, np.nan)
    for t in range(n):
        m = t - lag
        if m < BASELINE_BATCHES:
            continue
        lo = max(0, m - anchor_len + 1)
        anchor[t] = acc[lo:m + 1].mean()
        X[t] = S[t] - S[lo:m + 1].mean(axis=0)
    return X, anchor


def persist(x: np.ndarray, k: int = 2) -> np.ndarray:
    """Minimum over the last k values: mirrors the K-consecutive-batches event rule."""
    return np.array([np.min(x[max(0, t - k + 1):t + 1]) for t in range(len(x))])


class AnchoredEstimator:
    def __init__(self, delay: int) -> None:
        self.delay = delay
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))

    def fit(self, sigs: Sequence[Mapping[str, Any]]) -> AnchoredEstimator:
        xs, ys = [], []
        for s in sigs:
            X, anchor = delta_dataset(s, self.delay)
            ok = np.isfinite(anchor)
            xs.append(X[ok])
            ys.append((np.asarray(s["accuracy"]) - anchor)[ok])
        self.model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def estimate(self, sig: Mapping[str, Any]) -> np.ndarray:
        """Accuracy nowcast per batch; before labels exist, the clean-baseline accuracy."""
        X, anchor = delta_dataset(sig, self.delay)
        base = float(np.mean(sig["accuracy"][:BASELINE_BATCHES]))
        return np.where(np.isfinite(anchor), anchor + self.model.predict(X), base)

    def drop_scores(self, sig: Mapping[str, Any]) -> dict[str, np.ndarray]:
        base = float(np.mean(sig["accuracy"][:BASELINE_BATCHES]))
        drop = np.clip(base - self.estimate(sig), 0.0, None)
        return {"anchored": drop, "anchored_p2": persist(drop, 2)}
