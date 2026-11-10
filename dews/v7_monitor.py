"""v7 stream-normalized two-tier monitor.

Every signal is expressed in robust standard deviations relative to the stream's own
burn-in period (the first batches after deployment), so a threshold such as
"4 deviations above normal" means the same thing on every dataset.

* WATCH   = maximum z-score over label-free signals: "the inputs or the model's behaviour drifted".
* WARNING = logistic calibrator on z-scores plus the observed (delayed) label drop:
            "this drift is likely hurting accuracy".
Scores before the end of the burn-in are NaN (no alerts).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import v2
from . import v7_config as cfg
from .v6_eval import observed_accuracy, state_target

_IDX = {n: i for i, n in enumerate(v2.BASE_SIGNALS)}


def robust_z(x: np.ndarray, burn: int = cfg.BURN_IN_BATCHES, floor: float = cfg.SCALE_FLOOR) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.median(x[:burn])
    scale = max(1.4826 * np.median(np.abs(x[:burn] - med)), floor)
    return (x - med) / scale


def raw_signals(ep: Mapping[str, Any], delay: int, label_fraction: float) -> dict[str, np.ndarray]:
    s, a = np.asarray(ep["signals"], float), np.asarray(ep["aux"], float)
    view = {"signals": s, "accuracy": observed_accuracy(ep, label_fraction)}
    pos = s[:, _IDX["positive_prediction_rate"]]
    return {
        "drift_score": s[:, _IDX["drift_score"]],
        "mean_ks": np.asarray(ep["mean_ks_raw"], float),
        "rws": np.asarray(ep["rws"], float),
        "disagreement": a[:, :2].mean(axis=1),
        "confidence_drop": -s[:, _IDX["mean_confidence"]],
        "entropy": s[:, _IDX["mean_entropy"]],
        "positive_rate_shift": np.abs(pos - pos[: cfg.BURN_IN_BATCHES].mean()),
        "cbpe_drop": v2.cbpe_scores(view, delay)["da_cbpe"],
        "label_drop": v2.baseline_scores(view, delay)["delayed_label_only"],
    }


def _mask_burn_in(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    x[: cfg.BURN_IN_BATCHES] = np.nan
    return x


def watch_score(ep: Mapping[str, Any], delay: int, label_fraction: float) -> np.ndarray:
    r = raw_signals(ep, delay, label_fraction)
    z = np.column_stack([robust_z(r[k]) for k in cfg.WATCH_SIGNALS])
    return _mask_burn_in(z.max(axis=1))


def warning_features(ep: Mapping[str, Any], delay: int, label_fraction: float) -> np.ndarray:
    r = raw_signals(ep, delay, label_fraction)
    z = np.column_stack([robust_z(r[k]) for k in cfg.WATCH_SIGNALS] + [robust_z(r["cbpe_drop"]), r["label_drop"]])
    mean3 = np.vstack([z[max(t - 2, 0):t + 1].mean(axis=0) for t in range(len(z))])
    return np.hstack([z, mean3])


class WarningCalibrator:
    def __init__(self, delay: int, label_fraction: float) -> None:
        self.delay, self.label_fraction = delay, label_fraction
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))

    def fit(self, episodes: Sequence[Mapping[str, Any]]) -> WarningCalibrator:
        xs, ys = [], []
        for ep in episodes:
            X, y = warning_features(ep, self.delay, self.label_fraction), state_target(ep)
            keep = np.isfinite(y) & (np.arange(len(y)) >= cfg.BURN_IN_BATCHES)
            xs.append(X[keep])
            ys.append(y[keep])
        self.model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def score(self, ep: Mapping[str, Any]) -> np.ndarray:
        return _mask_burn_in(self.model.predict_proba(warning_features(ep, self.delay, self.label_fraction))[:, 1])
