"""Alerting rules on top of the v8 harm estimate, to reduce harmless-shift alarms (v10 branch).

All rules take the per-batch estimated accuracy drop (accuracy units, NaN during burn-in)
and return a score; an alert is score >= threshold. Every rule is causal.
"""
from __future__ import annotations

import numpy as np

BURN = 8


def raw(drop: np.ndarray, **_) -> np.ndarray:
    return drop


def ewma(drop: np.ndarray, alpha: float = 0.3, **_) -> np.ndarray:
    out = np.full(len(drop), np.nan)
    s = None
    for t, d in enumerate(drop):
        if np.isnan(d):
            continue
        s = d if s is None else alpha * d + (1 - alpha) * s
        out[t] = s
    return out


def rolling(drop: np.ndarray, k: int = 3, **_) -> np.ndarray:
    out = np.full(len(drop), np.nan)
    for t in range(len(drop)):
        window = drop[max(0, t - k + 1):t + 1]
        if np.isfinite(window).sum() == k:
            out[t] = window.mean()
    return out


def noise_scale(drop: np.ndarray, est: np.ndarray) -> float:
    """Robust batch-to-batch noise of the accuracy estimate, from the burn-in only."""
    base = est[:BURN]
    return max(1.4826 * float(np.median(np.abs(base - np.median(base)))), 0.003)


def zscore(drop: np.ndarray, est: np.ndarray = None, **_) -> np.ndarray:
    return drop / noise_scale(drop, est)


def cusum(drop: np.ndarray, est: np.ndarray = None, k: float = 0.5, **_) -> np.ndarray:
    """One-sided CUSUM on the standardized drop: S_t = max(0, S_{t-1} + z_t - k)."""
    z = drop / noise_scale(drop, est)
    out = np.full(len(drop), np.nan)
    s = 0.0
    for t, v in enumerate(z):
        if np.isnan(v):
            continue
        s = max(0.0, s + v - k)
        out[t] = s
    return out


RULES = {
    "raw (v8 rule)": (raw, {}),
    "ewma a=0.3": (ewma, {"alpha": 0.3}),
    "ewma a=0.5": (ewma, {"alpha": 0.5}),
    "rolling k=3": (rolling, {"k": 3}),
    "rolling k=5": (rolling, {"k": 5}),
    "z-score": (zscore, {}),
    "cusum k=0.5": (cusum, {"k": 0.5}),
    "cusum k=1.0": (cusum, {"k": 1.0}),
    "cusum k=1.5": (cusum, {"k": 1.5}),
}
