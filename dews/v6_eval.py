"""Scores, the primary risk calibrator, tier metrics and threshold selection for v6.

Implements docs/v6_protocol.md sections 5-6. All monitor inputs at batch t are causal:
label-derived features use only observed labels of batches b <= t - max(L, 1).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from . import v2
from . import v6_config as cfg

METHODS = ("primary_system", "fixed_dews_score", "statistical_only", "confidence_only",
           "mean_ks", "label_trend", "delay_aware_cbpe")
HARM_WINDOW = 3  # batches after onset used for the H3 harm score


def observed_accuracy(ep: Mapping[str, Any], label_fraction: float) -> np.ndarray:
    return np.asarray(ep["accuracy"] if label_fraction >= 1.0 else ep["accuracy_obs20"], float)


def _rolling(x: np.ndarray) -> np.ndarray:
    """current, 3-batch mean and 3-batch slope for each column of x (n, k)."""
    n = len(x)
    mean3 = np.vstack([x[max(t - 2, 0):t + 1].mean(axis=0) for t in range(n)])
    slope3 = np.zeros_like(x)
    slope3[2:] = (x[2:] - x[:-2]) / 2.0
    return np.hstack([x, mean3, slope3])


def system_features(ep: Mapping[str, Any], delay: int, label_fraction: float) -> np.ndarray:
    """Primary-system features per batch (protocol section 5)."""
    acc_obs = observed_accuracy(ep, label_fraction)
    aux = np.asarray(ep["aux"], float)
    aux_change = aux - aux[: cfg.BASELINE_BATCHES].mean(axis=0)
    rws = np.asarray(ep["rws"], float)[:, None]
    return np.hstack([
        v2.signal_features(np.asarray(ep["signals"], float)),
        v2.label_features(acc_obs, delay),
        aux, aux_change,
        _rolling(rws),
    ])


def state_target(ep: Mapping[str, Any]) -> np.ndarray:
    return v2.state_targets(np.asarray(ep["accuracy"], float))


def baseline_scores(ep: Mapping[str, Any], delay: int, label_fraction: float) -> dict[str, np.ndarray]:
    view = {"signals": np.asarray(ep["signals"], float), "accuracy": observed_accuracy(ep, label_fraction)}
    b = v2.baseline_scores(view, delay)
    return {
        "fixed_dews_score": b["fixed_score"],
        "statistical_only": b["stat_only"],
        "confidence_only": b["conf_only"],
        "mean_ks": np.asarray(ep["mean_ks_raw"], float),
        "label_trend": b["delayed_label_only"],
        "delay_aware_cbpe": v2.cbpe_scores(view, delay)["da_cbpe"],
    }


class RiskCalibrator:
    """Logistic risk model on the state target, trained on development episodes."""

    def __init__(self, delay: int, label_fraction: float) -> None:
        self.delay, self.label_fraction = delay, label_fraction
        self.model = v2._new_model()

    def fit(self, episodes: Sequence[Mapping[str, Any]]) -> RiskCalibrator:
        xs, ys = [], []
        for ep in episodes:
            X, y = system_features(ep, self.delay, self.label_fraction), state_target(ep)
            keep = np.isfinite(y) & (np.arange(len(y)) >= cfg.WARMUP_BATCHES)
            xs.append(X[keep])
            ys.append(y[keep])
        self.model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def score(self, ep: Mapping[str, Any]) -> np.ndarray:
        return self.model.predict_proba(system_features(ep, self.delay, self.label_fraction))[:, 1]


def score_matrix(episodes: Sequence[Mapping[str, Any]], delay: int, label_fraction: float,
                 calibrator_for: Any) -> dict[str, np.ndarray]:
    """{method: (E, N) scores}; ``calibrator_for(ep)`` returns the calibrator that may score ep."""
    out = {m: np.full((len(episodes), cfg.N_BATCHES), np.nan) for m in METHODS}
    for i, ep in enumerate(episodes):
        out["primary_system"][i] = calibrator_for(ep).score(ep)
        for name, values in baseline_scores(ep, delay, label_fraction).items():
            out[name][i] = values
    for values in out.values():
        values[:, : cfg.WARMUP_BATCHES] = np.nan
    return out


@dataclass
class Masks:
    """Evaluation masks over (episode, batch); see protocol section 6."""
    clean: np.ndarray        # batches counted for the false-alert rate
    t1: np.ndarray           # [onset, event)
    t2: np.ndarray           # [onset, confirmation)
    after_onset: np.ndarray  # [onset, end)
    harmful_gradual: np.ndarray
    harmful_abrupt: np.ndarray
    harmless_shift: np.ndarray
    event: np.ndarray        # event onset per episode (-1 if none)
    onset: np.ndarray


def build_masks(episodes: Sequence[Mapping[str, Any]], delay: int) -> Masks:
    E, N = len(episodes), cfg.N_BATCHES
    t = np.arange(N)
    clean = np.zeros((E, N), bool)
    t1, t2, after = (np.zeros((E, N), bool) for _ in range(3))
    hg, ha, hl = (np.zeros(E, bool) for _ in range(3))
    event, onset = np.full(E, -1), np.full(E, -1)
    for i, ep in enumerate(episodes):
        cat = ep["category"]
        if cat == "spurious":  # excluded everywhere (protocol section 3)
            continue
        on = ep["onset"]
        if on is None:
            clean[i] = t >= cfg.WARMUP_BATCHES
            continue
        onset[i] = on
        clean[i] = (t >= cfg.WARMUP_BATCHES) & (t < on)
        after[i] = t >= on
        if ep["event_onset"] is not None:
            ev = int(ep["event_onset"])
            event[i] = ev
            t1[i] = (t >= on) & (t < ev)
            t2[i] = (t >= on) & (t < cfg.confirmation_batch(ev, delay))
        hg[i] = cat == "harmful" and ep["kind"] == "gradual_shift"
        ha[i] = cat == "harmful" and ep["kind"] == "abrupt_shift"
        hl[i] = cat == "harmless"
    return Masks(clean, t1, t2, after, hg, ha, hl, event, onset)


def metrics(scores: np.ndarray, threshold: float, m: Masks) -> dict[str, Any]:
    alerts = np.where(np.isfinite(scores), scores >= threshold, False)
    fa = 100.0 * (alerts & m.clean).sum() / max(m.clean.sum(), 1)
    out: dict[str, Any] = {"threshold": float(threshold), "false_alerts_per_100_clean": float(fa)}
    for name, sel in (("gradual", m.harmful_gradual), ("abrupt", m.harmful_abrupt),
                      ("harmful", m.harmful_gradual | m.harmful_abrupt)):
        idx = np.flatnonzero(sel)
        t1_hit = (alerts & m.t1)[idx].any(axis=1)
        t2_hit = (alerts & m.t2)[idx].any(axis=1)
        leads = [int(m.event[i] - np.argmax(alerts[i] & m.t1[i])) for i, h in zip(idx, t1_hit) if h]
        out[name] = {
            "episodes": len(idx),
            "t1_share": float(t1_hit.mean()) if len(idx) else None,
            "t2_share": float(t2_hit.mean()) if len(idx) else None,
            "t1_median_lead": float(np.median(leads)) if leads else None,
            "t1_hits": t1_hit.astype(int).tolist(), "t2_hits": t2_hit.astype(int).tolist(),
        }
    hl = np.flatnonzero(m.harmless_shift)
    out["harmless_alert_share"] = float((alerts & m.after_onset)[hl].any(axis=1).mean()) if len(hl) else None
    return out


def harm_auroc(scores: np.ndarray, m: Masks) -> float | None:
    """H3: mean score over the first HARM_WINDOW post-onset batches, harmful vs harmless shifts."""
    pos = np.flatnonzero(m.harmful_gradual | m.harmful_abrupt)
    neg = np.flatnonzero(m.harmless_shift)
    if not len(pos) or not len(neg):
        return None
    def harm(i):
        on = m.onset[i]
        return float(np.nanmean(scores[i, on:on + HARM_WINDOW]))
    values = [harm(i) for i in pos] + [harm(i) for i in neg]
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), values))


def select_threshold(scores: np.ndarray, m: Masks, n_candidates: int = 200) -> float:
    """Highest development T1 share (harmful episodes) with false alerts <= budget; ties -> higher threshold."""
    finite = scores[np.isfinite(scores)]
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_candidates)))
    candidates = np.append(candidates, np.nextafter(finite.max(), np.inf))
    harmful = np.flatnonzero(m.harmful_gradual | m.harmful_abrupt)
    best, best_key = None, None
    for thr in candidates:
        alerts = np.where(np.isfinite(scores), scores >= thr, False)
        fa = 100.0 * (alerts & m.clean).sum() / max(m.clean.sum(), 1)
        if fa > cfg.FALSE_ALERT_BUDGET + 1e-12:
            continue
        key = ((alerts & m.t1)[harmful].any(axis=1).mean(), thr)
        if best_key is None or key > best_key:
            best, best_key = float(thr), key
    return best
