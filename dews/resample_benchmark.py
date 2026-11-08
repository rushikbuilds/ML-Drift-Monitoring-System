"""v6 benchmark: real data with controlled shifts, built by re-sampling real rows.

Implements the episode design frozen in ``dews.v6_config`` (see docs/v6_protocol.md).
Every streamed row keeps its true label (except in ``concept_change`` episodes, where
labels are flipped on purpose), so whether a shift hurts the monitored model is
measured from real labels rather than decided by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from . import v6_config as cfg
from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG
from .v2 import Episode, outcome

EPISODE_KINDS = tuple(cfg.EPISODES_PER_DATASET)  # clean, gradual_shift, abrupt_shift, concept_change
SHIFT_KINDS = ("gradual_shift", "abrupt_shift")
CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "v2"
# Cached (X, y) pickles written by scripts/run_v2_realdata.py, run_v3_confirmation.py and
# run_final_evaluation.py; the prefix records which script downloaded the dataset.
_CACHE_PREFIX = {
    "electricity": "real", "airlines": "real", "covertype": "real", "bank_marketing": "real",
    "run_or_walk": "fresh", "vehicle_sensit": "fresh", "kddcup99": "fresh", "ldpa": "fresh",
    "hyperplane": "new", "sea": "new", "covpokelec": "new", "poker_hand": "new", "led": "new",
}


@lru_cache(maxsize=4)
def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = pd.read_pickle(CACHE_DIR / f"{_CACHE_PREFIX[name]}_{name}.pkl")
    return X.reset_index(drop=True).astype(float), np.asarray(y, dtype=int)


def episode_plan(dataset: str) -> list[tuple[int, str, int]]:
    """(episode_index, kind, seed) for every episode of a dataset, in a fixed order."""
    base = cfg.HELD_OUT_SEED_BASE if dataset in cfg.HELD_OUT_DATASETS else cfg.DEV_SEED_BASE
    plan, index = [], 0
    for kind in EPISODE_KINDS:
        for _ in range(cfg.EPISODES_PER_DATASET[kind]):
            plan.append((index, kind, base + 100 * cfg.ALL_DATASETS.index(dataset) + index))
            index += 1
    return plan


def build_model(model_kind: str):
    if model_kind == "logistic_regression":
        return DEFAULT_EARLY_WARNING_CONFIG.classifier.build_pipeline()
    if model_kind == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=0)
    raise ValueError(f"unknown model kind: {model_kind}")


@dataclass
class V6Episode:
    dataset: str
    kind: str
    seed: int
    model_kind: str
    reference: pd.DataFrame        # all reference rows (detector reference)
    reference_labels: np.ndarray
    batches: list[pd.DataFrame]
    labels: list[np.ndarray]        # true labels, as they will arrive (after any concept flip)
    onset: int | None
    model: Any                      # monitored model, fit on the first fit rows of the reference
    importance: np.ndarray          # permutation importance of the monitored model, per feature
    meta: dict[str, Any] = field(default_factory=dict)

    def accuracy(self) -> np.ndarray:
        return np.array([np.mean(self.model.predict(b) == y) for b, y in zip(self.batches, self.labels)])

    def to_v2(self) -> Episode:
        return Episode(self.dataset, self.seed, self.reference, self.reference_labels,
                       self.batches, self.labels, self.onset)


def _draw_block(y: np.ndarray, rng: np.random.Generator) -> int:
    span = cfg.REFERENCE_ROWS + cfg.POOL_ROWS
    if len(y) < span:
        raise ValueError("dataset is shorter than one reference + pool block")
    valid = _valid_block_starts(y)
    if len(valid) == 0:
        raise RuntimeError("no block with both classes found")
    # uniform over valid starts: identical to redrawing until valid, without an attempt limit
    return int(rng.choice(valid))


def _valid_block_starts(y: np.ndarray) -> np.ndarray:
    """Starts where both classes make up >= 5% of the model-fit rows and of the pool, and appear in the holdout."""
    span = cfg.REFERENCE_ROWS + cfg.POOL_ROWS
    fit_len = cfg.REFERENCE_ROWS - cfg.IMPORTANCE_HOLDOUT_ROWS
    csum = np.concatenate([[0], np.cumsum(y)])
    starts = np.arange(len(y) - span + 1)

    def share(offset, length):
        return (csum[starts + offset + length] - csum[starts + offset]) / length

    fit, hold, pool = share(0, fit_len), share(fit_len, cfg.IMPORTANCE_HOLDOUT_ROWS), share(cfg.REFERENCE_ROWS, cfg.POOL_ROWS)
    lo, hi = cfg.MIN_CLASS_SHARE, 1 - cfg.MIN_CLASS_SHARE
    ok = (fit >= lo) & (fit <= hi) & (pool >= lo) & (pool <= hi) & (hold > 0) & (hold < 1)
    return starts[ok]


def _ramp(t: int, onset: int) -> float:
    return float(np.clip((t - onset + 1) / cfg.RAMP_BATCHES, 0.0, 1.0)) if t >= onset else 0.0


def build_episode(X: pd.DataFrame, y: np.ndarray, *, dataset: str, kind: str, seed: int,
                  model_kind: str = "logistic_regression", lambda_max: float = cfg.LAMBDA_MAX) -> V6Episode:
    if kind not in EPISODE_KINDS:
        raise ValueError(f"unknown episode kind: {kind}")
    rng = np.random.default_rng(seed)
    start = _draw_block(y, rng)
    ref_end = start + cfg.REFERENCE_ROWS
    reference, reference_labels = X.iloc[start:ref_end].reset_index(drop=True), y[start:ref_end]
    pool, pool_labels = X.iloc[ref_end:ref_end + cfg.POOL_ROWS].reset_index(drop=True), y[ref_end:ref_end + cfg.POOL_ROWS]

    fit_rows = cfg.REFERENCE_ROWS - cfg.IMPORTANCE_HOLDOUT_ROWS
    model = build_model(model_kind).fit(reference.iloc[:fit_rows], reference_labels[:fit_rows])
    imp = permutation_importance(model, reference.iloc[fit_rows:], reference_labels[fit_rows:],
                                 n_repeats=5, random_state=seed, scoring="accuracy").importances_mean

    mean, std = reference.mean().to_numpy(), reference.std().to_numpy()
    usable = np.flatnonzero(std > 0)
    z = (pool.to_numpy() - mean) / np.where(std > 0, std, 1.0)
    order = usable[np.argsort(-imp[usable], kind="stable")]
    top, bottom = order[: (len(order) + 1) // 2], order[(len(order) + 1) // 2:]

    onset = None if kind == "clean" else int(rng.integers(cfg.ONSET_RANGE[0], cfg.ONSET_RANGE[1] + 1))
    meta: dict[str, Any] = {"block_start": start, "kind": kind}
    sign, feature = 0, None
    if kind in SHIFT_KINDS:
        from_top = rng.random() < cfg.IMPORTANT_FEATURE_PROB or len(bottom) == 0
        feature = int(rng.choice(top if from_top else bottom))
        sign = int(rng.choice([-1, 1]))
        meta.update(feature=X.columns[feature], feature_index=feature, from_important_half=bool(from_top),
                    sign=sign, lambda_max=lambda_max)
    elif kind == "concept_change":
        feature = int(rng.choice(usable))
        meta.update(feature=X.columns[feature], feature_index=feature, flip_max=cfg.CONCEPT_FLIP_MAX)

    batches, labels, sampled = [], [], []
    for t in range(cfg.N_BATCHES):
        if kind in SHIFT_KINDS:
            lam = lambda_max * (_ramp(t, onset) if kind == "gradual_shift" else float(t >= onset))
            logits = lam * sign * z[:, feature]
            w = np.exp(logits - logits.max())
            idx = rng.choice(len(pool), size=cfg.BATCH_SIZE, replace=True, p=w / w.sum())
        else:
            idx = rng.choice(len(pool), size=cfg.BATCH_SIZE, replace=True)
        lab = pool_labels[idx].copy()
        if kind == "concept_change" and t >= onset:
            flip = (z[idx, feature] > 0) & (rng.random(len(idx)) < cfg.CONCEPT_FLIP_MAX * _ramp(t, onset))
            lab[flip] = 1 - lab[flip]
        batches.append(pool.iloc[idx].reset_index(drop=True))
        labels.append(lab)
        sampled.append(idx)
    meta["sampled_pool_indices"] = sampled
    meta["pool_labels"] = pool_labels
    return V6Episode(dataset, kind, seed, model_kind, reference, reference_labels, batches, labels,
                     onset, model, imp, meta)


def classify_outcome(ep: V6Episode) -> dict[str, Any]:
    """Event rule on the true accuracy; categories as defined in docs/v6_protocol.md section 3."""
    acc = ep.accuracy()
    out = outcome(acc)
    event = out["event_onset"]
    if ep.kind == "clean":
        category = "spurious" if event is not None else "clean"
    elif event is not None and event < ep.onset:
        category = "spurious"
    elif ep.kind in SHIFT_KINDS:
        category = "harmful" if event is not None else "harmless"
    else:
        category = "concept_event" if event is not None else "concept_no_event"
    return {"category": category, "event_onset": event, "baseline_accuracy": out["baseline"],
            "min_accuracy_after_onset": float(acc[(ep.onset or 0):].min()), "accuracy": acc}
