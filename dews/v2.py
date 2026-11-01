"""v2 experiment: delayed-label-aware degradation warning.

Revision of the v1 protocol after its negative result (see docs/v2_protocol.md):

* scenarios have noisy labels, randomized onset and genuine lead time between
  the input change and the accuracy event, plus a concept-drift probe where
  P(X) never changes;
* the risk target is state based: "a confirmed degraded run (>= K consecutive
  batches at least 5 points under baseline) starts or continues within the next
  H batches", so post-onset batches are positives instead of negatives;
* matured (delayed) accuracies are a causal input feature, so label delay
  genuinely matters for the monitor;
* a "wait for delayed labels" baseline answers whether drift signals add
  anything beyond simply waiting for labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .alerts import AlertEngine
from .config import MonitorConfig
from .data import FeatureSchema
from .early_warning import _prediction_summary, _signal_payload
from .early_warning_metrics import accuracy_baseline, find_degradation_onset
from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG
from .forecasting import DriftForecaster
from .integrations import IntegrationManager
from .models import SklearnModelAdapter
from .monitoring import DriftMonitoringSystem
from .storage import DriftStore

H = 5  # warning horizon in batches
K = 2  # consecutive degraded batches that confirm an event
DROP = 0.05
BASELINE_BATCHES = 5
WARMUP = 6  # alerts/scores are ignored before this batch index
DELAYS = (0, 2, 5)
SCENARIOS = ("clean", "harmless_shift", "harmful_gradual", "harmful_abrupt", "concept_gradual")
TRAIN_SEEDS = tuple(range(100, 140))
VALID_SEEDS = tuple(range(140, 160))
TEST_SEEDS = tuple(range(200, 250))

PRED_COLS = ("x0", "x1", "x2", "x3")
NUIS_COLS = ("n0", "n1", "n2")
FEATURES = PRED_COLS + NUIS_COLS
BASE_SIGNALS = (
    "drift_score", "s_stat", "s_conf", "mean_ks", "mean_confidence",
    "mean_entropy", "low_confidence_rate", "positive_prediction_rate",
)


# --------------------------------------------------------------------------- data
@dataclass
class Episode:
    scenario: str
    seed: int
    reference: pd.DataFrame
    reference_labels: np.ndarray
    batches: list[pd.DataFrame]
    labels: list[np.ndarray]
    onset: int | None  # intervention onset (None for clean)


def _latent(x: np.ndarray, w_flip: np.ndarray | None = None) -> np.ndarray:
    # the quadratic term in x3 is invisible to the linear classifier until x3 shifts
    z = x[:, 0] + 0.8 * x[:, 1] - 0.6 * x[:, 2] + 0.4 * x[:, 0] * x[:, 1] + 0.8 * (x[:, 3] ** 2 - 1.0)
    if w_flip is not None:  # concept drift: coefficient of x0 rotates 1 -> -1
        z = z - 2.0 * w_flip * x[:, 0]
    return z


def build_episode(scenario: str, seed: int, *, n_batches: int = 45,
                  batch_size: int = 300, reference_size: int = 3000) -> Episode:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario)])
    rows = n_batches * batch_size

    def draw(n: int) -> np.ndarray:
        return rng.normal(size=(n, len(FEATURES)))

    def sample_labels(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-2.5 * _latent(x, w)))
        return (rng.random(len(x)) < p).astype(int)

    ref_x = draw(reference_size)
    ref_y = sample_labels(ref_x)
    x = draw(rows)
    onset = None if scenario == "clean" else int(rng.integers(10, 17))
    batch_idx = np.repeat(np.arange(n_batches), batch_size)
    w = None
    if onset is not None:
        since = np.clip(batch_idx - onset + 1, 0, None).astype(float)
        if scenario == "harmless_shift":
            active = since > 0
            x[active, 4] += 2.0
            x[active, 5] *= 2.0
        elif scenario == "harmful_gradual":
            x[:, 3] += 1.5 * np.clip(since / 24.0, 0.0, 1.0)
        elif scenario == "harmful_abrupt":
            x[:, 3] += 1.2 * (since > 0)
        elif scenario == "concept_gradual":
            w = np.clip(since / 24.0, 0.0, 1.0)
    y = sample_labels(x, w)
    frame = pd.DataFrame(x, columns=FEATURES)
    return Episode(
        scenario=scenario,
        seed=seed,
        reference=pd.DataFrame(ref_x, columns=FEATURES),
        reference_labels=ref_y,
        batches=[frame.iloc[i * batch_size:(i + 1) * batch_size].reset_index(drop=True) for i in range(n_batches)],
        labels=[y[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)],
        onset=onset,
    )


# ------------------------------------------------------------------- signal replay
def extract_signals(ep: Episode, *, detector_reference_rows: int = 512) -> dict[str, Any]:
    """Run the frozen classifier + DEWS detectors over an episode (labels hidden).

    Returns per-batch unlabeled signals and, separately, realized accuracies,
    which downstream code may only expose after the configured label delay.
    """
    classifier = DEFAULT_EARLY_WARNING_CONFIG.classifier.build_pipeline()
    classifier.fit(ep.reference, ep.reference_labels)
    model = SklearnModelAdapter(classifier)
    schema = FeatureSchema(continuous=list(ep.reference.columns), categorical=[])
    config = MonitorConfig(batch_size=len(ep.batches[0]), window_size=len(ep.batches[0]))
    ref = ep.reference
    if len(ref) > detector_reference_rows:
        ref = ref.sample(n=detector_reference_rows, random_state=42).sort_index().reset_index(drop=True)
    system = DriftMonitoringSystem(
        config=config, schema=schema, model=model, store=DriftStore(":memory:"),
        alert_engine=AlertEngine(config),
        forecaster=DriftForecaster(horizon=config.forecast_horizon),
        integration_manager=IntegrationManager(),
    )
    system.fit_detectors(ref)
    signals, accuracies = [], []
    for i, (batch, labels) in enumerate(zip(ep.batches, ep.labels)):
        predictions, summary = _prediction_summary(model, batch, config.low_confidence_threshold)
        result = system.process_batch(i, ref, batch, batch_labels=None, evaluate_forecasts=False)
        sig = _signal_payload(result, summary)
        signals.append([sig[name] for name in BASE_SIGNALS])
        accuracies.append(float(np.mean(predictions == labels)))
    return {
        "scenario": ep.scenario, "seed": ep.seed, "onset": ep.onset,
        "signals": np.asarray(signals, dtype=float),
        "accuracy": np.asarray(accuracies, dtype=float),
    }


# ---------------------------------------------------------- outcomes and targets
def outcome(accuracy: np.ndarray) -> dict[str, Any]:
    """Evaluation-only outcome: baseline, event onset, per-batch degraded flags."""
    base = accuracy_baseline(list(accuracy), BASELINE_BATCHES)
    onset = find_degradation_onset(list(accuracy), base, start_batch=BASELINE_BATCHES,
                                   accuracy_drop=DROP, consecutive_batches=K)
    degraded = accuracy <= base - DROP
    return {"baseline": base, "event_onset": onset, "degraded": degraded}


def state_targets(accuracy: np.ndarray) -> np.ndarray:
    """y_t = 1 if a confirmed degraded run starts/continues in (t, t+H]; NaN if unobservable."""
    out = outcome(accuracy)
    deg = out["degraded"]
    n = len(accuracy)
    pair = np.array([deg[i] and deg[i + 1] for i in range(n - 1)] + [False])
    y = np.full(n, np.nan)
    for t in range(n - H - 1):
        y[t] = float(pair[t + 1:t + H + 1].any())
    return y


def target_mature_batch(t: int, delay: int) -> int:
    """First prediction time at which target t may be used for an update."""
    return t + H + 1 + max(delay, 1)


# ------------------------------------------------------------------- features
def label_features(accuracy: np.ndarray, delay: int) -> np.ndarray:
    """Causal accuracy-drop features usable at each time t (labels delayed)."""
    n = len(accuracy)
    out = np.zeros((n, 2))
    lag = max(delay, 1)
    for t in range(n):
        m = t - lag  # latest batch whose labels have arrived
        if m < BASELINE_BATCHES - 1:
            continue
        base = float(np.mean(accuracy[:BASELINE_BATCHES]))
        out[t, 0] = base - accuracy[m]
        out[t, 1] = base - float(np.mean(accuracy[max(m - 1, 0):m + 1]))
    return out


def signal_features(signals: np.ndarray) -> np.ndarray:
    n, k = signals.shape
    cols = [signals]
    mean3 = np.vstack([signals[max(t - 2, 0):t + 1].mean(axis=0) for t in range(n)])
    slope3 = np.zeros_like(signals)
    for t in range(2, n):
        slope3[t] = (signals[t] - signals[t - 2]) / 2.0
    cols += [mean3, slope3]
    return np.hstack(cols)


def episode_features(sig: Mapping[str, Any], delay: int, *, use_labels: bool) -> np.ndarray:
    base = signal_features(sig["signals"])
    if not use_labels:
        return base
    return np.hstack([base, label_features(sig["accuracy"], delay)])


# ------------------------------------------------------------------- scores
def baseline_scores(sig: Mapping[str, Any], delay: int) -> dict[str, np.ndarray]:
    s = sig["signals"]
    idx = {n: i for i, n in enumerate(BASE_SIGNALS)}
    ref_conf = s[:BASELINE_BATCHES, idx["mean_confidence"]].mean()
    ref_ent = s[:BASELINE_BATCHES, idx["mean_entropy"]].mean()
    fixed = s[:, idx["drift_score"]]
    slope = np.zeros_like(fixed)
    for t in range(2, len(fixed)):
        slope[t] = (fixed[t] - fixed[t - 2]) / 2.0
    return {
        "stat_only": s[:, idx["s_stat"]],
        "conf_only": np.abs(s[:, idx["mean_confidence"]] - ref_conf) + np.abs(s[:, idx["mean_entropy"]] - ref_ent),
        "fixed_score": fixed,
        "slope_forecast": np.clip(fixed + H * slope, 0.0, 1.0),
        "delayed_label_only": np.clip(label_features(sig["accuracy"], delay)[:, 1], 0.0, None),
    }


def _new_model() -> Any:
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                         StandardScaler(), LogisticRegression(max_iter=300, C=0.3))


def training_pool(train_sigs: Sequence[Mapping[str, Any]], delay: int, *, use_labels: bool):
    xs, ys = [], []
    for sig in train_sigs:
        feats = episode_features(sig, delay, use_labels=use_labels)
        y = state_targets(sig["accuracy"])
        keep = (~np.isnan(y)) & (np.arange(len(y)) >= WARMUP)
        xs.append(feats[keep]); ys.append(y[keep])
    return np.vstack(xs), np.concatenate(ys)


def calibrator_scores(sig: Mapping[str, Any], delay: int, pool, *, use_labels: bool,
                      online: bool, online_weight: float = 5.0) -> np.ndarray:
    """Risk probability per batch; the online variant refits on matured targets only."""
    px, py = pool
    feats = episode_features(sig, delay, use_labels=use_labels)
    y = state_targets(sig["accuracy"])
    n = len(feats)
    model = _new_model().fit(px, py)
    scores = np.full(n, np.nan)
    used: set[int] = set()
    for t in range(WARMUP, n):
        if online:
            ready = [e for e in range(n) if not np.isnan(y[e]) and e >= WARMUP
                     and target_mature_batch(e, delay) <= t]
            if set(ready) != used and ready:
                used = set(ready)
                x_all = np.vstack([px, feats[ready]])
                y_all = np.concatenate([py, y[ready]])
                w = np.concatenate([np.ones(len(py)), np.full(len(ready), online_weight)])
                if len(set(y_all)) == 2:
                    model = _new_model()
                    model.fit(x_all, y_all, logisticregression__sample_weight=w)
        scores[t] = model.predict_proba(feats[t:t + 1])[0, 1]
    return scores
