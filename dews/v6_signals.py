"""Signal extraction for v6 episodes, using each episode's own monitored model.

Per batch it records only what a live monitor could compute without labels
(DEWS drift/confidence signals, auxiliary-model disagreement, relevance-weighted
shift), plus the evaluation-only accuracies: the true batch accuracy and the
accuracy of a random 20% labeled subsample. Delays are applied later, when
features and scores are built, so one extraction serves every label delay.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from . import v6_config as cfg
from .alerts import AlertEngine
from .config import MonitorConfig
from .data import FeatureSchema
from .early_warning import _prediction_summary, _signal_payload
from .forecasting import DriftForecaster
from .integrations import IntegrationManager
from .models import SklearnModelAdapter
from .monitoring import DriftMonitoringSystem
from .resample_benchmark import V6Episode, classify_outcome
from .storage import DriftStore
from .v2 import BASE_SIGNALS

DETECTOR_REFERENCE_ROWS = 512  # MMD cost is quadratic in reference rows (same cap as v2)
LABELED_FRACTION = 0.2


def _monitor(ep: V6Episode) -> tuple[DriftMonitoringSystem, Any]:
    schema = FeatureSchema(continuous=list(ep.reference.columns), categorical=[])
    config = MonitorConfig(batch_size=cfg.BATCH_SIZE, window_size=cfg.BATCH_SIZE)
    ref = ep.reference.sample(n=min(DETECTOR_REFERENCE_ROWS, len(ep.reference)), random_state=42)
    ref = ref.sort_index().reset_index(drop=True)
    system = DriftMonitoringSystem(
        config=config, schema=schema, model=SklearnModelAdapter(ep.model), store=DriftStore(":memory:"),
        alert_engine=AlertEngine(config), forecaster=DriftForecaster(horizon=config.forecast_horizon),
        integration_manager=IntegrationManager(),
    )
    system.fit_detectors(ref)
    return system, ref


def extract_episode(ep: V6Episode) -> dict[str, Any]:
    fit_rows = cfg.REFERENCE_ROWS - cfg.IMPORTANCE_HOLDOUT_ROWS
    X_fit, y_fit = ep.reference.iloc[:fit_rows], ep.reference_labels[:fit_rows]
    rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X_fit, y_fit)
    hgb = HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=0).fit(X_fit, y_fit)

    weights = np.clip(ep.importance, 0.0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1.0 / len(weights))
    ref_values = ep.reference.to_numpy()

    system, det_ref = _monitor(ep)
    config = system.config
    rng = np.random.default_rng(ep.seed + 777)
    n_labeled = round(LABELED_FRACTION * cfg.BATCH_SIZE)
    signals, aux, rws, mean_ks, acc, acc20 = [], [], [], [], [], []
    for t, (batch, labels) in enumerate(zip(ep.batches, ep.labels)):
        predictions, summary = _prediction_summary(system.model, batch, config.low_confidence_threshold)
        result = system.process_batch(t, det_ref, batch, batch_labels=None, evaluate_forecasts=False)
        sig = _signal_payload(result, summary)
        signals.append([sig[name] for name in BASE_SIGNALS])

        p_rf, p_hgb = rf.predict(batch), hgb.predict(batch)
        aux_conf = (rf.predict_proba(batch).max(axis=1).mean() + hgb.predict_proba(batch).max(axis=1).mean()) / 2
        aux.append([np.mean(predictions != p_rf), np.mean(predictions != p_hgb), np.mean(p_rf != p_hgb), aux_conf])

        values = batch.to_numpy()
        ks = np.array([ks_2samp(ref_values[:, j], values[:, j]).statistic for j in range(values.shape[1])])
        rws.append(float(ks @ weights))
        mean_ks.append(float(ks.mean()))

        correct = predictions == labels
        acc.append(float(correct.mean()))
        acc20.append(float(correct[rng.choice(len(correct), size=n_labeled, replace=False)].mean()))

    out = classify_outcome(ep)
    return {
        "dataset": ep.dataset, "kind": ep.kind, "seed": ep.seed, "model_kind": ep.model_kind,
        "onset": ep.onset, "category": out["category"], "event_onset": out["event_onset"],
        "signals": np.asarray(signals), "aux": np.asarray(aux), "rws": np.asarray(rws),
        "mean_ks_raw": np.asarray(mean_ks), "accuracy": np.asarray(acc), "accuracy_obs20": np.asarray(acc20),
        "importance": ep.importance,
        "meta": {k: v for k, v in ep.meta.items() if k not in ("sampled_pool_indices", "pool_labels")},
    }
