"""Causal risk features, matched baseline scores, and delayed-label calibration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isfinite, nextafter, inf
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .early_warning_metrics import (
    evaluate_episode_alerts,
    risk_target_available_batch,
    stable_clean_batch_indices,
)
from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG


logger = logging.getLogger(__name__)

_BASE_SIGNAL_NAMES = (
    "s_stat",
    "s_conf",
    "mean_confidence",
    "low_confidence_rate",
    "positive_prediction_rate",
    "mean_entropy",
)
_FEATURE_SUFFIXES = ("", "_lag_1", "_lag_2", "_mean_3", "_slope_3")
_FORBIDDEN_FEATURE_KEYS = {
    "label",
    "target",
    "accuracy",
    "realized_accuracy",
    "event_onset_batch",
    "risk_target_label",
}


class RiskFeatureBuilder:
    """Build a fixed causal feature vector from current and prior signals."""

    feature_names = tuple(
        f"{signal}{suffix}"
        for signal in _BASE_SIGNAL_NAMES
        for suffix in _FEATURE_SUFFIXES
    )

    def build(self, signal_history: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
        if not signal_history:
            raise ValueError("signal_history must contain the current batch")
        for batch_signals in signal_history:
            leaked = _FORBIDDEN_FEATURE_KEYS.intersection(batch_signals)
            if leaked:
                raise ValueError(f"risk feature input contains outcome fields: {sorted(leaked)}")

        result: dict[str, float | None] = {}
        for name in _BASE_SIGNAL_NAMES:
            values: list[float | None] = []
            for batch_signals in signal_history:
                value = batch_signals.get(name)
                numeric = None if value is None else float(value)
                values.append(numeric if numeric is not None and isfinite(numeric) else None)

            result[name] = values[-1]
            result[f"{name}_lag_1"] = values[-2] if len(values) >= 2 else None
            result[f"{name}_lag_2"] = values[-3] if len(values) >= 3 else None
            trailing = values[-3:]
            result[f"{name}_mean_3"] = (
                float(np.mean(trailing))
                if len(trailing) == 3 and all(value is not None for value in trailing)
                else None
            )
            result[f"{name}_slope_3"] = (
                float((trailing[-1] - trailing[0]) / (len(trailing) - 1))
                if len(trailing) >= 2 and all(value is not None for value in trailing)
                else None
            )
        return result

    def vector(self, features: Mapping[str, float | None]) -> np.ndarray:
        values = []
        for name in self.feature_names:
            value = features.get(name)
            if value is None:
                values.append(np.nan)
                continue
            numeric = float(value)
            values.append(numeric if isfinite(numeric) else np.nan)
        return np.asarray(values, dtype=float)


def compute_baseline_scores(
    signal_history: Sequence[Mapping[str, float]],
    *,
    horizon_batches: int = DEFAULT_EARLY_WARNING_CONFIG.protocol.horizon_batches,
    slope_window: int = 3,
) -> dict[str, float]:
    """Compute all score baselines from unlabeled current/prior signals."""
    if not signal_history:
        raise ValueError("signal_history must contain the current batch")
    if horizon_batches < 1:
        raise ValueError("horizon_batches must be at least 1")
    if slope_window < 2:
        raise ValueError("slope_window must be at least 2")

    current = signal_history[-1]
    statistical = float(np.clip(current["s_stat"], 0.0, 1.0))
    confidence = float(np.clip(current["s_conf"], 0.0, 1.0))
    fixed_score = float(np.clip(current["drift_score"], 0.0, 1.0))
    fixed_history = [
        float(np.clip(item["drift_score"], 0.0, 1.0))
        for item in signal_history[-slope_window:]
    ]
    slope = (
        (fixed_history[-1] - fixed_history[0]) / (len(fixed_history) - 1)
        if len(fixed_history) >= 2
        else 0.0
    )
    slope_forecast = float(np.clip(fixed_score + horizon_batches * slope, 0.0, 1.0))
    return {
        "statistical_only": statistical,
        "confidence_only": confidence,
        "fixed_weighted_score": fixed_score,
        "recent_slope_forecast": slope_forecast,
    }


@dataclass(frozen=True, slots=True)
class ValidationEpisodeScores:
    """One validation episode's score timeline and known event outcome."""

    scenario: str
    seed: int
    scores: tuple[float | None, ...]
    event_onset_batch: int | None
    clean_batch_indices: tuple[int, ...]


def validation_episodes_from_runs(
    run_results: Sequence[Mapping[str, Any]],
    *,
    score_name: str,
) -> tuple[ValidationEpisodeScores, ...]:
    """Extract a score timeline from validation-split replay outputs."""
    output = []
    for run in run_results:
        metadata = run.get("metadata", {}).get("run", {})
        if metadata.get("split") != "validation":
            raise ValueError("threshold selection accepts validation episodes only")
        records = run.get("batch_records", [])
        if score_name == "risk_probability":
            values = tuple(
                None if record.get("risk_probability") is None else float(record["risk_probability"])
                for record in records
            )
        else:
            values = tuple(
                None
                if record.get("baseline_scores", {}).get(score_name) is None
                else float(record["baseline_scores"][score_name])
                for record in records
            )
        event_onset = run.get("summary", {}).get("event_onset_batch")
        event_batches = [int(event_onset)] if event_onset is not None else []
        output.append(
            ValidationEpisodeScores(
                scenario=str(metadata.get("scenario", "unknown")),
                seed=int(metadata.get("seed", -1)),
                scores=values,
                event_onset_batch=int(event_onset) if event_onset is not None else None,
                clean_batch_indices=stable_clean_batch_indices(
                    total_batches=len(records),
                    event_batches=event_batches,
                    horizon_batches=DEFAULT_EARLY_WARNING_CONFIG.protocol.horizon_batches,
                ),
            )
        )
    if not output:
        raise ValueError("at least one validation episode is required")
    return tuple(output)


def select_warning_threshold(
    episodes: Sequence[ValidationEpisodeScores],
    *,
    max_false_alerts_per_100_clean_batches: float = (
        DEFAULT_EARLY_WARNING_CONFIG.protocol.false_alert_budget_per_100_clean_batches
    ),
    horizon_batches: int = DEFAULT_EARLY_WARNING_CONFIG.protocol.horizon_batches,
) -> dict[str, Any]:
    """Choose the validation threshold with best event recall under the alert budget.

    Threshold candidates are generated only from validation scores. When event
    recall and warning timing tie, the selector prefers fewer alerts and then
    the higher threshold. ``nextafter(max_score, +inf)`` provides a no-alert
    candidate for the clean/no-event case.
    """
    if not episodes:
        raise ValueError("at least one validation episode is required")
    if not isfinite(max_false_alerts_per_100_clean_batches) or max_false_alerts_per_100_clean_batches < 0:
        raise ValueError("false-alert budget must be a finite non-negative number")
    if horizon_batches < 1:
        raise ValueError("horizon_batches must be at least 1")

    score_values = [
        float(score)
        for episode in episodes
        for score in episode.scores
        if score is not None and isfinite(float(score))
    ]
    if not score_values:
        raise ValueError("validation episodes contain no available scores for threshold selection")
    if any(score < 0.0 for score in score_values):
        raise ValueError("baseline and risk scores must be non-negative")

    candidates = sorted(set(score_values))
    candidates.append(nextafter(max(score_values), inf))
    clean_batch_total = sum(len(set(episode.clean_batch_indices)) for episode in episodes)
    event_total = sum(episode.event_onset_batch is not None for episode in episodes)
    if clean_batch_total == 0:
        raise ValueError("validation episodes contain no stable clean batches")

    evaluated = []
    for threshold in candidates:
        total_false_clean_alerts = 0
        total_alerts = 0
        matched_events = 0
        warning_leads: list[float] = []
        detection_lags: list[float] = []
        for episode in episodes:
            alerts = [
                batch_index
                for batch_index, score in enumerate(episode.scores)
                if score is not None and isfinite(float(score)) and float(score) >= threshold
            ]
            total_alerts += len(alerts)
            event_batches = [episode.event_onset_batch] if episode.event_onset_batch is not None else []
            metrics = evaluate_episode_alerts(
                alerts,
                event_batches,
                horizon_batches=horizon_batches,
                clean_batch_indices=episode.clean_batch_indices,
            )
            clean_indices = set(episode.clean_batch_indices)
            total_false_clean_alerts += sum(
                alert_batch in clean_indices
                for alert_batch in metrics["unmatched_alert_batches"]
            )
            matched_events += metrics["matched_event_count"]
            if metrics["warning_lead_batches"] is not None:
                warning_leads.append(float(metrics["warning_lead_batches"]))
            if metrics["detection_lag_batches"] is not None:
                detection_lags.append(float(metrics["detection_lag_batches"]))

        false_rate = 100.0 * total_false_clean_alerts / clean_batch_total
        warning_lead = float(np.mean(warning_leads)) if warning_leads else None
        detection_lag = float(np.mean(detection_lags)) if detection_lags else None
        recall = matched_events / event_total if event_total else None
        metrics = {
            "threshold": float(threshold),
            "false_alerts_per_100_clean_batches": false_rate,
            "false_alerts_in_clean_batches": total_false_clean_alerts,
            "clean_batches": clean_batch_total,
            "alerts_generated": total_alerts,
            "event_count": event_total,
            "matched_event_count": matched_events,
            "event_recall": recall,
            "mean_warning_lead_batches": warning_lead,
            "mean_detection_lag_batches": detection_lag,
        }
        metrics["within_budget"] = false_rate <= max_false_alerts_per_100_clean_batches + 1e-12
        evaluated.append(metrics)

    feasible = [result for result in evaluated if result["within_budget"]]
    if not feasible:
        raise RuntimeError("no validation threshold satisfies the false-alert budget")

    def rank(result: Mapping[str, Any]) -> tuple[float, float, float, float, int, float]:
        recall_value = result["event_recall"]
        mean_lead = result["mean_warning_lead_batches"]
        mean_lag = result["mean_detection_lag_batches"]
        return (
            -(float(recall_value) if recall_value is not None else -1.0),
            -(float(mean_lead) if mean_lead is not None else -1.0),
            float(mean_lag) if mean_lag is not None else inf,
            float(result["false_alerts_per_100_clean_batches"]),
            int(result["alerts_generated"]),
            -float(result["threshold"]),
        )

    selected = min(feasible, key=rank)
    return {
        **selected,
        "budget_per_100_clean_batches": max_false_alerts_per_100_clean_batches,
        "candidate_count": len(candidates),
        "selection_split": "validation",
        "selection_rule": "maximize event recall, then warning lead; minimize lag, false alerts, and alert count",
    }


class DelayedFeedbackRiskCalibrator:
    """Logistic future-risk model trained and updated from mature targets only."""

    def __init__(
        self,
        *,
        min_mature_examples: int = 40,
        min_examples_per_class: int = 5,
        random_state: int | None = None,
    ) -> None:
        if min_mature_examples < 2:
            raise ValueError("min_mature_examples must be at least 2")
        if min_examples_per_class < 1:
            raise ValueError("min_examples_per_class must be at least 1")
        if 2 * min_examples_per_class > min_mature_examples:
            raise ValueError("min_mature_examples must allow the minimum examples from both classes")
        self.min_mature_examples = min_mature_examples
        self.min_examples_per_class = min_examples_per_class
        self.random_state = (
            DEFAULT_EARLY_WARNING_CONFIG.classifier.random_state
            if random_state is None
            else random_state
        )
        self.feature_builder = RiskFeatureBuilder()
        self._examples: list[tuple[np.ndarray, int]] = []
        self._identities: set[tuple[str, int, int, int]] = set()
        self._model = None
        self._model_version = 0
        self._training_label_delay_batches: int | None = None
        self._last_insufficient_log_count = -1

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def status(self) -> dict[str, Any]:
        labels = [label for _, label in self._examples]
        positives = sum(labels)
        negatives = len(labels) - positives
        return {
            "fitted": self.is_fitted,
            "mature_example_count": len(labels),
            "positive_example_count": positives,
            "negative_example_count": negatives,
            "minimum_mature_examples": self.min_mature_examples,
            "minimum_examples_per_class": self.min_examples_per_class,
            "model_version": self._model_version,
            "training_label_delay_batches": self._training_label_delay_batches,
        }

    def fit_initial(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        training_label_delay_batches: int = 0,
    ) -> dict[str, Any]:
        """Fit the initial model from training-seed examples at one delay only."""
        if training_label_delay_batches not in DEFAULT_EARLY_WARNING_CONFIG.protocol.label_delays_batches:
            raise ValueError(f"unsupported training label delay: {training_label_delay_batches}")
        self._examples.clear()
        self._identities.clear()
        self._model = None
        self._model_version = 0
        self._last_insufficient_log_count = -1
        self._training_label_delay_batches = training_label_delay_batches
        for example in examples:
            if example.get("label_delay_batches") != training_label_delay_batches:
                continue
            seed = example.get("seed")
            if seed is None or DEFAULT_EARLY_WARNING_CONFIG.seed_split.split_for_seed(int(seed)) != "train":
                raise ValueError("initial calibrator fit accepts training-split seeds only")
            self._append_mature_example(example)
        self._fit_if_ready()
        if not self.is_fitted:
            self._log_insufficient_data()
        return self.status

    def update(self, example: Mapping[str, Any]) -> bool:
        """Add one target after its recorded availability time and refit if ready."""
        added = self._append_mature_example(example)
        if added:
            self._fit_if_ready()
        if not self.is_fitted:
            self._log_insufficient_data()
        return self.is_fitted

    def predict(self, features: Mapping[str, float | None]) -> float | None:
        """Return P(event within H batches), or None while model data is insufficient."""
        if not self.is_fitted:
            return None
        vector = self.feature_builder.vector(features).reshape(1, -1)
        return float(self._model.predict_proba(vector)[0, 1])

    def _append_mature_example(self, example: Mapping[str, Any]) -> bool:
        label = example.get("label")
        if label not in (0, 1):
            raise ValueError("risk target label must be 0 or 1")
        scenario = str(example.get("scenario", ""))
        seed = int(example.get("seed", -1))
        delay = int(example.get("label_delay_batches", -1))
        example_batch = int(example.get("example_batch", -1))
        if not scenario or seed < 0 or delay < 0 or example_batch < 0:
            raise ValueError("mature risk example is missing its episode identity")
        if delay not in DEFAULT_EARLY_WARNING_CONFIG.protocol.label_delays_batches:
            raise ValueError("mature risk example has an unsupported label delay")
        available = example.get("available_batch")
        released = example.get("released_at_batch")
        expected_available = risk_target_available_batch(
            example_batch,
            horizon_batches=DEFAULT_EARLY_WARNING_CONFIG.protocol.horizon_batches,
            label_delay_batches=delay,
            event_consecutive_batches=(
                DEFAULT_EARLY_WARNING_CONFIG.protocol.event.consecutive_batches
            ),
        )
        if (
            available is None
            or released is None
            or int(available) != expected_available
            or int(available) > int(released)
        ):
            raise ValueError("risk targets may be used only at or after their protocol availability batch")
        identity = (scenario, seed, delay, example_batch)
        if identity in self._identities:
            return False
        features = example.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("mature risk example must include its causal feature vector")
        self._identities.add(identity)
        self._examples.append((self.feature_builder.vector(features), int(label)))
        return True

    def _fit_if_ready(self) -> None:
        labels = np.asarray([label for _, label in self._examples], dtype=int)
        if len(labels) < self.min_mature_examples:
            return
        class_counts = np.bincount(labels, minlength=2)
        if np.any(class_counts < self.min_examples_per_class):
            return
        matrix = np.vstack([features for features, _ in self._examples])
        self._model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=self.random_state),
        )
        self._model.fit(matrix, labels)
        self._model_version += 1

    def _log_insufficient_data(self) -> None:
        count = len(self._examples)
        if count == self._last_insufficient_log_count:
            return
        self._last_insufficient_log_count = count
        status = self.status
        logger.info(
            "risk calibrator unavailable; using fixed-score fallback "
            "(mature examples=%d, positives=%d, negatives=%d, minimum=%d, per_class=%d)",
            status["mature_example_count"],
            status["positive_example_count"],
            status["negative_example_count"],
            self.min_mature_examples,
            self.min_examples_per_class,
        )


def fit_initial_risk_calibrator(
    training_runs: Sequence[Mapping[str, Any]],
    *,
    training_label_delay_batches: int = 0,
    min_mature_examples: int = 40,
    min_examples_per_class: int = 5,
) -> DelayedFeedbackRiskCalibrator:
    """Collect mature targets from training-split runs and fit one initial model."""
    calibrator = DelayedFeedbackRiskCalibrator(
        min_mature_examples=min_mature_examples,
        min_examples_per_class=min_examples_per_class,
    )
    examples = []
    for run in training_runs:
        metadata = run.get("metadata", {}).get("run", {})
        if metadata.get("split") != "train":
            raise ValueError("initial calibrator training runs must use training seeds only")
        examples.extend(run.get("released_risk_targets", []))
    calibrator.fit_initial(
        examples,
        training_label_delay_batches=training_label_delay_batches,
    )
    return calibrator
