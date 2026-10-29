"""Causal batch replay and delayed-feedback ledger for controlled episodes."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from math import isfinite, log
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import MonitorConfig
from .early_warning_metrics import (
    accuracy_baseline,
    complete_risk_target_batches,
    evaluate_episode_alerts,
    find_degradation_onset,
    risk_target_available_batch,
    stable_clean_batch_indices,
)
from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG
from .models import SklearnModelAdapter
from .monitoring import DriftMonitoringSystem
from .risk import RiskFeatureBuilder, compute_baseline_scores
from .scenarios import ScenarioEpisode
from .storage import DriftStore


RiskPredictor = Callable[[Mapping[str, float | None]], float | None]
RiskTargetUpdater = Callable[[Mapping[str, Any]], None]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _signal_payload(monitoring_result, prediction_summary: dict[str, float]) -> dict[str, float]:
    statistical = monitoring_result.statistical or {}
    embedding = monitoring_result.embedding or {}
    confidence = monitoring_result.confidence or {}
    return {
        "drift_score": float(monitoring_result.drift_score),
        "s_stat": float(statistical.get("s_stat", 0.0)),
        "s_emb": float(embedding.get("s_emb", 0.0)),
        "s_conf": float(confidence.get("s_conf", 0.0)),
        "mean_ks": float(statistical.get("mean_ks", 0.0)),
        "mean_psi": float(statistical.get("mean_psi", 0.0)),
        "low_confidence_rate": float(confidence.get("low_confidence_rate", 0.0)),
        "mean_confidence": prediction_summary["mean_confidence"],
        "positive_prediction_rate": prediction_summary["positive_prediction_rate"],
        "mean_entropy": prediction_summary["mean_entropy"],
    }


def _prediction_summary(model, batch: pd.DataFrame, low_confidence_threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    predictions = np.asarray(model.predict(batch))
    probabilities = np.asarray(model.predict_proba(batch), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(batch):
        raise ValueError("classifier predict_proba must return one probability row per example")
    confidences = probabilities.max(axis=1)
    normalized_entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    ) / max(log(probabilities.shape[1]), 1e-12)
    return predictions, {
        "mean_confidence": float(np.mean(confidences)),
        "low_confidence_rate": float(np.mean(confidences < low_confidence_threshold)),
        "positive_prediction_rate": float(np.mean(predictions == 1)),
        "mean_entropy": float(np.mean(normalized_entropy)),
    }


def _event_state(accuracies: list[float | None]) -> tuple[float | None, int | None, int]:
    visible_accuracies: list[float] = []
    for value in accuracies:
        if value is None:
            break
        visible_accuracies.append(value)

    event_definition = DEFAULT_EARLY_WARNING_CONFIG.protocol.event
    baseline = accuracy_baseline(visible_accuracies, event_definition.baseline_batches)
    onset = find_degradation_onset(
        visible_accuracies,
        baseline,
        start_batch=event_definition.baseline_batches,
        accuracy_drop=event_definition.accuracy_drop,
        consecutive_batches=event_definition.consecutive_batches,
    )
    return baseline, onset, len(visible_accuracies)


def run_early_warning_episode(
    episode: ScenarioEpisode,
    *,
    label_delay_batches: int,
    monitor_config: MonitorConfig | None = None,
    detector_reference_rows: int = 512,
    risk_predictor: RiskPredictor | None = None,
    risk_threshold: float | None = None,
    fallback_threshold: float | None = None,
    baseline_thresholds: Mapping[str, float] | None = None,
    risk_target_updater: RiskTargetUpdater | None = None,
    output_directory: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Replay one episode while keeping labels outside the live monitor.

    The optional predictor sees only causal risk features. The optional target
    updater receives examples only after their risk labels mature. The base
    classifier is fit on the episode reference rows once and remains frozen.
    """
    protocol = DEFAULT_EARLY_WARNING_CONFIG.protocol
    if label_delay_batches not in protocol.label_delays_batches:
        raise ValueError(f"unsupported label delay: {label_delay_batches}")
    if len(episode.stream_batches) != len(episode.stream_labels):
        raise ValueError("episode stream batches and labels must have equal lengths")
    if not episode.stream_batches:
        raise ValueError("episode must contain at least one stream batch")
    if detector_reference_rows < 2:
        raise ValueError("detector_reference_rows must be at least 2")
    allowed_baselines = {
        "statistical_only",
        "confidence_only",
        "fixed_weighted_score",
        "recent_slope_forecast",
    }
    thresholds = dict(baseline_thresholds or {})
    if set(thresholds) - allowed_baselines:
        raise ValueError("baseline_thresholds contains an unknown baseline")
    for name, threshold in (("risk_threshold", risk_threshold), ("fallback_threshold", fallback_threshold), *thresholds.items()):
        if threshold is not None and (not isfinite(threshold) or threshold < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if fallback_threshold is None:
        fallback_threshold = thresholds.get("fixed_weighted_score")

    feature_columns = episode.feature_schema.continuous + episode.feature_schema.categorical
    batch_size = episode.batch_size
    if batch_size < 2:
        raise ValueError("scenario batch size must be at least 2")
    for index, (batch, labels) in enumerate(zip(episode.stream_batches, episode.stream_labels)):
        if len(batch) != len(labels):
            raise ValueError(f"labels for batch {index} do not match its feature rows")
        if list(batch.columns) != feature_columns:
            raise ValueError(f"batch {index} columns do not match the episode feature schema")

    classifier = DEFAULT_EARLY_WARNING_CONFIG.classifier.build_pipeline()
    episode_started_at = perf_counter()
    classifier.fit(episode.reference_frame[feature_columns], episode.reference_labels)
    model = SklearnModelAdapter(classifier)

    detector_reference = episode.reference_frame[feature_columns]
    if len(detector_reference) > detector_reference_rows:
        detector_reference = detector_reference.sample(
            n=detector_reference_rows,
            random_state=42,
        ).sort_index().reset_index(drop=True)

    active_config = deepcopy(monitor_config) if monitor_config is not None else MonitorConfig()
    active_config.batch_size = batch_size
    active_config.window_size = batch_size
    store = DriftStore(":memory:")
    monitoring_system = DriftMonitoringSystem(
        config=active_config,
        schema=episode.feature_schema,
        model=model,
        store=store,
    )

    run_metadata = DEFAULT_EARLY_WARNING_CONFIG.metadata_for_run(
        dataset=DEFAULT_EARLY_WARNING_CONFIG.primary_dataset,
        scenario=episode.scenario,
        seed=episode.seed,
        label_delay_batches=label_delay_batches,
        feature_columns=feature_columns,
    )
    scenario_metadata = {
        key: value
        for key, value in episode.metadata.items()
        if key != "evaluation_only"
    }

    batch_count = len(episode.stream_batches)
    horizon = protocol.horizon_batches
    confirmation_batches = protocol.event.consecutive_batches
    complete_target_batches = set(
        complete_risk_target_batches(
            batch_count,
            horizon_batches=horizon,
            event_consecutive_batches=confirmation_batches,
        )
    )
    records: list[dict[str, Any]] = []
    signal_history: list[dict[str, float]] = []
    risk_feature_builder = RiskFeatureBuilder()
    realized_accuracies: list[float | None] = [None] * batch_count
    pending_labels: dict[int, dict[str, Any]] = {}
    pending_targets: dict[int, dict[str, Any]] = {}
    released_targets: list[dict[str, Any]] = []
    event_baseline_accuracy: float | None = None
    event_baseline_available_batch: int | None = None
    event_onset_batch: int | None = None
    event_confirmation_available_batch: int | None = None

    def update_event_state(clock_batch: int) -> None:
        nonlocal event_baseline_accuracy, event_baseline_available_batch, event_onset_batch
        nonlocal event_confirmation_available_batch
        baseline, onset, visible_count = _event_state(realized_accuracies)
        if baseline is not None and event_baseline_available_batch is None:
            event_baseline_accuracy = baseline
            event_baseline_available_batch = clock_batch
        if onset is not None and event_onset_batch is None:
            event_onset_batch = onset
            event_confirmation_available_batch = clock_batch
            confirmation_batch = onset + confirmation_batches - 1
            records[onset]["evaluation_only"]["event_onset_batch"] = onset
            records[confirmation_batch]["evaluation_only"]["event_confirmed_at_batch"] = clock_batch
            records[confirmation_batch]["evaluation_only"]["event_confirmation_batch"] = confirmation_batch
        if visible_count >= protocol.event.baseline_batches:
            for record in records:
                record["evaluation_only"]["baseline_accuracy"] = event_baseline_accuracy
                record["evaluation_only"]["baseline_accuracy_available_batch"] = event_baseline_available_batch

    def release_due_labels(clock_batch: int, *, include_current_batch: bool, phase: str) -> None:
        due_sources = [
            source_batch
            for source_batch, queued in pending_labels.items()
            if queued["available_batch"] <= clock_batch
            and (include_current_batch or source_batch < clock_batch)
        ]
        for source_batch in sorted(due_sources):
            queued = pending_labels.pop(source_batch)
            accuracy = float(np.mean(queued["predictions"] == queued["labels"]))
            realized_accuracies[source_batch] = accuracy
            evaluation = records[source_batch]["evaluation_only"]
            evaluation["realized_accuracy"] = accuracy
            evaluation["accuracy_released_at_batch"] = clock_batch
            evaluation["accuracy_release_phase"] = phase
            records[source_batch]["label_matured"] = True
            update_event_state(clock_batch)

    def release_due_targets(clock_batch: int, *, phase: str) -> None:
        due_examples = [
            example_batch
            for example_batch, queued in pending_targets.items()
            if queued["available_batch"] <= clock_batch
            and realized_accuracies[queued["confirmation_batch"]] is not None
        ]
        for example_batch in sorted(due_examples):
            queued = pending_targets.pop(example_batch)
            target = int(
                event_onset_batch is not None
                and example_batch < event_onset_batch <= example_batch + horizon
            )
            packet = {
                "scenario": episode.scenario,
                "seed": episode.seed,
                "label_delay_batches": label_delay_batches,
                "example_batch": example_batch,
                "label": target,
                "confirmation_batch": queued["confirmation_batch"],
                "available_batch": queued["available_batch"],
                "released_at_batch": clock_batch,
                "release_phase": phase,
                "features": deepcopy(records[example_batch]["risk_features"]),
            }
            records[example_batch]["evaluation_only"]["risk_target_label"] = target
            records[example_batch]["evaluation_only"]["risk_target_released_at_batch"] = clock_batch
            records[example_batch]["evaluation_only"]["risk_target_release_phase"] = phase
            records[example_batch]["risk_target_matured"] = True
            released_targets.append(packet)
            if risk_target_updater is not None:
                risk_target_updater(deepcopy(packet))

    try:
        monitoring_system.fit_detectors(detector_reference)

        for batch_index, (batch, labels) in enumerate(zip(episode.stream_batches, episode.stream_labels)):
            release_due_labels(
                batch_index,
                include_current_batch=False,
                phase="before_batch",
            )
            release_due_targets(batch_index, phase="before_batch")

            predictions, prediction_summary = _prediction_summary(
                model,
                batch,
                active_config.low_confidence_threshold,
            )
            # Labels are intentionally omitted from this call. Accuracy is
            # computed later by the private delayed-label evaluator above.
            monitor_result = monitoring_system.process_batch(
                batch_index,
                detector_reference,
                batch,
                batch_labels=None,
                evaluate_forecasts=False,
            )
            if monitor_result.accuracy is not None:
                raise RuntimeError("live monitor unexpectedly received a realized accuracy")

            signals = _signal_payload(monitor_result, prediction_summary)
            signal_history.append(signals)
            risk_features = risk_feature_builder.build(signal_history)
            baseline_scores = compute_baseline_scores(
                signal_history,
                horizon_batches=horizon,
            )
            risk_probability = risk_predictor(deepcopy(risk_features)) if risk_predictor else None
            if risk_probability is not None:
                risk_probability = float(risk_probability)
                if not isfinite(risk_probability) or not 0.0 <= risk_probability <= 1.0:
                    raise ValueError("risk_predictor must return a finite probability in [0, 1]")
            if risk_probability is None:
                active_warning_score = baseline_scores["fixed_weighted_score"]
                active_warning_threshold = fallback_threshold
                active_warning_source = "fixed_score_fallback"
            else:
                active_warning_score = risk_probability
                active_warning_threshold = risk_threshold
                active_warning_source = "risk_calibrator"
            risk_alarm = (
                bool(active_warning_score >= active_warning_threshold)
                if active_warning_threshold is not None
                else None
            )

            target_available_batch = (
                risk_target_available_batch(
                    batch_index,
                    horizon_batches=horizon,
                    label_delay_batches=label_delay_batches,
                    event_consecutive_batches=confirmation_batches,
                )
                if batch_index in complete_target_batches
                else None
            )
            evaluation_only = {
                "realized_accuracy": None,
                "accuracy_released_at_batch": None,
                "accuracy_release_phase": None,
                "baseline_accuracy": event_baseline_accuracy,
                "baseline_accuracy_available_batch": event_baseline_available_batch,
                "event_onset_batch": None,
                "risk_target_label": None,
                "risk_target_released_at_batch": None,
                "risk_target_release_phase": None,
            }
            record = {
                "episode": {
                    "scenario": episode.scenario,
                    "seed": episode.seed,
                },
                "batch_index": batch_index,
                "signals": signals,
                "baseline_scores": baseline_scores,
                "baseline_alarms": {
                    name: (
                        bool(baseline_scores[name] >= threshold)
                        if threshold is not None
                        else None
                    )
                    for name, threshold in thresholds.items()
                },
                "prediction_summaries": prediction_summary,
                "risk_features": risk_features,
                "risk_probability": risk_probability,
                "risk_alarm": risk_alarm,
                "active_warning_score": active_warning_score,
                "active_warning_threshold": active_warning_threshold,
                "active_warning_source": active_warning_source,
                "label_release_batch": batch_index + label_delay_batches,
                "label_matured": False,
                "risk_target_available_batch": target_available_batch,
                "risk_target_matured": False,
                "evaluation_only": evaluation_only,
            }
            records.append(record)

            pending_labels[batch_index] = {
                "available_batch": batch_index + label_delay_batches,
                "predictions": predictions.copy(),
                "labels": labels.to_numpy(copy=True),
            }
            if target_available_batch is not None:
                confirmation_batch = batch_index + horizon + confirmation_batches - 1
                pending_targets[batch_index] = {
                    "available_batch": target_available_batch,
                    "confirmation_batch": confirmation_batch,
                }

            if label_delay_batches == 0:
                release_due_labels(
                    batch_index,
                    include_current_batch=True,
                    phase="after_batch",
                )
            release_due_targets(batch_index, phase="after_batch")

        final_release_clock = batch_count - 1 + label_delay_batches
        for clock_batch in range(batch_count, final_release_clock + 1):
            release_due_labels(
                clock_batch,
                include_current_batch=False,
                phase="after_episode",
            )
            release_due_targets(clock_batch, phase="after_episode")

        event_batches = [event_onset_batch] if event_onset_batch is not None else []
        risk_alert_batches = [
            int(record["batch_index"])
            for record in records
            if record["risk_alarm"] is True
        ]
        risk_alert_metrics = evaluate_episode_alerts(
            risk_alert_batches,
            event_batches,
            horizon_batches=horizon,
            clean_batch_indices=stable_clean_batch_indices(
                total_batches=batch_count,
                event_batches=event_batches,
                horizon_batches=horizon,
            ),
        )
        baseline_method_metrics = {}
        for method_name in allowed_baselines:
            if method_name not in thresholds:
                continue
            method_alert_batches = [
                int(record["batch_index"])
                for record in records
                if record["baseline_alarms"].get(method_name) is True
            ]
            method_metrics = evaluate_episode_alerts(
                method_alert_batches,
                event_batches,
                horizon_batches=horizon,
                clean_batch_indices=stable_clean_batch_indices(
                    total_batches=batch_count,
                    event_batches=event_batches,
                    horizon_batches=horizon,
                ),
            )
            baseline_method_metrics[method_name] = {
                "threshold": thresholds[method_name],
                "alert_batches": method_alert_batches,
                "alerts_generated": len(method_alert_batches),
                "alerts_without_confirmed_event": method_metrics["alerts_without_confirmed_event"],
                "false_alerts_per_100_clean_batches": method_metrics["false_alerts_per_100_clean_batches"],
                "event_precision": method_metrics["event_precision"],
                "event_recall": method_metrics["event_recall"],
                "matched_event_count": method_metrics["matched_event_count"],
                "missed_event_count": len(event_batches) - method_metrics["matched_event_count"],
                "warning_lead_batches": method_metrics["warning_lead_batches"],
                "detection_lag_batches": method_metrics["detection_lag_batches"],
            }
        complete_accuracy_series = [
            float(value) for value in realized_accuracies if value is not None
        ]
        elapsed_seconds = perf_counter() - episode_started_at
        summary = {
            "batches_processed": batch_count,
            "runtime_seconds": elapsed_seconds,
            "runtime_seconds_per_batch": elapsed_seconds / batch_count,
            "baseline_accuracy": event_baseline_accuracy,
            "baseline_accuracy_available_batch": event_baseline_available_batch,
            "accuracy_by_batch": realized_accuracies,
            "event_present": event_onset_batch is not None,
            "event_onset_batch": event_onset_batch,
            "event_confirmation_available_batch": event_confirmation_available_batch,
            "risk_pipeline_alerts": risk_alert_batches,
            "risk_pipeline_alerts_generated": len(risk_alert_batches),
            "risk_pipeline_alerts_without_confirmed_event": risk_alert_metrics["alerts_without_confirmed_event"],
            "risk_pipeline_false_alerts_per_100_clean_batches": risk_alert_metrics["false_alerts_per_100_clean_batches"],
            "risk_pipeline_event_precision": risk_alert_metrics["event_precision"],
            "risk_pipeline_event_recall": risk_alert_metrics["event_recall"],
            "risk_pipeline_matched_event_count": risk_alert_metrics["matched_event_count"],
            "risk_pipeline_missed_event_count": len(event_batches) - risk_alert_metrics["matched_event_count"],
            "risk_pipeline_warning_lead_batches": risk_alert_metrics["warning_lead_batches"],
            "risk_pipeline_detection_lag_batches": risk_alert_metrics["detection_lag_batches"],
            "baseline_methods": baseline_method_metrics,
            "released_label_count": sum(value is not None for value in realized_accuracies),
            "released_risk_target_count": len(released_targets),
            "pending_label_count": len(pending_labels),
            "pending_risk_target_count": len(pending_targets),
            "mean_accuracy": float(np.mean(complete_accuracy_series)) if complete_accuracy_series else None,
        }
        payload: dict[str, Any] = {
            "metadata": {
                "run": run_metadata,
                "scenario": scenario_metadata,
                "detector_reference_rows": len(detector_reference),
                "monitor": {
                    "weights": list(active_config.weights),
                    "fixed_score_threshold": active_config.ds_crit,
                    "low_confidence_threshold": active_config.low_confidence_threshold,
                    "alert_confirmation_batches": active_config.alert_confirmation_batches,
                    "drift_cooldown_batches": active_config.drift_cooldown_batches,
                    "statistical_bins": active_config.statistical_bins,
                },
                "risk_model": "not_fitted" if risk_predictor is None else "provided_callback",
                "risk_threshold": risk_threshold,
                "fallback_threshold": fallback_threshold,
                "baseline_thresholds": thresholds,
                "live_monitor_receives_labels": False,
                "risk_features_use_labels": False,
            },
            "label_delay_batches": label_delay_batches,
            "summary": summary,
            "batch_records": records,
            "released_risk_targets": released_targets,
        }
    finally:
        store.close()

    output_root = Path(output_directory) if output_directory is not None else DEFAULT_EARLY_WARNING_CONFIG.output_directory
    output_path = (
        output_root
        / episode.scenario
        / f"seed-{episode.seed:02d}"
        / f"delay-{label_delay_batches}"
        / "episode.json"
    )
    if persist:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = str(output_path)
        output_path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    else:
        payload["output_path"] = None
    return payload
