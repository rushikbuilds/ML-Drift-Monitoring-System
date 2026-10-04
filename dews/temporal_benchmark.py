"""Temporal benchmark support for the public Electricity dataset.

The benchmark keeps the Digits demo intact and evaluates DEWS on a genuinely
ordered stream. The loader accepts a local CSV for offline/reproducible runs;
when no path is supplied it uses scikit-learn's OpenML cache/download.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .alerts import AlertEngine
from .calibration import simplex_weights
from .config import MonitorConfig
from .data import FeatureSchema
from .forecasting import DriftForecaster
from .integrations import IntegrationManager
from .models import SklearnModelAdapter
from .monitoring import DriftMonitoringSystem
from .retraining import RetrainingConfig
from .storage import DriftStore


@dataclass(slots=True)
class TemporalBenchmarkResult:
    dataset: str
    reference_rows: int
    stream_rows: int
    batches_processed: int
    baseline_accuracy: float
    final_accuracy: float
    accuracy_degradation_batch: int | None
    first_drift_alert_batch: int | None
    detection_lag: float | None
    false_alerts_before_degradation: int
    alerts_generated: int
    warning_lead_batches: float | None = None
    alert_batches: list[int] | None = None


def calibrate_electricity_weights(
    frame: pd.DataFrame,
    *,
    candidates: list[tuple[float, float, float]] | None = None,
    reference_fractions: tuple[float, ...] = (0.50, 0.60),
    batch_size: int = 256,
    max_detector_reference_rows: int = 2000,
) -> dict:
    """Select fusion weights across chronological Electricity folds.

    Lower scores are better. False alerts are penalized more than detection
    lag because a useful monitor must remain quiet during normal periods.
    """
    candidate_list = list(candidates or simplex_weights(step=0.5))
    baseline = (0.45, 0.35, 0.20)
    if baseline not in candidate_list:
        candidate_list.append(baseline)
    reports = []
    for weights in candidate_list:
        fold_reports = []
        for fraction in reference_fractions:
            config = MonitorConfig(
                window_size=batch_size,
                batch_size=batch_size,
                forecast_horizon=8,
                ds_crit=0.30,
                weights=weights,
                retraining_enabled=False,
            )
            fold_reports.append(run_electricity_benchmark(
                frame,
                config=config,
                reference_fraction=fraction,
                batch_size=batch_size,
                max_detector_reference_rows=max_detector_reference_rows,
            ))
        false_alerts = float(np.mean([report["false_alerts_before_degradation"] for report in fold_reports]))
        lag = float(np.mean([
            report["detection_lag"] if report["detection_lag"] is not None else report["batches_processed"]
            for report in fold_reports
        ]))
        objective = 10.0 * false_alerts + lag
        reports.append({
            "weights": weights,
            "objective": objective,
            "false_alerts_before_degradation": false_alerts,
            "detection_lag": lag,
            "folds": fold_reports,
        })
    reports.sort(key=lambda report: report["objective"])
    baseline_report = next(report for report in reports if tuple(report["weights"]) == baseline)
    return {
        "best": reports[0],
        "baseline": baseline_report,
        "candidates": reports,
        "protocol": {
            "reference_fractions": reference_fractions,
            "batch_size": batch_size,
            "max_detector_reference_rows": max_detector_reference_rows,
            "objective": "10*false_alerts_before_degradation + detection_lag",
        },
    }


def load_electricity(path: str | Path | None = None) -> pd.DataFrame:
    """Load Electricity from a local CSV or OpenML.

    Expected columns are the standard Electricity stream fields, including a
    binary ``class`` target. Column matching is case-insensitive.
    """
    if path is None:
        frame = fetch_openml("electricity", version=1, as_frame=True).frame
    else:
        frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    target = next((column for column in ("class", "target", "label") if column in frame.columns), None)
    if target is None:
        raise ValueError("Electricity data must contain a class, target, or label column")
    frame = frame.rename(columns={target: "target"}).dropna().reset_index(drop=True)
    frame["target"] = frame["target"].astype(str)
    return frame


def run_electricity_benchmark(
    frame: pd.DataFrame,
    *,
    config: MonitorConfig | None = None,
    reference_fraction: float = 0.60,
    batch_size: int = 256,
    degradation_tolerance: float = 0.05,
    max_detector_reference_rows: int = 2000,
    degradation_consecutive_batches: int = 2,
) -> dict:
    """Train chronologically and replay the remaining rows as live batches."""
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    target = next((column for column in ("class", "target", "label") if column in frame.columns), None)
    if target is None:
        raise ValueError("Electricity data must contain a class, target, or label column")
    frame = frame.rename(columns={target: "target"})
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must be between 0 and 1")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if max_detector_reference_rows < 2:
        raise ValueError("max_detector_reference_rows must be at least 2")

    feature_columns = [column for column in frame.columns if column not in {"target", "date", "datetime", "timestamp"}]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & frame["target"].notna()
    numeric = numeric.loc[valid].reset_index(drop=True)
    labels = frame.loc[valid, "target"].reset_index(drop=True)
    split = int(len(numeric) * reference_fraction)
    if split < batch_size or len(numeric) - split < batch_size:
        raise ValueError("Electricity dataset is too small for the selected reference and batch sizes")

    reference = numeric.iloc[:split].reset_index(drop=True)
    reference_labels = labels.iloc[:split]
    stream = numeric.iloc[split:].reset_index(drop=True)
    stream_labels = labels.iloc[split:].reset_index(drop=True)
    active_config = config or MonitorConfig(batch_size=batch_size, weights=(0.45, 0.35, 0.20))
    active_config.batch_size = batch_size
    active_config.window_size = max(active_config.window_size, batch_size)
    schema = FeatureSchema(continuous=feature_columns, categorical=[])
    detector_reference = reference
    if len(detector_reference) > max_detector_reference_rows:
        detector_reference = detector_reference.sample(
            n=max_detector_reference_rows,
            random_state=42,
        ).sort_index().reset_index(drop=True)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    model.fit(reference, reference_labels)
    adapter = SklearnModelAdapter(model)

    with tempfile.TemporaryDirectory(prefix="dews-electricity-") as directory:
        store = DriftStore(str(Path(directory) / "electricity.sqlite3"))
        system = DriftMonitoringSystem(
            config=active_config,
            schema=schema,
            model=adapter,
            store=store,
            alert_engine=AlertEngine(active_config),
            forecaster=DriftForecaster(horizon=active_config.forecast_horizon),
            integration_manager=IntegrationManager(
                retraining_config=RetrainingConfig(enabled=False),
            ),
        )
        # MMD uses pairwise distances, so bound detector reference size while
        # retaining all chronological rows for model training.
        system.fit_detectors(detector_reference)
        baseline_accuracy = float(np.mean(adapter.predict(reference) == reference_labels.to_numpy()))
        baseline_window: list[pd.DataFrame] = []
        accuracies: list[float] = []
        for batch_index, start in enumerate(range(0, len(stream), batch_size)):
            batch = stream.iloc[start:start + batch_size]
            batch_target = stream_labels.iloc[start:start + batch_size]
            if len(batch) < 2:
                continue
            baseline_window.append(batch)
            live_window = pd.concat(baseline_window, ignore_index=True).tail(active_config.window_size)
            result = system.process_batch(batch_index, detector_reference, live_window, batch_target)
            accuracies.append(result.accuracy if result.accuracy is not None else float("nan"))

        production_baseline = float(np.nanmean(accuracies[:min(5, len(accuracies))])) if accuracies else baseline_accuracy
        degradation_batch = None
        for index in range(5, len(accuracies) - degradation_consecutive_batches + 1):
            window = accuracies[index:index + degradation_consecutive_batches]
            if all(accuracy <= production_baseline - degradation_tolerance for accuracy in window):
                degradation_batch = index
                break
        alerts = store.load_alerts()
        first_alert = alerts[0]["batch_index"] if alerts else None
        false_alerts = sum(
            1 for alert in alerts
            if degradation_batch is not None and alert["batch_index"] < degradation_batch
        )
        detection_lag = None
        warning_lead = None
        if degradation_batch is not None and first_alert is not None:
            warning_lead = float(degradation_batch - first_alert)
            detection_lag = float(first_alert - degradation_batch)
        result = TemporalBenchmarkResult(
            dataset="electricity",
            reference_rows=len(reference),
            stream_rows=len(stream),
            batches_processed=len(accuracies),
            baseline_accuracy=production_baseline,
            final_accuracy=float(accuracies[-1]) if accuracies else float("nan"),
            accuracy_degradation_batch=degradation_batch,
            first_drift_alert_batch=first_alert,
            detection_lag=detection_lag,
            false_alerts_before_degradation=false_alerts,
            alerts_generated=len(alerts),
            warning_lead_batches=warning_lead,
            alert_batches=[int(alert["batch_index"]) for alert in alerts],
        )
        store.close()
    return result.__dict__ if hasattr(result, "__dict__") else {
        "dataset": result.dataset,
        "reference_rows": result.reference_rows,
        "stream_rows": result.stream_rows,
        "batches_processed": result.batches_processed,
        "baseline_accuracy": result.baseline_accuracy,
        "final_accuracy": result.final_accuracy,
        "accuracy_degradation_batch": result.accuracy_degradation_batch,
        "first_drift_alert_batch": result.first_drift_alert_batch,
        "detection_lag": result.detection_lag,
        "false_alerts_before_degradation": result.false_alerts_before_degradation,
        "alerts_generated": result.alerts_generated,
        "warning_lead_batches": result.warning_lead_batches,
        "alert_batches": result.alert_batches,
    }
