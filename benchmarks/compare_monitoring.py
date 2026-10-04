"""Compare DEWS, Evidently, and NannyML on identical temporal batches.

The benchmark uses the Electricity dataset, trains one shared classifier on the
chronological reference split, and sends the same current batches to each tool.
Optional tools are reported as unavailable instead of aborting the run.
"""
from __future__ import annotations

import argparse
import gc
import json
try:
    import resource
except ImportError:
    resource = None
import sys
import time
from pathlib import Path
from typing import Any

# Allow ``python benchmarks/compare_monitoring.py`` as well as
# ``python -m benchmarks.compare_monitoring`` from the repository root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dews.temporal_benchmark import load_electricity, run_electricity_benchmark


def _prepare(frame: pd.DataFrame, reference_fraction: float, batch_size: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    target = next((column for column in ("class", "target", "label") if column in frame.columns), None)
    if target is None:
        raise ValueError("Dataset must contain class, target, or label")
    frame = frame.rename(columns={target: "target"}).dropna().reset_index(drop=True)
    feature_columns = [column for column in frame.columns if column not in {"target", "date", "datetime", "timestamp"}]
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid = features.notna().all(axis=1)
    features = features.loc[valid].reset_index(drop=True)
    labels = frame.loc[valid, "target"].astype(str).reset_index(drop=True)
    split = int(len(features) * reference_fraction)
    if split < batch_size or len(features) - split < batch_size:
        raise ValueError("Dataset is too small for the selected reference fraction and batch size")
    return features.iloc[:split].reset_index(drop=True), labels.iloc[:split], features.iloc[split:].reset_index(drop=True), labels.iloc[split:]


def _base_metrics(tool: str, status: str, started: float, **values: Any) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": status,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2) if resource else None,
        "false_alerts_before_degradation": None,
        "first_alert_batch": None,
        "accuracy_degradation_batch": None,
        "detection_lag": None,
        "warning_lead_batches": None,
        "alerts_generated": None,
        "input_drift_alerts": None,
        "performance_alerts": None,
        "estimated_accuracy_mae": None,
        **values,
    }


def _accuracy_degradation(model, stream: pd.DataFrame, labels: pd.Series, batch_size: int, baseline_batches: int = 5, consecutive_batches: int = 2) -> tuple[list[float], int | None, float, int | None]:
    accuracies = []
    for start in range(0, len(stream), batch_size):
        batch = stream.iloc[start:start + batch_size]
        target = labels.iloc[start:start + batch_size]
        if len(batch) < 2:
            continue
        accuracies.append(float(np.mean(model.predict(batch) == target.to_numpy())))
    baseline = float(np.mean(accuracies[:min(baseline_batches, len(accuracies))]))
    degradation = None
    for index in range(baseline_batches, len(accuracies) - consecutive_batches + 1):
        if all(value <= baseline - 0.05 for value in accuracies[index:index + consecutive_batches]):
            degradation = index
            break
    rolling_degradation = None
    for index in range(baseline_batches + 1, len(accuracies) - consecutive_batches + 1):
        rolling_baseline = float(np.mean(accuracies[index - baseline_batches:index]))
        if all(value <= rolling_baseline - 0.05 for value in accuracies[index:index + consecutive_batches]):
            rolling_degradation = index
            break
    return accuracies, degradation, baseline, rolling_degradation


def run_dews(frame: pd.DataFrame, reference_fraction: float, batch_size: int, detector_reference_rows: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = run_electricity_benchmark(
            frame,
            reference_fraction=reference_fraction,
            batch_size=batch_size,
            max_detector_reference_rows=detector_reference_rows,
        )
        result["first_alert_batch"] = result.get("first_drift_alert_batch")
        result["warning_lead_batches"] = result.get("warning_lead_batches")
        result["input_drift_alerts"] = result.get("alerts_generated")
        result["performance_alerts"] = 0
        return _base_metrics("dews", "available", started, **result)
    except Exception as exc:
        return _base_metrics("dews", "error", started, error=f"{type(exc).__name__}: {exc}")


def run_evidently(reference: pd.DataFrame, stream: pd.DataFrame, batch_size: int, degradation_batch: int | None, drift_share_threshold: float = 0.25) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        return _base_metrics("evidently", "not_installed", started, accuracy_degradation_batch=degradation_batch, error=str(exc))

    try:
        alerts = []
        for batch_index, start in enumerate(range(0, len(stream), batch_size)):
            current = stream.iloc[start:start + batch_size]
            if len(current) < 2:
                continue
            result = Report(metrics=[DataDriftPreset()]).run(reference_data=reference, current_data=current)
            payload = result.dict() if hasattr(result, "dict") else json.loads(result.json())
            drift_metric = next(
                (metric for metric in payload.get("metrics", []) if metric.get("metric_name", "").startswith("DriftedColumnsCount")),
                None,
            )
            drifted = bool(drift_metric and drift_metric.get("value", {}).get("share", 0.0) >= drift_share_threshold)
            if drifted:
                alerts.append(batch_index)
        first_alert = alerts[0] if alerts else None
        false_alerts = sum(batch < degradation_batch for batch in alerts) if degradation_batch is not None else None
        lag = float(first_alert - degradation_batch) if first_alert is not None and degradation_batch is not None else None
        lead = float(degradation_batch - first_alert) if first_alert is not None and degradation_batch is not None else None
        return _base_metrics("evidently", "available", started, accuracy_degradation_batch=degradation_batch, first_alert_batch=first_alert, false_alerts_before_degradation=false_alerts, detection_lag=lag, warning_lead_batches=lead, alerts_generated=len(alerts), input_drift_alerts=len(alerts), performance_alerts=0, drift_share_threshold=drift_share_threshold)
    except Exception as exc:
        return _base_metrics("evidently", "error", started, accuracy_degradation_batch=degradation_batch, error=f"{type(exc).__name__}: {exc}")


def run_nannyml(reference: pd.DataFrame, reference_labels: pd.Series, stream: pd.DataFrame, labels: pd.Series, batch_size: int, model, degradation_batch: int | None, actual_accuracies: list[float]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import nannyml as nml
    except ImportError as exc:
        return _base_metrics("nannyml", "not_installed", started, accuracy_degradation_batch=degradation_batch, error=str(exc))

    try:
        reference_data = reference.copy()
        analysis = stream.copy()
        reference_predictions = model.predict(reference)
        analysis_predictions = model.predict(stream)
        reference_probabilities = model.predict_proba(reference).max(axis=1)
        analysis_probabilities = model.predict_proba(stream).max(axis=1)
        classes = {label: index for index, label in enumerate(sorted(set(reference_labels.astype(str))))}
        reference_predictions = np.asarray([classes[str(value)] for value in reference_predictions])
        analysis_predictions = np.asarray([classes.get(str(value), -1) for value in analysis_predictions])
        reference_targets = np.asarray([classes[str(value)] for value in reference_labels.astype(str)])
        analysis_targets = np.asarray([classes.get(str(value), -1) for value in labels.astype(str)])
        if len(classes) != 2 or np.any(analysis_predictions < 0) or np.any(analysis_targets < 0):
            return _base_metrics("nannyml", "unsupported", started, accuracy_degradation_batch=degradation_batch, error="CBPE comparison requires exactly two labels shared by reference and stream")
        reference_data["timestamp"] = pd.date_range("2020-01-01", periods=len(reference_data), freq="min")
        analysis["timestamp"] = pd.date_range(reference_data["timestamp"].iloc[-1] + pd.Timedelta(minutes=1), periods=len(analysis), freq="min")
        reference_data["prediction"] = reference_predictions
        analysis["prediction"] = analysis_predictions
        reference_data["prediction_probability"] = reference_probabilities
        analysis["prediction_probability"] = analysis_probabilities
        reference_data["target"] = reference_targets
        analysis["target"] = analysis_targets
        estimator = nml.CBPE(
            y_pred_proba="prediction_probability",
            y_pred="prediction",
            y_true="target",
            timestamp_column_name="timestamp",
            problem_type="classification_binary",
            metrics=["accuracy"],
            chunk_size=batch_size,
        )
        estimator.fit(reference_data)
        estimated = estimator.estimate(analysis)
        estimated_frame = estimated.to_df() if hasattr(estimated, "to_df") else pd.DataFrame(estimated)
        accuracy_columns = [column for column in estimated_frame.columns if isinstance(column, tuple) and column[0] == "accuracy"]
        value_column = next((column for column in accuracy_columns if column[1] == "value"), None)
        alert_column = next((column for column in accuracy_columns if column[1] == "alert"), None)
        period_column = next((column for column in estimated_frame.columns if isinstance(column, tuple) and column == ("chunk", "period")), None)
        analysis_rows = estimated_frame[period_column].eq("analysis") if period_column else pd.Series(True, index=estimated_frame.index)
        alerts = [index for index, flagged in enumerate(estimated_frame.loc[analysis_rows, alert_column].fillna(False)) if flagged] if alert_column else []
        first_alert = alerts[0] if alerts else None
        false_alerts = sum(alert < degradation_batch for alert in alerts) if degradation_batch is not None else None
        lag = float(first_alert - degradation_batch) if first_alert is not None and degradation_batch is not None else None
        lead = float(degradation_batch - first_alert) if first_alert is not None and degradation_batch is not None else None
        estimated_accuracy = estimated_frame.loc[analysis_rows, value_column].dropna().tolist() if value_column else []
        comparable = min(len(estimated_accuracy), len(actual_accuracies))
        mae = float(np.mean(np.abs(np.asarray(estimated_accuracy[:comparable]) - np.asarray(actual_accuracies[:comparable])))) if comparable else None
        return _base_metrics("nannyml", "available", started, accuracy_degradation_batch=degradation_batch, first_alert_batch=first_alert, false_alerts_before_degradation=false_alerts, detection_lag=lag, warning_lead_batches=lead, alerts_generated=len(alerts), input_drift_alerts=0, performance_alerts=len(alerts), estimated_accuracy=estimated_accuracy, realized_accuracy=actual_accuracies, estimated_accuracy_mae=mae)
    except Exception as exc:
        return _base_metrics("nannyml", "error", started, accuracy_degradation_batch=degradation_batch, error=f"{type(exc).__name__}: {exc}")


def _aggregate_results(fold_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated = []
    for tool in sorted({result["tool"] for result in fold_results}):
        rows = [result for result in fold_results if result["tool"] == tool]
        available = [row for row in rows if row["status"] == "available"]
        summary = dict(rows[0])
        summary["folds"] = len(rows)
        summary["status"] = "available" if len(available) == len(rows) else rows[0]["status"]
        for field in ("runtime_seconds", "peak_memory_mb", "false_alerts_before_degradation", "detection_lag", "warning_lead_batches", "alerts_generated", "input_drift_alerts", "performance_alerts", "estimated_accuracy_mae"):
            values = [row[field] for row in available if isinstance(row.get(field), (int, float))]
            summary[field] = round(float(np.mean(values)), 3) if values else None
            summary[f"{field}_std"] = round(float(np.std(values)), 3) if len(values) > 1 else 0.0 if values else None
        first_alerts = [row["first_alert_batch"] for row in available if row.get("first_alert_batch") is not None]
        summary["first_alert_batch"] = min(first_alerts) if first_alerts else None
        summary["accuracy_degradation_batch"] = [row.get("accuracy_degradation_batch") for row in rows]
        summary["rolling_accuracy_degradation_batch"] = [row.get("rolling_accuracy_degradation_batch") for row in rows]
        aggregated.append(summary)
    return aggregated


def run_comparison(data_path: str | None, output_dir: str, reference_fraction: float | tuple[float, ...], batch_size: int, detector_reference_rows: int, evidently_drift_share: float = 0.25, model_factory: str | None = None) -> dict[str, Any]:
    frame = load_electricity(data_path)
    fractions = (reference_fraction,) if isinstance(reference_fraction, float) else reference_fraction
    fold_results = []
    
    def get_model():
        if model_factory:
            import importlib
            module_name, func_name = model_factory.split(":")
            module = importlib.import_module(module_name)
            return getattr(module, func_name)()
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))

    for fraction in fractions:
        reference, reference_labels, stream, stream_labels = _prepare(frame, fraction, batch_size)
        model = get_model()
        model.fit(reference, reference_labels)
        actual_accuracies, degradation_batch, _, rolling_degradation_batch = _accuracy_degradation(model, stream, stream_labels, batch_size)
        dews_result = run_dews(frame, fraction, batch_size, detector_reference_rows)
        dews_result["accuracy_degradation_batch"] = degradation_batch
        dews_result["rolling_accuracy_degradation_batch"] = rolling_degradation_batch
        alert_batches = dews_result.get("alert_batches") or []
        dews_result["false_alerts_before_degradation"] = sum(batch < degradation_batch for batch in alert_batches) if degradation_batch is not None else None
        first_alert = dews_result.get("first_alert_batch")
        dews_result["detection_lag"] = float(first_alert - degradation_batch) if first_alert is not None and degradation_batch is not None else None
        dews_result["warning_lead_batches"] = float(degradation_batch - first_alert) if first_alert is not None and degradation_batch is not None else None
        fold_results.extend([
            dict(dews_result, fold_reference_fraction=fraction),
            dict(run_evidently(reference, stream, batch_size, degradation_batch, evidently_drift_share), fold_reference_fraction=fraction, rolling_accuracy_degradation_batch=rolling_degradation_batch),
            dict(run_nannyml(reference, reference_labels, stream, stream_labels, batch_size, model, degradation_batch, actual_accuracies), fold_reference_fraction=fraction, rolling_accuracy_degradation_batch=rolling_degradation_batch),
        ])
        gc.collect()
    results = _aggregate_results(fold_results)
    report = {
        "protocol": {
            "dataset": "electricity",
            "reference_fraction": fractions,
            "batch_size": batch_size,
            "detector_reference_rows": detector_reference_rows,
            "accuracy_degradation_batch": degradation_batch,
            "rolling_accuracy_degradation_batch": rolling_degradation_batch,
            "same_reference_and_batches": True,
            "fold_count": len(fractions),
            "evidently_drift_share_threshold": evidently_drift_share,
        },
        "results": results,
        "fold_results": fold_results,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(results).to_csv(destination / "comparison.csv", index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DEWS, Evidently, and NannyML on Electricity")
    parser.add_argument("--data", default="", help="Local Electricity CSV; empty uses OpenML")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--reference-fractions", default="0.5,0.6,0.7", help="Comma-separated chronological reference fractions; use 0.6 for a quick run")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--detector-reference-rows", type=int, default=500)
    parser.add_argument("--evidently-drift-share", type=float, default=0.25, help="Minimum share of drifted columns for an Evidently input-drift alert")
    parser.add_argument("--model-factory", type=str, default=None, help="Custom model factory in format module:function")
    args = parser.parse_args()
    fractions = tuple(float(value) for value in args.reference_fractions.split(",") if value.strip())
    report = run_comparison(args.data or None, args.output_dir, fractions, args.batch_size, args.detector_reference_rows, args.evidently_drift_share, args.model_factory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
