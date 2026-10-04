"""MLflow integration for experiment tracking and model versioning.

This module provides a lightweight wrapper around MLflow to:
- Track drift scores, component signals, and forecasts as MLflow metrics
- Log model versions and reference data hashes as run parameters
- Register model artifacts in the MLflow Model Registry

Usage:
    tracker = MLflowTracker(config)
    tracker.start_run()
    tracker.log_drift_metrics(batch_index, drift_score, s_stat, s_emb, s_conf)
    tracker.log_alert(alert_record)
    tracker.end_run()

When MLflow is not installed, all operations silently no-op so the rest of
the system continues to work without the dependency.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import MonitorConfig

logger = logging.getLogger(__name__)

# Lazy import: MLflow is an optional dependency
_mlflow = None
_mlflow_available = False


def _ensure_mlflow():
    global _mlflow, _mlflow_available
    if _mlflow is not None:
        return _mlflow_available
    try:
        import mlflow as _m
        _mlflow = _m
        _mlflow_available = True
    except ImportError:
        _mlflow_available = False
        logger.info("MLflow is not installed — tracking is disabled. Install with: pip install mlflow")
    return _mlflow_available


class MLflowTracker:
    """Thin wrapper around MLflow for drift monitoring experiment tracking."""

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self._run = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and _mlflow_available

    def start_run(self, run_name: str | None = None) -> None:
        """Start an MLflow tracking run. No-ops if MLflow is unavailable."""
        if not _ensure_mlflow():
            return
        assert _mlflow is not None

        if self.config.mlflow_tracking_uri:
            _mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        _mlflow.set_experiment(self.config.mlflow_experiment_name)
        self._run = _mlflow.start_run(run_name=run_name or f"dews-{self.config.model_version}")
        self._enabled = True

        # Log config as parameters
        _mlflow.log_params({
            "model_version": self.config.model_version,
            "reference_data_hash": self.config.reference_data_hash,
            "window_size": self.config.window_size,
            "batch_size": self.config.batch_size,
            "forecast_horizon": self.config.forecast_horizon,
            "ds_crit": self.config.ds_crit,
            "weights": str(self.config.weights),
            "top_k_features": self.config.top_k_features,
        })
        logger.info("MLflow run started: %s", self._run.info.run_id)

    def log_drift_metrics(
        self,
        batch_index: int,
        drift_score: float,
        s_stat: float,
        s_emb: float,
        s_conf: float,
        accuracy: float | None = None,
    ) -> None:
        """Log drift metrics for a single batch as MLflow step metrics."""
        if not self.enabled:
            return
        assert _mlflow is not None

        metrics: dict[str, float] = {
            "drift_score": drift_score,
            "s_stat": s_stat,
            "s_emb": s_emb,
            "s_conf": s_conf,
        }
        if accuracy is not None:
            metrics["accuracy"] = accuracy

        _mlflow.log_metrics(metrics, step=batch_index)

    def log_forecast(self, batch_index: int, ttd_batches: float | None, model_name: str) -> None:
        """Log forecast metadata."""
        if not self.enabled:
            return
        assert _mlflow is not None

        metrics: dict[str, float] = {}
        if ttd_batches is not None:
            metrics["ttd_batches"] = ttd_batches
        if metrics:
            _mlflow.log_metrics(metrics, step=batch_index)

    def log_alert(self, alert_dict: dict[str, Any]) -> None:
        """Log an alert as tagged metadata."""
        if not self.enabled:
            return
        assert _mlflow is not None

        _mlflow.set_tag(f"alert_batch_{alert_dict.get('batch_index', 'unknown')}", alert_dict.get("severity", "unknown"))
        if alert_dict.get("top_features"):
            features_str = ", ".join(f["feature"] for f in alert_dict["top_features"][:5])
            _mlflow.set_tag(f"alert_features_{alert_dict.get('batch_index', 'unknown')}", features_str)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log a final evaluation summary."""
        if not self.enabled:
            return
        assert _mlflow is not None

        safe_metrics = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)) and value is not None:
                safe_metrics[f"summary_{key}"] = value
        if safe_metrics:
            _mlflow.log_metrics(safe_metrics)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enabled:
            return
        assert _mlflow is not None

        _mlflow.end_run()
        self._enabled = False
        logger.info("MLflow run ended.")
