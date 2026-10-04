"""Tests for MLflow tracking wrapper — graceful no-op behavior."""
from __future__ import annotations

from dews.config import MonitorConfig
from dews.mlflow_tracking import MLflowTracker


def test_tracker_disabled_without_start_run() -> None:
    config = MonitorConfig()
    tracker = MLflowTracker(config)
    assert not tracker.enabled
    # All methods should no-op gracefully
    tracker.log_drift_metrics(batch_index=0, drift_score=0.5, s_stat=0.2, s_emb=0.1, s_conf=0.1)
    tracker.log_alert({"batch_index": 0, "severity": "Critical"})
    tracker.log_summary({"drift_lag": 2})
    tracker.end_run()


def test_tracker_noop_when_mlflow_unavailable() -> None:
    config = MonitorConfig()
    tracker = MLflowTracker(config)
    # Simulate MLflow not being available
    import dews.mlflow_tracking as mt
    original = mt._mlflow_available
    mt._mlflow_available = False
    try:
        assert not tracker.enabled
        tracker.log_drift_metrics(batch_index=0, drift_score=0.5, s_stat=0.2, s_emb=0.1, s_conf=0.1)
        tracker.end_run()
    finally:
        mt._mlflow_available = original


def test_tracker_log_forecast_noop() -> None:
    config = MonitorConfig()
    tracker = MLflowTracker(config)
    # Should not raise even when disabled
    tracker.log_forecast(batch_index=0, ttd_batches=3.5, model_name="holtwinters")
    tracker.log_forecast(batch_index=1, ttd_batches=None, model_name="linear")
