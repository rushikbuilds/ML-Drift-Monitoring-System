"""Tests for the IntegrationManager alert dispatch flow."""
from __future__ import annotations

from dews.alerts import AlertRecord
from dews.integrations import IntegrationManager
from dews.retraining import RetrainingConfig


def _make_alert(severity: str = "Critical", batch_index: int = 10) -> AlertRecord:
    return AlertRecord(
        timestamp="2024-01-01T00:00:00Z",
        batch_index=batch_index,
        severity=severity,
        message="Test alert",
        trigger_value=0.45,
        predicted_ttd=2.0,
        model_version="v1.0",
    )


def test_handle_alert_returns_action_records() -> None:
    mgr = IntegrationManager()
    alert = _make_alert("Warning")
    records = mgr.handle_alert(alert)
    # A Warning alert should produce at least a notification action
    assert len(records) >= 1
    assert records[0].target == "notification"


def test_notification_and_ticket_skipped_without_urls() -> None:
    mgr = IntegrationManager()
    alert = _make_alert("Critical")
    records = mgr.handle_alert(alert)
    targets = [r.target for r in records]
    assert "notification" in targets
    # Without URLs, actions should be skipped
    for r in records:
        if r.target in ("notification", "ticket"):
            assert r.status == "skipped"


def test_critical_alert_triggers_retraining_when_enabled() -> None:
    retrain_config = RetrainingConfig(
        enabled=True,
        runner="in_process",
        cooldown_seconds=0,
    )
    mgr = IntegrationManager(
        retraining_config=retrain_config,
        retrain_fn=lambda ctx: {"model_version": "v2"},
    )
    alert = _make_alert("Critical")
    records = mgr.handle_alert(alert)
    retrain_records = [r for r in records if r.target == "retraining"]
    assert len(retrain_records) == 1
    assert retrain_records[0].status == "success"


def test_non_critical_alert_does_not_trigger_retraining() -> None:
    retrain_config = RetrainingConfig(enabled=True, runner="in_process")
    mgr = IntegrationManager(
        retraining_config=retrain_config,
        retrain_fn=lambda ctx: {},
    )
    alert = _make_alert("Warning")
    records = mgr.handle_alert(alert)
    retrain_records = [r for r in records if r.target == "retraining"]
    assert len(retrain_records) == 0


def test_manual_retrain_trigger() -> None:
    retrain_config = RetrainingConfig(
        enabled=True,
        runner="in_process",
        cooldown_seconds=0,
    )
    mgr = IntegrationManager(
        retraining_config=retrain_config,
        retrain_fn=lambda ctx: {"model_version": "manual_v2"},
    )
    result = mgr.trigger_retrain_manually({"model_version": "v1", "drift_score": 0.5})
    assert result.status == "success"
    assert result.model_version_new == "manual_v2"
