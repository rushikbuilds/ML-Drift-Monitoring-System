"""Tests for the retraining orchestrator and runners."""
from __future__ import annotations

from dews.retraining import (
    InProcessRunner,
    RetrainingConfig,
    RetrainingOrchestrator,
    WebhookRunner,
)


def test_orchestrator_skips_when_disabled() -> None:
    config = RetrainingConfig(enabled=False)
    orch = RetrainingOrchestrator(config=config)
    result = orch.trigger({"model_version": "v1"})
    assert result.status == "skipped"
    assert "disabled" in result.details.get("reason", "")


def test_orchestrator_respects_cooldown() -> None:
    config = RetrainingConfig(enabled=True, runner="in_process", cooldown_seconds=1000)
    orch = RetrainingOrchestrator(config=config, retrain_fn=lambda ctx: {"model_version": "v2"})
    r1 = orch.trigger({"model_version": "v1"})
    assert r1.status == "success"
    r2 = orch.trigger({"model_version": "v1"})
    assert r2.status == "cooldown"


def test_orchestrator_respects_max_attempts() -> None:
    config = RetrainingConfig(enabled=True, runner="in_process", cooldown_seconds=0, max_retrain_attempts=1)
    orch = RetrainingOrchestrator(config=config, retrain_fn=lambda ctx: {"model_version": "v2"})
    r1 = orch.trigger({"model_version": "v1"})
    assert r1.status == "success"
    r2 = orch.trigger({"model_version": "v1"})
    assert r2.status == "skipped"
    assert "max attempts" in r2.details.get("reason", "")


def test_orchestrator_reset_attempts() -> None:
    config = RetrainingConfig(enabled=True, runner="in_process", cooldown_seconds=0, max_retrain_attempts=1)
    orch = RetrainingOrchestrator(config=config, retrain_fn=lambda ctx: {})
    orch.trigger({"model_version": "v1"})
    orch.reset_attempts()
    r = orch.trigger({"model_version": "v1"})
    assert r.status == "success"


def test_in_process_runner_success() -> None:
    def my_retrain(ctx):
        return {"model_version": "v2_retrained", "epochs": 10}
    runner = InProcessRunner(retrain_fn=my_retrain)
    result = runner.execute({"model_version": "v1"})
    assert result.status == "success"
    assert result.model_version_new == "v2_retrained"


def test_in_process_runner_skips_without_fn() -> None:
    runner = InProcessRunner(retrain_fn=None)
    result = runner.execute({"model_version": "v1"})
    assert result.status == "skipped"


def test_in_process_runner_handles_exception() -> None:
    def bad_retrain(ctx):
        raise ValueError("training exploded")
    runner = InProcessRunner(retrain_fn=bad_retrain)
    result = runner.execute({"model_version": "v1"})
    assert result.status == "failed"
    assert "training exploded" in result.details.get("error", "")


def test_webhook_runner_skips_without_url() -> None:
    config = RetrainingConfig(webhook_url="")
    runner = WebhookRunner(config)
    result = runner.execute({"model_version": "v1"})
    assert result.status == "skipped"


def test_orchestrator_to_dict() -> None:
    config = RetrainingConfig(enabled=True, runner="local", max_retrain_attempts=5)
    orch = RetrainingOrchestrator(config=config)
    state = orch.to_dict()
    assert state["enabled"] is True
    assert state["runner"] == "local"
    assert state["max_attempts"] == 5
    assert state["attempt_count"] == 0
