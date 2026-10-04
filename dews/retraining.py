"""Automated retraining pipeline triggered by critical drift alerts.

Provides three execution strategies:
  1. **LocalRunner** — Runs retraining as a subprocess (dev/testing).
  2. **WebhookRunner** — POSTs to an external orchestrator (Airflow, Prefect Cloud, Argo).
  3. **InProcessRunner** — Runs the retrain callable directly in-process.

The ``RetrainingOrchestrator`` selects the appropriate runner based on
configuration and manages the retrain lifecycle, including cooldowns,
retrain history, and status tracking.

Usage:
    orch = RetrainingOrchestrator(config)
    result = orch.trigger(alert, reference_frame, reference_labels)
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib import request

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrainingResult:
    """Result of a retraining attempt."""
    triggered_at: str
    status: str  # "success", "failed", "skipped", "cooldown"
    runner: str  # "local", "webhook", "in_process"
    model_version_old: str
    model_version_new: str
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingConfig:
    """Configuration for the retraining pipeline."""
    enabled: bool = True
    runner: str = "local"  # "local", "webhook", "in_process"
    cooldown_seconds: float = 300.0
    retrain_command: str = "python train_and_monitor_pytorch.py"
    webhook_url: str = ""
    max_retrain_attempts: int = 3
    # Script path for local runner (relative to project root)
    script_path: str = "train_and_monitor_pytorch.py"


class LocalRunner:
    """Runs retraining as a local subprocess."""

    def __init__(self, config: RetrainingConfig) -> None:
        self.config = config

    def execute(self, context: dict[str, Any]) -> RetrainingResult:
        start = time.monotonic()
        cmd = self.config.retrain_command
        logger.info("LocalRunner: executing '%s'", cmd)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=600,
            )
            elapsed = time.monotonic() - start
            if result.returncode == 0:
                logger.info("LocalRunner: retraining succeeded in %.1fs", elapsed)
                return RetrainingResult(
                    triggered_at=datetime.now(timezone.utc).isoformat(),
                    status="success",
                    runner="local",
                    model_version_old=context.get("model_version", "unknown"),
                    model_version_new=context.get("model_version_new", "retrained"),
                    duration_seconds=round(elapsed, 2),
                    details={"stdout": result.stdout[-500:], "returncode": 0},
                )
            else:
                logger.error("LocalRunner: retraining failed (rc=%d)", result.returncode)
                return RetrainingResult(
                    triggered_at=datetime.now(timezone.utc).isoformat(),
                    status="failed",
                    runner="local",
                    model_version_old=context.get("model_version", "unknown"),
                    model_version_new="",
                    duration_seconds=round(elapsed, 2),
                    details={"stderr": result.stderr[-500:], "returncode": result.returncode},
                )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                runner="local",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=round(elapsed, 2),
                details={"error": "timeout"},
            )


class WebhookRunner:
    """Triggers retraining via an external orchestrator webhook (Airflow, Prefect, Argo)."""

    def __init__(self, config: RetrainingConfig) -> None:
        self.config = config

    def execute(self, context: dict[str, Any]) -> RetrainingResult:
        start = time.monotonic()
        url = self.config.webhook_url
        if not url:
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="skipped",
                runner="webhook",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=0.0,
                details={"error": "no webhook_url configured"},
            )

        payload = json.dumps({
            "trigger": "drift_alert",
            "model_version": context.get("model_version", "unknown"),
            "drift_score": context.get("drift_score", 0.0),
            "batch_index": context.get("batch_index", -1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")

        req = request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=10) as resp:
                status_code = resp.status
            elapsed = time.monotonic() - start
            logger.info("WebhookRunner: triggered %s (status=%d)", url, status_code)
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="success",
                runner="webhook",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="pending",
                duration_seconds=round(elapsed, 2),
                details={"url": url, "status_code": status_code},
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("WebhookRunner: failed to trigger %s: %s", url, exc)
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                runner="webhook",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=round(elapsed, 2),
                details={"url": url, "error": str(exc)},
            )


class InProcessRunner:
    """Runs a retraining callable directly in the current process."""

    def __init__(self, retrain_fn: Callable | None = None) -> None:
        self.retrain_fn = retrain_fn

    def execute(self, context: dict[str, Any]) -> RetrainingResult:
        start = time.monotonic()
        if self.retrain_fn is None:
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="skipped",
                runner="in_process",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=0.0,
                details={"error": "no retrain_fn provided"},
            )
        try:
            result_info = self.retrain_fn(context)
            elapsed = time.monotonic() - start
            new_version = result_info.get("model_version", "retrained") if isinstance(result_info, dict) else "retrained"
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="success",
                runner="in_process",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new=new_version,
                duration_seconds=round(elapsed, 2),
                details=result_info if isinstance(result_info, dict) else {},
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                runner="in_process",
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=round(elapsed, 2),
                details={"error": str(exc)},
            )


class RetrainingOrchestrator:
    """Manages the retraining lifecycle with cooldowns and history tracking."""

    def __init__(
        self,
        config: RetrainingConfig | None = None,
        retrain_fn: Callable | None = None,
    ) -> None:
        self.config = config or RetrainingConfig()
        self.retrain_fn = retrain_fn
        self.history: list[RetrainingResult] = []
        self._last_trigger_time: float = 0.0
        self._attempt_count: int = 0

        # Select runner based on config
        if self.config.runner == "webhook":
            self._runner = WebhookRunner(self.config)
        elif self.config.runner == "in_process":
            self._runner = InProcessRunner(retrain_fn)
        else:
            self._runner = LocalRunner(self.config)

    def trigger(self, context: dict[str, Any]) -> RetrainingResult:
        """Attempt to trigger a retraining run.

        Respects cooldown periods and max retry limits.
        """
        if not self.config.enabled:
            result = RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="skipped",
                runner=self.config.runner,
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=0.0,
                details={"reason": "retraining disabled"},
            )
            self.history.append(result)
            return result

        # Check cooldown
        now = time.monotonic()
        if self._last_trigger_time > 0 and (now - self._last_trigger_time) < self.config.cooldown_seconds:
            remaining = self.config.cooldown_seconds - (now - self._last_trigger_time)
            result = RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="cooldown",
                runner=self.config.runner,
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=0.0,
                details={"cooldown_remaining_seconds": round(remaining, 1)},
            )
            self.history.append(result)
            return result

        # Check max attempts
        if self._attempt_count >= self.config.max_retrain_attempts:
            result = RetrainingResult(
                triggered_at=datetime.now(timezone.utc).isoformat(),
                status="skipped",
                runner=self.config.runner,
                model_version_old=context.get("model_version", "unknown"),
                model_version_new="",
                duration_seconds=0.0,
                details={"reason": f"max attempts reached ({self.config.max_retrain_attempts})"},
            )
            self.history.append(result)
            return result

        # Execute the retraining
        self._last_trigger_time = now
        self._attempt_count += 1
        result = self._runner.execute(context)
        self.history.append(result)

        logger.info(
            "Retraining attempt #%d: status=%s runner=%s old=%s new=%s",
            self._attempt_count, result.status, result.runner,
            result.model_version_old, result.model_version_new,
        )
        return result

    def reset_attempts(self) -> None:
        """Reset the attempt counter (e.g., after a successful model swap)."""
        self._attempt_count = 0

    def to_dict(self) -> dict:
        """Serialize orchestrator state for API responses."""
        return {
            "enabled": self.config.enabled,
            "runner": self.config.runner,
            "attempt_count": self._attempt_count,
            "max_attempts": self.config.max_retrain_attempts,
            "cooldown_seconds": self.config.cooldown_seconds,
            "total_triggers": len(self.history),
            "last_result": asdict(self.history[-1]) if self.history else None,
        }
