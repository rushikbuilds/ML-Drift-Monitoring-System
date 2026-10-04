from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib import request

from .alerts import AlertRecord
from .retraining import RetrainingConfig, RetrainingOrchestrator, RetrainingResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IntegrationActionRecord:
    timestamp: str
    target: str
    status: str
    payload: dict


class IntegrationManager:
    def __init__(
        self,
        webhook_url: str | None = None,
        ticket_url: str | None = None,
        retraining_hook_url: str | None = None,
        retraining_config: RetrainingConfig | None = None,
        retrain_fn: Callable | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.ticket_url = ticket_url
        self.retraining_hook_url = retraining_hook_url
        self.history: list[IntegrationActionRecord] = []
        # Retraining orchestrator
        self.retraining_orchestrator = RetrainingOrchestrator(
            config=retraining_config, retrain_fn=retrain_fn,
        )

    def handle_alert(self, alert: AlertRecord) -> list[IntegrationActionRecord]:
        records: list[IntegrationActionRecord] = []
        payload = {
            "timestamp": alert.timestamp,
            "batch_index": alert.batch_index,
            "severity": alert.severity,
            "message": alert.message,
            "trigger_value": alert.trigger_value,
            "predicted_ttd": alert.predicted_ttd,
            "model_version": getattr(alert, "model_version", "unversioned"),
        }

        if self.webhook_url:
            records.append(self._post_json(self.webhook_url, payload, "notification"))
        else:
            records.append(self._record("notification", "skipped", payload))

        if alert.severity == "Critical" and self.ticket_url:
            records.append(self._post_json(self.ticket_url, payload, "ticket"))
        elif alert.severity == "Critical":
            records.append(self._record("ticket", "skipped", payload))

        # --- Retraining trigger on Critical alerts ---
        if alert.severity == "Critical":
            if self.retraining_orchestrator.config.enabled:
                context = {
                    "model_version": getattr(alert, "model_version", "unversioned"),
                    "drift_score": alert.trigger_value,
                    "batch_index": alert.batch_index,
                    "severity": alert.severity,
                }
                retrain_result = self.retraining_orchestrator.trigger(context)
                records.append(self._record(
                    "retraining",
                    retrain_result.status,
                    asdict(retrain_result),
                ))
                logger.info("Retraining trigger: status=%s", retrain_result.status)
            elif self.retraining_hook_url:
                records.append(self._post_json(self.retraining_hook_url, payload, "retraining"))
            else:
                records.append(self._record("retraining", "skipped", payload))

        self.history.extend(records)
        return records

    def trigger_retrain_manually(self, context: dict | None = None) -> RetrainingResult:
        """Manually trigger a retraining run (e.g., from API endpoint)."""
        ctx = context or {"model_version": "manual", "drift_score": 0.0, "batch_index": -1}
        return self.retraining_orchestrator.trigger(ctx)

    def _record(self, target: str, status: str, payload: dict) -> IntegrationActionRecord:
        return IntegrationActionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            target=target,
            status=status,
            payload=payload,
        )

    def _post_json(self, url: str, payload: dict, target: str) -> IntegrationActionRecord:
        body = json.dumps(payload).encode("utf-8")
        request_obj = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(request_obj, timeout=5):
                status = "sent"
        except Exception as exc:  # pragma: no cover - network dependent branch
            status = f"failed: {exc.__class__.__name__}"
        return IntegrationActionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            target=target,
            status=status,
            payload=payload,
        )
