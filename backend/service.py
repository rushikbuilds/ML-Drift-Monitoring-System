"""Monitoring service layer for the FastAPI backend.

Manages the demo lifecycle (reset / step / run), exposes read-only views
of drift scores, alerts, forecasts, and feature drift, and provides
methods for benchmarking and manual retraining triggers.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

import pandas as pd

from dews import MonitorConfig, build_demo_system, run_demo_evaluation
from dews.benchmarks import run_benchmark_suite
from dews.storage import DriftStore

logger = logging.getLogger(__name__)


class MonitoringService:
    def __init__(self, store_path: str | None = None) -> None:
        self._store_path = store_path or os.getenv("DRIFT_DB_PATH", "artifacts/dews.sqlite3")
        self.store = DriftStore(self._store_path)

        # Demo state (lazy-initialized via reset_demo)
        self._system = None
        self._scenario = None
        self._batch_index: int = 0
        self._live_batches: list[pd.DataFrame] = []

    # ── Demo lifecycle ──────────────────────────────────────────────────

    def reset_demo(self, config_overrides: dict | None = None) -> dict:
        """Initialize (or re-initialize) the demo system and clear old data."""
        config = MonitorConfig(**(config_overrides or {}))
        self.store.clear_run_data()
        self._system, self._scenario = build_demo_system(
            config=config, store_path=self._store_path,
        )
        self._batch_index = 0
        self._live_batches = []
        logger.info("Demo reset — %d batches available", len(self._scenario.stream_batches))
        return self.status()

    def _ensure_demo(self) -> None:
        if self._system is None:
            self.reset_demo()

    def step_demo(self) -> dict | None:
        """Process exactly one demo batch and return the result."""
        self._ensure_demo()
        total = len(self._scenario.stream_batches)
        if self._batch_index >= total:
            return None

        schema = self._scenario.feature_schema
        reference_frame = self._scenario.reference_frame[schema.continuous + schema.categorical]

        self._live_batches.append(self._scenario.stream_batches[self._batch_index])
        live_window = (
            pd.concat(self._live_batches, ignore_index=True)
            .tail(self._system.config.window_size)
            .reset_index(drop=True)
        )

        result = self._system.process_batch(
            batch_index=self._batch_index,
            reference_frame=reference_frame,
            live_window=live_window,
            batch_labels=self._scenario.stream_labels[self._batch_index],
        )
        self._batch_index += 1

        return {
            "processed": self._batch_index,
            "total_batches": total,
            "result": {
                "batch_index": result.batch_index,
                "drift_score": result.drift_score,
                "alert": result.alert,
                "accuracy": result.accuracy,
            },
        }

    def run_demo(self, batches: int | None = None) -> dict:
        """Run the full demo evaluation (or up to *batches* batches)."""
        self._ensure_demo()
        summary = run_demo_evaluation(
            system=self._system,
            scenario=self._scenario,
            max_batches=batches,
        )
        total = len(self._scenario.stream_batches)
        self._batch_index = min(batches or total, total)
        return summary

    # ── Read-only data views ────────────────────────────────────────────

    def scores(self) -> list[dict]:
        import time
        scores = self.store.load_drift_scores()
        now_ms = int(time.time() * 1000)
        total = len(scores)
        for i, s in enumerate(scores):
            # Use real timestamp when available, fall back to synthetic spacing
            created_at = s.get("created_at", "")
            if created_at:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(created_at)
                    s["timestamp"] = int(dt.timestamp() * 1000)
                except ValueError:
                    s["timestamp"] = now_ms - ((total - 1 - i) * 60000)
            else:
                s["timestamp"] = now_ms - ((total - 1 - i) * 60000)
        return scores

    def alerts(self) -> list[dict]:
        return self.store.load_alerts()

    def forecasts(self) -> list[dict]:
        return self.store.load_forecasts()

    def feature_drift(self) -> list[dict]:
        scores = self.store.load_drift_scores()
        if not scores:
            return []
        latest_payload = scores[-1].get("payload", {})
        if isinstance(latest_payload, str):
            latest_payload = json.loads(latest_payload)
        return latest_payload.get("statistical", {}).get("per_feature", [])

    def model_registry(self) -> list[dict]:
        return self.store.load_model_registry()

    def retraining_history(self) -> list[dict]:
        return self.store.load_retraining_history()

    def state(self) -> dict:
        """Aggregate summary consumed by the Grafana stat/gauge panels."""
        scores = self.store.load_drift_scores()
        alerts = self.store.load_alerts()
        forecasts = self.store.load_forecasts()

        latest_ds = scores[-1]["ds"] if scores else 0
        model_version = scores[-1].get("model_version", "unknown") if scores else "unknown"

        # Latest TTD from the most recent forecast
        latest_ttd: float | None = None
        if forecasts:
            last_fc = forecasts[-1]
            fc_json = last_fc.get("forecast_json", "{}")
            if isinstance(fc_json, str):
                fc_json = json.loads(fc_json)
            latest_ttd = fc_json.get("ttd_batches")

        # Determine effective threshold from stored config
        ds_crit = 0.3  # default
        if scores:
            latest_payload = scores[-1].get("payload", "{}")
            if isinstance(latest_payload, str):
                latest_payload = json.loads(latest_payload)
            ds_crit = latest_payload.get("ds_crit", 0.3)

        # Drift-start batch: first batch where ds exceeded the effective threshold
        drift_start: int | None = None
        for s in scores:
            if s["ds"] >= ds_crit:
                drift_start = s["batch_index"]
                break

        return {
            "processed": len(scores),
            "total_batches": len(scores),
            "latest_drift_score": round(latest_ds, 4),
            "latest_ttd": latest_ttd,
            "model_version": model_version,
            "drift_start_batch": drift_start,
            "total_alerts": len(alerts),
        }

    def adaptive_threshold(self) -> list[dict]:
        """Return per-batch adaptive threshold data from stored processing results."""
        import time
        scores = self.store.load_drift_scores()
        result = []
        now_ms = int(time.time() * 1000)
        total = len(scores)
        for i, s in enumerate(scores):
            payload = s.get("payload", "{}")
            if isinstance(payload, str):
                payload = json.loads(payload)

            # Use real adaptive threshold if stored, else fall back to static default
            at_value = payload.get("adaptive_threshold")
            threshold = at_value if at_value is not None else payload.get("ds_crit", 0.3)

            # Use real timestamp when available
            created_at = s.get("created_at", "")
            if created_at:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(created_at)
                    ts = int(dt.timestamp() * 1000)
                except ValueError:
                    ts = now_ms - ((total - 1 - i) * 60000)
            else:
                ts = now_ms - ((total - 1 - i) * 60000)

            result.append({
                "timestamp": ts,
                "batch_index": s["batch_index"],
                "ds": round(s["ds"], 4),
                "adaptive_threshold": round(threshold, 4),
            })
        return result

    def status(self) -> dict:
        """Demo-aware status including system readiness."""
        base = self.state()
        total = len(self._scenario.stream_batches) if self._scenario else base["total_batches"]
        return {
            "ready": self._system is not None,
            "processed": self._batch_index,
            "total_batches": total,
            "drift_start_batch": self._scenario.drift_start_batch if self._scenario else base.get("drift_start_batch"),
            "latest_drift_score": base["latest_drift_score"],
            "latest_ttd": base["latest_ttd"],
            "model_version": base["model_version"],
            "total_alerts": base["total_alerts"],
        }

    # ── Benchmark & retrain ─────────────────────────────────────────────

    def run_benchmark(self) -> dict:
        """Run the internal benchmark suite."""
        config = MonitorConfig()
        return run_benchmark_suite(config)

    def trigger_retrain(self) -> dict:
        """Manually trigger a retraining run via the integration manager."""
        if self._system and self._system.integration_manager:
            result = self._system.integration_manager.trigger_retrain_manually()
            return asdict(result)
        return {"status": "no_system", "message": "Demo system not initialized. Call POST /demo/reset first."}

    # ── Kafka-fallback buffer ───────────────────────────────────────────

    def buffer_records(self, records: list[dict]) -> None:
        """Buffer incoming records when Kafka is unavailable."""
        self.store.buffer_records(records)


service = MonitoringService()
