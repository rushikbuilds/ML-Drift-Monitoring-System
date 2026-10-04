from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .alerts import AlertRecord
from .forecasting import ForecastResult


class DriftStore:
    def __init__(self, path: str = "artifacts/dews.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS drift_scores (
                batch_index INTEGER PRIMARY KEY,
                ds REAL NOT NULL,
                s_stat REAL NOT NULL,
                s_emb REAL NOT NULL,
                s_conf REAL NOT NULL,
                model_version TEXT NOT NULL DEFAULT 'unversioned',
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS forecasts (
                batch_index INTEGER PRIMARY KEY,
                forecast_json TEXT NOT NULL,
                model_name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                trigger_value REAL NOT NULL,
                predicted_ttd REAL,
                model_version TEXT NOT NULL DEFAULT 'unversioned',
                top_features TEXT NOT NULL DEFAULT '[]',
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS integration_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                target TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_registry (
                model_version TEXT PRIMARY KEY,
                reference_data_hash TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS retraining_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at TEXT NOT NULL,
                status TEXT NOT NULL,
                runner TEXT NOT NULL,
                model_version_old TEXT NOT NULL,
                model_version_new TEXT NOT NULL DEFAULT '',
                duration_seconds REAL NOT NULL DEFAULT 0,
                details TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS ingested_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                received_at TEXT NOT NULL
            );
            """
        )
        # Migrate existing tables: add columns if missing
        self._migrate_column("drift_scores", "model_version", "TEXT NOT NULL DEFAULT 'unversioned'")
        self._migrate_column("drift_scores", "created_at", "TEXT NOT NULL DEFAULT ''")
        self._migrate_column("alerts", "model_version", "TEXT NOT NULL DEFAULT 'unversioned'")
        self._migrate_column("alerts", "top_features", "TEXT NOT NULL DEFAULT '[]'")
        self.connection.commit()

    def _migrate_column(self, table: str, column: str, col_type: str) -> None:
        """Add a column to an existing table if it does not exist (safe migration)."""
        try:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    def save_drift_score(
        self, batch_index: int, ds: float, s_stat: float, s_emb: float, s_conf: float,
        payload: dict, model_version: str = "unversioned",
    ) -> None:
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "INSERT OR REPLACE INTO drift_scores(batch_index, ds, s_stat, s_emb, s_conf, model_version, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (batch_index, ds, s_stat, s_emb, s_conf, model_version, json.dumps(payload), created_at),
        )
        self.connection.commit()

    def save_forecast(self, batch_index: int, forecast: ForecastResult) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO forecasts(batch_index, forecast_json, model_name) VALUES (?, ?, ?)",
            (batch_index, json.dumps(asdict(forecast)), forecast.model_name),
        )
        self.connection.commit()

    def save_alert(self, alert: AlertRecord) -> None:
        top_features_json = json.dumps(alert.top_features) if alert.top_features else "[]"
        self.connection.execute(
            "INSERT INTO alerts(timestamp, batch_index, severity, message, trigger_value, predicted_ttd, model_version, top_features, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (alert.timestamp, alert.batch_index, alert.severity, alert.message,
             alert.trigger_value, alert.predicted_ttd, alert.model_version,
             top_features_json, json.dumps(asdict(alert))),
        )
        self.connection.commit()

    def save_integration_actions(self, actions: list) -> None:
        for action in actions:
            payload = asdict(action) if is_dataclass(action) else action
            self.connection.execute(
                "INSERT INTO integration_actions(timestamp, target, status, payload) VALUES (?, ?, ?, ?)",
                (payload["timestamp"], payload["target"], payload["status"], json.dumps(payload)),
            )
        self.connection.commit()

    def save_model_registration(self, model_version: str, reference_data_hash: str, metadata: dict | None = None) -> None:
        """Register a model version with its reference data hash."""
        from datetime import datetime, timezone
        self.connection.execute(
            "INSERT OR REPLACE INTO model_registry(model_version, reference_data_hash, registered_at, metadata) VALUES (?, ?, ?, ?)",
            (model_version, reference_data_hash, datetime.now(timezone.utc).isoformat(), json.dumps(metadata or {})),
        )
        self.connection.commit()

    def save_retraining_event(self, event: dict) -> None:
        """Persist a retraining attempt result."""
        from datetime import datetime, timezone
        self.connection.execute(
            "INSERT INTO retraining_history(triggered_at, status, runner, model_version_old, model_version_new, duration_seconds, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event.get("triggered_at", datetime.now(timezone.utc).isoformat()),
             event.get("status", "unknown"), event.get("runner", "unknown"),
             event.get("model_version_old", ""), event.get("model_version_new", ""),
             event.get("duration_seconds", 0), json.dumps(event.get("details", {}))),
        )
        self.connection.commit()

    def load_drift_scores(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM drift_scores ORDER BY batch_index").fetchall()
        return [dict(row) for row in rows]

    def load_alerts(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM alerts ORDER BY id").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            # Deserialize top_features from JSON string
            if isinstance(d.get("top_features"), str):
                try:
                    d["top_features"] = json.loads(d["top_features"])
                except (json.JSONDecodeError, TypeError):
                    d["top_features"] = []
            results.append(d)
        return results

    def load_forecasts(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM forecasts ORDER BY batch_index").fetchall()
        return [dict(row) for row in rows]

    def load_integration_actions(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM integration_actions ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def load_model_registry(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM model_registry ORDER BY registered_at").fetchall()
        return [dict(row) for row in rows]

    def load_retraining_history(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM retraining_history ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def clear_run_data(self) -> None:
        """Remove time-series data before starting an isolated demo run."""
        self.connection.executescript(
            """
            DELETE FROM drift_scores;
            DELETE FROM forecasts;
            DELETE FROM alerts;
            DELETE FROM integration_actions;
            DELETE FROM retraining_history;
            DELETE FROM ingested_buffer;
            """
        )
        self.connection.commit()

    # ── Ingestion buffer (Kafka fallback) ──

    def buffer_records(self, records: list[dict]) -> None:
        """Store raw ingested records when Kafka is unavailable."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for record in records:
            self.connection.execute(
                "INSERT INTO ingested_buffer(data, received_at) VALUES (?, ?)",
                (json.dumps(record), now),
            )
        self.connection.commit()

    def load_buffered_records(self, limit: int = 1000) -> list[dict]:
        """Retrieve buffered records for replay."""
        rows = self.connection.execute(
            "SELECT id, data, received_at FROM ingested_buffer ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        return [{"id": r["id"], "data": json.loads(r["data"]), "received_at": r["received_at"]} for r in rows]

    def delete_buffered_records(self, ids: list[int]) -> None:
        """Remove processed buffered records."""
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(f"DELETE FROM ingested_buffer WHERE id IN ({placeholders})", ids)
            self.connection.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
