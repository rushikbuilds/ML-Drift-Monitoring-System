"""PostgreSQL / TimescaleDB storage backend for DEWS.

Requires ``psycopg2`` (install with ``pip install psycopg2-binary``).

When TimescaleDB is available, the ``drift_scores`` table is automatically
converted to a hypertable for optimized time-series queries and built-in
retention policies.

Usage:
    store = PostgresDriftStore("postgresql://user:pass@localhost:5432/dews")
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Lazy import — psycopg2 is optional
_psycopg2 = None


def _ensure_psycopg2():
    global _psycopg2
    if _psycopg2 is not None:
        return _psycopg2
    try:
        import psycopg2
        import psycopg2.extras
        _psycopg2 = psycopg2
        return _psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for PostgreSQL storage. "
            "Install with: pip install psycopg2-binary"
        ) from exc


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS drift_scores (
    batch_index INTEGER PRIMARY KEY,
    ds DOUBLE PRECISION NOT NULL,
    s_stat DOUBLE PRECISION NOT NULL,
    s_emb DOUBLE PRECISION NOT NULL,
    s_conf DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'unversioned',
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forecasts (
    batch_index INTEGER PRIMARY KEY,
    forecast_json JSONB NOT NULL,
    model_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    batch_index INTEGER NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    trigger_value DOUBLE PRECISION NOT NULL,
    predicted_ttd DOUBLE PRECISION,
    model_version TEXT NOT NULL DEFAULT 'unversioned',
    top_features JSONB NOT NULL DEFAULT '[]',
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS integration_actions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    target TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,
    reference_data_hash TEXT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS retraining_history (
    id SERIAL PRIMARY KEY,
    triggered_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL,
    runner TEXT NOT NULL,
    model_version_old TEXT NOT NULL,
    model_version_new TEXT NOT NULL DEFAULT '',
    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    details JSONB NOT NULL DEFAULT '{}'
);
"""

_TIMESCALE_SQL = """
SELECT create_hypertable('drift_scores', 'batch_index',
    chunk_time_interval => 1000,
    if_not_exists => TRUE,
    migrate_data => TRUE
);
"""


class PostgresDriftStore:
    """PostgreSQL / TimescaleDB storage backend."""

    def __init__(self, dsn: str) -> None:
        psycopg2 = _ensure_psycopg2()
        self.dsn = dsn
        self.connection = psycopg2.connect(dsn)
        self.connection.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            # Try to enable TimescaleDB hypertable (no-op if not available)
            try:
                cur.execute(_TIMESCALE_SQL)
                logger.info("TimescaleDB hypertable created for drift_scores")
            except Exception:
                logger.debug("TimescaleDB not available — using plain PostgreSQL")
                self.connection.rollback() if not self.connection.autocommit else None

    def save_drift_score(
        self, batch_index: int, ds: float, s_stat: float, s_emb: float,
        s_conf: float, payload: dict, model_version: str = "unversioned",
    ) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """INSERT INTO drift_scores (batch_index, ds, s_stat, s_emb, s_conf, model_version, payload)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (batch_index) DO UPDATE SET
                       ds=EXCLUDED.ds, s_stat=EXCLUDED.s_stat, s_emb=EXCLUDED.s_emb,
                       s_conf=EXCLUDED.s_conf, model_version=EXCLUDED.model_version,
                       payload=EXCLUDED.payload""",
                (batch_index, ds, s_stat, s_emb, s_conf, model_version, json.dumps(payload)),
            )

    def save_forecast(self, batch_index, forecast) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """INSERT INTO forecasts (batch_index, forecast_json, model_name)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (batch_index) DO UPDATE SET
                       forecast_json=EXCLUDED.forecast_json, model_name=EXCLUDED.model_name""",
                (batch_index, json.dumps(asdict(forecast)), forecast.model_name),
            )

    def save_alert(self, alert) -> None:
        top_features_json = json.dumps(alert.top_features) if hasattr(alert, 'top_features') and alert.top_features else "[]"
        model_version = alert.model_version if hasattr(alert, 'model_version') else "unversioned"
        with self.connection.cursor() as cur:
            cur.execute(
                """INSERT INTO alerts (timestamp, batch_index, severity, message, trigger_value,
                   predicted_ttd, model_version, top_features, payload)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (alert.timestamp, alert.batch_index, alert.severity, alert.message,
                 alert.trigger_value, alert.predicted_ttd, model_version,
                 top_features_json, json.dumps(asdict(alert))),
            )

    def save_integration_actions(self, actions: list) -> None:
        with self.connection.cursor() as cur:
            for action in actions:
                payload = asdict(action) if is_dataclass(action) else action
                cur.execute(
                    "INSERT INTO integration_actions (timestamp, target, status, payload) VALUES (%s, %s, %s, %s)",
                    (payload["timestamp"], payload["target"], payload["status"], json.dumps(payload)),
                )

    def save_model_registration(self, model_version: str, reference_data_hash: str, metadata: dict | None = None) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """INSERT INTO model_registry (model_version, reference_data_hash, registered_at, metadata)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (model_version) DO UPDATE SET
                       reference_data_hash=EXCLUDED.reference_data_hash, metadata=EXCLUDED.metadata""",
                (model_version, reference_data_hash, datetime.now(timezone.utc).isoformat(), json.dumps(metadata or {})),
            )

    def save_retraining_event(self, event: dict) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """INSERT INTO retraining_history (triggered_at, status, runner, model_version_old,
                   model_version_new, duration_seconds, details) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (event.get("triggered_at", datetime.now(timezone.utc).isoformat()),
                 event.get("status", "unknown"), event.get("runner", "unknown"),
                 event.get("model_version_old", ""), event.get("model_version_new", ""),
                 event.get("duration_seconds", 0), json.dumps(event.get("details", {}))),
            )

    def load_drift_scores(self) -> list[dict]:
        with self.connection.cursor() as cur:
            cur.execute("SELECT batch_index, ds, s_stat, s_emb, s_conf, model_version, payload FROM drift_scores ORDER BY batch_index")
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def load_alerts(self) -> list[dict]:
        with self.connection.cursor() as cur:
            cur.execute("SELECT * FROM alerts ORDER BY id")
            columns = [desc[0] for desc in cur.description]
            results = []
            for row in cur.fetchall():
                d = dict(zip(columns, row))
                if isinstance(d.get("top_features"), str):
                    try:
                        d["top_features"] = json.loads(d["top_features"])
                    except (json.JSONDecodeError, TypeError):
                        d["top_features"] = []
                results.append(d)
            return results

    def load_forecasts(self) -> list[dict]:
        with self.connection.cursor() as cur:
            cur.execute("SELECT * FROM forecasts ORDER BY batch_index")
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def load_integration_actions(self) -> list[dict]:
        with self.connection.cursor() as cur:
            cur.execute("SELECT * FROM integration_actions ORDER BY id")
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def load_model_registry(self) -> list[dict]:
        with self.connection.cursor() as cur:
            cur.execute("SELECT * FROM model_registry ORDER BY registered_at")
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def load_retraining_history(self) -> list[dict]:
        with self.connection.cursor() as cur:
            cur.execute("SELECT * FROM retraining_history ORDER BY id")
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
