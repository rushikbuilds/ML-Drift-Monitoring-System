"""Abstract storage protocol for DEWS.

Defines the interface that all storage backends (SQLite, PostgreSQL,
TimescaleDB) must implement.  This allows swapping backends via config
without changing any monitoring logic.

Usage:
    store = create_store("postgresql://user:pass@host/db")
    store = create_store("artifacts/dews.sqlite3")  # SQLite (default)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .alerts import AlertRecord
from .forecasting import ForecastResult


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol that all DEWS storage backends must implement."""

    def save_drift_score(
        self, batch_index: int, ds: float, s_stat: float, s_emb: float,
        s_conf: float, payload: dict, model_version: str = "unversioned",
    ) -> None: ...

    def save_forecast(self, batch_index: int, forecast: ForecastResult) -> None: ...

    def save_alert(self, alert: AlertRecord) -> None: ...

    def save_integration_actions(self, actions: list) -> None: ...

    def save_model_registration(
        self, model_version: str, reference_data_hash: str,
        metadata: dict | None = None,
    ) -> None: ...

    def save_retraining_event(self, event: dict) -> None: ...

    def load_drift_scores(self) -> list[dict]: ...

    def load_alerts(self) -> list[dict]: ...

    def load_forecasts(self) -> list[dict]: ...

    def load_integration_actions(self) -> list[dict]: ...

    def load_model_registry(self) -> list[dict]: ...

    def load_retraining_history(self) -> list[dict]: ...


def create_store(dsn: str = "artifacts/dews.sqlite3") -> StorageBackend:
    """Factory: create the appropriate storage backend from a DSN string.

    - Strings starting with ``postgresql://`` or ``postgres://`` use the
      PostgreSQL / TimescaleDB backend.
    - All other strings are treated as SQLite file paths.
    """
    if dsn.startswith(("postgresql://", "postgres://")):
        from .storage_postgres import PostgresDriftStore
        return PostgresDriftStore(dsn)
    else:
        from .storage import DriftStore
        return DriftStore(dsn)
