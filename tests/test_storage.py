"""Tests for DriftStore SQLite round-trip persistence."""
from __future__ import annotations

from pathlib import Path

from dews.alerts import AlertRecord
from dews.forecasting import ForecastResult
from dews.storage import DriftStore


def test_drift_score_round_trip(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    store.save_drift_score(
        batch_index=0, ds=0.25, s_stat=0.20, s_emb=0.30, s_conf=0.10,
        model_version="v1.0",
        payload={"statistical": {"s_stat": 0.20}},
    )
    scores = store.load_drift_scores()
    assert len(scores) == 1
    assert scores[0]["ds"] == 0.25
    assert scores[0]["model_version"] == "v1.0"


def test_forecast_round_trip(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    forecast = ForecastResult(
        forecast=[0.3, 0.35, 0.4],
        lower_bound=[0.2, 0.25, 0.3],
        upper_bound=[0.4, 0.45, 0.5],
        ttd_batches=2.0,
        ttd_lower_bound=1.5,
        ttd_upper_bound=3.0,
        model_name="holtwinters",
    )
    store.save_forecast(batch_index=0, forecast=forecast)
    forecasts = store.load_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0]["model_name"] == "holtwinters"


def test_alert_round_trip(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    alert = AlertRecord(
        timestamp="2024-01-01T00:00:00Z",
        batch_index=5,
        severity="Critical",
        message="Drift threshold exceeded",
        trigger_value=0.45,
        predicted_ttd=1.5,
        model_version="v1.0",
        top_features=[{"feature": "f0", "kind": "continuous", "score": 0.8}],
    )
    store.save_alert(alert)
    alerts = store.load_alerts()
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "Critical"
    assert isinstance(alerts[0]["top_features"], list)
    assert alerts[0]["top_features"][0]["feature"] == "f0"


def test_model_registry_round_trip(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    store.save_model_registration("v2.0", "abc123def456", metadata={"source": "test"})
    registry = store.load_model_registry()
    assert len(registry) == 1
    assert registry[0]["model_version"] == "v2.0"
    assert registry[0]["reference_data_hash"] == "abc123def456"


def test_retraining_history_round_trip(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    store.save_retraining_event({
        "triggered_at": "2024-01-01T00:00:00Z",
        "status": "success",
        "runner": "local",
        "model_version_old": "v1.0",
        "model_version_new": "v2.0",
        "duration_seconds": 12.5,
        "details": {"epochs": 50},
    })
    history = store.load_retraining_history()
    assert len(history) == 1
    assert history[0]["status"] == "success"
    assert history[0]["runner"] == "local"


def test_integration_actions_round_trip(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    from dataclasses import dataclass
    @dataclass
    class FakeAction:
        timestamp: str
        target: str
        status: str
        payload: dict

    action = FakeAction("2024-01-01T00:00:00Z", "notification", "skipped", {"key": "value"})
    store.save_integration_actions([action])
    actions = store.load_integration_actions()
    assert len(actions) == 1
    assert actions[0]["target"] == "notification"


def test_multiple_drift_scores(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    for i in range(5):
        store.save_drift_score(
            batch_index=i, ds=0.1 * (i + 1), s_stat=0.1, s_emb=0.1, s_conf=0.1,
            model_version="v1", payload={},
        )
    scores = store.load_drift_scores()
    assert len(scores) == 5
    assert scores[0]["batch_index"] == 0
    assert scores[4]["batch_index"] == 4
    assert scores[4]["ds"] == 0.5


def test_clear_run_data_removes_time_series_rows(tmp_path: Path) -> None:
    store = DriftStore(str(tmp_path / "test.sqlite3"))
    store.save_drift_score(0, 0.2, 0.1, 0.1, 0.1, {})
    store.save_forecast(0, ForecastResult([], [], [], None, None, None, "empty"))
    store.save_alert(AlertRecord(
        timestamp="2024-01-01T00:00:00Z",
        batch_index=0,
        severity="Info",
        message="test",
        trigger_value=0.2,
        predicted_ttd=None,
    ))

    store.clear_run_data()

    assert store.load_drift_scores() == []
    assert store.load_forecasts() == []
    assert store.load_alerts() == []
