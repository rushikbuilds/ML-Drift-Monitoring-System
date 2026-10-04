from __future__ import annotations

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


# ── Read-only endpoints ─────────────────────────────────────────────────

def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "kafka" in body
    assert "database" in body


def test_scores_endpoint() -> None:
    response = client.get("/scores")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_alerts_endpoint() -> None:
    response = client.get("/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_forecasts_endpoint() -> None:
    response = client.get("/forecasts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_feature_drift_endpoint() -> None:
    response = client.get("/feature-drift")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_model_registry_endpoint() -> None:
    response = client.get("/model-registry")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_state_endpoint() -> None:
    response = client.get("/state")
    assert response.status_code == 200
    body = response.json()
    assert "processed" in body
    assert "latest_drift_score" in body


def test_adaptive_threshold_endpoint() -> None:
    response = client.get("/adaptive-threshold")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_retraining_history_endpoint() -> None:
    response = client.get("/retraining-history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ── Ingestion ───────────────────────────────────────────────────────────

@patch("backend.main.producer")
def test_ingest_with_kafka(mock_producer: MagicMock) -> None:
    """When Kafka is available, records should be sent via producer."""
    payload = {"data": [{"feature_0": 0.1, "feature_1": 1.0, "segment": 0}]}
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "enqueued"
    assert body["records"] == 1
    assert body["backend"] == "kafka"
    mock_producer.send.assert_called_once()
    mock_producer.flush.assert_called_once()


@patch("backend.main.producer", None)
def test_ingest_without_kafka() -> None:
    """When Kafka is unavailable, records should be buffered locally."""
    payload = {"data": [{"feature_0": 0.2, "feature_1": 0.8, "segment": 1}]}
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "buffered"
    assert body["records"] == 1
    assert body["backend"] == "local"


# ── Demo lifecycle ──────────────────────────────────────────────────────

def test_demo_reset_endpoint() -> None:
    response = client.post("/demo/reset")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["processed"] == 0


def test_demo_step_endpoint() -> None:
    # Ensure demo is initialized
    client.post("/demo/reset")
    response = client.post("/demo/step")
    assert response.status_code == 200
    body = response.json()
    assert body["processed"] == 1
    assert body["result"] is not None
    assert "drift_score" in body["result"]


def test_benchmark_endpoint() -> None:
    response = client.get("/benchmark")
    assert response.status_code == 200
    body = response.json()
    assert "report" in body
