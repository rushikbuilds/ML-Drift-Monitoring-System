"""FastAPI backend for the ML Drift Monitoring system.

Endpoints:
    Ingestion:
        POST /api/v1/ingest          Ingest live data (Kafka or local buffer fallback)

    Dashboard / Read-only:
        GET  /health                 Health check
        GET  /scores                 Drift score timeline
        GET  /alerts                 Alert history
        GET  /forecasts              Forecast history
        GET  /feature-drift          Per-feature drift breakdown
        GET  /model-registry         Registered model versions
        GET  /state                  Aggregate state for Grafana stat panels
        GET  /adaptive-threshold     Adaptive threshold evolution
        GET  /retraining-history     Retraining event log

    Demo lifecycle:
        POST /demo/reset             Initialize / reset the demo system
        POST /demo/step              Process one demo batch
        POST /demo/run               Run the full demo evaluation

    Actions:
        GET  /benchmark              Run internal benchmark suite
        POST /retrain                Manually trigger retraining

    Observability:
        /metrics                     Prometheus metrics (mounted ASGI sub-app)
"""
import json
import logging
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from .schemas import DemoResetRequest, StepResponse, StatusResponse, BenchmarkResponse
from .service import service

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Drift Monitoring API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics ──────────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Kafka producer (best-effort) ────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
producer = None
try:
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    logger.info("Kafka producer connected to %s", KAFKA_BROKER)
except Exception as exc:
    logger.warning("Kafka producer unavailable (%s) — ingestion will buffer locally.", exc)


# ── Pydantic models ────────────────────────────────────────────────────

class IngestPayload(BaseModel):
    data: list[dict]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        {"feature_0": 0.5, "feature_1": 1.2, "segment": 1, "confidence": 0.89, "prediction": 1}
                    ]
                }
            ]
        }
    }


# ── Health check ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "kafka": producer is not None,
        "database": service.store.connection is not None,
    }


# ── Ingestion ───────────────────────────────────────────────────────────

@app.post("/api/v1/ingest")
def ingest_data(payload: IngestPayload):
    """Ingest live data. Uses Kafka when available, otherwise buffers locally."""
    if producer:
        for row in payload.data:
            producer.send("dews-live-stream", row)
        producer.flush()
        return {"status": "enqueued", "records": len(payload.data), "backend": "kafka"}
    else:
        service.buffer_records(payload.data)
        return {"status": "buffered", "records": len(payload.data), "backend": "local"}


# ── Dashboard / read-only views ─────────────────────────────────────────

@app.get("/scores", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"batch_index": 1, "ds": 0.15, "timestamp": 1690000000000, "model_version": "v1.0"}]
            }
        }
    }
})
def get_scores() -> list[dict]:
    return service.scores()

@app.get("/alerts", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"batch_index": 12, "severity": "CRITICAL", "message": "Drift score 0.35 exceeded threshold 0.30", "timestamp": "2023-10-01T12:00:00Z"}]
            }
        }
    }
})
def get_alerts() -> list[dict]:
    return service.alerts()

@app.get("/forecasts", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"batch_index": 12, "forecast_json": {"ttd_batches": 5, "forecast": [0.35, 0.38, 0.40]}}]
            }
        }
    }
})
def get_forecasts() -> list[dict]:
    return service.forecasts()

@app.get("/feature-drift", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"feature_name": "feature_0", "drift_score": 0.45}]
            }
        }
    }
})
def feature_drift() -> list[dict]:
    return service.feature_drift()

@app.get("/model-registry", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"version": "v1.0", "deployed_at": "2023-10-01T10:00:00Z"}]
            }
        }
    }
})
def model_registry() -> list[dict]:
    return service.model_registry()

@app.get("/state", responses={
    200: {
        "content": {
            "application/json": {
                "example": {"processed": 120, "total_batches": 120, "latest_drift_score": 0.25, "latest_ttd": 12.5, "model_version": "v1.0", "drift_start_batch": 105, "total_alerts": 2}
            }
        }
    }
})
def get_state() -> dict:
    return service.state()

@app.get("/adaptive-threshold", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"timestamp": 1690000000000, "batch_index": 5, "ds": 0.12, "adaptive_threshold": 0.28}]
            }
        }
    }
})
def get_adaptive_threshold() -> list[dict]:
    return service.adaptive_threshold()

@app.get("/retraining-history", responses={
    200: {
        "content": {
            "application/json": {
                "example": [{"batch_index": 12, "status": "SUCCESS", "triggered_at": "2023-10-01T12:05:00Z"}]
            }
        }
    }
})
def get_retraining_history() -> list[dict]:
    return service.retraining_history()


# ── Demo lifecycle ──────────────────────────────────────────────────────

@app.post("/demo/reset")
def demo_reset(request: DemoResetRequest | None = None):
    """Initialize (or re-initialize) the demo system and clear old data."""
    overrides = request.config.model_dump() if request and request.config else None
    return service.reset_demo(config_overrides=overrides)


@app.post("/demo/step")
def demo_step():
    """Process exactly one demo batch. Returns null result when all batches are consumed."""
    result = service.step_demo()
    if result is None:
        return {"processed": service._batch_index, "total_batches": 0, "result": None, "message": "All batches consumed."}
    return result


@app.post("/demo/run")
def demo_run(batches: int | None = Query(default=None, description="Max batches to process")):
    """Run the full demo evaluation (or up to *batches* batches)."""
    return service.run_demo(batches=batches)


# ── Benchmark & retraining ──────────────────────────────────────────────

@app.get("/benchmark")
def run_benchmark():
    """Run the internal benchmark suite and return results."""
    report = service.run_benchmark()
    return {"report": report}


@app.post("/retrain")
def trigger_retrain():
    """Manually trigger a model retraining run."""
    return service.trigger_retrain()
