# DEWS Architecture

The **ML Model Drift Early-Warning System (DEWS)** is composed of nine cooperating modules.  Two operational modes are supported: a **local demo** that runs entirely in-process, and a **distributed streaming** deployment orchestrated via Docker Compose.

---

## System Diagram

```mermaid
flowchart TD
    subgraph Ingestion
        A["Live Inference Data"] -->|POST /api/v1/ingest| B["FastAPI Gateway"]
        B -->|Kafka available| C["Apache Kafka"]
        B -->|Kafka unavailable| BUF["Local SQLite Buffer"]
        C --> D["Kafka Consumer"]
        D --> E["Celery Worker"]
    end

    subgraph Core["Core Drift Detection Engine"]
        F["Reference Window"] --> G["Statistical Detector\n(KS, PSI, JSD, χ²)"]
        F --> H["Embedding Detector\n(Centroid, Mahalanobis, MMD)"]
        F --> I["Confidence Detector\n(Low-Conf Rate, KL Div)"]
        G --> J["Drift Fusion\nWeighted Average → DS ∈ [0,1]"]
        H --> J
        I --> J
    end

    subgraph Forecasting & Alerts
        J --> K["Holt-Winters Forecaster\n→ Time-to-Degradation"]
        J --> L["Alert Engine\n(Critical / Warning / Info)"]
        K --> L
        L --> M["Integration Manager"]
        M --> N["Slack / Webhook"]
        M --> O["Ticket Creation"]
        M --> P["Automated Retraining"]
    end

    subgraph Persistence
        J --> Q["SQLite / PostgreSQL"]
        K --> Q
        L --> Q
    end

    subgraph Visualization
        Q --> R["FastAPI Read Endpoints\n(/scores, /alerts, /forecasts, ...)"]
        R --> S["Grafana Dashboard\n(Infinity Plugin)"]
    end

    E --> F
    BUF -.->|"Replay"| F
```

---

## Module Descriptions

| # | Module | Responsibility |
|---|--------|---------------|
| 1 | **REST Ingestion** | FastAPI gateway exposing `/api/v1/ingest`. Sends to Kafka when available; buffers locally otherwise. |
| 2 | **Event Queue** | Apache Kafka topic `dews-live-stream` for high-throughput, durable event ingestion. |
| 3 | **Async Processing** | Kafka Consumer + Celery Workers process batches out-of-band using Redis for sliding-window state. |
| 4 | **Statistical Drift** | KS-statistic, PSI, JSD (continuous features); Chi-Square, PSI, JSD (categorical features). |
| 5 | **Embedding-Space Drift** | Centroid Distance, Mahalanobis Distance, and MMD (RBF kernel) on penultimate-layer embeddings. |
| 6 | **Prediction-Confidence Drift** | Low Confidence Rate and KL Divergence on model output probabilities. |
| 7 | **Drift Fusion** | Normalizes each signal onto [0, 1] and computes a weighted average Drift Score. Default weights: `(0.45, 0.35, 0.20)`. |
| 8 | **TTD Forecasting** | Holt-Winters exponential smoothing with linear regression fallback forecasts when the Drift Score will breach the critical threshold. |
| 9 | **Alert & Integration Engine** | Severity-graded alerts (Critical/Warning/Info) with cooldown, top-K feature attribution, and hooks for retraining, webhooks, and ticket creation. |

---

## Operational Modes

### Local Demo (`python main.py --demo`)

Runs entirely in-process using synthetic data with an injected drift episode:

```
main.py --demo
  → build_demo_system()       # Train model, fit detectors on reference data
  → run_demo_evaluation()     # Loop through synthetic batches, process each
  → SQLite                    # Results persisted to artifacts/dews.sqlite3
```

No external infrastructure (Kafka, Redis, Celery) is required.  This mode is useful for development, benchmarking, and demonstration.

### API Server (`python main.py --serve`)

Starts the FastAPI backend:

```
main.py --serve
  → uvicorn backend.main:app  # Starts HTTP server on :8000
  → /api/v1/ingest            # Accepts live data (Kafka or local buffer)
  → /demo/reset + /demo/run   # Can also run the demo via API
  → /scores, /alerts, ...     # Read-only endpoints for Grafana
```

Works locally without Kafka (ingestion buffers locally).  For the full distributed pipeline, use Docker Compose.

### Full Stack (`docker compose up`)

Orchestrates all services:

```
docker compose up --build
  → Zookeeper + Kafka         # Event queue
  → Redis                     # Sliding window state + Celery broker
  → FastAPI (api)             # REST gateway
  → Celery Worker             # Async drift processing
  → Kafka Consumer            # Bridges Kafka → Celery
  → Grafana                   # Dashboard at :3000
```

---

## Storage Backends

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| **SQLite** (default) | Development, demos | `--store artifacts/dews.sqlite3` |
| **PostgreSQL** | Production | `--storage-dsn postgresql://user:pass@host/db` |
| **TimescaleDB** | Time-series optimized production | Same DSN; hypertables created automatically |

All backends implement the `StorageBackend` protocol defined in `dews/storage_backend.py`.

---

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 160 | Sliding window size (samples) |
| `batch_size` | 80 | Rows per live batch |
| `forecast_horizon` | 8 | Forecast horizon (batches) |
| `ds_crit` | 0.30 | Critical drift score threshold |
| `weights` | (0.45, 0.35, 0.20) | Fusion weights (stat/emb/conf) |
| `adaptive_threshold_enabled` | False | Enable adaptive threshold learning |
| `retraining_enabled` | False | Enable automated retraining on Critical drift |
