# ML Model Drift Early-Warning System (DEWS)

## Problem Statement
Machine learning models often suffer from performance degradation once deployed in production due to evolving data patterns — a phenomenon known as **Concept Drift** and **Data Drift**. Detecting this degradation typically happens *after* the model has already made poor predictions, impacting business outcomes. There is a need for a proactive monitoring solution that identifies early signs of drift and forecasts potential degradation before it severely affects the system.

## Solution
The **ML Model Drift Early-Warning System (DEWS)** is a comprehensive monitoring pipeline that continuously evaluates incoming live data streams against reference (training) distributions. It combines multiple drift detection strategies, fuses their results into a unified "Drift Score," forecasts future degradation, and triggers alerts proactively. This enables data science and ML engineering teams to intervene, retrain, or fallback before a critical failure occurs.

## Architecture

The system is designed as a robust pipeline comprising nine major modules:

1. **REST Ingestion**: The FastAPI backend acts as a highly scalable gateway, exposing a `/api/v1/ingest` endpoint that immediately receives live streaming data.
2. **Synchronous Processing**: The system processes the incoming batches directly in-memory, computing the heavy statistical drift signals synchronously.
3. **State Management**: Live batches and drift metrics are buffered and persisted securely using a local `SQLite` database, ensuring easy setup and portability.
4. **Multi-layer Drift Detection**: Analyzes drift across three complementary signals:
   - **Statistical Drift** — KS statistic, PSI, JSD (continuous) and Chi-Square, PSI, JSD (categorical)
   - **Embedding-Space Drift** — Centroid Distance, Mahalanobis Distance, and MMD (RBF kernel)
   - **Prediction-Confidence Drift** — Low Confidence Rate and KL Divergence
5. **Drift Fusion**: Normalizes each raw metric onto a common [0, 1] scale and fuses them using a weighted average into a single **Drift Score**.
6. **Time-to-Degradation Forecasting**: Holt-Winters exponential smoothing with linear regression fallback.
7. **Alert Generation**: Severity levels (Critical/Warning/Info), cooldown, and explainable top-K feature attribution.
8. **Dashboard**: **Grafana** dashboard that queries the FastAPI backend to visualize drift scores, component signals, forecasts, per-feature breakdowns, alerts, model registry, adaptive thresholds, and retraining history.
9. **External Actions**: Hooks for automated retraining, webhook notifications, and ticket creation.

## Distributed Production Architecture
The codebase includes a distributed production architecture using Docker Compose:
- **Event-Driven Ingestion**: Apache Kafka for highly scalable event queuing.
- **Asynchronous Processing**: Celery Workers for out-of-band statistical processing.
- **Distributed State Management**: RedisSlidingWindow for persisting reference windows across distributed workers.

When Kafka is unavailable (e.g. running locally without Docker), the `/api/v1/ingest` endpoint gracefully falls back to local buffering.

*(See [`docs/architecture.md`](docs/architecture.md) for a detailed Mermaid diagram and module descriptions.)*

## How It Works

DEWS supports two operational modes:

### Local Demo Mode (`python main.py --demo`)
1. **Initialization**: A `MonitorConfig` is created with batch sizes, sliding window sizes, and forecast horizon.
2. **Synthetic Data Generation**: A reference dataset and streaming batches with injected drift are generated in-memory.
3. **In-Process Evaluation**: Each batch is processed through the multi-layer drift detection engine, fused into a Drift Score, and forecasted.
4. **Persistence**: Results are written to SQLite (`artifacts/dews.sqlite3`).
5. **Visualization**: Run `python main.py --serve` and open Grafana (via Docker) or Swagger UI to visualize.

### Streaming Production Mode (`docker compose up`)
1. **Data Ingestion**: Live data is POSTed to `/api/v1/ingest` and queued onto Apache Kafka.
2. **Asynchronous Evaluation**: A Kafka consumer batches the data and dispatches to Celery workers, which compute drift signals against the reference window stored in Redis.
3. **Persistence**: Computed scores, forecasts, and alerts are written to the database.
4. **Visualization**: Grafana queries the FastAPI backend for real-time dashboard updates.

### API Server Mode (`python main.py --serve`)
The FastAPI server works locally without Kafka or Docker. It exposes demo lifecycle endpoints (`/demo/reset`, `/demo/step`, `/demo/run`), read-only data endpoints, and an ingestion endpoint that buffers locally when Kafka is unavailable.

### Demo Behavior
The project includes a synthetic end-to-end demo. The demo stream deliberately injects a measurable drift episode halfway through the synthetic batches.

## Getting Started

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (for Grafana dashboard)
- Requirements listed in `requirements.txt`

### Installation
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

**1. Start the Full Stack (Kafka, Redis, Celery, API, Grafana)**
The entire streaming architecture is orchestrated via Docker Compose:
```bash
docker compose up --build
```
Then open **http://localhost:3000** in your browser (login: `admin` / `admin`).

**2. Simulate Live Traffic**
You can post real streaming inference data to the API:
```bash
curl -X POST http://localhost:8000/api/v1/ingest -H "Content-Type: application/json" -d '{"data": [{"feature_0": 0.5, "feature_1": 1.2, "segment": 1}]}'
```
You will instantly receive an `enqueued` status as the data hits Kafka!

**3. Run the End-to-End Demo (Optional)**
```bash
python main.py --demo
```
Optional arguments: `--batches`, `--batch-size`, `--window-size`, `--forecast-horizon`

The Grafana dashboard includes panels for:
- Drift Score timeline with component signal decomposition (S_stat, S_emb, S_conf)
- Drift forecast with confidence intervals
- Per-feature drift breakdown (bar chart + details table)
- Alert history with severity color-coding
- Model Registry with version tracking
- Adaptive Threshold state and evolution
- Retraining History with status indicators

> **Quick local alternative**: Run `python main.py --serve` and open
> `http://localhost:8000/docs` for the interactive Swagger UI.

**4. Run Benchmarks**
```bash
python main.py --benchmark
```

**5. Calibrate Fusion Weights**

DEWS can search normalized `(S_stat, S_emb, S_conf)` weight candidates using
clean and labeled gradual-drift episodes. The calibration uses the reproducible
demo generator for regression testing and initial tuning; production weights
should be recalibrated with representative temporal data such as Electricity,
Airlines, Weather, or WILDS.

```bash
python main.py --calibrate-weights \
  --batches 18 --batch-size 40 \
  --calibration-step 0.25 \
  --calibration-output artifacts/weight_calibration.json
```

The report compares the selected weights with the current `(0.45, 0.35, 0.20)`
baseline using false-alert rate, detection delay, and missed-detection rate.

For a real temporal benchmark, provide an Electricity CSV or allow the loader
to fetch the public OpenML copy:

```bash
python main.py --electricity-benchmark --batch-size 256 \
  --output artifacts/electricity_benchmark.json
```

The Electricity run is complementary to Digits: it validates chronological
drift and accuracy degradation with a tabular classifier, while Digits remains
the PyTorch embedding/deployment test.

To tune fusion weights on Electricity, start with a bounded calibration run:

```bash
python main.py --calibrate-electricity \
  --batch-size 512 \
  --calibration-step 0.5 \
  --electricity-detector-reference 500 \
  --output artifacts/electricity_weight_calibration.json
```

The reference cap is important because the embedding MMD calculation is
quadratic in detector-reference rows. The selected weights should be treated
as a benchmark result until validated on a later time period.

## Advanced Features

### Adaptive Threshold
Instead of a fixed critical drift threshold, the adaptive engine learns what "normal" drift looks like during a warm-up phase:
```
ds_crit_adaptive = mean(normal_scores) + sensitivity * std(normal_scores)
```
This significantly reduces false positives while preserving recall. Enable with:
```bash
python main.py --demo --adaptive-threshold --adaptive-sensitivity 2.5
```

### Automated Retraining
DEWS can automatically trigger model retraining when Critical drift is detected. Three runners are supported:
- **local**: Runs a shell command (e.g., `python train_and_monitor_pytorch.py`)
- **webhook**: POSTs to an external retraining orchestrator URL
- **in_process**: Calls a Python function directly

```bash
python main.py --demo --enable-retraining --retrain-runner local
```

### MLflow Experiment Tracking
All drift metrics, alerts, and model versions can be logged to MLflow:
```bash
python main.py --demo --mlflow-uri http://localhost:5000 --mlflow-experiment my-experiment
```
Disable with `--no-mlflow`.

### PostgreSQL / TimescaleDB Backend
For production deployments, swap SQLite for PostgreSQL with optional TimescaleDB hypertables:
```bash
python main.py --demo --storage-dsn postgresql://user:pass@host:5432/dews
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 160 | Sliding window size (samples) |
| `batch_size` | 80 | Rows per live batch |
| `forecast_horizon` | 8 | Forecast horizon (batches) |
| `ds_crit` | 0.30 | Critical drift score threshold |
| `weights` | (0.45, 0.35, 0.20) | Fusion weights for stat/emb/conf signals |
| `model_version` | "unversioned" | Model version identifier |
| `adaptive_threshold_enabled` | False | Enable adaptive threshold learning |
| `adaptive_sensitivity` | 2.5 | Std multiplier for adaptive threshold |
| `adaptive_warmup_batches` | 6 | Warm-up batches before learning kicks in |
| `retraining_enabled` | False | Enable automatic retraining on Critical drift |
| `retraining_runner` | "local" | Runner type: local, webhook, in_process |
| `storage_dsn` | "" | Storage DSN (empty = SQLite, postgresql:// for PG) |
| `top_k_features` | 5 | Number of top drifting features in alert explanations |

## Directory Structure

```
dews/               Core drift detection, forecasting, alerts, retraining, storage, and Redis sliding windows
worker/             Celery worker application and Kafka consumer for asynchronous processing
backend/            FastAPI application exposing monitoring and ingestion endpoints
dashboard/          Python API client for the monitoring backend
grafana/            Grafana provisioning configs and dashboard JSON
  dashboards/       Pre-built dashboard panels
  provisioning/     Datasource and dashboard loader configs
docs/               Architecture docs, developer guide, deployment guide
notebooks/          Jupyter notebooks for exploratory analysis
tests/              Unit and integration tests
artifacts/          SQLite databases and output summaries
main.py             CLI entry point for demos, backend, and benchmarks
docker-compose.yml  Full-stack deployment with API + Grafana
```

## Built With
- **FastAPI** and **Uvicorn** — Backend API serving
- **Apache Kafka** — High-throughput event ingestion queue
- **Redis** — Distributed state management for sliding windows
- **Celery** — Asynchronous background task workers
- **Grafana** and **Infinity Plugin** — Interactive monitoring dashboard
- **Scikit-Learn**, **SciPy**, **Statsmodels** — Core drift detection and forecasting
- **SQLite** / **PostgreSQL** / **TimescaleDB** — Data persistence (migrations via **Alembic**)
- **MLflow** — Optional experiment tracking and model versioning
- **Docker Compose** — Container orchestration
