# PyTorch Model Deployment, Integration, and Testing Guide

This guide provides step-by-step instructions for integrating a custom **PyTorch model** into the ML Model Drift Early-Warning System, running **development testing**, and setting up **production deployment**.

---

## Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Dev Testing Guide (Local Verification)](#2-dev-testing-guide-local-verification)
   - [Run Unit Tests](#step-1-run-unit-tests)
   - [Run CLI Synthetic Demo](#step-2-run-cli-synthetic-demo)
   - [Run Benchmark Comparison](#step-3-run-benchmark-comparison)
   - [Test Backend API & Grafana Dashboard](#step-4-test-backend-api--grafana-dashboard)
3. [PyTorch Model Integration Guide](#3-pytorch-model-integration-guide)
   - [Step 1: Load PyTorch Model](#step-1-load-pytorch-model)
   - [Step 2: Define Penultimate Embedding Extractor](#step-2-define-penultimate-embedding-extractor)
   - [Step 3: Wrap with TorchMLPAdapter](#step-3-wrap-with-torchmlpadapter)
   - [Step 4: Initialize Monitoring Engine](#step-4-initialize-monitoring-engine)
   - [Step 5: Feed Live Production Batches](#step-5-feed-live-production-batches)
4. [Production Deployment Guide](#4-production-deployment-guide)
   - [Option A: Containerized Deployment (Docker Compose)](#option-a-containerized-deployment-docker-compose)
   - [Option B: Standalone API & Background Monitoring Worker](#option-b-standalone-api--background-monitoring-worker)
   - [Option C: Webhooks for Notifications, Tickets, and Retraining](#option-c-webhooks-for-notifications-tickets-and-retraining)

---

## 1. Overview & Architecture

The system monitors deployed tabular machine learning models in a **label-free** manner across three detection layers:
1. **Statistical Layer ($S_{stat}$)**: Univariate KS-tests, PSI, Chi-Square, aggregated via JSD.
2. **Embedding Layer ($S_{emb}$)**: Multivariate shift in hidden feature representations (Mahalanobis distance + MMD).
3. **Confidence Layer ($S_{conf}$)**: Model prediction uncertainty distribution (KL divergence + low-confidence rate).

These signals are fused into a single **Drift Score ($DS$)**, which is forecasted using exponential smoothing (Holt-Winters) to estimate **Time-to-Degradation ($TTD$)**.

---

## 2. Dev Testing Guide (Local Verification)

Follow these steps to test and verify the monitoring framework on your local machine.

### Prerequisites

```bash
# Navigate to project directory
cd "/home/rushikesh/ML Drift Monitoring"

# Activate Python virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 1: Run Unit Tests

Execute the automated test suite:

```bash
python -m pytest tests/ -v
```

*Verifies detection layers, forecasting algorithms, API endpoints, benchmarks, and model adapters.*

### Step 2: Run CLI Synthetic Demo

Run an end-to-end synthetic monitoring pipeline:

```bash
python main.py --demo --batches 18 --batch-size 80 --output artifacts/demo_summary.json
```

*Simulates 18 production batches with drift injected at batch 9. Check `artifacts/demo_summary.json` for drift scores, alerts, and detection lag metrics.*

### Step 3: Run Benchmark Comparison

Compare the fused Drift Score against baseline detectors (standalone KS-test and PSI):

```bash
python main.py --benchmark
```

### Step 4: Test Backend API & Grafana Dashboard

**Terminal 1 — Launch FastAPI backend:**
```bash
python main.py --serve --host 0.0.0.0 --port 8000
```

**Terminal 2 — Launch Grafana dashboard:**
```bash
docker compose up -d grafana
```

Open `http://localhost:3000` in your web browser (default credentials: `admin` / `admin`):
- View real-time Drift Score timeline with component decomposition ($S_{stat}$, $S_{emb}$, $S_{conf}$).
- Monitor forecast trajectories and estimated Time-to-Degradation ($TTD$).
- Inspect alert history, per-feature drift attribution, adaptive thresholds, and model registry.

---

## 3. PyTorch Model Integration Guide

### Step 1: Load PyTorch Model

Suppose you have a PyTorch classification model saved as `model.pth`:

```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# 1. Define model architecture
class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 16)  # Penultimate hidden layer (16-dim)
        self.output = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.output(h)

# 2. Instantiate and load weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TabularMLP(input_dim=6, hidden_dim=32, num_classes=2).to(device)
model.load_state_dict(torch.load("path/to/model.pth", map_location=device))
model.eval()
```

### Step 2: Define Penultimate Embedding Extractor

To compute **Embedding-Space Drift ($S_{emb}$)**, extract activations from the penultimate layer (`fc2`):

```python
def extract_penultimate_embeddings(frame: pd.DataFrame) -> np.ndarray:
    """Extract hidden features from the penultimate layer of the PyTorch MLP."""
    feature_cols = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
    tensor_input = torch.tensor(frame[feature_cols].values, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        h1 = model.relu(model.fc1(tensor_input))
        embeddings = model.relu(model.fc2(h1))  # 16-dim representation
    
    return embeddings.cpu().numpy()
```

### Step 3: Wrap with TorchMLPAdapter

Use the built-in `TorchMLPAdapter` to bridge your PyTorch model with the DEWS monitoring layer:

```python
from dews import TorchMLPAdapter

feature_columns = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

adapter = TorchMLPAdapter(
    model=model,
    feature_columns=feature_columns,
    device=device,
    embedding_fn=extract_penultimate_embeddings,  # Pass embedding hook
)
```

### Step 4: Initialize Monitoring Engine

Set up `DriftMonitoringSystem` with reference dataset (e.g. training/validation data):

```python
from dews import (
    MonitorConfig, FeatureSchema, DriftMonitoringSystem,
    DriftStore, AlertEngine, DriftForecaster, IntegrationManager
)

# Configuration
config = MonitorConfig(
    window_size=160,              # Sliding live window size
    batch_size=80,                # Batch size
    forecast_horizon=8,           # Forecast horizon in batches
    ds_crit=0.30,                 # Critical drift threshold
    low_confidence_threshold=0.55,# Confidence cutoff
    weights=(0.45, 0.35, 0.20),   # (S_stat, S_emb, S_conf) weights
)

# Feature schema
schema = FeatureSchema(
    continuous=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
    categorical=["segment"],
)

# Persistence store & components
store = DriftStore("artifacts/pytorch_monitoring.sqlite3")
alert_engine = AlertEngine(config)
forecaster = DriftForecaster(horizon=config.forecast_horizon)
integrations = IntegrationManager(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    ticket_url="https://your-jira-or-service-desk-webhook",
    retraining_hook_url="https://your-airflow-or-kfp-retraining-trigger"
)

# Initialize system
system = DriftMonitoringSystem(
    config=config,
    schema=schema,
    model=adapter,
    store=store,
    alert_engine=alert_engine,
    forecaster=forecaster,
    integration_manager=integrations,
)

# Load reference dataset (training baseline)
reference_df = pd.read_csv("data/training_reference.csv")
reference_labels = reference_df["target"]

system.initialize(reference_df, reference_labels)
print("✅ DEWS monitoring system initialized for PyTorch model.")
```

### Step 5: Feed Live Production Batches

Process production streaming batches as inference occurs:

```python
def process_live_production_batch(batch_index: int, live_batch_df: pd.DataFrame):
    """Call this function whenever a new production batch arrives."""
    result = system.process_batch(
        batch_index=batch_index,
        batch_frame=live_batch_df,
        batch_labels=None,  # Label-free monitoring!
    )
    
    print(f"--- Batch {batch_index} Monitoring Result ---")
    print(f"Drift Score (DS): {result.drift_score:.4f}")
    print(f"  - S_stat (Statistical): {result.statistical['s_stat']:.4f}")
    print(f"  - S_emb (Embedding):   {result.embedding['s_emb']:.4f}")
    print(f"  - S_conf (Confidence): {result.confidence['s_conf']:.4f}")
    print(f"Estimated TTD: {result.forecast['ttd_batches']} batches")
    
    if result.alert:
        print(f"🚨 ALERT [{result.alert['severity']}]: {result.alert['message']}")

# Example loop over streaming batches
for batch_idx, batch_data in enumerate(production_stream_generator()):
    process_live_production_batch(batch_idx, batch_data)
```

---

## 4. Production Deployment Guide

### Option A: Containerized Deployment (Docker Compose)

The repository provides a multi-container setup via `Dockerfile` and `docker-compose.yml`.

1. **Configure Environment Variables (`.env`)**:
   ```ini
   API_BASE_URL=http://127.0.0.1:8000
   DRIFT_DB_PATH=artifacts/backend.sqlite3
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   TICKET_WEBHOOK_URL=https://jira.company.com/webhook/...
   RETRAINING_WEBHOOK_URL=https://airflow.company.com/api/v1/dags/retrain/dagRuns
   ```

2. **Build and Run**:
   ```bash
   docker compose up --build -d
   ```

3. **Verify Deployment**:
   - Backend API: `http://localhost:8000/health`
   - Grafana Dashboard: `http://localhost:3000`

### Option B: Standalone API & Background Monitoring Worker

In production MLOps environments, run FastAPI with Uvicorn under Systemd or Gunicorn:

```bash
# Launch FastAPI backend for monitoring & telemetry endpoints
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

To run the Grafana dashboard pointing to an external or dedicated monitoring backend, configure the backend API URL in `grafana/provisioning/datasources/datasources.yml`:

```bash
# Launch Grafana container
docker compose up -d grafana
```

### Option C: Webhooks for Notifications, Tickets, and Retraining

When drift is forecasted or breached, `IntegrationManager` emits structured JSON HTTP POST requests to configured URLs:

- **Slack / Teams Webhook**: Sends `Info`, `Warning`, or `Critical` alerts to engineering channels.
- **Jira / ServiceNow Ticket Webhook**: Automatically opens a ticket on `Critical` alerts.
- **Airflow / Kubeflow Pipeline Trigger**: Automatically triggers a model retraining workflow when a `Critical` drift alert is triggered.

---

## Summary File Reference

- **PyTorch & Sklearn Integration Examples**: `docs/pytorch_deployment_and_testing_guide.md` (this file)
- **General Architecture**: [`docs/architecture.md`](architecture.md)
- **Developer Guide**: [`docs/developer_guide.md`](developer_guide.md)
- **Main Usage & CLI Notes**: [`docs/usage.md`](usage.md)
- **Full Project README**: [`README.md`](../README.md)

