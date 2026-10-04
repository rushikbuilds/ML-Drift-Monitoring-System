# DEWS: Project Deployment & Demo Guide

This guide provides step-by-step instructions for deploying the **ML Model Drift Early-Warning System (DEWS)** and running the end-to-end demo to showcase its capabilities.

---

## 1. Prerequisites

Before starting, ensure you have the following installed on your system:
- **Python 3.10+**
- **Docker** and **Docker Compose** (Required for the full-stack deployment including Kafka, Redis, Celery, and Grafana).
- **Git** (to clone the repository, if not already done).

---

## 2. Local Setup & Installation

1. **Navigate to the project directory:**
   ```bash
   cd "ML Drift Monitoring"
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Full-Stack Deployment

DEWS is designed as a scalable, distributed system. The easiest way to spin up the entire architecture (FastAPI Backend, Kafka, Redis, Celery Workers, and Grafana) is using Docker Compose.

1. **Start the infrastructure:**
   ```bash
   docker compose up --build -d
   ```
   *Note: The `-d` flag runs the containers in detached mode. Remove it if you want to see the live logs in your terminal.*

2. **Verify Services:**
   - **Grafana Dashboard:** Open [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`).
   - **FastAPI Interactive Docs (Swagger UI):** Open [http://localhost:8000/docs](http://localhost:8000/docs).
   - **API Health Check:** Open [http://localhost:8000/health](http://localhost:8000/health).

---

## 4. Running the End-to-End Demo

The project includes a built-in synthetic demo that simulates a streaming ML pipeline. It deliberately injects a measurable drift episode halfway through the stream so you can observe the system detecting and forecasting the degradation.

### 4.1 Standard Demo
To run the standard synthetic demo:
```bash
python main.py --demo
```
**What to expect in Grafana:**
- Navigate to the **DEWS Monitoring Dashboard**.
- You will see the **Drift Score** timeline rise as the injected drift occurs.
- The **Forecast panel** will project the trajectory of the drift score, calculating the exact "Time-to-Degradation" (TTD) before the critical threshold is breached.
- The **Component Signals** (Statistical, Embedding, Confidence) will break down exactly *why* the score is rising.

### 4.2 Advanced Demo: Adaptive Thresholding
By default, the critical threshold is static (e.g., `0.30`). To showcase the **Dynamic Adaptive Threshold** (which learns normal variance during a warm-up phase to prevent false alarms):
```bash
python main.py --demo --adaptive-threshold --adaptive-sensitivity 2.5
```
**What to expect in Grafana:**
- The threshold line on the main chart will no longer be flat. It will dynamically adjust based on the EMA (Exponential Moving Average) of recent clean data.

### 4.3 Advanced Demo: Automated Retraining
To demonstrate the system's closed-loop capability (automatically triggering a remediation action when critical drift is forecasted):
```bash
python main.py --demo --enable-retraining --retrain-runner local
```
**What to expect in Grafana:**
- Once the Drift Score breaches the threshold, a Critical Alert will be generated.
- The **Retraining History** panel will update, showing a triggered retraining event (respecting cooldown periods).

### 4.4 Running the Demo via API (No CLI Required)
If the FastAPI server is already running (`python main.py --serve` or via Docker), you can trigger and step through the demo entirely via HTTP:

```bash
# Initialize the demo system
curl -X POST http://localhost:8000/demo/reset

# Process one batch at a time
curl -X POST http://localhost:8000/demo/step

# Or run all batches at once
curl -X POST http://localhost:8000/demo/run

# View results
curl http://localhost:8000/scores
curl http://localhost:8000/alerts
curl http://localhost:8000/state
```

> **Note:** When running locally without Docker, `python main.py --serve` works without Kafka. The `/api/v1/ingest` endpoint will buffer data locally instead of sending to Kafka.

---

## 5. Simulating Live Traffic

If you want to simulate live traffic hitting your API without using the built-in demo generator, you can use `curl` or Postman to send JSON payloads to the ingestion endpoint.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
     -H "Content-Type: application/json" \
     -d '{
           "data": [
             {
               "feature_0": 0.5, 
               "feature_1": 1.2, 
               "segment": 1,
               "confidence": 0.89,
               "prediction": 1
             }
           ]
         }'
```
You will receive a response confirming the payload was enqueued to Kafka for processing.

---

## 6. Benchmarking (Optional)

To demonstrate the performance and accuracy of DEWS compared to industry standards (like Evidently AI), you can run the benchmark suite:

**1. Synthetic Benchmark:**
```bash
python main.py --benchmark
```

**2. Real-World Benchmark (Australian Electricity Dataset):**
```bash
python main.py --electricity-benchmark --batch-size 256 --output artifacts/electricity_benchmark.json
```

---

## 7. Teardown

When you are finished with the demo, you can spin down the Docker containers and clean up the volumes:

```bash
docker compose down -v
```
*(The `-v` flag removes the named volumes, clearing the database and message queues for a fresh start next time).*
