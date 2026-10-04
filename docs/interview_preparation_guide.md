# ML Model Drift Early-Warning System (DEWS) — Interview & System Design Guide

## 30-Second Elevator Pitch

DEWS is a proactive ML model monitoring system that detects data drift and concept drift in production without requiring ground-truth labels. It fuses three complementary drift signals — statistical distribution tests, embedding-space distances, and prediction-confidence divergence — into a single Drift Score, then uses Holt-Winters forecasting to predict *when* the model will degrade critically. This gives engineering teams advance warning to retrain before users are affected.

## 2-Minute Pitch

Production ML models fail silently: unlike software bugs, a drifting model still returns valid predictions — they're just increasingly *wrong*. Traditional monitoring waits for labeled ground truth, which can take weeks. DEWS solves this by running alongside any classifier as a "sidecar monitor" that continuously compares incoming live data against the training reference distribution.

It detects drift across three layers: **statistical tests** (KS, PSI, JSD for feature distributions), **embedding-space distances** (Centroid, Mahalanobis, MMD on penultimate-layer representations), and **prediction-confidence shifts** (low-confidence rate, KL divergence on output probabilities). These are normalized to [0,1] and fused into a weighted Drift Score.

The key differentiator is the **Time-to-Degradation (TTD) forecasting** using Holt-Winters exponential smoothing — the system predicts how many batches until the model hits critical failure. When thresholds are breached, severity-graded alerts fire with top-K feature attribution, and the system can automatically trigger retraining via local commands, webhooks, or in-process functions.

Benchmarks against Evidently AI show DEWS achieves **0% false alerts**, **identical detection lag** (2.0 batches), and **15x faster runtime** (2.1s vs 32.2s).

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Three-layer fusion** over single-metric detection | Reduces false positives — a single metric can spike spuriously, but three independent signals agreeing is robust |
| **Label-free design** | Ground truth labels have latency (days/weeks); DEWS works on feature distributions and model confidence alone |
| **Holt-Winters forecasting** | Better than simple trend extrapolation for time-series with level changes; linear regression fallback when insufficient data |
| **Configurable weighted average fusion** | Weights `(0.45, 0.35, 0.20)` are calibratable via grid search; allows domain-specific tuning |
| **Adaptive thresholds** | Fixed thresholds cause false positives in low-variance environments; EMA + sensitivity multiplier adapts to the deployment context |
| **SQLite default with PostgreSQL option** | SQLite needs zero setup for demos; Protocol-based storage backend allows production swap without code changes |
| **Kafka + Celery for distributed mode** | Decouples ingestion throughput from processing latency; Celery provides retry semantics |

## Common Interview Questions

### Q1: Why not just use Evidently AI or WhyLabs?
DEWS combines three detection methods into a fused score, reducing false alarm rates. It also adds **predictive forecasting** (TTD), which existing tools lack. In benchmarks, DEWS matches Evidently's detection lag at 0% false alerts and 15x faster runtime.

### Q2: How does the system work without ground-truth labels?
It monitors input distributions (statistical tests), learned representations (embedding distances), and model confidence (output probability shifts). These are all computable from inputs and predictions alone — no labels needed.

### Q3: What happens when Kafka goes down?
The `/api/v1/ingest` endpoint detects Kafka unavailability and gracefully falls back to local SQLite buffering. Data is preserved for later replay.

### Q4: How do you prevent false alarms?
Three independent mechanisms: (1) multi-signal fusion — all three detectors must agree, (2) alert cooldown — suppresses duplicate alerts within N batches, (3) adaptive thresholds — the system learns normal variance during warmup and only alerts on statistically significant deviations.

### Q5: How does Time-to-Degradation (TTD) work?
The forecaster applies Holt-Winters exponential smoothing to the drift score history, projecting forward by `forecast_horizon` batches. It finds the earliest projected batch where the score exceeds `ds_crit` and reports that as TTD. If the history is too short, it falls back to linear regression.

### Q6: What is the computational complexity?
Per-batch: O(W·F) for statistical tests across F features and W window size, O(W·d) for embedding distances in d dimensions, and O(W²) for MMD kernel computation. In practice, ~2 seconds for 160-sample windows with 5 features.

### Q7: How would you scale this for 100x more data?
The Kafka→Celery→Redis architecture already supports horizontal scaling. Add more Celery workers for processing parallelism, partition the Kafka topic for ingestion parallelism, and swap SQLite for TimescaleDB for time-series query performance.

### Q8: Why a weighted average instead of a learned fusion model?
Interpretability and calibratability. A weighted average has clear semantics (each weight reflects a detector's importance), can be tuned via grid search on labeled drift episodes, and doesn't require training data for the fuser itself.

### Q9: What are the limitations?
(1) Embedding drift requires access to model internals (penultimate layer), limiting it to models you control. (2) The system doesn't detect *which type* of concept drift is occurring (gradual vs sudden). (3) Adaptive thresholds need a clean warmup period. (4) Fusion weights calibrated on synthetic data may not transfer perfectly to production distributions.

### Q10: If you had 6 more months, what would you add?
(1) Online learning for the adaptive threshold using Bayesian changepoint detection. (2) Concept drift detection via accuracy proxy models. (3) A/B testing integration to correlate drift scores with live business metrics. (4) Multi-model monitoring (track multiple deployed models simultaneously). (5) Automated root-cause analysis suggesting *which* upstream data source changed.
