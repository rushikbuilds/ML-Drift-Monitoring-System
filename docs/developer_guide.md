# Developer Guide

This project is intentionally modular. Each part can be replaced without changing the rest of the system.

## Main modules

- `dews/data.py` - feature schema and synthetic demo scenario generation
- `dews/detection.py` - statistical, embedding, and confidence drift detectors
- `dews/forecasting.py` - drift trajectory forecasting and time-to-degradation estimation
- `dews/alerts.py` - thresholding, severity classification, and cooldown handling
- `dews/integrations.py` - webhook-style notification, ticketing, and retraining hooks
- `dews/storage.py` - local SQLite persistence
- `dews/monitoring.py` - orchestration of the full monitoring loop
- `dews/models.py` - model adapter abstractions for sklearn and PyTorch
- `streamlit_app.py` - live dashboard

## How to adapt this to a real model

1. Replace the demo classifier in `dews/monitoring.py` with your deployed model or wrapper.
2. Replace `build_demo_scenario()` with a loader that reads your reference and production batches.
3. Provide the correct feature schema so the detector knows which fields are continuous and which are categorical.
4. Configure the fusion weights and thresholds using your validation results.
5. Wrap your model with `SklearnModelAdapter`, `TorchMLPAdapter`, or a custom adapter that implements the same methods.
6. Wire `IntegrationManager` to your notificatzwant outbound actions.

## Validation workflow

Suggested checks for future changes:

```bash
./.venv/bin/python main.py --demo --batches 6
./.venv/bin/python main.py --serve
./.venv/bin/python -m streamlit run streamlit_app.py
```

The dashboard expects the API server to be running on `http://127.0.0.1:8000` by default.

## Storage

The default SQLite file lives at `artifacts/dews.sqlite3`. It is safe to delete when you want a clean demo run.

## Extensibility points

- Swap `PCAEmbeddingExtractor` for a model-specific embedding extractor.
- Swap `DriftForecaster` for Prophet or a custom regression model.
- Replace the webhook integration layer with a real Slack, Teams, Jira, or ServiceNow adapter.

## PyTorch MLP & Production Deployment

For complete, end-to-end code examples on deploying PyTorch models, setting up hidden embedding hooks with `TorchMLPAdapter`, development testing, and production deployment (Docker Compose, webhooks, API), see:

👉 **[PyTorch Deployment & Testing Guide](pytorch_deployment_and_testing_guide.md)**