from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class ApiClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 10.0

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def get(self, path: str):
        with self._client() as client:
            response = client.get(path)
            response.raise_for_status()
            return response.json()

    def post(self, path: str, json: dict | None = None):
        with self._client() as client:
            response = client.post(path, json=json)
            response.raise_for_status()
            return response.json()

    def health(self):
        return self.get("/health")

    def state(self):
        return self.get("/state")

    def reset_demo(self, payload: dict):
        return self.post("/demo/reset", json=payload)

    def step_demo(self):
        return self.post("/demo/step")

    def run_demo(self, batches: int | None = None):
        params = {"batches": batches} if batches is not None else None
        with self._client() as client:
            response = client.post("/demo/run", params=params)
            response.raise_for_status()
            return response.json()

    def scores(self):
        return self.get("/scores")

    def alerts(self):
        return self.get("/alerts")

    def forecasts(self):
        return self.get("/forecasts")

    def benchmark(self):
        return self.get("/benchmark")

    def feature_drift(self):
        return self.get("/feature-drift")

    def model_registry(self):
        """Fetch all registered model versions."""
        return self.get("/model-registry")

    def adaptive_threshold(self):
        """Fetch current adaptive threshold state."""
        return self.get("/adaptive-threshold")

    def retraining_history(self):
        """Fetch retraining event history."""
        return self.get("/retraining-history")

    def trigger_retrain(self):
        """Manually trigger a model retraining run."""
        return self.post("/retrain")
