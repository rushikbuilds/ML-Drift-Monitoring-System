from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dews import (
    AlertEngine,
    DriftForecaster,
    DriftMonitoringSystem,
    DriftStore,
    FeatureSchema,
    IntegrationManager,
    MonitorConfig,
)


@dataclass
class RuleBasedAdapter:
    feature_columns: list[str]

    def fit_reference(self, reference_frame: pd.DataFrame) -> None:
        self.reference_mean = reference_frame[self.feature_columns].mean().to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame[self.feature_columns[0]] > 0).astype(int).to_numpy()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = frame[self.feature_columns[0]].to_numpy(dtype=float)
        probs = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probs, probs])

    def embedding_features(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.feature_columns].to_numpy(dtype=float)


def test_monitor_accepts_custom_model_adapter(tmp_path) -> None:
    reference = pd.DataFrame({"feature_0": [0.1, 0.2, 0.3, 0.4], "feature_1": [1.0, 1.1, 1.2, 1.3], "segment": [0, 1, 0, 1]})
    live_batch = pd.DataFrame({"feature_0": [0.7, 0.9], "feature_1": [1.8, 2.0], "segment": [1, 1]})

    config = MonitorConfig(window_size=4, batch_size=2, forecast_horizon=4)
    schema = FeatureSchema(continuous=["feature_0", "feature_1"], categorical=["segment"])
    adapter = RuleBasedAdapter(feature_columns=["feature_0", "feature_1", "segment"])
    store = DriftStore(str(tmp_path / "adapter.sqlite3"))
    system = DriftMonitoringSystem(
        config=config,
        schema=schema,
        model=adapter,
        store=store,
        alert_engine=AlertEngine(config),
        forecaster=DriftForecaster(horizon=4),
        integration_manager=IntegrationManager(),
    )

    system.fit_detectors(reference)
    result = system.process_batch(0, reference, live_batch, pd.Series([1, 1]))

    assert result.drift_score >= 0.0
    assert result.embedding["s_emb"] >= 0.0
    assert result.confidence["s_conf"] >= 0.0
