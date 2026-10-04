from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

import numpy as np
import pandas as pd

from .adaptive_threshold import AdaptiveThreshold
from .alerts import AlertEngine
from .config import MonitorConfig
from .data import DemoScenario, FeatureSchema, build_demo_scenario
from .detection import (
    ConfidenceDriftDetector,
    EmbeddingDriftDetector,
    StatisticalDriftDetector,
)
from .forecasting import DriftForecaster
from .integrations import IntegrationManager
from .models import ModelAdapter, build_model_adapter
from .retraining import RetrainingConfig
from .storage import DriftStore
from .storage_backend import create_store


@dataclass(slots=True)
class MonitoringResult:
    batch_index: int
    drift_score: float
    statistical: dict
    embedding: dict
    confidence: dict
    forecast: dict
    alert: dict | None
    accuracy: float | None
    model_version: str = "unversioned"
    adaptive_threshold: float | None = None

class DriftMonitoringSystem:
    def __init__(
        self,
        *,
        config: MonitorConfig,
        schema: FeatureSchema,
        model: ModelAdapter,
        store,
        alert_engine: AlertEngine,
        forecaster: DriftForecaster,
        integration_manager: IntegrationManager,
    ) -> None:
        self.config = config
        self.schema = schema
        self.model = model
        self.store = store
        self.alert_engine = alert_engine
        self.forecaster = forecaster
        self.integration_manager = integration_manager
        
        self.statistical_detector = StatisticalDriftDetector(config)
        self.embedding_detector = EmbeddingDriftDetector(config)
        self.confidence_detector = ConfidenceDriftDetector(config)
        
        self.drift_history: list[float] = []
        self.batch_results: list[MonitoringResult] = []
        
        self.adaptive_threshold: AdaptiveThreshold | None = None
        if config.adaptive_threshold_enabled:
            self.adaptive_threshold = AdaptiveThreshold(
                warmup_batches=config.adaptive_warmup_batches,
                sensitivity=config.adaptive_sensitivity,
                initial_threshold=config.ds_crit,
                ema_alpha=config.adaptive_ema_alpha,
                floor=config.adaptive_floor,
                ceiling=config.adaptive_ceiling,
            )

    def fit_detectors(self, reference_frame: pd.DataFrame) -> None:
        reference_view = reference_frame[self.schema.continuous + self.schema.categorical]
        self.model.fit_reference(reference_view)
        self.embedding_detector.fit(self.model.embedding_features(reference_view))
        self.confidence_detector.fit(self.model, reference_view)
        
        if not self.config.reference_data_hash:
            self.config.reference_data_hash = MonitorConfig.compute_reference_hash(reference_view)
        
        self.store.save_model_registration(
            model_version=self.config.model_version,
            reference_data_hash=self.config.reference_data_hash,
        )

    def process_batch(self, batch_index: int, reference_frame: pd.DataFrame, live_window: pd.DataFrame, batch_labels: pd.Series | None = None) -> MonitoringResult:
        if len(live_window) == 0:
            raise ValueError("Live window cannot be empty")
            
        reference_view = reference_frame[self.schema.continuous + self.schema.categorical]
        live_view = live_window[reference_view.columns]

        statistical = self.statistical_detector.evaluate(reference_frame, live_window, self.schema)
        embedding = self.embedding_detector.evaluate(self.model.embedding_features(reference_view), self.model.embedding_features(live_view))
        confidence = self.confidence_detector.evaluate(self.model, reference_view, live_view)

        s_stat, s_emb, s_conf = statistical.s_stat, embedding.s_emb, confidence.s_conf
        w_stat, w_emb, w_conf = self.config.normalized_weights()
        drift_score = float(np.clip(w_stat * s_stat + w_emb * s_emb + w_conf * s_conf, 0.0, 1.0))

        adaptive_threshold_value = None
        if self.adaptive_threshold is not None:
            adaptive_threshold_value = self.adaptive_threshold.update(drift_score, batch_index)
            effective_ds_crit = adaptive_threshold_value
        else:
            effective_ds_crit = self.config.ds_crit

        # Keep forecast history scoped to this monitoring-system instance. The
        # store may contain scores from an earlier run with the same batch IDs.
        self.drift_history.append(drift_score)

        stat_payload = self._as_payload(statistical)
        self.store.save_drift_score(
            batch_index=batch_index,
            ds=drift_score,
            s_stat=s_stat,
            s_emb=s_emb,
            s_conf=s_conf,
            model_version=self.config.model_version,
            payload={
                "statistical": stat_payload,
                "embedding": self._as_payload(embedding),
                "confidence": self._as_payload(confidence),
                "adaptive_threshold": adaptive_threshold_value,
                "ds_crit": effective_ds_crit,
            },
        )

        # Forecast against the same threshold used by alert evaluation. Using
        # the fixed threshold here made forecasts disagree with adaptive alerts.
        forecast = self.forecaster.forecast(self.drift_history, effective_ds_crit)
        self.store.save_forecast(batch_index, forecast)

        per_feature = stat_payload.get("per_feature", []) if isinstance(stat_payload, dict) else []
        # Suppress alerts during adaptive warmup — not enough baseline data yet
        if self.adaptive_threshold is not None and not self.adaptive_threshold.is_warmed_up:
            alert = None
        else:
            alert = self.alert_engine.evaluate(
                batch_index=batch_index,
                current_ds=drift_score,
                forecast=forecast,
                per_feature=per_feature,
                effective_threshold=effective_ds_crit,
                forecast_eligible=len(self.drift_history) >= self.config.forecast_min_history,
            )
        if alert is not None:
            self.store.save_alert(alert)
            actions = self.integration_manager.handle_alert(alert)
            self.store.save_integration_actions(actions)
            for action in actions:
                if action.target == "retraining" and action.status not in ("skipped",):
                    self.store.save_retraining_event(action.payload)

        accuracy = None
        if batch_labels is not None:
            batch_view = live_view.tail(len(batch_labels))
            predictions = self.model.predict(batch_view)
            accuracy = float(np.mean(predictions == batch_labels.to_numpy()))

        result = MonitoringResult(
            batch_index=batch_index,
            drift_score=drift_score,
            statistical=stat_payload,
            embedding=self._as_payload(embedding),
            confidence=self._as_payload(confidence),
            forecast=self._as_payload(forecast),
            alert=self._as_payload(alert) if alert else None,
            accuracy=accuracy,
            model_version=self.config.model_version,
            adaptive_threshold=adaptive_threshold_value,
        )
        self.batch_results.append(result)
        return result

    def _as_payload(self, value):
        if value is None:
            return None
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

def train_demo_model(reference_frame: pd.DataFrame, reference_labels: pd.Series) -> ModelAdapter:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )

    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    et = ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42)
    model = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("et", et)], voting="soft")
    model.fit(reference_frame, reference_labels)
    return build_model_adapter(model, list(reference_frame.columns))

def build_demo_system(config: MonitorConfig, store_path: str = "artifacts/dews.sqlite3") -> tuple[DriftMonitoringSystem, DemoScenario]:
    scenario = build_demo_scenario(reference_size=max(280, config.window_size * 2), stream_batches=18, batch_size=config.batch_size)
    feature_frame = scenario.reference_frame[scenario.feature_schema.continuous + scenario.feature_schema.categorical]
    model = train_demo_model(feature_frame, scenario.reference_labels)
    store = create_store(config.storage_dsn) if config.storage_dsn else DriftStore(store_path)
    
    system = DriftMonitoringSystem(
        config=config,
        schema=scenario.feature_schema,
        model=model,
        store=store,
        alert_engine=AlertEngine(config),
        forecaster=DriftForecaster(horizon=config.forecast_horizon),
        integration_manager=IntegrationManager(retraining_config=RetrainingConfig(enabled=config.retraining_enabled, runner=config.retraining_runner)),
    )
    system.fit_detectors(feature_frame)
    return system, scenario
