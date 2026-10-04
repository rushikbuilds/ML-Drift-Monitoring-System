"""ML Model Drift Early-Warning System package."""

from .adaptive_threshold import AdaptiveThreshold, ThresholdSnapshot
from .alerts import AlertEngine, AlertRecord, FeatureDriftSummary
from .config import MonitorConfig
from .data import DemoScenario, FeatureSchema, build_demo_scenario
from .detection import (
    ConfidenceDriftDetector,
    ConfidenceDriftResult,
    EmbeddingDriftDetector,
    EmbeddingDriftResult,
    StatisticalDriftDetector,
    StatisticalDriftResult,
)
from .evaluation import run_demo_evaluation
from .forecasting import DriftForecaster, ForecastResult
from .integrations import IntegrationManager
from .mlflow_tracking import MLflowTracker
from .models import (
    ModelAdapter,
    SklearnModelAdapter,
    TorchMLPAdapter,
    build_model_adapter,
)
from .monitoring import DriftMonitoringSystem, build_demo_system
from .retraining import RetrainingConfig, RetrainingOrchestrator, RetrainingResult
from .storage import DriftStore
from .storage_backend import StorageBackend, create_store

__all__ = [
    # Adaptive threshold
    "AdaptiveThreshold",
    # Alerts
    "AlertEngine",
    "AlertRecord",
    "ConfidenceDriftDetector",
    "ConfidenceDriftResult",
    # Data
    "DemoScenario",
    # Forecasting
    "DriftForecaster",
    # Core monitoring
    "DriftMonitoringSystem",
    # Storage
    "DriftStore",
    "EmbeddingDriftDetector",
    "EmbeddingDriftResult",
    "FeatureDriftSummary",
    "FeatureSchema",
    "ForecastResult",
    # Integrations
    "IntegrationManager",
    "MLflowTracker",
    # Model adapters
    "ModelAdapter",
    "MonitorConfig",
    # Retraining
    "RetrainingConfig",
    "RetrainingOrchestrator",
    "RetrainingResult",
    "SklearnModelAdapter",
    # Detection
    "StatisticalDriftDetector",
    "StatisticalDriftResult",
    "StorageBackend",
    "ThresholdSnapshot",
    "TorchMLPAdapter",
    "build_demo_scenario",
    "build_demo_system",
    "build_model_adapter",
    "create_store",
    # Evaluation
    "run_demo_evaluation",
]
