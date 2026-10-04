from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MonitorConfigPayload(BaseModel):
    model_config = {"protected_namespaces": ()}

    window_size: int = 160
    batch_size: int = 80
    forecast_horizon: int = 8
    ds_crit: float = 0.30
    low_confidence_threshold: float = 0.55
    drift_cooldown_batches: int = 3
    statistical_bins: int = 10
    confidence_bins: int = 10
    embedding_components: int = 5
    weights: tuple[float, float, float] = (0.45, 0.35, 0.20)
    time_unit: str = "batches"
    model_version: str = "unversioned"
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "dews-drift-monitoring"
    # Priority 2
    adaptive_threshold_enabled: bool = False
    adaptive_warmup_batches: int = 6
    adaptive_sensitivity: float = 2.5
    adaptive_ema_alpha: float = 0.15
    adaptive_floor: float = 0.15
    adaptive_ceiling: float = 0.70
    retraining_enabled: bool = False
    retraining_runner: str = "local"
    retraining_cooldown_seconds: float = 300.0
    retraining_command: str = "python train_and_monitor_pytorch.py"
    retraining_webhook_url: str = ""
    retraining_max_attempts: int = 3
    storage_dsn: str = ""


class DemoResetRequest(BaseModel):
    config: MonitorConfigPayload | None = Field(default=None)
    store_path: str | None = None


class StepResponse(BaseModel):
    processed: int
    total_batches: int
    result: dict[str, Any] | None


class StatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ready: bool
    processed: int
    total_batches: int
    drift_start_batch: int
    config: MonitorConfigPayload
    latest_drift_score: float | None
    latest_ttd: float | None
    latest_alert: dict[str, Any] | None
    model_version: str = "unversioned"


class BenchmarkResponse(BaseModel):
    report: dict[str, Any]


class ModelRegistryEntry(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_version: str
    reference_data_hash: str
    registered_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
