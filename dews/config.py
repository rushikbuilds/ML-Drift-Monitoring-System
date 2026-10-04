from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class MonitorConfig:
    window_size: int = 160
    batch_size: int = 80
    forecast_horizon: int = 8
    ds_crit: float = 0.30
    low_confidence_threshold: float = 0.55
    drift_cooldown_batches: int = 3
    alert_confirmation_batches: int = 2
    forecast_min_history: int = 4
    critical_score_margin: float = 0.10
    statistical_bins: int = 10
    confidence_bins: int = 10
    embedding_components: int = 5
    weights: tuple[float, float, float] = field(default_factory=lambda: (0.45, 0.35, 0.20))
    time_unit: str = "batches"
    # --- MLOps extensions (Priority 1) ---
    model_version: str = "unversioned"
    reference_data_hash: str = ""
    top_k_features: int = 5
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "dews-drift-monitoring"
    # --- Adaptive threshold (Priority 2) ---
    adaptive_threshold_enabled: bool = False
    adaptive_warmup_batches: int = 6
    adaptive_sensitivity: float = 2.5
    adaptive_ema_alpha: float = 0.15
    adaptive_floor: float = 0.15
    adaptive_ceiling: float = 0.70
    # --- Retraining (Priority 2) ---
    retraining_enabled: bool = False
    retraining_runner: str = "local"  # "local", "webhook", "in_process"
    retraining_cooldown_seconds: float = 300.0
    retraining_command: str = "python train_and_monitor_pytorch.py"
    retraining_webhook_url: str = ""
    retraining_max_attempts: int = 3
    # --- Storage backend (Priority 2) ---
    storage_dsn: str = ""  # empty = use store_path (SQLite); set to postgresql:// for PG

    def normalized_weights(self) -> tuple[float, float, float]:
        total = sum(max(weight, 0.0) for weight in self.weights)
        if total == 0:
            return (0.45, 0.35, 0.20)
        return tuple(max(weight, 0.0) / total for weight in self.weights)

    @staticmethod
    def compute_reference_hash(reference_frame: pd.DataFrame) -> str:
        """Compute a deterministic SHA-256 hash of the reference dataset for versioning."""
        content_arr = reference_frame.to_numpy()
        raw_bytes = content_arr.tobytes()
        return hashlib.sha256(raw_bytes).hexdigest()[:16]
