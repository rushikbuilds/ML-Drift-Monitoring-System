from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import MonitorConfig
from .forecasting import ForecastResult


@dataclass(slots=True)
class FeatureDriftSummary:
    """Summary of a single feature's drift contribution."""
    feature: str
    kind: str  # "continuous" or "categorical"
    score: float
    ks: float | None = None
    psi: float | None = None
    chi_square: float | None = None
    jsd: float | None = None


@dataclass(slots=True)
class AlertRecord:
    timestamp: str
    batch_index: int
    severity: str
    message: str
    trigger_value: float
    predicted_ttd: float | None
    source: str = "forecast"
    model_version: str = "unversioned"
    top_features: list[dict] = field(default_factory=list)


class AlertEngine:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self._last_alert_batch: int | None = None
        self._candidate_streak = 0
        self._last_candidate_batch: int | None = None
        self.history: list[AlertRecord] = []

    def evaluate(
        self,
        batch_index: int,
        current_ds: float,
        forecast: ForecastResult,
        per_feature: list[dict] | None = None,
        effective_threshold: float | None = None,
        forecast_eligible: bool = True,
    ) -> AlertRecord | None:
        threshold = effective_threshold if effective_threshold is not None else self.config.ds_crit
        projected_ttd = forecast.ttd_batches
        if not forecast_eligible:
            projected_ttd = None
        if current_ds >= threshold:
            severity = "Critical" if current_ds >= threshold + self.config.critical_score_margin else "Warning"
            level = "critical" if severity == "Critical" else "warning"
            message = f"Current drift score {current_ds:.3f} exceeds the {level} threshold."
        elif projected_ttd is not None and projected_ttd <= self.config.forecast_horizon:
            if projected_ttd <= 2:
                severity = "Critical"
            elif projected_ttd <= 4:
                severity = "Warning"
            else:
                severity = "Info"
            message = f"Forecasted drift score will cross the critical threshold in about {projected_ttd:.1f} {self.config.time_unit}."
        else:
            self._candidate_streak = 0
            self._last_candidate_batch = None
            return None

        if self._last_candidate_batch is not None and batch_index == self._last_candidate_batch + 1:
            self._candidate_streak += 1
        else:
            self._candidate_streak = 1
        self._last_candidate_batch = batch_index
        if self._candidate_streak < max(self.config.alert_confirmation_batches, 1):
            return None

        if self._last_alert_batch is not None and batch_index - self._last_alert_batch < self.config.drift_cooldown_batches:
            return None

        # --- Explainable drift: top-K drifting features ---
        top_features = _extract_top_features(per_feature, self.config.top_k_features)
        if top_features:
            feature_names = ", ".join(f["feature"] for f in top_features)
            message += f" Top drifting features: [{feature_names}]."

        record = AlertRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            batch_index=batch_index,
            severity=severity,
            message=message,
            trigger_value=current_ds,
            predicted_ttd=projected_ttd,
            model_version=self.config.model_version,
            top_features=top_features,
        )
        self.history.append(record)
        self._last_alert_batch = batch_index
        return record


def _extract_top_features(per_feature: list[dict] | None, top_k: int) -> list[dict]:
    """Extract the top-K features by drift score from per-feature records."""
    if not per_feature:
        return []
    sorted_features = sorted(per_feature, key=lambda f: f.get("score", 0.0), reverse=True)
    return [
        {
            "feature": f["feature"],
            "kind": f["kind"],
            "score": round(f["score"], 4),
            "ks": f.get("ks"),
            "psi": round(f.get("psi", 0.0), 4) if f.get("psi") is not None else None,
            "jsd": round(f.get("jsd", 0.0), 4) if f.get("jsd") is not None else None,
        }
        for f in sorted_features[:top_k]
    ]
