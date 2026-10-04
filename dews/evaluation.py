from __future__ import annotations

import numpy as np

from .data import DemoScenario
from .monitoring import DriftMonitoringSystem


def _detection_lag(drift_start_batch: int, triggered_batches: list[int]) -> float | None:
    if not triggered_batches:
        return None
    return float(max(triggered_batches[0] - drift_start_batch, 0))


def run_demo_evaluation(system: DriftMonitoringSystem, scenario: DemoScenario, max_batches: int | None = None) -> dict:
    total_batches = len(scenario.stream_batches)
    limit = min(max_batches or total_batches, total_batches)

    drift_score_triggers: list[int] = []
    baseline_ks_triggers: list[int] = []
    baseline_psi_triggers: list[int] = []
    accuracy_series: list[float] = []

    live_batches = []
    import pandas as pd
    reference_frame = scenario.reference_frame[scenario.feature_schema.continuous + scenario.feature_schema.categorical]

    for batch_index in range(limit):
        live_batches.append(scenario.stream_batches[batch_index])
        live_window = pd.concat(live_batches, ignore_index=True).tail(system.config.window_size).reset_index(drop=True)
        
        result = system.process_batch(
            batch_index=batch_index,
            reference_frame=reference_frame,
            live_window=live_window,
            batch_labels=scenario.stream_labels[batch_index]
        )
        if result.accuracy is not None:
            accuracy_series.append(result.accuracy)
        if result.drift_score >= system.config.ds_crit:
            drift_score_triggers.append(batch_index)
        if result.statistical["mean_ks"] >= 0.25:
            baseline_ks_triggers.append(batch_index)
        if result.statistical["mean_psi"] >= 0.15:
            baseline_psi_triggers.append(batch_index)

    alerts = system.store.load_alerts()
    forecasts = system.store.load_forecasts()
    drift_scores = system.store.load_drift_scores()

    alert_batches = [a["batch_index"] for a in alerts]
    post_drift_alerts = [b for b in alert_batches if b >= scenario.drift_start_batch]
    first_post = post_drift_alerts[0] if post_drift_alerts else None
    alert_detection_lag = float(first_post - scenario.drift_start_batch) if first_post is not None else None

    first_breach = drift_score_triggers[0] if drift_score_triggers else None
    first_alert = alert_batches[0] if alert_batches else None
    warning_lead = None
    if first_alert is not None and first_breach is not None and first_alert < first_breach:
        warning_lead = float(first_breach - first_alert)
    elif alerts and alerts[0].get("predicted_ttd") is not None and alerts[0]["predicted_ttd"] > 0:
        warning_lead = round(float(alerts[0]["predicted_ttd"]), 1)

    false_alerts = sum(1 for b in alert_batches if b < scenario.drift_start_batch)
    far = false_alerts / max(scenario.drift_start_batch, 1)

    return {
        "drift_start_batch": scenario.drift_start_batch,
        "processed_batches": limit,
        "final_drift_score": drift_scores[-1]["ds"] if drift_scores else None,
        "alerts_generated": len(alerts),
        "critical_alerts": sum(1 for alert in alerts if alert["severity"] == "Critical"),
        "detection_lag": _detection_lag(scenario.drift_start_batch, drift_score_triggers),
        "alert_detection_lag": alert_detection_lag,
        "warning_lead": warning_lead,
        "false_alert_rate": far,
        "first_alert_batch": first_alert,
        "threshold_breach_batch": first_breach,
        "baseline_ks_detection_lag": _detection_lag(scenario.drift_start_batch, baseline_ks_triggers),
        "baseline_psi_detection_lag": _detection_lag(scenario.drift_start_batch, baseline_psi_triggers),
        "average_accuracy": float(np.mean(accuracy_series)) if accuracy_series else None,
        "forecast_model": forecasts[-1]["model_name"] if forecasts else None,
        "last_forecast": forecasts[-1]["forecast_json"] if forecasts else None,
        "alert_severities": [alert["severity"] for alert in alerts],
    }
