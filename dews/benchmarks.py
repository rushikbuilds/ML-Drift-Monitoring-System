from __future__ import annotations

from dataclasses import dataclass

from .config import MonitorConfig
from .evaluation import run_demo_evaluation
from .monitoring import build_demo_system


@dataclass(slots=True)
class BenchmarkReport:
    drift_lag: float | None
    ks_lag: float | None
    psi_lag: float | None
    alerts_generated: int
    critical_alerts: int
    average_accuracy: float | None


def run_benchmark_suite(config: MonitorConfig | None = None) -> dict:
    active_config = config or MonitorConfig()
    system, scenario = build_demo_system(active_config)
    summary = run_demo_evaluation(system, scenario)
    report = BenchmarkReport(
        drift_lag=summary["detection_lag"],
        ks_lag=summary["baseline_ks_detection_lag"],
        psi_lag=summary["baseline_psi_detection_lag"],
        alerts_generated=summary["alerts_generated"],
        critical_alerts=summary["critical_alerts"],
        average_accuracy=summary["average_accuracy"],
    )
    return {
        "report": {
            "drift_lag": report.drift_lag,
            "ks_lag": report.ks_lag,
            "psi_lag": report.psi_lag,
            "alerts_generated": report.alerts_generated,
            "critical_alerts": report.critical_alerts,
            "average_accuracy": report.average_accuracy,
        }
    }
