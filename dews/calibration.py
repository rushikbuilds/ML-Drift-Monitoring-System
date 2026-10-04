"""Data-driven calibration of DEWS fusion weights.

The calibration episodes use the labeled demo generator as a reproducible
benchmark. They are useful for regression testing and initial calibration;
production weights should be recalibrated with representative temporal data.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import MonitorConfig
from .data import build_demo_scenario
from .evaluation import run_demo_evaluation
from .monitoring import build_demo_system


@dataclass(slots=True)
class WeightCalibration:
    weights: tuple[float, float, float]
    objective: float
    false_alert_rate: float
    detection_lag: float | None
    missed_detection_rate: float
    seeds_evaluated: int


def simplex_weights(step: float = 0.25) -> list[tuple[float, float, float]]:
    """Return non-negative weights that sum to one."""
    if step <= 0 or step > 1:
        raise ValueError("step must be greater than 0 and no greater than 1")
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly")
    return [
        (round(stat * step, 10), round(emb * step, 10), round((units - stat - emb) * step, 10))
        for stat in range(units + 1)
        for emb in range(units - stat + 1)
    ]


def _evaluate_weights(weights: tuple[float, float, float], seed: int, *, stream_batches: int, batch_size: int) -> tuple[float, float, float]:
    config = MonitorConfig(
        window_size=max(80, batch_size * 2),
        batch_size=batch_size,
        forecast_horizon=8,
        ds_crit=0.30,
        weights=weights,
        adaptive_threshold_enabled=False,
        retraining_enabled=False,
    )
    with tempfile.TemporaryDirectory(prefix="dews-calibration-") as directory:
        clean = build_demo_scenario(seed=seed, stream_batches=stream_batches, batch_size=batch_size, inject_drift=False)
        clean_system, _ = build_demo_system(config, store_path=str(Path(directory) / "clean.sqlite3"))
        clean_summary = run_demo_evaluation(clean_system, clean)
        clean_system.store.close()

        drift = build_demo_scenario(seed=seed, stream_batches=stream_batches, batch_size=batch_size, inject_drift=True)
        drift_system, _ = build_demo_system(config, store_path=str(Path(directory) / "drift.sqlite3"))
        drift_summary = run_demo_evaluation(drift_system, drift)
        drift_system.store.close()

    clean_alerts = clean_summary["alerts_generated"]
    false_alert_rate = clean_alerts / max(stream_batches, 1)
    detection_lag = drift_summary["detection_lag"]
    missed = 1.0 if detection_lag is None else 0.0
    return false_alert_rate, detection_lag if detection_lag is not None else float(stream_batches), missed


def calibrate_fusion_weights(
    *,
    seeds: Iterable[int] = (42, 43),
    candidates: Iterable[tuple[float, float, float]] | None = None,
    stream_batches: int = 18,
    batch_size: int = 40,
) -> dict:
    """Select fusion weights using clean and drifted labeled episodes.

    The objective penalizes false alerts heavily, then detection delay, with a
    missed-detection penalty. Lower objective values are better.
    """
    seed_list = list(seeds)
    candidate_list = list(candidates or simplex_weights())
    baseline_weights = (0.45, 0.35, 0.20)
    if baseline_weights not in candidate_list:
        candidate_list.append(baseline_weights)
    reports: list[WeightCalibration] = []
    for weights in candidate_list:
        measurements = [_evaluate_weights(weights, seed, stream_batches=stream_batches, batch_size=batch_size) for seed in seed_list]
        false_alert_rate = float(np.mean([measurement[0] for measurement in measurements]))
        detection_lag = float(np.mean([measurement[1] for measurement in measurements]))
        missed_rate = float(np.mean([measurement[2] for measurement in measurements]))
        objective = 10.0 * false_alert_rate + detection_lag + 10.0 * missed_rate
        reports.append(WeightCalibration(weights, objective, false_alert_rate, detection_lag, missed_rate, len(seed_list)))

    reports.sort(key=lambda report: report.objective)
    return {
        "best": asdict(reports[0]),
        "baseline": asdict(next(report for report in reports if report.weights == baseline_weights)),
        "candidates": [asdict(report) for report in reports],
        "protocol": {
            "seeds": seed_list,
            "stream_batches": stream_batches,
            "batch_size": batch_size,
            "objective": "10*false_alert_rate + detection_lag + 10*missed_detection_rate",
        },
    }


def save_calibration(report: dict, path: str | Path) -> None:
    """Write calibration results, including the selected weights, as JSON."""
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
