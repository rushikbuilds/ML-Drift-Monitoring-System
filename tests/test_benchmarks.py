from __future__ import annotations

import pandas as pd

from dews import MonitorConfig
from dews.benchmarks import run_benchmark_suite
from dews.calibration import calibrate_fusion_weights, simplex_weights
from dews.temporal_benchmark import (
    calibrate_electricity_weights,
    run_electricity_benchmark,
)


def test_benchmark_suite_returns_comparison_metrics() -> None:
    report = run_benchmark_suite(MonitorConfig(window_size=80, batch_size=40, forecast_horizon=8))

    assert "report" in report
    payload = report["report"]
    assert payload["drift_lag"] is not None
    assert payload["psi_lag"] is not None
    assert payload["alerts_generated"] >= 1


def test_simplex_weights_sum_to_one() -> None:
    candidates = simplex_weights(step=0.5)

    assert len(candidates) == 6
    assert all(abs(sum(weights) - 1.0) < 1e-9 for weights in candidates)


def test_calibration_returns_baseline_and_best() -> None:
    report = calibrate_fusion_weights(
        seeds=(42,),
        candidates=[(0.45, 0.35, 0.20), (1.0, 0.0, 0.0)],
        stream_batches=18,
        batch_size=20,
    )

    assert tuple(report["best"]["weights"]) in ((0.45, 0.35, 0.20), (1.0, 0.0, 0.0))
    assert report["baseline"] is not None


def test_electricity_benchmark_replays_temporal_csv() -> None:
    rows = []
    for index in range(80):
        rows.append({
            "date": f"2020-01-{index + 1:02d}",
            "feature_a": index / 80,
            "feature_b": float(index % 4),
            "class": ("UP" if index % 2 else "DOWN") if index < 48 else "DOWN",
        })
    frame = pd.DataFrame(rows)

    report = run_electricity_benchmark(
        frame,
        reference_fraction=0.60,
        batch_size=8,
    )

    assert report["dataset"] == "electricity"
    assert report["batches_processed"] == 4


def test_electricity_weight_calibration_compares_baseline() -> None:
    rows = []
    for index in range(80):
        rows.append({
            "feature_a": index / 80,
            "feature_b": float(index % 4),
            "class": ("UP" if index % 2 else "DOWN") if index < 48 else "DOWN",
        })
    report = calibrate_electricity_weights(
        pd.DataFrame(rows),
        candidates=[(1.0, 0.0, 0.0)],
        reference_fractions=(0.60,),
        batch_size=8,
        max_detector_reference_rows=32,
    )

    assert report["best"]["weights"] in ((1.0, 0.0, 0.0), (0.45, 0.35, 0.20))
    assert report["baseline"] is not None
