from __future__ import annotations

from pathlib import Path

from dews import MonitorConfig, build_demo_system, run_demo_evaluation


def test_demo_pipeline_generates_alerts(tmp_path: Path) -> None:
    config = MonitorConfig(window_size=80, batch_size=40, forecast_horizon=8)
    system, scenario = build_demo_system(config, store_path=str(tmp_path / "demo.sqlite3"))

    summary = run_demo_evaluation(system, scenario)

    assert summary["processed_batches"] == len(scenario.stream_batches)
    assert summary["alerts_generated"] >= 1
    assert summary["critical_alerts"] >= 1
    assert summary["detection_lag"] is not None
    assert summary["final_drift_score"] is not None
