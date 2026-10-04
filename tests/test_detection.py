from __future__ import annotations

from dews import MonitorConfig, build_demo_system


def test_statistical_detector_scores_drifted_batch_higher(tmp_path) -> None:
    config = MonitorConfig(window_size=80, batch_size=40, forecast_horizon=8)
    system, scenario = build_demo_system(config, store_path=str(tmp_path / "detector.sqlite3"))
    reference = scenario.reference_frame[scenario.feature_schema.continuous + scenario.feature_schema.categorical]

    baseline = system.statistical_detector.evaluate(reference, scenario.stream_batches[0], scenario.feature_schema)
    drifted = system.statistical_detector.evaluate(reference, scenario.stream_batches[-1], scenario.feature_schema)

    assert drifted.s_stat >= baseline.s_stat or drifted.mean_jsd >= baseline.mean_jsd
