from __future__ import annotations

import pandas as pd
import pytest

from dews.early_warning import run_early_warning_episode
from dews.early_warning_metrics import (
    build_risk_targets,
    evaluate_episode_alerts,
    risk_target_available_batch,
    stable_clean_batch_indices,
)
from dews.risk import (
    DelayedFeedbackRiskCalibrator,
    RiskFeatureBuilder,
    ValidationEpisodeScores,
    select_warning_threshold,
    validation_episodes_from_runs,
)
from dews.scenarios import build_scenario_episode


def _signals(value: float) -> dict[str, float]:
    return {
        "s_stat": value,
        "s_conf": value / 2,
        "mean_confidence": 0.9 - value / 10,
        "low_confidence_rate": value / 4,
        "positive_prediction_rate": 0.5,
        "mean_entropy": value / 3,
    }


def _mature_example(*, seed: int, batch: int, label: int, released_at: int | None = None) -> dict:
    available = risk_target_available_batch(batch, label_delay_batches=0)
    return {
        "scenario": "clean",
        "seed": seed,
        "label_delay_batches": 0,
        "example_batch": batch,
        "label": label,
        "available_batch": available,
        "released_at_batch": available if released_at is None else released_at,
        "features": RiskFeatureBuilder().build([_signals(float(label + batch / 10))]),
    }


def test_risk_targets_wait_for_horizon_confirmation_and_label_delay() -> None:
    targets = build_risk_targets(
        total_batches=20,
        event_onset_batch=10,
        horizon_batches=5,
        label_delay_batches=2,
        event_consecutive_batches=2,
    )

    assert targets[4].label == 0
    assert targets[5].label == 1
    assert targets[4].available_batch == 12
    assert targets[-1].example_batch == 13
    assert targets[-1].available_batch == 21


def test_alert_metrics_keep_no_event_and_clean_false_alerts_explicit() -> None:
    clean = stable_clean_batch_indices(total_batches=30, event_batches=[15], horizon_batches=5)
    metrics = evaluate_episode_alerts(
        [2, 11, 12, 18],
        [15],
        horizon_batches=5,
        clean_batch_indices=clean,
    )
    no_event = evaluate_episode_alerts(
        [3, 8],
        [],
        clean_batch_indices=tuple(range(10)),
    )

    assert metrics["event_recall"] == 1.0
    assert metrics["warning_lead_batches"] == 4.0
    assert metrics["false_alerts_per_100_clean_batches"] == pytest.approx(100 / 10)
    assert no_event["event_recall"] is None
    assert no_event["warning_lead_batches"] is None
    assert no_event["alerts_generated"] == 2


def test_risk_features_use_temporal_positions_and_reject_outcomes() -> None:
    builder = RiskFeatureBuilder()
    features = builder.build([_signals(0.1), {**_signals(0.2), "s_stat": None}, _signals(0.5)])

    assert features["s_stat"] == 0.5
    assert features["s_stat_lag_1"] is None
    assert features["s_stat_lag_2"] == 0.1
    assert features["s_stat_mean_3"] is None
    with pytest.raises(ValueError, match="outcome fields"):
        builder.build([_signals(0.1) | {"realized_accuracy": 1.0}])


def test_calibrator_enforces_maturity_and_training_seed_split() -> None:
    calibrator = DelayedFeedbackRiskCalibrator(min_mature_examples=2, min_examples_per_class=1)
    examples = [
        _mature_example(seed=0, batch=0, label=0),
        _mature_example(seed=1, batch=1, label=1),
    ]

    status = calibrator.fit_initial(examples)
    assert status["fitted"] is True
    assert calibrator.predict(_signals(0.5) | {"s_stat_lag_1": None}) is not None
    with pytest.raises(ValueError, match="availability batch"):
        calibrator.update(_mature_example(seed=2, batch=3, label=0, released_at=3))
    with pytest.raises(ValueError, match="training-split seeds only"):
        calibrator.fit_initial([_mature_example(seed=40, batch=0, label=0)])


def test_threshold_selection_uses_validation_outputs_and_alert_budget() -> None:
    event_episode = ValidationEpisodeScores(
        scenario="harmful_gradual",
        seed=30,
        scores=(0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1),
        event_onset_batch=10,
        clean_batch_indices=tuple(range(5)),
    )
    harmless_episode = ValidationEpisodeScores(
        scenario="harmless_shift",
        seed=31,
        scores=(0.1,) * 12,
        event_onset_batch=None,
        clean_batch_indices=tuple(range(12)),
    )
    selected = select_warning_threshold((event_episode, harmless_episode))

    assert selected["selection_split"] == "validation"
    assert selected["event_recall"] == 1.0
    assert selected["false_alerts_per_100_clean_batches"] <= 5.0
    with pytest.raises(ValueError, match="validation episodes only"):
        validation_episodes_from_runs(
            [{"metadata": {"run": {"split": "test"}}, "batch_records": []}],
            score_name="risk_probability",
        )


def test_scenario_is_reproducible_and_matches_expected_event() -> None:
    first = build_scenario_episode("harmful_abrupt", seed=0)
    second = build_scenario_episode("harmful_abrupt", seed=0)

    assert first.expected_event_onset_batch is not None
    assert second.expected_event_onset_batch == first.expected_event_onset_batch
    for left, right in zip(first.stream_batches, second.stream_batches):
        pd.testing.assert_frame_equal(left, right)


def test_replay_releases_labels_and_risk_targets_at_maturity() -> None:
    episode = build_scenario_episode(
        "clean",
        seed=0,
        reference_size=512,
        stream_batches=12,
        batch_size=64,
    )
    updates = []
    predictor_features = []
    result = run_early_warning_episode(
        episode,
        label_delay_batches=2,
        detector_reference_rows=32,
        risk_predictor=lambda features: predictor_features.append(features) or None,
        risk_target_updater=updates.append,
        persist=False,
    )

    assert result["metadata"]["live_monitor_receives_labels"] is False
    assert len(predictor_features) == len(episode.stream_batches)
    assert all("realized_accuracy" not in features for features in predictor_features)
    assert "fixed_score_alerts" not in result["summary"]
    assert all("fixed_score_alert" not in record for record in result["batch_records"])
    for index, record in enumerate(result["batch_records"]):
        assert record["label_release_batch"] == index + 2
        assert record["evaluation_only"]["accuracy_released_at_batch"] == index + 2
    assert len(updates) == len(result["released_risk_targets"]) == 6
    assert all(packet["released_at_batch"] >= packet["available_batch"] for packet in updates)
