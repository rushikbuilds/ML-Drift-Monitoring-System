"""Deterministic controlled episodes for the early-warning experiment.

These scenarios are separate from the legacy dashboard/demo generator in
``dews.data``. Labels are generated from a latent rule and remain fixed when
observed features are intervened on, so harmful scenarios change the observed
feature/label relationship while the classifier stays frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

import numpy as np
import pandas as pd

from .data import FeatureSchema
from .early_warning_metrics import accuracy_baseline, find_degradation_onset
from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG


ScenarioName = Literal["clean", "harmless_shift", "harmful_gradual", "harmful_abrupt"]
FEATURE_COLUMNS = ("signal_0", "signal_1", "noise_0", "nuisance_0")


@dataclass(frozen=True, slots=True)
class ScenarioEpisode:
    """One generated episode plus explicit evaluation-only event metadata.

    ``expected_event_onset_batch`` and the ``evaluation_only`` metadata are
    outcome annotations. A live monitor must not read them while producing
    signals or risk predictions.
    """

    scenario: ScenarioName
    seed: int
    reference_frame: pd.DataFrame
    reference_labels: pd.Series
    stream_batches: tuple[pd.DataFrame, ...]
    stream_labels: tuple[pd.Series, ...]
    feature_schema: FeatureSchema
    intervention_onset_batch: int | None
    expected_event_onset_batch: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.stream_batches[0]) if self.stream_batches else 0

    @property
    def total_batches(self) -> int:
        return len(self.stream_batches)


def _sample_base_episode(
    rng: np.random.Generator,
    *,
    reference_size: int,
    stream_rows: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Sample data whose label boundary is a stable, known linear rule."""

    def sample_rows(count: int) -> tuple[pd.DataFrame, pd.Series]:
        values = rng.normal(size=(count, 3))
        frame = pd.DataFrame(values, columns=FEATURE_COLUMNS[:3])
        # This column is constant during reference training, so its fitted
        # linear coefficient is exactly zero. Shifting it is demonstrably
        # harmless to the frozen classifier while remaining a visible input
        # distribution change to drift detectors.
        frame[FEATURE_COLUMNS[3]] = 0.0
        labels = pd.Series(
            (values[:, 0] + 0.8 * values[:, 1] > 0.0).astype(int),
            name="target",
        )
        return frame.loc[:, list(FEATURE_COLUMNS)], labels

    reference_frame, reference_labels = sample_rows(reference_size)
    stream_frame, stream_labels = sample_rows(stream_rows)
    return reference_frame, reference_labels, stream_frame, stream_labels


def _apply_intervention(
    stream: pd.DataFrame,
    *,
    scenario: ScenarioName,
    intervention_onset_batch: int | None,
    batch_size: int,
    gradual_ramp_batches: int,
    harmless_shift_magnitude: float,
    harmful_shift_magnitude: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transformed = stream.copy()
    intervention: dict[str, Any] = {
        "type": "none",
        "feature": None,
        "onset_batch": None,
        "latent_label_rule_changed": False,
        "observed_feature_label_relationship_changed": False,
    }
    if scenario == "clean":
        return transformed, intervention

    if intervention_onset_batch is None:
        raise ValueError("this scenario requires an intervention onset batch")
    onset_row = intervention_onset_batch * batch_size

    if scenario == "harmless_shift":
        transformed.loc[onset_row:, "nuisance_0"] += harmless_shift_magnitude
        intervention = {
            "type": "constant_location_shift",
            "feature": "nuisance_0",
            "magnitude": float(harmless_shift_magnitude),
            "onset_batch": intervention_onset_batch,
            "latent_label_rule_changed": False,
            "observed_feature_label_relationship_changed": False,
        }
    elif scenario == "harmful_abrupt":
        transformed.loc[onset_row:, "signal_0"] += harmful_shift_magnitude
        intervention = {
            "type": "abrupt_predictive_feature_shift",
            "feature": "signal_0",
            "magnitude": float(harmful_shift_magnitude),
            "onset_batch": intervention_onset_batch,
            "latent_label_rule_changed": False,
            "observed_feature_label_relationship_changed": True,
        }
    elif scenario == "harmful_gradual":
        offsets = []
        for batch_index in range(intervention_onset_batch, len(stream) // batch_size):
            ramp_fraction = min(
                (batch_index - intervention_onset_batch + 1) / gradual_ramp_batches,
                1.0,
            )
            offset = harmful_shift_magnitude * ramp_fraction
            start = batch_index * batch_size
            stop = min(start + batch_size, len(stream))
            transformed.loc[start:stop - 1, "signal_0"] += offset
            offsets.append(float(offset))
        intervention = {
            "type": "gradual_predictive_feature_shift",
            "feature": "signal_0",
            "maximum_magnitude": float(harmful_shift_magnitude),
            "ramp_batches": gradual_ramp_batches,
            "batch_offsets": offsets,
            "onset_batch": intervention_onset_batch,
            "latent_label_rule_changed": False,
            "observed_feature_label_relationship_changed": True,
        }
    return transformed, intervention


def _expected_event_onset(
    reference_frame: pd.DataFrame,
    reference_labels: pd.Series,
    stream_batches: tuple[pd.DataFrame, ...],
    stream_labels: tuple[pd.Series, ...],
) -> tuple[int | None, float | None, list[float]]:
    """Measure the expected event using only the reference-fitted classifier."""
    classifier_spec = DEFAULT_EARLY_WARNING_CONFIG.classifier
    classifier = classifier_spec.build_pipeline()
    classifier.fit(reference_frame, reference_labels)
    accuracies = [
        float(np.mean(classifier.predict(batch) == labels.to_numpy()))
        for batch, labels in zip(stream_batches, stream_labels)
    ]
    event_definition = DEFAULT_EARLY_WARNING_CONFIG.protocol.event
    baseline_accuracy = accuracy_baseline(
        accuracies,
        event_definition.baseline_batches,
    )
    onset = find_degradation_onset(
        accuracies,
        baseline_accuracy,
        start_batch=event_definition.baseline_batches,
        accuracy_drop=event_definition.accuracy_drop,
        consecutive_batches=event_definition.consecutive_batches,
    )
    return onset, baseline_accuracy, accuracies


def build_scenario_episode(
    scenario: ScenarioName,
    *,
    seed: int = 0,
    reference_size: int = 2048,
    stream_batches: int = 30,
    batch_size: int = 128,
    intervention_onset_batch: int | None = None,
    gradual_ramp_batches: int = 8,
    harmless_shift_magnitude: float = 3.0,
    harmful_shift_magnitude: float = 5.0,
) -> ScenarioEpisode:
    """Build one reproducible scenario using the Phase 0 classifier protocol.

    The returned labels describe the latent data before intervention. Harmful
    interventions alter the observed predictive feature while preserving those
    labels; the harmless intervention shifts only the classifier-irrelevant
    nuisance column. Event metadata is measured with a classifier fit on the
    reference partition only.
    """
    allowed_scenarios = DEFAULT_EARLY_WARNING_CONFIG.synthetic_scenarios
    if scenario not in allowed_scenarios:
        raise ValueError(f"unknown scenario: {scenario}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if seed not in (
        DEFAULT_EARLY_WARNING_CONFIG.seed_split.train_seeds
        + DEFAULT_EARLY_WARNING_CONFIG.seed_split.validation_seeds
        + DEFAULT_EARLY_WARNING_CONFIG.seed_split.test_seeds
    ):
        raise ValueError("seed must belong to the configured train/validation/test groups")
    if reference_size < 2:
        raise ValueError("reference_size must be at least 2")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    minimum_stream_batches = (
        DEFAULT_EARLY_WARNING_CONFIG.protocol.event.baseline_batches
        + DEFAULT_EARLY_WARNING_CONFIG.protocol.horizon_batches
        + DEFAULT_EARLY_WARNING_CONFIG.protocol.event.consecutive_batches
    )
    if stream_batches < minimum_stream_batches:
        raise ValueError("stream_batches must cover the baseline, warning horizon, and confirmation window")
    if gradual_ramp_batches < 1:
        raise ValueError("gradual_ramp_batches must be at least 1")
    if (
        not isfinite(harmless_shift_magnitude)
        or not isfinite(harmful_shift_magnitude)
        or harmless_shift_magnitude <= 0.0
        or harmful_shift_magnitude <= 0.0
    ):
        raise ValueError("intervention magnitudes must be positive")

    if scenario == "clean":
        onset_batch = None
    elif intervention_onset_batch is None:
        onset_batch = max(
            DEFAULT_EARLY_WARNING_CONFIG.protocol.event.baseline_batches,
            stream_batches // 3,
        )
    else:
        onset_batch = intervention_onset_batch

    if onset_batch is not None:
        if not isinstance(onset_batch, int) or isinstance(onset_batch, bool):
            raise ValueError("intervention_onset_batch must be an integer")
        baseline_batches = DEFAULT_EARLY_WARNING_CONFIG.protocol.event.baseline_batches
        if onset_batch < baseline_batches:
            raise ValueError("intervention_onset_batch must follow the clean baseline window")
        if onset_batch >= stream_batches:
            raise ValueError("intervention_onset_batch must be within the stream")
        required_tail = (
            DEFAULT_EARLY_WARNING_CONFIG.protocol.horizon_batches
            + DEFAULT_EARLY_WARNING_CONFIG.protocol.event.consecutive_batches
        )
        if stream_batches - onset_batch < required_tail:
            raise ValueError("stream must include post-intervention horizon and event-confirmation batches")

    rng = np.random.default_rng(seed)
    reference_frame, reference_labels, base_stream, stream_target = _sample_base_episode(
        rng,
        reference_size=reference_size,
        stream_rows=stream_batches * batch_size,
    )
    if reference_labels.nunique() != 2:
        raise ValueError("reference data must contain both target classes")
    observed_stream, intervention = _apply_intervention(
        base_stream,
        scenario=scenario,
        intervention_onset_batch=onset_batch,
        batch_size=batch_size,
        gradual_ramp_batches=gradual_ramp_batches,
        harmless_shift_magnitude=harmless_shift_magnitude,
        harmful_shift_magnitude=harmful_shift_magnitude,
    )

    batches = tuple(
        observed_stream.iloc[start:start + batch_size].reset_index(drop=True)
        for start in range(0, len(observed_stream), batch_size)
    )
    labels = tuple(
        stream_target.iloc[start:start + batch_size].reset_index(drop=True)
        for start in range(0, len(stream_target), batch_size)
    )
    expected_onset, baseline_accuracy, accuracies = _expected_event_onset(
        reference_frame,
        reference_labels,
        batches,
        labels,
    )
    harmful_scenario = scenario in {"harmful_gradual", "harmful_abrupt"}
    if harmful_scenario and expected_onset is None:
        raise RuntimeError(
            f"{scenario} did not produce a confirmed accuracy event for seed {seed}; "
            "increase the intervention magnitude or stream length"
        )
    if not harmful_scenario and expected_onset is not None:
        raise RuntimeError(
            f"{scenario} unexpectedly produced an accuracy event for seed {seed}; "
            "increase the reference or batch size"
        )

    split = DEFAULT_EARLY_WARNING_CONFIG.seed_split.split_for_seed(seed)
    metadata = {
        "schema_version": 1,
        "experiment": "label_delay_early_warning",
        "dataset": DEFAULT_EARLY_WARNING_CONFIG.primary_dataset,
        "scenario": scenario,
        "seed": seed,
        "split": split,
        "generator": "linear_tabular_v1",
        "latent_label_rule": "signal_0 + 0.8 * signal_1 > 0",
        "reference_rows": reference_size,
        "stream_batches": stream_batches,
        "batch_size": batch_size,
        "feature_columns": list(FEATURE_COLUMNS),
        "classifier": DEFAULT_EARLY_WARNING_CONFIG.classifier.to_dict(),
        "protocol": DEFAULT_EARLY_WARNING_CONFIG.protocol.to_dict(),
        "intervention": intervention,
        "evaluation_only": {
            "expected_event_present": expected_onset is not None,
            "expected_event_onset_batch": expected_onset,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_by_batch": accuracies,
        },
    }
    return ScenarioEpisode(
        scenario=scenario,
        seed=seed,
        reference_frame=reference_frame.reset_index(drop=True),
        reference_labels=reference_labels.reset_index(drop=True),
        stream_batches=batches,
        stream_labels=labels,
        feature_schema=FeatureSchema(
            continuous=list(FEATURE_COLUMNS),
            categorical=[],
        ),
        intervention_onset_batch=onset_batch,
        expected_event_onset_batch=expected_onset,
        metadata=metadata,
    )


def build_scenario_suite(
    *,
    seed: int,
    scenario_names: tuple[str, ...] = DEFAULT_EARLY_WARNING_CONFIG.synthetic_scenarios,
    **episode_options: Any,
) -> dict[str, ScenarioEpisode]:
    """Build matched episodes that share the same seed and base data."""
    if not scenario_names:
        raise ValueError("scenario_names cannot be empty")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario_names must be unique")
    return {
        scenario: build_scenario_episode(
            scenario,
            seed=seed,
            **episode_options,
        )
        for scenario in scenario_names
    }
