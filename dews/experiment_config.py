"""Frozen configuration for the label-delay early-warning experiment.

This module defines the experiment contract shared by the scenario runner,
the temporal benchmark, and the comparison scripts. It does not run an
experiment or create output files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DegradationEventDefinition:
    """Rule for declaring a sustained accuracy-degradation event."""

    baseline_batches: int = 5
    accuracy_drop: float = 0.05
    consecutive_batches: int = 2

    def __post_init__(self) -> None:
        if self.baseline_batches < 1:
            raise ValueError("baseline_batches must be at least 1")
        if not 0.0 < self.accuracy_drop < 1.0:
            raise ValueError("accuracy_drop must be between 0 and 1")
        if self.consecutive_batches < 1:
            raise ValueError("consecutive_batches must be at least 1")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EarlyWarningProtocol:
    """Evaluation constants fixed by the final project scope."""

    horizon_batches: int = 5
    label_delays_batches: tuple[int, ...] = (0, 2, 5)
    false_alert_budget_per_100_clean_batches: float = 5.0
    event: DegradationEventDefinition = field(default_factory=DegradationEventDefinition)

    def __post_init__(self) -> None:
        if self.horizon_batches < 1:
            raise ValueError("horizon_batches must be at least 1")
        if not self.label_delays_batches:
            raise ValueError("at least one label delay is required")
        if any(delay < 0 for delay in self.label_delays_batches):
            raise ValueError("label delays cannot be negative")
        if len(set(self.label_delays_batches)) != len(self.label_delays_batches):
            raise ValueError("label delays must be unique")
        if self.false_alert_budget_per_100_clean_batches < 0.0:
            raise ValueError("false-alert budget cannot be negative")

    def to_dict(self) -> dict:
        event_confirmation_batches = self.event.consecutive_batches - 1
        return {
            "horizon_batches": self.horizon_batches,
            "label_delays_batches": list(self.label_delays_batches),
            "false_alert_budget_per_100_clean_batches": self.false_alert_budget_per_100_clean_batches,
            "degradation_event": self.event.to_dict(),
            "event_confirmation_batches": event_confirmation_batches,
            "risk_target_available_after_batches": {
                str(delay): self.horizon_batches + event_confirmation_batches + delay
                for delay in self.label_delays_batches
            },
        }


@dataclass(frozen=True, slots=True)
class EpisodeSeedSplit:
    """Deterministic seed groups; all scenarios with a seed share its split."""

    train_seeds: tuple[int, ...] = tuple(range(30))
    validation_seeds: tuple[int, ...] = tuple(range(30, 40))
    test_seeds: tuple[int, ...] = tuple(range(40, 50))

    def __post_init__(self) -> None:
        groups = {
            "train": self.train_seeds,
            "validation": self.validation_seeds,
            "test": self.test_seeds,
        }
        for name, seeds in groups.items():
            if not seeds:
                raise ValueError(f"{name} seed group cannot be empty")
            if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
                raise ValueError(f"{name} seeds must be non-negative integers")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} seeds must be unique")

        all_seeds = [seed for seeds in groups.values() for seed in seeds]
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("train, validation, and test seed groups must not overlap")

    def split_for_seed(self, seed: int) -> str:
        if seed in self.train_seeds:
            return "train"
        if seed in self.validation_seeds:
            return "validation"
        if seed in self.test_seeds:
            return "test"
        raise ValueError(f"seed {seed} is not assigned to a data split")

    def to_dict(self) -> dict[str, list[int] | str]:
        return {
            "method": "fixed_disjoint_episode_seed_groups",
            "train_seeds": list(self.train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "test_seeds": list(self.test_seeds),
        }


@dataclass(frozen=True, slots=True)
class ClassifierSpec:
    """Stable tabular classifier shared by the controlled and real streams."""

    estimator: str = "LogisticRegression"
    preprocessing: str = "StandardScaler"
    max_iter: int = 500
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.estimator != "LogisticRegression":
            raise ValueError("the Phase 0 classifier is LogisticRegression")
        if self.preprocessing != "StandardScaler":
            raise ValueError("the Phase 0 preprocessing is StandardScaler")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")

    def build_pipeline(self):
        """Create a fresh pipeline; callers fit it on reference data only."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=self.max_iter, random_state=self.random_state),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            **asdict(self),
            "fit_partition": "reference_training_rows_only",
            "update_base_classifier_during_stream": False,
        }


@dataclass(frozen=True, slots=True)
class EarlyWarningExperimentConfig:
    """Shared protocol, split, model, and run-metadata contract."""

    protocol: EarlyWarningProtocol = field(default_factory=EarlyWarningProtocol)
    seed_split: EpisodeSeedSplit = field(default_factory=EpisodeSeedSplit)
    classifier: ClassifierSpec = field(default_factory=ClassifierSpec)
    synthetic_scenarios: tuple[str, ...] = (
        "clean",
        "harmless_shift",
        "harmful_gradual",
        "harmful_abrupt",
    )
    primary_dataset: str = "controlled_synthetic"
    reality_check_dataset: str = "electricity"
    output_directory: Path = Path("artifacts/early_warning")

    def __post_init__(self) -> None:
        if len(set(self.synthetic_scenarios)) != len(self.synthetic_scenarios):
            raise ValueError("synthetic scenario names must be unique")
        if not self.synthetic_scenarios:
            raise ValueError("at least one synthetic scenario is required")

    def split_for_run(self, dataset: str, scenario: str, seed: int) -> str:
        if dataset == self.primary_dataset:
            if scenario not in self.synthetic_scenarios:
                raise ValueError(f"unknown controlled scenario: {scenario}")
        elif dataset == self.reality_check_dataset:
            if scenario != "chronological":
                raise ValueError("the Electricity reality check uses scenario='chronological'")
        else:
            raise ValueError(f"unknown dataset: {dataset}")
        return self.seed_split.split_for_seed(seed)

    def metadata_for_run(
        self,
        *,
        dataset: str,
        scenario: str,
        seed: int,
        label_delay_batches: int,
        feature_columns: tuple[str, ...] | list[str],
    ) -> dict:
        """Return the required JSON-serializable metadata for one episode."""
        if label_delay_batches not in self.protocol.label_delays_batches:
            raise ValueError(f"unsupported label delay: {label_delay_batches}")
        if not feature_columns or any(not column for column in feature_columns):
            raise ValueError("feature_columns must contain the model input columns")
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns must be unique")

        return {
            "schema_version": 1,
            "experiment": "label_delay_early_warning",
            "dataset": dataset,
            "scenario": scenario,
            "seed": seed,
            "split": self.split_for_run(dataset, scenario, seed),
            "label_delay_batches": label_delay_batches,
            "feature_columns": list(feature_columns),
            "classifier": self.classifier.to_dict(),
            "protocol": self.protocol.to_dict(),
            "seed_split": self.seed_split.to_dict(),
            "output_directory": str(self.output_directory),
        }

    def to_dict(self) -> dict:
        """Return the full protocol manifest for a preflight display."""
        return {
            "experiment": "label_delay_early_warning",
            "protocol": self.protocol.to_dict(),
            "seed_split": self.seed_split.to_dict(),
            "classifier": self.classifier.to_dict(),
            "primary_dataset": self.primary_dataset,
            "reality_check_dataset": self.reality_check_dataset,
            "synthetic_scenarios": list(self.synthetic_scenarios),
            "electricity_scenario": "chronological",
            "output_directory": str(self.output_directory),
        }


DEFAULT_EARLY_WARNING_CONFIG = EarlyWarningExperimentConfig()
