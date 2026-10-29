"""Run the seed-split, label-delay early-warning evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dews import (
    build_scenario_episode,
    fit_initial_risk_calibrator,
    run_early_warning_episode,
    select_warning_threshold,
    validation_episodes_from_runs,
)
from dews.experiment_config import DEFAULT_EARLY_WARNING_CONFIG


_CONFIG = DEFAULT_EARLY_WARNING_CONFIG
_BASELINES = (
    "statistical_only",
    "confidence_only",
    "fixed_weighted_score",
    "recent_slope_forecast",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled delayed-feedback early-warning evaluation."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=_CONFIG.synthetic_scenarios,
        default=list(_CONFIG.synthetic_scenarios),
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(_CONFIG.seed_split.train_seeds))
    parser.add_argument(
        "--validation-seeds",
        nargs="+",
        type=int,
        default=list(_CONFIG.seed_split.validation_seeds),
    )
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(_CONFIG.seed_split.test_seeds))
    parser.add_argument(
        "--label-delays",
        nargs="+",
        type=int,
        default=list(_CONFIG.protocol.label_delays_batches),
    )
    parser.add_argument("--stream-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reference-size", type=int, default=2048)
    parser.add_argument("--detector-reference-rows", type=int, default=512)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=_CONFIG.output_directory,
        help="Directory for per-episode ledgers and aggregate_results.json.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for split_name, seeds in (
        ("train", args.train_seeds),
        ("validation", args.validation_seeds),
        ("test", args.test_seeds),
    ):
        if not seeds:
            raise ValueError(f"at least one {split_name} seed is required")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{split_name} seeds must be unique")
        for seed in seeds:
            actual_split = _CONFIG.seed_split.split_for_seed(seed)
            if actual_split != split_name:
                raise ValueError(f"seed {seed} belongs to {actual_split}, not {split_name}")
    if not args.scenarios or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("scenario names must be non-empty and unique")
    if not args.label_delays or len(set(args.label_delays)) != len(args.label_delays):
        raise ValueError("label delays must be non-empty and unique")
    if any(delay not in _CONFIG.protocol.label_delays_batches for delay in args.label_delays):
        raise ValueError(f"label delays must be selected from {_CONFIG.protocol.label_delays_batches}")
    minimum_batches = (
        _CONFIG.protocol.event.baseline_batches
        + _CONFIG.protocol.horizon_batches
        + _CONFIG.protocol.event.consecutive_batches
    )
    if args.stream_batches < minimum_batches:
        raise ValueError(f"stream batches must be at least {minimum_batches}")
    if args.batch_size < 2 or args.reference_size < 2:
        raise ValueError("batch size and reference size must be at least 2")
    if args.detector_reference_rows < 2:
        raise ValueError("detector reference rows must be at least 2")


def _episode_run(
    *,
    scenario: str,
    seed: int,
    delay: int,
    args: argparse.Namespace,
    calibrator=None,
    risk_threshold: float | None = None,
    fallback_threshold: float | None = None,
    baseline_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    episode = build_scenario_episode(
        scenario,
        seed=seed,
        reference_size=args.reference_size,
        stream_batches=args.stream_batches,
        batch_size=args.batch_size,
    )
    active_calibrator = deepcopy(calibrator) if calibrator is not None else None
    return run_early_warning_episode(
        episode,
        label_delay_batches=delay,
        detector_reference_rows=args.detector_reference_rows,
        risk_predictor=active_calibrator.predict if active_calibrator is not None else None,
        risk_threshold=risk_threshold,
        fallback_threshold=fallback_threshold,
        baseline_thresholds=baseline_thresholds,
        risk_target_updater=active_calibrator.update if active_calibrator is not None else None,
        output_directory=args.output_directory,
    )


def _run_split(
    *,
    split_name: str,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    delays: Sequence[int],
    args: argparse.Namespace,
    calibrator=None,
    risk_thresholds: dict[int, float] | None = None,
    fallback_threshold: float | None = None,
    baseline_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    runs = []
    total = len(seeds) * len(scenarios) * len(delays)
    completed = 0
    for seed in seeds:
        for scenario in scenarios:
            for delay in delays:
                run = _episode_run(
                    scenario=scenario,
                    seed=seed,
                    delay=delay,
                    args=args,
                    calibrator=calibrator,
                    risk_threshold=(risk_thresholds or {}).get(delay),
                    fallback_threshold=fallback_threshold,
                    baseline_thresholds=baseline_thresholds,
                )
                runs.append(run)
                completed += 1
                print(
                    f"{split_name}: {completed}/{total} "
                    f"{scenario}, seed={seed}, delay={delay}",
                    flush=True,
                )
    return runs


def _risk_probability_targets(run: dict[str, Any]) -> list[tuple[float, int]]:
    records = {record["batch_index"]: record for record in run["batch_records"]}
    scored = []
    for target in run["released_risk_targets"]:
        probability = records[target["example_batch"]].get("risk_probability")
        if probability is not None:
            scored.append((float(probability), int(target["label"])))
    return scored


def _brier_metrics(run: dict[str, Any]) -> dict[str, Any]:
    scored = _risk_probability_targets(run)
    total_targets = len(run["released_risk_targets"])
    return {
        "brier_score": (
            mean((probability - label) ** 2 for probability, label in scored)
            if scored
            else None
        ),
        "brier_scored_examples": len(scored),
        "risk_target_examples": total_targets,
        "risk_probability_coverage": len(scored) / total_targets if total_targets else None,
    }


def _reliability_bins(
    scored: Sequence[tuple[float, int]],
    *,
    bin_count: int = 10,
) -> list[dict[str, Any]]:
    bins = []
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        members = [
            (probability, label)
            for probability, label in scored
            if lower <= probability < upper or (bin_index == bin_count - 1 and probability == 1.0)
        ]
        bins.append(
            {
                "lower_bound": lower,
                "upper_bound": upper,
                "count": len(members),
                "mean_predicted_probability": (
                    mean(probability for probability, _ in members) if members else None
                ),
                "observed_event_rate": mean(label for _, label in members) if members else None,
            }
        )
    return bins


def _aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        metadata = run["metadata"]["run"]
        grouped.setdefault((metadata["scenario"], metadata["label_delay_batches"]), []).append(run)

    result: dict[str, Any] = {}
    metric_names = (
        "alerts_generated",
        "false_alerts_per_100_clean_batches",
        "event_precision",
        "event_recall",
        "matched_event_count",
        "missed_event_count",
        "warning_lead_batches",
        "detection_lag_batches",
    )
    for (scenario, delay), episodes in sorted(grouped.items()):
        method_values: dict[str, dict[str, list[float]]] = {}
        method_names = [*_BASELINES, "risk_calibrator"]
        for name in method_names:
            method_values[name] = {metric: [] for metric in metric_names}
        episode_rows = []
        reliability_pairs = []
        for run in episodes:
            summary = run["summary"]
            brier = _brier_metrics(run)
            reliability_pairs.extend(_risk_probability_targets(run))
            baseline_metrics = summary["baseline_methods"]
            for method in _BASELINES:
                metrics = baseline_metrics.get(method, {})
                for metric in metric_names:
                    value = metrics.get(metric)
                    if value is not None:
                        method_values[method][metric].append(float(value))
            risk_mapping = {
                "alerts_generated": "risk_pipeline_alerts_generated",
                "false_alerts_per_100_clean_batches": "risk_pipeline_false_alerts_per_100_clean_batches",
                "event_precision": "risk_pipeline_event_precision",
                "event_recall": "risk_pipeline_event_recall",
                "matched_event_count": "risk_pipeline_matched_event_count",
                "missed_event_count": "risk_pipeline_missed_event_count",
                "warning_lead_batches": "risk_pipeline_warning_lead_batches",
                "detection_lag_batches": "risk_pipeline_detection_lag_batches",
            }
            for metric, summary_key in risk_mapping.items():
                value = summary.get(summary_key)
                if value is not None:
                    method_values["risk_calibrator"][metric].append(float(value))
            episode_rows.append(
                {
                    "seed": run["metadata"]["run"]["seed"],
                    "event_present": summary["event_present"],
                    "event_onset_batch": summary["event_onset_batch"],
                    "baseline_methods": baseline_metrics,
                    "risk_pipeline": {
                        key: summary.get(value)
                        for key, value in risk_mapping.items()
                    },
                    "risk_calibration": brier,
                    "runtime_seconds": summary["runtime_seconds"],
                    "runtime_seconds_per_batch": summary["runtime_seconds_per_batch"],
                }
            )

        methods = {}
        for method, metric_values in method_values.items():
            methods[method] = {
                metric: {
                    "mean": mean(values) if values else None,
                    "population_std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
                    "episode_count": len(values),
                }
                for metric, values in metric_values.items()
            }
        brier_values = [
            row["risk_calibration"]["brier_score"]
            for row in episode_rows
            if row["risk_calibration"]["brier_score"] is not None
        ]
        runtime_values = [row["runtime_seconds_per_batch"] for row in episode_rows]
        result.setdefault(scenario, {})[str(delay)] = {
            "episode_count": len(episodes),
            "methods": methods,
            "runtime_seconds_per_batch": {
                "mean": mean(runtime_values),
                "population_std": pstdev(runtime_values) if len(runtime_values) > 1 else 0.0,
            },
            "risk_brier_score": {
                "mean": mean(brier_values) if brier_values else None,
                "population_std": pstdev(brier_values) if len(brier_values) > 1 else (0.0 if brier_values else None),
                "episode_count": len(brier_values),
            },
            "risk_reliability_bins": _reliability_bins(reliability_pairs),
            "episodes": episode_rows,
        }
    return result


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    print("Fitting initial calibrator on training seeds at label delay 0.", flush=True)
    training_runs = _run_split(
        split_name="train",
        seeds=args.train_seeds,
        scenarios=args.scenarios,
        delays=(0,),
        args=args,
    )
    calibrator = fit_initial_risk_calibrator(training_runs)
    if not calibrator.is_fitted:
        raise RuntimeError(
            "training targets did not meet the calibrator's minimum sample and class counts; "
            f"status={calibrator.status}"
        )
    calibrator_path = args.output_directory / "initial_risk_calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)

    validation_runs = _run_split(
        split_name="validation",
        seeds=args.validation_seeds,
        scenarios=args.scenarios,
        delays=args.label_delays,
        args=args,
        calibrator=calibrator,
    )
    baseline_validation_runs = [
        run for run in validation_runs
        if run["metadata"]["run"]["label_delay_batches"] == args.label_delays[0]
    ]
    selected_baselines = {
        method: select_warning_threshold(
            validation_episodes_from_runs(
                baseline_validation_runs,
                score_name=method,
            )
        )
        for method in _BASELINES
    }
    risk_thresholds = {}
    risk_selection_metrics = {}
    for delay in args.label_delays:
        delay_runs = [
            run for run in validation_runs
            if run["metadata"]["run"]["label_delay_batches"] == delay
        ]
        selected = select_warning_threshold(
            validation_episodes_from_runs(delay_runs, score_name="risk_probability")
        )
        risk_thresholds[delay] = float(selected["threshold"])
        risk_selection_metrics[delay] = selected

    baseline_thresholds = {
        name: float(selected["threshold"])
        for name, selected in selected_baselines.items()
    }
    test_runs = _run_split(
        split_name="test",
        seeds=args.test_seeds,
        scenarios=args.scenarios,
        delays=args.label_delays,
        args=args,
        calibrator=calibrator,
        risk_thresholds=risk_thresholds,
        fallback_threshold=baseline_thresholds["fixed_weighted_score"],
        baseline_thresholds=baseline_thresholds,
    )

    summary = {
        "experiment": "label_delay_early_warning",
        "protocol": _CONFIG.protocol.to_dict(),
        "classifier": _CONFIG.classifier.to_dict(),
        "scenarios": args.scenarios,
        "label_delays_batches": args.label_delays,
        "episode_config": {
            "stream_batches": args.stream_batches,
            "batch_size": args.batch_size,
            "reference_size": args.reference_size,
            "detector_reference_rows": args.detector_reference_rows,
        },
        "seed_groups": {
            "train": args.train_seeds,
            "validation": args.validation_seeds,
            "test": args.test_seeds,
        },
        "training_calibrator_status": calibrator.status,
        "initial_calibrator_artifact": str(calibrator_path),
        "online_update_policy": (
            "reset to the training-fitted calibrator for each episode; update only from "
            "that episode's risk targets after protocol maturity"
        ),
        "threshold_selection": {
            "split": "validation",
            "false_alert_budget_per_100_clean_batches": (
                _CONFIG.protocol.false_alert_budget_per_100_clean_batches
            ),
            "risk_by_delay": {str(delay): value for delay, value in risk_thresholds.items()},
            "risk_selection_metrics_by_delay": {
                str(delay): value for delay, value in risk_selection_metrics.items()
            },
            "baselines": selected_baselines,
        },
        "test_aggregate": _aggregate_runs(test_runs),
        "output_directory": str(args.output_directory),
    }
    output_path = args.output_directory / "aggregate_results.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved held-out aggregate results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
