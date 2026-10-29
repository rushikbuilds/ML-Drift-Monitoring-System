"""Run the paired held-out harmful and harmless early-warning demonstration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dews import build_scenario_episode, run_early_warning_episode
from dews.experiment_config import DEFAULT_EARLY_WARNING_CONFIG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one held-out harmful and harmless episode with frozen validation choices."
    )
    parser.add_argument(
        "--experiment-directory",
        type=Path,
        default=DEFAULT_EARLY_WARNING_CONFIG.output_directory,
        help="Directory containing aggregate_results.json and initial_risk_calibrator.joblib.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EARLY_WARNING_CONFIG.seed_split.test_seeds[0],
        help="Held-out test seed used for both matched scenarios.",
    )
    return parser.parse_args()


def _demo_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    return {
        "scenario": run["metadata"]["run"]["scenario"],
        "seed": run["metadata"]["run"]["seed"],
        "label_delay_batches": run["metadata"]["run"]["label_delay_batches"],
        "event_present": summary["event_present"],
        "event_onset_batch": summary["event_onset_batch"],
        "risk_pipeline_alerts": summary["risk_pipeline_alerts"],
        "risk_pipeline_event_recall": summary["risk_pipeline_event_recall"],
        "risk_pipeline_warning_lead_batches": summary["risk_pipeline_warning_lead_batches"],
        "risk_pipeline_detection_lag_batches": summary["risk_pipeline_detection_lag_batches"],
        "fixed_score_alerts": summary["baseline_methods"].get("fixed_weighted_score", {}).get("alert_batches", []),
        "fixed_score_event_recall": summary["baseline_methods"].get("fixed_weighted_score", {}).get("event_recall"),
        "runtime_seconds": summary["runtime_seconds"],
        "episode_ledger": run["output_path"],
    }


def main() -> None:
    args = _parse_args()
    config = DEFAULT_EARLY_WARNING_CONFIG
    if config.seed_split.split_for_seed(args.seed) != "test":
        raise ValueError(f"demo seed {args.seed} must belong to the held-out test split")

    experiment_directory = args.experiment_directory
    aggregate_path = experiment_directory / "aggregate_results.json"
    calibrator_path = experiment_directory / "initial_risk_calibrator.joblib"
    if not aggregate_path.exists() or not calibrator_path.exists():
        raise FileNotFoundError(
            "run scripts/run_early_warning_experiment.py first to create the fitted model "
            "and validation thresholds"
        )

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    delay = 2
    risk_threshold = aggregate["threshold_selection"]["risk_by_delay"].get(str(delay))
    baseline_selection = aggregate["threshold_selection"]["baselines"]
    if risk_threshold is None:
        raise ValueError("the experiment results do not contain a validation threshold for delay 2")
    baseline_thresholds = {
        name: float(selection["threshold"])
        for name, selection in baseline_selection.items()
    }

    episode_config = aggregate.get("episode_config", {})
    reference_size = int(episode_config.get("reference_size", 2048))
    stream_batches = int(episode_config.get("stream_batches", 30))
    batch_size = int(episode_config.get("batch_size", 128))
    detector_reference_rows = int(episode_config.get("detector_reference_rows", 512))
    calibrator = joblib.load(calibrator_path)
    demo_directory = experiment_directory / "demo"
    demo_directory.mkdir(parents=True, exist_ok=True)

    plot_script = PROJECT_ROOT / "scripts" / "plot_early_warning_episode.py"
    demo_results = []
    for scenario in ("harmful_gradual", "harmless_shift"):
        episode = build_scenario_episode(
            scenario,
            seed=args.seed,
            reference_size=reference_size,
            stream_batches=stream_batches,
            batch_size=batch_size,
        )
        active_calibrator = deepcopy(calibrator)
        run = run_early_warning_episode(
            episode,
            label_delay_batches=delay,
            detector_reference_rows=detector_reference_rows,
            risk_predictor=active_calibrator.predict,
            risk_threshold=float(risk_threshold),
            fallback_threshold=baseline_thresholds["fixed_weighted_score"],
            baseline_thresholds=baseline_thresholds,
            risk_target_updater=active_calibrator.update,
            output_directory=demo_directory,
        )
        plot_path = Path(run["output_path"]).with_suffix(".png")
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                str(run["output_path"]),
                "--output",
                str(plot_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        demo_results.append(_demo_summary(run))

    summary = {
        "demo": "paired_held_out_harmful_and_harmless",
        "seed": args.seed,
        "label_delay_batches": delay,
        "risk_threshold": risk_threshold,
        "baseline_thresholds": baseline_thresholds,
        "calibrator_artifact": str(calibrator_path),
        "episodes": demo_results,
    }
    summary_path = demo_directory / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved demo summary to {summary_path}")
    for result in demo_results:
        print(
            f"{result['scenario']}: event={result['event_present']} "
            f"onset={result['event_onset_batch']} "
            f"risk_alerts={len(result['risk_pipeline_alerts'])} "
            f"fixed_score_alerts={len(result['fixed_score_alerts'])}"
        )


if __name__ == "__main__":
    main()
