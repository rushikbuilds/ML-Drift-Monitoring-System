"""Plot the warning timeline from one saved early-warning episode ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot risk, fixed score, delayed accuracy, and event timing for one episode."
    )
    parser.add_argument("episode_json", type=Path, help="Path to an episode.json ledger.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Image path (default: next to the episode ledger with a .png suffix).",
    )
    return parser.parse_args()


def _number_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.episode_json.read_text(encoding="utf-8"))
    records = payload.get("batch_records")
    if not records:
        raise ValueError("episode ledger does not contain batch_records")

    import matplotlib.pyplot as plt

    batches = [int(record["batch_index"]) for record in records]
    risk_probability = [_number_or_nan(record.get("risk_probability")) for record in records]
    fixed_score = [
        _number_or_nan(record.get("baseline_scores", {}).get("fixed_weighted_score"))
        for record in records
    ]
    active_threshold = [_number_or_nan(record.get("active_warning_threshold")) for record in records]
    active_alert_batches = [
        int(record["batch_index"])
        for record in records
        if record.get("risk_alarm") is True
    ]
    records_by_batch = {int(record["batch_index"]): record for record in records}
    active_alert_scores = [
        float(records_by_batch[batch]["active_warning_score"])
        for batch in active_alert_batches
    ]
    fixed_alert_batches = [
        int(record["batch_index"])
        for record in records
        if record.get("baseline_alarms", {}).get("fixed_weighted_score") is True
    ]
    fixed_alert_scores = [
        float(records_by_batch[batch]["baseline_scores"]["fixed_weighted_score"])
        for batch in fixed_alert_batches
    ]

    released_batch_indices = []
    released_accuracies = []
    for record in records:
        evaluation = record.get("evaluation_only", {})
        accuracy = evaluation.get("realized_accuracy")
        released_at = evaluation.get("accuracy_released_at_batch")
        if accuracy is not None and released_at is not None:
            released_batch_indices.append(int(released_at))
            released_accuracies.append(float(accuracy))

    metadata = payload.get("metadata", {})
    run_metadata = metadata.get("run", {})
    summary = payload.get("summary", {})
    event_onset = summary.get("event_onset_batch")
    delay = int(run_metadata.get("label_delay_batches", payload.get("label_delay_batches", 0)))
    scenario = run_metadata.get("scenario", "unknown")
    seed = run_metadata.get("seed", "unknown")
    threshold = metadata.get("baseline_thresholds", {}).get("fixed_weighted_score")

    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.plot(batches, risk_probability, label="Risk probability", color="#6f42c1", linewidth=2)
    axis.plot(batches, fixed_score, label="Fixed weighted score", color="#2878b5", linewidth=1.8)
    axis.plot(batches, active_threshold, label="Active warning threshold", color="#444444", linestyle="--")
    if threshold is not None:
        axis.axhline(
            float(threshold),
            color="#2878b5",
            linestyle=":",
            linewidth=1.2,
            label="Fixed-score threshold",
        )
    if active_alert_batches:
        axis.scatter(
            active_alert_batches,
            active_alert_scores,
            marker="x",
            s=60,
            linewidth=2,
            color="#d62728",
            label="Active warning",
            zorder=5,
        )
    if fixed_alert_batches:
        axis.scatter(
            fixed_alert_batches,
            fixed_alert_scores,
            marker="o",
            s=28,
            facecolors="none",
            edgecolors="#2878b5",
            label="Fixed-score baseline alert",
            zorder=5,
        )
    if released_batch_indices:
        axis.plot(
            released_batch_indices,
            released_accuracies,
            marker="o",
            markersize=3.5,
            color="#2ca02c",
            linewidth=1.5,
            label=f"Accuracy when labels arrive (L={delay})",
        )
    if event_onset is not None:
        axis.axvline(
            int(event_onset),
            color="#d62728",
            linestyle="-.",
            linewidth=1.5,
            label=f"Measured event onset: batch {event_onset}",
        )

    axis.set_title(f"{scenario} | seed {seed} | label delay L={delay}")
    axis.set_xlabel("Batch clock (accuracy points appear at label-release time)")
    axis.set_ylabel("Score, risk probability, or accuracy")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlim(0, max(batches + released_batch_indices) if released_batch_indices else max(batches))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)

    output_path = args.output or args.episode_json.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved episode plot to {output_path}")


if __name__ == "__main__":
    main()
