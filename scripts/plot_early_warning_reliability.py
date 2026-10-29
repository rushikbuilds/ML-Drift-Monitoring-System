"""Plot the held-out risk-probability reliability curve for a scenario and delay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_json", type=Path, help="aggregate_results.json from the experiment runner")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--delay", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.aggregate_json.read_text(encoding="utf-8"))
    try:
        group = payload["test_aggregate"][args.scenario][str(args.delay)]
    except KeyError as exc:
        raise ValueError("requested scenario/delay is absent from the aggregate results") from exc
    bins = [item for item in group["risk_reliability_bins"] if item["count"]]
    if not bins:
        raise ValueError("this scenario/delay has no held-out risk probabilities to plot")

    import matplotlib.pyplot as plt

    predicted = [item["mean_predicted_probability"] for item in bins]
    observed = [item["observed_event_rate"] for item in bins]
    counts = [item["count"] for item in bins]
    figure, axis = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    axis.plot(predicted, observed, marker="o", color="#6f42c1", label="Held-out risk model")
    for x_value, y_value, count in zip(predicted, observed, counts):
        axis.annotate(str(count), (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    brier = group["risk_brier_score"]["mean"]
    axis.set_title(f"Risk reliability: {args.scenario}, L={args.delay}, Brier={brier}")
    axis.set_xlabel("Mean predicted event probability")
    axis.set_ylabel("Observed event frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")

    output_path = args.output or args.aggregate_json.with_name(
        f"reliability_{args.scenario}_delay-{args.delay}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved reliability plot to {output_path}")


if __name__ == "__main__":
    main()
