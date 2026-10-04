from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Visualize DEWS demo results with publication-ready charts.

Runs the synthetic end-to-end demo and produces four charts saved to
artifacts/charts/:

1. Drift Score Timeline - fused score with component signals and threshold
2. Forecast vs Actual - predicted trajectory vs observed drift
3. Per-Feature Drift Heatmap - which features drifted and when
4. Alert Timeline - when alerts fired relative to drift onset and threshold breach

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --adaptive-threshold
    python scripts/visualize_results.py --alert-confirmation 1
    python scripts/visualize_results.py --output-dir artifacts/charts
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from dews import MonitorConfig, build_demo_system, run_demo_evaluation


# -- Colour palette (dark, premium feel) --------------------------------
PALETTE = {
    "bg": "#ffffff",
    "panel": "#f8f9fa",
    "grid": "#e2e8f0",
    "text": "#1e293b",
    "accent": "#4f46e5",       # indigo-600
    "accent2": "#7c3aed",      # violet-600
    "stat": "#0891b2",         # cyan-600
    "emb": "#d97706",          # amber-600
    "conf": "#db2777",         # pink-600
    "fused": "#4f46e5",        # indigo-600
    "threshold": "#dc2626",    # red-600
    "drift_zone": "#ef444422",
    "forecast": "#0284c7",     # light blue-600
    "forecast_ci": "#0284c718",
    "alert_critical": "#dc2626",
    "alert_warning": "#d97706",
    "alert_info": "#2563eb",
    "accuracy": "#059669",     # emerald-600
    "heatmap_low": "#f8f9fa",
    "heatmap_high": "#4f46e5",
}


def _apply_style(ax: plt.Axes) -> None:
    """Apply dark theme to an axis."""
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], alpha=0.4, linewidth=0.5)


def run_demo(
    adaptive: bool = False,
    batches: int = 18,
    batch_size: int = 80,
    window_size: int = 160,
    forecast_horizon: int = 8,
    adaptive_sensitivity: float = 2.5,
    ds_crit: float = 0.30,
    alert_confirmation: int = 2,
) -> tuple:
    """Run the demo and collect all results."""
    config = MonitorConfig(
        window_size=window_size,
        batch_size=batch_size,
        forecast_horizon=forecast_horizon,
        ds_crit=ds_crit,
        alert_confirmation_batches=alert_confirmation,
        adaptive_threshold_enabled=adaptive,
        adaptive_sensitivity=adaptive_sensitivity,
        retraining_enabled=False,
    )
    system, scenario = build_demo_system(config=config, store_path=":memory:")
    summary = run_demo_evaluation(system=system, scenario=scenario, max_batches=batches)
    return system, scenario, summary


def chart_drift_timeline(system, scenario, output_dir: Path) -> Path:
    """Chart 1: Drift Score Timeline with component signals."""
    results = system.batch_results
    batches = [r.batch_index for r in results]
    ds = [r.drift_score for r in results]
    s_stat = [r.statistical.get("s_stat", 0) for r in results]
    s_emb = [r.embedding.get("s_emb", 0) for r in results]
    s_conf = [r.confidence.get("s_conf", 0) for r in results]
    accuracies = [r.accuracy for r in results]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[3, 1],
        sharex=True, facecolor=PALETTE["bg"],
    )

    # -- Top: drift score + components --
    _apply_style(ax1)
    ax1.fill_between(batches, 0, s_stat, alpha=0.25, color=PALETTE["stat"], label="S_stat")
    ax1.fill_between(batches, 0, s_emb, alpha=0.25, color=PALETTE["emb"], label="S_emb")
    ax1.fill_between(batches, 0, s_conf, alpha=0.25, color=PALETTE["conf"], label="S_conf")
    ax1.plot(batches, ds, color=PALETTE["fused"], linewidth=2.5, label="Fused Drift Score", zorder=5)

    # Threshold line
    threshold = system.config.ds_crit
    ax1.axhline(y=threshold, color=PALETTE["threshold"], linestyle="--",
                linewidth=1.5, alpha=0.8, label=f"Threshold ({threshold})")

    # Adaptive threshold if available
    if system.adaptive_threshold is not None:
        at_values = [r.adaptive_threshold for r in results if r.adaptive_threshold is not None]
        at_batches = [r.batch_index for r in results if r.adaptive_threshold is not None]
        if at_values:
            ax1.plot(at_batches, at_values, color="#a78bfa", linestyle="-.",
                     linewidth=1.5, alpha=0.8, label="Adaptive Threshold")

    # Drift injection zone
    drift_start = scenario.drift_start_batch
    ax1.axvspan(drift_start, max(batches), alpha=0.08, color=PALETTE["threshold"])
    ax1.axvline(x=drift_start, color=PALETTE["threshold"], linestyle=":",
                linewidth=1, alpha=0.6)
    ax1.text(drift_start + 0.2, ax1.get_ylim()[1] * 0.92, "Drift Injected ->",
             fontsize=8, color=PALETTE["threshold"], alpha=0.8, fontweight="bold")

    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Drift Score Timeline - Component Signal Decomposition",
                   fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3,
               facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"])
    ax1.set_ylim(0, max(max(ds) * 1.15, 0.5))

    # -- Bottom: accuracy --
    _apply_style(ax2)
    valid_acc = [(b, a) for b, a in zip(batches, accuracies) if a is not None]
    if valid_acc:
        acc_b, acc_v = zip(*valid_acc)
        ax2.plot(acc_b, acc_v, color=PALETTE["accuracy"], linewidth=2, marker="o",
                 markersize=3, label="Batch Accuracy")
        ax2.axvspan(drift_start, max(batches), alpha=0.08, color=PALETTE["threshold"])
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_xlabel("Batch Index", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.3,
               facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"])

    plt.tight_layout()
    path = output_dir / "drift_score_timeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  + {path}")
    return path


def chart_forecast(system, scenario, output_dir: Path) -> Path:
    """Chart 2: Forecast vs Actual drift trajectory."""
    results = system.batch_results
    store = system.store
    batches = [r.batch_index for r in results]
    ds = [r.drift_score for r in results]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    _apply_style(ax)

    ax.plot(batches, ds, color=PALETTE["fused"], linewidth=2, label="Observed Drift Score",
            marker="o", markersize=4, zorder=5)

    # Overlay forecasts: find early warning forecast where TTD is positive
    forecasts = store.load_forecasts()
    if forecasts:
        ew_idx = None
        for i, f in enumerate(forecasts):
            fc_json = f["forecast_json"]
            f_data = json.loads(fc_json) if isinstance(fc_json, str) else fc_json
            ttd = f_data.get("ttd_batches")
            if ttd is not None and ttd > 0:
                ew_idx = i
                break
        if ew_idx is None:
            ew_idx = len(forecasts) // 2

        fc_data = json.loads(forecasts[ew_idx]["forecast_json"]) if isinstance(forecasts[ew_idx]["forecast_json"], str) else forecasts[ew_idx]["forecast_json"]
        fc_batch = forecasts[ew_idx]["batch_index"]
        horizon = len(fc_data["forecast"])
        fc_x = list(range(fc_batch + 1, fc_batch + 1 + horizon))
        ax.plot(fc_x, fc_data["forecast"], color=PALETTE["forecast"],
                linewidth=2.2, linestyle="--", label=f"Forecast (batch {fc_batch})",
                zorder=4)
        ax.fill_between(fc_x, fc_data["lower_bound"], fc_data["upper_bound"],
                        alpha=0.15, color=PALETTE["forecast"], label="95% CI")

        ttd = fc_data.get("ttd_batches")
        if ttd is not None and ttd > 0:
            ds_at_batch = ds[fc_batch] if fc_batch < len(ds) else ds[-1]
            ax.annotate(f"Early Warning: TTD ~ {ttd:.1f} batches",
                        xy=(fc_batch, ds_at_batch),
                        xytext=(max(0, fc_batch - 3), ds_at_batch + 0.12),
                        fontsize=9, color=PALETTE["forecast"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=PALETTE["forecast"], lw=1.5))

        # Also show terminal forecast
        last_fc = json.loads(forecasts[-1]["forecast_json"]) if isinstance(forecasts[-1]["forecast_json"], str) else forecasts[-1]["forecast_json"]
        last_batch = forecasts[-1]["batch_index"]
        if last_batch != fc_batch:
            last_x = list(range(last_batch + 1, last_batch + 1 + len(last_fc["forecast"])))
            ax.plot(last_x, last_fc["forecast"], color="#a78bfa",
                    linewidth=1.5, linestyle="--", alpha=0.7,
                    label=f"Forecast (batch {last_batch})")
            ax.fill_between(last_x, last_fc["lower_bound"], last_fc["upper_bound"],
                            alpha=0.08, color="#a78bfa")

    # Threshold
    ax.axhline(y=system.config.ds_crit, color=PALETTE["threshold"], linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Critical Threshold ({system.config.ds_crit})")

    # Drift zone
    ax.axvspan(scenario.drift_start_batch, max(batches) + 8, alpha=0.06,
               color=PALETTE["threshold"])

    ax.set_xlabel("Batch Index", fontsize=10)
    ax.set_ylabel("Drift Score", fontsize=10)
    ax.set_title("Drift Forecast vs Observed - Time-to-Degradation Estimation",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.set_ylim(0, max(max(ds) * 1.2, 0.5))

    plt.tight_layout()
    path = output_dir / "forecast_vs_actual.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  + {path}")
    return path


def chart_feature_heatmap(system, scenario, output_dir: Path) -> Path:
    """Chart 3: Per-feature drift heatmap (batch x feature)."""
    results = system.batch_results
    features = scenario.feature_schema.continuous + scenario.feature_schema.categorical

    # Build the heatmap matrix
    matrix = np.zeros((len(results), len(features)))
    for i, r in enumerate(results):
        per_feature = r.statistical.get("per_feature", [])
        for pf in per_feature:
            if pf["feature"] in features:
                j = features.index(pf["feature"])
                matrix[i, j] = pf["score"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    _apply_style(ax)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dews", [PALETTE["heatmap_low"], "#312e81", PALETTE["heatmap_high"], "#c4b5fd"])

    im = ax.imshow(matrix.T, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=max(matrix.max(), 0.5))
    ax.set_xlabel("Batch Index", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r.batch_index for r in results], fontsize=8)

    # Drift start marker
    drift_start = scenario.drift_start_batch
    ax.axvline(x=drift_start - 0.5, color=PALETTE["threshold"], linestyle="--",
               linewidth=1.5, alpha=0.8)
    ax.text(drift_start, -0.8, "Drift ->", fontsize=8, color=PALETTE["threshold"],
            fontweight="bold", ha="left")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE["text"], labelsize=8)
    cbar.set_label("Feature Drift Score", color=PALETTE["text"], fontsize=9)

    ax.set_title("Per-Feature Drift Heatmap - Batch x Feature",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    path = output_dir / "feature_drift_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  + {path}")
    return path


def chart_alert_timeline(system, scenario, output_dir: Path) -> Path:
    """Chart 4: Alert timeline relative to drift onset and threshold breach."""
    results = system.batch_results
    alerts = system.store.load_alerts()
    batches = [r.batch_index for r in results]
    ds = [r.drift_score for r in results]

    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=PALETTE["bg"])
    _apply_style(ax)

    # Drift score as backdrop
    ax.fill_between(batches, 0, ds, alpha=0.15, color=PALETTE["fused"])
    ax.plot(batches, ds, color=PALETTE["fused"], linewidth=1.5, alpha=0.6, label="Drift Score")

    # Threshold line
    threshold = system.config.ds_crit
    ax.axhline(y=threshold, color=PALETTE["threshold"], linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Critical Threshold ({threshold})")

    # Drift zone
    drift_start = scenario.drift_start_batch
    ax.axvspan(drift_start, max(batches), alpha=0.06, color=PALETTE["threshold"])
    ax.axvline(x=drift_start, color=PALETTE["threshold"], linestyle=":",
               linewidth=1, alpha=0.6)
    ax.text(drift_start + 0.1, threshold * 0.92, "Drift Injected ->",
            fontsize=8, color=PALETTE["threshold"], alpha=0.8, fontweight="bold")

    # Alert markers
    severity_colors = {
        "Critical": PALETTE["alert_critical"],
        "Warning": PALETTE["alert_warning"],
        "Info": PALETTE["alert_info"],
    }
    for alert in alerts:
        batch = alert["batch_index"]
        severity = alert["severity"]
        color = severity_colors.get(severity, PALETTE["alert_info"])
        ds_val = next((r.drift_score for r in results if r.batch_index == batch), 0)
        ax.scatter(batch, ds_val, color=color, s=130, zorder=6, edgecolors="white",
                   linewidths=1.2, marker="v")
        ax.annotate(severity[0], xy=(batch, ds_val), fontsize=7, fontweight="bold",
                    color="white", ha="center", va="center",
                    xytext=(0, 18), textcoords="offset points")

    # Critical threshold breach batch
    breach_batch = next((i for i, r in enumerate(results) if r.drift_score >= threshold), None)
    if breach_batch is not None:
        ax.axvline(x=breach_batch, color=PALETTE["threshold"], linestyle="-.",
                   linewidth=1.2, alpha=0.7)
        ax.text(breach_batch + 0.1, threshold * 1.05, f"Breach (batch {breach_batch})",
                fontsize=8, color=PALETTE["threshold"], fontweight="bold")

    # Detection lag annotation
    post_drift_alerts = [a["batch_index"] for a in alerts if a["batch_index"] >= drift_start]
    first_alert_batch = post_drift_alerts[0] if post_drift_alerts else None

    y_mid = threshold * 0.45
    if first_alert_batch is not None:
        lag = first_alert_batch - drift_start
        mid = (drift_start + first_alert_batch) / 2
        ax.annotate(
            f"Detection Lag = {lag:.0f} batches",
            xy=(mid, y_mid),
            fontsize=9, color=PALETTE["text"], ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["panel"],
                      edgecolor=PALETTE["accent"], alpha=0.9),
        )
        ax.annotate("", xy=(first_alert_batch, y_mid),
                     xytext=(drift_start, y_mid),
                     arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.5))

    # Early warning lead annotation (before critical threshold breach!)
    if first_alert_batch is not None and breach_batch is not None and first_alert_batch < breach_batch:
        lead = breach_batch - first_alert_batch
        y_lead = threshold * 0.78
        mid_lead = (first_alert_batch + breach_batch) / 2
        ax.annotate(
            f"Warning Lead = {lead:.0f} batches",
            xy=(mid_lead, y_lead),
            fontsize=9, color="#059669", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["panel"],
                      edgecolor="#059669", alpha=0.9),
        )
        ax.annotate("", xy=(breach_batch, y_lead),
                     xytext=(first_alert_batch, y_lead),
                     arrowprops=dict(arrowstyle="<->", color="#059669", lw=1.5))

    # Legend
    legend_handles = [
        mpatches.Patch(color=PALETTE["alert_critical"], label="Critical Alert"),
        mpatches.Patch(color=PALETTE["alert_warning"], label="Warning Alert"),
        mpatches.Patch(color=PALETTE["alert_info"], label="Info Alert"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.3,
              facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], loc="upper left")

    ax.set_xlabel("Batch Index", fontsize=10)
    ax.set_ylabel("Drift Score", fontsize=10)
    ax.set_title("Alert Timeline - Detection Lag & Early Warning Lead",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(max(ds) * 1.25, 0.5))

    plt.tight_layout()
    path = output_dir / "alert_timeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  + {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DEWS demo results")
    parser.add_argument("--adaptive-threshold", action="store_true",
                        help="Enable adaptive threshold in the demo")
    parser.add_argument("--batches", type=int, default=18,
                        help="Number of synthetic stream batches")
    parser.add_argument("--output-dir", type=str, default="artifacts/charts",
                        help="Directory for output charts")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=160)
    parser.add_argument("--forecast-horizon", type=int, default=8)
    parser.add_argument("--adaptive-sensitivity", type=float, default=2.5)
    parser.add_argument("--ds-crit", type=float, default=0.30,
                        help="Critical drift score threshold (default: 0.30)")
    parser.add_argument("--alert-confirmation", type=int, default=2,
                        help="Consecutive trigger batches required for alert (default: 2)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running DEWS demo...")
    system, scenario, summary = run_demo(
        adaptive=args.adaptive_threshold,
        batches=args.batches,
        batch_size=args.batch_size,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        adaptive_sensitivity=args.adaptive_sensitivity,
        ds_crit=args.ds_crit,
        alert_confirmation=args.alert_confirmation,
    )
    det_lag = summary.get("alert_detection_lag", summary.get("detection_lag"))
    warn_lead = summary.get("warning_lead")
    far_pct = (summary.get("false_alert_rate", 0.0) or 0.0) * 100.0

    print(f"  Processed {summary['processed_batches']} batches")
    print(f"  Drift injection at batch {scenario.drift_start_batch}")
    print(f"  Alerts generated: {summary['alerts_generated']} (Critical: {summary['critical_alerts']})")
    print(f"  Detection lag: {det_lag} batches")
    print(f"  Warning lead: {warn_lead if warn_lead is not None else '-'} batches")
    print(f"  False alarm rate: {far_pct:.1f}%")
    print()

    print("Generating charts...")
    chart_drift_timeline(system, scenario, output_dir)
    chart_forecast(system, scenario, output_dir)
    chart_feature_heatmap(system, scenario, output_dir)
    chart_alert_timeline(system, scenario, output_dir)

    print(f"\nAll charts saved to {output_dir.resolve()}/")

    # Also dump a JSON summary for the report
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
