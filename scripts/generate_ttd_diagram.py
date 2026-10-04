"""
Script to generate the mathematically accurate Time-to-Degradation (TTD)
Forecasting diagram for the B.Tech Major Project report and presentation slides.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from dews.forecasting import DriftForecaster


def generate_ttd_forecasting_diagram(output_paths: list[Path]):
    # Setup data
    # Batches 1 to 8: clean historical baseline transitioning into drift
    hist_batches = np.arange(1, 9, dtype=float)
    hist_scores = [0.09, 0.08, 0.10, 0.11, 0.12, 0.15, 0.18, 0.22]
    ds_crit = 0.30
    horizon = 6

    # Run DEWS DriftForecaster
    forecaster = DriftForecaster(horizon=horizon)
    result = forecaster.forecast(hist_scores, ds_crit=ds_crit)

    forecast_batches = np.arange(8, 8 + horizon + 1, dtype=float)
    # prepend current score so forecast trajectory connects smoothly
    forecast_trajectory = [hist_scores[-1]] + result.forecast
    lower_bound = [hist_scores[-1]] + result.lower_bound
    upper_bound = [hist_scores[-1]] + result.upper_bound

    ttd = result.ttd_batches  # ~2.38 batches
    crossing_batch = 8.0 + ttd  # ~10.38 batch

    # Configure plotting style
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#94A3B8"
    plt.rcParams["axes.linewidth"] = 1.0

    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Grid
    ax.grid(True, linestyle="--", alpha=0.55, color="#E2E8F0", zorder=0)

    # Shaded regions for status
    ax.axhspan(0.0, ds_crit, facecolor="#F8FAFC", alpha=0.6, zorder=0)
    ax.axhspan(ds_crit, 0.60, facecolor="#FEF2F2", alpha=0.45, zorder=0)

    # Critical Threshold Line
    ax.axhline(
        y=ds_crit,
        color="#059669",
        linestyle="--",
        linewidth=2.0,
        zorder=3,
        label=rf"Critical Threshold ($DS_{{crit}} = {ds_crit:.2f}$)",
    )

    # Confidence Interval Ribbon
    ax.fill_between(
        forecast_batches,
        lower_bound,
        upper_bound,
        color="#EF4444",
        alpha=0.15,
        zorder=2,
        label=r"95% Confidence Interval ($\pm 1.96\sigma_{res}$)",
    )
    ax.plot(forecast_batches, lower_bound, color="#F87171", linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
    ax.plot(forecast_batches, upper_bound, color="#F87171", linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)

    # Forecast Trajectory
    ax.plot(
        forecast_batches,
        forecast_trajectory,
        color="#DC2626",
        linestyle="--",
        linewidth=2.4,
        marker="o",
        markersize=7.5,
        markerfacecolor="#EF4444",
        markeredgecolor="#B91C1C",
        markeredgewidth=1.5,
        zorder=5,
        label=r"Forecasted Trajectory ($\hat{y}_{t+h}$)",
    )

    # Historical Observations
    ax.plot(
        hist_batches,
        hist_scores,
        color="#1E40AF",
        linestyle="-",
        linewidth=2.4,
        marker="o",
        markersize=7.5,
        markerfacecolor="#3B82F6",
        markeredgecolor="#1D4ED8",
        markeredgewidth=1.5,
        zorder=6,
        label=r"Historical Drift Score ($DS_t$)",
    )

    # Highlight current batch point
    ax.scatter(
        [8.0],
        [hist_scores[-1]],
        s=140,
        facecolors="none",
        edgecolors="#1E3A8A",
        linewidths=2.5,
        zorder=7,
    )

    # Vertical split at current batch (t=8)
    ax.axvline(x=8.0, color="#64748B", linestyle="--", linewidth=1.4, alpha=0.85, zorder=3)
    ax.text(
        8.0,
        0.53,
        "Current Batch\n($t = 8$)",
        color="#334155",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.0),
        zorder=8,
    )

    # Breach Point Marker
    ax.scatter(
        [crossing_batch],
        [ds_crit],
        s=180,
        color="#D97706",
        marker="*",
        edgecolors="#78350F",
        linewidths=1.2,
        zorder=8,
        label=f"Projected Threshold Breach (Batch {crossing_batch:.1f})",
    )

    # Vertical drop from breach point
    ax.vlines(
        x=crossing_batch,
        ymin=0.08,
        ymax=ds_crit,
        color="#D97706",
        linestyle=":",
        linewidth=1.5,
        alpha=0.85,
        zorder=4,
    )

    # Annotation Callout for Breach Point
    ax.annotate(
        f"Projected Breach Point\nBatch {crossing_batch:.1f} ($DS = {ds_crit:.2f}$)",
        xy=(crossing_batch, ds_crit),
        xytext=(crossing_batch + 0.6, ds_crit - 0.08),
        fontsize=9.5,
        fontweight="bold",
        color="#78350F",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=1.0),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#B45309", lw=1.5),
        zorder=9,
    )

    # Critical Threshold Label on Right
    ax.text(
        14.2,
        ds_crit,
        rf"$DS_{{crit}} = {ds_crit:.2f}$" + "\n(Critical)",
        color="#047857",
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    # TTD Arrow at y = 0.08
    arrow_y = 0.08
    ax.annotate(
        "",
        xy=(crossing_batch, arrow_y),
        xytext=(8.0, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=2.2, mutation_scale=14),
        zorder=8,
    )
    # TTD Label above the arrow
    ax.text(
        (8.0 + crossing_batch) / 2.0,
        arrow_y + 0.02,
        f"TTD = {ttd:.1f} Batches\n(Advance Early Warning)",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#DC2626",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=1.0),
        zorder=9,
    )

    # Historical Observation Span at bottom
    ax.annotate(
        "",
        xy=(8.0, 0.015),
        xytext=(1.0, 0.015),
        arrowprops=dict(arrowstyle="<->", color="#1E40AF", lw=1.6, mutation_scale=10),
        zorder=8,
    )
    ax.text(
        4.5,
        0.025,
        r"Historical Window ($W_{live}$, Batches 1 to 8)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#1E40AF",
    )

    # Forecast Horizon Span at bottom
    ax.annotate(
        "",
        xy=(14.0, 0.015),
        xytext=(8.0, 0.015),
        arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.6, mutation_scale=10),
        zorder=8,
    )
    ax.text(
        11.0,
        0.025,
        f"Forecast Horizon ($h = {horizon}$ Batches)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#475569",
    )

    # Safe Zone vs Drift Zone text in background
    ax.text(
        1.5,
        0.26,
        r"Normal Operation Zone ($DS < DS_{crit}$)",
        fontsize=9,
        color="#64748B",
        fontstyle="italic",
        alpha=0.8,
    )
    ax.text(
        1.5,
        0.33,
        r"Critical Degradation Zone ($DS \geq DS_{crit}$)",
        fontsize=9,
        color="#DC2626",
        fontstyle="italic",
        alpha=0.75,
    )

    # Axis limits & ticks
    ax.set_xlim(0.5, 14.8)
    ax.set_ylim(0.0, 0.60)
    ax.set_xticks(np.arange(1, 15, 1))
    ax.set_yticks(np.linspace(0.0, 0.60, 7))

    ax.set_xlabel("Production Streaming Batches", fontsize=11, fontweight="bold", labelpad=10, color="#1E293B")
    ax.set_ylabel(r"Composite Drift Score ($DS$)", fontsize=11, fontweight="bold", labelpad=10, color="#1E293B")
    ax.tick_params(colors="#334155", labelsize=10)

    # Title & Subtitle - Clean separation
    ax.set_title(
        "Time-to-Degradation (TTD) Forecasting",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        pad=28,
    )
    fig.text(
        0.51,
        0.925,
        "Holt-Winters Additive Damped-Trend Extrapolation with 95% Confidence Bounds",
        fontsize=10.5,
        color="#64748B",
        ha="center",
    )

    # Legend
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.5,
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(1.0)

    # Spines styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#94A3B8")

    plt.tight_layout()

    # Save to all target paths
    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")

    plt.close()


if __name__ == "__main__":
    targets = [
        project_root / "BTech Report" / "assets" / "ttl_forecasting.png",
        project_root / "BTech Report" / "assets" / "ttd_forecasting.png",
        project_root / "docs" / "assets" / "ttl_forecasting.png",
        project_root / "docs" / "assets" / "ttd_forecasting.png",
    ]
    generate_ttd_forecasting_diagram(targets)
