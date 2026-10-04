from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Benchmark DEWS against baselines and competing tools.

Produces a clean comparison on the **synthetic demo scenario** (where we
control exactly when drift starts) and optionally includes Evidently / NannyML
if installed.  The output is a comparison table + chart saved to
``artifacts/charts/``.

Baselines compared
------------------
- **DEWS** - multi-signal fusion + forecasting + adaptive threshold
- **KS-only** - standalone Kolmogorov-Smirnov test (mean across features)
- **PSI-only** - standalone Population Stability Index (mean across features)
- **Evidently** - DataDriftPreset (if installed)
- **NannyML** - Data reconstruction drift or CBPE (if installed)

Usage::

    python visualize_benchmark.py
    python visualize_benchmark.py --output-dir artifacts/charts
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dews import MonitorConfig, build_demo_system, run_demo_evaluation
from dews.data import DemoScenario, build_demo_scenario
from dews.detection import StatisticalDriftDetector, population_stability_index, _safe_hist, jensen_shannon_divergence
from dews.config import MonitorConfig as _MC
from scipy import stats


PALETTE = {
    "bg": "#ffffff",
    "panel": "#f8f9fa",
    "grid": "#e2e8f0",
    "text": "#1e293b",
    "dews": "#4f46e5",
    "ks": "#0891b2",
    "psi": "#d97706",
    "evidently": "#db2777",
    "nannyml": "#059669",
    "threshold": "#dc2626",
}


def _apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], alpha=0.4, linewidth=0.5)


@dataclass
class BaselineResult:
    name: str
    detection_lag: float | None  # batches after drift_start to first alert AFTER drift
    false_alert_rate: float      # fraction of pre-drift batches with alerts
    alerts_total: int
    runtime_seconds: float
    has_forecasting: bool
    label_free: bool
    warning_lead: float | None = None  # batches of early warning BEFORE drift (positive = good)
    status: str = "available"    # available | not_installed


def _run_dews(
    scenario: DemoScenario,
    batches: int,
    batch_size: int = 80,
    window_size: int = 160,
    ds_crit: float = 0.30,
    alert_confirmation: int = 2,
    forecast_horizon: int = 8,
) -> tuple[BaselineResult, list[float]]:
    """Run full DEWS pipeline."""
    t0 = time.perf_counter()
    config = MonitorConfig(
        window_size=window_size,
        batch_size=batch_size,
        forecast_horizon=forecast_horizon,
        ds_crit=ds_crit,
        alert_confirmation_batches=alert_confirmation,
        retraining_enabled=False,
        adaptive_threshold_enabled=False,
    )
    system, _ = build_demo_system(config=config, store_path=":memory:")
    summary = run_demo_evaluation(system=system, scenario=scenario, max_batches=batches)
    elapsed = time.perf_counter() - t0

    drift_scores = [r.drift_score for r in system.batch_results]
    alerts = system.store.load_alerts()
    alert_batches = [a["batch_index"] for a in alerts]

    # First alert after drift onset = detection lag
    post_drift_alerts = [b for b in alert_batches if b >= scenario.drift_start_batch]
    first_post = post_drift_alerts[0] if post_drift_alerts else None
    lag = float(first_post - scenario.drift_start_batch) if first_post is not None else None

    # First alert before critical threshold breach = warning lead time (DEWS's advantage)
    breach_batch = next((i for i, r in enumerate(system.batch_results) if r.drift_score >= config.ds_crit), None)
    first_alert = alert_batches[0] if alert_batches else None
    warning_lead = None
    if first_alert is not None and breach_batch is not None and first_alert < breach_batch:
        warning_lead = float(breach_batch - first_alert)
    elif alerts and alerts[0].get("predicted_ttd") is not None and alerts[0]["predicted_ttd"] > 0:
        warning_lead = round(float(alerts[0]["predicted_ttd"]), 1)

    # Pre-drift alerts (alerts before synthetic drift was injected) = false alerts
    false_alerts = sum(1 for b in alert_batches if b < scenario.drift_start_batch)
    far = false_alerts / max(scenario.drift_start_batch, 1)

    return BaselineResult(
        name="DEWS",
        detection_lag=lag,
        false_alert_rate=far,
        alerts_total=len(alerts),
        runtime_seconds=round(elapsed, 3),
        has_forecasting=True,
        label_free=True,
        warning_lead=warning_lead,
    ), drift_scores


def _run_ks_baseline(scenario: DemoScenario, threshold: float = 0.25, window_size: int = 160) -> tuple[BaselineResult, list[float]]:
    """Standalone KS-test baseline (mean KS across features > threshold = alert)."""
    t0 = time.perf_counter()
    reference = scenario.reference_frame
    continuous = scenario.feature_schema.continuous
    ref_view = reference[continuous]

    scores = []
    alert_batches = []
    live_batches = []
    for i, batch in enumerate(scenario.stream_batches):
        live_batches.append(batch)
        window = pd.concat(live_batches, ignore_index=True).tail(window_size)
        ks_values = []
        for feat in continuous:
            stat = float(stats.ks_2samp(ref_view[feat].to_numpy(), window[feat].to_numpy()).statistic)
            ks_values.append(stat)
        mean_ks = float(np.mean(ks_values))
        scores.append(mean_ks)
        if mean_ks >= threshold:
            alert_batches.append(i)

    elapsed = time.perf_counter() - t0
    post_drift_alerts = [b for b in alert_batches if b >= scenario.drift_start_batch]
    first_post = post_drift_alerts[0] if post_drift_alerts else None
    lag = float(first_post - scenario.drift_start_batch) if first_post is not None else None
    false_alerts = sum(1 for b in alert_batches if b < scenario.drift_start_batch)
    far = false_alerts / max(scenario.drift_start_batch, 1)

    return BaselineResult(
        name="KS-only",
        detection_lag=lag,
        false_alert_rate=far,
        alerts_total=len(alert_batches),
        runtime_seconds=round(elapsed, 3),
        has_forecasting=False,
        label_free=True,
    ), scores


def _run_psi_baseline(scenario: DemoScenario, threshold: float = 0.15, window_size: int = 160) -> tuple[BaselineResult, list[float]]:
    """Standalone PSI baseline (mean PSI across features > threshold = alert)."""
    t0 = time.perf_counter()
    reference = scenario.reference_frame
    continuous = scenario.feature_schema.continuous
    ref_view = reference[continuous]

    scores = []
    alert_batches = []
    live_batches = []
    for i, batch in enumerate(scenario.stream_batches):
        live_batches.append(batch)
        window = pd.concat(live_batches, ignore_index=True).tail(window_size)
        psi_values = []
        for feat in continuous:
            psi = population_stability_index(ref_view[feat].to_numpy(), window[feat].to_numpy(), bins=10)
            psi_values.append(psi)
        mean_psi = float(np.mean(psi_values))
        scores.append(mean_psi)
        if mean_psi >= threshold:
            alert_batches.append(i)

    elapsed = time.perf_counter() - t0
    post_drift_alerts = [b for b in alert_batches if b >= scenario.drift_start_batch]
    first_post = post_drift_alerts[0] if post_drift_alerts else None
    lag = float(first_post - scenario.drift_start_batch) if first_post is not None else None
    false_alerts = sum(1 for b in alert_batches if b < scenario.drift_start_batch)
    far = false_alerts / max(scenario.drift_start_batch, 1)

    return BaselineResult(
        name="PSI-only",
        detection_lag=lag,
        false_alert_rate=far,
        alerts_total=len(alert_batches),
        runtime_seconds=round(elapsed, 3),
        has_forecasting=False,
        label_free=True,
    ), scores


def _run_evidently(scenario: DemoScenario, share_threshold: float = 0.25, window_size: int = 160) -> tuple[BaselineResult, list[float]]:
    """Run Evidently DataDriftPreset if installed."""
    t0 = time.perf_counter()
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return BaselineResult(
            name="Evidently", detection_lag=None, false_alert_rate=0,
            alerts_total=0, runtime_seconds=0, has_forecasting=False,
            label_free=True, status="not_installed",
        ), []

    reference = scenario.reference_frame
    continuous = scenario.feature_schema.continuous + scenario.feature_schema.categorical
    ref_view = reference[continuous]

    scores = []
    alert_batches = []
    live_batches = []
    for i, batch in enumerate(scenario.stream_batches):
        live_batches.append(batch)
        window = pd.concat(live_batches, ignore_index=True).tail(window_size)
        try:
            report = Report(metrics=[DataDriftPreset()]).run(
                reference_data=ref_view, current_data=window[continuous],
            )
            payload = report.dict() if hasattr(report, "dict") else json.loads(report.json())
            drift_metric = next(
                (m for m in payload.get("metrics", [])
                 if m.get("metric_name", "").startswith("DriftedColumnsCount")), None,
            )
            share = drift_metric.get("value", {}).get("share", 0.0) if drift_metric else 0.0
            scores.append(float(share))
            if share >= share_threshold:
                alert_batches.append(i)
        except Exception:
            scores.append(0.0)

    elapsed = time.perf_counter() - t0
    post_drift_alerts = [b for b in alert_batches if b >= scenario.drift_start_batch]
    first_post = post_drift_alerts[0] if post_drift_alerts else None
    lag = float(first_post - scenario.drift_start_batch) if first_post is not None else None
    false_alerts = sum(1 for b in alert_batches if b < scenario.drift_start_batch)
    far = false_alerts / max(scenario.drift_start_batch, 1)

    return BaselineResult(
        name="Evidently",
        detection_lag=lag,
        false_alert_rate=far,
        alerts_total=len(alert_batches),
        runtime_seconds=round(elapsed, 3),
        has_forecasting=False,
        label_free=True,
    ), scores


def _run_nannyml(
    scenario: DemoScenario,
    batch_size: int = 80,
    method: str = "reconstruction",
    model: Any = None,
) -> tuple[BaselineResult, list[float]]:
    """Run NannyML DataReconstructionDriftCalculator or CBPE if installed."""
    t0 = time.perf_counter()
    try:
        import nannyml as nml
    except ImportError:
        return BaselineResult(
            name="NannyML", detection_lag=None, false_alert_rate=0.0,
            alerts_total=0, runtime_seconds=0.0, has_forecasting=False,
            label_free=True, status="not_installed",
        ), []

    continuous = scenario.feature_schema.continuous
    reference = scenario.reference_frame[continuous].copy()
    reference["timestamp"] = pd.date_range("2020-01-01", periods=len(reference), freq="min")

    stream_concat = pd.concat(scenario.stream_batches, ignore_index=True)[continuous].copy()
    stream_concat["timestamp"] = pd.date_range(
        reference["timestamp"].iloc[-1] + pd.Timedelta(minutes=1),
        periods=len(stream_concat),
        freq="min",
    )

    scores = []
    alert_batches = []

    try:
        if method == "cbpe":
            from sklearn.linear_model import LogisticRegression
            clf = model if model is not None else LogisticRegression(max_iter=1000).fit(
                scenario.reference_frame[continuous], scenario.reference_labels
            )

            ref_data = reference.copy()
            ref_data["target"] = scenario.reference_labels.to_numpy()
            ref_data["prediction"] = clf.predict(scenario.reference_frame[continuous])
            ref_data["prob_1"] = clf.predict_proba(scenario.reference_frame[continuous])[:, 1]

            stream_data = stream_concat.copy()
            stream_features = pd.concat(scenario.stream_batches, ignore_index=True)[continuous]
            stream_data["prediction"] = clf.predict(stream_features)
            stream_data["prob_1"] = clf.predict_proba(stream_features)[:, 1]

            estimator = nml.CBPE(
                y_pred_proba="prob_1",
                y_pred="prediction",
                y_true="target",
                timestamp_column_name="timestamp",
                problem_type="classification_binary",
                metrics=["accuracy"],
                chunk_size=batch_size,
            )
            estimator.fit(ref_data)
            est = estimator.estimate(stream_data).to_df()
            analysis_rows = est[est[("chunk", "period")] == "analysis"]

            acc_values = analysis_rows[("accuracy", "value")].tolist()
            # Estimated error rate (1 - estimated accuracy)
            scores = [round(max(0.0, 1.0 - float(v)), 4) for v in acc_values]
            alert_batches = [
                i for i, flagged in enumerate(analysis_rows[("accuracy", "alert")].fillna(False))
                if flagged
            ]
        else:
            calc = nml.DataReconstructionDriftCalculator(
                column_names=continuous,
                timestamp_column_name="timestamp",
                chunk_size=batch_size,
            )
            calc.fit(reference)
            recon_res = calc.calculate(stream_concat).to_df()
            analysis_rows = recon_res[recon_res[("chunk", "period")] == "analysis"]

            recon_values = analysis_rows[("reconstruction_error", "value")].tolist()
            upper_th = float(analysis_rows[("reconstruction_error", "upper_threshold")].iloc[0])
            lower_th = float(analysis_rows[("reconstruction_error", "lower_threshold")].iloc[0])
            th_range = max(upper_th - lower_th, 1e-6)

            # Scaled score: relative to lower threshold and alert boundary (1.0 = alert)
            scores = [round(max(0.0, (float(v) - lower_th) / th_range), 4) for v in recon_values]
            alert_batches = [
                i for i, flagged in enumerate(analysis_rows[("reconstruction_error", "alert")].fillna(False))
                if flagged
            ]

    except Exception as exc:
        return BaselineResult(
            name="NannyML", detection_lag=None, false_alert_rate=0.0,
            alerts_total=0, runtime_seconds=round(time.perf_counter() - t0, 3),
            has_forecasting=False, label_free=True, status=f"error: {exc}",
        ), []

    elapsed = time.perf_counter() - t0
    post_drift_alerts = [b for b in alert_batches if b >= scenario.drift_start_batch]
    first_post = post_drift_alerts[0] if post_drift_alerts else None
    lag = float(first_post - scenario.drift_start_batch) if first_post is not None else None
    false_alerts = sum(1 for b in alert_batches if b < scenario.drift_start_batch)
    far = false_alerts / max(scenario.drift_start_batch, 1)

    return BaselineResult(
        name="NannyML",
        detection_lag=lag,
        false_alert_rate=far,
        alerts_total=len(alert_batches),
        runtime_seconds=round(elapsed, 3),
        has_forecasting=False,
        label_free=True,
    ), scores


def chart_comparison_scores(all_scores: dict[str, list[float]], scenario: DemoScenario, output_dir: Path) -> Path:
    """Overlay detection scores from all tools on a single chart."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    _apply_style(ax)

    colors = {"DEWS": PALETTE["dews"], "KS-only": PALETTE["ks"],
              "PSI-only": PALETTE["psi"], "Evidently": PALETTE["evidently"],
              "NannyML": PALETTE["nannyml"]}
    linewidths = {"DEWS": 2.5}

    for name, scores in all_scores.items():
        if not scores:
            continue
        color = colors.get(name, "#888888")
        lw = linewidths.get(name, 1.5)
        alpha = 1.0 if name == "DEWS" else 0.7
        ax.plot(range(len(scores)), scores, color=color, linewidth=lw,
                alpha=alpha, label=name, zorder=6 if name == "DEWS" else 3)

    # Drift zone
    ax.axvspan(scenario.drift_start_batch, len(scenario.stream_batches),
               alpha=0.06, color=PALETTE["threshold"])
    ax.axvline(x=scenario.drift_start_batch, color=PALETTE["threshold"],
               linestyle=":", linewidth=1, alpha=0.6)
    ax.text(scenario.drift_start_batch + 0.2, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.4,
            "Drift Injected ->", fontsize=8, color=PALETTE["threshold"],
            alpha=0.8, fontweight="bold")

    ax.set_xlabel("Batch Index", fontsize=10)
    ax.set_ylabel("Detection Score", fontsize=10)
    ax.set_title("Detection Score Comparison - DEWS vs Baselines",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    path = output_dir / "benchmark_scores_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  + {path}")
    return path


def chart_comparison_table(results: list[BaselineResult], output_dir: Path) -> Path:
    """Render comparison table as a styled chart image."""
    fig, ax = plt.subplots(figsize=(11, 4.0), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    columns = ["Tool", "Detection Lag\n(batches)", "Warning Lead\n(batches)", "False Alert\nRate", "Total\nAlerts",
               "Runtime\n(seconds)", "Forecasting", "Label-Free"]

    cell_text = []
    cell_colors = []
    for r in results:
        row = [
            r.name,
            str(int(r.detection_lag)) if r.detection_lag is not None else "-",
            str(int(r.warning_lead)) if r.warning_lead is not None else "-",
            f"{r.false_alert_rate:.1%}" if r.false_alert_rate > 0 else "0%",
            str(r.alerts_total),
            f"{r.runtime_seconds:.3f}",
            "Y" if r.has_forecasting else "N",
            "Y" if r.label_free else "N",
        ]
        cell_text.append(row)
        ncols = len(columns)
        # Highlight DEWS row
        if r.name == "DEWS":
            cell_colors.append([PALETTE["panel"]] + ["#e0e7ff"] * (ncols - 1))
        else:
            cell_colors.append([PALETTE["panel"]] * ncols)

    table = ax.table(
        cellText=cell_text, colLabels=columns,
        cellLoc="center", loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style cells
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_text_props(color=PALETTE["text"])
        if key[0] == 0:  # Header row
            cell.set_facecolor("#312e81")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    ax.set_title("Benchmark Comparison - Synthetic Drift Scenario",
                 fontsize=13, fontweight="bold", pad=16, color=PALETTE["text"])

    plt.tight_layout()
    path = output_dir / "benchmark_comparison_table.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  + {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DEWS against baselines")
    parser.add_argument("--batches", type=int, default=18)
    parser.add_argument("--output-dir", type=str, default="artifacts/charts")
    parser.add_argument("--skip-evidently", action="store_true",
                        help="Skip Evidently comparison even if installed")
    parser.add_argument("--skip-nannyml", action="store_true",
                        help="Skip NannyML comparison even if installed")
    parser.add_argument("--nannyml-method", type=str, default="reconstruction",
                        choices=["reconstruction", "cbpe"],
                        help="NannyML detection method: 'reconstruction' (multivariate drift) or 'cbpe' (performance estimation)")
    parser.add_argument("--model-factory", type=str, default=None,
                        help="Custom model factory in format module:function for CBPE")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=160)
    parser.add_argument("--ds-crit", type=float, default=0.30,
                        help="DEWS critical drift threshold (default: 0.30)")
    parser.add_argument("--alert-confirmation", type=int, default=2,
                        help="DEWS consecutive alert confirmation batches (default: 2)")
    parser.add_argument("--forecast-horizon", type=int, default=8,
                        help="DEWS forecast horizon in batches (default: 8)")
    parser.add_argument("--ks-threshold", type=float, default=0.25)
    parser.add_argument("--psi-threshold", type=float, default=0.15)
    parser.add_argument("--evidently-share", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building synthetic demo scenario...")
    scenario = build_demo_scenario(stream_batches=args.batches, batch_size=args.batch_size)
    print(f"  Drift starts at batch {scenario.drift_start_batch}")
    print(f"  Total stream batches: {len(scenario.stream_batches)}")
    print()

    all_results: list[BaselineResult] = []
    all_scores: dict[str, list[float]] = {}

    # DEWS
    print("Running DEWS...")
    dews_res, dews_scores = _run_dews(
        scenario,
        args.batches,
        batch_size=args.batch_size,
        window_size=args.window_size,
        ds_crit=args.ds_crit,
        alert_confirmation=args.alert_confirmation,
        forecast_horizon=args.forecast_horizon,
    )
    all_results.append(dews_res)
    all_scores["DEWS"] = dews_scores
    lead_disp = str(int(dews_res.warning_lead)) if dews_res.warning_lead is not None else "-"
    print(f"  Detection lag: {dews_res.detection_lag}, Warn lead: {lead_disp}, FAR: {dews_res.false_alert_rate:.1%}")

    # KS baseline
    print("Running KS-only baseline...")
    ks_res, ks_scores = _run_ks_baseline(scenario, threshold=args.ks_threshold, window_size=args.window_size)
    all_results.append(ks_res)
    all_scores["KS-only"] = ks_scores
    print(f"  Detection lag: {ks_res.detection_lag}, FAR: {ks_res.false_alert_rate:.1%}")

    # PSI baseline
    print("Running PSI-only baseline...")
    psi_res, psi_scores = _run_psi_baseline(scenario, threshold=args.psi_threshold, window_size=args.window_size)
    all_results.append(psi_res)
    all_scores["PSI-only"] = psi_scores
    print(f"  Detection lag: {psi_res.detection_lag}, FAR: {psi_res.false_alert_rate:.1%}")

    # Evidently
    if not args.skip_evidently:
        print("Running Evidently...")
        ev_res, ev_scores = _run_evidently(scenario, share_threshold=args.evidently_share, window_size=args.window_size)
        all_results.append(ev_res)
        if ev_scores:
            all_scores["Evidently"] = ev_scores
        print(f"  Status: {ev_res.status}")
        if ev_res.status == "available":
            lag_disp = str(int(ev_res.detection_lag)) if ev_res.detection_lag is not None else "None"
            print(f"  Detection lag: {lag_disp}, FAR: {ev_res.false_alert_rate:.1%}")

    # NannyML
    if not args.skip_nannyml:
        print(f"Running NannyML ({args.nannyml_method})...")
        custom_model = None
        if args.model_factory:
            import importlib
            module_name, func_name = args.model_factory.split(":")
            factory = getattr(importlib.import_module(module_name), func_name)
            custom_model = factory()
        nml_res, nml_scores = _run_nannyml(
            scenario,
            batch_size=args.batch_size,
            method=args.nannyml_method,
            model=custom_model,
        )
        all_results.append(nml_res)
        if nml_scores:
            all_scores["NannyML"] = nml_scores
        print(f"  Status: {nml_res.status}")
        if nml_res.status == "available":
            lag_disp = str(int(nml_res.detection_lag)) if nml_res.detection_lag is not None else "None"
            print(f"  Detection lag: {lag_disp}, FAR: {nml_res.false_alert_rate:.1%}")

    print()
    print("Generating charts...")
    chart_comparison_scores(all_scores, scenario, output_dir)
    chart_comparison_table(all_results, output_dir)

    # Save raw results as JSON
    report = {
        "protocol": {
            "scenario": "synthetic_demo",
            "drift_start_batch": scenario.drift_start_batch,
            "total_batches": len(scenario.stream_batches),
            "batch_size": args.batch_size,
            "window_size": args.window_size,
            "features": len(scenario.feature_schema.continuous) + len(scenario.feature_schema.categorical),
        },
        "results": [
            {
                "name": r.name,
                "detection_lag": r.detection_lag,
                "warning_lead": r.warning_lead,
                "false_alert_rate": round(r.false_alert_rate, 4),
                "alerts_total": r.alerts_total,
                "runtime_seconds": r.runtime_seconds,
                "has_forecasting": r.has_forecasting,
                "label_free": r.label_free,
                "status": r.status,
            }
            for r in all_results
        ],
    }
    report_path = output_dir / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nResults saved to {report_path}")

    # Print summary table to console
    print("\n" + "=" * 90)
    print(f"{'Tool':<14} {'Det. Lag':>10} {'Warn Lead':>10} {'FAR':>8} {'Alerts':>8} {'Runtime':>10} {'Forecast':>10} {'Label-Free':>12}")
    print("-" * 90)
    for r in all_results:
        lag_str = f"{int(r.detection_lag)}" if r.detection_lag is not None else "-"
        lead_str = f"{int(r.warning_lead)}" if r.warning_lead is not None else "-"
        print(f"{r.name:<14} {lag_str:>10} {lead_str:>10} {r.false_alert_rate:>7.1%} {r.alerts_total:>8} {r.runtime_seconds:>9.3f}s {'Y' if r.has_forecasting else 'N':>10} {'Y' if r.label_free else 'N':>12}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
