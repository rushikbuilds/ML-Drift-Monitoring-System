"""Command-line entry point for the ML Model Drift Early-Warning System.

Primary commands:
    python main.py --demo               Run synthetic end-to-end demo
    python main.py --serve              Start FastAPI backend
    python main.py --benchmark          Run internal benchmark suite
    python main.py --visualize          Generate demo charts (artifacts/charts/)
    python main.py --compare            Run benchmark comparison vs baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from dews import MonitorConfig, MLflowTracker, build_demo_system, run_demo_evaluation
from dews.benchmarks import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML Model Drift Early-Warning System")

    # ── Primary modes ──
    mode = parser.add_argument_group("Mode (pick one)")
    mode.add_argument("--demo", action="store_true", help="Run the synthetic end-to-end demo")
    mode.add_argument("--serve", action="store_true", help="Run the FastAPI backend with uvicorn")
    mode.add_argument("--benchmark", action="store_true", help="Run the internal benchmark suite")
    mode.add_argument("--visualize", action="store_true", help="Generate demo visualization charts")
    mode.add_argument("--compare", action="store_true", help="Run benchmark comparison vs baselines")

    # ── Common parameters ──
    params = parser.add_argument_group("Parameters")
    params.add_argument("--batches", type=int, default=18, help="Number of synthetic stream batches (default: 18)")
    params.add_argument("--batch-size", type=int, default=80, help="Rows per live batch (default: 80)")
    params.add_argument("--window-size", type=int, default=160, help="Sliding window size (default: 160)")
    params.add_argument("--forecast-horizon", type=int, default=8, help="Forecast horizon in batches (default: 8)")
    params.add_argument("--store", type=str, default="artifacts/dews.sqlite3", help="SQLite store path")
    params.add_argument("--output", type=str, default="", help="Optional path to write JSON summary")

    # ── Server options ──
    server = parser.add_argument_group("Server options")
    server.add_argument("--host", type=str, default="127.0.0.1", help="Host for FastAPI (default: 127.0.0.1)")
    server.add_argument("--port", type=int, default=8000, help="Port for FastAPI (default: 8000)")

    # ── Feature toggles ──
    features = parser.add_argument_group("Feature toggles")
    features.add_argument("--model-version", type=str, default="v1.0.0", help="Model version identifier")
    features.add_argument("--adaptive-threshold", action="store_true", help="Enable adaptive drift threshold")
    features.add_argument("--adaptive-sensitivity", type=float, default=2.5, help="Adaptive threshold sensitivity")
    features.add_argument("--enable-retraining", action="store_true", help="Enable automated retraining on critical drift")
    features.add_argument("--retrain-runner", type=str, default="local", choices=["local", "webhook", "in_process"])
    features.add_argument("--storage-dsn", type=str, default="", help="Storage DSN (postgresql:// for PG, empty for SQLite)")

    # ── MLflow ──
    mlflow = parser.add_argument_group("MLflow (optional)")
    mlflow.add_argument("--mlflow-uri", type=str, default="", help="MLflow tracking URI")
    mlflow.add_argument("--mlflow-experiment", type=str, default="dews-drift-monitoring", help="MLflow experiment name")
    mlflow.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")

    # ── Advanced (calibration — hidden from casual use) ──
    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument("--calibrate-weights", action="store_true", help="Calibrate fusion weights (advanced)")
    advanced.add_argument("--calibration-output", type=str, default="artifacts/weight_calibration.json")
    advanced.add_argument("--calibration-step", type=float, default=0.25)
    advanced.add_argument("--electricity-benchmark", action="store_true", help="Run Electricity temporal benchmark (advanced)")
    advanced.add_argument("--calibrate-electricity", action="store_true", help="Calibrate weights on Electricity (advanced)")
    advanced.add_argument("--electricity-data", type=str, default="")
    advanced.add_argument("--electricity-detector-reference", type=int, default=500)
    advanced.add_argument("--retrain-command", type=str, default="python train_and_monitor_pytorch.py")
    advanced.add_argument("--retrain-webhook", type=str, default="")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MonitorConfig(
        window_size=args.window_size,
        batch_size=args.batch_size,
        forecast_horizon=args.forecast_horizon,
        model_version=args.model_version,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment_name=args.mlflow_experiment,
        adaptive_threshold_enabled=args.adaptive_threshold,
        adaptive_sensitivity=args.adaptive_sensitivity,
        retraining_enabled=args.enable_retraining,
        retraining_runner=args.retrain_runner,
        retraining_command=args.retrain_command,
        retraining_webhook_url=args.retrain_webhook,
        storage_dsn=args.storage_dsn,
    )

    # ── Serve ──
    if args.serve:
        uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)
        return 0

    # ── Benchmark ──
    if args.benchmark:
        summary = run_benchmark_suite(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 0

    # ── Visualize ──
    if args.visualize:
        import subprocess, sys
        print("\nPhase 1: Generating demo visualizations...")
        cmd = [sys.executable, "scripts/visualize_results.py"]
        if args.adaptive_threshold:
            cmd.append("--adaptive-threshold")
        cmd.extend(["--batches", str(args.batches)])
        return subprocess.run(cmd).returncode

    # ── Compare ──
    if args.compare:
        import subprocess, sys
        print(f"\nPhase 2: Benchmarking DEWS against {args.batches} batches...")
        cmd = [sys.executable, "scripts/visualize_benchmark.py", "--batches", str(args.batches)]
        return subprocess.run(cmd).returncode

    # ── Calibrate weights (advanced) ──
    if args.calibrate_weights:
        from dews.calibration import calibrate_fusion_weights, save_calibration, simplex_weights
        report = calibrate_fusion_weights(
            candidates=simplex_weights(args.calibration_step),
            stream_batches=args.batches,
            batch_size=args.batch_size,
        )
        save_calibration(report, args.calibration_output)
        print(json.dumps(report["best"], indent=2, sort_keys=True))
        print(f"Calibration report: {Path(args.calibration_output).resolve()}")
        return 0

    # ── Electricity benchmark (advanced) ──
    if args.electricity_benchmark:
        from dews.temporal_benchmark import load_electricity, run_electricity_benchmark
        electricity = load_electricity(args.electricity_data or None)
        report = run_electricity_benchmark(electricity, config=config, batch_size=args.batch_size)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return 0

    # ── Calibrate electricity (advanced) ──
    if args.calibrate_electricity:
        from dews.temporal_benchmark import load_electricity
        from dews.calibration import simplex_weights
        from dews.temporal_benchmark import calibrate_electricity_weights
        electricity = load_electricity(args.electricity_data or None)
        report = calibrate_electricity_weights(
            electricity,
            candidates=simplex_weights(args.calibration_step),
            batch_size=args.batch_size,
            max_detector_reference_rows=args.electricity_detector_reference,
        )
        output = args.output or "artifacts/electricity_weight_calibration.json"
        Path(output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report["best"], indent=2, sort_keys=True))
        print(f"Calibration report: {Path(output).resolve()}")
        return 0

    # ── Demo ──
    system, scenario = build_demo_system(config=config, store_path=args.store)
    if not args.demo:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  ML Model Drift Early-Warning System (DEWS)              ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        print("  Quick start:")
        print("    python main.py --demo                Run synthetic demo")
        print("    python main.py --visualize           Generate demo charts")
        print("    python main.py --compare             Benchmark vs baselines")
        print("    python main.py --serve               Start API server")
        print()
        print(f"  Storage:            {Path(args.store).resolve()}")
        print(f"  Model version:      {config.model_version}")
        print(f"  Adaptive threshold: {'enabled' if config.adaptive_threshold_enabled else 'disabled'}")
        print(f"  Retraining:         {'enabled' if config.retraining_enabled else 'disabled'}")
        return 0

    # --- MLflow tracking ---
    tracker = MLflowTracker(config)
    if not args.no_mlflow:
        tracker.start_run(run_name=f"demo-{config.model_version}")

    summary = run_demo_evaluation(system=system, scenario=scenario, max_batches=args.batches)

    # Log each batch result to MLflow
    if tracker.enabled:
        for result in system.batch_results:
            tracker.log_drift_metrics(
                batch_index=result.batch_index,
                drift_score=result.drift_score,
                s_stat=result.statistical.get("s_stat", 0.0),
                s_emb=result.embedding.get("s_emb", 0.0),
                s_conf=result.confidence.get("s_conf", 0.0),
                accuracy=result.accuracy,
            )
            if result.alert:
                tracker.log_alert(result.alert)
        tracker.log_summary(summary)
        tracker.end_run()

    # Print adaptive threshold info if enabled
    if system.adaptive_threshold is not None:
        summary["adaptive_threshold"] = system.adaptive_threshold.to_dict()

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
