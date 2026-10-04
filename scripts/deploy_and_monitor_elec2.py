from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dews import (
    DriftMonitoringSystem,
    FeatureSchema,
    MonitorConfig,
    TorchMLPAdapter,
)
from dews.alerts import AlertEngine
from dews.forecasting import DriftForecaster
from dews.integrations import IntegrationManager
from dews.storage import DriftStore


class Elec2MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)  # penultimate layer
        self.bn3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.relu(self.bn1(self.fc1(x))))
        h = self.dropout(self.relu(self.bn2(self.fc2(h))))
        h = self.dropout(self.relu(self.bn3(self.fc3(h))))
        return self.output(h)


def _make_embedding_extractor(model: Elec2MLP, feature_columns: list[str], preprocessor, device: str):
    def extract(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[feature_columns]
        scaled = preprocessor.transform(raw).astype(np.float32)
        tensor_input = torch.tensor(scaled, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            h = model.dropout(model.relu(model.bn1(model.fc1(tensor_input))))
            h = model.dropout(model.relu(model.bn2(model.fc2(h))))
            embeddings = model.relu(model.bn3(model.fc3(h)))  # 16-dim penultimate
        return embeddings.cpu().numpy()
    return extract


def build_system(config: MonitorConfig, store_path: str):
    if not config.model_version or config.model_version == "unversioned":
        config.model_version = "elec2-mlp-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
    
    with open(artifacts_dir / "elec2_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
        
    X_ref_df = pd.read_csv(artifacts_dir / "elec2_reference_features.csv")
    
    numeric_features = X_ref_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_ref_df.select_dtypes(include=['category', 'object']).columns.tolist()
    feature_columns = numeric_features + categorical_features

    dummy_scaled = preprocessor.transform(X_ref_df.iloc[:2]).astype(np.float32)
    model = Elec2MLP(input_dim=dummy_scaled.shape[1], num_classes=2)
    model.load_state_dict(torch.load(artifacts_dir / "elec2_model.pth", map_location=device, weights_only=True))
    model.to(device)

    embedding_fn = _make_embedding_extractor(model, feature_columns, preprocessor, device)
    preprocessor_fn = lambda frame: preprocessor.transform(frame[feature_columns]).astype(np.float32)
    adapter = TorchMLPAdapter(model=model, feature_columns=feature_columns, device=device, embedding_fn=embedding_fn, preprocessor_fn=preprocessor_fn)
    
    store = DriftStore(store_path)
    schema = FeatureSchema(continuous=numeric_features, categorical=categorical_features)
    
    system = DriftMonitoringSystem(
        config=config,
        schema=schema,
        model=adapter,
        store=store,
        alert_engine=AlertEngine(config),
        forecaster=DriftForecaster(config.forecast_horizon),
        integration_manager=IntegrationManager()
    )
    system.fit_detectors(X_ref_df)
    return system, X_ref_df


def main():
    parser = argparse.ArgumentParser(description="Elec2 Real-world Drift Monitoring")
    parser.add_argument("--batches", type=int, default=50, help="Number of batches to stream (default: 50)")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size (default: 200)")
    parser.add_argument("--ds-crit", type=float, default=0.35, help="Critical drift score threshold (default: 0.35)")
    parser.add_argument("--output-csv", type=str, default="artifacts/elec2_monitor_results.csv",
                        help="Path to write batch-by-batch CSV results")
    parser.add_argument("--output-json", type=str, default="artifacts/elec2_monitor_results.json",
                        help="Path to write comprehensive JSON summary")
    parser.add_argument("--store", type=str, default="artifacts/elec2_demo.sqlite3",
                        help="SQLite store path")
    args = parser.parse_args()

    artifacts_dir = Path("artifacts")
    print("Loading Model and Data Artifacts...")
    
    X_stream_df = pd.read_csv(artifacts_dir / "elec2_stream_features.csv")
    y_stream = np.load(artifacts_dir / "elec2_stream_labels.npy")

    store_path = args.store
    Path(store_path).unlink(missing_ok=True)
    
    config = MonitorConfig(
        window_size=args.batch_size * 2,
        batch_size=args.batch_size,
        model_version="elec2-v1",
        ds_crit=args.ds_crit,
        adaptive_threshold_enabled=True,
    )
    
    print("Initializing DEWS baseline (Computing Reference Stats)...")
    system, X_ref_df = build_system(config, store_path)

    print("\nStarting Streaming Evaluation (Chronological Data Drift)...")
    batch_records: list[dict] = []

    for batch_idx in range(args.batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        if end > len(X_stream_df):
            break
        
        batch_frame = X_stream_df.iloc[start:end].reset_index(drop=True)
        batch_labels = pd.Series(y_stream[start:end])
        
        live_window = X_stream_df.iloc[max(0, end - config.window_size):end].reset_index(drop=True)

        res = system.process_batch(
            batch_index=batch_idx,
            reference_frame=X_ref_df,
            live_window=live_window,
            batch_labels=batch_labels
        )

        alert_str = f"ALERT: {res.alert['severity']}" if res.alert else "Normal"
        print(f"Batch {batch_idx:03d} | Drift Score: {res.drift_score:.3f} | Accuracy: {res.accuracy:.3f} | {alert_str}")

        top_feats = []
        if res.alert and res.alert.get("top_features"):
            top_feats = [tf["feature"] if isinstance(tf, dict) else str(tf) for tf in res.alert["top_features"]]

        batch_records.append({
            "batch_index": batch_idx,
            "records_in_batch": len(batch_frame),
            "drift_score": round(float(res.drift_score), 4),
            "s_stat": round(float(res.statistical.get("s_stat", 0.0)), 4),
            "s_emb": round(float(res.embedding.get("s_emb", 0.0)), 4),
            "s_conf": round(float(res.confidence.get("s_conf", 0.0)), 4),
            "mean_ks": round(float(res.statistical.get("mean_ks", 0.0)), 4),
            "mean_psi": round(float(res.statistical.get("mean_psi", 0.0)), 4),
            "accuracy": round(float(res.accuracy), 4) if res.accuracy is not None else None,
            "adaptive_threshold": round(float(res.adaptive_threshold), 4) if res.adaptive_threshold is not None else None,
            "alert_severity": res.alert["severity"] if res.alert else "None",
            "alert_message": res.alert["message"] if res.alert else "",
            "predicted_ttd": round(float(res.alert["predicted_ttd"]), 2) if res.alert and res.alert.get("predicted_ttd") is not None else None,
            "top_drifting_features": ", ".join(top_feats),
        })

    print("\nStreaming complete. Run complete.")
    print(f"Stored DEWS results in {store_path}")

    # Export CSV results
    if args.output_csv and batch_records:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(batch_records)
        results_df.to_csv(csv_path, index=False)
        print(f"  + CSV results saved to: {csv_path.resolve()}")

    # Export JSON results
    if args.output_json and batch_records:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        alerts = system.store.load_alerts()
        forecasts = system.store.load_forecasts()
        
        accuracies = [r["accuracy"] for r in batch_records if r["accuracy"] is not None]
        summary_report = {
            "dataset": "elec2",
            "model_version": config.model_version,
            "processed_batches": len(batch_records),
            "batch_size": args.batch_size,
            "window_size": config.window_size,
            "ds_crit": config.ds_crit,
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a.get("severity") == "Critical"),
            "warning_alerts": sum(1 for a in alerts if a.get("severity") == "Warning"),
            "average_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "final_drift_score": batch_records[-1]["drift_score"] if batch_records else None,
            "batches": batch_records,
            "alerts": alerts,
            "forecasts": forecasts,
        }
        json_path.write_text(json.dumps(summary_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"  + JSON summary saved to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
