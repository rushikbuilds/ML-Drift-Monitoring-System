from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser(description="Stream Elec2 observations to DEWS API")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of records to send per request")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds to wait between sending batches")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/api/v1/ingest", help="DEWS API Ingestion URL")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="DEWS API Base URL")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum number of batches to send")
    parser.add_argument("--output-csv", type=str, default="artifacts/elec2_streaming_log.csv",
                        help="Path to write CSV streaming log")
    parser.add_argument("--output-json", type=str, default="artifacts/elec2_streaming_log.json",
                        help="Path to write JSON streaming summary")
    parser.add_argument("--fetch-results", action="store_true",
                        help="Fetch backend scores and alerts after streaming")
    args = parser.parse_args()

    data_path = Path("artifacts/elec2_stream_features.csv")
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Please run train_elec2_model.py first.")
        return

    print(f"Loading streaming data from {data_path}...")
    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"Loaded {total_records} records.")

    batches = total_records // args.batch_size
    if args.max_batches:
        batches = min(batches, args.max_batches)

    print(f"Starting to stream {batches} batches of size {args.batch_size} to {args.api_url}...")

    session = requests.Session()
    log_records: list[dict] = []

    for batch_idx in range(batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        
        batch_df = df.iloc[start:end]
        records = batch_df.to_dict(orient="records")
        payload = {"data": records}

        t_start = time.perf_counter()
        resp_status = "error"
        backend = "unknown"
        error_msg = None
        status_code = None

        try:
            response = session.post(args.api_url, json=payload, timeout=10.0)
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            status_code = response.status_code
            if response.status_code == 200:
                res_json = response.json()
                resp_status = res_json.get("status", "ok")
                backend = res_json.get("backend", "unknown")
                print(f"[{batch_idx:03d}/{batches:03d}] Successfully sent {len(records)} records ({latency_ms:.1f} ms). Response: {res_json}")
            else:
                error_msg = response.text[:200]
                print(f"[{batch_idx:03d}/{batches:03d}] Failed to send batch. Status code: {response.status_code}, Response: {error_msg}")
        except requests.exceptions.RequestException as e:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            error_msg = str(e)
            print(f"[{batch_idx:03d}/{batches:03d}] Connection error: {e}")
            print("Make sure the backend is running (python main.py --serve).")
            log_records.append({
                "batch_index": batch_idx,
                "records_sent": len(records),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "http_status": None,
                "response_status": "connection_error",
                "backend": "none",
                "latency_ms": latency_ms,
                "error": error_msg,
            })
            break

        log_records.append({
            "batch_index": batch_idx,
            "records_sent": len(records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "http_status": status_code,
            "response_status": resp_status,
            "backend": backend,
            "latency_ms": latency_ms,
            "error": error_msg,
        })

        if batch_idx < batches - 1:
            time.sleep(args.interval)

    print("\nStreaming finished.")

    # Write CSV streaming log
    if args.output_csv and log_records:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_records).to_csv(csv_path, index=False)
        print(f"  + CSV streaming log saved to: {csv_path.resolve()}")

    # Fetch backend state if requested
    backend_scores = []
    backend_alerts = []
    if args.fetch_results:
        try:
            scores_resp = session.get(f"{args.base_url}/scores", timeout=5.0)
            if scores_resp.status_code == 200:
                backend_scores = scores_resp.json()
            alerts_resp = session.get(f"{args.base_url}/alerts", timeout=5.0)
            if alerts_resp.status_code == 200:
                backend_alerts = alerts_resp.json()
        except Exception as exc:
            print(f"Note: Could not fetch backend results: {exc}")

    # Write JSON streaming summary
    if args.output_json and log_records:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        latencies = [r["latency_ms"] for r in log_records if r["latency_ms"] is not None]
        summary = {
            "api_url": args.api_url,
            "batches_attempted": len(log_records),
            "batches_successful": sum(1 for r in log_records if r["http_status"] == 200),
            "total_records_sent": sum(r["records_sent"] for r in log_records if r["http_status"] == 200),
            "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "batches": log_records,
            "backend_scores": backend_scores,
            "backend_alerts": backend_alerts,
        }
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"  + JSON streaming summary saved to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
