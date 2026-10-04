from __future__ import annotations

import pandas as pd

from benchmarks.compare_monitoring import run_comparison


def test_comparison_writes_normalized_outputs(tmp_path) -> None:
    rows = []
    for index in range(80):
        rows.append({
            "feature_a": index / 80,
            "feature_b": float(index % 4),
            "class": ("UP" if index % 2 else "DOWN") if index < 48 else "DOWN",
        })
    data_path = tmp_path / "electricity.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    report = run_comparison(str(data_path), str(tmp_path / "results"), 0.60, 8, 32)

    assert report["protocol"]["same_reference_and_batches"] is True
    assert {result["tool"] for result in report["results"]} == {"dews", "evidently", "nannyml"}
    assert (tmp_path / "results" / "comparison.json").exists()
    assert (tmp_path / "results" / "comparison.csv").exists()
