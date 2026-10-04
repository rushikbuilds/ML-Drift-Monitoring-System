# Monitoring Comparison Benchmark

This folder compares DEWS, Evidently, and NannyML on the same chronological Electricity batches.

## Install optional tools

```bash
./.venv/bin/pip install evidently nannyml
```

The runner still executes DEWS when either optional package is absent and records `not_installed` in the output.

## Run

```bash
./.venv/bin/python benchmarks/compare_monitoring.py \
  --batch-size 512 \
  --detector-reference-rows 500 \
  --output-dir benchmarks/results
```

The default comparison evaluates three chronological folds: `0.5`, `0.6`,
and `0.7` reference fractions. Use `--reference-fractions 0.6` for a single
fold or provide another comma-separated list.

Use `--data path/to/electricity.csv` to avoid an OpenML download.

Outputs:

- `benchmarks/results/comparison.json`: protocol and normalized results
- `benchmarks/results/comparison.csv`: one row per tool

The reference period, batch size, chronological stream, classifier,
accuracy-degradation point, and detector reference cap are shared. Results
include signed warning lead time, fold standard deviation, runtime, and peak
process memory. The report separates input-drift alerts from performance
alerts; NannyML also reports estimated-versus-realized accuracy MAE. Evidently
alerts require at least 25% of columns to be drifted by default, configurable
with `--evidently-drift-share`. Evidently and NannyML remain optional because
their APIs and dependencies are not part of the core DEWS installation.

Accuracy degradation is reported against both the first five production
batches and a rolling five-batch production baseline. A null degradation point
means the required two consecutive low-accuracy batches did not occur.
