# ML Model Drift Early-Warning System (DEWS)

A research codebase for **label-delay-aware early warning of ML performance degradation**: given a frozen classifier, a stream of unlabeled batches, and labels that arrive late, can we warn before accuracy drops? It contains drift/confidence detectors, a replay protocol that enforces label availability, controlled and real-data experiments, and a small FastAPI service.

## Status and findings

| Stage | Document | Outcome |
|---|---|---|
| v1 | [evaluation_results.md](docs/evaluation_results.md) | Negative (0/20 events detected). Kept as published. |
| v1 diagnosis and v2 protocol | [v2_protocol.md](docs/v2_protocol.md) | v1 scenarios allowed no lead time, and its risk target was inverted. |
| v2 controlled study | [v2_results.md](docs/v2_results.md) | On gradual covariate drift a signals-only calibrator warned before onset in 98% of episodes (about 4 batches ahead); concept drift needs labels; online updating adds nothing. |
| v2 real-data study | [v2_realdata.md](docs/v2_realdata.md) | Negative: drift scores rank accuracy risk weakly, and their thresholds do not transfer across datasets. |
| v3 confirmation | [v3_results.md](docs/v3_results.md) | Accuracy-unit, delay-aware CBPE transfers across four unseen streams and is suggestively best at label delay 5. No gain over label-waiting on synthetic data. |

Scope notes: [final_scope.md](docs/final_scope.md), [implementation_plan.md](docs/implementation_plan.md) (v1 plan, historical).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

```bash
pytest -q --ignore=tests/test_api.py             # unit tests
python scripts/run_v2_experiment.py              # synthetic study (about 25 min), writes artifacts/v2/
python scripts/run_v2_realdata.py                # leave-one-dataset-out real data (downloads OpenML data)
python scripts/run_v3_confirmation.py            # confirmation on unseen data
```

Reports and figures are written to `artifacts/v2/`.

## Other entry points

```bash
python main.py --demo          # legacy synthetic demo pipeline (SQLite)
python main.py --serve         # FastAPI service; Swagger UI at http://localhost:8000/docs
python main.py --benchmark     # legacy benchmark
```

## Layout

```
dews/            detectors, fusion, forecasting, alerts, replay protocol; v2.py holds the experiment library
scripts/         experiment runners and plotting
backend/         FastAPI service (read-only views, demo lifecycle, local ingestion buffer)
dashboard/       small Python client for the API
tests/           unit tests
docs/            protocol, results, scope
artifacts/v2/    result tables and figures
```

## Limitations

- Controlled scenarios are stylised, and real-data windows are few (90 development + 40 confirmation windows from 8 datasets).
- Row order in some OpenML streams may not be strictly temporal.
- Event-level results on real data have wide confidence intervals.
- The system estimates risk; it does not establish causes or guarantee failure. It is not evaluated for production use, and distributed ingestion and dashboards were removed from the repository.

## License

MIT.
