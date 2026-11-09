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
| v4 final method | [v4_results.md](docs/v4_results.md) | Delay-anchored estimator (last known accuracy corrected by label-free signals, incl. disagreement with auxiliary models): nowcast error down 5.5 / 12.5 / 15% vs stale labels at L = 0 / 2 / 5 on 8 streams and 12-19% on 5 held-out streams; far better than the DEWS drift score; a trade-off, not a win, against waiting for labels. |
| v5 harmless vs harmful | [v5_harmless_vs_harmful.md](docs/v5_harmless_vs_harmful.md) | Sustained-alert rule does not help. Relevance-weighted shift separates harmless from harmful input shifts perfectly in synthetic data but is not supported on 13 real streams; use it as an explanation, not an alert filter. |
| v6 frozen benchmark | [verdict.md](docs/verdict.md) | Real data with controlled shifts, pre-registered, held-out run once: **FAILURE** (H1–H5 all fail). Thresholds did not transfer across datasets. |
| v7 mitigation | [v7_verdict.md](docs/v7_verdict.md) | Stream-normalized two-tier monitor, confirmed once on 6 never-used datasets: **PARTIAL SUCCESS**. WATCH warns before the accuracy drop on 87% of harmful gradual shifts (median 6 batches ahead) within the false-alert budget and beats the raw DEWS score on 5 of 6 datasets; it does not separate harmful from harmless shifts and fails on natural streams. |
| v8 harm estimation | [v8_verdict.md](docs/v8_verdict.md) | HARM tier (importance-weighted reference accuracy), confirmed once on 5 never-used datasets: **PARTIAL SUCCESS**. Harmful-vs-harmless AUROC 0.78 [0.70, 0.86] (target 0.80, missed); harmless-shift alarms cut from 97% (WATCH) to 34% while still warning before the drop on 76% of harmful gradual shifts, within the false-alert budget. |
| Ablations | [ablations.md](docs/ablations.md) | Why each version failed or worked: raw-unit thresholds break the alert budget 4–10× on unseen data while normalized or accuracy-unit scores hold it; input-shift signals warn early but are harm-blind; harm depends on where the model is wrong, not on which feature moved; stream normalization needs a stable start. |

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
python scripts/compute_aux_signals.py            # auxiliary-model disagreement signals (8 real streams)
python scripts/run_final_evaluation.py           # final locked evaluation (v4)
```

Reports and figures are written to `artifacts/v2/`.

## Dashboard

```bash
streamlit run streamlit_app.py
```

Four pages: an overview of the results, a live monitor that replays saved episodes batch by batch with both warning tiers and delayed labels, every verdict, and the ablation findings. It reads the saved experiment outputs in `artifacts/`, so run the experiment scripts first.

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
