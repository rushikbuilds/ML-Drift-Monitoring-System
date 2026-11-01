# v3: Delay-Aware CBPE and Confirmation on Unseen Data

Script: `python scripts/run_v3_confirmation.py` (needs the v2 caches from `run_v2_experiment.py` and `run_v2_realdata.py`). Full tables: `artifacts/v2/confirmation_results.md` and `.json`. Test coverage: `tests/test_v2.py::test_cbpe_scores_use_only_matured_labels`.

## Diagnosis of the real-data failure

In [v2_realdata.md](v2_realdata.md) every drift score lived in a dataset-specific unit (KS statistics, fused scores, probabilities). A threshold chosen on other datasets therefore meant something different on the held-out one, and false alerts rose from the 5-per-100 target to 20–80. Drift is also only a weak proxy for accuracy loss, so a calibrator on such features could not rescue it.

## Method: delay-aware CBPE (DA-CBPE)

The score is expressed in **accuracy units**, which is what the event definition itself uses:

1. Estimate each batch's accuracy label-free as the mean top-class probability (the idea behind CBPE).
2. Correct the estimate's bias with the residual (realized accuracy − mean confidence) of the latest 5 batches whose labels have *already arrived*. Before enough have arrived, the clean 5-batch baseline provides the bias.
3. Score = clean baseline accuracy − corrected estimate (clipped at 0). `da_cbpe_forecast` extrapolates the estimate's 3-batch slope 5 batches ahead. `cbpe_raw` is the same estimate without the delayed-label correction.

A threshold of 0.05 means "estimated 5-point drop" on every dataset. No new learned model is involved, and the code only reads labels with b ≤ t − max(L, 1).

## Confirmation protocol

The method was fixed on development data (synthetic seeds 100–249, real datasets electricity / airlines / covertype / bank-marketing). It was then evaluated once on data it never saw:

- fresh synthetic seeds 300–349 (250 episodes);
- four fresh OpenML streams: `run_or_walk`, `vehicle_sensIT`, `KDDCup99`, `ldpa` (40 non-overlapping windows; 28 contain an event; windows with a single-class reference were dropped).

Alert thresholds come from the development data only. A second, fitting-free "natural" threshold (0.05) is also reported for the accuracy-unit scores.

## Results on the fresh real streams (dev-selected thresholds; 95% CI over windows)

| method | L | recall | early-warning rate | false alerts / 100 clean batches |
|---|---:|---:|---:|---:|
| fixed DEWS score | any | 0.54 [0.36, 0.71] | 0.18 | 11.4 |
| recent-slope forecast | any | 0.79 [0.64, 0.93] | 0.29 | 10.1 |
| wait for delayed labels | 2 | 1.00 | 0.14 | 12.5 |
| DA-CBPE | 2 | 0.86 [0.71, 0.96] | 0.14 | 5.8 |
| DA-CBPE forecast | 2 | 0.93 [0.82, 1.00] | 0.21 | 6.9 |
| wait for delayed labels | 5 | 0.71 [0.54, 0.86] | 0.14 | 1.3 |
| DA-CBPE | 5 | 0.71 [0.54, 0.86] | 0.21 | 0.55 |
| DA-CBPE forecast | 5 | **0.86 [0.71, 0.96]** | **0.32** | **0.55** |

## What the results support

1. **Accuracy-unit scores transfer far better than drift scores.** On unseen datasets DA-CBPE stays at 0.3–7 false alerts per 100 clean batches, against 10–11 for the fixed and slope scores and 46–82 for the cross-dataset calibrator in v2.
2. **At L = 5 the forecast variant is the best monitor tried:** higher recall and a higher early-warning rate than waiting for labels, with fewer false alerts. The confidence intervals overlap, and 28 events from 4 datasets is a small sample, so read this as suggestive, not conclusive.
3. **At L = 2, DA-CBPE gives up a little recall for about half the false alerts of label-waiting** (0.86 vs 1.00 recall; 5.8 vs 12.5 false alerts), but it does not improve the early-warning rate.
4. **The delayed-label correction matters.** Without it (`cbpe_raw`), AUROC is 0.48 and recall 0.32 on the fresh streams.

## What the results do not support

- **No gain on fresh synthetic data.** There DA-CBPE matches waiting for labels (AUROC 0.89 vs 0.90 at L = 2; 0.84 vs 0.85 at L = 5). It has the same recall but more false alerts at L = 2 (10.1 vs 7.5 per 100). The controlled-study advantage of the signals-only calibrator ([v2_results.md](v2_results.md)) is a different effect.
- **Ranking quality is not better than the baselines on real data.** Pooled AUROC for DA-CBPE is 0.60–0.61 vs 0.76–0.80 for the drift scores on the fresh streams, because pooled AUROC is dominated by differences between datasets. Per dataset, no method is consistently best: on `ldpa` every score is at or below 0.61, and on `vehicle_sensIT` the DA-CBPE and stat scores are best at 0.61–0.75.
- **Early-warning rates stay low (0.14–0.32) for every method.** Real streams give little advance notice from these signals.
- **A systematic limit of confidence-based estimation:** it cannot see pure concept drift where the model remains confident. The delayed-label correction only catches up after the labels arrive.
- **Natural threshold is not safe for the forecast variant.** At 0.05, `da_cbpe_forecast` raises 8–19 false alerts per 100 batches.

## Claim-to-evidence

| Claim | Status |
|---|---|
| Accuracy-unit scores make thresholds transfer across datasets | Supported (4 unseen streams, small sample) |
| DA-CBPE forecast beats label-waiting at L = 5 on real streams | Suggestive; CIs overlap |
| DA-CBPE improves ranking over drift scores | Not supported |
| Any method gives reliable early warning on real streams | Not supported |
| DA-CBPE improves on label-waiting in controlled synthetic data | Not supported (parity) |
