# v4: Delay-Anchored Accuracy Estimation

Method code: [dews/anchored.py](../dews/anchored.py). Final run: `python scripts/run_final_evaluation.py` (after `scripts/compute_aux_signals.py`). Tables: `artifacts/v2/final_results.md` and `.json`; figure: `artifacts/v2/nowcast_mae.png`. Exploration scripts (`scripts/explore_*.py`) show how the design was reached and are not reported results.

## Why a new method

v3 showed that a single linear model's confidence carries little information once data shifts a lot. An exploration on the 8 development streams found a much better label-free signal: **disagreement between the frozen classifier and auxiliary models trained on the same reference rows** (RandomForest, HistGradientBoosting; they only supply monitor-side signals and never replace the monitored model). Its within-window Spearman correlation with error was 0.41 on average (positive on all 8 streams), against 0.19 for confidence drop, 0.13 for the statistical drift score and 0.21 for the fixed DEWS score.

## Method

At time t, labels exist only for batches b ≤ t − max(L, 1). The estimator:

1. **Anchors** on the mean realized accuracy of the latest 3 matured batches.
2. **Corrects** the anchor with a ridge regression (alpha 50, in accuracy units) of the *change since the anchor* in 11 label-free signals: confidence, entropy, low-confidence rate, positive rate, statistical drift scores, and disagreement with the auxiliary models.
3. Alerts use the estimated drop from the clean baseline, taking the minimum over the last 2 batches (`anchored_p2`), which mirrors the 2-consecutive-batches event rule.

Hyperparameters were fixed before the held-out run. The correction model transfers across streams because it is fit on accuracy *changes*. Tests: `tests/test_v2.py` checks that the estimate at time t does not change when labels not yet matured are altered.

## Evaluation

- **Development:** leave-one-dataset-out over 8 real streams (electricity, airlines, covertype, bank-marketing, KDDCup99, ldpa, run_or_walk, vehicle_sensIT).
- **Held-out:** 5 standard drift-benchmark streams never used during design (Hyperplane, SEA, CovPokElec, poker-hand, LED; 40 windows each; 51 windows contain an event, mostly CovPokElec and Hyperplane). The estimator and alert thresholds are fit on the development streams only.
- 25 batches of 250 rows per window, frozen logistic classifier, labels delayed by L ∈ {0, 2, 5}.

## Result 1: accuracy nowcasting error (accuracy points; lower is better)

| | stale labels | smoothed stale labels | raw CBPE | delay-aware CBPE | **anchored (ours)** |
|---|---:|---:|---:|---:|---:|
| Development, L = 0 | 9.64 | 9.85 | 17.91 | 10.14 | **9.11** |
| Development, L = 2 | 11.72 | 11.08 | 17.91 | 11.31 | **10.26** |
| Development, L = 5 | 14.15 | 13.84 | 17.91 | 13.57 | **12.05** |
| Held-out, L = 0 | 4.66 | 3.95 | 4.87 | 3.76 | 3.84 |
| Held-out, L = 2 | 4.62 | 3.95 | 4.87 | 3.83 | 3.93 |
| Held-out, L = 5 | 4.74 | 4.26 | 4.87 | 4.11 | 4.18 |

- **Gain grows with label delay.** Against carrying the last known accuracy forward, the anchored estimator cuts error by 5.5% (L = 0), 12.5% (L = 2) and 14.8% (L = 5) on the development streams. It wins on 4/8, 7/8 and 7/8 streams, and on 5/5 held-out streams at every delay (by 12–19%).
- **Against stronger baselines the margin is small.** On the development streams it beats smoothed stale labels by 7–13% at every delay, but on the held-out streams it ties with them (3.93 vs 3.95 at L = 2), and delay-aware CBPE is marginally better than it there (3.83).
- **Raw CBPE is the worst estimator** (17.9 on development), so a plain confidence-based estimate without label correction should not be used on shifting streams.
- The development macro is dominated by `ldpa` (33–44 points for every method, where person changes make accuracy swing wildly).

## Result 2: alerting (thresholds from development data only)

| setting | method | macro AUROC | recall | FA / 100 clean |
|---|---|---:|---:|---:|
| Dev, L = 2 | fixed DEWS score | 0.70 | 0.37 | 7.6 |
| Dev, L = 2 | wait for labels | 0.70 | 1.00 | 7.7 |
| Dev, L = 2 | **anchored, persistence 2** | **0.76** | 0.92 [0.86, 0.98] | **4.9** |
| Held-out, L = 2 | fixed DEWS score | 0.60 | 0.02 | 0.6 |
| Held-out, L = 2 | wait for labels | 0.73 | 1.00 | 2.5 |
| Held-out, L = 2 | anchored, persistence 2 | 0.73 | 0.92 [0.84, 0.98] | 2.2 |
| Held-out, L = 5 | wait for labels | 0.64 | 0.90 | 3.7 |
| Held-out, L = 5 | anchored, persistence 2 | 0.64 | 0.55 | 1.7 |

- **Versus the existing DEWS drift score the improvement is large and consistent:** recall 0.92 vs 0.37 (development) and 0.92 vs 0.02 (held-out) at L = 2.
- **Versus waiting for labels it is a trade-off, not a win.** On development streams at L = 2 it keeps 92% of the recall with 36% fewer false alerts and a higher AUROC (0.76 vs 0.70). On held-out streams it matches label-waiting in AUROC and false alerts, but loses recall at L = 5 (0.55 vs 0.90).
- **Early warning stays limited.** Only 12–45% of events are warned before onset, depending on the stream, and it is not higher than for label-waiting on the held-out streams.

## Claim-to-evidence

| Claim | Status |
|---|---|
| Disagreement with auxiliary models is a better label-free error signal than confidence or drift scores | Supported (8 streams, correlation) |
| The anchored estimator reduces nowcast error vs stale labels, increasingly with delay | Supported (development and held-out) |
| It beats smoothed stale labels | Supported on development streams only; tie on held-out |
| It detects degradation better than the DEWS drift score | Supported |
| It detects degradation better or earlier than waiting for delayed labels | Not supported (trade-off) |
| Reliable early warning on real streams | Not supported |
