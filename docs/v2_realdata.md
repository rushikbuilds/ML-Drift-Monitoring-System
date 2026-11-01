# Real-Data Study (leave-one-dataset-out)

Script: `python scripts/run_v2_realdata.py` (needs `artifacts/v2/signals.pkl` from the synthetic run; downloads four OpenML datasets). Full tables: `artifacts/v2/realdata_results.md` and `.json`.

## Design

- **Datasets** (OpenML, natural row order, no injected drift): electricity, airlines, covertype (first 10 numeric features, class 2 vs rest), bank-marketing.
- **Windows:** non-overlapping chronological windows of 2000 reference rows followed by 25 batches of 250 rows. Evenly spaced, at most 40 per dataset: 5 / 40 / 40 / 5 windows (90 total).
- **Same protocol as the synthetic study:** frozen logistic classifier, 5-point/2-batch event rule, H = 5, delays 0/2/5.
- **Leave-one-dataset-out:** for each held-out dataset, the calibrators and every alert threshold are fit on the *other* datasets only. A second calibrator variant is trained on the synthetic data from the main study.
- **Caveats:** 63 of the 90 windows contain an event, so the "clean" region is small. Airlines and covertype row order is not guaranteed to be truly temporal. Electricity and bank-marketing have only 5 windows each. Thresholds are selected on the other datasets with in-sample calibrator scores, which is optimistic for the calibrator, so its false alerts below are, if anything, understated.

## Results at L = 2 (pooled over held-out datasets; 95% CI over windows)

| method | mean AUROC | event recall | early-warning rate | false alerts / 100 clean batches |
|---|---:|---:|---:|---:|
| statistical-only | 0.57 | 0.40 | 0.24 | 28.6 |
| confidence-only | 0.51 | 0.11 | 0.05 | 3.7 |
| fixed DEWS score | 0.70 | 0.43 [0.30, 0.56] | 0.29 | 21.3 |
| recent-slope forecast | 0.63 | 0.41 | 0.24 | 5.9 |
| **wait for delayed labels** | **0.79** | **0.97 [0.92, 1.00]** | 0.14 | **4.8** |
| calibrator, other real datasets, signals only | 0.44 | 0.79 | 0.59 | 82.5 |
| calibrator, other real datasets, signals + labels | 0.70 | 0.89 [0.81, 0.95] | 0.43 | 46.5 |
| calibrator, synthetic-trained, signals + labels | 0.71 | 0.24 | 0.13 | 10.7 |

## What this shows

1. **The synthetic gain does not transfer.** On real data, drift and confidence signals rank accuracy risk weakly or inconsistently (mean AUROC 0.44–0.70, with per-dataset values from 0.34 to 0.97). For example, the fixed score is excellent on bank-marketing (0.97) and below chance on covertype (0.45).
2. **Thresholds do not transfer across datasets.** Every signal-based method that was within the 5% budget on the other datasets raised 4–16× more false alerts on the held-out one (for example, the fixed score gave 21 per 100 clean batches). Results that look like high recall on drift signals come with unacceptable false-alert rates and should not be read as detection.
3. **Waiting for labels is the strongest monitor at L ≤ 2.** It keeps recall 0.97 at 4.8 false alerts per 100 batches, inside the budget, and ranks risk best (AUROC 0.79). At L = 5 its recall falls to 0.73 and its AUROC to 0.70, which is the regime where a label-free signal would be most useful. No signal-based method filled that gap within the budget.
4. **The calibrator does not beat label-waiting on real data.** The synthetic-trained full calibrator is within budget at L = 0 but misses most events (recall 0.14–0.24). The cross-dataset calibrator catches more events but with false-alert rates of 35–75 per 100.

## Conclusion for the paper

Controlled experiments show *when* drift signals can give early warning (gradual covariate drift into a region where the model is wrong). The real-data study shows that, on four natural streams, this does not turn into a reliable label-free warning: shift is a weak and dataset-specific proxy for accuracy loss, and alert thresholds do not carry over between datasets. That is consistent with the project's original caution that a drift alarm is not evidence of degradation. The defensible contributions are the delay-aware protocol, the diagnosis of why the first design failed, the controlled study, and this negative real-data result with its matched baselines. The next step would be per-dataset calibration of thresholds on a short labeled period, and more datasets with confirmed temporal order.
