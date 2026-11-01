# v2 Results

Protocol and the list of changes from v1: [v2_protocol.md](v2_protocol.md). Full tables: `artifacts/v2/results.md` and `results.json`. Figures: `artifacts/v2/recall_by_method.png` and `timeline_harmful_gradual.png`.

Reproduce:

```bash
python scripts/run_v2_experiment.py      # about 25 minutes; caches signals in artifacts/v2/signals.pkl
python scripts/run_v2_electricity.py     # real-data check; downloads Electricity from OpenML
```

Setup: 250 held-out test episodes (50 seeds × 5 scenarios, seeds 200–249), frozen logistic classifier, thresholds chosen on validation seeds (140–159) for ≤ 5 false alerts per 100 clean batches. Event counts on the test set: 50/50 harmful episodes of each harmful type, 2/50 clean episodes and 1/50 harmless-shift episodes with a noise-driven event.

## Headline results (L = 2; 95% bootstrap CI over episodes)

| method | recall | early-warning rate | AUROC (state target) | Brier |
|---|---:|---:|---:|---:|
| fixed DEWS score | 0.00 [0.00, 0.00] | 0.00 | 0.56 | n/a |
| statistical-only | 0.00 [0.00, 0.00] | 0.00 | 0.55 | n/a |
| confidence-only | 0.44 [0.36, 0.52] | 0.27 | 0.50 | n/a |
| recent-slope forecast | 0.33 [0.26, 0.41] | 0.03 | 0.57 | n/a |
| wait for delayed labels | 1.00 [1.00, 1.00] | 0.42 [0.35, 0.50] | 0.90 | n/a |
| calibrator, signals only | 0.73 [0.65, 0.80] | 0.42 [0.34, 0.50] | 0.83 | 0.161 |
| calibrator, signals + matured accuracy | 1.00 [1.00, 1.00] | 0.56 [0.48, 0.64] | 0.94 | 0.086 |
| same, with online updates | 1.00 [1.00, 1.00] | 0.57 [0.49, 0.65] | 0.94 | 0.086 |

"Recall" counts an alert within ±5 batches of the event onset. "Early-warning rate" counts only alerts strictly before onset.

## What the results support

1. **v1's failure was caused by the design, not by the idea.** With a lead-time-bearing scenario and a state target, a calibrated signal model gives early warnings where the fixed score gives none.
2. **Gradual covariate drift is where drift signals help.** The signals-only calibrator warned before onset in 98% of `harmful_gradual` episodes (mean lead about 4.3 batches), against 0% for the fixed score and 56–72% for waiting for labels. It does this without any labels, and it stays within budget on harmless shifts (about 0.6 false alerts per 100 batches).
3. **Concept drift cannot be seen from the inputs.** The signals-only calibrator reaches recall of only 0.20 on `concept_gradual`, as expected when P(X) is unchanged. Matured labels are required, and the full model reaches recall 1.00 / 0.96 / 0.96 for L = 0 / 2 / 5.
4. **Label delay hurts label-only monitoring, and signals soften it.** The label-only baseline's early-warning rate falls from 0.50 (L = 0) to 0.31 (L = 5), and its recall from 1.00 to 0.92. The full calibrator falls from 0.61 to 0.44 and keeps recall at 0.97.
5. **Abrupt shifts leave little room to warn.** Early-warning rates are 0.1–0.28 for all methods, which matches the physics: there is no precursor.

## What the results do not support

- **Online updating adds nothing here.** `cal_full_online` and `cal_full_static` are within noise of each other. The online rule is not a contribution; it is reported as a null result.
- **The fixed DEWS score looks bad partly because of the protocol.** Under a strict false-alert budget on episodes that include harmless shifts, its threshold must exceed the harmless-shift level, which sits above the harmful signal. Its AUROC on harmful episodes is a respectable 0.73. Its normalisation is ad hoc, so do not read this as a general statement about drift detectors.
- **The label-using methods slightly exceed the budget on test.** For example, `cal_full_static` shows 6.0 false alerts per 100 clean-episode batches at L = 2 and the label-only baseline 5.5, against the 5.0 target set on validation. Compare them as roughly matched, not exactly matched.
- **Synthetic scenarios are stylised.** Findings 2–5 are about a controlled generator.

## Real-data check: Electricity (exploratory)

18 overlapping chronological windows (3000 reference rows, then 25 batches of 300). Natural drift only. The event rule fires in 15 of 18 windows; the 8 test windows start at row 18,500 or later. AUROC against the state target (prevalence 0.56), L = 2:

| score | AUROC |
|---|---:|
| fixed DEWS score | **0.76** |
| calibrator trained on synthetic data, signals + labels | 0.72 |
| statistical-only | 0.72 |
| confidence-only | 0.72 |
| recent-slope forecast | 0.71 |
| wait for delayed labels | 0.64 |
| calibrator trained in-domain on earlier windows (signals / full) | 0.36 / 0.38 |
| calibrator trained on synthetic data, signals only | 0.56 |

On real data the simple DEWS score ranks risk best, the synthetic-trained calibrator is competitive but not better, and the in-domain calibrators fail (below 0.5), most likely because only five heavily overlapping training windows exist and the regime changes over time. **The calibrator's advantage is not demonstrated on real data.** Treat the Electricity result as a limitation and as motivation for the next step: more datasets and a larger number of non-overlapping chronological episodes.

## Claim-to-evidence

| Claim | Status |
|---|---|
| The v1 negative result was caused by scenario and target design | Supported (diagnostics in `v2_protocol.md`) |
| Signals + matured labels give earlier warnings than waiting for labels, on controlled drift | Supported on synthetic data, 50 test seeds per scenario |
| Signals-only warnings beat the fixed score on gradual covariate drift | Supported on synthetic data |
| Online updating improves the calibrator | Not supported |
| The calibrator generalises to real temporal data | Not supported (Electricity, exploratory) |
| Production readiness | Not evaluated |
