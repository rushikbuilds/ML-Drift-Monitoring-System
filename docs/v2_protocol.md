# v2 Protocol: Delayed-Label-Aware Degradation Warning

v2 is a revision of the v1 experiment ([final_scope.md](final_scope.md), [evaluation_results.md](evaluation_results.md)), made after v1 returned a negative result. **v1 is kept as published.** This document lists what changed and why, so the revision is auditable.

## Why v1 failed (diagnosed, not tuned away)

| Finding | Evidence |
|---|---|
| No lead time by construction | In 50/50 seeds the accuracy event began on the same batch as the intervention. A 5σ shift on noise-free labels collapsed accuracy from 1.00 to about 0.5 within 1–3 batches, so no monitor could warn before it. |
| Inverted risk target | The v1 target was "an event *starts* in the next 5 batches". Every post-onset batch (highest drift) got label 0. A logistic fit on the KS statistic gave weight −5.3 on training seeds. The calibrator learned that more drift means less risk. |
| Trivial generator | One feature, a deterministic linear rule, baseline accuracy 1.00. |
| Code and docs out of sync | `early_warning.py` called `DriftMonitoringSystem` with missing arguments, and `process_batch` had no `evaluate_forecasts` parameter. Fixed in v2; the suite passes (54 tests). |

## What v2 changes

1. **Scenarios** (`dews/v2.py`): noisy labels (baseline accuracy ≈ 0.78), randomized intervention onset (batch 10–16), four features that matter and three nuisance features, and a linear classifier facing a rule with a nonlinear term. Five scenarios:
   - `clean`
   - `harmless_shift` (nuisance features only)
   - `harmful_gradual` (x3 drifts into the region where the nonlinear term bites; the event follows the intervention by about 12–16 batches)
   - `harmful_abrupt` (a step change, so little or no lead is possible)
   - `concept_gradual` (a concept-drift probe: P(X) never changes, so only delayed labels can reveal it)
2. **State target:** y_t = 1 if a confirmed degraded run (K = 2 consecutive batches ≥ 5 points below the baseline from the first 5 batches) starts or continues within (t, t+5]. Post-onset batches are now positive.
3. **Label delay matters:** the latest *matured* accuracy drop (labels of batch b usable at time t only if b ≤ t − max(L, 1)) is a causal input. A target for batch t may be used for an online update only at prediction time t + H + 1 + max(L, 1).
4. **Added baseline `delayed_label_only`:** alert on the matured accuracy drop. This answers whether drift signals add value beyond waiting for labels.
5. **Methods:** `stat_only`, `conf_only`, `fixed_score` (existing DEWS fusion), `slope_forecast`, `delayed_label_only`, `cal_signals_static` (signals only), `cal_full_static`, `cal_full_online` (signals plus matured-accuracy features, with a weighted refit on matured targets).

## Unchanged from v1

- Classifier: StandardScaler + LogisticRegression, fit on reference rows only and frozen.
- Horizon H = 5, event rule (K = 2, 5-point drop), delays L ∈ {0, 2, 5}.
- Budget: ≤ 5 false alerts per 100 clean batches. Thresholds are selected **on validation seeds only**. Scores before batch 6 are ignored for every method.
- Seeds: train 100–139, validation 140–159, test 200–249. All are disjoint from v1's 0–49.

## Disclosure on test-seed exposure

A `--quick` smoke run (12 train, 6 validation, 10 test seeds) was executed before the full run. Its test seeds (200–209) are a subset of the final test set. No method, scenario or protocol parameter was changed after seeing it. The scenario design (switching the harmful shift from x0 to x3) was set using training seeds 100–105 only, because the first design made accuracy *rise* under shift.

## Metrics

Event recall (alert within ±5 batches of onset), early-warning rate (alert strictly before onset), mean lead, false alerts per 100 clean batches (clean and harmless-shift episodes, plus the pre-warning region of event episodes), AUROC of the score against the state target (all batches; harmful episodes only), and Brier score for the probability models. 95% bootstrap intervals over test episodes.

## Limitations

- Synthetic scenarios are stylised. The Electricity check ([v2_results.md](v2_results.md)) is exploratory because its windows overlap and no drift is injected.
- Natural noise occasionally produces an event in a clean episode. Each such event is counted as an event for all methods.
- The fixed-score baseline is DEWS's existing weighted fusion with arbitrary normalisation. Its poor result under a strict false-alert budget partly reflects that normalisation.
