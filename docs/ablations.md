# Ablations and Failure Analysis

Post-hoc and explanatory: these results explain the verdicts in [verdict.md](verdict.md) (v6), [v7_verdict.md](v7_verdict.md) and [v8_verdict.md](v8_verdict.md), and do not change them. Script: `python scripts/run_ablations.py`; numbers: `artifacts/ablations/ablations.json`.

**Setup.** Every variant re-selects its threshold on development data with the rule of the experiment it ablates, then applies it unchanged to the evaluation data:
- v7 confirmation data for WATCH;
- v8 confirmation data for HARM;
- v6 held-out data for the v6 system.

All runs use logistic regression, label delay 5 and all labels.

**Metrics:**
- **T1:** share of harmful gradual shifts warned before the accuracy drop.
- **T2:** share of harmful shifts warned before labels could confirm.
- **FA:** false alerts per 100 clean batches.
- **Harmless:** share of harmless shifts that raised an alert.
- **AUROC:** harmful vs harmless separation.

## A. Why v6 failed: thresholds in raw units do not transfer

False alerts per 100 clean batches, with the threshold chosen on development data:

| Method | Development | Unseen data | Within budget (≤ 5)? |
|---|---|---|---|
| Raw DEWS drift score | 4.4 | **36.9** | no |
| Statistical-only drift score | 3.6 | **17.5** | no |
| Mean KS | 4.4 | **44.6** | no |
| v6 risk calibrator | 4.6 | 1.2 | yes, but badly under-alerting |
| Confidence-only | 4.9 | 2.2 | yes |
| Label trend | 2.6 | 3.1 | yes |
| Delay-aware CBPE | 4.8 | 5.5 | marginal |
| **v7 WATCH (stream-normalized)** | 2.3 | **4.4** | **yes** |
| **v8 HARM (accuracy units)** | 3.4 | **2.8** | **yes** |

**Finding.** Scores in dataset-specific units (KS statistics, fused drift scores) break the false-alert budget 4–10× on unseen data. Scores in units that mean the same thing everywhere (standard deviations from the stream's own normal, or accuracy points) stay within budget. This is the single most important lesson of the project, and it is why v7 and v8 improved on v6.

## B. What drives WATCH (v7)

| WATCH variant | T1 | T2 | FA | Harmless | AUROC |
|---|---|---|---|---|---|
| All 7 signals (frozen) | 0.87 | 1.00 | 4.4 | 0.96 | 0.51 |
| Without any single signal | 0.82–0.90 | 0.98–1.00 | 3.8–6.7 | 0.94–0.98 | 0.50–0.52 |
| Only DEWS drift score | 0.90 | 1.00 | 7.8 | 0.95 | 0.52 |
| Only mean KS | 0.90 | 1.00 | 5.0 | 0.95 | 0.53 |
| Only relevance-weighted shift | 0.79 | 0.98 | 4.4 | 0.92 | 0.51 |
| Only positive-rate shift | 0.79 | 0.90 | 4.2 | 0.73 | 0.54 |
| Only disagreement | 0.49 | 0.73 | 3.9 | 0.42 | 0.72 |
| Only confidence drop | 0.46 | 0.63 | 2.8 | 0.24 | 0.77 |
| Only entropy | 0.41 | 0.60 | 5.0 | 0.25 | 0.75 |

**Findings:**
- **WATCH is robust:** removing any one signal changes little, so no single signal is a point of failure.
- **There are two families of signals with opposite strengths:**
  - **Input-shift signals** (drift score, KS, relevance-weighted shift) warn early and almost always (T1 0.79–0.90), but they are blind to harm (AUROC ≈ 0.5, about 95% harmless alarms).
  - **Model-behaviour signals** (confidence, entropy, disagreement) warn later (T1 0.41–0.49) but are harm-aware (AUROC 0.72–0.77, 24–42% harmless alarms).

  This trade-off is what the two-tier design exploits.
- **Entropy and positive-rate shift keep false alerts in check:** removing them raises false alerts to 6.7 and 5.4.

## C. Which harm estimator (v8)

| HARM variant | Threshold rule | T1 | T2 | FA | Harmless | AUROC |
|---|---|---|---|---|---|---|
| **Importance-weighted (frozen)** | separation (frozen) | 0.76 | 0.83 | 2.8 | 0.34 | 0.78 |
| Importance-weighted | early-warning | 0.76 | 0.83 | 3.6 | 0.38 | 0.78 |
| Nearest-neighbour | separation | 0.49 | 0.70 | 0.2 | **0.17** | 0.76 |
| Nearest-neighbour | early-warning | 0.59 | 0.75 | 0.4 | 0.23 | 0.76 |
| Average of both | separation | 0.71 | 0.83 | 1.0 | 0.29 | 0.78 |
| Average of both | early-warning | 0.76 | 0.83 | 1.4 | 0.33 | 0.78 |

**Findings:**
- **All reference-based estimators reach about the same separation (AUROC 0.76–0.78).** That is far above every drift signal (≈ 0.5–0.6), so the gain comes from using the labeled reference, not from the specific estimator.
- **The variants trade sensitivity for quietness:**
  - nearest-neighbour is the quietest (17% harmless alarms) but warns less (T1 0.49);
  - the importance-weighted version warns most;
  - the average of both lowers false alerts (1.0 vs 2.8) at the same T2.
- **The threshold rule matters less than the estimator.**

## D. What went wrong inside the v6 system

| v6 calibrator variant | T1 | T2 | FA | Harmless | AUROC |
|---|---|---|---|---|---|
| All feature groups (v6 system) | 0.22 | 0.83 | 1.2 | 0.37 | 0.59 |
| Without auxiliary-model disagreement | **0.42** | 0.89 | 4.5 | 0.66 | 0.57 |
| Without drift and confidence signals | 0.33 | 0.87 | 4.3 | 0.51 | 0.60 |
| Without matured-label features | 0.19 | 0.34 | 0.2 | 0.21 | 0.59 |
| Without relevance-weighted shift | 0.19 | 0.72 | 0.7 | 0.29 | 0.59 |
| Only matured-label features | 0.39 | 0.96 | 5.3 | 0.54 | 0.56 |
| Only drift and confidence signals | 0.08 | 0.30 | 0.4 | 0.19 | 0.62 |
| Only auxiliary-model disagreement | 0.22 | 0.34 | 3.1 | 0.40 | 0.56 |
| Only relevance-weighted shift | 0.33 | 0.45 | 9.1 | 0.24 | 0.65 |

**Findings:**
- **v6 under-alerted because of raw, unnormalized features.** Removing raw disagreement levels, which differ widely across datasets, doubles T1 (0.22 → 0.42) and brings false alerts back to the budget. The calibrator had learned dataset-specific levels that didn't carry over: the same transfer problem as in A, inside a model.
- **Most of its T2 came from delayed labels.** Without the matured-label features, T2 collapses (0.83 → 0.34), and the label features alone nearly match the full system.

## E. Why most shifts are harmless

Share of shift episodes that became harmful:

| Data | Shifts | Harmful | Shifted feature important | Shifted feature unimportant | Baseline accuracy predicts harm (AUROC) |
|---|---|---|---|---|---|
| v6 development | 351 | 0.18 | 0.19 | 0.17 | 0.41 |
| v6 held-out | 222 | 0.24 | 0.23 | 0.25 | 0.34 |
| v7 confirmation | 266 | 0.20 | 0.20 | 0.19 | 0.57 |
| v8 confirmation | 224 | 0.29 | 0.38 | 0.19 | 0.39 |

**Findings:**
- **About 80% of realistic input shifts don't hurt the model.** This was true in every data group.
- **Whether harm happens is barely related to how important the shifted feature is** (it differs only in v8), and not at all to the model's baseline accuracy.
- **Harm depends on whether the shift moves data into regions where this particular model is wrong.** That is why feature-importance weighting (v5) failed and why reference-based estimation (v8) is the first approach that separates harmful from harmless shifts.

## F. Why WATCH fails on natural streams (v7 C5)

| Natural-stream statistic (328 windows) | Value |
|---|---|
| Windows with an accuracy event | 132 |
| Event onset batch, quartiles | 5 / 8 / 12 |
| Events within 5 batches after the 8-batch burn-in | 100 of 132 |
| Coefficient of variation of accuracy during burn-in | 0.12 |
| Post-burn-in batches with a WATCH alert | 12% |

**Finding.** On natural streams the drift is already under way when monitoring starts. The median event begins at batch 8, the end of the burn-in, and accuracy varies by about 12% inside the burn-in itself. So the "normal" that WATCH measures against is not normal, and there is almost no room to warn early. This is a limit of the burn-in assumption, not of the signals: stream normalization needs a genuinely stable period after deployment.

## Summary

1. **Units matter more than models.** Every method in raw units failed to transfer; every method in transferable units (z-scores from the stream's own normal, or accuracy points) held its false-alert budget on unseen data.
2. **Drift signals and harm signals are different things.** Input-shift signals warn early but can't tell harm. Model-behaviour signals and reference-based accuracy estimates can partly tell harm, at the cost of later warnings.
3. **Harm comes from where the model is wrong, not from which feature moved.** Only methods that use the labeled reference to locate model errors separate harmful from harmless shifts (AUROC about 0.78).
4. **Stream normalization needs a stable start.** It works when monitoring begins in a stable period (the benchmark) and fails when drift is already under way (natural streams).
