# v8 Protocol: Reference-Based Harm Estimation

**Frozen on 2026-11-09**, after a development-only feasibility check and **before any v8 confirmation dataset was downloaded**. Machine-readable: [dews/v8_config.py](../dews/v8_config.py). Code: [dews/harm_estimation.py](../dews/harm_estimation.py). v6 ([verdict.md](verdict.md)) and v7 ([v7_verdict.md](v7_verdict.md)) are unchanged.

## 1. Why v8

v7's WATCH tier gives early warning of drift but alerts on 96% of harmless shifts; no earlier signal separated harmful from harmless shifts (AUROC ≤ 0.71). None of them used the information that defines harm: **where the model is wrong**. The labeled reference set shows it.

**HARM score (frozen):** importance-weighted reference accuracy.
1. For each batch, a logistic domain classifier separates reference rows from the batch.
2. Its odds (self-normalized, capped at 20) reweight each reference row's correctness under the monitored model.
3. The weighted mean is a label-free estimate of the batch's accuracy. Under pure covariate shift it targets the true accuracy.
4. HARM = (median estimate over the 8 burn-in batches) − (current estimate), in accuracy units.

Pure label changes are invisible to it by construction.

## 2. Development feasibility (8 v6 development datasets, L = 5)

Harmful vs harmless AUROC, over the first 3 post-onset batches:

| Score | AUROC |
|---|---|
| WATCH (v7) | 0.55 |
| k-nearest-neighbour reference accuracy drop | 0.68 |
| **Importance-weighted reference accuracy drop (HARM)** | **0.77** |
| Average of the two | 0.73 |

Thresholds chosen on 7 datasets and applied to the 8th:

| Score, threshold rule | T1 | T2 | False alerts / 100 | Harmless shifts alerting |
|---|---|---|---|---|
| WATCH, v6 rule | 0.82 | 0.97 | 3.2 | 0.89 |
| HARM, v6 rule (maximize T1) | 0.67 | 0.79 | 4.8 | 0.32 |
| **HARM, separation rule (frozen)** | 0.43 | 0.65 | 2.7 | 0.23 |

The frozen threshold rule maximizes the T2 share on harmful shifts minus the alert share on harmless shifts, with false alerts ≤ 5 per 100 clean batches. It uses pooled development episodes; HARM has no trained parameters.

## 3. Confirmation data (run once)

albert, default-of-credit-card-clients, kick, house_16H, 2dplanes and mv (OpenML 41147, 42477, 41162, 821, 727, 881). None was used before in this project. First 20 non-constant features; seeds 9500 + 100·index + episode; v6 episode design; logistic regression, label delay 5, all labels.

## 4. Hypotheses

| ID | Claim | PASS if |
|---|---|---|
| D1 | HARM separates harmful from harmless shifts | AUROC ≥ 0.80 |
| D2 | HARM rarely alerts on harmless shifts | harmless alert share ≤ 0.10, false alerts ≤ 5 |
| D3 | HARM cuts harmless alarms versus WATCH while still warning | harmless share drop ≥ 0.30 AND HARM T2 ≥ 0.50 AND false alerts ≤ 5 |
| D4 | HARM ranks harm better than WATCH | AUROC gain ≥ 0.10, paired 95% CI lower bound > 0 |

**Overall verdict:**
- **Success:** D1 and D2 pass.
- **Partial success:** not success, and D3 or D4 passes.
- **Failure:** otherwise.

**Expected from development, written before confirmation:** D1 borderline (0.77 on development), D2 fail, D3 and D4 likely pass.

## 5. Frozen development choices (2026-11-09, before download)

| File | sha256 (first 16) | Content |
|---|---|---|
| `artifacts/v8/frozen_v8.json` | b8005ca95d70cc4f | HARM threshold 0.0115 (estimated accuracy drop); WATCH threshold 4.011 z (from v7) |

Pooled development at the frozen threshold: T1 0.61, T2 0.78, false alerts 3.4 per 100 clean batches, harmless shifts alerting 0.30, AUROC 0.77.

## 6. Deviations

None yet.
