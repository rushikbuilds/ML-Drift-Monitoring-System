# v7 Protocol: Stream-Normalized Two-Tier Monitor (mitigation after v6)

**Frozen on 2026-11-09**, after the v6 verdict ([verdict.md](verdict.md): FAILURE) and **before any confirmation dataset was downloaded**. Machine-readable version: [dews/v7_config.py](../dews/v7_config.py). Code: [dews/v7_monitor.py](../dews/v7_monitor.py). v6 stays published as it is; v7 is a separate, follow-up experiment.

## 1. Why v7

v2, v3 and v6 failed in the same way: alert thresholds chosen on some datasets did not transfer to others, because every score is in dataset-specific units. In v6 the primary system under-alerted on held-out data (1.2 false alerts per 100 clean batches against a budget of 5), while the raw DEWS score over-alerted (36.9).

**Development-only feasibility check** (leave-one-dataset-out on the 8 v6 development datasets, L = 5):

| Score | T1 | T2 | False alerts / 100 on the unseen fold | Harmless shifts alerting | Harm AUROC |
|---|---|---|---|---|---|
| Raw DEWS score | 0.27 | 0.43 | 10.5 | 0.25 | 0.62 |
| Normalized max of label-free signals | 0.82 | 0.97 | 3.2 | 0.89 | 0.55 |
| Normalized calibrator | 0.49 | 0.89 | 4.4 | 0.40 | 0.59 |
| Waiting for labels | 0.43 | 0.95 | 3.1 | 0.40 | 0.52 |

Normalization made thresholds transfer and made early drift warning strong, but every score still alerts on harmless shifts. Hence two tiers.

## 2. System

- **Normalization:** each signal is converted to robust z-scores, (x − median) / max(1.4826·MAD, 0.01), using the stream's first 8 batches (burn-in). No alerts during burn-in.
- **WATCH tier:** the maximum z-score over 7 label-free signals: DEWS drift score, mean KS, relevance-weighted shift, auxiliary-model disagreement, confidence drop, entropy, and positive-rate shift.
- **WARNING tier:** a logistic calibrator (C = 0.3) on those z-scores plus the z-scored delay-aware CBPE drop and the observed label drop (current and 3-batch mean). It is trained on development episodes for the state target.
- **Thresholds:** chosen on development data with the v6 rule (highest development T1 with false alerts ≤ 5 per 100 clean batches). WARNING uses leave-one-dataset-out development scores, as in v6.

## 3. Data

| Role | Datasets |
|---|---|
| Development (fitting and thresholds) | the 8 v6 development datasets, existing extracted episodes |
| **Confirmation (run once)** | adult, MiniBooNE, higgs, nomao, numerai28.6, Click_prediction_small (OpenML 1590, 41150, 23512, 1486, 23517, 1220). Never used before; first 20 non-constant features |
| Post-hoc only | the 5 v6 held-out datasets (already seen, so they cannot confirm anything) |

Episode design, settings, event rule, tiers and metrics are those of [v6_protocol.md](v6_protocol.md). Confirmation seeds are 9000 + 100·index + episode. Primary setting: logistic regression, label delay 5, all labels. Only the primary monitored model is run for confirmation.

## 4. Hypotheses (targets unchanged from v6)

| ID | Claim | PASS if (confirmation datasets) |
|---|---|---|
| C1 | WATCH warns before the accuracy drop (gradual harmful shifts) | T1 ≥ 0.60, median lead ≥ 2, false alerts ≤ 5 |
| C2 | WATCH warns before labels confirm, better than waiting for labels | difference ≥ 0.10 and paired CI lower bound > 0 |
| C3 | WARNING separates harmful from harmless shifts | AUROC ≥ 0.80 and harmless alert share ≤ 0.10 |
| C4 | WATCH beats the raw DEWS fixed score | higher T1 and T2 on ≥ 70% of confirmation datasets |
| C5 | WATCH carries over to natural streams | T2 difference vs label-trend ≥ 0 |

**Overall verdict:**
- **Success:** C1 and C3 pass.
- **Partial success:** not success, and C1 or C4 passes.
- **Failure:** C1 and C4 both fail.

**Expected from development:** C1 and C4 likely pass; C2 and C3 likely fail. This expectation is written down before the confirmation run.

## 5. Frozen development choices (2026-11-09)

| File | sha256 (first 16) | Content |
|---|---|---|
| `artifacts/v7/frozen_v7.json` | 4a9ab970cb58fad9 | thresholds: WATCH 4.011 z, WARNING 0.164; raw DEWS score and label trend reuse their frozen v6 thresholds |
| `artifacts/v7/frozen_v7_calibrator.pkl` | dab467db1cc2045e | WARNING calibrator trained on all development episodes |

Development (pooled; WARNING leave-one-dataset-out):

| Method | T1 | T2 | False alerts / 100 | Harmless shifts alerting | Harm AUROC |
|---|---|---|---|---|---|
| WATCH | 0.84 | 0.97 | 2.3 | 0.89 | 0.55 |
| WARNING | 0.49 | 0.86 | 4.0 | 0.38 | 0.59 |

## 6. Deviations

| Date | Change | Reason | Effect |
|---|---|---|---|
| 2026-11-09 | MiniBooNE rows are shuffled once with `numpy.random.default_rng(2026).permutation` before blocks are drawn | Its rows are stored sorted by class (one label change in 130,064 rows), so no contiguous block contains both classes. MiniBooNE is not time-ordered, so the shuffle removes no temporal structure. Done before any confirmation episode was built | None on the frozen choices; no outcome was looked at |
