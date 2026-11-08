# v6 Protocol: Early-Warning Benchmark on Real Data with Controlled Shifts

**Frozen on 2026-11-08, before any v6 result exists.** The machine-readable version is [dews/v6_config.py](../dews/v6_config.py); `tests/test_v6_config.py` checks it. If this document and the config disagree, the config wins. Plan context: [final_two_week_plan.md](final_two_week_plan.md).

## 1. Question

Can the Drift Early-Warning System warn about model degradation (a) before accuracy actually drops and (b) before delayed labels could confirm the drop, and can it tell shifts that hurt the model from shifts that do not?

## 2. Data

13 cached binary datasets. The held-out set was drawn once with `numpy.random.default_rng(2026).choice(sorted(names), 5)`:

| Split | Datasets |
|---|---|
| Development (tuning allowed) | covertype, covpokelec, hyperplane, kddcup99, led, poker_hand, run_or_walk, vehicle_sensit |
| **Held-out (run once)** | airlines, bank_marketing, electricity, ldpa, sea |

## 3. Episode design

Each episode is built from one random contiguous block of a dataset, chosen by the episode seed and redrawn until both classes make up at least 5% of rows.

1. **Reference:** the first 2000 rows. The monitored model is fit on rows 1–1500. Rows 1501–2000 are used only for permutation importance of the monitored model. All 2000 rows form the detector reference.
2. **Pool:** the next 20,000 rows of the same block. Because reference and pool come from one contiguous period, a clean stream has no built-in drift.
3. **Stream:** 45 batches of 300 rows, drawn *with replacement* from the pool. Each row keeps its true label.
4. **Intervention onset:** a random batch from 10 to 16.

| Type | Episodes per dataset | Construction after onset |
|---|---:|---|
| clean | 10 | uniform sampling |
| gradual_shift | 30 | sampling weight ∝ exp(λ(t)·s·z_j), λ ramping linearly from 0 to 1.5 over 15 batches |
| abrupt_shift | 15 | same, λ = 1.5 from onset |
| concept_change | 10 | uniform sampling; labels flipped with probability ramping to 0.3 over 15 batches, for rows with z_k > 0 |

- z_j is feature j standardised with reference statistics, and s = ±1 is a random direction.
- The shifted feature j is drawn from the top half of features by permutation importance of the monitored model with probability 0.5, otherwise from the bottom half.
- The concept-change feature k is drawn uniformly.
- Episode seeds are 1000 + 100·(dataset index) + episode index for development and 5000 + … for held-out.

**Outcome labels**, from the true accuracy of every batch:

- **Event:** the first of 2 consecutive batches at least 5 points below the mean of batches 0–4.
- **Harmful shift episode:** a shift episode (gradual or abrupt) whose event starts at or after the intervention onset.
- **Harmless shift episode:** a shift episode with no event.
- **Spurious episode:** any episode with an event before the intervention onset, and any clean episode with an event. These are excluded from H1–H3 and their count is reported.

## 4. Settings

- Label delay L ∈ {0, 2, 5, 10} batches. Labels of batch b are usable from batch b + max(L, 1).
- Labeled fraction ∈ {100%, 20%}. The monitor sees the accuracy of a random 20% subsample of each batch.
- Monitored model: logistic regression (primary), gradient boosting (robustness).
- **Primary setting:** logistic regression, L = 5, 100% labels.

## 5. Methods

**Primary system (frozen):** a delay-aware risk calibrator.
- Logistic regression on the state target: a confirmed degraded run starts or continues within the next 5 batches.
- Features:
  - current, 3-batch mean and 3-batch slope of the drift and confidence signals;
  - matured-label accuracy-drop features (observed labels only);
  - disagreement of the monitored model with auxiliary random-forest and gradient-boosting models, current and change;
  - relevance-weighted shift Σ importance_j × KS_j, using the monitored model's permutation importance.
- It is trained on development episodes separately for each (model, delay, labeled fraction).
- A **WARNING** is raised when risk ≥ threshold.

**Baselines:**
- the original DEWS fixed drift score;
- statistical-only and confidence-only scores;
- unweighted mean KS;
- label-trend (observed matured accuracy drop);
- delay-aware CBPE.

**Allowed tuning on development data only:**
- one alert threshold per method and setting;
- λ_max, through the Day 2 gate.

**Threshold rule:** among thresholds keeping development false alerts ≤ 5 per 100 clean batches, choose the one with the highest T1 share on development harmful episodes. Ties go to the higher threshold.

## 6. Metrics

Alerts before batch 6 are ignored for every method.

**Clean batches** are all batches of clean episodes plus pre-onset batches of every other episode. False alerts are counted per 100 clean batches.

| Tier | Counted when |
|---|---|
| T1 predictive | an alert falls in [intervention onset, event onset); lead = event onset − first such alert |
| T2 pre-confirmation | an alert falls in [intervention onset, confirmation), confirmation = event onset + 1 + max(L, 1) |
| T3 confirmed | first alert at or after confirmation; lag = alert − event onset |

- **Harmless alert share:** the share of harmless shift episodes with any WARNING after onset.
- **Harm score for H3:** the mean risk over the first 3 batches after onset.
- **Uncertainty:** 95% bootstrap intervals over episodes, 2000 resamples. For H2, the bootstrap is paired, resampling episodes once for both methods.

## 7. Hypotheses and PASS criteria (held-out datasets, primary setting)

| ID | Claim | PASS if |
|---|---|---|
| H1 | Predictive warning for gradual harmful shifts | T1 share ≥ 0.60 AND median lead ≥ 2 batches, with clean false alerts ≤ 5 per 100 |
| H2 | Earlier than waiting for labels | T2(system) − T2(label-trend) ≥ 0.10 AND paired 95% CI lower bound > 0 |
| H3 | Separates harmful from harmless shifts | AUROC ≥ 0.80 AND harmless alert share ≤ 0.10 |
| H4 | Beats the original DEWS fixed score | higher T1 and higher T2 than the fixed score on ≥ 4 of 5 held-out datasets |
| H5 | Carries over to natural streams | T2(system) − T2(label-trend) ≥ 0 on the 13 cached natural-stream window sets, L = 5 |

**Overall verdict:**
- **Success:** H1 and H3 pass.
- **Partial success:** not success, and H1 or H4 passes.
- **Failure:** H1 and H4 both fail.

Secondary results (other delays, 20% labels, gradient boosting, abrupt shifts, concept changes) are reported in full but do not change the verdict.

## 8. Gates

- **Day 2:** on development data, at least 25% of shift episodes must be harmful and at least 25% harmless. Otherwise λ_max is adjusted using development datasets only, and the change is recorded below.
- **Day 4:** if the development T1 share on harmful gradual shifts is below 0.40, the plan's fallback applies. The held-out run still happens, so the verdict is still produced.

## 9. Execution rules

- The held-out datasets are generated, extracted and scored **once**, after all development choices are committed.
- Every verdict number is produced by `scripts/make_verdict.py` from saved results.
- Outputs go to `artifacts/v6/`.

## 10. Deviations

None so far. Any change after 2026-11-08 is listed here with its date and reason.
