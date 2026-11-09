# v10 Protocol: Fewer Harmless-Shift Alarms (experiment branch `v10-harmless-remedies`)

**Frozen on 2026-11-09, before any confirmation dataset was downloaded.** It is adopted into the main project only if the one-time confirmation succeeds; otherwise the branch stays unmerged.

## Why

The v8 harm tier alerted on 34% of harmless shifts. A development-only diagnosis showed this was mostly accumulated background false alarms: the estimate's batch-to-batch noise (about 0.8 accuracy points) crossing a low threshold (1.15 points) over about 33 batches. Clean streams alerted just as often as harmless ones.

## Remedies tested on development data (leave-one-dataset-out)

Every remedy's threshold was chosen on 7 development datasets to minimize harmless alarms subject to warning before labels confirm (T2) on at least 60% of harmful shifts, with at most 5 false alerts per 100 clean batches, and was then tested on the 8th dataset (`scripts/run_v10_dev.py`).

| Remedy | Harmless alarms | T2 | T1 (before the drop) | False alerts / 100 |
|---|---|---|---|---|
| v8 rule, stricter threshold | 0.15 | 0.57 | 0.35 | 1.0 |
| Exponential smoothing (0.3, 0.5) | 0.14–0.15 | 0.59–0.62 | 0.41 | 0.9–1.1 |
| Rolling mean (3, 5 batches) | 0.14–0.16 | 0.60–0.67 | 0.39–0.41 | 0.4–0.6 |
| **Stream-normalized (z-score)** | **0.13** | **0.60** | 0.29 | 0.0 |
| CUSUM (k = 0.5, 1.0, 1.5) | 0.18–0.24 | 0.60–0.62 | 0.29 | 0.0–0.2 |
| Any of the above, gated by drift watch | 0.14–0.25 | 0.60–0.63 | 0.31–0.37 | 0.0 |

**Finding:** every remedy reaches about the same point. Fewer harmless alarms come from a stricter operating point, and they cost early warning (T1 falls from about 0.61 to about 0.3 on development). Smoothing and CUSUM add little. **Development gate** (harmless ≤ 0.15, T2 ≥ 0.60, false alerts ≤ 5): passed by the z-score rule.

## Frozen choice

- **Rule:** z-score. The v8 importance-weighted accuracy drop is divided by its own burn-in noise scale (1.4826 × MAD of the first 8 batch estimates, floor 0.003).
- **Threshold:** 6.33, chosen on pooled development episodes with the objective above.
- **File:** `artifacts/v10/frozen_v10.json`, sha256 **c30860f3adda1561**….

Pooled development at this threshold: harmless 0.12, T2 0.60, T1 0.27, false alerts 0.0.

## Confirmation (run once)

- **Datasets:** six never used in this project: road-safety, Diabetes130US, jannis, rl, fried, jungle_chess (OpenML 42803, 4541, 41168, 41160, 901, 41027). Target is majority class vs rest; first 20 non-constant features.
- **Design:** v6 episode design; seeds 11000 + 100·index + episode; logistic regression, label delay 5.

**Success, all three required:**
- harmless-shift alarm share ≤ **0.15**;
- warned before labels confirm (T2) on ≥ **0.50** of harmful shifts;
- false alerts ≤ **5** per 100 clean batches.

The early-warning share before the drop (T1) and the v8 rule on the same data are reported alongside.

## Deviations

| Date | Change | Reason | Effect |
|---|---|---|---|
| 2026-11-09 | rl rows are shuffled once with `numpy.random.default_rng(2026).permutation` (now part of `run_v10.py download`) | Its rows are stored nearly sorted by class (19 label changes in 31,406 rows), so blocks were not representative; same precedent as MiniBooNE in v7. Done before any confirmation episode was built | None on the frozen choice; no outcome was looked at |
