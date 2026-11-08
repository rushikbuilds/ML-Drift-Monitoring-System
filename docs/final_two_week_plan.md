# Final Two-Week Plan: ML Model Drift Early-Warning System

The project title stays **"ML Model Drift Early-Warning System"**. This plan defines early warning in testable terms, runs one decisive experiment around it, and ends with a pass/fail verdict for every hypothesis. A failure is reported as a failure.

## 1. Where the project stands (evidence so far)

| Finding | Status |
|---|---|
| Controlled synthetic drift: the risk calibrator warns before onset in 98% of gradual covariate-drift episodes, about 4 batches ahead | Positive ([v2_results.md](v2_results.md)) |
| Natural real streams: drift scores predict accuracy loss weakly; thresholds do not transfer | Negative ([v2_realdata.md](v2_realdata.md)) |
| Delay-anchored accuracy estimator: error down 5.5 / 12.5 / 15% vs stale labels at L = 0 / 2 / 5 | Positive, modest ([v4_results.md](v4_results.md)) |
| Harmless vs harmful shift: perfect separation in synthetic data, not supported on real streams | Mixed ([v5_harmless_vs_harmful.md](v5_harmless_vs_harmful.md)) |
| Check (8 Nov): share of events warned **before delayed labels could confirm them**, natural streams, L = 5, held-out | Label-trend 0.90, anchored 0.55, delay-aware CBPE 0.59, fixed DEWS score 0.02 |
| Check (8 Nov): same with only 20% or 5% of rows labeled | Label-trend falls to 0.63 / 0.33, but no label-free method overtakes it |

**Diagnosis.** On natural streams the accuracy changes are dominated by changes in the label relationship and by noise, and the onset of a shift is unknown, so "early" cannot be measured cleanly. In controlled data the system works. The missing piece is a **realistic but controlled** benchmark on real data, where the shift onset is known and harm is measured with real labels, not decided by construction.

## 2. What "early warning" means in this project

Three tiers, each with its own metric. Every alert is reported at the tier it reaches.

| Tier | Definition | Metric |
|---|---|---|
| **T1 Predictive warning** | Alert before the accuracy event begins (onset = first of 2 consecutive batches ≥ 5 points below baseline) | Share of events warned before onset; lead in batches |
| **T2 Pre-confirmation warning** | Alert before delayed labels could confirm the event (confirmation time = onset + 1 + L) | Share of events warned before confirmation; lead over confirmation |
| **T3 Confirmed degradation** | Alert after labels confirm it | Detection lag |

All tiers are measured at the same false-alert budget: **≤ 5 alerts per 100 clean batches**, with thresholds chosen on development data only.

## 3. Hypotheses and pass/fail criteria

These are frozen on Day 1, before any new result is seen.

| ID | Hypothesis | PASS if (on held-out datasets) |
|---|---|---|
| **H1** | The system gives predictive warnings (T1) for gradual shifts on real data | ≥ 60% of gradual harmful events warned before onset, median lead ≥ 2 batches, within budget |
| **H2** | It warns earlier than waiting for labels (T2) | T2 rate ≥ label-trend baseline + 10 points at L = 5, paired bootstrap 95% CI excludes 0 |
| **H3** | It tells harmful from harmless shifts | AUROC ≥ 0.80 for "will this shift cause an event?" at the first 3 post-shift batches; harmless-shift false alerts ≤ 10% of episodes |
| **H4** | It beats the original DEWS fixed drift score | Higher T1 and T2 rates than the fixed score at equal budget on ≥ 70% of datasets |
| **H5** | Results carry over to natural streams | On natural streams (13 already collected), T2 rate ≥ label-trend baseline at L = 5 |

The 8 Nov check already suggests that **H5 will likely fail**. It stays in the table so the outcome is shown honestly.

## 4. The decisive experiment: real data with injected shifts

**Idea.** Use real datasets, a real frozen model and real labels, but build each stream by **re-sampling** real rows so that the input distribution drifts in a known way from a known batch. Re-sampling changes P(X) while keeping each row's true label, so whether a shift hurts the model is *measured from the labels*, not assumed.

**Datasets.** The 13 already cached (electricity, airlines, covertype, bank-marketing, KDDCup99, ldpa, run_or_walk, vehicle_sensIT, hyperplane, SEA, CovPokElec, poker-hand, LED), plus 2–3 more tabular sets if time allows. Split: **8 development datasets, 5 locked held-out datasets**, fixed on Day 1.

**Shift types**, about 20 episodes per type per dataset:

| Type | How the stream is built | Expected outcome |
|---|---|---|
| Clean | Rows sampled uniformly from the stream pool | No event |
| Gradual covariate shift | Sampling weight ∝ exp(λ(t)·z_j) on feature j, with λ ramping over 15 batches | Event if the model is worse in that region; measured |
| Abrupt covariate shift | Same, with λ stepping at onset | Measured |
| Shift in an irrelevant feature | Same procedure on a feature with near-zero permutation importance for the monitored model | Usually harmless; measured |
| Label-relationship change | Flip labels in a sub-population after onset | Event; invisible to input-only signals (control case) |

Feature j is drawn at random, weighted half toward important and half toward unimportant features, so both harmful and harmless shifts occur naturally. Each episode is labeled harmful or harmless **by the event rule on real labels**.

**Settings.** Label delay L ∈ {0, 2, 5, 10}; labeled fraction {100%, 20%}; 300 rows per batch; 35 batches per episode; onset at batch 10–16 (random).

**Monitored models.** Logistic regression (main) and gradient boosting (robustness check).

## 5. System components

| Component | Status | Change |
|---|---|---|
| Drift signals (KS, PSI, MMD, confidence) | Existing | Keep |
| Auxiliary-model disagreement | Existing (`dews/anchored.py`) | Keep |
| **Model-based relevance**: permutation importance of the *monitored* model on a held-out reference slice | New | Replaces random-forest importance, a likely cause of the v5 real-data failure |
| Relevance-weighted shift | Existing script | Re-weight with model-based relevance |
| Risk calibrator (state target, delay-aware) | Existing (`dews/v2.py`) | Retrain on development datasets of the new benchmark |
| Delay-anchored accuracy estimator | Existing | Keep for T2 and T3 |
| **Two-stage alert policy**: Watch (risk ≥ t1) → Warning (risk ≥ t2, or relevance-weighted shift sustained) → Confirmed (labels) | New | Thresholds from development data |

**Baselines**, all at the same alert budget:
- the original DEWS fixed drift score;
- statistical-only and confidence-only scores;
- label-trend (waiting for delayed labels);
- delay-aware CBPE;
- unweighted mean KS.

## 6. Day-by-day schedule

**Week 1: build and run the decisive experiment**

| Day | Work | Output | Gate |
|---|---|---|---|
| 1 | Freeze the hypotheses and pass criteria (this document). Fix the dataset split and seeds. Write `docs/v6_protocol.md`. | Protocol committed **before** any new results | — |
| 2 | Build the re-sampling stream generator and model-based relevance. Unit tests: label-preservation, causality, onset metadata. | `dews/resample_benchmark.py`, tests | Sanity check: at least 25% of shift episodes become harmful and 25% stay harmless on development data. If not, adjust λ on **development data only**. |
| 3 | Signal extraction for all episodes (background run, cached). Fix the Windows process-pool issue by running through the background tool. | Signal caches | — |
| 4 | Train the calibrator and alert policy on development datasets with leave-one-dataset-out. Tune only here. | Development results | **Gate 1:** if H1 is clearly failing on development data (T1 < 40%), switch to the fallback in section 8 |
| 5 | Locked held-out run, **once**: all delays, labeled fractions and both models. | Held-out tables | — |

**Week 2: verdict, demo and presentation**

| Day | Work | Output |
|---|---|---|
| 6 | Verdict script: computes each hypothesis's metric, CI and PASS/FAIL automatically from saved results. Also re-scores H5 on the 13 natural streams. | `scripts/make_verdict.py`, `docs/verdict.md` |
| 7 | Ablations: without disagreement, without relevance, without the calibrator; batch size 150 vs 300; logistic vs gradient boosting. Failure analysis per dataset. | Ablation table |
| 8–9 | Dashboard (Streamlit): stream picker, delay and label-fraction sliders, timeline with true accuracy (revealed late), risk, Watch/Warning/Confirmed markers, relevance explanation, and a **Verdict** page showing the pass/fail table. | Working demo |
| 10 | Reproducibility: one command for each stage, pinned requirements, cleaned scripts (exploration scripts moved to `scripts/exploration/`), README rewrite around the three tiers and the verdict. | Release candidate |
| 11 | Presentation: 10–12 slides (problem → v1 failure and diagnosis → tiers → benchmark design → results → verdict → limits), 3-minute demo script, viva Q&A sheet. | Presentation kit |
| 12 | Rehearsal and fixes. | — |
| 13 | Buffer for reruns or slips. | — |
| 14 | Freeze: tag `v-final`, export artifacts and slides, back up the repository. | Final release |

## 7. Final deliverable: the verdict

`docs/verdict.md` and the dashboard Verdict page show, for every hypothesis:

| ID | Metric | Target | Result (95% CI) | Verdict |
|---|---|---|---|---|
| H1 | T1 rate, gradual harmful | ≥ 60% | … | PASS / FAIL |
| H2 | T2 rate − label-trend | ≥ +10 pts | … | PASS / FAIL |
| H3 | AUROC harmful vs harmless | ≥ 0.80 | … | PASS / FAIL |
| H4 | Datasets where the system beats the fixed score | ≥ 70% | … | PASS / FAIL |
| H5 | T2 on natural streams vs label-trend | ≥ 0 | … | PASS / FAIL (likely FAIL) |

Overall statement, chosen by the numbers:
- **Success**, if H1 and H3 pass: "The early-warning system works for gradual input shifts and separates harmful from harmless shifts on real data with controlled shifts; it does not replace labels on natural streams."
- **Partial success**, if only H1 or only H4 passes.
- **Failure**, if H1 fails: "Early warning was not achieved beyond controlled synthetic data; the contribution is the diagnosis, the delay-aware protocol and the benchmark."

## 8. Fallback if Gate 1 fails

- Keep the title, and report the system as an **early-warning evaluation framework**: the three-tier definition, the re-sampling benchmark, and the verdict that label-free early warning is limited to controlled synthetic drift.
- Shift the demo emphasis to T2/T3 monitoring with the delay-anchored estimator, which has a positive result.
- No extra method work after Day 5 in either case; week 2 is for the verdict and presentation.

## 9. Rules

- Hypotheses and pass criteria are frozen on Day 1 and never edited afterwards; any later change is listed as a deviation.
- Held-out datasets are run once.
- Every number in the verdict is produced by the verdict script from saved results.
- The original DEWS fixed score is always reported alongside, because it is the project's original baseline.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Re-sampling produces too few harmful (or harmless) shifts | Day 2 sanity gate; adjust λ on development data only |
| Compute time on a laptop | Cache signals; run in the background; about 1–2 hours for the full held-out run |
| Windows process-pool crashes | Use the tool's background mode, or serial runs |
| Results come out negative | The verdict reports it; the fallback keeps the project complete |
