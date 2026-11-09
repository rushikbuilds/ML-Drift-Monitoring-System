# Slides: ML Model Drift Early-Warning System

Twelve slides, about 12 minutes plus a 3-minute live demo. Every number traces to a committed verdict or analysis file, cited under each slide.

---

## 1. Title

**ML Model Drift Early-Warning System**
Warning about model degradation when labels arrive late.

*Speaker notes:* One sentence: a deployed model can degrade for days before anyone has the labels to notice. We built and tested a system that tries to warn earlier, and we report honestly where it works and where it doesn't.

---

## 2. The problem

- Models are trained on yesterday's data and run on today's.
- Accuracy can only be measured once true labels arrive, often days or weeks later (fraud, credit, churn).
- Drift detectors fire on any change in the input data, but most changes don't hurt the model.

**Question:** can a monitor warn *before* the labels confirm a drop, and tell harmful drift from harmless drift?

*Speaker notes:* Stress the two goals: warn early, and warn about the right thing.

---

## 3. First attempt failed, and why (v1)

- Original DEWS system: drift statistics fused into one score, plus a learned risk calibrator.
- Result: 0 of 20 degradation events detected.
- **Diagnosis:**
  - the synthetic scenarios gave no time to warn (the drop began on the same batch as the drift);
  - the risk target was inverted (the model learned "more drift = less risk").

*Speaker notes:* Present this as a strength: we found the cause instead of tuning around it. Source: `docs/v2_protocol.md`.

---

## 4. Research discipline

- **Pre-registered:** hypotheses and pass/fail thresholds were frozen in code before each run.
- **Held-out data run once;** later confirmations used datasets never touched before.
- **The verdict is computed by a script,** not chosen by us.
- **Every deviation is recorded** with date and reason.
- **Everything reproduces** in a clean environment (`scripts/reproduce.py`).

*Speaker notes:* This is why the failures in the next slides are informative rather than embarrassing.

---

## 5. The benchmark (v6)

- **Data:** real datasets with controlled shifts. Real rows are re-sampled so the inputs drift from a known batch.
- **Labels:** every row keeps its true label, so **whether a drift hurts is measured, not assumed**.
- **Scale:** 13 datasets, 8 for development and 5 held out; 1,690 episodes; label delays 0, 2, 5 and 10 batches.

*Speaker notes:* Explain "early" in two tiers: before the accuracy drop, and before labels could confirm it.

---

## 6. v6 verdict: FAILURE

- All five hypotheses failed on held-out data.
- **Why:** thresholds learned on some datasets don't transfer to others.
  - The original drift score's false alerts jumped from 4.4 to **36.9** per 100 batches (budget: 5).
  - The learned system under-alerted (1.2): it warned before the drop on only 22% of harmful drifts.

*Speaker notes:* Source: `docs/verdict.md`, `docs/ablations.md` section A.

---

## 7. Fix 1: stream normalization (v7, drift watch)

- **Measure every signal relative to the stream's own first 8 batches.** "4 standard deviations above normal" means the same thing on every dataset.
- **Confirmed once on 6 unseen datasets:**
  - **87%** of harmful drifts warned before accuracy drops;
  - **6 batches** median lead;
  - **4.4** false alerts per 100 batches (within budget);
  - beats the original score on 5 of 6 datasets.
- **But:** it alerts on 96% of harmless drifts. It's a drift detector, not a harm detector.

*Speaker notes:* Verdict: PARTIAL SUCCESS. Source: `docs/v7_verdict.md`.

---

## 8. Fix 2: estimating harm from the labeled reference (v8, harm tier)

- **Idea:** a drift is harmful when it moves data into regions where the model makes mistakes, and the labeled training sample shows where those are.
- **Method:** reweight the reference rows to look like the current batch, then estimate accuracy without new labels.
- **Confirmed once on 5 unseen datasets:**
  - harmless-drift alarms drop from **97% to 34%**;
  - **76%** of harmful drifts still warned before the drop;
  - harm separation AUROC **0.78** vs 0.59 for drift statistics.
- **Missed:** the 0.80 AUROC target and the 10% harmless-alarm target.

*Speaker notes:* Verdict: PARTIAL SUCCESS. Source: `docs/v8_verdict.md`.

---

## 9. Live demo

Dashboard → Live monitor. Script: `docs/presentation/demo_script.md`.

---

## 10. What we learned (ablations)

1. **Units matter more than models.** Raw drift scores broke the alert budget 4–10× on new data; scores relative to the stream's normal, or in accuracy points, held it.
2. **Drift signals and harm signals are different families.** Input-shift signals warn early but are blind to harm. Model-behaviour signals are harm-aware but warn later.
3. **Harm depends on where the model is wrong, not on which feature moved.** About 80% of realistic drifts were harmless, and feature importance barely predicts which.
4. **Normalization needs a stable start.** On natural streams already drifting at deployment, it fails.

*Speaker notes:* Source: `docs/ablations.md`.

---

## 11. Limitations

- Harm separation is good, not perfect: 34% of harmless drifts still alert. Two follow-up experiments (v9 better estimators, v10 alarm remedies) were tested and rejected on fresh data.
- Changes in the label relationship (concept drift) are invisible until labels arrive.
- Fails on natural streams that are already drifting when monitoring starts.
- Controlled shifts are realistic re-sampling, but still constructed; the main model is logistic regression.

---

## 12. Conclusion

- **Delivered:** a working two-tier early-warning system, a pre-registered benchmark, an interactive dashboard, and a fully reproducible pipeline.
- **Verdicts:** v6 FAILURE → v7 PARTIAL SUCCESS → v8 PARTIAL SUCCESS. Each step diagnosed the previous failure and was confirmed on fresh data.
- **One-line takeaway:** drift can be flagged early and reliably across datasets; telling whether it will hurt is possible only partly without labels, and needs the labeled reference.
