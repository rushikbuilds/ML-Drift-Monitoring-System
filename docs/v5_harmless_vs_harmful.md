# v5: Harmless vs Harmful Shift Triage

Scripts: `run_harmless_vs_harmful.py`, `run_sustained_fix.py`, `run_relevance_gate.py`, `run_relevance_real.py`. Results: `artifacts/v2/harmless_vs_harmful.json`, `sustained_fix.json`, `relevance_gate.json`, `rws_real_results.json`.

## Question

Does the monitor tell apart a shift that hurts the model from one that does not?

## Findings

| Test | Setting | Result |
|---|---|---|
| Final anchored monitor | Synthetic, 50 fresh seeds | Alerts on 57–95% of post-shift batches when harmful, 11–19% when harmless, but at least once in **100%** of harmless episodes. Partial separation only. |
| Sustained-alert rule (m of w batches) | Synthetic + real | **Does not fix it.** At L = 2 the best rule cuts harmless episodes with an alert from 100% to 92%, and stricter rules cost large recall (real held-out 0.92 → 0.55). At L = 5 no rule helps. The cause is the signal, not the alert rule: auxiliary models use every feature, so irrelevant-feature shifts move them. |
| Relevance-weighted shift (RWS = Σ importance_j × KS_j) | Synthetic, 250 fresh episodes | **Perfect separation:** harmful 100% of episodes alert, harmless 0%, clean 0%; AUROC 1.0 (unweighted KS: 0.5). Irrelevant features carry ~0.03 importance vs 0.13–0.33 for relevant ones. |
| RWS | 13 real streams | **Not supported.** Macro Spearman with error 0.10 (development) and 0.04 (held-out); AUROC for the state target 0.67 / 0.53, against 0.70 / 0.60 for the fixed DEWS score. RWS beats the unweighted mean KS on only 4/13 streams. Adding RWS to the anchored estimator leaves nowcast error unchanged (differences ≤ 0.06 points). |

## Interpretation

- **Relevance weighting works when harmlessness is "this feature does not matter".** That is a clean property of the controlled generator.
- **On natural streams it does not carry over.** Real accuracy loss is largely driven by changes in the label relationship or in how the model uses features, not by shifts in the inputs the reference-trained forest ranks as important, so input-side relevance, even weighted, tracks it poorly.
- **Concept drift is invisible to any input-only score.** Only delayed labels reveal it.

## What to claim

- **Supportable:** "Feature-relevance weighting separates harmless from harmful input shifts in controlled settings; on real streams the effect is not demonstrated."
- **Not supportable:** "The system correctly identifies harmful and harmless shifts."
- **Practical use:** show relevance-weighted shift as an advisory explanation in the dashboard ("shift concentrated in features the model barely uses"), not as an automatic alert filter.
