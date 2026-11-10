# v9 (not adopted): Nonlinear Harm Estimators

Experiment branch `v9-harm-experiment`, not merged. Run on 2026-11-09 to see whether harm separation could clear the 0.80 AUROC target that v8 missed (0.78).

**Rule fixed before running:** continue to a fresh-data confirmation only if a candidate reaches harmful-vs-harmless AUROC ≥ 0.80 on the 8 development datasets. Otherwise stop, and leave the main project unchanged.

**Candidates** (`dews/harm_estimation_v9.py`):
- **A.** Importance-weighted reference accuracy with a gradient-boosting domain classifier, instead of v8's linear one.
- **B.** An error predictor: gradient boosting on features plus model confidence, trained on the model's out-of-fold correctness on the reference.

**Development result** (`scripts/run_v9_gate.py`, 63 harmful vs 288 harmless shift episodes):

| Estimator | AUROC |
|---|---|
| v8 importance-weighted, linear (frozen) | **0.767** |
| A: importance-weighted, gradient boosting | 0.702 |
| B: error predictor | 0.699 |

**Gate failed.** Both nonlinear estimators were worse than the frozen v8 method overall, though each was better on some datasets: B on covertype (0.81), A on led (0.93). Both collapsed on kddcup99 (0.43–0.45). More flexible models over-fit the 2000-row reference and do not transfer across datasets: the same pattern as every learned combination tried earlier.

**Ensembles** (post hoc, recorded 2026-11-10; `scripts/run_posthoc.py`, `artifacts/posthoc/v9_ensembles.json`). Every combination of 2–5 of {v8 importance-weighted, kNN reference accuracy, A, B, confidence drop}, combined by averaging ranks within each dataset. The best reached 0.760 (v8 + B + confidence), below the frozen v8 estimator's 0.767 and the 0.80 gate. Scored the same way (within-dataset ranks), v8 alone is 0.746, so the best ensemble is slightly better than v8 on that basis, but no ensemble came near the gate.

No confirmation data was downloaded or used. The v8 verdict and the main project are unchanged.
