"""v9 development gate: does a candidate reach harmful-vs-harmless AUROC >= 0.80 on development data?

    python scripts/run_v9_gate.py

Same metric as the v8 verdict (mean estimated accuracy drop over the first 3 post-onset
batches, harmful vs harmless shift episodes), on the 8 development datasets only.
Writes artifacts/v9/gate.json. Rule fixed before running: continue only if the best
candidate reaches 0.80; otherwise stop and leave the experiment unmerged.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v6_eval as ev  # noqa: E402
from dews.harm_estimation import estimated_drop  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v8_extract import load as load_v8  # noqa: E402
from scripts.run_v9_extract import load as load_v9  # noqa: E402

GATE = 0.80
BURN = 8


def main() -> None:
    eps = load_signals("development", "logistic_regression")
    v8, v9 = load_v8("development"), load_v9("development")
    if len(v9) != len(eps):
        raise SystemExit(f"v9 development signals incomplete: {len(v9)} of {len(eps)}")
    masks = ev.build_masks(eps, 5)
    candidates = {
        "v8 importance-weighted (linear), frozen": lambda e: estimated_drop(v8[e]["iw_accuracy"], BURN),
        "A: importance-weighted, gradient boosting": lambda e: estimated_drop(v9[e]["gbm_iw_accuracy"], BURN),
        "B: error predictor": lambda e: estimated_drop(v9[e]["error_model_accuracy"], BURN),
    }
    result = {}
    for name, fn in candidates.items():
        scores = np.vstack([fn((e["dataset"], e["seed"])) for e in eps])
        per = {}
        for d in sorted({e["dataset"] for e in eps}):
            idx = [i for i, e in enumerate(eps) if e["dataset"] == d]
            per[d] = ev.harm_auroc(scores[idx], ev.build_masks([eps[i] for i in idx], 5))
        result[name] = {"auroc": ev.harm_auroc(scores, masks), "per_dataset": per}
        print(f"{name:45s} AUROC {result[name]['auroc']:.3f}  " +
              "  ".join(f"{d[:6]} {('n/a' if a is None else f'{a:.2f}')}" for d, a in per.items()))
    best = max((k for k in candidates if not k.startswith("v8")), key=lambda k: result[k]["auroc"])
    passed = result[best]["auroc"] >= GATE
    print(f"\nBest candidate: {best} ({result[best]['auroc']:.3f}); gate {GATE}: {'PASS - continue' if passed else 'FAIL - stop'}")
    (ROOT / "artifacts" / "v9" / "gate.json").write_text(json.dumps({"gate": GATE, "best": best, "passed": passed, "results": result}, indent=2))


if __name__ == "__main__":
    main()
