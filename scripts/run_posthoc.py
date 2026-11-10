"""Post hoc analyses reported in the paper (explanatory; no verdict changes).

    python scripts/run_posthoc.py

1. Label-free baselines on the v8 confirmation data, scored exactly like hypothesis D1
   (mean score over the first 3 batches after onset, harmful vs harmless shifts), with
   paired bootstrap CIs for the harm tier minus each baseline.
   -> artifacts/posthoc/v8_baselines.json
2. Ensembles of harm estimators on the v9 development data (needs the v9 extraction
   pickles in artifacts/v9/, produced on branch v9-harm-experiment). Scores are combined
   by averaging ranks within each dataset.
   -> artifacts/posthoc/v9_ensembles.json
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v6_eval as ev  # noqa: E402
from dews import v7_monitor as mon  # noqa: E402
from dews import v8_config as cfg  # noqa: E402
from dews.harm_estimation import estimated_drop  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v8_extract import load as load_v8  # noqa: E402

OUT = ROOT / "artifacts" / "posthoc"
DELAY, FRACTION = cfg.PRIMARY["label_delay"], cfg.PRIMARY["label_fraction"]
B = 2000


def window_values(scores: np.ndarray, m: ev.Masks, idx: np.ndarray) -> np.ndarray:
    return np.array([np.nanmean(scores[i, m.onset[i]:m.onset[i] + ev.HARM_WINDOW]) for i in idx])


def v8_baselines() -> dict:
    files = sorted((ROOT / "artifacts" / "v8").glob("signals_v8_confirmation_*of*.pkl"))
    eps = sorted([e for f in files for e in pickle.loads(f.read_bytes())], key=lambda e: (e["dataset"], e["seed"]))
    m = ev.build_masks(eps, DELAY)
    pos, neg = np.flatnonzero(m.harmful_gradual | m.harmful_abrupt), np.flatnonzero(m.harmless_shift)
    idx, y = np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = {
        "harm tier (importance-weighted reference accuracy, frozen)":
            np.vstack([estimated_drop(e["iw_accuracy"], cfg.BURN_IN_BATCHES) for e in eps]),
        "kNN reference accuracy": np.vstack([estimated_drop(e["knn_accuracy"], cfg.BURN_IN_BATCHES) for e in eps]),
        "drift watch (frozen)": np.vstack([mon.watch_score(e, DELAY, FRACTION) for e in eps]),
    }
    for k in ("disagreement", "confidence_drop", "entropy", "cbpe_drop"):
        scores[k] = np.vstack([mon.robust_z(mon.raw_signals(e, DELAY, FRACTION)[k]) for e in eps])
    vals = {n: np.nan_to_num(window_values(s, m, idx)) for n, s in scores.items()}
    harm = next(iter(vals))
    rng = np.random.default_rng(20261110)
    boots = []
    for _ in range(B):
        ix = np.r_[rng.integers(0, len(pos), len(pos)), len(pos) + rng.integers(0, len(neg), len(neg))]
        boots.append({n: roc_auc_score(y[ix], v[ix]) for n, v in vals.items()})
    out = {"harmful": int(len(pos)), "harmless": int(len(neg)), "methods": {}}
    for n, v in vals.items():
        a = [b[n] for b in boots]
        d = [b[harm] - b[n] for b in boots]
        out["methods"][n] = {"auroc": float(roc_auc_score(y, v)),
                             "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                             "harm_minus_this_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def v9_ensembles() -> dict | None:
    files = sorted((ROOT / "artifacts" / "v9").glob("harm9_development_*of*.pkl"))
    if not files:
        print("v9 extraction pickles not found; skipping the ensemble analysis")
        return None
    v9 = {(r["dataset"], r["seed"]): r for f in files for r in pickle.loads(f.read_bytes())}
    eps = load_signals("development", "logistic_regression")
    v8 = load_v8("development")
    m = ev.build_masks(eps, DELAY)
    pos, neg = np.flatnonzero(m.harmful_gradual | m.harmful_abrupt), np.flatnonzero(m.harmless_shift)
    idx, y = np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    key = lambda e: (e["dataset"], e["seed"])  # noqa: E731
    makers = {
        "v8 importance-weighted (frozen)": lambda e: estimated_drop(v8[key(e)]["iw_accuracy"], 8),
        "kNN reference accuracy": lambda e: estimated_drop(v8[key(e)]["knn_accuracy"], 8),
        "A: importance-weighted, gradient boosting": lambda e: estimated_drop(v9[key(e)]["gbm_iw_accuracy"], 8),
        "B: error predictor": lambda e: estimated_drop(v9[key(e)]["error_model_accuracy"], 8),
        "confidence drop": lambda e: mon.robust_z(mon.raw_signals(e, DELAY, 1.0)["confidence_drop"]),
    }
    cols = {n: window_values(np.vstack([fn(e) for e in eps]), m, idx) for n, fn in makers.items()}
    names = np.array([eps[i]["dataset"] for i in idx])

    def ranks(x):
        r = np.zeros_like(x)
        for d in np.unique(names):
            s = names == d
            r[s] = rankdata(x[s]) / s.sum()
        return r

    single = {n: {"auroc_raw": float(roc_auc_score(y, x)), "auroc_within_dataset_ranks": float(roc_auc_score(y, ranks(x)))}
              for n, x in cols.items()}
    ens = sorted(([float(roc_auc_score(y, np.mean([ranks(cols[c]) for c in combo], axis=0))), list(combo)]
                  for r in range(2, len(cols) + 1) for combo in itertools.combinations(cols, r)), reverse=True)
    return {"harmful": int(len(pos)), "harmless": int(len(neg)), "single": single,
            "ensembles_best_first": [{"auroc": a, "members": c} for a, c in ens],
            "best_ensemble_auroc": ens[0][0], "gate": 0.80}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    b = v8_baselines()
    (OUT / "v8_baselines.json").write_text(json.dumps(b, indent=2))
    print(f"v8 confirmation data ({b['harmful']} harmful, {b['harmless']} harmless):")
    for n, r in b["methods"].items():
        print(f"  {n:60s} {r['auroc']:.3f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]  "
              f"harm minus this [{r['harm_minus_this_ci95'][0]:+.2f}, {r['harm_minus_this_ci95'][1]:+.2f}]")
    e = v9_ensembles()
    if e:
        (OUT / "v9_ensembles.json").write_text(json.dumps(e, indent=2))
        print(f"v9 development data ({e['harmful']} harmful, {e['harmless']} harmless):")
        for n, r in e["single"].items():
            print(f"  {n:45s} raw {r['auroc_raw']:.3f}  within-dataset ranks {r['auroc_within_dataset_ranks']:.3f}")
        for r in e["ensembles_best_first"][:3]:
            print(f"  ensemble {r['auroc']:.3f}  {' + '.join(r['members'])}")


if __name__ == "__main__":
    main()
