"""Relevance-weighted shift: can feature relevance separate harmless from harmful shifts?

    python scripts/run_relevance_gate.py

For each synthetic episode a RandomForest is fit on the reference rows; its feature
importances weight the per-feature KS shift of each batch:
    RWS_t = sum_j importance_j * KS_j(t)
The unweighted mean KS is the comparison. The alert threshold is chosen on validation
seeds 140-159 (highest threshold with <= 5% harmless-or-clean episodes alerting and
all-batch sustained 2-of-2 rule) and reported on fresh seeds 300-349. Concept drift
leaves P(X) unchanged, so input-based scores cannot see it by construction.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402

OUT = ROOT / "artifacts" / "v2"


def job(args):
    warnings.filterwarnings("ignore")
    ep = v2.build_episode(*args)
    rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1)
    rf.fit(ep.reference, ep.reference_labels)
    imp = rf.feature_importances_
    ref = ep.reference.to_numpy()
    ks = np.array([[ks_2samp(ref[:, j], b.iloc[:, j].to_numpy()).statistic for j in range(ref.shape[1])] for b in ep.batches])
    return {"scenario": ep.scenario, "onset": ep.onset, "rws": ks @ imp, "mean_ks": ks.mean(axis=1),
            "importance": imp.tolist()}


def collect(seeds, ex):
    return [job((sc, s)) for sc in v2.SCENARIOS for s in seeds]


def sustained2(score, thr):
    hit = score >= thr
    return np.array([hit[max(0, t - 1):t + 1].all() for t in range(len(hit))])


def rates(eps, key, thr):
    out = {}
    for scn in v2.SCENARIOS:
        any_alert = []
        for e in (x for x in eps if x["scenario"] == scn):
            start = e["onset"] if e["onset"] is not None else v2.WARMUP
            a = sustained2(e[key], thr)[max(start, v2.WARMUP):]
            any_alert.append(bool(a.any()))
        out[scn] = float(np.mean(any_alert))
    return out


def main():
    val = collect(range(140, 160), None)
    test = collect(range(300, 350), None)
    report = {}
    for key in ("rws", "mean_ks"):
        # separation: post-onset mean score, harmful-input-shift vs harmless
        def post_mean(e):
            start = e["onset"] if e["onset"] is not None else v2.WARMUP
            return float(np.mean(e[key][max(start, v2.WARMUP):]))
        pos = [post_mean(e) for e in test if e["scenario"] in ("harmful_gradual", "harmful_abrupt")]
        neg = [post_mean(e) for e in test if e["scenario"] in ("harmless_shift", "clean")]
        auc = float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))
        # threshold: smallest value with <=5% of validation harmless/clean episodes alerting
        grid = np.unique(np.quantile(np.concatenate([e[key] for e in val]), np.linspace(0.5, 1.0, 150)))
        chosen = None
        for thr in grid:
            r = rates(val, key, thr)
            if max(r["clean"], r["harmless_shift"]) <= 0.05:
                chosen = float(thr)
                break
        report[key] = {"episode_level_auroc_harmful_vs_harmless_or_clean": auc, "threshold": chosen,
                       "test_episodes_with_sustained_alert": rates(test, key, chosen) if chosen is not None else None}
    imp = np.mean([e["importance"] for e in test], axis=0)
    report["mean_rf_importance_by_feature"] = dict(zip(v2.FEATURES, [round(float(x), 3) for x in imp]))
    (OUT / "relevance_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
