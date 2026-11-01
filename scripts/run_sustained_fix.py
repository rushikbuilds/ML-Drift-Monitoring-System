"""Sustained-alert rule to reduce harmless-shift alarms.

    python scripts/run_sustained_fix.py

An alert fires at batch t only if at least m of the last w batches exceed the
threshold. Candidate rules are fixed in advance; the rule is chosen on synthetic
VALIDATION seeds (140-159) and then reported on fresh seeds 300-349 and on the real
development / held-out streams (to confirm detection is not broken). The estimator
and thresholds are fit on the 8 real development streams only.
Needs: run_final_evaluation.py and run_harmless_vs_harmful.py caches.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from dews.anchored import AnchoredEstimator, aux_signals  # noqa: E402
from scripts.run_harmless_vs_harmful import job as syn_job  # noqa: E402
from scripts.run_v2_experiment import episode_eval, pick_threshold  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
RULES = [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)]
VALID_SEEDS = tuple(range(140, 160))
DELAYS = (2, 5)


def sustained(score, thr, w, m):
    hit = np.where(np.isfinite(score), score >= thr, False).astype(int)
    out = np.zeros(len(hit))
    for t in range(len(hit)):
        out[t] = float(hit[max(0, t - w + 1):t + 1].sum() >= m)
    return out


def anchored_score(sig, est):
    s = est.drop_scores(sig)["anchored_p2"].copy()
    s[: v2.WARMUP] = np.nan
    return s


def prep(sigs):
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"], s["targets"] = o["event_onset"], v2.state_targets(s["accuracy"])
    return sigs


def synthetic_metrics(sigs, est, thr, w, m):
    out = {}
    for scn in v2.SCENARIOS:
        eps = [s for s in sigs if s["scenario"] == scn]
        share, anyal, matched, early = [], [], [], []
        for s in eps:
            a = sustained(anchored_score(s, est), thr, w, m)
            start = max(s["onset"] if s["onset"] is not None else v2.WARMUP, v2.WARMUP)
            share.append(float(a[start:].mean()))
            anyal.append(bool(a[start:].any()))
            if s["event_onset"] is not None:
                r = episode_eval(a, s["event_onset"], len(a), 0.5)
                matched.append(r["matched_event_count"] > 0)
                early.append(r["warning_lead_batches"] is not None)
        out[scn] = {"share_alerting": float(np.mean(share)), "episodes_with_alert": float(np.mean(anyal)),
                    "event_recall": float(np.mean(matched)) if matched else None,
                    "early_warning_rate": float(np.mean(early)) if early else None}
    return out


def real_metrics(sigs, est, thr, w, m):
    rows = []
    for s in sigs:
        a = sustained(anchored_score(s, est), thr, w, m)
        r = episode_eval(a, s["event_onset"], len(a), 0.5)
        tta = -r["warning_lead_batches"] if r["warning_lead_batches"] is not None else r["detection_lag_batches"]
        rows.append((s["event_onset"] is not None, r["matched_event_count"] > 0, r["warning_lead_batches"] is not None,
                     tta, r["false_alerts_per_100_clean_batches"]))
    ev = [r for r in rows if r[0]]
    ttas = [r[3] for r in ev if r[3] is not None]
    return {"recall": float(np.mean([r[1] for r in ev])), "early_warning_rate": float(np.mean([r[2] for r in ev])),
            "mean_time_to_alert": float(np.mean(ttas)) if ttas else None,
            "false_alerts_per_100_clean": float(np.mean([r[4] for r in rows if r[4] is not None]))}


def main():
    dev = prep(pickle.loads((OUT / "real_aux_signals.pkl").read_bytes()))
    held = prep(pickle.loads((OUT / "new_signals.pkl").read_bytes()))
    test_syn = prep(pickle.loads((OUT / "synthetic_aux_signals.pkl").read_bytes()))
    vcache = OUT / "synthetic_valid_aux_signals.pkl"
    if vcache.exists():
        val_syn = pickle.loads(vcache.read_bytes())
    else:
        with ProcessPoolExecutor(12) as ex:
            val_syn = list(ex.map(syn_job, [(sc, s) for sc in v2.SCENARIOS for s in VALID_SEEDS], chunksize=2))
        vcache.write_bytes(pickle.dumps(val_syn))
    prep(val_syn)
    dev_names = sorted({s["scenario"] for s in dev})
    results = {"rules": RULES, "by_delay": {}}
    for delay in DELAYS:
        est = AnchoredEstimator(delay).fit(dev)
        folds = {d: AnchoredEstimator(delay).fit([s for s in dev if s["scenario"] != d]) for d in dev_names}
        items = []
        for s in dev:
            sc = anchored_score(s, folds[s["scenario"]])
            items.append(dict(s, scores={"anchored_p2": sc}))
        thr = pick_threshold(items, "anchored_p2")[0]
        per_rule = {}
        for w, m in RULES:
            key = f"{m}of{w}"
            per_rule[key] = {"validation_synthetic": synthetic_metrics(val_syn, est, thr, w, m),
                             "test_synthetic": synthetic_metrics(test_syn, est, thr, w, m),
                             "real_held_out": real_metrics(held, est, thr, w, m),
                             "real_development_lodo": real_metrics_lodo(dev, folds, thr, w, m)}
        # choose on validation: lowest harmless episode-alert rate subject to harmful recall >= 0.9
        def ok(k):
            v = per_rule[k]["validation_synthetic"]
            return all(v[s]["event_recall"] >= 0.9 for s in ("harmful_gradual", "harmful_abrupt", "concept_gradual"))
        feasible = [f"{m}of{w}" for w, m in RULES if ok(f"{m}of{w}")]
        if not feasible:  # no rule keeps harmful recall >= 0.9; fall back to the current rule
            feasible = ["1of1"]
        chosen = min(feasible, key=lambda k: (per_rule[k]["validation_synthetic"]["harmless_shift"]["episodes_with_alert"],
                                              per_rule[k]["validation_synthetic"]["harmless_shift"]["share_alerting"]))
        results["by_delay"][str(delay)] = {"threshold": float(thr), "chosen_on_validation": chosen, "per_rule": per_rule}
        print(f"\nL={delay} threshold={thr:.3f} chosen on validation: {chosen}")
        print(f"{'rule':7s} | harmless any-alert / share | harmful_grad recall / early | abrupt recall | concept recall | real held-out recall / FA | dev recall / FA")
        for k, v in per_rule.items():
            t = v["test_synthetic"]
            print(f"{k:7s} | {t['harmless_shift']['episodes_with_alert']:.2f} / {t['harmless_shift']['share_alerting']:.2f}"
                  f" | {t['harmful_gradual']['event_recall']:.2f} / {t['harmful_gradual']['early_warning_rate']:.2f}"
                  f" | {t['harmful_abrupt']['event_recall']:.2f} | {t['concept_gradual']['event_recall']:.2f}"
                  f" | {v['real_held_out']['recall']:.2f} / {v['real_held_out']['false_alerts_per_100_clean']:.1f}"
                  f" | {v['real_development_lodo']['recall']:.2f} / {v['real_development_lodo']['false_alerts_per_100_clean']:.1f}")
    (OUT / "sustained_fix.json").write_text(json.dumps(results, indent=2))


def real_metrics_lodo(dev, folds, thr, w, m):
    rows = []
    for s in dev:
        a = sustained(anchored_score(s, folds[s["scenario"]]), thr, w, m)
        r = episode_eval(a, s["event_onset"], len(a), 0.5)
        tta = -r["warning_lead_batches"] if r["warning_lead_batches"] is not None else r["detection_lag_batches"]
        rows.append((s["event_onset"] is not None, r["matched_event_count"] > 0, r["warning_lead_batches"] is not None,
                     tta, r["false_alerts_per_100_clean_batches"]))
    ev = [r for r in rows if r[0]]
    ttas = [r[3] for r in ev if r[3] is not None]
    return {"recall": float(np.mean([r[1] for r in ev])), "early_warning_rate": float(np.mean([r[2] for r in ev])),
            "mean_time_to_alert": float(np.mean(ttas)) if ttas else None,
            "false_alerts_per_100_clean": float(np.mean([r[4] for r in rows if r[4] is not None]))}


if __name__ == "__main__":
    main()
