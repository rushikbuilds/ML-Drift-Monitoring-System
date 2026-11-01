"""Does the final monitor separate harmless from harmful shifts?

    python scripts/run_harmless_vs_harmful.py

Uses fresh synthetic seeds 300-349 (known scenario labels). The anchored estimator
and all alert thresholds are fit on the 8 real development streams only, so the
synthetic episodes are an out-of-domain test. Reports, per scenario, the share of
post-intervention batches that raise an alert and the event-level outcome.
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
from dews.anchored import AnchoredEstimator, aux_signals, persist  # noqa: E402
from scripts.run_v2_experiment import pick_threshold  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
SEEDS = tuple(range(300, 350))
METHODS = ["fixed_score", "stat_only", "delayed_label_only", "anchored_p2"]


def job(args):
    warnings.filterwarnings("ignore")
    ep = v2.build_episode(*args)
    sig = v2.extract_signals(ep)
    sig["aux"] = aux_signals(ep)
    return sig


def scores(sig, delay, est):
    sc = v2.baseline_scores(sig, delay)
    sc.update(est.drop_scores(sig))
    out = {m: np.asarray(sc[m], float).copy() for m in METHODS}
    for v in out.values():
        v[: v2.WARMUP] = np.nan
    return out


def main():
    cache = OUT / "synthetic_aux_signals.pkl"
    if cache.exists():
        syn = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor(12) as ex:
            syn = list(ex.map(job, [(sc, s) for sc in v2.SCENARIOS for s in SEEDS], chunksize=2))
        cache.write_bytes(pickle.dumps(syn))
    dev = pickle.loads((OUT / "real_aux_signals.pkl").read_bytes())
    for s in dev:
        o = v2.outcome(s["accuracy"])
        s["event_onset"], s["targets"] = o["event_onset"], v2.state_targets(s["accuracy"])
    results = {}
    for delay in (2, 5):
        est = AnchoredEstimator(delay).fit(dev)
        dev_items = [dict(s, scores=scores(s, delay, AnchoredEstimator(delay).fit([d for d in dev if d["scenario"] != s["scenario"]]))) for s in dev]
        thr = {m: pick_threshold(dev_items, m)[0] for m in METHODS}
        rows = {}
        for scn in v2.SCENARIOS:
            eps = [s for s in syn if s["scenario"] == scn]
            per = {}
            for m in METHODS:
                share, any_alert, drop_median = [], [], []
                for s in eps:
                    sc = scores(s, delay, est)[m]
                    start = max(s["onset"] if s["onset"] is not None else v2.WARMUP, v2.WARMUP)
                    post = sc[start:]
                    post = post[np.isfinite(post)]
                    share.append(float(np.mean(post >= thr[m])))
                    any_alert.append(bool(np.any(post >= thr[m])))
                per[m] = {"share_of_batches_alerting": float(np.mean(share)), "episodes_with_any_alert": float(np.mean(any_alert))}
            rows[scn] = per
        results[str(delay)] = {"thresholds": thr, "by_scenario": rows}
    (OUT / "harmless_vs_harmful.json").write_text(json.dumps(results, indent=2))
    for delay, r in results.items():
        print(f"\nL={delay}: share of post-intervention batches that alert / episodes with any alert")
        print(f"{'scenario':16s}" + "".join(f"{m:>26s}" for m in METHODS))
        for scn, per in r["by_scenario"].items():
            print(f"{scn:16s}" + "".join(f"{per[m]['share_of_batches_alerting']:>14.2f} / {per[m]['episodes_with_any_alert']:<8.2f}" for m in METHODS))


if __name__ == "__main__":
    main()
