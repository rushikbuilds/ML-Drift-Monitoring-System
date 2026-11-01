"""Confirmation of the delay-aware CBPE monitor on data not used for its design.

    python scripts/run_v3_confirmation.py

The method was fixed on the v2 development data (synthetic seeds 100-249; real
datasets electricity, airlines, covertype, bank_marketing). This script evaluates
on (a) fresh synthetic seeds 300-349 and (b) four fresh OpenML streams:
run_or_walk, vehicle_sensit, kddcup99, ldpa. Alert thresholds come from the
development data only, plus a no-fitting "natural" threshold equal to the
5-point event definition for scores expressed in accuracy units.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from scripts.run_v2_experiment import boot_ci, episode_eval, pick_threshold  # noqa: E402
from scripts.run_v2_realdata import BATCH, NB, REF, SPAN, _numeric  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
FRESH_SEEDS = tuple(range(300, 350))
DELAYS = (0, 2, 5)
FRESH = {  # name: (openml id, columns to drop, max features, target rule)
    "run_or_walk": (40922, ("date", "time", "username"), 8, "binary"),
    "vehicle_sensit": (357, (), 20, "binary"),
    "kddcup99": (1113, (), 15, "normal_vs_attack"),
    "ldpa": (1483, (), 8, "majority_vs_rest"),
}
METHODS = ["stat_only", "conf_only", "fixed_score", "slope_forecast", "delayed_label_only",
           "cbpe_raw", "da_cbpe", "da_cbpe_forecast"]
ACCURACY_UNIT = ("cbpe_raw", "da_cbpe", "da_cbpe_forecast", "delayed_label_only")


def load(name):
    cache = OUT / f"fresh_{name}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    did, drop, kmax, rule = FRESH[name]
    try:
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = d.data, d.target.astype(str)
    except ValueError:  # sparse ARFF
        d = fetch_openml(data_id=did, as_frame=False)
        dense = d.data.toarray() if hasattr(d.data, "toarray") else d.data
        X = pd.DataFrame(dense[:, :kmax], columns=[f"f{i}" for i in range(min(kmax, dense.shape[1]))])
        y = pd.Series(d.target).astype(str)
    X = X.drop(columns=[c for c in drop if c in X.columns])
    if rule == "normal_vs_attack":
        yy = (y.str.strip(".") != "normal").astype(int).to_numpy()
    elif rule == "majority_vs_rest":
        yy = (y == y.value_counts().index[0]).astype(int).to_numpy()
    else:
        yy = (y == sorted(y.unique())[-1]).astype(int).to_numpy()
    X = _numeric(X).fillna(0.0)
    X = X.loc[:, X.nunique() > 1].iloc[:, :kmax].reset_index(drop=True)
    res = (X, yy)
    pd.to_pickle(res, cache)
    return res


def real_job(args):
    warnings.filterwarnings("ignore")
    name, start = args
    X, y = load(name)
    ref = X.iloc[start:start + REF]
    s = X.iloc[start + REF:start + SPAN]
    b = [s.iloc[i * BATCH:(i + 1) * BATCH].reset_index(drop=True) for i in range(NB)]
    lab = [y[start + REF + i * BATCH:start + REF + (i + 1) * BATCH] for i in range(NB)]
    ep = v2.Episode(name, start, ref.reset_index(drop=True), y[start:start + REF], b, lab, None)
    return v2.extract_signals(ep)


def syn_job(args):
    warnings.filterwarnings("ignore")
    return v2.extract_signals(v2.build_episode(*args))


def prep(sigs):
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"] = o["event_onset"]
        s["targets"] = v2.state_targets(s["accuracy"])
    return sigs


def with_scores(sig, delay):
    s = v2.baseline_scores(sig, delay)
    s.update(v2.cbpe_scores(sig, delay))
    s = {k: np.asarray(v, float).copy() for k, v in s.items() if k in METHODS}
    for v in s.values():
        v[: v2.WARMUP] = np.nan
    return s


def auroc(items, method):
    ys = np.concatenate([it["targets"] for it in items])
    ss = np.concatenate([it["scores"][method] for it in items])
    ok = np.isfinite(ys) & np.isfinite(ss)
    return float(roc_auc_score(ys[ok], ss[ok])) if len(set(ys[ok])) == 2 else None


def evaluate(items, thresholds, rng):
    out = {}
    for m in METHODS:
        thr = thresholds[m]
        rows = []
        for it in items:
            n = len(it["accuracy"])
            r = episode_eval(it["scores"][m], it["event_onset"], n, thr)
            rows.append({"event": it["event_onset"] is not None, "matched": r["matched_event_count"] > 0,
                         "early": r["warning_lead_batches"] is not None,
                         "fa": r["false_alerts_per_100_clean_batches"]})
        ev = [r for r in rows if r["event"]]
        out[m] = {
            "threshold": float(thr),
            "auroc": auroc(items, m),
            "recall": float(np.mean([r["matched"] for r in ev])) if ev else None,
            "recall_ci95": boot_ci([r["matched"] for r in ev], rng),
            "early_warning_rate": float(np.mean([r["early"] for r in ev])) if ev else None,
            "early_warning_ci95": boot_ci([r["early"] for r in ev], rng),
            "false_alerts_per_100_clean": float(np.mean([r["fa"] for r in rows if r["fa"] is not None])),
            "events": len(ev), "episodes": len(rows),
        }
    return out


def main():
    rng = np.random.default_rng(1)
    cache = OUT / "fresh_signals.pkl"
    if cache.exists():
        fresh = pickle.loads(cache.read_bytes())
    else:
        real_jobs = []
        for name in FRESH:
            X, y = load(name)
            n = len(X)
            k = min(40, n // SPAN)
            # a classifier cannot be fit on a single-class reference window
            real_jobs += [(name, int(s)) for s in np.linspace(0, n - SPAN, k)
                          if 0.02 < y[int(s):int(s) + REF].mean() < 0.98]
        syn_jobs = [(sc, s) for sc in v2.SCENARIOS for s in FRESH_SEEDS]
        with ProcessPoolExecutor(12) as ex:
            fresh = {"real": list(ex.map(real_job, real_jobs, chunksize=2)),
                     "synthetic": list(ex.map(syn_job, syn_jobs, chunksize=2))}
        cache.write_bytes(pickle.dumps(fresh))
    prep(fresh["real"])
    prep(fresh["synthetic"])

    dev = pickle.loads((OUT / "signals.pkl").read_bytes())
    dev_syn_val = prep(dev["valid"])
    dev_real = prep(pickle.loads((OUT / "real_signals.pkl").read_bytes()))

    results = {"fresh_datasets": {
        n: {"windows": sum(s["scenario"] == n for s in fresh["real"]),
            "windows_with_event": sum(s["scenario"] == n and s["event_onset"] is not None for s in fresh["real"])}
        for n in FRESH}, "by_delay": {}}
    for delay in DELAYS:
        syn_val = [dict(s, scores=with_scores(s, delay)) for s in dev_syn_val]
        real_dev = [dict(s, scores=with_scores(s, delay)) for s in dev_real]
        syn_test = [dict(s, scores=with_scores(s, delay)) for s in fresh["synthetic"]]
        real_test = [dict(s, scores=with_scores(s, delay)) for s in fresh["real"]]
        entry = {}
        for tag, val, test in (("synthetic_fresh_seeds", syn_val, syn_test),
                               ("real_fresh_datasets", real_dev, real_test)):
            thr = {m: pick_threshold(val, m)[0] for m in METHODS}
            natural = {m: (v2.DROP if m in ACCURACY_UNIT else thr[m]) for m in METHODS}
            entry[tag] = {"dev_thresholds": evaluate(test, thr, rng),
                          "natural_threshold": evaluate(test, natural, rng)}
            if tag == "real_fresh_datasets":
                entry[tag]["per_dataset_auroc"] = {
                    d: {m: auroc([t for t in test if t["scenario"] == d], m) for m in METHODS} for d in FRESH}
        results["by_delay"][str(delay)] = entry
        print("delay", delay, "done")
    (OUT / "confirmation_results.json").write_text(json.dumps(results, indent=2))
    write_md(results, OUT / "confirmation_results.md")


def f(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def write_md(res, path):
    L = ["# v3 confirmation (data unseen during design)", "",
         "| fresh dataset | windows | with event |", "|---|---:|---:|"]
    for n, d in res["fresh_datasets"].items():
        L.append(f"| {n} | {d['windows']} | {d['windows_with_event']} |")
    for delay, entry in res["by_delay"].items():
        for tag, blocks in entry.items():
            for kind in ("dev_thresholds", "natural_threshold"):
                L += ["", f"## L = {delay} | {tag} | {kind}", "",
                      "| method | AUROC | recall | early-warning | FA/100 clean | threshold |",
                      "|---|---:|---:|---:|---:|---:|"]
                for m, r in blocks[kind].items():
                    rc, ew = r["recall_ci95"], r["early_warning_ci95"]
                    L.append(f"| {m} | {f(r['auroc'])} | {f(r['recall'])} [{f(rc[0])}, {f(rc[1])}] | "
                             f"{f(r['early_warning_rate'])} [{f(ew[0])}, {f(ew[1])}] | "
                             f"{f(r['false_alerts_per_100_clean'])} | {f(r['threshold'], 3)} |")
            if "per_dataset_auroc" in blocks:
                pdx = blocks["per_dataset_auroc"]
                L += ["", f"AUROC per fresh dataset (L = {delay}):", "",
                      "| method | " + " | ".join(pdx) + " |", "|---|" + "---:|" * len(pdx)]
                for m in METHODS:
                    L.append(f"| {m} | " + " | ".join(f(pdx[d][m]) for d in pdx) + " |")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
