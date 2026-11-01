"""Multi-dataset real-data study (leave-one-dataset-out).

    python scripts/run_v2_realdata.py

Datasets (OpenML, natural row order): electricity, airlines, covertype, bank-marketing.
Each yields non-overlapping chronological windows: 2000 reference rows then 25
batches of 250. No drift is injected. For each held-out dataset the calibrators
and alert thresholds are fit on the OTHER datasets only. Requires
artifacts/v2/signals.pkl (from run_v2_experiment.py) for the synthetic-trained variant.
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

OUT = ROOT / "artifacts" / "v2"
REF, BATCH, NB = 2000, 250, 25
SPAN = REF + NB * BATCH
MAX_WINDOWS = 40
DELAYS = (0, 2, 5)
DATASETS = {"electricity": 151, "airlines": 1169, "covertype": 150, "bank_marketing": 1461}


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in frame.columns:
        s = frame[col]
        if str(s.dtype) in ("category", "object"):
            s = s.astype("category").cat.codes.astype(float) if s.nunique() > 2 else s.astype("category").cat.codes.astype(float)
        out[col] = s.astype(float)
    return pd.DataFrame(out)


def load(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    cache = OUT / f"real_{name}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    d = fetch_openml(data_id=DATASETS[name], as_frame=True, parser="auto")
    X, y = d.data, d.target
    if name == "covertype":
        X = X.iloc[:, :10]
        y = (y.astype(int) == 2)
    elif name == "airlines":
        X = X.copy()
        for c in ("Airline", "AirportFrom", "AirportTo"):
            X[c] = X[c].map(X[c].value_counts(normalize=True))  # frequency encoding
        y = y.astype(str).isin(["1", "True"])
    else:
        classes = sorted(y.astype(str).unique())
        y = y.astype(str) == classes[-1]
    X = _numeric(X).fillna(0.0)
    X = X.loc[:, X.nunique() > 1].reset_index(drop=True)
    res = (X, np.asarray(y, dtype=int))
    pd.to_pickle(res, cache)
    return res


def window(name, start):
    X, y = load(name)
    ref = X.iloc[start:start + REF]
    s = X.iloc[start + REF:start + SPAN]
    batches = [s.iloc[i * BATCH:(i + 1) * BATCH].reset_index(drop=True) for i in range(NB)]
    labels = [y[start + REF + i * BATCH:start + REF + (i + 1) * BATCH] for i in range(NB)]
    return v2.Episode(name, start, ref.reset_index(drop=True), y[start:start + REF], batches, labels, None)


def job(args):
    warnings.filterwarnings("ignore")
    name, start = args
    return v2.extract_signals(window(name, start))


def scored(sig, delay, pool_sig, pool_full, syn_sig, syn_full):
    s = v2.baseline_scores(sig, delay)
    s["cal_signals_lodo"] = v2.calibrator_scores(sig, delay, pool_sig, use_labels=False, online=False)
    s["cal_full_lodo"] = v2.calibrator_scores(sig, delay, pool_full, use_labels=True, online=False)
    s["cal_signals_synthetic"] = v2.calibrator_scores(sig, delay, syn_sig, use_labels=False, online=False)
    s["cal_full_synthetic"] = v2.calibrator_scores(sig, delay, syn_full, use_labels=True, online=False)
    for k in s:
        s[k] = np.asarray(s[k], float).copy()
        s[k][: v2.WARMUP] = np.nan
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "real_signals.pkl"
    if cache.exists():
        sigs = pickle.loads(cache.read_bytes())
    else:
        jobs = []
        for name in DATASETS:
            n = len(load(name)[0])
            k = min(MAX_WINDOWS, n // SPAN)
            starts = np.linspace(0, n - SPAN, k).astype(int)
            jobs += [(name, int(s)) for s in starts]
        with ProcessPoolExecutor(12) as ex:
            sigs = list(ex.map(job, jobs, chunksize=2))
        cache.write_bytes(pickle.dumps(sigs))
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"] = o["event_onset"]
        s["targets"] = v2.state_targets(s["accuracy"])
        s["baseline_acc"] = o["baseline"]

    syn_train = pickle.loads((OUT / "signals.pkl").read_bytes())["train"]
    for s in syn_train:
        s["targets"] = v2.state_targets(s["accuracy"])

    summary = {n: {"windows": sum(s["scenario"] == n for s in sigs),
                   "windows_with_event": sum(s["scenario"] == n and s["event_onset"] is not None for s in sigs),
                   "mean_baseline_accuracy": float(np.mean([s["baseline_acc"] for s in sigs if s["scenario"] == n]))}
               for n in DATASETS}
    results = {"config": {"reference_rows": REF, "batch_size": BATCH, "batches": NB, "max_windows": MAX_WINDOWS},
               "datasets": summary, "by_delay": {}}
    rng = np.random.default_rng(0)
    methods = None
    for delay in DELAYS:
        syn_sig = v2.training_pool(syn_train, delay, use_labels=False)
        syn_full = v2.training_pool(syn_train, delay, use_labels=True)
        per_dataset = {}
        pooled_rows = {}
        for held in DATASETS:
            others = [s for s in sigs if s["scenario"] != held]
            test = [s for s in sigs if s["scenario"] == held]
            pool_sig = v2.training_pool(others, delay, use_labels=False)
            pool_full = v2.training_pool(others, delay, use_labels=True)
            val = [dict(s, scores=scored(s, delay, pool_sig, pool_full, syn_sig, syn_full)) for s in others]
            tst = [dict(s, scores=scored(s, delay, pool_sig, pool_full, syn_sig, syn_full)) for s in test]
            methods = list(tst[0]["scores"])
            per_dataset[held] = {}
            for m in methods:
                thr, _, _ = pick_threshold(val, m)
                ys = np.concatenate([t["targets"] for t in tst])
                ss = np.concatenate([t["scores"][m] for t in tst])
                ok = np.isfinite(ys) & np.isfinite(ss)
                auc = float(roc_auc_score(ys[ok], ss[ok])) if len(set(ys[ok])) == 2 else None
                rows = []
                for t in tst:
                    n = len(t["accuracy"])
                    r = episode_eval(t["scores"][m], t["event_onset"], n, thr)
                    rows.append({"event": t["event_onset"] is not None, "matched": r["matched_event_count"] > 0,
                                 "early": r["warning_lead_batches"] is not None,
                                 "lead": r["warning_lead_batches"], "fa": r["false_alerts_per_100_clean_batches"]})
                ev = [r for r in rows if r["event"]]
                per_dataset[held][m] = {
                    "auroc": auc, "threshold": thr,
                    "recall": float(np.mean([r["matched"] for r in ev])) if ev else None,
                    "early_warning_rate": float(np.mean([r["early"] for r in ev])) if ev else None,
                    "false_alerts_per_100_clean": float(np.mean([r["fa"] for r in rows if r["fa"] is not None])),
                }
                pooled_rows.setdefault(m, {"auc_pairs": [], "rows": []})
                pooled_rows[m]["rows"] += rows
                pooled_rows[m]["auc_pairs"].append((held, auc))
        pooled = {}
        for m, d in pooled_rows.items():
            ev = [r for r in d["rows"] if r["event"]]
            aucs = [a for _, a in d["auc_pairs"] if a is not None]
            pooled[m] = {
                "mean_auroc_over_datasets": float(np.mean(aucs)),
                "recall": float(np.mean([r["matched"] for r in ev])),
                "recall_ci95": boot_ci([r["matched"] for r in ev], rng),
                "early_warning_rate": float(np.mean([r["early"] for r in ev])),
                "early_warning_ci95": boot_ci([r["early"] for r in ev], rng),
                "false_alerts_per_100_clean": float(np.mean([r["fa"] for r in d["rows"] if r["fa"] is not None])),
                "events": len(ev), "windows": len(d["rows"]),
            }
        results["by_delay"][str(delay)] = {"per_dataset": per_dataset, "pooled": pooled}
        print("delay", delay, "done")
    (OUT / "realdata_results.json").write_text(json.dumps(results, indent=2))
    write_md(results, OUT / "realdata_results.md")


def f(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def write_md(res, path):
    L = ["# Real-data study (leave-one-dataset-out)", "", "| dataset | windows | windows with event | mean baseline accuracy |", "|---|---:|---:|---:|"]
    for n, d in res["datasets"].items():
        L.append(f"| {n} | {d['windows']} | {d['windows_with_event']} | {d['mean_baseline_accuracy']:.3f} |")
    for delay, d in res["by_delay"].items():
        L += ["", f"## Label delay L = {delay} (pooled over held-out datasets)", "",
              "| method | mean AUROC | recall | early-warning rate | FA/100 clean |", "|---|---:|---:|---:|---:|"]
        for m, r in d["pooled"].items():
            rc, ew = r["recall_ci95"], r["early_warning_ci95"]
            L.append(f"| {m} | {f(r['mean_auroc_over_datasets'])} | {f(r['recall'])} [{f(rc[0])}, {f(rc[1])}] | "
                     f"{f(r['early_warning_rate'])} [{f(ew[0])}, {f(ew[1])}] | {f(r['false_alerts_per_100_clean'])} |")
        L += ["", "AUROC by held-out dataset:", "", "| method | " + " | ".join(d["per_dataset"]) + " |", "|---|" + "---:|" * len(d["per_dataset"])]
        for m in d["pooled"]:
            L.append(f"| {m} | " + " | ".join(f(d["per_dataset"][h][m]["auroc"]) for h in d["per_dataset"]) + " |")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
