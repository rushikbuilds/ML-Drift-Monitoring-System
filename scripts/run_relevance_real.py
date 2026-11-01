"""Relevance-weighted shift (RWS) on real streams.

    python scripts/run_relevance_real.py

RWS_t = sum_j importance_j * KS_j(t), with RandomForest importances fit on the window's
reference rows (no fitting on other streams, so nothing to leak). Questions:
 1. Does RWS track accuracy loss better than the unweighted mean KS or the fixed DEWS score?
 2. Does adding RWS to the delay-anchored estimator reduce nowcast error (leave-one-dataset-out
    on 8 development streams; fit on them, applied to 5 held-out streams)?
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
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from dews.anchored import ANCHOR_LEN, RIDGE_ALPHA, signal_matrix  # noqa: E402
from scripts.run_v2_realdata import BATCH, NB, REF, SPAN, window  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
DEV_OLD = {"electricity", "airlines", "covertype", "bank_marketing"}
NEW = {"hyperplane", "sea", "covpokelec", "poker_hand", "led"}
IDX = {n: i for i, n in enumerate(v2.BASE_SIGNALS)}


def episode(name, start):
    if name in DEV_OLD:
        return window(name, start)
    if name in NEW:
        X, y = pd.read_pickle(OUT / f"new_{name}.pkl")
    else:
        X, y = pd.read_pickle(OUT / f"fresh_{name}.pkl")
    ref, s = X.iloc[start:start + REF], X.iloc[start + REF:start + SPAN]
    b = [s.iloc[i * BATCH:(i + 1) * BATCH].reset_index(drop=True) for i in range(NB)]
    lab = [y[start + REF + i * BATCH:start + REF + (i + 1) * BATCH] for i in range(NB)]
    return v2.Episode(name, start, ref.reset_index(drop=True), y[start:start + REF], b, lab, None)


def job(args):
    warnings.filterwarnings("ignore")
    ep = episode(*args)
    rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1)
    rf.fit(ep.reference, ep.reference_labels)
    imp = rf.feature_importances_
    ref = ep.reference.to_numpy()
    ks = np.array([[ks_2samp(ref[:, j], b.iloc[:, j].to_numpy()).statistic for j in range(ref.shape[1])] for b in ep.batches])
    return {"rws": ks @ imp, "mean_ks": ks.mean(axis=1), "max_ks": ks.max(axis=1)}


def delta_x(sig, delay, extra):
    S = np.column_stack([signal_matrix(sig), extra])
    acc = np.asarray(sig["accuracy"])
    n, lag = len(acc), max(delay, 1)
    X = np.zeros((n, S.shape[1]))
    anchor = np.full(n, np.nan)
    for t in range(n):
        m = t - lag
        if m < v2.BASELINE_BATCHES:
            continue
        lo = max(0, m - ANCHOR_LEN + 1)
        anchor[t] = acc[lo:m + 1].mean()
        X[t] = S[t] - S[lo:m + 1].mean(axis=0)
    return X, anchor


def fit_estimate(train, test, delay, use_rws):
    def ex(s):
        return np.column_stack([s["rws"], s["mean_ks"]]) if use_rws else np.zeros((len(s["accuracy"]), 0))
    xs, ys = [], []
    for s in train:
        X, a = delta_x(s, delay, ex(s))
        ok = np.isfinite(a)
        xs.append(X[ok]); ys.append((np.asarray(s["accuracy"]) - a)[ok])
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(np.vstack(xs), np.concatenate(ys))
    outs = []
    for s in test:
        X, a = delta_x(s, delay, ex(s))
        base = float(np.mean(s["accuracy"][:v2.BASELINE_BATCHES]))
        outs.append(np.where(np.isfinite(a), a + model.predict(X), base))
    return outs


def main():
    dev = pickle.loads((OUT / "real_aux_signals.pkl").read_bytes())
    held = pickle.loads((OUT / "new_signals.pkl").read_bytes())
    allsig = dev + held
    cache = OUT / "rws_real.pkl"
    if cache.exists():
        extras = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor(10) as ex:
            extras = list(ex.map(job, [(s["scenario"], s["seed"]) for s in allsig], chunksize=3))
        cache.write_bytes(pickle.dumps(extras))
    for s, e in zip(allsig, extras):
        s.update(e)
        o = v2.outcome(s["accuracy"])
        s["event_onset"], s["targets"] = o["event_onset"], v2.state_targets(s["accuracy"])
    dev_names = sorted({s["scenario"] for s in dev})
    held_names = sorted({s["scenario"] for s in held})
    report = {}
    # 1. correlation with error and macro AUROC (no fitting)
    sigfn = {"rws": lambda s: s["rws"], "mean_ks": lambda s: s["mean_ks"], "max_ks": lambda s: s["max_ks"],
             "fixed_score": lambda s: s["signals"][:, IDX["drift_score"]], "s_stat": lambda s: s["signals"][:, IDX["s_stat"]]}
    corr = {k: {} for k in sigfn}
    auc = {k: {} for k in sigfn}
    for names, group in ((dev_names, dev), (held_names, held)):
        for n in names:
            win = [s for s in group if s["scenario"] == n]
            for k, fn in sigfn.items():
                vals = []
                for s in win:
                    err, v = 1 - s["accuracy"], fn(s)
                    if np.std(v[5:]) > 0 and np.std(err[5:]) > 0:
                        vals.append(spearmanr(v[5:], err[5:])[0])
                corr[k][n] = float(np.nanmean(vals)) if vals else None
                ys = np.concatenate([s["targets"][v2.WARMUP:] for s in win]); ss = np.concatenate([fn(s)[v2.WARMUP:] for s in win])
                ok = np.isfinite(ys)
                auc[k][n] = float(roc_auc_score(ys[ok], ss[ok])) if len(set(ys[ok])) == 2 else None
    report["spearman_with_error"] = corr
    report["auroc_state_target_per_dataset"] = auc
    report["macro"] = {k: {"spearman_dev": float(np.nanmean([v for n, v in corr[k].items() if n in dev_names and v is not None])),
                           "auroc_dev": float(np.nanmean([v for n, v in auc[k].items() if n in dev_names and v is not None])),
                           "spearman_held_out": float(np.nanmean([v for n, v in corr[k].items() if n in held_names and v is not None])),
                           "auroc_held_out": float(np.nanmean([v for n, v in auc[k].items() if n in held_names and v is not None]))}
                       for k in sigfn}
    # 2. anchored estimator with and without RWS features
    report["nowcast_mae"] = {}
    for delay in (0, 2, 5):
        res = {}
        for use in (False, True):
            tag = "anchored_plus_rws" if use else "anchored"
            dev_err = {}
            for d in dev_names:
                test = [s for s in dev if s["scenario"] == d]
                est = fit_estimate([s for s in dev if s["scenario"] != d], test, delay, use)
                dev_err[d] = float(np.concatenate([np.abs(np.clip(e, 0, 1) - s["accuracy"])[v2.WARMUP:] for e, s in zip(est, test)]).mean())
            est = fit_estimate(dev, held, delay, use)
            held_err = {}
            for d in held_names:
                idx = [i for i, s in enumerate(held) if s["scenario"] == d]
                held_err[d] = float(np.concatenate([np.abs(np.clip(est[i], 0, 1) - held[i]["accuracy"])[v2.WARMUP:] for i in idx]).mean())
            res[tag] = {"development_lodo": dev_err, "held_out": held_err,
                        "macro_dev": float(np.mean(list(dev_err.values()))), "macro_held_out": float(np.mean(list(held_err.values())))}
        report["nowcast_mae"][str(delay)] = res
    (OUT / "rws_real_results.json").write_text(json.dumps(report, indent=2))
    print("macro correlation / AUROC (dev, held-out):")
    for k, v in report["macro"].items():
        print(f"  {k:12s} spearman {v['spearman_dev']:.2f} / {v['spearman_held_out']:.2f}   auroc {v['auroc_dev']:.2f} / {v['auroc_held_out']:.2f}")
    for delay, r in report["nowcast_mae"].items():
        print(f"L={delay} MAE x100  dev: anchored {100*r['anchored']['macro_dev']:.2f} -> +rws {100*r['anchored_plus_rws']['macro_dev']:.2f}   "
              f"held-out: {100*r['anchored']['macro_held_out']:.2f} -> {100*r['anchored_plus_rws']['macro_held_out']:.2f}")


if __name__ == "__main__":
    main()
