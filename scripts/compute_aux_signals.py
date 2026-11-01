"""Exploration: does disagreement with auxiliary reference-trained models track error?

Auxiliary models (RandomForest, HistGradientBoosting) are trained on the same
reference rows as the monitored classifier and are never used to predict for
users; they only supply monitor-side signals.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from dews.experiment_config import DEFAULT_EARLY_WARNING_CONFIG  # noqa: E402

OUT = ROOT / "artifacts" / "v2"


def aux_signals(ep: v2.Episode) -> np.ndarray:
    """Per batch: [dis_rf, dis_hgb, dis_aux_aux, aux_mean_conf]."""
    lr = DEFAULT_EARLY_WARNING_CONFIG.classifier.build_pipeline().fit(ep.reference, ep.reference_labels)
    rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1).fit(ep.reference, ep.reference_labels)
    hgb = HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=0).fit(ep.reference, ep.reference_labels)
    out = []
    for b in ep.batches:
        p0, p1, p2 = lr.predict(b), rf.predict(b), hgb.predict(b)
        conf = (rf.predict_proba(b).max(axis=1).mean() + hgb.predict_proba(b).max(axis=1).mean()) / 2
        out.append([np.mean(p0 != p1), np.mean(p0 != p2), np.mean(p1 != p2), conf])
    return np.asarray(out)


def job(args):
    warnings.filterwarnings("ignore")
    kind, name, start = args
    if kind == "dev":
        from scripts.run_v2_realdata import window
        ep = window(name, start)
    else:
        from scripts.run_v3_confirmation import load
        from scripts.run_v2_realdata import BATCH, NB, REF, SPAN
        X, y = load(name)
        ref, s = X.iloc[start:start + REF], X.iloc[start + REF:start + SPAN]
        b = [s.iloc[i * BATCH:(i + 1) * BATCH].reset_index(drop=True) for i in range(NB)]
        lab = [y[start + REF + i * BATCH:start + REF + (i + 1) * BATCH] for i in range(NB)]
        ep = v2.Episode(name, start, ref.reset_index(drop=True), y[start:start + REF], b, lab, None)
    return aux_signals(ep)


def main():
    dev = pickle.loads((OUT / "real_signals.pkl").read_bytes())
    fresh = pickle.loads((OUT / "fresh_signals.pkl").read_bytes())["real"]
    jobs = [("dev", s["scenario"], s["seed"]) for s in dev] + [("fresh", s["scenario"], s["seed"]) for s in fresh]
    with ProcessPoolExecutor(12) as ex:
        aux = list(ex.map(job, jobs, chunksize=2))
    sigs = dev + fresh
    for s, a in zip(sigs, aux):
        s["aux"] = a
    pickle.dump(sigs, open(OUT / "real_aux_signals.pkl", "wb"))
    names = sorted({s["scenario"] for s in sigs})
    idx = {n: i for i, n in enumerate(v2.BASE_SIGNALS)}
    print("within-window Spearman(signal, error), mean over windows, by dataset")
    cols = {"conf_drop": lambda s: -s["signals"][:, idx["mean_confidence"]],
            "stat": lambda s: s["signals"][:, idx["s_stat"]],
            "fixed": lambda s: s["signals"][:, idx["drift_score"]],
            "dis_rf": lambda s: s["aux"][:, 0], "dis_hgb": lambda s: s["aux"][:, 1],
            "dis_aa": lambda s: s["aux"][:, 2], "aux_conf_drop": lambda s: -s["aux"][:, 3]}
    print(f"{'signal':14s}" + "".join(f"{n[:7]:>9s}" for n in names) + "    macro")
    for k, fn in cols.items():
        row = []
        for n in names:
            vals = []
            for s in (x for x in sigs if x["scenario"] == n):
                err = 1 - s["accuracy"]
                v = fn(s)
                if np.std(v[5:]) > 0 and np.std(err[5:]) > 0:
                    vals.append(spearmanr(v[5:], err[5:])[0])
            row.append(np.nanmean(vals))
        print(f"{k:14s}" + "".join(f"{r:9.2f}" for r in row) + f"   {np.mean(row):6.2f}")


if __name__ == "__main__":
    main()
