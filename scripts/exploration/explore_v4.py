"""Exploration on cached development signals (not a reported result).

Compares smoothing variants and a learned accuracy-drop regressor under
leave-one-dataset-out, macro-averaged over datasets.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from scripts.run_v2_experiment import episode_eval, pick_threshold  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
IDX = {n: i for i, n in enumerate(v2.BASE_SIGNALS)}


def persist(x, k):
    out = x.copy()
    for t in range(len(x)):
        out[t] = np.min(x[max(0, t - k + 1):t + 1])
    return out


def rel_features(sig, delay):
    s = np.asarray(sig["signals"])
    ref = s[:5].mean(axis=0)
    rel = s - ref
    cb = v2.cbpe_scores(sig, delay)
    lab = v2.label_features(sig["accuracy"], delay)
    feats = [rel[:, [IDX["mean_confidence"], IDX["mean_entropy"], IDX["low_confidence_rate"], IDX["positive_prediction_rate"]]],
             s[:, [IDX["s_stat"], IDX["mean_ks"], IDX["drift_score"], IDX["s_conf"]]],
             cb["da_cbpe"][:, None], cb["cbpe_raw"][:, None], lab]
    f = np.hstack(feats)
    sl = np.zeros_like(f)
    for t in range(2, len(f)):
        sl[t] = (f[t] - f[t - 2]) / 2
    return np.hstack([f, sl])


def drop_target(sig):
    base = sig["accuracy"][:5].mean()
    return base - sig["accuracy"]


def main():
    dev = pickle.loads((OUT / "real_signals.pkl").read_bytes())
    fresh = pickle.loads((OUT / "fresh_signals.pkl").read_bytes())["real"]
    sigs = dev + fresh
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"] = o["event_onset"]
        s["targets"] = v2.state_targets(s["accuracy"])
    names = sorted({s["scenario"] for s in sigs})
    print("datasets", {n: sum(s["scenario"] == n for s in sigs) for n in names})
    for delay in (2, 5):
        scored = {}
        for s in sigs:
            sc = v2.baseline_scores(s, delay)
            sc.update(v2.cbpe_scores(s, delay))
            sc["da_cbpe_p2"] = persist(sc["da_cbpe"], 2)
            sc["da_cbpe_p3"] = persist(sc["da_cbpe"], 3)
            sc["da_cbpe_fc_p2"] = persist(sc["da_cbpe_forecast"], 2)
            sc["label_only_p2"] = persist(sc["delayed_label_only"], 2)
            s["_feat"] = rel_features(s, delay)
            scored[id(s)] = sc
        methods = ["fixed_score", "slope_forecast", "delayed_label_only", "label_only_p2", "da_cbpe", "da_cbpe_p2",
                   "da_cbpe_p3", "da_cbpe_forecast", "da_cbpe_fc_p2", "learned_gbr", "learned_gbr_p2"]
        res = {m: {"auc": [], "rows": []} for m in methods}
        for held in names:
            train = [s for s in sigs if s["scenario"] != held]
            test = [s for s in sigs if s["scenario"] == held]
            X = np.vstack([s["_feat"][v2.WARMUP:] for s in train])
            y = np.concatenate([drop_target(s)[v2.WARMUP:] for s in train])
            model = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0).fit(X, y)
            for group in (train, test):
                for s in group:
                    pred = np.clip(model.predict(s["_feat"]), 0, None)
                    scored[id(s)]["learned_gbr"] = pred
                    scored[id(s)]["learned_gbr_p2"] = persist(pred, 2)
            for m in methods:
                def items(group):
                    out = []
                    for s in group:
                        sc = {k: np.asarray(v, float).copy() for k, v in scored[id(s)].items()}
                        for v in sc.values():
                            v[:v2.WARMUP] = np.nan
                        out.append(dict(s, scores=sc))
                    return out
                thr, _, _ = pick_threshold(items(train), m)
                te = items(test)
                ys = np.concatenate([t["targets"] for t in te]); ss = np.concatenate([t["scores"][m] for t in te])
                ok = np.isfinite(ys) & np.isfinite(ss)
                if len(set(ys[ok])) == 2:
                    res[m]["auc"].append(roc_auc_score(ys[ok], ss[ok]))
                for t in te:
                    n = len(t["accuracy"])
                    r = episode_eval(t["scores"][m], t["event_onset"], n, thr)
                    tta = None
                    if r["warning_lead_batches"] is not None:
                        tta = -r["warning_lead_batches"]
                    elif r["detection_lag_batches"] is not None:
                        tta = r["detection_lag_batches"]
                    res[m]["rows"].append((t["event_onset"] is not None, r["matched_event_count"] > 0,
                                           r["warning_lead_batches"] is not None, tta, r["false_alerts_per_100_clean_batches"]))
        print(f"\n=== L={delay} (LODO over {len(names)} datasets) ===")
        print(f"{'method':20s} macroAUC recall early  tta  FA/100")
        for m in methods:
            rows = res[m]["rows"]
            ev = [r for r in rows if r[0]]
            ttas = [r[3] for r in ev if r[3] is not None]
            fa = np.mean([r[4] for r in rows if r[4] is not None])
            print(f"{m:20s} {np.mean(res[m]['auc']):.2f}     {np.mean([r[1] for r in ev]):.2f}   {np.mean([r[2] for r in ev]):.2f}  {np.mean(ttas):5.2f}  {fa:5.1f}")


if __name__ == "__main__":
    main()
