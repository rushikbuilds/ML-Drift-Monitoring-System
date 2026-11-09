"""Exploration: disagreement-based, delay-aware accuracy estimation (LODO, 8 datasets)."""
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
from scripts.exploration.explore_v4 import persist, rel_features  # noqa: E402
from scripts.run_v2_experiment import episode_eval, pick_threshold  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
PRIOR_K, RIDGE = 1.0, 0.05


def da_disagree(sig, delay, window=10):
    """Disagreement (mean of RF and HGB vs the frozen model) calibrated to accuracy units
    with a ridge-through-origin slope fitted on matured batches only."""
    dis = sig["aux"][:, :2].mean(axis=1)
    ref = dis[:5].mean()
    x = dis - ref
    acc = sig["accuracy"]
    drop = acc[:5].mean() - acc
    n, lag = len(acc), max(delay, 1)
    k_series = np.full(n, PRIOR_K)
    for t in range(n):
        m = t - lag
        if m >= 5:
            lo = max(5, m - window + 1)
            xs, ys = x[lo:m + 1], drop[lo:m + 1]
            k_series[t] = (np.dot(xs, ys) + RIDGE * PRIOR_K) / (np.dot(xs, xs) + RIDGE)
    k_series = np.clip(k_series, 0.0, 3.0)
    est_drop = np.clip(k_series * x, 0, None)
    return est_drop, np.clip(x, 0, None)


def feats_with_aux(sig, delay):
    base = rel_features(sig, delay)
    a = sig["aux"]
    ref = a[:5].mean(axis=0)
    rel = a - ref
    f = np.hstack([base[:, :base.shape[1] // 2], rel])
    sl = np.zeros_like(f)
    for t in range(2, len(f)):
        sl[t] = (f[t] - f[t - 2]) / 2
    return np.hstack([f, sl])


def main():
    sigs = pickle.loads((OUT / "real_aux_signals.pkl").read_bytes())
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"] = o["event_onset"]
        s["targets"] = v2.state_targets(s["accuracy"])
    names = sorted({s["scenario"] for s in sigs})
    for delay in (0, 2, 5):
        for s in sigs:
            sc = v2.baseline_scores(s, delay)
            sc.update(v2.cbpe_scores(s, delay))
            est, raw = da_disagree(s, delay)
            sc["dis_raw"] = raw
            sc["dis_raw_p2"] = persist(raw, 2)
            sc["da_dis"] = est
            sc["da_dis_p2"] = persist(est, 2)
            sc["da_blend"] = 0.5 * (est + sc["da_cbpe"])
            sc["da_blend_p2"] = persist(sc["da_blend"], 2)
            sc["da_cbpe_p2"] = persist(sc["da_cbpe"], 2)
            s["_sc"] = sc
            s["_feat"] = feats_with_aux(s, delay)
        methods = ["fixed_score", "delayed_label_only", "da_cbpe_p2", "dis_raw", "dis_raw_p2", "da_dis", "da_dis_p2",
                   "da_blend", "da_blend_p2", "gbr_dis", "gbr_dis_p2"]
        res = {m: {"auc": [], "rows": []} for m in methods}
        mae = {k: {} for k in ("stale_last", "da_cbpe", "da_dis", "da_blend", "gbr_dis")}
        for held in names:
            train = [s for s in sigs if s["scenario"] != held]
            test = [s for s in sigs if s["scenario"] == held]
            X = np.vstack([s["_feat"][v2.WARMUP:] for s in train])
            y = np.concatenate([(s["accuracy"][:5].mean() - s["accuracy"])[v2.WARMUP:] for s in train])
            gbr = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0).fit(X, y)
            for s in train + test:
                p = np.clip(gbr.predict(s["_feat"]), 0, None)
                s["_sc"]["gbr_dis"], s["_sc"]["gbr_dis_p2"] = p, persist(p, 2)

            def items(group):
                out = []
                for s in group:
                    sc = {k: np.asarray(v, float).copy() for k, v in s["_sc"].items()}
                    for v in sc.values():
                        v[:v2.WARMUP] = np.nan
                    out.append(dict(s, scores=sc))
                return out
            tr, te = items(train), items(test)
            for m in methods:
                thr, _, _ = pick_threshold(tr, m)
                ys = np.concatenate([t["targets"] for t in te]); ss = np.concatenate([t["scores"][m] for t in te])
                ok = np.isfinite(ys) & np.isfinite(ss)
                if len(set(ys[ok])) == 2:
                    res[m]["auc"].append(roc_auc_score(ys[ok], ss[ok]))
                for t in te:
                    n = len(t["accuracy"])
                    r = episode_eval(t["scores"][m], t["event_onset"], n, thr)
                    tta = -r["warning_lead_batches"] if r["warning_lead_batches"] is not None else r["detection_lag_batches"]
                    res[m]["rows"].append((t["event_onset"] is not None, r["matched_event_count"] > 0,
                                           r["warning_lead_batches"] is not None, tta, r["false_alerts_per_100_clean_batches"]))
            # nowcast MAE
            est_sets = {"stale_last": [], "da_cbpe": [], "da_dis": [], "da_blend": [], "gbr_dis": []}
            for s in test:
                base = s["accuracy"][:5].mean()
                e = v2.accuracy_estimates(s, delay)
                est_sets["stale_last"].append(e["stale_last"])
                est_sets["da_cbpe"].append(e["da_cbpe"])
                d = da_disagree(s, delay)[0]
                est_sets["da_dis"].append(base - d)
                est_sets["da_blend"].append(0.5 * ((base - d) + e["da_cbpe"]))
                est_sets["gbr_dis"].append(base - np.clip(gbr.predict(s["_feat"]), 0, None))
            for k, v in est_sets.items():
                mae[k][held] = float(np.mean(np.concatenate([np.abs(np.clip(est, 0, 1) - s["accuracy"])[v2.WARMUP:] for est, s in zip(v, test)])))
        print(f"\n=== L={delay} (LODO over {len(names)} datasets) ===")
        print(f"{'method':20s} macroAUC recall early   tta  FA/100")
        for m in methods:
            rows = res[m]["rows"]; ev = [r for r in rows if r[0]]
            ttas = [r[3] for r in ev if r[3] is not None]
            fa = np.mean([r[4] for r in rows if r[4] is not None])
            print(f"{m:20s} {np.mean(res[m]['auc']):.2f}     {np.mean([r[1] for r in ev]):.2f}   {np.mean([r[2] for r in ev]):.2f}  {np.mean(ttas):5.2f}  {fa:5.1f}")
        print("nowcast MAE (points), macro:", {k: round(100 * float(np.mean(list(v.values()))), 2) for k, v in mae.items()})


if __name__ == "__main__":
    main()
