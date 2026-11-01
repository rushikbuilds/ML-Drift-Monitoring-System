"""Exploration: label-anchored differential estimator (LODO, 8 datasets)."""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from scripts.explore_v4 import persist  # noqa: E402
from scripts.run_v2_experiment import episode_eval, pick_threshold  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
IDX = {n: i for i, n in enumerate(v2.BASE_SIGNALS)}


def signal_matrix(sig):
    s, a = np.asarray(sig["signals"]), np.asarray(sig["aux"])
    return np.column_stack([
        s[:, IDX["mean_confidence"]], s[:, IDX["mean_entropy"]], s[:, IDX["low_confidence_rate"]],
        s[:, IDX["positive_prediction_rate"]], s[:, IDX["s_stat"]], s[:, IDX["mean_ks"]], s[:, IDX["drift_score"]],
        a[:, 0], a[:, 1], a[:, 2], a[:, 3]])


def delta_dataset(sig, delay, anchor_len=3):
    """For each t: anchor = last `anchor_len` matured batches; features = signal change since
    the anchor; target = accuracy change since the anchor. Causal: uses labels b <= t - max(L,1)."""
    S, acc = signal_matrix(sig), np.asarray(sig["accuracy"])
    n, lag = len(acc), max(delay, 1)
    X = np.zeros((n, S.shape[1]))
    anchor = np.full(n, np.nan)
    for t in range(n):
        m = t - lag
        if m < 5:
            continue
        lo = max(0, m - anchor_len + 1)
        anchor[t] = acc[lo:m + 1].mean()
        X[t] = S[t] - S[lo:m + 1].mean(axis=0)
    return X, anchor


def main():
    sigs = pickle.loads((OUT / "real_aux_signals.pkl").read_bytes())
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"] = o["event_onset"]
        s["targets"] = v2.state_targets(s["accuracy"])
    names = sorted({s["scenario"] for s in sigs})
    for delay in (0, 2, 5):
        for s in sigs:
            s["_X"], s["_anchor"] = delta_dataset(s, delay)
            sc = v2.baseline_scores(s, delay)
            sc.update(v2.cbpe_scores(s, delay))
            sc["da_cbpe_p2"] = persist(sc["da_cbpe"], 2)
            s["_sc"] = sc
        methods = ["fixed_score", "delayed_label_only", "da_cbpe_p2", "anchored", "anchored_p2"]
        res = {m: {"auc": [], "rows": []} for m in methods}
        mae = {k: {} for k in ("stale_last", "stale_mean3", "da_cbpe", "anchored")}
        for held in names:
            train = [s for s in sigs if s["scenario"] != held]
            test = [s for s in sigs if s["scenario"] == held]
            ok = lambda s: np.isfinite(s["_anchor"])  # noqa: E731
            X = np.vstack([s["_X"][ok(s)] for s in train])
            y = np.concatenate([(s["accuracy"] - s["_anchor"])[ok(s)] for s in train])
            model = make_pipeline(StandardScaler(), Ridge(alpha=50.0)).fit(X, y)
            for s in train + test:
                base = s["accuracy"][:5].mean()
                est = np.where(np.isfinite(s["_anchor"]), s["_anchor"] + model.predict(s["_X"]), base)
                s["_est"] = est
                drop = np.clip(base - est, 0, None)
                s["_sc"]["anchored"], s["_sc"]["anchored_p2"] = drop, persist(drop, 2)

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
                good = np.isfinite(ys) & np.isfinite(ss)
                if len(set(ys[good])) == 2:
                    res[m]["auc"].append(roc_auc_score(ys[good], ss[good]))
                for t in te:
                    r = episode_eval(t["scores"][m], t["event_onset"], len(t["accuracy"]), thr)
                    tta = -r["warning_lead_batches"] if r["warning_lead_batches"] is not None else r["detection_lag_batches"]
                    res[m]["rows"].append((t["event_onset"] is not None, r["matched_event_count"] > 0,
                                           r["warning_lead_batches"] is not None, tta, r["false_alerts_per_100_clean_batches"]))
            for k in mae:
                errs = []
                for s in test:
                    e = v2.accuracy_estimates(s, delay)
                    est = s["_est"] if k == "anchored" else e[k]
                    errs.append(np.abs(np.clip(est, 0, 1) - s["accuracy"])[v2.WARMUP:])
                mae[k][held] = float(np.concatenate(errs).mean())
        print(f"\n=== L={delay} ===")
        print(f"{'method':20s} macroAUC recall early   tta  FA/100")
        for m in methods:
            rows = res[m]["rows"]; ev = [r for r in rows if r[0]]
            ttas = [r[3] for r in ev if r[3] is not None]
            fa = np.mean([r[4] for r in rows if r[4] is not None])
            print(f"{m:20s} {np.mean(res[m]['auc']):.2f}     {np.mean([r[1] for r in ev]):.2f}   {np.mean([r[2] for r in ev]):.2f}  {np.mean(ttas):5.2f}  {fa:5.1f}")
        print("nowcast MAE macro:", {k: round(100 * float(np.mean(list(v.values()))), 2) for k, v in mae.items()})
        print("per dataset anchored vs stale_last:", {n: (round(100 * mae['anchored'][n], 1), round(100 * mae['stale_last'][n], 1)) for n in names})


if __name__ == "__main__":
    main()
