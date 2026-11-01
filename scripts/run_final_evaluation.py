"""Final locked evaluation of the delay-anchored estimator.

    python scripts/run_final_evaluation.py

Development: leave-one-dataset-out cross-validation over 8 real streams
(electricity, airlines, covertype, bank_marketing, kddcup99, ldpa, run_or_walk,
vehicle_sensit). Held-out: 5 benchmark streams never used while designing the
method (hyperplane, sea, covpokelec, poker_hand, led). The held-out estimator and
alert thresholds are fit on the 8 development streams only.
Needs: artifacts/v2/real_aux_signals.pkl (scripts/compute_aux_signals.py).
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

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from dews.anchored import AnchoredEstimator, aux_signals  # noqa: E402
from scripts.run_v2_experiment import boot_ci, episode_eval, pick_threshold  # noqa: E402
from scripts.run_v2_realdata import BATCH, NB, REF, SPAN, _numeric  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
DELAYS = (0, 2, 5)
NEW = {  # name: (openml id, max features, target rule)
    "hyperplane": (152, 10, "binary"),
    "sea": (162, 3, "binary"),
    "covpokelec": (149, 15, "majority_vs_rest"),
    "poker_hand": (1569, 10, "majority_vs_rest"),
    "led": (154, 24, "majority_vs_rest"),
}
SCORE_METHODS = ["fixed_score", "delayed_label_only", "da_cbpe_p2", "anchored", "anchored_p2"]
NOWCAST = ["stale_last", "stale_mean3", "cbpe_raw", "da_cbpe", "anchored"]


def load_new(name):
    cache = OUT / f"new_{name}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    did, kmax, rule = NEW[name]
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X, y = d.data, d.target.astype(str)
    yy = (y == y.value_counts().index[0]).astype(int).to_numpy() if rule == "majority_vs_rest" \
        else (y == sorted(y.unique())[-1]).astype(int).to_numpy()
    X = _numeric(X).fillna(0.0)
    X = X.loc[:, X.nunique() > 1].iloc[:, :kmax].reset_index(drop=True)
    pd.to_pickle((X, yy), cache)
    return X, yy


def job(args):
    warnings.filterwarnings("ignore")
    name, start = args
    X, y = pd.read_pickle(OUT / f"new_{name}.pkl")
    ref, s = X.iloc[start:start + REF], X.iloc[start + REF:start + SPAN]
    b = [s.iloc[i * BATCH:(i + 1) * BATCH].reset_index(drop=True) for i in range(NB)]
    lab = [y[start + REF + i * BATCH:start + REF + (i + 1) * BATCH] for i in range(NB)]
    ep = v2.Episode(name, start, ref.reset_index(drop=True), y[start:start + REF], b, lab, None)
    sig = v2.extract_signals(ep)
    sig["aux"] = aux_signals(ep)
    return sig


def prep(sigs):
    for s in sigs:
        o = v2.outcome(s["accuracy"])
        s["event_onset"], s["targets"] = o["event_onset"], v2.state_targets(s["accuracy"])
    return sigs


def all_scores(sig, delay, estimator):
    sc = v2.baseline_scores(sig, delay)
    sc.update(v2.cbpe_scores(sig, delay))
    from dews.anchored import persist
    sc["da_cbpe_p2"] = persist(sc["da_cbpe"], 2)
    sc.update(estimator.drop_scores(sig))
    out = {k: np.asarray(sc[k], float).copy() for k in SCORE_METHODS}
    for v in out.values():
        v[: v2.WARMUP] = np.nan
    return out


def nowcasts(sig, delay, estimator):
    e = v2.accuracy_estimates(sig, delay)
    e["anchored"] = estimator.estimate(sig)
    return e


def mae_by(sigs, delay, estimators):
    """estimators: callable sig -> AnchoredEstimator. Returns {method: {dataset: MAE}}."""
    out = {m: {} for m in NOWCAST}
    errs = {m: {} for m in NOWCAST}
    for s in sigs:
        e = nowcasts(s, delay, estimators(s))
        for m in NOWCAST:
            errs[m].setdefault(s["scenario"], []).append(np.abs(np.clip(e[m], 0, 1) - s["accuracy"])[v2.WARMUP:])
    for m in NOWCAST:
        for d, lst in errs[m].items():
            out[m][d] = float(np.concatenate(lst).mean())
    return out


def detection(test_items, thresholds, rng):
    res = {}
    for m in SCORE_METHODS:
        rows, aucs = [], []
        for d in sorted({t["scenario"] for t in test_items}):
            grp = [t for t in test_items if t["scenario"] == d]
            ys = np.concatenate([t["targets"] for t in grp]); ss = np.concatenate([t["scores"][m] for t in grp])
            ok = np.isfinite(ys) & np.isfinite(ss)
            if len(set(ys[ok])) == 2:
                aucs.append(roc_auc_score(ys[ok], ss[ok]))
        for t in test_items:
            r = episode_eval(t["scores"][m], t["event_onset"], len(t["accuracy"]), thresholds[m])
            tta = -r["warning_lead_batches"] if r["warning_lead_batches"] is not None else r["detection_lag_batches"]
            rows.append({"event": t["event_onset"] is not None, "matched": r["matched_event_count"] > 0,
                         "early": r["warning_lead_batches"] is not None, "tta": tta,
                         "fa": r["false_alerts_per_100_clean_batches"]})
        ev = [r for r in rows if r["event"]]
        ttas = [r["tta"] for r in ev if r["tta"] is not None]
        res[m] = {"macro_auroc": float(np.mean(aucs)) if aucs else None, "events": len(ev), "windows": len(rows),
                  "recall": float(np.mean([r["matched"] for r in ev])) if ev else None,
                  "recall_ci95": boot_ci([r["matched"] for r in ev], rng),
                  "early_warning_rate": float(np.mean([r["early"] for r in ev])) if ev else None,
                  "mean_time_to_alert": float(np.mean(ttas)) if ttas else None,
                  "false_alerts_per_100_clean": float(np.mean([r["fa"] for r in rows if r["fa"] is not None])),
                  "threshold": float(thresholds[m])}
    return res


def main():
    rng = np.random.default_rng(3)
    dev = prep(pickle.loads((OUT / "real_aux_signals.pkl").read_bytes()))
    cache = OUT / "new_signals.pkl"
    if cache.exists():
        new = pickle.loads(cache.read_bytes())
    else:
        jobs = []
        for name in NEW:
            X, y = load_new(name)
            k = min(40, len(X) // SPAN)
            jobs += [(name, int(s)) for s in np.linspace(0, len(X) - SPAN, k) if 0.02 < y[int(s):int(s) + REF].mean() < 0.98]
        with ProcessPoolExecutor(12) as ex:
            new = list(ex.map(job, jobs, chunksize=2))
        cache.write_bytes(pickle.dumps(new))
    prep(new)
    dev_names = sorted({s["scenario"] for s in dev})
    new_names = sorted({s["scenario"] for s in new})
    results = {"streams": {"development": {n: sum(s["scenario"] == n for s in dev) for n in dev_names},
                           "held_out": {n: {"windows": sum(s["scenario"] == n for s in new),
                                            "with_event": sum(s["scenario"] == n and s["event_onset"] is not None for s in new)}
                                       for n in new_names}},
               "by_delay": {}}
    for delay in DELAYS:
        # --- development: leave-one-dataset-out
        fold_est = {d: AnchoredEstimator(delay).fit([s for s in dev if s["scenario"] != d]) for d in dev_names}
        dev_scored = [dict(s, scores=all_scores(s, delay, fold_est[s["scenario"]])) for s in dev]
        dev_mae = mae_by(dev, delay, lambda s: fold_est[s["scenario"]])
        dev_det = {}
        for d in dev_names:  # thresholds from the other development streams' (cross-fit) scores
            thr = {m: pick_threshold([x for x in dev_scored if x["scenario"] != d], m)[0] for m in SCORE_METHODS}
            dev_det[d] = (thr, [x for x in dev_scored if x["scenario"] == d])
        merged_thr = {m: float(np.mean([dev_det[d][0][m] for d in dev_names])) for m in SCORE_METHODS}
        dev_report = detection([x for x in dev_scored], merged_thr, rng)
        # --- held-out streams
        est = AnchoredEstimator(delay).fit(dev)
        new_scored = [dict(s, scores=all_scores(s, delay, est)) for s in new]
        thr_final = {m: pick_threshold(dev_scored, m)[0] for m in SCORE_METHODS}
        new_report = detection(new_scored, thr_final, rng)
        new_mae = mae_by(new, delay, lambda s: est)
        results["by_delay"][str(delay)] = {
            "development_lodo": {"nowcast_mae_by_dataset": dev_mae, "detection": dev_report},
            "held_out": {"nowcast_mae_by_dataset": new_mae, "detection": new_report}}
        print("delay", delay, "done")
    (OUT / "final_results.json").write_text(json.dumps(results, indent=2))
    write_md(results, OUT / "final_results.md")
    make_figure(results, OUT / "nowcast_mae.png")


def f(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def write_md(res, path):
    L = ["# Final locked evaluation: delay-anchored estimator", "", "| held-out stream | windows | with event |", "|---|---:|---:|"]
    for n, d in res["streams"]["held_out"].items():
        L.append(f"| {n} | {d['windows']} | {d['with_event']} |")
    for delay, blocks in res["by_delay"].items():
        for part, title in (("development_lodo", "Development (leave-one-dataset-out, 8 real streams)"),
                            ("held_out", "Held-out streams (5, unseen during design)")):
            b = blocks[part]
            L += ["", f"## L = {delay}: {title}", "", "Nowcast MAE in accuracy points x100 (macro over datasets):", "",
                  "| estimator | macro MAE | " + " | ".join(b["nowcast_mae_by_dataset"]["anchored"]) + " |",
                  "|---|---:|" + "---:|" * len(b["nowcast_mae_by_dataset"]["anchored"])]
            for m, per in b["nowcast_mae_by_dataset"].items():
                L.append(f"| {m} | {100 * np.mean(list(per.values())):.2f} | " + " | ".join(f"{100 * v:.1f}" for v in per.values()) + " |")
            L += ["", "Alerting (thresholds from development data only):", "",
                  "| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |",
                  "|---|---:|---:|---:|---:|---:|"]
            for m, r in b["detection"].items():
                rc = r["recall_ci95"]
                L.append(f"| {m} | {f(r['macro_auroc'])} | {f(r['recall'])} [{f(rc[0])}, {f(rc[1])}] | {f(r['mean_time_to_alert'])} | "
                         f"{f(r['early_warning_rate'])} | {f(r['false_alerts_per_100_clean'])} |")
    path.write_text("\n".join(L), encoding="utf-8")


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, part, title in zip(axes, ("development_lodo", "held_out"), ("8 development streams (LODO)", "5 held-out streams")):
        w = 0.16
        for j, m in enumerate(NOWCAST):
            vals = [100 * np.mean(list(res["by_delay"][str(d)][part]["nowcast_mae_by_dataset"][m].values())) for d in DELAYS]
            ax.bar(np.arange(len(DELAYS)) + (j - 2) * w, vals, w, label=m)
        ax.set_xticks(range(len(DELAYS)))
        ax.set_xticklabels([f"L={d}" for d in DELAYS])
        ax.set_title(title)
    axes[0].set_ylabel("nowcast MAE (accuracy points)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
