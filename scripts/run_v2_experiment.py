"""Run the v2 delayed-label-aware warning experiment end to end.

    python scripts/run_v2_experiment.py [--quick]

Writes signals cache, results JSON, markdown tables and figures to artifacts/v2/.
Thresholds are chosen on validation seeds only; test seeds are touched once.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dews import v2  # noqa: E402
from dews.early_warning_metrics import evaluate_episode_alerts  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
BUDGET = 5.0  # false alerts per 100 clean batches
METHODS = (
    "stat_only", "conf_only", "fixed_score", "slope_forecast", "delayed_label_only",
    "cal_signals_static", "cal_full_static", "cal_full_online",
)


def _job(args):
    scenario, seed = args
    warnings.filterwarnings("ignore")
    return v2.extract_signals(v2.build_episode(scenario, seed))


def collect(seeds, workers):
    jobs = [(sc, s) for sc in v2.SCENARIOS for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_job, jobs, chunksize=2))


def clean_region(n, onset):
    end = n if onset is None else max(onset - v2.H, v2.WARMUP)
    return list(range(v2.WARMUP, end))


def all_scores(sig, delay, pools):
    s = v2.baseline_scores(sig, delay)
    s["cal_signals_static"] = v2.calibrator_scores(sig, delay, pools["sig"], use_labels=False, online=False)
    s["cal_full_static"] = v2.calibrator_scores(sig, delay, pools[delay], use_labels=True, online=False)
    s["cal_full_online"] = v2.calibrator_scores(sig, delay, pools[delay], use_labels=True, online=True)
    for name in s:
        s[name] = np.asarray(s[name], float).copy()
        s[name][: v2.WARMUP] = np.nan
    return s


def episode_eval(score, onset, n, thr):
    alerts = [i for i in range(n) if np.isfinite(score[i]) and score[i] >= thr]
    events = [] if onset is None else [onset]
    return evaluate_episode_alerts(alerts, events, horizon_batches=v2.H,
                                   clean_batch_indices=clean_region(n, onset))


def pick_threshold(val_items, method):
    pooled = np.concatenate([it["scores"][method][v2.WARMUP:] for it in val_items])
    pooled = pooled[np.isfinite(pooled)]
    cands = np.unique(np.quantile(pooled, np.linspace(0.5, 1.0, 120)))
    best = None
    for thr in cands:
        fa = clean = matched = events = 0
        lead_sum = 0.0
        for it in val_items:
            n = len(it["accuracy"])
            m = episode_eval(it["scores"][method], it["event_onset"], n, thr)
            cr = clean_region(n, it["event_onset"])
            fa += sum(a in set(cr) for a in m["unmatched_alert_batches"])
            clean += len(cr)
            if it["event_onset"] is not None:
                events += 1
                matched += m["matched_event_count"]
                lead_sum += m["warning_lead_batches"] or 0.0
        rate = 100.0 * fa / max(clean, 1)
        if rate > BUDGET:
            continue
        key = (-(matched / max(events, 1)), -lead_sum, rate, -thr)
        if best is None or key < best[0]:
            best = (key, float(thr), rate, matched / max(events, 1))
    if best is None:
        return float(cands.max()) + 1e-9, 0.0, 0.0
    return best[1], best[2], best[3]


def boot_ci(values, rng, n_boot=2000):
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("signals_quick.pkl" if args.quick else "signals.pkl")
    tr_seeds, va_seeds, te_seeds = v2.TRAIN_SEEDS, v2.VALID_SEEDS, v2.TEST_SEEDS
    if args.quick:
        tr_seeds, va_seeds, te_seeds = tr_seeds[:12], va_seeds[:6], te_seeds[:10]
    if cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        t0 = time.time()
        data = {k: collect(s, args.workers) for k, s in
                (("train", tr_seeds), ("valid", va_seeds), ("test", te_seeds))}
        cache.write_bytes(pickle.dumps(data))
        print(f"signals extracted in {time.time() - t0:.0f}s")

    for split in data.values():
        for sig in split:
            o = v2.outcome(sig["accuracy"])
            sig["event_onset"] = o["event_onset"]
            sig["targets"] = v2.state_targets(sig["accuracy"])

    prevalence = {}
    for sc in v2.SCENARIOS:
        eps = [s for s in data["test"] if s["scenario"] == sc]
        prevalence[sc] = {"episodes": len(eps), "with_event": sum(e["event_onset"] is not None for e in eps)}

    results = {"protocol": {"H": v2.H, "K": v2.K, "drop": v2.DROP, "budget_per_100": BUDGET,
                            "delays": list(v2.DELAYS), "train_seeds": [tr_seeds[0], tr_seeds[-1]],
                            "valid_seeds": [va_seeds[0], va_seeds[-1]], "test_seeds": [te_seeds[0], te_seeds[-1]],
                            "test_event_prevalence": prevalence},
               "by_delay": {}}
    rng = np.random.default_rng(0)
    pools = {"sig": v2.training_pool(data["train"], 0, use_labels=False)}
    for delay in v2.DELAYS:
        pools[delay] = v2.training_pool(data["train"], delay, use_labels=True)

    for delay in v2.DELAYS:
        t0 = time.time()
        scored = {}
        for split in ("valid", "test"):
            scored[split] = [dict(sig, scores=all_scores(sig, delay, pools)) for sig in data[split]]
        per_method = {}
        for method in METHODS:
            thr, val_fa, val_recall = pick_threshold(scored["valid"], method)
            rows = []
            for it in scored["test"]:
                n = len(it["accuracy"])
                m = episode_eval(it["scores"][method], it["event_onset"], n, thr)
                rows.append({"scenario": it["scenario"], "seed": it["seed"], "event": it["event_onset"] is not None,
                             "matched": m["matched_event_count"] > 0,
                             "lead": m["warning_lead_batches"], "lag": m["detection_lag_batches"],
                             "fa100": m["false_alerts_per_100_clean_batches"]})
            by_scn = {}
            for sc in v2.SCENARIOS:
                r = [x for x in rows if x["scenario"] == sc]
                ev = [x for x in r if x["event"]]
                by_scn[sc] = {
                    "events": len(ev),
                    "recall": float(np.mean([x["matched"] for x in ev])) if ev else None,
                    "early_warning_rate": float(np.mean([x["lead"] is not None for x in ev])) if ev else None,
                    "mean_lead": float(np.mean([x["lead"] for x in ev if x["lead"] is not None])) if any(x["lead"] is not None for x in ev) else None,
                    "mean_lag": float(np.mean([x["lag"] for x in ev if x["lag"] is not None])) if any(x["lag"] is not None for x in ev) else None,
                    "fa_per_100_clean": float(np.mean([x["fa100"] for x in r if x["fa100"] is not None])) if r else None,
                }
            harm = [x for x in rows if x["event"]]
            noev = [x for x in rows if not x["event"] and x["scenario"] in ("clean", "harmless_shift")]
            # threshold-free quality on the state target (post warm-up)
            from sklearn.metrics import roc_auc_score, brier_score_loss
            ys, ss, hs = [], [], []
            for it in scored["test"]:
                mask = np.isfinite(it["targets"]) & np.isfinite(it["scores"][method])
                ys.append(it["targets"][mask]); ss.append(it["scores"][method][mask])
                hs.append(np.full(mask.sum(), it["scenario"] in ("harmful_gradual", "harmful_abrupt", "concept_gradual")))
            ys, ss, hs = map(np.concatenate, (ys, ss, hs))
            auc = float(roc_auc_score(ys, ss)) if len(set(ys)) == 2 else None
            auc_h = float(roc_auc_score(ys[hs], ss[hs])) if len(set(ys[hs])) == 2 else None
            brier = float(brier_score_loss(ys, np.clip(ss, 0, 1))) if method.startswith("cal_") else None
            per_method[method] = {
                "threshold": thr, "val_false_alerts_per_100": val_fa, "val_recall": val_recall,
                "overall_recall": float(np.mean([x["matched"] for x in harm])) if harm else None,
                "overall_recall_ci95": boot_ci([x["matched"] for x in harm], rng),
                "early_warning_rate": float(np.mean([x["lead"] is not None for x in harm])) if harm else None,
                "early_warning_ci95": boot_ci([x["lead"] is not None for x in harm], rng),
                "false_alerts_per_100_no_event_eps": float(np.mean([x["fa100"] for x in noev if x["fa100"] is not None])) if noev else None,
                "auroc_state_target": auc, "auroc_harmful_only": auc_h, "brier": brier,
                "by_scenario": by_scn,
            }
        results["by_delay"][str(delay)] = per_method
        print(f"delay {delay} done in {time.time() - t0:.0f}s")

    (OUT / ("results_quick.json" if args.quick else "results.json")).write_text(json.dumps(results, indent=2))
    write_markdown(results, OUT / ("results_quick.md" if args.quick else "results.md"))
    if not args.quick:
        make_figures(results, data, pools)
    return results


def fmt(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def write_markdown(results, path):
    lines = ["# v2 results (held-out test seeds)", "",
             f"Alert budget: {BUDGET} false alerts / 100 clean batches, thresholds selected on validation seeds.", "",
             "Test event prevalence: " + ", ".join(f"{k}: {v['with_event']}/{v['episodes']}" for k, v in results["protocol"]["test_event_prevalence"].items()), ""]
    for delay, methods in results["by_delay"].items():
        lines += [f"## Label delay L = {delay}", "",
                  "| method | recall | early-warning rate | FA/100 (no-event eps) | AUROC (state) | AUROC (harmful) | Brier |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m, r in methods.items():
            rc, ew = r["overall_recall_ci95"], r["early_warning_ci95"]
            lines.append(f"| {m} | {fmt(r['overall_recall'])} [{fmt(rc[0]) if rc else 'n/a'}, {fmt(rc[1]) if rc else 'n/a'}] | "
                         f"{fmt(r['early_warning_rate'])} [{fmt(ew[0]) if ew else 'n/a'}, {fmt(ew[1]) if ew else 'n/a'}] | "
                         f"{fmt(r['false_alerts_per_100_no_event_eps'])} | {fmt(r['auroc_state_target'])} | "
                         f"{fmt(r['auroc_harmful_only'])} | {fmt(r['brier'], 3)} |")
        lines += ["", "Per-scenario recall / early-warning rate / mean lead:", "",
                  "| method | " + " | ".join(s for s in v2.SCENARIOS if s not in ("clean", "harmless_shift")) + " | FA/100 clean | FA/100 harmless |",
                  "|---|" + "---:|" * 5]
        for m, r in methods.items():
            cells = []
            for s in ("harmful_gradual", "harmful_abrupt", "concept_gradual"):
                b = r["by_scenario"][s]
                cells.append(f"{fmt(b['recall'])} / {fmt(b['early_warning_rate'])} / {fmt(b['mean_lead'], 1)}")
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {fmt(r['by_scenario']['clean']['fa_per_100_clean'])} | {fmt(r['by_scenario']['harmless_shift']['fa_per_100_clean'])} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(results, data, pools):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. recall and early-warning rate by method at each delay
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    width = 0.25
    for ax, key, title in zip(axes, ("overall_recall", "early_warning_rate"),
                              ("Event recall (alert within +/-5 batches)", "Early-warning rate (alert before onset)")):
        for j, delay in enumerate(v2.DELAYS):
            vals = [results["by_delay"][str(delay)][m][key] or 0 for m in METHODS]
            ax.bar(np.arange(len(METHODS)) + (j - 1) * width, vals, width, label=f"L={delay}")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "recall_by_method.png", dpi=140)
    plt.close(fig)

    # 2. example timeline: gradual harmful episode, L=2
    delay = 2
    sig = next(s for s in data["test"] if s["scenario"] == "harmful_gradual")
    sc = all_scores(sig, delay, pools)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sig["accuracy"], color="tab:gray", label="realized accuracy (delayed)")
    ax.axvline(sig["event_onset"], color="red", ls="--", label="event onset")
    ax.axvline(sig["onset"], color="orange", ls=":", label="intervention onset")
    ax2 = ax.twinx()
    for name in ("fixed_score", "cal_full_online"):
        ax2.plot(sc[name], label=name)
        ax2.axhline(results["by_delay"][str(delay)][name]["threshold"], ls="--", lw=0.8, color=ax2.lines[-1].get_color())
    ax.set_xlabel("batch"); ax.set_ylabel("accuracy"); ax2.set_ylabel("score")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    ax.set_title(f"Held-out harmful_gradual, seed {sig['seed']}, L={delay} (dashed = validation thresholds)")
    fig.tight_layout()
    fig.savefig(OUT / "timeline_harmful_gradual.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    run(parser.parse_args())
