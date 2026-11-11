"""Charts for the B.Tech report, drawn only from saved results in artifacts/.

    python report/figures/make_report_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "legend.frameon": False, "pdf.fonttype": 42})
C = {"blue": "#2563EB", "orange": "#EA580C", "green": "#059669", "red": "#DC2626", "gray": "#64748B",
     "purple": "#7C3AED", "light": "#CBD5E1"}


def load(p):
    return json.loads((ART / p).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(name)


def label_bars(ax, bars, fmt="{:.2f}", dy=0.01):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7.5)


def day2_lambda():
    lams = [1.5, 2.0, 2.5, 3.0, 4.0]
    share = [load(f"v6/day2_balance_lambda{x}.json")["models"]["logistic_regression"]["harmful_share"] for x in lams]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bars = ax.bar([str(x) for x in lams], share, color=C["blue"], width=0.55)
    label_bars(ax, bars)
    ax.axhline(0.25, color=C["red"], ls="--", lw=1)
    ax.text(4.35, 0.255, "gate: 25% harmful", color=C["red"], ha="right", va="bottom", fontsize=8)
    ax.set(xlabel=r"maximum shift strength $\lambda_{max}$", ylabel="share of shifts that are harmful", ylim=(0, 0.32))
    save(fig, "ch_day2_lambda")


def v6_per_dataset():
    rows = [("airlines", .29, .86, .43, .86), ("bank_marketing", .33, 1.0, 0, .17), ("electricity", .11, .79, .74, 1.0),
            ("ldpa", .26, .79, .37, .68), ("sea", 1.0, 1.0, 0, 0)]
    v = load("v6/verdict.json")
    assert abs(v["hypotheses"]["H1"]["value"] - 0.2222) < 1e-3  # numbers below are the verdict.md table
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    x = np.arange(len(rows))
    for ax, k, t in [(axes[0], 1, "T1: before the accuracy drop"), (axes[1], 2, "T2: before labels confirm")]:
        ax.bar(x - 0.2, [r[k] for r in rows], 0.4, color=C["blue"], label="v6 risk calibrator")
        ax.bar(x + 0.2, [r[k + 2] for r in rows], 0.4, color=C["light"], label="original DEWS score")
        ax.set_xticks(x, [r[0].replace("_", chr(10)) for r in rows])
        ax.set_title(t)
    axes[0].set_ylabel("share of harmful shifts warned")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "ch_v6_per_dataset.pdf"); plt.close(fig); print("ch_v6_per_dataset")


def v7_methods():
    v = load("v7/v7_verdict.json")["confirmation"]
    names = {"watch": "Drift watch", "warning": "v7 warning\ncalibrator", "fixed_dews_score": "Original\nDEWS score", "label_trend": "Waiting for\nlabels"}
    metrics = [("t1_gradual", "T1 (gradual)"), ("t2_harmful", "T2"), ("harmless_alert_share", "harmless alarms")]
    cols = [C["green"], C["blue"], C["orange"]]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    x = np.arange(len(names))
    for j, ((k, lab), c) in enumerate(zip(metrics, cols)):
        bars = ax.bar(x + (j - 1) * 0.26, [v[m][k] for m in names], 0.26, color=c, label=lab)
        label_bars(ax, bars, dy=0.01)
    ax.set_xticks(x, names.values())
    ax.set(ylim=(0, 1.15), ylabel="share of episodes")
    ax.legend(ncol=3, loc="upper right", fontsize=7.5)
    save(fig, "ch_v7_methods")


def v7_per_dataset():
    v = load("v7/v7_verdict.json")["per_dataset"]
    ds = list(v)
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    x = np.arange(len(ds))
    for j, (m, lab, c) in enumerate([("watch", "Drift watch", C["green"]), ("fixed_dews_score", "Original DEWS score", C["light"]),
                                     ("label_trend", "Waiting for labels", C["gray"])]):
        ax.bar(x + (j - 1) * 0.26, [v[d][m]["t1"] for d in ds], 0.26, color=c, label=lab)
    ax.set_xticks(x, [f"{d.replace('_', ' ')}\n(n={v[d]['watch']['harmful']})" for d in ds])
    ax.set(ylabel="T1 share (harmful shifts)", ylim=(0, 1.12))
    ax.legend(ncol=3, loc="upper right", fontsize=7.5)
    save(fig, "ch_v7_per_dataset")


def v8_per_dataset():
    v = load("v8/v8_verdict.json")["per_dataset"]
    ds = [d for d in v if v[d]["auroc_harm"] is not None]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    x = np.arange(len(ds))
    b1 = ax.bar(x - 0.2, [v[d]["auroc_harm"] for d in ds], 0.4, color=C["orange"], label="Harm tier")
    b2 = ax.bar(x + 0.2, [v[d]["auroc_watch"] for d in ds], 0.4, color=C["green"], label="Drift watch")
    label_bars(ax, b1); label_bars(ax, b2)
    ax.axhline(0.5, color=C["gray"], lw=0.8, ls=":")
    ax.axhline(0.8, color=C["red"], lw=1, ls="--")
    ax.text(len(ds) - 0.55, 0.81, "target 0.80", color=C["red"], fontsize=8, ha="right")
    ax.set_xticks(x, [f"{d.replace('_', ' ')}\n({v[d]['harmful']} / {v[d]['harmless']})" for d in ds])
    ax.set(ylabel="harmful vs harmless AUROC", ylim=(0, 1.1))
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    save(fig, "ch_v8_per_dataset")


def baselines():
    m = load("posthoc/v8_baselines.json")["methods"]
    nice = {"harm tier (importance-weighted reference accuracy, frozen)": "Harm tier (frozen)", "kNN reference accuracy": "kNN reference accuracy",
            "disagreement": "Disagreement", "entropy": "Entropy", "drift watch (frozen)": "Drift watch (frozen)",
            "confidence_drop": "Confidence drop", "cbpe_drop": "Delay-aware CBPE"}
    uses_ref = {"Harm tier (frozen)", "kNN reference accuracy", "Disagreement"}
    items = sorted(((nice[k], r) for k, r in m.items()), key=lambda t: t[1]["auroc"])
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    y = np.arange(len(items))
    for i, (n, r) in enumerate(items):
        c = C["orange"] if n in uses_ref else C["gray"]
        ax.errorbar(r["auroc"], i, xerr=[[r["auroc"] - r["ci95"][0]], [r["ci95"][1] - r["auroc"]]], fmt="o", color=c, capsize=3)
        ax.text(r["ci95"][1] + 0.01, i, f"{r['auroc']:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [n for n, _ in items])
    ax.axvline(0.5, color=C["gray"], lw=0.8, ls=":")
    ax.set(xlabel="harmful vs harmless AUROC (95% CI), v8 confirmation data", xlim=(0.4, 0.95))
    ax.plot([], [], "o", color=C["orange"], label="uses labeled reference")
    ax.plot([], [], "o", color=C["gray"], label="label-free only")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "ch_baselines")


def signal_families():
    b = load("ablations/ablations.json")["B_watch_signals"]
    pts = {k.replace("only ", ""): r for k, r in b.items() if k.startswith("only ")}
    shift = {"drift_score", "mean_ks", "rws", "positive_rate_shift"}
    names = {"drift_score": "DEWS score", "mean_ks": "mean KS", "rws": "relevance-weighted", "positive_rate_shift": "positive rate",
             "disagreement": "disagreement", "confidence_drop": "confidence drop", "entropy": "entropy"}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for k, r in pts.items():
        c = C["blue"] if k in shift else C["orange"]
        ax.scatter(r["t1_gradual"], r["harm_auroc"], color=c, s=40, zorder=3)
        ax.annotate(names.get(k, k), (r["t1_gradual"], r["harm_auroc"]), xytext=(5, 3), textcoords="offset points", fontsize=7.5)
    allr = b["all 7 signals"]
    ax.scatter(allr["t1_gradual"], allr["harm_auroc"], marker="*", s=140, color=C["green"], zorder=3)
    ax.annotate("drift watch (all 7)", (allr["t1_gradual"], allr["harm_auroc"]), xytext=(-20, -14), textcoords="offset points", fontsize=7.5)
    ax.scatter([], [], color=C["blue"], label="input-shift signals")
    ax.scatter([], [], color=C["orange"], label="model-behaviour signals")
    ax.set(xlabel="T1: early warning on harmful gradual shifts", ylabel="harm AUROC", xlim=(0.35, 1.0), ylim=(0.45, 0.82))
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, "ch_signal_families")


def harm_rates():
    e = load("ablations/ablations.json")["E_harm_rates"]
    groups = list(e)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(groups))
    for j, (k, lab, c) in enumerate([("harmful_share", "all shifts", C["blue"]), ("harmful_share_important_feature", "important feature shifted", C["orange"]),
                                     ("harmful_share_unimportant_feature", "unimportant feature shifted", C["gray"])]):
        bars = ax.bar(x + (j - 1) * 0.26, [e[g][k] for g in groups], 0.26, color=c, label=lab)
        label_bars(ax, bars)
    ax.set_xticks(x, [f"{g}\n({e[g]['shift_episodes']} shifts)" for g in groups])
    ax.set(ylabel="share of shifts that harmed f", ylim=(0, 0.5))
    ax.legend(ncol=3, fontsize=7.5, loc="upper left")
    save(fig, "ch_harm_rates")


def tradeoff():
    v8 = load("v8/v8_verdict.json")["methods"]
    v10 = load("v10/v10_verdict.json")["results"]
    pts = [("Drift watch (v8 data)", v8["WATCH"]["harmless_alert_share"], v8["WATCH"]["t2_harmful"], v8["WATCH"]["false_alerts"], C["green"]),
           ("Harm tier (v8 data)", v8["HARM"]["harmless_alert_share"], v8["HARM"]["t2_harmful"], v8["HARM"]["false_alerts"], C["orange"]),
           ("Harm tier (v10 data)", v10["v8 rule, for comparison"]["harmless"], v10["v8 rule, for comparison"]["t2"], v10["v8 rule, for comparison"]["false_alerts"], C["red"]),
           ("v10 z-score rule (v10 data)", v10["v10 (frozen)"]["harmless"], v10["v10 (frozen)"]["t2"], v10["v10 (frozen)"]["false_alerts"], C["gray"])]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for n, hx, t2, fa, c in pts:
        ax.scatter(hx, t2, s=60, color=c, zorder=3)
        ax.annotate(f"{n}\nFA {fa:.1f}/100", (hx, t2), xytext=(6, -4), textcoords="offset points", fontsize=7.5)
    ax.axhline(0.5, color=C["gray"], ls=":", lw=0.8)
    ax.axvline(0.10, color=C["gray"], ls=":", lw=0.8)
    ax.set(xlabel="share of harmless shifts with an alert (lower is better)", ylabel="T2 on harmful shifts (higher is better)",
           xlim=(0, 1.25), ylim=(0.2, 1.08))
    save(fig, "ch_tradeoff")


def nowcast():
    f = load("v2/final_results.json")["by_delay"]
    meth = [("stale_last", "stale labels"), ("stale_mean3", "smoothed stale"), ("da_cbpe", "delay-aware CBPE"), ("anchored", "anchored (v4)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, part, title in [(axes[0], "development_lodo", "8 development streams"), (axes[1], "held_out", "5 held-out streams")]:
        x = np.arange(3)
        for j, (k, lab) in enumerate(meth):
            vals = []
            for L in ("0", "2", "5"):
                d = f[L][part]["nowcast_mae_by_dataset"]
                key = k if k in d else next(kk for kk in d if kk.startswith(k))
                vals.append(100 * np.mean(list(d[key].values())))
            ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, label=lab, color=[C["light"], C["gray"], C["blue"], C["orange"]][j])
        ax.set_xticks(x, ["L = 0", "L = 2", "L = 5"])
        ax.set_title(title)
    axes[0].set_ylabel("nowcast error (accuracy points)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "ch_v4_nowcast.pdf"); plt.close(fig); print("ch_v4_nowcast")


if __name__ == "__main__":
    for fn in (day2_lambda, v6_per_dataset, v7_methods, v7_per_dataset, v8_per_dataset, baselines, signal_families, harm_rates, tradeoff, nowcast):
        fn()
