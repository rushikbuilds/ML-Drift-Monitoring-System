"""Figures for the paper (paper/figures/), generated from saved, verified results.

    python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.app_data import load_episodes, replay  # noqa: E402

OUT = ROOT / "paper" / "figures"
BLUE, ORANGE, GRAY, DARK, RED = "#2F6FB0", "#C2611A", "#9AA3AF", "#24324A", "#A63A2B"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType, not Type 3 (IEEE PDF eXpress)


def transfer() -> None:
    ab = json.loads((ROOT / "artifacts" / "ablations" / "ablations.json").read_text())["A_threshold_transfer"]
    names = {"fixed_dews_score": "Fused drift score (raw)", "statistical_only": "Statistical drift (raw)",
             "mean_ks": "Mean KS (raw)", "primary_system": "Learned risk calibrator", "confidence_only": "Confidence only",
             "label_trend": "Delayed-label trend", "delay_aware_cbpe": "Delay-aware CBPE",
             "v7 WATCH (normalized)": "Drift watch (normalized)", "v8 HARM (accuracy units)": "Harm tier (accuracy units)"}
    order = ["mean_ks", "fixed_dews_score", "statistical_only", "delay_aware_cbpe", "label_trend",
             "confidence_only", "primary_system", "v7 WATCH (normalized)", "v8 HARM (accuracy units)"]
    dev = [ab[k]["development"] for k in order]
    unseen = [ab[k]["held_out"] for k in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.barh(y + 0.19, dev, height=0.36, color=GRAY, label="Development data")
    ax.barh(y - 0.19, unseen, height=0.36, color=DARK, label="Unseen data")
    ax.axvline(5, color=RED, ls="--", lw=0.9, label="Budget")
    for yi, v in zip(y, unseen):
        ax.text(v + 0.6, yi - 0.19, f"{v:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("False alerts per 100 clean batches")
    ax.set_xlim(0, 52)
    fig.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "transfer.pdf")
    plt.close(fig)


def timeline() -> None:
    eps = {(e["dataset"], e["seed"]): e for e in load_episodes("v8_confirmation")}
    e = eps[("credit_default", 9624)]
    f = replay(e, 5)
    event = int(e["event_onset"])
    confirm = event + 1 + 5
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    a1.plot(f["batch"], f["true_accuracy"], color=DARK, lw=1.0, label="True accuracy (hidden)")
    a1.plot(f["batch"], f["estimated_accuracy"], color=ORANGE, lw=1.0, label="Label-free estimate")
    a1.axhline(f["baseline_accuracy"].iloc[0] - 0.05, color=RED, ls=":", lw=0.8)
    a1.set_ylabel("Accuracy")
    a2.plot(f["batch"], f["watch_ratio"], color=BLUE, lw=1.0, label="Drift watch")
    a2.plot(f["batch"], f["harm_ratio"], color=ORANGE, lw=1.0, label="Harm warning")
    a2.axhline(1.0, color=GRAY, lw=0.8)
    a2.set_ylabel("Score / threshold")
    a2.set_xlabel("Batch")

    for ax in (a1, a2):
        for x, lbl in ((e["onset"], "shift"), (event, "drop"), (confirm, "labels")):
            ax.axvline(x, color=GRAY, ls="--", lw=0.7)
    top = a1.get_ylim()[1]
    for (x, lbl), dy in zip(((e["onset"], "shift"), (event, "drop"), (confirm, "labels")), (0, 0, 0)):
        a1.text(x, top, lbl, fontsize=6, va="bottom", ha="center", color="#5B6573")
    handles = a1.get_lines()[:2] + a2.get_lines()[:2]
    fig.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=6.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "timeline.pdf")
    plt.close(fig)


def operating_curves() -> None:
    """Post hoc: harmless-alarm share vs pre-confirmation warnings across thresholds, v8 confirmation data."""
    from dews import v6_eval as ev
    from dews import v7_monitor as mon
    from dews.harm_estimation import estimated_drop

    eps = sorted(load_episodes("v8_confirmation"), key=lambda e: (e["dataset"], e["seed"]))
    M = ev.build_masks(eps, 5)
    scores = {"Harm tier": np.vstack([estimated_drop(e["iw_accuracy"], 8) for e in eps]),
              "Drift watch": np.vstack([mon.watch_score(e, 5, 1.0) for e in eps])}
    frozen = {"Harm tier": json.loads((ROOT / "artifacts" / "v8" / "frozen_v8.json").read_text())["harm_threshold"],
              "Drift watch": json.loads((ROOT / "artifacts" / "v7" / "frozen_v7.json").read_text())["thresholds"]["watch"]}
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for (name, S), color in zip(scores.items(), (ORANGE, BLUE)):
        finite = S[np.isfinite(S)]
        pts = []
        for thr in np.unique(np.quantile(finite, np.linspace(0.0, 0.999, 120))):
            r = ev.metrics(S, thr, M)
            pts.append((r["harmless_alert_share"], r["harmful"]["t2_share"]))
        pts = np.array(sorted(pts))
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.1, label=name)
        r = ev.metrics(S, frozen[name], M)
        ax.plot(r["harmless_alert_share"], r["harmful"]["t2_share"], "o", color=color, ms=4)
    v10 = json.loads((ROOT / "artifacts" / "v10" / "v10_verdict.json").read_text())["results"]["v10 (frozen)"]
    ax.plot(v10["harmless"], v10["t2"], "s", color=DARK, ms=4, label="v10 rule (other datasets)")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel("Harmless shifts with an alert")
    ax.set_ylabel("Harmful shifts warned before\nlabels confirm (T2)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "operating.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    transfer()
    timeline()
    operating_curves()
    print("figures written to", OUT)
