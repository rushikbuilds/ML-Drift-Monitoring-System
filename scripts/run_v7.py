"""v7 runner: development thresholds, confirmation data, one-time confirmation and verdict.

    python scripts/run_v7.py develop                      # freeze thresholds + WARNING calibrator
    python scripts/run_v7.py download                     # fetch the 6 confirmation datasets
    python scripts/run_v7.py extract --shard i --shards k # confirmation episodes (primary model)
    python scripts/run_v7.py confirm                      # score once, write docs/v7_verdict.md

Protocol: docs/v7_protocol.md. ``confirm`` refuses to run if the frozen files changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import resample_benchmark as rb  # noqa: E402
from dews import v2  # noqa: E402
from dews import v6_config as v6cfg  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from dews import v7_config as cfg  # noqa: E402
from dews import v7_monitor as mon  # noqa: E402
from dews.v6_signals import extract_episode  # noqa: E402
from scripts.run_v6_develop import setting_key  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402

OUT = ROOT / "artifacts" / "v7"
V6 = ROOT / "artifacts" / "v6"
B = 2000
RNG = np.random.default_rng(20261109)
DELAY, FRACTION, MODEL = cfg.PRIMARY["label_delay"], cfg.PRIMARY["label_fraction"], cfg.PRIMARY["model"]
V6_KEY = setting_key(MODEL, DELAY, FRACTION)
METHODS = ("watch", "warning", "fixed_dews_score", "label_trend")


# ----------------------------------------------------------------------------- scoring
def score_all(episodes, calibrator):
    out = {m: np.full((len(episodes), v6cfg.N_BATCHES), np.nan) for m in METHODS}
    for i, ep in enumerate(episodes):
        n = len(ep["accuracy"])
        base = ev.baseline_scores(ep, DELAY, FRACTION)
        out["watch"][i, :n] = mon.watch_score(ep, DELAY, FRACTION)
        out["warning"][i, :n] = calibrator(ep).score(ep)
        for m in ("fixed_dews_score", "label_trend"):
            s = np.asarray(base[m], float).copy()
            s[: v6cfg.WARMUP_BATCHES] = np.nan
            out[m][i, :n] = s
    return out


def develop() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = load_signals("development", MODEL)
    datasets = sorted({e["dataset"] for e in eps})
    folds = {d: mon.WarningCalibrator(DELAY, FRACTION).fit([e for e in eps if e["dataset"] != d]) for d in datasets}
    scores = score_all(eps, lambda e: folds[e["dataset"]])
    masks = ev.build_masks(eps, DELAY)
    v6_thr = json.loads((V6 / "frozen_choices.json").read_text())["thresholds"][V6_KEY]
    thresholds = {"watch": ev.select_threshold(scores["watch"], masks),
                  "warning": ev.select_threshold(scores["warning"], masks),
                  "fixed_dews_score": v6_thr["fixed_dews_score"], "label_trend": v6_thr["label_trend"]}
    dev = {m: {k: v for k, v in ev.metrics(scores[m], thresholds[m], masks).items() if k in ("false_alerts_per_100_clean", "harmless_alert_share")}
           | {"t1_gradual": ev.metrics(scores[m], thresholds[m], masks)["gradual"]["t1_share"],
              "t2_harmful": ev.metrics(scores[m], thresholds[m], masks)["harmful"]["t2_share"],
              "harm_auroc": ev.harm_auroc(scores[m], masks)} for m in METHODS}
    (OUT / "frozen_v7.json").write_text(json.dumps({"thresholds": thresholds, "development": dev}, indent=2))
    (OUT / "frozen_v7_calibrator.pkl").write_bytes(pickle.dumps(mon.WarningCalibrator(DELAY, FRACTION).fit(eps)))
    for m, r in dev.items():
        print(f"{m:17s} thr {thresholds[m]:.4f}  T1 {r['t1_gradual']:.2f}  T2 {r['t2_harmful']:.2f}  FA {r['false_alerts_per_100_clean']:.1f}"
              f"  harmless {r['harmless_alert_share']:.2f}  AUROC {r['harm_auroc']:.2f}")
    for name in ("frozen_v7.json", "frozen_v7_calibrator.pkl"):
        print(name, hashlib.sha256((OUT / name).read_bytes()).hexdigest())


# ----------------------------------------------------------------------------- data
def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, did in cfg.CONFIRMATION_DATASETS.items():
        path = OUT / f"data_{name}.pkl"
        if path.exists():
            continue
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = d.data, d.target.astype(str)
        X = X.apply(lambda c: c.astype("category").cat.codes if str(c.dtype) in ("category", "object") else c).astype(float)
        X = X.fillna(X.median()).loc[:, X.nunique() > 1].iloc[:, : cfg.MAX_FEATURES].reset_index(drop=True)
        minority = y.value_counts().index[-1]
        yy = (y == minority).astype(int).to_numpy()
        if name == "miniboone":  # rows are stored sorted by class; recorded deviation in docs/v7_protocol.md
            perm = np.random.default_rng(2026).permutation(len(yy))
            X, yy = X.iloc[perm].reset_index(drop=True), yy[perm]
        pd.to_pickle((X, yy), path)
        print(f"{name}: {X.shape}, positive share {yy.mean():.3f}, valid block starts {len(rb._valid_block_starts(yy))}")


def plan():
    names = list(cfg.CONFIRMATION_DATASETS)
    jobs = []
    for di, name in enumerate(names):
        index = 0
        for kind in rb.EPISODE_KINDS:
            for _ in range(v6cfg.EPISODES_PER_DATASET[kind]):
                jobs.append((name, kind, cfg.CONFIRMATION_SEED_BASE + 100 * di + index))
                index += 1
    return jobs


def extract(shard: int, shards: int) -> None:
    out = []
    for name, kind, seed in plan()[shard::shards]:
        X, y = pd.read_pickle(OUT / f"data_{name}.pkl")
        ep = rb.build_episode(X, y, dataset=name, kind=kind, seed=seed, model_kind=MODEL)
        out.append(extract_episode(ep))
    (OUT / f"signals_confirmation_{shard}of{shards}.pkl").write_bytes(pickle.dumps(out))
    print(f"shard {shard}: {len(out)} episodes", flush=True)


# ----------------------------------------------------------------------------- confirmation
def boot(x):
    x = np.asarray(x, float)
    m = [RNG.choice(x, len(x)).mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def evaluate(episodes, scores, thresholds):
    masks = ev.build_masks(episodes, DELAY)
    res = {m: ev.metrics(scores[m], thresholds[m], masks) for m in METHODS}
    for m in METHODS:
        res[m]["harm_auroc"] = ev.harm_auroc(scores[m], masks)
    per = {}
    for d in sorted({e["dataset"] for e in episodes}):
        idx = [i for i, e in enumerate(episodes) if e["dataset"] == d]
        sub = ev.build_masks([episodes[i] for i in idx], DELAY)
        per[d] = {m: {"t1": ev.metrics(scores[m][idx], thresholds[m], sub)["harmful"]["t1_share"],
                      "t2": ev.metrics(scores[m][idx], thresholds[m], sub)["harmful"]["t2_share"],
                      "harmful": int((sub.harmful_gradual | sub.harmful_abrupt).sum())} for m in METHODS}
    return res, per, masks


def natural(thresholds):
    windows = [w for f in sorted(V6.glob("natural_signals_*of*.pkl")) for w in pickle.loads(f.read_bytes())]
    w_hits, l_hits, fa = [], [], {"watch": [0, 0], "label_trend": [0, 0]}
    for w in windows:
        n, t = len(w["accuracy"]), np.arange(len(w["accuracy"]))
        sc = {"watch": mon.watch_score(w, DELAY, FRACTION), "label_trend": ev.baseline_scores(w, DELAY, FRACTION)["label_trend"]}
        event = v2.outcome(np.asarray(w["accuracy"]))["event_onset"]
        clean = (t >= cfg.BURN_IN_BATCHES) & ((t < event - v6cfg.HORIZON_BATCHES) if event is not None else True)
        hits = {}
        for m, s in sc.items():
            alerts = np.where(np.isfinite(s), s >= thresholds[m], False) & (t >= cfg.BURN_IN_BATCHES)
            fa[m][0] += int((alerts & clean).sum()); fa[m][1] += int(clean.sum())
            if event is not None:
                win = (t >= max(event - v6cfg.HORIZON_BATCHES, cfg.BURN_IN_BATCHES)) & (t < v6cfg.confirmation_batch(event, DELAY))
                hits[m] = bool((alerts & win).any())
        if event is not None:
            w_hits.append(hits["watch"]); l_hits.append(hits["label_trend"])
    a, b = np.array(w_hits, float), np.array(l_hits, float)
    idx = [RNG.choice(len(a), len(a)) for _ in range(B)]
    ci = [float(np.percentile([a[i].mean() - b[i].mean() for i in idx], q)) for q in (2.5, 97.5)]
    return {"t2_watch": float(a.mean()), "t2_label_trend": float(b.mean()), "difference": float(a.mean() - b.mean()),
            "ci95": ci, "event_windows": len(a), "false_alerts": {m: 100 * x / max(y, 1) for m, (x, y) in fa.items()}}


def confirm() -> None:
    frozen = json.loads((OUT / "frozen_v7.json").read_text())
    expected = (ROOT / "docs" / "v7_protocol.md").read_text(encoding="utf-8")
    for name in ("frozen_v7.json", "frozen_v7_calibrator.pkl"):
        digest = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        if digest[:16] not in expected:
            raise SystemExit(f"{name} fingerprint {digest[:16]} is not recorded in docs/v7_protocol.md; refusing to run")
    thresholds = frozen["thresholds"]
    calibrator = pickle.loads((OUT / "frozen_v7_calibrator.pkl").read_bytes())
    files = sorted(OUT.glob("signals_confirmation_*of*.pkl"))
    episodes = sorted([e for f in files for e in pickle.loads(f.read_bytes())], key=lambda e: (e["dataset"], e["seed"]))
    if {e["dataset"] for e in episodes} != set(cfg.CONFIRMATION_DATASETS):
        raise SystemExit("confirmation signals incomplete")
    scores = score_all(episodes, lambda e: calibrator)
    res, per, masks = evaluate(episodes, scores, thresholds)
    w, lab, warn = res["watch"], res["label_trend"], res["warning"]
    hyp = {}
    hyp["C1"] = {"value": w["gradual"]["t1_share"], "ci95": boot(w["gradual"]["t1_hits"]), "median_lead": w["gradual"]["t1_median_lead"],
                 "false_alerts": w["false_alerts_per_100_clean"], "episodes": w["gradual"]["episodes"]}
    hyp["C1"]["pass"] = bool(hyp["C1"]["value"] >= 0.60 and (hyp["C1"]["median_lead"] or 0) >= 2 and hyp["C1"]["false_alerts"] <= 5)
    s, l = np.array(w["harmful"]["t2_hits"], float), np.array(lab["harmful"]["t2_hits"], float)
    idx = [RNG.choice(len(s), len(s)) for _ in range(B)]
    ci = [float(np.percentile([s[i].mean() - l[i].mean() for i in idx], q)) for q in (2.5, 97.5)]
    hyp["C2"] = {"value": float(s.mean() - l.mean()), "ci95": ci, "t2_watch": float(s.mean()), "t2_label_trend": float(l.mean()),
                 "episodes": len(s)}
    hyp["C2"]["pass"] = bool(hyp["C2"]["value"] >= 0.10 and ci[0] > 0)
    hyp["C3"] = {"value": warn["harm_auroc"], "harmless_alert_share": warn["harmless_alert_share"],
                 "harmful": int((masks.harmful_gradual | masks.harmful_abrupt).sum()), "harmless": int(masks.harmless_shift.sum())}
    hyp["C3"]["pass"] = bool(warn["harm_auroc"] >= 0.80 and warn["harmless_alert_share"] <= 0.10)
    with_harm = {d: r for d, r in per.items() if r["watch"]["harmful"] > 0}
    wins = {d: bool((r["watch"]["t1"] or 0) > (r["fixed_dews_score"]["t1"] or 0) and (r["watch"]["t2"] or 0) > (r["fixed_dews_score"]["t2"] or 0))
            for d, r in with_harm.items()}
    hyp["C4"] = {"value": sum(wins.values()), "datasets": len(wins), "per_dataset": wins,
                 "fixed_false_alerts": res["fixed_dews_score"]["false_alerts_per_100_clean"]}
    hyp["C4"]["pass"] = bool(len(wins) and sum(wins.values()) / len(wins) >= 0.70)
    nat = natural(thresholds)
    hyp["C5"] = {**nat, "value": nat["difference"], "pass": bool(nat["difference"] >= 0)}
    overall = ("SUCCESS" if hyp["C1"]["pass"] and hyp["C3"]["pass"] else
               "PARTIAL SUCCESS" if hyp["C1"]["pass"] or hyp["C4"]["pass"] else "FAILURE")

    # post-hoc on the v6 held-out datasets (already seen; cannot confirm anything)
    v6_held = load_signals("held_out", MODEL)
    post, _, _ = evaluate(v6_held, score_all(v6_held, lambda e: calibrator), thresholds)
    counts = {}
    for e in episodes:
        counts.setdefault(e["dataset"], {}).setdefault(f"{e['kind']}|{e['category']}", 0)
        counts[e["dataset"]][f"{e['kind']}|{e['category']}"] += 1
    verdict = {"overall": overall, "hypotheses": hyp, "confirmation": strip(res), "per_dataset": per,
               "post_hoc_v6_held_out": strip(post), "episode_counts": counts, "thresholds": thresholds}
    (OUT / "v7_verdict.json").write_text(json.dumps(verdict, indent=2))
    write_markdown(verdict, ROOT / "docs" / "v7_verdict.md")
    for h, v in hyp.items():
        print(h, "PASS" if v["pass"] else "FAIL", v["value"])
    print("OVERALL:", overall)


def strip(res):
    return {m: {"t1_gradual": r["gradual"]["t1_share"], "t1_median_lead": r["gradual"]["t1_median_lead"],
                "t1_abrupt": r["abrupt"]["t1_share"], "t2_harmful": r["harmful"]["t2_share"],
                "false_alerts": r["false_alerts_per_100_clean"], "harmless_alert_share": r["harmless_alert_share"],
                "harm_auroc": r["harm_auroc"], "harmful_gradual_episodes": r["gradual"]["episodes"]} for m, r in res.items()}


def f(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def write_markdown(v, path):
    h = v["hypotheses"]
    P = lambda k: "PASS" if h[k]["pass"] else "FAIL"  # noqa: E731
    L = ["# v7 Verdict: Stream-Normalized Two-Tier Monitor", "",
         "Generated by `scripts/run_v7.py confirm` from a one-time run on 6 datasets never used before "
         "(adult, MiniBooNE, higgs, nomao, numerai28.6, Click_prediction_small). Protocol: [v7_protocol.md](v7_protocol.md). "
         "The v6 verdict ([verdict.md](verdict.md), FAILURE) is unchanged.", "",
         "Setting: logistic regression, label delay 5 batches, all labels.", "",
         f"## Overall: **{v['overall']}**", "",
         "| ID | Hypothesis | Result | Target | Verdict |", "|---|---|---|---|---|",
         f"| C1 | WATCH warns before the accuracy drop (gradual harmful shifts) | {f(h['C1']['value'])} [{f(h['C1']['ci95'][0])}, {f(h['C1']['ci95'][1])}], "
         f"median lead {f(h['C1']['median_lead'], 1)} batches, {f(h['C1']['false_alerts'], 1)} false alerts / 100 clean batches (n = {h['C1']['episodes']}) | ≥ 0.60, lead ≥ 2, FA ≤ 5 | {P('C1')} |",
         f"| C2 | WATCH warns before labels confirm, better than waiting for labels | {f(h['C2']['t2_watch'])} vs {f(h['C2']['t2_label_trend'])}, "
         f"difference {f(h['C2']['value'])} [{f(h['C2']['ci95'][0])}, {f(h['C2']['ci95'][1])}] (n = {h['C2']['episodes']}) | ≥ +0.10, CI > 0 | {P('C2')} |",
         f"| C3 | WARNING separates harmful from harmless shifts | AUROC {f(h['C3']['value'])}; harmless shifts with an alert {f(h['C3']['harmless_alert_share'])} "
         f"({h['C3']['harmful']} harmful, {h['C3']['harmless']} harmless) | AUROC ≥ 0.80, ≤ 0.10 | {P('C3')} |",
         f"| C4 | WATCH beats the raw DEWS score | wins on {h['C4']['value']} of {h['C4']['datasets']} datasets with harmful shifts | ≥ 70% | {P('C4')} |",
         f"| C5 | WATCH carries over to natural streams | {f(h['C5']['t2_watch'])} vs {f(h['C5']['t2_label_trend'])}, difference {f(h['C5']['value'])} "
         f"[{f(h['C5']['ci95'][0])}, {f(h['C5']['ci95'][1])}] ({h['C5']['event_windows']} event windows) | ≥ 0 | {P('C5')} |",
         "", "## All methods on the confirmation datasets", "",
         "| Method | T1 gradual | Median lead | T2 harmful | False alerts / 100 clean | Harmless shifts alerting | Harm AUROC |", "|---|---|---|---|---|---|---|"]
    for m, r in v["confirmation"].items():
        L.append(f"| {m} | {f(r['t1_gradual'])} | {f(r['t1_median_lead'], 1)} | {f(r['t2_harmful'])} | {f(r['false_alerts'], 1)} | {f(r['harmless_alert_share'])} | {f(r['harm_auroc'])} |")
    L += ["", "## Per confirmation dataset (harmful shifts: T1 / T2)", "", "| Dataset | Harmful | WATCH | Raw DEWS score | Label trend |", "|---|---|---|---|---|"]
    for d, r in v["per_dataset"].items():
        L.append(f"| {d} | {r['watch']['harmful']} | {f(r['watch']['t1'])} / {f(r['watch']['t2'])} | {f(r['fixed_dews_score']['t1'])} / {f(r['fixed_dews_score']['t2'])} | {f(r['label_trend']['t1'])} / {f(r['label_trend']['t2'])} |")
    L += ["", "## Post-hoc: v6 held-out datasets (already seen; not confirmation)", "",
          "| Method | T1 gradual | T2 harmful | False alerts / 100 clean | Harmless shifts alerting | Harm AUROC |", "|---|---|---|---|---|---|"]
    for m, r in v["post_hoc_v6_held_out"].items():
        L.append(f"| {m} | {f(r['t1_gradual'])} | {f(r['t2_harmful'])} | {f(r['false_alerts'], 1)} | {f(r['harmless_alert_share'])} | {f(r['harm_auroc'])} |")
    L += ["", f"Natural streams (C5) false alerts per 100 clean batches: WATCH {f(h['C5']['false_alerts']['watch'], 1)}, "
          f"label trend {f(h['C5']['false_alerts']['label_trend'], 1)}. Raw DEWS score false alerts on confirmation data: {f(h['C4']['fixed_false_alerts'], 1)}."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["develop", "download", "extract", "confirm"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    {"develop": develop, "download": download, "confirm": confirm}.get(a.stage, lambda: extract(a.shard, a.shards))()
