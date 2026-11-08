"""v8 runner: development threshold, confirmation data, one-time confirmation and verdict.

    python scripts/run_v8.py develop                       # freeze the HARM threshold
    python scripts/run_v8.py download                      # fetch the 6 confirmation datasets
    python scripts/run_v8.py extract --shard i --shards k  # v6 signals + harm signals per episode
    python scripts/run_v8.py confirm                       # score once, write docs/v8_verdict.md
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
from dews import v6_config as v6cfg  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from dews import v7_monitor as mon  # noqa: E402
from dews import v8_config as cfg  # noqa: E402
from dews.harm_estimation import estimated_drop, harm_signals  # noqa: E402
from dews.v6_signals import extract_episode  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v8_extract import jobs, load  # noqa: E402

OUT = ROOT / "artifacts" / "v8"
B = 2000
RNG = np.random.default_rng(20261110)
DELAY, FRACTION = cfg.PRIMARY["label_delay"], cfg.PRIMARY["label_fraction"]
WATCH_THRESHOLD = json.loads((ROOT / "artifacts" / "v7" / "frozen_v7.json").read_text())["thresholds"]["watch"]


def harm_score(rec) -> np.ndarray:
    return estimated_drop(rec["iw_accuracy"], cfg.BURN_IN_BATCHES)


def separation_threshold(scores: np.ndarray, m: ev.Masks) -> float:
    finite = scores[np.isfinite(scores)]
    harmful, harmless = np.flatnonzero(m.harmful_gradual | m.harmful_abrupt), np.flatnonzero(m.harmless_shift)
    best, key = None, None
    for thr in np.unique(np.quantile(finite, np.linspace(0, 1, 200))):
        a = np.where(np.isfinite(scores), scores >= thr, False)
        if 100 * (a & m.clean).sum() / m.clean.sum() > v6cfg.FALSE_ALERT_BUDGET:
            continue
        k = ((a & m.t2)[harmful].any(axis=1).mean() - (a & m.after_onset)[harmless].any(axis=1).mean(), thr)
        if key is None or k > key:
            best, key = float(thr), k
    return best


def develop() -> None:
    eps = load_signals("development", "logistic_regression")
    harm = load("development")
    scores = np.vstack([harm_score(harm[(e["dataset"], e["seed"])]) for e in eps])
    masks = ev.build_masks(eps, DELAY)
    thr = separation_threshold(scores, masks)
    r = ev.metrics(scores, thr, masks)
    summary = {"harm_threshold": thr, "watch_threshold": WATCH_THRESHOLD,
               "development": {"t1_gradual": r["gradual"]["t1_share"], "t2_harmful": r["harmful"]["t2_share"],
                               "false_alerts": r["false_alerts_per_100_clean"], "harmless_alert_share": r["harmless_alert_share"],
                               "harm_auroc": ev.harm_auroc(scores, masks)}}
    (OUT / "frozen_v8.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("frozen_v8.json", hashlib.sha256((OUT / "frozen_v8.json").read_bytes()).hexdigest())


def download() -> None:
    for name, did in cfg.CONFIRMATION_DATASETS.items():
        path = OUT / f"data_{name}.pkl"
        if path.exists():
            continue
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = d.data, d.target.astype(str)
        X = X.apply(lambda c: c.astype("category").cat.codes if str(c.dtype) in ("category", "object") else c).astype(float)
        X = X.fillna(X.median()).loc[:, X.nunique() > 1].iloc[:, : cfg.MAX_FEATURES].reset_index(drop=True)
        yy = (y == y.value_counts().index[-1]).astype(int).to_numpy()
        pd.to_pickle((X, yy), path)
        print(f"{name}: {X.shape}, positive share {yy.mean():.3f}, label changes {int(np.sum(np.diff(yy) != 0))}, "
              f"valid block starts {len(rb._valid_block_starts(yy))}")


def extract(shard: int, shards: int) -> None:
    out = []
    for dataset, kind, seed, path in jobs("v8_confirmation")[shard::shards]:
        X, y = pd.read_pickle(path)
        ep = rb.build_episode(X, y, dataset=dataset, kind=kind, seed=seed, model_kind="logistic_regression")
        rec = extract_episode(ep)
        rec.update(harm_signals(ep.reference, ep.reference_labels, ep.model, ep.batches))
        out.append(rec)
    (OUT / f"signals_v8_confirmation_{shard}of{shards}.pkl").write_bytes(pickle.dumps(out))
    print(f"shard {shard}: {len(out)} episodes", flush=True)


def harm_values(scores, masks):
    def h(i):
        return float(np.nanmean(scores[i, masks.onset[i]:masks.onset[i] + ev.HARM_WINDOW]))
    pos = [h(i) for i in np.flatnonzero(masks.harmful_gradual | masks.harmful_abrupt)]
    neg = [h(i) for i in np.flatnonzero(masks.harmless_shift)]
    return np.array(pos), np.array(neg)


def confirm() -> None:
    digest = hashlib.sha256((OUT / "frozen_v8.json").read_bytes()).hexdigest()
    if digest[:16] not in (ROOT / "docs" / "v8_protocol.md").read_text(encoding="utf-8"):
        raise SystemExit("frozen_v8.json fingerprint not recorded in docs/v8_protocol.md; refusing to run")
    frozen = json.loads((OUT / "frozen_v8.json").read_text())
    files = sorted(OUT.glob("signals_v8_confirmation_*of*.pkl"))
    eps = sorted([e for f in files for e in pickle.loads(f.read_bytes())], key=lambda e: (e["dataset"], e["seed"]))
    if {e["dataset"] for e in eps} != set(cfg.CONFIRMATION_DATASETS):
        raise SystemExit("confirmation signals incomplete")
    masks = ev.build_masks(eps, DELAY)
    harm = np.vstack([harm_score(e) for e in eps])
    watch = np.vstack([mon.watch_score(e, DELAY, FRACTION) for e in eps])
    rh, rw = ev.metrics(harm, frozen["harm_threshold"], masks), ev.metrics(watch, frozen["watch_threshold"], masks)
    hp, hn = harm_values(harm, masks)
    wp, wn = harm_values(watch, masks)
    auc = lambda p, n: float(roc_auc_score([1] * len(p) + [0] * len(n), np.r_[p, n]))  # noqa: E731
    auc_h, auc_w = auc(hp, hn), auc(wp, wn)
    diffs = []
    for _ in range(B):
        ip, ineg = RNG.integers(0, len(hp), len(hp)), RNG.integers(0, len(hn), len(hn))
        diffs.append(auc(hp[ip], hn[ineg]) - auc(wp[ip], wn[ineg]))
    ci_d = [float(np.percentile(diffs, q)) for q in (2.5, 97.5)]
    boot_h = [auc(hp[RNG.integers(0, len(hp), len(hp))], hn[RNG.integers(0, len(hn), len(hn))]) for _ in range(B)]
    hyp = {
        "D1": {"value": auc_h, "ci95": [float(np.percentile(boot_h, 2.5)), float(np.percentile(boot_h, 97.5))], "pass": auc_h >= 0.80},
        "D2": {"value": rh["harmless_alert_share"], "false_alerts": rh["false_alerts_per_100_clean"],
               "pass": rh["harmless_alert_share"] <= 0.10 and rh["false_alerts_per_100_clean"] <= 5},
        "D3": {"value": rw["harmless_alert_share"] - rh["harmless_alert_share"], "watch_harmless": rw["harmless_alert_share"],
               "harm_harmless": rh["harmless_alert_share"], "harm_t2": rh["harmful"]["t2_share"], "harm_false_alerts": rh["false_alerts_per_100_clean"],
               "pass": (rw["harmless_alert_share"] - rh["harmless_alert_share"]) >= 0.30 and rh["harmful"]["t2_share"] >= 0.50
               and rh["false_alerts_per_100_clean"] <= 5},
        "D4": {"value": auc_h - auc_w, "ci95": ci_d, "auroc_harm": auc_h, "auroc_watch": auc_w,
               "pass": (auc_h - auc_w) >= 0.10 and ci_d[0] > 0},
    }
    for v in hyp.values():
        v["pass"] = bool(v["pass"])
    overall = "SUCCESS" if hyp["D1"]["pass"] and hyp["D2"]["pass"] else (
        "PARTIAL SUCCESS" if hyp["D3"]["pass"] or hyp["D4"]["pass"] else "FAILURE")
    per = {}
    for d in sorted(cfg.CONFIRMATION_DATASETS):
        idx = [i for i, e in enumerate(eps) if e["dataset"] == d]
        sub = ev.build_masks([eps[i] for i in idx], DELAY)
        p, n = harm_values(harm[idx], sub)
        q, r = harm_values(watch[idx], sub)
        per[d] = {"harmful": int(len(p)), "harmless": int(len(n)),
                  "auroc_harm": auc(p, n) if len(p) and len(n) else None, "auroc_watch": auc(q, r) if len(q) and len(r) else None}
    detail = {name: {"t1_gradual": r["gradual"]["t1_share"], "t2_harmful": r["harmful"]["t2_share"],
                     "false_alerts": r["false_alerts_per_100_clean"], "harmless_alert_share": r["harmless_alert_share"]}
              for name, r in (("HARM", rh), ("WATCH", rw))}
    verdict = {"overall": overall, "hypotheses": hyp, "methods": detail, "per_dataset": per}
    (OUT / "v8_verdict.json").write_text(json.dumps(verdict, indent=2))
    write_markdown(verdict, ROOT / "docs" / "v8_verdict.md")
    for h, v in hyp.items():
        print(h, "PASS" if v["pass"] else "FAIL", round(v["value"], 3))
    print("OVERALL:", overall)


def f(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def write_markdown(v, path):
    h = v["hypotheses"]
    P = lambda k: "PASS" if h[k]["pass"] else "FAIL"  # noqa: E731
    L = ["# v8 Verdict: Reference-Based Harm Estimation", "",
         "Generated by `scripts/run_v8.py confirm` from a one-time run on 6 datasets never used before "
         "(albert, default-of-credit-card-clients, kick, house_16H, 2dplanes, mv). Protocol: [v8_protocol.md](v8_protocol.md). "
         "Earlier verdicts are unchanged.", "", f"## Overall: **{v['overall']}**", "",
         "| ID | Claim | Result | Target | Verdict |", "|---|---|---|---|---|",
         f"| D1 | HARM separates harmful from harmless shifts | AUROC {f(h['D1']['value'])} [{f(h['D1']['ci95'][0])}, {f(h['D1']['ci95'][1])}] | ≥ 0.80 | {P('D1')} |",
         f"| D2 | HARM rarely alerts on harmless shifts | {f(h['D2']['value'])} of harmless shifts, {f(h['D2']['false_alerts'], 1)} false alerts / 100 | ≤ 0.10, FA ≤ 5 | {P('D2')} |",
         f"| D3 | HARM cuts harmless alarms versus WATCH while still warning | {f(h['D3']['watch_harmless'])} → {f(h['D3']['harm_harmless'])} "
         f"(−{f(h['D3']['value'])}); HARM T2 {f(h['D3']['harm_t2'])}, FA {f(h['D3']['harm_false_alerts'], 1)} | ≥ 0.30 drop, T2 ≥ 0.50 | {P('D3')} |",
         f"| D4 | HARM ranks harm better than WATCH | AUROC {f(h['D4']['auroc_harm'])} vs {f(h['D4']['auroc_watch'])}, gain {f(h['D4']['value'])} "
         f"[{f(h['D4']['ci95'][0])}, {f(h['D4']['ci95'][1])}] | ≥ 0.10, CI > 0 | {P('D4')} |",
         "", "## Both tiers on the confirmation datasets", "",
         "| Tier | T1 gradual | T2 harmful | False alerts / 100 clean | Harmless shifts alerting |", "|---|---|---|---|---|"]
    for name, r in v["methods"].items():
        L.append(f"| {name} | {f(r['t1_gradual'])} | {f(r['t2_harmful'])} | {f(r['false_alerts'], 1)} | {f(r['harmless_alert_share'])} |")
    L += ["", "## Per dataset (harmful vs harmless AUROC)", "", "| Dataset | Harmful | Harmless | HARM | WATCH |", "|---|---|---|---|---|"]
    for d, r in v["per_dataset"].items():
        L.append(f"| {d} | {r['harmful']} | {r['harmless']} | {f(r['auroc_harm'])} | {f(r['auroc_watch'])} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["develop", "download", "extract", "confirm"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    {"develop": develop, "download": download, "confirm": confirm}.get(a.stage, lambda: extract(a.shard, a.shards))()
