"""v10 runner (experiment branch): freeze threshold, fetch fresh data, extract, confirm once.

    python scripts/run_v10.py develop
    python scripts/run_v10.py download
    python scripts/run_v10.py extract --shard i --shards k
    python scripts/run_v10.py confirm
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

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import resample_benchmark as rb  # noqa: E402
from dews import v6_config as v6cfg  # noqa: E402
from dews import v10_config as cfg  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from dews.harm_alerting import RULES  # noqa: E402
from dews.harm_estimation import estimated_drop, harm_signals  # noqa: E402
from scripts.run_v10_dev import choose, rates, score_matrix  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v8_extract import load as load_v8  # noqa: E402

OUT = ROOT / "artifacts" / "v10"
B = 2000
RNG = np.random.default_rng(20261110)


def develop() -> None:
    eps = load_signals("development", "logistic_regression")
    S = score_matrix(eps, load_v8("development"), cfg.RULE, cfg.WATCH_GATE)
    M = ev.build_masks(eps, 5)
    thr = choose(S, M)
    r = rates(S, thr, M)
    frozen = {"rule": cfg.RULE, "threshold": thr, "development": {
        "harmless": float(r["harmless"].mean()), "t2": float(r["t2"].mean()), "t1": float(r["t1"].mean()),
        "false_alerts": 100 * r["fa_hits"] / r["fa_n"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_v10.json").write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen, indent=2))
    print("sha256", hashlib.sha256((OUT / "frozen_v10.json").read_bytes()).hexdigest())


def download() -> None:
    for name, did in cfg.CONFIRMATION_DATASETS.items():
        path = OUT / f"data_{name}.pkl"
        if path.exists() or name in cfg.UNAVAILABLE:
            continue
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = d.data, d.target.astype(str)
        X = X.apply(lambda c: c.astype("category").cat.codes if str(c.dtype) in ("category", "object") else c).astype(float)
        X = X.fillna(X.median()).loc[:, X.nunique() > 1].iloc[:, : cfg.MAX_FEATURES].reset_index(drop=True)
        yy = (y == y.value_counts().index[0]).astype(int).to_numpy()
        pd.to_pickle((X, yy), path)
        print(f"{name}: {X.shape}, positive share {yy.mean():.3f}, label changes {int(np.sum(np.diff(yy) != 0))}, "
              f"valid block starts {len(rb._valid_block_starts(yy))}", flush=True)


def plan():
    out = []
    for di, d in enumerate(cfg.CONFIRMATION_DATASETS):
        if d in cfg.UNAVAILABLE:
            continue
        index = 0
        for kind in rb.EPISODE_KINDS:
            for _ in range(v6cfg.EPISODES_PER_DATASET[kind]):
                out.append((d, kind, cfg.CONFIRMATION_SEED_BASE + 100 * di + index))
                index += 1
    return out


def extract(shard: int, shards: int) -> None:
    out = []
    for d, kind, seed in plan()[shard::shards]:
        X, y = pd.read_pickle(OUT / f"data_{d}.pkl")
        ep = rb.build_episode(X, y, dataset=d, kind=kind, seed=seed, model_kind="logistic_regression")
        res = rb.classify_outcome(ep)
        rec = {"dataset": d, "seed": seed, "kind": kind, "onset": ep.onset, "category": res["category"],
               "event_onset": res["event_onset"], "accuracy": res["accuracy"]}
        rec.update(harm_signals(ep.reference, ep.reference_labels, ep.model, ep.batches))
        out.append(rec)
    (OUT / f"signals_{shard}of{shards}.pkl").write_bytes(pickle.dumps(out))
    print(f"shard {shard}: {len(out)} episodes", flush=True)


def scores(eps, rule):
    fn, kw = RULES[rule]
    return np.vstack([fn(estimated_drop(np.asarray(e["iw_accuracy"], float), 8), est=np.asarray(e["iw_accuracy"], float), **kw)
                      for e in eps])


def boot(x):
    x = np.asarray(x, float)
    m = [RNG.choice(x, len(x)).mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def confirm() -> None:
    digest = hashlib.sha256((OUT / "frozen_v10.json").read_bytes()).hexdigest()
    if digest[:16] not in (ROOT / "docs" / "v10_protocol.md").read_text(encoding="utf-8"):
        raise SystemExit("frozen_v10.json fingerprint not recorded in docs/v10_protocol.md; refusing to run")
    frozen = json.loads((OUT / "frozen_v10.json").read_text())
    v8_thr = json.loads((ROOT / "artifacts" / "v8" / "frozen_v8.json").read_text())["harm_threshold"]
    eps = sorted([e for f in sorted(OUT.glob("signals_*of*.pkl")) for e in pickle.loads(f.read_bytes())],
                 key=lambda e: (e["dataset"], e["seed"]))
    M = ev.build_masks(eps, 5)
    res = {}
    for label, rule, thr in (("v10 (frozen)", cfg.RULE, frozen["threshold"]), ("v8 rule, for comparison", "raw (v8 rule)", v8_thr)):
        r = rates(scores(eps, rule), thr, M)
        res[label] = {"harmless": float(r["harmless"].mean()), "harmless_ci95": boot(r["harmless"]),
                      "t2": float(r["t2"].mean()), "t2_ci95": boot(r["t2"]), "t1": float(r["t1"].mean()),
                      "false_alerts": 100 * r["fa_hits"] / r["fa_n"], "harmful": int(len(r["t2"])),
                      "harmless_episodes": int(len(r["harmless"]))}
    v = res["v10 (frozen)"]
    s = cfg.SUCCESS
    passed = v["harmless"] <= s["harmless_max"] and v["t2"] >= s["t2_min"] and v["false_alerts"] <= s["false_alerts_max"]
    counts = {}
    for e in eps:
        counts.setdefault(e["dataset"], {}).setdefault(f"{e['kind']}|{e['category']}", 0)
        counts[e["dataset"]][f"{e['kind']}|{e['category']}"] += 1
    out = {"passed": bool(passed), "success_rule": s, "results": res, "episode_counts": counts}
    (OUT / "v10_verdict.json").write_text(json.dumps(out, indent=2))
    for k, r in res.items():
        print(f"{k:26s} harmless {r['harmless']:.2f} {r['harmless_ci95']}  T2 {r['t2']:.2f} {r['t2_ci95']}  T1 {r['t1']:.2f}  "
              f"FA {r['false_alerts']:.1f}  (harmful {r['harmful']}, harmless {r['harmless_episodes']})")
    print("CONFIRMATION:", "SUCCESS" if passed else "FAILURE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["develop", "download", "extract", "confirm"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    {"develop": develop, "download": download, "confirm": confirm}.get(a.stage, lambda: extract(a.shard, a.shards))()
