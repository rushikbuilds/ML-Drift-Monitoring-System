"""v10 development evaluation of harmless-alarm remedies (leave-one-dataset-out).

    python scripts/run_v10_dev.py

For each rule (dews/harm_alerting.py), with and without gating by the v7 drift watch,
the threshold is chosen on 7 development datasets to MINIMIZE the harmless-alarm share
subject to T2 (warned before labels confirm) >= 0.60 and <= 5 false alerts per 100 clean
batches, then applied to the held-out 8th dataset. Pooled held-out-fold results decide.
Gate fixed before running: best rule must reach harmless <= 0.15 with T2 >= 0.60, FA <= 5.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v6_eval as ev  # noqa: E402
from dews import v7_monitor as mon  # noqa: E402
from dews.harm_alerting import RULES  # noqa: E402
from dews.harm_estimation import estimated_drop  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v8_extract import load as load_v8  # noqa: E402

T2_MIN, FA_MAX, GATE_HARMLESS = 0.60, 5.0, 0.15
WATCH_THR = json.loads((ROOT / "artifacts" / "v7" / "frozen_v7.json").read_text())["thresholds"]["watch"]


def score_matrix(eps, harm, rule, gated):
    fn, kw = RULES[rule]
    rows = []
    for e in eps:
        est = np.asarray(harm[(e["dataset"], e["seed"])]["iw_accuracy"], float)
        s = fn(estimated_drop(est, 8), est=est, **kw)
        if gated:
            w = mon.watch_score(e, 5, 1.0)
            s = np.where(np.isfinite(w) & (w >= WATCH_THR), s, -np.inf)
        rows.append(s)
    return np.vstack(rows)


def rates(S, thr, M):
    A = np.where(np.isfinite(S), S >= thr, False)
    harmful = M.harmful_gradual | M.harmful_abrupt
    return {"t1": (A & M.t1)[M.harmful_gradual].any(axis=1), "t2": (A & M.t2)[harmful].any(axis=1),
            "harmless": (A & M.after_onset)[M.harmless_shift].any(axis=1),
            "fa_hits": int((A & M.clean).sum()), "fa_n": int(M.clean.sum())}


def choose(S, M):
    finite = S[np.isfinite(S)]
    best = None
    for thr in np.unique(np.quantile(finite, np.linspace(0.2, 0.999, 200))):
        r = rates(S, thr, M)
        fa = 100 * r["fa_hits"] / max(r["fa_n"], 1)
        if r["t2"].mean() < T2_MIN or fa > FA_MAX:
            continue
        key = (r["harmless"].mean(), -r["t2"].mean())
        if best is None or key < best[0]:
            best = (key, float(thr))
    return None if best is None else best[1]


def main():
    eps = load_signals("development", "logistic_regression")
    harm = load_v8("development")
    datasets = sorted({e["dataset"] for e in eps})
    out = {}
    for rule in RULES:
        for gated in (False, True):
            name = rule + (" + watch gate" if gated else "")
            S = score_matrix(eps, harm, rule, gated)
            agg = {"t1": [], "t2": [], "harmless": [], "fa_hits": 0, "fa_n": 0}
            feasible = True
            for d in datasets:
                tr = [i for i, e in enumerate(eps) if e["dataset"] != d]
                te = [i for i, e in enumerate(eps) if e["dataset"] == d]
                thr = choose(S[tr], ev.build_masks([eps[i] for i in tr], 5))
                if thr is None:
                    feasible = False
                    break
                r = rates(S[te], thr, ev.build_masks([eps[i] for i in te], 5))
                for k in ("t1", "t2", "harmless"):
                    agg[k] += list(r[k])
                agg["fa_hits"] += r["fa_hits"]; agg["fa_n"] += r["fa_n"]
            if not feasible:
                out[name] = None
                print(f"{name:32s} no threshold meets T2 >= {T2_MIN} within the budget")
                continue
            res = {"t1": float(np.mean(agg["t1"])), "t2": float(np.mean(agg["t2"])),
                   "harmless": float(np.mean(agg["harmless"])), "fa": 100 * agg["fa_hits"] / agg["fa_n"]}
            out[name] = res
            print(f"{name:32s} harmless {res['harmless']:.2f}  T2 {res['t2']:.2f}  T1 {res['t1']:.2f}  FA {res['fa']:.1f}")
    ok = {k: v for k, v in out.items() if v and v["t2"] >= T2_MIN and v["fa"] <= FA_MAX}
    best = min(ok, key=lambda k: (ok[k]["harmless"], -ok[k]["t2"])) if ok else None
    passed = bool(best and ok[best]["harmless"] <= GATE_HARMLESS)
    print(f"\nBest: {best} -> {ok.get(best)}; gate (harmless <= {GATE_HARMLESS}, T2 >= {T2_MIN}, FA <= {FA_MAX}): "
          f"{'PASS' if passed else 'FAIL'}")
    (ROOT / "artifacts" / "v10").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "v10" / "dev_gate.json").write_text(json.dumps({"results": out, "best": best, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
