"""Ablations and failure analysis (post-hoc, explanatory; no verdict changes).

    python scripts/run_ablations.py

Writes artifacts/ablations/ablations.json and prints the tables used in docs/ablations.md.
Thresholds for every variant are re-selected on development data with the rule of the
experiment being ablated, then applied unchanged to the evaluation data.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from dews import v7_config as v7cfg  # noqa: E402
from dews import v7_monitor as mon  # noqa: E402
from dews.harm_estimation import estimated_drop  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v8 import separation_threshold  # noqa: E402
from scripts.run_v8_extract import load as load_harm  # noqa: E402

OUT = ROOT / "artifacts" / "ablations"
DELAY, FRAC = 5, 1.0
BURN = v7cfg.BURN_IN_BATCHES


def summarize(scores, thr, masks):
    r = ev.metrics(scores, thr, masks)
    return {"t1_gradual": r["gradual"]["t1_share"], "t2_harmful": r["harmful"]["t2_share"],
            "false_alerts": r["false_alerts_per_100_clean"], "harmless_alert_share": r["harmless_alert_share"],
            "harm_auroc": ev.harm_auroc(scores, masks)}


def v7_signals(split):
    if split == "development":
        return load_signals("development", "logistic_regression")
    files = sorted((ROOT / "artifacts" / "v7").glob("signals_confirmation_*of*.pkl"))
    return sorted([e for f in files for e in pickle.loads(f.read_bytes())], key=lambda e: (e["dataset"], e["seed"]))


def v8_signals():
    files = sorted((ROOT / "artifacts" / "v8").glob("signals_v8_confirmation_*of*.pkl"))
    return sorted([e for f in files for e in pickle.loads(f.read_bytes())], key=lambda e: (e["dataset"], e["seed"]))


# ----------------------------------------------------------------------------- A. threshold transfer
def threshold_transfer():
    dev = json.loads((ROOT / "artifacts" / "v6" / "day4_development.json").read_text())["results"]
    held = json.loads((ROOT / "artifacts" / "v6" / "day5_held_out.json").read_text())["results"]
    key = "logistic_regression|L5|labels100"
    rows = {m: {"development": dev[key][m]["false_alerts_per_100_clean"], "held_out": held[key][m]["false_alerts_per_100_clean"]}
            for m in ev.METHODS}
    v7 = json.loads((ROOT / "artifacts" / "v7" / "v7_verdict.json").read_text())
    v7dev = json.loads((ROOT / "artifacts" / "v7" / "frozen_v7.json").read_text())["development"]
    v8 = json.loads((ROOT / "artifacts" / "v8" / "v8_verdict.json").read_text())
    v8dev = json.loads((ROOT / "artifacts" / "v8" / "frozen_v8.json").read_text())["development"]
    rows["v7 WATCH (normalized)"] = {"development": v7dev["watch"]["false_alerts_per_100_clean"],
                                     "held_out": v7["confirmation"]["watch"]["false_alerts"]}
    rows["v8 HARM (accuracy units)"] = {"development": v8dev["false_alerts"], "held_out": v8["methods"]["HARM"]["false_alerts"]}
    return rows


# ----------------------------------------------------------------------------- B. WATCH signal ablation
def watch_from(ep, signals):
    r = mon.raw_signals(ep, DELAY, FRAC)
    z = np.column_stack([mon.robust_z(r[k]) for k in signals]).max(axis=1)
    z[:BURN] = np.nan
    return z


def watch_ablation():
    dev, conf = v7_signals("development"), v7_signals("confirmation")
    mdev, mconf = ev.build_masks(dev, DELAY), ev.build_masks(conf, DELAY)
    variants = {"all 7 signals": v7cfg.WATCH_SIGNALS}
    variants.update({f"without {s}": tuple(x for x in v7cfg.WATCH_SIGNALS if x != s) for s in v7cfg.WATCH_SIGNALS})
    variants.update({f"only {s}": (s,) for s in v7cfg.WATCH_SIGNALS})
    out = {}
    for name, sig in variants.items():
        sdev = np.vstack([watch_from(e, sig) for e in dev])
        scf = np.vstack([watch_from(e, sig) for e in conf])
        out[name] = summarize(scf, ev.select_threshold(sdev, mdev), mconf)
    return out


# ----------------------------------------------------------------------------- C. HARM estimator variants
def harm_ablation():
    dev = load_signals("development", "logistic_regression")
    hdev = load_harm("development")
    conf = v8_signals()
    mdev, mconf = ev.build_masks(dev, DELAY), ev.build_masks(conf, DELAY)
    variants = {
        "importance-weighted (frozen HARM)": lambda r: estimated_drop(r["iw_accuracy"], BURN),
        "nearest-neighbour": lambda r: estimated_drop(r["knn_accuracy"], BURN),
        "average of both": lambda r: (estimated_drop(r["iw_accuracy"], BURN) + estimated_drop(r["knn_accuracy"], BURN)) / 2,
    }
    out = {}
    for name, fn in variants.items():
        sdev = np.vstack([fn(hdev[(e["dataset"], e["seed"])]) for e in dev])
        scf = np.vstack([fn(e) for e in conf])
        out[name] = {"separation rule": summarize(scf, separation_threshold(sdev, mdev), mconf),
                     "early-warning rule": summarize(scf, ev.select_threshold(sdev, mdev), mconf)}
    return out


# ----------------------------------------------------------------------------- D. v6 feature groups
GROUPS = {"drift and confidence signals": slice(0, 24), "matured-label features": slice(24, 26),
          "auxiliary-model disagreement": slice(26, 34), "relevance-weighted shift": slice(34, 37)}


class GroupCalibrator(ev.RiskCalibrator):
    def __init__(self, keep):
        super().__init__(DELAY, FRAC)
        self.keep = keep

    def _x(self, ep):
        return ev.system_features(ep, DELAY, FRAC)[:, self.keep]

    def fit(self, episodes):
        xs, ys = [], []
        for ep in episodes:
            X, y = self._x(ep), ev.state_target(ep)
            k = np.isfinite(y) & (np.arange(len(y)) >= 6)
            xs.append(X[k]); ys.append(y[k])
        self.model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def score(self, ep):
        return self.model.predict_proba(self._x(ep))[:, 1]


def v6_feature_ablation():
    dev = load_signals("development", "logistic_regression")
    held = load_signals("held_out", "logistic_regression")
    mdev, mheld = ev.build_masks(dev, DELAY), ev.build_masks(held, DELAY)
    all_cols = np.arange(37)
    variants = {"all groups (v6 system)": all_cols}
    for g, sl in GROUPS.items():
        variants[f"without {g}"] = np.setdiff1d(all_cols, np.arange(37)[sl])
        variants[f"only {g}"] = np.arange(37)[sl]
    datasets = sorted({e["dataset"] for e in dev})
    out = {}
    for name, keep in variants.items():
        folds = {d: GroupCalibrator(keep).fit([e for e in dev if e["dataset"] != d]) for d in datasets}
        sdev = np.vstack([folds[e["dataset"]].score(e) for e in dev]); sdev[:, :6] = np.nan
        final = GroupCalibrator(keep).fit(dev)
        sh = np.vstack([final.score(e) for e in held]); sh[:, :6] = np.nan
        out[name] = summarize(sh, ev.select_threshold(sdev, mdev), mheld)
    return out


# ----------------------------------------------------------------------------- E. why shifts are harmless
def harm_rates():
    groups = {"v6 development": load_signals("development", "logistic_regression"),
              "v6 held-out": load_signals("held_out", "logistic_regression"),
              "v7 confirmation": v7_signals("confirmation"), "v8 confirmation": v8_signals()}
    out = {}
    for name, eps in groups.items():
        shifts = [e for e in eps if e["kind"] in ("gradual_shift", "abrupt_shift") and e["category"] != "spurious"]
        imp = [e for e in shifts if e["meta"].get("from_important_half")]
        unimp = [e for e in shifts if not e["meta"].get("from_important_half")]
        rate = lambda g: float(np.mean([e["category"] == "harmful" for e in g])) if g else None  # noqa: E731
        base = [float(np.mean(e["accuracy"][:5])) for e in shifts]
        harmful = [e["category"] == "harmful" for e in shifts]
        out[name] = {"shift_episodes": len(shifts), "harmful_share": rate(shifts),
                     "harmful_share_important_feature": rate(imp), "harmful_share_unimportant_feature": rate(unimp),
                     "auroc_baseline_accuracy_predicts_harm": float(roc_auc_score(harmful, base)) if 0 < sum(harmful) < len(harmful) else None,
                     "gradual_harmful_share": rate([e for e in shifts if e["kind"] == "gradual_shift"]),
                     "abrupt_harmful_share": rate([e for e in shifts if e["kind"] == "abrupt_shift"])}
    return out


# ----------------------------------------------------------------------------- F. natural streams
def natural_diagnosis():
    windows = [w for f in sorted((ROOT / "artifacts" / "v6").glob("natural_signals_*of*.pkl")) for w in pickle.loads(f.read_bytes())]
    thr = json.loads((ROOT / "artifacts" / "v7" / "frozen_v7.json").read_text())["thresholds"]["watch"]
    events, early, alert_rate, burn_cv = [], 0, [], []
    for w in windows:
        acc = np.asarray(w["accuracy"])
        ev_on = v2.outcome(acc)["event_onset"]
        z = mon.watch_score(w, DELAY, FRAC)
        alert_rate.append(float(np.nanmean(z >= thr)))
        burn_cv.append(float(np.std(acc[:BURN]) / max(np.mean(acc[:BURN]), 1e-9)))
        if ev_on is not None:
            events.append(ev_on)
            early += ev_on < BURN + 5
    return {"windows": len(windows), "windows_with_event": len(events),
            "event_onset_quartiles": [float(np.percentile(events, q)) for q in (25, 50, 75)],
            "events_within_5_batches_after_burn_in": int(early),
            "share_of_post_burn_in_batches_alerting": float(np.mean(alert_rate)),
            "burn_in_accuracy_coefficient_of_variation": float(np.mean(burn_cv))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, fn in (("A_threshold_transfer", threshold_transfer), ("B_watch_signals", watch_ablation),
                     ("C_harm_estimators", harm_ablation), ("D_v6_feature_groups", v6_feature_ablation),
                     ("E_harm_rates", harm_rates), ("F_natural_streams", natural_diagnosis)):
        result[name] = fn()
        print(f"\n== {name}")
        print(json.dumps(result[name], indent=1)[:4000])
    (OUT / "ablations.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
