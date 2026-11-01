"""Real-data reality check on the chronological Electricity (elec2) stream.

Overlapping chronological windows (reference -> 25 batches). Natural drift only;
nothing is injected. Reports (a) how often the event rule fires, and (b)
threshold-free AUROC of each score for the state target, for a calibrator
transferred from synthetic training data and one trained in-domain on earlier
windows only. Exploratory: windows overlap, so episodes are not independent.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402

OUT = ROOT / "artifacts" / "v2"
REF, BATCH, NB, STRIDE = 3000, 300, 25, 2000
COLS = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "period", "day"]


def load():
    cache = OUT / "elec.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frame = fetch_openml("electricity", version=1, as_frame=True, parser="auto").frame
    frame["day"] = frame["day"].astype(float)
    frame["y"] = (frame["class"] == "UP").astype(int)
    frame = frame[COLS + ["y"]].reset_index(drop=True)
    frame.to_pickle(cache)
    return frame


def window(frame, start):
    ref = frame.iloc[start:start + REF]
    stream = frame.iloc[start + REF:start + REF + NB * BATCH]
    batches = [stream.iloc[i * BATCH:(i + 1) * BATCH][COLS].reset_index(drop=True) for i in range(NB)]
    labels = [stream.iloc[i * BATCH:(i + 1) * BATCH]["y"].to_numpy() for i in range(NB)]
    return v2.Episode("electricity", start, ref[COLS].reset_index(drop=True), ref["y"].to_numpy(), batches, labels, None)


def job(start):
    warnings.filterwarnings("ignore")
    frame = load()
    return v2.extract_signals(window(frame, start))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    starts = list(range(0, len(frame) - REF - NB * BATCH + 1, STRIDE))
    with ProcessPoolExecutor(12) as ex:
        sigs = list(ex.map(job, starts))
    for s in sigs:
        o = v2.outcome(s["accuracy"]); s["event_onset"] = o["event_onset"]; s["targets"] = v2.state_targets(s["accuracy"])
    span = REF + NB * BATCH
    train = [s for s in sigs if s["seed"] + span <= 18500]
    test = [s for s in sigs if s["seed"] >= 18500]
    delay = 2
    res = {"windows": len(sigs), "train_windows": len(train), "test_windows": len(test),
           "test_windows_with_event": int(sum(s["event_onset"] is not None for s in test)),
           "all_windows_with_event": int(sum(s["event_onset"] is not None for s in sigs)),
           "baseline_accuracy_mean": float(np.mean([v2.outcome(s["accuracy"])["baseline"] for s in sigs])),
           "delay": delay, "auroc": {}}
    # synthetic-trained transfer calibrators (cached signals from the main run)
    syn = pickle.loads((OUT / "signals.pkl").read_bytes())["train"]
    pools = {"syn_sig": v2.training_pool(syn, delay, use_labels=False), "syn_full": v2.training_pool(syn, delay, use_labels=True),
             "ind_sig": v2.training_pool(train, delay, use_labels=False), "ind_full": v2.training_pool(train, delay, use_labels=True)}
    names = {}
    for tag, (pool, ul) in {"transfer_signals": ("syn_sig", False), "transfer_full": ("syn_full", True),
                            "indomain_signals": ("ind_sig", False), "indomain_full": ("ind_full", True)}.items():
        names[tag] = [v2.calibrator_scores(s, delay, pools[pool], use_labels=ul, online=False) for s in test]
    for key in ("stat_only", "conf_only", "fixed_score", "slope_forecast", "delayed_label_only"):
        names[key] = [v2.baseline_scores(s, delay)[key] for s in test]
    ys = np.concatenate([s["targets"][v2.WARMUP:] for s in test])
    ok = np.isfinite(ys)
    res["state_target_prevalence"] = float(np.mean(ys[ok]))
    for tag, arrs in names.items():
        sc = np.concatenate([a[v2.WARMUP:] for a in arrs])
        res["auroc"][tag] = float(roc_auc_score(ys[ok], sc[ok])) if len(set(ys[ok])) == 2 else None
    (OUT / "electricity_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
