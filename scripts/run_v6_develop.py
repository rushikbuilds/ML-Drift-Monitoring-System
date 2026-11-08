"""Day 4: development evaluation, threshold selection and freezing of all choices.

    python scripts/run_v6_develop.py

For every (monitored model, label delay, labeled fraction):
  * the risk calibrator is scored leave-one-dataset-out on the 8 development datasets;
  * each method's alert threshold is selected on those development scores (protocol section 5);
  * a final calibrator is trained on all development datasets for the held-out run.
Writes artifacts/v6/day4_development.json, frozen_choices.json and frozen_calibrators.pkl.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v6_config as cfg  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402

OUT = ROOT / "artifacts" / "v6"


def setting_key(model: str, delay: int, fraction: float) -> str:
    return f"{model}|L{delay}|labels{int(fraction * 100)}"


def per_dataset(episodes, scores, delay, threshold):
    out = {}
    for d in sorted({e["dataset"] for e in episodes}):
        idx = [i for i, e in enumerate(episodes) if e["dataset"] == d]
        sub = [episodes[i] for i in idx]
        r = ev.metrics(scores[idx], threshold, ev.build_masks(sub, delay))
        out[d] = {"t1_share_harmful": r["harmful"]["t1_share"], "t2_share_harmful": r["harmful"]["t2_share"],
                  "harmful_episodes": r["harmful"]["episodes"]}
    return out


def main() -> None:
    results, frozen, calibrators = {}, {"frozen_on": "Day 4", "thresholds": {}}, {}
    for model in cfg.MONITORED_MODELS:
        episodes = load_signals("development", model)
        datasets = sorted({e["dataset"] for e in episodes})
        for delay in cfg.LABEL_DELAYS:
            for fraction in cfg.LABEL_FRACTIONS:
                key = setting_key(model, delay, fraction)
                folds = {d: ev.RiskCalibrator(delay, fraction).fit([e for e in episodes if e["dataset"] != d])
                         for d in datasets}
                scores = ev.score_matrix(episodes, delay, fraction, lambda e: folds[e["dataset"]])
                masks = ev.build_masks(episodes, delay)
                entry = {}
                for method in ev.METHODS:
                    thr = ev.select_threshold(scores[method], masks)
                    r = ev.metrics(scores[method], thr, masks)
                    r["harm_auroc"] = ev.harm_auroc(scores[method], masks)
                    r["per_dataset"] = per_dataset(episodes, scores[method], delay, thr)
                    entry[method] = r
                    frozen["thresholds"].setdefault(key, {})[method] = thr
                results[key] = entry
                calibrators[key] = ev.RiskCalibrator(delay, fraction).fit(episodes)
                p = entry["primary_system"]
                print(f"{key:36s} system T1 gradual {p['gradual']['t1_share']:.2f} (lead {p['gradual']['t1_median_lead']})"
                      f"  T2 harmful {p['harmful']['t2_share']:.2f}  FA {p['false_alerts_per_100_clean']:.1f}"
                      f"  harmless {p['harmless_alert_share']:.2f}  AUROC {p['harm_auroc']:.2f}"
                      f" | label_trend T2 {entry['label_trend']['harmful']['t2_share']:.2f}"
                      f" | fixed T1 {entry['fixed_dews_score']['gradual']['t1_share']:.2f}", flush=True)
    primary = setting_key(cfg.PRIMARY["model"], cfg.PRIMARY["label_delay"], cfg.PRIMARY["label_fraction"])
    t1 = results[primary]["primary_system"]["gradual"]["t1_share"]
    gate = {"setting": primary, "development_t1_share_gradual": t1, "threshold": 0.40,
            "fallback_triggered": bool(t1 < 0.40)}
    print(f"\nDay 4 gate ({primary}): development T1 share on harmful gradual shifts = {t1:.2f} -> "
          f"{'FALLBACK' if gate['fallback_triggered'] else 'PASS'}")
    (OUT / "day4_development.json").write_text(json.dumps({"gate": gate, "results": results}, indent=2))
    (OUT / "frozen_choices.json").write_text(json.dumps({**frozen, "gate": gate}, indent=2))
    (OUT / "frozen_calibrators.pkl").write_bytes(pickle.dumps(calibrators))


if __name__ == "__main__":
    main()
