"""Day 5: score the held-out datasets ONCE with the frozen Day 4 choices.

    python scripts/run_v6_heldout.py

Requires held-out signal shards (scripts/run_v6_extract.py --split held_out ...).
Refuses to run if the frozen thresholds or calibrators differ from the fingerprints
recorded in docs/v6_protocol.md. Nothing is fitted here. Writes
artifacts/v6/day5_held_out.json (results consumed by scripts/make_verdict.py).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v6_config as cfg  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from scripts.run_v6_develop import per_dataset, setting_key  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402

OUT = ROOT / "artifacts" / "v6"
FINGERPRINTS = {
    "frozen_choices.json": "93019702f3f59814e060092963ed535f9dfa26e5439fe00af0443befa278e4ad",
    "frozen_calibrators.pkl": "93f58147a20f2b3c9d114e7e44493ad06ebe1fb60b3870d521e42ecc5c52409e",
}


def check_frozen() -> None:
    for name, expected in FINGERPRINTS.items():
        actual = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"{name} changed since Day 4 ({actual}); refusing to score the held-out data")


def main() -> None:
    check_frozen()
    thresholds = json.loads((OUT / "frozen_choices.json").read_text())["thresholds"]
    calibrators = pickle.loads((OUT / "frozen_calibrators.pkl").read_bytes())
    results, identities = {}, {}
    for model in cfg.MONITORED_MODELS:
        episodes = load_signals("held_out", model)
        if {e["dataset"] for e in episodes} != set(cfg.HELD_OUT_DATASETS):
            raise SystemExit("held-out signals do not cover exactly the held-out datasets")
        for delay in cfg.LABEL_DELAYS:
            for fraction in cfg.LABEL_FRACTIONS:
                key = setting_key(model, delay, fraction)
                calibrator = calibrators[key]
                scores = ev.score_matrix(episodes, delay, fraction, lambda e: calibrator)
                masks = ev.build_masks(episodes, delay)
                entry = {}
                for method in ev.METHODS:
                    thr = thresholds[key][method]
                    r = ev.metrics(scores[method], thr, masks)
                    r["harm_auroc"] = ev.harm_auroc(scores[method], masks)
                    r["per_dataset"] = per_dataset(episodes, scores[method], delay, thr)
                    entry[method] = r
                results[key] = entry
                harmful = masks.harmful_gradual | masks.harmful_abrupt
                identities[key] = {
                    "harmful": [(episodes[i]["dataset"], episodes[i]["seed"]) for i in range(len(episodes)) if harmful[i]],
                    "gradual": [(episodes[i]["dataset"], episodes[i]["seed"]) for i in range(len(episodes)) if masks.harmful_gradual[i]],
                }
                p = entry["primary_system"]
                print(f"{key:36s} system T1 gradual {p['gradual']['t1_share']:.2f} (lead {p['gradual']['t1_median_lead']})"
                      f"  T2 harmful {p['harmful']['t2_share']:.2f}  FA {p['false_alerts_per_100_clean']:.1f}"
                      f"  harmless {p['harmless_alert_share']:.2f}  AUROC {p['harm_auroc']:.2f}"
                      f" | label_trend T2 {entry['label_trend']['harmful']['t2_share']:.2f}"
                      f" | fixed T1 {entry['fixed_dews_score']['gradual']['t1_share']:.2f}", flush=True)
        counts = {}
        for e in episodes:
            counts.setdefault(e["dataset"], {}).setdefault(f"{e['kind']}|{e['category']}", 0)
            counts[e["dataset"]][f"{e['kind']}|{e['category']}"] += 1
        results[f"{model}|episode_counts"] = counts
    (OUT / "day5_held_out.json").write_text(json.dumps({"results": results, "identities": identities}, indent=2))
    print("wrote day5_held_out.json")


if __name__ == "__main__":
    main()
