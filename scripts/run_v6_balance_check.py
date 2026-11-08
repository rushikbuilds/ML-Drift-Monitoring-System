"""Day 2 gate: do development shift episodes produce both harmful and harmless outcomes?

    python scripts/run_v6_balance_check.py [--lambda-max 1.5]

Gate (docs/v6_protocol.md section 8): on development datasets, with the primary model,
>= 25% of shift episodes must be harmful and >= 25% harmless. Uses only outcome labels
(no monitor signals), and never touches held-out datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import resample_benchmark as rb  # noqa: E402
from dews import v6_config as cfg  # noqa: E402

OUT = ROOT / "artifacts" / "v6"


def main(lambda_max: float, models: tuple[str, ...]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"lambda_max": lambda_max, "models": {}}
    for model_kind in models:
        per_dataset, totals = {}, Counter()
        for dataset in cfg.DEVELOPMENT_DATASETS:
            X, y = rb.load_dataset(dataset)
            counts, rows = Counter(), []
            for _, kind, seed in rb.episode_plan(dataset):
                ep = rb.build_episode(X, y, dataset=dataset, kind=kind, seed=seed,
                                      model_kind=model_kind, lambda_max=lambda_max)
                res = rb.classify_outcome(ep)
                counts[(kind, res["category"])] += 1
                rows.append({"kind": kind, "seed": seed, "category": res["category"],
                             "event_onset": res["event_onset"], "onset": ep.onset,
                             "from_important_half": ep.meta.get("from_important_half"),
                             "baseline_accuracy": res["baseline_accuracy"],
                             "min_accuracy_after_onset": res["min_accuracy_after_onset"]})
            totals.update(counts)
            per_dataset[dataset] = {"counts": {f"{k}|{c}": v for (k, c), v in sorted(counts.items())},
                                    "episodes": rows}
            shift = [r for r in rows if r["kind"] in rb.SHIFT_KINDS]
            print(f"{model_kind:19s} {dataset:15s} shift: harmful {sum(r['category'] == 'harmful' for r in shift):2d}"
                  f"  harmless {sum(r['category'] == 'harmless' for r in shift):2d}"
                  f"  spurious {sum(r['category'] == 'spurious' for r in shift):2d}"
                  f" | clean spurious {counts[('clean', 'spurious')]}"
                  f" | concept events {counts[('concept_change', 'concept_event')]}/{cfg.EPISODES_PER_DATASET['concept_change']}")
        shift_total = sum(v for (k, _), v in totals.items() if k in rb.SHIFT_KINDS)
        harmful = sum(v for (k, c), v in totals.items() if k in rb.SHIFT_KINDS and c == "harmful") / shift_total
        harmless = sum(v for (k, c), v in totals.items() if k in rb.SHIFT_KINDS and c == "harmless") / shift_total
        report["models"][model_kind] = {"per_dataset": per_dataset, "harmful_share": harmful,
                                        "harmless_share": harmless,
                                        "gate_pass": harmful >= 0.25 and harmless >= 0.25}
        print(f"==> {model_kind}: harmful {harmful:.2f}  harmless {harmless:.2f}  "
              f"gate {'PASS' if harmful >= 0.25 and harmless >= 0.25 else 'FAIL'}\n")
    (OUT / f"day2_balance_lambda{lambda_max}.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lambda-max", type=float, default=cfg.LAMBDA_MAX)
    p.add_argument("--models", nargs="+", default=list(cfg.MONITORED_MODELS))
    a = p.parse_args()
    main(a.lambda_max, tuple(a.models))
