"""Extract v9 candidate harm signals by regenerating episodes from their seeds (experiment branch).

    python scripts/run_v9_extract.py --split development --shard 0 --shards 6
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import resample_benchmark as rb  # noqa: E402
from dews.harm_estimation_v9 import harm_signals_v9  # noqa: E402
from scripts.run_v8_extract import jobs as v8_jobs  # noqa: E402

OUT = ROOT / "artifacts" / "v9"


def jobs(split: str):
    if split == "v9_confirmation":
        from dews import v9_config as cfg
        from dews import v6_config as v6cfg
        out = []
        for di, d in enumerate(cfg.CONFIRMATION_DATASETS):
            if d in cfg.UNAVAILABLE:
                continue
            index = 0
            for kind in rb.EPISODE_KINDS:
                for _ in range(v6cfg.EPISODES_PER_DATASET[kind]):
                    out.append((d, kind, cfg.CONFIRMATION_SEED_BASE + 100 * di + index, OUT / f"data_{d}.pkl"))
                    index += 1
        return out
    return v8_jobs(split)


def main(split: str, shard: int, shards: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for dataset, kind, seed, path in jobs(split)[shard::shards]:
        X, y = rb.load_dataset(dataset) if path is None else pd.read_pickle(path)
        ep = rb.build_episode(X, y, dataset=dataset, kind=kind, seed=seed, model_kind="logistic_regression")
        rec = {"dataset": dataset, "seed": seed, "kind": kind}
        rec.update(harm_signals_v9(ep.reference, ep.reference_labels, ep.model, ep.batches))
        out.append(rec)
    (OUT / f"harm9_{split}_{shard}of{shards}.pkl").write_bytes(pickle.dumps(out))
    print(f"{split} shard {shard}: {len(out)} episodes", flush=True)


def load(split: str) -> dict:
    return {(r["dataset"], r["seed"]): r for f in sorted(OUT.glob(f"harm9_{split}_*of*.pkl")) for r in pickle.loads(f.read_bytes())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["development", "v9_confirmation"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    main(a.split, a.shard, a.shards)
