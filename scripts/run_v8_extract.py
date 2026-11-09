"""Extract v8 harm-estimation signals by regenerating episodes deterministically from their seeds.

    python scripts/run_v8_extract.py --split development --shard 0 --shards 6
    python scripts/run_v8_extract.py --split v8_confirmation --shard 0 --shards 6

Splits: development (v6 dev datasets), v6_held_out, v7_confirmation, v8_confirmation.
Writes artifacts/v8/harm_<split>_<shard>of<shards>.pkl with one record per episode,
keyed by (dataset, seed) so it can be joined to the existing v6/v7 signal files.
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
from dews import v6_config as v6cfg  # noqa: E402
from dews.harm_estimation import harm_signals  # noqa: E402

OUT = ROOT / "artifacts" / "v8"
MODEL = "logistic_regression"


def jobs(split: str):
    if split in ("development", "v6_held_out"):
        names = v6cfg.DEVELOPMENT_DATASETS if split == "development" else v6cfg.HELD_OUT_DATASETS
        return [(d, kind, seed, None) for d in names for _, kind, seed in rb.episode_plan(d)]
    if split == "v7_confirmation":
        from scripts.run_v7 import plan
        return [(d, kind, seed, ROOT / "artifacts" / "v7" / f"data_{d}.pkl") for d, kind, seed in plan()]
    from dews import v8_config as cfg
    out = []
    for di, d in enumerate(cfg.CONFIRMATION_DATASETS):
        if d in cfg.UNAVAILABLE:
            continue  # seed numbering of the remaining datasets stays as frozen
        index = 0
        for kind in rb.EPISODE_KINDS:
            for _ in range(v6cfg.EPISODES_PER_DATASET[kind]):
                out.append((d, kind, cfg.CONFIRMATION_SEED_BASE + 100 * di + index, OUT / f"data_{d}.pkl"))
                index += 1
    return out


def main(split: str, shard: int, shards: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for dataset, kind, seed, path in jobs(split)[shard::shards]:
        X, y = rb.load_dataset(dataset) if path is None else pd.read_pickle(path)
        ep = rb.build_episode(X, y, dataset=dataset, kind=kind, seed=seed, model_kind=MODEL)
        rec = {"dataset": dataset, "seed": seed, "kind": kind}
        rec.update(harm_signals(ep.reference, ep.reference_labels, ep.model, ep.batches))
        out.append(rec)
    (OUT / f"harm_{split}_{shard}of{shards}.pkl").write_bytes(pickle.dumps(out))
    print(f"{split} shard {shard}: {len(out)} episodes", flush=True)


def load(split: str) -> dict:
    files = sorted(OUT.glob(f"harm_{split}_*of*.pkl"))
    return {(r["dataset"], r["seed"]): r for f in files for r in pickle.loads(f.read_bytes())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["development", "v6_held_out", "v7_confirmation", "v8_confirmation"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    main(a.split, a.shard, a.shards)
