"""Extract v6 signals for one split and monitored model, optionally sharded.

    python scripts/run_v6_extract.py --split development --model logistic_regression --shard 0 --shards 4

Shards run as separate processes (the Windows process pool is unreliable here) and
write artifacts/v6/signals_<split>_<model>_<shard>of<shards>.pkl. The held-out split
must only be extracted after development choices are committed (protocol section 9).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import resample_benchmark as rb  # noqa: E402
from dews import v6_config as cfg  # noqa: E402
from dews.v6_signals import extract_episode  # noqa: E402

OUT = ROOT / "artifacts" / "v6"


def jobs(split: str) -> list[tuple[str, str, int]]:
    datasets = cfg.DEVELOPMENT_DATASETS if split == "development" else cfg.HELD_OUT_DATASETS
    return [(d, kind, seed) for d in datasets for _, kind, seed in rb.episode_plan(d)]


def main(split: str, model: str, shard: int, shards: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mine = jobs(split)[shard::shards]
    out, started = [], time.time()
    for i, (dataset, kind, seed) in enumerate(mine):
        X, y = rb.load_dataset(dataset)
        ep = rb.build_episode(X, y, dataset=dataset, kind=kind, seed=seed, model_kind=model)
        out.append(extract_episode(ep))
        if i % 20 == 0:
            print(f"[{split} {model} {shard}/{shards}] {i + 1}/{len(mine)} {time.time() - started:.0f}s", flush=True)
    path = OUT / f"signals_{split}_{model}_{shard}of{shards}.pkl"
    path.write_bytes(pickle.dumps(out))
    print(f"wrote {path.name}: {len(out)} episodes in {time.time() - started:.0f}s", flush=True)


def load_signals(split: str, model: str) -> list[dict]:
    """Merge every shard file of a split and model, in a stable order."""
    files = sorted(OUT.glob(f"signals_{split}_{model}_*of*.pkl"))
    if not files:
        raise FileNotFoundError(f"no signal shards for {split}/{model}")
    shards = {int(f.stem.split("_")[-1].split("of")[1]) for f in files}
    if len(shards) != 1 or len(files) != shards.pop():
        raise RuntimeError(f"incomplete or mixed shard set for {split}/{model}: {[f.name for f in files]}")
    episodes = [e for f in files for e in pickle.loads(f.read_bytes())]
    return sorted(episodes, key=lambda e: (e["dataset"], e["seed"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["development", "held_out"], required=True)
    p.add_argument("--model", choices=list(cfg.MONITORED_MODELS), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    main(a.split, a.model, a.shard, a.shards)
