"""Extract v6 signals for the natural-stream windows used by H5.

    python scripts/run_v6_natural.py --shard 0 --shards 6

The 13 natural window sets are the windows already used in v2-v4 (2000 reference rows,
then 25 natural batches of 250 rows; no injected shift). Each window gets the v6
treatment: logistic regression fit on the first 1500 reference rows, permutation
importance on the last 500, and the same signal extraction as the benchmark.
Writes artifacts/v6/natural_signals_<shard>of<shards>.pkl.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v6_config as cfg  # noqa: E402
from dews.resample_benchmark import V6Episode, build_model  # noqa: E402
from dews.v6_signals import extract_episode  # noqa: E402
from scripts.run_relevance_real import episode as natural_window  # noqa: E402

OUT = ROOT / "artifacts" / "v6"


def windows() -> list[tuple[str, int]]:
    dev = pickle.loads((ROOT / "artifacts" / "v2" / "real_aux_signals.pkl").read_bytes())
    held = pickle.loads((ROOT / "artifacts" / "v2" / "new_signals.pkl").read_bytes())
    return [(s["scenario"], int(s["seed"])) for s in dev + held]


def to_v6(name: str, start: int) -> V6Episode | None:
    ep = natural_window(name, start)
    fit_rows = cfg.REFERENCE_ROWS - cfg.IMPORTANCE_HOLDOUT_ROWS
    ref, ref_y = ep.reference.astype(float), np.asarray(ep.reference_labels, int)
    if len(set(ref_y[:fit_rows])) < 2:
        return None  # a model cannot be fit on a single-class reference (window skipped)
    model = build_model("logistic_regression").fit(ref.iloc[:fit_rows], ref_y[:fit_rows])
    imp = permutation_importance(model, ref.iloc[fit_rows:], ref_y[fit_rows:], n_repeats=5,
                                 random_state=start, scoring="accuracy").importances_mean
    return V6Episode(name, "natural", start, "logistic_regression", ref, ref_y,
                     [b.astype(float) for b in ep.batches], [np.asarray(l, int) for l in ep.labels],
                     None, model, imp, {})


def main(shard: int, shards: int) -> None:
    mine = windows()[shard::shards]
    out = []
    for name, start in mine:
        ep = to_v6(name, start)
        if ep is None:
            continue
        r = extract_episode(ep)
        r["kind"] = "natural"
        out.append(r)
    (OUT / f"natural_signals_{shard}of{shards}.pkl").write_bytes(pickle.dumps(out))
    print(f"shard {shard}: {len(out)} of {len(mine)} windows", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    main(a.shard, a.shards)
