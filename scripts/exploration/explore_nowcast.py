"""Exploration (not a reported result): nowcast MAE of accuracy estimators."""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from scripts.exploration.explore_v4 import rel_features  # noqa: E402

OUT = ROOT / "artifacts" / "v2"


def main():
    dev = pickle.loads((OUT / "real_signals.pkl").read_bytes())
    fresh = pickle.loads((OUT / "fresh_signals.pkl").read_bytes())["real"]
    sigs = dev + fresh
    names = sorted({s["scenario"] for s in sigs})
    for delay in (0, 2, 5):
        for s in sigs:
            s["_est"] = v2.accuracy_estimates(s, delay)
            s["_feat"] = rel_features(s, delay)
            s["_base"] = s["accuracy"][:5].mean()
        methods = ["stale_last", "stale_mean3", "cbpe_raw", "da_cbpe", "learned_gbr", "learned+da"]
        mae = {m: {} for m in methods}
        mae_deg = {m: {} for m in methods}
        for held in names:
            train = [s for s in sigs if s["scenario"] != held]
            test = [s for s in sigs if s["scenario"] == held]
            X = np.vstack([s["_feat"][v2.WARMUP:] for s in train])
            y = np.concatenate([(s["_base"] - s["accuracy"])[v2.WARMUP:] for s in train])
            gbr = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0).fit(X, y)
            errs = {m: [] for m in methods}
            degr = {m: [] for m in methods}
            for s in test:
                est = dict(s["_est"])
                est["learned_gbr"] = s["_base"] - gbr.predict(s["_feat"])
                est["learned+da"] = 0.5 * est["learned_gbr"] + 0.5 * est["da_cbpe"]
                bad = s["accuracy"] <= s["_base"] - v2.DROP
                for m in methods:
                    e = np.abs(np.clip(est[m], 0, 1) - s["accuracy"])[v2.WARMUP:]
                    errs[m].append(e)
                    d = np.abs(np.clip(est[m], 0, 1) - s["accuracy"])[bad]
                    if len(d):
                        degr[m].append(d)
            for m in methods:
                mae[m][held] = float(np.concatenate(errs[m]).mean())
                mae_deg[m][held] = float(np.concatenate(degr[m]).mean()) if degr[m] else np.nan
        print(f"\n=== L={delay}: nowcast MAE (accuracy points x100), macro over {len(names)} datasets ===")
        print(f"{'method':14s} all   degraded  | " + " ".join(f"{n[:6]:>6s}" for n in names))
        for m in methods:
            print(f"{m:14s} {100*np.mean(list(mae[m].values())):4.2f}  {100*np.nanmean(list(mae_deg[m].values())):6.2f}   | "
                  + " ".join(f"{100*mae[m][n]:6.2f}" for n in names))


if __name__ == "__main__":
    main()
