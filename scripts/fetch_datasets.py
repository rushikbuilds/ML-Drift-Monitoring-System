"""Download and cache every dataset the v6-v8 experiments use (OpenML; needs network).

    python scripts/fetch_datasets.py

Uses each experiment's own loader, so the cached files match the preprocessing used in
the runs. Existing caches are kept. albert (v8) is skipped: OpenML returned HTTP 500 for it.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from scripts import run_final_evaluation, run_v2_realdata, run_v3_confirmation, run_v7, run_v8

    groups = [
        ("v2 real streams", lambda n: run_v2_realdata.load(n), list(run_v2_realdata.DATASETS)),
        ("v3 fresh streams", lambda n: run_v3_confirmation.load(n), list(run_v3_confirmation.FRESH)),
        ("v4 benchmark streams", lambda n: run_final_evaluation.load_new(n), list(run_final_evaluation.NEW)),
    ]
    for label, loader, names in groups:
        for name in names:
            loader(name)
            print(f"{label}: {name} cached", flush=True)
    run_v7.download()
    print("v7 confirmation datasets cached", flush=True)
    run_v8.download()
    print("v8 confirmation datasets cached", flush=True)


if __name__ == "__main__":
    main()
