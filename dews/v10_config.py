"""Frozen protocol for v10: fewer harmless-shift alarms from the harm tier (experiment branch).

Frozen on 2026-11-09 after the development gate (scripts/run_v10_dev.py) and BEFORE any
confirmation dataset was downloaded. See docs/v10_protocol.md.
"""
from __future__ import annotations

FROZEN_ON = "2026-11-09"

RULE = "z-score"            # v8 harm estimate (importance-weighted accuracy drop) / its burn-in noise scale
WATCH_GATE = False
THRESHOLD_OBJECTIVE = ("minimize the harmless-alarm share subject to T2 >= 0.60 and <= 5 false alerts "
                       "per 100 clean batches, on pooled development episodes")
PRIMARY = {"model": "logistic_regression", "label_delay": 5, "label_fraction": 1.0}

CONFIRMATION_DATASETS = {   # never used before in this project; target = majority class vs rest
    "road_safety": 42803, "diabetes130": 4541, "jannis": 41168, "rl": 41160, "fried": 901, "jungle_chess": 41027,
}
UNAVAILABLE: tuple[str, ...] = ()
MAX_FEATURES = 20
CONFIRMATION_SEED_BASE = 11000

SUCCESS = {"harmless_max": 0.15, "t2_min": 0.50, "false_alerts_max": 5.0}
