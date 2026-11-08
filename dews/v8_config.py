"""Frozen protocol for v8: reference-based harm estimation (last separation experiment).

Frozen on 2026-11-09, after a development-only feasibility check and before any v8
confirmation dataset was downloaded. Episode design, event rule, budget and tiers are
inherited from v6; the WATCH tier and its threshold are inherited from v7. See docs/v8_protocol.md.
"""
from __future__ import annotations

FROZEN_ON = "2026-11-09"

HARM_SCORE = "iw_drop"          # importance-weighted reference accuracy, drop vs burn-in median (accuracy units)
BURN_IN_BATCHES = 8
THRESHOLD_RULE = ("maximize T2 share on harmful shifts minus alert share on harmless shifts, "
                  "subject to <= 5 false alerts per 100 clean batches, on pooled development episodes")
PRIMARY = {"model": "logistic_regression", "label_delay": 5, "label_fraction": 1.0}

CONFIRMATION_DATASETS = {
    "albert": 41147, "credit_default": 42477, "kick": 41162, "house_16h": 821, "2dplanes": 727, "mv": 881,
}
MAX_FEATURES = 20
CONFIRMATION_SEED_BASE = 9500

HYPOTHESES = {
    "D1": ("HARM separates harmful from harmless shifts", "AUROC (mean over first 3 post-onset batches) >= 0.80"),
    "D2": ("HARM rarely alerts on harmless shifts", "harmless-episode alert share <= 0.10 with false alerts <= 5 per 100"),
    "D3": ("HARM cuts harmless alarms versus WATCH while still warning",
           "harmless share(WATCH) - harmless share(HARM) >= 0.30 AND HARM T2 on harmful shifts >= 0.50 AND HARM false alerts <= 5"),
    "D4": ("HARM ranks harm better than WATCH", "AUROC(HARM) - AUROC(WATCH) >= 0.10 with paired bootstrap 95% CI lower bound > 0"),
}
OVERALL_VERDICT = {"success": "D1 and D2 pass", "partial_success": "not success, and D3 or D4 passes",
                   "failure": "otherwise"}
