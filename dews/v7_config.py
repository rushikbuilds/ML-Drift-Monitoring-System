"""Frozen protocol for v7: the stream-normalized two-tier monitor (mitigation after the v6 failure).

Frozen on 2026-11-09, after the v6 verdict (FAILURE) and a development-only feasibility check,
and before any confirmation dataset was downloaded. Episode design, event rule, budget and
metrics are inherited unchanged from ``dews.v6_config``. See docs/v7_protocol.md.
"""
from __future__ import annotations

from . import v6_config as v6

FROZEN_ON = "2026-11-09"

# --------------------------------------------------------------------------- monitor
BURN_IN_BATCHES = 8          # first batches after deployment define each stream's own normal
SCALE_FLOOR = 0.01           # robust scale = max(1.4826 * MAD over burn-in, floor)
WATCH_SIGNALS = ("drift_score", "mean_ks", "rws", "disagreement", "confidence_drop", "entropy", "positive_rate_shift")
WARNING_FEATURES = "z-scores of WATCH_SIGNALS and of delay-aware CBPE drop, observed label drop; current and 3-batch mean"
PRIMARY = {"model": "logistic_regression", "label_delay": 5, "label_fraction": 1.0}

# --------------------------------------------------------------------------- data
DEVELOPMENT_DATASETS = v6.DEVELOPMENT_DATASETS        # thresholds and the WARNING calibrator are fit here only
CONFIRMATION_DATASETS = {                              # never used before v7; OpenML ids
    "adult": 1590, "miniboone": 41150, "higgs": 23512, "nomao": 1486, "numerai": 23517, "click_prediction": 1220,
}
MAX_FEATURES = 20                                      # first 20 non-constant features after numeric encoding
CONFIRMATION_SEED_BASE = 9000                          # episode seed = base + 100 * dataset index + episode index
POST_HOC_DATASETS = v6.HELD_OUT_DATASETS               # already seen in v6: reported as post-hoc only

# --------------------------------------------------------------------------- hypotheses (targets as in v6)
HYPOTHESES = {
    "C1": ("WATCH warns before the accuracy drop for gradual harmful shifts",
           "T1 share >= 0.60 AND median lead >= 2 AND false alerts <= 5 per 100 clean batches"),
    "C2": ("WATCH warns before labels can confirm, better than waiting for labels",
           "T2(WATCH) - T2(label_trend) >= 0.10 AND paired bootstrap CI lower bound > 0"),
    "C3": ("WARNING separates harmful from harmless shifts",
           "AUROC >= 0.80 AND harmless-episode alert share <= 0.10"),
    "C4": ("WATCH beats the original (raw) DEWS fixed score",
           "higher T1 AND higher T2 on >= 70% of confirmation datasets"),
    "C5": ("WATCH carries over to natural streams",
           "T2(WATCH) - T2(label_trend) >= 0 on the natural-stream windows, L = 5"),
}
OVERALL_VERDICT = {"success": "C1 and C3 pass", "partial_success": "not success, and C1 or C4 passes",
                   "failure": "C1 and C4 both fail"}
