"""Frozen protocol for the v6 early-warning benchmark (Day 1, 2026-11-08).

Everything an experiment may depend on is defined here, before any v6 result exists:
datasets and their split, episode design, seeds, settings, the primary system, the
early-warning tiers, and each hypothesis with its PASS criterion. Code for Days 2-7
reads these values; it must not redefine them. Any later change is a deviation and
must be listed in docs/v6_protocol.md under "Deviations".

The only parameter that may change after Day 1 is ``LAMBDA_MAX`` (shift strength), and
only through the Day 2 sanity gate, using development datasets.
"""
from __future__ import annotations

from dataclasses import dataclass

FROZEN_ON = "2026-11-08"

# --------------------------------------------------------------------------- datasets
# The 13 cached binary datasets. Held-out = numpy default_rng(2026).choice(sorted names, 5).
ALL_DATASETS = (
    "airlines", "bank_marketing", "covertype", "covpokelec", "electricity", "hyperplane",
    "kddcup99", "ldpa", "led", "poker_hand", "run_or_walk", "sea", "vehicle_sensit",
)
SPLIT_SEED = 2026
HELD_OUT_DATASETS = ("airlines", "bank_marketing", "electricity", "ldpa", "sea")
DEVELOPMENT_DATASETS = tuple(d for d in ALL_DATASETS if d not in HELD_OUT_DATASETS)

# --------------------------------------------------------------------------- episodes
REFERENCE_ROWS = 2000            # labeled reference: model fit + detector reference
IMPORTANCE_HOLDOUT_ROWS = 500    # last rows of the reference, used only for permutation importance
POOL_ROWS = 20000                # stream pool, contiguous with the reference block
BATCH_SIZE = 300
N_BATCHES = 45                   # long enough to confirm the latest possible event at L = 10
ONSET_RANGE = (10, 16)           # intervention onset batch, inclusive
RAMP_BATCHES = 15                # gradual shift / concept change reach full strength after this
LAMBDA_MAX = 1.5                 # re-sampling strength: weight = exp(lambda(t) * z_j)
SAMPLE_WITH_REPLACEMENT = True   # each batch is drawn from the full pool, so the shift never runs out of rows
CONCEPT_FLIP_MAX = 0.3           # max label-flip probability inside the affected subpopulation
MIN_CLASS_SHARE = 0.05           # a block is redrawn until both classes have >= 5% of rows

EPISODES_PER_DATASET = {"clean": 10, "gradual_shift": 30, "abrupt_shift": 15, "concept_change": 10}
IMPORTANT_FEATURE_PROB = 0.5     # chance the shifted feature comes from the top half by importance

DEV_SEED_BASE = 1000             # episode seed = base + 100 * dataset_index + episode_index
HELD_OUT_SEED_BASE = 5000

# --------------------------------------------------------------------------- settings
LABEL_DELAYS = (0, 2, 5, 10)
LABEL_FRACTIONS = (1.0, 0.2)
MONITORED_MODELS = ("logistic_regression", "gradient_boosting")
PRIMARY = {"model": "logistic_regression", "label_delay": 5, "label_fraction": 1.0}

# --------------------------------------------------------------------------- event rule
BASELINE_BATCHES = 5
ACCURACY_DROP = 0.05
CONSECUTIVE_BATCHES = 2
WARMUP_BATCHES = 6               # alerts before this batch are ignored for every method
FALSE_ALERT_BUDGET = 5.0         # per 100 clean batches


def confirmation_batch(event_onset: int, label_delay: int) -> int:
    """First batch at which delayed labels can confirm the event (labels of batch b usable at b + max(L, 1))."""
    return event_onset + CONSECUTIVE_BATCHES - 1 + max(label_delay, 1)


# --------------------------------------------------------------------------- system
PRIMARY_SYSTEM = (
    "Delay-aware risk calibrator: logistic regression on the state target "
    "(a confirmed degraded run starts or continues within the next 5 batches), features = "
    "v2 signal features (current, 3-batch mean, 3-batch slope of drift and confidence signals) "
    "+ matured-label accuracy-drop features + auxiliary-model disagreement (current and change) "
    "+ relevance-weighted shift using permutation importance of the monitored model. "
    "Trained on development episodes for each (model, delay, label fraction). "
    "An alert is the WARNING tier: risk >= threshold chosen on development data."
)
TUNABLE_ON_DEVELOPMENT = ("alert threshold per method and setting", "LAMBDA_MAX via the Day 2 gate")
HORIZON_BATCHES = 5

BASELINES = (
    "fixed_dews_score",        # original DEWS weighted drift score
    "statistical_only",
    "confidence_only",
    "mean_ks",                 # unweighted shift
    "label_trend",             # matured (observed) accuracy drop
    "delay_aware_cbpe",
)

# --------------------------------------------------------------------------- tiers
TIERS = {
    "T1_predictive": "alert in [intervention_onset, event_onset)",
    "T2_pre_confirmation": "alert in [intervention_onset, confirmation_batch(event_onset, L))",
    "T3_confirmed": "first alert at or after confirmation_batch; lag = alert - event_onset",
}


@dataclass(frozen=True)
class Hypothesis:
    id: str
    claim: str
    metric: str
    pass_rule: str


HYPOTHESES = (
    Hypothesis("H1", "Predictive warning (T1) for gradual harmful shifts",
               "share of held-out harmful gradual-shift episodes with a T1 warning; median lead",
               "share >= 0.60 AND median lead >= 2 batches, with clean false alerts <= 5 per 100 batches"),
    Hypothesis("H2", "Earlier than waiting for labels (T2)",
               "T2 share(system) - T2 share(label_trend) over held-out harmful shift episodes, primary setting",
               "difference >= 0.10 AND paired bootstrap 95% CI lower bound > 0"),
    Hypothesis("H3", "Separates harmful from harmless shifts",
               "AUROC of mean risk over the first 3 post-onset batches, harmful vs harmless shift episodes; "
               "share of harmless shift episodes with any WARNING after onset",
               "AUROC >= 0.80 AND harmless-episode alert share <= 0.10"),
    Hypothesis("H4", "Beats the original DEWS fixed drift score",
               "held-out datasets where the system has higher T1 AND higher T2 share than fixed_dews_score",
               ">= 70% of held-out datasets (>= 4 of 5)"),
    Hypothesis("H5", "Carries over to natural streams",
               "T2 share(system) - T2 share(label_trend) on the 13 cached natural-stream window sets, L = 5",
               "difference >= 0"),
)

OVERALL_VERDICT = {
    "success": "H1 and H3 pass",
    "partial_success": "exactly one of H1, H4 passes (and not success)",
    "failure": "H1 fails and H4 fails",
}

DAY4_GATE = "switch to fallback if development T1 share on harmful gradual shifts < 0.40"
DAY2_GATE = ("on development data, >= 25% of shift episodes are harmful and >= 25% harmless; "
             "otherwise adjust LAMBDA_MAX using development datasets only")
