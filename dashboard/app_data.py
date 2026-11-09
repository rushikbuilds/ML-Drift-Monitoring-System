"""Data and replay logic for the Streamlit dashboard (no Streamlit imports; unit-tested).

The live monitor replays saved episodes with the frozen v7 WATCH and v8 HARM thresholds.
Labels are shown only once they have arrived (batch b is visible from batch b + max(L, 1)).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dews import v2
from dews import v6_config as v6cfg
from dews import v7_config as v7cfg
from dews import v7_monitor as mon
from dews.harm_estimation import estimated_drop

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

SOURCES = {
    "Unseen data (v8 confirmation)": "v8_confirmation",
    "Development data": "development",
}
KIND_LABELS = {"clean": "Clean", "gradual_shift": "Gradual shift", "abrupt_shift": "Abrupt shift",
               "concept_change": "Label change"}
STATUS = {
    "calibrating": ("Calibrating", "Learning this stream's normal behaviour (burn-in)."),
    "healthy": ("Healthy", "No drift and no sign of harm."),
    "watch": ("Drift watch", "Inputs or model behaviour have drifted. Not necessarily harmful."),
    "harm": ("Harm warning", "Drift is moving data into regions where the model makes more mistakes."),
    "confirmed": ("Degradation confirmed", "Delayed labels confirm a sustained accuracy drop."),
}


def thresholds() -> dict[str, float]:
    v7 = json.loads((ART / "v7" / "frozen_v7.json").read_text())["thresholds"]
    v8 = json.loads((ART / "v8" / "frozen_v8.json").read_text())
    return {"watch": float(v7["watch"]), "harm": float(v8["harm_threshold"])}


def load_episodes(source: str) -> list[dict[str, Any]]:
    """Episodes with both monitor signals and harm signals (iw_accuracy)."""
    if source == "v8_confirmation":
        files = sorted((ART / "v8").glob("signals_v8_confirmation_*of*.pkl"))
        eps = [e for f in files for e in pickle.loads(f.read_bytes())]
    else:
        files = sorted((ART / "v6").glob("signals_development_logistic_regression_*of*.pkl"))
        eps = [e for f in files for e in pickle.loads(f.read_bytes())]
        harm = {}
        for f in sorted((ART / "v8").glob("harm_development_*of*.pkl")):
            for r in pickle.loads(f.read_bytes()):
                harm[(r["dataset"], r["seed"])] = r
        for e in eps:
            e.update({k: harm[(e["dataset"], e["seed"])][k] for k in ("iw_accuracy", "knn_accuracy")})
    if not eps:
        raise FileNotFoundError(f"no saved episodes for {source}; run the experiment scripts first")
    return sorted(eps, key=lambda e: (e["dataset"], e["seed"]))


def episode_label(e: dict[str, Any]) -> str:
    outcome = {"harmful": "harmful", "harmless": "harmless", "clean": "no event", "spurious": "noisy event",
               "concept_event": "accuracy event", "concept_no_event": "no event"}.get(e["category"], e["category"])
    return f"Seed {e['seed']}: {KIND_LABELS[e['kind']].lower()}, {outcome}"


def replay(e: dict[str, Any], delay: int) -> pd.DataFrame:
    """Per-batch monitor state as the system would see it, plus evaluation-only truth."""
    thr = thresholds()
    n = len(e["accuracy"])
    acc = np.asarray(e["accuracy"], float)
    watch = mon.watch_score(e, delay, 1.0)
    harm = estimated_drop(e["iw_accuracy"], v7cfg.BURN_IN_BATCHES)
    baseline = float(acc[: v6cfg.BASELINE_BATCHES].mean())
    lag = max(delay, 1)
    event = v2.outcome(acc)["event_onset"]
    confirmed_at = v6cfg.confirmation_batch(event, delay) if event is not None else None
    status = []
    for t in range(n):
        if t < v7cfg.BURN_IN_BATCHES:
            status.append("calibrating")
        elif confirmed_at is not None and t >= confirmed_at:
            status.append("confirmed")
        elif np.isfinite(harm[t]) and harm[t] >= thr["harm"]:
            status.append("harm")
        elif np.isfinite(watch[t]) and watch[t] >= thr["watch"]:
            status.append("watch")
        else:
            status.append("healthy")
    return pd.DataFrame({
        "batch": np.arange(n),
        "watch_score": watch, "watch_ratio": watch / thr["watch"],
        "harm_drop": harm, "harm_ratio": harm / thr["harm"],
        "estimated_accuracy": np.asarray(e["iw_accuracy"], float),
        "true_accuracy": acc,
        "labels_arrive_at": np.arange(n) + lag,
        "status": status,
        "baseline_accuracy": baseline,
    })


def visible_accuracy(frame: pd.DataFrame, t: int) -> pd.Series:
    """Observed accuracy at time t: only batches whose labels have arrived."""
    return frame["true_accuracy"].where(frame["labels_arrive_at"] <= t)


def signal_breakdown(e: dict[str, Any], delay: int, t: int) -> pd.DataFrame:
    """z-score of every WATCH signal at batch t (which signals drive the alert)."""
    raw = mon.raw_signals(e, delay, 1.0)
    names = {"drift_score": "DEWS drift score", "mean_ks": "Mean KS shift", "rws": "Relevance-weighted shift",
             "disagreement": "Model disagreement", "confidence_drop": "Confidence drop", "entropy": "Prediction entropy",
             "positive_rate_shift": "Positive-rate shift"}
    rows = [{"signal": names[k], "z": float(mon.robust_z(raw[k])[t])} for k in v7cfg.WATCH_SIGNALS]
    return pd.DataFrame(rows).sort_values("z", ascending=False)


def episode_facts(e: dict[str, Any]) -> dict[str, Any]:
    meta = e.get("meta", {})
    event = v2.outcome(np.asarray(e["accuracy"], float))["event_onset"]
    return {"Dataset": e["dataset"], "Episode type": KIND_LABELS[e["kind"]], "Outcome": e["category"],
            "Shift starts at batch": e["onset"], "Accuracy event at batch": event,
            "Shifted feature": meta.get("feature"),
            "Feature importance": ("top half" if meta.get("from_important_half") else "bottom half")
            if meta.get("from_important_half") is not None else None}


def load_results() -> dict[str, Any]:
    return {
        "v6": json.loads((ART / "v6" / "verdict.json").read_text()),
        "v7": json.loads((ART / "v7" / "v7_verdict.json").read_text()),
        "v8": json.loads((ART / "v8" / "v8_verdict.json").read_text()),
        "ablations": json.loads((ART / "ablations" / "ablations.json").read_text()),
    }


def doc_text(name: str) -> str:
    """A results document without its top-level heading (the page shows its own title)."""
    import re

    lines = (ROOT / "docs" / name).read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines[1:] if lines and lines[0].startswith("# ") else lines)
    return re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"\1", text)  # repository-relative links don't work in the app
