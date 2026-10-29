"""Event, risk-target, and alert metrics for delayed-label experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean
from typing import Sequence

from .experiment_config import DEFAULT_EARLY_WARNING_CONFIG


_PROTOCOL = DEFAULT_EARLY_WARNING_CONFIG.protocol
_EVENT = _PROTOCOL.event


@dataclass(frozen=True, slots=True)
class RiskTarget:
    """Future-event label for one signal row and the batch when it matures."""

    example_batch: int
    label: int
    available_batch: int


@dataclass(frozen=True, slots=True)
class EventAlertMatch:
    """One-to-one match between an event and its selected alert."""

    event_batch: int
    alert_batch: int | None
    warning_lead_batches: int | None
    detection_lag_batches: int | None


def complete_risk_target_batches(
    total_batches: int,
    *,
    horizon_batches: int = _PROTOCOL.horizon_batches,
    event_consecutive_batches: int = _EVENT.consecutive_batches,
) -> tuple[int, ...]:
    """Return examples whose horizon and event-confirmation tail are observable."""
    if total_batches < 0:
        raise ValueError("total_batches cannot be negative")
    if horizon_batches < 1:
        raise ValueError("horizon_batches must be at least 1")
    if event_consecutive_batches < 1:
        raise ValueError("event_consecutive_batches must be at least 1")
    count = max(total_batches - horizon_batches - event_consecutive_batches + 1, 0)
    return tuple(range(count))


def risk_target_available_batch(
    example_batch: int,
    *,
    horizon_batches: int = _PROTOCOL.horizon_batches,
    label_delay_batches: int = 0,
    event_consecutive_batches: int = _EVENT.consecutive_batches,
) -> int:
    """Return the first logical batch at which a future-risk target matures."""
    if example_batch < 0:
        raise ValueError("example_batch cannot be negative")
    if horizon_batches < 1:
        raise ValueError("horizon_batches must be at least 1")
    if label_delay_batches < 0:
        raise ValueError("label_delay_batches cannot be negative")
    if event_consecutive_batches < 1:
        raise ValueError("event_consecutive_batches must be at least 1")
    return example_batch + horizon_batches + event_consecutive_batches - 1 + label_delay_batches


def accuracy_baseline(
    accuracies: Sequence[float],
    baseline_batches: int = _EVENT.baseline_batches,
) -> float | None:
    """Return the mean of the first full clean baseline window, if available."""
    if baseline_batches < 1:
        raise ValueError("baseline_batches must be at least 1")
    if len(accuracies) < baseline_batches:
        return None
    baseline = [float(value) for value in accuracies[:baseline_batches]]
    if any(not isfinite(value) for value in baseline):
        return None
    return float(mean(baseline))


def find_degradation_onset(
    accuracies: Sequence[float],
    baseline_accuracy: float | None,
    *,
    start_batch: int,
    accuracy_drop: float = _EVENT.accuracy_drop,
    consecutive_batches: int = _EVENT.consecutive_batches,
) -> int | None:
    """Find the first sustained accuracy drop relative to a fixed baseline.

    The returned index is the first batch in the qualifying consecutive run.
    The baseline is supplied by the caller so controlled scenarios can use
    known-clean pre-intervention batches and real streams can declare their
    baseline window explicitly.
    """
    if not 0.0 < accuracy_drop < 1.0:
        raise ValueError("accuracy_drop must be between 0 and 1")
    if consecutive_batches < 1:
        raise ValueError("consecutive_batches must be at least 1")
    if start_batch < 0:
        raise ValueError("start_batch cannot be negative")
    if baseline_accuracy is None or not isfinite(float(baseline_accuracy)):
        return None

    threshold = float(baseline_accuracy) - accuracy_drop
    last_start = len(accuracies) - consecutive_batches
    for index in range(start_batch, last_start + 1):
        window = [float(value) for value in accuracies[index:index + consecutive_batches]]
        if len(window) == consecutive_batches and all(
            isfinite(value) and value <= threshold for value in window
        ):
            return index
    return None


def build_risk_targets(
    *,
    total_batches: int,
    event_onset_batch: int | None,
    horizon_batches: int = _PROTOCOL.horizon_batches,
    label_delay_batches: int = 0,
    event_consecutive_batches: int = _EVENT.consecutive_batches,
) -> list[RiskTarget]:
    """Build complete future-event targets and their release batch indices.

    A target at batch ``t`` is positive when an event begins in ``(t, t+H]``.
    It matures at ``t + H + (K - 1) + L``: the horizon must close, a possible
    event at its last batch needs ``K - 1`` later batches to confirm, and those
    labels must arrive. Targets whose horizon plus confirmation tail would
    extend beyond the episode are omitted.
    """
    if label_delay_batches < 0:
        raise ValueError("label_delay_batches cannot be negative")
    if event_consecutive_batches < 1:
        raise ValueError("event_consecutive_batches must be at least 1")
    if event_onset_batch is not None and not (
        0 <= event_onset_batch <= total_batches - event_consecutive_batches
    ):
        raise ValueError("event_onset_batch must have enough batches for confirmation")

    targets = []
    for batch_index in complete_risk_target_batches(
        total_batches,
        horizon_batches=horizon_batches,
        event_consecutive_batches=event_consecutive_batches,
    ):
        is_positive = (
            event_onset_batch is not None
            and batch_index < event_onset_batch <= batch_index + horizon_batches
        )
        targets.append(
            RiskTarget(
                example_batch=batch_index,
                label=int(is_positive),
                available_batch=risk_target_available_batch(
                    batch_index,
                    horizon_batches=horizon_batches,
                    label_delay_batches=label_delay_batches,
                    event_consecutive_batches=event_consecutive_batches,
                ),
            )
        )
    return targets


def stable_clean_batch_indices(
    *,
    total_batches: int,
    event_batches: Sequence[int],
    horizon_batches: int = _PROTOCOL.horizon_batches,
) -> tuple[int, ...]:
    """Return the stable pre-warning region used for false-alert rates.

    With no confirmed event, the whole episode is eligible as clean monitoring
    time, including episodes that contain a harmless input shift. When an event
    occurs, batches from ``event - horizon`` onward are excluded because a
    warning in that window may be a valid early alert.
    """
    if total_batches < 0:
        raise ValueError("total_batches cannot be negative")
    if horizon_batches < 0:
        raise ValueError("horizon_batches cannot be negative")
    events = [int(batch) for batch in event_batches]
    if any(batch < 0 or batch >= total_batches for batch in events):
        raise ValueError("event batches must be within the episode")
    if not events:
        return tuple(range(total_batches))
    clean_end = max(min(events) - horizon_batches, 0)
    return tuple(range(clean_end))


def match_alerts_to_events(
    alert_batches: Sequence[int],
    event_batches: Sequence[int],
    *,
    horizon_batches: int = _PROTOCOL.horizon_batches,
) -> tuple[list[EventAlertMatch], list[int]]:
    """Match at most one alert to each event within the warning/detection window.

    For a given event, the earliest alert in its preceding horizon is selected
    as the warning. If there is no preceding alert, the first alert from the
    event through the following horizon is selected as a detection. Repeated
    alerts remain unmatched and are preserved for alert-rate accounting.
    """
    if horizon_batches < 0:
        raise ValueError("horizon_batches cannot be negative")
    if any(batch < 0 for batch in alert_batches) or any(batch < 0 for batch in event_batches):
        raise ValueError("batch indices cannot be negative")

    indexed_alerts = sorted(enumerate(int(batch) for batch in alert_batches), key=lambda item: item[1])
    events = sorted(int(batch) for batch in event_batches)
    if len(events) != len(set(events)):
        raise ValueError("event batches must be unique")

    used_alert_indices: set[int] = set()
    matches: list[EventAlertMatch] = []
    for event_batch in events:
        candidates = [
            (index, alert_batch)
            for index, alert_batch in indexed_alerts
            if index not in used_alert_indices
            and event_batch - horizon_batches <= alert_batch <= event_batch + horizon_batches
        ]
        early = [(index, batch) for index, batch in candidates if batch < event_batch]
        at_or_after = [(index, batch) for index, batch in candidates if batch >= event_batch]
        selected = min(early, key=lambda item: item[1]) if early else (
            min(at_or_after, key=lambda item: item[1]) if at_or_after else None
        )

        if selected is None:
            matches.append(EventAlertMatch(event_batch, None, None, None))
            continue

        alert_index, alert_batch = selected
        used_alert_indices.add(alert_index)
        if alert_batch < event_batch:
            lead = event_batch - alert_batch
            lag = None
        else:
            lead = None
            lag = alert_batch - event_batch
        matches.append(EventAlertMatch(event_batch, alert_batch, lead, lag))

    unmatched = [batch for index, batch in indexed_alerts if index not in used_alert_indices]
    return matches, unmatched


def evaluate_episode_alerts(
    alert_batches: Sequence[int],
    event_batches: Sequence[int],
    *,
    horizon_batches: int = _PROTOCOL.horizon_batches,
    clean_batch_indices: Sequence[int] = (),
    harmless_shift_batch_indices: Sequence[int] = (),
) -> dict:
    """Summarize event matching without treating a missing event as no alarms."""
    alerts = [int(batch) for batch in alert_batches]
    events = [int(batch) for batch in event_batches]
    matches, unmatched = match_alerts_to_events(
        alerts,
        events,
        horizon_batches=horizon_batches,
    )
    matched_count = sum(match.alert_batch is not None for match in matches)
    warning_leads = [
        match.warning_lead_batches
        for match in matches
        if match.warning_lead_batches is not None
    ]
    detection_lags = [
        match.detection_lag_batches
        for match in matches
        if match.detection_lag_batches is not None
    ]
    clean_indices = set(int(batch) for batch in clean_batch_indices)
    harmless_indices = set(int(batch) for batch in harmless_shift_batch_indices)
    clean_unmatched_count = sum(batch in clean_indices for batch in unmatched)
    harmless_alert_count = sum(batch in harmless_indices for batch in alerts)
    first_event = min(events) if events else None
    false_before_event = (
        sum(batch < first_event for batch in unmatched)
        if first_event is not None
        else None
    )

    return {
        "event_present": bool(events),
        "event_count": len(events),
        "matched_event_count": matched_count,
        "event_recall": matched_count / len(events) if events else None,
        "event_precision": matched_count / len(alerts) if alerts else None,
        "alerts_generated": len(alerts),
        "matched_alert_batches": [match.alert_batch for match in matches if match.alert_batch is not None],
        "unmatched_alert_batches": unmatched,
        "alerts_without_confirmed_event": len(unmatched),
        "false_alerts_before_degradation": false_before_event,
        "false_alerts_per_100_clean_batches": (
            100.0 * clean_unmatched_count / len(clean_indices)
            if clean_indices
            else None
        ),
        "harmless_shift_alerts": harmless_alert_count if harmless_indices else None,
        "warning_lead_batches": float(mean(warning_leads)) if warning_leads else None,
        "detection_lag_batches": float(mean(detection_lags)) if detection_lags else None,
        "event_matches": [asdict(match) for match in matches],
    }
