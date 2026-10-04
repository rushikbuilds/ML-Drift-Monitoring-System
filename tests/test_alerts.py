from dews.alerts import AlertEngine
from dews.config import MonitorConfig
from dews.forecasting import ForecastResult


def _forecast(ttd: float | None) -> ForecastResult:
    return ForecastResult([0.4] * 8, [0.3] * 8, [0.5] * 8, ttd, ttd, ttd, "test")


def test_alert_requires_consecutive_confirmation() -> None:
    engine = AlertEngine(MonitorConfig(alert_confirmation_batches=2))

    assert engine.evaluate(0, 0.5, _forecast(None)) is None
    alert = engine.evaluate(1, 0.5, _forecast(None))

    assert alert is not None
    assert alert.severity == "Critical"


def test_forecast_alert_requires_minimum_history() -> None:
    engine = AlertEngine(MonitorConfig(alert_confirmation_batches=1, forecast_min_history=4))

    short_forecast = ForecastResult([0.2, 0.4], [0.1, 0.3], [0.3, 0.5], 1.0, 1.0, 1.0, "test")

    assert engine.evaluate(0, 0.2, short_forecast, forecast_eligible=False) is None


def test_small_threshold_breach_is_warning() -> None:
    engine = AlertEngine(MonitorConfig(alert_confirmation_batches=1, critical_score_margin=0.10))

    alert = engine.evaluate(0, 0.35, _forecast(None))

    assert alert is not None
    assert alert.severity == "Warning"
