from __future__ import annotations

import numpy as np

from dews.forecasting import DriftForecaster


def test_forecaster_estimates_forward_trend() -> None:
    forecaster = DriftForecaster(horizon=5)
    series = [0.1, 0.14, 0.19, 0.24, 0.31]

    result = forecaster.forecast(series, ds_crit=0.3)

    assert len(result.forecast) == 5
    assert result.model_name in {"ensemble", "holtwinters", "linear"}
    assert result.ttd_batches is not None
    assert result.forecast[-1] >= result.forecast[0]


def test_forecaster_uses_the_supplied_threshold() -> None:
    forecaster = DriftForecaster(horizon=5)
    series = [0.20, 0.22, 0.24, 0.26]

    result = forecaster.forecast(series, ds_crit=0.50)

    assert result.ttd_batches is None


def test_forecaster_interpolates_threshold_crossing() -> None:
    forecaster = DriftForecaster(horizon=3)

    crossing = forecaster._interpolated_crossing_time(
        0.30, np.array([0.40, 0.50]), 0.35
    )

    assert crossing is not None
    assert abs(crossing - 0.5) < 1e-9
