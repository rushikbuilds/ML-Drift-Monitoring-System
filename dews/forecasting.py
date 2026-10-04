from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency branch
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None


@dataclass(slots=True)
class ForecastResult:
    forecast: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    ttd_batches: float | None
    ttd_lower_bound: float | None
    ttd_upper_bound: float | None
    model_name: str


class DriftForecaster:
    def __init__(self, horizon: int = 8) -> None:
        self.horizon = horizon

    def forecast(self, drift_series: list[float], ds_crit: float) -> ForecastResult:
        series = np.asarray(drift_series, dtype=float)
        if series.size == 0:
            return ForecastResult([], [], [], None, None, None, "empty")
        if series.size == 1:
            value = float(series[-1])
            return ForecastResult([value] * self.horizon, [value] * self.horizon, [value] * self.horizon, None, None, None, "flat")

        # Use a recent window for the forecast model so it reacts to
        # new trends rather than being anchored by a long stable history.
        max_window = max(2 * self.horizon, 12)
        model_series = series[-max_window:] if series.size > max_window else series

        if ExponentialSmoothing is not None and model_series.size >= 4:
            try:
                model = ExponentialSmoothing(model_series, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated")
                fit = model.fit(optimized=True)
                holt_forecast = np.asarray(fit.forecast(self.horizon), dtype=float)
                residuals = model_series - fit.fittedvalues
                residual_scale = float(np.std(residuals)) if residuals.size > 1 else 0.05
                robust_forecast = self._robust_forecast(float(series[-1]), model_series)
                predicted = 0.5 * holt_forecast + 0.5 * robust_forecast
                model_name = "ensemble"
            except Exception:
                predicted, residual_scale, model_name = self._linear_forecast(model_series)
        else:
            predicted, residual_scale, model_name = self._linear_forecast(model_series)

        residual_scale = max(residual_scale, 0.01)
        lower = np.clip(predicted - 1.96 * residual_scale, 0.0, 1.5)
        upper = np.clip(predicted + 1.96 * residual_scale, 0.0, 1.5)

        ttd_batches, ttd_lower, ttd_upper = self._estimate_ttd(float(series[-1]), predicted, lower, upper, ds_crit)
        return ForecastResult(
            forecast=predicted.tolist(),
            lower_bound=lower.tolist(),
            upper_bound=upper.tolist(),
            ttd_batches=ttd_batches,
            ttd_lower_bound=ttd_lower,
            ttd_upper_bound=ttd_upper,
            model_name=model_name,
        )

    def _robust_forecast(self, current_value: float, series: np.ndarray) -> np.ndarray:
        """Forecast with a median recent slope to reduce one-batch overreaction."""
        recent = series[-min(6, series.size):]
        slope = float(np.median(np.diff(recent))) if recent.size > 1 else 0.0
        steps = np.arange(1, self.horizon + 1, dtype=float)
        return current_value + slope * steps

    def _linear_forecast(self, series: np.ndarray) -> tuple[np.ndarray, float, str]:
        x = np.arange(series.size, dtype=float)
        slope, intercept = np.polyfit(x, series, deg=1)
        future_x = np.arange(series.size, series.size + self.horizon, dtype=float)
        predicted = slope * future_x + intercept
        fitted = slope * x + intercept
        residual_scale = float(np.std(series - fitted)) if series.size > 1 else 0.05
        return predicted, residual_scale, "linear"

    def _estimate_ttd(
        self,
        current_value: float,
        forecast: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        ds_crit: float,
    ) -> tuple[float | None, float | None, float | None]:
        if current_value >= ds_crit:
            return 0.0, 0.0, 0.0

        crossing = self._interpolated_crossing_time(current_value, forecast, ds_crit)
        crossing_low = self._interpolated_crossing_time(current_value, lower, ds_crit)
        crossing_high = self._interpolated_crossing_time(current_value, upper, ds_crit)

        if crossing is not None:
            return float(crossing), float(crossing_low or crossing), float(crossing_high or crossing)

        # Linear extrapolation beyond the forecast horizon, capped to
        # avoid absurd projections (e.g. 1271 batches).
        last_value = float(forecast[-1])
        if len(forecast) >= 2:
            slope = float((forecast[-1] - forecast[0]) / max(len(forecast) - 1, 1))
            if slope > 0:
                extra = max((ds_crit - last_value) / slope, 0.0)
                projected = float(len(forecast) + extra)
                # Cap at 2x horizon — beyond that the projection is unreliable
                if projected > 2 * len(forecast):
                    projected = None
            else:
                projected = None
        else:
            projected = None
        return projected, None, None

    @staticmethod
    def _interpolated_crossing_time(
        current_value: float,
        forecast: np.ndarray,
        threshold: float,
    ) -> float | None:
        """Estimate a fractional crossing time from the current score onward."""
        trajectory = np.concatenate(([current_value], np.asarray(forecast, dtype=float)))
        for index in range(1, len(trajectory)):
            previous = float(trajectory[index - 1])
            value = float(trajectory[index])
            if value >= threshold:
                change = value - previous
                fraction = (threshold - previous) / change if change > 0 else 0.0
                return float((index - 1) + np.clip(fraction, 0.0, 1.0))
        return None
