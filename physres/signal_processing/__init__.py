"""Signal-to-task transformations for time-series forecasting."""

from physres.signal_processing.forecasting import (
    ForecastPairs,
    LaggedForecast,
    align_states_and_targets,
    build_lagged_forecast,
    make_forecast_pairs,
)

__all__ = [
    "ForecastPairs",
    "LaggedForecast",
    "align_states_and_targets",
    "build_lagged_forecast",
    "make_forecast_pairs",
]
