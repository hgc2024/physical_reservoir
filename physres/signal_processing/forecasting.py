"""Create causal forecasting inputs and targets from ordered signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ForecastPairs:
    """Current scalar inputs paired with later target values."""

    inputs: jax.Array
    targets: jax.Array
    horizon: int


@dataclass(frozen=True)
class LaggedForecast:
    """Causal lagged features paired with later target values."""

    features: jax.Array
    targets: jax.Array
    horizon: int
    lag_spacing: int


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _as_signal(signal: Any) -> jax.Array:
    values = jnp.asarray(signal)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.float32)
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("signal must contain only finite values")
    return values


def make_forecast_pairs(signal: Any, *, horizon: int = 1) -> ForecastPairs:
    """Pair each usable value with the value ``horizon`` samples ahead."""
    values = _as_signal(signal)
    _validate_positive_integer(horizon, "horizon")
    if values.size <= horizon:
        raise ValueError("signal must be longer than horizon")
    return ForecastPairs(values[:-horizon], values[horizon:], horizon)


def build_lagged_forecast(
    signal: Any,
    *,
    n_lags: int,
    horizon: int = 1,
    lag_spacing: int = 1,
) -> LaggedForecast:
    """Build causal lagged features ordered from oldest to current value."""
    values = _as_signal(signal)
    _validate_positive_integer(n_lags, "n_lags")
    _validate_positive_integer(horizon, "horizon")
    _validate_positive_integer(lag_spacing, "lag_spacing")

    history = (n_lags - 1) * lag_spacing
    if values.size <= history + horizon:
        raise ValueError("signal is too short for the requested lags and horizon")
    current_indices = jnp.arange(history, values.size - horizon)
    offsets = jnp.arange(n_lags - 1, -1, -1) * lag_spacing
    features = values[current_indices[:, None] - offsets[None, :]]
    targets = values[current_indices + horizon]
    return LaggedForecast(features, targets, horizon, lag_spacing)


def align_states_and_targets(
    states: Any,
    signal: Any,
    *,
    horizon: int = 1,
    washout: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Align reservoir states at time ``t`` with signal targets at ``t+h``."""
    state_values = jnp.asarray(states)
    signal_values = _as_signal(signal)
    _validate_positive_integer(horizon, "horizon")
    if not isinstance(washout, int) or isinstance(washout, bool) or washout < 0:
        raise ValueError("washout must be a non-negative integer")
    if state_values.ndim != 2:
        raise ValueError("states must be a two-dimensional matrix")
    if state_values.shape[0] != signal_values.size:
        raise ValueError("states and signal must contain the same number of samples")
    if washout + horizon >= signal_values.size:
        raise ValueError("washout and horizon leave no forecasting samples")
    if not bool(jnp.all(jnp.isfinite(state_values))):
        raise ValueError("states must contain only finite values")
    return state_values[washout:-horizon], signal_values[washout + horizon :]
