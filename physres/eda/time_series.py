"""Numerical and visual exploratory analysis for one-dimensional signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SignalSummary:
    """Compact descriptive statistics for a signal."""

    n_samples: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    median: float
    first_quartile: float
    third_quartile: float

    def as_dict(self) -> dict[str, int | float]:
        """Return the summary in a serialization-friendly form."""
        return asdict(self)


def _as_valid_signal(signal: Any, *, minimum_length: int = 1) -> jax.Array:
    values = jnp.asarray(signal)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if values.size < minimum_length:
        raise ValueError(f"signal must contain at least {minimum_length} samples")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.float32)
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("signal must contain only finite values")
    return values


def describe_signal(signal: Any) -> SignalSummary:
    """Calculate descriptive statistics without modifying the signal."""
    values = _as_valid_signal(signal)
    quartiles = jnp.quantile(values, jnp.asarray([0.25, 0.5, 0.75]))
    return SignalSummary(
        n_samples=int(values.size),
        minimum=float(jnp.min(values)),
        maximum=float(jnp.max(values)),
        mean=float(jnp.mean(values)),
        standard_deviation=float(jnp.std(values)),
        median=float(quartiles[1]),
        first_quartile=float(quartiles[0]),
        third_quartile=float(quartiles[2]),
    )


def autocorrelation(signal: Any, *, max_lag: int) -> jax.Array:
    """Return the normalized, biased autocorrelation from lag zero onward."""
    values = _as_valid_signal(signal, minimum_length=2)
    if not isinstance(max_lag, int) or isinstance(max_lag, bool):
        raise ValueError("max_lag must be an integer")
    if not 0 <= max_lag < values.size:
        raise ValueError("max_lag must be between 0 and len(signal) - 1")

    centered = values - jnp.mean(values)
    energy = jnp.dot(centered, centered)
    if float(energy) == 0.0:
        raise ValueError("autocorrelation is undefined for a constant signal")
    correlation = jnp.correlate(centered, centered, mode="full")
    non_negative_lags = correlation[values.size - 1 :]
    return non_negative_lags[: max_lag + 1] / energy


def plot_signal_overview(
    signal: Any,
    *,
    dt: float = 1.0,
    max_lag: int = 200,
    title: str = "Signal overview",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Plot a trace, value distribution, and autocorrelation diagnostic."""
    values = _as_valid_signal(signal, minimum_length=2)
    if dt <= 0:
        raise ValueError("dt must be positive")

    correlations = autocorrelation(values, max_lag=max_lag)
    time = jnp.arange(values.size) * dt
    lags = jnp.arange(correlations.size) * dt

    figure, axes_array = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    trace_axis, histogram_axis, correlation_axis = axes_array
    trace_axis.plot(time, values, linewidth=1.0)
    trace_axis.set(xlabel="Time (model time units)", ylabel="Value", title=title)
    histogram_axis.hist(values, bins=40)
    histogram_axis.set(
        xlabel="Observed value",
        ylabel="Number of samples",
        title="How often each value occurs",
    )
    correlation_axis.plot(lags, correlations, linewidth=1.0)
    correlation_axis.axhline(0.0, color="black", linewidth=0.75)
    correlation_axis.set(
        xlabel="Time shift (model time units)",
        ylabel="Similarity (-1 to 1)",
        title="Similarity to earlier values (autocorrelation)",
    )
    for axis in axes_array:
        axis.grid(alpha=0.2)
    return figure, (trace_axis, histogram_axis, correlation_axis)
