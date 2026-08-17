"""Forecast-quality metrics reported on held-out time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class ForecastMetrics:
    mse: float
    rmse: float
    nrmse: float
    r_squared: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "nrmse": self.nrmse,
            "r_squared": self.r_squared,
        }


def evaluate_forecast(targets: Any, predictions: Any) -> ForecastMetrics:
    """Evaluate predictions; NRMSE is normalized by target standard deviation."""
    observed = jnp.asarray(targets)
    predicted = jnp.asarray(predictions)
    if observed.ndim != 1 or predicted.shape != observed.shape or observed.size == 0:
        raise ValueError("targets and predictions must be non-empty matching vectors")
    if not bool(jnp.all(jnp.isfinite(observed))) or not bool(
        jnp.all(jnp.isfinite(predicted))
    ):
        raise ValueError("targets and predictions must contain only finite values")

    residual = observed - predicted
    mse = jnp.mean(residual**2)
    target_variance = jnp.mean((observed - jnp.mean(observed)) ** 2)
    if float(target_variance) == 0.0:
        raise ValueError("forecast metrics require variable target values")
    return ForecastMetrics(
        mse=float(mse),
        rmse=float(jnp.sqrt(mse)),
        nrmse=float(jnp.sqrt(mse / target_variance)),
        r_squared=float(1.0 - mse / target_variance),
    )
