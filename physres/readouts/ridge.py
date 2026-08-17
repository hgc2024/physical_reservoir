"""Closed-form ridge-regression readout for reservoir states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RidgeReadout:
    weights: jax.Array
    bias: jax.Array
    alpha: float

    def predict(self, features: Any) -> jax.Array:
        values = jnp.asarray(features)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("features have the wrong shape for this readout")
        return values @ self.weights + self.bias


def fit_ridge(
    features: Any,
    targets: Any,
    *,
    alpha: float = 1e-4,
) -> RidgeReadout:
    """Fit a centered ridge readout with an unregularized intercept."""
    x = jnp.asarray(features)
    y = jnp.asarray(targets)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or y.size == 0:
        raise ValueError("features and targets must have compatible non-empty shapes")
    if not bool(jnp.all(jnp.isfinite(x))) or not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError("features and targets must contain only finite values")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    x_mean = jnp.mean(x, axis=0)
    y_mean = jnp.mean(y)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    regularized = gram + alpha * jnp.eye(x.shape[1], dtype=x.dtype)
    weights = jnp.linalg.solve(regularized, centered_x.T @ centered_y)
    bias = y_mean - x_mean @ weights
    return RidgeReadout(weights=weights, bias=bias, alpha=alpha)
