"""General preprocessing primitives for one-dimensional time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TimeSeriesSplits:
    """Chronologically ordered train, validation, and test segments."""

    train: jax.Array
    validation: jax.Array
    test: jax.Array


@dataclass(frozen=True)
class StandardizationStats:
    """Training-set statistics used to standardize a signal."""

    mean: jax.Array
    scale: jax.Array

    def transform(self, signal: Any) -> jax.Array:
        """Standardize ``signal`` using the stored training statistics."""
        values = jnp.asarray(signal)
        return (values - self.mean) / self.scale

    def inverse_transform(self, signal: Any) -> jax.Array:
        """Restore standardized values to the original scale."""
        values = jnp.asarray(signal)
        return values * self.scale + self.mean


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


def chronological_split(
    signal: Any,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> TimeSeriesSplits:
    """Split a signal in temporal order without shuffling.

    The test fraction is the remainder after the train and validation
    fractions. Every requested segment must contain at least one sample.
    """
    values = _as_valid_signal(signal, minimum_length=2)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than 1")

    train_end = int(values.size * train_fraction)
    validation_end = train_end + int(values.size * validation_fraction)
    if train_end == 0:
        raise ValueError("train split is empty; provide more samples")
    if validation_fraction > 0.0 and validation_end == train_end:
        raise ValueError("validation split is empty; provide more samples")
    if validation_end == values.size:
        raise ValueError("test split is empty; provide more samples")

    return TimeSeriesSplits(
        train=values[:train_end],
        validation=values[train_end:validation_end],
        test=values[validation_end:],
    )


def fit_standardizer(signal: Any, *, epsilon: float = 1e-8) -> StandardizationStats:
    """Fit mean and standard deviation statistics to one signal segment."""
    values = _as_valid_signal(signal)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    scale = jnp.std(values)
    if float(scale) < epsilon:
        raise ValueError("cannot standardize a constant or near-constant signal")
    return StandardizationStats(mean=jnp.mean(values), scale=scale)


def standardize_splits(
    splits: TimeSeriesSplits,
    *,
    epsilon: float = 1e-8,
) -> tuple[TimeSeriesSplits, StandardizationStats]:
    """Standardize all splits using statistics fitted only on training data."""
    stats = fit_standardizer(splits.train, epsilon=epsilon)
    standardized = TimeSeriesSplits(
        train=stats.transform(splits.train),
        validation=stats.transform(splits.validation),
        test=stats.transform(splits.test),
    )
    return standardized, stats
