import jax.numpy as jnp
import pytest

from physres.preprocessing import (
    chronological_split,
    fit_standardizer,
    standardize_splits,
)


def test_chronological_split_preserves_order_and_uses_remainder_for_test():
    signal = jnp.arange(20, dtype=jnp.float32)

    splits = chronological_split(
        signal,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    assert jnp.array_equal(splits.train, signal[:10])
    assert jnp.array_equal(splits.validation, signal[10:15])
    assert jnp.array_equal(splits.test, signal[15:])


def test_standardization_uses_training_statistics_only_and_is_reversible():
    signal = jnp.arange(20, dtype=jnp.float32)
    splits = chronological_split(
        signal,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    standardized, stats = standardize_splits(splits)

    assert float(jnp.mean(standardized.train)) == pytest.approx(0.0, abs=1e-6)
    assert float(jnp.std(standardized.train)) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.mean(standardized.test)) > 0.0
    assert jnp.allclose(stats.inverse_transform(standardized.test), splits.test)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_fraction": 0.0},
        {"train_fraction": 1.0},
        {"validation_fraction": -0.1},
        {"train_fraction": 0.8, "validation_fraction": 0.2},
    ],
)
def test_chronological_split_rejects_invalid_fractions(kwargs):
    with pytest.raises(ValueError):
        chronological_split(jnp.arange(20), **kwargs)


def test_preprocessing_rejects_non_finite_and_constant_signals():
    with pytest.raises(ValueError, match="finite"):
        chronological_split(jnp.asarray([0.0, jnp.nan, 1.0]))
    with pytest.raises(ValueError, match="constant"):
        fit_standardizer(jnp.ones(10))
