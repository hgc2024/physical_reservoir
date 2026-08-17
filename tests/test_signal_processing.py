import jax.numpy as jnp
import pytest

from physres.signal_processing import (
    align_states_and_targets,
    build_lagged_forecast,
    make_forecast_pairs,
)


def test_forecast_pairs_shift_targets_by_horizon():
    pairs = make_forecast_pairs(jnp.arange(8), horizon=2)

    assert jnp.array_equal(pairs.inputs, jnp.arange(6))
    assert jnp.array_equal(pairs.targets, jnp.arange(2, 8))


def test_lagged_forecast_is_causal_and_ordered_oldest_first():
    task = build_lagged_forecast(jnp.arange(10), n_lags=3, horizon=2, lag_spacing=2)

    assert jnp.array_equal(task.features[0], jnp.asarray([0, 2, 4]))
    assert task.targets[0] == 6
    assert task.features.shape == (4, 3)


def test_align_states_applies_washout_and_future_target_shift():
    signal = jnp.arange(10, dtype=jnp.float32)
    states = jnp.column_stack((signal, signal**2))

    features, targets = align_states_and_targets(
        states, signal, horizon=2, washout=3
    )

    assert jnp.array_equal(features, states[3:-2])
    assert jnp.array_equal(targets, signal[5:])


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_forecast_pairs_reject_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="horizon"):
        make_forecast_pairs(jnp.arange(8), horizon=horizon)
