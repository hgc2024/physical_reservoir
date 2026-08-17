import jax.numpy as jnp
import pytest

from physres.metrics import calculate_linear_memory_capacity, evaluate_forecast
from physres.readouts import fit_ridge


def test_ridge_recovers_a_linear_mapping():
    x = jnp.linspace(-2.0, 2.0, 100)[:, None]
    y = 3.0 * x[:, 0] + 2.0

    model = fit_ridge(x, y, alpha=1e-6)
    prediction = model.predict(x)

    assert jnp.max(jnp.abs(prediction - y)) < 1e-4


def test_forecast_metrics_report_perfect_predictions():
    target = jnp.asarray([1.0, 2.0, 4.0])
    metrics = evaluate_forecast(target, target)

    assert metrics.mse == 0.0
    assert metrics.rmse == 0.0
    assert metrics.nrmse == 0.0
    assert metrics.r_squared == 1.0


def test_memory_capacity_detects_explicit_delay_features():
    inputs = jnp.sin(jnp.linspace(0.0, 50.0, 500))
    states = jnp.column_stack(
        [jnp.roll(inputs, delay) for delay in range(1, 6)]
    )

    result = calculate_linear_memory_capacity(states, inputs, max_delay=5)

    assert result.capacities.shape == (5,)
    assert result.total_capacity == pytest.approx(5.0, abs=0.05)
