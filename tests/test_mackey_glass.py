import jax
import jax.numpy as jnp
import pytest

from physres.datasets import MackeyGlassConfig, generate_mackey_glass


def test_default_generator_returns_requested_jax_signal():
    signal = generate_mackey_glass(n_steps=128)

    assert isinstance(signal, jax.Array)
    assert signal.shape == (128,)
    assert signal.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(signal))
    assert not jnp.all(signal == signal[0])


def test_first_step_matches_documented_euler_equation():
    signal = generate_mackey_glass(n_steps=1)
    initial = 1.2
    derivative = 0.2 * initial / (1.0 + initial**10.0) - 0.1 * initial
    expected = initial + 0.1 * derivative

    assert signal[0] == pytest.approx(expected)


def test_transient_steps_are_discarded_without_changing_trajectory():
    full_signal = generate_mackey_glass(n_steps=40)
    after_transient = generate_mackey_glass(n_steps=30, transient_steps=10)

    assert jnp.array_equal(after_transient, full_signal[10:])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_steps": 0}, "n_steps"),
        ({"transient_steps": -1}, "transient_steps"),
        ({"tau": 0.0}, "tau"),
        ({"dt": 0.0}, "dt"),
        ({"tau": 1.0, "dt": 0.3}, "integer multiple"),
        ({"dtype": jnp.int32}, "floating-point"),
    ],
)
def test_invalid_parameters_raise_clear_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_mackey_glass(**({"n_steps": 8} | kwargs))


def test_config_exposes_discrete_delay_length():
    config = MackeyGlassConfig()

    config.validate()
    assert config.delay_steps == 170
