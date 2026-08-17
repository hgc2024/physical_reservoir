import jax.numpy as jnp

from physres.engine import build_reservoir, solve_trajectory
from physres.substrates import MemristorNanonetwork


def test_memristor_reservoir_produces_bounded_diverse_states():
    inputs = jnp.sin(jnp.linspace(0.0, 10.0, 200))
    reservoir = build_reservoir(state_size=16, seed=3)

    states = reservoir.transform(inputs)

    assert states.shape == (200, 16)
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all((states >= 0.0) & (states <= 1.0))
    assert float(jnp.std(states)) > 0.0


def test_reservoir_is_reproducible_for_a_fixed_seed():
    inputs = jnp.linspace(-1.0, 1.0, 50)

    first = build_reservoir(state_size=8, seed=10).transform(inputs)
    second = build_reservoir(state_size=8, seed=10).transform(inputs)

    assert jnp.array_equal(first, second)


def test_solver_supports_euler_and_rk4():
    substrate = MemristorNanonetwork(state_size=6)
    inputs = jnp.ones(20)

    euler = solve_trajectory(substrate, inputs, method="euler")
    rk4 = solve_trajectory(substrate, inputs, method="rk4")

    assert euler.shape == rk4.shape == (20, 6)
    assert jnp.all(jnp.isfinite(euler))
    assert jnp.all(jnp.isfinite(rk4))
