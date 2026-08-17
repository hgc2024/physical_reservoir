"""Fixed-step integration for input-driven physical substrates."""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp

from physres.substrates.base import AbstractSubstrate


def solve_trajectory(
    substrate: AbstractSubstrate,
    inputs: Any,
    *,
    dt: float = 0.1,
    initial_state: Any | None = None,
    method: Literal["euler", "rk4"] = "rk4",
) -> jax.Array:
    """Integrate a substrate with each input held constant for one interval."""
    values = jnp.asarray(inputs)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("inputs must be a non-empty one-dimensional signal")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("inputs must contain only finite values")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")

    state = substrate.initial_state() if initial_state is None else jnp.asarray(initial_state)
    if state.shape != substrate.initial_state().shape:
        raise ValueError("initial_state has the wrong shape for this substrate")

    def step(carry: tuple[jax.Array, jax.Array], input_value: jax.Array):
        x, time = carry
        if method == "euler":
            next_state = x + dt * substrate.drift(time, x, input_value)
        else:
            k1 = substrate.drift(time, x, input_value)
            k2 = substrate.drift(time + dt / 2, x + dt * k1 / 2, input_value)
            k3 = substrate.drift(time + dt / 2, x + dt * k2 / 2, input_value)
            k4 = substrate.drift(time + dt, x + dt * k3, input_value)
            next_state = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        next_state = substrate.constrain_state(next_state)
        return (next_state, time + dt), substrate.output_map(next_state)

    (_, _), outputs = jax.lax.scan(
        step, (state, jnp.asarray(0.0, dtype=state.dtype)), values
    )
    return outputs
