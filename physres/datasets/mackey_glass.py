"""JAX generator for the chaotic Mackey-Glass time series."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MackeyGlassConfig:
    """Parameters for a discretized Mackey-Glass trajectory.

    The defaults match the chaotic regime documented in
    ``mackey_glass_data.md``. ``gamma`` is the nonlinear exponent in that
    document; ``b`` is the linear decay coefficient.
    """

    tau: float = 17.0
    dt: float = 0.1
    a: float = 0.2
    b: float = 0.1
    gamma: float = 10.0
    initial_value: float = 1.2

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration cannot be integrated."""
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.a < 0:
            raise ValueError("a must be non-negative")
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive")

        delay_steps = self.tau / self.dt
        if not math.isclose(delay_steps, round(delay_steps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("tau must be an integer multiple of dt")

    @property
    def delay_steps(self) -> int:
        """Number of discrete samples in the delay interval."""
        return round(self.tau / self.dt)


def generate_mackey_glass(
    n_steps: int = 5000,
    tau: float = 17.0,
    dt: float = 0.1,
    a: float = 0.2,
    b: float = 0.1,
    gamma: float = 10.0,
    *,
    initial_value: float = 1.2,
    transient_steps: int = 0,
    dtype: Any = jnp.float32,
) -> jax.Array:
    """Generate a Mackey-Glass trajectory using explicit Euler integration.

    The discretized delay-differential equation is

    ``dx/dt = a * x(t - tau) / (1 + x(t - tau)**gamma) - b * x(t)``.

    A constant history equal to ``initial_value`` is assumed for
    ``-tau <= t <= 0``. The first ``transient_steps`` integrated values are
    discarded, and the following ``n_steps`` values are returned as a JAX
    array. This function only generates the raw signal; chronological
    splitting and scaling are provided by ``physres.preprocessing``, while
    windowing and signal transformations belong to later pipeline stages.

    Args:
        n_steps: Number of samples to return.
        tau: Delay of the feedback term.
        dt: Euler integration step size.
        a: Delayed-feedback coefficient.
        b: Linear decay coefficient.
        gamma: Exponent controlling feedback nonlinearity.
        initial_value: Constant value of the initial history.
        transient_steps: Number of initial integrated samples to discard.
        dtype: Floating-point JAX dtype for the trajectory.

    Returns:
        A one-dimensional JAX array with shape ``(n_steps,)``.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if (
        not isinstance(transient_steps, int)
        or isinstance(transient_steps, bool)
        or transient_steps < 0
    ):
        raise ValueError("transient_steps must be a non-negative integer")

    config = MackeyGlassConfig(
        tau=tau,
        dt=dt,
        a=a,
        b=b,
        gamma=gamma,
        initial_value=initial_value,
    )
    config.validate()

    result_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(result_dtype, jnp.floating):
        raise ValueError("dtype must be a floating-point dtype")

    history = jnp.full(config.delay_steps + 1, initial_value, dtype=result_dtype)
    coefficients = tuple(
        jnp.asarray(value, dtype=result_dtype)
        for value in (dt, a, b, gamma)
    )

    def euler_step(
        carry: tuple[jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        delay_buffer, oldest_index = carry
        step_dt, feedback, decay, exponent = coefficients
        delayed_value = delay_buffer[oldest_index]
        current_value = delay_buffer[(oldest_index - 1) % delay_buffer.size]
        derivative = (
            feedback * delayed_value / (1.0 + delayed_value**exponent)
            - decay * current_value
        )
        next_value = current_value + step_dt * derivative
        delay_buffer = delay_buffer.at[oldest_index].set(next_value)
        oldest_index = (oldest_index + 1) % delay_buffer.size
        return (delay_buffer, oldest_index), next_value

    total_steps = transient_steps + n_steps
    (_, _), trajectory = jax.lax.scan(
        euler_step,
        (history, jnp.asarray(0, dtype=jnp.int32)),
        xs=None,
        length=total_steps,
    )
    return trajectory[transient_steps:]
