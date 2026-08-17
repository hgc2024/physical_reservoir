"""Volatile memristive nanonetwork reservoir substrate."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from physres.substrates.base import AbstractSubstrate


class MemristorNanonetwork(AbstractSubstrate):
    """A bounded network of volatile conductive-filament state variables.

    Each state is a normalized filament length in ``[0, 1]``. Input and
    recurrent voltages drive filament growth through a boundary window,
    while exponential decay supplies fading memory.
    """

    input_weights: jax.Array
    recurrent_weights: jax.Array
    eta: float
    tau_decay: float
    r_on: float
    r_off: float
    input_scale: float
    recurrent_scale: float
    window_power: int
    initial_filament: float

    def __init__(
        self,
        state_size: int = 50,
        *,
        seed: int = 0,
        eta: float = 0.2,
        tau_decay: float = 10.0,
        r_on: float = 0.1,
        r_off: float = 1.0,
        input_scale: float = 0.25,
        recurrent_scale: float = 0.15,
        window_power: int = 2,
        initial_filament: float = 0.1,
    ) -> None:
        if not isinstance(state_size, int) or isinstance(state_size, bool) or state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if eta <= 0 or tau_decay <= 0:
            raise ValueError("eta and tau_decay must be positive")
        if not 0 < r_on < r_off:
            raise ValueError("resistances must satisfy 0 < r_on < r_off")
        if input_scale <= 0 or recurrent_scale < 0:
            raise ValueError("input_scale must be positive and recurrent_scale non-negative")
        if not isinstance(window_power, int) or window_power <= 0:
            raise ValueError("window_power must be a positive integer")
        if not 0 < initial_filament < 1:
            raise ValueError("initial_filament must be between 0 and 1")

        input_key, recurrent_key = jax.random.split(jax.random.PRNGKey(seed))
        self.input_weights = jax.random.uniform(
            input_key, (state_size,), minval=-1.0, maxval=1.0
        )
        raw_recurrent = jax.random.normal(recurrent_key, (state_size, state_size))
        self.recurrent_weights = raw_recurrent / jnp.sqrt(float(state_size))
        self.eta = eta
        self.tau_decay = tau_decay
        self.r_on = r_on
        self.r_off = r_off
        self.input_scale = input_scale
        self.recurrent_scale = recurrent_scale
        self.window_power = window_power
        self.initial_filament = initial_filament

    @property
    def state_size(self) -> int:
        return self.input_weights.size

    def initial_state(self) -> jax.Array:
        return jnp.full(
            (self.state_size,), self.initial_filament, dtype=self.input_weights.dtype
        )

    def output_map(self, x: jax.Array) -> jax.Array:
        filament = jnp.clip(x, 0.0, 1.0)
        resistance = self.r_on * filament + self.r_off * (1.0 - filament)
        conductance = 1.0 / resistance
        return (conductance - 1.0 / self.r_off) / (1.0 / self.r_on - 1.0 / self.r_off)

    def drift(self, t: float, x: jax.Array, args: jax.Array) -> jax.Array:
        del t
        filament = jnp.clip(x, 0.0, 1.0)
        activity = self.output_map(filament)
        input_voltage = self.input_scale * self.input_weights * args
        recurrent_voltage = self.recurrent_scale * (
            self.recurrent_weights @ (activity - 0.5)
        )
        resistance = self.r_on * filament + self.r_off * (1.0 - filament)
        current = (input_voltage + recurrent_voltage) / resistance
        window = 1.0 - (2.0 * filament - 1.0) ** (2 * self.window_power)
        return self.eta * current * window - filament / self.tau_decay

    def constrain_state(self, x: jax.Array) -> jax.Array:
        return jnp.clip(x, 0.0, 1.0)
