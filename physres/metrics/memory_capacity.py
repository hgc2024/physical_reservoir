"""Linear memory-capacity evaluation for reservoir state trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from physres.readouts import fit_ridge


@dataclass(frozen=True)
class MemoryCapacityResult:
    delays: jax.Array
    capacities: jax.Array
    total_capacity: float


def calculate_linear_memory_capacity(
    states: Any,
    inputs: Any,
    *,
    max_delay: int = 50,
    train_fraction: float = 0.7,
    alpha: float = 1e-4,
) -> MemoryCapacityResult:
    """Measure reconstruction of past inputs from current reservoir states.

    Each delay score is the squared Pearson correlation on a chronological
    held-out segment. The total is the sum over delays ``1..max_delay``.
    """
    x = jnp.asarray(states)
    u = jnp.asarray(inputs)
    if x.ndim != 2 or u.ndim != 1 or x.shape[0] != u.size:
        raise ValueError("states and inputs must have aligned time dimensions")
    if not isinstance(max_delay, int) or isinstance(max_delay, bool) or max_delay <= 0:
        raise ValueError("max_delay must be a positive integer")
    if u.size <= max_delay + 2:
        raise ValueError("trajectory is too short for max_delay")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    usable_states = x[max_delay:]
    split_index = int(usable_states.shape[0] * train_fraction)
    if split_index == 0 or split_index == usable_states.shape[0]:
        raise ValueError("train_fraction produces an empty memory-capacity split")

    scores = []
    for delay in range(1, max_delay + 1):
        delayed_input = u[max_delay - delay : u.size - delay]
        model = fit_ridge(
            usable_states[:split_index], delayed_input[:split_index], alpha=alpha
        )
        prediction = model.predict(usable_states[split_index:])
        target = delayed_input[split_index:]
        centered_prediction = prediction - jnp.mean(prediction)
        centered_target = target - jnp.mean(target)
        denominator = jnp.sqrt(
            jnp.sum(centered_prediction**2) * jnp.sum(centered_target**2)
        )
        correlation = jnp.where(
            denominator > 0,
            jnp.sum(centered_prediction * centered_target) / denominator,
            0.0,
        )
        scores.append(jnp.clip(correlation**2, 0.0, 1.0))

    capacities = jnp.stack(scores)
    delays = jnp.arange(1, max_delay + 1)
    return MemoryCapacityResult(
        delays=delays,
        capacities=capacities,
        total_capacity=float(jnp.sum(capacities)),
    )
