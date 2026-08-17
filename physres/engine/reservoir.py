"""High-level physical reservoir construction and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from physres.engine.solver import solve_trajectory
from physres.substrates import AbstractSubstrate, MemristorNanonetwork


@dataclass(frozen=True)
class PhysicalReservoir:
    """An input-driven substrate with fixed integration settings."""

    substrate: AbstractSubstrate
    dt: float = 0.1
    method: str = "rk4"

    def transform(self, inputs: Any, *, initial_state: Any | None = None) -> jax.Array:
        """Map a scalar input sequence to a reservoir-state feature matrix."""
        return solve_trajectory(
            self.substrate,
            inputs,
            dt=self.dt,
            initial_state=initial_state,
            method=self.method,
        )


def build_reservoir(
    state_size: int = 50,
    *,
    seed: int = 0,
    dt: float = 0.1,
    **substrate_kwargs: Any,
) -> PhysicalReservoir:
    """Construct the default volatile memristive physical reservoir."""
    substrate = MemristorNanonetwork(
        state_size=state_size, seed=seed, **substrate_kwargs
    )
    return PhysicalReservoir(substrate=substrate, dt=dt)
