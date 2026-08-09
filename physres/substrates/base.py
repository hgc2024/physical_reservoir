"""Abstract base classes for reservoir substrates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import equinox as eqx
except ImportError:  # pragma: no cover - fallback for skeleton use
    class _Module:  # type: ignore[no-redef]
        pass

    eqx = None
    Module = _Module
else:
    Module = eqx.Module


class AbstractSubstrate(Module, ABC):
    """Common interface for differentiable physical substrates."""

    @abstractmethod
    def drift(self, t: float, x: Any, args: Any) -> Any:
        """Return the time derivative of the state."""

    @abstractmethod
    def output_map(self, x: Any) -> Any:
        """Map the internal state to an output vector."""
