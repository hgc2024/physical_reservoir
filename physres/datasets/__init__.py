"""Dataset generators used by physical-reservoir experiments."""

from physres.datasets.mackey_glass import (
    MackeyGlassConfig,
    generate_mackey_glass,
)

__all__ = ["MackeyGlassConfig", "generate_mackey_glass"]
