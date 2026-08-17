"""Physical substrate modules for reservoir dynamics."""

from physres.substrates.base import AbstractSubstrate
from physres.substrates.memristor import MemristorNanonetwork

__all__ = ["AbstractSubstrate", "MemristorNanonetwork"]
