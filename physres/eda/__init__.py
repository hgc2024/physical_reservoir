"""Exploratory analysis tools for reservoir input signals."""

from physres.eda.time_series import (
    SignalSummary,
    autocorrelation,
    describe_signal,
    plot_signal_overview,
)

__all__ = [
    "SignalSummary",
    "autocorrelation",
    "describe_signal",
    "plot_signal_overview",
]
