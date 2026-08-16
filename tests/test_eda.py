import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from physres.eda import autocorrelation, describe_signal, plot_signal_overview


def test_describe_signal_returns_expected_statistics():
    summary = describe_signal(jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    assert summary.n_samples == 4
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0
    assert summary.mean == 2.5
    assert summary.median == 2.5
    assert summary.standard_deviation == pytest.approx(1.1180339)


def test_autocorrelation_starts_at_one_and_respects_max_lag():
    correlations = autocorrelation(jnp.arange(10, dtype=jnp.float32), max_lag=4)

    assert correlations.shape == (5,)
    assert correlations[0] == pytest.approx(1.0)


def test_autocorrelation_rejects_constant_signal_and_invalid_lag():
    with pytest.raises(ValueError, match="constant"):
        autocorrelation(jnp.ones(5), max_lag=2)
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation(jnp.arange(5), max_lag=5)


def test_plot_signal_overview_builds_three_diagnostics():
    signal = jnp.sin(jnp.linspace(0.0, 8.0, 100))
    figure, axes = plot_signal_overview(signal, dt=0.1, max_lag=10)

    assert len(axes) == 3
    assert axes[0].get_title() == "Signal overview"
    assert axes[1].get_title() == "How often each value occurs"
    assert axes[2].get_title() == "Similarity to earlier values (autocorrelation)"
    plt.close(figure)
