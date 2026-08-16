import jax.numpy as jnp

from benchmarks.mackey_glass_eda import build_markdown_report
from physres.preprocessing import chronological_split, standardize_splits


def test_markdown_report_explains_results_in_plain_language():
    time = jnp.linspace(0.0, 20.0, 200)
    signal = 1.0 + 0.2 * jnp.sin(time)
    splits = chronological_split(signal)
    standardized, stats = standardize_splits(splits)

    report = build_markdown_report(
        signal,
        splits,
        standardized,
        stats,
        transient_steps=100,
        max_lag=50,
    )

    assert report.startswith("# Mackey-Glass exploratory data analysis")
    assert "no physical-computing background is required" in report
    assert "Autocorrelation measures" in report
    assert "future during training" in report
    assert "does not prove chaos" in report
    assert "**Training:** 140 samples" in report
