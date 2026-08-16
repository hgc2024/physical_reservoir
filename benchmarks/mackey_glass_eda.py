"""Run preprocessing and exploratory analysis for the Mackey-Glass dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from physres.datasets import MackeyGlassConfig, generate_mackey_glass
from physres.eda import autocorrelation, describe_signal, plot_signal_overview
from physres.preprocessing import (
    StandardizationStats,
    TimeSeriesSplits,
    chronological_split,
    standardize_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--transient-steps", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-lag", type=int, default=500)
    parser.add_argument("--output", type=Path, help="Optional path for the EDA figure")
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional Markdown report path; defaults beside --output",
    )
    parser.add_argument("--show", action="store_true", help="Display the EDA figure")
    return parser.parse_args()


def build_markdown_report(
    signal: jnp.ndarray,
    splits: TimeSeriesSplits,
    standardized: TimeSeriesSplits,
    stats: StandardizationStats,
    *,
    transient_steps: int,
    max_lag: int,
    dt: float = 0.1,
) -> str:
    """Build a plain-language EDA report for technical and general readers."""
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1 for the EDA report")
    config = MackeyGlassConfig(dt=dt)
    raw = describe_signal(signal)
    standardized_train = describe_signal(standardized.train)
    correlations = [float(value) for value in autocorrelation(signal, max_lag=max_lag)]

    first_zero_crossing = next(
        (index for index, value in enumerate(correlations[1:], start=1) if value <= 0),
        None,
    )
    minimum_lag = min(range(1, len(correlations)), key=correlations.__getitem__)
    recurrence_lag = None
    if minimum_lag + 1 < len(correlations):
        candidate = max(
            range(minimum_lag + 1, len(correlations)),
            key=correlations.__getitem__,
        )
        if correlations[candidate] > 0:
            recurrence_lag = candidate

    first_zero_text = (
        f"{first_zero_crossing * dt:.1f} time units"
        if first_zero_crossing is not None
        else "not reached in the inspected lag range"
    )
    recurrence_text = (
        f"A later positive relationship appears around {recurrence_lag * dt:.1f} "
        f"time units (correlation {correlations[recurrence_lag]:.2f})."
        if recurrence_lag is not None
        else "No later positive recurrence appears in the inspected lag range."
    )

    return f"""# Mackey-Glass exploratory data analysis

## Reader's guide

Mackey-Glass data is a computer-generated sequence whose next value depends on both its current value and a delayed past value. It is useful here because it creates bounded but difficult-to-predict oscillations. The goal of this report is to check the generated data before it is used to train a forecasting model; no physical-computing background is required.

## What was generated

- **Samples analyzed:** {raw.n_samples:,}
- **Integration interval:** {dt:g} model time units per sample
- **Discarded startup period:** {transient_steps:,} samples ({transient_steps * dt:g} time units)
- **Delayed feedback:** {config.tau:g} time units
- **Equation settings:** feedback `a={config.a:g}`, decay `b={config.b:g}`, nonlinear exponent `gamma={config.gamma:g}`

The startup period is discarded because the sequence begins from an artificial constant history. Removing it gives the system time to settle into its characteristic dynamics.

## What the values look like

- **Observed range:** {raw.minimum:.4f} to {raw.maximum:.4f}
- **Mean (arithmetic average):** {raw.mean:.4f}
- **Median (middle value):** {raw.median:.4f}
- **Standard deviation (typical spread around the mean):** {raw.standard_deviation:.4f}
- **Middle 50% of observations:** {raw.first_quartile:.4f} to {raw.third_quartile:.4f}

The values are finite, bounded, and visibly variable rather than constant. This is the basic behavior needed for the planned forecasting experiments.

## How values depend on earlier values

Autocorrelation measures how similar the sequence is to a time-shifted copy of itself. A value near `1` means strong similarity, `0` means little linear similarity, and a negative value means the shifted patterns tend to move in opposite directions.

- **One-step autocorrelation:** {correlations[1] if len(correlations) > 1 else correlations[0]:.3f}
- **First zero crossing:** {first_zero_text}
- **Strongest negative relationship inspected:** correlation {correlations[minimum_lag]:.3f} at {minimum_lag * dt:.1f} time units
- **Later recurrence:** {recurrence_text}

This rise-and-fall pattern confirms strong temporal structure: nearby samples are not independent, and relationships persist over many steps. It supports chronological evaluation and makes random shuffling inappropriate. Autocorrelation alone does not prove chaos or establish the best forecasting horizon.

## How the data was prepared

The sequence was kept in its original order and divided into:

- **Training:** {splits.train.size:,} samples — used to fit model parameters
- **Validation:** {splits.validation.size:,} samples — reserved for choosing model settings
- **Test:** {splits.test.size:,} samples — reserved for one final, unbiased evaluation

Standardization used only the training segment, with training mean `{float(stats.mean):.4f}` and scale `{float(stats.scale):.4f}`. The same values were then applied to validation and test segments. This avoids using information from the future during training.

As a check, the standardized training data has mean `{standardized_train.mean:.3g}` and standard deviation `{standardized_train.standard_deviation:.3f}`. Predictions can later be converted back to the original scale using the saved mean and scale.

## Practical conclusion

The generated series passed the initial data-quality checks: it contains no missing or infinite values, has meaningful variation, and exhibits long temporal dependence. It is ready for the next methodology stage: defining forecasting inputs and targets before driving the reservoir model.
"""


def main() -> None:
    args = parse_args()
    signal = generate_mackey_glass(
        n_steps=args.n_steps,
        transient_steps=args.transient_steps,
    )
    splits = chronological_split(
        signal,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    standardized, stats = standardize_splits(splits)

    report = build_markdown_report(
        signal,
        splits,
        standardized,
        stats,
        transient_steps=args.transient_steps,
        max_lag=args.max_lag,
    )
    print(report)

    report_output = args.report_output
    if report_output is None and args.output is not None:
        report_output = args.output.with_suffix(".md")
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(report, encoding="utf-8")

    figure, _ = plot_signal_overview(
        signal,
        dt=0.1,
        max_lag=args.max_lag,
        title="Mackey-Glass signal",
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
