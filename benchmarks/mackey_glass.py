"""End-to-end Mackey-Glass forecasting benchmark for the physical reservoir."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from physres.datasets import generate_mackey_glass
from physres.engine import build_reservoir
from physres.metrics import (
    ForecastMetrics,
    calculate_linear_memory_capacity,
    evaluate_forecast,
)
from physres.preprocessing import chronological_split, standardize_splits
from physres.readouts import RidgeReadout, fit_ridge
from physres.signal_processing import (
    align_states_and_targets,
    build_lagged_forecast,
    make_forecast_pairs,
)


RIDGE_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--transient-steps", type=int, default=1000)
    parser.add_argument("--state-size", type=int, default=75)
    parser.add_argument("--horizon", type=int, default=170)
    parser.add_argument("--washout", type=int, default=100)
    parser.add_argument("--max-memory-delay", type=int, default=50)
    parser.add_argument("--output", type=Path, help="Optional benchmark figure path")
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional Markdown report path; defaults beside --output",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def select_ridge_alpha(
    train_features: jax.Array,
    train_targets: jax.Array,
    validation_features: jax.Array,
    validation_targets: jax.Array,
) -> tuple[RidgeReadout, ForecastMetrics]:
    """Select ridge strength by validation NRMSE, then return the fitted model."""
    candidates = []
    for alpha in RIDGE_ALPHAS:
        model = fit_ridge(train_features, train_targets, alpha=alpha)
        metrics = evaluate_forecast(
            validation_targets, model.predict(validation_features)
        )
        candidates.append((metrics.nrmse, model, metrics))
    _, model, metrics = min(candidates, key=lambda candidate: candidate[0])
    return model, metrics


def _metrics_row(name: str, metrics: ForecastMetrics) -> str:
    return (
        f"| {name} | {metrics.mse:.6f} | {metrics.rmse:.4f} | "
        f"{metrics.nrmse:.3f} | {metrics.r_squared:.3f} |"
    )


def build_report(
    *,
    horizon: int,
    state_size: int,
    washout: int,
    evaluation_samples: int,
    reservoir_model: RidgeReadout,
    reservoir_validation: ForecastMetrics,
    lagged_model: RidgeReadout,
    lagged_validation: ForecastMetrics,
    reservoir_metrics: ForecastMetrics,
    lagged_metrics: ForecastMetrics,
    persistence_metrics: ForecastMetrics,
    memory_capacity: float,
    max_memory_delay: int,
) -> str:
    improvement = 100.0 * (
        1.0 - reservoir_metrics.rmse / persistence_metrics.rmse
    )
    return f"""# Mackey-Glass reservoir benchmark

## Evaluation question

Can a simulated volatile memristive reservoir use the observed Mackey-Glass sequence to predict its value `{horizon}` samples (`{horizon * 0.1:g}` model time units) into the future? This horizon matches the delayed-feedback timescale used to generate the data.

## Signal processing and evaluation protocol

1. Generate one trajectory, remove its startup transient, and split it chronologically into training, validation, and test segments.
2. Standardize every segment using only the training mean and standard deviation.
3. Feed the standardized scalar sequence into a `{state_size}`-state memristive reservoir. Discard the first `{washout}` training states as washout.
4. Pair each reservoir state at time `t` with the signal at `t + {horizon}`.
5. Choose ridge regularization on validation data only. Evaluate once on `{evaluation_samples}` common test timestamps.
6. Compare with persistence (reuse the current value) and a causal linear baseline using 10 past values spaced 10 samples apart.

Keeping all splits in time order matters because nearby Mackey-Glass values are strongly related. Random splitting would allow closely related future observations into training and overstate forecasting quality.

## Validation selection

- **Reservoir ridge alpha:** `{reservoir_model.alpha:g}` (validation NRMSE `{reservoir_validation.nrmse:.3f}`)
- **Lagged-linear ridge alpha:** `{lagged_model.alpha:g}` (validation NRMSE `{lagged_validation.nrmse:.3f}`)

## Held-out test results

Lower MSE, RMSE, and NRMSE are better. NRMSE measures error relative to the natural variation in the test target. An R² of `1` is perfect; `0` is equivalent to always predicting the test mean.

| Method | MSE | RMSE | NRMSE | R² |
|---|---:|---:|---:|---:|
{_metrics_row("Memristive reservoir", reservoir_metrics)}
{_metrics_row("Lagged linear baseline", lagged_metrics)}
{_metrics_row("Persistence baseline", persistence_metrics)}

The reservoir RMSE is `{improvement:.1f}%` lower than persistence at this forecast horizon. This is a comparative benchmark result, not yet a general claim about other trajectories, seeds, substrates, or forecasting horizons.

## Fading-memory evaluation

Linear memory capacity was evaluated separately with an independent random input, because the strong autocorrelation of Mackey-Glass data would otherwise inflate delayed-reconstruction scores. The summed capacity over delays 1 through {max_memory_delay} is **{memory_capacity:.3f}**. Individual delay scores are shown in the figure.

## Discussion

The held-out result shows that the reservoir contains useful information for this specific delayed forecast. It follows the broad trajectory more closely than either baseline and reduces RMSE by `{improvement:.1f}%` relative to persistence. Its advantage over the lagged-linear model suggests that the fixed nonlinear state transformation contributes information beyond the selected raw lags.

The memory-capacity result adds an important qualification. A summed linear capacity of **{memory_capacity:.3f}** across {max_memory_delay} tested delays is modest relative to the `{state_size}` available states. The forecasting improvement therefore should not be interpreted as evidence of uniformly high linear memory. It may instead reflect a task-specific combination of fading memory, nonlinear response, and the match between the chosen forecast horizon and Mackey-Glass feedback delay.

This is a direct, teacher-forced forecast: the reservoir observes the real sequence through time `t` and predicts one value at `t + horizon`. It is not an autonomous rollout in which earlier predictions are fed back as later inputs. Overlapping test targets also come from one continuous synthetic trajectory, so they should not be treated as independent experimental replications.

Several limits remain. The benchmark uses one generated trajectory, one reservoir seed, one forecast horizon, clean simulated data, and one phenomenological memristor model. Only ridge regularization is selected automatically on validation data; substrate settings and baseline structure are exploratory fixed choices and have not been subjected to nested model selection. The fixed-step RK4 solution has not yet been checked against smaller steps or adaptive integration. These results are therefore an initial implementation benchmark rather than a confirmatory estimate of general performance or physical-hardware behavior.

The next useful tests are repeated seeds and trajectories, horizon sweeps, stronger nonlinear and autoregressive baselines, integration-convergence checks, noise and parameter perturbations, and autonomous multi-step forecasting. Those experiments will show which conclusions survive beyond the favorable configuration used for this draft.
"""


def main() -> None:
    args = parse_args()
    signal = generate_mackey_glass(
        args.n_steps, transient_steps=args.transient_steps
    )
    splits = chronological_split(signal)
    standardized, stats = standardize_splits(splits)
    combined = jnp.concatenate(
        [standardized.train, standardized.validation, standardized.test]
    )

    reservoir = build_reservoir(
        args.state_size,
        seed=0,
        eta=0.3,
        tau_decay=10.0,
        input_scale=0.2,
        recurrent_scale=0.25,
    )
    states = reservoir.transform(combined)
    train_end = standardized.train.size
    validation_end = train_end + standardized.validation.size
    train_states = states[:train_end]
    validation_states = states[train_end:validation_end]
    test_states = states[validation_end:]

    reservoir_train = align_states_and_targets(
        train_states,
        standardized.train,
        horizon=args.horizon,
        washout=args.washout,
    )
    reservoir_validation = align_states_and_targets(
        validation_states, standardized.validation, horizon=args.horizon
    )
    reservoir_test = align_states_and_targets(
        test_states, standardized.test, horizon=args.horizon
    )
    reservoir_model, reservoir_validation_metrics = select_ridge_alpha(
        *reservoir_train, *reservoir_validation
    )

    lagged_train = build_lagged_forecast(
        standardized.train, n_lags=10, lag_spacing=10, horizon=args.horizon
    )
    lagged_validation = build_lagged_forecast(
        standardized.validation, n_lags=10, lag_spacing=10, horizon=args.horizon
    )
    lagged_test = build_lagged_forecast(
        standardized.test, n_lags=10, lag_spacing=10, horizon=args.horizon
    )
    lagged_model, lagged_validation_metrics = select_ridge_alpha(
        lagged_train.features,
        lagged_train.targets,
        lagged_validation.features,
        lagged_validation.targets,
    )

    common_offset = (10 - 1) * 10
    reservoir_test_features = reservoir_test[0][common_offset:]
    reservoir_targets = reservoir_test[1][common_offset:]
    persistence = make_forecast_pairs(
        standardized.test, horizon=args.horizon
    ).inputs[common_offset:]
    lagged_prediction = lagged_model.predict(lagged_test.features)
    reservoir_prediction = reservoir_model.predict(reservoir_test_features)

    original_targets = stats.inverse_transform(reservoir_targets)
    original_reservoir_prediction = stats.inverse_transform(reservoir_prediction)
    original_lagged_prediction = stats.inverse_transform(lagged_prediction)
    original_persistence = stats.inverse_transform(persistence)
    reservoir_metrics = evaluate_forecast(
        original_targets, original_reservoir_prediction
    )
    lagged_metrics = evaluate_forecast(original_targets, original_lagged_prediction)
    persistence_metrics = evaluate_forecast(original_targets, original_persistence)

    memory_input = jax.random.uniform(
        jax.random.PRNGKey(123), (2500,), minval=-1.0, maxval=1.0
    )
    memory_states = reservoir.transform(memory_input)
    memory_result = calculate_linear_memory_capacity(
        memory_states[args.washout :],
        memory_input[args.washout :],
        max_delay=args.max_memory_delay,
        alpha=reservoir_model.alpha,
    )

    report = build_report(
        horizon=args.horizon,
        state_size=args.state_size,
        washout=args.washout,
        evaluation_samples=original_targets.size,
        reservoir_model=reservoir_model,
        reservoir_validation=reservoir_validation_metrics,
        lagged_model=lagged_model,
        lagged_validation=lagged_validation_metrics,
        reservoir_metrics=reservoir_metrics,
        lagged_metrics=lagged_metrics,
        persistence_metrics=persistence_metrics,
        memory_capacity=memory_result.total_capacity,
        max_memory_delay=args.max_memory_delay,
    )
    print(report)

    time = jnp.arange(original_targets.size) * 0.1
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(time, original_targets, label="Observed", linewidth=1.5)
    axes[0].plot(
        time, original_reservoir_prediction, label="Reservoir forecast", linewidth=1.2
    )
    axes[0].set(
        title=f"Held-out forecast ({args.horizon * 0.1:g} time units ahead)",
        xlabel="Test time (model time units)",
        ylabel="Mackey-Glass value",
    )
    axes[0].legend()
    state_image = axes[1].imshow(
        test_states.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0, test_states.shape[0] * 0.1, 0, test_states.shape[1]),
    )
    axes[1].set(
        title="Memristive reservoir activity",
        xlabel="Test time (model time units)",
        ylabel="Reservoir state",
    )
    figure.colorbar(state_image, ax=axes[1], label="Normalized conductance")
    axes[2].bar(memory_result.delays, memory_result.capacities, width=0.8)
    axes[2].set(
        title="Reconstruction of past random inputs",
        xlabel="Delay (samples)",
        ylabel="Squared correlation",
    )
    for axis in axes:
        axis.grid(alpha=0.2)

    report_output = args.report_output
    if report_output is None and args.output is not None:
        report_output = args.output.with_suffix(".md")
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(report, encoding="utf-8")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
