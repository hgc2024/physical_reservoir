# Mackey-Glass reservoir benchmark

## Evaluation question

Can a simulated volatile memristive reservoir use the observed Mackey-Glass sequence to predict its value `170` samples (`17` model time units) into the future? This horizon matches the delayed-feedback timescale used to generate the data.

## Signal processing and evaluation protocol

1. Generate one trajectory, remove its startup transient, and split it chronologically into training, validation, and test segments.
2. Standardize every segment using only the training mean and standard deviation.
3. Feed the standardized scalar sequence into a `75`-state memristive reservoir. Discard the first `100` training states as washout.
4. Pair each reservoir state at time `t` with the signal at `t + 170`.
5. Choose ridge regularization on validation data only. Evaluate once on `490` common test timestamps.
6. Compare with persistence (reuse the current value) and a causal linear baseline using 10 past values spaced 10 samples apart.

Keeping all splits in time order matters because nearby Mackey-Glass values are strongly related. Random splitting would allow closely related future observations into training and overstate forecasting quality.

## Validation selection

- **Reservoir ridge alpha:** `1e-06` (validation NRMSE `0.203`)
- **Lagged-linear ridge alpha:** `10` (validation NRMSE `0.465`)

## Held-out test results

Lower MSE, RMSE, and NRMSE are better. NRMSE measures error relative to the natural variation in the test target. An R² of `1` is perfect; `0` is equivalent to always predicting the test mean.

| Method | MSE | RMSE | NRMSE | R² |
|---|---:|---:|---:|---:|
| Memristive reservoir | 0.004132 | 0.0643 | 0.318 | 0.899 |
| Lagged linear baseline | 0.019220 | 0.1386 | 0.686 | 0.530 |
| Persistence baseline | 0.103443 | 0.3216 | 1.590 | -1.529 |

The reservoir RMSE is `80.0%` lower than persistence at this forecast horizon. This is a comparative benchmark result, not yet a general claim about other trajectories, seeds, substrates, or forecasting horizons.

## Fading-memory evaluation

Linear memory capacity was evaluated separately with an independent random input, because the strong autocorrelation of Mackey-Glass data would otherwise inflate delayed-reconstruction scores. The summed capacity over delays 1 through 50 is **0.517**. Individual delay scores are shown in the figure.

## Discussion

The held-out result shows that the reservoir contains useful information for this specific delayed forecast. It follows the broad trajectory more closely than either baseline and reduces RMSE by `80.0%` relative to persistence. Its advantage over the lagged-linear model suggests that the fixed nonlinear state transformation contributes information beyond the selected raw lags.

The memory-capacity result adds an important qualification. A summed linear capacity of **0.517** across 50 tested delays is modest relative to the `75` available states. The forecasting improvement therefore should not be interpreted as evidence of uniformly high linear memory. It may instead reflect a task-specific combination of fading memory, nonlinear response, and the match between the chosen forecast horizon and Mackey-Glass feedback delay.

This is a direct, teacher-forced forecast: the reservoir observes the real sequence through time `t` and predicts one value at `t + horizon`. It is not an autonomous rollout in which earlier predictions are fed back as later inputs. Overlapping test targets also come from one continuous synthetic trajectory, so they should not be treated as independent experimental replications.

Several limits remain. The benchmark uses one generated trajectory, one reservoir seed, one forecast horizon, clean simulated data, and one phenomenological memristor model. Only ridge regularization is selected automatically on validation data; substrate settings and baseline structure are exploratory fixed choices and have not been subjected to nested model selection. The fixed-step RK4 solution has not yet been checked against smaller steps or adaptive integration. These results are therefore an initial implementation benchmark rather than a confirmatory estimate of general performance or physical-hardware behavior.

The next useful tests are repeated seeds and trajectories, horizon sweeps, stronger nonlinear and autoregressive baselines, integration-convergence checks, noise and parameter perturbations, and autonomous multi-step forecasting. Those experiments will show which conclusions survive beyond the favorable configuration used for this draft.
