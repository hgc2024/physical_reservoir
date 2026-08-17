# Physical Reservoir

A modular research scaffold for differentiable physical reservoir computing built around JAX and Equinox, using the Mackey-Glass chaotic time series as its canonical dataset. The initial reservoir uses a fixed-step JAX integrator; Diffrax remains available for later adaptive-solver experiments.

This repository is intended as a starting point for studying how continuous-time physical substrates encode nonlinear dynamics and retain temporal information. The initial methodology is centered on forecasting a generated Mackey-Glass signal: drive a reservoir with the signal, collect its dynamic states, train a readout on those states, and evaluate predictions on a later, unseen portion of the trajectory.

## Dataset and methodology

The project uses a synthetically generated Mackey-Glass delay-differential time series rather than a downloaded dataset. The reference configuration is the standard chaotic regime (the implementation names these coefficients `a`, `b`, and `gamma`, respectively):

```text
beta = 0.2
gamma = 0.1
n = 10
tau = 17.0
dt = 0.1
```

Generating the data locally makes experiments reproducible and allows sequence length, initial history, sampling interval, and forecasting horizon to be controlled directly. The intended experimental pipeline is:

1. Generate one continuous Mackey-Glass trajectory and discard a configurable initial transient.
2. Inspect the unmodified signal using descriptive statistics, its value distribution, temporal traces, and autocorrelation. Produce a plain-language Markdown report that explains these checks, their modeling implications, and their limitations for readers without a physical-computing background.
3. Split the signal chronologically into training, validation, and test segments. Samples are never shuffled, so later observations cannot leak into earlier stages.
4. Fit standardization statistics on the training segment only, then apply the same transformation to validation and test data. Preserve the fitted statistics so predictions can be returned to the original scale.
5. Convert the ordered sequence into a causal forecasting task. The primary benchmark predicts 170 samples, or 17 model time units, ahead because this matches the Mackey-Glass delayed-feedback timescale. Lagged features are constructed from past and current values only.
6. Drive a 75-state volatile memristive reservoir with the standardized scalar signal. Each bounded state represents normalized conductive-filament growth with input/recurrent current, a boundary window, and thermal decay. Inputs are held constant over each `dt=0.1` interval and integrated with RK4; normalized conductance is recorded as the reservoir feature.
7. Discard an initial training washout, then align each state at time `t` with the target at `t + 170`. Fit only the ridge readout; the substrate weights remain fixed.
8. Choose ridge regularization on validation data. Compare the final reservoir forecast with persistence and a causal lagged-linear baseline on identical held-out test timestamps using MSE, RMSE, NRMSE, and R².
9. Evaluate linear memory capacity separately with an independent random input stream. This avoids inflating delayed-reconstruction scores with the Mackey-Glass signal's own autocorrelation.

Mackey-Glass forecasting is therefore the basis for the initial dataset API, reservoir interface, readout training, metrics, and benchmark design. Other datasets and task types are future extensions rather than part of the core methodology. See [`mackey_glass_data.md`](mackey_glass_data.md) for the generator parameters and broader dataset notes.

## Signal processing

The raw trajectory is not shuffled, smoothed, or frequency-filtered. Signal processing is limited to causal transformations needed to define a forecasting experiment:

- `make_forecast_pairs` forms scalar input/target pairs where the target is `horizon` samples after the input.
- `build_lagged_forecast` creates baseline feature vectors from current and earlier observations only. Feature order runs from the oldest lag to the current value.
- `align_states_and_targets` pairs each reservoir state with its future target and removes an optional initial washout.

For the primary benchmark, the horizon is 170 samples. With `dt=0.1`, this is 17 model time units and matches the delayed term in the Mackey-Glass generator. The lagged-linear baseline uses 10 observations spaced 10 samples apart. All methods are evaluated on the common subset of test timestamps left after applying their history requirements.

## Reservoir computing model

The initial physical substrate is a network of volatile memristive elements. Element `i` has a normalized conductive-filament state `w_i` constrained to `[0, 1]`:

```text
dw_i/dt = eta * I_i(t) * f(w_i) - w_i / tau_decay
f(w_i)  = 1 - (2*w_i - 1)^(2*p)
I_i(t)  = (V_input,i + V_recurrent,i) / (R_on*w_i + R_off*(1 - w_i))
```

The boundary window `f(w_i)` reduces driven growth near the physical limits, while `tau_decay` creates fading memory. A seeded input mask gives elements different responses to the same scalar signal, and a fixed random recurrent matrix couples their normalized conductances. Numerical states are projected back into `[0, 1]` after each integration interval.

The engine holds each input sample constant for one interval and integrates the state with fixed-step RK4 by default; Euler is also available for comparison. The recorded reservoir feature is normalized conductance, not the internal filament length itself. Only the linear readout is trained:

```text
prediction(t + horizon) = reservoir_state(t) @ readout_weights + bias
```

The readout is fitted with centered ridge regression. Regularization strength is selected using validation NRMSE, leaving substrate parameters and random weights unchanged during training.

## Evaluation design

Forecasts are converted back to the original Mackey-Glass scale before evaluation. The benchmark reports MSE, RMSE, NRMSE normalized by the test-target standard deviation, and R². Persistence and lagged-linear forecasts use the same final test timestamps as the reservoir.

Linear memory capacity is a separate substrate diagnostic. The reservoir is driven with a seeded independent uniform-random stream, and ridge readouts reconstruct inputs from 1 through 50 steps in the past. Each delay contributes its squared held-out correlation, and the reported capacity is their sum. Using an independent stream is important because measuring this quantity on the autocorrelated Mackey-Glass trajectory would confound reservoir memory with predictability already present in the input.

## What this project includes

- A package skeleton for physical reservoir substrates, engine logic, readouts, and metrics
- A JAX-native Mackey-Glass generator with validated chaotic-regime defaults
- Leakage-safe chronological splitting and training-only standardization
- Numerical and visual EDA for raw Mackey-Glass trajectories, with a plain-language Markdown report
- Causal horizon shifting and lagged-feature construction for forecasting
- A bounded volatile-memristor substrate and JAX Euler/RK4 reservoir engine
- A closed-form ridge readout, held-out forecast metrics, baselines, and linear memory capacity
- A Mackey-Glass-first methodology for nonlinear time-series forecasting
- Starter files for benchmarks and tests so the project can grow into a full experimental framework

## Current package layout

```text
physres/
├── datasets/
│   └── mackey_glass.py
├── preprocessing/
│   └── time_series.py
├── eda/
│   └── time_series.py
├── signal_processing/
│   └── forecasting.py
├── substrates/
│   ├── base.py
│   └── memristor.py
├── engine/
│   ├── solver.py
│   └── reservoir.py
├── readouts/
│   └── ridge.py
└── metrics/
    ├── forecasting.py
    └── memory_capacity.py

benchmarks/
├── mackey_glass_eda.py
└── mackey_glass.py
```

## Planned research directions

- Volatile memristor nanonetwork dynamics
- Photonic or optoelectronic delay-line reservoirs
- Ridge and differentiable readout training for Mackey-Glass forecasting
- Forecast-error, memory-capacity, and nonlinearity metrics
- Additional forecasting and classification datasets after the core benchmark is established

## Development setup

```bash
python -m pip install -e .
pytest -q
```

Run the Mackey-Glass EDA report and save its diagnostic figure. When `--output` is supplied, a Markdown report with the same filename is saved beside the image; use `--report-output` to choose another location.

```bash
python -m benchmarks.mackey_glass_eda --output artifacts/mackey_glass_eda.png
```

The report introduces the dataset, explains each statistic and preprocessing decision, summarizes the observed temporal structure, and distinguishes basic data-quality evidence from stronger claims such as proving chaotic behavior.

Run the end-to-end physical reservoir benchmark and save its evaluation report and figure:

```bash
python -m benchmarks.mackey_glass --output artifacts/mackey_glass_benchmark.png
```

As with the EDA command, supplying an image path also writes a Markdown report beside it.

## Initial results and discussion

On the initial held-out benchmark, the 75-state memristive reservoir predicts 17 model time units ahead with NRMSE `0.318` and R² `0.899`. On the same 490 test timestamps, the causal lagged-linear baseline reaches NRMSE `0.686`, while persistence reaches `1.590`. The reservoir's RMSE is 80% lower than persistence for this task. The forecast plot shows that it captures the broad test trajectory, though it smooths or mistimes some turning points.

This result supports a narrow conclusion: the implemented reservoir state contains useful predictive information for this Mackey-Glass trajectory at the generator's delayed-feedback horizon. Its improvement over a linear lagged baseline is consistent with a benefit from the reservoir's nonlinear transformation and fading dynamics. It does not show that the same advantage will hold across reservoir seeds, different trajectories, other horizons, or physical devices.

The separate linear memory-capacity score is `0.517` across delays 1–50, which is modest compared with the 75 available reservoir states. Forecast quality should therefore not be read as evidence of generally high linear memory. The benefit may be task-specific, combining limited memory with nonlinear dynamics that happen to suit this delayed forecast.

The evaluation is a direct teacher-forced forecast: the reservoir observes the true sequence through time `t` and predicts the value at `t + 170`. It is not an autonomous simulation that feeds predictions back as future inputs. The overlapping targets belong to one clean synthetic trajectory and are not independent replications.

Other limitations are equally important. The draft uses one trajectory, one random reservoir seed, one substrate type, and a phenomenological normalized memristor model. Ridge strength is validation-selected, but the substrate settings and baseline structure are exploratory fixed choices rather than the result of nested model selection. Fixed-step RK4 has not yet been compared with smaller integration steps or adaptive Diffrax solutions. The reported test result should therefore be treated as an initial implementation benchmark, not a confirmatory performance estimate or evidence of hardware fidelity.

The most informative follow-up is a robustness study across seeds, independently generated trajectories, forecast horizons, noise levels, and substrate parameters. Stronger nonlinear/autoregressive baselines, numerical-convergence tests, and autonomous multi-step forecasts are also needed before drawing broader conclusions.

## Next steps

1. Evaluate robustness across reservoir seeds, independent trajectories, and forecast horizons.
2. Add stronger nonlinear baselines and autonomous multi-step forecasting.
3. Compare the fixed-step engine with smaller steps and adaptive Diffrax integration.
4. Test noise and substrate-parameter sensitivity before adding photonic dynamics.
