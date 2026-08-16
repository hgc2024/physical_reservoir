# Physical Reservoir

A modular research scaffold for differentiable physical reservoir computing built around JAX, Equinox, and Diffrax, using the Mackey-Glass chaotic time series as its canonical dataset.

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
5. Use the standardized signal to drive a simulated physical reservoir and record its state trajectory, allowing a washout period before fitting.
6. Fit a readout, initially ridge regression, to predict future Mackey-Glass values from the reservoir states. Forecast windowing and other signal transformations will be introduced as a separate signal-processing layer.
7. Select settings on the validation segment and report final forecasting error once on the held-out test segment, alongside memory-capacity and reservoir-dynamics diagnostics.

Mackey-Glass forecasting is therefore the basis for the initial dataset API, reservoir interface, readout training, metrics, and benchmark design. Other datasets and task types are future extensions rather than part of the core methodology. See [`mackey_glass_data.md`](mackey_glass_data.md) for the generator parameters and broader dataset notes.

## What this project includes

- A package skeleton for physical reservoir substrates, engine logic, readouts, and metrics
- A JAX-native Mackey-Glass generator with validated chaotic-regime defaults
- Leakage-safe chronological splitting and training-only standardization
- Numerical and visual EDA for raw Mackey-Glass trajectories, with a plain-language Markdown report
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
├── substrates/
│   └── base.py
├── engine/
│   ├── solver.py
│   └── reservoir.py
├── readouts/
└── metrics/

benchmarks/
└── mackey_glass_eda.py
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

## Next steps

1. Add forecasting-window and signal-processing utilities without coupling them to dataset generation or EDA.
2. Implement the core substrate interfaces and ODE integration wrappers with Diffrax.
3. Train a ridge readout on reservoir states and evaluate held-out forecasts.
4. Add memory-capacity diagnostics and benchmark visualizations.
