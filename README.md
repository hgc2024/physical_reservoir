# Physical Reservoir

A modular research scaffold for differentiable physical reservoir computing built around JAX, Equinox, and Diffrax.

This repository is intended as a starting point for simulating continuous-time physical substrates, benchmarking reservoir dynamics, and training readout layers for time-series tasks.

## What this project includes

- A package skeleton for physical reservoir substrates, engine logic, readouts, and metrics
- A documentation structure inspired by the reservoir-computing formulation in the seed note
- Starter files for benchmarks and tests so the project can grow into a full experimental framework

## Proposed architecture

```text
physres/
├── substrates/
│   ├── base.py
│   ├── memristor.py
│   └── photonic.py
├── engine/
│   ├── solver.py
│   └── reservoir.py
├── readouts/
│   ├── ridge.py
│   └── differentiable.py
└── metrics/
    ├── memory_capacity.py
    └── nonlinearity.py
```

## Planned research directions

- Volatile memristor nanonetwork dynamics
- Photonic or optoelectronic delay-line reservoirs
- Differentiable readout training
- Memory-capacity and nonlinearity metrics
- Benchmark scripts for forecasting and classification tasks

## Development setup

```bash
python -m pip install -e .
pytest -q
```

## Next steps

1. Implement the core substrate interfaces.
2. Add ODE integration wrappers with Diffrax.
3. Introduce readout training and metric evaluation.
4. Add benchmark scripts and visualization utilities.
