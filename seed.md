An agentic IDE (Cursor, Windsurf, Claude Dev/Roo Code, Devin) works best when given explicit structural boundaries, clean mathematical declarations, typed schema hints, and step-by-step roadmap prompts.

Save the code below as **`README.md`** at the root of your new repository to seed your physical reservoir engine project.

```markdown
# PhysRes-JAX: Differentiable Physical Reservoir Computing Engine

A modular, GPU-accelerated framework for simulating, benchmarking, and training **Physical Reservoir Computing (PRC)** substrates using **JAX**, **Equinox**, and **Diffrax**.

This repository models continuous-time non-linear physical systems (volatile memristive nanonetworks and photonic delay lines) as high-dimensional dynamical reservoirs with trainable linear/ridge readouts and differentiable hardware surrogate models.

---

## 1. System Architecture & Component Mapping


```

```
                           ┌─────────────────────────────────────────┐
                           │           Input Signal u(t)             │
                           └────────────────────┬────────────────────┘
                                                │
                                                ▼

```

┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Physical Substrate Simulation (Diffrax ODE Integration)                                        │
│                                                                                                │
│   \dot{x}(t) = -\frac{1}{\tau} x(t) + f_{non-linear}(W_{in} u(t) + W_{rec} x(t) + \xi_{noise}) │
│                                                                                                │
│   ┌─────────────────────────────────────────┐   ┌──────────────────────────────────────────┐   │
│   │   Substrate A: Volatile Memristor       │   │     Substrate B: Optoelectronic Loop     │   │
│   │   (Filament Growth & Thermal Decay)     │   │     (Ikeda Delay & Phase Shift)          │   │
│   └─────────────────────────────────────────┘   └──────────────────────────────────────────┘   │
└───────────────────────────────────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│       Substrate State Trajectory        │
│          X = [x(t_1), ..., x(t_N)]      │
└────────────────────┬────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Analysis & Readout Pipeline                                                                    │
│                                                                                                │
│   ┌─────────────────────────────────────────┐   ┌──────────────────────────────────────────┐   │
│   │          Readout Layer y(t)             │   │    Physical Capacity Benchmarks          │   │
│   │    y(t) = W_{out} X(t) + b_{out}        │   │    (Linear Memory Capacity MC,         │   │
│   │    (Ridge Reg. / Differentiable Adam)   │   │     Non-linear Separation Metric)       │   │
│   └─────────────────────────────────────────┘   └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Core Physics Formulations

### Substrate 1: Volatile Memristive Nanonetwork Dynamics
The internal state variable $w(t) \in [0, 1]$ represents the normalized conductive filament length governed by current-driven ion transport and spontaneous thermal dissolution:

$$\frac{dw(t)}{dt} = \eta \cdot I(t) \cdot f(w) - \frac{w(t)}{\tau_{decay}}$$

Where:
* $I(t) = \frac{V_{in}(t) + V_{rec}(t)}{R_{on} w(t) + R_{off}(1 - w(t))}$
* $f(w) = 1 - (2w - 1)^{2p}$ is a window function constraining filament growth.
* $\tau_{decay}$ controls the fading memory dynamics (relaxation timescale).

### Substrate 2: Photonic/Optoelectronic Delay Line Dynamics
The optical phase/intensity state $x(t)$ in a feedback-loop system is governed by the Ikeda delay differential equation:

$$\tau \frac{dx(t)}{dt} = -x(t) + \beta \sin^2\left(x(t - \tau_d) + \gamma u(t) + \phi_0\right)$$

Where:
* $\tau$ is the system response time.
* $\tau_d$ is the optical feedback loop time delay.
* $\beta$ is the non-linear gain factor, and $\gamma$ is the input coupling strength.

---

## 3. Project Structure & Module Plan

```text
physres_jax/
├── README.md                          <-- You are here
├── pyproject.toml                     <-- JAX, Equinox, Diffrax, Jaxtyping dependencies
├── physres/
│   ├── __init__.py
│   ├── substrates/                    <-- ODE & State Variable Definitions
│   │   ├── base.py                    <-- Abstract Base Substrate (PyTree)
│   │   ├── memristor.py               <-- Volatile Memristor Nanonetwork
│   │   └── photonic.py                <-- Ikeda Optoelectronic Loop
│   ├── engine/                        <-- Diffrax Integration Drivers
│   │   ├── solver.py                  <-- Adaptive ODE Solver wrappers
│   │   └── reservoir.py               <-- Continuous-time Physical Reservoir module
│   ├── readouts/                      <-- State Readout & Linear Mapping
│   │   ├── ridge.py                   <-- Fast Closed-form Ridge Regression
│   │   └── differentiable.py          <-- Adam-optimized Readout (PyTorch/JAX style)
│   └── metrics/                       <-- Physical Metrics & Capacity Measurement
│       ├── memory_capacity.py         <-- Linear Memory Capacity (MC) evaluation
│       └── nonlinearity.py            <-- State Separation & Rank Metrics
├── benchmarks/                        <-- Evaluation Scripts
│   ├── mackey_glass.py                <-- Chaotic Time-Series Forecasting
│   ├── lorenz.py                      <-- Attractor Reconstruction
│   └── gesture_dvs.py                 <-- Event-based Signal Classification
└── tests/
    ├── test_substrates.py             <-- ODE gradient and state bounds tests
    └── test_reservoir.py              <-- End-to-end integration tests

```

---

## 4. Key Dependencies

* **`jax`** & **`jaxlib`**: Accelerate linear algebra and dynamic parallelization across sub-nodes via `vmap` and `jit`.
* **`equinox`**: Neural network and dynamic model definition using pure JAX PyTrees.
* **`diffrax`**: Numerical differential equation solvers (Dormand-Prince, Tsitouras) with reverse-mode automatic differentiation.
* **`jaxtyping`**: Type annotations for array shapes, numerical types, and state dimensions.
* **`optax`**: Optimization algorithms for differentiable surrogate tuning and end-to-end input mask optimization.
* **`scikit-learn`** & **`matplotlib`**: Baseline ridge regression benchmarks and phase-space state trajectory visualization.

---

## 5. Agentic Implementation Roadmap (Prompt Sequence for IDE Agents)

Use the following prompts sequentially with your agentic IDE (e.g., Cursor Composer, Roo Code, or Windsurf) to auto-generate and verify each layer of the repository.

### **Phase 1: Environment & Base Interfaces**

> **Agent Prompt 1:**
> "Read `README.md`. Create `pyproject.toml` using `hatchling` or standard `setuptools` with dependencies for `jax`, `jaxlib`, `equinox`, `diffrax`, `jaxtyping`, `optax`, `scikit-learn`, `matplotlib`, and `pytest`. Create an abstract base class `AbstractSubstrate` in `physres/substrates/base.py` inheriting from `equinox.Module`. Define abstract methods `drift(self, t, x, args)` and `output_map(self, x)` with full `jaxtyping` annotations for Float arrays."

---

### **Phase 2: Memristor Substrate & Diffrax Engine**

> **Agent Prompt 2:**
> "Implement `MemristorNanonetwork` in `physres/substrates/memristor.py` inheriting from `AbstractSubstrate`. Implement the volatile memristor filament ODE mechanics described in Section 2 of `README.md`. Use `equinox.field(static=True)` for non-differentiable structural hyperparameters. Then, create `physres/engine/reservoir.py` using `diffrax.diffeqsolve`, `diffrax.ODETerm`, and `diffrax.Tsit5` to integrate state trajectories over a discrete time sequence $u(t)$."

---

### **Phase 3: Photonic Substrate & Parallelization**

> **Agent Prompt 3:**
> "Implement the Ikeda delay optoelectronic substrate in `physres/substrates/photonic.py`. Ensure state variable vectorization works seamlessly with `jax.vmap` across batch dimensions. Add a `test_substrates.py` file in `tests/` verifying that `diffrax` integration completes without NaNs or unstable state divergence for random input sequences."

---

### **Phase 4: Memory Capacity & Metrics Pipeline**

> **Agent Prompt 4:**
> "Implement `calculate_linear_memory_capacity` in `physres/metrics/memory_capacity.py`. The function should accept a reservoir state matrix $X \in \mathbb{R}^{N \times D}$ and input stream $u \in \mathbb{R}^N$, compute ridge regression $k$-step delay reconstruction metrics $r^2(k)$ up to maximum delay $k_{max}$, and return the total Linear Memory Capacity $MC = \sum r^2(k)$."

---

### **Phase 5: End-to-End Benchmark Execution**

> **Agent Prompt 5:**
> "Create `benchmarks/mackey_glass.py`. Generate a chaotic Mackey-Glass time-series dataset. Run the dataset through `MemristorNanonetwork` integrated via `diffrax`, train the readout layer in `physres/readouts/ridge.py`, and print test MSE and Linear Memory Capacity. Plot the ground truth versus predicted trajectory alongside the reservoir's dynamic state activity heatmaps."

---

## 6. Execution & Verification Commands

```bash
# Install package in editable mode
pip install -e .

# Run test suite to verify ODE solvers and substrates
pytest tests/ -v

# Run the Mackey-Glass physical benchmark
python benchmarks/mackey_glass.py

```

```

<FollowUp label="Ready to start generating code for Phase 1?" query="Generate the `pyproject.toml` and `physres/substrates/base.py` files for Phase 1 of the PhysRes-JAX framework."/>

```