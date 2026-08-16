Four tiers of datasets provide standard benchmarks for physical reservoir computing, progressing from clean mathematical chaos to temporal event streams.

---

### Tier 1: Continuous Chaos & Memory Capacity (Mathematical Benchmarks)

#### 1. Mackey-Glass Time-Series System

* **Primary Use Case:** Evaluates long-range dynamic forecasting and non-linear trajectory tracking.
* **Parameter Settings:**
* $\beta = 0.2, \gamma = 0.1, n = 10, \tau = 17.0$ (standard chaotic regime).


* **Quick Access (No Download Required):** Generate on-the-fly using `nolitsa` or native numerical ODE solvers in JAX.

```python
import jax.numpy as jnp

def generate_mackey_glass(n_steps=5000, tau=17.0, dt=0.1, a=0.2, b=0.1, gamma=10.0):
    # Generates chaotic time series via discrete RK4 or Euler integration
    history_length = int(tau / dt)
    x = jnp.zeros(n_steps + history_length)
    x = x.at[:history_length].set(1.2) # Initial condition
    
    for t in range(history_length, len(x) - 1):
        x_tau = x[t - history_length]
        dxdt = a * x_tau / (1.0 + jnp.power(x_tau, gamma)) - b * x[t]
        x = x.at[t + 1].set(x[t] + dxdt * dt)
        
    return x[history_length:]

```

#### 2. Lorenz 3D Attractor

* **Primary Use Case:** Multi-variable state-space reconstruction (input-driven vector fields).
* **Source & Generator:** Standard SciPy ODE Integrator or JAX `diffrax.ODETerm`.
* **Standard Equations:**

$$\frac{dx}{dt} = \sigma (y - x), \quad \frac{dy}{dt} = x (\rho - z) - y, \quad \frac{dz}{dt} = x y - \beta z$$



*(Parameters: $\sigma = 10.0, \rho = 28.0, \beta = 8/3$)*

---

### Tier 2: Spatiotemporal & Sequential Audio Signals

#### 3. Free Spoken Digit Dataset (FSDD)

* **Primary Use Case:** Low-latency sequence classification and temporal pattern separation.
* **Details:** 3,000 audio recordings (8kHz WAV files) of digits 0–9 spoken by 6 speakers.
* **Source:** [GitHub - jakobovski/free-spoken-digit-dataset](https://github.com/jakobovski/free-spoken-digit-dataset)

```bash
git clone https://github.com/jakobovski/free-spoken-digit-dataset.git

```

* **Python Loading:**

```python
import torchaudio
import glob

files = glob.glob("free-spoken-digit-dataset/recordings/*.wav")
waveform, sample_rate = torchaudio.load(files[0])

```

#### 4. Japanese Vowels Dataset (UCI Machine Learning Repository)

* **Primary Use Case:** Standard temporal classification benchmark for Echo State Networks (ESNs) and memristive substrates.
* **Details:** 9 male speakers uttering two Japanese vowels (`/ae/`) sequentially. Feature space consists of 12 LPC cepstrum coefficients over time.
* **Download Access:** Available directly via `scikit-learn` or openml:

```python
from sklearn.datasets import fetch_openml
japanese_vowels = fetch_openml(data_id=1240, as_frame=False)

```

---

### Tier 3: Asynchronous Event-Based Spiking Streams (Neuromorphic & DVS)

#### 5. IBM DVS128 Gesture Dataset

* **Primary Use Case:** Event-driven dynamic classification using neuromorphic sensors and volatile physical reservoirs.
* **Details:** 11 hand gesture classes captured with a Dynamic Vision Sensor (128x128 resolution, microsecond event temporal resolution) under varying illumination conditions.
* **Python Access via Tonic Library:**

```python
import tonic

dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)
events, label = dataset[0]  # Array of (t, x, y, p) tuples

```

#### 6. N-MNIST (Neuromorphic-MNIST)

* **Primary Use Case:** Benchmarking asynchronous event-stream mapping on physical hardware simulators.
* **Details:** Created by mounting an ATIS event camera on a motorized pan-tilt head and scanning original MNIST handwritten images across 3 saccades.
* **Python Access via Tonic:**

```python
import tonic

nmnist_train = tonic.datasets.NMNIST(save_to='./data', train=True)
events, label = nmnist_train[0]

```

---

### Tier 4: Real-World Physical & Industrial Telemetry

#### 7. NASA Prognostics Data Repository (C-MAPSS Turbofan Engine)

* **Primary Use Case:** Evaluating degradation, remaining useful life (RUL), and physical state tracking under continuous noisy sensor streams.
* **Details:** Simulated run-to-failure sensor records (temperature, pressure, fan speeds) under changing operating conditions.
* **Download Access:** [NASA Prognostics Data Center](https://www.google.com/search?q=https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/padd/prognostics-center-of-excellence-data-repository/)

---

### Suggested Directory Pipeline Strategy

Create a dedicated data loader directory inside your repository to mirror the architecture:

```text
physres_jax/
└── physres/
    └── datasets/
        ├── __init__.py
        ├── mackey_glass.py     <-- Synthetic Generator (JAX)
        ├── fsdd.py             <-- FSDD WAV to Spectrogram/Spike Converter
        └── tonic_loaders.py    <-- DVS Gesture & N-MNIST Event Stream Wrappers

```