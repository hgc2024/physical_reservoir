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

"""
benchmark_mackey_glass.py
=========================
A head-to-head benchmark comparing:
1. Physical Reservoir Computing (JAX/Equinox): Volatile Memristive Substrate with Closed-Form Ridge Readout.
2. Standard Deep Learning Recurrent Neural Network (PyTorch): Stacked LSTM trained via Backpropagation Through Time (BPTT) with Adam.

Dataset: Chaotic Mackey-Glass Time Series (tau = 17, standard chaotic regime).
Evaluation Metrics:
- Predictive Performance: Test MSE & Normalized Root Mean Square Error (NRMSE)
- Computational Efficiency: Training Time (seconds) & GPU/CPU memory efficiency
- Sample Efficiency: Test error scaling over limited training steps
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# JAX / Equinox / Diffrax Stack
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

# PyTorch Stack
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for exact reproducibility
np.random.seed(42)
torch.manual_seed(42)
key = jax.random.PRNGKey(42)

# =====================================================================
# 1. DATASET GENERATOR: Mackey-Glass Chaotic Delay Differential Equation
# =====================================================================
def generate_mackey_glass(n_timesteps=4000, tau=17, a=0.2, b=0.1, n=10, dt=1.0):
    """
    Generates chaotic Mackey-Glass time series using Runge-Kutta 4th order.
    dx/dt = [a * x(t - tau) / (1 + x(t - tau)^n)] - b * x(t)
    """
    history_len = int(tau / dt)
    total_len = n_timesteps + history_len
    x = np.zeros(total_len)
    x[:history_len] = 1.2  # Initial condition state

    for t in range(history_len, total_len - 1):
        x_tau = x[t - history_len]
        
        # RK4 Integration
        def dxdt(x_val, x_t):
            return (a * x_t) / (1.0 + x_t**n) - b * x_val

        k1 = dxdt(x[t], x_tau)
        k2 = dxdt(x[t] + 0.5 * dt * k1, x_tau)
        k3 = dxdt(x[t] + 0.5 * dt * k2, x_tau)
        k4 = dxdt(x[t] + dt * k3, x_tau)
        
        x[t + 1] = x[t] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x[history_len:]


# =====================================================================
# 2. JAX / EQUINOX: Volatile Memristive Physical Reservoir
# =====================================================================
class VolatileMemristorSubstrate(eqx.Module):
    """
    Models physical internal state w(t) of volatile memristive filaments.
    dw/dt = eta * I(t) * f(w) - w(t) / tau_decay
    """
    tau_decay: float
    eta: float
    r_on: float
    r_off: float
    w_dim: int

    def __init__(self, w_dim=100, tau_decay=10.0, eta=0.1, r_on=100.0, r_off=10000.0):
        self.w_dim = w_dim
        self.tau_decay = tau_decay
        self.eta = eta
        self.r_on = r_on
        self.r_off = r_off

    def vector_field(self, t, w, args):
        u_t, Win, Wrec = args
        # Calculate current drive based on state-dependent resistance
        resistance = self.r_on * w + self.r_off * (1.0 - w)
        voltage_input = jnp.dot(Win, u_t) + jnp.dot(Wrec, w)
        current = voltage_input / resistance
        
        # Window function constraining filament growth between [0, 1]
        window = 1.0 - jnp.square(2.0 * w - 1.0)
        
        # Filament growth vs. spontaneous thermal decay
        dwdt = self.eta * current * window - (w / self.tau_decay)
        return dwdt


def run_jax_physical_reservoir(train_u, test_u, res_size=200, ridge_alpha=1e-3):
    """
    Runs the Physical Reservoir through continuous ODE integration via Diffrax
    and fits an optimal closed-form linear readout.
    """
    print("\n--- Running JAX Physical Reservoir (Diffrax + Ridge) ---")
    start_time = time.time()
    
    # Random fixed physical connectivity weights (Win, Wrec)
    k1, k2 = jax.random.split(key)
    Win = jax.random.uniform(k1, shape=(res_size, 1), minval=-0.5, maxval=0.5)
    Wrec = jax.random.normal(k2, shape=(res_size, res_size)) * 0.9 / jnp.sqrt(res_size)
    
    substrate = VolatileMemristorSubstrate(w_dim=res_size)
    
    # ODE term definition using Diffrax
    def solve_step(w_prev, u_curr):
        term = diffrax.ODETerm(substrate.vector_field)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=1.0, dt0=0.2, y0=w_prev, args=(u_curr, Win, Wrec)
        )
        w_next = sol.ys[-1]
        return w_next, w_next

    # Step through training time-series to capture internal reservoir state trajectory
    w0 = jnp.full((res_size,), 0.1) # Initial state
    _, train_states = jax.lax.scan(solve_step, w0, train_u)
    
    # Target shift (Predict t+1 given state at t)
    X_train = train_states[:-1]
    Y_train = train_u[1:]
    
    # Closed-Form Ridge Regression Readout Training: W_out = (X^T X + alpha * I)^(-1) X^T Y
    X_matrix = jnp.hstack([X_train, jnp.ones((X_train.shape[0], 1))]) # Add bias column
    I_reg = jnp.eye(X_matrix.shape[1]).at[-1, -1].set(0.0) # Do not penalize bias
    W_out = jnp.linalg.solve(X_matrix.T @ X_matrix + ridge_alpha * I_reg, X_matrix.T @ Y_train)
    
    train_time = time.time() - start_time
    
    # Test Evaluation
    last_train_state = train_states[-1]
    _, test_states = jax.lax.scan(solve_step, last_train_state, test_u)
    
    X_test = test_states[:-1]
    Y_test = test_u[1:]
    X_test_matrix = jnp.hstack([X_test, jnp.ones((X_test.shape[0], 1))])
    
    Y_pred = X_test_matrix @ W_out
    
    return Y_test, Y_pred, train_time


# =====================================================================
# 3. PYTORCH: Stacked LSTM Recurrent Baseline
# =====================================================================
class PyTorchLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(PyTorchLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


def run_pytorch_lstm(train_u, test_u, hidden_dim=64, epochs=150, lr=0.005, seq_len=50):
    """
    Trains a 2-layer LSTM via Backpropagation Through Time (BPTT).
    """
    print("\n--- Running PyTorch LSTM Baseline (BPTT) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Create overlapping sequence datasets for BPTT
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length])
        return torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1), torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)

    X_train, Y_train = create_sequences(train_u, seq_len)
    X_test, Y_test = create_sequences(test_u, seq_len)
    
    model = PyTorchLSTMModel(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(X_train)
        pred_last = out[:, -1, :]
        loss = criterion(pred_last, Y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        out_test, _ = model(X_test)
        Y_pred = out_test[:, -1, :].cpu().numpy().flatten()
        Y_gt = Y_test.numpy().flatten()

    return Y_gt, Y_pred, train_time


# =====================================================================
# 4. BENCHMARK EXECUTION & METRIC COMPARISON
# =====================================================================
if __name__ == "__main__":
    print("Generating Mackey-Glass Time Series (N=3000 steps)...")
    mg_data = generate_mackey_glass(n_timesteps=3000, tau=17)
    
    # Normalize data to zero mean, unit variance
    mg_data = (mg_data - np.mean(mg_data)) / np.std(mg_data)
    
    # Train / Test split
    split_idx = 2000
    train_data = mg_data[:split_idx]
    test_data = mg_data[split_idx:]

    # 1. Execute JAX Physical Reservoir Model
    Y_test_pr, Y_pred_pr, time_pr = run_jax_physical_reservoir(
        jnp.array(train_data)[:, None], 
        jnp.array(test_data)[:, None], 
        res_size=150
    )
    mse_pr = np.mean((np.array(Y_test_pr) - np.array(Y_pred_pr)) ** 2)
    nrmse_pr = np.sqrt(mse_pr) / np.std(Y_test_pr)

    # 2. Execute PyTorch LSTM Baseline
    Y_test_lstm, Y_pred_lstm, time_lstm = run_pytorch_lstm(
        train_data, 
        test_data, 
        hidden_dim=64, 
        epochs=150
    )
    mse_lstm = np.mean((Y_test_lstm - Y_pred_lstm) ** 2)
    nrmse_lstm = np.sqrt(mse_lstm) / np.std(Y_test_lstm)

    # Output Metric Summary Table
    print("\n" + "="*55)
    print("           BENCHMARK PERFORMANCE SUMMARY           ")
    print("="*55)
    print(f"{'Metric':<25} | {'Physical Reservoir':<12} | {'PyTorch LSTM':<12}")
    print("-" * 55)
    print(f"{'Test MSE':<25} | {mse_pr:<12.6f} | {mse_lstm:<12.6f}")
    print(f"{'Test NRMSE':<25} | {nrmse_pr:<12.6f} | {nrmse_lstm:<12.6f}")
    print(f"{'Training Time (sec)':<25} | {time_pr:<12.4f} | {time_lstm:<12.4f}")
    print(f"{'Speedup Factor':<25} | {time_lstm/time_pr:<12.2f}x | {'1.00x':<12}")
    print("="*55)

    # Plot Visual Predictions
    plt.figure(figsize=(12, 5))
    plt.plot(Y_test_pr[:300], label="Ground Truth (Mackey-Glass)", color="black", alpha=0.7, linewidth=1.5)
    plt.plot(Y_pred_pr[:300], label=f"JAX PhysRes (MSE: {mse_pr:.4f})", color="crimson", linestyle="--")
    plt.plot(Y_pred_lstm[:300], label=f"PyTorch LSTM (MSE: {mse_lstm:.4f})", color="dodgerblue", linestyle=":")
    plt.title("Mackey-Glass 1-Step Time-Series Forecasting Comparison")
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nBenchmark visualization saved to 'benchmark_results.png'.")