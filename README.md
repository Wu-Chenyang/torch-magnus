# torch-linode: A PyTorch Solver for Linear ODEs

[![PyPI version](https://badge.fury.io/py/torch-linode.svg)](https://badge.fury.io/py/torch-linode)
[![Tests](https://github.com/Wu-Chenyang/torch-linode/actions/workflows/ci.yml/badge.svg)](https://github.com/Wu-Chenyang/torch-linode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`torch-linode` is a specialized PyTorch library for the efficient and differentiable batch solving of **linear ordinary differential equations (ODEs)**. It is designed to solve systems of the form:

- **Homogeneous**: `dy/dt = A(t)y(t)`
- **Non-homogeneous**: `dy/dt = A(t)y(t) + g(t)`

Leveraging high-order integrators and a memory-efficient adjoint method for backpropagation, this library is particularly well-suited for problems in physics, control theory, and for implementing "Neural ODE" models with linear dynamics.

## Key Features

- **Specialized for Linear ODEs**: Optimized for homogeneous and non-homogeneous linear systems.
- **Powerful Batch Processing**: Natively handles broadcasting between batches of systems (`A(t)`, `g(t)`) and batches of initial conditions (`y0`).
- **High-Order Integrators**: Includes 2nd, 4th, and 6th-order Magnus integrators and a generic `Collocation` solver (e.g., Gauss-Legendre).
- **Fully Differentiable**: Gradients can be backpropagated through the solvers, making it ideal for training.
- **Dense Output**: Provides continuous solutions for evaluation at any time point.
- **GPU Support**: Runs seamlessly on CUDA-enabled devices.

## Installation

```bash
pip install torch-linode
```

## Core API

This library provides two main solver functions: `odeint` and `odeint_adjoint`.

- **`odeint`**: The standard solver. Use this for simple forward passes (inference) where gradients are not required.
- **`odeint_adjoint`**: A memory-efficient solver for training. It uses the adjoint sensitivity method to compute gradients, which uses significantly less memory than storing the entire computation graph. **Always prefer `odeint_adjoint` for training.**

### Defining The System: Two Approaches

You can define your linear system in two ways:

#### Approach 1: Using `nn.Module` (Recommended)

This is the most flexible and standard method. The solver automatically detects if the system is homogeneous or non-homogeneous.

- **For Homogeneous Systems (`dy/dt = Ay`)**: Implement the `forward(self, t)` method to return your matrix `A(t)`.
- **For Non-Homogeneous Systems (`dy/dt = Ay + g`)**: Implement `forward(self, t)` for `A(t)` and add a `g(self, t)` method for the forcing term `g(t)`.

```python
import torch.nn as nn

class MyNonHomogeneousSystem(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A # Shape: (*batch_A, dim, dim)

    def forward(self, t):
        # Must return A(t) with shape (*batch_A, *t.shape, dim, dim)
        t_shape = t.shape
        A_view = self.A.view(
            *self.A.shape[:1],
            *((1,) * len(t_shape)),
            *self.A.shape[1:]
        )
        return A_view.expand(
            *self.A.shape[:1],
            *t_shape,
            *self.A.shape[1:]
        )

    def g(self, t):
        # Must return g(t) with shape (*batch_g, *t.shape, dim)
        return torch.sin(t).unsqueeze(-1).expand(*t.shape, 2)
```

#### Approach 2: Using a Plain Function

For simple systems without internal state, you can use a plain function with the signature `system_func(t, params)`.

```python
# A is passed via the `params` argument
def functional_system(t, params):
    A = params
    t_shape = t.shape
    A_view = A.view(*((1,) * len(t_shape)), *A.shape)
    return A_view.expand(*t_shape, *A.shape)
```

### Key API Rules (Common Pitfalls)

- **Strict `forward(self, t)` Signature**: The `forward` method of your `nn.Module` **must** accept only `t` (time) as an argument. The solver handles the `y` variable internally.
- **Shape Expansion is Critical**: Your returned `A(t)` and `g(t)` tensors must be explicitly expanded to match the shape of the input time tensor `t`.
- **Automatic Broadcasting**: The solver automatically broadcasts the batch dimensions of your system (`A(t)`, `g(t)`) and your initial conditions (`y0`). Ensure your batch dimensions are compatible.

## Complete Example: Learning an Unknown System

This example demonstrates the core power of `torch-linode`: learning the parameters of an unknown dynamical system from observed data using `odeint_adjoint`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_linode.solvers import odeint, odeint_adjoint

# 1. Define the learnable system and a ground truth system
class LearnableLinearODE(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim))

    def forward(self, t):
        t_shape = t.shape
        A_view = self.A.view(*((1,) * len(t_shape)), *self.A.shape)
        return A_view.expand(*t_shape, *self.A.shape)

class GroundTruthODE(nn.Module):
    def __init__(self, A_true):
        super().__init__()
        self.A = A_true

    def forward(self, t):
        t_shape = t.shape
        A_view = self.A.view(*((1,) * len(t_shape)), *self.A.shape)
        return A_view.expand(*t_shape, *self.A.shape)

# 2. Generate ground truth data from the true system
print("--- Generating ground truth data ---")
A_true = torch.tensor([[-0.1, -1.0], [1.0, -0.1]]) # Damped spiral
y0 = torch.tensor([2.0, 0.0])
t_span = torch.linspace(0, 10, 100)
true_system = GroundTruthODE(A_true)
with torch.no_grad():
    # Use standard odeint for inference
    y_true = odeint(true_system, y0, t_span)

# 3. Set up and train the learnable model
print("--- Training the learnable model ---")
model = LearnableLinearODE(dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize parameters near the true solution to ensure convergence
with torch.no_grad():
    model.A.data = A_true + torch.randn_like(A_true) * 0.3

for epoch in range(1000):
    optimizer.zero_grad()
    # Use odeint_adjoint for memory-efficient training
    y_pred = odeint_adjoint(model, y0, t_span)
    loss = nn.MSELoss()(y_pred, y_true)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

print(f"Learned Matrix A:\n{model.A.data}")
print(f"True Matrix A:\n{A_true}")

# 4. Visualize the results
with torch.no_grad():
    y_pred_final = odeint(model, y0, t_span)

plt.figure(figsize=(8, 4))
plt.plot(y_true[:, 0], y_true[:, 1], 'g-', label='Ground Truth', linewidth=2)
plt.plot(y_pred_final[:, 0], y_pred_final[:, 1], 'b--', label='Learned Trajectory')
plt.title("Phase Portrait: Learning an ODE System")
plt.xlabel("State 1")
plt.ylabel("State 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
