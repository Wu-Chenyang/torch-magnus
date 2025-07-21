# torch-magnus

[![PyPI version](https://badge.fury.io/py/torch-magnus.svg)](https://badge.fury.io/py/torch-magnus)
[![Tests](https://github.com/Wu-Chenyang/torch-magnus/actions/workflows/ci.yml/badge.svg)](https://github.com/Wu-Chenyang/torch-magnus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`torch_magnus` is a specialized PyTorch-based library for the efficient **batch solving of homogeneous linear ordinary differential equations (ODEs)** of the form `dy/dt = A(t)y`. It leverages Magnus-type integrators to provide high-precision, differentiable, and GPU-accelerated solutions.

This library is particularly well-suited for problems in quantum mechanics, control theory, and other areas of physics and engineering where such ODEs are common.

## Key Features

- **Batch Processing**: Natively handles batches of initial conditions and parameters for massive parallelization. A single call can solve thousands of ODEs simultaneously.
- **High-Order Integrators**: Includes 2nd and 4th-order Magnus integrators.
- **Adaptive Stepping**: Automatically adjusts step size to meet specified error tolerances (`rtol`, `atol`), even across batches.
- **Differentiable**: Gradients can be backpropagated through the solvers using a memory-efficient adjoint method.
- **Dense Output**: Provides continuous solutions for evaluation at any time point within the integration interval.
- **GPU Support**: Runs seamlessly on CUDA-enabled devices for significant performance gains.

## Installation

You can install the package directly from PyPI (once published):

```bash
pip install torch-magnus
```

Or, for development, clone this repository and install in editable mode. **Note**: Use quotes around `.[dev]` to prevent shell expansion issues.

```bash
git clone https://github.com/Wu-Chenyang/torch_magnus.git
cd torch_magnus
pip install -e ".[dev]"
```

## API and Usage

The primary functions are `magnus_odeint` and `magnus_odeint_adjoint`.

```python
magnus_odeint(
    A_func_or_module: Union[Callable, nn.Module], 
    y0: Tensor, 
    t: Union[Sequence[float], torch.Tensor],
    params: Tensor = None,
    order: int = 4, 
    rtol: float = 1e-6, 
    atol: float = 1e-8
) -> Tensor
```

### Parameters

- `A_func_or_module`: The function or `nn.Module` that defines the matrix `A(t)`.
  - **As a function**: It should have the signature `A(t, params)`. For batch solving, it must return a tensor of shape `(*batch_shape, dim, dim)` when `t` is a scalar, and `(*batch_shape, num_times, dim, dim)` when `t` is a vector (as used internally by the adjoint method).
  - **As an `nn.Module`**: The module's `forward` method should accept `t` as input. Parameters are handled automatically.
- `y0`: A tensor of initial conditions with shape `(*batch_shape, dim)`. The `batch_shape` determines the number of ODEs to solve in parallel.
- `t`: A 1D tensor or sequence of time points `(t_0, t_1, ..., t_N-1)` at which to evaluate the solution.
- `params`: Optional tensor of parameters to be passed to `A_func`.
- `order`: The order of the Magnus integrator to use (2 or 4).
- `rtol`, `atol`: Relative and absolute tolerances for the adaptive step-size controller.

### Returns

A tensor of shape `(*batch_shape, N, dim)` containing the solution trajectories for each ODE in the batch at each time point in `t`.

---

`magnus_odeint_adjoint` has the same signature but uses a more memory-efficient method for computing gradients, making it ideal for training and optimization.

## Example: Batch Solving and Parameter Learning

This example demonstrates how to solve a batch of ODEs simultaneously and use the adjoint method to learn the parameters of the system.

We will solve a batch of `(2, 3)` simple harmonic oscillators, each with a different initial condition, and learn the oscillation frequency `w`.

```python
import torch
import torch.nn as nn
from torch_magnus import magnus_odeint, magnus_odeint_adjoint

# 1. Define the problem for a batch of systems
dim = 2
batch_shape = (2, 3)  # Solve 6 ODEs at once
dtype = torch.float64

# Batch of initial conditions
y0 = torch.randn(*batch_shape, dim, dtype=dtype)
t_span = torch.linspace(0, 2 * torch.pi, 20, dtype=dtype)

# Define the true system to generate target data
with torch.no_grad():
    true_w = 1.5
    def A_target_func(t, params):
        A_base = torch.tensor([[0., true_w], [-true_w, 0.]], dtype=dtype)
        # Correctly broadcast for both scalar and vector t
        if isinstance(t, torch.Tensor) and t.ndim == 1:
            # Case for quadrature: t is a vector of shape (e.g., 15)
            return A_base.view(1, 1, dim, dim).expand(*batch_shape, t.shape[0], -1, -1)
        else:
            # Case for forward pass: t is a scalar
            return A_base.unsqueeze(0).expand(*batch_shape, -1, -1)

    y_target = magnus_odeint(A_target_func, y0, t_span)

# 2. Create a learnable system
w = nn.Parameter(torch.tensor(1.0, dtype=dtype))

def A_learnable_func(t, params):
    A_matrix = torch.zeros(dim, dim, dtype=dtype)
    A_matrix[0, 1] = params
    A_matrix[1, 0] = -params
    # Correctly broadcast for both scalar and vector t
    if isinstance(t, torch.Tensor) and t.ndim == 1:
        return A_matrix.view(1, 1, dim, dim).expand(*batch_shape, t.shape[0], -1, -1)
    else:
        return A_matrix.unsqueeze(0).expand(*batch_shape, -1, -1)

# 3. Set up the optimization loop
optimizer = torch.optim.Adam([w], lr=1e-2)
print("Starting parameter learning...")
print(f"Target w: {true_w}, Initial w: {w.item():.4f}")

for i in range(201):
    optimizer.zero_grad()
    y_pred = magnus_odeint_adjoint(A_learnable_func, y0, t_span, params=w)
    loss = torch.mean((y_pred - y_target)**2)
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(f"Iter {i:03d} | Loss: {loss.item():.6f} | Learned w: {w.item():.4f}")

print(f"\nFinal learned frequency: {w.item():.4f}")
```

This example shows how the solver seamlessly handles inputs with a batch shape `(2, 3)`, solving all 6 systems and aggregating the loss for gradient-based optimization.

## Running Tests

To run the test suite, first install the development dependencies. **Note**: Use quotes around `.[dev]` to prevent shell expansion issues.

```bash
pip install -e ".[dev]"
```

Then, run pytest:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
