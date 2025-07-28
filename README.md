# torch-linode

[![PyPI version](https://badge.fury.io/py/torch-linode.svg)](https://badge.fury.io/py/torch-linode)
[![Tests](https://github.com/Wu-Chenyang/torch-linode/actions/workflows/ci.yml/badge.svg)](https://github.com/Wu-Chenyang/torch-linode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`torch-linode` is a specialized PyTorch-based library for the efficient **batch solving of homogeneous and non-homogeneous linear ordinary differential equations (ODEs)**. It solves systems of the form:
- **Homogeneous**: `dy/dt = A(t)y(t)`
- **Non-homogeneous**: `dy/dt = A(t)y(t) + g(t)`

It leverages Magnus-type integrators to provide high-precision, differentiable, and GPU-accelerated solutions. This library is particularly well-suited for problems in quantum mechanics, control theory, and other areas of physics and engineering where such ODEs are common.

## Key Features

- **Solves Homogeneous & Non-homogeneous Systems**: Unified API for both types of linear ODEs.
- **Batch Processing**: Natively handles batches of initial conditions and parameters for massive parallelization.
- **High-Order Integrators**: Includes 2nd, 4th, and 6th-order Magnus and Gauss-Legendre Runge-Kutta (GLRK) integrators.
- **Adaptive Stepping**: Automatically adjusts step size to meet specified error tolerances (`rtol`, `atol`).
- **Differentiable**: Gradients can be backpropagated through the solvers using a memory-efficient adjoint method.
- **Dense Output**: Provides continuous solutions for evaluation at any time point.
- **GPU Support**: Runs seamlessly on CUDA-enabled devices.

## Installation

```bash
pip install torch-linode
```

Or, for development, clone this repository and install in editable mode:

```bash
git clone https://github.com/Wu-Chenyang/torch-linode.git
cd torch-linode
pip install -e ".[dev]"
```

## API and Usage

The primary functions are `odeint` and `odeint_adjoint`. The solver automatically detects whether the system is homogeneous or non-homogeneous based on the return value of the system function.

```python
odeint(
    system_func_or_module: Union[Callable, nn.Module], 
    y0: Tensor, 
    t: Union[Sequence[float], torch.Tensor],
    params: Tensor = None,
    # ... other options
) -> Tensor
```

### Parameters

- `system_func_or_module`: The function or `nn.Module` that defines the system.
  - **For homogeneous systems (`dy/dt = Ay`)**: Return a single tensor `A(t)` of shape `(*batch_shape, dim, dim)`.
  - **For non-homogeneous systems (`dy/dt = Ay + g`)**: Return a tuple `(A(t), g(t))`, where `g(t)` is a tensor of shape `(*batch_shape, dim)`.
- `y0`: A tensor of initial conditions with shape `(*batch_shape, dim)`.
- `t`: A 1D tensor or sequence of time points at which to evaluate the solution.
- `params`: Optional tensor of parameters to be passed to the system function.

### Returns

A tensor of shape `(*batch_shape, N, dim)` containing the solution trajectories.

---

`odeint_adjoint` has the same signature but uses a more memory-efficient method for computing gradients, making it ideal for training and optimization.

## Example: Solving a Non-homogeneous ODE

This example solves `dy/dt = A(t)y + g(t)` where `A` is a constant rotation matrix and `g` is a time-dependent vector.

```python
import torch
from torch_linode import odeint
import numpy as np

# 1. Define the non-homogeneous system
dim = 2
A = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)

def system_func(t, params):
    # This function returns a tuple (A, g)
    t_tensor = torch.as_tensor(t, dtype=torch.float64)
    A_t = A.expand(*t_tensor.shape, dim, dim)
    g_t = torch.stack([torch.sin(t_tensor), torch.cos(t_tensor)], dim=-1)
    return A_t, g_t

# 2. Set initial conditions and time points
y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
t_span = torch.linspace(0, 2 * np.pi, 30, dtype=torch.float64)

# 3. Call the solver
solution = odeint(
    system_func_or_module=system_func,
    y0=y0,
    t=t_span
)

# 4. The exact solution is y(t) = [cos(t) + t*sin(t), -sin(t) + t*cos(t)]
# The final point should be [1, 2*pi]
print(f"Computed y(2pi): {solution[-1].numpy()}")
# Expected: [1.         6.28318531]
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
