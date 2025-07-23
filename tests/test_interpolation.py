import torch
import math
import numpy as np
import pytest
from torch_magnus import magnus_odeint

# Define the highly oscillatory system for testing
w0, w1, w2 = 10.0, 5.0, 20.0

def A_func(t, params=None):
    """A(t) for the oscillatory system."""
    t = torch.as_tensor(t)
    wt = w0 + w1 * torch.cos(w2 * t)
    A = torch.zeros(t.shape + (2, 2), dtype=torch.float64)
    A[..., 0, 1] = wt
    A[..., 1, 0] = -wt
    return A


def analytical_solution(t):
    """Analytical solution for the oscillatory system."""
    t = torch.as_tensor(t, dtype=torch.float64)
    theta_t = w0 * t + (w1 / w2) * torch.sin(w2 * t)
    return torch.stack([torch.cos(theta_t), -torch.sin(theta_t)], dim=-1)


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("rtol", [1e-4, 1e-6])
def test_interpolation_accuracy(order, rtol):
    """
    Tests the accuracy of the dense output interpolation for a highly oscillatory system.

    1. Solves the ODE on a coarse time grid.
    2. Evaluates the solution on a much finer time grid using interpolation.
    3. Compares the interpolated solution to the analytical solution.
    4. Checks that the interpolation error is reasonably low.
    """
    print(f"\nTesting interpolation accuracy for order={order}, rtol={rtol}")

    # Initial conditions and time span for the solver
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    # Use a coarse grid to force the solver to take larger steps
    t_span = torch.linspace(0, 0.5, 10, dtype=torch.float64)

    # Solve the ODE and request dense output
    solution = magnus_odeint(
        A_func,
        y0,
        t_span,
        order=order,
        rtol=rtol,
        atol=rtol * 1e-1,
        dense_output=True
    )

    # Fine grid for evaluating the interpolation
    t_eval = torch.linspace(t_span.min(), t_span.max(), 100, dtype=torch.float64)

    # Get interpolated and analytical solutions
    y_interpolated = solution(t_eval)
    y_analytical = analytical_solution(t_eval)

    # Calculate the maximum interpolation error
    interpolation_error = torch.norm(y_interpolated - y_analytical, dim=-1)

    print(f" Solution error: {interpolation_error[-1]}")

    # Set a realistic error threshold for interpolation
    # Interpolation error is expected to be higher than solver step error
    error_threshold = 100 * (rtol * 1e-1 + rtol * torch.norm(y_analytical, dim=-1))

    print(f"  Max interpolation error: {interpolation_error.max().item():.2e}")
    print(f"  Max error threshold: {error_threshold.max().item():.2e}")

    success = interpolation_error < error_threshold
    assert torch.all(success), (
        "Interpolation failures:\n" +
        "\n".join([  # 使用 join 将多行错误合并成一个字符串
            f"  - Index {i}: Error {interpolation_error[i]:.4e} > Threshold {error_threshold[i]:.4e}"
            for i in (~success).nonzero(as_tuple=True)[0] # 列表推导式
        ])
    )

if __name__ == "__main__":
    test_interpolation_accuracy(2, 1e-6)
    for order in [2, 4, 6]:
        for rtol in [1e-4, 1e-6]:
            test_interpolation_accuracy(order, rtol)