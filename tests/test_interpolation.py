import torch
import math
import numpy as np
import pytest
from torch_linode import odeint

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
@pytest.mark.parametrize("rtol", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("dense_output_method", ["naive", "collocation_precompute", "collocation_ondemand"])
def test_interpolation_accuracy(order, rtol, dense_output_method):
    """
    Tests the accuracy of the dense output interpolation for a highly oscillatory system.

    1. Solves the ODE on a coarse time grid.
    2. Evaluates the solution on a much finer time grid using interpolation.
    3. Compares the interpolated solution to the analytical solution.
    4. Checks that the interpolation error is reasonably low.
    """
    print(f"\nTesting interpolation accuracy for order={order}, rtol={rtol}, interpolation method={dense_output_method}")

    # Initial conditions and time span for the solver
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    # Use a coarse grid to force the solver to take larger steps
    t_span = torch.tensor([0., 0.5], dtype=torch.float64)

    # Solve the ODE and request dense output
    solution = odeint(
        A_func,
        y0,
        t_span,
        order=order,
        rtol=rtol,
        atol=rtol * 1e-1,
        dense_output=True,
        dense_output_method=dense_output_method
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
    error_threshold = (rtol * 1e-1 + rtol * torch.norm(y_analytical, dim=-1))

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


@pytest.mark.parametrize("order", [4, 6])
@pytest.mark.parametrize("rtol", [1e-6, 1e-8])
@pytest.mark.parametrize("dense_output_method", ["collocation_precompute", "collocation_ondemand"])
def test_harmonic_oscillator_interpolation(order, rtol, dense_output_method):
    """
    Tests the interpolation accuracy for a simple harmonic oscillator system.
    dy/dt = [[0, 1], [-w^2, 0]]y
    This system has a known analytical solution, allowing for precise error checking.
    """
    # --- 1. Define the System and its Analytical Solution ---
    def A_func(t, params: torch.Tensor) -> torch.Tensor:
        """A = [[0, 1], [-ω², 0]]"""
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t)
        omega = params[0]
        A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
        A[..., 0, 1] += 1.0
        A[..., 1, 0] += -omega**2
        return A

    omega = 2.0
    def analytical_sol(t):
        # Use torch.cos/sin for vectorized operations on tensors
        t = torch.as_tensor(t, dtype=torch.float64)
        return torch.stack([
            torch.cos(omega * t),
            -omega * torch.sin(omega * t)
        ], dim=-1)

    # --- 2. Set up the Problem ---
    params = torch.tensor([omega], dtype=torch.float64)
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    T = 0.5
    t_span = torch.tensor([0.0, T], dtype=torch.float64)

    # --- 3. Solve and Get Dense Output ---
    solution = odeint(
        A_func,
        y0,
        t_span,
        params=params,
        order=order,
        rtol=rtol,
        atol=rtol * 1e-1,
        dense_output=True,
        dense_output_method=dense_output_method
    )

    # --- 4. Test Interpolation Accuracy ---
    # Choose some random points within the interval to test interpolation
    num_test_points = 50
    t_eval = torch.rand(num_test_points, dtype=torch.float64) * T

    y_interpolated = solution(t_eval)
    y_analytical = analytical_sol(t_eval)

    # Calculate the error and a consistent threshold
    error = torch.norm(y_interpolated - y_analytical, dim=-1)
    atol = rtol * 1e-1 # Consistent with other tests
    error_threshold = atol + rtol * torch.norm(y_analytical, dim=-1)

    assert torch.all(error < 10 * error_threshold), \
        f"Interpolation error ({error.max().item():.2e}) exceeds threshold ({10 * error_threshold.max().item():.2e})"

if __name__ == "__main__":
    test_interpolation_accuracy(2, 1e-6)
    for order in [2, 4, 6]:
        for rtol in [1e-4, 1e-6]:
            test_interpolation_accuracy(order, rtol)