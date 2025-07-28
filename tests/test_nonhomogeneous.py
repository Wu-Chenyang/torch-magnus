import torch
import numpy as np
from torch_linode import odeint, odeint_adjoint

# Define a tolerance for comparing floating point numbers
RTOL = 1e-5
ATOL = 1e-7

def test_odeint_nonhomogeneous_correctness():
    """
    Tests the forward pass correctness of odeint for a non-homogeneous ODE
    with a known analytical solution.
    """
    # 1. Define the problem: dy/dt = A(t)y(t) + g(t)
    # Let A(t) = [[0, 1], [-1, 0]] (constant rotation)
    # Let g(t) = [sin(t), cos(t)]
    dim = 2
    A = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)

    def system_func(t, params):
        t_tensor = torch.as_tensor(t, dtype=torch.float64)
        # Ensure system handles batched time inputs
        if t_tensor.ndim == 0:
            A_t = A.unsqueeze(0)
            g_t = torch.stack([torch.sin(t_tensor), torch.cos(t_tensor)], dim=-1).unsqueeze(0)
        else:
            A_t = A.expand(t_tensor.shape[0], dim, dim)
            g_t = torch.stack([torch.sin(t_tensor), torch.cos(t_tensor)], dim=-1)
        return A_t, g_t

    # 2. Set initial conditions and time points
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    t_span = torch.linspace(0, 2 * np.pi, 30, dtype=torch.float64)

    # 3. Call the solver
    solution_trajectory = odeint(
        system_func_or_module=system_func,
        y0=y0,
        t=t_span,
        params=None,
        method='magnus',
        order=6  # Use a higher order for better precision
    )

    # 4. Verification
    # The exact solution is y(t) = [cos(t) + t*sin(t), -sin(t) + t*cos(t)]
    def exact_solution(t):
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        y1 = cos_t + t * sin_t
        y2 = -sin_t + t * cos_t
        return torch.stack([y1, y2], dim=-1)

    y_exact = exact_solution(t_span)
    
    # Assert that the solution is close to the exact one
    assert torch.allclose(solution_trajectory, y_exact, rtol=RTOL, atol=ATOL), \
        f"Max error: {torch.max(torch.norm(solution_trajectory - y_exact, dim=-1)).item()}"

def test_odeint_adjoint_nonhomogeneous_gradient():
    """
    Tests the backward pass of odeint_adjoint for a non-homogeneous ODE
    by comparing the computed gradient with a numerical estimate.
    """
    # 1. Define a parameterized system
    # A(p) = [[0, p], [-p, 0]], g(t) = [t, t^2]
    # We want to find the gradient of a loss w.r.t. parameter p
    p_val = 0.5
    p = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)

    def system_func(t, params):
        t = torch.as_tensor(t, dtype=torch.float64)
        A = torch.zeros(*t.shape, 2, 2, dtype=torch.float64)
        A[..., 0, 1] = params
        A[..., 1, 0] = -params
        g = torch.stack([t, t**2], dim=-1)
        return A, g

    y0 = torch.tensor([1., 0.], dtype=torch.float64)
    t_span = torch.linspace(0., 1., 5, dtype=torch.float64)

    # 2. Compute gradient using odeint_adjoint
    y_final_adjoint = odeint_adjoint(system_func, y0, t_span, p)[-1]
    loss = y_final_adjoint.norm()
    loss.backward()
    grad_adjoint = p.grad.clone()

    # 3. Estimate gradient numerically using finite differences
    p.grad.zero_()
    eps = 1e-7
    
    # Calculate loss at p + eps
    p_plus = torch.tensor(p_val + eps, dtype=torch.float64)
    y_final_plus = odeint(system_func, y0, t_span, p_plus)[-1]
    loss_plus = y_final_plus.norm()
    
    # Calculate loss at p - eps
    p_minus = torch.tensor(p_val - eps, dtype=torch.float64)
    y_final_minus = odeint(system_func, y0, t_span, p_minus)[-1]
    loss_minus = y_final_minus.norm()

    grad_numerical = (loss_plus - loss_minus) / (2 * eps)

    # 4. Assert that the gradients are close
    assert torch.allclose(grad_adjoint, grad_numerical, rtol=RTOL, atol=ATOL), f"Adjoint gradient {grad_adjoint.item()} differs from numerical estimate {grad_numerical.item()}"

def test_odeint_batching():
    """
    Tests if the solver correctly handles batched initial conditions (y0).
    """
    batch_size = 5
    dim = 2
    A = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)

    def system_func(t, params):
        t_tensor = torch.as_tensor(t, dtype=torch.float64)
        # Ensure system handles batched time inputs
        if t_tensor.ndim == 0:
            A_t = A.unsqueeze(0)
            g_t = torch.stack([torch.sin(t_tensor), torch.cos(t_tensor)], dim=-1).unsqueeze(0)
        else:
            A_t = A.expand(t_tensor.shape[0], dim, dim)
            g_t = torch.stack([torch.sin(t_tensor), torch.cos(t_tensor)], dim=-1)
        return A_t, g_t

    # Create a batch of initial conditions
    y0_batch = torch.randn(batch_size, dim, dtype=torch.float64)
    t_span = torch.linspace(0, 1.0, 10, dtype=torch.float64)

    # Solve for the batch all at once
    solution_batch = odeint(system_func, y0_batch, t_span)

    # Solve for each item in the batch individually
    solutions_individual = []
    for i in range(batch_size):
        y0_i = y0_batch[i]
        sol_i = odeint(system_func, y0_i, t_span)
        solutions_individual.append(sol_i)
    solution_manual_batch = torch.stack(solutions_individual, dim=0)

    # Assert that the batched solution is identical to the manually stacked one
    assert solution_batch.shape == (batch_size, t_span.shape[0], dim), \
        f"Unexpected output shape: {solution_batch.shape}"
    assert torch.allclose(solution_batch, solution_manual_batch, rtol=RTOL, atol=ATOL), \
        "Batched solution does not match individually computed solutions."

def test_odeint_adjoint_batching_gradient():
    """
    Tests if odeint_adjoint correctly handles batched gradients.
    """
    batch_size = 3
    dim = 2
    # Each item in the batch has its own parameter
    p_vals = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64).view(batch_size, 1)
    p = p_vals.clone().requires_grad_(True)

    def system_func(t, params):
        t = torch.as_tensor(t, dtype=torch.float64)
        # A has a batch dimension corresponding to p
        A = torch.zeros(params.shape[0], 2, 2, dtype=torch.float64)
        A[:, 0, 1] = params.squeeze(-1)
        A[:, 1, 0] = -params.squeeze(-1)
        
        # g must be broadcastable with A's batch dimensions.
        g = torch.stack([t, t**2], dim=-1)
        if t.ndim > 0:
            A = A.unsqueeze(1).expand(-1, t.shape[0], -1, -1)
            g = g.unsqueeze(0).expand(params.shape[0], -1, -1)

        return A, g

    # y0 also has a batch dimension
    y0 = torch.randn(batch_size, dim, dtype=torch.float64)
    t_span = torch.linspace(0., 1., 5, dtype=torch.float64)

    # --- 1. Compute gradients in batch mode ---
    y_final_batch = odeint_adjoint(system_func, y0, t_span, p)
    loss_batch = y_final_batch[:, -1, :].norm(dim=-1).sum() # Sum of norms for total loss
    loss_batch.backward()
    grad_batch = p.grad.clone()

    # --- 2. Compute gradients individually ---
    grads_individual = []
    for i in range(batch_size):
        p_i = p_vals[i].clone().requires_grad_(True)
        y0_i = y0[i]
        
        # The system function needs to be adapted to take the i-th parameter
        def system_func_i(t, params_i):
            A_i, g_i = system_func(t, params_i.unsqueeze(0))
            return A_i.squeeze(0), g_i.squeeze(0)

        y_final_i = odeint_adjoint(system_func_i, y0_i, t_span, p_i)
        loss_i = y_final_i[-1, :].norm()
        loss_i.backward()
        grads_individual.append(p_i.grad)
    
    grad_manual_batch = torch.stack(grads_individual, dim=0)

    # --- 3. Compare results ---
    assert grad_batch.shape == grad_manual_batch.shape, \
        f"Shape mismatch: {grad_batch.shape} vs {grad_manual_batch.shape}"
    assert torch.allclose(grad_batch, grad_manual_batch, rtol=RTOL, atol=ATOL), \
        "Batched gradients do not match individually computed gradients."

