import torch
import pytest
from torch_linode import odeint

# --- Test System Definition ---

def A_func_batch(t, params):
    """A(t) for a batch of harmonic oscillators. `t` is ignored (time-independent)."""
    w = params
    if t.ndim > 0:
        shape = torch.broadcast_shapes(t.shape, w.shape+(1,))
    else:
        shape = w.shape
    A = torch.zeros(shape + (2, 2), dtype=w.dtype, device=w.device)
    if t.ndim > 0:
        A[..., 0, 1] = w.unsqueeze(-1)
        A[..., 1, 0] = -w.unsqueeze(-1)
    else:
        A[..., 0, 1] = w
        A[..., 1, 0] = -w
    print(t.shape, w.shape, A.shape)
    return A

def analytical_solution_batch(t, params):
    """
    Analytical solution for a batch of harmonic oscillators.
    t has shape (*t_shape)
    params has shape (*broadcasted_batch_shape)
    """
    w = params
    # Unsqueeze w to make it broadcastable with t's time dimension.
    w_unsqueezed = w.unsqueeze(-1)
    theta_t = w_unsqueezed * t
    return torch.stack([torch.cos(theta_t), -torch.sin(theta_t)], dim=-1)


@pytest.mark.parametrize("ode_batch_shape", [(), (2,), (3, 4)])
@pytest.mark.parametrize("dense_output_method", ["naive", "collocation_precompute"])
def test_interpolation_with_batched_t(ode_batch_shape, dense_output_method):
    """
    Tests dense output when t_eval has batch dimensions that need to broadcast
    with the ODE's batch dimensions.
    """
    dtype = torch.float64
    
    # 1. Set up and solve the batched ODE
    if ode_batch_shape:
        w = torch.linspace(1.0, 2.0, int(torch.prod(torch.tensor(ode_batch_shape))), dtype=dtype).reshape(ode_batch_shape)
    else:
        w = torch.tensor(1.5, dtype=dtype)

    y0 = torch.tensor([1.0, 0.0], dtype=dtype)
    if ode_batch_shape:
        y0 = y0.expand(ode_batch_shape + (2,))

    t_span = torch.tensor([0., 2.0], dtype=dtype)

    solution = odeint(
        A_func_batch,
        y0,
        t_span,
        params=w,
        order=6,
        rtol=1e-6,
        dense_output=True,
        dense_output_method=dense_output_method
    )

    # 2. Create a batched evaluation time tensor
    t_len = 10
    t_eval_flat = torch.linspace(t_span.min(), t_span.max(), t_len, dtype=dtype)
    
    t_eval = t_eval_flat.repeat(*ode_batch_shape, 1)

    # 3. Call dense output with batched t_eval
    y_interpolated = solution(t_eval)

    # 4. Check output shape
    expected_shape = ode_batch_shape + (t_len,) + (2,)
    assert y_interpolated.shape == expected_shape, \
        f"Shape mismatch! Got {y_interpolated.shape}, expected {expected_shape}"

    # 5. Check accuracy
    y_analytical = analytical_solution_batch(t_eval, w)
    
    error = torch.norm(y_interpolated - y_analytical, dim=-1)
    
    assert torch.all(error < 1e-4), f"High interpolation error: {error.max().item()}"
    print(f"Test passed for ode_batch={ode_batch_shape}. Output shape: {y_interpolated.shape}")