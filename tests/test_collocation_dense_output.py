import torch
import pytest
from torch_linode.solvers import odeint, merge_dense_outputs

# Define a simple time-varying linear ODE system
def system_func(t, params):
    t = torch.as_tensor(t)
    batch_shape = t.shape
    A = torch.zeros(*batch_shape, 2, 2, dtype=t.dtype, device=t.device)
    A[..., 0, 0] = -0.1
    A[..., 0, 1] = -t
    A[..., 1, 0] = t
    A[..., 1, 1] = -0.1
    g = torch.stack([torch.sin(t), torch.cos(t)], dim=-1)
    return A, g

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_merged_collocation_dense_output(dtype):
    """Tests the merged CollocationDenseOutput for a non-homogeneous system."""
    y0 = torch.tensor([1.0, 0.0], dtype=dtype)
    
    # Solve over two connected intervals
    t_span1 = torch.tensor([0.0, 1.5], dtype=dtype)
    t_span2 = torch.tensor([1.5, 3.0], dtype=dtype)

    # Get two separate dense output objects
    rtol = 1e-5
    atol = 1e-6
    dense_output1 = odeint(
        system_func, y0, t_span1, 
        method='glrk', order=4, dense_output=True, dense_output_method='collocation',
        rtol=rtol, atol=atol
    )
    y1_end = dense_output1(t_span1[-1])
    
    dense_output2 = odeint(
        system_func, y1_end, t_span2, 
        method='glrk', order=4, dense_output=True, dense_output_method='collocation',
        rtol=rtol, atol=atol
    )

    # Merge them
    merged_dense_output = merge_dense_outputs([dense_output1, dense_output2])

    # --- Verification ---
    # 1. Check if the time grid of the merged output is correct
    expected_ts = torch.cat([dense_output1.ts, dense_output2.ts[1:]])
    assert torch.allclose(merged_dense_output.ts, expected_ts)

    # 2. Evaluate at various points and compare with single-run integration
    t_eval = torch.linspace(0, 3.0, 15, dtype=dtype)
    
    # Get reference solution by solving over the whole interval at once
    reference_solution = odeint(
        system_func, y0, t_eval, 
        method='glrk', order=4, dense_output=False,
        rtol=rtol, atol=atol
    )
    
    # Get interpolated solution from the merged dense output
    interpolated_solution = merged_dense_output(t_eval)
    
    # The error should be small
    error = torch.norm(interpolated_solution - reference_solution)
    assert error < 1e-5, f"Interpolation error is too high: {error}"

    # 3. Check boundary conditions
    assert torch.allclose(merged_dense_output(torch.tensor(0.0, dtype=dtype)), y0, atol=1e-6)
    assert torch.allclose(merged_dense_output(torch.tensor(3.0, dtype=dtype)), dense_output2(t_span2[-1]), atol=1e-6)