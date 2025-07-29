import torch
import pytest
from torch_linode.solvers import odeint_adjoint

dtype = torch.float64
@pytest.mark.parametrize("method", ['glrk', 'magnus'])
def test_magnus_nonhomogeneous_gradcheck_simple(method):
    dim = 2
    y0 = torch.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    t_span = torch.tensor([0.0, 1.0], dtype=dtype)

    # Combine A and g into a single parameter tensor for gradcheck
    A_flat = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    g_flat = torch.tensor([0.5, 0.6], dtype=dtype)
    params_tensor = torch.cat([A_flat, g_flat]).requires_grad_(True)

    def system_func(t, params):
        # Unpack A and g from the single params tensor
        A_param = params[:4].view(dim, dim)
        g_param = params[4:]
        
        # Expand to match t's shape
        t_tensor = torch.as_tensor(t, dtype=t.dtype, device=t.device)
        A_t = A_param.expand(*t_tensor.shape, dim, dim)
        g_t = g_param.expand(*t_tensor.shape, dim)
        return A_t, g_t

    # The function to be checked by gradcheck
    def func_to_check(y0_in, params_in):
        # Pass the combined parameters tensor to odeint_adjoint
        # Note: The [0] slice was likely a bug, odeint_adjoint returns the full trajectory.
        solution = odeint_adjoint(system_func, y0_in, t_span, params=params_in, method=method)
        return solution

    # gradcheck now correctly checks the function with respect to y0 and the combined params_tensor
    # We test the gradient of the final state
    assert torch.autograd.gradcheck(lambda y, p: func_to_check(y, p)[-1], (y0, params_tensor), eps=1e-6, atol=1e-4)
    print(f"Magnus non-homogeneous gradcheck (simple, {dtype}) passed.")

