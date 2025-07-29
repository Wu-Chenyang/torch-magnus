import torch
import pytest
from unittest.mock import MagicMock, patch
from typing import Callable

from torch_linode import solvers

class MockCtx:
    def __init__(self, t, param_values, functional_system_func, param_keys, method, order, rtol, atol, quad_method, quad_options, is_nonhomogeneous, y_dense_traj_aug, y0_requires_grad):
        self.saved_tensors = (t, *param_values)
        self.functional_system_func = functional_system_func
        self.param_keys = param_keys
        self.method = method
        self.order = order
        self.rtol = rtol
        self.atol = atol
        self.quad_method = quad_method
        self.quad_options = quad_options
        self.is_nonhomogeneous = is_nonhomogeneous
        self.y_dense_traj_aug = y_dense_traj_aug
        self.y0_requires_grad = y0_requires_grad

def test_quad_integrator_is_nonhomogeneous_param():
    # Dummy data for non-homogeneous system
    dim = 2
    y0 = torch.randn(dim, requires_grad=True)
    t = torch.tensor([0.0, 1.0])
    grad_y_traj = torch.randn(2, dim) # T=2, dim=2

    # Dummy functional_system_func for non-homogeneous system
    def dummy_functional_system_func(t_val, p_dict):
        A = torch.eye(dim) * t_val
        g = torch.ones(dim) * t_val
        return A, g

    # Dummy y_dense_traj_aug (needs to be callable and return a tensor)
    class DummyDenseOutput:
        def __call__(self, t_nodes):
            # Simulate augmented trajectory for Magnus
            if isinstance(t_nodes, torch.Tensor):
                return torch.randn(*t_nodes.shape, dim + 1)
            else:
                return torch.randn(dim + 1)

    y_dense_traj_aug = DummyDenseOutput()

    # Dummy parameters
    param_keys = ['p1']
    p1 = torch.tensor(1.0, requires_grad=True)
    param_values = [p1]

    # Mock the adaptive_ode_solve to return a mock dense output object for a_dense_traj
    mock_a_dense_traj = MagicMock()
    mock_a_dense_traj.return_value = torch.randn(dim) # For a_dense_traj(t_prev)
    
    with patch('torch_linode.solvers.adaptive_ode_solve', return_value=mock_a_dense_traj) as mock_adaptive_ode_solve, \
         patch('torch_linode.solvers.AdaptiveGaussKronrod') as MockGK, \
         patch('torch_linode.solvers.FixedSimpson') as MockSimpson:

        # Create a mock instance for the quad_integrator
        mock_quad_integrator_instance = MagicMock()
        mock_quad_integrator_instance.return_value = {'p1': torch.zeros_like(p1)} # Return dummy gradients
        MockGK.return_value = mock_quad_integrator_instance
        MockSimpson.return_value = mock_quad_integrator_instance

        # Test with 'gk' method
        ctx_gk = MockCtx(
            t=t,
            param_values=param_values,
            functional_system_func=dummy_functional_system_func,
            param_keys=param_keys,
            method='magnus',
            order=4, rtol=1e-6, atol=1e-8,
            quad_method='gk', quad_options={},
            is_nonhomogeneous=True,
            y_dense_traj_aug=y_dense_traj_aug,
            y0_requires_grad=True
        )
        solvers._Adjoint.backward(ctx_gk, grad_y_traj)
        
        # Assert that the mock was called with is_nonhomogeneous=True
        # We'll check the type of the first two arguments instead of exact match
        assert isinstance(mock_quad_integrator_instance.call_args.args[0], Callable)
        assert isinstance(mock_quad_integrator_instance.call_args.args[1], Callable)
        assert mock_quad_integrator_instance.call_args.args[2] == t[1].item()
        assert mock_quad_integrator_instance.call_args.args[3] == t[0].item()
        assert mock_quad_integrator_instance.call_args.args[4] == ctx_gk.atol
        assert mock_quad_integrator_instance.call_args.args[5] == ctx_gk.rtol
        assert mock_quad_integrator_instance.call_args.args[6] == {'p1': p1}
        assert mock_quad_integrator_instance.call_args.args[7] is True # This is the key assertion

        # Reset mock for next test
        mock_quad_integrator_instance.reset_mock()
        mock_adaptive_ode_solve.reset_mock()

        # Test with 'simpson' method
        ctx_simpson = MockCtx(
            t=t,
            param_values=param_values,
            functional_system_func=dummy_functional_system_func,
            param_keys=param_keys,
            method='magnus',
            order=4, rtol=1e-6, atol=1e-8,
            quad_method='simpson', quad_options={},
            is_nonhomogeneous=True,
            y_dense_traj_aug=y_dense_traj_aug,
            y0_requires_grad=True
        )
        solvers._Adjoint.backward(ctx_simpson, grad_y_traj)
        
        assert isinstance(mock_quad_integrator_instance.call_args.args[0], Callable)
        assert isinstance(mock_quad_integrator_instance.call_args.args[1], Callable)
        assert mock_quad_integrator_instance.call_args.args[2] == t[1].item()
        assert mock_quad_integrator_instance.call_args.args[3] == t[0].item()
        assert mock_quad_integrator_instance.call_args.args[4] == ctx_simpson.atol
        assert mock_quad_integrator_instance.call_args.args[5] == ctx_simpson.rtol
        assert mock_quad_integrator_instance.call_args.args[6] == {'p1': p1}
        assert mock_quad_integrator_instance.call_args.args[7] is True
