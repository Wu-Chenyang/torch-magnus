import torch
import pytest
from torch_linode.solvers import _Adjoint, odeint

class TestMagnusAdjointForward:
    def test_magnus_adjoint_forward_homogeneous(self):
        def system_func(t, params):
            t_tensor = torch.as_tensor(t)
            if t_tensor.ndim == 0:
                return torch.eye(2) * t_tensor
            else:
                return torch.eye(2).unsqueeze(0).expand(t_tensor.shape[0], -1, -1) * t_tensor.unsqueeze(-1).unsqueeze(-1)

        y0 = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        t = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        param_keys = []
        param_values = []

        y_traj = _Adjoint.apply(y0, t, system_func, param_keys, 'magnus', 2, 1e-6, 1e-8, 'gk', None, *param_values)
        assert y_traj.shape == (3, 2)
        # Further assertions could check the actual values if a simple analytical solution is known

    def test_magnus_adjoint_forward_nonhomogeneous_magnus(self):
        def system_func(t, params):
            t_tensor = torch.as_tensor(t, dtype=torch.float64)
            if t_tensor.ndim == 0:
                A = torch.eye(2, dtype=torch.float64) * t_tensor
                g = torch.ones(2, dtype=torch.float64) * t_tensor
            else:
                A = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(t_tensor.shape[0], -1, -1) * t_tensor.unsqueeze(-1).unsqueeze(-1)
                g = torch.ones(t_tensor.shape[0], 2, dtype=torch.float64) * t_tensor.unsqueeze(-1)
            return A, g

        y0 = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        t = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        param_keys = []
        param_values = []

        y_traj = _Adjoint.apply(y0, t, system_func, param_keys, 'magnus', 2, 1e-6, 1e-8, 'gk', None, *param_values)
        # For Magnus non-homogeneous, the output should be sliced back to original dimension
        assert y_traj.shape == (3, 2)

    def test_magnus_adjoint_forward_nonhomogeneous_glrk(self):
        def system_func(t, params):
            t_tensor = torch.as_tensor(t, dtype=torch.float64)
            if t_tensor.ndim == 0:
                A = torch.eye(2, dtype=torch.float64) * t_tensor
                g = torch.ones(2, dtype=torch.float64) * t_tensor
            else:
                A = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(t_tensor.shape[0], -1, -1) * t_tensor.unsqueeze(-1).unsqueeze(-1)
                g = torch.ones(t_tensor.shape[0], 2, dtype=torch.float64) * t_tensor.unsqueeze(-1)
            return A, g

        y0 = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        t = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        param_keys = []
        param_values = []

        y_traj = _Adjoint.apply(y0, t, system_func, param_keys, 'glrk', 2, 1e-6, 1e-8, 'gk', None, *param_values)
        # For GLRK non-homogeneous, the output should NOT be sliced, as it operates on original dim
        assert y_traj.shape == (3, 2)

    def test_magnus_adjoint_forward_with_params(self):
        p = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)
        def system_func(t, params):
            t_tensor = torch.as_tensor(t, dtype=torch.float64)
            if t_tensor.ndim == 0:
                A = torch.eye(2, dtype=torch.float64) * t_tensor * params['p']
            else:
                A = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(t_tensor.shape[0], -1, -1) * t_tensor.unsqueeze(-1).unsqueeze(-1) * params['p']
            return A

        y0 = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        t = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        param_keys = ['p']
        param_values = [p]

        y_traj = _Adjoint.apply(y0, t, system_func, param_keys, 'magnus', 2, 1e-6, 1e-8, 'gk', None, *param_values)
        assert y_traj.shape == (3, 2)