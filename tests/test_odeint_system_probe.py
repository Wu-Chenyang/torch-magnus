import torch
import pytest
from torch_linode.solvers import odeint

class TestOdeintSystemProbe:
    def test_homogeneous_system_probe(self):
        def system_func_homogeneous(t, params):
            return torch.eye(2)

        y0 = torch.zeros(2)
        t = torch.tensor([0.0, 1.0])

        # We expect odeint to run without error for homogeneous system
        try:
            odeint(system_func_homogeneous, y0, t, method='magnus', order=2)
        except Exception as e:
            pytest.fail(f"odeint failed for homogeneous system: {e}")

    def test_nonhomogeneous_system_probe(self):
        def system_func_nonhomogeneous(t, params):
            A = torch.eye(2)
            g = torch.ones(2)
            return A, g

        y0 = torch.zeros(2)
        t = torch.tensor([0.0, 1.0])

        # We expect odeint to run without error for non-homogeneous system
        try:
            odeint(system_func_nonhomogeneous, y0, t, method='magnus', order=2)
        except Exception as e:
            pytest.fail(f"odeint failed for non-homogeneous system: {e}")

    def test_invalid_system_probe(self):
        def system_func_invalid(t, params):
            return "invalid_return_type"

        y0 = torch.zeros(2)
        t = torch.tensor([0.0, 1.0])

        # We expect odeint to raise TypeError for invalid return type
        with pytest.raises(TypeError, match="system_func_or_module must return a Tensor"): # or a Tuple[Tensor, Tensor]
            odeint(system_func_invalid, y0, t, method='magnus', order=2)
