import torch
import unittest
from torch_linode.collocation import Collocation
from torch_linode.butcher import GL4

class TestCollocationSolver(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.batch_shape = (2, 3)
        self.y0 = torch.randn(*self.batch_shape, self.dim, dtype=torch.float64)
        self.A_base = torch.randn(*self.batch_shape, self.dim, self.dim, device=self.y0.device, dtype=torch.float64)
        self.g_base = torch.randn(*self.batch_shape, self.dim, device=self.y0.device, dtype=torch.float64)
        self.tableau = GL4

    def A_func(self, t, params=None):
        if isinstance(t, float) or t.ndim == 0:
            return self.A_base * torch.sin(torch.as_tensor(t))
        elif t.ndim == 1:
            sin_t = torch.sin(t).view(-1, 1, 1)
            return self.A_base.unsqueeze(-3) * sin_t
        else:
            raise ValueError(f"Unsupported time shape: {t.shape}")

    def A_func_nonhomogeneous(self, t, params=None):
        if isinstance(t, float) or t.ndim == 0:
            A_t = self.A_base * torch.cos(torch.as_tensor(t))
            g_t = self.g_base * torch.sin(torch.as_tensor(t))
            return A_t, g_t
        elif t.ndim == 1:
            sin_t = torch.sin(t).view(-1, 1, 1)
            cos_t = torch.cos(t).view(-1, 1)
            return self.A_base.unsqueeze(-3) * sin_t, self.g_base.unsqueeze(-2) * cos_t
        else:
            raise ValueError(f"Unsupported time shape: {t.shape}")

if __name__ == '__main__':
    unittest.main()