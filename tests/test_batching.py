import torch
import torch.nn as nn
import unittest
import math
from torch_magnus import magnus_odeint, magnus_odeint_adjoint, magnus_solve

class LinearMatrixModule(nn.Module):
    """A simple nn.Module that defines a time-independent linear system."""
    def __init__(self, dim: int, batch_shape: tuple = ()):
        super().__init__()
        self.dim = dim
        self.batch_shape = batch_shape
        # Learnable parameter for the off-diagonal element
        self.omega = nn.Parameter(torch.randn(*batch_shape, 1))

    def forward(self, t):
        # Build the matrix A = [[0, w], [-w, 0]]
        # This module ignores t, creating a time-independent system.
        A = torch.zeros(*self.batch_shape, self.dim, self.dim, device=self.omega.device, dtype=self.omega.dtype)
        A[..., 0, 1] = self.omega.squeeze(-1)
        A[..., 1, 0] = -self.omega.squeeze(-1)
        
        # The solver might pass a vector of times for quadrature in the backward pass.
        # We need to expand our matrix to match that time dimension.
        if isinstance(t, torch.Tensor) and t.ndim == 1:
            return A.unsqueeze(-3).expand(*self.batch_shape, t.shape[0], -1, -1)
        return A

class TestMagnusSolver(unittest.TestCase):

    def test_batch_solve_shape(self):
        """测试 magnus_odeint 和 magnus_odeint_adjoint 在批处理模式下的输出形状是否正确。"""
        dim = 2
        batch_shape = (2, 3)
        y0 = torch.randn(*batch_shape, dim)
        t_span = torch.linspace(0, 1, 10)
        A_base = torch.randn(*batch_shape, dim, dim)

        def A_func(t, params=None):
            A_base_dev = A_base.to(t.device if isinstance(t, torch.Tensor) else y0.device)
            if isinstance(t, float) or t.ndim == 0:
                return A_base_dev * torch.sin(torch.tensor(t))
            elif t.ndim == 1:
                sin_t = torch.sin(t).view(-1, 1, 1)
                return A_base_dev.unsqueeze(-3) * sin_t
            else:
                raise ValueError(f"Unsupported time shape: {t.shape}")

        # Test magnus_odeint shape
        solution_odeint = magnus_odeint(A_func, y0, t_span)
        expected_shape = (*batch_shape, len(t_span), dim)
        self.assertEqual(solution_odeint.shape, expected_shape)

        # Test magnus_odeint_adjoint shape
        solution_adjoint = magnus_odeint_adjoint(A_func, y0, t_span)
        self.assertEqual(solution_adjoint.shape, expected_shape)

    def test_accuracy_batch(self):
        """测试求解一个已知解析解的批量谐振子系统的精度。"""
        dim = 2
        batch_shape = (2,) # Example batch shape
        y0 = torch.tensor([1.0, 0.0]).unsqueeze(0).expand(*batch_shape, -1) 
        t_span = torch.linspace(0, 2 * math.pi, 100)
        
        A_base = torch.tensor([[0., 1.], [-1., 0.]])
        def A_func(t, params=None):
            A_dev = A_base.to(t.device if isinstance(t, torch.Tensor) else y0.device)
            if isinstance(t, float) or t.ndim == 0:
                return A_dev.unsqueeze(0).expand(*batch_shape, -1, -1)
            elif t.ndim == 1:
                return A_dev.unsqueeze(0).unsqueeze(0).expand(*batch_shape, t.shape[0], -1, -1)
            else:
                raise ValueError(f"Unsupported time shape: {t.shape}")

        y_analytical_single = torch.stack([torch.cos(t_span), -torch.sin(t_span)], dim=-1)
        y_analytical = y_analytical_single.unsqueeze(0).expand(*batch_shape, -1, -1)

        atol = 1e-8
        rtol = 1e-6

        # Test magnus_odeint accuracy
        y_numerical_odeint = magnus_odeint(A_func, y0, t_span, order=4, rtol=rtol, atol=atol)
        error_odeint = torch.norm(y_numerical_odeint[..., -1, :] - y_analytical[..., -1, :])
        print(f"Accuracy test error (odeint, batch): {error_odeint.item()}")
        self.assertTrue(error_odeint < 10 * (atol + rtol * torch.norm(y_analytical[..., -1, :]).item()))

        # Test magnus_odeint_adjoint accuracy
        y_numerical_adjoint = magnus_odeint_adjoint(A_func, y0, t_span, order=4, rtol=rtol, atol=atol)
        error_adjoint = torch.norm(y_numerical_adjoint[..., -1, :] - y_analytical[..., -1, :])
        print(f"Accuracy test error (adjoint, batch): {error_adjoint.item()}")
        self.assertTrue(error_adjoint < 10 * (atol + rtol * torch.norm(y_analytical[..., -1, :]).item()))

    def test_gradient_backpropagation_batch(self):
        """测试使用 magnus_odeint_adjoint 进行批量梯度反向传播和参数学习，并分析收敛性能。"""
        dim = 2
        batch_shape = (2, 3) 
        y0 = torch.randn(*batch_shape, dim) # Initial condition
        t_span = torch.linspace(0, 2 * math.pi, 20)
        
        true_w = 1.0
        A_target_base = torch.tensor([[0., true_w], [-true_w, 0.]])
        def A_target_func(t, p=None):
            A_target_dev = A_target_base.to(t.device if isinstance(t, torch.Tensor) else y0.device)
            if batch_shape: # Only apply unsqueeze and expand if batch_shape is not empty
                if isinstance(t, float) or t.ndim == 0:
                    return A_target_dev.unsqueeze(0).expand(*batch_shape, -1, -1)
                elif t.ndim == 1:
                    return A_target_dev.unsqueeze(0).unsqueeze(0).expand(*batch_shape, t.shape[0], -1, -1)
                else:
                    raise ValueError(f"Unsupported time shape for A_target_func: {t.shape}")
            else: # If batch_shape is empty, return directly or unsqueeze for time dim if t is a vector
                if isinstance(t, float) or t.ndim == 0:
                    return A_target_dev
                elif t.ndim == 1:
                    return A_target_dev.unsqueeze(0).expand(t.shape[0], -1, -1)
                else:
                    raise ValueError(f"Unsupported time shape for A_target_func: {t.shape}")
        
        with torch.no_grad():
            y_target = magnus_odeint(A_target_func, y0, t_span)

        w = torch.nn.Parameter(torch.tensor(0.5))
        optimizer = torch.optim.Adam([w], lr=1e-2) 

        def A_learnable_func(t, params):
            A_matrix = torch.stack([
                torch.stack([torch.tensor(0., device=y0.device), params]),
                torch.stack([-params, torch.tensor(0., device=y0.device)])
            ])

            if batch_shape: # Only apply unsqueeze and expand if batch_shape is not empty
                if isinstance(t, float) or t.ndim == 0:
                    return A_matrix.unsqueeze(0).expand(*batch_shape, -1, -1)
                elif t.ndim == 1:
                    return A_matrix.unsqueeze(0).unsqueeze(0).expand(*batch_shape, t.shape[0], -1, -1)
                else:
                    raise ValueError(f"Unsupported time shape for A_learnable_func: {t.shape}")
            else: # If batch_shape is empty, return directly or unsqueeze for time dim if t is a vector
                if isinstance(t, float) or t.ndim == 0:
                    return A_matrix
                elif t.ndim == 1:
                    return A_matrix.unsqueeze(0).expand(t.shape[0], -1, -1)
                else:
                    raise ValueError(f"Unsupported time shape for A_learnable_func: {t.shape}")

        num_iterations = 100 
        for i in range(num_iterations):
            optimizer.zero_grad()
            y_pred = magnus_odeint_adjoint(A_learnable_func, y0, t_span, params=w) 
            loss = torch.mean((y_pred - y_target)**2)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 20 == 0 or i == 0 or i == num_iterations - 1:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}, Learned w: {w.item():.6f}")

        print(f"Final learned frequency (started at 0.5): {w.item()}")
        self.assertTrue(abs(w.item() - true_w) < 0.05) 

    def test_batch_vs_individual_steps(self):
        """对比批量求解和逐个求解的迭代步数差异。"""
        dim = 2
        batch_size = 5
        y0_batch = torch.randn(batch_size, dim)
        t_span_solve = (0.0, 10.0)
        A_base = torch.tensor([[0., 1.], [-1., 0.]])

        def A_func_batch(t, params=None):
            A_dev = A_base.to(t.device if isinstance(t, torch.Tensor) else y0_batch.device)
            if isinstance(t, float) or t.ndim == 0:
                return A_dev.unsqueeze(0).expand(batch_size, -1, -1)
            elif t.ndim == 1:
                return A_dev.unsqueeze(0).unsqueeze(0).expand(batch_size, t.shape[0], -1, -1)
            else:
                raise ValueError(f"Unsupported time shape: {t.shape}")

        traj_batch, _ = magnus_solve(y0_batch, t_span_solve, A_func_batch, {}, order=4, return_traj=True)
        steps_batch = len(traj_batch)

        total_steps_individual = 0
        for i in range(batch_size):
            y0_individual = y0_batch[i]
            def A_func_individual(t, params=None):
                A_dev = A_base.to(t.device if isinstance(t, torch.Tensor) else y0_individual.device)
                if isinstance(t, float) or t.ndim == 0:
                    return A_dev
                elif t.ndim == 1:
                    return A_dev.unsqueeze(0).expand(t.shape[0], -1, -1)
                else:
                    raise ValueError(f"Unsupported time shape: {t.shape}")
            
            traj_individual, _ = magnus_solve(y0_individual, t_span_solve, A_func_individual, {}, order=4, return_traj=True)
            total_steps_individual += len(traj_individual)

        print(f"Steps (batch solve): {steps_batch}")
        print(f"Steps (individual solve): {total_steps_individual}")
        self.assertTrue(steps_batch <= total_steps_individual)

    def test_nn_module_optimization(self):
        """Test optimization of a system defined by an nn.Module."""
        dim = 2
        batch_shape = (4,) # A batch of 4 systems
        dtype = torch.float64

        # 1. Generate target data from a module with known parameters
        true_omega = torch.tensor([1.5, -0.5, 2.0, 0.8], dtype=dtype).view(batch_shape[0], 1)
        target_module = LinearMatrixModule(dim, batch_shape).to(dtype)
        with torch.no_grad():
            target_module.omega.copy_(true_omega)
        
        y0 = torch.randn(*batch_shape, dim, dtype=dtype)
        t_span = torch.linspace(0, math.pi, 10, dtype=dtype)

        with torch.no_grad():
            y_target = magnus_odeint(target_module, y0, t_span)

        # 2. Create a learnable module
        learnable_module = LinearMatrixModule(dim, batch_shape).to(dtype)
        with torch.no_grad():
            learnable_module.omega.fill_(0.1)

        # 3. Optimize the learnable module's parameters
        optimizer = torch.optim.Adam(learnable_module.parameters(), lr=0.1)
        
        print("\nTesting nn.Module optimization...")
        
        # First step
        optimizer.zero_grad()
        y_pred_initial = magnus_odeint_adjoint(learnable_module, y0, t_span)
        loss_initial = torch.mean((y_pred_initial - y_target)**2)
        loss_initial.backward()
        optimizer.step()
        
        # Check that gradients were computed
        self.assertIsNotNone(learnable_module.omega.grad)
        print(f"Iter 000 | Loss: {loss_initial.item():.6f}")

        # Run a few more steps
        for i in range(1, 10):
            optimizer.zero_grad()
            y_pred = magnus_odeint_adjoint(learnable_module, y0, t_span)
            loss = torch.mean((y_pred - y_target)**2)
            loss.backward()
            optimizer.step()

        print(f"Iter 009 | Loss: {loss.item():.6f}")

        # 4. Check that the loss has decreased, proving optimization is working.
        self.assertLess(loss.item(), loss_initial.item())
        print("Optimization test passed: Gradients are present and loss is decreasing.")

if __name__ == '__main__':
    unittest.main()