"""
This test file verifies that all code examples provided in the README.md file
are correct and runnable.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for testing
import matplotlib.pyplot as plt
from torch_linode.solvers import odeint, odeint_adjoint

# --- Test for the Homogeneous System Example ---
def test_homogeneous_example():
    """Tests the MyHomogeneousSystem example from the README."""
    class MyHomogeneousSystem(nn.Module):
        def __init__(self, A):
            super().__init__()
            self.A = A

        def forward(self, t):
            A_view = self.A.view(*self.A.shape[:-2], *((1,) * t.ndim), *self.A.shape[-2:])
            return A_view.expand(*self.A.shape[:-2], *t.shape, *self.A.shape[-2:])

    A = torch.tensor([[[0., -1.], [1., 0.]]]) # Batch of 1
    y0 = torch.tensor([[1., 0.]])
    t_span = torch.linspace(0, 1, 10)
    system = MyHomogeneousSystem(A)
    
    solution = odeint(system, y0, t_span)
    assert solution.shape == (1, 10, 2)

# --- Test for the Full Learning Example ---
def test_learning_example():
    """Tests the full "Learning an Unknown System" example from the README."""
    class LearnableLinearODE(nn.Module):
        def __init__(self, dim=2):
            super().__init__()
            self.A = nn.Parameter(torch.randn(dim, dim))

        def forward(self, t):
            A_view = self.A.view(*self.A.shape[:-2], *((1,) * t.ndim), *self.A.shape[-2:])
            return A_view.expand(*self.A.shape[:-2], *t.shape, *self.A.shape[-2:])

    class GroundTruthODE(nn.Module):
        def __init__(self, A_true):
            super().__init__()
            self.A = A_true

        def forward(self, t):
            A_view = self.A.view(*self.A.shape[:-2], *((1,) * t.ndim), *self.A.shape[-2:])
            return A_view.expand(*self.A.shape[:-2], *t.shape, *self.A.shape[-2:])

    A_true = torch.tensor([[-0.1, -1.0], [1.0, -0.1]])
    y0 = torch.tensor([2.0, 0.0])
    t_span = torch.linspace(0, 10, 100)
    true_system = GroundTruthODE(A_true)
    with torch.no_grad():
        y_true = odeint(true_system, y0, t_span)

    model = LearnableLinearODE(dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        model.A.data = A_true + torch.randn_like(A_true) * 0.3

    initial_loss = loss_fn(odeint_adjoint(model, y0, t_span), y_true)

    for _ in range(10):
        optimizer.zero_grad()
        y_pred = odeint_adjoint(model, y0, t_span)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
    
    final_loss = loss_fn(odeint_adjoint(model, y0, t_span), y_true)
    assert final_loss < initial_loss

    try:
        with torch.no_grad():
            y_pred_final = odeint(model, y0, t_span)
        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, 0], y_true[:, 1], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(y_pred_final[:, 0], y_pred_final[:, 1], 'b--', label='Learned Trajectory')
        plt.title("Phase Portrait: Learning an ODE System")
        plt.xlabel("State 1")
        plt.ylabel("State 2")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.close()
    except Exception as e:
        assert False, f"Visualization code failed with error: {e}"

# --- Test for the Functional API with odeint_adjoint Example ---
def test_functional_adjoint_example():
    """Tests the functional API example with odeint_adjoint from the README."""
    def functional_system(t, params):
        A = params
        # A can have shape (*batch_shape, dim, dim)
        # t has shape (*t_shape)
        # We must return A_t with broadcast-compatible shapes.
        # A_t should be (*batch_shape, *t_shape, dim, dim)
        A_view = A.view(*A.shape[:-2], *((1,) * t.ndim), *A.shape[-2:])
        return A_view.expand(*A.shape[:-2], *t.shape, *A.shape[-2:])

    A_true = torch.tensor([[-0.1, -1.0], [1.0, -0.1]])
    y0 = torch.tensor([2.0, 0.0])
    t_span = torch.linspace(0, 10, 100)

    with torch.no_grad():
        y_true = odeint_adjoint(functional_system, y0, t_span, params=A_true)

    A_learnable = torch.randn(2, 2, requires_grad=True)
    with torch.no_grad():
        A_learnable.data = A_true + torch.randn_like(A_true) * 0.01
    optimizer = optim.Adam([A_learnable], lr=0.01)
    loss_fn = nn.MSELoss()

    initial_loss = loss_fn(odeint_adjoint(functional_system, y0, t_span, params=A_learnable), y_true)

    for _ in range(10):
        optimizer.zero_grad()
        y_pred = odeint_adjoint(functional_system, y0, t_span, params=A_learnable)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(odeint_adjoint(functional_system, y0, t_span, params=A_learnable), y_true)
    assert final_loss < initial_loss

# --- Test for the Non-Homogeneous Example ---
def test_readme_non_homogeneous_example():
    """Tests the "Solving a Non-Homogeneous System" example from the README."""
    class ForcedOscillator(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.tensor([[0., 1.], [-1., 0.]])

        def forward(self, t):
            A_view = self.A.view(*self.A.shape[:-2], *((1,) * t.ndim), *self.A.shape[-2:])
            A_t = A_view.expand(*self.A.shape[:-2], *t.shape, *self.A.shape[-2:])
            g_t = torch.stack([torch.sin(t), torch.cos(t)], dim=-1).expand(*self.A.shape[:-2], *t.shape, self.A.shape[-1])
            return A_t, g_t

    system = ForcedOscillator()
    y0 = torch.tensor([1.0, 0.0])
    t_span = torch.linspace(0, 20, 200)

    with torch.no_grad():
        solution = odeint(system, y0, t_span)

    assert solution.shape == (200, 2)

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(t_span, solution[:, 0], label='y_1(t)')
        plt.plot(t_span, solution[:, 1], label='y_2(t)')
        plt.title("Solution of a Forced Oscillator")
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.legend()
        plt.grid(True)
        plt.close()
    except Exception as e:
        assert False, f"Visualization code failed with error: {e}"