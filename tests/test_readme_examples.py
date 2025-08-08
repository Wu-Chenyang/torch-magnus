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
from torch_linode.solvers import odeint

# --- Test for the Homogeneous System Example ---
def test_homogeneous_example():
    """Tests the MyHomogeneousSystem example from the README."""
    class MyHomogeneousSystem(nn.Module):
        def __init__(self, A):
            super().__init__()
            self.A = A

        def forward(self, t):
            t_shape = t.shape
            A_view = self.A.view(*self.A.shape[:1], *((1,) * len(t_shape)), *self.A.shape[1:])
            return A_view.expand(*self.A.shape[:1], *t_shape, *self.A.shape[1:])

    A = torch.tensor([[[0., -1.], [1., 0.]]]) # Batch of 1
    y0 = torch.tensor([[1., 0.]])
    t_span = torch.linspace(0, 1, 10)
    system = MyHomogeneousSystem(A)
    
    # Ensure it runs without errors
    solution = odeint(system, y0, t_span)
    assert solution.shape == (1, 10, 2)

# --- Test for the Non-Homogeneous System Example ---
def test_non_homogeneous_example():
    """Tests the MyNonHomogeneousSystem example from the README."""
    class MyNonHomogeneousSystem(nn.Module):
        def __init__(self, A):
            super().__init__()
            self.A = A

        def forward(self, t):
            t_shape = t.shape
            A_view = self.A.view(*self.A.shape[:1], *((1,) * len(t_shape)), *self.A.shape[1:])
            return A_view.expand(*self.A.shape[:1], *t_shape, *self.A.shape[1:])

        def g(self, t):
            return torch.sin(t).unsqueeze(-1).expand(*t.shape, 2)

    A = torch.tensor([[[0., -1.], [1., 0.]]]) # Batch of 1
    y0 = torch.tensor([[1., 0.]])
    t_span = torch.linspace(0, 1, 10)
    system = MyNonHomogeneousSystem(A)
    
    # Ensure it runs without errors
    solution = odeint(system, y0, t_span)
    assert solution.shape == (1, 10, 2)

# --- Test for the Full Learning Example ---
def test_learning_example():
    """Tests the full "Learning an Unknown System" example from the README."""
    # 1. Define the modules
    class LearnableLinearODE(nn.Module):
        def __init__(self, dim=2):
            super().__init__()
            self.A = nn.Parameter(torch.randn(dim, dim))

        def forward(self, t):
            t_shape = t.shape
            A_view = self.A.view(*((1,) * len(t_shape)), *self.A.shape)
            return A_view.expand(*t_shape, *self.A.shape)

    class GroundTruthODE(nn.Module):
        def __init__(self, A_true):
            super().__init__()
            self.A = A_true

        def forward(self, t):
            t_shape = t.shape
            A_view = self.A.view(*((1,) * len(t_shape)), *self.A.shape)
            return A_view.expand(*t_shape, *self.A.shape)

    # 2. Generate ground truth data
    A_true = torch.tensor([[-0.1, -1.0], [1.0, -0.1]])
    y0 = torch.tensor([2.0, 0.0])
    t_span = torch.linspace(0, 10, 100)
    true_system = GroundTruthODE(A_true)
    with torch.no_grad():
        y_true = odeint(true_system, y0, t_span)

    # 3. Set up and train the model
    model = LearnableLinearODE(dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        model.A.data = A_true + torch.randn_like(A_true) * 0.3

    initial_loss = loss_fn(odeint(model, y0, t_span), y_true)

    for _ in range(10): # A few steps are enough to see loss decrease
        optimizer.zero_grad()
        y_pred = odeint(model, y0, t_span)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
    
    final_loss = loss_fn(odeint(model, y0, t_span), y_true)

    # Assert that the loss has decreased, proving learning is happening
    assert final_loss < initial_loss

    # 4. Test that the visualization code runs without error
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
        # plt.show() # Do not show plot in tests
        plt.close() # Close the figure to free memory
    except Exception as e:
        assert False, f"Visualization code failed with error: {e}"


# --- Test for the Functional API with odeint_adjoint Example ---
def test_functional_adjoint_example():
    """Tests the functional API example with odeint_adjoint from the README."""
    from torch_linode.solvers import odeint_adjoint

    # 1. Define the system as a function
    def functional_system(t, params):
        A = params
        t_shape = t.shape
        A_view = A.view(*((1,) * len(t_shape)), *A.shape)
        return A_view.expand(*t_shape, *A.shape)

    # 2. Set up the learning problem
    A_true = torch.tensor([[-0.1, -1.0], [1.0, -0.1]])
    y0 = torch.tensor([2.0, 0.0])
    t_span = torch.linspace(0, 10, 100)

    with torch.no_grad():
        y_true = odeint_adjoint(functional_system, y0, t_span, params=A_true)

    # 3. Initialize learnable parameters and optimizer
    A_learnable = torch.randn(2, 2, requires_grad=True)
    with torch.no_grad():
        A_learnable.data = A_true + torch.randn_like(A_true) * 0.01
    optimizer = optim.Adam([A_learnable], lr=0.01)
    loss_fn = nn.MSELoss()

    initial_loss = loss_fn(odeint_adjoint(functional_system, y0, t_span, params=A_learnable), y_true)

    # 4. Training loop
    for _ in range(10):
        optimizer.zero_grad()
        y_pred = odeint_adjoint(functional_system, y0, t_span, params=A_learnable)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(odeint_adjoint(functional_system, y0, t_span, params=A_learnable), y_true)

    assert final_loss < initial_loss
