"""
A runnable and meaningful example of a "Neural ODE" using the torch_linode library.

This example demonstrates how to learn the parameters of an ODE system from data.

1.  **Ground Truth**: We define a "true" linear ODE system (a damped spiral)
    and generate a trajectory from it. This serves as our training data.

2.  **Learnable Model**: We create an instance of a `LearnableLinearODE` with
    randomly initialized parameters (a matrix A).

3.  **Training**: We use a standard PyTorch training loop. In each step:
    a. We use `odeint` to solve for the trajectory predicted by our model.
    b. We calculate the Mean Squared Error between our model's trajectory
       and the ground truth trajectory.
    c. We use backpropagation (`loss.backward()`) and an optimizer to update
       our model's parameters (the matrix A).

4.  **Visualization**: We plot the ground truth trajectory alongside the model's
    predicted trajectory before and after training. This visually demonstrates
    that the model has successfully learned the underlying dynamics of the system.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_linode.solvers import odeint_adjoint as odeint
# from torch_linode.solvers import odeint

# 1. Define the ODE System Modules
class LearnableLinearODE(nn.Module):
    """
    Represents a linear ODE system dy/dt = A*y. The matrix A is learnable.
    """
    def __init__(self, dim=2):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim))

    def forward(self, t):
        t_shape = t.shape
        A_view = self.A.view(*((1,) * len(t_shape)), *self.A.shape)
        return A_view.expand(*t_shape, *self.A.shape)

class GroundTruthODE(nn.Module):
    """
    A fixed, non-learnable ODE system to generate our target data.
    """
    def __init__(self, A_true):
        super().__init__()
        self.A = A_true

    def forward(self, t):
        t_shape = t.shape
        A_view = self.A.view(*((1,) * len(t_shape)), *self.A.shape)
        return A_view.expand(*t_shape, *self.A.shape)

def run_learning_example():
    """
    Sets up and runs the ODE learning example.
    """
    # 2. Generate Ground Truth Data
    print("--- Step 1: Generating ground truth data ---")
    A_true = torch.tensor([[-0.1, -1.0],
                           [1.0, -0.1]])
    y0 = torch.tensor([2.0, 0.0])
    t_span = torch.linspace(0, 10, 100)

    # Create an instance of the ground truth system
    true_system = GroundTruthODE(A_true)
    
    with torch.no_grad():
        y_true = odeint(true_system, y0, t_span)
    print("Ground truth data generated.")

    # 3. Set up the Learning Problem
    print("--- Step 2: Setting up the learning problem ---")
    # Instantiate the model
    model = LearnableLinearODE(dim=2)

    # Initialize the model's parameters to be 'close' to the true values.
    # A good initialization is crucial to avoid local minima and ensure convergence
    # to the correct solution. Here, we add some noise to the true matrix.
    with torch.no_grad():
        noise = torch.randn_like(A_true) * 0.3  # Add some noise
        model.A.data = A_true + noise
    
    # A lower learning rate and more epochs are often needed for convergence.
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    print("Initial (random) matrix A:")
    print(model.A.data)

    # Get the model's initial prediction before training
    with torch.no_grad():
        y_pred_initial = odeint(model, y0, t_span)

    # 4. Run the Training Loop
    print("--- Step 3: Running the training loop ---")
    epochs = 100  # Increased epochs for better convergence
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get the model's current prediction
        y_pred = odeint(model, y0, t_span)
        
        # Calculate the loss
        loss = loss_fn(y_pred, y_true)
        
        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:  # Print progress less frequently
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    print("Training complete.")
    print("Learned matrix A:")
    print(model.A.data)
    print("True matrix A:")
    print(A_true)

    # Get the model's final prediction after training
    with torch.no_grad():
        y_pred_final = odeint(model, y0, t_span)

    # 5. Visualize the Results
    print("--- Step 4: Visualizing the results ---")
    plt.figure(figsize=(10, 8))
    
    # Plot phase portraits
    plt.subplot(2, 1, 1)
    plt.plot(y_true[:, 0], y_true[:, 1], 'g-', label='Ground Truth', linewidth=2)
    plt.plot(y_pred_initial[:, 0], y_pred_initial[:, 1], 'r--', label='Prediction (Before Training)')
    plt.plot(y_pred_final[:, 0], y_pred_final[:, 1], 'b-', label='Prediction (After Training)', linewidth=2)
    plt.title("Phase Portrait: Learning an ODE System")
    plt.xlabel("State 1 (y_0)")
    plt.ylabel("State 2 (y_1)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Plot states over time
    plt.subplot(2, 1, 2)
    plt.plot(t_span, y_true[:, 0], 'g-', label='Ground Truth y_0(t)', linewidth=2)
    plt.plot(t_span, y_pred_final[:, 0], 'b-', label='Learned y_0(t)')
    plt.plot(t_span, y_true[:, 1], 'g--', label='Ground Truth y_1(t)', linewidth=2)
    plt.plot(t_span, y_pred_final[:, 1], 'b--', label='Learned y_1(t)')
    plt.title("State Values Over Time")
    plt.xlabel("Time")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("Plot displayed. Close the plot window to exit.")

if __name__ == "__main__":
    # To run this example, you need to have matplotlib installed:
    # pip install matplotlib
    run_learning_example()