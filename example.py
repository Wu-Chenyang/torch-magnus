import torch
from torch_magnus import magnus_odeint

def main():
    """Example of solving a simple time-dependent ODE."""
    # 1. Define the problem: dy/dt = A(t) * y
    # Let's define a simple time-dependent matrix A(t)
    # A(t) = [[0, -t], [t, 0]]
    # This corresponds to a 2D rotation with increasing angular velocity.
    def A_func(t, params):
        # This function can be as complex as needed.
        # It can be a simple function or a full torch.nn.Module.
        t_tensor = torch.as_tensor(t, dtype=torch.float64)
        A = torch.zeros(*t_tensor.shape, 2, 2, dtype=torch.float64)
        A[..., 0, 1] = -t
        A[..., 1, 0] = t
        return A

    # 2. Set initial conditions and time points
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)  # Initial state
    t_span = torch.linspace(0, 2 * torch.pi, 10)      # Time points to evaluate

    print("Solving ODE: dy/dt = A(t) * y")
    print(f"Initial condition y(0): {y0.numpy()}")
    print(f"Time points: {t_span.numpy()}")

    # 3. Call the solver
    # We use magnus_odeint, which is imported from our new package.
    # Since A_func has no trainable parameters, we pass `params=None`.
    solution_trajectory = magnus_odeint(
        A_func_or_module=A_func,
        y0=y0,
        t=t_span,
        params=None, # No trainable parameters in this example
        order=4,     # Use a 4th-order integrator
        rtol=1e-6,
        atol=1e-8
    )

    # 4. Print the results
    print("\n--- Solution ---")
    for i, t in enumerate(t_span):
        print(f"t = {t:.2f}, y(t) = {solution_trajectory[i].numpy()}")

    # The exact solution is y(t) = [cos(t^2/2), sin(t^2/2)]
    # Let's check the final point for accuracy
    t_final = t_span[-1]
    y_exact_final = torch.tensor([torch.cos(t_final**2 / 2), torch.sin(t_final**2 / 2)])
    y_computed_final = solution_trajectory[-1]
    error = torch.norm(y_computed_final - y_exact_final)

    print("\n--- Verification ---")
    print(f"Computed y({t_final:.2f}) = {y_computed_final.numpy()}")
    print(f"Exact y({t_final:.2f})    = {y_exact_final.numpy()}")
    print(f"Final error: {error.item():.2e}")

if __name__ == "__main__":
    main()
