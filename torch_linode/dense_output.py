import torch
from typing import Callable, Union, List
from .stepper import Magnus2nd, Magnus4th, Magnus6th, Collocation
from .butcher import GL2, GL4, GL6
Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Dense Output (Continuous Extension)
# -----------------------------------------------------------------------------

class DenseOutputNaive:
    """
    Provides continuous interpolation between Magnus integration steps by re-running
    the integrator for a single step from the last grid point. It requires s extra function
    evaluations for each interpolation but maintains the 2s order accuracy of the solver.
    """
    
    def __init__(self, ys: Tensor, ts: Tensor, order: int, A_func: Callable, method: str):
        """
        Initialize dense output interpolator.
        
        Args:
            ys: Tensor of states
            ts: Tensor of times
            order: Order of Magnus integrator (2 or 4).
            A_func: The matrix function A(t) used for integration.
        """
        self.order = order
        self.A_func = A_func
        self.ys = ys
        self.ts = ts
        if self.ts[0] > self.ts[-1]:
             self.ts = torch.flip(self.ts, dims=[0])
             self.ys = torch.flip(self.ys, dims=[-2])

        if method == 'magnus':
            if self.order == 2: self.integrator = Magnus2nd()
            elif self.order == 4: self.integrator = Magnus4th()
            elif self.order == 6: self.integrator = Magnus6th()
            else: raise ValueError(f"Invalid order: {order} for Magnus")
        elif method == 'glrk':
            if self.order == 2: self.integrator = Collocation(GL2)
            elif self.order == 4: self.integrator = Collocation(GL4)
            elif self.order == 6: self.integrator = Collocation(GL6)
            else: raise ValueError(f"Invalid order: {order} for GLRK")
        else: raise ValueError(f"Invalid method: {method}")

    def __call__(self, t_batch: Tensor) -> Tensor:
        """
        Evaluate solution at given time points by performing a single integration
        step from the nearest previous time grid point.
        
        Args:
            t_batch: Time points of shape (*time_shape,)
            
        Returns:
            Solution tensor of shape (*batch_shape, *time_shape, dim)
        """
        # Find the interval each t_batch point falls into
        indices = torch.searchsorted(self.ts, t_batch, right=True) - 1
        
        # Get the starting points (t0, y0) for each interpolation
        t0 = self.ts[indices]
        if indices.ndim == 0:
            y0 = self.ys[..., indices, :]
        else:
            y0 = torch.gather(self.ys, -2, indices.unsqueeze(-1).expand((*self.ys.shape[:-2], indices.shape[0], self.ys.shape[-1])))

        # Calculate the new step size h_new for each point
        h_new = t_batch - t0

        # Perform a single integration step for each point
        y_interp = self.integrator(self.A_func, t0, h_new, y0)
        
        return y_interp

class CollocationDenseOutput:
    def __init__(self, ys: Tensor, ts: Tensor, t_nodes_traj: Tensor, A_nodes_traj: Tensor, g_nodes_traj: Union[None, Tensor], order: int):
        self.order = order
        self.ys = ys # [*batch_shape, n_intervals+1, dim]
        self.ts = ts # [n_intervals+1]
        self.hs = ts[1:] - ts[:-1] # [n_intervals]
        self.t_nodes_traj = t_nodes_traj # [s_nodes, n_intervals]
        self.A_nodes_traj = A_nodes_traj # [*batch_shape, s_nodes, n_intervals, dim, dim]
        self.g_nodes_traj = g_nodes_traj # [*batch_shape, s_nodes, n_intervals, dim]

        if self.ts[0] > self.ts[-1]:
            self.ts = torch.flip(self.ts, dims=[0])
            self.ys = torch.flip(self.ys, dims=[-2])
            self.hs = torch.flip(self.hs, dims=[0])
            self.t_nodes_traj = torch.flip(self.t_nodes_traj, dims=[-1])
            self.A_nodes_traj = torch.flip(self.A_nodes_traj, dims=[-3])
            if self.g_nodes_traj is not None:
                self.g_nodes_traj = torch.flip(self.g_nodes_traj, dims=[-2])


    def __call__(self, t_batch: Tensor) -> Tensor:
        """
        Evaluate solution at given time points using pre-computed data.
        """
        t_batch = torch.as_tensor(t_batch, dtype=self.ts.dtype, device=self.ts.device)
        indices = torch.searchsorted(self.ts, t_batch, right=True) - 1
        indices = torch.clamp(indices, 0, len(self.ts) - 2)

        t0 = self.ts[indices]
        h = self.hs[indices]
        if indices.ndim == 0:
            y0 = self.ys[..., indices, :]
            y1 = self.ys[..., 1+indices, :]
            t_nodes = self.t_nodes_traj[:, indices]
            A_nodes = self.A_nodes_traj[..., indices, :, :]
            g_nodes = self.g_nodes_traj[..., indices, :] if self.g_nodes_traj is not None else None
        else:
            y0 = torch.gather(self.ys, -2, indices.unsqueeze(-1).expand(self.ys.shape[:-2] + (indices.shape[0], self.ys.shape[-1])))
            y1 = torch.gather(self.ys, -2, 1+indices.unsqueeze(-1).expand(self.ys.shape[:-2] + (indices.shape[0], self.ys.shape[-1])))
            t_nodes = torch.gather(self.t_nodes_traj, -1, indices.expand((self.t_nodes_traj.shape[0], indices.shape[0])))
            A_nodes = torch.gather(self.A_nodes_traj, -3, indices.unsqueeze(-1).unsqueeze(-1).expand(self.A_nodes_traj.shape[:-3] + (indices.shape[0],) + self.A_nodes_traj.shape[-2:]))
            g_nodes = torch.gather(self.g_nodes_traj, -2, indices.unsqueeze(-1).expand(self.g_nodes_traj.shape[:-2] + (indices.shape[0], self.g_nodes_traj.shape[-1]))) if self.g_nodes_traj is not None else None
        
        # 2. Define system parameters
        # ode_batch_shape: The batch shape of the ODE system itself.
        # t_batch_shape: The batch shape of the evaluation time points.
        # batch_shape: The combined batch shape for the output.
        ode_batch_shape = self.ys.shape[:-2]
        t_batch_shape = t_batch.shape
        batch_shape = ode_batch_shape + t_batch_shape
        dim = self.ys.shape[-1]
        n_stages = self.t_nodes_traj.shape[0]
        poly_degree = n_stages + 1
        n_coeffs = poly_degree + 1

        # 3. Build M and D for the linear system
        # M should have shape (*batch_shape, n_coeffs * dim, n_coeffs * dim)
        # D should have shape (*batch_shape, n_coeffs * dim)
        eye = torch.eye(dim, dtype=y0.dtype, device=y0.device)
        M = eye.repeat(*batch_shape, n_coeffs, n_coeffs).reshape(*batch_shape, n_coeffs, dim, n_coeffs, dim)
        D = torch.zeros(batch_shape + (n_coeffs * dim,), dtype=y0.dtype, device=y0.device)

        # Eq 1: y(t_a) = y_0
        # Formula: M_1j = t_a^j * I. With t_a=0, this is I for j=0 and 0 otherwise.
        # D_1 = y_a
        M[..., 0, :, 1:, :] = 0.0
        D[..., :dim] = y0

        # Eq 2: y(t_b) = y_1
        # Formula: M_2j = t_b^j * I. With t_b=h.
        # D_2 = y_b
        power = torch.pow(h.reshape(*t_batch_shape, 1, 1).expand(*t_batch_shape, 1, n_coeffs), torch.arange(n_coeffs).expand(*t_batch_shape, 1, n_coeffs)).unsqueeze(-1)
        M[..., 1, :, :, :] *= power
        D[..., dim:2*dim] = y1

        # Eqs 3 to n+2: Collocation constraints y'(t_i) = A(t_i)y(t_i) + g(t_i)
        # Formula: M_k0 = -A(t_i), M_kj = (j*t_i^(j-1)*I - t_i^j*A(t_i)) for j>0
        # D_k = g(t_i)
        if t_batch.ndim == 1:
            t_nodes = t_nodes.transpose(0, 1)
            A_nodes = A_nodes.transpose(-3, -4)
            if g_nodes is not None:
                g_nodes = g_nodes.transpose(-2, -3)

        power = torch.pow(t_nodes.reshape(*t_batch_shape, n_stages, 1, 1).expand(*t_batch_shape, n_stages, 1, n_coeffs), torch.arange(n_coeffs).expand(*t_batch_shape, n_stages, 1, n_coeffs))
        power[..., :-1]
        coeff = torch.arange(n_coeffs)[1:] * power[..., :-1]
        M[..., 2:, :, 1:, :] *= coeff.unsqueeze(-1)
        M[..., 2:, :, 0, :] = 0.0
        M[..., 2:, :, :, :] -= A_nodes.unsqueeze(-2) * power.unsqueeze(-1)
        M = M.reshape(*batch_shape, n_coeffs*dim, n_coeffs*dim)
        if g_nodes is not None:
            D[..., 2*dim:] = g_nodes.flatten(start_dim=-2)
            
        # 4. Solve for coefficients and evaluate the polynomial
        # C should have shape (*batch_shape, n_coeffs, dim)
        C_flat = torch.linalg.solve(M, D)
        C = C_flat.reshape(batch_shape + (n_coeffs, dim))

        # t_eval should be broadcastable to batch_shape
        t_eval = (t_batch - t0).view((1,) * len(ode_batch_shape) + t_batch_shape)

        # y_interp should have shape (*batch_shape, dim)
        y_interp = torch.zeros(batch_shape + (dim,), dtype=y0.dtype, device=y0.device)
        for j in range(n_coeffs):
            y_interp += C[..., j, :] * torch.pow(t_eval.unsqueeze(-1), j)
            
        return y_interp
        
def merge_dense_outputs(dense_outputs: List[Union['DenseOutputNaive', 'CollocationDenseOutput']]) -> Union['DenseOutputNaive', 'CollocationDenseOutput']:
    """
    Merge multiple dense output instances with connected time intervals.
    
    Args:
        dense_outputs: List of dense output instances to merge
        
    Returns:
        A new merged dense output instance
        
    Raises:
        ValueError: If intervals are not properly connected or instances are incompatible
    """
    if not dense_outputs:
        raise ValueError("Cannot merge empty list of dense outputs")
    
    if len(dense_outputs) == 1:
        return dense_outputs[0]
    
    # Check that all instances are of the same type
    output_type = type(dense_outputs[0])
    if not all(isinstance(output, output_type) for output in dense_outputs):
        raise ValueError("All dense outputs must be of the same type")
    
    # Verify intervals are connected
    for i in range(len(dense_outputs) - 1):
        current_end = dense_outputs[i].ts[-1]
        next_start = dense_outputs[i + 1].ts[0]
        if not torch.allclose(current_end, next_start, atol=1e-10):
            raise ValueError(f"Intervals are not connected: gap between {current_end} and {next_start}")
    
    # Merge based on the type
    if output_type == DenseOutputNaive:
        return _merge_naive_dense_outputs(dense_outputs)
    elif output_type == CollocationDenseOutput:
        return _merge_collocation_dense_outputs(dense_outputs)
    else:
        raise ValueError(f"Unknown dense output type: {output_type}")

def _merge_naive_dense_outputs(dense_outputs: List['DenseOutputNaive']) -> 'DenseOutputNaive':
    """Merge DenseOutputNaive instances."""
    first_output = dense_outputs[0]
    
    # Collect time grids and states, removing duplicate boundary points
    merged_ts = [dense_outputs[0].ts]
    merged_ys = [dense_outputs[0].ys]
    
    for i in range(1, len(dense_outputs)):
        # Skip the first time point of subsequent intervals (it's duplicate)
        merged_ts.append(dense_outputs[i].ts[1:])
        merged_ys.append(dense_outputs[i].ys[..., 1:, :])
    
    # Concatenate along time dimension
    merged_t_grid = torch.cat(merged_ts, dim=0)
    merged_y_states = torch.cat(merged_ys, dim=-2)
    
    # Create new merged instance
    # We need to determine the method from the integrator type
    if isinstance(first_output.integrator, Magnus2nd):
        method = 'magnus'
    elif isinstance(first_output.integrator, Magnus4th):
        method = 'magnus'
    elif isinstance(first_output.integrator, Magnus6th):
        method = 'magnus'
    elif isinstance(first_output.integrator, Collocation):
        method = 'glrk'
    else:
        raise ValueError("Unknown integrator type")
    
    return DenseOutputNaive(
        ys=merged_y_states,
        ts=merged_t_grid,
        order=first_output.order,
        A_func=first_output.A_func,
        method=method
    )


def _merge_collocation_dense_outputs(dense_outputs: List['CollocationDenseOutput']) -> 'CollocationDenseOutput':
    """Merge CollocationDenseOutput instances."""
    first_output = dense_outputs[0]
    
    # Collect time grids, states, and cached data
    merged_ts = [first_output.ts]
    merged_ys = [first_output.ys]
    merged_t_nodes = [first_output.t_nodes_traj]
    merged_A_nodes = [first_output.A_nodes_traj]
    
    has_g_nodes = first_output.g_nodes_traj is not None
    if has_g_nodes:
        merged_g_nodes = [first_output.g_nodes_traj]

    for i in range(1, len(dense_outputs)):
        next_output = dense_outputs[i]
        
        # Skip the first time point of subsequent intervals (it's a duplicate)
        merged_ts.append(next_output.ts[1:])
        merged_ys.append(next_output.ys[..., 1:, :])
        
        # For trajectory data, the number of intervals is ts.shape[0] - 1
        # The shapes are:
        # t_nodes_traj: [s_nodes, n_intervals]
        # A_nodes_traj: [*batch, s_nodes, n_intervals, dim, dim]
        # g_nodes_traj: [*batch, s_nodes, n_intervals, dim]
        
        # We concatenate along the interval dimension
        merged_t_nodes.append(next_output.t_nodes_traj)
        merged_A_nodes.append(next_output.A_nodes_traj)
        if has_g_nodes:
            if next_output.g_nodes_traj is None:
                 raise ValueError("Inconsistent g_nodes_traj in dense_outputs to merge.")
            merged_g_nodes.append(next_output.g_nodes_traj)

    # Concatenate along appropriate dimensions
    merged_t_grid = torch.cat(merged_ts, dim=0)
    merged_y_states = torch.cat(merged_ys, dim=-2)
    
    # Concatenate trajectory data along the interval dimension
    merged_t_nodes_traj = torch.cat(merged_t_nodes, dim=-1)
    merged_A_nodes_traj = torch.cat(merged_A_nodes, dim=-3)
    
    merged_g_nodes_traj = None
    if has_g_nodes:
        merged_g_nodes_traj = torch.cat(merged_g_nodes, dim=-2)

    return CollocationDenseOutput(
        ys=merged_y_states,
        ts=merged_t_grid,
        t_nodes_traj=merged_t_nodes_traj,
        A_nodes_traj=merged_A_nodes_traj,
        g_nodes_traj=merged_g_nodes_traj,
        order=first_output.order
    )
