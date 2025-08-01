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
    
    def __init__(self, ts: Tensor, ys: Tensor, order: int, A_func: Callable, method: str):
        """
        Initialize dense output interpolator.
        
        Args:
            ts: Tensor of times
            ys: Tensor of states
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
        y0 = self.ys[..., indices, :]

        # Calculate the new step size h_new for each point
        h_new = t_batch - t0

        # Perform a single integration step for each point
        y_interp = self.integrator(self.A_func, t0, h_new, y0)
        
        return y_interp

class CollocationDenseOutput:
    def __init__(self, ts: Tensor, ys: Union[None, Tensor] = None, t_nodes_traj: Union[None, Tensor] = None, A_nodes_traj: Union[None, Tensor] = None, g_nodes_traj: Union[None, Tensor] = None, order: int = None, dense_mode: str = 'precompute', precomputed_P: Union[None, Tensor] = None):
        self.order = order
        self.dense_mode = dense_mode
        self.ys = ys # [*batch_shape, n_intervals+1, dim]
        self.ts = ts # [n_intervals+1]

        # If precomputed_P is provided, use it directly
        if precomputed_P is not None:
            self.P = precomputed_P
            self.t_nodes_traj = None
            self.A_nodes_traj = None
            self.g_nodes_traj = None
            self.dense_mode = 'precompute'  # Force precompute mode when P is provided
            return

        if self.ts[0] > self.ts[-1]:
            self.ts = torch.flip(self.ts, dims=[0])
            self.ys = torch.flip(self.ys, dims=[-2])
            t_nodes_traj = torch.flip(t_nodes_traj, dims=[-1])
            A_nodes_traj = torch.flip(A_nodes_traj, dims=[-3])
            if g_nodes_traj is not None:
                g_nodes_traj = torch.flip(g_nodes_traj, dims=[-2])
        self.hs = ts[1:] - ts[:-1] # [n_intervals]

        ode_batch_shape = self.ys.shape[:-2]
        t_batch_shape = (self.ts.shape[0] - 1,)
        dim = self.ys.shape[-1]
        n_stages = t_nodes_traj.shape[0]

        if self.dense_mode == 'precompute':
            t0 = self.ts[:-1]
            y0 = self.ys[..., :-1, :]
            y1 = self.ys[..., 1:, :]
            h = self.hs

            self.P = _solve_collocation_system(
                y0, y1, h, t_nodes_traj-t0, A_nodes_traj, g_nodes_traj, n_stages, dim, 
                ode_batch_shape, t_batch_shape
            )
            self.t_nodes_traj = None
            self.A_nodes_traj = None
            self.g_nodes_traj = None
        else:
            self.t_nodes_traj = t_nodes_traj # [n_stages, n_intervals]
            self.A_nodes_traj = A_nodes_traj # [*batch_shape, n_stages, n_intervals, dim, dim]
            self.g_nodes_traj = g_nodes_traj # [*batch_shape, n_stages, n_intervals, dim]
            self.P = torch.zeros(self.ys.shape[:-2] + t_batch_shape + (n_stages+2, dim), dtype=ys.dtype, device=ys.device)
            self.P_available = torch.zeros(t_batch_shape, dtype=torch.bool, device=ys.device)

    def __call__(self, t_batch: Tensor) -> Tensor:
        """
        Evaluate solution at given time points using pre-computed data.
        """
        t_batch = torch.as_tensor(t_batch, dtype=self.ts.dtype, device=self.ts.device)
        indices = torch.searchsorted(self.ts, t_batch, right=True) - 1
        indices = torch.clamp(indices, max=len(self.ts) - 2)

        ode_batch_shape = self.P.shape[:-3]
        t_batch_shape = indices.shape
        dim = self.P.shape[-1]
        n_coeffs = self.P.shape[-2]
        n_stages = n_coeffs - 2

        if self.dense_mode == 'ondemand':
            unique_indices = torch.unique(indices)
            not_avail_indices = unique_indices[~self.P_available[unique_indices]]
            if not_avail_indices.numel():
                t0 = self.ts[not_avail_indices]
                y0 = self.ys[..., not_avail_indices, :]
                y1 = self.ys[..., not_avail_indices + 1, :]
                h = self.hs[not_avail_indices]
                t_nodes = self.t_nodes_traj[:, not_avail_indices] - t0
                A_nodes = self.A_nodes_traj[..., not_avail_indices, :, :]
                g_nodes = self.g_nodes_traj[..., not_avail_indices, :] if self.g_nodes_traj is not None else None

                coeffs = _solve_collocation_system(
                    y0, y1, h, t_nodes, A_nodes, g_nodes, n_stages, dim, 
                    ode_batch_shape, not_avail_indices.shape
                )
                self.P[..., not_avail_indices, :, :] = coeffs
                self.P_available[not_avail_indices] = True
            
        C = self.P[..., indices, :, :]
        t0 = self.ts[indices]

        # Evaluate the polynomial with the computed coefficients
        # t_eval should be broadcastable to batch_shape
        t_eval = (t_batch - t0).view((1,) * len(ode_batch_shape) + t_batch_shape)

        # y_interp should have shape (*batch_shape, dim)
        y_interp = torch.zeros(ode_batch_shape + t_batch_shape + (dim,), dtype=self.ts.dtype, device=self.ts.device)
        for j in range(n_coeffs):
            y_interp += C[..., j, :] * torch.pow(t_eval.unsqueeze(-1), j)
            
        return y_interp
        
def _solve_collocation_system(y0, y1, h, t_nodes, A_nodes, g_nodes, n_stages, dim, ode_batch_shape, t_batch_shape):
    """
    Solves the linear system to find polynomial coefficients for collocation.
    This is a helper function to keep the main __call__ method cleaner.
    """
    # Define system parameters
    batch_shape = ode_batch_shape + t_batch_shape
    poly_degree = n_stages + 1
    n_coeffs = poly_degree + 1

    # Build M and D for the linear system
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
    power = torch.pow(h.reshape(*t_batch_shape, 1, 1).expand(*t_batch_shape, 1, n_coeffs), torch.arange(n_coeffs, device=h.device).expand(*t_batch_shape, 1, n_coeffs)).unsqueeze(-1)
    M[..., 1, :, :, :] *= power
    D[..., dim:2*dim] = y1

    # Eqs 3 to n+2: Collocation constraints y'(t_i) = A(t_i)y(t_i) + g(t_i)
    # Formula: M_k0 = -A(t_i), M_kj = (j*t_i^(j-1)*I - t_i^j*A(t_i)) for j>0
    # D_k = g(t_i)
    if t_batch_shape:
        t_nodes = t_nodes.transpose(0, 1)
        A_nodes = A_nodes.transpose(-3, -4)
        if g_nodes is not None:
            g_nodes = g_nodes.transpose(-2, -3)

    power = torch.pow(t_nodes.reshape(*t_batch_shape, n_stages, 1, 1).expand(*t_batch_shape, n_stages, 1, n_coeffs), torch.arange(n_coeffs, device=t_nodes.device).expand(*t_batch_shape, n_stages, 1, n_coeffs))
    coeff = (torch.arange(n_coeffs, device=power.device)[1:] * power[..., :-1]).unsqueeze(-1)
    M[..., 2:, :, 1:, :] *= coeff
    M[..., 2:, :, 0, :] = 0.0
    M[..., 2:, :, :, :] -= A_nodes.unsqueeze(-2) * power.unsqueeze(-1)
    
    M = M.reshape(*batch_shape, n_coeffs*dim, n_coeffs*dim)
    if g_nodes is not None:
        D[..., 2*dim:] = g_nodes.flatten(start_dim=-2)
        
    # Solve for coefficients
    C_flat = torch.linalg.solve(M, D)
    C = C_flat.reshape(batch_shape + (n_coeffs, dim))
    return C

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
        ts=merged_t_grid,
        ys=merged_y_states,
        order=first_output.order,
        A_func=first_output.A_func,
        method=method
    )

def _merge_collocation_dense_outputs(dense_outputs: List['CollocationDenseOutput'], dense_mode) -> 'CollocationDenseOutput':
    """Merge CollocationDenseOutput instances."""
    first_output = dense_outputs[0]
    if first_output.dense_mode == "precompute":
        merged_ts = [first_output.ts]
        merged_P = [dense_outputs[0].P]

        for i in range(1, len(dense_outputs)):
            next_output = dense_outputs[i]
            merged_P.append(next_output.P)
            merged_ts.append(next_output.ts[1:])
        merged_P_tensor = torch.cat(merged_P, dim=-3)
        merged_t_grid = torch.cat(merged_ts, dim=0)
        return CollocationDenseOutput(
            ts=merged_t_grid,
            dense_mode="precompute",
            precomputed_P=merged_P_tensor
        )

    
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
        ts=merged_t_grid,
        ys=merged_y_states,
        t_nodes_traj=merged_t_nodes_traj,
        A_nodes_traj=merged_A_nodes_traj,
        g_nodes_traj=merged_g_nodes_traj,
        order=first_output.order,
        dense_mode=dense_mode
    )
