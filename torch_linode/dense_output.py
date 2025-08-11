import torch
from typing import Callable, Union, List
from .stepper import Magnus2nd, Magnus4th, Magnus6th, Collocation
from .butcher import GL2, GL4, GL6
Tensor = torch.Tensor
import scipy
import numpy as np
from functools import lru_cache

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
            t_batch: Time points of shape (*batch_shape, *t_shape) or (*t_shape)
            
        Returns:
            Solution tensor of shape (*batch_shape, *t_shape, dim)
        """
        t_batch = torch.as_tensor(t_batch, dtype=self.ts.dtype, device=self.ts.device)
        # 1. Disentangle batch and time dimensions from t_batch
        ode_batch_shape = self.ys.shape[:-2]
        
        was_scalar = (t_batch.ndim == 0)
        if was_scalar:
            t_batch = t_batch.unsqueeze(0)

        # Let's assume last dim is time
        t_time_shape = t_batch.shape[-1:]

        t_batch_broadcasted = t_batch.broadcast_to(*ode_batch_shape, *t_time_shape)

        # Find intervals
        indices = torch.searchsorted(self.ts, t_batch_broadcasted, right=True) - 1
        
        # Get start points t0, y0
        t0 = self.ts[indices]
        y0 = torch.gather(self.ys, -2, indices.unsqueeze(-1).expand(*indices.shape, self.ys.shape[-1]))
        
        # Calculate step size
        h_new = t_batch_broadcasted - t0

        # 4. Perform integration
        y_interp = self.integrator(self.A_func, t0, h_new, y0)
        
        if was_scalar:
            y_interp = y_interp.squeeze(-2)

        return y_interp

class CollocationDenseOutput:
    def __init__(self, ts: Tensor, ys: Union[None, Tensor] = None, t_nodes_traj: Union[None, Tensor] = None, A_nodes_traj: Union[None, Tensor] = None, g_nodes_traj: Union[None, Tensor] = None, order: int = None, dense_mode: str = 'precompute', precomputed_P: Union[None, Tensor] = None, interpolation_method: str = 'lifting'):
        self.order = order
        self.dense_mode = dense_mode
        self.interpolation_method = interpolation_method
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

            if self.interpolation_method == 'power':
                self.P = _solve_collocation_system(
                    y0, y1, h, t_nodes_traj-t0, A_nodes_traj, g_nodes_traj, n_stages, dim, 
                    ode_batch_shape, t_batch_shape
                )
            elif self.interpolation_method == 'lifting':
                self.P = _solve_collocation_system_lifting(
                    y0, y1, h, t_nodes_traj-t0, A_nodes_traj, g_nodes_traj, n_stages, dim, 
                    ode_batch_shape, t_batch_shape
                )
            else:
                raise ValueError(f"Unknown interpolation_method: {self.interpolation_method}")

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

        # 1. Disentangle batch and time dimensions from t_batch
        ode_batch_shape = self.P.shape[:-3]
        
        was_scalar = (t_batch.ndim == 0)
        if was_scalar:
            t_batch = t_batch.unsqueeze(0)

        # Let's assume last dim is time
        t_time_shape = t_batch.shape[-1:]

        # 2. Broadcast batch shapes

        t_batch_broadcasted = t_batch.broadcast_to(*ode_batch_shape, *t_time_shape)

        indices = torch.searchsorted(self.ts, t_batch_broadcasted, right=True) - 1
        indices = torch.clamp(indices, max=len(self.ts) - 2)

        dim = self.P.shape[-1]
        n_coeffs = self.P.shape[-2]
        n_stages = n_coeffs - 2

        if self.dense_mode == 'ondemand':
            unique_indices = torch.unique(indices.view(-1))
            not_avail_indices = unique_indices[~self.P_available[unique_indices]]
            if not_avail_indices.numel():
                t0 = self.ts[not_avail_indices]
                y0 = self.ys[..., not_avail_indices, :]
                y1 = self.ys[..., not_avail_indices + 1, :]
                h = self.hs[not_avail_indices]
                t_nodes = self.t_nodes_traj[:, not_avail_indices] - t0
                A_nodes = self.A_nodes_traj[..., not_avail_indices, :, :]
                g_nodes = self.g_nodes_traj[..., not_avail_indices, :] if self.g_nodes_traj is not None else None

                if self.interpolation_method == 'power':
                    coeffs = _solve_collocation_system(
                        y0, y1, h, t_nodes, A_nodes, g_nodes, n_stages, dim,
                        ode_batch_shape, not_avail_indices.shape
                    )
                elif self.interpolation_method == 'lifting':
                    coeffs = _solve_collocation_system_lifting(
                        y0, y1, h, t_nodes, A_nodes, g_nodes, n_stages, dim,
                        ode_batch_shape, not_avail_indices.shape
                    )
                else:
                    raise ValueError(f"Unknown interpolation_method: {self.interpolation_method}")

                self.P[..., not_avail_indices, :, :] = coeffs
                self.P_available[not_avail_indices] = True
        
        # 3. Use torch.gather to safely get coefficients C
        C = torch.gather(self.P, -3, indices.unsqueeze(-1).unsqueeze(-1).expand(indices.shape + self.P.shape[-2:]))

        # 4. Evaluate the polynomial with the computed coefficients
        t0 = self.ts[indices]
        t_eval = t_batch_broadcasted - t0
        powers = torch.arange(n_coeffs, device=t_eval.device, dtype=t_eval.dtype)
        t_powers = torch.pow(t_eval.unsqueeze(-1), powers).unsqueeze(-1)
        y_interp = torch.sum(C * t_powers, dim=-2)
            
        if was_scalar:
            y_interp = y_interp.squeeze(-2)

        return y_interp

@lru_cache(maxsize=None)
def _get_legendre_coeffs_and_deriv_coeffs(max_order, device, dtype):
    """
    Calculates and caches the power series coefficients for Legendre polynomials
    and their derivatives.
    """
    # Get the cached power series coefficients for Legendre polynomials P_n(x)
    poly_coeffs = _get_legendre_coeffs(max_order)

    # Compute coefficients for the derivatives P'_n(x)
    # If P_n(x) = sum_k c_k * x^k, then P'_n(x) = sum_k k * c_k * x^(k-1)
    k = np.arange(max_order + 1)
    deriv_coeffs_times_power = poly_coeffs * k
    # The coefficient of x^j in P'_n(x) is (j+1)*c_{j+1}
    deriv_coeffs = np.roll(deriv_coeffs_times_power, -1, axis=1)
    deriv_coeffs[:, -1] = 0
    
    return torch.from_numpy(poly_coeffs).to(device=device, dtype=dtype), torch.from_numpy(deriv_coeffs).to(device=device, dtype=dtype)

def _eval_legendre_poly_and_deriv(max_order, x):
    """
    Evaluates Legendre polynomials and their derivatives up to a given order
    at specified points x.

    This implementation uses the power series coefficients of the Legendre
    polynomials to evaluate them in a vectorized manner. The coefficients
    are pre-calculated and cached via `_get_legendre_coeffs_and_deriv_coeffs`.
    """
    x = torch.as_tensor(x)
    dtype = x.dtype
    device = x.device

    # Get the cached coefficients for the polynomials and their derivatives
    poly_coeffs, deriv_coeffs = _get_legendre_coeffs_and_deriv_coeffs(max_order, device, dtype)

    # Compute powers of x, i.e., [1, x, x^2, ..., x^max_order]
    # Shape: [max_order + 1, *x.shape]
    x_powers_flat = x.flatten().unsqueeze(0).pow(torch.arange(max_order + 1, device=device, dtype=dtype).unsqueeze(1))
    x_powers = x_powers_flat.reshape((max_order + 1,) + x.shape)

    # Evaluate the polynomials using the coefficients and powers of x
    # P[n](x) = sum_k poly_coeffs[n, k] * x^k
    P = torch.einsum("nk,k...->n...", poly_coeffs, x_powers)

    # Evaluate the derivatives
    P_deriv = torch.einsum("nk,k...->n...", deriv_coeffs, x_powers)
        
    return P, P_deriv

@lru_cache(maxsize=None)
def _get_legendre_coeffs(max_order):
    coeffs = np.zeros((max_order + 1, max_order + 1))
    if max_order >= 0:
        coeffs[0, 0] = 1.0
    if max_order >= 1:
        coeffs[1, 1] = 1.0
    for k in range(1, max_order):
        coeffs[k+1, 1:] = (2*k+1)/(k+1) * coeffs[k, :-1]
        coeffs[k+1, :] -= k/(k+1) * coeffs[k-1, :]
    return coeffs

@lru_cache(maxsize=None)
def _get_legendre_to_monomial_transformation(n_stages, device, dtype):
    """
    Computes the h-independent part of the transformation matrix from the
    shifted Legendre basis to the monomial basis.

    This function calculates a matrix `M` where `M[l, j]` is a component of the
    coefficient for the `t^l` term in the expansion of `P_j(2*t/h - 1)`.
    Specifically, the coefficient of `t^l` is `(2/h)^l * M[l, j]`.

    The matrix is constructed such that its j-th column contains the coefficients
    for the j-th basis function, which is a standard convention.
    """
    legendre_coeffs = _get_legendre_coeffs(n_stages - 1)
    
    # Pre-compute all possible combinations
    l_idx, j_idx, m_idx = np.meshgrid(
        np.arange(n_stages), np.arange(n_stages), np.arange(n_stages),
        indexing='ij'
    )
    
    # Create mask
    mask = (m_idx >= l_idx) & (m_idx <= j_idx)
    
    # Compute terms
    terms = np.where(mask,
                    legendre_coeffs[j_idx, m_idx] * 
                    scipy.special.comb(m_idx, l_idx) * 
                    (-1.0) ** (m_idx - l_idx),
                    0)
    
    # Sum over m dimension
    M = np.sum(terms, axis=2)
    return torch.from_numpy(M).to(device=device, dtype=dtype)

def _compute_phi_to_power_coeffs_matrix(n_stages, h, dtype, device):
    """
    Computes the matrix of power series coefficients for the lifting basis functions phi_j(t).
    The returned tensor has shape [*t_batch_shape, n_coeffs, n_stages], where
    phi_coeffs[..., k, j] is the coefficient of t^k for the basis function phi_j(t).
    """
    t_batch_shape = h.shape
    poly_degree = n_stages + 1
    n_coeffs = poly_degree + 1

    # Get the h-independent transformation matrix M.
    # M[l, j] is the part of the t^l coefficient for the j-th basis function.
    M = _get_legendre_to_monomial_transformation(n_stages, device, dtype)

    a = 2 / h
    a_powers = torch.pow(a.unsqueeze(-1), torch.arange(n_stages, device=device, dtype=dtype))

    # Compute q_coeffs. q_coeffs[..., l, j] is the coeff of t^l in P_j(a*t-1).
    # This is M[l, j] * a_powers[..., l].
    q_coeffs = M.view((1,) * len(t_batch_shape) + (n_stages, n_stages)) * a_powers.unsqueeze(-1)

    # Get power series for phi_j(t) = (t^2 - h*t) * P_j(tau(t))
    # phi_coeffs[..., k, j] is the k-th power coeff for phi_j(t)
    phi_coeffs = torch.zeros(*t_batch_shape, n_coeffs, n_stages, dtype=dtype, device=device)

    # from t^2 * P_j
    phi_coeffs[..., 2:, :] += q_coeffs
    # from -h*t * P_j
    phi_coeffs[..., 1:-1, :] -= h.unsqueeze(-1).unsqueeze(-1) * q_coeffs
    
    return phi_coeffs


def _solve_collocation_system_lifting(y0, y1, h, t_nodes, A_nodes, g_nodes, n_stages, dim, ode_batch_shape, t_batch_shape):
    # --- Argument Shapes ---
    # y0, y1: [*ode_batch_shape, *t_batch_shape, dim]
    # h: [*t_batch_shape]
    # t_nodes: [n_stages, *t_batch_shape]
    # A_nodes: [*ode_batch_shape, n_stages, *t_batch_shape, dim, dim]
    # g_nodes: [*ode_batch_shape, n_stages, *t_batch_shape, dim] or None
    
    batch_shape = ode_batch_shape + t_batch_shape

    # --- 1. Construct y_0(t) and its derivative y_0'(t) ---
    h_ = h.unsqueeze(-1)
    y0_prime = (y1 - y0) / h_

    # --- 2. Evaluate Basis Functions phi_j(t_i) and their derivatives phi_j'(t_i) ---
    tau = 2 * t_nodes / h - 1
    legendre_polys, legendre_derivs = _eval_legendre_poly_and_deriv(n_stages - 1, tau)
    
    phi = t_nodes * (t_nodes - h) * legendre_polys
    phi_deriv = (2 * t_nodes - h) * legendre_polys + t_nodes * (t_nodes - h) * legendre_derivs * (2 / h)
    
    phi = phi.permute(*range(2, tau.dim() + 1), 1, 0)
    phi_deriv = phi_deriv.permute(*range(2, tau.dim() + 1), 1, 0)

    # --- 3. Construct and Solve the Linear System M*c = D ---
    eye = torch.eye(dim, dtype=y0.dtype, device=y0.device)
    permute_dims = list(range(len(ode_batch_shape))) + [len(ode_batch_shape) + i + 1 for i in range(len(t_batch_shape))] + [len(ode_batch_shape)] + list(range(A_nodes.dim() - 2, A_nodes.dim()))
    A_nodes_T = A_nodes.permute(*permute_dims)
    
    M = phi_deriv.unsqueeze(-1).unsqueeze(-1) * eye - phi.unsqueeze(-1).unsqueeze(-1) * A_nodes_T.unsqueeze(-3)
    M = M.transpose(-2, -3).reshape(*batch_shape, n_stages * dim, n_stages * dim)

    y0_at_nodes = y0.unsqueeze(-2) + (y1 - y0).unsqueeze(-2) * (t_nodes / h).permute(*range(1, t_nodes.dim()), 0).unsqueeze(-1)
    D = torch.einsum('...isde,...ise->...isd', A_nodes_T, y0_at_nodes) - y0_prime.unsqueeze(-2)
    if g_nodes is not None:
        g_nodes_T = g_nodes.permute(*permute_dims[:-2], g_nodes.dim() - 1)
        D += g_nodes_T
    
    D = D.reshape(*batch_shape, n_stages * dim)
    
    c_flat = torch.linalg.solve(M, D)
    c = c_flat.reshape(*batch_shape, n_stages, dim)

    # --- 4. Fully Vectorized Coefficient Conversion ---
    # Final polynomial: y(t) = y_0(t) + sum_{j=0}^{n_stages-1} c_j * phi_j(t)
    # We represent this as a matrix-vector product. The "vector" is the coefficients
    # in the lifting basis {1, t, phi_0, ...} and the matrix transforms them to the
    # power basis {t^k}.
    
    poly_degree = n_stages + 1
    n_coeffs = poly_degree + 1

    # Construct the conversion matrix T
    # T has shape [*t_batch_shape, n_coeffs_out, n_coeffs_in]
    # T[k, i] is the coeff of t^k in the i-th basis function.
    # The basis is {1, t, phi_0, ..., phi_{s-1}}
    T = torch.zeros(*h.shape, n_coeffs, n_coeffs, dtype=y0.dtype, device=y0.device)
    T[..., 0, 0] = 1.0  # Basis func 1 -> 1 * t^0
    T[..., 1, 1] = 1.0  # Basis func t -> 1 * t^1

    # Basis funcs phi_j(t)
    # phi_coeffs has shape [*t_batch_shape, n_coeffs, n_stages], where the (k,j) entry
    # is the coefficient of t^k for basis function phi_j. This is exactly what we need
    # for the columns of our transformation matrix T.
    phi_coeffs = _compute_phi_to_power_coeffs_matrix(n_stages, h, y0.dtype, y0.device)
    T[..., :, 2:] = phi_coeffs

    # Construct the coefficient vector in the lifting basis
    # The coefficients are {y0, (y1-y0)/h, c_0, ...}
    # C_lift shape: [*batch_shape, n_coeffs, dim]
    C_lift = torch.cat([
        y0.unsqueeze(-2),
        ((y1 - y0) / h_).unsqueeze(-2),
        c
    ], dim=-2)

    # Transform coefficients to the power basis using batched matrix multiplication
    # T shape: [*t_batch, n_coeffs, n_coeffs]
    # C_lift shape: [*ode_batch, *t_batch, n_coeffs, dim]
    # We need to expand T to match the batch dimensions of C_lift
    ode_batch_rank = len(ode_batch_shape)
    T_expanded = T.view((1,) * ode_batch_rank + T.shape)

    final_coeffs = torch.matmul(T_expanded, C_lift)

    return final_coeffs
        
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
            precomputed_P=merged_P_tensor,
            interpolation_method=first_output.interpolation_method
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
        dense_mode=dense_mode,
        interpolation_method=first_output.interpolation_method
    )
