
import torch
import torch.nn as nn
from typing import Callable, Tuple, Union, Sequence
import math
Tensor = torch.Tensor

from .butcher import ButcherTableau, GL2, GL4, GL6
from .utils import _apply_matrix, _commutator, _matrix_exp

# -----------------------------------------------------------------------------
# Single-Step Integrators
# -----------------------------------------------------------------------------

class BaseStepper(nn.Module):
    """
    Abstract base class for single-step integrators for linear ODEs.
    Defines the interface for all steppers, such as Magnus or GLRK.
    """
    order: int

    def forward(self, A_func: Callable, t0: float, h: float, y0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform a single integration step.

        Args:
            A_func: A function that returns the matrix A for a given time t.
            t0: The initial time of the step.
            h: The step size.
            y0: The initial state tensor of shape (..., *batch_shape, dim).

        Returns:
            A tuple containing:
            - y_next (Tensor): The solution at time t0 + h.
            - aux_data (Tensor): Auxiliary data computed during the step,
                                 such as matrix evaluations at quadrature nodes,
                                 which can be used for dense output.
        """
        raise NotImplementedError("step() must be implemented by subclasses.")

class Magnus2nd(BaseStepper):
    """Second-order Magnus integrator using midpoint rule."""
    order = 2
    tableau = GL2.clone()

    def forward(self, A: Callable[..., Tensor], t0: Union[Sequence[float], torch.Tensor, float], h: Union[Sequence[float], torch.Tensor, float], y0: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        t0 = torch.as_tensor(t0, dtype=y0.dtype, device=y0.device)
        h = torch.as_tensor(h, dtype=y0.dtype, device=y0.device)
        h = h.reshape(torch.broadcast_shapes(h.shape, t0.shape))
        t_nodes = self.tableau.get_t_nodes(t0, h)

        A = A(t_nodes)
        y_next = self.get_next_y(A, h, y0)

        return y_next
    
    def get_next_y(self, A: Tensor, h: Union[Sequence[float], torch.Tensor, float], y0: Tensor) -> Tensor:
        if h.ndim == 0:
            A = A.squeeze(-3)

        h_expanded = h.unsqueeze(-1).unsqueeze(-1)
        Omega = h_expanded * A
        U = _matrix_exp(Omega)
        y_next = _apply_matrix(U, y0)
        return y_next

class Magnus4th(BaseStepper):
    """Fourth-order Magnus integrator using two-point Gauss quadrature."""
    order = 4
    tableau = GL4.clone()
    _sqrt3 = math.sqrt(3.0)

    def forward(self, A: Callable[..., Tensor], t0: Union[Sequence[float], torch.Tensor, float], h: Union[Sequence[float], torch.Tensor, float], y0: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        t0 = torch.as_tensor(t0, dtype=y0.dtype, device=y0.device)
        h = torch.as_tensor(h, dtype=y0.dtype, device=y0.device)
        h = h.reshape(torch.broadcast_shapes(h.shape, t0.shape))
        t_nodes = self.tableau.get_t_nodes(t0, h)
        
        A = A(t_nodes)
        y_next = self.get_next_y(A, h, y0)
        
        return y_next

    def get_next_y(self, A: Tensor, h: Union[Sequence[float], torch.Tensor, float], y0: Tensor) -> Tensor:
        A = A.reshape(A.shape[:-3] + (-1, 2) + A.shape[-2:])
        A1, A2 = A[..., 0, :, :], A[..., 1, :, :]
        if h.ndim == 0:
            A1, A2 = A1.squeeze(-3), A2.squeeze(-3)
        
        h_expanded = h.unsqueeze(-1).unsqueeze(-1)
        alpha1 = h_expanded / 2.0 * (A1 + A2)
        alpha2 = h_expanded * self._sqrt3 * (A2 - A1)
        
        Omega = alpha1 - (1/12) * _commutator(alpha1, alpha2)
        
        U = _matrix_exp(Omega)
        y_next = _apply_matrix(U, y0)
        return y_next

class Magnus6th(BaseStepper):
    """Sixth-order Magnus integrator using three-point Gauss quadrature."""
    order = 6
    tableau = GL6.clone()
    _sqrt15 = math.sqrt(15.0)

    def forward(self, A: Callable[..., Tensor], t0: Union[Sequence[float], torch.Tensor, float], h: Union[Sequence[float], torch.Tensor, float], y0: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        t0 = torch.as_tensor(t0, dtype=y0.dtype, device=y0.device)
        h = torch.as_tensor(h, dtype=y0.dtype, device=y0.device)
        h = h.reshape(torch.broadcast_shapes(h.shape, t0.shape))
        t_nodes = self.tableau.get_t_nodes(t0, h)

        A = A(t_nodes)
        y_next = self.get_next_y(A, h, y0)
        
        return y_next

    def get_next_y(self, A: Tensor, h: Union[Sequence[float], torch.Tensor, float], y0: Tensor) -> Tensor:
        A = A.reshape(A.shape[:-3] + (-1, 3) + A.shape[-2:])
        A1, A2, A3 = A[..., 0, :, :], A[..., 1, :, :], A[..., 2, :, :]
        if h.ndim == 0:
            A1, A2, A3 = A1.squeeze(-3), A2.squeeze(-3), A3.squeeze(-3)

        h_expanded = h.unsqueeze(-1).unsqueeze(-1)
        alpha1 = h_expanded * A2
        alpha2 = h_expanded * self._sqrt15 / 3.0 * (A3 - A1)
        alpha3 = h_expanded * 10.0 / 3.0 * (A1 - 2 * A2 + A3)

        C1 = _commutator(alpha1, alpha2)
        C2 = -1/60 * _commutator(alpha1, 2 * alpha3 + C1)
        
        term_in_comm = -20*alpha1 - alpha3 + C1
        term_with_comm = alpha2 + C2
        Omega = alpha1 + alpha3/12 + (1/240)*_commutator(term_in_comm, term_with_comm)

        U = _matrix_exp(Omega)
        y_next = _apply_matrix(U, y0)
        return y_next

class Collocation(BaseStepper):
    def __init__(self, tableau: ButcherTableau):
        super().__init__()
        self.tableau = tableau.clone()
        self.order = tableau.order

    def forward(self, A: Callable[..., torch.Tensor], t0: Union[Sequence[float], torch.Tensor, float], h: Union[Sequence[float], torch.Tensor, float], y0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t0 = torch.as_tensor(t0, dtype=y0.dtype, device=y0.device)
        h = torch.as_tensor(h, dtype=y0.dtype, device=y0.device)
        h = h.reshape(torch.broadcast_shapes(h.shape, t0.shape))
        t_nodes = self.tableau.get_t_nodes(t0, h)
        
        A_nodes_out = A(t_nodes)
        y_next = self.get_next_y(A_nodes_out, h, y0)

        return y_next

    def get_next_y(self, A: Union[Tensor, Tuple[Tensor, Tensor]], h: torch.Tensor, y0: Tensor) -> Tensor:
        self.tableau = self.tableau.to(dtype=y0.dtype, device=y0.device)
        is_nonhomogeneous = isinstance(A, tuple) and len(A) == 2
        if is_nonhomogeneous:
            A_nodes_flat, g_nodes_flat = A
        else:
            A_nodes_flat, g_nodes_flat = A, None

        g_nodes = None
        A_nodes = A_nodes_flat.reshape(A_nodes_flat.shape[:-3] + (h.numel(), -1) + A_nodes_flat.shape[-2:])
        if g_nodes_flat is not None:
            g_nodes = g_nodes_flat.view(g_nodes_flat.shape[:-2] + (h.numel(), -1) + g_nodes_flat.shape[-1:])

        num_stages = self.tableau.c.shape[0]
        d = y0.shape[-1]

        # Vectorized construction of L matrix: L_ij = delta_ij * I - h * a_ij * A_i
        h_expanded = h.reshape(-1).unsqueeze(-1).unsqueeze(-1)
        L = torch.einsum("lij,...likm->...likjm", (-h_expanded) * self.tableau.a, A_nodes)
        L = L.reshape(L.shape[:-4] + (num_stages * d, num_stages * d))
        L_diag = torch.diagonal(L, 0, -1, -2)
        L_diag += 1.0

        # Construct the R vector (concatenation of R_i)
        # R_i = A_i @ y0 (+ g_i)
        R = _apply_matrix(A_nodes, y0.unsqueeze(-2))
        if g_nodes is not None:
            R += g_nodes
        R = R.flatten(start_dim=-2)

        # Solve for the stage derivatives K
        K_flat = torch.linalg.solve(L, R)
        k_stages = K_flat.view(K_flat.shape[:-1] + (num_stages, d))

        # Compute the final solution y_next
        p = h.unsqueeze(-1) * torch.einsum("i,...id->...d", self.tableau.b, k_stages)
        if h.ndim == 0:
            p = p.squeeze(-2)

        y_next = y0 + p

        return y_next