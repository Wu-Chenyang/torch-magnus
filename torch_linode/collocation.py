
import torch
import torch.nn as nn
from typing import Callable, Tuple, Union, Sequence
Tensor = torch.Tensor

from .butcher import ButcherTableau
from .utils import _apply_matrix

class Collocation(nn.Module):
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