
import torch
import torch.nn as nn
from typing import Callable, Tuple, Union, Sequence

from .butcher import ButcherTableau
from .utils import _apply_matrix

class Collocation(nn.Module):
    def __init__(self, tableau: ButcherTableau):
        super().__init__()
        self.tableau = tableau.clone()
        self.order = tableau.order

    def forward(self, A: Callable[..., torch.Tensor], t0: Union[Sequence[float], torch.Tensor, float], h: Union[Sequence[float], torch.Tensor, float], y0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.tableau = self.tableau.to(dtype=y0.dtype, device=y0.device)
        c, b, a_matrix = self.tableau.c, self.tableau.b, self.tableau.a
        t0 = torch.as_tensor(t0, dtype=y0.dtype, device=y0.device)
        h = torch.as_tensor(h, dtype=y0.dtype, device=y0.device)
        h_expanded = h.reshape(-1).unsqueeze(-1).unsqueeze(-1)
        t_nodes = self.tableau.get_t_nodes(t0, h)
        
        A_nodes_out = A(t_nodes)
        
        is_nonhomogeneous = isinstance(A_nodes_out, tuple) and len(A_nodes_out) == 2
        if is_nonhomogeneous:
            A_nodes_flat, g_nodes_flat = A_nodes_out
        else:
            A_nodes_flat, g_nodes_flat = A_nodes_out, None

        tend = t0+h
        g_nodes = None
        A_nodes = A_nodes_flat.reshape(A_nodes_flat.shape[:-3] + (-1, tend.numel()) + A_nodes_flat.shape[-2:])
        if g_nodes_flat is not None:
            g_nodes = g_nodes_flat.view(g_nodes_flat.shape[:-2] + (-1, tend.numel()) + g_nodes_flat.shape[-1:])

        num_stages = c.shape[0]
        d = y0.shape[-1]

        # Vectorized construction of L matrix: L_ij = delta_ij * I - h * a_ij * A_i
        L = torch.einsum("lij,...ilkm->...likjm", (-h_expanded) * a_matrix, A_nodes)
        L = L.reshape(L.shape[:-4] + (num_stages * d, num_stages * d))
        L_diag = torch.diagonal(L, 0, -1, -2)
        L_diag += 1.0

        # Construct the R vector (concatenation of R_i)
        # R_i = A_i @ y0 (+ g_i)
        R = _apply_matrix(A_nodes, y0)
        if g_nodes is not None:
            R += g_nodes
        R = R.transpose(-2, -3)
        R = R.flatten(start_dim=-2)

        # Solve for the stage derivatives K
        K_flat = torch.linalg.solve(L, R)
        k_stages = K_flat.view(K_flat.shape[:-1] + (num_stages, d))

        # Compute the final solution y_next
        d = h.unsqueeze(-1) * torch.einsum("i,...id->...d", b, k_stages)
        if tend.ndim == 0:
            d = d.squeeze(-2)

        y_next = y0 + d
        
        return y_next, A_nodes
