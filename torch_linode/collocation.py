
import torch
import torch.nn as nn
from typing import Callable, Tuple, Union, Sequence

from .butcher import ButcherTableau

def _apply_matrix(U: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Helper function from solvers.py, needed here as well.
    Apply matrix or batch of matrices to vector or batch of vectors.
    """
    return (U @ y.unsqueeze(-1)).squeeze(-1)

class Collocation(nn.Module):
    def __init__(self, tableau: ButcherTableau):
        super().__init__()
        self.tableau = tableau.clone()
        self.order = tableau.order

    def forward(self, A: Callable[..., torch.Tensor], t0: Union[Sequence[float], torch.Tensor, float], h: Union[Sequence[float], torch.Tensor, float], y0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure tensors are on the correct device and dtype
        t0 = torch.as_tensor(t0, device=y0.device, dtype=y0.dtype)
        h_tensor = torch.as_tensor(h, device=y0.device, dtype=y0.dtype)

        # Get Butcher tableau coefficients
        self.tableau = self.tableau.to(dtype=y0.dtype, device=y0.device)
        c, b, a_matrix = self.tableau.c, self.tableau.b, self.tableau.a
        num_stages = c.shape[0]
        
        # 1. Calculate collocation time points
        t_nodes = t0.unsqueeze(-1) + h_tensor.unsqueeze(-1) * c
        
        # 2. Evaluate the system matrix A(t) at each time point
        t_nodes = t0.unsqueeze(-1) + h_tensor.unsqueeze(-1) * c
        A_nodes_out = A(t_nodes.reshape(-1))
        
        is_nonhomogeneous = isinstance(A_nodes_out, tuple) and len(A_nodes_out) == 2
        if is_nonhomogeneous:
            A_nodes_flat, g_nodes_flat = A_nodes_out
        else:
            A_nodes_flat, g_nodes_flat = A_nodes_out, None

        # Reshape evaluated matrices and vectors to match batch dimensions
        d = y0.shape[-1]
        eval_batch_shape = torch.broadcast_shapes(y0.shape[:-1], h_tensor.shape)

        A_nodes = A_nodes_flat.view(*eval_batch_shape, num_stages, d, d)
        g_nodes = None
        if g_nodes_flat is not None:
            g_nodes = g_nodes_flat.view(*eval_batch_shape, num_stages, d)

        # 3. Construct the linear system L K = R
        I_d = torch.eye(d, device=y0.device, dtype=y0.dtype)

        # Vectorized construction of L matrix: L_ij = delta_ij * I - h * a_ij * A_i

        # Step 1: Form the tensor of blocks from a_ij and A_i using einsum.
        # Result L_T has shape (*eval_batch_shape, num_stages, num_stages, d, d)
        # where L_T[..., i, j, :, :] corresponds to a_matrix[i, j] * A_nodes[..., i, :, :]
        L_T = torch.einsum("ij,...ikm->...ijkm", a_matrix, A_nodes)

        # Step 2: Permute and reshape to form the large matrix of blocks.
        # Permute from (*b, i, j, k, m) to (*b, i, k, j, m) and then reshape.
        N_batch = len(eval_batch_shape)
        permute_dims = list(range(N_batch)) + [N_batch, N_batch + 2, N_batch + 1, N_batch + 3]
        L = L_T.permute(*permute_dims).reshape(*eval_batch_shape, num_stages * d, num_stages * d)

        # Step 3: Multiply by -h. h_tensor is expanded to broadcast with L.
        h_exp = h_tensor
        while h_exp.ndim < L.ndim:
            h_exp = h_exp.unsqueeze(-1)
        L = -h_exp * L

        # Step 4: Add identity matrix to the diagonal blocks.
        # A loop is clear and robust against varying batch dimensions.
        for i in range(num_stages):
            L[..., i*d:(i+1)*d, i*d:(i+1)*d].add_(I_d)

        # 4. Construct the R vector (concatenation of R_i)
        # R_i = A_i @ y0 (+ g_i)
        R = torch.einsum('...sij,...j->...si', A_nodes, y0)
        if g_nodes is not None:
            R += g_nodes
        R = R.flatten(start_dim=-2)


        # 5. Solve for the stage derivatives K
        K_flat = torch.linalg.solve(L, R)
        k_stages = K_flat.view(*eval_batch_shape, num_stages, d)

        # 6. Compute the final solution y_next
        h_final_exp = h_tensor.view(*h_tensor.shape, 1)
        y_next = y0 + h_final_exp * torch.einsum("i,...id->...d", b, k_stages)
        
        return y_next, A_nodes
