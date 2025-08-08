import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple

from .butcher import GL2, GL4, GL6, GL8 # Import necessary tableaus
from .utils import _apply_matrix

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Modular Integration Backends
# -----------------------------------------------------------------------------

class BaseQuadrature(nn.Module):
    """Base class for quadrature integration methods."""
    
    def forward(self, system: nn.Module, dense_output_segment: object, params_req: Dict[str, Tensor], buffers_dict: Dict[str, Tensor], atol: float, rtol: float) -> Dict[str, Tensor]:
        """
        Integrate vector-Jacobian product over the interval of the dense output segment.
        
        Args:
            system: The original user-provided system module.
            dense_output_segment: A dense output object covering the integration interval [t_i, t_{i-1}].
            params_req: Dictionary of parameters requiring gradients.
            buffers_dict: Dictionary of buffers.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            
        Returns:
            Dictionary of integrated gradients for the segment.
        """
        raise NotImplementedError

class GaussLegendreQuadrature(BaseQuadrature):
    """
    Computes the gradient integral by iterating through the adaptive steps of the
    adjoint ODE solution and applying a high-order Gauss-Legendre quadrature
    within each sub-interval.
    """
    _TABLEAU_MAP = {2: GL2, 4: GL4, 6: GL6, 8: GL8}

    def __init__(self, order_offset=2):
        super().__init__()
        self.order_offset = order_offset

    def forward(self, system: nn.Module, a_dense_segment: object, y_dense_traj: object, interval: Tuple[float, float], params_req: Dict[str, Tensor], buffers_dict: Dict[str, Tensor], atol: float, rtol: float) -> Dict[str, Tensor]:
        # To use torch.func.grad, we define a function that computes the integral from parameters.
        def compute_integral_from_params(p_req):
            # --- Vectorized Quadrature Setup ---
            ts = a_dense_segment.ts
            t_starts = ts[:-1]
            t_ends = ts[1:]
            h = t_ends - t_starts

            quad_order = a_dense_segment.order + self.order_offset
            if quad_order not in self._TABLEAU_MAP:
                raise ValueError(f"Gauss-Legendre quadrature of order {quad_order} is not available.")
            tableau = self._TABLEAU_MAP[quad_order]
            
            # Get quadrature nodes and weights for all sub-intervals at once
            # t_nodes shape: (num_sub_intervals, num_quad_nodes)
            c_tensor = tableau.c.to(h.device, h.dtype)
            t_nodes = t_starts.unsqueeze(1) + h.unsqueeze(1) * c_tensor
            
            # Flatten for batch evaluation
            t_nodes_flat = t_nodes.flatten()

            # --- Batch Evaluation ---
            a_nodes_flat = a_dense_segment(t_nodes_flat)
            y_nodes_flat = y_dense_traj(t_nodes_flat)

            sys_out = system(t_nodes_flat, {**p_req, **buffers_dict})
            A_nodes_flat, g_nodes_flat = (sys_out[0], sys_out[1]) if isinstance(sys_out, tuple) else (sys_out, None)

            # --- Vectorized Integration ---
            integrand = _apply_matrix(A_nodes_flat, y_nodes_flat)
            if g_nodes_flat is not None:
                integrand = integrand + g_nodes_flat

            # Contract with adjoint state: a^T * f
            sub_integral_nodes_flat = torch.einsum("...d,...d->...", a_nodes_flat, integrand)

            # Weighted sum for each sub-interval's integral
            weights = tableau.b.to(h.device, h.dtype)
            # h shape: (num_sub_intervals, 1), weights shape: (num_quad_nodes,)
            sub_integrals = torch.einsum("d, ...d->...", (weights * h.unsqueeze(-1)).view(-1), sub_integral_nodes_flat)
            
            # Accumulate the final scalar integral sum
            total_integral_sum = torch.sum(sub_integrals)

            return total_integral_sum

        # Compute the gradient using the functional API
        grad_of_integral_func = torch.func.grad(compute_integral_from_params)
        integral_dict = grad_of_integral_func(params_req)

        return integral_dict