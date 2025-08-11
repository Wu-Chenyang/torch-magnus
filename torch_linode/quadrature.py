import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple

from .butcher import GL2, GL4, GL6, GL8, GL10, GK15
from .utils import _apply_matrix
import warnings
import heapq
import math

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
    _TABLEAU_MAP = {2: GL2, 4: GL4, 6: GL6, 8: GL8, 10: GL10}

    def __init__(self):
        super().__init__()

    def forward(self, system: nn.Module, a_dense_segment: object, y_dense_traj: object, params_req: Dict[str, Tensor], buffers_dict: Dict[str, Tensor], atol: float, rtol: float) -> Dict[str, Tensor]:
        # --- Vectorized Quadrature Setup ---
        ts = a_dense_segment.ts
        t_starts = ts[:-1]
        t_ends = ts[1:]
        h = t_ends - t_starts

        quad_order = a_dense_segment.order
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

        # To use torch.func.grad, we define a function that computes the integral from parameters.
        def compute_integral_from_params(p_req):
            
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

class AdaptiveGaussKronrod(BaseQuadrature):
    """Adaptive Gauss-Kronrod quadrature integration with batching."""
    
    _TABLEAU_MAP = {15: GK15}

    def __init__(self):
        super().__init__()

    def _eval_segment_batch(self, system, a_dense_segment, y_dense_traj, intervals, params_req, buffers_dict, tableau):
        """Evaluate integral over a batch of segments using Gauss-Kronrod rule."""
        a = intervals[:, 0]
        b = intervals[:, 1]
        h = b - a
        
        c_tensor = tableau.c.to(h.device, h.dtype)
        t_nodes = a.unsqueeze(1) + h.unsqueeze(1) * c_tensor.unsqueeze(0)
        t_nodes_flat = t_nodes.flatten()
        
        a_nodes_flat = a_dense_segment(t_nodes_flat)
        y_nodes_flat = y_dense_traj(t_nodes_flat)

        def compute_integrals_from_params(p_req):
            sys_out = system(t_nodes_flat, {**p_req, **buffers_dict})
            A_nodes_flat, g_nodes_flat = (sys_out[0], sys_out[1]) if isinstance(sys_out, tuple) else (sys_out, None)
            integrand = _apply_matrix(A_nodes_flat, y_nodes_flat)
            if g_nodes_flat is not None:
                integrand = integrand + g_nodes_flat
            
            sub_integral_nodes_flat = torch.einsum("...d,...d->...", a_nodes_flat, integrand)
            sub_integral_nodes = sub_integral_nodes_flat.view(a_nodes_flat.shape[:-2] + t_nodes.shape)

            weights_k = tableau.b.to(h.device, h.dtype)
            integral_k = h * torch.einsum("...n,n->...", sub_integral_nodes, weights_k)
            
            weights_g = (tableau.b - tableau.b_error).to(h.device, h.dtype)
            integral_g = h * torch.einsum("...n,n->...", sub_integral_nodes, weights_g)
            
            return integral_k.view(-1, intervals.shape[0]).sum(0), integral_g.view(-1, intervals.shape[0]).sum(0)

        grad_funcs = torch.func.jacrev(compute_integrals_from_params)
        grads = grad_funcs(params_req)
        I_K = grads[0]
        I_G = grads[1]
        
        error = torch.zeros(len(intervals), device=a.device, dtype=a.dtype)
        for k in I_K:
            error += (I_K[k] - I_G[k]).square().sum(dim=list(range(1, I_K[k].ndim)))
        error = torch.sqrt(error)
        
        return I_K, error

    def forward(self, system: nn.Module, a_dense_segment: object, y_dense_traj: object, params_req: Dict[str, Tensor], buffers_dict: Dict[str, Tensor], atol: float, rtol: float) -> Dict[str, Tensor]:
        """Adaptive Gauss-Kronrod integration with error control."""
        ts = a_dense_segment.ts
        t_starts = ts[:-1]
        t_ends = ts[1:]

        I_total = {k: torch.zeros_like(v) for k, v in params_req.items()}
        
        if len(t_starts) == 0:
            return I_total

        quad_order = 15
        tableau = self._TABLEAU_MAP[quad_order]
        
        initial_intervals = torch.stack([t_starts, t_ends], dim=1)
        I_K_batch, error_batch = self._eval_segment_batch(system, a_dense_segment, y_dense_traj, initial_intervals, params_req, buffers_dict, tableau)

        heap = []
        for i in range(len(initial_intervals)):
            I_K_i = {k: v[i] for k, v in I_K_batch.items()}
            heapq.heappush(heap, (-error_batch[i].item(), tuple(initial_intervals[i].tolist()), I_K_i, error_batch[i].item()))

        for k in I_total:
            I_total[k] = torch.sum(I_K_batch[k], dim=0)
        E_total = torch.sum(error_batch)

        ref_param = next(iter(params_req.values()))
        machine_eps = torch.finfo(ref_param.dtype).eps
        max_segments_per_iter = 32

        while E_total > atol + rtol * math.sqrt(sum(v.square().sum().item() for v in I_total.values())):
            if not heap:
                break

            num_to_pop = min(max_segments_per_iter, len(heap))
            intervals_to_process = []
            intervals_to_refine = []
            I_K_parents = {k: [] for k in I_total.keys()}
            err_parents = []
            for i in range(num_to_pop):
                interval_to_process = heapq.heappop(heap)
                intervals_to_process.append(interval_to_process)
                E_total -= interval_to_process[3]
                intervals_to_refine.append(interval_to_process[1])
                for k in I_total:
                    I_K_parents[k].append(interval_to_process[2][k])
                err_parents.append(interval_to_process[3])

            intervals_to_refine = torch.tensor(intervals_to_refine, device=ts.device, dtype=ts.dtype)
            I_K_parents = {k: torch.stack(v) for k, v in I_K_parents.items()}
            err_parents = torch.tensor(err_parents, device=ts.device, dtype=ts.dtype)

            a = intervals_to_refine[:, 0]
            b = intervals_to_refine[:, 1]
            
            if torch.any(torch.abs(b - a) < machine_eps * 100):
                warnings.warn("An interval is too small to subdivide further.")
                # Filter out intervals that are too small
                mask = torch.abs(b-a) >= machine_eps * 100
                if not torch.any(mask):
                    continue
                a = a[mask]
                b = b[mask]
                I_K_parents = {k: v[mask] for k,v in I_K_parents.items()}
                err_parents = err_parents[mask]
                num_to_pop = mask.sum().item()

            mid = (a + b) / 2.0
            intervals_left = torch.stack([a, mid], dim=1)
            intervals_right = torch.stack([mid, b], dim=1)
            new_intervals = torch.cat([intervals_left, intervals_right])
            
            I_K_new, err_new = self._eval_segment_batch(system, a_dense_segment, y_dense_traj, new_intervals, params_req, buffers_dict, tableau)
            
            I_K_left = {k: v[:num_to_pop] for k, v in I_K_new.items()}
            I_K_right = {k: v[num_to_pop:] for k, v in I_K_new.items()}
            err_left = err_new[:num_to_pop]
            err_right = err_new[num_to_pop:]

            diff = {k: I_K_left[k] + I_K_right[k] - I_K_parents[k] for k in I_total}
            for k in I_total:
                I_total[k] += torch.sum(diff[k], dim=0)
            
            posterior_error = torch.zeros(num_to_pop, device=ts.device, dtype=ts.dtype)
            for k in diff:
                posterior_error += diff[k].square().sum(dim=list(range(1, diff[k].ndim)))
            posterior_error = torch.sqrt(posterior_error)

            # Avoid division by zero
            err_parents[err_parents == 0] = 1.0
            refined_err_left = err_left * posterior_error / err_parents
            refined_err_right = err_right * posterior_error / err_parents
            
            E_total += torch.sum(refined_err_left) + torch.sum(refined_err_right)

            for i in range(num_to_pop):
                I_K_left_i = {k: I_K_left[k][i] for k in I_total}
                I_K_right_i = {k: I_K_right[k][i] for k in I_total}
                heapq.heappush(heap, (-refined_err_left[i].item(), tuple(intervals_left[i].tolist()), I_K_left_i, refined_err_left[i].item()))
                heapq.heappush(heap, (-refined_err_right[i].item(), tuple(intervals_right[i].tolist()), I_K_right_i, refined_err_right[i].item()))

        return I_total