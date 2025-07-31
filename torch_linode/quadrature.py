from typing import Callable, Dict
import torch
import torch.nn as nn
import math
import warnings
import heapq
Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Modular Integration Backends
# -----------------------------------------------------------------------------

class BaseQuadrature(nn.Module):
    """Base class for quadrature integration methods."""
    
    def forward(self, f_for_vjp: Callable, a_interp_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Dict[str, Tensor], is_nonhomogeneous: bool) -> Dict[str, Tensor]:
        """
        Integrate vector-Jacobian product over interval [a, b].
        
        Args:
            f_for_vjp: Function to compute the VJP target.
            a_interp_func: Interpolation function for the adjoint state.
            a: Integration start time.
            b: Integration end time.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            params_req: Dictionary of parameters requiring gradients.
            is_nonhomogeneous: Flag indicating system type.
            
        Returns:
            Dictionary of integrated gradients.
        """
        raise NotImplementedError


class AdaptiveGaussKronrod(BaseQuadrature):
    """Adaptive Gauss-Kronrod quadrature integration."""
    
    # 15-point Gauss-Kronrod rule coefficients
    _GK_NODES_RAW = [-0.99145537112081263920685469752598, -0.94910791234275852452618968404809, -0.86486442335976907278971278864098, -0.7415311855993944398638647732811, -0.58608723546769113029414483825842, -0.40584515137739716690660641207707, -0.20778495500789846760068940377309, 0.0]
    _GK_WEIGHTS_K_RAW = [0.022935322010529224963732008059913, 0.063092092629978553290700663189093, 0.10479001032225018383987632254189, 0.14065325971552591874518959051021, 0.16900472663926790282658342659795, 0.19035057806478540991325640242055, 0.20443294007529889241416199923466, 0.20948214108472782801299917489173]
    _GK_WEIGHTS_G_RAW = [0.12948496616886969327061143267787, 0.2797053914892766679014677714229, 0.38183005050511894495036977548818, 0.41795918367346938775510204081658]
    _rule_cache = {}

    @classmethod
    def _get_rule(cls, dtype, device):
        """Get cached quadrature rule for given dtype and device."""
        if (dtype, device) in cls._rule_cache: 
            return cls._rule_cache[(dtype, device)]
            
        nodes_neg = torch.tensor(cls._GK_NODES_RAW, dtype=dtype, device=device)
        nodes = torch.cat([-nodes_neg[0:-1].flip(0), nodes_neg])
        weights_k_half = torch.tensor(cls._GK_WEIGHTS_K_RAW, dtype=dtype, device=device)
        weights_k = torch.cat([weights_k_half[0:-1].flip(0), weights_k_half])
        weights_g_half = torch.tensor(cls._GK_WEIGHTS_G_RAW, dtype=dtype, device=device)
        weights_g_embedded = torch.cat([weights_g_half[0:-1].flip(0), weights_g_half])
        weights_g = torch.zeros_like(weights_k)
        weights_g[1::2] = weights_g_embedded
        rule = (nodes, weights_k.unsqueeze(1), weights_g.unsqueeze(1))
        cls._rule_cache[(dtype, device)] = rule
        return rule

    def _eval_segment(self, f_for_vjp, a_interp_func, a, b, params_req, nodes, weights_k, weights_g, is_nonhomogeneous):
        """Evaluate integral over a single segment using Gauss-Kronrod rule."""
        h = (b - a) / 2.0
        c = (a + b) / 2.0
        segment_nodes = c + h * nodes
        
        # Get evaluations of the adjoint and the VJP target function
        a_eval = a_interp_func(segment_nodes)
        
        with torch.enable_grad():
            vjp_target = f_for_vjp(segment_nodes, params_req)
            _, vjp_fn = torch.func.vjp(lambda p: f_for_vjp(segment_nodes, p), params_req)

        # Prepare cotangents for VJP
        cotangent_K = h * weights_k * a_eval
        cotangent_G = h * weights_g * a_eval

        I_K = vjp_fn(cotangent_K)[0]
        I_G = vjp_fn(cotangent_G)[0]
        
        diff_dict = {k: I_K[k] - I_G[k] for k in I_K}
        error = math.sqrt(sum(v.square().sum().item() for v in diff_dict.values()))
        return I_K, error

    def forward(self, f_for_vjp: Callable, a_interp_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Dict[str, Tensor], is_nonhomogeneous: bool, max_segments: int = 100) -> Dict[str, Tensor]:
        """Adaptive Gauss-Kronrod integration with error control."""
        if a == b:
            return {k: torch.zeros_like(v) for k, v in params_req.items()}

        ref_param = next(iter(params_req.values()))
        nodes, weights_k, weights_g = self._get_rule(ref_param.dtype, ref_param.device)
        
        I_total = {k: torch.zeros_like(v) for k, v in params_req.items()}
        E_total = 0.0
        
        I_K, error = self._eval_segment(f_for_vjp, a_interp_func, a, b, params_req, nodes, weights_k, weights_g, is_nonhomogeneous)
        heap = [(-error, a, b, I_K, error)]

        for k in I_total: I_total[k] += I_K[k]
        E_total += error
        
        machine_eps = torch.finfo(ref_param.dtype).eps

        while heap:
            I_total_norm = torch.sqrt(sum(v.square().sum() for v in I_total.values())).item()
            if E_total <= atol + rtol * I_total_norm:
                break
            if len(heap) >= max_segments:
                warnings.warn(f"Max segments ({max_segments}) reached. Result may be inaccurate. atol: {atol} rtol: {rtol} error: {E_total} tolerance: {atol + rtol * I_total_norm}")
                break

            _, a_parent, b_parent, I_K_parent, err_parent = heapq.heappop(heap)
            
            if abs(b_parent - a_parent) < machine_eps * 100:
                warnings.warn(f"Interval {b_parent - a_parent} too small to subdivide further.")
                continue

            mid = (a_parent + b_parent) / 2.0

            I_K_left, err_left = self._eval_segment(f_for_vjp, a_interp_func, a_parent, mid, params_req, nodes, weights_k, weights_g, is_nonhomogeneous)
            I_K_right, err_right = self._eval_segment(f_for_vjp, a_interp_func, mid, b_parent, params_req, nodes, weights_k, weights_g, is_nonhomogeneous)

            posterior_error = 0.0
            for k in I_total:
                diff = I_K_left[k] + I_K_right[k] - I_K_parent[k]
                I_total[k] += diff
                posterior_error += diff.square().sum().item()
            posterior_error = math.sqrt(posterior_error)

            refined_err_left = err_left * posterior_error / err_parent if err_parent > 0 else err_left
            refined_err_right = err_right * posterior_error / err_parent if err_parent > 0 else err_right

            E_total += refined_err_left + refined_err_right - err_parent

            heapq.heappush(heap, (-refined_err_left, a_parent, mid, I_K_left, refined_err_left))
            heapq.heappush(heap, (-refined_err_right, mid, b_parent, I_K_right, refined_err_right))
            
        return I_total


class FixedSimpson(BaseQuadrature):
    """Fixed-step composite Simpson's rule integrator."""
    
    def __init__(self, N=100):
        """
        Initialize Simpson integrator.
        
        Args:
            N: Number of intervals (should be even)
        """
        super().__init__()
        if N % 2 != 0:
            warnings.warn("N should be even for Simpson's rule; incrementing N by 1.")
            N += 1
        self.N = N

    def forward(self, f_for_vjp: Callable, a_interp_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Dict[str, Tensor], is_nonhomogeneous: bool) -> Dict[str, Tensor]:
        """Fixed-step Simpson integration."""
        if a == b:
            return {k: torch.zeros_like(v) for k, v in params_req.items()}

        ref_param = next(iter(params_req.values()))
        nodes = torch.linspace(a, b, self.N + 1, device=ref_param.device, dtype=ref_param.dtype)
        h = (b - a) / self.N

        a_eval = a_interp_func(nodes)

        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(lambda p: f_for_vjp(nodes, p), params_req)
            
            weights = torch.ones(self.N + 1, device=a_eval.device, dtype=a_eval.dtype)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            weights *= (h / 3.0)
            
            cotangent = weights.unsqueeze(1) * a_eval

            integral_dict = vjp_fn(cotangent)[0]

        return integral_dict
