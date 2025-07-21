import math
from typing import Callable, Sequence, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
import heapq

Tensor = torch.Tensor
TimeSpan = Union[Tuple[float, float], List[float], torch.Tensor]

# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------
def _commutator(A: Tensor, B: Tensor) -> Tensor:
    return A @ B - B @ A


def _matrix_exp(A: Tensor) -> Tensor:
    if A.size(-1) != A.size(-2):
        raise ValueError("matrix_exp 仅支持方阵")
    return torch.linalg.matrix_exp(A)


def _apply_matrix(U: Tensor, y: Tensor) -> Tensor:
    """
    Applies a matrix or a batch of matrices to a vector or a batch of vectors.
    y : (..., *batch_shape, dim)
    U : (..., *batch_shape, dim, dim) or (dim, dim)
    """
    return (U @ y.unsqueeze(-1)).squeeze(-1)

def _prepare_functional_call(A_func_or_module: Union[Callable, nn.Module], params: Tensor = None) -> Tuple[Callable, Dict[str, Tensor]]:
    """
    將用戶輸入的 A_func (Module 或 Callable) 統一轉換為函數式接口。
    
    返回:
        functional_A_func (Callable): 一個接受 (t, p_dict) 的函數。
        params_and_buffers_dict (Dict): 包含所有參數和緩衝區的字典。
    """
    if isinstance(A_func_or_module, torch.nn.Module):
        module = A_func_or_module
        # 合併參數和緩衝區以支持 functional_call
        params_and_buffers = {
            **dict(module.named_parameters()),
            **dict(module.named_buffers())
        }
        
        def functional_A_func(t_val, p_and_b_dict):
            # 使用 functional_call 執行無狀態的模塊
            return torch.func.functional_call(module, p_and_b_dict, (t_val,))
        
        return functional_A_func, params_and_buffers
    else:
        # 處理舊的 (Callable, params) 接口
        A_func = A_func_or_module

        # --- CORRECTED: Handle case where params is None (no parameters) ---
        if params is None:
            # 系統沒有可訓練參數
            params_dict = {}
            def functional_A_func(t_val, p_dict):
                # 原始的 A_func 可能仍然期望 params 參數，即使它是 None
                return A_func(t_val, None)
            return functional_A_func, params_dict

        elif isinstance(params, torch.Tensor):
            # 帶有單個 params 張量的舊接口
            params_dict = {'params': params}
            def functional_A_func(t_val, p_dict):
                # 從字典中解包並調用原始函數
                return A_func(t_val, p_dict['params'])
            return functional_A_func, params_dict
        
        else:
            # params 的類型無效
            raise TypeError(f"The 'params' argument must be a torch.Tensor or None, but got {type(params)}")

# -----------------------------------------------------------------------------
# Magnus 单步积分器
# -----------------------------------------------------------------------------
class BaseMagnus(nn.Module):
    order: int

    def forward(self, A: Callable[[float], Tensor],
                t0: float, h: float, y0: Tensor) -> Tensor:   # abstract
        raise NotImplementedError

class Magnus2nd(BaseMagnus):
    order = 2
    def forward(self, A: Callable[..., Tensor], t0: float, h: float, y0: Tensor) -> Tensor:
        A1 = A(t0 + 0.5 * h)
        Omega = h * A1
        U = _matrix_exp(Omega)
        # --- 核心修改: 返回字典以保持數據結構一致 ---
        return _apply_matrix(U, y0), {'A1': A1}

class Magnus4th(BaseMagnus):
    order = 4
    _sqrt3 = math.sqrt(3.0)
    _c1, _c2 = 0.5 - _sqrt3 / 6, 0.5 + _sqrt3 / 6

    def forward(self, A: Callable[..., Tensor], t0: float, h: float, y0: Tensor) -> Tensor:
        t1, t2 = t0 + self._c1 * h, t0 + self._c2 * h
        A1h = h * A(t1)
        A2h = h * A(t2)
        
        B = 0.5 * (A1h + A2h)
        C = (A2h - A1h) * self._sqrt3 / 12.0
        K = _commutator(B, C)
        
        Omega = B - K / 12.0
        U = _matrix_exp(Omega)
        y_next = _apply_matrix(U, y0)
        
        # 返回主步結果和用於密集輸出的中間矩陣
        return y_next, {'B': B, 'C': C}

# -----------------------------------------------------------------------------
# 連續延拓 (密集輸出) - 簡化為只支持一種方案
# -----------------------------------------------------------------------------
class DenseOutput:
    def __init__(self, steps_data: list, order: int):
        self.steps_data = steps_data
        self.order = order
        self.t_grid = torch.tensor([s[0] for s in steps_data] + [steps_data[-1][0] + steps_data[-1][1]])
        if len(self.steps_data) > 1 and self.t_grid[0] > self.t_grid[-1]:
             self.t_grid = torch.flip(self.t_grid, dims=[0])

    def __call__(self, t_batch: Tensor) -> Tensor:
        
        indices = torch.searchsorted(self.t_grid, t_batch, right=True) - 1
        indices = torch.clamp(indices, 0, len(self.steps_data) - 1)
        
        t0, h, y0, data_dicts = zip(*[self.steps_data[i] for i in indices])
        
        t0 = torch.tensor(t0, device=t_batch.device, dtype=t_batch.dtype)
        h = torch.tensor(h, device=t_batch.device, dtype=t_batch.dtype)
        y0 = torch.stack(y0, dim=-2)
        
        theta = (t_batch - t0) / h
        theta_b = theta.view(-1, 1, 1)

        if self.order == 2:
            A1 = torch.stack([d['A1'] for d in data_dicts], dim=-3)
            Omega_theta_batch = theta_b * A1 * h.view(-1, 1, 1)
            U_batch = _matrix_exp(Omega_theta_batch)
            return _apply_matrix(U_batch, y0)
        
        elif self.order == 4:
            B = torch.stack([d['B'] for d in data_dicts], dim=-3)
            C = torch.stack([d['C'] for d in data_dicts], dim=-3)
            K = _commutator(B, C)
            Omega_theta_batch = theta_b * B - (theta_b**2) / 12.0 * K
            U_batch = _matrix_exp(Omega_theta_batch)
            return _apply_matrix(U_batch, y0)
        else:
            raise NotImplementedError(f"Dense output for order {self.order} is not implemented.")

# -----------------------------------------------------------------------------
# ODE 求解器接口 - 簡化
# -----------------------------------------------------------------------------
def magnus_solve(
    y0: Tensor, t_span: TimeSpan, 
    functional_A_func: Callable, p_dict: Dict[str, Tensor],
    order: int = 4, rtol: float = 1e-6, atol: float = 1e-8, 
    return_traj: bool = False, dense_output: bool = False, 
    max_steps: int = 10_000
):
    """
    使用自適應步長的Magnus積分器求解一批（或單個）線性ODE。
    採用了統一的邏輯來處理正向/反向積分以及批次化/非批次化情況。
    """
    if order == 2: integrator = Magnus2nd()
    elif order == 4: integrator = Magnus4th()
    else: raise ValueError("order must be 2 or 4")

    t0, t1 = float(t_span[0]), float(t_span[1])
    if t0 == t1:
        if return_traj: return y0.unsqueeze(-2), torch.tensor([t0])
        return y0

    # 使用帶符號的步長 dt 來統一處理正向和反向積分
    dt = t1 - t0
    t, y = t0, y0.clone()
    ts, ys, steps_data = [t], [y], []
    step_cnt = 0
    
    # 將 p_dict 綁定到 A_func
    A_func_bound = lambda tau: functional_A_func(tau, p_dict)

    while (t - t1) * dt < 0:
        if step_cnt >= max_steps:
            raise RuntimeError("Maximum number of steps reached.")
        if (t + dt - t1) * dt > 0:
            dt = t1 - t

        dt_half = 0.5 * dt
        y_big, data_big = integrator(A_func_bound, t, dt, y)
        y_half, _ = integrator(A_func_bound, t, dt_half, y)
        y_small, _ = integrator(A_func_bound, t + dt_half, dt_half, y_half)

        # --- 統一的誤差控制和步長調整 ---
        # err 和 tol 的形狀為 batch_shape
        err = torch.norm(y_small - y_big, dim=-1)
        tol = atol + rtol * torch.norm(y_small, dim=-1)
        accept_step = torch.all(err <= tol)

        if accept_step or abs(dt) < 1e-12:
            if dense_output:
                steps_data.append((t, dt, y.detach(), data_big))
            t += dt
            y = y_small
            if return_traj:
                ts.append(t)
                ys.append(y)

        safety, fac_min, fac_max = 0.9, 0.2, 5.0
        
        # 為 err 增加一個極小的 epsilon 以保證數值穩定性
        err_safe = err + 1e-16
        
        # 計算所有系統的步長調整因子
        factors = safety * (tol / err_safe).pow(1.0 / (integrator.order + 1))
        
        # 選擇最保守（最小）的因子來確保對所有系統都安全
        factor = torch.min(factors)
        
        dt = dt * float(max(fac_min, min(fac_max, factor)))
        
        step_cnt += 1

    if dense_output:
        if not steps_data:
            _, data_big = integrator(A_func_bound, t0, t1 - t0, y0)
            return DenseOutput([(t0, t1 - t0, y0, data_big)], order, functional_A_func, p_dict)
        return DenseOutput(steps_data, order)
    
    if return_traj:
        return torch.stack(ys, dim=-2), torch.tensor(ts)
    
    return y

def _magnus_odeint_core(
    functional_A_func: Callable, p_dict: Dict, 
    y0: Tensor, t_vec: Tensor,
    order: int, rtol: float, atol: float
) -> Tensor:
    """Internal integration loop without input preparation."""
    ys_out = [y0]
    y_curr = y0
    for i in range(len(t_vec) - 1):
        t0, t1 = float(t_vec[i]), float(t_vec[i + 1])
        y_next = magnus_solve(y_curr, (t0, t1), functional_A_func, p_dict, order, rtol, atol)
        ys_out.append(y_next)
        y_curr = y_next
    return torch.stack(ys_out, dim=-2)

def magnus_odeint(
    A_func_or_module: Union[Callable, nn.Module], y0: Tensor, t: Union[Sequence[float], torch.Tensor],
    params: Tensor = None, # MODIFIED: 允許為 None
    order: int = 4, rtol: float = 1e-6, atol: float = 1e-8
) -> Tensor:
    # MODIFIED: 調用接口適配器
    functional_A_func, p_dict = _prepare_functional_call(A_func_or_module, params)
    
    t_vec = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)
    # MODIFIED: Call the refactored core loop
    return _magnus_odeint_core(functional_A_func, p_dict, y0, t_vec, order, rtol, atol)

# -----------------------------------------------------------------------------
# 模块化的积分后端
# -----------------------------------------------------------------------------
class BaseQuadrature(nn.Module):
    def forward(self, interp_func: Callable, functional_A_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

class AdaptiveGaussKronrod(BaseQuadrature):
    _GK_NODES_RAW = [-0.99145537112081263920685469752598, -0.94910791234275852452618968404809, -0.86486442335976907278971278864098, -0.7415311855993944398638647732811, -0.58608723546769113029414483825842, -0.40584515137739716690660641207707, -0.20778495500789846760068940377309, 0.0]
    _GK_WEIGHTS_K_RAW = [0.022935322010529224963732008059913, 0.063092092629978553290700663189093, 0.10479001032225018383987632254189, 0.14065325971552591874518959051021, 0.16900472663926790282658342659795, 0.19035057806478540991325640242055, 0.20443294007529889241416199923466, 0.20948214108472782801299917489173]
    _GK_WEIGHTS_G_RAW = [0.12948496616886969327061143267787, 0.2797053914892766679014677714229, 0.38183005050511894495036977548818, 0.41795918367346938775510204081658]
    _rule_cache = {}

    @classmethod
    def _get_rule(cls, dtype, device):
        if (dtype, device) in cls._rule_cache: return cls._rule_cache[(dtype, device)]
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

    def _eval_segment(self, interp_func, functional_A_func, a, b, params_req, nodes, weights_k, weights_g):
        h = (b - a) / 2.0
        c = (a + b) / 2.0
        segment_nodes = c + h * nodes
        z_eval = interp_func(segment_nodes)
        y_eval, a_eval = z_eval.tensor_split(2, dim=-1)

        def f_batched_for_vjp(p_dict):
            A_batch = functional_A_func(segment_nodes, p_dict)
            return _apply_matrix(A_batch, y_eval)

        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(f_batched_for_vjp, params_req)
            cotangent_K = h * weights_k * a_eval
            cotangent_G = h * weights_g * a_eval
            I_K = vjp_fn(cotangent_K)[0]
            I_G = vjp_fn(cotangent_G)[0]
        
        # MODIFIED: 計算字典結構的誤差
        diff_dict = {k: I_K[k] - I_G[k] for k in I_K}
        error = torch.sqrt(sum(torch.norm(v)**2 for v in diff_dict.values()))
        return I_K, error

    def forward(self, interp_func: Callable, functional_A_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Dict[str, Tensor], max_segments: int = 100) -> Dict[str, Tensor]:
        if a == b:
            return {k: torch.zeros_like(v) for k, v in params_req.items()}

        # 獲取第一個參數的 dtype 和 device 作為參考
        ref_param = next(iter(params_req.values()))
        nodes, weights_k, weights_g = self._get_rule(ref_param.dtype, ref_param.device)
        
        # MODIFIED: 初始化字典結構的積分值和誤差
        I_total = {k: torch.zeros_like(v) for k, v in params_req.items()}
        E_total = torch.tensor(0.0, dtype=ref_param.dtype, device=ref_param.device)
        
        I_K, error = self._eval_segment(interp_func, functional_A_func, a, b, params_req, nodes, weights_k, weights_g)
        heap = [(-error.item(), a, b, I_K, error)]

        # MODIFIED: 字典累加
        for k in I_total: I_total[k] += I_K[k]
        E_total += error
        
        machine_eps = torch.finfo(ref_param.dtype).eps

        while heap:
            # MODIFIED: 計算字典結構的總範數
            I_total_norm = torch.sqrt(sum(torch.norm(v)**2 for v in I_total.values()))
            if E_total <= atol + rtol * I_total_norm:
                break
            if len(heap) >= max_segments:
                warnings.warn(f"Max segments ({max_segments}) reached. Result may be inaccurate.")
                break

            neg_err_parent, a_parent, b_parent, I_K_parent, err_parent = heapq.heappop(heap)
            
            for k in I_total: I_total[k] -= I_K_parent[k]
            E_total -= err_parent

            mid = (a_parent + b_parent) / 2.0
            if abs(b_parent - a_parent) < machine_eps * 100:
                for k in I_total: I_total[k] += I_K_parent[k]
                E_total += err_parent
                warnings.warn(f"Interval {b_parent - a_parent} too small to subdivide further.")
                continue

            I_K_left, err_left = self._eval_segment(interp_func, functional_A_func, a_parent, mid, params_req, nodes, weights_k, weights_g)
            I_K_right, err_right = self._eval_segment(interp_func, functional_A_func, mid, b_parent, params_req, nodes, weights_k, weights_g)

            heapq.heappush(heap, (-err_left.item(), a_parent, mid, I_K_left, err_left))
            heapq.heappush(heap, (-err_right.item(), mid, b_parent, I_K_right, err_right))
            
            for k in I_total: I_total[k] += I_K_left[k] + I_K_right[k]
            E_total += err_left + err_right
            
        return I_total

class FixedSimpson(BaseQuadrature):
    """固定步長的複合辛普森法則積分器。"""
    def __init__(self, N=100):
        super().__init__()
        if N % 2 != 0:
            warnings.warn("N should be even for Simpson's rule; incrementing N by 1.")
            N += 1
        self.N = N

    def forward(self, interp_func: Callable, functional_A_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if a == b:
            return {k: torch.zeros_like(v) for k, v in params_req.items()}

        ref_param = next(iter(params_req.values()))
        nodes = torch.linspace(a, b, self.N + 1, device=ref_param.device, dtype=ref_param.dtype)
        h = (b - a) / self.N

        z_eval = interp_func(nodes)
        y_eval, a_eval = z_eval.tensor_split(2, dim=-1)

        def f_batched_for_vjp(p_dict):
            A_batch = functional_A_func(nodes, p_dict)
            return _apply_matrix(A_batch, y_eval)

        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(f_batched_for_vjp, params_req)
            
            weights = torch.ones(self.N + 1, device=a_eval.device, dtype=a_eval.dtype)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            weights *= (h / 3.0)
            
            cotangent = weights.unsqueeze(1) * a_eval
            integral_dict = vjp_fn(cotangent)[0]

        return integral_dict

# -----------------------------------------------------------------------------
# 解耦伴隨法 (採用連續 Magnus 延拓)
# -----------------------------------------------------------------------------
class _MagnusAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, t, functional_A_func, param_keys, order, rtol, atol, quad_method, quad_options, *param_values):
        # --- CORRECTED: Reconstruct dictionary from unpacked args ---
        params_and_buffers_dict = dict(zip(param_keys, param_values))
        
        t = t.to(y0.dtype)
        with torch.no_grad():
            # Call the core integration loop
            y_traj = _magnus_odeint_core(
                functional_A_func, 
                params_and_buffers_dict,
                y0, t, order, rtol, atol
            )

        # Save context for the backward pass
        ctx.functional_A_func = functional_A_func
        ctx.param_keys = param_keys
        ctx.order, ctx.rtol, ctx.atol = order, rtol, atol
        ctx.quad_method, ctx.quad_options = quad_method, quad_options
        ctx.y0_requires_grad = y0.requires_grad
        
        # Save all tensors that might be needed for gradient computation
        ctx.save_for_backward(t, y_traj, *param_values)
        
        return y_traj

    @staticmethod
    def backward(ctx, grad_y_traj: Tensor):
        # Unpack saved tensors
        saved_tensors = ctx.saved_tensors
        t, y_traj = saved_tensors[0], saved_tensors[1]
        param_values = saved_tensors[2:]
        
        # Unpack non-tensor context
        functional_A_func = ctx.functional_A_func
        param_keys = ctx.param_keys
        order, rtol, atol = ctx.order, ctx.rtol, ctx.atol
        quad_method, quad_options = ctx.quad_method, ctx.quad_options

        # Reconstruct dictionaries
        full_p_and_b_dict = dict(zip(param_keys, param_values))
        params_req = {k: v for k, v in full_p_and_b_dict.items() if v.requires_grad}
        buffers_dict = {k: v for k, v in full_p_and_b_dict.items() if not v.requires_grad}

        if not params_req:
            # If no parameters require gradients, no need to do any work
            num_params = len(param_values)
            return (None,) * (9 + num_params)


        if quad_method == 'gk':
            quad_integrator = AdaptiveGaussKronrod()
        elif quad_method == 'simpson':
            quad_integrator = FixedSimpson(**quad_options)
        else:
            raise ValueError(f"Unknown quadrature method: {quad_method}")

        T, dim = y_traj.shape[-2], y_traj.shape[-1]
        adj_y = grad_y_traj[..., -1, :].clone()
        # Initialize the gradient dictionary
        adj_params = {k: torch.zeros_like(v) for k, v in params_req.items()}

        def augmented_A_func(t_val: Union[float, Tensor], p_and_b_dict: Dict) -> Tensor:
            A = functional_A_func(t_val, p_and_b_dict)
            A_T_neg = -A.transpose(-1, -2)
            zeros = torch.zeros_like(A)
            top = torch.cat([A, zeros], dim=-1)
            bottom = torch.cat([zeros, A_T_neg], dim=-1)
            return torch.cat([top, bottom], dim=-2)

        for i in range(T - 1, 0, -1):
            t_i, t_prev = float(t[i]), float(t[i - 1])
            y_i = y_traj[..., i, :]
            z_i = torch.cat([y_i, adj_y], dim=-1)
            
            full_p_dict_for_solve = {**params_req, **buffers_dict}
            
            with torch.no_grad():
                dense_output_solver = magnus_solve(
                    z_i, (t_i, t_prev), 
                    lambda t_val, p: augmented_A_func(t_val, p), full_p_dict_for_solve,
                    order=order, rtol=rtol, atol=atol, dense_output=True
                )

            def A_func_for_quadrature(t_val, p_dict_req):
                full_dict = {**p_dict_req, **buffers_dict}
                return functional_A_func(t_val, full_dict)

            quad_atol = atol * 0.1
            quad_rtol = rtol * 0.1
            
            integral_val_dict = quad_integrator(
                dense_output_solver, A_func_for_quadrature, 
                t_i, t_prev, quad_atol, quad_rtol, params_req=params_req
            )
            
            for k in adj_params:
                adj_params[k].sub_(integral_val_dict[k])

            z_prev = dense_output_solver(torch.tensor([t_prev], device=z_i.device, dtype=z_i.dtype)).squeeze(-2)
            adj_y = z_prev.narrow(-1, dim, dim).clone()
            adj_y.add_(grad_y_traj[..., i-1, :])

        grad_y0 = adj_y if ctx.y0_requires_grad else None
        
        # Build the tuple of gradients for *param_values
        grad_param_values = tuple(adj_params.get(key) for key in param_keys)

        # The return tuple must match the inputs to forward()
        return (
            grad_y0, # grad for y0
            None,    # grad for t
            None,    # grad for functional_A_func
            None,    # grad for param_keys
            None,    # grad for order
            None,    # grad for rtol
            None,    # grad for atol
            None,    # grad for quad_method
            None,    # grad for quad_options
            *grad_param_values # UNPACKED grads for *param_values
        )

# -----------------------------------------------------------------------------
# 用戶友好接口 (MODIFIED)
# -----------------------------------------------------------------------------
def magnus_odeint_adjoint(
    A_func_or_module: Union[Callable, nn.Module], y0: Tensor, t: Union[Sequence[float], torch.Tensor],
    params: Tensor = None, # MODIFIED: 允許為 None
    order: int = 4, rtol: float = 1e-6, atol: float = 1e-8,
    quad_method: str = 'gk', quad_options: dict = None
) -> Tensor:
    t_vec = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)
    if t_vec.ndim != 1 or t_vec.numel() < 2:
        raise ValueError("t 必須是一維且至少包含兩個時間點")
    
    # Prepare the functional form of A and the parameter dictionary
    functional_A_func, p_and_b_dict = _prepare_functional_call(A_func_or_module, params)
    
    if quad_options is None:
        quad_options = {}
    
    # --- CORRECTED: Unpack parameter tensors as direct arguments to apply ---
    param_keys = list(p_and_b_dict.keys())
    param_values = list(p_and_b_dict.values())
        
    # Pass all tensors and options as a flat list of arguments
    return _MagnusAdjoint.apply(
        y0, t_vec, 
        functional_A_func, 
        param_keys, order, rtol, atol, 
        quad_method, quad_options,
        *param_values # UNPACK the tensors here
    )

