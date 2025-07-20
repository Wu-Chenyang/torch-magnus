import math
from typing import Callable, Sequence, Tuple, Union, List

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
    y0: Tensor, t_span: TimeSpan, A_func: Callable[..., Tensor],
    params: torch.nn.Parameter = torch.tensor([]), order: int = 4,
    rtol: float = 1e-6, atol: float = 1e-8, return_traj: bool = False, 
    dense_output: bool = False, max_steps: int = 10_000
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
        if return_traj:
            # 處理批次化和非批次化的情況
            return y0.unsqueeze(-2), torch.tensor([t0])
        return y0

    # 使用帶符號的步長 dt 來統一處理正向和反向積分
    dt = t1 - t0
    t, y = t0, y0.clone()

    ts, ys, steps_data = [t], [y], []
    step_cnt = 0

    while (t - t1) * dt < 0:
        if step_cnt >= max_steps:
            raise RuntimeError("Maximum number of steps reached.")

        # 統一的邏輯防止超出 t1
        if (t + dt - t1) * dt > 0:
            dt = t1 - t

        dt_half = 0.5 * dt
        y_big, data_big = integrator(lambda tau: A_func(tau, params), t, dt, y)
        y_half, _ = integrator(lambda tau: A_func(tau, params), t, dt_half, y)
        y_small, _ = integrator(lambda tau: A_func(tau, params), t + dt_half, dt_half, y_half)

        # --- 統一的誤差控制和步長調整 ---
        # err 和 tol 的形狀為 [batch_size] 或 []
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
            _, data_big = integrator(lambda tau: A_func(tau, params), t0, t1 - t0, y0)
            return DenseOutput([(t0, t1 - t0, y0, data_big)], order)
        return DenseOutput(steps_data, order)
    
    if return_traj:
        return torch.stack(ys, dim=-2), torch.tensor(ts)
    
    return y

def magnus_odeint(
    A_func: Callable[..., Tensor], y0: Tensor, t: Union[Sequence[float], torch.Tensor],
    params: torch.nn.Parameter = torch.tensor([]), order: int = 4, rtol: float = 1e-6, atol: float = 1e-8
) -> Tensor:
    t_vec = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)
    ys_out = [y0]
    y_curr = y0
    for i in range(len(t_vec) - 1):
        t0, t1 = float(t_vec[i]), float(t_vec[i + 1])
        y_next = magnus_solve(y_curr, (t0, t1), A_func, params, order, rtol, atol)
        ys_out.append(y_next)
        y_curr = y_next
    return torch.stack(ys_out, dim=-2)

# -----------------------------------------------------------------------------
# 模块化的积分后端
# -----------------------------------------------------------------------------
class BaseQuadrature(nn.Module):
    """积分器的抽象基类。"""
    def forward(self, interp_func: Callable, A_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Tensor) -> Tensor:
        raise NotImplementedError

class AdaptiveGaussKronrod(BaseQuadrature):
    """
    自适应高斯-克龙罗德积分器，其设计严格参考了 Julia 的 QuadGK.jl 库。
    此实现采用迭代式算法、优先队列、缓存和全局误差控制，以实现高鲁棒性和效率。
    同时，它保留了通过 `torch.func.vjp` 实现的可微分性。
    """
    # n=7 (15点) 规则的高精度节点和权重, 直接取自 QuadGK.jl
    _GK_NODES_RAW = [-0.99145537112081263920685469752598, -0.94910791234275852452618968404809, -0.86486442335976907278971278864098, -0.7415311855993944398638647732811, -0.58608723546769113029414483825842, -0.40584515137739716690660641207707, -0.20778495500789846760068940377309, 0.0]
    _GK_WEIGHTS_K_RAW = [0.022935322010529224963732008059913, 0.063092092629978553290700663189093, 0.10479001032225018383987632254189, 0.14065325971552591874518959051021, 0.16900472663926790282658342659795, 0.19035057806478540991325640242055, 0.20443294007529889241416199923466, 0.20948214108472782801299917489173]
    _GK_WEIGHTS_G_RAW = [0.12948496616886969327061143267787, 0.2797053914892766679014677714229, 0.38183005050511894495036977548818, 0.41795918367346938775510204081658]

    _rule_cache = {}

    @classmethod
    def _get_rule(cls, dtype, device):
        """获取或创建并缓存积分规则（节点和权重）。"""
        if (dtype, device) in cls._rule_cache:
            return cls._rule_cache[(dtype, device)]

        nodes_neg = torch.tensor(cls._GK_NODES_RAW, dtype=dtype, device=device)
        nodes = torch.cat([-nodes_neg[0:-1].flip(0), nodes_neg])

        weights_k_half = torch.tensor(cls._GK_WEIGHTS_K_RAW, dtype=dtype, device=device)
        weights_k = torch.cat([weights_k_half[0:-1].flip(0), weights_k_half])

        weights_g_half = torch.tensor(cls._GK_WEIGHTS_G_RAW, dtype=dtype, device=device)
        # For n=7, there are 7 Gauss points (excluding the center). The weights are symmetric.
        weights_g_embedded = torch.cat([weights_g_half[0:-1].flip(0), weights_g_half])

        weights_g = torch.zeros_like(weights_k)
        weights_g[1::2] = weights_g_embedded

        rule = (nodes, weights_k.unsqueeze(1), weights_g.unsqueeze(1))
        cls._rule_cache[(dtype, device)] = rule
        return rule

    def _eval_segment(self, interp_func, A_func, a, b, params_req, nodes, weights_k, weights_g):
        """评估单个积分区间，返回积分估计和误差。"""
        h = (b - a) / 2.0
        c = (a + b) / 2.0
        
        segment_nodes = c + h * nodes
        z_eval = interp_func(segment_nodes)
        y_eval, a_eval = z_eval.tensor_split(2, dim=-1)

        def f_batched_for_vjp(p):
            A_batch = A_func(segment_nodes, p)
            return _apply_matrix(A_batch, y_eval)

        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(f_batched_for_vjp, params_req)
            cotangent_K = h * weights_k * a_eval
            cotangent_G = h * weights_g * a_eval
            I_K = vjp_fn(cotangent_K)[0]
            I_G = vjp_fn(cotangent_G)[0]

        error = torch.norm(I_K - I_G)
        return I_K, error

    def forward(self, interp_func: Callable, A_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Tensor, max_segments: int = 100) -> Tensor:
        """
        使用迭代式自适应算法和优先队列执行积分。
        """
        if a == b:
            return torch.zeros_like(params_req)

        nodes, weights_k, weights_g = self._get_rule(params_req.dtype, params_req.device)
        
        # C++ Core a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z
        I_total = torch.zeros_like(params_req)
        E_total = torch.tensor(0.0, dtype=params_req.dtype, device=params_req.device)
        
        # 使用最小堆模拟优先队列，存储 (-error, a, b) 以优先处理大误差区间
        # C++ Core a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z
        I_K, error = self._eval_segment(interp_func, A_func, a, b, params_req, nodes, weights_k, weights_g)
        heap = [(-error.item(), a, b, I_K, error)]

        # 初始全局估计
        I_total += I_K
        E_total += error
        
        # 机器精度，用于判断区间是否过小
        machine_eps = torch.finfo(params_req.dtype).eps

        while heap:
            # 检查是否满足全局容忍度
            if E_total <= atol + rtol * torch.norm(I_total):
                break

            if len(heap) >= max_segments:
                warnings.warn(f"Max segments ({max_segments}) reached. Result may be inaccurate.")
                break

            # 从优先队列中取出误差最大的区间
            neg_err_parent, a_parent, b_parent, I_K_parent, err_parent = heapq.heappop(heap)

            # 从全局估计中移除父区间贡献
            I_total -= I_K_parent
            E_total -= err_parent

            # 分裂区间
            mid = (a_parent + b_parent) / 2.0

            # 如果区间过小，则不再分裂，将其贡献加回并继续
            if abs(b_parent - a_parent) < machine_eps * 100:
                I_total += I_K_parent
                E_total += err_parent
                warnings.warn(f"Interval {b_parent - a_parent} too small to subdivide further.")
                continue

            # 评估左、右两个子区间
            I_K_left, err_left = self._eval_segment(interp_func, A_func, a_parent, mid, params_req, nodes, weights_k, weights_g)
            I_K_right, err_right = self._eval_segment(interp_func, A_func, mid, b_parent, params_req, nodes, weights_k, weights_g)

            # 将子区间加入优先队列
            heapq.heappush(heap, (-err_left.item(), a_parent, mid, I_K_left, err_left))
            heapq.heappush(heap, (-err_right.item(), mid, b_parent, I_K_right, err_right))

            # 更新全局估计
            I_total += I_K_left + I_K_right
            E_total += err_left + err_right
            
        return I_total

class FixedSimpson(BaseQuadrature):
    """固定步长的复合辛普森法则积分器。"""
    def __init__(self, N=100):
        super().__init__()
        if N % 2 != 0:
            warnings.warn("N should be even for Simpson's rule; incrementing N by 1.")
            N += 1
        self.N = N

    def forward(self, interp_func: Callable, A_func: Callable, a: float, b: float, atol: float, rtol: float, params_req: Tensor) -> Tensor:
        if a == b:
            return torch.zeros_like(params_req)

        nodes = torch.linspace(a, b, self.N + 1, device=params_req.device, dtype=params_req.dtype)
        h = (b - a) / self.N

        z_eval = interp_func(nodes)
        y_eval, a_eval = z_eval.tensor_split(2, dim=-1)

        def f_batched_for_vjp(p):
            A_batch = A_func(nodes, p)
            return _apply_matrix(A_batch, y_eval)

        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(f_batched_for_vjp, params_req)
            
            weights = torch.ones(self.N + 1, device=a_eval.device, dtype=a_eval.dtype)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            weights *= (h / 3.0)
            
            cotangent = weights.unsqueeze(1) * a_eval
            integral = vjp_fn(cotangent)[0]

        return integral

# -----------------------------------------------------------------------------
# 解耦伴隨法 (採用連續 Magnus 延拓)
# -----------------------------------------------------------------------------
class _MagnusAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, t, params, order, rtol, atol, quad_method, quad_options, A_func):
        t = t.to(y0.dtype)
        with torch.no_grad():
            y_traj = magnus_odeint(A_func, y0, t, params, order=order, rtol=rtol, atol=atol)
        ctx.save_for_backward(t, y_traj, params)
        ctx.order, ctx.rtol, ctx.atol = order, rtol, atol
        ctx.A_func = A_func
        ctx.quad_method = quad_method
        ctx.quad_options = quad_options
        return y_traj

    @staticmethod
    def backward(ctx, grad_y_traj: Tensor):
        t, y_traj, params = ctx.saved_tensors
        order, rtol, atol, A_func = ctx.order, ctx.rtol, ctx.atol, ctx.A_func
        quad_method, quad_options = ctx.quad_method, ctx.quad_options
        
        if quad_method == 'gk':
            quad_integrator = AdaptiveGaussKronrod()
        elif quad_method == 'simpson':
            quad_integrator = FixedSimpson(**quad_options)
        else:
            raise ValueError(f"Unknown quadrature method: {quad_method}")

        T, dim = y_traj.shape[-2], y_traj.shape[-1]
        adj_y = grad_y_traj[..., -1, :].clone()
        adj_params = torch.zeros_like(params)
        params_req = params.detach().requires_grad_(True)

        def augmented_A_func(t_val: Union[float, Tensor], p: Tensor) -> Tensor:
            A = A_func(t_val, p)
            A_T_neg = -A.transpose(-1, -2)
            zeros = torch.zeros_like(A)
            top = torch.cat([A, zeros], dim=-1)
            bottom = torch.cat([zeros, A_T_neg], dim=-1)
            return torch.cat([top, bottom], dim=-2)

        for i in range(T - 1, 0, -1):
            t_i, t_prev = float(t[i]), float(t[i - 1])
            y_i = y_traj[..., i, :]
            
            z_i = torch.cat([y_i, adj_y], dim=-1)
            
            with torch.no_grad():
                dense_output_solver = magnus_solve(
                    z_i, (t_i, t_prev), lambda t,p: augmented_A_func(t,p), params_req,
                    order=order, rtol=rtol, atol=atol, dense_output=True
                )

            quad_atol = atol * 0.1
            quad_rtol = rtol * 0.1
            
            integral_val = quad_integrator(
                dense_output_solver, A_func, t_i, t_prev, quad_atol, quad_rtol, params_req=params_req
            )
            
            adj_params.sub_(integral_val)
            z_prev = dense_output_solver(torch.tensor([t_prev], device=z_i.device, dtype=z_i.dtype)).squeeze(-2)
            adj_y = z_prev.narrow(-1, dim, dim).clone()
            adj_y.add_(grad_y_traj[..., i-1, :])

        grad_y0 = adj_y
        
        return grad_y0, None, adj_params, None, None, None, None, None, None

# -----------------------------------------------------------------------------
# 用戶友好接口 - 移除 interp_method
# -----------------------------------------------------------------------------
def magnus_odeint_adjoint(
    A_func : Callable[..., Tensor], y0: Tensor, t: Union[Sequence[float], torch.Tensor],
    params : Tensor = torch.tensor([]), order: int = 4, rtol: float = 1e-6, atol: float = 1e-8,
    quad_method: str = 'gk', quad_options: dict = None
) -> Tensor:
    """
    使用 Magnus 展開和解耦伴隨法求解線性 ODE: y'(t) = A(t,θ) y(t)，並支持梯度計算。

    重要 API 約定:
    為了達到最佳性能，用戶提供的 `A_func` 必須支持批處理時間輸入。
    即，當 `t` 是一個形狀為 `[N]` 的張量時，`A_func(t, params)` 必須返回一個形狀為 `[N, dim, dim]` 的張量。

    參數:
    ...
    quad_method (str): 用於計算參數梯度的積分方法。
                       可選: 'gk' (自適應高斯-克龍羅德), 'simpson'。
    quad_options (dict): 傳遞給固定步長積分器的選項，例如 {'N': 200}。
    """
    t_vec = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)
    if t_vec.ndim != 1 or t_vec.numel() < 2:
        raise ValueError("t 必須是一維且至少包含兩個時間點")
    
    if quad_options is None:
        quad_options = {}
        
    return _MagnusAdjoint.apply(y0, t_vec, params, order, rtol, atol, quad_method, quad_options, A_func)
