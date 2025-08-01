from typing import Callable, Sequence, Tuple, Union, List, Dict

import torch
import torch.nn as nn

from .butcher import GL2, GL4, GL6
from .stepper import Collocation, Magnus2nd, Magnus4th, Magnus6th
from .quadrature import AdaptiveGaussKronrod, FixedSimpson
from .utils import _apply_matrix
from .dense_output import CollocationDenseOutput, DenseOutputNaive, _merge_collocation_dense_outputs, _merge_naive_dense_outputs

Tensor = torch.Tensor
TimeSpan = Union[Tuple[float, float], List[float], torch.Tensor]

def _prepare_functional_call(A_func_or_module: Union[Callable, nn.Module], params: Tensor = None) -> Tuple[Callable, Dict[str, Tensor]]:
    """
    Convert user input A_func (Module or Callable) to unified functional interface.
    
    Args:
        A_func_or_module: Either a torch.nn.Module or a callable
        params: Optional parameter tensor for callable interface
        
    Returns:
        functional_A_func: A function that accepts (t, p_dict)
        params_and_buffers_dict: Dictionary containing all parameters and buffers
    """
    if isinstance(A_func_or_module, torch.nn.Module):
        module = A_func_or_module
        # Combine parameters and buffers to support functional_call
        params_and_buffers = {
            **dict(module.named_parameters()),
            **dict(module.named_buffers())
        }
        
        def functional_A_func(t_val, p_and_b_dict):
            # Use functional_call for stateless module execution
            return torch.func.functional_call(module, p_and_b_dict, (t_val,))
        
        return functional_A_func, params_and_buffers
    else:
        # Handle legacy (Callable, params) interface
        A_func = A_func_or_module

        if params is None:
            # System has no trainable parameters
            params_dict = {}
            def functional_A_func(t_val, p_dict):
                return A_func(t_val, None)
            return functional_A_func, params_dict

        elif isinstance(params, torch.Tensor):
            # Legacy interface with single params tensor
            params_dict = {'params': params}
            def functional_A_func(t_val, p_dict):
                # Unpack from dictionary and call original function
                return A_func(t_val, p_dict['params'])
            return functional_A_func, params_dict
        
        else:
            raise TypeError(f"The 'params' argument must be a torch.Tensor or None, but got {type(params)}")

# -----------------------------------------------------------------------------
# Adaptive Stepping
# -----------------------------------------------------------------------------
def _richardson_step(integrator, sys_func, t: float, dt: float, y):
    """
    Performs a step using Richardson extrapolation for adaptive step sizing.

    This function computes the solution with one full step (y_big) and two 
    half-steps (y_small). The difference is used to estimate the error and
    a higher-order solution (y_next) for error control and dense output.

    Args:
        integrator: The Magnus integrator instance.
        sys_func: The function returning A(t) or (A(t), g(t)).
        t: Current time.
        dt: Current step size.
        y: Current solution tensor.

    Returns:
        A tuple containing:
        - y_next (Tensor): Higher-order solution estimate (extrapolated).
        - err (Tensor): Norm of the estimated local error.
        - A_nodes_step (Tensor): Matrix values at quadrature nodes
        - g_nodes_step (Tensor): Vectors values at quadrature nodes
        - t_nodes_step (Tensor): Time values at quadrature nodes
    """

    t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    dt = torch.as_tensor(dt, dtype=y.dtype, device=y.device)
    dt_half = 0.5 * dt

    big_t_nodes = integrator.tableau.get_t_nodes(t, dt)
    half_t_nodes = integrator.tableau.get_t_nodes(t, dt_half)
    small_t_nodes = integrator.tableau.get_t_nodes(t+dt_half, dt_half)
    t_nodes = torch.cat([big_t_nodes, half_t_nodes, small_t_nodes], dim=0)

    syn_stages = sys_func(t_nodes)

    is_nonhomogeneous = isinstance(syn_stages, tuple) and len(syn_stages) == 2
    if is_nonhomogeneous:
        A_nodes_flat, g_nodes_flat = syn_stages
    else:
        A_nodes_flat, g_nodes_flat = syn_stages, None

    g_nodes = None
    A_nodes = A_nodes_flat.reshape(A_nodes_flat.shape[:-3] + (3, -1) + A_nodes_flat.shape[-2:])
    if g_nodes_flat is not None:
        g_nodes = g_nodes_flat.view(g_nodes_flat.shape[:-2] + (3, -1) + g_nodes_flat.shape[-1:])

        y_bighalf = integrator.get_next_y((A_nodes[..., 0:2, :, :, :].flatten(start_dim=-4, end_dim=-3), g_nodes[..., 0:2, :, :].flatten(start_dim=-3, end_dim=-2)), torch.stack([dt, dt_half]), y.unsqueeze(-2))
        y_big, y_half = y_bighalf[..., 0, :], y_bighalf[..., 1, :]
        y_small = integrator.get_next_y((A_nodes[..., 2, :, :, :], g_nodes[..., 2, :, :]), dt_half, y_half)
    else:
        y_bighalf = integrator.get_next_y(A_nodes[..., 0:2, :, :, :].flatten(start_dim=-4, end_dim=-3), torch.stack([dt, dt_half]), y.unsqueeze(-2))
        y_big, y_half = y_bighalf[..., 0, :], y_bighalf[..., 1, :]
        y_small = integrator.get_next_y(A_nodes[..., 2, :, :, :], dt_half, y_half)

    # Richardson extrapolation for a higher-order solution and error estimation
    y_extrap = y_small + (y_small - y_big) / (2**integrator.order - 1)
    err = torch.norm(y_extrap - y_big, dim=-1)
    
    return y_extrap, err, t_nodes, A_nodes_flat, g_nodes_flat

# -----------------------------------------------------------------------------
# ODE Solver Interface
# -----------------------------------------------------------------------------

def adaptive_ode_solve(
    y0: Tensor, t_span: TimeSpan, 
    functional_A_func: Callable, p_dict: Dict[str, Tensor],
    method: str = 'magnus', order: int = 4, rtol: float = 1e-6, atol: float = 1e-8, 
    return_traj: bool = False, dense_output: bool = False, 
    max_steps: int = 10_000, dense_output_method: str = 'naive',

):
    """
    Generic adaptive step-size solver for linear ODEs.
    
    Solves dy/dt = A(t, params) * y with a given stepper module.
    
    Args:
        y0: Initial conditions of shape (*batch_shape, dim)
        t_span: Integration interval (t0, t1)
        functional_A_func: Matrix function A(t, params) returning (*batch_shape, *time_shape, dim, dim)
        p_dict: Parameter dictionary
        method: Integration method ('magnus' or 'glrk')
        order: Integrator order (2, 4, or 6)
        rtol: Relative tolerance for adaptive stepping
        atol: Absolute tolerance for adaptive stepping
        return_traj: If True, return trajectory at all time steps
        dense_output: If True, return DenseOutput object for continuous interpolation
        max_steps: Maximum number of integration steps
        
    Returns:
        If return_traj=True: Tuple of (solution_trajectory, time_points)
            where solution has shape (*batch_shape, len(times), dim)
        If dense_output=True: DenseOutput object
        Otherwise: Final solution of shape (*batch_shape, dim)
    """
    if method == 'magnus':
        if order == 2: integrator = Magnus2nd()
        elif order == 4: integrator = Magnus4th()
        elif order == 6: integrator = Magnus6th()
        else: raise ValueError(f"Invalid order {order} for Magnus method")
    elif method == 'glrk':
        if order == 2: integrator = Collocation(GL2)
        elif order == 4: integrator = Collocation(GL4)
        elif order == 6: integrator = Collocation(GL6)
        else: raise ValueError(f"Invalid order {order} for GLRK method")
    else:
        raise ValueError(f"Unknown integration method: {method}")

    # Bind p_dict to A_func
    A_func_bound = lambda tau: functional_A_func(tau, p_dict)

    t0, t1 = float(t_span[0]), float(t_span[1])
    assert t0 != t1

    # Use signed step size dt to unify forward and backward integration
    dt = t1 - t0
    t, y = t0, y0.clone()
    ts, ys = [t], [y]
    t_nodes_traj, A_nodes_traj, g_nodes_traj = [], [], []
    step_cnt = 0

    while (t - t1) * dt < 0:
        if step_cnt >= max_steps:
            raise RuntimeError("Maximum number of steps reached.")
        if (t + dt - t1) * dt > 0:
            dt = t1 - t

        y_next, err, t_nodes_step, A_nodes_step, g_nodes_step = _richardson_step(
            integrator, A_func_bound, t, dt, y
        )

        tol = atol + rtol * torch.norm(y_next, dim=-1)
        accept_step = torch.all(err <= tol)

        if accept_step or abs(dt) < 1e-12:
            y = y_next
            if return_traj or dense_output:
                ts.append(t+dt)
                ys.append(y)
                if dense_output:
                    t_nodes_traj.append(t_nodes_step)
                    A_nodes_traj.append(A_nodes_step)
                    g_nodes_traj.append(g_nodes_step)
            t += dt

        safety, fac_min, fac_max = 0.9, 0.2, 5.0
        
        # Add small epsilon to err for numerical stability
        err_safe = err + torch.finfo(err.dtype).eps
        
        # Calculate step size adjustment factors for all systems
        factors = safety * (tol / err_safe).pow(1.0 / (integrator.order + 1))
        
        # Choose the most conservative (smallest) factor to ensure safety for all systems
        factor = torch.min(factors).item()
        
        dt = dt * float(max(fac_min, min(fac_max, factor)))
        
        step_cnt += 1


    if return_traj or dense_output:
        ys_out = torch.stack(ys, dim=-2)
        ts_out = torch.tensor(ts, device=y0.device, dtype=y0.dtype)

        if dense_output:
            if dense_output_method.startswith('collocation'):
                if 'ondemand' in dense_output_method:
                    dense_mode = 'ondemand'
                else:
                    dense_mode = 'precompute'
                t_nodes_out = torch.stack(t_nodes_traj, dim=-1)
                A_nodes_out = torch.stack(A_nodes_traj, dim=-3)
                g_nodes_out = torch.stack(g_nodes_traj, dim=-2) if g_nodes_traj[0] is not None else None
                dense_sol = CollocationDenseOutput(ts_out, ys_out, t_nodes_out, A_nodes_out, g_nodes_out, order, dense_mode=dense_mode)
            else:
                dense_sol = DenseOutputNaive(ts_out, ys_out, order, A_func_bound, method)

        if return_traj and dense_output:
            return dense_sol, ys_out, ts_out
        elif return_traj:
            return ys_out, ts_out
        else:
            return dense_sol
    return y


def odeint(
    system_func_or_module: Union[Callable, nn.Module], y0: Tensor, t: Union[Sequence[float], torch.Tensor],
    params: Tensor = None,
    method: str = 'magnus', order: int = 4, rtol: float = 1e-6, atol: float = 1e-8,
    dense_output: bool = False,
    dense_output_method: str = 'collocation_precompute',
    return_traj: bool = False
) -> Union[Tensor, DenseOutputNaive, CollocationDenseOutput]:
    """
    Solve linear ODE system at specified time points.

    This function solves initial value problems of the form:
    dy/dt = A(t)y(t)  (homogeneous)
    or
    dy/dt = A(t)y(t) + g(t) (non-homogeneous)

    The system type is determined by the return value of `system_func_or_module`.

    Args:
        system_func_or_module: A callable or `torch.nn.Module` that defines the system.
            - For homogeneous systems, it should return the matrix A(t) of shape
              (*batch_shape, dim, dim).
            - For non-homogeneous systems, it should return a tuple (A(t), g(t)), where
              A(t) is the matrix and g(t) is the vector of shape (*batch_shape, dim).
        y0: Initial conditions of shape (*batch_shape, dim).
        t: Time points of shape (N,). If dense_output is True, only the first and
           last time points are used to define the integration interval.
        params: Parameter tensor (for callable interface) or None.
        method: Integration method ('magnus' or 'glrk').
        order: Integrator order (2, 4, or 6).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        dense_output: If True, return a `DenseOutput` object for interpolation.
                      Otherwise, return a tensor with solutions at time points `t`.
        dense_output_method: Method for dense output ('naive', 'collocation_precompute', 'collocation_ondemand').
        return_traj: If True, return trajectory at all time steps

    Returns:
        If dense_output is False (default):
            Solution trajectory of shape (*batch_shape, N, dim).
        If dense_output is True:
            A `DenseOutput` object capable of interpolating the solution.
    """
    functional_system_func, p_dict = _prepare_functional_call(system_func_or_module, params)
    t_vec = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)

    # --- Probe system function to determine mode (homogeneous vs. non-homogeneous) ---
    with torch.no_grad():
        probe_result = functional_system_func(t_vec[0], p_dict)

    if isinstance(probe_result, torch.Tensor):
        is_nonhomogeneous = False
        solver_func = functional_system_func
        y_in = y0
        output_slicer = lambda sol: sol # Default no slicing
    elif isinstance(probe_result, tuple) and len(probe_result) == 2:
        is_nonhomogeneous = True
        A_probe, g_probe = probe_result
        dim = y0.shape[-1]

        # Validate shapes
        if not (A_probe.ndim >= 2 and A_probe.shape[-2:] == (dim, dim)):
            raise ValueError(f"Expected A(t) to have shape (..., {dim}, {dim}), but got {A_probe.shape}")
        if not (g_probe.ndim >= 1 and g_probe.shape[-1] == dim):
            raise ValueError(f"Expected g(t) to have shape (..., {dim}), but got {g_probe.shape}")

        if method == 'glrk':
            # For GLRK, pass the original functional_system_func (which returns (A, g))
            solver_func = functional_system_func
            y_in = y0 # y_in remains original y0
            output_slicer = lambda sol: sol # No slicing needed for GLRK non-homogeneous
            
        elif method == 'magnus':
            # For Magnus, use the augmented system
            def augmented_B_func(t_val, p_dict_combined):
                A_t, g_t = functional_system_func(t_val, p_dict_combined)
                batch_dims = A_t.shape[:-2]
                B_t = torch.zeros(*batch_dims, dim + 1, dim + 1, dtype=A_t.dtype, device=A_t.device)
                B_t[..., :dim, :dim] = A_t
                B_t[..., :dim, dim] = g_t
                return B_t
            
            solver_func = augmented_B_func
            ones = torch.ones_like(y0[..., :1])
            y_in = torch.cat([y0, ones], dim=-1) # y_in becomes augmented
            output_slicer = lambda sol: sol[..., :-1] # Slicing needed for Magnus augmented
        else:
            raise ValueError(f"Unknown integration method: {method}")
    else:
        raise TypeError(
            "system_func_or_module must return a Tensor (for homogeneous systems) "
            f"or a Tuple[Tensor, Tensor] (for non-homogeneous systems), but got {type(probe_result)}"
        )

    # --- Solve the ODE ---
    if dense_output:
        sol_out = []
        y_curr = y_in
        ys_traj, ts_traj = [y_in.unsqueeze(-2)], [t_vec[0:1]]
        if dense_output_method.startswith("collocation"):
            if 'ondemand' in dense_output_method:
                dense_mode = 'ondemand'
            else:
                dense_mode = 'precompute'
            dense_output_method = "collocation_ondemand"
        for i in range(len(t_vec) - 1):
            t0, t1 = float(t_vec[i]), float(t_vec[i + 1])
            output = adaptive_ode_solve(
                y_curr, (t0, t1), solver_func, p_dict,
                method, order, rtol, atol, dense_output=True, dense_output_method=dense_output_method, return_traj=return_traj
            )
            if return_traj:
                sol, ys, ts = output[0], output[1], output[2]
                y_next = ys[..., -1, :]
                ys_traj.append(ys[..., 1:, :])
                ts_traj.append(ts[..., 1:])
            else:
                sol = output
                y_next = sol(t1)
            sol_out.append(sol)
            y_curr = y_next

        if dense_output_method.startswith("collocation"):
            solution = _merge_collocation_dense_outputs(sol_out, dense_mode)
        else:
            solution = _merge_naive_dense_outputs(sol_out)
    else:
        ys_out = [y_in]
        y_curr = y_in
        ys_traj, ts_traj = [y_in.unsqueeze(-2)], [t_vec[0:1]]
        for i in range(len(t_vec) - 1):
            t0, t1 = float(t_vec[i]), float(t_vec[i + 1])
            output = adaptive_ode_solve(y_curr, (t0, t1), solver_func, p_dict, method, order, rtol, atol, return_traj=return_traj)
            if return_traj:
                ys, ts = output[0], output[1]
                y_next = ys[..., -1, :]
                ys_traj.append(ys[..., 1:, :])
                ts_traj.append(ts[..., 1:])
            else:
                y_next = output
            ys_out.append(y_next)
            y_curr = y_next
        solution = torch.stack(ys_out, dim=-2)

    if return_traj:
        ys_traj = torch.cat(ys_traj, dim=-2)
        ts_traj = torch.cat(ts_traj, dim=-1)

    # --- Handle output slicing for non-homogeneous case ---
    if is_nonhomogeneous and method == 'magnus':
        if dense_output:
            # Wrap the dense output object to slice the result on-the-fly
            class _SlicedDenseOutput:
                def __init__(self, dense_output_obj, slicer_func):
                    self.dense_output_obj = dense_output_obj
                    self.slicer_func = slicer_func
                def __call__(self, t_batch: Tensor) -> Tensor:
                    Y_interp = self.dense_output_obj(t_batch)
                    return self.slicer_func(Y_interp)
            solution = _SlicedDenseOutput(solution, output_slicer)
        else:
            solution = output_slicer(solution)
    
    if return_traj:
        return solution, ys_traj, ts_traj
    else:
        return solution

# -----------------------------------------------------------------------------
# Decoupled Adjoint Method
# -----------------------------------------------------------------------------

class _Adjoint(torch.autograd.Function):
    """Magnus integrator with memory-efficient adjoint gradient computation."""
    
    @staticmethod
    def forward(ctx, y0, t, functional_system_func, param_keys, method, order, rtol, atol, quad_method, quad_options, *param_values):
        # Reconstruct dictionary from unpacked arguments
        params_and_buffers_dict = dict(zip(param_keys, param_values))
        
        t = t.to(y0.dtype)

        # --- Probe system function to determine mode ---
        with torch.no_grad():
            probe_result = functional_system_func(t[0], params_and_buffers_dict)

        if isinstance(probe_result, torch.Tensor):
            is_nonhomogeneous = False
            solver_func = functional_system_func
            y_in = y0
        elif isinstance(probe_result, tuple) and len(probe_result) == 2:
            is_nonhomogeneous = True
            dim = y0.shape[-1]
            
            if method == 'glrk':
                # For GLRK, pass the original functional_system_func (which returns (A, g))
                solver_func = functional_system_func
                y_in = y0 # y_in remains original y0
                
            elif method == 'magnus':
                def augmented_B_func(t_val, p_dict_combined):
                    A_t, g_t = functional_system_func(t_val, p_dict_combined)
                    g_t = g_t.unsqueeze(-1)
                    *batch_dims, _, _ = A_t.shape
                    B_t = torch.zeros(*batch_dims, dim + 1, dim + 1, dtype=A_t.dtype, device=A_t.device)
                    B_t[..., :dim, :dim] = A_t
                    B_t[..., :dim, dim] = g_t.squeeze(-1)
                    return B_t
                
                solver_func = augmented_B_func
                ones = torch.ones_like(y0[..., :1])
                y_in = torch.cat([y0, ones], dim=-1)
            else:
                raise ValueError(f"Unknown integration method: {method}")
        else:
            raise TypeError(f"System function must return a Tensor or a Tuple[Tensor, Tensor], but got {type(probe_result)}")

        # --- Solve ODE and save context ---
        with torch.no_grad():
            y_dense_traj = adaptive_ode_solve(
                y_in, (t[0], t[-1]), solver_func, params_and_buffers_dict, 
                method=method, order=order, rtol=rtol, atol=atol, dense_output=True
            )
            y_traj_maybe_aug = y_dense_traj(t)

        # --- Save context for backward pass ---
        ctx.is_nonhomogeneous = is_nonhomogeneous
        ctx.functional_system_func = functional_system_func # Save original user func
        ctx.param_keys = param_keys
        ctx.method, ctx.order, ctx.rtol, ctx.atol = method, order, rtol, atol
        ctx.quad_method, ctx.quad_options = quad_method, quad_options
        ctx.y0_requires_grad = y0.requires_grad
        
        # For backward, we need the augmented trajectory for y and the original for a
        ctx.y_dense_traj_aug = y_dense_traj 
        
        ctx.save_for_backward(t, *param_values)
        
        return y_traj_maybe_aug[..., :-1] if is_nonhomogeneous and method == 'magnus' else y_traj_maybe_aug

    @staticmethod
    def backward(ctx, grad_y_traj: Tensor):
        # --- Unpack saved context ---
        t, *param_values = ctx.saved_tensors
        functional_system_func = ctx.functional_system_func
        param_keys = ctx.param_keys
        method, order, rtol, atol = ctx.method, ctx.order, ctx.rtol, ctx.atol
        quad_method, quad_options = ctx.quad_method, ctx.quad_options
        is_nonhomogeneous = ctx.is_nonhomogeneous
        y_dense_traj_aug = ctx.y_dense_traj_aug

        # --- Reconstruct parameter dictionaries ---
        full_p_and_b_dict = dict(zip(param_keys, param_values))
        params_req = {k: v for k, v in full_p_and_b_dict.items() if v.requires_grad}
        buffers_dict = {k: v for k, v in full_p_and_b_dict.items() if not v.requires_grad}

        if not params_req and not ctx.y0_requires_grad:
            num_params = len(param_values)
            return (None,) * (10 + num_params)

        # --- Prepare for backward integration ---
        if quad_method == 'gk':
            quad_integrator = AdaptiveGaussKronrod()
        elif quad_method == 'simpson':
            quad_integrator = FixedSimpson(**quad_options)
        else:
            raise ValueError(f"Unknown quadrature method: {quad_method}")

        T, dim = grad_y_traj.shape[-2], grad_y_traj.shape[-1]
        adj_y = grad_y_traj[..., -1, :].clone()
        adj_params = {k: torch.zeros_like(v) for k, v in params_req.items()}
        full_p_dict_for_solve = {**params_req, **buffers_dict}

        # --- Define the backward dynamics function (Optimized) ---
        def neg_trans_A_func(t_val: Union[float, Tensor], p_and_b_dict: Dict) -> Tensor:
            sys_out = functional_system_func(t_val, p_and_b_dict)
            A = sys_out if not is_nonhomogeneous else sys_out[0]
            return -A.transpose(-1, -2)

        # --- Main backward loop ---
        for i in range(T - 1, 0, -1):
            t_i, t_prev = float(t[i]), float(t[i - 1])

            # Solve the adjoint ODE backward in time (d-dimensional)
            with torch.no_grad():
                a_dense_traj = adaptive_ode_solve(
                    adj_y, (t_i, t_prev), neg_trans_A_func, full_p_dict_for_solve, 
                    method=method, order=order, rtol=rtol, atol=atol, dense_output=True
                )

            # --- Define VJP target and cotangents for quadrature ---
            def f_for_vjp(t_nodes, p_dict_req):
                full_dict = {**p_dict_req, **buffers_dict}
                y_eval_aug = y_dense_traj_aug(t_nodes)
                if is_nonhomogeneous and method == 'magnus':
                    y_eval = y_eval_aug[..., :-1]
                else:
                    y_eval = y_eval_aug
                
                sys_out = functional_system_func(t_nodes, full_dict)
                if is_nonhomogeneous:
                    A, g = sys_out
                    return _apply_matrix(A, y_eval) + g
                else:
                    return _apply_matrix(sys_out, y_eval)

            # Update adjoint state for next segment
            adj_y = a_dense_traj(t_prev)
            adj_y.add_(grad_y_traj[..., i-1, :])

            integral_val_dict = quad_integrator(
                f_for_vjp, a_dense_traj, t_i, t_prev, atol, rtol, params_req, is_nonhomogeneous
            )
            
            for k in adj_params:
                adj_params[k].sub_(integral_val_dict[k])

        grad_y0 = adj_y
        if not ctx.y0_requires_grad:
            grad_y0 = None
        
        grad_param_values = tuple(adj_params.get(key) for key in param_keys)

        return (grad_y0, None, None, None, None, None, None, None, None, None, *grad_param_values)

# -----------------------------------------------------------------------------
# User-Friendly Interface
# -----------------------------------------------------------------------------

def odeint_adjoint(
    system_func_or_module: Union[Callable, nn.Module], y0: Tensor, t: Union[Sequence[float], torch.Tensor],
    params: Tensor = None,
    method: str = 'magnus', order: int = 4, rtol: float = 1e-6, atol: float = 1e-8,
    quad_method: str = 'gk', quad_options: dict = None
) -> Tensor:
    """
    Solve linear ODE system with memory-efficient adjoint gradient computation.
    
    This function provides the same interface as odeint but uses the adjoint
    sensitivity method for efficient gradient computation through the ODE solution.
    
    Args:
        system_func_or_module: A callable or `torch.nn.Module` that defines the system.
            - For homogeneous systems, it should return the matrix A(t) of shape
              (*batch_shape, dim, dim).
            - For non-homogeneous systems, it should return a tuple (A(t), g(t)), where
              A(t) is the matrix and g(t) is the vector of shape (*batch_shape, dim).
        y0: Initial conditions of shape (*batch_shape, dim).
        t: Time points of shape (N,).
        params: Parameter tensor (for callable interface) or None.
        order: Magnus integrator order (2, 4, or 6).
        rtol: Relative tolerance for integration.
        atol: Absolute tolerance for integration.
        quad_method: Quadrature method for adjoint integration ('gk' or 'simpson').
        quad_options: Options dictionary for quadrature method.
        
    Returns:
        Solution trajectory of shape (*batch_shape, N, dim).
    """
    t_vec = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)
    if t_vec.ndim != 1 or t_vec.numel() < 2:
        raise ValueError("t must be 1-dimensional and contain at least two time points")
    
    # Prepare the functional form of A and the parameter dictionary
    functional_system_func, p_and_b_dict = _prepare_functional_call(system_func_or_module, params)
    
    if quad_options is None:
        quad_options = {}
    
    # Unpack parameter tensors as direct arguments to apply
    param_keys = list(p_and_b_dict.keys())
    param_values = list(p_and_b_dict.values())
        
    # Pass all tensors and options as a flat list of arguments
    return _Adjoint.apply(
        y0, t_vec, 
        functional_system_func, 
        param_keys, method, order, rtol, atol, 
        quad_method, quad_options,
        *param_values  # unpack the tensors here
    )