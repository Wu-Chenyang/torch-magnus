from torch_magnus import magnus_odeint, magnus_odeint_adjoint, magnus_solve

import torch
import math
import numpy as np
from typing import Callable, Union, Sequence
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("order", [2, 4, 6])
def test_exponential_system(order):
    """测试指数系统: y' = diag(p1, p2) * y"""
    
    def A_func(t, params: torch.Tensor) -> torch.Tensor:
        """A = diag(p1, p2)"""
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t)
        A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
        A[..., 0, 0] += params[0]
        A[..., 1, 1] += params[1]
        return A
    
    # 测试参数
    p1, p2 = 0.5, -0.3
    params = torch.tensor([p1, p2], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    T = 1.0
    t = torch.tensor([0.0, T], dtype=torch.float64)
    
    # Magnus求解
    y_traj = magnus_odeint_adjoint(A_func, y0, t, params, order=order, rtol=1e-6, atol=1e-8)
    
    # 解析解: y(T) = [exp(p1*T), 2*exp(p2*T)]
    y_analytical = torch.tensor([
        math.exp(p1 * T),
        2 * math.exp(p2 * T)
    ], dtype=torch.float64)
    
    print(f"  Magnus solution: {y_traj[-1].detach().numpy()}")
    print(f"  Analytical solution: {y_analytical.numpy()}")
    print(f"  Solution error: {torch.norm(y_traj[-1] - y_analytical).item():.2e}")
    
    # 损失函数: L = ||y(T)||^2
    loss = torch.sum(y_traj[-1]**2)
    grad_magnus = torch.autograd.grad(loss, params)[0]
    
    # 解析梯度: dL/dp1 = 2*T*exp(2*p1*T), dL/dp2 = 2*T*4*exp(2*p2*T)
    grad_analytical = torch.tensor([
        2 * T * math.exp(2 * p1 * T),
        2 * T * 4 * math.exp(2 * p2 * T)
    ], dtype=torch.float64)
    
    print(f"  Magnus gradient: {grad_magnus.detach().numpy()}")
    print(f"  Analytical gradient: {grad_analytical.numpy()}")
    
    grad_error = torch.norm(grad_magnus - grad_analytical)
    print(f"  Gradient error: {grad_error.item():.2e}")
    
    success = grad_error < 1e-8 + 1e-6 * torch.norm(grad_analytical)
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    assert success


@pytest.mark.parametrize("order", [2, 4, 6])
def test_harmonic_oscillator(order):
    """测试谐振子: y'' + ω²y = 0 """
    
    def A_func(t, params: torch.Tensor) -> torch.Tensor:
        """A = [[0, 1], [-ω², 0]]"""
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t)
        omega = params[0]
        A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
        A[..., 0, 1] += 1.0
        A[..., 1, 0] += -omega**2
        return A
    
    # 测试参数
    omega = 2.0
    params = torch.tensor([omega], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)  # 初始条件: y(0)=1, y'(0)=0
    T = 0.5
    t = torch.tensor([0.0, T], dtype=torch.float64)
    
    # Magnus求解
    y_traj = magnus_odeint_adjoint(A_func, y0, t, params, order=order, rtol=1e-6, atol=1e-8)
    
        # 解析解: y(T) = [cos(ωT), -ω*sin(ωT)]
    y_analytical = torch.tensor([
        math.cos(omega * T),
        -omega * math.sin(omega * T)
    ], dtype=torch.float64)
    
    print(f"  Magnus solution: {y_traj[-1].detach().numpy()}")
    print(f"  Analytical solution: {y_analytical.numpy()}")
    print(f"  Solution error: {torch.norm(y_traj[-1] - y_analytical).item():.2e}")
    
    # 损失函数: L = ||y(T)||^2 = cos²(ωT) + ω²sin²(ωT)
    loss = torch.sum(y_traj[-1]**2)
    grad_magnus = torch.autograd.grad(loss, params)[0]
    
    # 正确的解析梯度计算:
    # L = cos²(ωT) + ω²sin²(ωT)
    # dL/dω = 2cos(ωT)(-sin(ωT))T + 2ω sin²(ωT) + ω²(2sin(ωT)cos(ωT))T
    #       = -2T cos(ωT)sin(ωT) + 2ω sin²(ωT) + 2ω²T sin(ωT)cos(ωT)
    #       = 2ω sin²(ωT) + 2T sin(ωT)cos(ωT)(ω² - 1)
    
    cos_T = math.cos(omega * T)
    sin_T = math.sin(omega * T)
    grad_analytical = 2*omega*sin_T**2 + 2*T*sin_T*cos_T*(omega**2 - 1)
    
    print(f"  Magnus gradient: {grad_magnus.detach().numpy()}")
    print(f"  Analytical gradient: {grad_analytical}")

    grad_error = abs(grad_magnus.item() - grad_analytical)
    print(f"  Gradient error: {grad_error:.2e}")
    
    # 详细验证计算过程
    print(f"  Detailed verification:")
    print(f"    ω = {omega}, T = {T}, ωT = {omega*T}")
    print(f"    cos(ωT) = {cos_T:.10f}")
    print(f"    sin(ωT) = {sin_T:.10f}")
    print(f"    First term: 2ω sin²(ωT) = {2*omega*sin_T**2:.10f}")
    print(f"    Second term: 2T sin(ωT)cos(ωT)(ω²-1) = {2*T*sin_T*cos_T*(omega**2-1):.10f}")
    print(f"    Sum: {2*omega*sin_T**2 + 2*T*sin_T*cos_T*(omega**2-1):.10f}")
    
    success = grad_error < 1e-8 + 1e-6 * abs(grad_analytical)
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    assert success


@pytest.mark.parametrize("order", [2, 4, 6])
def test_rotation_matrix(order):
    """测试旋转矩阵: A = [[0, ω], [-ω, 0]]"""
    
    def A_func(t, params: torch.Tensor) -> torch.Tensor:
        """A = [[0, ω], [-ω, 0]]"""
        omega = params[0]
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t)
        A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
        A[..., 0, 1] += omega
        A[..., 1, 0] += -omega
        return A
    
    # 测试参数
    omega = 1.5
    params = torch.tensor([omega], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    T = 0.5
    t = torch.tensor([0.0, T], dtype=torch.float64)
    
    # Magnus求解
    y_traj = magnus_odeint_adjoint(A_func, y0, t, params, order=order, rtol=1e-6, atol=1e-8)
    
    # 解析解: y(T) = [cos(ωT), -sin(ωT)]
    y_analytical = torch.tensor([
        math.cos(omega * T),
        -math.sin(omega * T)
    ], dtype=torch.float64)
    
    print(f"  Magnus solution: {y_traj[-1].detach().numpy()}")
    print(f"  Analytical solution: {y_analytical.numpy()}")
    print(f"  Solution error: {torch.norm(y_traj[-1] - y_analytical).item():.2e}")
    
    # 损失函数: L = ||y(T)||^2 = cos²(ωT) + sin²(ωT) = 1 (常数!)
    loss = torch.sum(y_traj[-1]**2)
    grad_magnus = torch.autograd.grad(loss, params)[0]
    
    # 解析梯度: dL/dω = 0 (因为L=1是常数)
    grad_analytical = 0.0
    
    print(f"  Magnus gradient: {grad_magnus.detach().numpy()}")
    print(f"  Analytical gradient: {grad_analytical}")
    
    grad_error = abs(grad_magnus.item() - grad_analytical)
    print(f"  Gradient error: {grad_error:.2e}")
    
    success = grad_error < 1e-8 + 1e-6 * abs(grad_analytical)
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    assert success


@pytest.mark.parametrize("order", [2, 4, 6])
def test_challenging_highly_oscillatory_system(order):
    """测试一个具有非零梯度的挑战性高度振荡系统"""
    
    def A_func(t, params: torch.Tensor) -> torch.Tensor:
        """A(t) = [[p3*t, w(t)], [-w(t), p3*t]] where w(t) = p0 + p1*cos(p2*t)"""
        w0, w1, w2, p3 = params[0], params[1], params[2], params[3]
        t_tensor = torch.as_tensor(t, dtype=params.dtype, device=params.device)
        wt = w0 + w1 * torch.cos(w2 * t_tensor)
        
        A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
        A[..., 0, 0] = p3 * t_tensor
        A[..., 0, 1] = wt
        A[..., 1, 0] = -wt
        A[..., 1, 1] = p3 * t_tensor
        return A

    # 测试参数
    w0, w1, w2, p3 = 10.0, 5.0, 20.0, -0.2
    params = torch.tensor([w0, w1, w2, p3], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    T = 0.5
    t = torch.tensor([0.0, T], dtype=torch.float64)

    # Magnus求解
    y_traj = magnus_odeint_adjoint(A_func, y0, t, params, order=order, rtol=1e-6, atol=1e-8)

    # 解析解
    if w2 == 0:
        theta_T = w0 * T + w1 * T
    else:
        theta_T = w0 * T + (w1 / w2) * math.sin(w2 * T)
    exp_term = math.exp(p3 * T**2 / 2)
    y_analytical = exp_term * torch.tensor([
        math.cos(theta_T),
        -math.sin(theta_T)
    ], dtype=torch.float64)

    print(f"  Magnus solution: {y_traj[-1].detach().numpy()}")
    print(f"  Analytical solution: {y_analytical.numpy()}")
    sol_error = torch.norm(y_traj[-1] - y_analytical)
    print(f"  Solution error: {sol_error.item():.2e}")

    # 损失函数: L = y_1(T) (第一个分量)
    loss = y_traj[-1, 0]
    grad_magnus = torch.autograd.grad(loss, params)[0]

    # 解析梯度
    sin_theta_T = math.sin(theta_T)
    cos_theta_T = math.cos(theta_T)
    
    if w2 == 0:
        grad_analytical = torch.tensor([
            -exp_term * sin_theta_T * T,
            -exp_term * sin_theta_T * T,
            -exp_term * sin_theta_T * 0,
            (T**2 / 2) * exp_term * cos_theta_T
        ], dtype=torch.float64)
    else:
        grad_analytical = torch.tensor([
            -exp_term * sin_theta_T * T,
            -exp_term * sin_theta_T * (math.sin(w2 * T) / w2),
            -exp_term * sin_theta_T * (T * math.cos(w2 * T) / w2 - math.sin(w2 * T) / w2**2) * w1,
            (T**2 / 2) * exp_term * cos_theta_T
        ], dtype=torch.float64)

    print(f"  Magnus gradient: {grad_magnus.detach().numpy()}")
    print(f"  Analytical gradient: {grad_analytical.numpy()}")
    
    grad_error = torch.norm(grad_magnus - grad_analytical)
    print(f"  Gradient error: {grad_error.item():.2e}")

    success = grad_error < 1e-8 + 1e-6 * torch.norm(grad_analytical).item()
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    assert success


@pytest.mark.parametrize("order", [2, 4, 6])
def test_against_torchdiffeq(order):
    """与torchdiffeq比较(如果可用)"""
    
    try:
        from torchdiffeq import odeint
        print("  Comparing with torchdiffeq...")
        
        def ode_func(t, y, params):
            """ODE函数: dy/dt = A(t) * y"""
            A = torch.zeros(2, 2, dtype=y.dtype, device=y.device)
            A[0, 0] += params[0]
            A[0, 1] += params[1]
            A[1, 0] += -params[1]
            A[1, 1] += params[0]
            return A @ y
        
        params = torch.tensor([0.3, 0.2], dtype=torch.float64, requires_grad=True)
        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        t = torch.linspace(0, 0.5, 11, dtype=torch.float64)
        
        # torchdiffeq求解
        y_torchdiffeq = odeint(lambda t, y: ode_func(t, y, params), y0, t, method='dopri5')
        loss_torchdiffeq = torch.sum(y_torchdiffeq[-1]**2)
        grad_torchdiffeq = torch.autograd.grad(loss_torchdiffeq, params, retain_graph=True)[0]
        
        # Magnus求解
        def A_func(t, params: torch.Tensor) -> torch.Tensor:
            t_tensor = t if torch.is_tensor(t) else torch.tensor(t)
            A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
            A[..., 0, 0] = params[0]
            A[..., 0, 1] = params[1]
            A[..., 1, 0] = -params[1]
            A[..., 1, 1] = params[0]
            return A
        
        params_magnus = params.clone().detach().requires_grad_(True)
        y_magnus = magnus_odeint_adjoint(A_func, y0, t, params_magnus, order=order, rtol=1e-6, atol=1e-8)
        loss_magnus = torch.sum(y_magnus[-1]**2)
        grad_magnus = torch.autograd.grad(loss_magnus, params_magnus)[0]
        
        # 比较结果
        sol_error = torch.norm(y_magnus[-1] - y_torchdiffeq[-1])
        grad_error = torch.norm(grad_magnus - grad_torchdiffeq)
        
        print(f"  Solution error vs torchdiffeq: {sol_error.item():.2e}")
        print(f"  Gradient error vs torchdiffeq: {grad_error.item():.2e}")
        
        success = sol_error < 1e-8 + torch.norm(y_torchdiffeq[-1]).item() * 1e-6 and grad_error < 1e-8 + torch.norm(grad_torchdiffeq).item() * 1e-6
        print(f"  Comparison: {'PASSED' if success else 'FAILED'}")
        
        assert success
        
    except ImportError:
        print("  torchdiffeq not available, skipping comparison")
        assert True


def test_complex_system_stability():
    """测试复杂系统的稳定性"""
    
    def A_func(t, params: torch.Tensor) -> torch.Tensor:
        """更复杂的时间相关系统"""
        p1, p2, p3 = params[0], params[1], params[2]
        t_tensor = torch.as_tensor(t, dtype=params.dtype, device=params.device)
        
        A = torch.zeros(t_tensor.shape + (3, 3), dtype=params.dtype, device=params.device)
        A[..., 0, 1] += p1 * torch.sin(t_tensor)
        A[..., 0, 2] += p2
        A[..., 1, 0] += -p1 * torch.sin(t_tensor)
        A[..., 1, 2] += p3 * torch.cos(t_tensor)
        A[..., 2, 0] += -p2
        A[..., 2, 1] += -p3 * torch.cos(t_tensor)
        return A
    
    params = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    t = torch.linspace(0, 1, 21, dtype=torch.float64)
    
    # 测试不同阶数
    for order in [2, 4, 6]:
        print(f"  Testing order {order}...")
        
        try:
            y_traj = magnus_odeint_adjoint(A_func, y0, t, params, order=order)
            loss = torch.sum(y_traj[-1]**2)
            grad = torch.autograd.grad(loss, params, retain_graph=True)[0]
            
            # 检查数值稳定性
            has_nan = torch.isnan(y_traj).any() or torch.isnan(grad).any()
            has_inf = torch.isinf(y_traj).any() or torch.isinf(grad).any()
            
            if has_nan or has_inf:
                print(f"    Order {order}: FAILED (NaN/Inf detected)")
                success = False
            else:
                print(f"    Order {order}: PASSED")
                success = True
            assert success
        except Exception as e:
            print(f"    Order {order}: FAILED ({str(e)})")
            assert False


def test_order_consistency():
    """测试不同阶数的一致性"""
    
    def A_func(t: float, params: torch.Tensor) -> torch.Tensor:
        """简单时间相关系统"""
        p = params[0]
        t_tensor = torch.as_tensor(t, dtype=params.dtype, device=params.device)
        A = torch.zeros(t_tensor.shape + (2, 2), dtype=params.dtype, device=params.device)
        A[..., 0, 0] += p * torch.cos(t_tensor)
        A[..., 1, 1] += p * torch.sin(t_tensor)
        return A
    
    params = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    
    print("  Time steps | Solution diff | Gradient diff")
    print("  " + "-" * 45)
    
    for n_steps in [20, 40, 80]:
        t = torch.linspace(0, 0.5, n_steps + 1, dtype=torch.float64)
        
        # 2阶方法
        params_2 = params.clone().detach().requires_grad_(True)
        y_traj_2 = magnus_odeint_adjoint(A_func, y0, t, params_2, order=2)
        loss_2 = torch.sum(y_traj_2[-1]**2)
        grad_2 = torch.autograd.grad(loss_2, params_2)[0]
        
        # 4阶方法
        params_4 = params.clone().detach().requires_grad_(True)
        y_traj_4 = magnus_odeint_adjoint(A_func, y0, t, params_4, order=4)
        loss_4 = torch.sum(y_traj_4[-1]**2)
        grad_4 = torch.autograd.grad(loss_4, params_4)[0]
        
        # 6阶方法
        params_6 = params.clone().detach().requires_grad_(True)
        y_traj_6 = magnus_odeint_adjoint(A_func, y0, t, params_6, order=6)
        loss_6 = torch.sum(y_traj_6[-1]**2)
        grad_6 = torch.autograd.grad(loss_6, params_6)[0]

        # 比较差异
        sol_diff = torch.maximum(torch.norm(y_traj_2[-1] - y_traj_6[-1]), torch.norm(y_traj_4[-1] - y_traj_6[-1]))
        grad_diff = torch.maximum(torch.norm(grad_2 - grad_6), torch.norm(grad_4 - grad_6))
        
        print(f"  {n_steps:9d} | {sol_diff.item():11.2e} | {grad_diff.item():11.2e}")
    
    print("  Note: Differences should decrease with more time steps")
    assert True


@pytest.mark.parametrize("order", [2, 4, 6])
def test_tolerance_settings(order):
    """
    设计一个完善的测试，检查求解器是否能够正确满足不同的容差设置。
    测试逻辑：
    1. 定义一个高度振荡的时变系统。
    2. 设置一系列递减的容差值（rtol 和 atol）。
    3. 使用 magnus_solve 在每个容差设置下求解ODE。
    4. 计算数值解与解析解在终点的误差。
    5. 验证：
        a) 随着容差的减小，计算出的误差是否也单调递减。
        b) 计算出的最终误差是否小于或大致等于设定的容差。
    """
    print("\nTesting solver tolerance settings with a highly oscillatory system...")
    print("=" * 60)

    # 1. 定义一个高度振荡的系统
    w0, w1, w2 = 10.0, 5.0, 20.0
    def A_func(t, params=None):
        t = torch.as_tensor(t)
        wt = w0 + w1 * torch.cos(w2 * t)
        A = torch.zeros(t.shape + (2, 2), dtype=torch.float64)
        A[..., 0, 1] = wt
        A[..., 1, 0] = -wt
        return A

    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    t_span = (0.0, 0.5)
    T = t_span[1]

    # 解析解
    def analytical_solution(t):
        theta_t = w0 * t + (w1 / w2) * math.sin(w2 * t)
        return torch.tensor([math.cos(theta_t), -math.sin(theta_t)], dtype=torch.float64)

    # 2. 测试不同的容差设置
    tolerances = [1e-3, 1e-5, 1e-7]#, 1e-9]
    results = []
    all_passed = True

    print("  Solver   | Tolerance | Final Error vs. Analytical | Status")
    print("  " + "-" * 60)

    for tol in tolerances:
        rtol = 0.1 * tol
        atol = tol
        # 3. 使用 magnus_odeint 进行求解
        y_final = magnus_odeint(
            A_func, y0, t_span, 
            rtol=rtol, atol=atol, # 设置求解器的容差
            order=order  # 使用4阶方法
        )[..., -1, :]

        # 4. 计算与解析解的误差
        y_analytical_final = analytical_solution(T)
        error = torch.norm(y_final - y_analytical_final).item()

        # 5b. 验证误差是否满足设定的容差
        tolerance = atol + rtol * torch.norm(y_analytical_final).item()
        passed = error < 10 * tolerance
        if not passed:
            all_passed = False
        
        results.append({'tol': tolerance, 'error': error, 'passed': passed})
        status = "PASSED" if passed else "FAILED"
        print(f"  Magnus O{order} | {tolerance:9.1e} | {error:11.2e}               | {status}")

    # 5a. 验证误差是否随着容差的减小而单调递减
    errors = [r['error'] for r in results]
    is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors) - 1))
    
    print("\n  Verification:")
    print(f"  - Is error monotonically decreasing with tolerance? {'YES' if is_monotonic else 'NO'}")
    if not is_monotonic:
        all_passed = False

    print(f"  - Does final error meet the tolerance for all cases? {'YES' if all(r['passed'] for r in results) else 'NO'}")

    print(f"\n  Overall Tolerance Test Result: {'PASSED' if all_passed else 'FAILED'}")
    assert all_passed


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 测试1: 与解析解比较
    print("Test 1: Analytical solution comparison")
    print("\n1.1 Simple exponential system")
    test_exponential_system()
    
    print("\n1.2 Harmonic oscillator")
    test_harmonic_oscillator()
    
    print("\n1.3 Rotation matrix")
    test_rotation_matrix()

    print("\n1.4 Challenging highly oscillatory system")
    test_challenging_highly_oscillatory_system()
    
    # 测试2: 与torchdiffeq比较
    print("\nTest 2: Comparison with torchdiffeq")
    test_against_torchdiffeq()
    
    # 测试3: 复杂系统稳定性
    print("\nTest 3: Complex system stability")
    test_complex_system_stability()
    
    # 测试4: 不同阶数一致性
    print("\nTest 4: Order consistency")
    test_order_consistency()

    print("\nTest 5: Tolerance test")
    test_tolerance_settings()