import torch
import pytest
from torch_linode.solvers import _Adjoint

class MockSystemFunc(torch.nn.Module):
    def forward(self, t, p_dict):
        # Simulate a non-homogeneous system returning (A, g)
        A = torch.eye(2, dtype=t.dtype, device=t.device).unsqueeze(0) # Batch dim for A
        g = torch.tensor([t, t], dtype=t.dtype, device=t.device).unsqueeze(0) # Batch dim for g
        return A, g

class MockDenseOutput:
    def __init__(self, y_traj_aug):
        self.y_traj_aug = y_traj_aug

    def __call__(self, t_nodes):
        # Simulate dense output for specific t_nodes
        # For simplicity, we'll just return a pre-defined trajectory
        # In a real scenario, this would interpolate based on t_nodes
        return self.y_traj_aug

@pytest.fixture
def setup_backward_context():
    # Mock context object for _Adjoint.backward
    ctx = type('Context', (object,), {})

    # Simulate a non-homogeneous Magnus case
    ctx.is_nonhomogeneous = True
    ctx.method = 'magnus'

    # Simulate y_dense_traj_aug with an augmented dimension
    # Original y would be (batch, dim), augmented is (batch, dim + 1)
    # Example: y_original = [[1.0, 2.0]], y_augmented = [[1.0, 2.0, 1.0]]
    y_traj_aug_data = torch.tensor([[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]], dtype=torch.float32)
    ctx.y_dense_traj_aug = MockDenseOutput(y_traj_aug_data)

    # Mock other necessary context attributes
    ctx.functional_system_func = MockSystemFunc()
    ctx.param_keys = []
    ctx.saved_tensors = (torch.tensor([0.0, 1.0]),) # Mock t
    ctx.rtol = 1e-6
    ctx.atol = 1e-8
    ctx.quad_method = 'gk'
    ctx.quad_options = {}
    ctx.y0_requires_grad = False

    return ctx

def test_f_for_vjp_y_eval_slicing(setup_backward_context):
    ctx = setup_backward_context

    # Manually call the f_for_vjp function from _Adjoint.backward
    # We need to simulate the environment where f_for_vjp is defined
    # This involves extracting the relevant part of the backward method

    t_nodes = torch.tensor([0.5], dtype=torch.float32) # Mock time node
    params_req = {}
    buffers_dict = {}
    full_dict = {**params_req, **buffers_dict}

    # The f_for_vjp function as it appears in _Adjoint.backward
    def f_for_vjp(t_nodes, p_dict_req):
        full_dict_inner = {**p_dict_req, **buffers_dict}
        y_eval_aug = ctx.y_dense_traj_aug(t_nodes)
        if ctx.is_nonhomogeneous and ctx.method == 'magnus':
            y_eval = y_eval_aug[..., :-1]
        else:
            y_eval = y_eval_aug
        
        sys_out = ctx.functional_system_func(t_nodes, full_dict_inner)
        if ctx.is_nonhomogeneous:
            A, g = sys_out
            return A, g, y_eval # Return y_eval for assertion
        else:
            return sys_out, y_eval # Return y_eval for assertion

    # Call f_for_vjp and get the returned y_eval
    _, _, y_eval_result = f_for_vjp(t_nodes, params_req)

    # Assert that y_eval_result has the augmented dimension sliced off
    expected_y_eval = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    assert torch.allclose(y_eval_result, expected_y_eval), f"Expected {expected_y_eval}, got {y_eval_result}"
    assert y_eval_result.shape[-1] == 2 # Original dimension was 2, augmented was 3


def test_f_for_vjp_y_eval_no_slicing_homogeneous(setup_backward_context):
    ctx = setup_backward_context
    ctx.is_nonhomogeneous = False # Simulate homogeneous case
    ctx.method = 'magnus'

    # Simulate y_dense_traj_aug without an augmented dimension
    y_traj_aug_data = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    ctx.y_dense_traj_aug = MockDenseOutput(y_traj_aug_data)

    t_nodes = torch.tensor([0.5], dtype=torch.float32)
    params_req = {}
    buffers_dict = {}
    full_dict = {**params_req, **buffers_dict}

    def f_for_vjp(t_nodes, p_dict_req):
        full_dict_inner = {**p_dict_req, **buffers_dict}
        y_eval_aug = ctx.y_dense_traj_aug(t_nodes)
        if ctx.is_nonhomogeneous and ctx.method == 'magnus':
            y_eval = y_eval_aug[..., :-1]
        else:
            y_eval = y_eval_aug
        
        sys_out = ctx.functional_system_func(t_nodes, full_dict_inner)
        if ctx.is_nonhomogeneous:
            A, g = sys_out
            return A, g, y_eval
        else:
            return sys_out, y_eval

    # Call f_for_vjp and get the returned y_eval
    _, y_eval_result = f_for_vjp(t_nodes, params_req)

    # Assert that y_eval_result is the same as y_traj_aug_data (no slicing)
    expected_y_eval = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    assert torch.allclose(y_eval_result, expected_y_eval), f"Expected {expected_y_eval}, got {y_eval_result}"
    assert y_eval_result.shape[-1] == 2

def test_f_for_vjp_y_eval_no_slicing_glrk_nonhomogeneous(setup_backward_context):
    ctx = setup_backward_context
    ctx.is_nonhomogeneous = True # Simulate non-homogeneous case
    ctx.method = 'glrk' # Simulate GLRK method

    # Simulate y_dense_traj_aug without an augmented dimension (GLRK doesn't augment)
    y_traj_aug_data = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    ctx.y_dense_traj_aug = MockDenseOutput(y_traj_aug_data)

    t_nodes = torch.tensor([0.5], dtype=torch.float32)
    params_req = {}
    buffers_dict = {}
    full_dict = {**params_req, **buffers_dict}

    def f_for_vjp(t_nodes, p_dict_req):
        full_dict_inner = {**p_dict_req, **buffers_dict}
        y_eval_aug = ctx.y_dense_traj_aug(t_nodes)
        if ctx.is_nonhomogeneous and ctx.method == 'magnus':
            y_eval = y_eval_aug[..., :-1]
        else:
            y_eval = y_eval_aug
        
        sys_out = ctx.functional_system_func(t_nodes, full_dict_inner)
        if ctx.is_nonhomogeneous:
            A, g = sys_out
            return A, g, y_eval
        else:
            return sys_out, y_eval

    # Call f_for_vjp and get the returned y_eval
    _, _, y_eval_result = f_for_vjp(t_nodes, params_req)

    # Assert that y_eval_result is the same as y_traj_aug_data (no slicing)
    expected_y_eval = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    assert torch.allclose(y_eval_result, expected_y_eval), f"Expected {expected_y_eval}, got {y_eval_result}"
    assert y_eval_result.shape[-1] == 2
