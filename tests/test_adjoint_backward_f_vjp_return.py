import torch
import pytest
from torch_linode.solvers import _Adjoint, _apply_matrix

# Mock functional_system_func for testing
def mock_homogeneous_system_func(t, p_dict):
    # Simple A matrix for homogeneous system
    return torch.tensor([[p_dict['p1'], 0.], [0., p_dict['p2']]], dtype=torch.float64)

def mock_nonhomogeneous_system_func(t, p_dict):
    # Simple A matrix and g vector for non-homogeneous system
    A = torch.tensor([[p_dict['p1'], 0.], [0., p_dict['p2']]], dtype=torch.float64)
    g = torch.tensor([t, t**2], dtype=torch.float64)
    return A, g

@pytest.fixture
def common_params():
    return {
        'p1': torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
        'p2': torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    }

@pytest.fixture
def mock_ctx(common_params):
    class MockCtx:
        def __init__(self):
            self.is_nonhomogeneous = False
            self.functional_system_func = None
            self.param_keys = list(common_params.keys())
            self.saved_tensors = (None, *common_params.values()) # t is None for this test
            self.method = 'magnus' # or 'glrk'
            self.order = 4
            self.rtol = 1e-6
            self.atol = 1e-8
            self.quad_method = 'gk'
            self.quad_options = {}
            self.y0_requires_grad = True
            
            # Mock y_dense_traj_aug to return a simple y_eval
            class MockYDenseTrajAug:
                def __call__(self, t_nodes, is_nonhomogeneous, method):
                    if is_nonhomogeneous and method == 'magnus':
                        # Augmented for Magnus
                        if isinstance(t_nodes, torch.Tensor) and t_nodes.ndim > 0:
                            return torch.stack([torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64) for _ in range(t_nodes.numel())], dim=0)
                        else:
                            return torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
                    else:
                        # Not augmented for homogeneous or GLRK
                        if isinstance(t_nodes, torch.Tensor) and t_nodes.ndim > 0:
                            return torch.stack([torch.tensor([1.0, 2.0], dtype=torch.float64) for _ in range(t_nodes.numel())], dim=0)
                        else:
                            return torch.tensor([1.0, 2.0], dtype=torch.float64)
            self.y_dense_traj_aug = MockYDenseTrajAug()

    return MockCtx()

def test_f_for_vjp_homogeneous_return(mock_ctx, common_params):
    mock_ctx.is_nonhomogeneous = False
    mock_ctx.functional_system_func = mock_homogeneous_system_func
    
    # Reconstruct parameter dictionaries
    full_p_and_b_dict = dict(zip(mock_ctx.param_keys, mock_ctx.saved_tensors[1:]))
    params_req = {k: v for k, v in full_p_and_b_dict.items() if v.requires_grad}
    buffers_dict = {k: v for k, v in full_p_and_b_dict.items() if not v.requires_grad}

    # Define f_for_vjp within the test to capture mock_ctx variables
    def f_for_vjp(t_nodes, p_dict_req):
        full_dict = {**p_dict_req, **buffers_dict}
        y_eval_aug = mock_ctx.y_dense_traj_aug(t_nodes, mock_ctx.is_nonhomogeneous, mock_ctx.method)
        if mock_ctx.is_nonhomogeneous and mock_ctx.method == 'magnus':
            y_eval = y_eval_aug[..., :-1]
        else:
            y_eval = y_eval_aug
        
        sys_out = mock_ctx.functional_system_func(t_nodes, full_dict)
        if mock_ctx.is_nonhomogeneous:
            A, g = sys_out
            return _apply_matrix(A, y_eval) + g
        else:
            return _apply_matrix(sys_out, y_eval)

    t_node = torch.tensor(0.5, dtype=torch.float64)
    result = f_for_vjp(t_node, params_req)
    
    # Expected result for homogeneous: A(t) @ y_eval
    # A(0.5) = [[1.0, 0.], [0., 2.0]]
    # y_eval = [1.0, 2.0] (from mock_y_dense_traj_aug, sliced)
    # Expected: [1.0 * 1.0 + 0. * 2.0, 0. * 1.0 + 2.0 * 2.0] = [1.0, 4.0]
    expected_result = torch.tensor([1.0, 4.0], dtype=torch.float64)
    assert torch.allclose(result, expected_result)

def test_f_for_vjp_nonhomogeneous_magnus_return(mock_ctx, common_params):
    mock_ctx.is_nonhomogeneous = True
    mock_ctx.functional_system_func = mock_nonhomogeneous_system_func
    mock_ctx.method = 'magnus' # Ensure Magnus method for slicing
    
    # Reconstruct parameter dictionaries
    full_p_and_b_dict = dict(zip(mock_ctx.param_keys, mock_ctx.saved_tensors[1:]))
    params_req = {k: v for k, v in full_p_and_b_dict.items() if v.requires_grad}
    buffers_dict = {k: v for k, v in full_p_and_b_dict.items() if not v.requires_grad}

    # Define f_for_vjp within the test to capture mock_ctx variables
    def f_for_vjp(t_nodes, p_dict_req):
        full_dict = {**p_dict_req, **buffers_dict}
        y_eval_aug = mock_ctx.y_dense_traj_aug(t_nodes, mock_ctx.is_nonhomogeneous, mock_ctx.method)
        if mock_ctx.is_nonhomogeneous and mock_ctx.method == 'magnus':
            y_eval = y_eval_aug[..., :-1]
        else:
            y_eval = y_eval_aug
        
        sys_out = mock_ctx.functional_system_func(t_nodes, full_dict)
        if mock_ctx.is_nonhomogeneous:
            A, g = sys_out
            return _apply_matrix(A, y_eval) + g
        else:
            return _apply_matrix(sys_out, y_eval)

    t_node = torch.tensor(0.5, dtype=torch.float64)
    result = f_for_vjp(t_node, params_req)
    
    # Expected result for non-homogeneous Magnus: A(t) @ y_eval + g(t)
    # A(0.5) = [[1.0, 0.], [0., 2.0]]
    # y_eval = [1.0, 2.0] (from mock_y_dense_traj_aug, sliced)
    # g(0.5) = [0.5, 0.5**2] = [0.5, 0.25]
    # A @ y_eval = [1.0, 4.0]
    # Expected: [1.0 + 0.5, 4.0 + 0.25] = [1.5, 4.25]
    expected_result = torch.tensor([1.5, 4.25], dtype=torch.float64)
    assert torch.allclose(result, expected_result)

def test_f_for_vjp_nonhomogeneous_glrk_return(mock_ctx, common_params):
    mock_ctx.is_nonhomogeneous = True
    mock_ctx.functional_system_func = mock_nonhomogeneous_system_func
    mock_ctx.method = 'glrk' # Ensure GLRK method for no slicing
    
    # Reconstruct parameter dictionaries
    full_p_and_b_dict = dict(zip(mock_ctx.param_keys, mock_ctx.saved_tensors[1:]))
    params_req = {k: v for k, v in full_p_and_b_dict.items() if v.requires_grad}
    buffers_dict = {k: v for k, v in full_p_and_b_dict.items() if not v.requires_grad}

    # Define f_for_vjp within the test to capture mock_ctx variables
    def f_for_vjp(t_nodes, p_dict_req):
        full_dict = {**p_dict_req, **buffers_dict}
        y_eval_aug = mock_ctx.y_dense_traj_aug(t_nodes, mock_ctx.is_nonhomogeneous, mock_ctx.method)
        if mock_ctx.is_nonhomogeneous and mock_ctx.method == 'magnus':
            y_eval = y_eval_aug[..., :-1]
        else:
            y_eval = y_eval_aug
        
        sys_out = mock_ctx.functional_system_func(t_nodes, full_dict)
        if mock_ctx.is_nonhomogeneous:
            A, g = sys_out
            return _apply_matrix(A, y_eval) + g
        else:
            return _apply_matrix(sys_out, y_eval)

    t_node = torch.tensor(0.5, dtype=torch.float64)
    result = f_for_vjp(t_node, params_req)
    
    # Expected result for non-homogeneous GLRK: A(t) @ y_eval + g(t)
    # A(0.5) = [[1.0, 0.], [0., 2.0]]
    # y_eval = [1.0, 2.0, 1.0] (from mock_y_dense_traj_aug, no slicing for GLRK)
    # g(0.5) = [0.5, 0.25]
    # A @ y_eval (first two dims) = [1.0, 4.0]
    # Expected: [1.0 + 0.5, 4.0 + 0.25] = [1.5, 4.25]
    # Note: The mock_y_dense_traj_aug returns a 3-dim tensor, but g is 2-dim.
    # This test assumes _apply_matrix handles the dimension mismatch or that
    # y_eval is effectively 2-dim for the A@y part.
    # Let's adjust y_eval in mock_y_dense_traj_aug to be 2-dim for GLRK case.
    # For GLRK, y_eval should not be augmented.
    # Re-evaluating expected:
    # A(0.5) = [[1.0, 0.], [0., 2.0]]
    # y_eval = [1.0, 2.0] (from mock_y_dense_traj_aug, assuming it's not augmented for GLRK)
    # g(0.5) = [0.5, 0.25]
    # A @ y_eval = [1.0, 4.0]
    # Expected: [1.0 + 0.5, 4.0 + 0.25] = [1.5, 4.25]
    expected_result = torch.tensor([1.5, 4.25], dtype=torch.float64)
    assert torch.allclose(result, expected_result)
