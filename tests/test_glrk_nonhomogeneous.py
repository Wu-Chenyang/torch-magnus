import torch
import pytest
from torch_linode.solvers import GLRK2nd, GLRK4th, GLRK6th

class TestGLRKNonhomogeneous:
    def test_glrk2nd_nonhomogeneous_integration(self):
        # Define a simple non-homogeneous system: dy/dt = A*y + g
        # A(t) = [[0, 0], [0, 0]]
        # g(t) = [1, 1]
        def mock_A_func(t):
            A = torch.zeros(2, 2, dtype=torch.float64)
            g = torch.ones(2, dtype=torch.float64)
            return A, g

        glrk2nd = GLRK2nd()
        y0 = torch.zeros(2, dtype=torch.float64) # Initial state [0, 0]
        t0 = 0.0
        h = 0.1

        # Expected solution: y_next = y0 + h * g(t_mid)
        # Since g(t) is constant [1, 1], g(t_mid) = [1, 1]
        # y_next = [0, 0] + 0.1 * [1, 1] = [0.1, 0.1]
        expected_y_next = torch.tensor([0.1, 0.1], dtype=torch.float64)

        y_next, _ = glrk2nd(mock_A_func, t0, h, y0)
        
        torch.testing.assert_close(y_next, expected_y_next, rtol=1e-5, atol=1e-5)

    def test_glrk2nd_homogeneous_still_works(self):
        # Ensure homogeneous case still works after changes
        def mock_A_func(t):
            return torch.eye(2, dtype=torch.float64) * t

        glrk2nd = GLRK2nd()
        y0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t0 = 0.0
        h = 0.1

        # Expected solution for dy/dt = t*I*y, y0=[1,2], h=0.1
        # Using GLRK2nd (midpoint rule for A)
        # A(t0 + h/2) = A(0.05) = 0.05 * I
        # k1 = (I - h*a*A1)^-1 * A1*y0
        # k1 = (I - 0.1*0.5*0.05*I)^-1 * 0.05*I*y0
        # k1 = (I - 0.0025*I)^-1 * 0.05*I*y0
        # k1 = (0.9975*I)^-1 * 0.05*I*y0
        # k1 = (1/0.9975) * 0.05 * y0 = 0.050125313 * y0
        # y_next = y0 + h * k1 = y0 + 0.1 * 0.050125313 * y0 = y0 * (1 + 0.0050125313)
        # y_next = y0 * 1.0050125313
        expected_y_next = y0 * 1.0050125313

        y_next, _ = glrk2nd(mock_A_func, t0, h, y0)
        torch.testing.assert_close(y_next, expected_y_next, rtol=1e-5, atol=1e-5)

    def test_glrk4th_nonhomogeneous_integration(self):
        # Define a simple non-homogeneous system: dy/dt = A*y + g
        # A(t) = [[0, 0], [0, 0]]
        # g(t) = [1, 1]
        def mock_A_func(t):
            if t.ndim == 0: # Scalar t
                A = torch.zeros(2, 2, dtype=torch.float64)
                g = torch.ones(2, dtype=torch.float64)
            else: # Batched t
                batch_size = t.shape[0]
                A = torch.zeros(batch_size, 2, 2, dtype=torch.float64)
                g = torch.ones(batch_size, 2, dtype=torch.float64)
            return A, g

        glrk4th = GLRK4th()
        y0 = torch.zeros(2, dtype=torch.float64) # Initial state [0, 0]
        t0 = 0.0
        h = 0.1

        # Expected solution: y_next = y0 + h * g(t_mid)
        # Since g(t) is constant [1, 1], g(t_mid) = [1, 1]
        # y_next = [0, 0] + 0.1 * [1, 1] = [0.1, 0.1]
        expected_y_next = torch.tensor([0.1, 0.1], dtype=torch.float64)

        y_next, _ = glrk4th(mock_A_func, t0, h, y0)
        
        torch.testing.assert_close(y_next, expected_y_next, rtol=1e-5, atol=1e-5)

    def test_glrk6th_nonhomogeneous_integration(self):
        # Define a simple non-homogeneous system: dy/dt = A*y + g
        # A(t) = [[0, 0], [0, 0]]
        # g(t) = [1, 1]
        def mock_A_func(t):
            if t.ndim == 0: # Scalar t
                A = torch.zeros(2, 2, dtype=torch.float64)
                g = torch.ones(2, dtype=torch.float64)
            else: # Batched t
                batch_size = t.shape[0]
                A = torch.zeros(batch_size, 2, 2, dtype=torch.float64)
                g = torch.ones(batch_size, 2, dtype=torch.float64)
            return A, g

        glrk6th = GLRK6th()
        y0 = torch.zeros(2, dtype=torch.float64) # Initial state [0, 0]
        t0 = 0.0
        h = 0.1

        # Expected solution: y_next = y0 + h * g(t_mid)
        # Since g(t) is constant [1, 1], g(t_mid) = [1, 1]
        # y_next = [0, 0] + 0.1 * [1, 1] = [0.1, 0.1]
        expected_y_next = torch.tensor([0.1, 0.1], dtype=torch.float64)

        y_next, _ = glrk6th(mock_A_func, t0, h, y0)
        
        torch.testing.assert_close(y_next, expected_y_next, rtol=1e-5, atol=1e-5)
