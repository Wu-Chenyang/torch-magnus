import torch
import unittest
import math
from torch_linode.butcher import DOPRI5, RK4, GL2, GL4, GL6, RADAU2, RADAU4, RADAU6

class TestButcherTableau(unittest.TestCase):
    def test_gl4_tableau(self):
        """Tests the correctness of the GL4 Butcher Tableau."""
        tableau = GL4
        
        # Check types
        self.assertIsInstance(tableau.c, torch.Tensor)
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertIsInstance(tableau.a, torch.Tensor)

        # Check shapes
        self.assertEqual(tableau.c.shape, (2,))
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertEqual(tableau.a.shape, (2, 2))

        # Check values
        sqrt3 = math.sqrt(3)
        expected_c = torch.tensor([0.5 - sqrt3 / 6, 0.5 + sqrt3 / 6], dtype=torch.float64)
        expected_b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        expected_a = torch.tensor([[1/4, 1/4 - sqrt3/6],
                                     [1/4 + sqrt3/6, 1/4]], dtype=torch.float64)
        
        self.assertTrue(torch.allclose(tableau.c, expected_c))
        self.assertTrue(torch.allclose(tableau.b, expected_b))
        self.assertTrue(torch.allclose(tableau.a, expected_a))
        self.assertEqual(tableau.order, 4)

    def test_gl2_tableau(self):
        """Tests the correctness of the GL2 Butcher Tableau."""
        tableau = GL2
        self.assertIsInstance(tableau.c, torch.Tensor)
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertIsInstance(tableau.a, torch.Tensor)
        self.assertEqual(tableau.c.shape, (2,))
        self.assertEqual(tableau.b.shape, (2,))
        self.assertEqual(tableau.a.shape, (2, 2))
        self.assertEqual(tableau.order, 4)

        sqrt3 = math.sqrt(3)
        expected_c = torch.tensor([1 / 2 - sqrt3 / 6, 1 / 2 + sqrt3 / 6], dtype=torch.float64)
        expected_b = torch.tensor([1 / 2, 1 / 2], dtype=torch.float64)
        expected_a = torch.tensor([\
            [1 / 4, 1 / 4 - sqrt3 / 6],\
            [1 / 4 + sqrt3 / 6, 1 / 4],\
        ], dtype=torch.float64)
        
        self.assertTrue(torch.allclose(tableau.c, expected_c))
        self.assertTrue(torch.allclose(tableau.b, expected_b))
        self.assertTrue(torch.allclose(tableau.a, expected_a))

    def test_gl6_tableau(self):
        """Tests the correctness of the GL6 Butcher Tableau."""
        tableau = GL6
        self.assertIsInstance(tableau.c, torch.Tensor)
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertIsInstance(tableau.a, torch.Tensor)
        self.assertEqual(tableau.c.shape, (3,))
        self.assertEqual(tableau.b.shape, (3,))
        self.assertEqual(tableau.a.shape, (3, 3))
        self.assertEqual(tableau.order, 6)

        sqrt15 = math.sqrt(15)
        expected_c = torch.tensor([1 / 2 - sqrt15 / 10, 1 / 2, 1 / 2 + sqrt15 / 10], dtype=torch.float64)
        expected_b = torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64)
        expected_a = torch.tensor([\
            [5 / 36, 2 / 9 - sqrt15 / 15, 5 / 36 - sqrt15 / 30],\
            [5 / 36 + sqrt15 / 24, 2 / 9, 5 / 36 - sqrt15 / 24],\
            [5 / 36 + sqrt15 / 30, 2 / 9 + sqrt15 / 15, 5 / 36],\
        ], dtype=torch.float64)

        self.assertTrue(torch.allclose(tableau.c, expected_c))
        self.assertTrue(torch.allclose(tableau.b, expected_b))
        self.assertTrue(torch.allclose(tableau.a, expected_a))

    def test_radau2_tableau(self):
        """Tests the correctness of the RADAU2 Butcher Tableau."""
        tableau = RADAU2
        self.assertIsInstance(tableau.c, torch.Tensor)
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertIsInstance(tableau.a, torch.Tensor)
        self.assertEqual(tableau.c.shape, (1,))
        self.assertEqual(tableau.b.shape, (1,))
        self.assertEqual(tableau.a.shape, (1, 1))
        self.assertEqual(tableau.order, 1)

        expected_c = torch.tensor([1], dtype=torch.float64)
        expected_b = torch.tensor([1], dtype=torch.float64)
        expected_a = torch.tensor([[1]], dtype=torch.float64)

        self.assertTrue(torch.allclose(tableau.c, expected_c))
        self.assertTrue(torch.allclose(tableau.b, expected_b))
        self.assertTrue(torch.allclose(tableau.a, expected_a))

    def test_radau4_tableau(self):
        """Tests the correctness of the RADAU4 Butcher Tableau."""
        tableau = RADAU4
        self.assertIsInstance(tableau.c, torch.Tensor)
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertIsInstance(tableau.a, torch.Tensor)
        self.assertEqual(tableau.c.shape, (2,))
        self.assertEqual(tableau.b.shape, (2,))
        self.assertEqual(tableau.a.shape, (2, 2))
        self.assertEqual(tableau.order, 3)

        expected_c = torch.tensor([1 / 3, 1], dtype=torch.float64)
        expected_b = torch.tensor([3 / 4, 1 / 4], dtype=torch.float64)
        expected_a = torch.tensor([\
            [5 / 12, -1 / 12],\
            [3 / 4, 1 / 4],\
        ], dtype=torch.float64)

        self.assertTrue(torch.allclose(tableau.c, expected_c))
        self.assertTrue(torch.allclose(tableau.b, expected_b))
        self.assertTrue(torch.allclose(tableau.a, expected_a))

    def test_radau6_tableau(self):
        """Tests the correctness of the RADAU6 Butcher Tableau."""
        tableau = RADAU6
        self.assertIsInstance(tableau.c, torch.Tensor)
        self.assertIsInstance(tableau.b, torch.Tensor)
        self.assertIsInstance(tableau.a, torch.Tensor)
        self.assertEqual(tableau.c.shape, (3,))
        self.assertEqual(tableau.b.shape, (3,))
        self.assertEqual(tableau.a.shape, (3, 3))
        self.assertEqual(tableau.order, 5)

        sqrt6 = math.sqrt(6)
        expected_c = torch.tensor([(4 - sqrt6) / 10, 1 / 2, (4 + sqrt6) / 10], dtype=torch.float64)
        expected_b = torch.tensor([1 / 9, (16 + sqrt6) / 36, (16 - sqrt6) / 36], dtype=torch.float64)
        expected_a = torch.tensor([\
            [(88 - 7 * sqrt6) / 360, (296 - 169 * sqrt6) / 1800, (-2 + 3 * sqrt6) / 225],\
            [(296 + 169 * sqrt6) / 1800, (88 + 7 * sqrt6) / 360, (-2 - 3 * sqrt6) / 225],\
            [1 / 9, (16 + sqrt6) / 36, (16 - sqrt6) / 36],\
        ], dtype=torch.float64)

        self.assertTrue(torch.allclose(tableau.c, expected_c))
        self.assertTrue(torch.allclose(tableau.b, expected_b))
        self.assertTrue(torch.allclose(tableau.a, expected_a))

if __name__ == '__main__':
    unittest.main()