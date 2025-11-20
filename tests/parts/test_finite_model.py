"""
Core unit tests for FiniteModel class.

Tests the basic functionality of FiniteModel including:
- Initialization with various configurations
- Forward pass behavior and shapes
- Metric properties
- Convexity and concavity guarantees

These tests are fast (<1s each) and test individual model components.
"""

import unittest
import torch
from models import FiniteModel
from tests import TimedTestCase


class TestFiniteModelInitialization(TimedTestCase):
    """Test suite for FiniteModel initialization."""
    
    def test_convex_initialization(self):
        def kernel(x, y):
            return ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=10,
            num_dims=3,
            kernel=kernel,
            mode="convex",
            is_y_parameter=True,
        )
        
        self.assertEqual(model.mode, "convex")
        self.assertEqual(model.num_candidates, 10)
        self.assertEqual(model.num_dims, 3)
        self.assertTrue(model.is_y_parameter)
        
        Y = model.full_Y()
        self.assertEqual(Y.shape, (1, 10, 3))
    
    def test_concave_initialization(self):
        def kernel(x, y):
            return ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="concave",
        )
        
        self.assertEqual(model.mode, "concave")
        self.assertEqual(model.num_candidates, 5)
    
    def test_fixed_y_initialization(self):
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=8,
            num_dims=2,
            kernel=kernel,
            mode="convex",
            is_y_parameter=False,
        )
        
        self.assertFalse(model.is_y_parameter)
        
        param_names = [name for name, _ in model.named_parameters()]
        self.assertNotIn("Y_rest", param_names)
        
        buffer_names = [name for name, _ in model.named_buffers()]
        self.assertIn("Y_rest", buffer_names)
    
    def test_bounded_initialization(self):
        def kernel(x, y):
            return (x * y).sum(dim=-1)
        
        y_min, y_max = -2.0, 2.0
        model = FiniteModel(
            num_candidates=10,
            num_dims=3,
            kernel=kernel,
            mode="convex",
            y_min=y_min,
            y_max=y_max,
        )
        
        Y = model.full_Y()
        self.assertTrue(Y.min() >= y_min - 0.1)
        self.assertTrue(Y.max() <= y_max + 0.1)
    
    def test_default_candidate(self):
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="convex",
            is_there_default=True,
        )
        
        self.assertTrue(model.is_there_default)
        
        Y = model.full_Y()
        self.assertEqual(Y.shape[1], 6)
        
        torch.testing.assert_close(Y[:, 0, :], torch.zeros(1, 2))
    
    def test_invalid_mode_raises_error(self):
        def kernel(x, y):
            return (x * y).sum(dim=-1)
        
        with self.assertRaises(AssertionError):
            FiniteModel(
                num_candidates=5,
                num_dims=2,
                kernel=kernel,
                mode="invalid",
            )


class TestFiniteModelForward(TimedTestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.quad = lambda x, y: -((x - y)**2).sum(dim=-1)
        self.lin = lambda x, y: (x * y).sum(dim=-1)
    
    def test_convex_forward_shape(self):
        model = FiniteModel(5, 3, self.quad, mode="convex")
        X = torch.randn(10, 3)
        choice, fx = model.forward(X, "soft")
        self.assertEqual(choice.shape, (10, 3))
        self.assertEqual(fx.shape, (10,))
    
    def test_concave_forward_shape(self):
        model = FiniteModel(5, 2, self.quad, mode="concave")
        X = torch.randn(7, 2)
        choice, fx = model.forward(X, "soft")
        self.assertEqual(choice.shape, (7, 2))
        self.assertEqual(fx.shape, (7,))
    
    def test_soft_weights_sum_to_one(self):
        model = FiniteModel(3, 2, self.quad, mode="convex")
        X = torch.randn(5, 2)
        _, _ = model.forward(X, "soft")
        w = model._last_weights
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(5)))
    
    def test_hard_mode(self):
        model = FiniteModel(5, 2, self.quad, mode="convex")
        X = torch.randn(8, 2)
        _, fx = model.forward(X, "hard")
        self.assertIsNotNone(model._last_idx)
        self.assertEqual(model._last_mean_max_weight, 1.0)
    
    def test_gradient_flow(self):
        model = FiniteModel(5, 2, self.quad, mode="convex")
        X = torch.randn(4, 2)
        _, fx = model.forward(X, "soft")
        fx.sum().backward()
        self.assertIsNotNone(model.Y_rest.grad)
        self.assertIsNotNone(model.intercept_rest.grad)
    
    def test_convex_max_property(self):
        model = FiniteModel(3, 1, self.lin, mode="convex")
        with torch.no_grad():
            model.Y_rest.data = torch.tensor([[[1.],[2.],[3.]]])
            model.intercept_rest.data = torch.tensor([[0.5,1.0,1.5]])
        X = torch.tensor([[1.],[2.]])
        _, fx = model.forward(X, "hard")
        expected = torch.tensor([1.5, 4.5])
        torch.testing.assert_close(fx, expected)
    
    def test_concave_min_property(self):
        model = FiniteModel(3, 1, self.lin, mode="concave")
        with torch.no_grad():
            model.Y_rest.data = torch.tensor([[[1.],[2.],[3.]]])
            model.intercept_rest.data = torch.tensor([[0.5,1.0,1.5]])
        X = torch.tensor([[1.],[2.]])
        _, fx = model.forward(X, "hard")
        expected = torch.tensor([0.5, 1.5])
        torch.testing.assert_close(fx, expected)


class TestFiniteModelMetricProperties(TimedTestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_metric_inf_transform(self):
        eps = 1e-8
        d = lambda x, y: ((x - y)**2).sum(-1).add(eps).sqrt()
        model = FiniteModel(5, 2, d, mode="concave", is_y_parameter=False)
        with torch.no_grad():
            model.intercept_rest.zero_()

        Y = model.full_Y()
        Z = Y[:,0:1,:].squeeze(0)
        _, fz = model.forward(Z, "soft")
        self.assertLess(fz.item(), 0.01)

        _, v, _ = model.inf_transform(Z, steps=200, lr=5e-3, optimizer="adam")
        self.assertTrue(torch.isfinite(v).all())
        self.assertLess(abs(v.item() + fz.item()), 0.1)

    def test_negative_metric_sup_transform(self):
        eps = 1e-8
        negd = lambda x, y: -((x - y)**2).sum(-1).add(eps).sqrt()
        model = FiniteModel(5, 2, negd, mode="convex", is_y_parameter=False)
        with torch.no_grad():
            model.intercept_rest.zero_()

        Y = model.full_Y()
        Z = Y[:,0:1,:].squeeze(0)
        _, fz = model.forward(Z, "soft")
        self.assertLess(abs(fz.item()), 0.01)

        _, v, _ = model.sup_transform(Z, steps=200, lr=5e-3, optimizer="adam")
        self.assertTrue(torch.isfinite(v).all())
        self.assertLess(abs(v.item() + fz.item()), 0.1)

    def test_quadratic_surplus_conjugate_symmetry(self):
        surplus = lambda x, y: -0.5*((x - y)**2).sum(-1)
        model = FiniteModel(10, 2, surplus, mode="convex")
        X = torch.randn(5, 2)

        _, fx = model.forward(X)
        _, fcx, _ = model.sup_transform(X, steps=50, lr=1e-2, optimizer="adam")

        corr = torch.corrcoef(torch.stack([fx, fcx]))[0,1]
        self.assertGreater(abs(corr), 0.3)

    def test_inf_sup_are_finite(self):
        kernel = lambda x, y: -((x - y)**2).sum(-1)
        model = FiniteModel(8, 2, kernel)
        Z = torch.randn(3, 2)

        _, infv, _ = model.inf_transform(Z)
        _, supv, _ = model.sup_transform(Z)

        self.assertTrue(torch.isfinite(infv).all())
        self.assertTrue(torch.isfinite(supv).all())


class TestFiniteModelConvexityConcavity(TimedTestCase):
    def test_convex_piecewise_linear(self):
        torch.manual_seed(42)
        k = lambda x,y: (x*y).sum(-1)
        model = FiniteModel(5,1,k,mode="convex")

        X = torch.linspace(-2,2,100).unsqueeze(-1)
        _, f = model.forward(X)

        sec = f[2:] - 2*f[1:-1] + f[:-2]
        self.assertTrue((sec >= -1e-4).all())

    def test_concave_piecewise_linear(self):
        torch.manual_seed(43)
        k = lambda x,y: (x*y).sum(-1)
        model = FiniteModel(5,1,k,mode="concave")

        X = torch.linspace(-2,2,100).unsqueeze(-1)
        _, f = model.forward(X)

        sec = f[2:] - 2*f[1:-1] + f[:-2]
        self.assertTrue((sec <= 1e-4).all())


if __name__ == "__main__":
    unittest.main()
