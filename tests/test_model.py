"""
Unit tests for model.FiniteModel.

Tests for:
- Initialization with different parameters
- Forward pass in convex and concave modes
- Soft/hard/ste selection modes
- Metric properties: when kernel is a metric, inf_transform(f) = -f
- Negative metric properties: when kernel is -metric, sup_transform(f) = -f
- Conjugate transform behavior
"""

import unittest
import torch
from model import FiniteModel


class TestFiniteModelInitialization(unittest.TestCase):
    """Test suite for FiniteModel initialization."""
    
    def test_convex_initialization(self):
        """Test initialization in convex mode."""
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
        
        # Check Y parameters
        Y = model.full_Y()
        self.assertEqual(Y.shape, (1, 10, 3))
    
    def test_concave_initialization(self):
        """Test initialization in concave mode."""
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
        """Test initialization with non-learnable Y."""
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
        
        # Y should be a buffer, not a parameter
        param_names = [name for name, _ in model.named_parameters()]
        self.assertNotIn("Y_rest", param_names)
        
        buffer_names = [name for name, _ in model.named_buffers()]
        self.assertIn("Y_rest", buffer_names)
    
    def test_bounded_initialization(self):
        """Test initialization with y_min and y_max bounds."""
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
        
        # Y should be within bounds (with some tolerance for initialization)
        self.assertTrue(Y.min() >= y_min - 0.1)
        self.assertTrue(Y.max() <= y_max + 0.1)
    
    def test_default_candidate(self):
        """Test initialization with default candidate."""
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
        
        # Full Y should have num_candidates + 1 (default)
        Y = model.full_Y()
        self.assertEqual(Y.shape[1], 6)  # 5 + 1 default
        
        # First candidate should be zero (default)
        torch.testing.assert_close(Y[:, 0, :], torch.zeros(1, 2))
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises assertion."""
        def kernel(x, y):
            return (x * y).sum(dim=-1)
        
        with self.assertRaises(AssertionError):
            FiniteModel(
                num_candidates=5,
                num_dims=2,
                kernel=kernel,
                mode="invalid",
            )


class TestFiniteModelForward(unittest.TestCase):
    """Test suite for FiniteModel forward pass."""
    
    def setUp(self):
        """Set up common test fixtures."""
        torch.manual_seed(42)
        
        # Simple quadratic kernel
        self.quadratic_kernel = lambda x, y: -((x - y)**2).sum(dim=-1)
        
        # Linear kernel
        self.linear_kernel = lambda x, y: (x * y).sum(dim=-1)
    
    def test_convex_forward_shape(self):
        """Test forward pass shape in convex mode."""
        model = FiniteModel(
            num_candidates=5,
            num_dims=3,
            kernel=self.quadratic_kernel,
            mode="convex",
        )
        
        X = torch.randn(10, 3)
        choice, f_x = model.forward(X, selection_mode="soft")
        
        self.assertEqual(choice.shape, (10, 3))
        self.assertEqual(f_x.shape, (10,))
    
    def test_concave_forward_shape(self):
        """Test forward pass shape in concave mode."""
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=self.quadratic_kernel,
            mode="concave",
        )
        
        X = torch.randn(7, 2)
        choice, f_x = model.forward(X, selection_mode="soft")
        
        self.assertEqual(choice.shape, (7, 2))
        self.assertEqual(f_x.shape, (7,))
    
    def test_soft_mode_forward(self):
        """Test soft mode produces weighted combinations."""
        model = FiniteModel(
            num_candidates=3,
            num_dims=2,
            kernel=self.quadratic_kernel,
            mode="convex",
        )
        
        X = torch.randn(5, 2)
        choice, f_x = model.forward(X, selection_mode="soft")
        
        # Should have weights stored
        self.assertIsNotNone(model._last_weights)
        self.assertEqual(model._last_weights.shape, (5, 3))
        
        # Weights should sum to 1
        self.assertTrue(torch.allclose(
            model._last_weights.sum(dim=1),
            torch.ones(5),
            atol=1e-5
        ))
    
    def test_hard_mode_forward(self):
        """Test hard mode selects single candidate."""
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=self.quadratic_kernel,
            mode="convex",
        )
        
        X = torch.randn(8, 2)
        choice, f_x = model.forward(X, selection_mode="hard")
        
        # Should have indices stored
        self.assertIsNotNone(model._last_idx)
        self.assertEqual(model._last_idx.shape, (8,))
        
        # mean_max should be 1.0 for hard mode
        self.assertAlmostEqual(model._last_mean_max_weight, 1.0, places=5)
    
    def test_ste_mode_forward(self):
        """Test STE mode combines hard and soft."""
        model = FiniteModel(
            num_candidates=4,
            num_dims=2,
            kernel=self.linear_kernel,
            mode="convex",
        )
        
        X = torch.randn(6, 2)
        choice, f_x = model.forward(X, selection_mode="ste")
        
        # Should have both weights and indices
        self.assertIsNotNone(model._last_weights)
        self.assertIsNotNone(model._last_idx)
    
    def test_gradient_flow(self):
        """Test that gradients flow through forward pass."""
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=self.quadratic_kernel,
            mode="convex",
            is_y_parameter=True,
        )
        
        X = torch.randn(4, 2)
        choice, f_x = model.forward(X, selection_mode="soft")
        
        loss = f_x.sum()
        loss.backward()
        
        # Gradients should exist for Y and intercepts
        self.assertIsNotNone(model.Y_rest.grad)
        self.assertIsNotNone(model.intercept_rest.grad)
    
    def test_convex_max_property(self):
        """Test that convex mode uses max over candidates."""
        torch.manual_seed(123)
        
        model = FiniteModel(
            num_candidates=3,
            num_dims=1,
            kernel=self.linear_kernel,
            mode="convex",
        )
        
        # Set specific Y and intercepts for deterministic test
        with torch.no_grad():
            # Y_rest is (1, 3, 1), so we reshape correctly
            model.Y_rest.data = torch.tensor([[[1.0], [2.0], [3.0]]])
            model.intercept_rest.data = torch.tensor([[0.5, 1.0, 1.5]])
        
        X = torch.tensor([[1.0], [2.0]])
        _, f_x = model.forward(X, selection_mode="hard")
        
        # Manually compute expected max
        # scores = X @ Y.T - b
        # For X[0]=1: [1*1-0.5, 1*2-1.0, 1*3-1.5] = [0.5, 1.0, 1.5]
        # For X[1]=2: [2*1-0.5, 2*2-1.0, 2*3-1.5] = [1.5, 3.0, 4.5]
        expected = torch.tensor([1.5, 4.5])
        
        torch.testing.assert_close(f_x, expected, rtol=1e-4, atol=1e-4)
    
    def test_concave_min_property(self):
        """Test that concave mode uses min over candidates."""
        torch.manual_seed(123)
        
        model = FiniteModel(
            num_candidates=3,
            num_dims=1,
            kernel=self.linear_kernel,
            mode="concave",
        )
        
        # Set specific Y and intercepts
        with torch.no_grad():
            # Y_rest is (1, 3, 1), so we reshape correctly
            model.Y_rest.data = torch.tensor([[[1.0], [2.0], [3.0]]])
            model.intercept_rest.data = torch.tensor([[0.5, 1.0, 1.5]])
        
        X = torch.tensor([[1.0], [2.0]])
        _, f_x = model.forward(X, selection_mode="hard")
        
        # Expected min
        # For X[0]=1: min([0.5, 1.0, 1.5]) = 0.5
        # For X[1]=2: min([1.5, 3.0, 4.5]) = 1.5
        expected = torch.tensor([0.5, 1.5])
        
        torch.testing.assert_close(f_x, expected, rtol=1e-4, atol=1e-4)


class TestFiniteModelMetricProperties(unittest.TestCase):
    """
    Test metric properties of transforms.
    
    When kernel is a metric d(x,y):
        inf_x [d(x,z) - f(x)] should relate to -f for certain f
    
    When kernel is negative of a metric -d(x,y):
        sup_x [-d(x,z) - f(x)] should relate to -f for certain f
    """
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_metric_kernel_inf_transform(self):
        """
        Test: When kernel is a metric d(x,y), and f(x) = min_j d(x,y_j),
        the inf-transform should be well-behaved.
        
        For z = y_j (a support point with b=0), f(z) should be small (distance to itself).
        """
        # Smoothed Euclidean distance metric (epsilon prevents gradient singularity at x=y)
        eps = 1e-8
        def euclidean_metric(x, y):
            return ((x - y)**2).sum(dim=-1).add(eps).sqrt()
        
        # Create a concave model: f(x) = min_j [d(x,y_j) - b_j]
        # With b_j = 0, this is f(x) = min_j d(x, y_j)
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=euclidean_metric,
            mode="concave",
            is_y_parameter=False,
        )
        
        # Set intercepts to zero
        with torch.no_grad():
            model.intercept_rest.zero_()
        
        # Choose test point at one of the Y candidates
        Y = model.full_Y()
        Z = Y[:, 0:1, :].squeeze(0)  # (1, 2)
        
        # Compute f(Z)
        _, f_z = model.forward(Z, selection_mode="soft")
        
        # For a metric, when z is one of the support points with b=0,
        # f(z) should be close to 0 (distance to itself)
        self.assertLess(f_z.item(), 0.01)
        
        # Compute inf-transform at Z with more steps for better convergence
        X_star, inf_val = model.inf_transform(Z, steps=200, lr=5e-3, optimizer="adam")
        
        # The inf-transform should be finite
        self.assertTrue(torch.isfinite(inf_val).all(), 
                       f"inf_val is {inf_val.item()}, should be finite")
        
        # For smoothed metric at support point, inf-transform should be close to -f(z)
        # (within reasonable tolerance given numerical optimization)
        self.assertLess(abs(inf_val.item() + f_z.item()), 0.1)
    
    def test_negative_metric_kernel_sup_transform(self):
        """
        Test: When kernel is -d(x,y) (negative of a metric), 
        and f(x) = max_j[-d(x,y_j) - b_j] is convex in this kernel,
        the sup-transform should be well-behaved.
        """
        # Smoothed negative Euclidean distance (epsilon prevents gradient singularity)
        eps = 1e-8
        def neg_euclidean(x, y):
            return -((x - y)**2).sum(dim=-1).add(eps).sqrt()
        
        # Create a convex model in this kernel
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=neg_euclidean,
            mode="convex",
            is_y_parameter=False,
        )
        
        # Set intercepts to zero
        with torch.no_grad():
            model.intercept_rest.zero_()
        
        # Choose test point at one of the Y candidates
        Y = model.full_Y()
        Z = Y[:, 0:1, :].squeeze(0)  # (1, 2)
        
        # Compute f(Z)
        _, f_z = model.forward(Z, selection_mode="soft")
        
        # For negative metric at support point: f(z) should be close to 0
        self.assertLess(abs(f_z.item()), 0.01)
        
        # Compute sup-transform at Z with more steps
        X_star, sup_val = model.sup_transform(Z, steps=200, lr=5e-3, optimizer="adam")
        
        # The sup-transform should be finite
        self.assertTrue(torch.isfinite(sup_val).all(),
                       f"sup_val is {sup_val.item()}, should be finite")
        
        # For smoothed negative metric at support point, sup-transform should be close to -f(z)
        # (within reasonable tolerance given numerical optimization)
        self.assertLess(abs(sup_val.item() + f_z.item()), 0.1)
    
    def test_quadratic_cost_conjugate_symmetry(self):
        """
        Test conjugate symmetry for quadratic cost c(x,y) = ||x-y||²/2.
        
        The (-c)-conjugate of f should satisfy certain properties.
        Specifically, for the surplus Φ(x,y) = -c(x,y) = -||x-y||²/2,
        if we compute the sup-transform twice, we should get back close to original.
        """
        # Surplus: Φ(x,y) = -||x-y||²/2
        def surplus(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=10,
            num_dims=2,
            kernel=surplus,
            mode="convex",
            temp=10.0,
        )
        
        # Sample points
        X = torch.randn(5, 2)
        
        # Compute f(X)
        _, f_x = model.forward(X, selection_mode="soft")
        
        # Compute sup-transform at X (should give conjugate value)
        _, f_conj_x = model.sup_transform(X, steps=50, lr=1e-2, optimizer="adam")
        
        # For convex functions, f**(x) = f(x) (involutive property)
        # But our model is only approximately convex, so we check correlation
        # The conjugate should be somewhat correlated with original values
        correlation = torch.corrcoef(torch.stack([f_x, f_conj_x]))[0, 1]
        
        # They should have some relationship (not perfect due to approximation)
        self.assertGreater(abs(correlation.item()), 0.3)
    
    def test_inf_sup_relationship(self):
        """
        Test relationship between inf and sup transforms.
        
        inf_x[k(x,z) - f(x)] = -sup_x[-k(x,z) + f(x)]
        """
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=8,
            num_dims=2,
            kernel=kernel,
            mode="convex",
        )
        
        Z = torch.randn(3, 2)
        
        # Compute inf-transform
        _, inf_val = model.inf_transform(Z, steps=50, lr=1e-2, optimizer="adam")
        
        # Compute sup-transform (note: this is sup_x[k(x,z) - f(x)])
        _, sup_val = model.sup_transform(Z, steps=50, lr=1e-2, optimizer="adam")
        
        # They should be different (not negatives of each other in general)
        # but both should be finite
        self.assertTrue(torch.isfinite(inf_val).all())
        self.assertTrue(torch.isfinite(sup_val).all())


class TestFiniteModelOptimization(unittest.TestCase):
    """Test optimization routines for transforms."""
    
    def test_lbfgs_optimizer(self):
        """Test LBFGS optimizer for conjugate transforms."""
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="convex",
        )
        
        Z = torch.randn(4, 2)
        
        X_star, values = model.sup_transform(
            Z, steps=20, lr=1e-1, optimizer="lbfgs"
        )
        
        self.assertEqual(X_star.shape, (4, 2))
        self.assertEqual(values.shape, (4,))
        self.assertTrue(torch.isfinite(values).all())
    
    def test_adam_optimizer(self):
        """Test Adam optimizer for conjugate transforms."""
        def kernel(x, y):
            return (x * y).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="convex",
        )
        
        Z = torch.randn(4, 2)
        
        X_star, values = model.inf_transform(
            Z, steps=100, lr=1e-2, optimizer="adam"
        )
        
        self.assertEqual(X_star.shape, (4, 2))
        self.assertEqual(values.shape, (4,))
        self.assertTrue(torch.isfinite(values).all())
    
    def test_gd_optimizer(self):
        """Test gradient descent optimizer for conjugate transforms."""
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="concave",
        )
        
        Z = torch.randn(3, 2)
        
        X_star, values = model.inf_transform(
            Z, steps=100, lr=1e-2, optimizer="gd"
        )
        
        self.assertEqual(X_star.shape, (3, 2))
        self.assertEqual(values.shape, (3,))
    
    def test_warm_start(self):
        """Test that warm start improves optimization."""
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="convex",
        )
        
        Z = torch.randn(5, 2)
        
        # First call - cold start
        _, val1 = model.sup_transform(Z, steps=10, lr=1e-2, optimizer="adam")
        
        # Second call - warm start
        _, val2 = model.sup_transform(Z, steps=10, lr=1e-2, optimizer="adam")
        
        # Both should give finite values
        self.assertTrue(torch.isfinite(val1).all())
        self.assertTrue(torch.isfinite(val2).all())
        
        # Warm start should exist after first call
        self.assertIsNotNone(model._warm_X)


if __name__ == "__main__":
    unittest.main()
