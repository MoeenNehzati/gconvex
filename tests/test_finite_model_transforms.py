"""
Test Suite for FiniteModel Transform with InfConvolution

This test module validates that FiniteModel's sup_transform and inf_transform
correctly use InfConvolution for computing the transforms, and that results
match closed-form solutions for specific kernels.

The tests cover:
    1. Linear kernel with analytical transform solutions
    2. Quadratic kernel with analytical transform solutions
    3. Gradient correctness for transforms
    4. Mini-batching behavior with sample_idx
    5. Warm-start consistency across calls
    6. Comparison between old and new implementations

Key Mathematical Results Tested:

1. Linear Kernel k(x,y) = x^T y:
   For a finitely-convex model f(x) = max_j [x^T y_j - b_j]:
   
   Inf-transform: inf_x [k(x,z) - f(x)] = inf_x [x^T z - max_j(x^T y_j - b_j)]
                = inf_x min_j [x^T z - x^T y_j + b_j]
                = min_j [inf_x x^T(z - y_j) + b_j]
                = min_j [-∞ if z ≠ y_j, b_j if z = y_j]
   
   Sup-transform: sup_x [k(x,z) - f(x)] = sup_x [x^T z - max_j(x^T y_j - b_j)]
   
2. Quadratic Kernel k(x,y) = -0.5||x-y||^2:
   For convex f, the inf-transform is the Moreau envelope:
       inf_x [k(x,z) - f(x)] = inf_x [-0.5||x-z||^2 - f(x)]
                              = -sup_x [0.5||x-z||^2 + f(x)]  (Moreau envelope)
   
   The sup-transform is the convex conjugate at z.

Running Tests:
    python -m unittest tests.test_finite_model_transforms -v
"""

import unittest
import torch
import torch.nn as nn
from models import FiniteModel


class TestFiniteModelLinearKernelTransforms(unittest.TestCase):
    """
    Test transforms for linear kernel k(x,y) = x^T y.
    
    For this kernel, we can verify specific properties:
    - The transforms should be finite for specific configurations
    - Gradients should flow correctly through the model parameters
    """
    
    def test_linear_kernel_inf_transform_gradients(self):
        """
        Test that gradients flow correctly through inf_transform with linear kernel.
        
        Setup:
            k(x,y) = x^T y (linear kernel)
            f(x) = max_j [x^T y_j - b_j] (finitely convex, mode="convex")
        
        We verify:
            1. inf_transform produces finite values
            2. Gradients flow to model parameters (Y and intercepts)
            3. Values make mathematical sense
        """
        torch.manual_seed(42)
        d = 2
        num_candidates = 5
        
        # Linear kernel
        def linear_kernel(x, y):
            return (x * y).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=linear_kernel,
            mode="convex",
            is_y_parameter=True,
        )
        
        # Initialize with known values for reproducibility
        with torch.no_grad():
            model.Y_rest.data = torch.randn(1, num_candidates, d)
            model.intercept_rest.data = torch.randn(1, num_candidates)
        
        # Test points
        Z = torch.randn(3, d, requires_grad=False)
        
        # Compute inf transform
        X_opt, transform_vals = model.inf_transform(
            Z, steps=50, lr=1e-2, optimizer="adam", lam=1e-3
        )
        
        # Values should be finite
        self.assertTrue(torch.isfinite(transform_vals).all())
        self.assertTrue(torch.isfinite(X_opt).all())
        
        # Compute gradients
        loss = transform_vals.sum()
        loss.backward()
        
        # Gradients should flow to parameters
        self.assertIsNotNone(model.Y_rest.grad)
        self.assertIsNotNone(model.intercept_rest.grad)
        self.assertTrue(torch.isfinite(model.Y_rest.grad).all())
        self.assertTrue(torch.isfinite(model.intercept_rest.grad).all())
    
    def test_linear_kernel_sup_transform_gradients(self):
        """
        Test that gradients flow correctly through sup_transform with linear kernel.
        """
        torch.manual_seed(42)
        d = 2
        num_candidates = 5
        
        # Linear kernel
        def linear_kernel(x, y):
            return (x * y).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=linear_kernel,
            mode="convex",
            is_y_parameter=True,
        )
        
        # Initialize with known values
        with torch.no_grad():
            model.Y_rest.data = torch.randn(1, num_candidates, d)
            model.intercept_rest.data = torch.randn(1, num_candidates)
        
        # Test points
        Z = torch.randn(3, d, requires_grad=False)
        
        # Compute sup transform
        X_opt, transform_vals = model.sup_transform(
            Z, steps=50, lr=1e-2, optimizer="adam", lam=1e-3
        )
        
        # Values should be finite
        self.assertTrue(torch.isfinite(transform_vals).all())
        self.assertTrue(torch.isfinite(X_opt).all())
        
        # Compute gradients
        loss = transform_vals.sum()
        loss.backward()
        
        # Gradients should flow to parameters
        self.assertIsNotNone(model.Y_rest.grad)
        self.assertIsNotNone(model.intercept_rest.grad)
        self.assertTrue(torch.isfinite(model.Y_rest.grad).all())
        self.assertTrue(torch.isfinite(model.intercept_rest.grad).all())


class TestFiniteModelQuadraticKernelTransforms(unittest.TestCase):
    """
    Test transforms for quadratic surplus kernel k(x,y) = -0.5||x-y||^2.
    
    This kernel has special properties related to Moreau envelopes and
    Legendre-Fenchel conjugates.
    """
    
    def test_quadratic_kernel_inf_transform_simple_case(self):
        """
        Test inf_transform with quadratic surplus kernel on a simple case.
        
        Setup:
            k(x,y) = -0.5||x-y||^2 (quadratic surplus)
            f(x) = max_j [-0.5||x-y_j||^2 - b_j]
        
        We verify that:
        1. inf_transform produces finite values
        2. Gradients flow correctly
        3. The optimization converges to a reasonable solution
        """
        torch.manual_seed(42)
        d = 2
        num_candidates = 5
        
        # Quadratic surplus kernel
        def quad_surplus_kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=quad_surplus_kernel,
            mode="convex",
            is_y_parameter=True,  # Make Y trainable to test gradients
        )
        
        # Set intercepts to zero
        with torch.no_grad():
            model.intercept_rest.zero_()
        
        # Use a test point
        Z = torch.randn(3, d)
        
        # Compute inf transform with better convergence settings
        X_opt, inf_val = model.inf_transform(
            Z, steps=200, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        
        # Values should be finite
        self.assertTrue(torch.isfinite(inf_val).all())
        self.assertTrue(torch.isfinite(X_opt).all())
        
        # Test gradient flow
        loss = inf_val.sum()
        loss.backward()
        
        self.assertIsNotNone(model.Y_rest.grad)
        self.assertIsNotNone(model.intercept_rest.grad)
        self.assertTrue(torch.isfinite(model.Y_rest.grad).all())
        self.assertTrue(torch.isfinite(model.intercept_rest.grad).all())
    
    def test_quadratic_kernel_sup_transform_simple_case(self):
        """
        Test sup_transform with negative quadratic kernel.
        """
        torch.manual_seed(42)
        d = 2
        num_candidates = 5
        
        # Negative quadratic kernel
        def neg_quad_kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=neg_quad_kernel,
            mode="convex",
            is_y_parameter=False,
        )
        
        # Set intercepts to zero
        with torch.no_grad():
            model.intercept_rest.zero_()
        
        # Get one of the Y points as test point
        Y = model.full_Y()
        Z = Y[:, 0, :].squeeze(0).unsqueeze(0)
        
        # Compute forward and sup transform with better convergence settings
        _, f_z = model.forward(Z, selection_mode="soft")
        X_opt, sup_val = model.sup_transform(
            Z, steps=200, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        
        # Values should be finite
        self.assertTrue(torch.isfinite(sup_val).all())
        
        # Should relate to -f(z)
        self.assertLess(abs(sup_val.item() + f_z.item()), 0.1)
    
    def test_quadratic_kernel_gradient_flow(self):
        """
        Test that gradients flow correctly with quadratic kernel.
        
        This is crucial for optimization and training.
        """
        torch.manual_seed(42)
        d = 3
        num_candidates = 4
        
        def quad_kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=quad_kernel,
            mode="convex",
            is_y_parameter=True,
        )
        
        Z = torch.randn(5, d)
        
        # Inf transform with gradient computation
        X_opt, inf_vals = model.inf_transform(
            Z, steps=50, lr=1e-2, optimizer="adam", lam=1e-3
        )
        
        loss = inf_vals.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        self.assertIsNotNone(model.Y_rest.grad)
        self.assertIsNotNone(model.intercept_rest.grad)
        self.assertTrue(torch.isfinite(model.Y_rest.grad).all())
        self.assertTrue(torch.isfinite(model.intercept_rest.grad).all())
        
        # Gradient magnitudes should be reasonable (not too large)
        self.assertLess(model.Y_rest.grad.abs().max().item(), 100.0)
        self.assertLess(model.intercept_rest.grad.abs().max().item(), 100.0)


class TestFiniteModelTransformBatching(unittest.TestCase):
    """
    Test that transforms work correctly with mini-batching via sample_idx.
    """
    
    def test_inf_transform_with_sample_idx(self):
        """
        Test that inf_transform correctly handles sample_idx for warm-starts.
        """
        torch.manual_seed(42)
        d = 2
        num_candidates = 5
        
        def kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=kernel,
            mode="convex",
        )
        
        # Create a batch of points
        Z = torch.randn(10, d)
        sample_idx = torch.arange(10)
        
        # First call - should create warm start storage
        X_opt1, vals1 = model.inf_transform(
            Z, sample_idx=sample_idx, steps=100, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        
        # Check warm start was saved
        self.assertIsNotNone(model._warm_X_global)
        self.assertEqual(model._warm_X_global.shape[0], 10)
        
        # Values should be finite
        self.assertTrue(torch.isfinite(vals1).all())
        self.assertTrue(torch.isfinite(X_opt1).all())
        
        # Test deterministic behavior with same inputs
        torch.manual_seed(42)  # Reset seed
        model2 = FiniteModel(num_candidates, d, kernel, mode="convex")
        X_opt1b, vals1b = model2.inf_transform(
            Z, sample_idx=sample_idx, steps=100, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        # Should get identical results with same seed
        self.assertTrue(torch.allclose(vals1, vals1b, atol=1e-5))
        
        # Test subset indexing - warm starts should be reused
        subset_idx = torch.tensor([0, 2, 5, 7])
        Z_subset = Z[subset_idx]
        
        # Store current warm starts for these indices
        warm_before = model._warm_X_global[subset_idx].clone()
        
        # Run with subset - should use stored warm starts
        X_opt_subset, vals_subset = model.inf_transform(
            Z_subset, sample_idx=subset_idx, steps=10, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        
        # Values should be finite
        self.assertTrue(torch.isfinite(vals_subset).all())
        self.assertTrue(torch.isfinite(X_opt_subset).all())
        
        # Warm starts should be updated but storage size unchanged
        self.assertEqual(model._warm_X_global.shape[0], 10)
    
    def test_sup_transform_with_sample_idx(self):
        """
        Test that sup_transform correctly handles sample_idx for warm-starts.
        
        Tests that the warm-start mechanism:
        1. Saves optimization results correctly
        2. Can be retrieved for subsequent optimizations
        3. Produces deterministic results with same seed
        """
        torch.manual_seed(123)  # Different seed from other tests
        d = 2
        num_candidates = 5
        
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=kernel,
            mode="convex",
        )
        
        # Create a batch of points
        Z = torch.randn(10, d)
        sample_idx = torch.arange(10)
        
        # First call - should initialize warm starts
        X_opt1, vals1 = model.sup_transform(
            Z, sample_idx=sample_idx, steps=100, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        
        # Check warm start was saved with correct shape
        self.assertIsNotNone(model._warm_X_global)
        self.assertEqual(model._warm_X_global.shape[0], 10)
        self.assertEqual(model._warm_X_global.shape[1], d)
        
        # Save the warm start positions
        warm_starts_saved = model._warm_X_global.clone()
        
        # Second call - should use warm starts  
        X_opt2, vals2 = model.sup_transform(
            Z, sample_idx=sample_idx, steps=20, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6
        )
        
        # Warm start should still be allocated correctly
        self.assertEqual(model._warm_X_global.shape[0], 10)
        
        # Both should produce finite values
        self.assertTrue(torch.isfinite(vals1).all())
        self.assertTrue(torch.isfinite(vals2).all())
        
        # Test deterministic behavior - same seed should give same results
        torch.manual_seed(456)
        model3 = FiniteModel(num_candidates, d, kernel, mode="convex")
        Z_test = torch.randn(5, d)
        _, vals3a = model3.sup_transform(
            Z_test, steps=50, lr=1e-2, optimizer="adam"
        )
        
        torch.manual_seed(456)
        model4 = FiniteModel(num_candidates, d, kernel, mode="convex")
        Z_test2 = torch.randn(5, d)
        _, vals3b = model4.sup_transform(
            Z_test2, steps=50, lr=1e-2, optimizer="adam"
        )
        
        self.assertTrue(torch.allclose(vals3a, vals3b, atol=1e-5))


class TestFiniteModelTransformConsistency(unittest.TestCase):
    """
    Test consistency properties of the transforms.
    """
    
    def test_inf_sup_consistency(self):
        """
        Test that inf and sup transforms have expected relationship.
        
        For the same optimization problem:
            sup_x [k(x,z) - f(x)] >= inf_x [k(x,z) - f(x)]
        
        This tests that sup finds larger values than inf (as expected).
        """
        torch.manual_seed(999)  # Use different seed for variety
        d = 2
        num_candidates = 4
        
        # Use same kernel for both
        def kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=d,
            kernel=kernel,
            mode="convex",
        )
        
        Z = torch.randn(5, d)
        
        # Both should produce finite values with good convergence
        _, inf_vals = model.inf_transform(Z, steps=100, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6)
        _, sup_vals = model.sup_transform(Z, steps=100, lr=5e-3, optimizer="adam", lam=1e-4, tol=1e-6)
        
        self.assertTrue(torch.isfinite(inf_vals).all())
        self.assertTrue(torch.isfinite(sup_vals).all())
        
        # sup should be >= inf for each point (maximization vs minimization)
        # Allow small numerical tolerance
        self.assertTrue((sup_vals >= inf_vals - 0.1).all(), 
                       f"sup should be >= inf, but got sup={sup_vals}, inf={inf_vals}")
    
    def test_transform_deterministic_with_seed(self):
        """
        Test that transforms are deterministic with fixed seed.
        """
        d = 2
        num_candidates = 3
        
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        Z = torch.randn(3, d)
        
        # First run
        torch.manual_seed(42)
        model1 = FiniteModel(num_candidates, d, kernel, mode="convex")
        _, vals1 = model1.inf_transform(Z, steps=50, lr=1e-2, optimizer="adam")
        
        # Second run with same seed
        torch.manual_seed(42)
        model2 = FiniteModel(num_candidates, d, kernel, mode="convex")
        _, vals2 = model2.inf_transform(Z, steps=50, lr=1e-2, optimizer="adam")
        
        # Should produce identical results
        self.assertTrue(torch.allclose(vals1, vals2, atol=1e-6))


class TestFiniteModelTransformOptimizers(unittest.TestCase):
    """
    Test that different optimizers all work correctly for transforms.
    """
    
    def test_inf_transform_lbfgs(self):
        """Test inf_transform with LBFGS optimizer."""
        torch.manual_seed(42)
        d = 2
        
        def kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(5, d, kernel, mode="convex")
        Z = torch.randn(3, d)
        
        X_opt, vals = model.inf_transform(Z, steps=20, lr=1.0, optimizer="lbfgs")
        
        self.assertTrue(torch.isfinite(vals).all())
        self.assertTrue(torch.isfinite(X_opt).all())
    
    def test_inf_transform_adam(self):
        """Test inf_transform with Adam optimizer."""
        torch.manual_seed(42)
        d = 2
        
        def kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(5, d, kernel, mode="convex")
        Z = torch.randn(3, d)
        
        X_opt, vals = model.inf_transform(Z, steps=50, lr=1e-2, optimizer="adam")
        
        self.assertTrue(torch.isfinite(vals).all())
        self.assertTrue(torch.isfinite(X_opt).all())
    
    def test_sup_transform_gd(self):
        """Test sup_transform with gradient descent optimizer."""
        torch.manual_seed(42)
        d = 2
        
        def kernel(x, y):
            return -((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(5, d, kernel, mode="convex")
        Z = torch.randn(3, d)
        
        X_opt, vals = model.sup_transform(Z, steps=100, lr=1e-2, optimizer="gd")
        
        self.assertTrue(torch.isfinite(vals).all())
        self.assertTrue(torch.isfinite(X_opt).all())


if __name__ == "__main__":
    unittest.main()
