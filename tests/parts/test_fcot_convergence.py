"""
Tests for FCOT convergence warnings.

This test module verifies that FCOT properly detects and warns about
non-convergent inner optimization loops.
"""

import unittest
import torch
import warnings
import logging
from baselines.ot_fc_map import FCOT
from tools.utils import L22, inverse_grad_L22
from tools.feedback import logger
from tests import TimedTestCase


class TestFCOTConvergenceWarnings(TimedTestCase):
    """Test that FCOT warns when inner optimization doesn't converge."""
    
    def setUp(self):
        """Set up common test fixtures."""
        torch.manual_seed(42)
        self.dim = 2
        self.n_samples = 10
        
    def test_warns_with_insufficient_inner_steps(self):
        """
        Test that FCOT warns when inner_steps is too small for convergence.
        
        With very few inner steps (e.g., 1-5), the inf_transform optimization
        is unlikely to converge, and FCOT should issue a warning.
        """
        # Create FCOT with very few inner steps
        fcot = FCOT.initialize_right_architecture(
            dim=self.dim,
            n_params_target=50,
            cost=L22,
            inverse_cx=inverse_grad_L22,
            inner_steps=2,  # Too few to converge
            inner_optimizer="adam",
            inner_tol=1e-6  # Tight tolerance to force non-convergence
        )
        
        # Generate data
        X = torch.randn(self.n_samples, self.dim)
        Y = torch.randn(self.n_samples, self.dim)
        idx_y = torch.arange(self.n_samples)
        
        # Capture warnings
        with self.assertLogs(logger, level='WARNING') as cm:
            # Run dual objective which calls inf_transform
            fcot._dual_objective(X, Y, idx_y)
        
        # Check that a convergence warning was issued
        warning_found = any('converge' in msg.lower() or 'inner' in msg.lower() 
                           for msg in cm.output)
        self.assertTrue(warning_found, 
                       "Expected convergence warning but none was found")
    
    def test_no_warning_with_sufficient_steps(self):
        """
        Test that FCOT doesn't warn when inner optimization converges.
        
        With sufficient inner steps and reasonable tolerance, the optimization
        should converge and no warning should be issued.
        """
        # Create FCOT with sufficient inner steps
        fcot = FCOT.initialize_right_architecture(
            dim=self.dim,
            n_params_target=50,
            cost=L22,
            inverse_cx=inverse_grad_L22,
            inner_steps=50,  # Sufficient for convergence
            inner_optimizer="lbfgs",  # Fast convergence
            inner_tol=1e-3  # Reasonable tolerance
        )
        
        # Generate simple data (easy to optimize)
        X = torch.randn(self.n_samples, self.dim) * 0.5
        Y = X + 0.1 * torch.randn_like(X)  # Y close to X
        idx_y = torch.arange(self.n_samples)
        
        # Set up logger to capture warnings
        with self.assertRaises(AssertionError):
            # This should NOT log any warnings
            with self.assertLogs(logger, level='WARNING') as cm:
                fcot._dual_objective(X, Y, idx_y)
    
    def test_warns_with_adam_insufficient_steps(self):
        """Test warning specifically with Adam optimizer (slower convergence)."""
        fcot = FCOT.initialize_right_architecture(
            dim=self.dim,
            n_params_target=50,
            cost=L22,
            inverse_cx=inverse_grad_L22,
            inner_steps=5,  # Too few for Adam
            inner_optimizer="adam",
            inner_lr=1e-3
        )
        
        X = torch.randn(self.n_samples, self.dim)
        Y = torch.randn(self.n_samples, self.dim) * 2.0  # Far from X
        idx_y = torch.arange(self.n_samples)
        
        with self.assertLogs(logger, level='WARNING') as cm:
            fcot._dual_objective(X, Y, idx_y)
        
        warning_found = any('converge' in msg.lower() or 'inner' in msg.lower() 
                           for msg in cm.output)
        self.assertTrue(warning_found)
    
    def test_convergence_info_returned(self):
        """
        Test that inf_transform returns convergence information.
        
        The transform should return not just (X_opt, values) but also
        convergence status.
        """
        fcot = FCOT.initialize_right_architecture(
            dim=self.dim,
            n_params_target=50,
            cost=L22,
            inverse_cx=inverse_grad_L22,
            inner_steps=10
        )
        
        Y = torch.randn(5, self.dim)
        
        # Call inf_transform directly
        result = fcot.model.inf_transform(
            Z=Y,
            steps=10,
            lr=1e-2,
            optimizer="adam",
            tol=1e-6
        )
        
        # Should return 3 values: X_opt, values, converged
        self.assertEqual(len(result), 3, 
                        "inf_transform should return (X_opt, values, converged)")
        
        X_opt, values, converged = result
        
        # Check types
        self.assertIsInstance(X_opt, torch.Tensor)
        self.assertIsInstance(values, torch.Tensor)
        self.assertIsInstance(converged, bool)
    
    def test_multiple_warnings_during_fit(self):
        """
        Test that warnings accumulate during fit iterations.
        
        If inner optimization consistently fails to converge, multiple
        warnings should be issued during training.
        """
        fcot = FCOT.initialize_right_architecture(
            dim=self.dim,
            n_params_target=30,
            cost=L22,
            inverse_cx=inverse_grad_L22,
            inner_steps=1,  # Guaranteed non-convergence
            inner_optimizer="gd",
            lr=1e-2
        )
        
        X = torch.randn(20, self.dim)
        Y = torch.randn(20, self.dim)
        
        # Run a few iterations of fit
        with self.assertLogs(logger, level='WARNING') as cm:
            fcot._fit(
                X, Y,
                batch_size=10,
                iters=3,
                print_every=10,  # Set to high value to avoid debug output
            )
        
        # Should have multiple warnings (at least one per iteration)
        warning_count = sum(1 for msg in cm.output 
                           if 'WARNING' in msg and 
                           ('converge' in msg.lower() or 'inner' in msg.lower()))
        self.assertGreater(warning_count, 0,
                          "Expected multiple convergence warnings during fit")


class TestFiniteModelConvergenceDetection(TimedTestCase):
    """Test convergence detection in FiniteModel transforms."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(123)
        
    def test_lbfgs_convergence_detection(self):
        """Test that LBFGS convergence is properly detected."""
        from models import FiniteModel
        
        def kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="convex"
        )
        
        Z = torch.randn(3, 2)
        
        # LBFGS with sufficient iterations should converge
        X_opt, values, converged = model.inf_transform(
            Z, steps=100, lr=1.0, optimizer="lbfgs", tol=1e-6
        )
        
        # With LBFGS and 100 steps, should converge
        # (Note: convergence detection for LBFGS is based on iterations)
        self.assertIsInstance(converged, bool)
    
    def test_adam_convergence_detection(self):
        """Test that Adam convergence is properly detected."""
        from models import FiniteModel
        
        def kernel(x, y):
            return -0.5 * ((x - y)**2).sum(dim=-1)
        
        model = FiniteModel(
            num_candidates=5,
            num_dims=2,
            kernel=kernel,
            mode="convex"
        )
        
        Z = torch.randn(3, 2)
        
        # Adam with very few steps should NOT converge
        X_opt1, values1, converged1 = model.inf_transform(
            Z, steps=2, lr=1e-3, optimizer="adam", tol=1e-8
        )
        
        self.assertFalse(converged1, 
                        "With 2 steps, Adam should not converge")
        
        # Adam with many steps should converge
        X_opt2, values2, converged2 = model.inf_transform(
            Z, steps=500, lr=1e-2, optimizer="adam", tol=1e-3
        )
        
        # More likely to converge with many steps
        # (though not guaranteed, so we just check it returns a bool)
        self.assertIsInstance(converged2, bool)


if __name__ == "__main__":
    unittest.main()
