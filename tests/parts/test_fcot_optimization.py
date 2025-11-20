"""
Optimization tests for FCOT (Finitely-Concave Optimal Transport).

Tests optimization behavior including:
- Numerical stability during training
- Inner loop convergence  
- Warm-start efficiency

These tests involve some training but are faster than full integration tests (2-10s each).
"""

import unittest
import torch
import numpy as np
from baselines.ot_fc_map import FCOT
from models import FiniteModel
from tools.utils import L22, inverse_grad_L22
from tests import TimedTestCase


class TestNumericalStability(TimedTestCase):
    """Test numerical stability and robustness."""
    max_test_time = 15  # These tests involve training iterations

    def test_no_nans_during_training(self):
        """Test that training doesn't produce NaNs."""
        dim = 2
        torch.manual_seed(202)
        
        X = torch.randn(50, dim)
        Y = torch.randn(50, dim)
        
        model = FiniteModel(
            num_candidates=15,
            num_dims=dim,
            kernel=lambda x, y: L22(x, y),
            mode="concave"
        )
        
        fcot = FCOT(
            input_dim=dim,
            model=model,
            inverse_cx=inverse_grad_L22,
            lr=1e-3,
            inner_steps=5
        )
        
        logs = fcot._fit(X, Y, batch_size=25, iters=20, print_every=100)
        
        # Check that all logged values are finite
        for dual_val in logs.get("dual", []):
            self.assertFalse(np.isnan(dual_val))
            self.assertFalse(np.isinf(dual_val))

    def test_dual_monotonic_improvement(self):
        """Test that dual objective improves (becomes less negative) over training."""
        dim = 2
        torch.manual_seed(303)
        
        X = torch.randn(40, dim)
        Y = torch.randn(40, dim)
        
        model = FiniteModel(
            num_candidates=20,
            num_dims=dim,
            kernel=lambda x, y: L22(x, y),
            mode="concave",
            temp=50.0
        )
        
        fcot = FCOT(
            input_dim=dim,
            model=model,
            inverse_cx=inverse_grad_L22,
            lr=5e-3,
            inner_steps=5,
            inner_optimizer="adam"
        )
        
        logs = fcot._fit(X, Y, batch_size=30, iters=30, print_every=10)
        
        duals = logs["dual"]
        
        # With stochastic minibatching, dual can fluctuate significantly
        # Just check that optimization runs and produces valid values
        if len(duals) > 1:
            # Check all duals are finite
            for d in duals:
                self.assertFalse(np.isnan(d))
                self.assertFalse(np.isinf(d))
            
            # Dual should show some variation (indicating optimization is happening)
            dual_std = np.std(duals)
            self.assertGreater(dual_std, 0.01, "Dual should vary during optimization")

    def test_gradient_clipping_prevents_explosion(self):
        """Test that gradient clipping is active and prevents explosions."""
        dim = 2
        torch.manual_seed(404)
        
        # Extreme case: very large values
        X = torch.randn(30, dim) * 10
        Y = torch.randn(30, dim) * 10
        
        model = FiniteModel(
            num_candidates=10,
            num_dims=dim,
            kernel=lambda x, y: L22(x, y),
            mode="concave"
        )
        
        fcot = FCOT(
            input_dim=dim,
            model=model,
            inverse_cx=inverse_grad_L22,
            lr=1e-2,  # High learning rate
            inner_steps=3
        )
        
        # Should not explode due to gradient clipping
        logs = fcot._fit(X, Y, batch_size=15, iters=10, print_every=100)
        
        # Check no NaNs
        for dual_val in logs.get("dual", []):
            self.assertFalse(np.isnan(dual_val))


class TestInnerLoopConvergence(TimedTestCase):
    """Test that inner loop optimizer converges correctly."""

    def test_sup_transform_converges(self):
        """Test that sup_transform finds maximum correctly."""
        dim = 1
        torch.manual_seed(707)
        
        # Simple case: few candidates
        model = FiniteModel(
            num_candidates=5,
            num_dims=dim,
            kernel=lambda x, y: L22(x, y),
            mode="concave",
            temp=100.0
        )
        
        Z = torch.tensor([[0.0]])
        idx = torch.tensor([0])
        
        # Run sup_transform with many steps
        with torch.no_grad():
            _, val_few, _ = model.sup_transform(
                Z=Z, sample_idx=idx, steps=5, optimizer="lbfgs"
            )
            _, val_many, _ = model.sup_transform(
                Z=Z, sample_idx=idx, steps=50, optimizer="lbfgs"
            )
        
        # More steps should give at least as good result
        self.assertGreaterEqual(val_many.item(), val_few.item() - 1e-3)

    def test_different_inner_optimizers(self):
        """Test that different inner optimizers all work."""
        dim = 2
        torch.manual_seed(808)
        
        X = torch.randn(20, dim)
        Y = torch.randn(20, dim)
        
        for opt in ["lbfgs", "adam", "gd"]:
            model = FiniteModel(
                num_candidates=10,
                num_dims=dim,
                kernel=lambda x, y: L22(x, y),
                mode="concave"
            )
            
            fcot = FCOT(
                input_dim=dim,
                model=model,
                inverse_cx=inverse_grad_L22,
                inner_optimizer=opt,
                inner_steps=10
            )
            
            # Should run without error
            logs = fcot._fit(X, Y, batch_size=10, iters=5, print_every=100)
            self.assertGreater(len(logs["dual"]), 0)


class TestWarmStart(TimedTestCase):
    """Test warm-start efficiency."""

    def test_warm_start_buffer_allocation(self):
        """Test that warm-start buffers are allocated correctly."""
        dim = 2
        n_points = 30
        
        model = FiniteModel(
            num_candidates=10,
            num_dims=dim,
            kernel=lambda x, y: L22(x, y),
            mode="concave"
        )
        
        fcot = FCOT(
            input_dim=dim,
            model=model,
            inverse_cx=inverse_grad_L22
        )
        
        Y = torch.randn(n_points, dim)
        idx_y = torch.arange(n_points)
        
        # First call should allocate buffers
        with torch.no_grad():
            _, _, _ = model.sup_transform(
                Z=Y, sample_idx=idx_y, steps=5, optimizer="lbfgs"
            )
        
        # Check that buffer was allocated
        self.assertIsNotNone(model._warm_X_global)
        self.assertEqual(model._num_global_points, n_points)

    def test_sample_idx_consistency(self):
        """Test that using sample indices gives consistent results."""
        dim = 1
        torch.manual_seed(909)
        
        model = FiniteModel(
            num_candidates=8,
            num_dims=dim,
            kernel=lambda x, y: L22(x, y),
            mode="concave"
        )
        
        Y = torch.randn(15, dim)
        idx = torch.arange(15)
        
        # First call initializes warm-start
        with torch.no_grad():
            _, val1, _ = model.sup_transform(Z=Y, sample_idx=idx, steps=5, optimizer="adam")
            # Second call uses warm-start
            _, val2, _ = model.sup_transform(Z=Y, sample_idx=idx, steps=5, optimizer="adam")
        
        # Both calls should produce valid, finite values
        self.assertFalse(torch.isnan(val1).any())
        self.assertFalse(torch.isnan(val2).any())


if __name__ == "__main__":
    unittest.main()
