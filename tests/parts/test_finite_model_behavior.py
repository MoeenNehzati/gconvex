"""
Behavior tests for FiniteModel class.

Tests behavioral aspects of FiniteModel including:
- Optimization convergence
- Warm-start efficiency
- Numerical stability
- Batch consistency
- Transform improvement

These tests involve some optimization and are slower (1-5s each).
"""

import unittest
import torch
import random
from models import FiniteModel
from tests import TimedTestCase


class TestFiniteModelOptimization(TimedTestCase):
    def test_lbfgs(self):
        k = lambda x, y: -((x - y)**2).sum(-1)
        model = FiniteModel(5, 2, k)
        Z = torch.randn(4,2)
        Xs, vals, _ = model.sup_transform(Z, steps=20, lr=1e-1, optimizer="lbfgs")
        self.assertEqual(Xs.shape, (4,2))
        self.assertTrue(torch.isfinite(vals).all())

    def test_adam(self):
        k = lambda x, y: (x * y).sum(-1)
        model = FiniteModel(5, 2, k)
        Z = torch.randn(4,2)
        Xs, vals, _ = model.inf_transform(Z, steps=100, lr=1e-2, optimizer="adam")
        self.assertEqual(Xs.shape, (4,2))
        self.assertTrue(torch.isfinite(vals).all())

    def test_gd(self):
        k = lambda x, y: -((x - y)**2).sum(-1)
        model = FiniteModel(5,2,k,mode="concave")
        Z = torch.randn(3,2)
        Xs, vals, _ = model.inf_transform(Z, steps=100, lr=1e-2, optimizer="gd")
        self.assertEqual(Xs.shape, (3,2))


class TestFiniteModelWarmStartBehavior(TimedTestCase):

    def test_warm_start_reuse_same_batch(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(10, 2, k)

        Z = torch.randn(32,2)
        sample_idx = torch.arange(32)

        # First call - cold start
        X1, v1, _ = model.sup_transform(Z, sample_idx=sample_idx, steps=30, optimizer="adam", lr=5e-3)

        # Verify warm start buffer was created and populated
        self.assertTrue(hasattr(model, "_warm_X_global"))
        self.assertEqual(model._warm_X_global.shape, (32,2))
        
        # Store the warm start values
        warm_start_after_first = model._warm_X_global.clone()

        # Second call - should use warm start
        X2, v2, _ = model.sup_transform(Z, sample_idx=sample_idx, steps=30, optimizer="adam", lr=5e-3)
        
        # Verify that X2 started from the warm start (X1)
        # Since we're starting from X1 and optimizing further, X2 should be different from X1
        # but the warm start buffer should have been used (which we verify implicitly by
        # checking that the optimization produces reasonable values)
        
        # The warm start should have been updated
        warm_start_after_second = model._warm_X_global.clone()
        
        # Warm starts should be different (optimization continued)
        self.assertFalse(torch.allclose(warm_start_after_first, warm_start_after_second, atol=1e-6))
        
        # Both values should be finite and reasonable
        self.assertTrue(torch.all(torch.isfinite(v1)))
        self.assertTrue(torch.all(torch.isfinite(v2)))


    def test_warm_start_resets_if_batch_size_changes(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(5,2,k)

        Z1 = torch.randn(20,2)
        sample_idx1 = torch.arange(20)
        model.sup_transform(Z1, sample_idx=sample_idx1)
        self.assertEqual(model._warm_X_global.shape,(20,2))

        Z2 = torch.randn(10,2)
        sample_idx2 = torch.arange(30, 40)  # Different indices
        model.sup_transform(Z2, sample_idx=sample_idx2)
        # Buffer should expand to accommodate new indices
        self.assertEqual(model._warm_X_global.shape,(40,2))


    def test_warm_start_does_not_leak_between_models(self):
        k = lambda x,y: -((x-y)**2).sum(-1)
        m1 = FiniteModel(4,2,k)
        m2 = FiniteModel(4,2,k)

        Z = torch.randn(12,2)

        m1.sup_transform(Z)

        # m2 should be clean
        self.assertFalse(hasattr(m2, "_warm_X_fallback"))
        self.assertIsNone(m2._warm_X_global)


    def test_warm_start_subset_batching(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(6,2,k)

        Z = torch.randn(30,2)

        idx_full = torch.arange(0,30)
        model.sup_transform(Z, sample_idx=idx_full)

        self.assertEqual(model._warm_X_global.shape,(30,2))

        idx_subset = torch.tensor(random.sample(range(30),15))
        Z_small = Z[idx_subset]

        _, v_small, _ = model.sup_transform(Z_small, sample_idx=idx_subset)

        self.assertTrue(torch.isfinite(v_small).all())
        self.assertEqual(model._warm_X_global.shape,(30,2))
        self.assertEqual(model._warm_X_global[idx_subset].shape,(15,2))


class TestFiniteModelStability(TimedTestCase):

    def test_softmin_large_values(self):
        k = lambda x,y: (x*y).sum(-1)
        m = FiniteModel(5,3,k,mode="concave")
        X = torch.randn(8,3)*1000
        c, fx = m.forward(X,"soft")
        self.assertTrue(torch.isfinite(fx).all())
        self.assertTrue(torch.isfinite(c).all())

    def test_temp_scaling(self):
        k = lambda x,y: -((x-y)**2).sum(-1)
        m = FiniteModel(7,2,k,temp=1e6)
        X = torch.randn(10,2)
        _, fx = m.forward(X,"soft")
        self.assertTrue(torch.isfinite(fx).all())


class TestFiniteModelBatchConsistency(TimedTestCase):
    def test_forward_batch_consistency(self):
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(10,2,k)

        X = torch.randn(40,2)
        X1 = X[:20]
        X2 = X[20:]

        _, f_full = model.forward(X)
        _, f1 = model.forward(X1)
        _, f2 = model.forward(X2)

        torch.testing.assert_close(f_full, torch.cat([f1,f2]))

    def test_inf_sup_batched_vs_full(self):
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(8,2,k)

        Z = torch.randn(32,2)

        _, inf_full, _ = model.inf_transform(Z, steps=40, lr=5e-3, optimizer="adam")

        vals = []
        for i in range(0,32,8):
            _, v, _ = model.inf_transform(Z[i:i+8], steps=40, lr=5e-3, optimizer="adam")
            vals.append(v)

        inf_batched = torch.cat(vals)
        self.assertTrue(torch.isfinite(inf_batched).all())
        self.assertTrue(torch.allclose(inf_batched, inf_full, atol=0.15))


class TestFiniteModelTransformImprovement(TimedTestCase):
    def test_sup_transform_improves(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(6,2,k)

        Z = torch.randn(12,2)

        _, v5, _ = model.sup_transform(Z, steps=5, lr=1e-2, optimizer="adam")
        _, v50, _ = model.sup_transform(Z, steps=50, lr=1e-2, optimizer="adam")

        self.assertTrue((v50 >= v5 - 1e-5).all())


if __name__ == "__main__":
    unittest.main()
