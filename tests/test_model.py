import unittest
import torch
import random
from models import FiniteModel


#######################################################################
# =====================================================================
#  ORIGINAL TEST SUITE (UNCHANGED)
# =====================================================================
#######################################################################

class TestFiniteModelInitialization(unittest.TestCase):
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


class TestFiniteModelForward(unittest.TestCase):
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


class TestFiniteModelMetricProperties(unittest.TestCase):
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

        _, v = model.inf_transform(Z, steps=200, lr=5e-3, optimizer="adam")
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

        _, v = model.sup_transform(Z, steps=200, lr=5e-3, optimizer="adam")
        self.assertTrue(torch.isfinite(v).all())
        self.assertLess(abs(v.item() + fz.item()), 0.1)

    def test_quadratic_surplus_conjugate_symmetry(self):
        surplus = lambda x, y: -0.5*((x - y)**2).sum(-1)
        model = FiniteModel(10, 2, surplus, mode="convex")
        X = torch.randn(5, 2)

        _, fx = model.forward(X)
        _, fcx = model.sup_transform(X, steps=50, lr=1e-2, optimizer="adam")

        corr = torch.corrcoef(torch.stack([fx, fcx]))[0,1]
        self.assertGreater(abs(corr), 0.3)

    def test_inf_sup_are_finite(self):
        kernel = lambda x, y: -((x - y)**2).sum(-1)
        model = FiniteModel(8, 2, kernel)
        Z = torch.randn(3, 2)

        _, infv = model.inf_transform(Z)
        _, supv = model.sup_transform(Z)

        self.assertTrue(torch.isfinite(infv).all())
        self.assertTrue(torch.isfinite(supv).all())


class TestFiniteModelOptimization(unittest.TestCase):
    def test_lbfgs(self):
        k = lambda x, y: -((x - y)**2).sum(-1)
        model = FiniteModel(5, 2, k)
        Z = torch.randn(4,2)
        Xs, vals = model.sup_transform(Z, steps=20, lr=1e-1, optimizer="lbfgs")
        self.assertEqual(Xs.shape, (4,2))
        self.assertTrue(torch.isfinite(vals).all())

    def test_adam(self):
        k = lambda x, y: (x * y).sum(-1)
        model = FiniteModel(5, 2, k)
        Z = torch.randn(4,2)
        Xs, vals = model.inf_transform(Z, steps=100, lr=1e-2, optimizer="adam")
        self.assertEqual(Xs.shape, (4,2))
        self.assertTrue(torch.isfinite(vals).all())

    def test_gd(self):
        k = lambda x, y: -((x - y)**2).sum(-1)
        model = FiniteModel(5,2,k,mode="concave")
        Z = torch.randn(3,2)
        Xs, vals = model.inf_transform(Z, steps=100, lr=1e-2, optimizer="gd")
        self.assertEqual(Xs.shape, (3,2))


#######################################################################
# =====================================================================
#  NEW TESTS (UPDATED FOR NEW WARM-START SYSTEM)
# =====================================================================
#######################################################################

class TestFiniteModelWarmStartBehavior(unittest.TestCase):

    def test_warm_start_reuse_same_batch(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(10, 2, k)

        Z = torch.randn(32,2)

        # cold
        _, v1 = model.sup_transform(Z, steps=5, optimizer="adam", lr=5e-3)

        self.assertTrue(hasattr(model, "_warm_X_fallback"))
        self.assertEqual(model._warm_X_fallback.shape, (32,2))

        # warm
        _, v2 = model.sup_transform(Z, steps=3, optimizer="adam", lr=5e-3)
        self.assertTrue(torch.allclose(v1, v2, atol=5e-2, rtol=1e-3))


    def test_warm_start_resets_if_batch_size_changes(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(5,2,k)

        Z1 = torch.randn(20,2)
        model.sup_transform(Z1)
        self.assertEqual(model._warm_X_fallback.shape,(20,2))

        Z2 = torch.randn(10,2)
        model.sup_transform(Z2)
        self.assertEqual(model._warm_X_fallback.shape,(10,2))


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

        _, v_small = model.sup_transform(Z_small, sample_idx=idx_subset)

        self.assertTrue(torch.isfinite(v_small).all())
        self.assertEqual(model._warm_X_global.shape,(30,2))
        self.assertEqual(model._warm_X_global[idx_subset].shape,(15,2))


#######################################################################
# Stability / batch consistency / transform tests
#######################################################################

class TestFiniteModelStability(unittest.TestCase):

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


class TestFiniteModelBatchConsistency(unittest.TestCase):
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

        _, inf_full = model.inf_transform(Z, steps=40, lr=5e-3, optimizer="adam")

        vals = []
        for i in range(0,32,8):
            _, v = model.inf_transform(Z[i:i+8], steps=40, lr=5e-3, optimizer="adam")
            vals.append(v)

        inf_batched = torch.cat(vals)
        self.assertTrue(torch.isfinite(inf_batched).all())
        self.assertTrue(torch.allclose(inf_batched, inf_full, atol=0.15))


class TestFiniteModelConvexityConcavity(unittest.TestCase):
    def test_convex_piecewise_linear(self):
        k = lambda x,y: (x*y).sum(-1)
        model = FiniteModel(5,1,k,mode="convex")

        X = torch.linspace(-2,2,100).unsqueeze(-1)
        _, f = model.forward(X)

        sec = f[2:] - 2*f[1:-1] + f[:-2]
        self.assertTrue((sec >= -1e-4).all())

    def test_concave_piecewise_linear(self):
        k = lambda x,y: (x*y).sum(-1)
        model = FiniteModel(5,1,k,mode="concave")

        X = torch.linspace(-2,2,100).unsqueeze(-1)
        _, f = model.forward(X)

        sec = f[2:] - 2*f[1:-1] + f[:-2]
        self.assertTrue((sec <= 1e-4).all())


class TestFiniteModelTransformImprovement(unittest.TestCase):
    def test_sup_transform_improves(self):
        torch.manual_seed(0)
        k = lambda x,y: -((x-y)**2).sum(-1)
        model = FiniteModel(6,2,k)

        Z = torch.randn(12,2)

        _, v5 = model.sup_transform(Z, steps=5, lr=1e-2, optimizer="adam")
        _, v50 = model.sup_transform(Z, steps=50, lr=1e-2, optimizer="adam")

        self.assertTrue((v50 >= v5 - 1e-5).all())


#######################################################################
# END FULL SUITE
#######################################################################


if __name__ == "__main__":
    unittest.main()
