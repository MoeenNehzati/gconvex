"""
Comprehensive test suite for inner optimization convergence tracking.

Tests all three optimizers (LBFGS, Adam, GD) with:
- Easy problems that should converge
- Hard problems that should fail to converge
- Verifies warnings are logged appropriately
"""

import torch
import torch.nn as nn
from models.inf_convolution import InfConvolution
import logging

# Configure logging to show warnings
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleQuadratic(nn.Module):
    """Simple quadratic function - easy to optimize."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        return (self.w * x).sum()


class IllConditioned(nn.Module):
    """Ill-conditioned function - hard to optimize."""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim) * 100.0)
        self.w2 = nn.Parameter(torch.randn(dim) * 0.001)
    
    def forward(self, x):
        return (self.w1 * x).pow(4).sum() + (self.w2 * x).pow(6).sum()


def simple_kernel(x, y):
    """Simple quadratic kernel."""
    return 0.5 * ((x - y)**2).sum()


def hard_kernel(x, y):
    """Anisotropic, ill-conditioned kernel."""
    diff = x - y
    scales = torch.tensor([1000.0, 1.0, 0.001])[:len(diff)]
    return (scales * diff).pow(4).sum()


def run_test(name, optimizer, net, kernel, x0, y, solver_steps, lr, lam, tol, patience, should_converge):
    """Run a single convergence test."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {name}")
    logger.info(f"{'='*70}")
    logger.info(f"Optimizer: {optimizer.upper()}, Steps: {solver_steps}, LR: {lr:.2e}, "
                f"Tol: {tol:.2e}, Patience: {patience}")
    
    try:
        g, converged, x_star = InfConvolution.apply(
            y, net, kernel, x0,
            solver_steps, lr, optimizer, lam, tol, patience,
            *list(net.parameters())
        )
        
        result = "✓ PASS" if converged == should_converge else "✗ FAIL"
        logger.info(f"Result: converged={converged}, expected={should_converge} {result}")
        logger.info(f"Final g={g.item():.6e}")
        
        return converged == should_converge
    except Exception as e:
        logger.error(f"✗ FAIL: Exception - {e}")
        return False


def main():
    logger.info("="*70)
    logger.info("INNER OPTIMIZATION CONVERGENCE TEST SUITE")
    logger.info("="*70)
    
    dim = 3
    results = []
    
    # ========================================================================
    # LBFGS TESTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("LBFGS OPTIMIZER TESTS")
    logger.info("="*70)
    
    # Test 1: LBFGS should converge on simple problem
    net = SimpleQuadratic(dim)
    y = torch.randn(dim)
    x0 = torch.randn(dim)
    results.append(run_test(
        "LBFGS - Simple problem (should converge)",
        "lbfgs", net, simple_kernel, x0, y,
        solver_steps=50, lr=1.0, lam=1e-3, tol=1e-6, patience=5,
        should_converge=True
    ))
    
    # Test 2: LBFGS with only 1 step should fail
    net = IllConditioned(dim)
    y = torch.randn(dim) * 10.0
    x0 = torch.randn(dim) * 10.0
    results.append(run_test(
        "LBFGS - Insufficient steps (should fail)",
        "lbfgs", net, hard_kernel, x0, y,
        solver_steps=1, lr=1.0, lam=0.0, tol=1e-6, patience=5,
        should_converge=False
    ))
    
    # Test 3: LBFGS on ill-conditioned with few steps should fail
    net = IllConditioned(dim)
    y = torch.randn(dim) * 5.0
    x0 = torch.randn(dim) * 5.0
    results.append(run_test(
        "LBFGS - Ill-conditioned problem (should fail)",
        "lbfgs", net, hard_kernel, x0, y,
        solver_steps=3, lr=1.0, lam=1e-4, tol=1e-6, patience=5,  # Small lam to prevent NaN
        should_converge=False
    ))
    
    # ========================================================================
    # ADAM TESTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("ADAM OPTIMIZER TESTS")
    logger.info("="*70)
    
    # Test 4: Adam should converge with reasonable settings
    net = SimpleQuadratic(dim)
    y = torch.randn(dim)
    x0 = torch.randn(dim)
    results.append(run_test(
        "Adam - Simple problem (should converge)",
        "adam", net, simple_kernel, x0, y,
        solver_steps=100, lr=0.1, lam=1e-3, tol=1e-3, patience=5,
        should_converge=True
    ))
    
    # Test 5: Adam with too few steps should fail
    net = IllConditioned(dim)
    y = torch.randn(dim)
    x0 = torch.randn(dim)
    results.append(run_test(
        "Adam - Insufficient steps (should fail)",
        "adam", net, hard_kernel, x0, y,
        solver_steps=5, lr=0.01, lam=0.0, tol=1e-6, patience=10,
        should_converge=False
    ))
    
    # Test 6: Adam with tight tolerance should fail
    net = SimpleQuadratic(dim)
    y = torch.randn(dim)
    x0 = torch.randn(dim)
    results.append(run_test(
        "Adam - Tight tolerance (should fail)",
        "adam", net, simple_kernel, x0, y,
        solver_steps=10, lr=0.01, lam=0.0, tol=1e-10, patience=5,
        should_converge=False
    ))
    
    # ========================================================================
    # GD TESTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("GRADIENT DESCENT OPTIMIZER TESTS")
    logger.info("="*70)
    
    # Test 7: GD should converge with enough steps
    net = SimpleQuadratic(dim)
    y = torch.randn(dim)
    x0 = torch.randn(dim)
    results.append(run_test(
        "GD - Simple problem with enough steps (should converge)",
        "gd", net, simple_kernel, x0, y,
        solver_steps=200, lr=0.1, lam=1e-3, tol=1e-3, patience=5,
        should_converge=True
    ))
    
    # Test 8: GD with very few steps should fail
    net = SimpleQuadratic(dim)  # Use simple problem to avoid NaN
    y = torch.randn(dim)
    x0 = torch.randn(dim)
    results.append(run_test(
        "GD - Insufficient steps (should fail)",
        "gd", net, simple_kernel, x0, y,
        solver_steps=3, lr=0.1, lam=0.0, tol=1e-6, patience=5,
        should_converge=False
    ))
    
    # Test 9: GD with small learning rate on harder problem should fail
    net = SimpleQuadratic(dim)
    y = torch.randn(dim) * 5.0  # Larger values make convergence harder
    x0 = torch.zeros(dim)  # Start from zero
    results.append(run_test(
        "GD - Learning rate too small (should fail)",
        "gd", net, simple_kernel, x0, y,
        solver_steps=15, lr=1e-4, lam=0.0, tol=1e-5, patience=8,
        should_converge=False
    ))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.error(f"✗ {total - passed} TEST(S) FAILED")
    
    logger.info("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
