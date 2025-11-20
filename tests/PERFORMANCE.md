# Test Performance Optimization Guide

## Current Performance

After optimization, the test suite runs significantly faster:
- **Total tests**: 126
- **Sequential time**: ~70-90 seconds
- **Parallel time (4 workers)**: ~20-30 seconds
- **Speedup from original**: ~4-5x
- **Speedup with parallelization**: ~3-4x additional

## Quick Start

Install test dependencies:
```bash
pip install pytest pytest-xdist pytest-timeout
```

Or install with the package:
```bash
pip install -e ".[dev]"
```

Run tests in parallel:
```bash
pytest tests/ -n auto           # Recommended: auto-detect CPUs
pytest tests/ -n 4              # Use 4 workers
pytest tests/parts/ -n auto     # Only unit tests

# If using a virtual environment:
python -m pytest tests/ -n auto # Ensures correct Python is used
```

## Test Timing Breakdown

### Fast Tests (<1s)
Most unit tests in `tests/parts/` are fast and test individual components.

### Medium Tests (1-5s)
- `TestNumericalStability` tests
- `TestFiniteModelBatchConsistency` tests
- `TestFiniteModelTransform` tests
- `TestEdgeCases.test_high_dimensional`

### Slow Tests (5-20s)
Integration tests in `tests/integration/`:
- `TestInvariance.test_translation_invariance_of_cost` (~16s)
- `TestClosedFormOT.test_1d_gaussian_transport` (~10s)
- `TestDualConstraints.test_dual_feasibility_constraint` (~5-7s)
- `TestMongeMap.test_identity_transport_approximately` (~3-5s)
- `TestStochasticRobustness` tests (~5s each)

## Parallel Test Execution

### Using pytest-xdist (Recommended)

pytest-xdist enables parallel test execution.

Install:
```bash
pip install pytest pytest-xdist pytest-timeout
```

Or install with the package:
```bash
pip install -e ".[dev]"
```

Run all tests in parallel:
```bash
pytest tests/ -n auto                # Auto-detect CPU cores (recommended)
pytest tests/ -n 4                   # Use 4 workers
pytest tests/parts/ -n auto          # Only unit tests in parallel
pytest tests/integration/ -n 2       # Integration tests (fewer workers)
pytest tests/                        # Sequential (no -n flag)

# If using a virtual environment and getting import errors:
python -m pytest tests/ -n auto      # Use Python from current environment
```

Additional useful options:
```bash
pytest tests/ -n auto -v             # Verbose output
pytest tests/ -n auto --durations=10 # Show 10 slowest tests
pytest tests/ -n auto -k "test_finite"  # Run tests matching pattern
pytest tests/ -n auto --lf           # Run last failed tests
```

### Configuration

The test suite is configured via `pyproject.toml`:
- Timeout: 60 seconds per test (global safety limit)
- Output format: Short tracebacks, summary of all outcomes
- Test discovery: Automatic for test_*.py files

### Alternative: unittest (no parallelization)

For compatibility or debugging, you can still use unittest:
```bash
python -m unittest discover -s tests        # All tests
python -m unittest discover -s tests/parts  # Only unit tests
```

## Optimization Techniques Used

### 1. Reduced Training Iterations
- Original: 200-400 iterations
- Optimized: 15-30 iterations
- Tests still validate correctness while running faster

### 2. Reduced Sample Sizes
- Original: 100-200 samples
- Optimized: 20-50 samples
- Smaller datasets converge faster

### 3. Reduced Inner Optimization Steps
- Original: 15-50 steps
- Optimized: 3-10 steps
- Sufficient for test validation

### 4. Faster Optimizers
- Changed from LBFGS → Adam where appropriate
- LBFGS is more accurate but slower
- Adam is faster for test validation

### 5. Increased Learning Rates
- Higher learning rates (1e-3 → 5e-3) for faster convergence
- Still within stable range for tests

### 6. Test Organization
- Separated slow integration tests
- Can skip integration tests during rapid development:
  ```bash
  python -m unittest discover -s tests/parts
  ```

## Further Optimization Ideas

### 1. Conditional Skip for CI
```python
import os
import unittest

@unittest.skipIf(os.getenv("FAST_TESTS"), "Skipping slow test in fast mode")
class TestSlowIntegration(TimedTestCase):
    ...
```

Usage:
```bash
FAST_TESTS=1 python -m unittest discover -s tests
```

### 2. Test Fixtures/Caching
Cache expensive setups:
```python
class TestWithCache(TimedTestCase):
    @classmethod
    def setUpClass(cls):
        # Expensive setup once per class
        cls.model = create_expensive_model()
```

### 3. Reduce Assertion Precision
For floating-point comparisons, use looser tolerances in tests:
```python
self.assertAlmostEqual(a, b, places=4)  # Instead of places=6
torch.allclose(a, b, atol=1e-3)         # Instead of atol=1e-6
```

## Recommended Workflow

### During Development (Fast Iteration)
```bash
# Run only unit tests sequentially
python -m unittest discover -s tests/parts

# Or run specific test modules
python -m unittest tests.parts.test_finite_model

# Or use pytest for specific tests
pytest tests/parts/test_finite_model.py -v
```

### Pre-Commit (Full Validation)
```bash
# Run all tests in parallel (recommended)
pytest tests/ -n auto

# With verbose output to see what's running
pytest tests/ -n auto -v
```

### CI/CD Pipeline
```bash
# Run all tests with maximum parallelism and coverage
pytest tests/ -n auto --cov=models --cov=baselines --cov=tools
```

## Monitoring Test Performance

The `TimedTestCase` base class automatically tracks and reports:
- Individual test times
- Top 20 slowest tests  
- Total test time

View timing summary:
```bash
python -m unittest discover -s tests 2>&1 | grep -A 25 "Test Timing Summary"
```

With pytest, use duration reports:
```bash
pytest tests/ --durations=20        # Show 20 slowest tests
pytest tests/ --durations=0         # Show all test durations
```
