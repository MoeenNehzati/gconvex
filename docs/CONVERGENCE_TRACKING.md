# Inner Optimization Convergence Tracking

## Summary

Implemented comprehensive convergence detection and logging for all three inner optimizers (LBFGS, Adam, GD) to ensure you always know when inner optimization fails.

## Key Changes

### 1. Patience Mechanism (Adam/GD)
- **Before**: Single step below tolerance → converged
- **After**: Requires `patience` consecutive steps below tolerance
- **Default**: `patience=5`
- **Benefit**: Prevents premature convergence from temporary slow steps

### 2. LBFGS Convergence Detection
- Checks both iteration count AND gradient norm
- Converged only if: stopped early AND gradient < tolerance
- Two types of warnings:
  - Used all iterations → need more steps
  - Stopped early but gradient large → ill-conditioned problem

### 3. NaN/Inf Detection
- All optimizers check for numerical issues
- Raises `RuntimeError` with diagnostic info
- Suggests fixes (reduce lr, increase lam)

### 4. Training-Level Tracking
- FCOT tracks inner convergence throughout training
- Logs summary at end showing failure rate
- New log keys: `inner_converged`, `inner_failures`
- CRITICAL warning if failure rate > 10%

### 5. Error Handling Option
- New parameter: `raise_on_inner_divergence` (default=False)
- When True: stops training on convergence failure
- When False: logs warning and continues

## API Changes

### New Parameters

**`InfConvolution.forward()`:**
- `patience` (int, default=5): Consecutive steps needed for convergence

**`FCOT.__init__()` and `FCOT.initialize_right_architecture()`:**
- `inner_patience` (int, default=5): Patience for inner solver
- `raise_on_inner_divergence` (bool, default=False): Raise errors vs warnings

## Usage

```python
from baselines.ot_fc_map import FCOT
from tools.utils import L22, inverse_grad_L22

# Default: warnings only
fcot = FCOT.initialize_right_architecture(
    dim=2,
    n_params_target=1000,
    cost=L22,
    inverse_cx=inverse_grad_L22,
    inner_optimizer="adam",
    inner_steps=50,
    inner_patience=5,  # Require 5 consecutive converged steps
    raise_on_inner_divergence=False  # Just warn, don't stop
)

logs = fcot.fit(X, Y, iters=1000)

# Check convergence
print(f"Inner failures: {logs['inner_failures']} / 1000")
print(f"Failure rate: {logs['inner_failures']/10:.1f}%")
```

## Testing

Run comprehensive test suite:
```bash
python -m tests.test_convergence
```

Tests cover:
- **LBFGS**: Success on simple problems, failure with insufficient steps
- **Adam**: Success with adequate settings, failure with tight tolerance
- **GD**: Success with enough steps, failure with small learning rate

All 9 tests pass, verifying correct convergence detection across all optimizers.

## Warning Examples

**LBFGS - Insufficient iterations:**
```
[InfConvolution] LBFGS did not converge after 1 iterations (27 function evaluations).
Final objective: 6.04e+13, grad_norm: 1.67e+14, tol: 1.00e-06.
Consider increasing solver_steps (current: 1).
```

**Adam - Failed to converge:**
```
[InfConvolution] ADAM did not converge after 5 steps.
Final objective: 9.43e+03, final rel_change: 2.05e-02, tol: 1.00e-06, patience: 0/10.
Consider increasing solver_steps or relaxing tol.
```

**Training Summary (FCOT):**
```
======================================================================
INNER OPTIMIZATION CONVERGENCE SUMMARY
======================================================================
Optimizer: ADAM
Total iterations: 1000
Inner convergence failures: 234 (23.40%)
Inner steps per iteration: 50
Inner learning rate: 1.00e-02
Inner tolerance: 1.00e-03
Inner lambda (regularization): 1.00e-03

RECOMMENDATION: CRITICAL - High failure rate!
  - Current failures compromise 23.4% of training steps
  - Consider: increasing inner_steps from 50 to 100+
  - Or: relaxing inner_tol from 1.00e-03 to 1.00e-02
  - Or: trying different inner_optimizer (current: adam)
======================================================================
```

## Files Modified

- `models/inf_convolution.py`: Core convergence detection
- `models/finite_model.py`: Pass patience parameter
- `baselines/ot_fc_map.py`: Training-level tracking and summaries
- `tests/test_convergence.py`: Comprehensive test suite

## Backward Compatibility

All changes are backward compatible:
- New parameters have sensible defaults
- Existing code sees improved warnings
- No breaking API changes
