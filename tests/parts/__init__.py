"""
Unit tests organized by component functionality.

This directory contains fast to medium-speed unit tests (0-10s per test)
that test individual components and their interactions.

Test modules:
- test_finite_model.py: Core FiniteModel unit tests
- test_finite_model_behavior.py: FiniteModel behavior tests
- test_finite_model_transforms.py: Transform batching and consistency
- test_fcot_core.py: Core FCOT unit tests
- test_fcot_optimization.py: FCOT optimization behavior
- test_fcot_convergence.py: FCOT convergence properties
- test_inf_convolution.py: InfConvolution autograd function
- test_utils.py: Utility function tests

To run only these unit tests:
    python -m unittest discover -s tests/parts
"""
