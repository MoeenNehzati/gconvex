"""
Models Module

This module contains neural network models for representing finitely convex/concave
functions and computing infimal convolutions with implicit differentiation.

Classes:
    - FiniteModel: Unified finite representation for convex/concave functions
    - InfConvolution: Differentiable infimal convolution operation
"""

from .finite_model import FiniteModel
from .inf_convolution import InfConvolution

__all__ = ["FiniteModel", "InfConvolution"]
