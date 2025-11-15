"""
Portable wrapper that adapts a simple external map trainer to a GNOT-style API.

If the full GNOT repo training pipeline is desired, replace the implementation
of `BaselineGNOTExternal` here with calls into the cloned `baselines/external/GNOT` code.

For now this file re-uses the MMD-based MapNet implementation from `fan_map`
so that `baselines.gnot` can offer a working method out-of-the-box for
downstream analysis.
"""
from typing import Optional, Dict

try:
    from baselines.external.fan_map import BaselineFanExternal  # type: ignore
except Exception:
    BaselineFanExternal = None

import torch


class BaselineGNOTExternal:
    """Adapter that uses `BaselineFanExternal` as a fallback working implementation.

    Replace internals with calls into the full GNOT code for a proper integration.
    """

    def __init__(self, dim: int, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if BaselineFanExternal is None:
            raise ImportError("Required external fan_map implementation not found.")
        self._inner = BaselineFanExternal(dim=dim, device=self.device)

    def train(self, source: torch.Tensor, target: torch.Tensor, **kwargs):
        return self._inner.train(source, target, **kwargs)

    def compute_mechanism(self, sample: torch.Tensor) -> Dict:
        return self._inner.compute_mechanism(sample)
