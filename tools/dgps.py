"""Data generators with caching for DGPS experiments, under `tools`.

Provides `generate_gaussian_pairs(...)` which creates paired (x,y) datasets
and caches them in a data directory (default `tmp/`). Call via
`from tools.dgps import generate_gaussian_pairs`.
"""
from __future__ import annotations
from torch.distributions.multivariate_normal import MultivariateNormal
import os
import json
import hashlib
from typing import Optional, Tuple, Union
import numpy as np
import torch


def _stable_hash(params) -> str:
    h = hashlib.sha256()
    for key in sorted(params.keys()):
        h.update(str(key).encode())
        v = params[key]
        if torch.is_tensor(v):
            h.update(v.detach().cpu().numpy().tobytes())
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
        else:
            h.update(str(v).encode())

    return h.hexdigest()

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def make_path(dir='tmp', **params):
    h = _stable_hash(params)
    fname = f'gaussian_pairs_n{params["n"]}_d{params["d"]}_{h}.npz'
    fpath = os.path.join(dir, fname)
    return fpath

def is_square(t):
    return t.ndim == 2 and t.shape[0] == t.shape[1]

def mean_cov_dims_match(μ, Σ):
    return (
        μ.ndim == 1 and
        Σ.ndim == 2 and
        Σ.shape[0] == Σ.shape[1] == μ.shape[0]
    )


def generate_gaussian_pairs(
    n,
    μ_x,
    Σ_x,
    μ_y,
    Σ_y,
    data_dir = 'tmp',
    force = False,
):
    """Generate paired Gaussian datasets with caching.

    Args:
      n: number of samples.
      d: dimensionality of each sample.
      mean_x: scalar or length-d sequence for X mean.
      scale_x: scalar or length-d sequence for X standard deviation.
      mean_y: scalar or length-d sequence for Y mean.
      scale_y: scalar or length-d sequence for Y standard deviation.
      data_dir: directory to save cached datasets (default `tmp`).
      seed: random seed for reproducibility. Different seeds produce different files.
      force: if True, regenerate and overwrite existing cached file.

    Returns:
      (x, y, path) where `x` and `y` are numpy arrays of shape `(n, d)` and
      `path` is the `.npz` file path used for caching.
    """
    _ensure_dir(data_dir)
    if (not torch.is_tensor(μ_x)) or (not torch.is_tensor(Σ_x)) or (not torch.is_tensor(μ_y)) or (not torch.is_tensor(Σ_y)):
        raise AssertionError("μ_x, Σ_x, μ_y, Σ_y should be tensors")
    if μ_x.shape != μ_y.shape:
        raise AssertionError("μ_x and μ_y must have the same shape")
    if Σ_x.shape != Σ_y.shape:
        raise AssertionError("Σ_x and Σ_y must have the same shape")
    if not is_square(Σ_x) or not is_square(Σ_y):
        raise AssertionError("Σ_x and Σ_y must be square tensors")
    if not mean_cov_dims_match(μ_x, Σ_x):
        raise AssertionError("dimension of μ_x and Σ_x must match")
    d = len(μ_x)
    params = {
        'func': 'generate_gaussian_pairs',
        'n': n,
        'd': d,
        'μ_x': μ_x,
        'Σ_x': Σ_x,
        'μ_y': μ_y,
        'Σ_y': Σ_y,
    }
    fpath = make_path(data_dir, **params)
    if os.path.exists(fpath) and not force:
        data = torch.load(fpath)
        return data['x'], data['y'], fpath
    
    x = MultivariateNormal(μ_x, Σ_x).sample((n,))
    y = MultivariateNormal(μ_y, Σ_y).sample((n,))
    torch.save({'x': x, 'y': y, 'params': params}, fpath)
    return x, y, fpath
