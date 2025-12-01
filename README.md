gconvex: Finitely Convex Models and OT Solvers
==============================================

This repository implements finitely convex/concave parametrizations on grids, plus optimal-transport solvers and mechanism-design experiments used in the accompanying research.

Layout
------
- `models/finite_model.py` – grid-based models:
  - `FiniteSeparableModel` for separable kernels with exact grid transforms and optional coarse-to-fine search.
  - `FiniteModel` for generic finite convex/concave potentials.
- `mech_design/` – mechanism-design wrappers around `FiniteModel` (e.g. `Mechanism`, `Trainer`) plus training/checkpointing helpers.
- `optimal_transport/ot_fc_sep_map.py` – separable OT solver (`FCOTSeparable`) built on `FiniteSeparableModel`.
- `optimal_transport/ot_icnn_map.py` – ICNN-based OT baseline (`ICNNOT`) with alternating f/g updates.
- `tools/` – utilities (kernels, loaders, NaN hooks), synthetic data generators (`dgps.py`), visualization helpers (`visualize.py`), logging (`feedback.py`), and selection ops (`utils.py`).
- `scripts/` – runnable experiments:
  - `solve_mechanisms.py` sweeps mechanism-design configs across dimensions/kernels.
  - `run_finite_ot.py` runs FCOT-Separable on DGPS data with plotting hooks.
  - `compare_baselines.py` compares OT baselines on Gaussian pairs.
  - `plot_training_log.py` renders training curves from log files.
  - `plot_monge_map.py` visualizes learned Monge maps / transport plans from OT runs.
- `tests/` – regression/consistency checks: `test_separable.py` (separable model), `test_mech1d.py` (mechanism tests).
- `config.py` – default `WRITING_ROOT` for artifacts (default `tmp/`).

Install
-------
```bash
pip install -e .            # core
pip install -e ".[dev]"     # + tests/linters
# optional wandb logging
pip install -e ".[wandb]"
```

Quick usage
-----------
Finite separable model forward:
```python
import torch
from models import FiniteSeparableModel

def quadratic_kernel(x, y):  # 1D kernel
    return -(x - y)**2

model = FiniteSeparableModel(
    kernel=quadratic_kernel,
    num_dims=2,
    radius=1.0,
    y_accuracy=0.1,
    x_accuracy=0.1,
    mode="convex",
    temp=10.0,
)
X = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float32)
choice, values = model.forward(X, selection_mode="soft")
```

FCOT-Separable solver on synthetic Gaussian pairs:
```python
from optimal_transport.ot_fc_sep_map import FCOTSeparable
from tools.dgps import generate_gaussian_pairs
from tools.utils import L22_1d, inverse_L22x

X, Y, _ = generate_gaussian_pairs(n=1000)
solver = FCOTSeparable.initialize_right_architecture(
    dim=X.shape[1],
    radius=4.0,
    n_params=200,          # must be divisible by dim
    x_accuracy=0.05,
    kernel_1d=L22_1d,
    inverse_kx=inverse_L22x,
    cache_gradients=True,
)
solver.fit(X, Y, iters=500, print_every=50)
```

Running scripts
---------------
- Mechanism sweep (writes under `WRITING_ROOT/mech/`):
  ```bash
  python scripts/solve_mechanisms.py --niters 20000 --batch-size 512
  ```
- FCOT-Separable DGPS experiment:
  ```bash
  python scripts/run_finite_ot.py
  ```
- OT baseline comparison on Gaussian pairs:
  ```bash
  python scripts/compare_baselines.py --iters 2000 --nparams 5000
  ```
- Plot training logs:
  ```bash
  python scripts/plot_training_log.py --log training.log
  ```

Testing
-------
To run the main unit/regression test suite (excluding slow manual tests), use:

```bash
pytest tests/ -n auto
```

Manual end-to-end tests (e.g. Gaussian OT checks) live under `tests/manual/` and
are excluded from default runs. To execute them explicitly:

```bash
pytest tests/manual/ -v -s
```

Artifacts
---------
Most scripts write checkpoints/plots under paths prefixed by `WRITING_ROOT` (see `config.py`). Set it to a desired location before running experiments. `scripts/compare_baselines.py --out ...` and `scripts/solve_mechanisms.py --clear` control output directories and cleanup.

Reproducing paper results
-------------------------
The main experiments and figures from the paper can be regenerated with:

- Mechanism-design experiments:
  ```bash
  python -m scripts.solve_mechanisms
  ```
- Optimal transport experiments:
  ```bash
  python -m scripts.run_finite_ot
  ```
- Monge map / visualization plots (from the OT runs):
  ```bash
  python -m scripts.plot_monge_map
  ```
