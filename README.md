Implements the Class of Finitely Convex For Optimizing Over Generalized Convex Functions
=================================================

Overview
--------
This repository contains code used for the experiments reported in the accompanying paper "Universal Representation of Generalized Convex Functions and their Gradients". The code implements the finitely convex parametrization of generalized convex functions. Additionally, it takes in constraints and solves for the optimal mechanism for selling multiple items to a single buyer.


Repository layout
-----------------
- `optim.py` — model, selection helpers, Trainer (checkpointing, wandb integration, penalties, switch logic).
- `runner.py` — example driver that generates or loads samples and runs experiments specified in config.py.
- `config.py` — settings for experiments in the paper.
- `visualize.py` — creates plots in the paper from the results produced by runner.py.
- `utilities.py` — helper utilities: kernels, constraints, directory name generation, data loaders etc.
- `feedback.py` _ includes scripts for custom logging and update panel during training

Quick start
-----------
1. Install dependencies:

```bash
pip install -e .                    # Install package
pip install -e ".[dev]"             # Install with dev dependencies (includes pytest)
# optional: wandb if you want experiment tracking
pip install -e ".[wandb]"
```

2. Configure experiments in `config.py`

3. Run the example runner (creates `snapshots/…` directories and saves artifacts):

```bash
python runner.py
```

4. After a run finishes, create visualizations and plots by `visualize.py`

```bash
python visualize.py
```

or inspect the results by loading the saved snapshots by `utilities.loader()`.

API snippets
------------
Under the hood the simple entry point is `optim.run(...)` which returns `(mechanism, mechanism_data)`. mechanism is the fitted model and mechanism data is a dictionary containing its performance.

Example usage from a notebook:

```python
import torch
import optim

# sample: torch.Tensor with shape (N, d)
sample = torch.rand(2000, 2)
mechanism, data = optim.run(
    sample,
    modes=["soft", "ste"],
    compile=False,
    model_kwargs={"npoints":50, "y_dim":2},
    train_kwargs={"nsteps":1000, "writing_dir":"snapshots/"},
    optimizers_kwargs_dict={"soft":{"lr":1e-2}, "ste":{"lr":1e-3}},
    schedulers_kwargs_dict={},
    with_hook=False,
)
```

Key shapes / conventions
- `sample`: Tensor (S, n) where S is the sample size and n is the number of goods(dimension of Y)
- Model `compute_mechanism(sample)` returns a dict with entries used by logging and evaluation: `choice`, `v`, `profits`, `kernel`, `y`, `intercept`.
- Checkpoints are saved as `{writing_dir}/{run_id}_epoch_{epoch}.pt` and the final artifact is `{writing_dir}/{run_id}_final.pt`.

Testing
-------
The test suite is organized into unit tests (`tests/parts/`) and integration tests (`tests/integration/`).

Install test dependencies:
```bash
pip install pytest pytest-xdist pytest-timeout
# OR install with the package: pip install -e ".[dev]"
```

Run all tests in parallel:
```bash
pytest tests/ -n auto                # Parallel (recommended, auto-detect CPUs)
pytest tests/ -n 4                   # Parallel with 4 workers
pytest tests/                        # Sequential

# If using a virtual environment and pytest isn't finding modules:
python -m pytest tests/ -n auto      # Use Python from current environment
```

Run specific test groups:
```bash
pytest tests/parts/ -n auto          # Only unit tests (fast)
pytest tests/integration/            # Only integration tests (slow)
pytest tests/parts/test_finite_model.py -v  # Specific module
```

Alternatively, use unittest:
```bash
python -m unittest discover -s tests/parts  # Fast unit tests only
```

See `tests/PERFORMANCE.md` for more details on test optimization and parallel execution.

Checkpoints and resuming
------------------------
- `Trainer` automatically looks for `*final*` and recent `*epoch_*.pt` files in `writing_dir` and will resume if found.
- Run IDs are encoded into the prefix of checkpoint filenames; resumed runs are given an `R` suffix by the loader to keep traces distinct.

Logging and wandb
-----------------
- WandB integration is optional. Enable by setting `use_wandb=True` in `TRAIN_KWARGS` or passing `train_kwargs={"use_wandb": True}` to `optim.run`.
- Calls to wandb are guarded so that failed wandb logging will not crash training.

Notes for reproducibility
-------------------------
- Seeds: `runner.py` seeds torch for reproducibility. For full reproducibility fix NumPy and Python RNG seeds if needed.

Citing the paper
----------------
If you reuse this code in published work, please cite the accompanying paper.

License
-------
This repository is licensed under Apache 2.0.



https://github.com/AmirTag/OT-ICNN


Optimal transport mapping via input convex neural networks — Ashok V. Makkuva, Amirhossein Taghvaei, Sewoong Oh & Jason D. Lee. ICML (Proceedings of Machine Learning Research), 2020. 
Proceedings of Machine Learning Research
+1

Focus: learning OT maps under quadratic cost using input-convex neural networks (ICNNs).

Why include: baseline method for neural OT map learning under classical cost.

Scalable Computation of Monge Maps with General Costs — Jiaojiao Fan, Shu Liu, Shaojun Ma, Yongxin Chen & Haomin Zhou. ICLR Workshop / arXiv, 2021/2022. 
OpenReview
+1

Focus: computing Monge maps for general cost functions (beyond quadratic) in a scalable fashion.

Why include: one of the few works explicitly addressing non-quadratic costs.

Neural Optimal Transport with General Cost Functionals — Arip Asadulaev, Alexander Korotin, Vage Egiazarian, Petr Mokrov & Evgeny Burnaev. ICLR (or arXiv submission) 2022/2024. 
arXiv
+1

Focus: neural network-based OT for general cost functionals (including class‐guided, pair‐guided) with theoretical error analysis.

Why include: the most directly related work if you’re working on non-Euclidean costs; this paper claims “general cost” setting.

External optimal_transport — clone commands
----------------------------------
If you want to pull external baseline repositories into this project under
`optimal_transport/external/` (so adapters in `optimal_transport/external` can use them), run
the following commands from the repository root:

```bash
# Clone OT-ICNN (Input-Convex Neural Networks for OT maps)
git clone https://github.com/AmirTag/OT-ICNN optimal_transport/external/OT-ICNN

# Clone GNOT (Neural OT with general cost functionals)
git clone https://github.com/machinestein/GNOT optimal_transport/external/GNOT

# (Optional) If you find an official Fan et al. implementation, clone similarly
# git clone <fan-repo-url> optimal_transport/external/FAN
```

Notes:
- Cloning these projects may require additional dependencies (e.g. PyTorch).
- We recommend using a virtualenv and installing pinned versions from
    `setup.py` or manually, for example:

```bash
python -m pip install torch==1.9.0 torchvision==0.10.0
```

```