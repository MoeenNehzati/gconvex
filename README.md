Mechanism-design experiments — code and reproduction
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
1. Install dependencies (typical):

<!-- ```bash
python -m pip install torch matplotlib numpy
# optional: wandb if you want experiment tracking
python -m pip install wandb
``` -->

2. Configure experiments in `config.py`

3. Run the example runner (creates `snapshots/…` directories and saves artifacts):

```bash
python runner.py
```

4. After a run finishes, create visualizations and plots by `visualize.py`

```bash
python runner.py
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
This repository is provided for research reproducibility.
