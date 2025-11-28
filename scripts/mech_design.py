import argparse
import os
import shutil
from math import exp
import numpy as np
import torch
from config import WRITING_ROOT
import mech_design.mechanism as mechanism_module

parser = argparse.ArgumentParser(description="Run mechanism design sweep.")
parser.add_argument(
    "-c",
    "--clear",
    action="store_true",
    help="Remove existing snapshots in the writing_dir before training.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    help="Chunk size used when computing the kernel to limit memory.",
)
parser.add_argument(
    "--niters",
    type=int,
    default=20_000,
    help="Number of training iterations (passed as nsteps).",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=5e-4,
    help="Barrier epsilon threshold for the constraints.",
)
args = parser.parse_args()
writing_dir_base = f"{WRITING_ROOT}mech/"
if args.clear:
    shutil.rmtree(writing_dir_base, ignore_errors=True)
    print(f"Cleared {writing_dir_base}")

torch.manual_seed(0)

def candidate_scale(dim: int, base: float = 5.0, target: float = 999.0, growth: float = 0.01) -> float:
    """Return a smoothly growing candidate size as dim increases.

    The value starts at `base` when `dim == 1` and approaches `base + target`
    as `dim` grows.  The `growth` rate controls how quickly the curve rises
    (larger values push it toward the ceiling faster).  The formula uses an
    exponential ramp so the list comprehension stays lightweight and monotonic.
    """
    return base + target * (1 - exp(-growth * (dim - 1)))

def dot_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (X * Y).sum(dim=-1)

def mismatch_kernel(X: torch.Tensor, Y: torch.Tensor, α=4., κ=2.) -> torch.Tensor:
    """Returns a bounded surplus that grows when the bundle exceeds the type."""
    return κ*torch.sigmoid(α * (Y - X)).sum(dim=-1)

def euclidean_cost(Y, β=1.0, ε=1e-4):
    return β * (torch.sqrt(Y**2 + ε**2) - ε).sum(dim=-1)

epsilon = args.epsilon
dims = [1, 2, 4, 6, 20, 100, 250, 500]#[::-1]
npoints_per_dim = [int(candidate_scale(d, base=2, target=275, growth=0.015)) for d in dims]
print("npoints per dim:", npoints_per_dim)
patience = 400
factor = 0.6
nsteps = args.niters
max_samples = 10_000
temp = 60.
temp_warmup_steps = 500
temp_schedule_initial = 1.0
lr = 0.05
convergence_tolerance = 1e-6
sorted_model = True
is_Y_parameter = True
is_there_default = True
window = 1000
default_utility = 4*epsilon
scheduler_threshold = 1e-2
max_clipping_norm = 10
unif_sample = torch.rand(max_samples, max(dims))
lognormal_sample = torch.exp(torch.randn(max_samples, max(dims)) * 0.5)

costs = [None,euclidean_cost, ]
kernels = [dot_kernel, mismatch_kernel]
samples = [unif_sample, lognormal_sample]
y_maxes = [1.0, None]

for y_max, cost, kernel, big_sample in zip(y_maxes, costs, kernels, samples):
    for dim, npoints in zip(dims, npoints_per_dim):
        torch.manual_seed(1)
        print(y_max, cost, kernel, big_sample, dim, npoints)
        print(f"Running mechanism design for dim={dim}, npoints={npoints}")
        sample = big_sample[:, :dim]
        if sorted_model:
            sample, _ = torch.sort(sample, dim=-1)
        model_kwargs = {
            "npoints": npoints,
            "kernel": kernel,
            "cost_fn": cost,
            "y_dim": dim,
            "temp": temp,
            "is_Y_parameter": is_Y_parameter,
            "is_there_default": is_there_default,
            "default_intercept": -default_utility,
            "y_min": 0.0,
            "y_max": y_max,
            "sorted_model": sorted_model,
        }
        mechanism_instance = mechanism_module.Mechanism(**model_kwargs, kernel_batch_size=args.batch_size)
        # with torch.no_grad():
        #     mechanism_instance.Y_rest_raw.fill_(1.0)      # push all candidate bundles to 1
        #     mechanism_instance.intercept_rest.fill_(0.0)   # start every price at zero

        writing_dir_dim = os.path.join(writing_dir_base, f"dim{dim}/")

        mechanism, mechanism_data = mechanism_instance.fit(
            sample,
            already_sorted=True,
            modes=["soft"],
            compile=True,
            optimizers_kwargs_dict={"soft": {"lr": lr}},
            schedulers_kwargs_dict={
                "soft": {
                    "patience": patience,
                    "threshold": scheduler_threshold,
                    "factor": factor,
                    "cooldown": patience,
                    "eps": 1e-8,
                },
            },
            train_kwargs={
                "nsteps": nsteps,
                "max_clipping_norm": max_clipping_norm,
                "steps_per_snapshot": 200,
                "steps_per_update": 5,
                "window": window,
                "constraint_fns": [],
                "use_wandb": False,
                "writing_dir": writing_dir_dim,
                "convergence_tolerance": convergence_tolerance,
                "epsilon": epsilon,
                "temp_warmup_steps": temp_warmup_steps,
                "temp_schedule_initial": temp_schedule_initial,
            },
        )
