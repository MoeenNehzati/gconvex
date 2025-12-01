"""Command line driver for sweeping mechanism design experiments over multiple dimensions.

Runs a grid of dimensions/kernels/costs, writing checkpoints and snapshots under
`WRITING_ROOT/mech/`. Use --clear to remove previous runs in that directory
before starting a new sweep.
"""

import argparse
import os
import shutil
from math import exp
import numpy as np
import torch
from config import WRITING_ROOT
from math import sqrt
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
    default=20_000,
    help="Chunk size used when computing the kernel to limit memory.",
)
parser.add_argument(
    "--niters",
    type=int,
    default=60_000,
    help="Number of training iterations (passed as nsteps).",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=2e-3,
    help="Barrier epsilon threshold for the constraints.",
)
args = parser.parse_args()
writing_dir_base = f"{WRITING_ROOT}mech/"
if args.clear:
    shutil.rmtree(writing_dir_base, ignore_errors=True)
    print(f"Cleared {writing_dir_base}")

torch.manual_seed(0)

def candidate_scale(dim: int, base: float = 5.0, target: float = 999.0, growth: float = 0.01) -> float:
    """Return a smoothly growing candidate size that approaches `base+target` as `dim` increases."""
    return base + target * (1 - exp(-growth * (dim - 1)))

def dot_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Simple dot product kernel used when bundling goods independent of constraints."""
    return (X * Y).sum(dim=-1)

# def mismatch_kernel(X: torch.Tensor, Y: torch.Tensor, α=.5, κ=2.) -> torch.Tensor:
#     """Returns a bounded surplus that grows when the bundle exceeds the type."""
#     return κ*torch.sigmoid(α * (Y - X)).sum(dim=-1)

def mismatch_kernel(X: torch.Tensor, Y: torch.Tensor, κ: float = 2.0, α: float = 1.0) -> torch.Tensor:
    return κ * torch.nn.functional.softplus(α * (Y - X), beta=κ).sum(dim=-1)
    # """Quadratic hinge surplus: zero when Y<=X, grows quadratically when Y>X."""
    # gap = (Y - X).clamp_min(0.0)
    # return gap.sum(dim=-1)#κ * (α * gap).pow(2).sum(dim=-1)


def euclidean_cost(Y, β=.5, ε=1e-4):
    return β * (torch.sqrt(Y**2 + ε**2) - ε).sum(dim=-1)

def quadratic_cost(Y, β=.1):
    return β * (Y**2).sum(dim=-1)

def hinge_kernel(X: torch.Tensor, Y: torch.Tensor, κ=1.0) -> torch.Tensor:
    # reward only above the type, linear growth
    return κ * (Y - X).clamp_min(0).sum(dim=-1)





epsilon = args.epsilon
# Sweep dimensions with smoothly growing candidate counts per dim.
dims = [20, 40]#[1, 5, 10, 20, 30, 40, 50, 100, 250]#[1, 2, 4, 6, 12, 50, 100, 250]
npoints_per_dim1 = [6*d+5 for d in dims]#[int(candidate_scale(d, base=2, target=275, growth=0.015)) for d in dims]
npoints_per_dim2 = npoints_per_dim1
initial_penalty_factors = [100./d for d in dims]
# Training / scheduler defaults shared across sweeps.
patience = 600
cooldown = 300
factor = 0.85
nsteps = args.niters
max_samples = 1_000
temp = 60.
temp_warmup_steps = 1000
temp_schedule_initial = 5.0
lrs = [0.05 for d in dims]  # dimension-scaled learning rate
convergence_tolerance = 1e-8
sorted_model = True
is_Y_parameter = True
window = 1000
default_utility = 1.5*epsilon
scheduler_threshold = 1e-2
max_clipping_norm = 5
unif_sample = torch.rand(max_samples, max(dims))
lognormal_sample = torch.exp(torch.randn(max_samples, max(dims)) * 0.25)

y_maxes = [None]#[1.0, None]
costs =   [quadratic_cost]#[None, euclidean_cost]
kernels = [hinge_kernel]#[dot_kernel, mismatch_kernel]
samples = [lognormal_sample]#[unif_sample, lognormal_sample]
npoints_per_dim_per_config = [npoints_per_dim1]#, npoints_per_dim2]
are_there_defaults = [True]#[True, False]

# Each tuple (y_max, cost, kernel, sample) defines one sweep family:
#   - dot_kernel + no cost on [0,1]
#   - mismatch_kernel + euclidean_cost on unbounded lognormal types
for y_max, cost, kernel, big_sample, npoints_per_dim, is_there_default in zip(y_maxes, costs, kernels, samples, npoints_per_dim_per_config, are_there_defaults):
    for dim, npoints, lr, initial_penalty_factor in zip(dims, npoints_per_dim, lrs, initial_penalty_factors):
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
        with torch.no_grad():
            q = torch.linspace(0.3, 0.9, mechanism_instance.num_candidates, device=sample.device)
            Y_target = torch.quantile(sample, q, dim=0).unsqueeze(0)  # sample is sorted already
            # keep this to satisfy the sorted parametrization
            Y_target, _ = Y_target.sort(dim=-1)
            diffs = torch.cat([Y_target[..., :1], Y_target[..., 1:] - Y_target[..., :-1]], dim=-1)
            raw = torch.log(torch.expm1(diffs.clamp_min(1e-8)))
            mechanism_instance.Y_rest_raw.copy_(raw)
            if cost is not None:
                costs_per_candidate = cost(Y_target.squeeze(0)).view(1, -1)  # shape (1, npoints)
                mechanism_instance.intercept_rest.copy_(costs_per_candidate)
            else:
                mechanism_instance.intercept_rest.zero_()



        # with torch.no_grad():
        #     # initial_y = (torch.randn_like(mechanism_instance.Y_rest_raw)+1).clamp(min=0.)
        #     # mechanism_instance.Y_rest_raw.copy_(initial_y)
        #     Y0 = torch.exp(torch.randn_like(mechanism_instance.Y_rest_raw) * 0.5)/10  # match your sample scale
        #     if sorted_model:
        #         Y0, _ = Y0.sort(dim=-1)
        #     mechanism_instance.Y_rest_raw.copy_(Y0)
        #     mechanism_instance.intercept_rest.fill_(0.0)
        #     # mechanism_instance.intercept_rest.fill_(.0)
        #     # mechanism_instance.Y_rest_raw.fill_()      # push all candidate bundles to 1
        #     # mechanism_instance.intercept_rest.fill_(0.)   # start every price at zero
        #     # mechanism_instance.Y_rest_raw.fill_(1.0)      # push all candidate bundles to 1
        #     # mechanism_instance.intercept_rest.fill_(1.0)   # start every price at zero

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
                    "cooldown": cooldown,
                    "eps": 1e-8,
                },
            },
            train_kwargs={
                "nsteps": nsteps,
                "max_clipping_norm": max_clipping_norm,
                "initial_penalty_factor": initial_penalty_factor,
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
                "switch_threshold": 0.995,

        },
    )
        # Artifacts: snapshots/metrics written under writing_dir_dim (CSV/pt files).
