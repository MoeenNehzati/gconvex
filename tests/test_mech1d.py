import numpy as np
import torch

import models.mechanism as mechanism_module
import tools.visualize as viz


def dot_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (X * Y).sum(dim=-1)


# def test_stats_work():
#     """Uniform types over [0,1] should accept offers priced at 0.5."""
#     mech = mechanism_module.Mechanism(
#         npoints=1,
#         kernel=dot_kernel,
#         y_dim=1,
#         temp=1.0,
#         is_Y_parameter=False,
#         is_there_default=True,
#         y_min=0.0,
#         y_max=1.0,
#     )
#     with torch.no_grad():
#         mech.Y_rest.fill_(1.0)
#         mech.intercept_rest.fill_(0.5)

#     xs, q, _, _, _ = viz.eval_mech_1d(mech, N=401)
#     cutoff = viz.estimate_cutoff(xs, q, level=0.5)
#     assert abs(cutoff - 0.5) < 5e-3
#     q = np.array(q)
#     below_price = q[xs < 0.49]
#     above_price = q[xs > 0.51]
#     assert np.allclose(below_price, 0.0, atol=5e-3)
#     assert np.allclose(above_price, 1.0, atol=5e-3)


def test_training_mechanism(tmp_path):
    """Short Mechanism training run produces valid mechanism metadata."""
    dim = 100
    npoints = 200
    patience = 100
    nsteps = 10000
    sample = torch.rand(10_000, dim)
    model_kwargs = {
        "npoints": npoints,
        "kernel": dot_kernel,
        "y_dim": dim,
        "temp": 80.0,
        "is_Y_parameter": True,
        "is_there_default": True,
        "y_min": 0.0,
        "y_max": 1.0,
    }
    mechanism_instance = mechanism_module.Mechanism(**model_kwargs)
    mechanism, mechanism_data = mechanism_instance.fit(
        sample,
        modes=["soft"],
        compile=False,
        optimizers_kwargs_dict={"soft": {"lr": 0.05}},
        schedulers_kwargs_dict={
            "soft": {
                "patience": patience,
                "threshold": 1e-3,
                "factor": 0.5,
                "cooldown": 1,
                "eps": 1e-6,
            },
        },
        train_kwargs={
            "nsteps": nsteps,
            "steps_per_snapshot": 20,
            "steps_per_update": 20,
            "window": 10,
            "constraint_fns": [],
            "use_wandb": False,
            "writing_dir": str(tmp_path / "mech1d"),
            "convergence_tolerance": 1e-5,
        },
    )
    assert "profits" in mechanism_data
    assert mechanism_data["profits"].shape[0] == sample.shape[0]
    revenue_mean = float(mechanism_data["revenue"].mean().item())
    print(f"mean revenue = {revenue_mean:.6f}")
