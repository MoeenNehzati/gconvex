import model
import torch
from tools import utils
import os
from tools.feedback import logger
import math
import config
import torch
import argparse

if __name__ == "__main__":
    torch.manual_seed(2)
    # Argparser
    parser = argparse.ArgumentParser(description="Run matching and genetics experiment.")
    parser.add_argument("-c", "--correlated", action="store_true", help="Use correlated sample")
    parser.set_defaults(correlated=False)
    args = parser.parse_args()
    if args.correlated:
        desc = "correlated"
    else:
        desc = "independant"
    # Initialize from config
    max_dim = config.MAX_DIM
    rank = config.RANK
    joint_dir = utils.generate_dir_and_name(f"{config.WRITING_ROOT}{desc}/", **config.PATH_RELEVANT_KWARGS)
    sample_path = f"{joint_dir}sample.pt"
    if os.path.exists(sample_path):
        data = torch.load(sample_path)
        all_sample = data["all_sample"]
        logger.info(f"Loaded sample from {sample_path}")
    else:
        if desc == "correlated":
            L = torch.randn(max_dim, rank)
            row_norms = torch.norm(L, dim=1, keepdim=True).clamp_min(1e-12)
            L = L * (config.ROW_NORM_TARGET / row_norms)
            LLt_diag = (L * L).sum(dim=1)
            uniq = (1.0 - LLt_diag).clamp_min(1e-9)
            R = L @ L.T + torch.diag(uniq)
            jitter = 1e-5
            R = R + jitter * torch.eye(max_dim)
            R,_ = utils.greedy_neg_order(R)
            Lc = torch.linalg.cholesky(R)
            z = torch.randn(config.EXPECTATION_SAMPLE_SIZE, max_dim) @ Lc.T
            all_sample = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
            torch.save({"L": L,
                        "R": R,
                        "Lc": Lc,
                        "all_sample": all_sample
                        }, sample_path)
            logger.info(f"Generated correlated sample with all_sample.shape={all_sample.shape}, R={R}, Lc={Lc}")
        if desc == "independant":
            all_sample = torch.rand(config.EXPECTATION_SAMPLE_SIZE, max_dim)
            torch.save({"all_sample": all_sample}, sample_path)
            logger.info(f"Generated independant sample with all_sample.shape={all_sample.shape}")

    for dim,npoints in zip(config.DIMS, config.NPOINTS):
        model_kwargs = config.MODEL_KWARGS.copy()
        model_kwargs["y_dim"] = dim
        model_kwargs["npoints"] = npoints
        dir = utils.generate_dir_and_name(joint_dir, dim=dim)
        mechanism, mechanism_data = model.run(
                                            all_sample[:, :dim],
                                            config.MODES,
                                            compile=True,
                                            model_kwargs=model_kwargs,
                                            optimizers_kwargs_dict=config.OPTIMIZERS_KWARGS_DICT,
                                            schedulers_kwargs_dict=config.SCHEDULERS_KWARGS_DICT,
                                            train_kwargs={**config.TRAIN_KWARGS, "writing_dir": dir},
                                        )