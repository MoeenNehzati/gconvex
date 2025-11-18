import model
import torch
from tools import utils
import os
from tools.feedback import logger
import math
import config
import torch
import argparse
import glob
import re
import numpy as np
import torch
from torch import nn
from datetime import datetime
import tools.feedback
import os
import tools.utils
import math
from typing import Dict, Any, Tuple, Optional, List, Callable
from model import FinitelyConvexModel
logger = tools.feedback.logger


class Trainer:
    """Encapsulate training loop state and behavior for mechanism optimization."""
    def __init__(
        self,
        sample: torch.Tensor,
        mechanism: nn.Module,
        optimizers: Dict[str, Any],
        schedulers: Dict[str, Any],
        modes: List[str],
        constraint_fns: Optional[List[Callable[[nn.Module, Dict[str, Any]], torch.Tensor]]] = None,
        initial_penalty_factor: float = 1.0,
        nsteps: int = 10000,
        steps_per_snapshot: int = 200,
        steps_per_update: int = 50,
        window: int = 500,
        detect_anomaly: bool = False,
        epsilon: float = 1e-2,
        use_wandb: bool = True,
        wandb_project: str = "mechanism-design",
        writing_dir: str = "tmp",
        convergence_tolerance: float = 1e-5,
        switch_threshold: float = 0.98,
        switch_patience: int = 200,
    ) -> None:
        self.sample = sample
        self.mechanism = mechanism
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.modes = modes
        self.constraint_fns = constraint_fns or []
        self.initial_penalty_factor = initial_penalty_factor
        self.nsteps = nsteps
        self.steps_per_snapshot = steps_per_snapshot
        self.steps_per_update = steps_per_update
        self.window = window
        self.detect_anomaly = detect_anomaly
        self.epsilon = epsilon
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.writing_dir = os.fspath(writing_dir)
        self.convergence_tolerance = convergence_tolerance
        self.switch_threshold = switch_threshold
        self.switch_patience = switch_patience

        # state
        self.mode = modes[0]
        self.optimizer = optimizers[self.mode]
        self.scheduler = schedulers[self.mode]
        self.nmodes = len(modes)
        self.above_thresh_streak = 0

        # penalties
        self.nconstraints = len(self.constraint_fns)
        self.base_penalty = torch.full((self.nconstraints,), initial_penalty_factor)
        self.penalty_factors = self.base_penalty.clone()
        self.max_penalty = 2000.0 * self.base_penalty
        self.min_penalty = 0.1 * self.base_penalty
        self.alpha = 2.0
        self.beta_ema = 0.9
        self.violation2_ema = torch.zeros(self.nconstraints)

        # initialize & bookkeeping
        self.ls = []
        self.start_epoch = 0
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.writing_dir, exist_ok=True)

        # wandb setup
        self.wandb_project_url = None
        self.wandb_run_url = None
        self.live_wandb_dict = {}
        if self.use_wandb:
            import wandb
            wandb_dir = os.path.join(self.writing_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(project=self.wandb_project, name=f"run_{self.run_id}", dir=wandb_dir, config={"nsteps": self.nsteps})
            wandb.watch(self.mechanism, log="all", log_freq=self.steps_per_snapshot)
            self.wandb_project_url = f"https://wandb.ai/{wandb.run.entity}/{self.wandb_project}"
            self.wandb_run_url = wandb.run.url
            # keep a reference to the wandb module so other methods can safely
            # call log/finish even outside the __init__ local scope
            self._wandb = wandb
            self.live_wandb_dict = {"wandb_project_url": self.wandb_project_url,
                                    "wandb_run_url": self.wandb_run_url}
            logger.info(f"🚀 WandB Project: {self.wandb_project_url}")
            logger.info(f"📊 WandB Run: {self.wandb_run_url}")

    def _load_final_if_exists(self) -> Optional[nn.Module]:
        """If a completed final snapshot exists, load and return it.

        This allows the Trainer to short-circuit training when a previous run
        produced a final artifact in `writing_dir`.
        """
        pattern = os.path.join(self.writing_dir, "*final*")
        finals = glob.glob(pattern)
        if finals:
            latest_final = max(finals, key=os.path.getmtime)
            logger.info(f"Loading final snapshot: {latest_final}")
            mech = torch.load(latest_final, map_location="cpu")
            return mech
        return None

    def _load_latest_checkpoint(self) -> Optional[Tuple[Any, Any, Any]]:
        """Load the most recent checkpoint (if any) and return (mech, opt, sch).

        The function searches for files matching '*epoch_*.pt', extracts the
        epoch number from the filename and sets `self.start_epoch`. If a
        checkpoint is found the saved mechanism, optimizer and scheduler are
        returned so the trainer can resume.
        """
        pattern = os.path.join(self.writing_dir, "*epoch_*.pt")
        snapshots = glob.glob(pattern)
        if not snapshots:
            return None
        def extract_epoch(path):
            match = re.search(r'epoch_(\d+)\\.pt', path)
            return int(match.group(1)) if match else -1
        # the latest snapshot
        latest_snapshot = max(snapshots, key=extract_epoch)
        logger.info(f"Loading checkpoint: {latest_snapshot}")
        checkpoint = torch.load(latest_snapshot, map_location="cpu")
        mech = checkpoint.get('mechanism', None)
        opt = checkpoint.get('optimizer', None)
        sch = checkpoint.get('scheduler', None)
        self.start_epoch = extract_epoch(latest_snapshot)
        # new run_id will be run_id of loaded + "R"
        m = re.search(r'([^/]+)_epoch_\d+\\.pt', os.path.basename(latest_snapshot))
        if m:
            self.run_id = m.group(1) + "R"
        return mech, opt, sch

    def _save_snapshot(self, prefix: str, epoch: int) -> None:
        """Persist a checkpoint with the mechanism, optimizer and scheduler.

        The saved file uses the convention `{prefix}epoch_{epoch}.pt` so resume
        logic can find and order checkpoints by epoch number.
        """
        snapshot_path = f"{prefix}epoch_{epoch}.pt"
        torch.save({'mechanism': self.mechanism, 'optimizer': self.optimizer, 'scheduler': self.scheduler}, snapshot_path)
        logger.info(f"Epoch {epoch}: Saved snapshot to {snapshot_path}")
        if self.use_wandb:
            try:
                self._wandb.log({"checkpoint_saved": epoch}, step=epoch)
            except Exception:
                # be defensive: don't crash checkpoint saving if wandb logger fails
                logger.debug("wandb.log failed during checkpoint save")

    def run(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Execute the training loop and return (mechanism, mech_data).

        The loop supports resume from checkpoints in `writing_dir`, periodic
        snapshots, wandb logging and a simple mode-switching heuristic based
        on the peakedness of the selection distribution.
        """
        # checking for complete optimization results
        final = self._load_final_if_exists()
        if final is not None:
            logger.info("Final mechanism found; returning it.")
            mech_data = final.compute_mechanism(self.sample)
            return final, mech_data

        #Loading the latest existing checkpoint
        ck = self._load_latest_checkpoint()
        if ck is not None:
            mech, opt, sch = ck
            if mech is not None:
                self.mechanism = mech
            if opt is not None:
                self.optimizer = opt
            if sch is not None:
                self.scheduler = sch
            logger.info(f"Starting from epoch {self.start_epoch}")

        prefix = os.path.join(self.writing_dir, f"{self.run_id}_")
        remaining_epochs = self.nsteps - self.start_epoch

        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        with feedback.LiveOrJupyter() as live:
            for epoch in range(remaining_epochs):
                epoch_global = epoch + self.start_epoch
                self.optimizer.zero_grad()
                mechanism_data = self.mechanism.compute_mechanism(self.sample, mode=self.mode)

                # Compute constraints and penalties
                constraint_vals = [fn(self.mechanism, mechanism_data).reshape(-1) for fn in self.constraint_fns]
                violations = [(-c).clamp_min(0.0) for c in constraint_vals]
                penalties = torch.stack([barrier(c, epsilon=self.epsilon, reduction="sum") for c in constraint_vals]) if constraint_vals else torch.tensor([])

                with torch.no_grad():
                    mean_violation2 = torch.stack([viol.pow(2).mean() for viol in violations]) if violations else torch.tensor([])
                    if mean_violation2.numel() > 0:
                        self.violation2_ema = self.beta_ema * self.violation2_ema + (1 - self.beta_ema) * mean_violation2
                        scale = 1.0 + self.alpha * (self.violation2_ema / (self.epsilon**2 + 1e-12))
                        target = self.base_penalty * scale
                        self.penalty_factors = self.beta_ema * self.penalty_factors + (1.0 - self.beta_ema) * target
                        self.penalty_factors.clamp_(min=self.min_penalty, max=self.max_penalty)
                        minimum_constraint_value = [v.min().item() for v in constraint_vals]
                    else:
                        minimum_constraint_value = []

                weighted_penalties = (penalties * self.penalty_factors).sum() if penalties.numel() > 0 else torch.tensor(0.0)
                if penalties.numel() > 0 and penalties.isnan().any():
                    logger.error(f"Epoch {epoch_global}: penalties became nan {penalties.detach().cpu().numpy()} with constraints being {constraint_vals.detach().cpu().numpy() if constraint_vals.numel()>0 else 'N/A'}")

                # Compute gradients
                mean_profit = mechanism_data["profits"].mean()
                g = mean_profit + weighted_penalties
                l = -g
                l.backward()
                l_scaler = l.item()
                self.ls.append(l_scaler)

                total_gradient_norm = 0.0
                for p in self.mechanism.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_gradient_norm += param_norm.item() ** 2
                total_gradient_norm = total_gradient_norm ** 0.5
                max_clipping_norm = 5
                max_logging_norm = 100
                torch.nn.utils.clip_grad_norm_(self.mechanism.parameters(), max_clipping_norm)
                if total_gradient_norm > max_logging_norm:
                    logger.warning(f"Epoch {epoch_global}: Gradient norm exceeded {max_logging_norm}: {total_gradient_norm:.4f}")

                # Update parameters
                self.optimizer.step()
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        try:
                            self.scheduler.step(g.item())
                        except Exception:
                            self.scheduler.step()
                    else:
                        self.scheduler.step()

                # Check for how peaked softmax is and whether to switch optimizer/scheduler
                last_mean_max_weight = getattr(self.mechanism, '_last_mean_max_weight', float('nan'))
                if (self.nmodes > 1) and (self.mode == self.modes[0]) and (not math.isnan(last_mean_max_weight)):
                    if last_mean_max_weight >= self.switch_threshold:
                        self.above_thresh_streak += 1
                    else:
                        self.above_thresh_streak = 0

                    if self.above_thresh_streak >= self.switch_patience:
                        self.mode = self.modes[1]
                        self.optimizer = self.optimizers[self.mode]
                        self.scheduler = self.schedulers[self.mode]
                        logger.info(f"🔀 Switched to {self.mode} at epoch {epoch_global} (mean_max≈{last_mean_max_weight:.3f}) based on max_weight")

                # data to be logged
                optimization_data = {
                    "mean_violation2": mean_violation2 if 'mean_violation2' in locals() else torch.tensor([]),
                    "minimum_constraint_value": minimum_constraint_value,
                    "penalty": penalties,
                    "penalty_factors": self.penalty_factors,
                    "penalized gain": g,
                    "l_scaler": l_scaler,
                    "epoch": epoch_global,
                    "total_gradient_norm": total_gradient_norm,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "last_mean_max_weight": last_mean_max_weight,
                }
                if self.use_wandb:
                    log_data = {k:(v.detach().cpu().numpy() if hasattr(v, 'detach') else v) for k,v in optimization_data.items()}
                    try:
                        self._wandb.log(log_data, step=epoch_global)
                    except Exception:
                        logger.debug("wandb.log failed during training step")

                #Check for convergence
                if epoch >= self.window:
                    recent = self.ls[-self.window:]
                    max_diff = max(abs(recent[i] - recent[i-1]) for i in range(1, self.window))
                    relative_max_diff = max_diff/abs(recent[-1])
                    if relative_max_diff < self.convergence_tolerance:
                        if self.mode == self.modes[-1]:
                            logger.info(f"Converged at epoch {epoch_global} (loss stable over last {self.window} epochs)")
                            if self.use_wandb:
                                try:
                                    self._wandb.log({"convergence_epoch": epoch_global}, step=epoch_global)
                                except Exception:
                                    logger.debug("wandb.log failed during convergence logging")
                        elif (self.nmodes>1) and (self.mode==self.modes[0]):
                            #if there are remaining unexplored modes, switch to them
                            self.ls[-1] = 0
                            self.mode = self.modes[1]
                            self.optimizer = self.optimizers[self.mode]
                            self.scheduler = self.schedulers[self.mode]
                            logger.info(f"🔀 Switched to {self.mode} at epoch {epoch_global} (mean_max≈{last_mean_max_weight:.3f}) based on convergence")
                        break

                #saving a snapshot
                if epoch>0 and (epoch % self.steps_per_snapshot == 0 or torch.isnan(l).any()):
                    self._save_snapshot(prefix, epoch_global)

                # updating the live panel
                if epoch % self.steps_per_update == 0:
                    panel_data = {**mechanism_data, **optimization_data, **self.live_wandb_dict, "desc":self.writing_dir}
                    panel = feedback.make_status_panel(panel_data)
                    live.update(panel)
            else:
                logger.warning(f"Finished training for {self.nsteps} epochs without convergence.")
                self.mechanism.converged = False

        self.mechanism.steps_taken = epoch
        self.mechanism.max_steps = self.nsteps
        if self.use_wandb:
            try:
                self._wandb.finish()
            except Exception:
                logger.debug("wandb.finish failed")
        torch.save(self.mechanism, f"{prefix}final.pt")
        mech_data = self.mechanism.compute_mechanism(self.sample)
        return self.mechanism, mech_data
        
def run(
    sample: torch.Tensor,
    modes: List[str] = ["soft"],
    compile: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    optimizers_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    schedulers_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    with_hook: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:    
    """Convenience entrypoint that constructs a `Trainer` and runs training.
    optimizers_kwargs_dict and schedulers_kwargs_dict are dicts whose keys include value in modes with the values being the corresponding kwargs dict
    Parameters mirror the Trainer constructor and common defaults for
    experiments. Returns the trained mechanism and the final mechanism data
    produced by `mechanism.compute_mechanism(sample)`.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}
    if optimizers_kwargs_dict is None:
        optimizers_kwargs_dict = {}
    if schedulers_kwargs_dict is None:
        schedulers_kwargs_dict = {}
    for mode in modes:
        if mode not in optimizers_kwargs_dict:
            raise KeyError(f"Mode '{mode}' not found in optimizers_kwargs_dict keys: {list(optimizers_kwargs_dict.keys())}")
        if mode not in schedulers_kwargs_dict:
            raise KeyError(f"Mode '{mode}' not found in schedulers_kwargs_dict keys: {list(schedulers_kwargs_dict.keys())}")
    #initialize the model
    if with_hook:
        mechanism = FinitelyConvexModel.with_hooks(**model_kwargs)
    else:
        mechanism = FinitelyConvexModel(**model_kwargs)
    compiled_mechanism = mechanism
    if compile:
        compiled_mechanism = torch.compile(mechanism)
    
    #initialize optimizers of not provided
    optimizers = train_kwargs.pop("optimizers", None)
    if optimizers is None:
        optimizers = {}
        for mode in modes:
            if mode == "soft" or mode == "hard":
                optimizers[mode] = torch.optim.AdamW(
                    compiled_mechanism.parameters(),
                    **optimizers_kwargs_dict.get(mode, {}),
                )
            if mode == "ste":
                optimizers[mode] = torch.optim.SGD(
                    compiled_mechanism.parameters(),                    
                    **optimizers_kwargs_dict.get(mode, {}),
                )
    #initialize schedulers if not provided
    schedulers = train_kwargs.pop("schedulers", None)
    if schedulers is None:
        schedulers = {}
        for mode in modes:
            scheduler_kwargs = schedulers_kwargs_dict.get(mode, {})
            if mode=="soft" or mode=="hard":
                schedulers[mode] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[mode],
                    mode="max",
                    **scheduler_kwargs,
                )
            if mode=="ste":
                schedulers[mode] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[mode],
                                                                              **scheduler_kwargs,
                                                                          )
    #initialize and run the trainer
    trainer = Trainer(
        sample=sample,
        mechanism=compiled_mechanism,
        optimizers=optimizers,
        schedulers=schedulers,
        modes=modes,
        **train_kwargs,
    )
    mechanism, mechanism_data = trainer.run()
    return mechanism, mechanism_data

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