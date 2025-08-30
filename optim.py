"""
optim.py
---------
Implements finitely convex functions and provides helpers for training

Key exports:
- moded_max: soft/hard/ste selection helper
- FinitelyConvexModel: compact parametric mechanism used in experiments
- Trainer: encapsulates the training loop (checkpoints, penalties, wandb, switching)
"""
import glob
import re
import numpy as np
import torch
from torch import nn
from datetime import datetime
import feedback
import os
import utilities
import math
from typing import Dict, Any, Tuple, Optional, List, Callable
logger = feedback.logger

def moded_max(
    x: torch.Tensor,                 # (S, B)
    Y: torch.Tensor,                 # (S, B, n)
    dim: int = 1,
    temp: float = 5.0,
    max_effective_temp: float = 5000.0,
    mode: str = "soft",              # "soft" | "hard" | "ste"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    S, B, n are sample size, number of bundles, number of items
    Returns:
      choice: (S, n)
      v:      (S,)
      aux: {
        "weights": (S,B) or None,   # soft/ste: softmax weights; hard: None
        "idx":     (S,)  or None,   # hard/ste: argmax indices; soft: None
        "mean_max": float,          # avg max weight across batch
        "eff_temp": (S,1) or None,  # effective temperature per row (soft/ste)
      }
    """
    # Assumptions matching pipeline
    assert dim == 1, "This implementation assumes choices are on dim=1."
    assert Y.dim() == 3 and Y.size(0) == 1, "Expected Y with shape (1, K, d)."

    Y_squeezed = Y.squeeze(0)  # (B, n)
    aux: Dict[str, Any] = {"weights": None, "idx": None, "mean_max": float("nan"), "eff_temp": None}

    if mode == "soft":
        #uses softmax
        # Row-wise scale-invariant temperature
        # make temperature proportional to inverse of max(x)-min(x)
        span = (x.amax(dim=dim, keepdim=True) - x.amin(dim=dim, keepdim=True)).clamp_min(1e-3)  # (S,1)
        eff_temp = (temp / span).clamp(max=max_effective_temp, min=temp)  # (S,1)
        z = x * eff_temp
        #subtract max so the maximum is 0
        z = z - z.amax(dim=dim, keepdim=True)
        w = torch.softmax(z, dim=dim)                          # (S, K)
        v = (w * x).sum(dim=dim)                               # (S,)
        choice = w @ Y_squeezed                                        # (S, n)
        aux["weights"]  = w
        aux["mean_max"] = float(w.max(dim=dim).values.mean().item())
        aux["eff_temp"] = eff_temp
        return choice, v, aux

    elif mode == "hard":
        #uses hardmax and argmax
        v, idx = x.max(dim=1)               # (S,), (S,)
        choice = Y_squeezed[idx]
        aux["idx"] = idx
        aux["mean_max"] = 1.0

    elif mode == "ste":
        #Do hardmax for forward, softmax for backward
        # Reuse the function recursively for soft and hard parts
        choice_soft, v_soft, aux_soft = moded_max(
            x, Y, dim=dim, temp=temp, max_effective_temp=max_effective_temp, mode="soft"
        )
        choice_hard, v_hard, aux_hard = moded_max(
            x, Y, dim=dim, temp=temp, max_effective_temp=max_effective_temp, mode="hard"
        )

        # Straight-through glue: forward=hard, backward=soft
        choice = choice_soft + (choice_hard - choice_soft).detach()  # (S, n)
        v      = v_soft      + (v_hard      - v_soft).detach()       # (S,)
        # Merge aux (prefer soft weights/eff_temp and hard idx)
        aux["weights"]  = aux_soft["weights"]
        aux["eff_temp"] = aux_soft["eff_temp"]
        aux["idx"]      = aux_hard["idx"]
        aux["mean_max"] = aux_soft["mean_max"]
    else:
        raise ValueError(f"mode must be 'soft', 'hard', or 'ste', got {mode!r}")
    return choice, v, aux



def forward_hook(module: nn.Module,
                 input: tuple,
                 output: Any) -> None:
    """Forward hook that checks for NaNs in module outputs.

    Expects `output` to be a tuple (weights, v) produced by selection layers.
    If NaNs are found this raises a RuntimeError to fail fast and make
    debugging easier.
    """
    weights, v = output
    if torch.isnan(weights).any():
        raise RuntimeError(f"[NaN Detected] in weights output of {module.__class__.__name__}")
    if torch.isnan(v).any():
        raise RuntimeError(f"[NaN Detected] in v of output of {module.__class__.__name__}")
    logger.debug(f"[Forward] {module.__class__.__name__}: output shape {v.shape} for v and {weights.shape} for weights")

def grad_hook(grad: torch.Tensor) -> torch.Tensor:
    """Backward hook to detect NaNs in gradients.

    Register this hook on parameters to raise early when numerical issues
    appear during backward passes.
    """
    if torch.isnan(grad).any():
        raise RuntimeError("[NaN Detected] in gradient of output tensor")
    logger.debug(f"[Backward] Gradient shape: {grad.shape}")
    return grad

def attach_nan_hooks(model: nn.Module,
                     log_shapes: bool = False) -> None:
    """
    Attaches forward and gradient hooks to all modules in a model
    to detect NaNs and optionally log tensor shapes.
    """
    # Register forward hooks
    for name, module in model.named_modules():
        # Skip container modules like nn.Sequential
        if len(list(module.children())) == 0:
            module.register_forward_hook(forward_hook)

    # Register backward hooks on model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(grad_hook)

    logger.info("✅ NaN hooks attached to model.")

class FinitelyConvexModel(nn.Module):
    """
    The model implements finitely convex functions from the papers. It stores a finite candidate set Y (optionally with a default
    option) and per-candidate intercepts. The forward function computes the conjugation and argmax.

    The maximum forward can use soft, hard, or straight-through modes.

    Public attributes used by the Trainer:
    - `_last_mean_max_weight`, `_last_idx`, `_last_weights`, `_last_eff_temp`: diagnostics
    - `converged`, `steps_taken`, `max_steps`: training bookkeeping
    """
    def __init__(self,
                 npoints: int = 1000, #size of the finite subset of Y used for conjugation
                 kernel: Callable[..., torch.Tensor] = utilities.linear_kernel, #the surplus kernel
                 y_dim: int = 1, #i.e. number of goods
                 cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 temp: float = 50.0, #used in softmax
                 is_Y_parameter: bool = False, #whether to compute gradients wrp to Y or (Y, intercept)
                 is_there_default: bool = False, #whether to include a default option (Y=0, intercept=0)
                 y_min: Optional[float] = None, #bounds on Y
                 y_max: Optional[float] = None, #bounds on Y
                 original_distance_to_bounds: float = 1e-1,
                 X: Optional[torch.Tensor] = None,
                 ) -> None:
        super(FinitelyConvexModel, self).__init__()
        self.cost_fn = cost_fn
        self.y_min = y_min
        self.y_max = y_max
        self.y_dim = y_dim
        self.X = X
        self.is_there_default = is_there_default
        self.kernel = kernel
        self.temp = temp
        self.npoints = npoints
        self.converged = None
        
        # The non-default part of Y and intercept
        self.Y_rest = torch.randn((1, npoints, y_dim))
        initial_intercept_scale = y_dim
        self.intercept_rest = nn.Parameter(torch.rand((1, npoints))*initial_intercept_scale)

        #Bring Y within bounds
        sampled_max = torch.max(self.Y_rest)
        sampled_min = torch.min(self.Y_rest)
        if self.y_min is not None and self.y_max is not None:
            demeaned_Y_rest = self.Y_rest - self.Y_rest.mean()
            max_abs_demeaned = demeaned_Y_rest.abs().max() + 1e-4
            middle = (self.y_max + self.y_min)/2
            gap = self.y_max - self.y_min
            self.Y_rest = middle + (demeaned_Y_rest / max_abs_demeaned)*gap/2
        elif y_min is not None:
            self.Y_rest = self.Y_rest - sampled_min + self.y_min + original_distance_to_bounds
        elif y_max is not None:
            self.Y_rest = sampled_max - self.Y_rest + self.y_max - original_distance_to_bounds

        # Register Y_rest as parameter or buffer
        if is_Y_parameter:
            self.Y_rest = nn.Parameter(self.Y_rest)
        else:
            self.register_buffer("Y", self.Y_rest)
        
        # Adding the default values
        self.register_buffer("Y0", torch.zeros(1, 1, y_dim))
        self.register_buffer("intercept0", torch.zeros(1, 1))
        

    def full_Y(self) -> torch.Tensor:
        """Return the complete Y tensor (includes default if existing)."""
        if self.is_there_default:
            return torch.cat([self.Y0, self.Y_rest], dim=1)
        else:
            return self.Y_rest
    

    def full_intercept(self) -> torch.Tensor:
        """Return the complete intercept tensor (includes default if existing)."""
        if self.is_there_default:
            return torch.cat([self.intercept0, self.intercept_rest], dim=1)
        else:
            return self.intercept_rest

    @classmethod
    def with_hooks(cls,
                   *args: Any,
                   **kwargs: Any) -> "FinitelyConvexModel":
        """Construct the model and attach NaN-detection hooks.

        Useful when running experiments interactively to get immediate
        feedback when numerical problems occur.
        """
        model = cls(*args, **kwargs)
        attach_nan_hooks(model, log_shapes=True)
        return model

    def forward(self,
                X: torch.Tensor,
                selection_mode: str = "soft")-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with selectable mode:
        - "soft": softmax-weighted expectation (differentiable)
        - "ste" : straight-through estimator (hard forward, soft backward)
        - "hard": argmax selection (non-differentiable in selection)

        Args:
            X: (S, B) batch of input types
            selection_mode: one of {"soft", "ste", "hard"}

        Returns:
            choice: (S, n)   selected Y points
            v:      (S,)     corresponding values
        """
        Y = self.full_Y()                           # (1, B, n)
        intercept = self.full_intercept()           # (1, B)
        affines = self.kernel(X[:, None, :], Y, dim=2) - intercept  # (S, B)

        #Finding max and argmax, the max is generalized convex
        choice, v, aux = moded_max(
            x=affines,
            Y=Y,
            dim=1,
            temp=self.temp,            
            mode=selection_mode,
        )

        # stash diagnostics for trainer/logging
        self._last_mean_max_weight = aux.get("mean_max", float("nan"))
        self._last_idx = aux.get("idx", None)            # (S,) for hard/ste, else None
        self._last_weights = aux.get("weights", None)    # (S,B) for soft/ste, else None
        self._last_eff_temp = aux.get("eff_temp", None)  # (S,1) for soft/ste, else None
        return choice, v

    def compute_mechanism(self,
                          sample: torch.Tensor,
                          mode: str = "soft") -> Dict[str, Any]:
        """Compute outputs useful for logging and evaluation.

        Returns a dict containing choice, v, kernel values, revenue, cost and
        profits along with the underlying Y and intercept tensors. This
        structure is what the Trainer expects when computing loss and
        constraints.
        """
        choice, v = self.forward(sample, mode)
        ker = self.kernel(sample, choice, dim=1)
        revenue =  ker - v
        if self.cost_fn is not None:
            cost = self.cost_fn(choice)
        else:
            cost = torch.zeros_like(revenue)
        profits = revenue - cost
        return {
            "sample": sample,
            "choice": choice,
            "v": v,
            "revenue": revenue,
            "kernel": ker,
            "cost": cost,
            "profits": profits,
            "y": self.full_Y(),
            "intercept": self.full_intercept(),
        }

def barrier(c: torch.Tensor,
            epsilon: float = 1e-2,
            reduction: str = "none") -> torch.Tensor:
    """A simple hinge-squared barrier penalty for constraint enforcement.

    Parameters
    ----------
    c : torch.Tensor
        Constraint tensor where values >= 0 are feasible.
    epsilon : float
        A slack threshold; values below epsilon are penalized.
    reduction : {'none', 'mean', 'sum'}
        How to aggregate the per-element penalties.
    """
    # c >= 0 is feasible
    pen = -(epsilon-c).clamp_min(0.0).pow(2)
    if reduction == "mean":
        return pen.mean()
    if reduction == "sum":
        return pen.sum()
    return pen


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