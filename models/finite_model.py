from typing import Optional
from torch import nn
import torch
from tools.utils import moded_max, moded_min
# =====================================================================
# FiniteModel: unified finitely-convex / finitely-concave representation
# =====================================================================

class FiniteModel(nn.Module):
    """
    Unified finite model representing either finitely convex or finitely concave functions.

    **convex mode**:
        f(x) = max_j [ k(x, y_j) - b_j ]

    **concave mode**:
        f(x) = min_j [ k(x, y_j) - b_j ]

    Supports:
    - soft / hard / ste selection (via moded_max and moded_min)
    - numerical sup/inf transforms
    - full batching support with *global warm-start* storage indexed by sample_idx
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        num_candidates: int,
        num_dims: int,
        kernel,                    # k(x,y)
        mode: str = "convex",
        temp: float = 50.0,
        is_y_parameter: bool = True,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        original_dist_to_bounds: float = 1e-1,
        is_there_default: bool = False,
    ):
        super().__init__()
        assert mode in ("convex", "concave")

        self.num_dims = num_dims
        self.num_candidates = num_candidates
        self.temp = temp
        self.mode = mode
        self.is_y_parameter = is_y_parameter
        self.is_there_default = is_there_default
        self.y_min = y_min
        self.y_max = y_max
        self.original_dist_to_bounds = original_dist_to_bounds

        self.kernel_fn = kernel

        # ---------------------------------------------------------
        # Initialize Y candidates
        # ---------------------------------------------------------
        Y_init = torch.randn(1, num_candidates, num_dims)
        min_val, max_val = Y_init.min(), Y_init.max()

        if (y_min is not None) and (y_max is not None):
            centered = Y_init - Y_init.mean()
            max_abs = centered.abs().max() + 1e-4
            mid = 0.5 * (y_min + y_max)
            half = 0.5 * (y_max - y_min)
            Y_init = mid + half * centered / max_abs
        elif y_min is not None:
            Y_init = Y_init - min_val + (y_min + original_dist_to_bounds)
        elif y_max is not None:
            Y_init = max_val - Y_init + (y_max - original_dist_to_bounds)

        if is_y_parameter:
            self.Y_rest = nn.Parameter(Y_init)
        else:
            self.register_buffer("Y_rest", Y_init)

        self.intercept_rest = nn.Parameter(torch.zeros(1, num_candidates))

        # Optional default option
        self.register_buffer("Y0", torch.zeros(1,1,num_dims))
        self.register_buffer("intercept0", torch.zeros(1,1))

        # ---------------------------------------------------------
        # GLOBAL per-datapoint warm start buffers
        # Will be lazily allocated when sample_idx is first seen.
        # ---------------------------------------------------------
        self._warm_X_global = None
        self._num_global_points = None


    # ============================================================
    # Helpers: full Y / full intercept
    # ============================================================

    def full_Y(self):
        if self.is_there_default:
            return torch.cat([self.Y0, self.Y_rest], dim=1)
        return self.Y_rest

    def full_intercept(self):
        if self.is_there_default:
            return torch.cat([self.intercept0, self.intercept_rest], dim=1)
        return self.intercept_rest


    # ============================================================
    # Forward: MAX or MIN over candidates
    # ============================================================

    def forward(self, X: torch.Tensor, selection_mode: str = "soft"):
        """
        Compute f(x) for x ∈ R^{num_samples × num_dims} using either
        max_j or min_j of the kernel scores.
        """
        Y = self.full_Y()             # (1, num_candidates, num_dims)
        b = self.full_intercept()     # (1, num_candidates)

        scores = self.kernel_fn(X[:,None,:], Y) - b

        if self.mode == "convex":
            choice, f_x, aux = moded_max(scores, Y, dim=1, temp=self.temp, mode=selection_mode)
        else:
            choice, f_x, aux = moded_min(scores, Y, dim=1, temp=self.temp, mode=selection_mode)

        # Diagnostics
        self._last_weights = aux.get("weights")
        self._last_idx = aux.get("idx")
        self._last_mean_max_weight = aux.get("mean_max")
        self._last_eff_temp = aux.get("eff_temp")

        return choice, f_x


    # ============================================================
    # INTERNAL: closure for sup/inf optimization
    # ============================================================

    def _closure_x(self, X_var, Z, maximize, lam, optimizer_obj):
        """
        Closure for optimizing x → k(x,z) - f(x).

        maximize=True → sup
        maximize=False → inf
        """
        optimizer_obj.zero_grad()

        # f(x)
        _, f_x = self.forward(X_var, selection_mode="soft")

        # k(x,z)
        k_x_z = self.kernel_fn(X_var, Z)

        obj = (k_x_z - f_x).mean()
        reg = 0.5 * lam * (X_var.pow(2).sum(dim=-1)).mean()

        if maximize:
            loss = -(obj - reg)
        else:
            loss = obj + reg

        loss.backward()
        return loss


    # ============================================================
    # PUBLIC sup/inf transforms (batch-aware sample_idx added)
    # ============================================================

    def sup_transform(self, Z, sample_idx=None, **kw):
        """ Compute sup_x [k(x,z) - f(x)] for each z. """
        return self._transform_core(Z, sample_idx, maximize=True, **kw)

    def inf_transform(self, Z, sample_idx=None, **kw):
        """ Compute inf_x [k(x,z) - f(x)] for each z. """
        return self._transform_core(Z, sample_idx, maximize=False, **kw)


    # ============================================================
    # CORE routine with per-sample warm start via sample_idx
    # ============================================================

    def _transform_core(
        self,
        Z: torch.Tensor,
        sample_idx: Optional[torch.Tensor],
        maximize: bool,
        steps: int = 50,
        lr: float = 1e-3,
        optimizer: str = "lbfgs",
        lam: float = 1e-3,
        tol: float = 1e-3,
    ):
        """
        Main routine for sup_x or inf_x:

            maximize=True  → sup_x
            maximize=False → inf_x

        Supports random mini-batching through sample_idx,
        enabling per-datapoint warm-start reuse.
        """
        num_samples, num_dims = Z.shape
        device = Z.device

        # -------------------------------------------------------------
        # Allocate GLOBAL warm-start storage if first call
        # -------------------------------------------------------------
        if sample_idx is not None:
            max_idx = int(sample_idx.max())

            if self._warm_X_global is None:
                # First time: allocate storage for all possible Y indices
                self._num_global_points = max_idx + 1
                self._warm_X_global = torch.zeros(
                    self._num_global_points, num_dims, device=device
                )
            elif max_idx + 1 > self._num_global_points:
                # Expand storage if new larger index appears
                new_size = max_idx + 1
                new_tensor = torch.zeros(new_size, num_dims, device=device)
                new_tensor[:self._num_global_points] = self._warm_X_global
                self._warm_X_global = new_tensor
                self._num_global_points = new_size

            # Extract warm starts for this batch
            X_init = self._warm_X_global[sample_idx].clone()

        else:
            # Fallback: no sample_idx → use Z as initialization
            X_init = Z.clone()

        X_var = X_init.detach().clone().requires_grad_(True)

        # -------------------------------------------------------------
        # Optimizer selection
        # -------------------------------------------------------------
        opt_name = optimizer.lower()
        if opt_name == "lbfgs":
            optim_obj = torch.optim.LBFGS(
                [X_var],
                lr=lr,
                max_iter=steps,
                tolerance_grad=tol,
                tolerance_change=tol,
                line_search_fn="strong_wolfe"
            )
            optim_obj.step(lambda: self._closure_x(X_var, Z, maximize, lam, optim_obj))

        else:
            if opt_name == "adam":
                optim_obj = torch.optim.Adam([X_var], lr=lr)
            elif opt_name == "gd":
                optim_obj = torch.optim.SGD([X_var], lr=lr)
            else:
                raise ValueError(f"Unknown optimizer {optimizer}")

            prev = None
            for _ in range(steps):
                loss = self._closure_x(X_var, Z, maximize, lam, optim_obj)
                optim_obj.step()
                if prev is not None and abs(prev - loss.item()) < tol:
                    break
                prev = loss.item()

        # -------------------------------------------------------------
        # Compute transform value and save warm starts
        # -------------------------------------------------------------
        with torch.no_grad():
            _, f_x_detached = self.forward(X_var)
            k_x_z_detached = self.kernel_fn(X_var, Z)
            
            if sample_idx is not None:
                self._warm_X_global[sample_idx] = X_var.detach()
            else:
                self._warm_X_fallback = X_var.detach()
        
        # Compute values WITH gradients for backprop
        _, f_x = self.forward(X_var)
        k_x_z = self.kernel_fn(X_var, Z)
        values = k_x_z - f_x

        return X_var.detach(), values
