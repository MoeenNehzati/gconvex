from typing import Optional
from torch import nn
import torch
from tools.utils import moded_max, moded_min
from models.inf_convolution import InfConvolution
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
    # PUBLIC sup/inf transforms (batch-aware sample_idx added)
    # ============================================================

    def sup_transform(self, Z, sample_idx=None, **kw):
        """ 
        Compute sup_x [k(x,z) - f(x)] for each z.
        
        Returns:
            tuple: (X_opt, values, converged) where
                - X_opt: Optimized positions (detached)
                - values: Transform values (with gradients)
                - converged: Boolean indicating if optimization converged
        """
        return self._transform_core(Z, sample_idx, maximize=True, **kw)

    def inf_transform(self, Z, sample_idx=None, **kw):
        """ 
        Compute inf_x [k(x,z) - f(x)] for each z.
        
        Returns:
            tuple: (X_opt, values, converged) where
                - X_opt: Optimized positions (detached)
                - values: Transform values (with gradients)
                - converged: Boolean indicating if optimization converged
        """
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

            maximize=True  → sup_x (negated inf)
            maximize=False → inf_x

        Supports random mini-batching through sample_idx,
        enabling per-datapoint warm-start reuse.
        
        Uses InfConvolution for implicit differentiation.
        
        Returns:
            tuple: (X_opt, values, converged) where
                - X_opt: Optimized positions (detached)
                - values: Transform values (with gradients)
                - converged: Boolean indicating if optimization converged
                  
        Convergence Detection:
            - LBFGS: Always considered converged (uses internal line search)
            - Adam/GD: Converged if |loss[i] - loss[i-1]| < tol for any step i
            - If max steps reached without meeting tolerance, converged=False
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

        # -------------------------------------------------------------
        # Create a wrapper module for f(x) to work with InfConvolution
        # InfConvolution expects f_net(x) to return scalar value
        # and x will be a 1D tensor of shape (num_dims,)
        # -------------------------------------------------------------
        class FWrapper(nn.Module):
            def __init__(self, finite_model, negate=False):
                super().__init__()
                self.finite_model = finite_model
                self.negate = negate
            
            def forward(self, x):
                # x is (num_dims,) - need to add batch dimension
                # FiniteModel.forward expects (num_samples, num_dims)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                # self.finite_model.forward returns (choice, f_x)
                _, f_x = self.finite_model.forward(x, selection_mode="soft")
                # f_x is (num_samples,) or scalar - return scalar
                result = f_x.squeeze()
                return -result if self.negate else result
        
        # -------------------------------------------------------------
        # Define kernel function that handles maximize flag and dimensions
        # InfConvolution expects K(x, y) where x and y are 1D tensors
        # 
        # For maximize=True (sup):
        #   sup_x [k(x,z) - f(x)] = -inf_x [-(k(x,z) - f(x))]
        #                         = -inf_x [f(x) - k(x,z)]
        #   InfConvolution computes: inf_x [K(x,z) - F(x)]
        #   So we need: K(x,z) = -k(x,z) and F(x) = -f(x)
        #   Then: inf_x [-k(x,z) - (-f(x))] = inf_x [f(x) - k(x,z)]
        #   And negate result: -inf_x [f(x) - k(x,z)] = sup_x [k(x,z) - f(x)]
        # -------------------------------------------------------------
        if maximize:
            f_wrapper = FWrapper(self, negate=True)
            
            def K_wrapper(x, z):
                # Ensure x and z are 2D for kernel_fn
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                result = -self.kernel_fn(x, z)
                return result.squeeze()
        else:
            f_wrapper = FWrapper(self, negate=False)
            
            def K_wrapper(x, z):
                # Ensure x and z are 2D for kernel_fn
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                result = self.kernel_fn(x, z)
                return result.squeeze()

        # -------------------------------------------------------------
        # Apply InfConvolution for each sample in Z
        # InfConvolution now returns (g_value, converged, x_star) so we
        # don't need to re-solve the optimization
        # -------------------------------------------------------------
        values_list = []
        converged_list = []
        X_opt_list = []

        for i in range(num_samples):
            z_i = Z[i]  # 1D tensor of shape (num_dims,)
            x_init_i = X_init[i]  # 1D tensor of shape (num_dims,)
            
            # InfConvolution.apply returns (g_value, converged, x_star)
            # This computes the value with proper gradients via implicit differentiation
            # and returns the optimal x for warm-starting
            g_i, conv_i, x_star_i = InfConvolution.apply(
                z_i,
                f_wrapper,
                K_wrapper,
                x_init_i,
                steps,
                lr,
                optimizer,
                lam,
                tol,
                *list(self.parameters())
            )
            
            values_list.append(g_i)
            converged_list.append(conv_i)
            X_opt_list.append(x_star_i)
        
        values = torch.stack(values_list)
        converged = all(converged_list)
        X_opt = torch.stack(X_opt_list)
        
        # Negate values if maximize
        if maximize:
            values = -values

        # -------------------------------------------------------------
        # Save warm starts
        # -------------------------------------------------------------
        with torch.no_grad():
            if sample_idx is not None:
                self._warm_X_global[sample_idx] = X_opt.detach()

        return X_opt.detach(), values, converged
