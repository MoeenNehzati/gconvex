from typing import Optional
from torch import nn
import torch
from tools.utils import moded_max, moded_min
from models.inf_convolution import InfConvolution
from tools.feedback import logger
from models.helpers import ZeroMean 

# =====================================================================
# FiniteSeparableModel: finitely-convex/concave on product kernels
# =====================================================================
class FiniteSeparableModel(nn.Module):

    def __init__(
        self,
        kernel,
        num_dims: int,
        radius: float,
        y_accuracy: float = 1e-2,
        x_accuracy: float = 1e-2,
        mode: str = "convex",
        temp: float = 50.0,
        epsilon: float = 1e-4,
        cache_gradients: bool = False,
        coarse_x_factor: int = None,
        coarse_top_k: int = 1,
        coarse_window: int = 0,
    ):
        super().__init__()
        assert mode in ("convex", "concave")
        self.kernel_fn = kernel
        self.num_dims = num_dims
        self.radius = radius
        self.y_accuracy = y_accuracy
        self.x_accuracy = x_accuracy
        self.mode = mode
        self.temp = temp
        self.epsilon = epsilon
        self.cache_gradients = cache_gradients

        # ---------------------------------------------------------------
        # Build X/Y grids
        # ---------------------------------------------------------------
        ny = int(2 * radius / y_accuracy) + 1
        nx = int(2 * radius / x_accuracy) + 1
        Y_grid = torch.linspace(-radius, radius, ny)
        X_grid = torch.linspace(-radius, radius, nx)

        kernel_tensor = kernel(
            X_grid.reshape(-1, 1),
            Y_grid.reshape(1, -1)
        )  # shape (nx, ny)

        self.register_buffer("Y_grid", Y_grid)
        self.register_buffer("X_grid", X_grid)
        self.register_buffer("kernel_tensor", kernel_tensor)

        # ---------------------------------------------------------------
        # Cache k_x, k_y derivatives if requested
        # ---------------------------------------------------------------
        if cache_gradients:
            dx_tensor, dy_tensor = self._precompute_kernel_derivatives()
        else:
            dx_tensor = torch.empty(0)
            dy_tensor = torch.empty(0)

        self.register_buffer("kernel_dx", dx_tensor)
        self.register_buffer("kernel_dy", dy_tensor)

        # ---------------------------------------------------------------
        # INTERCEPT PARAMETRIZATION (ZeroMean)
        # ---------------------------------------------------------------
        # ny b-values but only ny−1 parameters (theta)
        self.intercepts_param = ZeroMean(nrows=ny - 1, ndims=num_dims)

        # ---------------------------------------------------------------
        # Coarse search parameters
        # ---------------------------------------------------------------
        stride = coarse_x_factor if (coarse_x_factor and coarse_x_factor > 1) else None
        if stride:
            coarse_idx = torch.arange(0, nx, stride, dtype=torch.long)
            if coarse_idx[-1] != nx - 1:
                coarse_idx = torch.cat([coarse_idx, coarse_idx.new_tensor([nx - 1])])

            coarse_top_k = max(1, coarse_top_k)
            coarse_window = max(0, coarse_window)
            window_radius = coarse_window * stride
            offsets = torch.arange(-window_radius, window_radius + 1, dtype=torch.long)
        else:
            coarse_idx = torch.empty(0, dtype=torch.long)
            offsets = torch.empty(0, dtype=torch.long)
            coarse_top_k = 0
            coarse_window = 0

        self.register_buffer("coarse_idx", coarse_idx)
        self.register_buffer("coarse_offsets", offsets)
        self.coarse_stride = stride if stride else 1
        self.coarse_top_k = coarse_top_k
        self.coarse_window = coarse_window
        self.use_coarse_search = coarse_idx.numel() > 0


    # =====================================================================
    # Intercept refresh (c-conjugacy refresh)
    # =====================================================================
    def refresh_intercepts_via_transform(self):
        pass
        # K = self.kernel_tensor.to(self.intercepts_param.theta.device)
        # b_old = self.intercepts_param.value        # (ny, d)
        # b_new = torch.zeros_like(b_old)
        # ny, d = b_old.shape

        # for dim in range(d):
        #     column = b_old[:, dim]

        #     scores = K - column.unsqueeze(0)
        #     if self.mode == "concave":
        #         u_grid = scores.min(dim=1).values
        #         diff = K - u_grid.unsqueeze(1)
        #         refreshed = diff.min(dim=0).values
        #     else:
        #         u_grid = scores.max(dim=1).values
        #         diff = K - u_grid.unsqueeze(1)
        #         refreshed = diff.max(dim=0).values

        #     b_new[:, dim] = refreshed

        # # Write back consistently in ZeroMean gauge
        # self.intercepts_param.project_from_b(b_new)

        # changed = (b_new - b_old).abs().gt(1e-12).sum().item()
        # return changed


    # =====================================================================
    # Reactivation step
    # =====================================================================
    def reactivation_step(self, dead_mask, eps):

        with torch.no_grad():
            b = self.intercepts_param.value.clone()
            b[dead_mask] += eps
            self.intercepts_param.project_from_b(b)

    # =====================================================================
    # Utility clamp
    # =====================================================================
    def project(self, T):
        return torch.clamp(T, -self.radius + self.epsilon, self.radius - self.epsilon)

    # =====================================================================
    # Cached derivative precomputation
    # =====================================================================
    def _precompute_kernel_derivatives(self):
        device = self.X_grid.device
        dtype = self.X_grid.dtype

        # dk/dx on grid
        dx_list = []
        for y in self.Y_grid:
            x_vals = self.X_grid.clone().detach().requires_grad_(True)
            v = self.kernel_fn(x_vals.unsqueeze(-1),
                               y.to(device=device, dtype=dtype).view(1, 1)).squeeze()
            g = torch.autograd.grad(
                v, x_vals,
                grad_outputs=torch.ones_like(v),
                create_graph=False,
                retain_graph=False
            )[0]
            dx_list.append(g.detach())
        kernel_dx = torch.stack(dx_list, dim=1)

        # dk/dy on grid
        dy_list = []
        for x in self.X_grid:
            y_vals = self.Y_grid.clone().detach().requires_grad_(True)
            v = self.kernel_fn(
                x.to(device=device, dtype=dtype).view(1, 1),
                y_vals.unsqueeze(0)
            ).squeeze()
            g = torch.autograd.grad(
                v, y_vals,
                grad_outputs=torch.ones_like(v),
                retain_graph=False,
                create_graph=False
            )[0]
            dy_list.append(g.detach())
        kernel_dy = torch.stack(dy_list, dim=0)

        return kernel_dx, kernel_dy



    # =====================================================================
    # Forward: per-dimension finitely-convex or finitely-concave function
    # =====================================================================
    def _per_dim_forward(self, X, along=0, selection_mode="soft"):

        x_vals = X[:, along]
        x_vals_clamped = self.project(x_vals)

        if selection_mode == "hard":
            x_idx = self.get_indices_for_x(x_vals_clamped.detach()).long()
            kernel_scores = self.kernel_tensor[x_idx, :]
        else:
            kernel_scores = self.kernel_fn(
                x_vals_clamped.unsqueeze(-1),
                self.Y_grid.unsqueeze(0)
            )

        # ==============================================================
        # FIX: direct ZeroMean access
        # ==============================================================

        b = self.intercepts_param.value[:, along].unsqueeze(0)
        scores = kernel_scores - b
        candidates = self.Y_grid.view(1, -1, 1)

        if self.mode == "convex":
            choice, f_x, aux = moded_max(scores, candidates, dim=1,
                                         temp=self.temp,
                                         mode=selection_mode)
        else:
            choice, f_x, aux = moded_min(scores, candidates, dim=1,
                                         temp=self.temp,
                                         mode=selection_mode)

        choice = choice.squeeze(-1)

        # STE hard-gradient patch
        if selection_mode == "hard" and x_vals.requires_grad:
            if self.cache_gradients and self.kernel_dx.numel() > 0:
                approx = self._cached_dx(x_vals_clamped, choice.detach())
            else:
                tmp = []
                for xs, ys in zip(x_vals_clamped, choice.detach()):
                    tmp.append(self._kernel_grad_wrt_x(xs, ys))
                approx = torch.stack(tmp, dim=0)
            f_x = f_x + approx * (x_vals_clamped - x_vals_clamped.detach())

        return choice, f_x


    def forward(self, X, selection_mode="soft"):
        args = []
        vals = []
        for dim in range(self.num_dims):
            arg, val = self._per_dim_forward(X, along=dim, selection_mode=selection_mode)
            args.append(arg)
            vals.append(val)

        f_x = torch.stack(vals, dim=1).sum(dim=1)
        choice = torch.stack(args, dim=1)
        return choice, f_x



    # =====================================================================
    # Transform core (unchanged, intercept reads patched)
    # =====================================================================
    def sup_transform(self, Z):
        return self._transform_core(Z, maximize=True)

    def inf_transform(self, Z):
        return self._transform_core(Z, maximize=False)

    def _transform_core(self, Z, maximize=True):
        num_samples, num_dims = Z.shape
        device = Z.device
        dtype = Z.dtype

        X_points = []
        total_values = torch.zeros(num_samples, device=device, dtype=dtype)

        for d in range(num_dims):
            x_d, value_d = self._per_dimension_transform_batch(Z[:, d], along=d,
                                                               maximize=maximize)
            X_points.append(x_d)
            total_values += value_d

        X_opt = torch.stack(X_points, dim=1)
        return X_opt.detach(), total_values, True



    # =====================================================================
    # Per-dimension transform
    # =====================================================================
    def _per_dimension_transform_batch(self, z_vals, along, maximize):

        device = z_vals.device
        dtype = z_vals.dtype

        z_clamped = self.project(z_vals)
        z_idx = self.get_indices_for_y(z_clamped.detach()).long()

        kernel_vals = self.kernel_tensor[:, z_idx]

        # ==============================================================
        # FIX: direct ZeroMean access
        # ==============================================================
        b_col = self.intercepts_param.value[:, along]

        rows = self.kernel_tensor
        scores = rows - b_col.unsqueeze(0)

        if self.mode == "convex":
            f_subset = scores.max(dim=1).values
        else:
            f_subset = scores.min(dim=1).values

        objective_all = kernel_vals - f_subset.unsqueeze(1)
        nx, batch = objective_all.shape

        if not self.use_coarse_search:
            if maximize:
                best_pos = objective_all.argmax(dim=0)
            else:
                best_pos = objective_all.argmin(dim=0)

            best_values = objective_all[best_pos, torch.arange(batch, device=device)]
            best_idx = best_pos
        else:
            coarse_idx = self.coarse_idx
            coarse_obj = objective_all.index_select(0, coarse_idx)
            k = min(self.coarse_top_k, coarse_obj.size(0))

            if maximize:
                _, top_pos = torch.topk(coarse_obj, k=k, dim=0, largest=True)
            else:
                _, top_pos = torch.topk(-coarse_obj, k=k, dim=0, largest=False)

            seed_idx = coarse_idx[top_pos]

            if self.coarse_offsets.numel() == 0:
                refine_idx = seed_idx.view(-1, batch)
            else:
                offsets = self.coarse_offsets.view(1, 1, -1)
                refine_idx = seed_idx.unsqueeze(-1) + offsets
                refine_idx = refine_idx.view(-1, batch)

            refine_idx = refine_idx.clamp(0, nx - 1)
            candidate_obj = objective_all.gather(0, refine_idx)

            if maximize:
                best_rows = candidate_obj.argmax(dim=0)
            else:
                best_rows = candidate_obj.argmin(dim=0)

            best_values = candidate_obj[best_rows, torch.arange(batch, device=device)]
            best_idx = refine_idx[best_rows, torch.arange(batch, device=device)]

        x_opt = self.X_grid[best_idx].to(device=device, dtype=dtype)
        values = best_values

        # gradient wrt z
        if z_vals.requires_grad:
            if self.cache_gradients and self.kernel_dy.numel() > 0:
                approx_grad = self._cached_dy(best_idx, z_clamped)
            else:
                tmp = []
                for xo, zc in zip(x_opt, z_clamped):
                    tmp.append(self._kernel_grad_wrt_y(xo, zc))
                approx_grad = torch.stack(tmp, dim=0)

            values = values + approx_grad * (z_clamped - z_clamped.detach())

        return x_opt, values



    # =====================================================================
    # X/Y index helpers
    # =====================================================================
    def get_indices_for_x(self, X):
        idx = ((X + self.radius) / self.x_accuracy).round().long()
        nx = self.X_grid.numel()
        return idx.clamp(0, nx - 1)

    def get_indices_for_y(self, Y):
        idx = ((Y + self.radius) / self.y_accuracy).round().long()
        ny = self.Y_grid.numel()
        return idx.clamp(0, ny - 1)



    # =====================================================================
    # Cached grads interpolation
    # =====================================================================
    def _cached_dx(self, x_vals, y_vals):
        if self.kernel_dx.numel() == 0:
            raise RuntimeError("Cached dx requested but kernel_dx empty.")

        nx = self.X_grid.numel()
        ny = self.Y_grid.numel()

        x_pos = (x_vals + self.radius) / self.x_accuracy
        x_low = torch.floor(x_pos).long().clamp(0, nx - 1)
        x_high = torch.clamp(x_low + 1, 0, nx - 1)
        weight = (x_pos - x_low.float()).clamp(0.0, 1.0)

        y_idx = ((y_vals + self.radius) / self.y_accuracy).round().long().clamp(0, ny - 1)

        g_low = self.kernel_dx[x_low, y_idx]
        g_high = self.kernel_dx[x_high, y_idx]
        return g_low + (g_high - g_low) * weight

    def _cached_dy(self, x_idx, y_vals):
        if self.kernel_dy.numel() == 0:
            raise RuntimeError("Cached dy requested but kernel_dy empty.")

        ny = self.Y_grid.numel()
        y_pos = (y_vals + self.radius) / self.y_accuracy
        y_low = torch.floor(y_pos).long().clamp(0, ny - 1)
        y_high = torch.clamp(y_low + 1, 0, ny - 1)
        weight = (y_pos - y_low.float()).clamp(0.0, 1.0)

        g_low = self.kernel_dy[x_idx, y_low]
        g_high = self.kernel_dy[x_idx, y_high]
        return g_low + (g_high - g_low) * weight


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
        patience: int = 5,
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
                patience,
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
