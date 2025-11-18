"""
model.py
---------
Implements finitely convex functions and their kernel-conjugates.

We represent a finitely k-convex function as

    f(x) = max_j [ k(x, y_j) - b_j ],

where the user supplies a *surplus* function:

    surplus(x, y) : (..., #dim), (..., #dim) -> (...,)

and we simply take

    k(x,y) = surplus(x,y).

The kernel-conjugate is

    f^k(z) = max_x [ k(x,z) - f(x) ]
           = - min_x [ f(x) - k(x,z) ].

Since f(x) is a max over affine functions k(x,y_j)-b_j and k(x,z) is
supplied by the user, the inner optimization is convex in x whenever
-x ↦ k(x,z) is convex (e.g. many OT costs such as p-norms^p).

Features
--------
- Soft/Hard/STE maximization via moded_max
- Optional learnable vs fixed Y support (is_y_parameter)
- Optional bounded/recentered initialization for Y (y_min, y_max)
- Efficient LBFGS/Adam/GD inner conjugate solver
- Per-sample warm starts for stochastic minibatches
"""

import torch
from torch import nn
from typing import Optional, Tuple
from tools.utils import moded_max

# ============================================================
# Finitely Convex Model
# ============================================================
import torch
from torch import nn
from typing import Optional, Tuple
from tools.utils import moded_max, moded_min


# =====================================================================
# FiniteModel: unified finitely-convex / finitely-concave representation
# =====================================================================

class FiniteModel(nn.Module):
    """
    Unified finite model representing either finitely convex or finitely concave functions.
    
    This class provides a flexible representation for functions of the form:
    
        **Finitely k-CONVEX** (mode="convex"):
            f(x) = max_j [ k(x, y_j) - b_j ]
    
        **Finitely k-CONCAVE** (mode="concave"):
            f(x) = min_j [ k(x, y_j) - b_j ]
    
    where:
    - k(x, y) is a user-supplied kernel function
    - {y_j} are support points (learnable or fixed)
    - {b_j} are intercept parameters (always learnable)
    
    The class supports three selection modes via moded_max/moded_min:
    - **soft**: Temperature-scaled softmax/softmin (fully differentiable)
    - **hard**: Argmax/argmin selection (discrete, no gradient through selection)
    - **ste**: Straight-through estimator (hard forward, soft backward)
    
    Additionally, the class provides numerical computation of kernel transforms:
    - **sup_transform(z)**: Computes sup_x [k(x, z) - f(x)] via optimization
    - **inf_transform(z)**: Computes inf_x [k(x, z) - f(x)] via optimization
    
    These transforms are related to conjugate functions in convex analysis:
    - For surplus Φ(x,y) = -c(x,y), the sup-transform gives the Φ-conjugate
    - For cost c(x,y), the inf-transform relates to the c-conjugate
    
    **Mathematical Properties**:
    
    When the kernel has special structure, interesting properties emerge:
    
    1. **Metric kernel**: If k(x,y) = d(x,y) is a metric, and f(x) = min_j d(x,y_j),
       then inf_x[d(x,z) - f(x)] relates to the negative of f evaluated at z.
    
    2. **Negative metric**: If k(x,y) = -d(x,y) where d is a metric, and 
       f(x) = max_j[-d(x,y_j)], then sup_x[-d(x,z) - f(x)] has similar properties.
    
    These properties are verified in the test suite (see tests/test_model.py).
    
    Attributes:
        num_candidates: Number of support points y_j.
        num_dims: Dimensionality of input/output space.
        mode: Either "convex" (use max) or "concave" (use min).
        temp: Temperature parameter for soft selection.
        is_y_parameter: Whether support points Y are learnable parameters.
        kernel_fn: The kernel function k(x,y).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        num_candidates: int,
        num_dims: int,
        kernel,                    # function: kernel(x,y)
        mode: str = "convex",      # "convex" or "concave"
        temp: float = 50.0,
        is_y_parameter: bool = True,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        original_dist_to_bounds: float = 1e-1,
        is_there_default: bool = False,
    ):
        """
        Parameters
        ----------
        num_candidates : int
            Number of support points y_j.
        num_dims : int
            Dimensionality of x and y.
        kernel : callable
            Kernel k(x,y): shapes (num_samples,1,num_dims) and (1,num_candidates,num_dims)
                           → (num_samples,num_candidates).
        mode : {"convex", "concave"}
            Whether f(x) is max or min over candidates.
        temp : float
            Soft/hard selection temperature.
        is_y_parameter : bool
            Whether Y is learnable.
        y_min, y_max : floats
            Optional bounding box for initialization.
        is_there_default : bool
            Whether to include y=0,b=0 as an extra candidate.
        """
        super().__init__()
        assert mode in ("convex", "concave"), "mode must be 'convex' or 'concave'."

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
        sampled_min, sampled_max = Y_init.min(), Y_init.max()

        if (y_min is not None) and (y_max is not None):
            centered = Y_init - Y_init.mean()
            max_abs = centered.abs().max() + 1e-4
            mid = 0.5 * (y_min + y_max)
            half_range = 0.5 * (y_max - y_min)
            Y_init = mid + (half_range * centered / max_abs)
        elif y_min is not None:
            Y_init = Y_init - sampled_min + (y_min + original_dist_to_bounds)
        elif y_max is not None:
            Y_init = sampled_max - Y_init + (y_max - original_dist_to_bounds)

        if is_y_parameter:
            self.Y_rest = nn.Parameter(Y_init)
        else:
            self.register_buffer("Y_rest", Y_init)

        self.intercept_rest = nn.Parameter(torch.zeros(1, num_candidates))

        # Optional default candidate y=0,b=0
        self.register_buffer("Y0", torch.zeros(1,1,num_dims))
        self.register_buffer("intercept0", torch.zeros(1,1))

        # Warm-start storage for sup/inf transform
        self._warm_X = None


    # ============================================================
    # Helpers: constructing candidate sets
    # ============================================================

    def full_Y(self):
        """Return full candidate set (default + learned)."""
        if self.is_there_default:
            return torch.cat([self.Y0, self.Y_rest], dim=1)
        return self.Y_rest

    def full_intercept(self):
        """Return full intercept vector."""
        if self.is_there_default:
            return torch.cat([self.intercept0, self.intercept_rest], dim=1)
        return self.intercept_rest


    # ============================================================
    # Forward: MAX or MIN over candidates using moded_max/min
    # ============================================================

    def forward(self, X: torch.Tensor, selection_mode: str = "soft"):
        """
        Compute f(x) using either max or min over candidates.
        
        Depending on the mode:
        - **convex**: f(x) = max_j [ k(x,y_j) - b_j ]
        - **concave**: f(x) = min_j [ k(x,y_j) - b_j ]
        
        The selection can be soft (differentiable weighted combination),
        hard (discrete argmax/argmin), or ste (straight-through estimator).
        
        Args:
            X: Input tensor of shape (num_samples, num_dims).
            selection_mode: One of "soft", "hard", or "ste".
                - "soft": Weighted combination using temperature-scaled softmax/softmin.
                  Fully differentiable with smooth gradients.
                - "hard": Discrete selection (argmax for convex, argmin for concave).
                  Forward pass is discrete, no gradients through selection.
                - "ste": Straight-through estimator. Hard selection in forward pass,
                  but gradients from soft selection in backward pass.
        
        Returns:
            choice: Tensor of shape (num_samples, num_dims).
                The selected support point(s). In soft mode, this is a weighted
                combination. In hard mode, it's a single selected y_j per sample.
            f_x: Tensor of shape (num_samples,).
                Function values f(x) for each input sample.
        
        Side Effects:
            Updates internal diagnostic attributes:
            - self._last_weights: Softmax/softmin weights (soft/ste only).
            - self._last_idx: Argmax/argmin indices (hard/ste only).
            - self._last_mean_max_weight: Mean of maximum weight per sample.
            - self._last_eff_temp: Effective temperature used (soft/ste only).
        
        Examples:
            >>> model = FiniteModel(num_candidates=5, num_dims=2, 
            ...                     kernel=lambda x,y: -((x-y)**2).sum(-1),
            ...                     mode="convex")
            >>> X = torch.randn(10, 2)
            >>> choice, f_x = model.forward(X, selection_mode="soft")
            >>> print(f_x.shape)  # (10,)
            >>> print(model._last_weights.sum(dim=1))  # All close to 1.0
        """
        Y = self.full_Y()                      # (1, num_candidates, num_dims)
        b = self.full_intercept()              # (1, num_candidates)

        scores = self.kernel_fn(X[:,None,:], Y) - b  # (num_samples, num_candidates)

        if self.mode == "convex":
            choice, f_x, aux = moded_max(scores, Y, dim=1, temp=self.temp, mode=selection_mode)
        else:
            choice, f_x, aux = moded_min(scores, Y, dim=1, temp=self.temp, mode=selection_mode)

        # Diagnostics
        self._last_mean_max_weight = aux.get("mean_max")
        self._last_idx = aux.get("idx")
        self._last_weights = aux.get("weights")
        self._last_eff_temp = aux.get("eff_temp")

        return choice, f_x


    # ============================================================
    # INTERNAL: closure for optimization over x
    # ============================================================

    def _closure_x(self, X_var, Z, maximize: bool, lam: float, optimizer_obj):
        """
        Optimize x → k(x,z) − f(x).

        maximize=True  → sup_x
        maximize=False → inf_x
        """
        optimizer_obj.zero_grad()

        # f(x)
        _, f_x = self.forward(X_var, selection_mode="soft")

        # k(x,z)
        k_x_z = self.kernel_fn(X_var, Z)

        objective = (k_x_z - f_x).mean()
        reg = 0.5 * lam * (X_var.pow(2).sum(dim=-1)).mean()

        if maximize:
            # maximize objective → minimize -(obj - reg)
            loss = -(objective - reg)
        else:
            # minimize objective → minimize (obj + reg)
            loss = objective + reg

        loss.backward()
        return loss


    # ============================================================
    # Numerical conjugates: sup-transform / inf-transform
    # ============================================================

    def sup_transform(
        self,
        Z: torch.Tensor,
        steps: int = 50,
        lr: float = 1e-3,
        optimizer: str = "lbfgs",
        lam: float = 1e-3,
        tol: float = 1e-3,
    ):
        """
        Compute the supremum transform: sup_x [ k(x,z) - f(x) ] for each z.
        
        This is a numerical optimization over x to find the supremum. The result
        is related to the kernel-conjugate of f in convex analysis.
        
        For surplus kernels Φ(x,y) = -c(x,y), this computes the Φ-conjugate:
            f^Φ(z) = sup_x [ Φ(x,z) - f(x) ]
        
        **Mathematical Property** (verified in tests):
        When k(x,y) = -d(x,y) where d is a metric, and f(x) = max_j[-d(x,y_j)],
        the sup-transform satisfies special properties. In particular, for z = y_j
        (a support point), sup_x[-d(x,z) - f(x)] ≈ -f(z) ≈ 0.
        
        Args:
            Z: Points at which to evaluate the transform, shape (num_samples, num_dims).
            steps: Maximum number of optimization steps.
            lr: Learning rate for the optimizer.
            optimizer: Optimization algorithm - "lbfgs", "adam", or "gd".
                - "lbfgs": L-BFGS with line search (generally fastest, best for smooth problems).
                - "adam": Adaptive moment estimation (good default, handles noise well).
                - "gd": Gradient descent (simplest, may need more steps).
            lam: L2 regularization coefficient on x (prevents unbounded solutions).
            tol: Convergence tolerance for gradient/change (early stopping).
        
        Returns:
            X_star: Tensor of shape (num_samples, num_dims).
                Optimal x values that achieve the supremum for each z.
            values: Tensor of shape (num_samples,).
                The supremum values: sup_x[k(x,z) - f(x)] for each z.
        
        Implementation Notes:
            - Uses warm-starting: previous solution initializes next optimization.
            - Adds small L2 regularization to ensure bounded solutions.
            - Automatically switches between LBFGS and iterative optimizers.
        
        Examples:
            >>> model = FiniteModel(num_candidates=10, num_dims=2,
            ...                     kernel=lambda x,y: -((x-y)**2).sum(-1),
            ...                     mode="convex")
            >>> Z = torch.randn(5, 2)
            >>> X_star, values = model.sup_transform(Z, steps=50, optimizer="adam")
            >>> print(values.shape)  # (5,)
        """
        return self._transform_core(Z, maximize=True, steps=steps,
                                    lr=lr, optimizer=optimizer,
                                    lam=lam, tol=tol)

    def inf_transform(
        self,
        Z: torch.Tensor,
        steps: int = 50,
        lr: float = 1e-3,
        optimizer: str = "lbfgs",
        lam: float = 1e-3,
        tol: float = 1e-3,
    ):
        """
        Compute the infimum transform: inf_x [ k(x,z) - f(x) ] for each z.
        
        This is a numerical optimization over x to find the infimum. The result
        is related to the negative kernel-conjugate.
        
        For cost functions c(x,y), with kernel k(x,y) = -c(x,y), this is related to:
            -f^c(z) where f^c(z) = inf_x [ c(x,z) - f(x) ]
        
        **Mathematical Property** (verified in tests):
        When k(x,y) = d(x,y) is a metric, and f(x) = min_j d(x,y_j), the
        inf-transform has special properties. For z = y_j (a support point),
        inf_x[d(x,z) - f(x)] ≈ -f(z) ≈ 0.
        
        Args:
            Z: Points at which to evaluate the transform, shape (num_samples, num_dims).
            steps: Maximum number of optimization steps.
            lr: Learning rate for the optimizer.
            optimizer: Optimization algorithm - "lbfgs", "adam", or "gd".
            lam: L2 regularization coefficient on x.
            tol: Convergence tolerance for gradient/change.
        
        Returns:
            X_star: Tensor of shape (num_samples, num_dims).
                Optimal x values that achieve the infimum for each z.
            values: Tensor of shape (num_samples,).
                The infimum values: inf_x[k(x,z) - f(x)] for each z.
        
        Relationship to sup_transform:
            inf_x[k(x,z) - f(x)] = -sup_x[-k(x,z) + f(x)]
        
        Examples:
            >>> model = FiniteModel(num_candidates=10, num_dims=2,
            ...                     kernel=lambda x,y: ((x-y)**2).sum(-1).sqrt(),
            ...                     mode="concave")
            >>> Z = torch.randn(5, 2)
            >>> X_star, values = model.inf_transform(Z, steps=50, optimizer="lbfgs")
            >>> print(values.shape)  # (5,)
        """
        return self._transform_core(Z, maximize=False, steps=steps,
                                    lr=lr, optimizer=optimizer,
                                    lam=lam, tol=tol)


    # ============================================================
    # Shared numerical routine for sup/inf-conjugates
    # ============================================================

    def _transform_core(
        self,
        Z: torch.Tensor,
        maximize: bool,
        steps: int,
        lr: float,
        optimizer: str,
        lam: float,
        tol: float,
    ):
        """
        Internal routine for sup/inf transform.

        maximize=True  → sup_x [k(x,z) − f(x)]
        maximize=False → inf_x [k(x,z) − f(x)]
        """
        num_samples, num_dims = Z.shape

        # Warm start
        if self._warm_X is None or self._warm_X.shape[0] != num_samples:
            X_init = Z.clone()
        else:
            X_init = self._warm_X.clone()

        X_var = X_init.detach().clone().requires_grad_(True)

        # ------------------------------
        # Choose optimizer (LBFGS / Adam)
        # ------------------------------
        opt_name = optimizer.lower()
        if opt_name == "lbfgs":
            optimizer_obj = torch.optim.LBFGS(
                [X_var],
                lr=lr,
                max_iter=steps,
                tolerance_grad=tol,
                tolerance_change=tol,
                line_search_fn="strong_wolfe"
            )
            optimizer_obj.step(lambda: self._closure_x(X_var, Z, maximize, lam, optimizer_obj))

        else:
            if opt_name == "adam":
                optimizer_obj = torch.optim.Adam([X_var], lr=lr)
            elif opt_name == "gd":
                optimizer_obj = torch.optim.SGD([X_var], lr=lr)
            else:
                raise ValueError(f"Unknown optimizer {optimizer}")

            prev_loss = None
            for _ in range(steps):
                loss = self._closure_x(X_var, Z, maximize, lam, optimizer_obj)
                optimizer_obj.step()
                if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
                    break
                prev_loss = loss.item()

        # ------------------------------
        # Evaluate conjugate value
        # ------------------------------
        with torch.no_grad():
            _, f_x = self.forward(X_var)
            k_x_z = self.kernel_fn(X_var, Z)
            values = k_x_z - f_x
            self._warm_X = X_var.detach()

        return X_var.detach(), values.detach()
