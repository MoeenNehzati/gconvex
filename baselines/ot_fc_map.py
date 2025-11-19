import torch
from torch import nn
from baselines.ot import OT
from models import FiniteModel


class FCOT(OT):
    """
    Finitely-concave OT solver using the Kantorovich dual:

        D = sup_{u c-concave} E_mu[u(X)] + E_nu[u^c(Y)].

    Parameterization:
      - We represent u directly as a c-concave potential using a FiniteModel
        with kernel Φ(x,y) = c(x,y) and mode="concave".

    Dual objective:
        D = E_x[u(x)] + E_y[u^c(y)].

    This class:
      - Maximizes D with Adam on the parameters of u.
      - Uses FiniteModel.inf_transform with sample_idx for per-sample warm starts.
      - Supports stochastic minibatching in _fit.
      - Recovers the Monge map via the c-gradient of u with inverse_cx.
    
    Example:
        >>> from models import FiniteModel
        >>> from tools.utils import L22, inverse_grad_L22
        >>> 
        >>> # Create model
        >>> model = FiniteModel(
        ...     num_candidates=50,
        ...     num_dims=2,
        ...     kernel=lambda X, Y: L22(X, Y),
        ...     mode="concave"
        ... )
        >>> 
        >>> # Create FCOT solver
        >>> fcot = FCOT(
        ...     input_dim=2,
        ...     model=model,
        ...     inverse_cx=inverse_grad_L22,
        ...     lr=1e-3
        ... )
        >>> 
        >>> # Or use factory method to match parameter budget
        >>> fcot = FCOT.initialize_right_architecture(
        ...     dim=2,
        ...     n_params_target=1000,
        ...     cost=L22,
        ...     inverse_cx=inverse_grad_L22
        ... )
        >>> 
        >>> # Fit to data
        >>> import torch
        >>> X = torch.randn(100, 2)
        >>> Y = torch.randn(100, 2)
        >>> logs = fcot.fit(X, Y, iters=1000)
        >>> 
        >>> # Transport points
        >>> Y_pred = fcot.transport_X_to_Y(X)
    """

    # ----------------------------------------------------------------------
    @staticmethod
    def initialize_right_architecture(
        dim: int,
        n_params_target: int,
        cost,                        # Cost function c(x,y)
        inverse_cx,                  # Inverse gradient: (x,p) ↦ y solving ∇_x c(x,y) = p
        lr: float = 1e-3,
        betas=(0.5, 0.9),
        device: str = "cpu",
        inner_optimizer: str = "lbfgs",
        inner_steps: int = 5,
        inner_lam: float = 1e-3,
        inner_tol: float = 1e-3,
        inner_lr: float | None = None,
        temp: float = 50.0,
        is_cost_metric: bool = False,
        **kwargs
    ):
        """
        Factory method: creates FCOT instance matching a parameter budget.
        
        Automatically determines the number of candidates in the FiniteModel
        to approximately match the desired parameter count.
        
        Parameters
        ----------
        dim : int
            Input/output dimension
        n_params_target : int
            Target number of learnable parameters
        cost : callable
            Cost function c(X, Y) where X, Y have shape (batch, dim)
        inverse_cx : callable
            Maps (X, grad_c) → Y solving ∇_x c(X,Y) = grad_c
        lr : float, default=1e-3
            Learning rate for outer optimizer (Adam on model parameters)
        betas : tuple, default=(0.5, 0.9)
            Beta parameters for Adam optimizer
        device : str, default="cpu"
            Device to run on ("cpu" or "cuda")
        inner_optimizer : str, default="lbfgs"
            Inner loop optimizer: "lbfgs", "adam", or "gd"
        inner_steps : int, default=5
            Number of inner optimization steps for conjugate
        inner_lam : float, default=1e-3
            Regularization for inner solver
        inner_tol : float, default=1e-3
            Tolerance for inner solver convergence
        inner_lr : float or None, default=None
            Learning rate for inner solver (defaults to outer lr if None)
        temp : float, default=50.0
            Temperature for soft selection mode
        is_cost_metric : bool, default=False
            Whether the cost is a metric (unused currently, for future extensions)
        **kwargs
            Additional arguments (for compatibility)
            
        Returns
        -------
        FCOT
            Initialized FCOT solver with approximately n_params_target parameters
            
        Notes
        -----
        Parameter count formula for FiniteModel:
            n_params ≈ num_candidates * (dim + 1)
        
        where:
            - num_candidates * dim parameters for Y positions
            - num_candidates parameters for intercepts
        """
        # Estimate number of candidates needed
        # Each candidate has: dim parameters (position) + 1 parameter (intercept)
        params_per_candidate = dim + 1
        num_candidates = max(1, n_params_target // params_per_candidate)
        
        actual_params = num_candidates * params_per_candidate
        
        print(f"[FCOT ARCH] dim={dim}, target_params={n_params_target}")
        print(f"            num_candidates = {num_candidates}")
        print(f"            actual_params  = {actual_params}")
        
        # Build FiniteModel with kernel = cost (for c-concave representation)
        model = FiniteModel(
            num_candidates=num_candidates,
            num_dims=dim,
            kernel=lambda X, Y: cost(X, Y),
            mode="concave",
            temp=temp,
        ).to(device)
        
        return FCOT(
            input_dim=dim,
            model=model,
            inverse_cx=inverse_cx,
            lr=lr,
            betas=betas,
            device=device,
            inner_optimizer=inner_optimizer,
            inner_steps=inner_steps,
            inner_lam=inner_lam,
            inner_tol=inner_tol,
            inner_lr=inner_lr
        )

    # ----------------------------------------------------------------------
    def __init__(
        self,
        input_dim: int,
        model: nn.Module,        # FiniteModel(mode="concave", kernel = c)
        inverse_cx,              # (x,p) ↦ y solving ∇_x c(x,y) = p
        lr: float = 1e-3,
        betas=(0.5, 0.9),
        device: str = "cpu",
        inner_optimizer: str = "lbfgs",
        inner_steps: int = 5,
        inner_lam: float = 1e-3,
        inner_tol: float = 1e-3,
        inner_lr: float | None = None,   # NEW: separate LR for inner solver (optional)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.model = model.to(device)
        self.device = torch.device(device)

        self.inverse_cx = inverse_cx

        # Outer optimizer (on f parameters)
        self.lr = lr
        self.betas = betas
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)

        # Inner conjugate solver hyperparams
        self.inner_optimizer = inner_optimizer
        self.inner_steps = inner_steps
        self.inner_lam = inner_lam
        self.inner_tol = inner_tol
        # If inner_lr is None, fall back to outer lr (no behavior change)
        self.inner_lr = lr if inner_lr is None else inner_lr


    # ----------------------------------------------------------------------
    # Dual objective on a given batch
    # ----------------------------------------------------------------------
    def _dual_objective(self, X_batch, Y_batch, idx_y):
        """
        Compute on a minibatch:

            D = E_x[u(X)] + E_y[u^c(Y)].

        where:
          - u is represented by self.model.forward
          - u^c is computed by self.model.inf_transform

        Y_batch must be accompanied by idx_y (global indices into Y)
        so that FiniteModel can use per-sample warm starts for LBFGS/Adam.
        """
        # u(X)
        _, u_vals = self.model.forward(X_batch, selection_mode="soft")

        # u^c(Y): numerical inf_x [ c(x,y) - u(x) ]
        _, uc_vals = self.model.inf_transform(
            Z=Y_batch,
            sample_idx=idx_y,                # <--- crucial for warm-start
            steps=self.inner_steps,
            lr=self.inner_lr,                # <--- separate inner LR (tunable)
            optimizer=self.inner_optimizer,
            lam=self.inner_lam,
            tol=self.inner_tol,
        )

        u_mean = u_vals.mean()
        uc_mean = uc_vals.mean()

        D = u_mean + uc_mean
        return D, u_mean, uc_mean


    # ----------------------------------------------------------------------
    # One stochastic dual step
    # ----------------------------------------------------------------------
    def _step(self, X_batch, Y_batch, idx_y):
        """
        Perform one stochastic gradient step w.r.t. D on a minibatch.

        We *maximize* D, hence do gradient ascent: -D.backward() is equivalent to
        ascending on D.
        """
        self.optimizer.zero_grad()

        D, u_mean, uc_mean = self._dual_objective(X_batch, Y_batch, idx_y)
        # Maximize D by minimizing -D
        loss = -D
        loss.backward()

        # Optional: small, safe grad clipping (helps stability; negligible speed cost)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        self.optimizer.step()

        return {
            "dual": float(D.detach().item()),
            "u_mean": float(u_mean.detach().item()),
            "uc_mean": float(uc_mean.detach().item()),
        }


    # ----------------------------------------------------------------------
    # Monge map T(X): via ∇u and inverse_cx
    # ----------------------------------------------------------------------
    def transport_X_to_Y(self, X):
        """
        Recover the Monge map T(x) from the learned potential.

        Our model stores u as a c-concave function.

        For the optimal c-concave potential u, the map satisfies:

            ∇_x c(x, T(x)) = ∇u(x).

        So:
            T(x) = inverse_cx(x, ∇u(x)).
        """
        X = X.to(self.device).requires_grad_(True)

        _, u_vals = self.model.forward(X, selection_mode="soft")
        grad_u = torch.autograd.grad(u_vals.sum(), X)[0]

        with torch.no_grad():
            Y = self.inverse_cx(X, grad_u)

        return Y


    # ----------------------------------------------------------------------
    # Training loop: this is what OT.fit(...) will call
    # ----------------------------------------------------------------------
    def _fit(
        self,
        X,
        Y,
        batch_size: int = 512,
        iters: int = 2000,
        print_every: int = 50,
        callback=None,
        convergence_tol: float = 1e-4,
        convergence_patience: int = 50,
    ):
        """
        Stochastic training of the Kantorovich dual using random minibatches.

        X, Y: full datasets (num_x, dim), (num_y, dim).
        This method is called by OT.fit(...).

        Returns:
            logs: dict with keys "dual", "f", "g" tracking training trajectory.
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        nX, nY = X.shape[0], Y.shape[0]

        logs = {"dual": [], "u": [], "uc": []}

        prev_dual = None
        patience = 0

        for it in range(iters):
            # -------------------------------------------------------
            # Sample random mini-batches for X and Y
            # -------------------------------------------------------
            idx_x = torch.randint(0, nX, (batch_size,), device=self.device)
            idx_y = torch.randint(0, nY, (batch_size,), device=self.device)

            Xb = X[idx_x]
            Yb = Y[idx_y]

            stats = self._step(Xb, Yb, idx_y)

            # -------------------------------------------------------
            # Logging and convergence check
            # -------------------------------------------------------
            if it % print_every == 0:
                dual = stats["dual"]
                logs["dual"].append(dual)
                logs["u"].append(stats["u_mean"])
                logs["uc"].append(stats["uc_mean"])

                print(
                    f"[Iter {it}] dual={dual:.6f} "
                    f"u={stats['u_mean']:.4f} "
                    f"uc={stats['uc_mean']:.4f}"
                )

                if prev_dual is not None:
                    rel = abs(dual - prev_dual) / (abs(prev_dual) + 1e-12)
                    if rel < convergence_tol:
                        patience += 1
                    else:
                        patience = 0
                    if patience >= convergence_patience:
                        print(
                            f"[CONVERGED] dual rel change < {convergence_tol} "
                            f"for {convergence_patience} checks."
                        )
                        break

                prev_dual = dual

            if callback is not None:
                callback(it)

        return logs
