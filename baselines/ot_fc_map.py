import torch
import torch.nn.functional as F  # (may be unused but left for consistency)
from config import WRITING_ROOT     # (may be unused but left for consistency)
from baselines.ot import OT
from model import FinitelyConvexModel
from torch.utils.data import DataLoader, TensorDataset

class FCOT(OT):
    """
    Finitely-convex OT solver using the (-c)-convex Kantorovich dual.

    We represent a (-c)-convex potential f with a FinitelyConvexModel using
    surplus(x, y) = -c(x, y). The associated (-c)-dual functional is:

        J(f) = E_mu[f(X)] + E_nu[f^{(-c)}(Y)],

    where f^{(-c)} is the sup-transform with respect to the surplus kernel.

    - We **minimize** J(f) with Adam.
    - The OT cost is approximated by:

          OT(mu, nu) ≈ -min J(f).

    Logging convention:
      - `loss`   = J(f)       (what we minimize)
      - `dual`   = -J(f)      (estimate of the Kantorovich value, ~ OT cost)
      - `f_mean` = E[f(X)]
      - `g_mean` = E[f^{(-c)}(Y)]
    """

    def __init__(
        self,
        input_dim,
        ncandidates,
        cost,
        inverse_cx,
        lr: float = 1.,
        betas=(0.5, 0.9),
        device: str = "cpu",
        inner_optimizer: str = "lbfgs",
        is_cost_metric: bool = False,
    ):
        # surplus(x,y) = -c(x,y)
        surplus = lambda x, y: -cost(x, y)

        self.input_dim   = input_dim
        self.ncandidates = ncandidates
        self.is_cost_metric = is_cost_metric
        self.lr = lr
        self.inner_optimizer = inner_optimizer

        # Finitely convex potential f(x) = max_j k(x, y_j) - b_j with k = surplus
        self.model = FinitelyConvexModel(
            ncandidates=self.ncandidates,
            dim=self.input_dim,
            surplus=surplus,
            temp=50.0,
            is_there_default=False,
            is_y_parameter=True,        # Y is trainable
            is_cost_metric=self.is_cost_metric,        # kept for compatibility; may be unused internally
        ).to(device)

        # inverse_cx(x, p) returns y solving ∇_x c(x, y) = p
        self.inverse_cx = inverse_cx
        self.device     = torch.device(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=betas
        )
        self.betas = betas

    ############################################################
    # Architecture helper
    ############################################################
    @staticmethod
    def initialize_right_architecture(
        dim,
        n_params_target,
        cost,
        inverse_cx,
        lr: float = 1.,
        betas=(0.5, 0.9),
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        Heuristically choose the number of candidate Y points (ncandidates)
        so that the total number of parameters is around `n_params_target`.
        """
        ncandidates = n_params_target // (dim + 1)

        return FCOT(
            input_dim=dim,
            ncandidates=ncandidates,
            cost=cost,
            inverse_cx=inverse_cx,
            lr=lr,
            betas=betas,
            device=device,
            *args,
            **kwargs,
        )

    ############################################################
    # Core dual objective and training utilities
    ############################################################
    def _dual_objective(self, X, Y, inner_steps):
        """
        Compute the Φ-dual functional:

            J(f) = E_mu[f(X)] + E_nu[f^{Φ}(Y)],

        where:
          - model.forward(X)   returns f(X)  (a Φ-convex potential),
          - model.conjugate(Y) returns f^{Φ}(Y) (its sup-transform).
        Returns:
          d  = -J(f)     (scalar tensor),
          j  = J(f)         (scalar tensor),
          fmean = E[f(X)],
          gmean = E[f^{Φ}(Y)].
        """
        # f(X): (-c)-convex potential
        _, f_vals = self.model.forward(X,
                                       selection_mode="soft")
        _, g_vals = self.model.conjugate(
            Z=Y,
            sample_idx=None,
            selection_mode="soft",
            steps=inner_steps,
            optimizer=self.inner_optimizer,
            lr=self.lr,
        )  # (#samples,)

        fmean = f_vals.mean()
        gmean = g_vals.mean()
        j  = fmean + gmean
        d = -j      # J(f)
        return d, j, fmean, gmean

    def step(self, x_batch, y_batch, inner_steps):
        """
        Single gradient step maximizing the kantorovitch dual
        We have definve the surplus Φ = -c
        For cost c we define f^c(y) = inf_x [c(x,y) - f(x)]
        For surplus Φ we define f^Φ(y) = sup_x [Φ(x,y) - f(x)]
        We have f^c = -(-f)^Φ and f^Φ = -(-f)^c
        If f is c-concave, then -f is Φ-convex

        The Kantorovitch dual D is:
        D = sup_{f finitely c-concave} E_x[f(X)] + E_y[f^c(Y)]
        D = -inf_{f finitely c-concave} E_x[-f(X)] + E_y[-f^c(Y)]
        setting g = -f we have
        D = - inf_{g finitely Φ-convex} E_x[g(X)] + E_y[g^Φ(Y)]
        So we need to minimize J(g) = E_x[g(X)] + E_y[g^Φ(Y)]

        Returns a dict of detached scalars for logging:
          - J: J(f)
          - dual: -J(f)     (estimate of the OT cost)
          - f_mean: E[f(X)]
          - g_mean: E[f^Φ(Y)]
        """
        self.optimizer.zero_grad()
        d, j, f_mean, g_mean = self._dual_objective(x_batch, y_batch, inner_steps)
        j.backward()
        self.optimizer.step()

        return {
            "dual":  d.detach().item(),  # OT estimate
            "J":   j.detach().item(),
            "f_mean": float(f_mean.detach().item()),
            "g_mean": float(g_mean.detach().item()),
        }

    def transport_X_to_Y(self, X):
        """
        Compute the Monge map T(X) using the c-gradient of the c-convex potential u.

        Model represents a (-c)-convex f. We set:

            u(x) = -f(x)   (c-convex),

        and the optimal map satisfies:

            ∇_x c(x, T(x)) = ∇u(x).

        Given an oracle inverse_cx(x, p) solving ∇_x c(x, y) = p,
        we obtain T(x) as:

            T(x) = inverse_cx(x, ∇u(x)).
        """
        X = X.to(self.device)
        X = X.requires_grad_(True)
        _, f_vals = self.model.forward(X, selection_mode="soft")
        grad_f = torch.autograd.grad(f_vals.sum(), X, create_graph=False)[0]
        with torch.no_grad():
            Y = self.inverse_cx(X, -grad_f)
        return Y

    def debug_losses(self, X, Y):
        """
        Convenience helper for quick diagnostics.

        Returns a list:
          [dual_estimate, loss, f_mean, g_mean]
        where:
          - dual_estimate = -loss ≈ OT cost,
          - loss          = J(f),
          - f_mean        = E[f(X)],
          - g_mean        = E[f^{(-c)}(Y)].
        """
        with torch.no_grad():
            d, j, f_mean, g_mean = self._dual_objective(
                X.to(self.device), Y.to(self.device)
            )
        return [d.item(), j.item(), f_mean.item(), g_mean.item()]

    ############################################################
    # Save / Load
    ############################################################
    def save(self, address, iters_done):
        torch.save({
            "model_state": self.model.state_dict(),
            "iters_done": iters_done,
            "betas": self.betas,
            "input_dim": self.input_dim,
            "ncandidates": self.ncandidates,
        }, address)

    def load(self, address):
        data = torch.load(address, map_location=self.device)
        self.model.load_state_dict(data["model_state"])
        self.betas = data.get("betas", self.betas)
        self.input_dim = data.get("input_dim", self.input_dim)
        self.ncandidates = data.get("ncandidates", self.ncandidates)
        return data.get("iters_done", 0)

    ############################################################
    # Training loop
    ############################################################
    ############################################################
    # Training loop (with minibatches)
    ############################################################
    def _fit(self,
             X,
             Y,
             iters_done=0,
             iters=10000,
             inner_steps=5,
             print_every=50,
             callback=None,
             convergence_tol=1e-4,
             convergence_patience=50,
             batch_size=256):
        """
        Run the training loop using minibatches of (X, Y).

        Arguments
        ---------
        X, Y : tensors of shape (N, dim)
        iters : total number of optimizer steps
        batch_size : minibatch size for both X and Y

        Logs:
          - logs["dual"]: sequence of -J(f) (OT estimates),
          - logs["f"]:    E[f(X)]  on the last seen batch,
          - logs["g"]:    E[f^{(-c)}(Y)] on the last seen batch.
        """

        # Move full data once to device
        X = X.to(self.device)
        Y = Y.to(self.device)

        # One dataset over pairs; we only use X for f and Y for conjugate,
        # but sampling pairs is still unbiased for the marginals.
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

        data_iter = iter(loader)

        logs = {"dual": [], "f": [], "g": []}
        prev_dual = None
        patience_counter = 0

        for it in range(iters_done, iters):
            # ------------------------------------------------
            # Get next minibatch, recycle iterator if needed
            # ------------------------------------------------
            try:
                x_batch, y_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_batch, y_batch = next(data_iter)

            # ------------------------------------------------
            # One SGD/Adam step on the minibatch
            # ------------------------------------------------
            last_stats = self.step(x_batch, y_batch, inner_steps=inner_steps)

            # ------------------------------------------------
            # Logging + simple convergence check
            # ------------------------------------------------
            if it % print_every == 0 and last_stats is not None:
                dual_val = last_stats["dual"]
                logs["dual"].append(dual_val)
                logs["f"].append(last_stats["f_mean"])
                logs["g"].append(last_stats["g_mean"])

                print(
                    f"[Iter {it}] dual={dual_val:.6f} j={last_stats['J']:.6f} "
                    f"f={last_stats['f_mean']:.4f} g={last_stats['g_mean']:.4f}"
                )

                # Convergence check on dual (OT estimate, on minibatch)
                if prev_dual is not None:
                    rel_change = abs(dual_val - prev_dual) / (abs(prev_dual) + 1e-12)
                    if rel_change < convergence_tol:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    if patience_counter >= convergence_patience:
                        print(
                            f"[CONVERGED] dual rel change < {convergence_tol} "
                            f"for {convergence_patience} checks (minibatch)."
                        )
                        break
                prev_dual = dual_val

            if callback is not None:
                callback(it)

        return logs
