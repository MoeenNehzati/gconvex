import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from config import WRITING_ROOT
from tools.utils import hash_dict
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn


def count_icnn_params(dim, hidden_sizes):
    """
    hidden_sizes: list like [h1, h2, ..., hL] where hL is typically 1
    """
    total = 0
    # A_k and b_k
    for h in hidden_sizes:
        total += dim * h     # A_k
        total += h           # b_k (1×h)
    # W_k for k>=2
    for h_prev, h_next in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        total += h_prev * h_next
    return total

def choose_icnn_architecture(dim, n_params_target, 
                             depth_range=(2, 8), 
                             width_range=(4, 2048)):
    """
    Returns (depth, hidden_sizes, param_count)

    hidden_sizes is a list like [h, h, ..., h, 1]
    so the ICNN ends with a scalar output.
    """

    best = None
    best_diff = float("inf")

    for depth in range(depth_range[0], depth_range[1] + 1):
        # We will search width for layers 1..(depth-1), last layer is output=1
        for width in range(width_range[0], width_range[1] + 1):
            hidden_sizes = [width] * (depth - 1) + [1]
            count = count_icnn_params(dim, hidden_sizes)
            diff = abs(count - n_params_target)

            if diff < best_diff:
                best_diff = diff
                best = (depth, hidden_sizes, count)

            # exact match possible
            if diff == 0:
                return best

    return best

# # ==================================================
# #  ICNN ARCHITECTURE
# # ==================================================

# class ICNN(nn.Module):
#     def __init__(self, input_dim, hidden_dims):
#         super().__init__()
#         self.input_dim = input_dim

#         self.W = nn.ModuleList()
#         self.A = nn.ModuleList()
#         self.b = nn.ParameterList()

#         dims = [input_dim] + hidden_dims
#         for k in range(len(hidden_dims)):
#             Wk = nn.Linear(dims[k], dims[k+1], bias=False)
#             Wk.weight.data.uniform_(0, 0.1)
#             self.W.append(Wk)

#             Ak = nn.Linear(input_dim, dims[k+1], bias=False)
#             self.A.append(Ak)

#             bk = nn.Parameter(torch.zeros(dims[k+1]))
#             self.b.append(bk)

#         self.final_w = nn.Linear(hidden_dims[-1], 1, bias=False)
#         self.final_a = nn.Linear(input_dim, 1, bias=False)
#         self.final_b = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         z = x
#         for Wk, Ak, bk in zip(self.W, self.A, self.b):
#             z = F.leaky_relu(Wk(z) + Ak(x) + bk)
#         out = self.final_w(z) + self.final_a(x) + self.final_b
#         return out.squeeze(-1)

#     @torch.no_grad()
#     def project_weights(self):
#         # Clamp all convexity-critical weights ≥ 0
#         for layer in list(self.W) + [self.final_w]:
#             layer.weight.clamp_(min=0)



# # ==================================================
# #  UTILS
# # ==================================================

# def grad_g(g, Y, create_graph):
#     Y_leaf = Y.detach().requires_grad_(True)
#     gY = g(Y_leaf)
#     gradY = torch.autograd.grad(gY.sum(), Y_leaf, create_graph=create_graph)[0]
#     return gradY


# def J(f, g, X, Y, create_graph_for_g):
#     nabla_gY = grad_g(g, Y, create_graph=create_graph_for_g)
#     f_gradgY = f(nabla_gY)
#     dot_term = (Y * nabla_gY).sum(dim=1)
#     fX = f(X)
#     return (f_gradgY - dot_term - fX).mean()


# def count_icnn_params(dim, hidden_dims):
#     total = 0
#     dims = [dim] + hidden_dims
#     for k in range(len(hidden_dims)):
#         total += dims[k] * dims[k+1]
#         total += dim * dims[k+1]
#         total += dims[k+1]
#     total += hidden_dims[-1]
#     total += dim
#     total += 1
#     return total


# def choose_depth(dim):
#     if dim <= 5: return 2
#     elif dim <= 20: return 3
#     elif dim <= 50: return 4
#     else: return 6


# def choose_hidden_width(dim, depth, target_params):
#     best_h = None
#     best_diff = float('inf')

#     for h in range(4, 4000):
#         hidden_dims = [h] * depth
#         p = count_icnn_params(dim, hidden_dims)
#         diff = abs(p - target_params)

#         if diff < best_diff:
#             best_diff = diff
#             best_h = h

#         if p > target_params and diff > best_diff:
#             break

#     return best_h


# # ==================================================
# #  EXTRA WEIGHT/CONVEXITY DEBUG HELPERS
# # ==================================================

# def _debug_icnn_weight_stats(icnn: ICNN, name: str):
#     with torch.no_grad():
#         # Constrained Wk weights
#         W_mins = [Wk.weight.min().item() for Wk in icnn.W]
#         W_maxs = [Wk.weight.max().item() for Wk in icnn.W]
#         minW = min(W_mins)
#         maxW = max(W_maxs)

#         any_neg_W = any(wmin < 0 for wmin in W_mins)

#         # Final_w weights (not clamped in your code, so we care about sign)
#         final_min = icnn.final_w.weight.min().item()
#         final_max = icnn.final_w.weight.max().item()
#         any_neg_final = final_min < 0

#         # NaN / Inf checks
#         any_nan = any(torch.isnan(p).any().item() for p in icnn.parameters())
#         any_inf = any(torch.isinf(p).any().item() for p in icnn.parameters())

#     print(f"[DEBUG][{name}] Wk.weight min/max     : {minW:.6f} / {maxW:.6f}")
#     print(f"[DEBUG][{name}] any neg in Wk.weight : {any_neg_W}")
#     print(f"[DEBUG][{name}] final_w min/max      : {final_min:.6f} / {final_max:.6f}")
#     print(f"[DEBUG][{name}] any neg in final_w   : {any_neg_final}")
#     print(f"[DEBUG][{name}] any NaN / any Inf    : {any_nan} / {any_inf}")


# # ==================================================
# #  DEBUGGING BLOCK
# # ==================================================

# def debug_diagnostics(model, X, Y):
#     f = model.f
#     g = model.g

#     # ---- 1. mean ||∇g(Y)|| and distribution ----
#     Y_tmp = Y.detach().clone().requires_grad_(True)
#     gY = g.forward(Y_tmp)
#     nabla_gY = torch.autograd.grad(gY.sum(), Y_tmp, create_graph=False)[0]

#     gradg_norm_vec = nabla_gY.norm(dim=1)
#     gradg_norm_mean = gradg_norm_vec.mean().item()
#     gradg_norm_std = gradg_norm_vec.std().item()
#     nabla_min = nabla_gY.min().item()
#     nabla_max = nabla_gY.max().item()

#     # ---- 2. parameter norms ----
#     with torch.no_grad():
#         f_norm = sum(p.norm().item() for p in f.parameters())
#         g_norm = sum(p.norm().item() for p in g.parameters())

#     # ---- 3. convexity-ish checks on constrained Wk and final_w ----
#     with torch.no_grad():
#         negW_f = any((Wk.weight < 0).any().item() for Wk in f.W)
#         negW_g = any((Wk.weight < 0).any().item() for Wk in g.W)
#         neg_final_f = (f.final_w.weight < 0).any().item()
#         neg_final_g = (g.final_w.weight < 0).any().item()

#     # ---- 4. f(X), f(∇g(Y)), dot term ----
#     fX = f(X).mean().item()
#     f_gradgY = f(nabla_gY).mean().item()
#     dot_term = (Y * nabla_gY).sum(dim=1).mean().item()

#     # ---- 5. J ----
#     J_val = (f_gradgY - dot_term - fX)

#     print("\n===================== DEBUG =====================")
#     print(f"mean ||∇g(Y)||        : {gradg_norm_mean:.6f}")
#     print(f"std  ||∇g(Y)||        : {gradg_norm_std:.6f}")
#     print(f"∇g(Y) min / max       : {nabla_min:.6f} / {nabla_max:.6f}")
#     print(f"||θ_f||, ||θ_g||      : {f_norm:.3f}, {g_norm:.3f}")
#     print(f"any neg W_f?          : {negW_f}")
#     print(f"any neg W_g?          : {negW_g}")
#     print(f"any neg final_w_f?    : {neg_final_f}")
#     print(f"any neg final_w_g?    : {neg_final_g}")
#     print(f"f(X)                  : {fX:.4f}")
#     print(f"f(∇g(Y))              : {f_gradgY:.4f}")
#     print(f"dot_term              : {dot_term:.4f}")
#     print(f"J(f,g)                : {J_val:.6f}")
#     print("-------------------------------------------------")
#     _debug_icnn_weight_stats(f, "f")
#     _debug_icnn_weight_stats(g, "g")
#     print("=================================================\n")


# # ==================================================
# #  AUTO OT-ICNN
# # ==================================================

# class OTICNN_Auto(nn.Module):
#     def __init__(self, dim, num_params):
#         super().__init__()
#         self.dim = dim
#         self.num_params = num_params

#         depth = choose_depth(dim)
#         width = choose_hidden_width(dim, depth, num_params//2)
#         hidden_dims = [width] * depth

#         print(f"[Auto-ICNN] dim={dim}, depth={depth}, width={width}, params≈{count_icnn_params(dim, hidden_dims)}")

#         self.f = ICNN(dim, hidden_dims)
#         self.g = ICNN(dim, hidden_dims)

#     # ----------------------------
#     # LOSS
#     # ----------------------------
#     def loss(self, X, Y, create_graph_for_g):
#         return J(self.f, self.g, X, Y, create_graph_for_g)

#     # ----------------------------
#     # ONE FULL ALGORITHM 1 STEP
#     # ----------------------------
#     def alternate_steps(self, X, Y, opt_g, K):
#         for _ in range(K):
#             opt_g.zero_grad()
#             loss_g = self.loss(X, Y, create_graph_for_g=True)
#             (-loss_g).backward()
#             torch.nn.utils.clip_grad_norm_(self.g.parameters(), max_norm=1.0)
#             opt_g.step()
#             self.g.project_weights()

#     def step(self, X, Y, opt_f, opt_g, K):
#         # Inner loop: minimize J wrt g
#         self.alternate_steps(X, Y, opt_g, K)

#         # Outer loop: maximize J wrt f
#         opt_f.zero_grad()
#         loss_f = self.loss(X, Y, create_graph_for_g=False)
#         loss_f.backward()
#         torch.nn.utils.clip_grad_norm_((self.f).parameters(), max_norm=1.0)
#         opt_f.step()
#         self.f.project_weights()
#         return float(loss_f.item())

#     # ----------------------------
#     # FULL TRAINING
#     # ----------------------------
#     def fit(self, X, Y, iters=500, K=5, lr_f=1e-4, lr_g=1e-3, save_dir=WRITING_ROOT, force=False):
#         os.makedirs(save_dir, exist_ok=True)

#         h = dataset_hash(X, Y)
#         save_path = os.path.join(save_dir, f"oticnn_dim{self.dim}_params{self.num_params}_{h}.pt")

#         if os.path.exists(save_path) and not force:
#             print(f"[✓] Found saved model. Loading: {save_path}")
#             ckpt = torch.load(save_path, map_location=X.device)
#             self.load_state_dict(ckpt["model_state_dict"])
#             for p in self.parameters():
#                 p.requires_grad_(True)
#             return ckpt.get("losses", [])

#         print(f"[→] No saved model found. Training new OT-ICNN (hash={h}).")

#         opt_f = torch.optim.SGD(self.f.parameters(), lr=lr_f)
#         opt_g = torch.optim.SGD(self.g.parameters(), lr=lr_g, weight_decay=1e-3)

#         # Initial diagnostics BEFORE any training
#         debug_diagnostics(self, X, Y)

#         losses = []
#         with Progress(
#             TextColumn("[bold blue]Training"),
#             BarColumn(),
#             TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
#             TextColumn(" • loss = {task.fields[loss]:8.4f}"),
#             TimeElapsedColumn(),
#             TimeRemainingColumn(),
#         ) as progress:

#             task = progress.add_task("OT-ICNN", total=iters, loss=float('inf'))

#             for t in range(iters):
#                 loss = self.step(X, Y, opt_f, opt_g, K=K)
#                 losses.append(loss)
#                 progress.update(task, advance=1, loss=loss)

#                 # Diagnostics every 50 iters
#                 if t % 50 == 0 and t > 0:
#                     debug_diagnostics(self, X, Y)

#         ckpt = {
#             "model_state_dict": self.state_dict(),
#             "losses": losses,
#             "dim": self.dim,
#             "num_params": self.num_params,
#         }
#         torch.save(ckpt, save_path)
#         print(f"[✓] Saved model to: {save_path}")

#         return losses

#     # ----------------------------
#     # TRANSPORT MAP ∇f
#     # ----------------------------
#     def transport_map(self, X):
#         X = X.requires_grad_(True)
#         fX = self.f(X)
#         grad = torch.autograd.grad(fX.sum(), X)[0]
#         return grad

# #============================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class KantorovichPotential(nn.Module):
    """
    Modelling the Kantorovich potential as an Input Convex Neural Network (ICNN).

    input:  y (batch_size, input_size)
    output: z = h_L (batch_size, 1) if hidden_size_list ends with 1

    Architecture:
        h_1     = ReLU^2(A_0 y + b_0)
        h_{l+1} = LeakyReLU(A_l y + b_l + W_{l-1} h_l),  for l >= 1

    Constraint: W_l >= 0 (enforced by projection and/or penalty).
    """

    def __init__(self, input_size, hidden_size_list):
        super().__init__()

        # hidden_size_list always contains 1 at the end (scalar output)
        self.input_size = input_size
        self.num_hidden_layers = len(hidden_size_list)

        # A_k: matrices that interact with inputs (input_size x hidden_k)
        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(input_size, hidden_size_list[k]))
            for k in range(self.num_hidden_layers)
        ])

        # b_k: bias vectors (1 x hidden_k)
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_size_list[k]))
            for k in range(self.num_hidden_layers)
        ])

        # W_k: matrices between consecutive layers (hidden_{k-1} x hidden_k)
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size_list[k - 1], hidden_size_list[k]))
            for k in range(1, self.num_hidden_layers)
        ])

        # Initialization similar to tf.random_uniform(maxval=0.1)
        for A_k in self.A:
            nn.init.uniform_(A_k, a=0.0, b=0.1)
        for W_k in self.W:
            nn.init.uniform_(W_k, a=0.0, b=0.1)

    @property
    def positive_constraint_loss(self):
        """
        Equivalent to:
            tf.add_n([tf.nn.l2_loss(tf.nn.relu(-w)) for w in self.W])

        tf.nn.l2_loss(x) = sum(x^2) / 2
        """
        loss = 0.0
        for w in self.W:
            neg_part = F.relu(-w)          # relu(-w) >= 0 where w < 0
            loss = loss + 0.5 * (neg_part ** 2).sum()
        return loss

    @torch.no_grad()
    def proj(self):
        """
        Projection step to enforce W_l >= 0:
            w.assign(tf.nn.relu(w)) in TF.
        """
        for w in self.W:
            w.clamp_(min=0.0)

    def forward(self, input_y):
        """
        input_y: (batch_size, input_size)
        returns: (batch_size, hidden_size_list[-1]) (usually (batch_size, 1))
        """
        # First layer: leaky ReLU then squared
        z = input_y @ self.A[0] + self.b[0]             # (B, hidden_0)
        z = F.leaky_relu(z, negative_slope=0.2)
        z = z * z                                       # ReLU^2 as in TF code

        # Subsequent layers: leaky ReLU(A_k y + b_k + W_{k-1} z)
        for k in range(1, self.num_hidden_layers):
            z = input_y @ self.A[k] + self.b[k] + z @ self.W[k - 1]
            z = F.leaky_relu(z, negative_slope=0.2)

        return z

    def get_hidden_size(self):
        """
        Returns:
            depth: number of layers
            hidden_sizes: list of hidden layer widths (includes final scalar 1)
            n_params: number of parameters in this ICNN
        """
        hidden_sizes = [A.shape[1] for A in self.A]   # hidden sizes from A matrices
        return hidden_sizes


class ICNNOT:
    """
    PyTorch version of the TensorFlow ComputeOT driver.

    Supports:
      - alternating f/g optimization steps
      - transport_X_to_Y, transport_Y_to_X
      - compute_W2
      - architecture auto-selection via initialize_right_architecture
    """

    def __init__(self, input_dim, f_model, g_model, lr,
                 lambda_reg=1.0, betas=(0.5, 0.9), device="cpu"):

        self.device  = torch.device(device)
        self.lr      = lr
        self.betas   = betas
        self.lambda_reg = lambda_reg
        self.input_dim  = input_dim

        # Models
        self.f_model = f_model.to(self.device) if f_model is not None else None
        self.g_model = g_model.to(self.device) if g_model is not None else None

        # Optimizers (only if models exist)
        if self.f_model is not None:
            self.f_optimizer = torch.optim.Adam(
                self.f_model.parameters(), lr=lr, betas=betas
            )
        if self.g_model is not None:
            self.g_optimizer = torch.optim.Adam(
                self.g_model.parameters(), lr=lr, betas=betas
            )


    # ----------------------------------------------------------------------
    #  Architecture Auto-Initialization
    # ----------------------------------------------------------------------
    @staticmethod
    def initialize_right_architecture(dim,
                                      n_params_target,
                                      depth_range=(2, 8),
                                      width_range=(4, 2048),
                                      device="cpu"):
        """
        Static method.
        Finds the ICNN architecture that matches the parameter budget
        and returns fully initialized (f_model, g_model).

        Returns:
            f_model, g_model, depth, hidden_sizes, actual_param_count
        """
        # local imports or replace with your file paths
        # Search for architecture
        depth, hidden_sizes, actual = choose_icnn_architecture(
            dim,
            n_params_target,
            depth_range=depth_range,
            width_range=width_range
        )

        print(f"[ARCH] dim={dim}, target={n_params_target}")
        print(f"       depth        = {depth}")
        print(f"       hidden_sizes = {hidden_sizes}")
        print(f"       n_params     = {actual}")

        # Build two ICNN models
        f_model = KantorovichPotential(dim, hidden_sizes).to(device)
        g_model = KantorovichPotential(dim, hidden_sizes).to(device)
        return ICNNOT(
            dim,
            f_model,
            g_model,
            lr=1e-3,
            lambda_reg=1.0,
            betas=(0.5, 0.9),
            device=device
        )
    # ----------------------------------------------------------------------
    #  Core Computations
    # ----------------------------------------------------------------------
    def _compute_core(self, x, y, create_graph=True):
        """
        Equivalent to the TensorFlow ops x→fx, y→gy, gradients, losses, W2.
        """

        x = x.to(self.device)
        y = y.to(self.device)
        x.requires_grad_(True)
        y.requires_grad_(True)

        fx = self.f_model(x)         # (B,1)
        gy = self.g_model(y)         # (B,1)

        # ∇f(x)
        grad_fx = torch.autograd.grad(
            fx.sum(), x, create_graph=create_graph
        )[0]

        # ∇g(y)
        grad_gy = torch.autograd.grad(
            gy.sum(), y, create_graph=create_graph
        )[0]

        # f(∇g(y))
        f_grad_gy = self.f_model(grad_gy)

        # <y, ∇g(y)>
        y_dot_grad_gy = (y * grad_gy).sum(dim=1, keepdim=True)

        # Norms
        x_squared = (x * x).sum(dim=1, keepdim=True)
        y_squared = (y * y).sum(dim=1, keepdim=True)

        # Losses
        f_loss = (fx - f_grad_gy).mean()
        g_loss = (f_grad_gy - y_dot_grad_gy).mean()

        # Dual W₂ objective
        W2 = (f_grad_gy - fx - y_dot_grad_gy
              + 0.5 * x_squared + 0.5 * y_squared).mean()

        return {
            "fx": fx,
            "gy": gy,
            "grad_fx": grad_fx,
            "grad_gy": grad_gy,
            "f_grad_gy": f_grad_gy,
            "y_dot_grad_gy": y_dot_grad_gy,
            "x_squared": x_squared,
            "y_squared": y_squared,
            "f_loss": f_loss,
            "g_loss": g_loss,
            "W2": W2,
        }


    # ----------------------------------------------------------------------
    #  Training Steps
    # ----------------------------------------------------------------------
    def step_g(self, x_batch, y_batch):
        """
        g-step: minimize g_loss + λ * positivity_penalty
        """

        self.g_optimizer.zero_grad()
        out = self._compute_core(x_batch, y_batch, create_graph=True)

        g_loss = out["g_loss"]
        if self.lambda_reg > 0:
            g_loss = g_loss + self.g_model.positive_constraint_loss

        g_loss.backward()
        self.g_optimizer.step()

        if self.lambda_reg <= 0:
            self.g_model.proj()   # TF mirrors this case

        return g_loss.item()


    def step_f(self, x_batch, y_batch):
        """
        f-step: minimize f_loss, always project W≥0 afterwards (like TF)
        """

        self.f_optimizer.zero_grad()
        out = self._compute_core(x_batch, y_batch, create_graph=True)

        f_loss = out["f_loss"]
        f_loss.backward()
        self.f_optimizer.step()

        # Always project
        self.f_model.proj()

        return f_loss.item()


    # ----------------------------------------------------------------------
    #  Evaluation Utilities
    # ----------------------------------------------------------------------
    def compute_W2(self, X, Y):
        out = self._compute_core(X, Y, create_graph=False)
        return out["W2"].item()

    def transport_X_to_Y(self, X):
        X = X.to(self.device)
        X.requires_grad_(True)
        fx = self.f_model(X)
        grad_fx = torch.autograd.grad(fx.sum(), X)[0]
        return grad_fx

    def transport_Y_to_X(self, Y):
        Y = Y.to(self.device)
        Y.requires_grad_(True)
        gy = self.g_model(Y)
        grad_gy = torch.autograd.grad(gy.sum(), Y)[0]
        return grad_gy

    def debug_losses(self, X, Y):
        out = self._compute_core(X, Y, create_graph=False)
        return (out["f_loss"].item(),
                out["g_loss"].item(),
                out["W2"].item())
    
    def get_architecture_info(self):
        f_hidden_sizes = self.f_model.get_hidden_size()
        g_hidden_sizes = self.g_model.get_hidden_size()
        return {
            "f_hidden_sizes": f_hidden_sizes,
            "g_hidden_sizes": g_hidden_sizes,
        }
    
    def get_hash(self, X, Y):
        arch = {
            "X": X,
            "Y": Y,
            "lr": self.lr,
            "betas": self.betas,
            "lambda_reg": self.lambda_reg,
            **self.get_architecture_info()
        }
        return hash_dict(arch)

    def get_address(self, X, Y, dir=WRITING_ROOT):
        h = self.get_hash(X, Y)
        filename = f"icnnot_dim{self.input_dim}_{h}.pt"
        return os.path.join(dir, filename)

    def fit(self,
            X,
            Y,
            iters=10000,
            inner_steps=5,
            print_every=50,
            callback=None,
            force_retrain=False,
            convergence_tol=1e-4,
            convergence_patience=50):
        """
        Full-batch ICNN-OT training with:
        - caching by architecture+data hash
        - checkpoint resume
        - early stopping by W2 convergence
        """

        # --------------------------------------------------------
        # Convert data to tensors
        # --------------------------------------------------------
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        if not torch.is_tensor(Y):
            Y = torch.from_numpy(Y).float()

        X = X.to(self.device)
        Y = Y.to(self.device)

        # --------------------------------------------------------
        # Compute save location
        # --------------------------------------------------------
        address = self.get_address(X, Y)
        os.makedirs(os.path.dirname(address), exist_ok=True)

        # --------------------------------------------------------
        # Load checkpoint if exists
        # --------------------------------------------------------
        iters_done = 0

        if os.path.exists(address) and not force_retrain:
            print(f"[CACHE] Found saved ICNN-OT model:\n  {address}")

            checkpoint = torch.load(address, map_location=self.device)
            self.f_model.load_state_dict(checkpoint["f_model"])
            self.g_model.load_state_dict(checkpoint["g_model"])

            iters_done = checkpoint.get("iters_done", 0)
            print(f"[CACHE] Model previously trained for {iters_done} iterations.")

            # If we already reached or exceeded desired iters → return
            if iters_done >= iters:
                print("[CACHE] Requested iterations already satisfied. Skipping training.")
                return {"f_losses": [], "g_losses": [], "W2": []}

            print(f"[RESUME] Resuming training for {iters - iters_done} more iterations.")
        else:
            if force_retrain:
                print("[FORCE] Forced retrain from scratch.")
            else:
                print("[TRAIN] No checkpoint found. Training from scratch.")

        # --------------------------------------------------------
        # Begin training (fresh or resumed)
        # --------------------------------------------------------
        logs = {"f_losses": [], "g_losses": [], "W2": []}

        prev_w2 = None
        convergence_counter = 0

        # MAIN LOOP
        for it in range(iters_done, iters):

            # ---------------------------
            # g steps
            # ---------------------------
            for _ in range(inner_steps):
                self.step_g(X, Y)

            # ---------------------------
            # f step
            # ---------------------------
            self.step_f(X, Y)

            # ---------------------------
            # Compute losses (required for convergence)
            # ---------------------------
            f_loss, g_loss, w2 = self.debug_losses(X, Y)

            # ---------------------------
            # Logging
            # ---------------------------
            if it % print_every == 0:
                logs["f_losses"].append(f_loss)
                logs["g_losses"].append(g_loss)
                logs["W2"].append(w2)
                print(f"[Iter {it}] f={f_loss:.4f}, g={g_loss:.4f}, W2={w2:.6f}")

            if callback is not None:
                callback(it)

            # ---------------------------
            # Early stopping by convergence
            # ---------------------------
            if prev_w2 is not None:
                rel_change = abs(w2 - prev_w2) / (abs(prev_w2) + 1e-12)

                if rel_change < convergence_tol:
                    convergence_counter += 1
                else:
                    convergence_counter = 0

                if convergence_counter >= convergence_patience:
                    print(f"[CONVERGED] Relative W2 change < {convergence_tol} "
                        f"for {convergence_patience} iterations.")
                    iters = it + 1
                    break

            prev_w2 = w2

        # --------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------
        print(f"[SAVE] Writing checkpoint to:\n  {address}")

        torch.save(
            {
                "f_model": self.f_model.state_dict(),
                "g_model": self.g_model.state_dict(),
                "hash": self.get_hash(X, Y),
                "arch": self.get_architecture_info(),
                "iters_done": iters,   # actual completed iterations
            },
            address
        )

        return logs
