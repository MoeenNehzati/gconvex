import os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from tools.utils import hash_dict
from baselines.ot import OT
from tools.feedback import logger


# ------------------------------------------------------------
# Utility: Count parameters
# ------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.
    Correct for:
    - linear layers
    - residual blocks
    - spectral norm wrapped layers
    - any sequential/recursive submodules
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------
# Simple flexible MLP blocks for T and D
# ------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class ResNetMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_blocks):
        super().__init__()
        self.fc_in = nn.Linear(d_in, d_hidden)
        
        self.blocks = nn.Sequential(*[
            ResBlock(d_hidden) for _ in range(n_blocks)
        ])
        
        self.fc_out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x

class SNResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = U.spectral_norm(nn.Linear(width, width))
        self.fc2 = U.spectral_norm(nn.Linear(width, width))
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))

class SNResNetMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_blocks):
        super().__init__()
        self.fc_in = U.spectral_norm(nn.Linear(d_in, d_hidden))
        
        self.blocks = nn.Sequential(*[
            SNResBlock(d_hidden) for _ in range(n_blocks)
        ])
        
        self.fc_out = U.spectral_norm(nn.Linear(d_hidden, d_out))

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x


# ------------------------------------------------------------
# The GNOT/NOT Solver (vector version)
# ------------------------------------------------------------

class GNOTOT(OT):
    """
    Exact refactor of Korotin's NOT/GNOT algorithm into your OT interface.

    - architecture selection: initialize_right_architecture()
    - same losses as NOT:
          T_loss = cost(X, T(X)) - D(T(X)).mean()
          D_loss = D(T(X)).mean() - D(Y).mean()
    - same alternating updates:
          T_ITERS generator steps, 1 critic step
    """

    DEFAULT_T_ITERS = 5
    DEFAULT_T_LR = 1e-4
    DEFAULT_D_LR = 1e-4
    DEFAULT_BATCH = 128
    # ------------------------------------------------------------
    # Architecture selection
    # ------------------------------------------------------------
    @staticmethod
    def initialize_right_architecture(
        dim,
        n_params_target,
        cost_fn,                 # cost(X,Y) function, NOT a string
        T_ITERS=10,
        T_lr=1e-4,
        D_lr=1e-4,
        batch_size=128,
        device=None,
        seed=0x12345
    ):
        """
        Choose widths/depths of T and D so that
             params(T) + params(D) ≈ n_params_target.

        Simple search over hidden_dim ∈ {32, 64, ..., 1024}
        and n_layers ∈ {1,2,3,4}.
        """

        best_cfg = None
        best_diff = 1e18

        for width in [4, 16, 32, 64, 128, 256, 384, 512, 768, 1024]:
            for layers in [2, 3, 4]:
                # param counts for tentative T and D
                T_tmp = ResNetMLP(dim, width, dim, layers)
                D_tmp = SNResNetMLP(dim, width, 1, layers)

                total = count_params(T_tmp) + count_params(D_tmp)
                diff = total - n_params_target

                if diff < best_diff:
                    best_diff = diff
                    best_cfg = (width, layers)

        width, layers = best_cfg
        logger.info(f"[GNOTOT] Selected architecture with width={width}, layers={layers}, total_params≈{n_params_target} (diff={best_diff})")
        return GNOTOT(
            input_dim=dim,
            hidden_dim=width,
            n_layers=layers,
            n_params_target=n_params_target,
            cost_fn=cost_fn,
            T_ITERS=T_ITERS,
            T_lr=T_lr,
            D_lr=D_lr,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )

    # ------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_layers,
                 n_params_target,
                 cost_fn,        # the real cost function c(X,Y)
                 T_ITERS=10,
                 T_lr=1e-4,
                 D_lr=1e-4,
                 batch_size=128,
                 device=None,
                 seed=0x12345):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_params_target = n_params_target
        self.cost = cost_fn                                # IMPORTANT
        self.T_ITERS = T_ITERS
        self.T_lr = T_lr
        self.D_lr = D_lr
        self.batch_size = batch_size
        self.seed = seed

        # Device selection
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        torch.manual_seed(seed)

        # Networks
        self.T = ResNetMLP(input_dim, hidden_dim, input_dim, n_layers).to(device)
        self.D = SNResNetMLP(input_dim, hidden_dim, 1,        n_layers).to(device)

        # Optimizers
        self.T_opt = torch.optim.Adam(self.T.parameters(), lr=T_lr, weight_decay=1e-10)
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=D_lr, weight_decay=1e-10)

    # ------------------------------------------------------------
    # Monge map
    # ------------------------------------------------------------
    def transport_X_to_Y(self, X):
        self.T.eval()
        with torch.no_grad():
            return self.T(X)

    # ------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------
    def debug_losses(self, X, Y):
        self.T.eval()
        self.D.eval()
        with torch.no_grad():
            T_X = self.T(X)
            cost_val = self.cost(X, T_X).mean()
            T_loss = cost_val - self.D(T_X).mean()
            D_loss = self.D(T_X).mean() - self.D(Y).mean()
        return [T_loss.item(), D_loss.item(), cost_val.item()]


    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------
    def save(self, address, iters_done):
        state = {
            "T_state": self.T.state_dict(),
            "D_state": self.D.state_dict(),
            "iters_done": iters_done,
        }
        torch.save(state, address)

    def load(self, address):
        if not os.path.exists(address):
            return 0
        ckpt = torch.load(address, map_location=self.device)
        self.T.load_state_dict(ckpt["T_state"])
        self.D.load_state_dict(ckpt["D_state"])
        return ckpt.get("iters_done", 0)

    # ------------------------------------------------------------
    # Core training (identical to Korotin code logic)
    # ------------------------------------------------------------
    def _fit(
        self, X, Y,
        iters_done=0,
        iters=10000,
        inner_steps=5,             # not used (NOT uses T_ITERS instead)
        print_every=50,
        callback=None,
        convergence_tol=1e-4,
        convergence_patience=50
    ):

        N_X = X.shape[0]
        N_Y = Y.shape[0]

        def minibatch(data):
            idx = torch.randint(0, data.shape[0], (self.batch_size,), device=self.device)
            return data[idx]

        logs = {"T_loss": [], "D_loss": [], "mean_cost": []}

        for step in range(iters_done, iters):

            # ------------------------------------------------
            # (1) Optimize T (generator): inner T_ITERS steps
            # ------------------------------------------------
            self.T.train()
            self.D.eval()

            last_T_loss = None
            for _ in range(self.T_ITERS):
                Xb = minibatch(X)
                self.T_opt.zero_grad()

                T_X = self.T(Xb)
                T_loss = self.cost(Xb, T_X).mean() - self.D(T_X).mean()  # identical to NOT
                T_loss.backward()
                self.T_opt.step()

                last_T_loss = T_loss.detach().item()

            # ------------------------------------------------
            # (2) Optimize D (critic): 1 step
            # ------------------------------------------------
            self.T.eval()
            self.D.train()

            Xb = minibatch(X)
            Yb = minibatch(Y)

            with torch.no_grad():
                T_X = self.T(Xb)

            self.D_opt.zero_grad()
            D_loss = self.D(T_X).mean() - self.D(Yb).mean()      # identical to NOT
            D_loss.backward()
            self.D_opt.step()
            average_cost = self.cost(Xb, T_X).mean().item()
            logs["T_loss"].append(last_T_loss)
            logs["D_loss"].append(D_loss.detach().item())
            logs["mean_cost"].append(average_cost)

            if callback:
                callback(step, logs)

            if print_every and step % print_every == 0:
                logger.debug(f"[step {step}] T_loss={last_T_loss:.4f}  D_loss={float(D_loss):.4f}  mean_cost={average_cost:.4f}")

        return logs
