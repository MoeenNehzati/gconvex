import torch
from typing import Any
import os
from config import WRITING_ROOT
from tools.utils import hash_dict
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn


class OT:
    def __init__(self) -> None:
        # Default placeholders to satisfy type checkers; subclasses typically override
        self.input_dim: int = 0
        self.device = torch.device("cpu")

    @staticmethod
    def initialize_right_architecture(dim,
                                      n_params_target,
                                      *args,
                                      **kwargs):
        #chooses the models that combined have the target number of parameters
        return OT(*args, **kwargs)

    def step(self, x_batch, y_batch) -> Any:
        # perform one training step on batch, to be used in the training loop of fit
        # subclasses may return metrics for logging
        pass
    
    def transport_X_to_Y(self, X):
       return X
    
    def debug_losses(self, X, Y):
        return []
    
    def get_hash(self, X, Y):
        # returns a hash associated with the model specification and data
        info = {'X': X,
                'Y': Y,
                **self.__dict__}
        return hash_dict(info)

    def get_address(self, X, Y, dir=WRITING_ROOT):
        # returns address associated with the model specification and data
        h = self.get_hash(X, Y)
        classname = self.__class__.__name__
        filename = f"{classname}_dim{self.input_dim}_{h}.pt"
        return os.path.join(dir, filename)

    def save(self, address, iters_done):
        # saves the models and itersdone to address with torch.save
        return None

    def load(self, address):
        # loads the models and itersdone from address with torch.load and returns iters_done
        return 0
    
    def _fit(self,
            X,
            Y,
            iters_done=0,
            iters=10000,
            inner_steps=5,
            print_every=50,
            callback=None,
            convergence_tol=1e-4,
            convergence_patience=50):
        # internal fit function that does not handle caching/resuming
        return {}
        
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
            iters_done = self.load(address)             
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
                print(f"[TRAIN] No checkpoint found at `{address}`. Training from scratch.")

        logs = self._fit(X, Y, iters_done, iters, inner_steps, print_every, callback, convergence_tol, convergence_patience)
        # --------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------
        print(f"[SAVE] Writing checkpoint to:\n  {address}")
        self.save(address, iters)
        return logs
