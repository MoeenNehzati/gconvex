"""
Manual end-to-end test for FCOTSeparable on 1D Gaussian → Gaussian optimal transport.

This test is meant for **manual debugging, regression checks, and performance tuning**.
All tuning knobs are grouped in a CONFIG BLOCK at the top, each parameter is explained.

Run manually with:

    pytest tests/manual/test_gaussian_end_to_end.py -v -s
"""

import unittest
import os
import logging
import threading
import time
import torch
import numpy as np

from baselines.ot_fc_sep_map import FCOTSeparable
from tools.dgps import generate_gaussian_pairs
from tools.utils import inverse_grad_L22
from tools.feedback import set_log_level


class TestGaussianEndToEnd(unittest.TestCase):

    def test_gaussian_transport_pipeline(self):

        # =====================================================================
        #                           CONFIG BLOCK
        # =====================================================================
        #
        # Every parameter below affects accuracy, speed, or stability.
        # Tweak them freely while debugging. Comments explain recommended ranges.
        #
        # ----------------------------------------------------------------------
        # Randomness / Logging
        # ----------------------------------------------------------------------
        SEED = 42                      # Ensures reproducibility across runs
        LOG_LEVEL = "DEBUG"            # "INFO", "DEBUG", or "ERROR"
        ENABLE_LOG_FILE = True         # Set False to avoid writing log file
        LOG_FILE = "test_gaussian_sep.log"

        # ----------------------------------------------------------------------
        # Data distribution settings
        # Samples drawn from Gaussian → Gaussian OT problem.
        # ----------------------------------------------------------------------
        N_SAMPLES = 2000               # Larger → more stable gradients but slower
        MU_X = torch.tensor([0.0])     # Source mean
        SIGMA_X = torch.tensor([[1.0]])# Source covariance
        MU_Y = torch.tensor([-1.0])    # Target mean
        SIGMA_Y = torch.tensor([[2.0]])# Target covariance

        # ----------------------------------------------------------------------
        # Domain / grid resolution
        # Grid is [-RADIUS, RADIUS], discretized at X_ACCURACY and Y_ACCURACY.
        #
        # Smaller accuracy → finer grid → more accurate transforms but slower.
        # ----------------------------------------------------------------------
        RADIUS = 5.0
        X_ACCURACY = 0.01              # Step size for X-grid
        Y_ACCURACY = 0.01              # Step size for Y-grid

        # ----------------------------------------------------------------------
        # Optimizer (intercept optimizer)
        # ----------------------------------------------------------------------
        OUTER_LR = 1e-3                # Adam learning rate for intercepts

        # ----------------------------------------------------------------------
        # Softmax temperature schedule
        # max/min selection during forward pass becomes sharper over time.
        #
        # TEMP_MIN → start smooth
        # TEMP_MAX → final sharp selection
        #
        # TEMP_WARMUP_ITERS controls annealing schedule.
        # ----------------------------------------------------------------------
        TEMP_MIN = 1.0
        TEMP_MAX = 40.0
        TEMP_WARMUP_ITERS = 4_000

        # ----------------------------------------------------------------------
        # Reactivation and full refresh rules
        #
        # Reactivation: Occasionally nudge “dead” intercepts (zero gradient)
        # so they re-enter competition.
        #
        # Full refresh: Replace intercepts by exact b ← (b^c)^c,
        # then reset momentum + add jitter to break ties.
        # ----------------------------------------------------------------------
        REACTIVATE_EVERY = 50         # Steps between reactivations
        REACTIVATE_EPS = 5e-3          # How much to raise dead intercepts
        FULL_REFRESH_EVERY = 300       # Steps between b ← (b^c)^c full refresh

        # ----------------------------------------------------------------------
        # Coarse-to-fine transform speedup
        # ----------------------------------------------------------------------
        COARSE_X_FACTOR = 40           # Stride: check every nth X-grid point
        COARSE_TOP_K = 4               # Keep n best coarse candidates
        COARSE_WINDOW = 1              # Refine candidates ±window coarse stride

        # ----------------------------------------------------------------------
        # Training and evaluation set│ convergence       0/300                                                                          │tings
        # ----------------------------------------------------------------------
        TRAIN_ITERS = 10_000             # Total training iterations
        TEST_POINTS = 25               # Number of test evaluation points
        MONOTONIC_TOL = -0.5           # Allowed violation of monotonicity
        MAE_THR = 1.5                  # Acceptable mean absolute error tolerance
        # ======================================================================
        #                        Fit params
        # ======================================================================
        PRINT_EVERY=10
        LOG_EVERY=1
        CONVERGENCE_TOL=1e-4
        CONVERGENCE_PATIENCE=100
        FORCE_RETRAIN=True
        # =====================================================================
        #                        END CONFIG BLOCK
        # =====================================================================

        # ---- Randomness ----
        torch.manual_seed(SEED)
        set_log_level(LOG_LEVEL)

        # ---- Reduce thread usage for stability ----
        cpu = max(1, (os.cpu_count() or 1) // 2)
        try:
            torch.set_num_threads(cpu)
            torch.set_num_interop_threads(max(1, cpu // 2))
        except RuntimeError:
            pass

        # ----------------------------------------------------------------------
        # Optional log-to-file for debugging
        # ----------------------------------------------------------------------
        if ENABLE_LOG_FILE:
            root_logger = logging.getLogger()
            fh = logging.FileHandler(LOG_FILE, mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
            ))
            root_logger.addHandler(fh)
            self.addCleanup(lambda: (root_logger.removeHandler(fh), fh.close()))

        # ----------------------------------------------------------------------
        # Generate Gaussian samples for OT
        # ----------------------------------------------------------------------
        params = {
            "n": N_SAMPLES,
            "μ_x": MU_X,
            "Σ_x": SIGMA_X,
            "μ_y": MU_Y,
            "Σ_y": SIGMA_Y,
        }
        X, Y, _ = generate_gaussian_pairs(**params)

        # Closed-form transport map for 1D Gaussian → Gaussian
        sx = SIGMA_X[0, 0].item()
        sy = SIGMA_Y[0, 0].item()
        sigma_ratio = np.sqrt(sy / sx)

        def T_theory(x):
            """Exact Monge map for Gaussian → Gaussian in 1D."""
            return sigma_ratio * x - 1.0

        # ----------------------------------------------------------------------
        # Build FCOTSeparable instance
        # ----------------------------------------------------------------------
        ny = int((2 * RADIUS) / Y_ACCURACY) + 1
        n_params = ny * 1  # dim=1

        fcot = FCOTSeparable.initialize_right_architecture(
            dim=1,
            radius=RADIUS,
            n_params=n_params,
            x_accuracy=X_ACCURACY,
            kernel_1d=lambda x, y: (x - y) ** 2,
            inverse_cx = lambda g, y: y + 0.5 * g,
            outer_lr=OUTER_LR,
            temp_min=TEMP_MIN,
            temp_max=TEMP_MAX,
            temp_warmup_iters=TEMP_WARMUP_ITERS,
            reactivate_every=REACTIVATE_EVERY,
            reactivate_eps=REACTIVATE_EPS,
            full_refresh_every=FULL_REFRESH_EVERY,
            cache_gradients=True,
            coarse_x_factor=COARSE_X_FACTOR,
            coarse_top_k=COARSE_TOP_K,
            coarse_window=COARSE_WINDOW,
        )

        # ----------------------------------------------------------------------
        # Train FCOTSeparable
        # ----------------------------------------------------------------------
        fcot.fit(
            X, Y,
            iters=TRAIN_ITERS,
            print_every=PRINT_EVERY,
            log_every=LOG_EVERY,
            convergence_tol=CONVERGENCE_TOL,
            convergence_patience=CONVERGENCE_PATIENCE,
            force_retrain=FORCE_RETRAIN,
        )

        # ----------------------------------------------------------------------
        # Evaluate transport quality
        # ----------------------------------------------------------------------
        X_test = torch.linspace(-3, 3, TEST_POINTS).reshape(-1, 1)
        Y_pred = fcot.transport_X_to_Y(X_test)
        Y_true = T_theory(X_test)

        # --- Sanity checks ---
        self.assertFalse(torch.isnan(Y_pred).any())
        self.assertFalse(torch.isinf(Y_pred).any())
        self.assertEqual(Y_pred.shape, X_test.shape)

        # --- Mean absolute error ---
        mae = (Y_pred - Y_true).abs().mean().item()
        print(f"[GAUSSIAN TEST] MAE = {mae:.4f}")
        self.assertLess(mae, MAE_THR, f"Transport MAE too high: {mae:.4f}")

        # --- Monotonicity (OT in 1D is monotone) ---
        diffs = Y_pred[1:] - Y_pred[:-1]
        monotone = (diffs > MONOTONIC_TOL).float().mean().item()
        print(f"[GAUSSIAN TEST] Monotonicity = {monotone:.2%}")
        self.assertGreater(monotone, 0.8)

        print("✓ Gaussian end-to-end test PASSED.")


if __name__ == "__main__":
    unittest.main()
