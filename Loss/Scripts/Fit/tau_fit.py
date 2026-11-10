import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import emcee
from scipy.interpolate import RegularGridInterpolator
from schwimmbad import MPIPool

# =====================================
# 1. DIRECTORY & CONFIGURATION
# =====================================
# Inject the root 'Scripts' directory into Python's path and import the config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import get_paths

# Fetch paths for the currently active route
cfg = get_paths()

DATA_DIR = cfg["data_dir"]
FIT_DIR = cfg["fit_dir"]
SPATIAL_MODE = cfg["spatial_mode"]

GRID_FILE = cfg["tau_grid"]

# Outputs
BACKEND_FILE = cfg["tau_backend"]
CSV_CHAIN_FILE = cfg["tau_chain"]

TAN0 = 4.51e-6 #8.6577e-6
TAN_BG = 0.0 #3.3573e-6

# Set False for ignorant flat priors (recommended for spatial fits),
# or True to apply the legacy Gaussian priors from the uniform fit.
USE_LEGACY_GAUSSIAN_PRIORS = False

# MCMC Hyperparameters
N_DIM = 3
N_WALKERS = 32
N_STEPS = 5000
N_BATCH = 1000

# =====================================
# 2. LOAD PRECOMPUTED GRID & SPLINE
# =====================================
grid_archive = np.load(GRID_FILE)
P_GRID = grid_archive["p_grid"]
LOG_P_GRID = np.log10(P_GRID)
G_GRID = grid_archive["g_grid"]
LOSS_GRID = grid_archive["loss_grid"]
LOSS_DATA = grid_archive["loss_data"]
NU_DATA = grid_archive["nu"]
NPH_DATA = grid_archive["n_ph"]

# Dynamic sigma (std of point-to-point diff along n_ph)
SIGMA_DATA = np.zeros_like(LOSS_DATA)
for nu in np.unique(NU_DATA):
    mask = (NU_DATA == nu)
    order = np.argsort(NPH_DATA[mask])
    y_sorted = LOSS_DATA[mask][order]
    if len(y_sorted) > 1:
        local_sigma = max(np.std(np.diff(y_sorted)) / np.sqrt(2.0), 1.0e-7)
    else:
        local_sigma = 2.5e-7
    SIGMA_DATA[mask] = local_sigma

# Cubic spline over uniform(log10(p), log10(Gamma_1)) axes
LOSS_SPLINE = RegularGridInterpolator(
    (LOG_P_GRID, G_GRID),
    LOSS_GRID,
    method="cubic",
    bounds_error=False,
    fill_value=None
)

P_MIN, P_MAX = float(P_GRID[0]), float(P_GRID[-1])
G_MIN, G_MAX = float(G_GRID[0]), float(G_GRID[-1])

# =====================================
# 3. MCMC LOG PROBABILITY
# =====================================
def log_probability(theta):
    p_val, log_k1_MHz, A_val = theta
    
    if not (P_MIN <= p_val <= P_MAX and -1.0 < log_k1_MHz < G_MAX and 0.3 < A_val < 1.7):
        return -np.inf

    unit_loss = LOSS_SPLINE(np.array([[np.log10(p_val), log_k1_MHz]]))[0]
    model = TAN_BG + TAN0 * A_val * unit_loss

    chi2 = np.sum(((LOSS_DATA - model) / SIGMA_DATA) ** 2)
    if np.isnan(chi2) or not np.isfinite(chi2):
        return -np.inf

    log_prob = -0.5 * chi2
    
    if USE_LEGACY_GAUSSIAN_PRIORS:
        log_prob += -0.5 * ((p_val - 18.0) / 2.0) ** 2
        log_prob += -0.5 * ((log_k1_MHz - 0.7) / 0.2) ** 2
        log_prob += -0.5 * ((A_val - 1.0) / 0.0018) ** 2

    return log_prob
        
# =====================================
# 4. MPI EXECUTION
# =====================================
if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print(f"--- STARTING SPLINE_ACCELERATED LOSS MCMC ({SPATIAL_MODE}) ---", flush=True)

        backend = emcee.backends.HDFBackend(str(BACKEND_FILE))
        backend.reset(N_WALKERS, N_DIM)

        p0_center = np.array([15.0, 0.2, 1.0])
        jitter = np.array([1.5, 0.15, 0.05])

        p0 = []
        for _ in range(N_WALKERS):
            while True:
                pos = p0_center + jitter * np.random.randn(N_DIM)
                if np.isfinite(log_probability(pos)):
                    p0.append(pos)
                    break

        sampler = emcee.EnsembleSampler(
            N_WALKERS, N_DIM, log_probability,
            moves=[
                (emcee.moves.StretchMove(a=4.0), 0.7),
                (emcee.moves.DEMove(), 0.2),
                (emcee.moves.DESnookerMove(), 0.1)
            ],
            pool=pool,
            backend=backend
        )

        state, start_time = p0, time.time()
        prev_accepted = np.zeros(N_WALKERS)  # Track rolling acceptance

        for batch_start in range(0, N_STEPS, N_BATCH):
            state, prob, rstate = sampler.run_mcmc(
                state,
                N_BATCH,
                progress=False,
                skip_initial_state_check=(batch_start > 0)
            )

            # Calculate this batch's rolling acceptance rate
            current_accepted = sampler.backend.accepted
            batch_accepted = current_accepted - prev_accepted
            rolling_acc = np.mean(batch_accepted) / N_BATCH
            
            # Update for the next iteration
            prev_accepted = current_accepted.copy()

            elapsed = (time.time() - start_time) / 60
            print(
                f"Completed {batch_start + N_BATCH}/{N_STEPS} steps | "
                f"Rolling Acc: {rolling_acc:.3f} | "
                f"Cumulative Acc: {np.mean(sampler.acceptance_fraction):.3f} | "
                f"Time: {elapsed:.2f} mins",
                flush=True
            )

        # Export post-burnin chain to CSV for compatibility with existing scripts
        burn_in = N_STEPS // 4
        flat_chain = sampler.get_chain(discard=burn_in, flat=True)
        flat_probs = sampler.get_log_prob(discard=burn_in, flat=True)

        df_out = pd.DataFrame({
            "p_val": flat_chain[:, 0],
            "log_k1_MHz": flat_chain[:, 1],
            "A_val": flat_chain[:, 2],
            "log_prob": flat_probs
        })
        df_out.to_csv(CSV_CHAIN_FILE, index=False)
        print(f"--- LOSS FIT COMPLETE | Saved HDF5 & CSV to {FIT_DIR} ---", flush=True)
