import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import emcee
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
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

GRID_FILE = cfg["s_grid"]

# Outputs
BACKEND_FILE = cfg["s_backend"]
CSV_CHAIN_FILE = cfg["s_chain"]

# S-TLS config parameters
s_cfg = cfg["s_tls"]

PWR_MIN = s_cfg["pwr_min"]
PWR_MAX = s_cfg["pwr_max"]

FIT_NORMALIZED_RATIO = s_cfg["fit_normalized_ratio"]
USE_P_PRIOR = s_cfg["use_p_prior"]
PRIOR_P_MEAN = s_cfg["prior_p_mean"]
PRIOR_P_STD = s_cfg["prior_p_std"]

USE_GAMMA1_PRIOR = s_cfg["use_gamma1_prior"]
PRIOR_LOG_K1_MEAN = s_cfg["prior_log_k1_mean"]
PRIOR_LOG_K1_STD = s_cfg["prior_log_k1_std"]

# MCMC Hyperparameters
N_DIM = 3
N_WALKERS = s_cfg["n_walkers"]
N_STEPS = s_cfg["n_steps"]
N_BATCH = s_cfg["n_batch"]

# =====================================
# 2. LOAD PRECOMPUTED GRID & SPLINE
# =====================================
grid_archive = np.load(GRID_FILE)
P_GRID = grid_archive["p_grid"]
LOG_P_GRID = np.log10(P_GRID)
G_GRID = grid_archive["g_grid"]

PWR_DATA = grid_archive["power"]
NU_DATA = grid_archive["nu"]

LOSS_GRID = grid_archive["loss_grid"]
LOSS_DATA = grid_archive["loss_data"]
TAN_S0_DATA = grid_archive["tan0_data"]# * 0.556396  # Fitted Absorber
NORM_LOSS_DATA = LOSS_DATA / TAN_S0_DATA

EB_DATA = grid_archive["dE_dt"]
NPH_DATA = grid_archive["n_ph"]

# Dynamic sigma: point-to-point diff along n_ph at each fixed bias frequency
SIGMA_RAW = np.zeros_like(LOSS_DATA)

for nu in np.unique(NU_DATA):
    m = (NU_DATA == nu)
    order = np.argsort(NPH_DATA[m])
    y_sorted = LOSS_DATA[m][order]

    if len(y_sorted) > 2:
        local_sigma = max(
            float(np.std(np.diff(y_sorted)) / np.sqrt(2.0)),
            s_cfg["sigma_min_floor"],
        )
    elif len(y_sorted) == 2:
        local_sigma = max(
            float(abs(y_sorted[1] - y_sorted[0]) / np.sqrt(2.0)),
            s_cfg["sigma_min_floor"],
        )
    else:
        local_sigma = s_cfg.get("sigma_single_fallback", 2.5e-7)

    SIGMA_RAW[m] = local_sigma

# Exact error propagation for NORM_LOSS_DATA = LOSS_DATA / TAN_S0_DATA
SIGMA_NORM = SIGMA_RAW / np.maximum(TAN_S0_DATA, 1e-12)

# print(f"SIGMA_RAW:  {SIGMA_RAW}")
# print(f"SIGMA_NORM: {SIGMA_NORM}")

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
    
    if not (P_MIN <= p_val <= P_MAX and G_MIN < log_k1_MHz < G_MAX and 0.8 < A_val < 1.2):
        return -np.inf

    unit_loss = LOSS_SPLINE(np.array([[np.log10(p_val), log_k1_MHz]]))[0]

    if FIT_NORMALIZED_RATIO:
        model = A_val * unit_loss
        chi2 = np.sum(((NORM_LOSS_DATA - model) / SIGMA_NORM) ** 2)
    else:
        model = A_val * TAN_S0_DATA * unit_loss
        chi2 = np.sum(((LOSS_DATA - model) / SIGMA_RAW) ** 2)

    if np.isnan(chi2) or not np.isfinite(chi2):
        return -np.inf

    log_prob = -0.5 * chi2

    # Tan_S0 is already optimized
    log_prob += -0.5 * ((A_val - 1.0) / 0.05) ** 2

    if USE_P_PRIOR and USE_GAMMA1_PRIOR:
        log_prob += -0.5 * ((p_val - PRIOR_P_MEAN) / PRIOR_P_STD) ** 2
        log_prob += -0.5 * ((log_k1_MHz - PRIOR_LOG_K1_MEAN) / PRIOR_LOG_K1_STD) ** 2
    elif USE_P_PRIOR:
        log_prob += -0.5 * ((p_val - PRIOR_P_MEAN) / PRIOR_P_STD) ** 2
    elif USE_GAMMA1_PRIOR:
        log_prob += -0.5 * ((log_k1_MHz - PRIOR_LOG_K1_MEAN) / PRIOR_LOG_K1_STD) ** 2
    
    return log_prob
        
# =====================================
# 4. MPI EXECUTION
# =====================================
if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print(
            f"--- STARTING HIGH-POWER S-LOSS MCMC ({SPATIAL_MODE}), "
            f"Powers [{PWR_MIN:.0f}, {PWR_MAX:.0f}] dBm, "
            f"N={len(LOSS_DATA)} ---",
            flush=True
        )

        backend = emcee.backends.HDFBackend(str(BACKEND_FILE))
        backend.reset(N_WALKERS, N_DIM)

        p0_center = np.array([150.0, 3.0, 1.0])
        jitter = np.array([2.0, 0.03, 0.005])
        
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
                (emcee.moves.StretchMove(a=3.0), 0.5),
                (emcee.moves.DEMove(), 0.3),
                (emcee.moves.DESnookerMove(), 0.2)
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
        print(f"--- S-LOSS FIT COMPLETE | Saved HDF5 & CSV to {FIT_DIR} ---", flush=True)
