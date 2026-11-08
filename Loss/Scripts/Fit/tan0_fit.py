import sys
import emcee
import numpy as np
import pandas as pd
from pathlib import Path
import time
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

INPUT_FILE = cfg["raw_csv"]
BACKEND_FILE = FIT_DIR / "tan_0_fit.h5"
CHAIN_FILE = FIT_DIR / "tan_0_fit.csv"

# --- Get Spatial Field Data from 2D HFSS ---
spatial_data = cfg["spatial_npz"]
E_k = spatial_data["E_k"]
W_k = spatial_data["W_k"]
E_mean = float(spatial_data["E_mean"])
E_vac = float(spatial_data["E_vac"])
V = 5

# MCMC Hyperparameters
N_DIM = 3
N_WALKERS = 32
N_STEPS = 10000
N_BATCH = 500

# ==================================================
# 2. Log Probability of Saturated + Background Loss
# ==================================================
def log_probability(theta, n_data, loss_data):
    log_tan0, nc, log_tan_bg = theta

    if not (-7.0 < log_tan0 < -2.0 and 1.0 < nc < 1e6 and -8.0 < log_tan_bg < -4.0):
        return -np.inf

    tan0 = 10**log_tan0
    tan_bg = 10**log_tan_bg

    model_loss = (tan0 / np.sqrt(1.0 + (n_data / nc))) + tan_bg

    sigma = 1e-7
    chi2 = ((loss_data - model_loss) / sigma) ** 2

    log_prob = -0.5 * np.sum(chi2)

    if np.isnan(log_prob) or not np.isfinite(log_prob):
        return -np.inf

    return log_prob

# ==================================================
# 3. Load Data & Calculate Fields
# ==================================================
def load_data(filepath):
    df = pd.read_csv(filepath)

    n_data = df['photon_number'].values
    loss_data = df['loss_tangent'].values
    nu = df['bias_frequency'].values

    df['dE_dt'] = 2 * V * nu * E_mean
    df['E_ac'] = E_vac * np.sqrt(n_data)

    valid_mask = np.isfinite(n_data) & np.isfinite(loss_data)

    return n_data[valid_mask], loss_data[valid_mask], df["dE_dt"][valid_mask], df["E_ac"][valid_mask]

# ==================================================
# 4. Execution Wrapper
# ==================================================
if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("--- STARTING TAN0 FIT FIT ---", flush=True)
        print("MPI Pool successfully established! Starting Sampler...", flush=True)

        n_data, loss_data, Edot, Eac = load_data(INPUT_FILE)

        backend = emcee.backends.HDFBackend(str(BACKEND_FILE))
        backend.reset(N_WALKERS, N_DIM)
    
        p0_center = np.array([-4.0, 10.0, -6.0])
        jitter = np.array([0.5, 2.0, 0.5])

        p0, initial_time = [], time.time()
        for _ in range(N_WALKERS):
            valid_spawn = False
            attempts = 0
            while not valid_spawn:
                guess = p0_center + jitter * np.random.randn(N_DIM)

                lp = log_probability(guess, n_data, loss_data)

                if np.isfinite(lp):
                    p0.append(guess)
                    valid_spawn = True
                attempts += 1

        initialized = time.time() - initial_time
        
        print(f"All {N_WALKERS} walkers successfully initialized! | "
              f"Time: {initialized:.2f} s | "
              f"Attempts: {attempts}", flush=True
        )

        sampler = emcee.EnsembleSampler(
            N_WALKERS, N_DIM, log_probability,
            args=(n_data, loss_data),
            moves=[
                (emcee.moves.StretchMove(a=3.0), 0.8),
                (emcee.moves.DESnookerMove(), 0.2)
            ],
            pool=pool,
            backend=backend
        )

        state, start_time = p0, time.time()
        prev_accepted = np.zeros(N_WALKERS)

        for batch_start in range(0, N_STEPS, N_BATCH):
            state, prob, rstate = sampler.run_mcmc(state, N_BATCH, progress=False)

            current_accepted = sampler.backend.accepted
            batch_accepted = current_accepted - prev_accepted
            rolling_acc = np.mean(batch_accepted) / N_BATCH

            prev_accepted = current_accepted.copy()
            elapsed = (time.time() - start_time) / 60
            print(f"Completed {batch_start + N_BATCH}/{N_STEPS} steps | "
                  f"Rolling Acc: {rolling_acc:.3f} | "
                  f"Cumulative Acc: {np.mean(sampler.acceptance_fraction):.3f} | "
                  f"Time: {elapsed:.2f} mins", flush=True
            )

    print(f"--- Intrinsic tau-Loss Fit COMPLETE | Saved HDF5 to {FIT_DIR} ---", flush=True)
