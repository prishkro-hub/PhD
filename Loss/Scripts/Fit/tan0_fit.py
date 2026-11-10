import sys
import emcee
import numpy as np
import pandas as pd
from pathlib import Path
import time
from scipy.stats import linregress
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
spatial_data = np.load(cfg["spatial_npz"])
E_mean = float(spatial_data["E_mean"])
E_vac = float(spatial_data["E_vac"])
V = 5

# E_k = spatial_data["E_k"]
# W_k = spatial_data["W_k"]

# MCMC Hyperparameters
N_DIM = 3
N_WALKERS = 32
N_STEPS = 10000
N_BATCH = 500

# ==================================================
# 2. Log Probability of Saturated + Background Loss
# ==================================================
def log_probability(theta, n_data, loss_data):
    log_tan0, log_nc, log_tan_bg = theta

    if not (-7.0 < log_tan0 < -4.0 and 0.0 < log_nc < 6.0 and -8.0 < log_tan_bg < -4.0):
        return -np.inf

    tan0 = 10**log_tan0
    nc = 10**log_nc
    tan_bg = 10**log_tan_bg

    model_loss = (tan0 / np.sqrt(1.0 + (n_data / nc))) + tan_bg

    sigma = 3e-7
    chi2 = np.sum(((loss_data - model_loss) / sigma) ** 2)

    if np.isnan(chi2) or not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2

# ==================================================
# 3. Load Data & Calculate Fields
# ==================================================
def load_data(filepath, low_power_thresh=-55.0):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # 1. Find the tau-TLS regime
    low_pwr = df[df['drive_power'] <= low_power_thresh]
    y_low = low_pwr.groupby('bias_frequency')['loss_tangent'].mean().values * 1e6
    nu_arr = np.sort(low_pwr['bias_frequency'].unique())

    tau_baseline = np.mean(y_low[:3])
    log_nu = np.log10(nu_arr)
    log_y = np.log10(y_low)
    slope, intercept, _, _, _ = linregress(log_nu[-4:], log_y[-4:])

    log_nu_cut = (np.log10(tau_baseline) - intercept) / slope
    nu_cut = 10 ** log_nu_cut

    print(f"Dynamically calculated tau-TLS cutoff: {nu_cut:.0f} Hz", flush=True)

    # 1. Filter dataset
    valid_mask = (
        np.isfinite(df['photon_number'])
        & np.isfinite(df['loss_tangent'])
        & (df['bias_frequency'] <= nu_cut)
    )

    df_active = df[valid_mask].copy()

    n_data = df_active['photon_number'].values
    loss_data = df_active['loss_tangent'].values

    print(f"Restricted fit to {len(loss_data)} slow-sweep data points.", flush=True)

    return n_data, loss_data

# ==================================================
# 4. Execution Wrapper
# ==================================================
if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("--- STARTING TAN0 + BACKGROUND FIT ---", flush=True)
        print("MPI Pool successfully established! Starting Sampler...", flush=True)

        n_data, loss_data= load_data(INPUT_FILE)

        backend = emcee.backends.HDFBackend(str(BACKEND_FILE))
        backend.reset(N_WALKERS, N_DIM)
    
        p0_center = np.array([-5.0, 2.3, -5.5])
        jitter = np.array([0.05, 0.2, 0.05])

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
                (emcee.moves.StretchMove(a=2.0), 0.7),
                (emcee.moves.DEMove(), 0.2)
                (emcee.moves.DESnookerMove(), 0.1)
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

        # Save CSV for posterity
        flat_samples = sampler.get_chain(discard=N_STEPS//4, flat=True)
        df_out = pd.DataFrame(flat_samples, columns=["log_tan0", "log_nc", "log_tan_bg"])
        df_out.to_csv(CHAIN_FILE, index=False)

    print(f"--- Intrinsic tau-Loss Fit COMPLETE | Saved HDF5 & CSV to {FIT_DIR} ---", flush=True)
