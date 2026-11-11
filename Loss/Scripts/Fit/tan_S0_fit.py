import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import exp1
from scipy.optimize import minimize
import emcee
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

INPUT_FILE = cfg["processed_csv"]

BACKEND_FILE = FIT_DIR / f"tan_s0_fit.h5"
CSV_CHAIN_FILE = FIT_DIR / f"tan_s0_fit.csv"

# --- S-regime parameters
MIN_PWR = -75.0     # dBm
MAX_PWR = -60.0     # dBm
NU_CUT = 9702.0     # Hz
# --- MCMC Hyperparameters ---
N_DIM = 4
N_WALKERS = 32
N_STEPS = 20000
N_BATCH = 1000

# =====================================
# 2. INTRINSIC LOSS MODEL
# =====================================
def calc_model(Eb_array, A, B, C, D):
    ratio = D / Eb_array
    u0 = ratio
    u1 = ratio * np.exp(2 * C)

    integral_val = 0.5 * (exp1(u0) - exp1(u1))

    result = A * np.exp(B * integral_val)
    return result

def log_probability(theta, data_Eb, data_loss, sigma_arr):
    logA, B, C, logD = theta
    A = 10**logA
    D = 10**logD
    
    # Boundary constraints
    if not (-10.0 < logA < -4.0 and 0.0 < B < 5.0 and 0.0 < C < 12.0 and 5.0 < logD < 11.0):
        return -np.inf
        
    # 1. Full model numpy array
    model = calc_model(data_Eb, A, B, C, D)
    if not np.all(np.isfinite(model)):
        return -np.inf
    
    chi2 = np.sum(((data_loss - model) / sigma_arr) ** 2)
    if np.isnan(chi2) or not np.isfinite(chi2):
        return -np.inf

    # Soft Gaussian on C ~ 5.5 +/- 1.0 since C > 7 plateaus
    log_prior_C = -0.5 * ((C - 5.5) / 1.0) ** 2

    return -0.5 * chi2 + log_prior_C

# =====================================
# 3. LOAD AND FILTER DATA
# =====================================
def load_data(filepath):
    df = pd.read_csv(filepath)

    S_mask = (
        (df["power"] >= MIN_PWR)
        & (df["power"] <= MAX_PWR)
        & (df["bias_frequency"] >= NU_CUT)
    )
    sub = df[S_mask].copy()

    # Empirical per-frequency nois across low-power points
    sub["sigma"] = sub.groupby("bias_frequency")["y_S"].transform(
        lambda x: max(x.std(), 1.0e-7)
    )
    dE_dt_val = sub["dE_dt"].values
    loss_data_val = sub["y_S"].values
    sigma_val = sub["sigma"].values

    return dE_dt_val, loss_data_val, sigma_val

# =====================================
# 4. MCMC EXECUTION
# =====================================
if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("--- STARTING TAN_S0 FIT ---", flush=True)

        backend = emcee.backends.HDFBackend(str(BACKEND_FILE))
        backend.reset(N_WALKERS, N_DIM)
        
        dE_dt, loss_data, sigma_data = load_data(INPUT_FILE)
        
        # Start near theorized values
        p0_center = np.array([-6.8, 0.88, 5.5, 7.2])
        jitter = np.array([0.25, 0.1, 0.1, 0.25])

        p0, init = [], time.time()
        for _ in range(N_WALKERS):
            valid_spawn = False
            attempts = 0
            while not valid_spawn:
                guess = p0_center + jitter * np.random.randn(N_DIM)
                lp = log_probability(guess, dE_dt, loss_data, sigma_data)

                if np.isfinite(lp):
                    p0.append(guess)
                    valid_spawn = True
                attempts += 1

        initialized = time.time() - init
        print(f"All {N_WALKERS} walkers successfully initialized! | Time: {initialized:.2e} seconds | Attempts: {attempts}", flush=True)

        sampler = emcee.EnsembleSampler(
            N_WALKERS, N_DIM, log_probability,
            args=(dE_dt, loss_data, sigma_data),
            moves=[
                (emcee.moves.StretchMove(a=2.0), 0.7),
                (emcee.moves.DEMove(), 0.2),
                (emcee.moves.DESnookerMove(), 0.1)
            ],
            pool=pool,
            backend=backend
        )

        state, start_time = p0, time.time()
        prev_accepted = np.zeros(N_WALKERS)
    
        for batch_start in range(0, N_STEPS, N_BATCH):
            state, prob, rstate = sampler.run_mcmc(state, N_BATCH, progress=False)

            # Calculate the batch's rolling acceptance rate
            current_accepted = sampler.backend.accepted
            batch_accepted = current_accepted - prev_accepted
            rolling_acc = np.mean(batch_accepted) / N_BATCH

            # Update for the next iteration
            prev_accepted = current_accepted.copy()

            elapsed = time.time() - start_time
            print(
                f"Completed {batch_start + N_BATCH}/{N_STEPS} steps | "
                f"Rolling Acc: {rolling_acc:.3f} | "
                f"Cumulative Acc: {np.mean(sampler.acceptance_fraction):.3f} | "
                f"Time: {elapsed:.2f} sec",
                flush=True
            )

        # Export post burnin for clean analysis
        burnin = N_STEPS // 4
        flat_chain = sampler.get_chain(discard=burnin, flat=True)
        flat_probs = sampler.get_log_prob(discard=burnin, flat=True)

        df = pd.DataFrame({
            "log_A": flat_chain[:, 0],
            "B": flat_chain[:, 1],
            "C": flat_chain[:, 2],
            "log_D": flat_chain[:, 3],
            "log_prob": flat_probs
        }).dropna()

        df.to_csv(CSV_CHAIN_FILE, index=False)
        print(f"--- S-TLS INTRINSIC LOSS FIT COMPLETE | Saved HDF5 & CSV to {FIT_DIR} ---", flush=True)
