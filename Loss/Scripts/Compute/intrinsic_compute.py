import numpy as np
import os
import sys
import time
import pandas as pd
from scipy.special import exp1
import emcee
from pathlib import Path

# =====================================
# 1. DIRECTORY & CONFIGURATION
# =====================================
# Inject the root 'Scripts' directory into Python's path and import the config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import get_paths

# Fetch paths for the currently active route
cfg = get_paths()

DATA_DIR = cfg["data_dir"]
COMP_DIR = cfg["comp_dir"]

INPUT_FILE = cfg["processed_csv"]
OUTPUT_FILE = COMP_DIR / "S0_Compute.csv"

# --- Best Fit Parameters ---
A = 1.2795e-6
B = 0.7895
C = 3.5424
D = 1.1125e9    # V/(m s)

# --- Intrinsic Loss Model ---
def calc_model(Eb_array, A, B, C, D):
    try:
        ratio = D / Eb_array
        u0 = ratio
        u1 = ratio * np.exp(2 * C)

        integral_val = 0.5 * (exp1(u0) - exp1(u1))

        result = A * np.exp(B * integral_val)
        return result - A   # Matching the processed data's zero anchor
    
    except Exception as e:
        print(f"Error in calc_model: {e}")
        return np.full_like(Eb_array, np.inf)

def load_data(filepath):
    df = pd.read_csv(filepath)

    Power = df['power'].values
    n_ph = df['n_ph'].values
    bias_data = df['dE_dt'].values
    exp_loss_data = df['y_exp'].values
    tau_loss_data = df['y_tau'].values
    S_loss_data = df['new_y_S'].values
        
    return n_ph, bias_data, exp_loss_data, tau_loss_data, S_loss_data, Power


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("--- STARTING TAN_S0 COMPUTATION ---", flush=True)
    n_data, dE_dt, y_exp, y_tau, y_S, power = load_data(INPUT_FILE)
    
    y_S0 = calc_model(dE_dt, A, B, C, D)

    df_out = pd.DataFrame({
        "power": power,
        "n_ph": n_data,
        "dE_dt": dE_dt,
        "y_exp": y_exp,
        "y_tau": y_tau,
        "y_S": y_S,
        "tan_S0": y_S0
    })

    df_out.to_csv(OUTPUT_FILE, index=False)
