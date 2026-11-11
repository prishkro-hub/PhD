import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths
cfg = get_paths()

DATAFILE = cfg["raw_csv"]

def find_tls_cutoff(df, min_power=-75.0, max_power=-60.0):
    """
    Calculates the tau-TLS to S-TLS crossover frequency using 
    asymptote intersection on the unsaturated low-power envelope.
    """
    # 1. Isolate fully unsaturated envelope
    window = df[(df['drive_power'] >= min_power) & (df['drive_power'] <= max_power)]
    y_low = window.groupby('bias_frequency')['loss_tangent'].mean().values * 1e6
    nu = np.sort(window['bias_frequency'].unique())
    # eb = low_pwr.groupby('bias_frequency')['dE_dt'].mean().values
    
    # 2. Flat tau-TLS baseline (average of first 3 frequencies)
    tau_baseline = np.mean(y_low[:3])
    
    # 3. Rising S-TLS asymptote (log-log fit of last 4 frequencies)
    log_nu = np.log10(nu)
    log_y = np.log10(y_low)
    slope, intercept, _, _, _ = linregress(log_nu[-4:], log_y[-4:])
    
    # 4. Intersection point
    log_nu_cut = (np.log10(tau_baseline) - intercept) / slope
    nu_cut = 10**log_nu_cut
    
    # # Map back to sweep rate
    # eb_cut = np.interp(nu_cut, nu)
    
    return {
        "nu_cutoff_Hz": nu_cut,
        # "eb_cutoff_Vms": eb_cut,
        "tau_baseline_ppm": tau_baseline,
        "s_tls_slope": slope
    }

if __name__ == "__main__":
    df = pd.read_csv(DATAFILE)
    cutoffs = find_tls_cutoff(df)
    print(
        f"Baseline tau-TLS Loss: {cutoffs['tau_baseline_ppm']:.4f}\n"
        f"S-TLS Loss Slope: {cutoffs['s_tls_slope']:.4f}\n"
        f"Cutoff at {cutoffs['nu_cutoff_Hz']:.0f} Hz"
    )
