import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Scripts"
if not SCRIPTS_DIR.exists():
    SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths, update_config_cutoff

cfg = get_paths()
DATAFILE = cfg["clean_csv"] if cfg["clean_csv"].exists() else cfg["raw_csv"]
PWR_MIN = cfg["tan_s0"]["pwr_min"]
PWR_MAX = cfg["tan_s0"]["pwr_max"]


def find_tls_cutoff(df, min_power=PWR_MIN, max_power=PWR_MAX):
    """
    Calculates the tau-TLS to S-TLS crossover frequency using
    asymptote intersection on the unsaturated low-power envelope.
    """
    # 1. Isolate fully unsaturated envelope
    window = df[(df["drive_power"] >= min_power) & (df["drive_power"] <= max_power)]
    grouped = window.groupby("bias_frequency")["loss_tangent"].mean().sort_index()
    nu = grouped.index.to_numpy()
    y_low = grouped.to_numpy() * 1e6

    if len(nu) < 8:
        print(
            f"[WARNING] Only {len(nu)} unique frequencies found (max = {nu.max():.0f} Hz). "
            f"Asymptote fit (-4:) overlaps or sits too close to baseline (:3)!"
        )

    # 2. Flat tau-TLS baseline (average of first 3 frequencies)
    tau_baseline = float(np.mean(y_low[:3]))

    # 3. Rising S-TLS asymptote (log-log fit of last 4 frequencies)
    log_nu = np.log10(nu)
    log_y = np.log10(y_low)
    slope, intercept, _, _, _ = linregress(log_nu[-4:], log_y[-4:])

    # 4. Intersection point
    log_nu_cut = (np.log10(tau_baseline) - intercept) / slope
    nu_cut = float(10.0 ** log_nu_cut)

    return {
        "nu_cutoff_Hz": nu_cut,
        "tau_baseline_ppm": tau_baseline,
        "s_tls_slope": float(slope),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATAFILE)
    df.columns = df.columns.str.strip()

    cutoffs = find_tls_cutoff(df)
    print(
        f"Baseline tau-TLS Loss : {cutoffs['tau_baseline_ppm']:.4f} ppm\n"
        f"S-TLS Loss Slope      : {cutoffs['s_tls_slope']:.4f}\n"
        f"Cutoff Frequency      : {cutoffs['nu_cutoff_Hz']:.1f} Hz"
    )

    # Automatically write the new cutoff frequency into config.toml
    update_config_cutoff(cutoffs["nu_cutoff_Hz"])
