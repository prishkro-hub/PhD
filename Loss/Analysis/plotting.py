import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from pathlib import Path

# =====================================
# 1. DIRECTORY & CONFIGURATION
# =====================================
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths
cfg = get_paths()

DATA_DIR = cfg["data_dir"]
COMP_DIR = cfg["comp_dir"]
FINAL_DIR = cfg["comparison_dir"]
VOLT = f"{cfg['V_bias']}V"
TAN0 = 9.04   # ppm
TAN_BG = 3.10e-6

PLOT_MODE = "comp"  # Choose from: "data", "comp", "fit"
if PLOT_MODE == "data":
    DATAFILE = cfg["raw_csv"]
    PLOTFILE = DATA_DIR / "Plots" / "Raw_Plot.png"
elif PLOT_MODE == "comp":
    DATAFILE = cfg["processed_csv"]
    PLOTFILE = COMP_DIR / "Plots" / "Comp_Plot.png"
else:
    DATAFILE = cfg["final_csv"]
    PLOTFILE = FINAL_DIR / f"{VOLT}_Plot.png"

# =====================================
# 2. RAW DATA PLOTTER
# =====================================
def plot_raw_data(input_file, save_path):
    """Plots the raw, unbroken experimental loss tangent sweeps."""
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by='bias_frequency')
    unique_powers = np.sort(df_sorted['drive_power'].unique())

    # --- Colorbar Setup ---
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=unique_powers.min(), vmax=unique_powers.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=[12, 9])
    
    for P in unique_powers:
        if -60 >= P >= -75:
            subset = df_sorted[df_sorted["drive_power"] == P]
            color = cmap(norm(P))
            ax.plot(
                subset['bias_frequency'], 
                subset['loss_tangent'] * 1e6,
                color=color, 
                marker='o', 
                markersize=4,
                linewidth=1.5
            )

    # Add tan_0 reference line
    ax.axhline(TAN0, color='k', linestyle='--')

    # --- Formatting ---
    ax.set_xscale('log')
    ax.set_xlabel(r'Bias Voltage Frequency $\nu$ (Hz)', fontsize=20)
    ax.set_ylabel(r'Raw $\tan\delta$ (x $10^6$)', fontsize=20)
    ax.set_title('Raw TLS Loss Tangent vs Bias Frequency', fontsize=24, fontweight='bold')
    ax.tick_params(labelsize=16)

    # --- Add Colorbar ---
    cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.03)
    cbar.set_label('Drive Power (dBm)', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Raw data plot saved to: {save_path}")
    plt.show()

# =====================================
# 3. PROCESSED / COMPONENT PLOTTER
# =====================================
def plot_processed_data(input_file, save_path):
    """Plots the separated tau-TLS and clamped S-TLS loss components."""
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by='dE_dt')
    unique_powers = np.sort(df_sorted['power'].unique())

    # --- Colorbar Setup ---
    cmap = cm.plasma # Or cm.viridis / cm.plasma if preferred
    norm = mcolors.Normalize(vmin=unique_powers.min(), vmax=unique_powers.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[16, 9])
    fig.suptitle('Separated TLS Components vs Bias Sweep Rate', fontsize=24, fontweight='bold')
    fig.supxlabel(r'$\dot{E}_{DC}$ ($V/(\mu m\cdot s)$)', fontsize=20)
    fig.supylabel(r'$\tan\delta$ (x $10^6$)', fontsize=20)
    
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    for P in unique_powers:
        if P >= -95:
            subset = df_sorted[df_sorted["power"] == P]
            color = cmap(norm(P))
            
            # Left Subplot: tau-TLS
            ax1.plot(
                subset['dE_dt'] * 1e-6, 
                subset['y_tau'] * 1e6,
                color=color, 
                marker='s', 
                markersize=4,
                linewidth=1.0
            )
            
            # Right Subplot: Clamped S-TLS
            ax2.plot(
                subset['dE_dt'] * 1e-6, 
                subset['y_S'] * 1e6,
                color=color, 
                marker='o', 
                markersize=4,
                linewidth=1.0
            )
    
    # Add tan_0 reference line
    ax1.axhline(TAN0, color='k', linestyle='--', label=r"$\tan(\delta_0^\tau)$")
    ax2.axhline(TAN0, color='k', linestyle='--', label=r"$\tan(\delta_0^\tau)$")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    ax1.set_title(r'$\tau$-TLS Component', fontsize=20)
    ax2.set_title('Clamped S-TLS Component', fontsize=20)
    
    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)

    # --- Add Colorbar to Right Axis ---
    cbar = fig.colorbar(sm, ax=ax2, aspect=30, pad=0.03)
    cbar.set_label('Drive Power (dBm)', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Processed data plot saved to: {save_path}")
    plt.show()

# =====================================
# 4. EXECUTION
# =====================================
if __name__ == "__main__":
    if PLOT_MODE == "data":
        plot_raw_data(DATAFILE, PLOTFILE)
    elif PLOT_MODE == "comp":
        plot_processed_data(DATAFILE, PLOTFILE)
    else:
        print("Final plotting mode not yet configured.")
