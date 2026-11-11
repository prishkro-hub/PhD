import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

# Filepath Configuration
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths

cfg = get_paths()

DATA_DIR = cfg["data_dir"]
COMP_DIR = cfg["comp_dir"]
FINAL_DIR = cfg["comparison_dir"]
VOLT = f"{cfg["V_bias"]}V"

PLOT_MODE = "data"  # data/comp/fit
if PLOT_MODE == "data":
    DATAFILE = cfg["raw_csv"]
    PLOTFILE = DATA_DIR / "Plots" / "Raw_Plot.png"
elif PLOT_MODE == "comp":
    DATAFILE = cfg["processed_csv"]
    PLOTFILE = COMP_DIR / "Plots" / "Comp_Plot.png"
else:
    DATAFILE = cfg["final_csv"]
    PLOTFILE = FINAL_DIR / f"{VOLT}_Plot.png"

LEGACY_POWERS = np.linspace(-65, -5, num=5)

def plot_curves_separate(input_file):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by='dE_dt')
    unique_powers = df_sorted['power'].unique()

    colors = [cm.tab20(x) for x in np.linspace(0, 1, len(unique_powers))]
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True, figsize=[14,10])
    fig.suptitle('Tau-TLS and S-TLS Loss Tangents vs Bias Field Sweep Rate', fontsize=24, fontweight='bold')
    fig.supxlabel(r'$\dot{E}_{DC}$ ($V/(\mu m\cdot s)$)', fontsize=20)
    fig.supylabel(r'$\tan\delta$ (x $10^6$)', fontsize=20)
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    TAN0 = 9.19363263     # With Fitted Absorber

    for i, P in enumerate(unique_powers):
        subset = df_sorted[df_sorted["power"] == P]
        if P > -95:
            # ax1.scatter(subset['dE_dt']*1e-6, subset['y_exp']*1e6,
            #             color=colors[i], label=rf'{P} dBm: Exp')
            # ax1.scatter(subset['dE_dt']*1e-6, subset['Raw_y_S']*1e6,
            #             color=colors[i], marker='*', label=f'{P} dBm')
            ax2.scatter(subset['dE_dt']*1e-6, subset['y_S']*1e6,
                        color=colors[i], label=f'{P} dBm: Clamped S-TLS')
            ax2.scatter(subset['dE_dt']*1e-6, subset['tan_S0']*1e6,
                     color=colors[i], marker='^', label=f'{P} dBm: Intrinsic S-TLS')

        # ax1.set_title('Raw S Loss', fontsize=20)
        ax2.set_title('Clamped S Loss', fontsize=20)
        # ax1.legend(loc='best', ncols=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTFILE, dpi=300)
    plt.show()

def plot_curves_new(input_file):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by='bias_frequency')
    unique_powers = df_sorted['drive_power'].unique()

    colors = [cm.plasma(x) for x in np.linspace(0, 1, len(unique_powers))]
    plt.figure(figsize=[14, 10])

    # TAN0 = 1.14169475    # With Fitted Absorber

    for i, P in enumerate(unique_powers):
        subset = df_sorted[df_sorted["drive_power"] == P]

        # y_total = subset['y_tau'] + subset['y_S']
        xi = subset['bias_frequency'] / subset['photon_number']

        if P >= -100:
            plt.plot(subset['bias_frequency'], subset['loss_tangent']*1e6,
                        color=colors[i], marker='o', label=f'{P} dBm: Exp')
            # plt.plot(subset_sorted['dE_dt']*1e-6, subset_sorted['y_tau']*1e5,
            #             color=colors[i], linestyle='--', label=rf'{P} dBm: $\tau$-TLS')
            # plt.scatter(subset_sorted['dE_dt']*1e-6, subset_sorted['y_S']*1e5,
            #           color=colors[i], label=f'{P} dBm: S-TLS')
        #     plt.plot(subset['dE_dt']*1e-6, y_total*1e5,
        #               color=colors[i], linestyle='--', label=f'{P} dBm: Total Loss')
        # plt.axhline(TAN0, color='k', linestyle='--')

        plt.xlabel(r'Bias Voltage Frequency $\nu$ (Hz)', fontsize=20)
        plt.ylabel(r'$\tan\delta$ (x $10^6$)', fontsize=20)
        plt.title('TLS Loss Tangent vs Voltage Sweep Rate', fontsize=24, fontweight='bold')

        plt.xscale('log')
        # plt.legend(bbox_to_anchor=(1.22, 1), loc='upper right', fontsize=12)
        plt.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(PLOTFILE, dpi=300)
    plt.show()

if __name__ == "__main__":
    # plot_curves_all(DATAFILE)
    # plot_curves_separate(DATAFILE)
    plot_curves_new(DATAFILE)
