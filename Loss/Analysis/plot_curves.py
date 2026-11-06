import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

# Filepath Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "Outputs" / "Computes"
DATAFILE = OUTPUT_DIR / "processed_compute.csv"
# DATAFILE = SCRIPT_DIR.parent / "Data" / "New" / "res6978__bias5V_powersweep_Dec4.csv" #"5umGapRes_6789MHz_data" / "resonator_data_5V.csv"
ALL_PLOTFILE = OUTPUT_DIR / "Loss_Plot.png"
S_PLOTFILE = OUTPUT_DIR / "S_Loss_Plot.png"

LEGACY_POWERS = np.linspace(-65, -5, num=5)

# Data parsing and plotting
def plot_curves_all(input_file):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by='Power')
    unique_powers = df_sorted['Power'].unique()

    colors = [cm.tab20(x) for x in np.linspace(0, 1, len(unique_powers))]
    plt.figure(figsize=[14, 10])

    TAN0 = 1.14169475    # With Fitted Absorber

    for i, P in enumerate(unique_powers):
        subset = df_sorted[df_sorted["Power"] == P]

        # if P == -75:
        plt.scatter(subset['dE_dt']*1e-6, subset['y_exp']*1e5,
                    color=colors[i], label=f'{P} dBm: Exp')
        # plt.scatter(subset['dE_dt']*1e-6, subset['y_tau']*1e5,
        #             color=colors[i], marker='*', label=rf'{P} dBm: $\tau$-TLS')
        # plt.scatter(subset['dE_dt']*1e-6, subset['y_S']*1e5,
                #   color=colors[i], label=f'{P} dBm: S-TLS')
        plt.axhline(TAN0, color='k', linestyle='--')

        plt.xscale('log')
        plt.xlabel(r'$\dot{E}_{DC}$ (V/$\mu$m s)', fontsize=20)
        plt.ylabel(r'$\tan\delta$ (x $10^6$)', fontsize=20)
        plt.title('TLS Loss Tangent vs Bias Field Sweep Rate', fontsize=24, fontweight='bold')

        plt.legend(bbox_to_anchor=(1.22, 1), loc='upper right', fontsize=12)
        plt.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(ALL_PLOTFILE, dpi=300)
    plt.show()

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
    plt.savefig(S_PLOTFILE, dpi=300)
    plt.show()

def plot_curves_new(input_file):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by='dE_dt')
    unique_powers = df_sorted['power'].unique()

    colors = [cm.plasma(x) for x in np.linspace(0, 1, len(unique_powers))]
    plt.figure(figsize=[14, 10])

    TAN0 = 1.14169475    # With Fitted Absorber

    for i, P in enumerate(unique_powers):
        subset = df_sorted[df_sorted["power"] == P]

        y_total = subset['y_tau'] + subset['y_S']
        xi = subset['dE_dt'] / subset['n_ph']

        if P >= -100:
            plt.plot(xi*1e-6, subset['y_exp']*1e5,
                        color=colors[i], marker='o', label=f'{P} dBm: Exp')
            # plt.plot(subset_sorted['dE_dt']*1e-6, subset_sorted['y_tau']*1e5,
            #             color=colors[i], linestyle='--', label=rf'{P} dBm: $\tau$-TLS')
            # plt.scatter(subset_sorted['dE_dt']*1e-6, subset_sorted['y_S']*1e5,
            #           color=colors[i], label=f'{P} dBm: S-TLS')
        #     plt.plot(subset['dE_dt']*1e-6, y_total*1e5,
        #               color=colors[i], linestyle='--', label=f'{P} dBm: Total Loss')
        # plt.axhline(TAN0, color='k', linestyle='--')

        plt.xlabel(r'Bias Voltage Frequency $\dot{E}_b$ (V / ($\mu$m s))', fontsize=20)
        plt.ylabel(r'$\tan\delta$ (x $10^5$)', fontsize=20)
        plt.title('TLS Loss Tangent vs Bias Field Sweep Rate', fontsize=24, fontweight='bold')

        plt.xscale('log')
        # plt.legend(bbox_to_anchor=(1.22, 1), loc='upper right', fontsize=12)
        plt.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(S_PLOTFILE, dpi=300)
    plt.show()

if __name__ == "__main__":
    # plot_curves_all(DATAFILE)
    # plot_curves_separate(DATAFILE)
    plot_curves_new(DATAFILE)
