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
if not SCRIPTS_DIR.exists():
    SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths
cfg = get_paths()

DATA_DIR = cfg["data_dir"]
COMP_DIR = cfg["comp_dir"]
FINAL_DIR = cfg["comparison_dir"]
VOLT = f"{cfg['V_bias']:g}V"
TAN0 = 10.0**(6.0 - 5.0251)   # ppm
TAN_BG = 10.0**(-5.6801)

PLOT_MODE = "comp"  # Choose from: "data", "comp", "fit"

# --- High-Power Spike Filter Settings (Used in PLOT_MODE == "data") ---
APPLY_FILTER = True
SHOW_EXCISED_POINTS = True   # Marks dropped spikes with red 'x' on Raw_Plot.png
CLEAN_CSV_OUT = DATA_DIR / "resonator_data_clean.csv"

ANCHOR_MIN_PWR = -35.0       # dBm
ANCHOR_MAX_PWR = -27.0       # Powers <= -27 dBm form the clean upper ceiling
EPS_PPM = 0.25               # VNA jitter tolerance (ppm)
MIN_SLOPE_FRAC = 0.20        # Min fractional rise vs. anchor during steep S-TLS climb
MAX_SLOPE_FRAC = 1.20        # Max fractional rise vs. anchor over any frequency step

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
# 2. 2D PHYSICAL MONOTONICITY FILTER
# =====================================
def filter_raw_dataframe(df):
    """
    Flags VNA circle-fit spikes, flares, and Duffing plunges at P > ANCHOR_MAX_PWR
    using the clean [-35, -27] dBm curves as a physical monotonicity reference.
    Returns (df_clean, df_dropped).
    """
    piv = (
        df.pivot(index="drive_power", columns="bias_frequency", values="loss_tangent")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    powers = piv.index.to_numpy()
    y_mat = piv.to_numpy() * 1e6  # Evaluate in ppm

    n_p, n_f = y_mat.shape
    is_clean = np.isfinite(y_mat)

    anchor_rows = np.where((powers >= ANCHOR_MIN_PWR) & (powers <= ANCHOR_MAX_PWR))[0]
    if len(anchor_rows) == 0:
        return df.copy(), df.iloc[0:0].copy()

    y_ceil = np.nanmin(y_mat[anchor_rows, :], axis=0)
    y_anchor_curve = y_mat[anchor_rows[-1], :]
    running_ceil = y_ceil.copy()

    for i in range(n_p):
        if powers[i] <= ANCHOR_MAX_PWR:
            continue

        y_curve = y_mat[i, :]

        # Pass A: Cannot exceed the clean anchor ceiling (-27 dBm)
        for k in range(n_f):
            if is_clean[i, k] and (y_curve[k] > y_ceil[k] + EPS_PPM):
                is_clean[i, k] = False

        # Pass B: Along-trace frequency monotonicity, spike lookahead, and slope bounds
        valid_ks = np.where(is_clean[i, :])[0]
        if len(valid_ks) == 0:
            continue
        last_valid_k = valid_ks[0]

        for k in range(last_valid_k + 1, n_f):
            if not is_clean[i, k]:
                continue
            y_prev = y_curve[last_valid_k]
            ref_dy = y_anchor_curve[k] - y_anchor_curve[last_valid_k]

            # 1. Lookahead for interior upward spikes
            future_candidates = [
                m for m in range(k + 1, n_f)
                if is_clean[i, m] and (y_curve[m] >= y_prev - EPS_PPM)
            ]
            if any(y_curve[k] > y_curve[m] + EPS_PPM for m in future_candidates):
                is_clean[i, k] = False
                continue

            # 2. Downward plunge or high-frequency stagnation check
            min_dy = MIN_SLOPE_FRAC * max(ref_dy - 0.5, 0.0)
            if (y_curve[k] - y_prev) < min_dy - EPS_PPM:
                is_clean[i, k] = False
                continue

            # 3. Excessive upward flare check
            if (y_curve[k] - y_prev) > max(MAX_SLOPE_FRAC * ref_dy, ref_dy + 0.5):
                is_clean[i, k] = False
                continue

            last_valid_k = k

        # Pass C: Cross-power monotonicity against running ceiling from lower powers
        for k in range(n_f):
            if not is_clean[i, k]:
                continue
            if y_curve[k] > running_ceil[k] + EPS_PPM:
                is_clean[i, k] = False
            else:
                if k == 0 or is_clean[i, k - 1]:
                    running_ceil[k] = min(running_ceil[k], y_curve[k])

    clean_lookup = pd.DataFrame(is_clean, index=piv.index, columns=piv.columns)
    mask = np.array(
        [clean_lookup.loc[p, nu] for p, nu in zip(df["drive_power"], df["bias_frequency"])],
        dtype=bool
    )
    return df[mask].copy(), df[~mask].copy()


# =====================================
# 3. RAW DATA PLOTTER & CLEAN EXPORTER
# =====================================
def plot_raw_data(input_file, save_path):
    """Filters corrupted high-power points, exports clean CSV, and plots sweeps."""
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    if APPLY_FILTER:
        df_plot, df_dropped = filter_raw_dataframe(df)
        df_plot.to_csv(CLEAN_CSV_OUT, index=False)

        EXCISED_CSV_OUT = DATA_DIR / "resonator_data_excised.csv"
        df_dropped.sort_values(by=["drive_power", "bias_frequency"]).to_csv(EXCISED_CSV_OUT, index=False)

        n_total_Freqs = df["bias_frequency"].nunique()
        in_window = (df_dropped["drive_power"] >= -35.0) & (df_dropped["drive_power"] <= -11.0)
        n_drop_window = int(in_window.sum())
        n_drop_high = int((~in_window).sum())

        print(f"\nSaved cleaned dataset to: {CLEAN_CSV_OUT}")
        print(f"Saved excised audit log to: {EXCISED_CSV_OUT}")
        print(
            f"Total Excised Points: {len(df_dropped)} "
            f"({n_drop_window} in [-35, -11] dBm fit window | {n_drop_high} at P > -11 dBm)"
        )

        if not df_dropped.empty:
            print("\n=== EXCISED POINTS BY DRIVE POWER ===")
            print(f"{'Power (dBm)':<13} | {'Window':<10} | {'Count':<8} | {'Excised (Freq [Hz] : Loss [ppm])'}")
            print("-" * 88)

            for pwr, grp in df_dropped.sort_values("bias_frequency").groupby("drive_power"):
                win_tag = "FIT [-35,-11]" if (-35.0 <= pwr <= -11.0) else "HIGH (>-11)"
                cnt_str = f"{len(grp)}/{n_total_Freqs}"
                pts_str = ", ".join(
                    f"{row['bias_frequency']:.0f} Hz ({row['loss_tangent']*1e6:.2f} ppm)"
                    for _, row in grp.iterrows()
                )
                print(f"{pwr:>7.1f} dBm   | {win_tag:<13} | {cnt_str:<8} | {pts_str}")
            print("-" * 88 + "\n")
    else:
        df_plot, df_dropped = df.copy(), df.iloc[0:0].copy()

    df_sorted = df_plot.sort_values(by="bias_frequency")
    unique_powers = np.sort(df["drive_power"].unique())

    # --- Colorbar Setup ---
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=unique_powers.min(), vmax=unique_powers.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=[12, 9])

    for P in unique_powers:
        if 11 >= P >= -135:
            subset = df_sorted[df_sorted["drive_power"] == P]
            color = cmap(norm(P))

            xi = subset["bias_frequency"] / subset["photon_number"]

            ax.plot(
                subset["bias_frequency"],
                subset["loss_tangent"] * 1e6,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5
            )

            if APPLY_FILTER and SHOW_EXCISED_POINTS and not df_dropped.empty:
                drop_sub = df_dropped[df_dropped["drive_power"] == P]
                if not drop_sub.empty:
                    ax.scatter(
                        drop_sub["bias_frequency"],
                        drop_sub["loss_tangent"] * 1e6,
                        color="crimson",
                        marker="x",
                        s=55,
                        linewidths=1.5,
                        zorder=5
                    )

    if APPLY_FILTER and SHOW_EXCISED_POINTS and not df_dropped.empty:
        ax.scatter([], [], color="crimson", marker="x", s=55, label="Excised Spike/Plunge")
        ax.legend(loc="upper left", fontsize=14)

    # Add tan_0 reference line
    ax.axhline(TAN0, color="k", linestyle="--")

    # --- Formatting ---
    ax.set_xscale("log")
    ax.set_xlabel(r"Bias Voltage Frequency $\nu$ (Hz)", fontsize=20)
    ax.set_ylabel(r"Raw $\tan\delta$ ($\times 10^6$)", fontsize=20)
    ax.set_title("Raw TLS Loss Tangent vs Bias Frequency", fontsize=24, fontweight="bold")
    ax.tick_params(labelsize=16)

    # --- Add Colorbar ---
    cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.03)
    cbar.set_label("Drive Power (dBm)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Raw data plot saved to: {save_path}")
    # plt.show()


# =====================================
# 4. PROCESSED / COMPONENT PLOTTER
# =====================================
def plot_processed_data(input_file, save_path):
    """Plots the separated tau-TLS and clamped S-TLS loss components."""
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df_sorted = df.sort_values(by="dE_dt")
    unique_powers = np.sort(df_sorted["power"].unique())

    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=unique_powers.min(), vmax=unique_powers.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[16, 9])
    fig.suptitle("Separated TLS Components vs Bias Sweep Rate", fontsize=24, fontweight="bold")
    fig.supxlabel(r"$\dot{E}_{DC}$ ($V/(\mu m\cdot s)$)", fontsize=20)
    fig.supylabel(r"$\tan\delta$ ($\times 10^6$)", fontsize=20)

    ax1.set_xscale("log")
    ax2.set_xscale("log")

    for P in unique_powers:
        if 7 >= P > -110:
            subset = df_sorted[df_sorted["power"] == P]
            color = cmap(norm(P))

            ax1.plot(
                subset["dE_dt"] * 1e-6,
                subset["y_tau"] * 1e6,
                color=color,
                marker="s",
                markersize=4,
                linewidth=1.0
            )
            ax2.plot(
                subset["dE_dt"] * 1e-6,
                subset["y_S"] * 1e6,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.0
            )

    ax1.axhline(TAN0, color="k", linestyle="--", label=r"$\tan(\delta_0^\tau)$")
    ax2.axhline(TAN0, color="k", linestyle="--", label=r"$\tan(\delta_0^\tau)$")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    ax1.set_title(r"$\tau$-TLS Component", fontsize=20)
    ax2.set_title("Clamped S-TLS Component", fontsize=20)

    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)

    cbar = fig.colorbar(sm, ax=ax2, aspect=30, pad=0.03)
    cbar.set_label("Drive Power (dBm)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Processed data plot saved to: {save_path}")
    # plt.show()


# =====================================
# 5. EXECUTION
# =====================================
if __name__ == "__main__":
    if PLOT_MODE == "data":
        plot_raw_data(DATAFILE, PLOTFILE)
    elif PLOT_MODE == "comp":
        plot_processed_data(DATAFILE, PLOTFILE)
    else:
        print("Final plotting mode not yet configured.")
