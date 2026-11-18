import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

# =====================================
# 1. DIRECTORY & CONFIGURATION
# =====================================
# Resolve the root 'Scripts' directory whether run from Scripts/ or a subfolder
THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR if (THIS_DIR / "project_config.py").exists() else THIS_DIR.parent
if not (SCRIPTS_DIR / "project_config.py").exists() and (SCRIPTS_DIR / "Scripts").exists():
    SCRIPTS_DIR = SCRIPTS_DIR / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths

# Fetch paths for the currently active route
cfg = get_paths()
COMP_DIR = Path(cfg["comp_dir"])
FIT_DIR = Path(cfg["fit_dir"])
COMPARISON_DIR = Path(cfg["comparison_dir"])
DATAFILE = Path(cfg["final_csv"])          # comp_dir / "final_results.csv"

SPATIAL_MODE = cfg["spatial_mode"]
V_BIAS = cfg.get("V_bias", 5.0)
V_TAG = f"{float(V_BIAS):g}V" if isinstance(V_BIAS, (int, float)) or str(V_BIAS).replace(".", "").isdigit() else str(V_BIAS)

PLOTS_DIR = COMP_DIR / "Plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# Local voltage-folder plots & resonator-level comparison plots
LOCAL_PLOTFILE = PLOTS_DIR / "Loss_Plot.png"
COMP_PLOTFILE = COMPARISON_DIR / f"Loss_Plot_{V_TAG}_{SPATIAL_MODE}.png"

# Fallback tau-TLS parameters (in absolute units) if summary CSV is not found
DEFAULT_TAN0_TAU = 10.0**(-5.0440)
DEFAULT_TAN_BG = 10.0**(-5.5085)


def _load_tau_params(fit_dir):
    """Loads tan0_tau and tan_bg from the tau-TLS best-fit CSV if available."""
    for candidate in ("Best_Parameters_tau.csv", "Best_Parameters_Tau.csv", "tau_best_params.csv"):
        csv_path = fit_dir / candidate
        if csv_path.exists():
            try:
                df_p = pd.read_csv(csv_path).set_index("Parameter")
                tan0 = 10.0 ** float(df_p.loc["log_tan0", "MAP (Best Fit)"])
                tan_bg = 10.0 ** float(df_p.loc["log_tan_bg", "MAP (Best Fit)"])
                return tan0, tan_bg
            except Exception:
                pass
    return DEFAULT_TAN0_TAU, DEFAULT_TAN_BG


def _regime_stats(res_ppm, y_exp_ppm, global_range_ppm):
    rmse = float(np.sqrt(np.mean(res_ppm ** 2)))
    mae = float(np.mean(np.abs(res_ppm)))
    bias = float(np.mean(res_ppm))
    ss_res = float(np.sum(res_ppm ** 2))
    ss_tot = float(np.sum((y_exp_ppm - np.mean(y_exp_ppm)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    loc_range = float(np.max(y_exp_ppm) - np.min(y_exp_ppm))
    nrmse_loc = 100.0 * rmse / loc_range if loc_range > 0 else np.nan
    nrmse_glob = 100.0 * rmse / global_range_ppm
    return {
        "N": int(len(res_ppm)),
        "RMSE (ppm)": rmse,
        "MAE (ppm)": mae,
        "Bias (ppm)": bias,
        "R^2": r2,
        "Local NRMSE (%)": nrmse_loc,
        "Global NRMSE (%)": nrmse_glob,
    }


def plot_curves_new(
    input_file,
    output_file=None,
    comparison_file=None,
    sweep_cuts=(1.698e9, 4.80e10),   # [V/(m*s)] -> 1.70e3 and 4.80e4 V/(um*s)
    power_cuts=(-55.0, -25.0),       # [dBm]     -> Low / Mid / High drive power
):
    df = pd.read_csv(input_file)

    # Optional stripping of noisy params
    # df = df[(df["power"] >= -75.0) & (df["power"] <= -11.0)].copy()
    
    df.columns = df.columns.str.strip()

    if "y_S" not in df.columns:
        raise KeyError("Column 'y_S' not found in the input CSV.")

    tan0_tau, tan_bg = _load_tau_params(FIT_DIR)

    df_sorted = df.sort_values(by=["power", "dE_dt"]).copy()

    # Reconstruct total model loss, adding tan_bg if y_tau is purely the saturable component
    if "y_total" not in df_sorted.columns:
        if df_sorted["y_tau"].min() < 0.5 * tan_bg:
            df_sorted["y_total"] = df_sorted["y_tau"] + df_sorted["y_S"] + tan_bg
        else:
            df_sorted["y_total"] = df_sorted["y_tau"] + df_sorted["y_S"]

    unique_powers = np.sort(df_sorted["power"].unique())

    # --- 1. Global & 3-Regime Fit Quality Calculations ---
    y_exp_ppm = df_sorted["y_exp"].to_numpy() * 1e6
    y_tot_ppm = df_sorted["y_total"].to_numpy() * 1e6
    res_ppm = y_exp_ppm - y_tot_ppm
    global_range_ppm = float(np.max(y_exp_ppm) - np.min(y_exp_ppm))

    glob_stats = _regime_stats(res_ppm, y_exp_ppm, global_range_ppm)

    # 3 Regimes by Bias Field Sweep Rate (dE_dt)
    sw_cut1, sw_cut2 = sweep_cuts
    eb_arr = df_sorted["dE_dt"].to_numpy()
    sweep_masks = {
        f"Slow (<= {sw_cut1*1e-6:.2e} V/um*s)": eb_arr <= sw_cut1,
        f"Mid ({sw_cut1*1e-6:.2e} - {sw_cut2*1e-6:.2e} V/um*s)": (eb_arr > sw_cut1) & (eb_arr <= sw_cut2),
        f"Fast (> {sw_cut2*1e-6:.2e} V/um*s)": eb_arr > sw_cut2,
    }
    sweep_results = {
        k: _regime_stats(res_ppm[m], y_exp_ppm[m], global_range_ppm)
        for k, m in sweep_masks.items()
    }

    # 3 Regimes by Drive Power (P [dBm])
    pw_cut1, pw_cut2 = power_cuts
    pwr_arr = df_sorted["power"].to_numpy()
    power_masks = {
        f"Low (P <= {pw_cut1:.0f} dBm)": pwr_arr <= pw_cut1,
        f"Mid ({pw_cut1:.0f} < P < {pw_cut2:.0f} dBm)": (pwr_arr > pw_cut1) & (pwr_arr < pw_cut2),
        f"High (P >= {pw_cut2:.0f} dBm)": pwr_arr >= pw_cut2,
    }
    power_results = {
        k: _regime_stats(res_ppm[m], y_exp_ppm[m], global_range_ppm)
        for k, m in power_masks.items()
    }

    # Print formatted summary tables to console
    print("\n=== GLOBAL FIT QUALITY ===")
    print(pd.Series(glob_stats).round(4).to_string())
    print("\n=== 3 REGIMES BY BIAS FIELD SWEEP RATE ===")
    print(pd.DataFrame(sweep_results).T.round(4).to_string())
    print("\n=== 3 REGIMES BY DRIVE POWER ===")
    print(pd.DataFrame(power_results).T.round(4).to_string())

    # --- 2. Setup 2-Panel GridSpec with Dedicated Colorbar Axis ---
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.2, 1.0],
        width_ratios=[40, 1.2],
        hspace=0.08,
        wspace=0.03
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)
    cax = fig.add_subplot(gs[:, 1])

    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=unique_powers.min(), vmax=unique_powers.max())
    TAN_PLATEAU = (tan0_tau + tan_bg) * 1e5  # Unsaturated slow-sweep plateau (x 10^-5)

    # --- 3. Plot Curves Grouped by Drive Power ---
    for P in unique_powers:
        if P < -100:
            continue
        subset = df_sorted[df_sorted["power"] == P]
        x_eb = subset["dE_dt"].to_numpy() * 1e-6       # V / (um * s)
        y_exp_sub = subset["y_exp"].to_numpy() * 1e5   # 10^-5
        y_tot_sub = subset["y_total"].to_numpy() * 1e5
        res_sub = y_exp_sub - y_tot_sub
        col = cmap(norm(P))

        ax_main.scatter(x_eb, y_exp_sub, color=col, s=26, alpha=0.85, edgecolors="none", zorder=3)
        ax_main.plot(x_eb, y_tot_sub, color=col, linestyle="--", linewidth=1.5, alpha=0.9, zorder=2)
        ax_res.plot(x_eb, res_sub, "o-", color=col, markersize=3.5, linewidth=0.9, alpha=0.75)

    # --- 4. Reference Lines & Side-by-Side Regime Metric Boxes ---
    ax_main.axhline(
        TAN_PLATEAU, color="k", linestyle=":", linewidth=1.8,
        label=rf"$\tan\delta_0^\tau + \tan\delta_{{\mathrm{{bg}}}} = {TAN_PLATEAU:.2f} \times 10^{{5}}$"
    )
    ax_main.axvline(
        sw_cut1 * 1e-6, color="crimson", linestyle="-.", linewidth=1.5, alpha=0.75,
        label=rf"Slow/Mid Sweep Cut (${sw_cut1*1e-9:.2f}\times 10^3\ \mathrm{{V/(\mu m\cdot s)}}$)"
    )
    ax_main.axvline(
        sw_cut2 * 1e-6, color="teal", linestyle="-.", linewidth=1.5, alpha=0.75,
        label=rf"Mid/Fast Sweep Cut (${sw_cut2*1e-6/1e4:.2f}\times 10^4\ \mathrm{{V/(\mu m\cdot s)}}$)"
    )
    ax_res.axvline(sw_cut1 * 1e-6, color="crimson", linestyle="-.", linewidth=1.2, alpha=0.6)
    ax_res.axvline(sw_cut2 * 1e-6, color="teal", linestyle="-.", linewidth=1.2, alpha=0.6)

    ax_main.scatter([], [], color="gray", s=35, label=r"Experimental Data ($y_{\mathrm{exp}}$)")
    ax_main.plot([], [], color="gray", linestyle="--", linewidth=1.8, label=r"Total Model ($y_\tau + y_S + \tan\delta_{\mathrm{bg}}$)")

    sw_vals = list(sweep_results.values())
    pw_vals = list(power_results.values())

    box_left_text = (
        f"Global Fit ($N = {glob_stats['N']}$)\n"
        f"• $R^2 = {glob_stats['R^2']:.4f}$ | NRMSE $= {glob_stats['Global NRMSE (%)']:.2f}\\%$\n"
        f"• RMSE $= {glob_stats['RMSE (ppm)']:.2f}$ ppm | MAE $= {glob_stats['MAE (ppm)']:.2f}$ ppm\n"
        f"─────────────────────────────\n"
        f"By Sweep Rate $\\dot{{E}}_b$ [V/($\\mu$m$\\cdot$s)]:\n"
        f"• Slow ($\\leq 1.7\\times 10^3$): {sw_vals[0]['RMSE (ppm)']:.2f} ppm\n"
        f"• Mid ($1.7\\times 10^3–4.8\\times 10^4$): {sw_vals[1]['RMSE (ppm)']:.2f} ppm\n"
        f"• Fast ($> 4.8\\times 10^4$): {sw_vals[2]['RMSE (ppm)']:.2f} ppm"
    )
    box_right_text = (
        f"By Drive Power $P$ [dBm]:\n"
        f"• Low ($P \\leq {pw_cut1:.0f}$ dBm):\n"
        f"  RMSE $= {pw_vals[0]['RMSE (ppm)']:.2f}$ ppm ($N={pw_vals[0]['N']}$)\n"
        f"• Mid (${pw_cut1:.0f} < P < {pw_cut2:.0f}$ dBm):\n"
        f"  RMSE $= {pw_vals[1]['RMSE (ppm)']:.2f}$ ppm ($N={pw_vals[1]['N']}$)\n"
        f"• High ($P \\geq {pw_cut2:.0f}$ dBm):\n"
        f"  RMSE $= {pw_vals[2]['RMSE (ppm)']:.2f}$ ppm ($N={pw_vals[2]['N']}$)"
    )

    ax_main.text(
        0.015, 0.81, box_left_text,
        transform=ax_main.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="gray", alpha=0.94),
        zorder=5
    )
    ax_main.text(
        0.345, 0.81, box_right_text,
        transform=ax_main.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="gray", alpha=0.94),
        zorder=5
    )

    ax_main.set_xscale("log")
    ax_main.set_ylim(-0.08, 3.05)
    ax_main.set_ylabel(r"Loss Tangent $\tan\delta$ ($10^{-5}$)", fontsize=16)
    ax_main.set_title(
        rf"Total TLS Loss Tangent ($\tau$-TLS + $S$-TLS) — {V_TAG}, {SPATIAL_MODE}",
        fontsize=17.5, fontweight="bold", pad=12
    )
    ax_main.legend(loc="upper left", fontsize=10.5, framealpha=0.94, ncol=2)
    ax_main.tick_params(labelsize=13, labelbottom=False)
    ax_main.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- 5. Residual Panel Formatting ---
    rmse_e5_glob = glob_stats["RMSE (ppm)"] * 0.1
    ax_res.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax_res.axhline(rmse_e5_glob, color="crimson", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_res.axhline(-rmse_e5_glob, color="crimson", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_res.set_xscale("log")
    ax_res.set_xlabel(r"Bias Field Sweep Rate $\dot{E}_b$ [$\mathrm{V / (\mu m \cdot s)}$]", fontsize=16)
    ax_res.set_ylabel(r"Residual ($10^{-5}$)", fontsize=13)
    ax_res.tick_params(labelsize=13)
    ax_res.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- 6. Colorbar with Drive Power Regime Dividers ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Drive Power [dBm]", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.axhline(pw_cut1, color="white", linestyle="--", linewidth=1.8)
    cbar.ax.axhline(pw_cut2, color="black", linestyle="--", linewidth=1.8)

    # Save to both the voltage-specific Plots folder and the resonator-level Comparisons folder
    save_local = output_file if output_file is not None else LOCAL_PLOTFILE
    fig.savefig(save_local, dpi=300, bbox_inches="tight")
    print(f"\nSaved local plot to: {save_local}")

    save_comp = comparison_file if comparison_file is not None else COMP_PLOTFILE
    if save_comp is not None and Path(save_comp) != Path(save_local):
        fig.savefig(save_comp, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_comp}")


if __name__ == "__main__":
    plot_curves_new(DATAFILE, LOCAL_PLOTFILE, COMP_PLOTFILE)
