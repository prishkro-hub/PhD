import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emcee
from pathlib import Path

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    print("Warning: 'corner' module not found. Run 'pip install corner' for contour plots.")
    CORNER_AVAILABLE = False

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_BASE = SCRIPT_DIR.parent / "Outputs"
OUTPUT_DIR = OUTPUTS_BASE / "Fits"
PLOTS_DIR = OUTPUT_DIR / "Plots"

CHAIN_FILE = OUTPUT_DIR / "tan_s0_fit.csv"
PLOT_PREFIX = "Tan_S0"

N_WALKERS = 32
NDIM = 4
MANUAL_CUT = 200
MIN_BURN_IN = 500  # Increased for a 10000-step MCMC run

def detect_straggler_burnin(chain, buffer_steps=100, mad_Limit=6.0):
    """
    Scans active walkers to find the last step where any walker was still traveling
    into the final equilibrium band (defined by the last 25% of the chain).
    """
    n_steps = chain.shape[0]
    tail_start = max(1, int(0.75 * n_steps))
    tail_samples = chain[tail_start:, :, :].reshape(-1, chain.shape[2])

    eq_med = np.median(tail_samples, axis=0)
    eq_mad = np.median(np.abs(tail_samples - eq_med), axis=0)
    eq_mad = np.where(eq_mad < 1e-12, 1.0, eq_mad)

    # Max normalized deviation across all walkers and parameters at each step t
    dev_per_step = np.max(np.abs(chain - eq_med[None, None, :]) / eq_mad[None, None, :], axis=(1, 2))
    out_of_eq = np.where(dev_per_step > mad_Limit)[0]

    if len(out_of_eq) == 0:
        return 0

    last_bad_step = int(out_of_eq[-1])
    # Only auto-trim if the straggler converged within the first 70% of the remaining chain
    if last_bad_step < int(0.70 * n_steps):
        return min(n_steps - 100, last_bad_step + buffer_steps)
    return 0

# Walker pruning thresholds
MAX_DELTA_LOGPROB = 25.0  # Reject walkers with median log_prob > 25 below best walker
MAD_THRESHOLD = 5.0       # Reject walkers > 5 MAD away from ensemble median


def load_chain_data(chain_path, n_walkers=32, ndim=3):
    """
    Loads 3D chain (n_steps, n_walkers, ndim) and 2D log_probs (n_steps, n_walkers)
    directly from an emcee .h5 backend or a flat .csv file.
    """
    if not chain_path.exists():
        print(f"Error: Could not find {chain_path}. Skipping...")
        return None, None

    if chain_path.suffix in [".h5", ".hdf5"]:
        reader = emcee.backends.HDFBackend(str(chain_path), read_only=True)
        chain = reader.get_chain()          # Native shape: (n_steps, n_walkers, ndim)
        log_probs = reader.get_log_prob()   # Native shape: (n_steps, n_walkers)
        print(f"Successfully loaded HDF5 backend from {chain_path.name} | Shape: {chain.shape}")
        return chain, log_probs

    # Fallback for older CSV chains
    df = pd.read_csv(chain_path)
    df.columns = df.columns.str.strip()
    cols = ["log_A", "B", "C", "log_D"]
    if not all(c in df.columns for c in cols):
        print(f"Error: CSV missing required columns. Found: {df.columns.to_list()}")
        return None, None

    n_steps = len(df) // n_walkers
    df = df.iloc[: n_steps * n_walkers]
    chain = df[cols].values.reshape(n_steps, n_walkers, ndim)
    log_probs = df["log_prob"].values.reshape(n_steps, n_walkers)
    print(f"Successfully loaded CSV data from {chain_path.name} | Shape: {chain.shape}")
    return chain, log_probs


def prune_stuck_walkers(chain, log_probs, max_delta_lp=25.0, mad_thresh=5.0):
    """
    Identifies and removes stranded walkers along axis=1 using median log-probability
    and Median Absolute Deviation (MAD) across walkers.
    """
    half_idx = max(1, chain.shape[0] // 2)
    walker_med_lp = np.median(log_probs[half_idx:, :], axis=0)
    best_med_lp = np.max(walker_med_lp)

    # 1. Log-probability filter
    lp_mask = (best_med_lp - walker_med_lp) <= max_delta_lp

    # 2. Parameter MAD filter across walkers
    walker_med_params = np.median(chain[half_idx:, :, :], axis=0)
    ens_median = np.median(walker_med_params, axis=0)
    mad = np.median(np.abs(walker_med_params - ens_median), axis=0)
    mad = np.where(mad < 1e-12, 1.0, mad)

    max_dev = np.max(np.abs(walker_med_params - ens_median) / mad, axis=1)
    mad_mask = max_dev <= mad_thresh

    good_mask = lp_mask & mad_mask
    bad_indices = np.where(~good_mask)[0]

    if len(bad_indices) > 0:
        print(f"\n[PRUNING] Filtering out {len(bad_indices)} stuck walker(s): {bad_indices.tolist()}")
        for idx in bad_indices:
            A_stuck, B_stuck, C_stuck, D_stuck = walker_med_params[idx]
            print(
                f"   -> Walker {idx:2d}: A={A_stuck:.2f}, "
                f"B={B_stuck:.4f}, C={C_stuck:.4f}, D={D_stuck:.2f}"
                f"(median log_prob={walker_med_lp[idx]:.1f} vs best={best_med_lp:.1f})"
            )

    return chain[:, good_mask, :], log_probs[:, good_mask]


def analyze_bloch_fit():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}\nANALYZING STEP 3 FIT ({CHAIN_FILE})\n{'='*50}")

    chain_file = OUTPUT_DIR / CHAIN_FILE
    chain, log_probs = load_chain_data(chain_file, N_WALKERS, NDIM)
    if chain is None:
        return None

    n_steps_total, n_walkers_loaded, _ = chain.shape

    # # Convert log10 parameters to physical units: A and D
    # chain[:, :, 0] = 10.0 ** chain[:, :, 0]
    # chain[:, :, 3] = 10.0 ** chain[:, :, 3]

    # 1. Apply initial manual warm-up cut
    if n_steps_total > MANUAL_CUT:
        chain = chain[MANUAL_CUT:, :, :]
        log_probs = log_probs[MANUAL_CUT:, :]
        print(f"\n[INFO] Sliced off first {MANUAL_CUT} warm-up steps ({chain.shape[0]} steps remaining).")

    # 2a. Filter out stuck walkers along axis=1
    chain, log_probs = prune_stuck_walkers(
        chain, log_probs, max_delta_lp=MAX_DELTA_LOGPROB, mad_thresh=MAD_THRESHOLD
    )

    # 2b. Auto-trim any late-converging straggler steps (like the jump at step ~1600)
    straggler_cut = detect_straggler_burnin(chain)
    if straggler_cut > 0:
        chain = chain[straggler_cut:, :, :]
        log_probs = log_probs[straggler_cut:, :]
        MANUAL_CUT_EFFECTIVE = MANUAL_CUT + straggler_cut
        print(f"[AUTO-BURN] Trimmed {straggler_cut} additional straggler convergence steps (total cut = {MANUAL_CUT_EFFECTIVE}).")
    else:
        MANUAL_CUT_EFFECTIVE = MANUAL_CUT

    n_steps, n_good_walkers, _ = chain.shape
    print(f"[INFO] Active ensemble: {n_good_walkers}/{n_walkers_loaded} walkers over {n_steps} steps.")

    # 3. Autocorrelation Analysis
    labels_text = ["A", "B", "C", "D"]
    units = ["", "", "", "V/(um s)"]
    plot_labels = ["A", "B", "C", r"D (V/$\mu$m s)"]

    try:
        tau_array = emcee.autocorr.integrated_time(chain, tol=0)
        max_tau = float(np.max(tau_array))

        print("\nAutocorrelation Times (tau):")
        for lbl, tau_val in zip(labels_text, tau_array):
            print(f"  {lbl:<10} : {tau_val:.2f} steps")

        rec_burn = int(5 * max_tau)
        rec_len = int(50 * max_tau)
        print("\nConvergence Diagnostics:")
        print(f" -> Recommended total burn-in : {rec_burn} steps")
        print(f" -> Recommended chain length  : {rec_len} steps")

        if n_steps + MANUAL_CUT_EFFECTIVE < rec_len:
            print(f"[WARNING] Chains shorter than 50*tau ({n_steps + MANUAL_CUT_EFFECTIVE} < {rec_len})")
        else:
            print(f"[SUCCESS] Chains well-converged! ({n_steps} post-cut steps)")
    except Exception as e:
        print(f"\n[WARNING] Could not calculate autocorrelation: {e}")
        max_tau = 0.0

    # 4. Determine remaining burn-in to discard
    target_total_burn = max(MIN_BURN_IN, int(5 * max_tau))
    extra_burn = max(0, target_total_burn - MANUAL_CUT_EFFECTIVE)

    if n_steps <= extra_burn:
        print(f"[ERROR] Remaining chain ({n_steps}) shorter than extra burn-in ({extra_burn})")
        return None

    if extra_burn > 0:
        print(f"\nDiscarding {extra_burn} additional burn-in steps (total burn-in = {MANUAL_CUT_EFFECTIVE + extra_burn})...")
        chain_post = chain[extra_burn:, :, :]
        log_probs_post = log_probs[extra_burn:, :]
    else:
        print(f"\nMANUAL_CUT ({MANUAL_CUT_EFFECTIVE}) already exceeds recommended burn-in ({target_total_burn}). Keeping all {n_steps} post-cut steps.")
        chain_post = chain
        log_probs_post = log_probs

    flat_samples = chain_post.reshape(-1, NDIM)
    flat_log_probs = log_probs_post.flatten()

    # 5. Parameter Extraction (MAP & Posterior Percentiles)
    map_idx = np.argmax(flat_log_probs)
    map_vals = flat_samples[map_idx]

    p16, median, p84 = np.percentile(flat_samples, [16, 50, 84], axis=0)
    lower = median - p16
    upper = p84 - median

    print(f"\n{'Parameter':<15} | {'MAP (Best Fit)':<15} | {'Median (Posterior)':<25}")
    print("-" * 65)
    summary_data = []
    for i in range(NDIM):
        val_fmt = "{:.4f}"# if (i == 1 or i == 2) else "{:.4e}"
        print(
            f"{labels_text[i]:<15} | {val_fmt.format(map_vals[i]):<15} | "
            f"{val_fmt.format(median[i])} (-{val_fmt.format(lower[i])} / +{val_fmt.format(upper[i])}) {units[i]}"
        )
        summary_data.append({
            "Parameter": labels_text[i],
            "MAP (Best Fit)": val_fmt.format(map_vals[i]),
            "Median (Posterior)": val_fmt.format(median[i]),
            "Error (-)": val_fmt.format(lower[i]),
            "Error (+)": val_fmt.format(upper[i]),
            "Units": units[i]
        })

    # 6. Plotting: Post-Cut Walker Traces
    n_post = chain_post.shape[0]
    step_axis = np.arange(MANUAL_CUT_EFFECTIVE + extra_burn, MANUAL_CUT_EFFECTIVE + extra_burn + n_post)

    fig, axes = plt.subplots(NDIM, figsize=(12, 10), sharex=True)
    for i in range(NDIM):
        ax = axes[i]
        ax.plot(step_axis, chain_post[:, :, i], "k", alpha=0.3)
        ax.axhline(median[i], color="red", linestyle="--", linewidth=1.2, label="Median" if i == 0 else None)
        ax.set_xlim(step_axis[0], step_axis[-1])
        ax.set_ylabel(plot_labels[i], fontsize=11)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Step Number (Post-Cut)", fontsize=12)
    plt.suptitle(f"Post-Cut Walker Traces ({n_good_walkers} Converged Walkers)", fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{PLOT_PREFIX}_Traces.png", dpi=300)
    plt.close()

    # 7. Plotting: Corner Plot
    if CORNER_AVAILABLE:
        corner_labels = [f"{lbl}\n{med:.3g}" for lbl, med in zip(labels_text, median)]
        fig = corner.corner(
            flat_samples,
            labels=corner_labels,
            truths=map_vals,
            truth_color="red",
            quantiles=[0.16, 0.5, 0.84],
            color="darkred",
            label_kwargs={"fontsize": 10}
        )
        fig.subplots_adjust(top=0.92)
        plt.suptitle("Corner Plot Posteriors (Global Fit)", fontsize=16, fontweight="bold")
        plt.savefig(PLOTS_DIR / f"{PLOT_PREFIX}_Corner.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 8. Plotting: 1D Posterior Histograms (Percentage of Samples per Bin)
    fig_hist, axes_hist = plt.subplots(1, NDIM, figsize=(15, 5))
    weights_pct = np.full(len(flat_samples), 100.0 / len(flat_samples))

    for i in range(NDIM):
        ax = axes_hist[i]
        ax.hist(flat_samples[:, i], bins=80, weights=weights_pct, alpha=0.6, color="maroon")
        ax.axvline(median[i], color="k", linestyle="dashed", linewidth=2, label=f"Median: {median[i]:.3g}")
        ax.axvline(p16[i], color="red", linestyle="-", linewidth=2, alpha=0.8, label=r"1$\sigma$")
        ax.axvline(p84[i], color="red", linestyle="-", linewidth=2, alpha=0.8)
        ax.axvline(map_vals[i], color="green", linestyle="-.", linewidth=2, label=f"MAP: {map_vals[i]:.3g}")
        ax.set_title(plot_labels[i], fontsize=14)
        if i == 0:
            ax.set_ylabel("Posterior Probability (%)", fontsize=12)
        ax.legend(fontsize="small", loc="best")

    plt.suptitle("Global Posterior Distributions", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{PLOT_PREFIX}_Hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    df_summary = analyze_bloch_fit()

    if df_summary is not None:
        print(f"\n{'='*50}\nCOMPILING SUMMARY TABLE\n{'='*50}")
        summary_path = OUTPUT_DIR / f"Best_Parameters_{PLOT_PREFIX}.csv"
        df_summary.to_csv(summary_path, index=False)
        print(df_summary.to_string(index=False))
        print(f"\nSaved summary to: {summary_path}")
