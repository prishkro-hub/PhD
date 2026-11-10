from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================
# 1. LOAD DATA
# ====================================================
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "resonator_data_5V.csv").exists():
    DATA_DIR = CURRENT_DIR
elif (CURRENT_DIR.parent / "Data" / "5umGapRes_6789MHz_data").exists():
    DATA_DIR = CURRENT_DIR.parent / "Data" / "5umGapRes_6789MHz_data"
else:
    DATA_DIR = CURRENT_DIR.parents[2] / "Data" / "5umGapRes_6789MHz_data"

INPUT_FILE = DATA_DIR / "resonator_data_5V.csv"
OUT_PLOT = DATA_DIR / "Loss_Cutoff_Clean_3Panel.png"

df = pd.read_csv(INPUT_FILE)
col_nu = "bias_frequency" if "bias_frequency" in df.columns else "Voltage Frequency"
col_nph = "photon_number" if "photon_number" in df.columns else "Photon Occupation"
col_loss = "loss_tangent" if "loss_tangent" in df.columns else "Loss Tangent"

df = df.dropna(subset=[col_nu, col_nph, col_loss]).sort_values([col_nu, col_nph]).copy()
df["power_idx"] = df.groupby(col_nu).cumcount()
df["loss_ppm"] = df[col_loss] * 1e6

# Pivot table: rows = bias_frequency (12), cols = power_idx (43)
piv_loss = df.pivot(index=col_nu, columns="power_idx", values="loss_ppm")
piv_nph = df.pivot(index=col_nu, columns="power_idx", values=col_nph)
freqs = piv_loss.index.values
nph_med = piv_nph.median(axis=0).values

# Subtract 100 Hz baseline for each power index: Delta tan(delta) [ppm]
delta_loss = piv_loss.subtract(piv_loss.iloc[0, :], axis=1)

# ====================================================
# 2. BUILD 3-PANEL DIAGNOSTIC FIGURE
# ====================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.0, 11.5), sharex=True)

# --- PANEL 1: 6 Decadal Representative Curves ---
target_nphs = [0.02, 0.2, 2.0, 50.0, 2000.0, 200000.0]
cmap = plt.get_cmap("plasma")
chosen_indices = sorted(list({int(np.argmin(np.abs(nph_med - tn))) for tn in target_nphs}))

for k, idx in enumerate(chosen_indices):
    color = cmap(k / max(1, len(chosen_indices) - 1))
    ax1.plot(
        freqs, piv_loss[idx].values, "o-",
        color=color, linewidth=2.0, markersize=5.5,
        label=rf"$n_{{\mathrm{{ph}}}} \approx {nph_med[idx]:.2g}$"
    )

ax1.set_ylabel(r"$\tan\delta$ [ppm]", fontsize=12)
ax1.set_title("1. Six Representative Drive Powers (1 Per Decade)", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", ncol=3, fontsize=10)

# --- PANEL 2: Excess Loss Above 100 Hz in 4 Photon Bands ---
bands = [
    ("Linear Baseline ($n_{\\mathrm{ph}} \\leq 1$): Pure S-TLS Onset Indicator", nph_med <= 1.0, "navy"),
    ("Early Knee ($1 < n_{\\mathrm{ph}} \\leq 100$)", (nph_med > 1.0) & (nph_med <= 100.0), "teal"),
    ("LZ Transition ($100 < n_{\\mathrm{ph}} \\leq 10^4$)", (nph_med > 100.0) & (nph_med <= 1e4), "darkorange"),
    ("Deep Saturation ($n_{\\mathrm{ph}} > 10^4$)", nph_med > 1e4, "crimson"),
]

for label, mask, color in bands:
    mean_delta = delta_loss.loc[:, mask].mean(axis=1).values
    std_delta = delta_loss.loc[:, mask].std(axis=1).values
    ax2.plot(freqs, mean_delta, "s-", color=color, linewidth=2.2, markersize=5.5, label=label)
    ax2.fill_between(freqs, mean_delta - std_delta, mean_delta + std_delta, color=color, alpha=0.15)

ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
ax2.set_ylabel(r"$\Delta\tan\delta(\nu) - \tan\delta(100\mathrm{\ Hz})$ [ppm]", fontsize=12)
ax2.set_title("2. Change Relative to 100 Hz (Band-Averaged ± 1 SD)", fontsize=13, fontweight="bold")
ax2.legend(loc="upper left", fontsize=10)

# --- PANEL 3: Local Slope d(tan delta) / d(log10 nu) ---
log_freqs = np.log10(freqs)
for label, mask, color in bands:
    mean_curve = piv_loss.loc[:, mask].mean(axis=1).values
    slope = np.gradient(mean_curve, log_freqs)
    ax3.plot(freqs, slope, "^-", color=color, linewidth=2.0, markersize=5.5, label=label)

ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
ax3.set_ylabel(r"Slope $d(\tan\delta)/d(\log_{10}\nu)$ [ppm/dec]", fontsize=12)
ax3.set_xlabel(r"Bias Voltage Sweep Frequency $\nu$ [Hz]", fontsize=13)
ax3.set_title("3. Local Logarithmic Slope (Onset of S-TLS vs. LZ De-Saturation)", fontsize=13, fontweight="bold")
ax3.legend(loc="upper left", fontsize=10)

for ax in (ax1, ax2, ax3):
    ax.axvline(1233.0, color="darkorange", linestyle=":", linewidth=1.8, label="1233 Hz (4th freq)")
    ax.axvline(2848.0, color="crimson", linestyle="--", linewidth=1.5, label="2848 Hz (5th freq)")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved clean 3-panel cutoff plot to: {OUT_PLOT}")
