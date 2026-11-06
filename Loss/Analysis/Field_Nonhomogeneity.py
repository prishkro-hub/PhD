import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Load Data ---
YANIV_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = YANIV_DIR / "Data" / "New" / "6978MHzRes_5Vbias" / "share"
OUTPUT_DIR = YANIV_DIR / "Outputs"

spatial_data = np.load(DATA_DIR / 'hfss_spatial_data.npz')
E_k = spatial_data['E_k']
W_k = spatial_data['W_k']
E_mean = float(spatial_data['E_mean'])

# --- Calculate Field Amplitude (|E|) ---
# Prevent division by zero for the exact center of the gap
epsilon = 1e-12
E_k_safe = np.where(E_k == 0, epsilon, E_k)
abs_E_k = np.abs(E_k_safe)

# --- Calculate Distributions ---
# 1. Electrical Distribution (Energy Participation)
W_k_norm = W_k / np.sum(W_k)

# 2. Geometric Distribution (Physical Volume Fraction)
# Volume is proportional to Energy / Field^2
V_k = W_k / (abs_E_k ** 2)
V_k_norm = V_k / np.sum(V_k)

# Sort arrays to ensure clean plotting
sort_idx = np.argsort(abs_E_k)
sorted_abs_E = abs_E_k[sort_idx]
sorted_W_norm = W_k_norm[sort_idx]
sorted_V_norm = V_k_norm[sort_idx]

normalized_E_amp = sorted_abs_E / E_mean
width_abs = (np.max(normalized_E_amp) - np.min(normalized_E_amp)) / len(normalized_E_amp)

# --- Check spatial profile's effect on bias sweep rates ---
bias = np.array([100, 231, 534, 1233, 2848, 6579, 15199, 35112, 81113, 187382, 432876, 1e6])
R_min = np.min(normalized_E_amp)
R_max = np.max(normalized_E_amp)
bias_R_min = R_min * bias
bias_R_max = R_max * bias
print(
    f"Minimum R (E_min / E_mean): {R_min:.2f}\n"
    f"Maximum R (E_max / E_mean): {R_max:.2f}\n"
)

print(
    f"Base Bias Voltage Frequencies [Hz]: {bias}\n"
    f"Bias Voltage Frequencies at R_min [Hz]: {bias_R_min}\n"
    f"Bias Voltage Frequencies at R_max [Hz]: {bias_R_max}\n"
)

dE_dt = 10 * bias * E_mean
dE_dt_min = 10 * bias_R_min * E_mean
dE_dt_max = 10 * bias_R_max * E_mean

print(
    f"RMS field sweep rates [V/(m*s)]: {dE_dt}\n"
    f"Minimum field sweep rates [V/(m*s)]: {dE_dt_min}\n"
    f"Maximum field sweep rates [V/(m*s)]: {dE_dt_max}\n"
)

tau_cutoff = 1.698e9

dE_dt_tau = dE_dt[dE_dt <= tau_cutoff]
dE_dt_min_tau = dE_dt_min[dE_dt_min <= tau_cutoff]
dE_dt_max_tau = dE_dt_max[dE_dt_max <= tau_cutoff]

print(
    f"Tau RMS field sweep rates [V/(m*s)]: {dE_dt_tau}\n"
    f"Tau Minimum field sweep rates [V/(m*s)]: {dE_dt_min_tau}\n"
    f"Tau Maximum field sweep rates [V/(m*s)]: {dE_dt_max_tau}"
)
# # --- Generate Side-by-Side Plot ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# # Panel 1: Physical Volume
# ax1.bar(normalized_E_amp, sorted_V_norm * 100, width=width_abs, color='teal', alpha=0.8)
# ax1.axvline(1.0, color='k', linestyle='--', label='Macroscopic Mean Field')
# ax1.set_xlabel('Field Amplitude (|E| / E_mean)', fontsize=14)
# ax1.set_ylabel('Physical Volume Fraction (%)', fontsize=14)
# ax1.set_title('Geometric Distribution (Volume)', fontsize=16, fontweight='bold')
# ax1.legend(fontsize=12)
# ax1.grid(axis='y', alpha=0.3)

# # Panel 2: Electrical Energy
# ax2.bar(normalized_E_amp, sorted_W_norm * 100, width=width_abs, color='maroon', alpha=0.8)
# ax2.axvline(1.0, color='k', linestyle='--', label='Macroscopic Mean Field')
# ax2.set_xlabel('Field Amplitude (|E| / E_mean)', fontsize=14)
# ax2.set_ylabel('Energy Participation (%)', fontsize=14)
# ax2.set_title('Electrical Distribution (Energy)', fontsize=16, fontweight='bold')
# ax2.legend(fontsize=12)
# ax2.grid(axis='y', alpha=0.3)

# # Master Title & Formatting
# plt.suptitle('Al-O Coplanar Waveguide Field Amplitude Distribution', fontsize=18, fontweight='bold', y=1.05)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / 'field_amplitude_distribution_comparison.png', dpi=300, bbox_inches='tight')

# print("Plot successfully saved to outputs folder.")
# plt.show()
