import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

length=187.5e-6*4*18
hbar = 1.054571817e-34
Edc_EnergyInJ_2dSim=5.8015E-011
Edc_EnergyInJ_2dSim_Total=Edc_EnergyInJ_2dSim*length

Capacitance=Edc_EnergyInJ_2dSim_Total*2
V_1_ph = np.sqrt(hbar*2*np.pi*6.789e9*2/Capacitance)   

filename_add = 'LS5LW5_'
num_bins = 100

# --- GET FIELD FIGURES ---
# 1. Metal-Subtrate (MS)
ms_path = DATA_DIR / f"{filename_add}MS.fld"
Es_MS = np.loadtxt(ms_path, skiprows=2)[:, 3]
Es_MS = Es_MS[Es_MS != 0]
Es_MSenergy = 0.5 * 11 * 8.85e-12 * Es_MS**3

# 2. Metal-Air Top (MAt)
mat_path = DATA_DIR / f"{filename_add}MAt.fld"
Es_MAt = np.loadtxt(mat_path, skiprows=2)[:, 3] / 10
Es_MAt = Es_MAt[Es_MAt != 0]
Es_MAtenergy = 0.5 * 10 * 8.85e-12 * Es_MAt**3

# 3. Metal-Air Side (MAs)
mas_path = DATA_DIR / f"{filename_add}MAs.fld"
Es_MAs = np.loadtxt(mas_path, skiprows=2)[:, 3] / 10
Es_MAs = Es_MAs[Es_MAs != 0]
Es_MAsenergy = 0.5 * 10 * 8.85e-12 * Es_MAs**3

# 4. Subtrate-Air (SA)
sa_perp_path = DATA_DIR / f"{filename_add}SA_perp.fld"
Es_SA_perp = np.loadtxt(sa_perp_path, skiprows=2)[:, 3]
Es_SA_perp = Es_SA_perp[Es_SA_perp != 0]

sa_para_path = DATA_DIR / f"{filename_add}SA_para.fld"
Es_SA_para = np.loadtxt(sa_para_path, skiprows=2)[:, 3]
Es_SA_para = Es_SA_para[Es_SA_para != 0]

# Ensure arrays alifn
Es_SA_mag = np.sqrt(Es_SA_para**2 + Es_SA_perp**2)
Es_SA_energy = 0.5 * 11 * 8.85e-12 * Es_SA_mag**3

# 5. Concatenate all interfaces together
Es_total = np.concatenate((Es_MS, Es_MAt, Es_MAs, Es_SA_mag))
Energy_total = np.concatenate((Es_MSenergy, Es_MAtenergy, Es_MAsenergy, Es_SA_energy))

# 6. Generate Energy-Weighted Histogram
counts, bin_edges = np.histogram(Es_total, bins=num_bins, weights=Energy_total)

# Calculate discrete field strength (E_k) and normalized volume/energy fractions (W_k)
E_k = (bin_edges[:-1] + bin_edges[1:]) / 2
W_k = counts / np.sum(counts)

# Isolate populated bins to lighten integration load
valid_bins = W_k > 0
E_k_active = E_k[valid_bins]
W_k_active = W_k[valid_bins]

# 7. Calculate average and vacuum fields
E_mean = np.mean(np.abs(Es_total))
E_vac = V_1_ph * E_mean 

# --- SAVE FOR EASY EXTRACTION ---
np.savez(DATA_DIR / 'hfss_spatial_data.npz',
         E_k=E_k_active,
         W_k=W_k_active,
         E_mean=E_mean,
         E_vac=E_vac)
