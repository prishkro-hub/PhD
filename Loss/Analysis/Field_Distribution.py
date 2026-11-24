import sys
import argparse
import numpy as np
from pathlib import Path

# ===========================================
# 1. DIRECTORY & CONFIGURATION
# ===========================================
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from project_config import get_paths

HBAR = 1.054571817e-34
EPS0 = 8.854187812e-12
NUM_BINS = 100

def build_field_histogram(mat=None, res=None, num_bins=NUM_BINS):
    kwargs = {}
    if mat is not None:
        kwargs["mat"] = mat
    if res is not None:
        kwargs["res"] = res

    cfg = get_paths(**kwargs)
    spec = cfg["spec"]

    res_dir = cfg["data_dir"].parent

    f0_hz = float(spec["f0_GHz"]) * 1e9
    ls_um = float(spec["LS_um"])
    lw_um = float(spec["LW_um"])
    prefix = spec.get("hfss_prefix", f"LS{ls_um:g}LW{lw_um:g}_")
    length_m = float(spec["hfss_length_m"])
    edc_energy_2d = float(spec["hfss_edc_energy_2d"])

    # Vacuum voltage per single photon
    edc_energy_total = edc_energy_2d * length_m
    capacitance = edc_energy_total * 2.0
    v_1_ph = np.sqrt(HBAR * 2.0 * np.pi * f0_hz * 2.0 / capacitance)

    def load_fld(suffix):
        path = res_dir / f"{prefix}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing HFSS field file: {path}")
        return np.loadtxt(path, skiprows=2)[:, 3]


    # --- GET FIELD FIGURES ---
    # 1. Metal-Subtrate (MS)
    Es_MS = load_fld("MS.fld")
    Es_MS = Es_MS[Es_MS != 0]
    Es_MSenergy = 0.5 * 11.0 * EPS0 * Es_MS**2

    # 2. Metal-Air Top (MAt)
    Es_MAt = load_fld("MAt.fld") / 10.0
    Es_MAt = Es_MAt[Es_MAt != 0]
    Es_MAtenergy = 0.5 * 10 * EPS0 * Es_MAt**2

    # 3. Metal-Air Side (MAs)
    Es_MAs = load_fld("MAs.fld") / 10.0
    Es_MAs = Es_MAs[Es_MAs != 0]
    Es_MAsenergy = 0.5 * 10 * EPS0 * Es_MAs**2

    # 4. Subtrate-Air (SA)
    Es_SA_perp_raw = load_fld("SA_perp.fld")
    Es_SA_para_raw = load_fld("SA_para.fld")
    if Es_SA_perp_raw.shape != Es_SA_para_raw.shape:
        raise ValueError(
            f"SA_perp ({Es_SA_perp_raw.shape}) and SA_para ({Es_SA_para_raw.shape}) lengths do not match."
        )
    Es_SA_mag = np.sqrt(Es_SA_para_raw**2 + Es_SA_perp_raw**2)
    Es_SA_mag = Es_SA_mag[Es_SA_mag != 0]
    Es_SA_energy = 0.5 * 11 * EPS0 * Es_SA_mag**2

    # 5. Concatenate all interfaces together
    Es_total = np.concatenate((Es_MS, Es_MAt, Es_MAs, Es_SA_mag))
    Energy_total = np.concatenate((Es_MSenergy, Es_MAtenergy, Es_MAsenergy, Es_SA_energy))

    # 6. Generate Energy-Weighted Histogram
    counts, bin_edges = np.histogram(np.abs(Es_total), bins=num_bins, weights=Energy_total)
    E_k = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    W_k = counts / np.sum(counts)

    # Isolate populated bins to lighten integration load
    valid_bins = W_k > 0
    E_k_active = E_k[valid_bins]
    W_k_active = W_k[valid_bins]

    # 7. Calculate average and vacuum fields
    E_mean = np.mean(np.abs(Es_total))
    E_vac = v_1_ph * E_mean 

    # --- SAVE BACK INTO RESONATOR DATA FOLDER ---
    out_path = res_dir / "hfss_spatial_data.npz"
    np.savez(
        out_path,
        E_k=E_k_active,
        W_k=W_k_active,
        E_mean=E_mean,
        E_vac=E_vac,
    )

    print(f"[HFSS HISTOGRAM] Resonator Dir : {res_dir}")
    print(f"  Prefix        : {prefix}")
    print(f"  f0            : {f0_hz * 1e-9:.3f} GHz")
    print(f"  Capacitance   : {capacitance * 1e12:.4f} pF")
    print(f"  V_1_ph        : {v_1_ph * 1e6:.4f} uV")
    print(f"  E_mean        : {E_mean:.4e} V/m (per 1 V)")
    print(f"  E_vac         : {E_vac:.4e} V/m")
    print(f"  Active Bins   : {len(E_k_active)}/{num_bins}")
    print(f"  Saved to      : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HFSS energy-weighted field histogram")
    parser.add_argument("--mat", type=str, default=None, help="Override active material (e.g. Al-Sapp)")
    parser.add_argument("--res", type=str, default=None, help="Override active resonator (e.g. 6789_MHz)")
    args = parser.parse_args()

    build_field_histogram(mat=args.mat, res=args.res)
