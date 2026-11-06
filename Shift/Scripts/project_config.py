from pathlib import Path

# The root PhD/Loss directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Global variables controlling the active pipeline target
ACTIVE_MATERIAL = "Al-Al2O3"
ACTIVE_RES = "Res_1"
ACTIVE_VOLTAGE = "5V"

# Hardware / Physics parameters tied to specific resonators
RESONATOR_SPECS = {
    "Al-Al2O3": {
        "Res_1": {"f0_MHz": 6637, "LS_um": 10.0},
        "Res_2": {"f0_MHz": 6987, "LS_um": 5.0},
        "Res_3": {"f0_MHz": 7524, "Ls_um": 3.0},
        "Res_4": {"f0_MHz": 8037, "Ls_um": 2.0}
    },
    "Al-Sapp": {
        "Res_1": {"f0_MHz": 6427, "LS_um": 10.0},
        "Res_2": {"f0_MHz": 6789, "LS_um": 5.0},
        "Res_3": {"f0_MHz": 7308, "Ls_um": 3.0},
        "Res_4": {"f0_MHz": 7835, "Ls_um": 2.0}
    },
    "Ta-Nb-Sapp": {
        "Res_1": {"f0_MHz": 6550, "LS_um": 10.0},
        "Res_2": {"f0_MHz": 6939, "Ls_um": 2.0},
        "Res_3": {"f0_MHz": 7506, "LS_um": 5.0},
        "Res_4": {"f0_MHz": 8037, "Ls_um": 3.0}
    },
    "Ta-Sapp": {
        "Res_1": {"f0_MHz": 6870, "LS_um": 10.0},
        "Res_2": {"f0_MHz": 7411, "Ls_um": 2.0},
        "Res_3": {"f0_MHz": 8022, "LS_um": 5.0},
        "Res_4": {"f0_MHz": 8575, "Ls_um": 3.0}
    },
}

def get_paths(mat=ACTIVE_MATERIAL, res=ACTIVE_RES, volt=ACTIVE_VOLTAGE):
    """Dynamically builds all Data and Output paths for a specific combination."""

    # 1. Raw Data Paths (Read-Only)
    data_dir = ROOT_DIR / "Data" / mat / res / volt

    # 2. Output Paths (Compute & Fit)
    out_base = ROOT_DIR / "Outputs" / mat / res / volt
    comp_dir = out_base / "Compute"
    fit_dir = out_base / "Fit"

    # Ensure output folders exist before before the scripts try to write to them
    for d in [comp_dir / "Plots", fit_dir / "Plots"]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract numerical voltage (e.g. "5V" -> 5.0)
    v_bias = float(volt.replace("V", ""))

    return {
        "spec": RESONATOR_SPECS[mat][res],
        "V_bias": v_bias,

        # Files
        "raw_csv": data_dir / f"resonator_data_{v_bias}.csv",
        "spatial_npz": data_dir / "hfss_spatial_data.npz",
        "processd_csv": comp_dir / "processed_compute.csv",
        "tau_chain": fit_dir / "tau_fit.csv",
        "s_chain": fit_dir / "s_fit.csv",

        # Directories
        "comp_dir": comp_dir,
        "fit_dir": fit_dir,
        "comparison_dir": ROOT_DIR / "Outputs" / mat / res / "Comparisons"
    }
