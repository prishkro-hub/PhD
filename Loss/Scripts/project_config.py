from pathlib import Path

# The root PhD/Loss directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Global variables controlling the active pipeline target
ACTIVE_MATERIAL = "Al-Sapp"
ACTIVE_RES = "6789_MHz"
ACTIVE_VOLTAGE = "5V"
ACTIVE_SPATIAL_MODE = "uniform"

# Hardware / Physics parameters tied to specific resonators
RESONATOR_SPECS = {
    "Al-Al2O3": {
        "6637_MHz": {"f0_GHz": 6.637, "LS_um": 10.0},
        "6987_MHz": {"f0_GHz": 6.987, "LS_um": 5.0},
        "7524_MHz": {"f0_GHz": 7.524, "LS_um": 3.0},
        "8037_MHz": {"f0_GHz": 8.037, "LS_um": 2.0}
    },
    "Al-Sapp": {
        "6427_MHz": {"f0_GHz": 6.427, "LS_um": 10.0},
        "6789_MHz": {"f0_GHz": 6.789, "LS_um": 5.0},
        "7308_MHz": {"f0_GHz": 7.308, "LS_um": 3.0},
        "7835_MHz": {"f0_GHz": 7.835, "LS_um": 2.0}
    },
    "Ta-Nb-Sapp": {
        "6550_MHz": {"f0_GHz": 6.550, "LS_um": 10.0},
        "6939_MHz": {"f0_GHz": 6.939, "LS_um": 2.0},
        "7506_MHz": {"f0_GHz": 7.506, "LS_um": 5.0},
        "8037_MHz": {"f0_GHz": 8.037, "LS_um": 3.0}
    },
    "Ta-Sapp": {
        "6870_MHz": {"f0_GHz": 6.870, "LS_um": 10.0},
        "7411_MHz": {"f0_GHz": 7.411, "LS_um": 2.0},
        "8022_MHz": {"f0_GHz": 8.022, "LS_um": 5.0},
        "8575_MHz": {"f0_GHz": 8.575, "LS_um": 3.0}
    },
}

def get_paths(mat=ACTIVE_MATERIAL, res=ACTIVE_RES, volt=ACTIVE_VOLTAGE, spatial=ACTIVE_SPATIAL_MODE):
    """Dynamically builds all Data and Output paths for a specific combination."""

    # 1. Base Paths
    data_dir = ROOT_DIR / "Data" / mat / res / volt
    out_base = ROOT_DIR / "Outputs" / mat / res / volt

    # 2. Output Paths (Compute & Fit)
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
        "spatial_mode": spatial,

        # Data & Grids (Read-Only inputs)
        "data_dir": data_dir,
        "raw_csv": data_dir / f"resonator_data.csv",
        "spatial_npz": data_dir.parent / "hfss_spatial_data.npz",
        "tau_grid": data_dir / f"loss_grid_tau_{spatial}.npz",
        "s_grid": data_dir / f"loss_grid_S_{spatial}.npz",


        # Computed Outputs & Chains
        "comp_dir": comp_dir,
        "fit_dir": fit_dir,
        "processed_csv": comp_dir / "processed_compute.csv",
        "final_csv": comp_dir / "final_results.csv",

        "tau_backend": fit_dir / f"tau_fit_{spatial}.h5",
        "tau_chain": fit_dir / f"tau_fit_{spatial}.csv",

        "s_backend": fit_dir / f"S_fit_{spatial}.h5",
        "s_chain": fit_dir / f"S_fit_{spatial}.csv",

        "comparison_dir": ROOT_DIR / "Outputs" / mat / res / "Comparisons"
    }
