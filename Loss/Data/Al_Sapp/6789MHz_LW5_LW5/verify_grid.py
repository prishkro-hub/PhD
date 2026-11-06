import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# DIRECTORY CONFIGURATION
# =====================================
DATA_DIR = Path(__file__).resolve().parent
GRID_FILE = DATA_DIR / "loss_grid_burin_log_pdf.npz"

def run_diagnostics():
    if not GRID_FILE.exists():
        print(f"[ERROR] Could not find grid file at: {GRID_FILE}")
        sys.exit(1)

    print(f"Loading grid from: {GRID_FILE.name}... \n")
    data = np.load(GRID_FILE)

    p_grid = data['p_grid']
    g_grid = data['g_grid']

    print(f"Grid Boundaries:")
    print(f" -> p:      {len(p_grid)} points [{p_grid[0]:.1f} to {p_grid[-1]:.1f}]")
    print(f" -> log_G1:      {len(g_grid)} points [{g_grid[0]:.2f} to {g_grid[-1]:.2f}]")
    print("-" * 60)

    freq_keys = [k for k in data.files if k.startswith("nu")]
    all_clean = True

    # Setup plotting for visual verification
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, key in enumerate(freq_keys):
        grid_3d = data[key]

        has_nan = np.isnan(grid_3d).any()
        has_inf = np.isinf(grid_3d).any()

        print(f"[key] Shape: {grid_3d.shape}")
        print(f"  -> Contains NaN: {has_nan}")
        print(f"  -> Contains Inf: {has_inf}")

        if has_nan or has_inf:
            all_clean = False
            # Find exactly where it broke for debugging
            if has_nan:
                nan_idx = np.argwhere(np.isnan(grid_3d))[0]
                print(f"  -> FIRST NaN AT: p={p_grid[nan_idx[0]]:.1f}, g={g_grid[nan_idx[1]]:.2f}, field_idx={nan_idx[2]}")
        else:
            print(f"  -> Loss Range:  [{np.min(grid_3d):.3e}, {np.max(grid_3d):.3e}]")

        # Plot a 3D slice at the highest microwave power (last field index)
        # This is where RK4 explosions are most likely
        ax = axes[idx]
        slice_2d = grid_3d[:, :, -1]

        # Transpose so the points are (p,log_G1)
        c = ax.pcolormesh(p_grid, g_grid, slice_2d.T, shading='auto', cmap='viridis')
        ax.set_title(f"{key} (Max Power Field)")
        ax.set_xlabel("p (Debye)")
        ax.set_ylabel(r"$\log_{10}(\Gamma_1)$")
        fig.colorbar(c, ax=ax, label="Loss")

    print("-" * 60)
    if all_clean:
        print("[SUCCESS] Grid is mathematically sound. No NaNs or Infs detected!")
    else:
        print("[WARNING] Grid contains invalid values. Do not use for MCMC interpolation.")

    # Save the diagnostic plot
    plot_path = DATA_DIR / "grid_landscape_diagnostic.jpg"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved landscape heatmaps to: {plot_path.name}")

if __name__ == "__main__":
    run_diagnostics()
