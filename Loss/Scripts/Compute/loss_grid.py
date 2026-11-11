import sys
from pathlib import Path
import time

import numpy as np
import numpy.polynomial.legendre as leg
import pandas as pd
from scipy.stats import linregress
from numba import njit, prange
from schwimmbad import MPIPool

# =====================================
# 1. DIRECTORY & MODE CONFIGURATION
# =====================================
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import get_paths
cfg = get_paths()

DATA_DIR = cfg["data_dir"]
SPATIAL_MODE = cfg["spatial_mode"]
INPUT_FILE = cfg["raw_csv"]
OUTPUT_FILE = cfg["tau_grid"]
V_BIAS = float(cfg["V_bias"])

# Spatial Integration Mode:
#   "burin_log_pdf"  -> Step 3 (Analytical): P(r) ~ r^{-3} => flat weight in u = ln(r)
#   "hfss_histogram" -> Step 3 (Empirical) : All 83 active HFSS bins (|E_k| / E_mean, W_k)
#   "burin_ac_only"  -> Step 2 (Analytical): P(r) ~ r^{-3}, Spatial AC (r_ac = r), Uniform DC
#   "hfss_ac_only"        -> Step 2 (Empirical)   : Spatial AC (r_ac = r), Uniform DC (r_dc = 1, /sqrt(3))
#   "uniform"        -> Step 1 (Uniform)   : Uniform AC/DC (r_ac = 1, r_dc = 1)
N_SPATIAL = 12
N_GRID = 45

# =====================================
# 2. CONSTANTS & QUADRATURE ARRAYS
# =====================================
DEBYE = 3.33564e-30
HBAR = 1.054571817e-34

RABI_COEFF = (DEBYE / HBAR) * 1e-9          # Converts (p [D] * E_ac [v/m]) to GHz
V_COEFF = (2.0 * DEBYE / HBAR) * 1e-18      # Converts (2 * p [D] * dE_dt [V/(m*s)]) to GHz^2

NN = 100.0  # Stable cutoff
SQRT_NN = np.sqrt(NN)
N_QUAD = 40
x_leg, w_leg = leg.leggauss(N_QUAD)

U_N = (np.pi / 4.0) * x_leg + (np.pi / 4.0)
U_W = (np.pi / 4.0) * w_leg
Y_N = 0.5 * x_leg + 0.5
Y_W = 0.5 * w_leg

SIN_U = np.sin(U_N)
COS_U = np.cos(U_N)
SIN2_U = SIN_U ** 2

# Load HFSS spatial field data
spatial_data = np.load(cfg["spatial_npz"])
E_k = spatial_data["E_k"]
W_k = spatial_data["W_k"]
E_mean = float(spatial_data["E_mean"])
E_vac = float(spatial_data["E_vac"])

r_abs = np.abs(E_k) / E_mean
R_MAX = float(np.max(r_abs))

if SPATIAL_MODE in ("burin_log_pdf", "burin_ac_only"):
    R_MIN = R_MAX / (2.0 * R_MAX - 1.0)
    x_u, w_u = leg.leggauss(N_SPATIAL)
    u_min, u_max = np.log(R_MIN), np.log(R_MAX)
    u_nodes = 0.5 * (u_max - u_min) * x_u + 0.5 * (u_max + u_min)
    R_AC_NODES = np.exp(u_nodes)
    if SPATIAL_MODE == "burin_log_pdf":
        R_DC_NODES = R_AC_NODES.copy()
    else:
        R_DC_NODES = np.ones_like(R_AC_NODES)
    W_R_NODES = 0.5 * w_u
elif SPATIAL_MODE in ("hfss_histogram", "hfss_ac_only"):
    mask = r_abs > 1e-6
    R_AC_NODES = r_abs[mask]
    if SPATIAL_MODE == "hfss_histogram":
        R_DC_NODES = R_AC_NODES.copy()
    else:
        R_DC_NODES = np.ones_like(R_AC_NODES)
    W_R_NODES = W_k[mask] / np.sum(W_k[mask])
    R_MIN = float(np.min(R_AC_NODES))
elif SPATIAL_MODE == "uniform":
    R_AC_NODES = np.array([1.0], dtype=np.float64)
    R_DC_NODES = np.array([1.0], dtype=np.float64)
    W_R_NODES = np.array([1.0], dtype=np.float64)
    R_MIN = 1.0
else:
    raise ValueError(f"Unknown SPATIAL_MODE: {SPATIAL_MODE}")

# =====================================
# 3. CRACK-NICOLSON & HYBRID LOSS ENGINE
# =====================================
@njit(fastmath=True, inline="always")
def cn_step(y1, y2, y3, y4, t, dt, Om, k1, k2, v):
    vt_old = v * t
    dy1_old = -k2 * y1 + vt_old * y2
    dy2_old = -vt_old * y1 - k2 * y2 + Om * y3
    dy3_old = -Om * y2 - k1 * y3

    dt2 = 0.5 * dt
    V1 = y1 + dy1_old * dt2
    V2 = y2 + dy2_old * dt2
    V3 = y3 + dy3_old * dt2 + k1 * dt

    t_new = t + dt
    vt_new = v * t_new

    a = 1.0 + k2 * dt2
    b = vt_new * dt2
    c = Om * dt2
    d = 1.0 + k1 * dt2

    det = d * (a**2 + b**2) + a * (c**2)
    inv_det = 1.0 / det

    y1_new = ((a * d + c**2) * V1 + b * d * V2 + b * c * V3) * inv_det
    y2_new = (-b * d * V1 + a * d * V2 + a * c * V3) * inv_det
    y3_new = (b * c * V1 - a * c * V2 + (a**2 + b**2) * V3) * inv_det
    y4_new = y4 - k1 * dt2 * (y3 + y3_new) + k1 * dt

    return y1_new, y2_new, y3_new, y4_new, t_new

@njit(fastmath=True)
def LTI_cn(Om, k1, v, NN_val):
    k2 = k1 / 2.0
    sq_v = np.sqrt(v)
    inv_sq_v = 1.0 / sq_v

    t_min = -NN_val * inv_sq_v
    t_max = NN_val * inv_sq_v + np.log(NN_val) / k1
    kk_n = k1 * inv_sq_v
    Om_n = Om * inv_sq_v

    tt_n_min_sq = NN_val**2
    kk_n_sq = kk_n**2
    Om_n_sq = Om_n**2

    D = kk_n_sq + 2.0 * Om_n_sq + 4.0 * tt_n_min_sq
    sqrt_val = np.sqrt(kk_n_sq + 2.0 * Om_n_sq)

    y04 = -(kk_n * Om_n_sq * np.arctan(2.0 * NN_val / sqrt_val)) / sqrt_val
    y1 = (-4.0 * Om_n * NN_val) / D
    y2 = (2.0 * kk_n * Om_n) / D
    y3 = (kk_n_sq + 4.0 * tt_n_min_sq) / D
    y4 = y04

    t = t_min
    dt_base = 0.02 * inv_sq_v
    dt_relax = 0.1 / k1

    while t < t_max:
        dist = abs(t * sq_v)
        scale_base = min(1.0 + 0.8 * dist, 25.0)
        scale_relax = min(1.0 + 0.133 * dist, 5.0)
        dt = min(dt_base * scale_base, dt_relax * scale_relax)
        if t + dt > t_max:
            dt = t_max - t
        y1, y2, y3, y4, t = cn_step(y1, y2, y3, y4, t, dt, Om, k1, k2, v)

    tt_n_max = sq_v * t_max
    y04_max = (kk_n * Om_n_sq * np.arctan(2.0 * tt_n_max / sqrt_val)) / sqrt_val
    missing_tails = (np.pi * kk_n * Om_n_sq) / sqrt_val - y04_max

    return y4 + missing_tails

@njit(fastmath=True, parallel=True)
def compute_unit_loss_curve(
    p_val, k1_ghz, Eb_arr, E_ac,
    u_w, y_w, y_n, sin_u, cos_u, sin2_u,
    nn_val, sqrt_nn, rabi_c, v_c,
    r_ac_nodes, r_dc_nodes, w_r_nodes
):
    """
    Computes the normalized spatial AC/DC loss curve L_i(p, k1) in [0, 1]
    for all experimental data points i.
    """
    n_points = len(Eb_arr)
    total_loss = np.zeros(n_points)
    n_quad_u = len(u_w)
    n_quad_y = len(y_w)
    n_radii = len(r_ac_nodes)
    inv_nn_val = 1.0 / nn_val

    for i in prange(n_points):
        Om_base = rabi_c * p_val * E_ac[i]
        if Om_base <= 0.0:
            continue
        v_base = v_c * p_val * Eb_arr[i]

        point_loss = 0.0
        active_w_sum = 0.0

        # Spatial quadrature loop over r = |E_local| / E_mean
        for r_idx in range(n_radii):
            
            Om_local = Om_base * r_ac_nodes[r_idx]
            v_local = v_base * r_dc_nodes[r_idx]
            w_r = w_r_nodes[r_idx]

            if Om_local <= 0.0:
                continue

            local_sum = 0.0

            # Internal tunneling angle u: sin(u) = Delta_0 / E_0
            for ui in range(n_quad_u):
                s_u = sin_u[ui]
                c_u = cos_u[ui]
                s2_u = sin2_u[ui]

                Om_ui = Om_local * s_u
                v_ui = v_local * c_u
                k1_l = k1_ghz * s2_u

                v_ui_cond1 = v_ui < 0.01 * Om_ui * k1_l
                if not v_ui_cond1:
                    adiab_coeff = np.pi * (Om_ui**2) / (2.0 * v_ui)
                    k1_l_cond3 = k1_l < 0.3 * v_ui / (Om_ui * sqrt_nn)
                else:
                    adiab_coeff = 0.0
                    k1_l_cond3 = False

                inv_k1_l_sq_2 = 2.0 / (k1_l**2)
                cond2_thresh = k1_l / (3.0 * sqrt_nn)

                int_y = 0.0

                # Dipole orientation projection y = |cos(theta)|
                for yi in range(n_quad_y):
                    y_val = y_n[yi]
                    Om_l = Om_ui * y_val
                    v_l = v_ui * y_val
                    Om_l_sq = Om_l**2

                    if v_ui_cond1:
                        denom = np.sqrt(1.0 + 2.0 * Om_l_sq * inv_k1_l_sq_2)
                        val = np.pi * Om_l_sq / denom
                    elif (Om_l_sq < v_l * inv_nn_val) or (Om_l < cond2_thresh):
                        val = np.pi * Om_l_sq
                    elif k1_l_cond3:
                        val = 2.0 * v_l * (1.0 - np.exp(-adiab_coeff * y_val))
                    else:
                        val = LTI_cn(Om_l, k1_l, v_l, nn_val) * v_l

                    int_y += val * y_w[yi]

                local_sum += (int_y / s_u) * u_w[ui]

            # Normalized loss factor at spatial node r_idx, weighted by energy fraction w_r
            norm_local_loss = 3.0 * local_sum / max(np.pi * (Om_local**2), 1e-100)
            point_loss += w_r * norm_local_loss
            active_w_sum += w_r

        if active_w_sum > 0.0:
            total_loss[i] = point_loss / active_w_sum

    return total_loss


# =====================================
# 4. LOAD EXPERIMENTAL DATA & MPI WORKER
# =====================================
def load_loss_data(filepath, low_power_thresh=-55.0):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # 1. Find the tau-TLS regime
    low_pwr = df[df['drive_power'] <= low_power_thresh]
    y_low = low_pwr.groupby('bias_frequency')['loss_tangent'].mean().values * 1e6
    nu_arr = np.sort(low_pwr['bias_frequency'].unique())

    tau_baseline = np.mean(y_low[:3])
    log_nu = np.log10(nu_arr)
    log_y = np.log10(y_low)
    slope, intercept, _, _, _ = linregress(log_nu[-4:], log_y[-4:])

    log_nu_cut = (np.log10(tau_baseline) - intercept) / slope
    nu_cut = 10 ** log_nu_cut

    print(f"Dynamically calculated tau-TLS cutoff: {nu_cut:.0f} Hz", flush=True)

    # 2. Filter dataset
    valid_mask = (
        np.isfinite(df['photon_number'])
        & np.isfinite(df['loss_tangent'])
        & (df['bias_frequency'] <= nu_cut)
    )

    df_active = df[valid_mask].copy()

    n_data = df_active['photon_number'].values
    loss_data = df_active['loss_tangent'].values
    nu_data = df_active['bias_frequency'].values
    drive_data = df_active['drive_power'].values

    print(f"Restricted fit to {len(loss_data)} slow-sweep data points.", flush=True)

    # Unprojected DC sweep rate and AC field
    Eb_data = 2.0 * V_BIAS * nu_data * E_mean
    Eac_data = E_vac * np.sqrt(n_data)

    return drive_data, n_data, loss_data, nu_data, Eac_data, Eb_data

GLOBAL_DRIVE, GLOBAL_NPH, GLOBAL_LOSS, GLOBAL_NU, GLOBAL_E_AC, GLOBAL_EB = load_loss_data(INPUT_FILE)

def compute_grid_point(task):
    i, p_val, j, g_val = task
    # g_val is log10(Gamma_1 in MHz) -> convert to GHz for solver
    k1_ghz = (10.0 ** g_val) * 1e-3

    unit_loss = compute_unit_loss_curve(
        p_val, k1_ghz, GLOBAL_EB, GLOBAL_E_AC,
        U_W, Y_W, Y_N, SIN_U, COS_U, SIN2_U,
        NN, SQRT_NN, RABI_COEFF, V_COEFF,
        R_AC_NODES, R_DC_NODES, W_R_NODES
    )

    return (i, j, unit_loss)

# =====================================
# 5. MPI EXECUTION & SAVE
# =====================================
if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print(f"--- STARTING MPI LOSS TANGENT GRID BUILDER ({SPATIAL_MODE}) ---", flush=True)
        print(f"ACTIVE data points: {len(GLOBAL_LOSS)} | Spatial nodes: {len(R_AC_NODES)}", flush=True)

        p_grid = np.geomspace(1.0, 150.0, N_GRID)
        g_grid = np.linspace(-2.5, 3.0, N_GRID)

        loss_grid_matrix = np.zeros((len(p_grid), len(g_grid), len(GLOBAL_LOSS)), dtype=np.float64)

        tasks = [
            (i, p_grid[i], j, g_grid[j])
            for i in range(len(p_grid))
            for j in range(len(g_grid))
        ]
        total_evals = len(tasks)
        batch_size = 64
        start_time = time.time()
        completed = 0

        for b_start in range(0, total_evals, batch_size):
            batch_tasks = tasks[b_start : b_start + batch_size]
            batch_results = list(pool.map(compute_grid_point, batch_tasks))

            for i, j, unit_curve in batch_results:
                loss_grid_matrix[i, j, :] = unit_curve

            completed += len(batch_results)
            elapsed_min = (time.time() - start_time) / 60
            est_total_min = (elapsed_min / completed) * total_evals
            rem_min = est_total_min - elapsed_min

            print(
                f" Completed {completed}/{total_evals} ({100.0*completed/total_evals:.2f}%) | "
                f"Elapsed: {elapsed_min:.2f} min | ETA: {rem_min:.2f} min",
                flush=True
            )

            np.savez_compressed(
                OUTPUT_FILE,
                p_grid=p_grid,
                g_grid=g_grid,
                loss_grid=loss_grid_matrix,
                dE_dt=GLOBAL_EB,
                loss_data=GLOBAL_LOSS,
                E_ac=GLOBAL_E_AC,
                n_ph=GLOBAL_NPH,
                nu=GLOBAL_NU,
                drive=GLOBAL_DRIVE,
                spatial_mode=SPATIAL_MODE
            )
        elapsed_total = (time.time() - start_time) / 60
        print(f"\nLoss Tangent Spline Table saved to: {OUTPUT_FILE}", flush=True)
        print(f"Total Computation Time: {elapsed_total:.2f} minutes.", flush=True)
