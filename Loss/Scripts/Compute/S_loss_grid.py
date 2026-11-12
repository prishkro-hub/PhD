import sys
from pathlib import Path
import time

import numpy as np
import numpy.polynomial.legendre as leg
import pandas as pd
from scipy.special import exp1
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
FIT_DIR = cfg["fit_dir"]
INPUT_FILE = cfg["processed_csv"]
OUTPUT_FILE = cfg["s_grid"]
V_BIAS = float(cfg["V_bias"])

# Specific S-TLS Parameters
NU_CUT = 9702.0     # Hz
MIN_PWR = -35.0     # dBm
MAX_PWR = -11.0     # dBm

BEST_TAN_S0_CSV = FIT_DIR / "Best_Parameters_tan_S0.csv" 
# Fallback MAP parameters from tan_s0 fit if the CSV is missing
DEFAULT_TAN_S0_PARAMS = (4.9283e-8, 1.0556, 5.5915, 1.1094e7)

# Spatial Integration Mode:
#   "burin_log_pdf"  -> Step 3 (Analytical): P(r) ~ r^{-3} => flat weight in u = ln(r)
#   "hfss_histogram" -> Step 3 (Empirical) : All 83 active HFSS bins (|E_k| / E_mean, W_k)
#   "burin_ac_only"  -> Step 2 (Analytical): P(r) ~ r^{-3}, Spatial AC (r_ac = r), Uniform DC
#   "hfss_ac_only"        -> Step 2 (Empirical)   : Spatial AC (r_ac = r), Uniform DC (r_dc = 1, /sqrt(3))
#   "uniform"        -> Step 1 (Uniform)   : Uniform AC/DC (r_ac = 1, r_dc = 1)
SPATIAL_FIELD = cfg["spatial_npz"]
SPATIAL_MODE = cfg["spatial_mode"]
N_SPATIAL = 12
N_GRID = 45

# =====================================
# 2. CONSTANTS & QUADRATURE ARRAYS
# =====================================
DEBYE = 3.33564e-30
HBAR = 1.054571817e-34
# V_BIAS = 5.0

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
spatial_data = np.load(SPATIAL_FIELD)
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
# 4. LOAD S-TLS DATA & TAN_S0 ENVELOPE
# =====================================
def get_tan_s0_params(chain_filepath):
    if chain_filepath.exists():
        df_map = pd.read_csv(chain_filepath)
        df_map.set_index("Parameter", inplace=True)

        MAP_log_A = df_map.loc["logA", "MAP (Best Fit)"]
        MAP_B = df_map.loc["B", "MAP (Best Fit)"]
        MAP_C = df_map.loc["C", "MAP (Best Fit)"]
        MAP_log_D = df_map.loc["logD", "MAP (Best Fit)"]

        MAP_A = 10.0 ** MAP_log_A
        MAP_D = 10.0 ** MAP_log_D
        
        return MAP_A, MAP_B, MAP_C, MAP_D
    return DEFAULT_TAN_S0_PARAMS

def calc_tan_s0(Eb_array, A, B, C, D):
    u0 = D / Eb_array
    u1 = u0 * np.exp(2.0 * C)
    integral_val = 0.5 * (exp1(u0) - exp1(u1))

    return A * np.exp(B * integral_val)

def load_loss_data(filepath, chain_filepath):
    df = pd.read_csv(filepath)

    pwr_data = df["power"].values
    n_data = df["n_ph"].values
    loss_data = df["y_S"].values
    dE_dt = df["dE_dt"].values
    nu = df["bias_frequency"]
    E_ac = E_vac * np.sqrt(n_data)

    # Precompute across all high-power S-TLS points (power >= -33 dBm, dE_dt > 1.698e9)
    valid_mask = (
        np.isfinite(n_data)
        & np.isfinite(loss_data)
        & (nu > NU_CUT)
        & (pwr_data >= MIN_PWR)
        & (pwr_data <= MAX_PWR)
    )

    eb_active = dE_dt[valid_mask]
    A_s0, B_s0, C_s0, D_s0 = get_tan_s0_params(chain_filepath)
    tan_s0_active = calc_tan_s0(eb_active, A_s0, B_s0, C_s0, D_s0)

    return (
        pwr_data[valid_mask],
        eb_active,
        loss_data[valid_mask],
        E_ac[valid_mask],
        n_data[valid_mask],
        nu[valid_mask],
        tan_s0_active,
        np.array([A_s0, B_s0, C_s0, D_s0], dtype=np.float64)
    )


(
    GLOBAL_PWR,
    GLOBAL_EB,
    GLOBAL_LOSS,
    GLOBAL_E_AC,
    GLOBAL_NPH,
    GLOBAL_NU,
    GLOBAL_TAN_S0,
    TAN_S0_PARAMS
) = load_loss_data(INPUT_FILE, BEST_TAN_S0_CSV)

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

        print(f"--- STARTING S-TLS LOSS GRID BUILDER ({SPATIAL_MODE}) ---", flush=True)
        print(
            f"ACTIVE S-TLS points: {len(GLOBAL_LOSS)} | "
            f"Powers: {np.unique(GLOBAL_PWR)} dBm | "
            f"Frequencies: {np.unique(GLOBAL_NU)} Hz",
            flush=True
        )

        p_grid = np.geomspace(1.0, 400.0, N_GRID)
        g_grid = np.linspace(1.5, 5.0, N_GRID)

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
                power=GLOBAL_PWR,
                dE_dt=GLOBAL_EB,
                loss_data=GLOBAL_LOSS,
                tan0_data=GLOBAL_TAN_S0,
                tan_s0_params=TAN_S0_PARAMS,
                E_ac=GLOBAL_E_AC,
                n_ph=GLOBAL_NPH,
                nu=GLOBAL_NU,
                spatial_mode=SPATIAL_MODE
            )
        elapsed_total = (time.time() - start_time) / 60
        print(f"\nLoss Tangent Spline Table saved to: {OUTPUT_FILE}", flush=True)
        print(f"Total Computation Time: {elapsed_total:.2f} minutes.", flush=True)
