import sys
from pathlib import Path
import time

import numpy as np
import numpy.polynomial.legendre as leg
import pandas as pd
from numba import njit, prange
from schwimmbad import MPIPool

# =====================================
# 1. DIRECTORY & CONFIGURATION
# =====================================
# Inject the root 'Scripts' directory into Python's path and import the config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import get_paths

# Fetch paths for the currently active route
cfg = get_paths()

DATA_DIR = cfg["data_dir"]
COMP_DIR = cfg["comp_dir"]

INPUT_FILE = cfg["raw_csv"]
GRID_FILE = cfg["tau_grid"]
OUTPUT_FILE = cfg["processed_csv"]

# Spatial Integration Mode:
#   "burin_log_pdf"  -> Step 3 (Analytical): P(r) ~ r^{-3} => flat weight in u = ln(r)
#   "hfss_histogram" -> Step 3 (Empirical) : All 83 active HFSS bins (|E_k| / E_mean, W_k)
#   "burin_ac_only"  -> Step 2 (Analytical): P(r) ~ r^{-3}, Spatial AC (r_ac = r), Uniform DC
#   "hfss_ac_only"        -> Step 2 (Empirical)   : Spatial AC (r_ac = r), Uniform DC (r_dc = 1, /sqrt(3))
#   "uniform"        -> Step 1 (Uniform)   : Uniform AC/DC (r_ac = 1, r_dc = 1)
SPATIAL_MODE = cfg["spatial_mode"]
SPATIAL_FILE = cfg["spatial_npz"]
N_SPATIAL = 12
N_GRID = 45

# =====================================
# 2. CONSTANTS & QUADRATURE ARRAYS
# =====================================
V_BIAS = cfg["V_bias"]
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
spatial_data = np.load(SPATIAL_FILE)
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

# MCMC Parameters
TAN0 = 9.04e-6
TAN_BG = 3.10e-6   
P_BEST = 28.5689            # D
K1_BEST = 3.0556e-4         # GHz

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
def load_loss_data(filepath):
    df = pd.read_csv(filepath)
    n_data = df["photon_number"].values
    loss_data = df["loss_tangent"].values
    nu = df["bias_frequency"].values
    power = df["drive_power"].values

    # Unprojected DC sweep rate
    dE_dt = 2.0 * V_BIAS * nu * E_mean
    E_ac = E_vac * np.sqrt(n_data)

    valid_mask = np.isfinite(n_data) & np.isfinite(loss_data)

    return (
        dE_dt[valid_mask],
        loss_data[valid_mask],
        E_ac[valid_mask],
        n_data[valid_mask],
        nu[valid_mask],
        power[valid_mask]
    )


GLOBAL_EB, GLOBAL_LOSS, GLOBAL_E_AC, GLOBAL_NPH, GLOBAL_NU, GLOBAL_DRIVE = load_loss_data(INPUT_FILE)

if __name__ == "__main__":
    start_time = time.time()

    unit_loss = compute_unit_loss_curve(
        P_BEST, K1_BEST, GLOBAL_EB, GLOBAL_E_AC,
        U_W, Y_W, Y_N, SIN_U, COS_U, SIN2_U,
        NN, SQRT_NN, RABI_COEFF, V_COEFF,
        R_AC_NODES, R_DC_NODES, W_R_NODES
    )

    tau_loss = TAN0 * unit_loss

    S_loss = GLOBAL_LOSS - tau_loss - TAN_BG


    df_out = pd.DataFrame({
        "power": GLOBAL_DRIVE,
        "bias_frequency": GLOBAL_NU,
        "n_ph": GLOBAL_NPH,
        "dE_dt": GLOBAL_EB,
        "y_exp": GLOBAL_LOSS,
        "y_tau": tau_loss,
        "y_S": S_loss
    })

    elapsed = time.time() - start_time

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"--- COMPUTATION COMPLETE IN {elapsed:.2f} SECONDS!", flush=True)
    print(f"--- CURVES SAVED TO {OUTPUT_FILE} ---", flush=True)
