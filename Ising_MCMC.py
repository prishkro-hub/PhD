import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not found; falling back to pure Python (will be much slower for L=100).")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit
def wolff_step(spins, L, p_add, stack):
    """Perform a single Wolff cluster flip with periodic boundary conditions."""
    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    seed_spin = spins[i0, j0]

    # Flip seed immediately to mark it as visited
    spins[i0, j0] = -seed_spin
    stack[0] = i0 * L + j0
    top = 1
    cluster_size = 1

    while top > 0:
        top -= 1
        idx = stack[top]
        i = idx // L
        j = idx % L

        # Four nearest neighbors with periodic BCs
        ni = i + 1 if i + 1 < L else 0
        if spins[ni, j] == seed_spin and np.random.random() < p_add:
            spins[ni, j] = -seed_spin
            stack[top] = ni * L + j
            top += 1
            cluster_size += 1

        ni = i - 1 if i > 0 else L - 1
        if spins[ni, j] == seed_spin and np.random.random() < p_add:
            spins[ni, j] = -seed_spin
            stack[top] = ni * L + j
            top += 1
            cluster_size += 1

        nj = j + 1 if j + 1 < L else 0
        if spins[i, nj] == seed_spin and np.random.random() < p_add:
            spins[i, nj] = -seed_spin
            stack[top] = i * L + nj
            top += 1
            cluster_size += 1

        nj = j - 1 if j > 0 else L - 1
        if spins[i, nj] == seed_spin and np.random.random() < p_add:
            spins[i, nj] = -seed_spin
            stack[top] = i * L + nj
            top += 1
            cluster_size += 1

    return cluster_size, seed_spin


@njit
def simulate_single_T(T, L, n_therm_sweeps, n_meas_sweeps, seed):
    """Run Wolff MC at temperature T and return (chi_full, chi_para, <|m|>, err_chi_para)."""
    np.random.seed(seed)
    N = L * L
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    # Start ordered below ~Tc and disordered above ~Tc for faster equilibration
    spins = np.empty((L, L), dtype=np.int8)
    M = 0
    if T < 2.27:
        for i in range(L):
            for j in range(L):
                spins[i, j] = 1
        M = N
    else:
        for i in range(L):
            for j in range(L):
                s = 1 if np.random.random() < 0.5 else -1
                spins[i, j] = s
                M += s

    stack = np.empty(N, dtype=np.int32)

    # 1. Brief warm-up + cluster-size calibration
    for _ in range(300):
        c, s0 = wolff_step(spins, L, p_add, stack)
        M -= 2 * s0 * c

    total_c = 0
    calib_flips = 200
    for _ in range(calib_flips):
        c, s0 = wolff_step(spins, L, p_add, stack)
        M -= 2 * s0 * c
        total_c += c

    mean_c = total_c / calib_flips
    flips_per_sweep = max(1, int(round(N / mean_c)))

    # 2. Thermalization
    for _ in range(n_therm_sweeps * flips_per_sweep):
        c, s0 = wolff_step(spins, L, p_add, stack)
        M -= 2 * s0 * c

    # 3. Measurements (with 20 blocks for standard error estimation)
    n_blocks = 20
    sweeps_per_block = n_meas_sweeps // n_blocks
    block_M2 = np.empty(n_blocks, dtype=np.float64)

    sum_M2 = 0.0
    sum_absM = 0.0

    for b in range(n_blocks):
        b_M2 = 0.0
        for _ in range(sweeps_per_block):
            for _ in range(flips_per_sweep):
                c, s0 = wolff_step(spins, L, p_add, stack)
                M -= 2 * s0 * c
            Mf = float(M)
            M2 = Mf * Mf
            sum_M2 += M2
            sum_absM += abs(Mf)
            b_M2 += M2
        block_M2[b] = b_M2 / sweeps_per_block

    total_samples = n_blocks * sweeps_per_block
    mean_M2 = sum_M2 / total_samples
    mean_absM = sum_absM / total_samples

    chi_para = (beta / N) * mean_M2
    chi_full = (beta / N) * (mean_M2 - mean_absM * mean_absM)
    mag_abs = mean_absM / N

    # Block standard error for chi_para
    var_block = 0.0
    for b in range(n_blocks):
        diff = block_M2[b] - mean_M2
        var_block += diff * diff
    err_chi_para = (beta / N) * np.sqrt(var_block / (n_blocks * (n_blocks - 1)))

    return chi_full, chi_para, mag_abs, err_chi_para


@njit(parallel=True)
def run_sweep(temps, L, n_therm_sweeps, n_meas_sweeps, base_seed=42):
    n_T = len(temps)
    chi_full = np.empty(n_T, dtype=np.float64)
    chi_para = np.empty(n_T, dtype=np.float64)
    mag_abs = np.empty(n_T, dtype=np.float64)
    err_chi_para = np.empty(n_T, dtype=np.float64)

    for idx in prange(n_T):
        cf, cp, m, err = simulate_single_T(
            temps[idx], L, n_therm_sweeps, n_meas_sweeps, base_seed + idx * 1009
        )
        chi_full[idx] = cf
        chi_para[idx] = cp
        mag_abs[idx] = m
        err_chi_para[idx] = err

    return chi_full, chi_para, mag_abs, err_chi_para


def main():
    L = 100
    n_therm_sweeps = 500
    n_meas_sweeps = 4000

    # Dense grid near Tc and in the T > Tc power-law scaling window
    temps = np.unique(np.concatenate([
        np.linspace(1.90, 2.20, 8),
        np.linspace(2.22, 2.33, 18),
        np.linspace(2.34, 2.58, 22),
        np.linspace(2.62, 3.00, 8),
    ]))

    t0 = time.time()
    chi_full, chi_para, mag_abs, err_chi_para = run_sweep(
        temps, L, n_therm_sweeps, n_meas_sweeps
    )
    print(f"Simulation finished in {time.time() - t0:.2f} s (L={L}, {len(temps)} temperatures)")

    # --- 1. Estimate Tc from the susceptibility peak ---
    peak_idx = np.argmax(chi_full)
    Tc_peak = temps[peak_idx]

    # Refine peak with a local quadratic fit around the maximum
    sl = slice(max(0, peak_idx - 2), min(len(temps), peak_idx + 3))
    poly = np.polyfit(temps[sl], chi_full[sl], 2)
    Tc_est = -poly[1] / (2.0 * poly[0]) if poly[0] < 0 else Tc_peak

    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
    gamma_exact = 7.0 / 4.0

    # --- 2. Extract gamma in the T > Tc scaling window ---
    # Window chosen so correlation length satisfies a << xi(T) << L:
    # For L=100, T in [2.34, 2.56] corresponds to xi ~ 18 down to 5.
    fit_mask = (temps >= 2.34) & (temps <= 2.56)
    T_fit = temps[fit_mask]
    chi_fit = chi_para[fit_mask]
    err_fit = err_chi_para[fit_mask]

    # Method A: Power-law fit with Tc fixed to Tc_est from peak
    def power_law_fixed_Tc(T, A, gamma):
        return A * (T - Tc_est) ** (-gamma)

    popt_fixed, pcov_fixed = curve_fit(
        power_law_fixed_Tc, T_fit, chi_fit, p0=[0.02, 1.75], sigma=err_fit, absolute_sigma=True
    )
    A_fix, gamma_fix = popt_fixed
    gamma_fix_err = np.sqrt(pcov_fixed[1, 1])

    # Method B: Free 3-parameter fit (A, Tc, gamma)
    def power_law_free(T, A, Tc, gamma):
        return A * (T - Tc) ** (-gamma)

    popt_free, pcov_free = curve_fit(
        power_law_free, T_fit, chi_fit, p0=[0.02, Tc_est, 1.75], sigma=err_fit, absolute_sigma=True
    )
    A_free, Tc_free, gamma_free = popt_free
    Tc_free_err = np.sqrt(pcov_free[1, 1])
    gamma_free_err = np.sqrt(pcov_free[2, 2])

    print("\n--- Results Comparison ---")
    print(f"Exact Tc:                   {Tc_exact:.5f}")
    print(f"Peak Tc (L={L}):            {Tc_est:.5f}  (rel. diff: {(Tc_est - Tc_exact)/Tc_exact:+.2%})")
    print(f"Free-fit Tc (T > Tc):       {Tc_free:.5f} +/- {Tc_free_err:.5f}")
    print(f"Exact gamma:                {gamma_exact:.4f}")
    print(f"Gamma (fixed Tc_est):       {gamma_fix:.4f} +/- {gamma_fix_err:.4f}")
    print(f"Gamma (free 3-param fit):   {gamma_free:.4f} +/- {gamma_free_err:.4f}")

    # --- 3. Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Full susceptibility curve
    ax1.plot(temps, chi_full, "o-", ms=4, label=r"$\chi' = \frac{\beta}{N}(\langle M^2 \rangle - \langle |M| \rangle^2)$")
    ax1.axvline(Tc_exact, color="k", ls="--", lw=1.2, label=rf"$T_c^{{\mathrm{{exact}}}} = {Tc_exact:.4f}$")
    ax1.axvline(Tc_est, color="r", ls=":", lw=1.5, label=rf"$T_c^{{\mathrm{{peak}}}} = {Tc_est:.4f}$")
    ax1.set_xlabel(r"Temperature $T$ ($J/k_B$)")
    ax1.set_ylabel(r"Susceptibility $\chi$")
    ax1.set_title(rf"2D Ising Susceptibility ($L={L}\times{L}$, Wolff)")
    ax1.legend()
    ax1.grid(True, ls=":", alpha=0.6)

    # Right: Log-log scaling plot for T > Tc
    t_reduced = (T_fit - Tc_est) / Tc_est
    ax2.errorbar(t_reduced, chi_fit, yerr=err_fit, fmt="o", ms=5, capsize=3, label=r"MC $\chi_{\mathrm{para}}$ ($T > T_c$)")
    t_line = np.linspace(t_reduced.min(), t_reduced.max(), 200)
    T_line = Tc_est * (1.0 + t_line)
    ax2.plot(
        t_line, power_law_fixed_Tc(T_line, A_fix, gamma_fix), "r-",
        label=rf"Fit: $\gamma = {gamma_fix:.3f} \pm {gamma_fix_err:.3f}$"
    )
    ax2.plot(
        t_line, A_fix * (T_line - Tc_est) ** (-gamma_exact) * ((T_line[len(T_line)//2] - Tc_est) ** (gamma_exact - gamma_fix)),
        "k--", alpha=0.7, label=r"Exact slope ($\gamma = 1.75$)"
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Reduced temperature $(T - T_c)/T_c$")
    ax2.set_ylabel(r"$\chi_{\mathrm{para}} = \frac{\beta}{N}\langle M^2 \rangle$")
    ax2.set_title(r"Power-law scaling above $T_c$")
    ax2.legend()
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
