import os
import sys
import re
import shutil
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Configure root and scripts folders
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
CONFIG_TOML_PATH = SCRIPTS_DIR / "config.toml"

with open(CONFIG_TOML_PATH, "rb") as f:
    TOML_DATA = tomllib.load(f)

# Global variables controlling the active pipeline target with qsub -v override option
ACTIVE_MATERIAL = os.environ.get("ACTIVE_MATERIAL", TOML_DATA["active"]["material"])
ACTIVE_RES = os.environ.get("ACTIVE_RES", TOML_DATA["active"]["resonator"])
ACTIVE_VOLTAGE = os.environ.get("ACTIVE_VOLTAGE", TOML_DATA["active"]["voltage"])
ACTIVE_SPATIAL_MODE = os.environ.get("ACTIVE_SPATIAL_MODE", TOML_DATA["active"]["spatial_mode"])

# Hardware / Physics parameters tied to specific resonators
RESONATOR_SPECS = TOML_DATA["resonators"]


def update_config_cutoff(new_cutoff_hz, mat=ACTIVE_MATERIAL, res=ACTIVE_RES, volt=ACTIVE_VOLTAGE):
    """
    Updates nu_cut_hz for the specified (mat, res, volt) in-place inside config.toml
    without stripping comments or altering file formatting.
    """
    volt_str = f"{float(str(volt).upper().replace('V', '')):g}V"
    val_str = f"{float(new_cutoff_hz):.1f}"

    toml_text = CONFIG_TOML_PATH.read_text(encoding="utf-8")

    # Locate the target [resonators.<mat>.<res>] block
    block_pattern = re.compile(
        rf"(\[resonators\.{re.escape(mat)}\.{re.escape(res)}\].*?)(?=\n\[|\Z)",
        re.DOTALL
    )
    match = block_pattern.search(toml_text)
    if not match:
        raise KeyError(f"Section [resonators.{mat}.{res}] not found in {CONFIG_TOML_PATH}")

    block = match.group(1)
    volt_pattern = re.compile(rf'("{re.escape(volt_str)}"\s*=\s*)([0-9.eE+-]+)')

    if "nu_cut_hz" in block and volt_pattern.search(block):
        # Update existing voltage key in-place
        new_block = volt_pattern.sub(rf"\g<1>{val_str}", block, count=1)
    elif "nu_cut_hz" in block:
        # Insert new voltage key into existing inline table
        new_block = re.sub(
            r"(nu_cut_hz\s*=\s*\{[^}]*?)(\s*\})",
            rf'\1, "{volt_str}" = {val_str}\2',
            block,
            count=1
        )
    else:
        # Append nu_cut_hz inline table to the resonator block
        new_block = block.rstrip() + f'\nnu_cut_hz = {{ "{volt_str}" = {val_str} }}\n'

    updated_text = toml_text[:match.start(1)] + new_block + toml_text[match.end(1):]
    CONFIG_TOML_PATH.write_text(updated_text, encoding="utf-8")

    # Sync in-memory dictionary as well
    TOML_DATA["resonators"][mat][res].setdefault("nu_cut_hz", {})[volt_str] = round(float(new_cutoff_hz), 1)
    print(f"[CONFIG UPDATED] [resonators.{mat}.{res}] nu_cut_hz['{volt_str}'] = {val_str} Hz")


def get_paths(mat=ACTIVE_MATERIAL, res=ACTIVE_RES, volt=ACTIVE_VOLTAGE, spatial=ACTIVE_SPATIAL_MODE):
    """Dynamically builds all Data, Output, and Log paths for a specific combination."""

    # 1. Base Paths
    data_dir = ROOT_DIR / "Data" / mat / res / volt
    out_base = ROOT_DIR / "Outputs" / mat / res / volt
    log_base = ROOT_DIR / "Logs" / mat / res / volt

    # 2. Output Paths (Compute & Fit)
    comp_dir = out_base / "Compute"
    fit_dir = out_base / "Fit"
    log_comp_dir = log_base / "Compute"
    log_fit_dir = log_base / "Fit"

    # Ensure output folders exist before before the scripts try to write to them
    for d in [comp_dir / "Plots", fit_dir / "Plots", log_comp_dir, log_fit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    v_bias = float(volt.replace("V", ""))
    res_spec = RESONATOR_SPECS[mat][res]

    # Empirical tau-S cutoff frequency for this specific resonator and voltage
    nu_cut_map = res_spec.get("nu_cut_hz", {})
    nu_cut_val = float(nu_cut_map.get(volt, 9702.0))

    tau_cfg = dict(TOML_DATA["tau"])
    tau_cfg["nu_cut"] = nu_cut_val

    return {
        "material": mat,
        "resonator": res,
        "spec": res_spec,
        "V_bias": v_bias,
        "spatial_mode": spatial,
        "nu_cut": nu_cut_val,

        # Stage hyperparameter dictionaries
        "solver": TOML_DATA["solver"],
        "tau": tau_cfg,
        "tan_s0": dict(TOML_DATA["tan_s0"]),
        "s_tls": dict(TOML_DATA["s_tls"]),
        "analysis": dict(TOML_DATA["analysis"]),

        # Data & Grids (Read-Only inputs)
        "data_dir": data_dir,
        "raw_csv": data_dir / "resonator_data.csv",
        "clean_csv": data_dir / "resonator_data_clean.csv",
        "spatial_npz": data_dir.parent / "hfss_spatial_data.npz",
        "tau_grid": data_dir / f"loss_grid_tau_{spatial}.npz",
        "s_grid": data_dir / f"loss_grid_S_{spatial}.npz",


        # Computed Outputs & Chains
        "comp_dir": comp_dir,
        "fit_dir": fit_dir,
        "log_comp_dir": log_comp_dir,
        "log_fit_dir": log_fit_dir,
        "processed_csv": comp_dir / "processed_compute.csv",
        "final_csv": comp_dir / "final_results.csv",

        "tau_backend": fit_dir / f"tau_fit_{spatial}.h5",
        "tau_chain": fit_dir / f"tau_fit_{spatial}.csv",

        "s_backend": fit_dir / f"S_fit_{spatial}.h5",
        "s_chain": fit_dir / f"S_fit_{spatial}.csv",

        "comparison_dir": ROOT_DIR / "Outputs" / mat / res / "Comparisons"
    }

FITTED_PARAMS_PATH = SCRIPTS_DIR / "fitted_params.toml"

if FITTED_PARAMS_PATH.exists():
    with open(FITTED_PARAMS_PATH, "rb") as f:
        FITTED_DATA = tomllib.load(f)
else:
    FITTED_DATA = {}

def save_fitted_stage_params(
        stage_prefix,
        params_dict,
        mat=ACTIVE_MATERIAL,
        res=ACTIVE_RES,
        volt=ACTIVE_VOLTAGE,
        toml_path=FITTED_PARAMS_PATH
):
    """
    Writes a dictionary of posterior medians into fitted_params.tomml under:
    [<stage_prefix>.<mat>.<res>.<volt>]
    Example stage_prefix values: "tau_results", "tan_s0_results", "s_tls_results"
    """
    v_tag = f"{float(str(volt).upper().replace('V', '')):g}V"
    section_header = f"[{stage_prefix}.{mat}.{res}.{v_tag}]"

    lines = [section_header]

    for k, v in params_dict.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lines.append(f"{k} = {float(v):.6g}" if abs(float(v)) < 1e-3 or abs(float(v)) >= 1e5 else f"{k} = {float(v):.4f}")
        else:
            lines.append(f"{k} = {v}")

    new_block = "\n".join(lines).rstrip()

    if toml_path.exists():
        text = toml_path.read_text(encoding="utf-8")
    else:
        text = "# Fitted MCMC posterior medians across pipeline stages\n\n"

    # Match from section_header up to the next [section] header (or end of file)
    pattern = re.compile(
        rf"({re.escape(section_header)}.*?)(?=\n\[|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        matched_chunk = match.group(1)
        remainder = text[match.end(1):]

        # Pull any trailing comment/blank lines off the old section so they stay with the next section
        lines_in_chunk = matched_chunk.splitlines()
        trailing = []
        while lines_in_chunk and (
            lines_in_chunk[-1].strip().startswith("#") or lines_in_chunk[-1].strip() == ""
        ):
            trailing.append(lines_in_chunk.pop())
        trailing.reverse()

        # Drop leading blank lines before the next section's comment banner
        while trailing and trailing[0].strip() == "":
            trailing.pop(0)

        if trailing:
            if matched_chunk.endswith("\n"):
                trailing.append("")
            updated_text = text[:match.start(1)] + new_block + "\n\n" + "\n".join(trailing) + remainder
        elif remainder:
            updated_text = text[:match.start(1)] + new_block + "\n\n" + remainder.lstrip("\n")
        else:
            updated_text = text[:match.start(1)] + new_block + "\n"
    else:
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        updated_text = text + sep + new_block + "\n"

    toml_path.write_text(updated_text, encoding="utf-8")
    print(f"[TOML SYNC] Updated {section_header} in {toml_path.name}")

def get_fitted_stage_params(
        stage_prefix,
        mat=ACTIVE_MATERIAL,
        res=ACTIVE_RES,
        volt=ACTIVE_VOLTAGE,
):
    """Retrieves stored stage parameters for (stage_prefix, mat, res, volt)."""
    volt_key = f"{float(str(volt).upper().replace('V', "")):g}V"
    entry = (
        FITTED_DATA.get(stage_prefix, {})
        .get(mat, {})
        .get(res, {})
        .get(volt_key)
    )
    if entry is None:
        raise KeyError(f"Missing [{stage_prefix}.{mat}.{res}.{volt_key}] in {FITTED_PARAMS_PATH}")
    return dict(entry)

def snapshot_config(target_dir, tag="run"):
    """Copies config.toml (and fitted_params.toml if exists) into target_dir."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_TOML_PATH, target / f"config_snapshot_{tag}.toml")
    if FITTED_PARAMS_PATH.exists():
        shutil.copy2(FITTED_PARAMS_PATH, target / f"fitted_params_snapshot_{tag}.toml")



if __name__ == "__main__":
    # Called by bas scripts: eval "$(python Loss/Scripts/project_config.py Compute)"
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "compute"
    cfg = get_paths()

    if mode in ("fit", "fits"):
        logs_path = cfg["log_fit_dir"]
        outs_path = cfg["fit_dir"]
    elif mode in ("grid", "grids"):
        logs_path = cfg["log_comp_dir"]
        outs_path = cfg["data_dir"]
    else:
        logs_path = cfg["log_comp_dir"]
        outs_path = cfg["comp_dir"]

    print(f'export LOGS="{logs_path}"')
    print(f'export OUTPUTS="{outs_path}"')
