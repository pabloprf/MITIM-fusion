"""
test_eped_fuel_impurity.py — effect of fuel mass and impurity charge on EPED
----------------------------------------------------------------------------
EPED does not receive the fuel fraction: it reconstructs the fuel-vs-impurity
split from quasineutrality + Zeff, so the impurity charge `zi` sets the main-ion
(fuel) dilution at fixed Zeff, while the main-ion mass `m` sets the isotope mass
(2.0 = pure D, 2.5 = 50/50 D-T). Both used to be hardcoded (m=2.5, neon zi=10);
`EPEDtools.EPED.run` now exposes them, and they can be supplied/scanned through
`input_params`.

This runs two scans at a fixed engineering point — over `zi` and over `m` — each
submitted as ONE SLURM job array (`scan_param` + `job_array_limit`), so the cases
run concurrently rather than one at a time, exactly like the eped_01 capability
test scans `neped`. It then tabulates the EPED pedestal (ptop, wtop) for each.

REQUIREMENT: runs full EPED, so it needs the "eped" machine configured in
config_user.json (like eped_01).

Usage:
    python tests/dev_tests/test_eped_fuel_impurity.py
"""

from mitim_tools.eped_tools import EPEDtools
from mitim_tools.misc_tools import IOtools
from mitim_tools import __mitimroot__

cold_start = True
folder = __mitimroot__ / "tests" / "scratch" / "dev_eped_fuel_impurity"
if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# Fixed engineering point (SPARC-like), shared across both scans
base_params = {
    "ip": 8.7, "bt": 12.16, "r": 1.85, "a": 0.57,
    "kappa": 1.9, "delta": 0.5,
    "neped": 30.0, "betan": 1.0, "zeffped": 1.5,
    "nesep": 12.0, "tesep": 100.0,
}

zi_values = [6.0, 8.0, 10.0, 12.0]   # impurity charge at fixed mass m=2.5
m_values = [2.0, 2.25, 2.5]          # main-ion mass at fixed impurity zi=10

eped = EPEDtools.EPED(folder=folder)


def run_scan(subfolder, variable, values, fixed):
    """Submit one EPED job array scanning `variable` over `values` (concurrent),
    with the composition held at `fixed`. Returns [(value, ptop, wtop), ...]."""
    eped.run(
        subfolder=subfolder,
        input_params={**base_params, **fixed},   # composition rides through input_params
        scan_param={"variable": variable, "values": values},
        job_array_limit=len(values),              # all cases at once
        nproc_per_run=64,
        minutes_slurm=240,
        cold_start=cold_start,
        removeScratchFolders=True,
    )
    eped.read(subfolder=subfolder)
    out = []
    for i, val in enumerate(values):
        data = eped.results[subfolder][f"run{i + 1}"]
        out.append((val, float(data["ptop"].values[0]), float(data["wptop"].values[0])))
    return out


# Impurity-charge scan (fixed m=2.5, mi=20) and fuel-mass scan (fixed zi=10, mi=20)
res_zi = run_scan("scan_zi", "zi", zi_values, fixed={"m": 2.5, "mi": 20})
res_m = run_scan("scan_m", "m", m_values, fixed={"zi": 10, "mi": 20})

# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------
def table(title, var, rows):
    print("\n" + "=" * 44)
    print(title)
    print("=" * 44)
    print(f"{var:>8}{'ptop (kPa)':>16}{'wtop (psi)':>16}")
    print("-" * 44)
    for v, ptop, wtop in rows:
        print(f"{v:>8.2f}{ptop:>16.1f}{wtop:>16.4f}")
    print("=" * 44)

table("EPED vs impurity charge zi  (m=2.5)", "zi", res_zi)
table("EPED vs main-ion mass m  (zi=10)", "m", res_m)

# ---------------------------------------------------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for col, (rows, xlabel) in enumerate([(res_zi, "impurity charge $z_i$"), (res_m, "main-ion mass $m$ (amu)")]):
    x = [r[0] for r in rows]
    axs[0, col].plot(x, [r[1] for r in rows], "-o", color="r")
    axs[1, col].plot(x, [r[2] for r in rows], "-o", color="b")
    axs[0, col].set_ylabel("$p_{top}$ (kPa)"); axs[0, col].set_ylim(bottom=0)
    axs[1, col].set_ylabel("$w_{top}$ ($\\psi_{pol}$)"); axs[1, col].set_ylim(bottom=0)
    axs[1, col].set_xlabel(xlabel)
    for row in (0, 1):
        axs[row, col].grid(alpha=0.3)
fig.suptitle("EPED pedestal sensitivity to fuel mass and impurity charge")
plt.tight_layout()
plt.show()
