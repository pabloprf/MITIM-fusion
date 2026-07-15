"""
CAPABILITY: Standalone QuaLiKiz runs from an input.gacode
-----------------------------------------------------------
This script teaches how to run the QuaLiKiz quasilinear gyrokinetic code
starting from a plasma state (input.gacode). QuaLiKiz runs on the machine
configured for "qualikiz" in config_user.json (possibly remote, possibly via
SLURM), and requires the external `qualikiz_tools` (QuaLiKiz-pythontools)
package to be installed so that `import qualikiz_tools` resolves.

Key teaching points:
    1. QuaLiKiz(rhos=[...]) + prep(input.gacode, ...) maps the plasma state
       onto QuaLiKiz's own internal "parallel" scan (dimx): unlike TGLF/NEO/
       CGYRO, ALL requested radii are packed into a SINGLE execution/folder
       rather than one folder per rho.
    2. The same three-level settings hierarchy applies: controls file
       (templates/input.qualikiz.controls, full "meta" defaults) ->
       `code_settings` preset (templates/input.qualikiz.models.yaml, e.g.
       "FAST" -> fewer eigenvalue solutions and looser tolerances, cheap for
       teaching) -> `extraOptions` (individual parameters, the final word).
    3. `multipliers` is an alternative to `extraOptions` for plasma
       parameters: instead of setting an absolute value, it multiplies the
       base value that prep() derived from the plasma state, same convention
       as NEOtools/TGLFtools. Scanned quantities use QuaLiKiz's own naming
       (e.g. "Ati0" = normalized temperature gradient of ion species 0).
    4. QLKtools is scoped to run/read/save only: no in-process/ctypes engine
       and no built-in plotting (see the module docstring in QLKtools.py).
       read() returns a fully-dimensioned xarray.Dataset (one slice per rho
       in qlk.results[label]['output']); plot it directly with matplotlib.
"""

import matplotlib.pyplot as plt
from mitim_tools.qualikiz_tools import QLKtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: each run() call below creates a subfolder in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_qualikiz_run_from_inputgacode"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare QuaLiKiz at a few radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state and builds a QuaLiKizPlan scanned in parallel across
# all requested rhos (a single run directory, not one per radius), attaching the
# experimental normalizations needed to recover fluxes in physical units
qlk = QLKtools.QuaLiKiz(rhos=[0.35, 0.5, 0.65])
qlk.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run with different settings and read each one under a label
# ---------------------------------------------------------------------------------------------------------------------

# "FAST" preset (level 2 of the hierarchy, templates/input.qualikiz.models.yaml):
# fewer eigenvalue solutions (numsols=2) and looser convergence tolerances, cheap
# for demonstration. The first argument is the name of the subfolder (inside the
# working folder) where this run lives
qlk.run("fast/", code_settings="FAST", cold_start=cold_start)
# read() parses the QuaLiKiz output folder and stores the results in the object under the label
qlk.read(label="FAST")

# Same preset, now increasing the main-ion temperature gradient by 50% via
# multipliers (level 3, alternative to extraOptions): "Ati0" is QuaLiKiz's scan-key
# name for the normalized temperature gradient (a/LTi) of ion species 0
qlk.run(
    "fast_aLTi/",
    code_settings="FAST",
    multipliers={"Ati0": 1.5},
    cold_start=cold_start,
)
qlk.read(label="FAST + 50% aLTi")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot (plain matplotlib: QLKtools has no built-in plot() / FigureNotebook)
# ---------------------------------------------------------------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for label in ["FAST", "FAST + 50% aLTi"]:
    ds = qlk.results[label]["dataset"]
    rho = qlk.results[label]["x"]
    # efe_GB/efi_GB: electron/ion heat flux in gyro-Bohm units (summed over ion species)
    axs[0].plot(rho, ds["efe_GB"].values, "-o", label=label)
    axs[1].plot(rho, ds["efi_GB"].values.sum(axis=-1), "-o", label=label)

axs[0].set_xlabel("$\\rho$"); axs[0].set_ylabel("$Q_e$ (GB)"); axs[0].set_title("Electron heat flux")
axs[1].set_xlabel("$\\rho$"); axs[1].set_ylabel("$Q_i$ (GB)"); axs[1].set_title("Ion heat flux")
for ax in axs:
    ax.legend()

plt.tight_layout()
plt.show()
