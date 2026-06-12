"""
CAPABILITY: CGYRO linear ky-spectrum scan
-----------------------------------------
This script teaches how to build a linear spectrum with CGYRO: since CGYRO
computes one linear mode per execution (unlike GX, which does the whole
spectrum in one run), the spectrum is obtained by scanning KY — one cheap
linear run per wavenumber — and collecting the converged eigenvalues.

Key teaching points:
    1. run_scan() over the variable KY with relativeChanges=False: varUpDown
       contains the ABSOLUTE ky values to run, not multipliers of the base
       value (contrast with the TGLF scans, see tglf_05_scan.py).
    2. read_linear_scan() walks the scan subfolders and collects the
       converged eigenvalue (growth rate and real frequency) of every KY run
       into a single spectrum object, stored under `store_as_label`.
    3. plot_quick_linear() shows the assembled gamma(ky) and omega(ky)
       spectrum; the individual runs can also be plotted as usual under their
       per-ky labels.
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one scan subfolder per ky lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_kyscan"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare CGYRO at one radius from the plasma state (one radius to keep it cheap)
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.cgyro per requested rho into the folder
# and attaches the experimental normalizations
cgyro = CGYROtools.CGYRO(rhos=[0.5])
cgyro.prep(input_gacode, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Scan KY: one linear run per wavenumber (a small, cheap spectrum here)
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run_scan(
    # Name of the subfolders (inside the working folder) where the scan points live
    "kyscan",
    # Preset from templates/input.cgyro.models.yaml: NONLINEAR_FLAG=0, single toroidal mode
    code_settings="Linear",
    extraOptions={
        "MAX_TIME": 10.0,  # very short, just for demonstration (real linear runs need eigenvalue convergence)
    },
    # ABSOLUTE ky values (relativeChanges=False), a small set to keep this example cheap
    variable="KY",
    varUpDown=[0.3, 0.5, 0.7],
    relativeChanges=False,
    # Resources of each CGYRO instance (one per ky): cores or GPUs per call, and SLURM time limit
    allocation={"resources_per_call": 8, "minutes": 10},
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type="normal",
)

# Collect the converged eigenvalue of every ky run into a spectrum (for the rho of index irho)
cgyro.read_linear_scan(label="kyscan", variable="KY", store_as_label="spectrum_rho05", irho=0)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the assembled linear spectrum
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (cgyro.fn); show() opens the GUI:
# first the standard per-ky tabs, then the quick gamma/omega vs ky spectrum view
cgyro.plot(labels=["kyscan_KY_0.3", "kyscan_KY_0.5", "kyscan_KY_0.7"])
fig = cgyro.fn.add_figure(label="Linear spectrum", tab_color=1)
cgyro.plot_quick_linear(labels=["spectrum_rho05"], fig=fig)
cgyro.fn.show()
