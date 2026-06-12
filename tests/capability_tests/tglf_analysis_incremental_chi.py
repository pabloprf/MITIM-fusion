"""
CAPABILITY: Incremental diffusivity (stiffness) analysis with TGLF
------------------------------------------------------------------
This script teaches how to extract transport coefficients from the local
flux-gradient response of TGLF, starting from an input.gacode file.

Key teaching points:
    1. runAnalysis(analysisType='chi_e') scans the electron temperature
       gradient (RLTS_1) by +-5% around its base value (11 points) and, from
       the Qe response, computes at each radius:
         - chi_inc : incremental (stiffness) diffusivity, dQ/d(grad T)
         - chi_eff : effective diffusivity, Q/(n grad T)
         - chi_pb  : power-balance diffusivity at the base gradient
         - Vpinch  : the equivalent heat pinch
       This requires the experimental normalizations, so the object must be
       prepared from a plasma state (input.gacode), not from a bare input.tglf.
    2. Other analysis types: 'chi_i' (RLTS_2 -> Qi), 'chi_ei' (cross-term:
       ion temperature gradient on electron heat flux) and 'Z' (trace-impurity
       D and V from a particle-flux scan; the trace charge and physical mass
       are given with trace=[Z, A]).
    3. plotAnalysis() shows the flux response and the derived coefficients.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: the analysis scan subfolder lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_incremental_chi"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations (required by runAnalysis to produce
# diffusivities in physical units)
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the electron incremental-diffusivity analysis
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    # Name of the subfolder (inside the working folder) where the scan lives
    subfolder="chi_e",
    # Results are stored in the object under this label
    label="chi_e_sat2",
    # Which analysis to perform (see docstring for the available types)
    analysisType="chi_e",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy);
    # extraOptions could also be passed here, exactly as in a single run
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the flux response and the derived diffusivities
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI.
# plotTGLFs=False skips the per-scan-point TGLF notebooks (11 points x 2 radii of them);
# set it to True to inspect the spectra of each individual scan point
tglf.plotAnalysis(labels=["chi_e_sat2"], analysisType="chi_e", plotTGLFs=False)
tglf.fn.show()
