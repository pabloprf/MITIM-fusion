"""
CAPABILITY: Trace-impurity transport coefficients (D, V) with TGLF
------------------------------------------------------------------
This script teaches the 'Z' flavor of runAnalysis(): extract the diffusion
coefficient D and convective velocity (pinch) V of a trace impurity from the
linear response of its TGLF particle flux to its own density gradient.

Key teaching points:
    1. A trace of the requested impurity (charge Z, PHYSICAL mass A in amu;
       here tungsten-like) is added to the input.tglf at negligible
       concentration (1e-6), so it does not perturb the underlying turbulence
       — it only rides on it.
    2. runAnalysis() scans the trace's density gradient (RLNS of the added
       species) and fits the flux-gradient relation
           Gamma_Z/n_Z = -D * (grad n_Z)/n_Z + V
       storing D (m^2/s), V (m/s) and the V/D ratio per radius — the
       steady-state zero-flux impurity peaking is exp(integral of V/D).
    3. ApplyCorrections is forced off internally so the species ordering of
       the input.tglf (with the appended trace) is preserved.
    4. plotAnalysis(analysisType='Z') shows the flux response and the derived
       D, V and V/D at each radius.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: the trace-impurity scan lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_traceDV"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations (required by runAnalysis)
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Trace-impurity D/V analysis (tungsten-like trace)
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    subfolder="traceW",
    label="traceW_sat2",
    analysisType="Z",
    # Charge and PHYSICAL mass (amu) of the trace; the mass is converted internally
    # to the deuterium-normalized convention of the input.tglf file
    trace=[74.0, 184.0],
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the flux response and the derived D, V, V/D per radius
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI.
# plotTGLFs=False skips the per-scan-point TGLF notebooks
tglf.plotAnalysis(labels=["traceW_sat2"], analysisType="Z", plotTGLFs=False)
tglf.fn.show()
