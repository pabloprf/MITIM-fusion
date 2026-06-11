"""
CAPABILITY: TGLF runs from a TRANSP output (CDF), with experimental fluctuations
--------------------------------------------------------------------------------
This script teaches the third TGLF entry point (besides input.tglf and
input.gacode, see the other tglf_run_* capabilities): starting directly from a
TRANSP output file, extracting the plasma state at a chosen time. It also
teaches how to compare saturation rules against experimental fluctuation
measurements through the synthetic diagnostic.

NOTE: the preparation goes through the TGYRO route (prep_using_tgyro), which
prints a deprecation notice — this entry point will evolve in a future
release. Requires gacode (profiles_gen) configured.

Key teaching points:
    1. TGLF(cdf=..., time=..., avTime=...) extracts the plasma state from the
       TRANSP run at `time`, averaging over the `avTime` window (smooths
       sawteeth/noise); prep_using_tgyro() then builds the per-rho inputs.
    2. Several saturation rules can be run and read under different labels,
       each with eigenfunction waveforms at requested ky's (see
       tglf_run_waveforms.py).
    3. d_perp_cm at read() time activates the synthetic fluctuation
       diagnostic (perpendicular resolution of the measurement at each rho),
       and the measured fluctuation levels with error bars are attached via
       NormalizationSets["EXP"] — the plot then shows model vs experiment for
       fluxes AND fluctuations (same machinery that VITALS exploits, see
       vitals_tglf_validation.py).
"""

import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Example TRANSP output distributed with MITIM (C-Mod shot 12345)
cdf_file = __mitimroot__ / "tests" / "data" / "12345.CDF"

# Working folder of the run
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_from_cdf"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF from the TRANSP output at t = 2.5 s
# ---------------------------------------------------------------------------------------------------------------------

# The plasma state is extracted from the CDF at `time`, averaged over the `avTime`
# window; experimental normalizations come from the TRANSP run itself
tglf = TGLFtools.TGLF(cdf=cdf_file, time=2.5, avTime=0.02, rhos=np.array([0.6, 0.8]))
tglf.prep_using_tgyro(folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run three saturation rules, with waveforms, and read each with the synthetic diagnostic
# ---------------------------------------------------------------------------------------------------------------------

# d_perp_cm: perpendicular resolution (cm) of the fluctuation measurement at each rho,
# used by the synthetic diagnostic to filter the TGLF spectrum like the instrument does
d_perp_cm = {0.6: 0.5, 0.8: 0.5}

tglf.run(
    subfolder="runSAT0",
    code_settings="SAT0",
    runWaveForms=[0.5],
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf.read(label="runSAT0", d_perp_cm=d_perp_cm)

tglf.run(
    subfolder="runSAT2",
    code_settings="SAT2",
    runWaveForms=[0.1, 0.3],
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf.read(label="runSAT2", d_perp_cm=d_perp_cm)

tglf.run(
    subfolder="runSAT3",
    code_settings="SAT3",
    runWaveForms=[0.5],
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf.read(label="runSAT3", d_perp_cm=d_perp_cm)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Attach the experimental fluctuation measurements and error bars
# ---------------------------------------------------------------------------------------------------------------------

# Measured Te fluctuation levels (%) at each radius, with their error
tglf.NormalizationSets["EXP"]["exp_TeFluct_rho"] = [0.6, 0.8]
tglf.NormalizationSets["EXP"]["exp_TeFluct"] = [1.12, 1.49]
tglf.NormalizationSets["EXP"]["exp_TeFluct_error"] = 0.2

# Error bars on the power-balance fluxes (the values come from the TRANSP run)
tglf.NormalizationSets["EXP"]["exp_Qe_error"] = 0.005
tglf.NormalizationSets["EXP"]["exp_Qi_error"] = 0.005

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot the three saturation rules against each other and against the measurements
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI
tglf.plot(labels=["runSAT0", "runSAT2", "runSAT3"])
tglf.fn.show()
