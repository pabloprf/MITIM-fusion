"""
CAPABILITY: Pedestal prediction with EPED (including a parameter scan)
----------------------------------------------------------------------
This script teaches how to predict the pedestal height and width with EPED
from a set of scalar engineering and pedestal parameters, scanning one of
them. EPED runs on the machine configured for "eped" in config_user.json.

Key teaching points:
    1. EPED does not start from an input.gacode: its inputs are the scalars in
       `input_params` (machine: ip, bt, r, a, kappa, delta, zeta; pedestal:
       neped, betan, zeffped; separatrix: nesep, tesep).
    2. `scan_param` repeats the run over a list of values of one input (here
       the pedestal density), submitted as a SLURM job array. `keep_nsep_ratio`
       ties the separatrix density to the scanned pedestal density.
    3. `eped_params_override` modifies parameters of the EPED configuration
       file itself (the EPED analog of extraOptions).
    4. read() parses the scan results and plot() shows pedestal height/width
       vs the scanned parameter(s).
    5. read() also chooses the DIAMAGNETIC STABILIZATION RULE that turns the
       ELITE growth-rate spectrum into a single predicted pedestal. Two rules
       are read here from the very same EPED output (no extra runs) and
       overplotted (see section 3).
    6. The two cases differ ONLY in the toroidal mode set (NMODES up to 30 vs
       up to 60), demonstrating which rule is converged in the mode ceiling.
"""

from mitim_tools.eped_tools import EPEDtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = False

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "capability_eped"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize EPED and run a pedestal-density scan
# ---------------------------------------------------------------------------------------------------------------------

eped = EPEDtools.EPED(folder=folder)

# Scanned pedestal densities (10^19 m^-3). The scan deliberately runs deep into the
# high-density end, where the pedestal becomes limited by the high-n ballooning branch
# (Hughes et al., J. Plasma Phys. 86, 865860504 (2020)) and the stability rule matters most.
neped_values = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0]

eped.run(
    subfolder="case1",
    # Base scalar inputs of EPED (SPARC-like values here)
    input_params={
        "ip": 8.7,        # plasma current (MA)
        "bt": 12.16,      # toroidal field (T)
        "r": 1.85,        # major radius (m)
        "a": 0.57,        # minor radius (m)
        "kappa": 1.9,     # elongation
        "delta": 0.5,     # triangularity
        "zeta": 0.01,     # squareness (if your EPED implementation supports it)
        "neped": 30.0,    # pedestal density (10^19 m^-3), overridden by the scan
        "betan": 1.5,     # normalized beta
        "zeffped": 1.5,   # pedestal Zeff
        "nesep": 10.0,    # separatrix density (10^19 m^-3); superseded by keep_nsep_ratio below
        "tesep": 100.0,   # separatrix temperature (eV)
    },
    scan_param={"variable": "neped", "values": neped_values},
    keep_nsep_ratio=0.4,
    nproc_per_run=64,
    # Search window of the pedestal-temperature march. Floor at 0.1 keV (not the usual 0.3):
    # the highest-neped points are already unstable at 0.3 keV, and would return nan.
    # NOTE: this gives 131 heights, and heights x modes must stay <= 1024 (EPED runner job
    # limit, checked at submission) -- hence 7-mode NMODES sets in both cases.
    eped_params_override={"TEPED_BOUND": [0.1, 1.4, 0.01]},
    cold_start=cold_start,
    job_array_limit=5,
    # EPED scratch trees are enormous: only set to False for debugging
    removeScratchFolders=True,
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. The same scan with a higher toroidal-mode ceiling (n up to 60 instead of 30)
# ---------------------------------------------------------------------------------------------------------------------

eped.run(
    subfolder="case2",
    input_params={
        "ip": 8.7,
        "bt": 12.16,
        "r": 1.85,
        "a": 0.57,
        "kappa": 1.9,
        "delta": 0.5,
        "zeta": 0.01,
        "neped": 30.0,
        "betan": 1.5,
        "zeffped": 1.5,
        "nesep": 10.0,
        "tesep": 100.0,
    },
    scan_param={"variable": "neped", "values": neped_values},
    keep_nsep_ratio=0.4,
    nproc_per_run=64,
    # Still 7 modes (job limit): the added n = 40, 50, 60 displace the near-redundant low-n ones
    eped_params_override={"TEPED_BOUND": [0.1, 1.4, 0.01], "NMODES": [5, 10, 20, 30, 40, 50, 60]},
    cold_start=cold_start,
    job_array_limit=5,
    removeScratchFolders=True,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Read both cases with both diamagnetic stabilization rules and overplot them
# ---------------------------------------------------------------------------------------------------------------------

# Two rules to pick the pedestal from the gamma(height, n) spectrum:
#   'G' (default, threshold 0.03): flat cut on gamma/omega_A -- same bar for every n, so the
#       prediction chases the highest n in the set (NOT converged in the mode ceiling).
#   'W' (threshold = O(1) calibration factor C): EPED1 diamagnetic criterion
#       gamma > C * omega_*i(n)/2 (Snyder et al., Phys. Plasmas 16, 056118 (2009)). The bar
#       rises linearly with n, so the mode set self-truncates and the answer converges.
# In the overlay: the two flat curves should differ (higher ceiling -> lower pedestal), the
# two 'W' curves should coincide; the annotated limiting n makes this explicit.
#
# The 'W' rule needs a companion plasma state, purely to supply the flux/density
# normalizations of omega_*; it must describe the same plasma as the EPED scalars.
gacode_for_omega_star = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"

# ptop = nan means no height in the TEPED_BOUND window was stable under that rule; such a
# point is omitted from the overlay and its profile tab shows no selected point.
for label_tag, case in (("n30", "case1"), ("n60", "case2")):
    eped.read(subfolder=case, label=f"{label_tag}_flat")
    eped.read(
        subfolder=case,
        label=f"{label_tag}_omegastar",
        diamagnetic_stab_rule="W",
        stability_threshold=1.0,
        gacode_state=gacode_for_omega_star,
    )

eped.plot(
    labels=["n30_flat", "n30_omegastar", "n60_flat", "n60_omegastar"],
    scan_params=["neped"],
    scan_params_labels=["$n_{e,ped}\\ (10^{19}m^{-3})$"],
)
eped.fn.show()
