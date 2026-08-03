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
       the pedestal density), submitted as a SLURM job array with at most
       `job_array_limit` concurrent cases. `keep_nsep_ratio` ties the
       separatrix density to the scanned pedestal density (nesep = ratio *
       neped) so the scan stays physically consistent.
    3. `eped_params_override` modifies parameters of the EPED configuration
       file itself (the EPED analog of extraOptions), e.g. the bounds and
       resolution of the pedestal-temperature search.
    4. read() parses the scan results and plot() shows pedestal height/width
       vs the scanned parameter(s).
    5. read() also chooses the DIAMAGNETIC STABILIZATION RULE that turns the
       ELITE growth-rate spectrum into a single predicted pedestal. Two rules
       are read here from the very same EPED output (no extra runs) and
       overplotted, see the discussion in section 3 below.
    6. Two EPED cases are run that differ ONLY in the set of toroidal mode
       numbers explored (NMODES up to 30 vs up to 40). Reading both cases under
       both rules is the cleanest demonstration of why the rule matters: one
       rule is converged with respect to the mode ceiling and the other is not.
"""

from mitim_tools.eped_tools import EPEDtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = False

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one subfolder per case lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_eped"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize EPED and run a pedestal-density scan
# ---------------------------------------------------------------------------------------------------------------------

eped = EPEDtools.EPED(folder=folder)

# Scanned pedestal densities (10^19 m^-3), shared by both cases below. The scan deliberately
# runs deep into the high-density end: as neped rises the pedestal becomes limited by the
# high-n BALLOONING branch rather than by the low-n current-driven peeling branch (the SPARC
# behavior described in Hughes et al., J. Plasma Phys. 86, 865860504 (2020)). That high-n
# ballooning corner is precisely where the choice of stability rule below changes the answer,
# because the two rules treat high n completely differently.
neped_values = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0]

eped.run(
    # Name of the subfolder (inside the working folder) where this case lives
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
        "neped": 30.0,    # pedestal density (10^19 m^-3)
        "betan": 1.5,     # normalized beta. NOTE on high-density scan points: neped >= 100 (1e19)
                          # makes TOQ's fixed-width peddata output glue adjacent fields, and the
                          # EPED driver's whitespace-split parser (toq_io.read_peddata) then fails
                          # SILENTLY on every height (eq_* = -1 in the netCDF, gamma still fine)
                          # -> ptop = nan under any rule. Same bug hits cold pedestals via large
                          # nu*. It is a driver parse bug, not physics -- TOQ converges; fixed by
                          # passing itype=1/2 in read_peddata (see study notes). betan itself was
                          # NOT the culprit (an earlier hypothesis); 1.5 kept for continuity.
        "zeffped": 1.5,   # pedestal Zeff
        "nesep": 10.0,    # separatrix density (10^19 m^-3); superseded by keep_nsep_ratio below
        "tesep": 100.0,   # separatrix temperature (eV)
    },
    # Scan one of the inputs over these values (each value is one EPED case)
    scan_param={"variable": "neped", "values": neped_values},
    # Tie the separatrix density to the scanned pedestal density: nesep = 0.4 * neped
    keep_nsep_ratio=0.4,
    # Cores for each EPED case
    nproc_per_run=64,
    # Override parameters of the EPED configuration file itself, e.g. the [min, max, step]
    # bounds of the pedestal-temperature search. The floor is 0.1 keV rather than the more
    # usual 0.3: at neped = 105e19 the pedestal is ALREADY unstable at 0.3 keV under both
    # stability rules, so EPED returns "no stable solution" (the deep-ballooning fallback
    # regime) and the marginal point is never resolved. Dropping the floor to 0.1 recovers it.
    # This sets num_heights = (1.4 - 0.1)/0.01 = 130, and 130 x 7 modes = 910 stays under the
    # 1024-job limit of the EPED runner -- which is exactly why NMODES must stay at 7 (see below).
    # NMODES is left at the template default here (5 6 8 10 15 20 30), i.e. a ceiling of n = 30.
    eped_params_override={"TEPED_BOUND": [0.1, 1.4, 0.01]},
    cold_start=cold_start,
    # At most this many cases of the scan run concurrently (SLURM job array)
    job_array_limit=5,
    # EPED scratch trees are enormous: only set to False for debugging
    removeScratchFolders=True,
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. The same scan with a higher toroidal-mode ceiling (n up to 40 instead of 30)
# ---------------------------------------------------------------------------------------------------------------------

# Identical to case1 except for NMODES: n = 40 is added and n = 6 is REMOVED, so the set stays
# at SEVEN modes. That is not cosmetic -- it is a hard constraint of the EPED job runner:
#
#   run_parallel.exe dispatches one ELITE job per (pedestal height, mode number) pair and has a
#   HARDCODED 1024-job limit with NO bounds check. If num_heights * num_modes exceeds 1024,
#   ELITE simply never runs for the excess: the run still exits 0 and the output netCDF is
#   written, but with gamma = -1 everywhere. The failure is completely silent.
#
#   num_heights comes from TEPED_BOUND: with the [0.1, 1.4, 0.01] window used here that is
#   130 heights, so 130 x 7 = 910 is safe but 130 x 8 = 1040 is already over the cliff.
#   Hence 7 modes, not 8.
#
# n = 6 is the right one to sacrifice: at the low-n end the spectrum is smooth and n = 6 is
# nearly redundant with n = 5, whereas the whole question here is what happens at high n.
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
        "betan": 1.5,     # must match case1 (see the betan feasibility note there)
        "zeffped": 1.5,
        "nesep": 10.0,
        "tesep": 100.0,
    },
    scan_param={"variable": "neped", "values": neped_values},
    keep_nsep_ratio=0.4,
    nproc_per_run=64,
    eped_params_override={"TEPED_BOUND": [0.1, 1.4, 0.01], "NMODES": [5, 8, 10, 15, 20, 30, 40]},
    cold_start=cold_start,
    job_array_limit=5,
    removeScratchFolders=True,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Read both cases with both diamagnetic stabilization rules and overplot them
# ---------------------------------------------------------------------------------------------------------------------

# EPED gives, for every trial pedestal height, the growth rate gamma/omega_A of each toroidal
# mode number n. Picking THE pedestal means deciding when a mode counts as unstable, and MITIM
# offers two rules:
#
#   'G' (default, threshold 0.03): a FLAT cut. The pedestal is the first height where the
#       largest gamma/omega_A over all n exceeds 0.03 -- the same bar for every n, so the rule
#       is always set by whichever n happens to grow fastest, typically the highest n computed.
#
#   'W' (threshold = calibration factor C): the EPED1 diamagnetic criterion
#       gamma > C * omega_*i(n)/2, with omega_*i = (n/(Z_i e n_i)) dp_i/dpsi maximized across
#       the pedestal barrier (Snyder et al., Phys. Plasmas 16, 056118 (2009)). Because
#       omega_*i grows linearly with n, the bar RISES with n: a high-n mode must grow much
#       faster than a low-n one to limit the pedestal, and the mode set self-truncates.
#       Physically this is diamagnetic (omega_*) stabilization of the short-wavelength modes.
#       C is an O(1) calibration factor (nominal 1.0), not a growth rate: the absolute
#       normalization against ELITE's internal Alfven normalization is uncertain.
#
# WHAT TO LOOK FOR IN THE OVERLAY -- this contrast is the whole point of the exercise. The two
# cases differ only in the mode ceiling (n <= 30 vs n <= 40), so a rule that is converged with
# respect to the mode set must give the SAME pedestal for both:
#
#   * the two FLAT curves should NOT coincide: raising the ceiling 30 -> 40 should LOWER the
#     predicted pedestal, because a flat bar is easiest to clear for the fastest-growing mode
#     and that is typically the highest n available. The flat rule therefore keeps chasing the
#     top of whatever mode set was requested -- it is not converged in n, and the "prediction"
#     partly reflects the NMODES choice. The effect is largest at high neped, in the
#     ballooning-limited corner where the high-n modes dominate.
#   * the two 'W' curves should lie on top of each other: n = 40 has to beat a threshold ~4/3
#     that of n = 30 (and ~8x that of n = 5), so the added mode is declared diamagnetically
#     stabilized and never becomes the limiter. The criterion self-truncates the mode set, and
#     the answer stops depending on where the mode list was cut.
#
# The limiting n annotated next to each marker makes this explicit: it should climb to the top
# of the set under the flat rule, and stay put under the 'W' rule.
#
# The 'W' rule needs a companion plasma state, purely to supply the flux/density
# normalizations of omega_* (the dimensional psi from torfluxa, and the mass density). In a
# real application this must be the plasma the EPED case describes; here we reuse the SPARC
# input.gacode shipped with the tests, which matches the SPARC-like scalars given above.
gacode_for_omega_star = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"

# A case can still come back with ptop = nan: that means no pedestal height in the scanned
# TEPED_BOUND window was stable under that rule, so there is genuinely nothing to report. Such
# a point is simply omitted from the overlay (it shows up as a cross at zero instead of a
# marker), and its profile tab draws the whole height stack with no selected point marked.
#
# read() parses the EPED output files of every case of the scan. Reading a case twice under
# two labels re-postprocesses the same files, it does not re-run anything.
for label_tag, case in (("n30", "case1"), ("n40", "case2")):
    eped.read(subfolder=case, label=f"{label_tag}_flat")
    eped.read(
        subfolder=case,
        label=f"{label_tag}_omegastar",
        diamagnetic_stab_rule="W",
        stability_threshold=1.0,
        gacode_state=gacode_for_omega_star,
    )

# All figures go into a multi-tab MITIM FigureNotebook (eped.fn); show() opens the GUI.
# All four labels are overlaid in the Pedestal Top tab, and each gets its own Stability tab
# showing the flat horizontal cut versus the per-n (dashed, one per mode) thresholds.
eped.plot(
    labels=["n30_flat", "n30_omegastar", "n40_flat", "n40_omegastar"],
    scan_params=["neped"],
    scan_params_labels=["$n_{e,ped}\\ (10^{19}m^{-3})$"],
)
eped.fn.show()
