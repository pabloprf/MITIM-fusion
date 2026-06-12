"""
CAPABILITY: MAESTRO chain with fixed-boundary-condition initialization
----------------------------------------------------------------------
This script teaches how to launch MAESTRO — the multi-beat integrated-modeling
orchestrator — programmatically, customizing the namelist in-situ. To keep the
example self-contained and machine-independent, the template's production
chain (TRANSP -> EPED -> PORTALS -> EPED -> PORTALS) is replaced by a minimal
one: a FreeGS-initialized plasma with a CONSTANT (user-fixed) boundary
condition, followed by a single (cheap) PORTALS beat — TGLF and NEO run
locally, no TRANSP/EPED machines needed.

Key teaching points:
    1. The namelist is the single definition of the simulation: engineering
       parameters, the beat chain, and per-beat settings. Start from
       templates/namelist.maestro.yaml (the per-knob comments are the
       documentation) and modify a copy — here done in-situ with
       read_mitim_yaml/write_mitim_yaml, like the PORTALS dictionaries.
    2. Profiles initialization is split into the equilibrium method
       (initialization_type: freegs here) and the profile creator
       (creator_type): 'eped_initializer' runs EPED for the pedestal, while
       'fixed_bc' (used here) pins Te/Ti at a chosen location x_bc and still
       matches BetaN and density peaking by adjusting the gradients — no
       pedestal code involved.
    3. run_maestro_local() is the programmatic equivalent of the CLI
       `mitim_run_maestro <folder> --namelist <file>`. MAESTRO is checkpointed
       and idempotent: re-running with the same folder (force_cold_start=False)
       skips completed beats and resumes the interrupted one.
    4. Progress can be checked at any time with `mitim_check_maestro <folder>`
       and results plotted with `mitim_plot_maestro <folder>` (or m.plot()).

*** WARNING ***: the PORTALS beat is capped at 2 BO iterations here ONLY so
that this teaching script finishes quickly — far too few for a converged flux
match. For physics runs, use the template chain and convergence defaults.
"""

import torch
from mitim_modules.maestro.scripts import run_maestro
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False resumes
# the chain from the last completed beat (see teaching point 3)
cold_start = True

# Working folder of the run: Beats/<n>_<type>/ subfolders and Outputs/ live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_maestro"

template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# Avoid consuming the entire machine with pytorch threading during the PORTALS beat
torch.set_num_threads(8)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Build the namelist: template + in-situ modifications
# ---------------------------------------------------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(template)

# --- Initialization: FreeGS equilibrium + CONSTANT boundary condition ------------------
# 'fixed_bc' pins the temperatures at x_bc; ne at the BC comes from plasma.parameters
# (neped_20) and the separatrix values from Tesep_eV and ne_ratio_sep_ped. BetaN and
# nu_ne (density peaking) are still matched by adjusting the core gradients.
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV (Ti_bc: null -> same as Te_bc)

# --- Beat chain: a single PORTALS beat instead of the production template chain --------
nml["maestro"]["beats"] = ["portals"]

# --- Cheapen the PORTALS beat (see the WARNING in the docstring) ------------------------
pp = nml["maestro"]["portals"]["parameters_prepare"]["portals_parameters"]
pp["solution"]["predicted_roa"] = [0.35, 0.55, 0.75, 0.9]
pp.setdefault("optimization_options", {}).setdefault("convergence_options", {})["maximum_iterations"] = 2

# The exact namelist used is written alongside the run for traceability
namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the chain
# ---------------------------------------------------------------------------------------------------------------------

m = run_maestro.run_maestro_local(
    namelist_file,
    folder=folder,
    # Echo the per-beat logs to the terminal (they always go to Outputs/Logs/ too)
    terminal_outputs=True,
    force_cold_start=cold_start,
    # CPUs available to the beats run locally
    cpus=8,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the result
# ---------------------------------------------------------------------------------------------------------------------

m.plot(num_beats=1)
