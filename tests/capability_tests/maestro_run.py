"""
CAPABILITY: Full MAESTRO chain from a namelist
----------------------------------------------
This script teaches how to launch MAESTRO — the multi-beat integrated-modeling
orchestrator — programmatically from a namelist. The template namelist chains
TRANSP (equilibrium + sources) -> EPED (pedestal) -> PORTALS (core transport)
-> EPED -> PORTALS to a self-consistent plasma, so this is a HEAVY run (hours,
and it dispatches TRANSP/EPED/PORTALS to the machines configured for them).

Key teaching points:
    1. The namelist is the single definition of the simulation: engineering
       parameters (Bt, Ip, shape, heating), the beat chain, and per-beat
       settings. Start from templates/namelist.maestro.yaml and modify a copy
       (the per-knob comments in the template are the documentation).
    2. run_maestro_local() is the programmatic equivalent of the CLI
       `mitim_run_maestro <folder> --namelist <file>`; terminal_outputs=True
       echoes the per-beat logs to the terminal instead of only to
       Outputs/Logs/.
    3. MAESTRO is checkpointed and idempotent: re-running with the same folder
       (force_cold_start=False) skips completed beats and resumes the
       interrupted one — this is why preempted/requeued SLURM jobs can simply
       re-execute it.
    4. Progress can be checked at any time from a terminal with
       `mitim_check_maestro <folder>`, and results plotted with
       `mitim_plot_maestro <folder>` (or m.plot() as below).
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

# The full namelist: plasma definition + beat chain + per-beat parameters
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# Avoid consuming the entire machine with pytorch threading during the PORTALS beats
torch.set_num_threads(8)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Run the chain
# ---------------------------------------------------------------------------------------------------------------------

m = run_maestro.run_maestro_local(
    template,
    folder=folder,
    # Echo the per-beat logs to the terminal (they always go to Outputs/Logs/ too)
    terminal_outputs=True,
    force_cold_start=cold_start,
    # CPUs available to the beats run locally
    cpus=8,
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Plot the evolution across beats
# ---------------------------------------------------------------------------------------------------------------------

# Profiles, equilibrium and performance figures comparing the last `num_beats` beats
m.plot(num_beats=4)
