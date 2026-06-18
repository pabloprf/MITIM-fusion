"""
DEV TEST: MAESTRO rotation flow (TRANSP -> PORTALS -> TRANSP -> PORTALS)
-----------------------------------------------------------------------
Exercises and lets you MONITOR the toroidal rotation (w0) as it moves through a
MAESTRO chain, using the rotation plumbing added alongside this test:
  - PORTALS predicts w0 (rotation added to predicted_channels), so each PORTALS
    beat evolves the rotation profile.
  - each TRANSP beat passes the incoming w0 INTO TRANSP as the 'omg' U-File
    (gacode_state.to_transp auto-writes it when w0 != 0) and writes the TRANSP
    rotation back out to w0 (OMEGA, with the NCLASS neoclassical Er/omega now in
    the CDF by default).
So rotation should flow: (seed w0=0) -> PORTALS predicts a w0 -> next TRANSP
ingests it -> next PORTALS evolves it. The point is to WATCH that propagation,
not to converge it.

*** WARNING ***: both the TRANSP flattop and the PORTALS iteration cap are cut
to the bone here ONLY so the chain finishes fast enough to inspect. These are
FAR too short for converged physics — do not read the numbers as results.

*** REQUIREMENTS ***: the "transp" machine in config_user.json (TRANSP runs) and
TGLF/NEO for the PORTALS beats (same dependencies as maestro_01_run.py).

Monitor afterwards with `mitim_plot_maestro <folder> --beats 4` (the profile tabs
include w0), or read the per-beat printout at the end of this script.
"""

import numpy as np
import torch

from mitim_modules.maestro.scripts import run_maestro
from mitim_modules.maestro.utils import MAESTROplot
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True
folder = __mitimroot__ / "tests" / "scratch" / "dev_maestro_rotation"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(8)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Namelist: template + in-situ edits
# ---------------------------------------------------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(template)

# Constant-BC initialization (FreeGS + fixed_bc), as in maestro_01 — avoids the EPED
# dependency so the chain is just TRANSP + PORTALS.
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0

# The chain to monitor: TRANSP -> PORTALS -> TRANSP -> PORTALS. Each beat type appears
# twice and shares its single config block below.
nml["maestro"]["beats"] = ["transp", "portals", "transp", "portals"]

# --- PORTALS beats: PREDICT ROTATION (add w0), kept very short --------------------------
pp = nml["maestro"]["portals"]["parameters_prepare"]["portals_parameters"]
pp["solution"]["predicted_roa"] = [0.4, 0.6, 0.8]
pp["solution"]["predicted_channels"] = ["te", "ti", "ne", "w0"]   # <-- rotation is now predicted
pp.setdefault("optimization_options", {}).setdefault("convergence_options", {})["maximum_iterations"] = 2

# --- TRANSP beats: short flattop (see the WARNING) --------------------------------------
# NUBEAM/TORIC still dominate the wall time; the flattop is the main length knob.
nml["maestro"]["transp"]["parameters_prepare"]["flattop_window"] = 0.5

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the chain
# ---------------------------------------------------------------------------------------------------------------------

m = run_maestro.run_maestro_local(
    namelist_file,
    folder=folder,
    terminal_outputs=True,
    force_cold_start=cold_start,
    cpus=8,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Monitor the rotation across beats
# ---------------------------------------------------------------------------------------------------------------------

objs, _, _ = MAESTROplot.collect_beat_states(m)

print("\n" + "=" * 64)
print(" Toroidal rotation w0(rad/s) across the MAESTRO chain")
print("=" * 64)
print(f" {'state':<22}{'w0(0)':>13}{'w0(rho=0.5)':>15}")
print("-" * 64)
for label, st in objs.items():
    if st is None:
        continue
    rho, w0 = st.profiles["rho(-)"], st.profiles["w0(rad/s)"]
    print(f" {label:<22}{w0[0]:>13.3e}{np.interp(0.5, rho, w0):>15.3e}")
print("=" * 64)

# Full plots (the profile tabs include w0)
m.plot(num_beats=4)
