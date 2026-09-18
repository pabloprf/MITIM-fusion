"""
CAPABILITY: MAESTRO chain with HARVEST (archive the evaluations of every beat into a database)
-----------------------------------------------------------------------------------------------
A short MAESTRO chain that opts into the "harvest" program:

    init (FreeGS + fixed BC) -> portals -> eped -> portals

Every individual TGLF and NEO evaluation of BOTH PORTALS beats (base points and scan-trick
members) and every full-EPED evaluation of the EPED beat are recorded as input -> output
records with their provenance, and pushed ONCE, by MAESTRO at the end of the chain, into a
per-user netCDF-4 database. See portals_04_harvest.py for the standalone-PORTALS version and
the record layout.

Key teaching points:
    1. One switch for the whole chain: `maestro.harvest.enabled: true` (+ optional `file`,
       `scan_trick_members`). MAESTRO forwards it to each PORTALS beat with its own run id, and
       the EPED beat records itself; nothing else changes in the beat configs.
    2. Records of every beat (PORTALS and EPED alike) are staged as the run goes in ONE folder,
       <run>/Outputs/harvest/, compressed on the fly, each tagged with its `maestro_beat`; the
       folder sits outside Beats/ so it survives every prune_level. MAESTRO pushes at finalize;
       a chain that died can be pushed later with `mitim_harvest <folder>`.
    3. Only FULL EPED evaluations are harvested (EPED-NN is a surrogate itself, and surrogate
       outputs must never enter a training set). Full EPED runs on the machine configured for
       `eped` in config_user.json (TGLF/NEO of the PORTALS beats run wherever `tglf`/`neo`
       point to). The EPED record holds the complete eped.input namelist as run, the effective
       driver settings (NMODES, WIDTHS, TEPED_BOUND), the MITIM stability rule and threshold that
       picked the pedestal, and the pedestal outputs.

*** WARNING ***: the PORTALS beats are capped at 2 BO iterations ONLY so that this teaching
script finishes quickly — far too few for converged results.
"""

import torch
from mitim_modules.maestro.scripts import run_maestro
from mitim_tools.harvest_tools import HARVESTtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

folder = __mitimroot__ / "tests" / "scratch" / "capability_maestro_harvest"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

# The central database of this example (NOT the user's real one, which the template default points to)
harvest_file = __mitimroot__ / "tests" / "scratch" / "capability_harvest_maestro.nc"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)
if cold_start and harvest_file.exists():
    harvest_file.unlink()

torch.set_num_threads(8)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Build the namelist: template + in-situ modifications (see maestro_01_run.py for the details of each block)
# ---------------------------------------------------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(template)

# Initialization: FreeGS equilibrium + constant boundary condition (no pedestal code at init)
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0

# Beat chain: two PORTALS beats around one EPED beat
nml["maestro"]["beats"] = ["portals", "eped", "portals"]

# The ONLY harvest-specific lines: enable it for the whole chain and choose the central file
nml["maestro"]["harvest"]["enabled"] = True
nml["maestro"]["harvest"]["file"] = str(harvest_file)

# PORTALS beats: cheap Te/Ti prediction with SAT0 (both "portals" entries share this block)
pp = nml["maestro"]["portals"]["parameters_prepare"]["portals_parameters"]
pp["solution"]["predicted_roa"] = [0.35, 0.55, 0.75, 0.9]
pp["solution"]["predicted_channels"] = ["te", "ti"]
pp["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT0"
pp["transport"]["options"]["tglf"]["run"]["extraOptions"] = {"USE_BPER": False}
pp.setdefault("optimization_options", {}).setdefault("convergence_options", {})["maximum_iterations"] = 2

# EPED beat: template defaults (full EPED, 'standard' TOQ equilibrium, stability rule ['G', 0.03])

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the chain (MAESTRO pushes the harvest at finalize)
# ---------------------------------------------------------------------------------------------------------------------

m = run_maestro.run_maestro_local(namelist_file, folder=folder, terminal_outputs=True, force_cold_start=cold_start, cpus=8)

# ---------------------------------------------------------------------------------------------------------------------
# 3. What was harvested, and where it was staged
# ---------------------------------------------------------------------------------------------------------------------

print("\nStaging folders of this MAESTRO run (all pushed, archives kept):")
for f in HARVESTtools.staging_folders_of(folder):
    print(f"   {f.relative_to(folder)}: {[p.name for p in sorted(f.glob('*.jsonl*'))]}")

db = HARVESTtools.harvest_database(harvest_file)
print("\nDatabase summary:")
print(db.summary().to_string(index=False))

# Both PORTALS beats share the MAESTRO run id: one provenance row per (run, code), records from both beats
print("\nRuns table:")
print(db.runs()[["run", "code", "machine", "maestro_beat", "code_version"]].assign(code_version=lambda d: d["code_version"].str.split("\n").str[0]).to_string(index=False))

tglf = db.load("tglf")

# Records contributed by each beat: all beats stage into <run>/Outputs/harvest, each record tagged with its maestro_beat
print("\nTGLF records per beat:")
print(tglf.groupby("maestro_beat").size().to_string())

print(f"\nTGLF records in the database: {len(tglf)} ({sum(c.startswith('in_') for c in tglf.columns)} inputs -> {sum(c.startswith('out_') for c in tglf.columns)} outputs)")

eped = db.load("eped")
print(f"\nEPED records: {len(eped)} (one per full-EPED evaluation, retries included)")
if len(eped):
    cols = [c for c in ["in_ip", "in_bt", "in_neped", "in_betan", "in_zeffped", "in_teped", "in_cfg_TEPED_BOUND_0", "in_cfg_TEPED_BOUND_1",
                        "in_stability_rule", "in_stability_threshold", "out_ptop_kPa", "out_wtop_psipol", "out_n_limiting", "out_limiting_mode"] if c in eped]
    print(eped[cols].to_string(index=False))
    print("EPED inputs stored:", [c[3:] for c in eped.columns if c.startswith("in_")])

# ---------------------------------------------------------------------------------------------------------------------
# 4. Interpret and plot the database (same as `mitim_plot_harvest <file>`)
# ---------------------------------------------------------------------------------------------------------------------

db.interpret()
fn = db.plotDatabase()   # Overview + one tab per code (+ EPED tab when full EPED ran); fn.show() is the only display call
fn.show()
