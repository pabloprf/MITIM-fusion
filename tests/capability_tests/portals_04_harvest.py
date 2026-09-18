"""
CAPABILITY: PORTALS run with HARVEST (archive every TGLF/NEO evaluation into a database)
-----------------------------------------------------------------------------------------
Same cheap PORTALS run as portals_01_tglf_standard.py, but opting into the "harvest"
program: every individual TGLF and NEO evaluation of the run (the base point at each radius
AND every perturbed member of the TGLF std scan trick) is recorded with its full input file,
its scalar GB fluxes and provenance (code version, machine, MITIM commit), and appended at
the end of the run into a per-user netCDF-4 database that can be reused off-line (surrogate
training, physics studies).

Key teaching points:
    1. Opt in with the `harvest:` block of the PORTALS namelist (`enabled: true`). The central
       file is `file:`, else config_user.json `preferences.harvest_file`, else
       ~/mitim_harvest/mitim_harvest.nc. Inside MAESTRO, the same block lives under
       `maestro.harvest` and MAESTRO pushes once at the end for all its beats (+ full EPED).
    2. During the run, records are STAGED as JSON-lines under Outputs/harvest/ (one file per
       code) and pushed to the central file when the run finishes (under an NFS-safe lock, so
       many runs can share the file). A run that died can be pushed later with
       `mitim_harvest <run folder>`.
    3. The central file is inspected with `HARVESTtools.harvest_database` (load / summary /
       interpret / plotDatabase) or from the terminal with `mitim_plot_harvest <file>`.
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.harvest_tools import HARVESTtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_harvest"

# The central database of this example (NOT the user's real one, which the template default points to)
harvest_file = __mitimroot__ / "tests" / "scratch" / "capability_harvest.nc"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)
if cold_start and harvest_file.exists():
    harvest_file.unlink()

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize PORTALS (defaults from templates/namelist.portals.yaml) and opt into harvesting
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# The ONLY harvest-specific lines: enable it and (optionally) choose the central file
portals_fun.portals_parameters["harvest"]["enabled"] = True
portals_fun.portals_parameters["harvest"]["file"] = str(harvest_file)

# Cheap run, exactly as in portals_01_tglf_standard.py (see that script for what each knob does)
portals_fun.optimization_options["initialization_options"]["initial_training"] = 3
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 1
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT2"
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["extraOptions"] = {"USE_BPER": False, "USE_BPAR": False}

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare and run
# ---------------------------------------------------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})
portals_fun.prep(plasma_state)

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()
# -> at the end of run(), PORTALS pushed Outputs/harvest/*.jsonl into harvest_file and renamed them *.pushed-<ts>

# ---------------------------------------------------------------------------------------------------------------------
# 3. What was harvested
# ---------------------------------------------------------------------------------------------------------------------

staging = folderWork / "Outputs" / "harvest"
print("\nStaged files (renamed after the push):")
for f in sorted(staging.iterdir()):
    print(f"   {f.name}")

db = HARVESTtools.harvest_database(harvest_file)
print("\nDatabase summary:")
print(db.summary().to_string(index=False))

tglf = db.load("tglf")
neo = db.load("neo")
in_cols = [c for c in tglf.columns if c.startswith("in_")]
out_cols = [c for c in tglf.columns if c.startswith("out_")]
print(f"\nTGLF records: {len(tglf)}  ({len(in_cols)} inputs -> {len(out_cols)} outputs, keys: run, hash)")
print(f"NEO  records: {len(neo)}")

# A record is just inputs -> outputs. Every individual TGLF run is one record: the base point of each
# evaluation and radius, plus each scan-trick member with its own perturbed input. The members of one
# scan trick are the records that differ from their base point in only a few inputs (the scanned
# variable and the species TGLF ties to it, e.g. RLTS_2 together with RLTS_3):
ref = tglf.iloc[0]
n_diff = (tglf[in_cols] != ref[in_cols]).sum(axis=1)
siblings = tglf[n_diff <= 3]
print(f"\nRecords differing from the first record in at most 3 inputs (base point + its 10 scan-trick members): {len(siblings)}")
print(siblings[["in_RLTS_1", "in_RLTS_2", "in_TAUS_2", "in_XNUE", "in_BETAE", "out_Qe", "out_Qi"]].to_string(index=False))

# Who produced them: once per run and code in the `runs` table, joined on load
print("\nProvenance (runs table):")
for k in ["code_version", "machine", "mitim_version", "git_commit", "run"]:
    print(f"   {k:15s} {(str(tglf[k].iloc[0]).splitlines() or ['(not available)'])[0]}")

# ---------------------------------------------------------------------------------------------------------------------
# 4. Interpret and plot the database (same as `mitim_plot_harvest <file>`)
# ---------------------------------------------------------------------------------------------------------------------

db.interpret("tglf")

# (a) The multi-tab notebook: Overview (records per code/run, timeline, machine + code version), one
#     tab per code (flux histograms; Qe/Qi/Ge vs their main drives) and an EPED tab when present
fn = db.plotDatabase()

# (b) A custom scatter of any output vs any input of a code, colored by run, added as one more tab of
#     the same notebook (names with or without the in_/out_ prefix). Here Qi (GB) vs a/LTi (RLTS_2):
#     every dot is one individual TGLF run, so the +-2% scan-trick members cluster around each base point.
#     NOTE: notebook figures are pyplot figures; do NOT call plt.show() next to a FigureNotebook, or every
#     tab pops up again as a separate window. fn.show() is the only display call needed.
ax = fn.add_figure(label="Qi vs a/LTi").add_subplot(111)
db.plot("tglf", "RLTS_2", "Qi", ax=ax)
ax.set_title("Every TGLF run harvested: Qi vs a/LTi")

fn.show()

# Manual push of a run whose push did not happen (killed run, or a MAESTRO beat): from the terminal
#   mitim_harvest <run folder> [--file <central file>]
# which is what this does:
print("\nManual re-push (nothing left to push, as expected):", db.push(HARVESTtools.staging_folders_of(folderWork)))
