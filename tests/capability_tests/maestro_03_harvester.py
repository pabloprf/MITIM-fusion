"""
CAPABILITY: HARVESTER (rebuild the harvest records of MAESTRO runs that did NOT enable harvesting)
---------------------------------------------------------------------------------------------------
maestro_02_harvest.py shows harvesting switched on BEFORE a run. Most existing MAESTRO runs never
had it. `mitim_harvester` rebuilds, from the files those runs left on disk, the same records a live
run would have staged, and appends them to a harvest file:

    mitim_harvester <run folder(s) or parent folder of runs> <harvest_file.nc> [--dry-run] [--stage DIR]

Key teaching points:
    1. What it reads. TGLF and NEO: every run folder of every PORTALS evaluation still on disk
       (Initialization/initialization_simple_relax/portals_sr_ev_*/ and Execution/Evaluation.*/,
       transport_simulation_folder/{base_tglf, base_neo, turb_drives_*}), with MITIM's own readers and
       the same harvest_recorder a live run uses, so fields and input hashes are identical. EPED: every
       full-EPED output_run1.nc (each EPED beat, plus the eped_initializer creator as maestro_beat 0);
       when pruning removed eped.input.1 / eped.config1 they are rebuilt from the .nc.
    2. What it cannot give back. Pruned runs keep no PORTALS run folders (prune_level >= 2, or the old
       keep_all_files: false): their TGLF/NEO evaluations are gone, only EPED survives. Also lost: EPED
       retries overwritten by the final attempt, the machine/modules of each code, and the MITIM
       version of the run. The runs table marks recovered runs with `recovered_by`.
    3. Dedup. Each run keeps a stable id (the live one if the run staged anything, else a hash of its
       folder path), and every (run, hash) already in the file is skipped: running the harvester twice
       appends nothing, and a run that DID harvest live adds nothing that is already there.
    4. Big scans: pass the scan's parent folder (runs are found up to 3 levels down); records are
       pushed in batches of --batch runs; --dry-run stages in a temporary folder and only reports.

This script runs the harvester on the chain of maestro_02_harvest.py (which harvested LIVE), into a
separate file, and checks that the recovered TGLF/NEO/EPED records are the live ones. Pass any other
MAESTRO folder as the first argument to harvest that instead.
"""

import sys
from pathlib import Path
from mitim_tools import __mitimroot__
from mitim_tools.harvest_tools import HARVESTtools, HARVESTrecover

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else __mitimroot__ / "tests" / "scratch" / "capability_maestro_harvest"
live_file = __mitimroot__ / "tests" / "scratch" / "capability_harvest_maestro.nc"          # written by maestro_02_harvest.py
recovered_file = __mitimroot__ / "tests" / "scratch" / "capability_harvester_recovered.nc"  # this example's own file

if not folder.exists():
    raise SystemExit(f"Run maestro_02_harvest.py first (or pass a MAESTRO folder): {folder} does not exist")
if recovered_file.exists():
    recovered_file.unlink()

# ---------------------------------------------------------------------------------------------------------------------
# 1. What is on disk, and what would be recorded (same as `mitim_harvester <folder> <file> --dry-run`)
# ---------------------------------------------------------------------------------------------------------------------

h = HARVESTrecover.harvester(folder, staging_folder=None)
print(f"\nRun {folder} (run id {h.run_id}):")
for beat, portals in h.portals_folders():
    tsfs = h.transport_folders(portals)
    print(f"   Beat_{beat} PORTALS: {len(tsfs)} evaluation(s) with transport folders on disk "
          f"({sum(len(h.code_folders(t)) for t in tsfs)} TGLF/NEO run folders)")
for beat, eped_folder in h.eped_folders():
    print(f"   maestro_beat {beat} EPED: {eped_folder.relative_to(folder)}")

HARVESTrecover.harvest_runs([folder], recovered_file, dry_run=True)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Harvest for real (same as `mitim_harvester <folder> <file>`), then once more to see the dedup
# ---------------------------------------------------------------------------------------------------------------------

appended = HARVESTrecover.harvest_runs([folder], recovered_file)
again = HARVESTrecover.harvest_runs([folder], recovered_file)
print(f"\nFirst pass appended {appended}; second pass appended {again} (dedup against the file)")

db = HARVESTtools.harvest_database(recovered_file)
print("\nRecovered database summary:")
print(db.summary().to_string(index=False))
print("\nRuns table (recovered_by marks records rebuilt from disk):")
print(db.runs()[["run", "code", "maestro_beat", "recovered_by"]].to_string(index=False))
for code in db.codes():
    df = db.load(code, with_run_info=False)
    print(f"\n{code}: {len(df)} records per maestro_beat {df.groupby('maestro_beat').size().to_dict()}, "
          f"{sum(c.startswith('in_') for c in df.columns)} inputs -> {[c for c in df.columns if c.startswith('out_')]}")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Against the live records (only for the maestro_02 chain, which harvested live)
# ---------------------------------------------------------------------------------------------------------------------

if live_file.exists() and len(sys.argv) == 1:
    live = HARVESTtools.harvest_database(live_file)
    for code in [c for c in db.codes() if c in live.codes()]:
        rec = db.load(code, with_run_info=False).set_index("hash")
        liv = live.load(code, with_run_info=False).set_index("hash")
        common = rec.index.intersection(liv.index)
        outs = [c for c in rec.columns if c.startswith("out_") and c in liv.columns and rec[c].dtype != object]
        same = all(((rec.loc[common, c] - liv.loc[common, c]).abs() <= 1e-12 * liv.loc[common, c].abs().clip(lower=1)).all() for c in outs)
        print(f"\n{code}: {len(common)} of {len(liv)} live records recovered with the same input hash; outputs identical: {same}")
        print(f"   live-only records (not on disk any more, e.g. EPED retries, pruned folders): {len(liv.index.difference(rec.index))}")
