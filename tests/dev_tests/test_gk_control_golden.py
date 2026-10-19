"""
test_gk_control_golden.py
=========================
Golden snapshot of everything the gyrokinetic run-control machinery writes or declares: the bash
of every CGYRO radial call (launch shape x watchdog mode x cost guard), the job scripts built from
it (bash loop, standard SLURM, SLURM array with its per-folder rescue bodies), the in-allocation
scheduler's call script, the requeue trim, the remote probe, the staged cost-guard file, the GX
call, and the completion / rescue / retrieval lists of the CGYRO class.

The reference texts live in tests/dev_tests/data/gk_control_golden/, one file per entry, generated
from the code before the run-control code was gathered into simulation_tools/gk_control. Moving
code must leave every byte unchanged; a deliberate change regenerates the files and shows up as a
reviewable diff:

    python tests/dev_tests/test_gk_control_golden.py              # compare
    python tests/dev_tests/test_gk_control_golden.py --regenerate # rewrite the reference files
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools import CONFIGread, SLURMtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.physics import GXtools
from mitim_tools.simulation_tools.utils import SCHEDULERtools

GOLDEN = Path(__file__).resolve().parent / "data" / "gk_control_golden"

EXEC = "/scratch/test"
FOLDERS = ["base_cgyro/rho_0.5500", "base_cgyro/rho_0.6712"]
HOSTS = ["node01", "node02"]

SHAPES = ("plain", "numa", "bash", "srun_wrap", "cpu")
MODES = {
    "manual": (None, None),
    "budget": (None, {"strategy": "wall_budget", "minutes_per_call": 45, "min_time": 120}),
    "stop": ("stop", {"strategy": "extra_points", "min_time": 300}),
}
GUARD = {"enabled": True, "targets_GB": {0.55: {"QeGB": 3.0, "QiGB": 4.0}, 0.6712: {"QeGB": 27.3, "QiGB": 14.4}}}


# ---------------------------------------------------------------------------
# Deterministic machine, MPI layout and allocation
# ---------------------------------------------------------------------------

def fake_machine(shape):
    if shape == "cpu":
        return {"machine": "fake_cpu", "gpus_per_node": 0, "cores_per_node": 128}
    return {"machine": "fake", "gpus_per_node": 4, "srun_wrap_calls": shape == "srun_wrap"}


def fake_resolved(shape):
    if shape == "plain":
        return types.SimpleNamespace(submission_type="slurm_array", concurrency=2, mpi={"n": 4, "nomp": 8})
    if shape == "cpu":
        return types.SimpleNamespace(submission_type="slurm_standard", concurrency=2, mpi={"n": 64, "nomp": 2})
    if shape == "numa":
        return types.SimpleNamespace(submission_type="slurm_array", concurrency=2,
                                     mpi={"n": 4, "nomp": 8, "numa": 4, "mpinuma": 1, "nodes": 1})
    return types.SimpleNamespace(submission_type="bash", concurrency=2,
                                 mpi={"n": 4, "nomp": 8, "numa": 4, "mpinuma": 1, "nodes": 1})


@contextlib.contextmanager
def patched(shape):
    hosts = HOSTS if shape in ("bash", "srun_wrap") else []
    keep = (CONFIGread.machineSettings, SLURMtools.resolve, SIMtools.slurm_allocation_hostnames,
            os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    CONFIGread.machineSettings = lambda *a, **k: fake_machine(shape)
    SLURMtools.resolve = lambda *a, **k: fake_resolved(shape)
    SIMtools.slurm_allocation_hostnames = lambda: list(hosts)
    os.environ["SLURM_JOB_CPUS_PER_NODE"] = "128"
    try:
        yield hosts
    finally:
        CONFIGread.machineSettings, SLURMtools.resolve, SIMtools.slurm_allocation_hostnames = keep[:3]
        if keep[3] is None:
            os.environ.pop("SLURM_JOB_CPUS_PER_NODE", None)
        else:
            os.environ["SLURM_JOB_CPUS_PER_NODE"] = keep[3]


def new_cgyro(load_balance=None, guard=False, allocation=None):
    with contextlib.redirect_stdout(io.StringIO()):
        cgyro = CGYROtools.CGYRO()
        cgyro._load_balance = load_balance
        cgyro._allocation = allocation
        cgyro._cost_guard = CGYROtools.CostGuard(GUARD) if guard else None
    return cgyro


def normalize(text):
    return text.replace(str(__mitimroot__), "<MITIMROOT>")


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------

def snapshot():
    '''{entry name: text}, every text normalized so it does not depend on where the repo lives.'''
    out = {}
    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        for shape in SHAPES:
            for mode_name, (mode, load_balance) in MODES.items():
                for guard in (False, True):
                    tag = f"{shape}__{mode_name}__guard{int(guard)}"
                    allocation = {"omp_threads_cpu": 2} if shape == "cpu" else None
                    with patched(shape) as hosts:
                        cgyro = new_cgyro(load_balance, guard, allocation)
                        code_call = cgyro.run_specifications["code_call"]
                        out[f"code_call__{tag}"] = code_call(
                            folder=FOLDERS[1], p=EXEC, n=4, additional_command="&& echo done", watchdog=mode)

                        if mode_name == "manual" or guard:
                            continue
                        call = lambda folder, n, p, **kw: code_call(folder=folder, n=n, p=p, watchdog=mode, **kw)
                        if shape in ("bash", "srun_wrap"):
                            script = SIMtools.BashScript(FOLDERS, call, 4, EXEC, max_parallel=2, hosts=hosts)
                            out[f"bash_script__{tag}"] = script.command
                            out[f"bash_bodies__{tag}"] = json.dumps(script.folder_bodies(), indent=1, sort_keys=True)
                            out[f"scheduler_call__{tag}"] = scheduler_call_script(script, hosts)
                        elif shape == "cpu":
                            out[f"standard_script__{tag}"] = SIMtools.StandardSlurmScript(FOLDERS, call, 4, EXEC).command
                        else:
                            script = SIMtools.ArraySlurmScript(FOLDERS, call, 4, EXEC)
                            out[f"array_script__{tag}"] = script.command
                            out[f"array_rescue__{tag}"] = json.dumps(
                                {"per_folder_commands": script.per_folder_commands,
                                 "array_index_by_folder": script.array_index_by_folder}, indent=1, sort_keys=True)

        body = CGYROtools.CgyroLaunchBody.__new__(CGYROtools.CgyroLaunchBody)
        body.p, body.folder = EXEC, FOLDERS[1]
        marker_cmd, cleanup_cmd = body.markers()
        out["launch_markers"] = marker_cmd + "\n----\n" + cleanup_cmd
        out["exit_verdict"] = body.exit_verdict()
        out["body_keeping_exit_status"] = CGYROtools.body_keeping_exit_status("PRE", "MAIN", "CLEAN", "VERDICT")
        out["requeue_trim"] = CGYROtools._ShellTemplate(CGYROtools._REQUEUE_TRIM_BASH).substitute(run_dir=f"{EXEC}/{FOLDERS[1]}")
        out["probe"] = CGYROtools.CgyroProbe(types.SimpleNamespace(folderExecution=EXEC)).script(FOLDERS)
        out["watchdog_template"] = CGYROtools.CGYRO._WALL_BUDGET_WATCHDOG
        out["cost_guard_stage"] = cost_guard_stage()

        cgyro = new_cgyro()
        out["cgyro_run_specifications"] = run_specifications(cgyro)
        out["cgyro_output_files"] = json.dumps(cgyro.output_files_simulation, indent=1, sort_keys=True)
        out["cgyro_submission_metadata_filename"] = CGYROtools.CGYRO._submission_metadata_filename

        gx = GXtools.GX()
        out["gx_code_call"] = gx.run_specifications["code_call"](folder=FOLDERS[1], n=2, p=EXEC)
        out["gx_submission_metadata_filename"] = GXtools.GX._submission_metadata_filename
    return {k: normalize(v) for k, v in out.items()}


def scheduler_call_script(script, hosts):
    '''The mitim_call_<rel>.sh the in-allocation scheduler writes for the first main call.'''
    sched = SCHEDULERtools.InAllocationScheduler(script.folder_bodies(), hosts, 2)
    rel, body = next(iter(script.folder_bodies().items()))
    keep = SCHEDULERtools.subprocess.Popen
    SCHEDULERtools.subprocess.Popen = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            sched._launch(Path(tmp), "module load cgyro", None, rel, body, 1)
        finally:
            SCHEDULERtools.subprocess.Popen = keep
        text = (Path(tmp) / f"mitim_call_{rel.replace('/', '_')}.sh").read_text()
        return text.replace(tmp, "<CWD>") + f"\n----\naccepted_marker={sched.accepted_marker}\nstop_file={sched.stop_file}\n"


def cost_guard_stage():
    guard = CGYROtools.CostGuard({**GUARD, "flux_ratio": 6, "std_floor": 0.25, "seconds_per_acs": 100})
    with tempfile.TemporaryDirectory() as tmp:
        files = guard.stage(tmp, {0.55: [("/x/bin.cgyro.restart", "bin.cgyro.restart")]})
        listing = {f"{rho:.4f}": [(str(src).replace(tmp, "<TMP>"), dst) for src, dst in entries]
                   for rho, entries in sorted(files.items())}
        contents = {p.name: p.read_text() for p in sorted(Path(tmp).iterdir())}
    return json.dumps({"files": listing, "contents": contents, "thresholds": guard.thresholds,
                       "std_floor": guard.std_floor, "FILE": guard.FILE, "SCRIPT": guard.SCRIPT,
                       "TAG_TOKEN": guard.TAG_TOKEN}, indent=1, sort_keys=True)


def run_specifications(sim):
    '''The run_specifications of a simulation object, callables reduced to "<callable>".'''
    def clean(v):
        if callable(v) and not isinstance(v, type):
            return "<callable>"
        if isinstance(v, type):
            return f"<class {v.__name__}>"
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(sim.run_specifications), indent=1, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Compare / regenerate
# ---------------------------------------------------------------------------

def regenerate():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for old in GOLDEN.glob("*.txt"):
        old.unlink()
    for name, text in snapshot().items():
        (GOLDEN / f"{name}.txt").write_text(text)


def test_golden_snapshot():
    current = snapshot()
    stored = {p.stem: p.read_text() for p in GOLDEN.glob("*.txt")}
    assert stored, f"no reference files in {GOLDEN}; run with --regenerate"
    assert set(current) == set(stored), (
        f"entries differ: new {sorted(set(current) - set(stored))}, gone {sorted(set(stored) - set(current))}")
    changed = [k for k in sorted(current) if current[k] != stored[k]]
    assert not changed, f"{len(changed)} entries changed, e.g. {changed[:5]}"
    print(f"PASS test_golden_snapshot ({len(current)} entries byte-identical)")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
        print(f"regenerated {len(list(GOLDEN.glob('*.txt')))} files in {GOLDEN}")
    else:
        test_golden_snapshot()
