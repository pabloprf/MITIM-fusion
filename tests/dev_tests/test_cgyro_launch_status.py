"""
test_cgyro_launch_status.py
===========================
The CGYRO launch body, the per-task probe and the auto-resubmit ledger, as small units.

The launch body and the enforced input values are compared against the implementation of the
commit BEFORE the refactor (read out of git, so nothing has to be kept in sync by hand): the bash
MITIM ships to the cluster for every launch shape and every watchdog mode must be byte-identical,
and the three _enforce_* must write the same values. The rest is pure-python: a probe line becomes
a RadiusStatus, a ledger entry round-trips through the JSON shape stored in cgyro_submission.json.

Run as:

    python tests/dev_tests/test_cgyro_launch_status.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
mitim_root = repo_root / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.gacode_tools.utils import GACODEdefaults
from mitim_tools.misc_tools import CONFIGread, SLURMtools
from mitim_tools.simulation_tools import SIMtools

# The commit this refactor started from; its CGYROtools is the reference for the bash and the
# enforced values. The old-vs-new tests are skipped when the object is not in the repo.
REFERENCE_COMMIT = "20564ef1"

FOLDER = "base_cgyro/rho_0.6712"
EXEC = "/scratch/test"


# ---------------------------------------------------------------------------
# Fakes and the reference module
# ---------------------------------------------------------------------------


def reference_module():
    '''CGYROtools as of REFERENCE_COMMIT, imported under its own name. None when unavailable.'''
    try:
        text = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{REFERENCE_COMMIT}:src/mitim_tools/gacode_tools/CGYROtools.py"],
            capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    path = Path(tempfile.mkdtemp()) / "old_cgyrotools.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("old_cgyrotools", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def fake_machine(srun_wrap=False):
    return {"machine": "fake", "gpus_per_node": 4, "srun_wrap_calls": srun_wrap}


def fake_resolved(shape):
    '''The SLURMtools.resolve result for each launch shape the builder distinguishes.'''
    if shape == "plain":
        return types.SimpleNamespace(submission_type="slurm_array", mpi={"n": 4, "nomp": 8})
    if shape == "numa":
        return types.SimpleNamespace(submission_type="slurm_array",
                                     mpi={"n": 4, "nomp": 8, "numa": 4, "mpinuma": 1, "nodes": 1})
    return types.SimpleNamespace(submission_type="bash",
                                 mpi={"n": 4, "nomp": 8, "numa": 4, "mpinuma": 1, "nodes": 1})


@contextlib.contextmanager
def patched(shape, hosts):
    '''Machine block, MPI resolution and allocation hostnames, for both modules at once.'''
    keep = (CONFIGread.machineSettings, SLURMtools.resolve, SIMtools.slurm_allocation_hostnames,
            os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    CONFIGread.machineSettings = lambda *a, **k: fake_machine(srun_wrap=(shape == "srun_wrap"))
    SLURMtools.resolve = lambda *a, **k: fake_resolved(shape)
    SIMtools.slurm_allocation_hostnames = lambda: list(hosts)
    os.environ["SLURM_JOB_CPUS_PER_NODE"] = "128"
    try:
        yield
    finally:
        CONFIGread.machineSettings, SLURMtools.resolve, SIMtools.slurm_allocation_hostnames = keep[:3]
        if keep[3] is None:
            os.environ.pop("SLURM_JOB_CPUS_PER_NODE", None)
        else:
            os.environ["SLURM_JOB_CPUS_PER_NODE"] = keep[3]


def build_body(module, shape, hosts, mode, load_balance, additional_command=""):
    with patched(shape, hosts), contextlib.redirect_stdout(io.StringIO()):
        cgyro = module.CGYRO()
        cgyro._load_balance = load_balance
        return cgyro.run_specifications["code_call"](
            folder=FOLDER, p=EXEC, n=4, additional_command=additional_command, watchdog=mode)


SHAPES = (("plain", []), ("numa", []), ("bash", ["node01", "node02"]), ("srun_wrap", ["node01", "node02"]))
MODES = ((None, None),
         (None, {"strategy": "wall_budget", "minutes_per_call": 45, "min_time": 120}),
         ("stop", {"strategy": "extra_points", "min_time": 300}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_launch_bodies_are_byte_identical():
    '''Every launch shape x watchdog mode produces exactly the bash of the reference commit.'''
    old = reference_module()
    if old is None:
        print(f"SKIP: {REFERENCE_COMMIT} not available in this repo")
        return
    n = 0
    for shape, hosts in SHAPES:
        for mode, load_balance in MODES:
            new_body = build_body(CGYROtools, shape, hosts, mode, load_balance, additional_command="&& echo done")
            old_body = build_body(old, shape, hosts, mode, load_balance, additional_command="&& echo done")
            assert new_body == old_body, (
                f"{shape}/{mode}:\n--- reference ---\n{old_body}\n--- now ---\n{new_body}")
            n += 1
    print(f"PASS: {n} launch bodies (shape x watchdog mode) byte-identical to {REFERENCE_COMMIT}")


def test_watchdog_modes_render_their_switches():
    '''The three knobs the watchdog takes reach the bash, and only they differ between modes.'''
    manual = CGYROtools.Watchdog(f"{EXEC}/{FOLDER}").wrap("cgyro -e .")
    budget = CGYROtools.Watchdog.from_load_balance(
        f"{EXEC}/{FOLDER}", {"strategy": "wall_budget", "minutes_per_call": 45, "min_time": 120}).wrap("cgyro -e .")
    stop = CGYROtools.Watchdog(f"{EXEC}/{FOLDER}", mode=CGYROtools.Watchdog.STOP, min_time=300).wrap("cgyro -e .")

    assert manual.splitlines()[0].endswith("_lb_budget=0; _lb_min=0; _lb_discard=0"), manual.splitlines()[0]
    assert budget.splitlines()[0].endswith("_lb_budget=2700; _lb_min=120; _lb_discard=0"), budget.splitlines()[0]
    assert stop.splitlines()[0].endswith("_lb_budget=0; _lb_min=300; _lb_discard=1"), stop.splitlines()[0]
    assert manual.splitlines()[1:] == stop.splitlines()[1:], "modes must differ only in the first line"
    assert "cgyro -e ." in manual and "@{" not in manual, manual

    try:
        CGYROtools.Watchdog(f"{EXEC}/{FOLDER}", mode=CGYROtools.Watchdog.BUDGET)
    except ValueError as e:
        assert "minutes_per_call" in str(e), str(e)
    else:
        raise AssertionError("wall_budget without minutes_per_call was accepted")
    print("PASS: watchdog modes render their switches, wall_budget without minutes_per_call raises")


def test_radius_status_from_probe_line():
    running = CGYROtools.RadiusStatus.from_probe_line(
        f"{FOLDER}|RUNNING|5.000|120|3600|30|-|0")
    assert running.state == CGYROtools.TaskState.RUNNING, running
    assert running.stale_threshold == 300 and running.seconds_per_step == 5.0, running
    assert "running — 120 step(s)" in running.describe()[0], running.describe()

    stalled = CGYROtools.RadiusStatus.from_probe_line(
        f"{FOLDER}|RUNNING|200.000|12|9000|4000|-|0")
    assert stalled.state == CGYROtools.TaskState.STALLED, stalled
    assert stalled.stale_threshold == 600, stalled          # 3*avg capped at 600 s
    assert stalled.describe()[1] == 'w', stalled.describe()
    assert stalled.to_row()["effective"] == "STALLED", stalled.to_row()

    timed_out = CGYROtools.RadiusStatus.from_probe_line(
        f"{FOLDER}|RUNNING|5.000|12|9000|10|-|0", slurm_state="TIMEOUT", job_terminal=True)
    assert timed_out.state == CGYROtools.TaskState.TIMED_OUT, timed_out

    exited = CGYROtools.RadiusStatus.from_probe_line(f"{FOLDER}|RUNNING|5.000|900|9000|4000|-|1")
    assert exited.state == CGYROtools.TaskState.FINISHED, exited

    init = CGYROtools.RadiusStatus.from_probe_line(f"{FOLDER}|INITIALIZED|NA|0|4000|4000|-|0")
    assert init.state == CGYROtools.TaskState.STALLED_INIT and init.stale_threshold == 180, init

    pending = CGYROtools.RadiusStatus.from_probe_line(f"{FOLDER}|NOT_STARTED|NA|0|0|0|-|0")
    assert pending.state == CGYROtools.TaskState.NOT_STARTED, pending
    assert "pending" in pending.describe()[0], pending.describe()

    # a phase token is informational: the state stays what the files say, the token is appended
    phase = CGYROtools.RadiusStatus.from_probe_line(f"{FOLDER}|RUNNING|5.000|120|3600|30|100|0")
    assert phase.state == CGYROtools.TaskState.RUNNING and "[out.cgyro.tag=100]" in phase.describe()[0], phase

    assert CGYROtools.RadiusStatus.from_probe_line("bash: stat: command not found") is None
    assert CGYROtools.RadiusStatus.from_probe_line("") is None
    print("PASS: probe lines parse, reclassify and print; a malformed line is dropped")


def test_ledger_entry_is_closed_and_json():
    entry = CGYROtools.LedgerEntry.from_json(None)
    assert entry.to_json() == {"n_attempts": 0, "child_jobids": [], "last_action_at": None, "status": "active"}
    assert not entry.is_closed(1)

    stored = {"n_attempts": 1, "child_jobids": ["67890"], "last_action_at": "2026-05-07T00:00:00Z", "status": "active"}
    entry = CGYROtools.LedgerEntry.from_json(stored)
    assert entry.to_json() == stored, entry.to_json()
    assert entry.is_closed(1) and not entry.is_closed(2), entry     # the cap closes it

    for status in ("EXHAUSTED", "TERMINAL_NO_RESCUE:COMPLETED"):
        closed = CGYROtools.LedgerEntry.from_json({"status": status})
        assert closed.is_closed(5), status
    assert CGYROtools.LedgerEntry.from_json({"status": "TERMINAL_NO_RESCUE:NODE_FAIL"}).terminal_no_rescue
    assert not CGYROtools.LedgerEntry.from_json({"status": "active"}).terminal_no_rescue
    print("PASS: ledger entry closes on EXHAUSTED / TERMINAL_NO_RESCUE / cap, and round-trips")


def test_enforced_values_match_reference():
    '''The three _enforce_* write the same values as the reference commit, for the same inputs.'''
    old = reference_module()
    if old is None:
        print(f"SKIP: {REFERENCE_COMMIT} not available in this repo")
        return

    cases = [
        ({}, "Nonlinear_high", {"resources_per_call": 4}),
        ({}, "Nonlinear_high", {"resources_per_call": 8}),
        ({}, "Linear", {"resources_per_call": 4}),
        ({"N_RADIAL": [128, 256], "KY": [0.05, 0.1]}, "Nonlinear_high", {"resources_per_call": 8}),
        ({"MAX_TIME": [400.0, 800.0]}, "Nonlinear_high", {"resources_per_call": 4}),
        ({"DELTA_T": 0.006, "MAX_TIME": 250.0}, "Nonlinear_high", {"resources_per_call": 4}),
        ({"TOROIDALS_PER_PROC": 5}, "Nonlinear_high", {"resources_per_call": 4}),
        ({"PRINT_STEP": 50, "RESTART_STEP": 7}, "Nonlinear_high", {"resources_per_call": 4}),
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        new_sim, old_sim = CGYROtools.CGYRO(), old.CGYRO()
        for extraOptions, code_settings, allocation in cases:
            results = []
            for sim in (new_sim, old_sim):
                options = dict(extraOptions)
                options = sim._enforce_toroidals_per_proc(options, allocation, code_settings=code_settings)
                options = sim._enforce_print_step(options, code_settings=code_settings)
                options = sim._enforce_restart_step(options, code_settings=code_settings)
                results.append(options)
            assert results[0] == results[1], (extraOptions, code_settings, allocation, results)

    # the rescue hook re-derives RESTART_STEP the same way _enforce_restart_step sizes it
    text = "MAX_TIME=1200\nDELTA_T=0.04\nPRINT_STEP=25\nRESTART_STEP=1200\n"
    assert CGYROtools.CGYRO._restart_step_after_trim(text, 100.0) == old.CGYRO._restart_step_after_trim(text, 100.0)
    print(f"PASS: {len(cases)} enforced-value cases identical to {REFERENCE_COMMIT}")


def test_probe_script_is_portable_bash():
    '''The probe reads mtimes on GNU and BSD remotes and never compares a grep count to a literal.'''
    job = types.SimpleNamespace(folderExecution=EXEC)
    script = CGYROtools.CgyroProbe(job).script([FOLDER, "base_cgyro/rho_0.8334"])
    assert "stat -c %Y" in script and "stat -f %m" in script, script
    assert "grep -c" not in script and 'grep -q "^EXIT"' in script, script
    assert "@{" not in script and f'"{FOLDER}"' in script, script
    with tempfile.TemporaryDirectory() as d:
        rc = subprocess.run(["bash", "-n", "-c", script], cwd=d, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()
    print("PASS: probe script parses as bash and uses the portable mtime/EXIT checks")


def test_shared_node_calls_get_their_own_gpus():
    '''Several 1-GPU calls on a 4-GPU node: call k takes host k // 4, and each srun step owns its
    GPU exclusively (--exact, no --overlap: overlapping steps were all handed the node's first GPU,
    and gacode's wrapper re-pins CUDA_VISIBLE_DEVICES to the local rank, so only the step's device
    cgroup can make the choice stick).'''
    import subprocess
    body = CGYROtools.CgyroLaunchBody.__new__(CGYROtools.CgyroLaunchBody)
    body.machine = {"machine": "local", "gpus_per_node": 4, "srun_wrap_calls": True}
    body.mpi = {"n": 1, "nomp": 16, "numa": 1, "mpinuma": 1}
    body.hosts = ["node01", "node02"]; body.nodes = 1; body.bash_mode = True; body.srun_wrap = True
    body.folder = "base_cgyro/rho_0.4808"; body.p = "/scratch/x"; body.additional_command = ""; body.cpus_per_node = 128
    txt = body.launch()
    assert "-c16 --gpus-per-node=1 --cpu-bind=none ${_sel:+-w $_sel} --exact" in txt and "--overlap" not in txt and "CUDA_VISIBLE" not in txt, txt

    def host(k, sel):
        out = subprocess.run(["bash", "-c", f"MITIM_HOSTS=(node01 node02); MITIM_CALL={k}\n{sel}echo $_sel"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    sel = body.host_selection()
    assert [host(k, sel) for k in (1, 4, 5, 9)] == ["node01", "node01", "node02", "node01"]
    body.mpi["numa"] = 2; sel = body.host_selection()
    assert [host(k, sel) for k in (1, 2, 3)] == ["node01", "node01", "node02"]
    body.mpi["numa"] = 4
    assert "--overlap" in body.launch() and "-c32 --gpus-per-node=4" in body.launch()
    print("PASS: shared-node calls own their GPU exclusively per step; whole-node shape unchanged")


if __name__ == "__main__":
    test_watchdog_modes_render_their_switches()
    test_radius_status_from_probe_line()
    test_ledger_entry_is_closed_and_json()
    test_probe_script_is_portable_bash()
    test_launch_bodies_are_byte_identical()
    test_enforced_values_match_reference()
    test_shared_node_calls_get_their_own_gpus()
    print("\nALL PASS")
