"""
test_check_waits_for_children.py
================================
SIMtools.mitim_simulation.check() derives "job finished" from squeue on the PARENT jobid
only. The auto-resubmit orchestrator (CGYROtools._cgyro_handle_stalled_tasks) rescues a
stalled radius with an INDEPENDENT job whose id lives in the resubmit ledger, so the parent
array can drain while that radius is still integrating — check() then returned and fetch()
pulled the rescued radius half-done.

Covered here:
- check() keeps polling while a ledger child is still in the queue, and exits once it is gone.
- a liveness probe that cannot reach the remote counts as ALIVE in the poll loop (a VPN flap
  must not end the poll), with a warning.
- _local_results_complete() also requires the code's completion marker, not just the files.
- fetch() runs the same completion gate as the 'normal' run path and raises on a truncated radius.

Run as:

    python tests/dev_tests/test_check_waits_for_children.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools import SIMtools

CHILD = "67890"
RHO = 0.6712
MARKER, ALT = ("out.cgyro.info", "EXIT"), "mitim_budget.tag"
FILE = "out.cgyro.gbflux"


class FakeJob:
    """mitim_job stand-in: parent already gone, child liveness driven by `child_polls_alive`."""

    def __init__(self, child_polls_alive=0, probe_raises=False):
        self.jobid = "12345"
        self.launchSlurm = True
        self.status = 2
        self.infoSLURM = {"STATE": "NOT FOUND"}
        self.child_polls_alive = child_polls_alive
        self.probe_raises = probe_raises
        self.n_squeue_child = 0

    def check(self, file_output=None):
        pass

    def connect(self):
        if self.probe_raises:
            raise RuntimeError("ssh boom")

    def close(self):
        pass

    def execute(self, cmd, printYN=False):
        assert cmd.startswith("squeue"), cmd
        self.n_squeue_child += 1
        if self.n_squeue_child <= self.child_polls_alive:
            return f"          {CHILD}    RUNNING\n".encode(), b""
        return b"", b""


def _sim(job, ledger=None):
    sim = types.SimpleNamespace(simulation_job=job, slurm_output="slurm_output.dat",
                                _resubmit_ledger=ledger or {})
    for name in ("check", "_child_jobids", "_live_child_jobids", "_any_child_job_alive"):
        setattr(sim, name, types.MethodType(getattr(SIMtools.mitim_simulation, name), sim))
    return sim


@contextlib.contextmanager
def _no_sleep():
    """check() sleeps every_n_minutes*60 between polls; count the naps instead of taking them."""
    naps = []
    original = SIMtools.time.sleep
    SIMtools.time.sleep = naps.append
    try:
        yield naps
    finally:
        SIMtools.time.sleep = original


LEDGER = {"base_cgyro/rho_0.6712": {"n_attempts": 1, "child_jobids": [CHILD]}}


def test_keeps_polling_while_child_alive():
    job = FakeJob(child_polls_alive=2)
    sim = _sim(job, LEDGER)
    log = io.StringIO()
    with _no_sleep() as naps, contextlib.redirect_stdout(log):
        sim.check(every_n_minutes=1)
    out = log.getvalue()

    assert len(naps) == 2, f"expected 2 waits while the child was alive, got {naps}"
    assert job.n_squeue_child == 3, job.n_squeue_child
    assert f"rescue child jobid(s) ['{CHILD}'] are still in it" in out, out
    assert "Job considered finished" in out, out
    print("PASS: parent gone + live rescue child -> check() keeps polling until the child leaves")


def test_no_children_exits_immediately():
    job = FakeJob()
    sim = _sim(job, {})
    log = io.StringIO()
    with _no_sleep() as naps, contextlib.redirect_stdout(log):
        sim.check(every_n_minutes=1)

    assert naps == [], naps
    assert job.n_squeue_child == 0, "no ledger entries -> no squeue for children"
    assert "Job considered finished" in log.getvalue()
    print("PASS: parent gone + no rescue children -> check() exits on the first poll")


def test_unreachable_probe_counts_as_alive():
    job = FakeJob(probe_raises=True)
    sim = _sim(job, LEDGER)
    log = io.StringIO()

    # The probe never recovers, so stop it after the first nap
    with _no_sleep() as naps, contextlib.redirect_stdout(log):
        def stop(_):
            job.probe_raises = False
            naps.append(1)
        SIMtools.time.sleep = stop
        sim.check(every_n_minutes=1)
    out = log.getvalue()

    assert naps == [1], naps
    assert out.count("squeue failed") == 1, out
    assert "treating rescue children as alive" in out, out
    assert f"['{CHILD}'] are still in it" in out, out
    print("PASS: unreachable liveness probe -> children treated as alive, one warning, poll continues")


def test_reattach_probe_keeps_not_alive_on_failure():
    '''The re-attach decision tree (transport_cgyro) keeps the old "no signal" semantics.'''
    sim = _sim(FakeJob(probe_raises=True), LEDGER)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        alive = sim._any_child_job_alive()
    assert alive is False
    assert "treating rescue children as not alive" in log.getvalue(), log.getvalue()
    print("PASS: default (re-attach) semantics unchanged — unreachable probe means not alive")


# ---------------------------------------------------------------------------


def _local_sim(info_text=None, budget_tag=False, with_file=True):
    d = Path(tempfile.mkdtemp())
    if with_file:
        (d / f"{FILE}_{RHO:.4f}").write_text("fluxes\n")
    if info_text is not None:
        (d / f"{MARKER[0]}_{RHO:.4f}").write_text(info_text)
    if budget_tag:
        (d / f"{ALT}_{RHO:.4f}").write_text("t=300\n")
    sim = types.SimpleNamespace(
        kwargs_organize={"code_executor": {"base_cgyro": {RHO: {"folder": d}}}, "filesToRetrieve": [FILE, MARKER[0]]},
        run_specifications={"completion_marker": MARKER, "completion_alt_file": ALT},
    )
    sim._local_results_complete = types.MethodType(SIMtools.mitim_simulation._local_results_complete, sim)
    return d, sim


def test_local_results_complete_requires_marker():
    cases = {
        "truncated, no budget tag": ("[t: 5.260E+02]\n", False, False),
        "EXIT present": ("[t: 5.260E+02]\nEXIT: (CGYRO) Normal\n", False, True),
        "budget tag instead": ("[t: 5.260E+02]\n", True, True),
    }
    for label, (info, tag, expected) in cases.items():
        d, sim = _local_sim(info, budget_tag=tag)
        try:
            got = sim._local_results_complete()
            assert got is expected, f"{label}: expected {expected}, got {got}"
        finally:
            shutil.rmtree(d, ignore_errors=True)

    d, sim = _local_sim(None, with_file=False)
    try:
        assert sim._local_results_complete() is False, "missing files must still fail"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: _local_results_complete requires the EXIT marker (or mitim_budget.tag), not just files")


class FetchJob(FakeJob):
    def retrieve(self):
        pass


def test_fetch_raises_on_unfinished_radius():
    d = Path(tempfile.mkdtemp())
    try:
        results, tmpFolder = d / "base_cgyro", d / "tmp_cgyro"
        results.mkdir()
        rho_dir = tmpFolder / "base_cgyro" / f"rho_{RHO:.4f}"
        rho_dir.mkdir(parents=True)
        (rho_dir / MARKER[0]).write_text("INFO: (CGYRO) GPU-aware code triggered.\n[t: 5.260E+02]\n")

        sim = types.SimpleNamespace(
            simulation_job=FetchJob(), FolderGACODE=d,
            run_specifications={"code": "cgyro", "completion_marker": MARKER, "completion_alt_file": ALT},
            kwargs_organize={"code_executor": {"base_cgyro": {RHO: {"folder": results}}},
                             "tmpFolder": tmpFolder, "filesToRetrieve": [MARKER[0]]},
        )
        for name in ("fetch", "_organize_results", "_verify_completion"):
            setattr(sim, name, types.MethodType(getattr(SIMtools.mitim_simulation, name), sim))

        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                sim.fetch()
        except RuntimeError as e:
            assert "returned without finishing at 1 radius" in str(e), str(e)
        else:
            raise AssertionError("fetch() must raise on a radius without the completion marker")

        assert (results / f"{MARKER[0]}_{RHO:.4f}").exists(), "files are still organized before the gate fires"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: fetch() applies the completion gate and raises on a truncated radius")


if __name__ == "__main__":
    test_keeps_polling_while_child_alive()
    test_no_children_exits_immediately()
    test_unreachable_probe_counts_as_alive()
    test_reattach_probe_keeps_not_alive_on_failure()
    test_local_results_complete_requires_marker()
    test_fetch_raises_on_unfinished_radius()
    print("\nALL TESTS PASSED")
