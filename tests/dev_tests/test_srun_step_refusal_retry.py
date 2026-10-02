"""
test_srun_step_refusal_retry.py
===============================
What mitim_job.full_process does when an in-allocation bash dispatch comes back without its
outputs because SLURM refused to create the job steps ("srun: error: Unable to create step for
job N: Job/step already completing or completed"), as seen on Perlmutter for the NEO call of a
PORTALS-CGYRO chain link while the job still had hours left.

- The same dispatch is re-run after the waits in FARMINGtools.SRUN_STEP_REFUSAL_WAITS, with a
  warning per re-run, and those re-runs are not charged against attempts_execution.
- In a non-interactive session (no tty) the "Not all expected files received" prompt raises a
  MissingOutputsError that says what happened, instead of the bare InteractiveTerminalError;
  SIMtools._dispatch_blocking carries that cause into its final RuntimeError.
- In an interactive session the prompt is still asked, exactly as before.
- Inside an allocation (SLURM_JOB_ID set), each re-run and the final give-up append `scontrol show job` and
  `squeue -s` to mitim_slurm_snapshot.txt, so the refusal can be traced; nothing is written outside one.

Run as:

    python tests/dev_tests/test_srun_step_refusal_retry.py

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

from mitim_tools.misc_tools import CONFIGread, FARMINGtools, LOGtools
from mitim_tools.simulation_tools import SIMtools

FOLDER = "base_neo/rho_0.3316"
OUTPUT = "out.neo.transport_flux"
REFUSAL = (b"srun: warning: can't run 1 processes on 10 nodes, setting nnodes to 1\n"
           b"srun: error: Unable to create step for job 58888652: Job/step already completing or completed\n")


@contextlib.contextmanager
def _harness(tty):
    """Verbose 3 (so 'q' reaches the prompt whatever config_user.json says), no sleeps, chosen stdin."""
    slept = []
    saved = (FARMINGtools.time.sleep, CONFIGread.read_verbose_level, sys.stdin, LOGtools.query_yes_no)
    FARMINGtools.time.sleep = lambda seconds: slept.append(seconds)
    CONFIGread.read_verbose_level = lambda: 3
    asked = []
    if tty:
        # Interactive laptop session: the user answers 'y'
        LOGtools.query_yes_no = lambda question, extra="": asked.append(question) or True
    else:
        sys.stdin = io.StringIO()  # not a tty, as inside a batch job
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            yield slept, log, asked
    finally:
        FARMINGtools.time.sleep, CONFIGread.read_verbose_level, sys.stdin, LOGtools.query_yes_no = saved


def _job(fail_times):
    """Local in-place job whose execute() is refused `fail_times` times, then writes the output."""
    folder = Path(tempfile.mkdtemp())
    job = FARMINGtools.mitim_job(folder)
    job.machineSettings = {"machine": "local"}
    job.folderExecution = folder
    job.run_in_place = True
    job.spec = FARMINGtools.RetrievalSpec(folders=[FOLDER])
    (folder / FOLDER).mkdir(parents=True)
    calls = []

    def execute(command_str, **kwargs):
        calls.append(command_str)
        if len(calls) <= fail_times:
            return b"NEO b49339750\n", REFUSAL * 5
        (folder / FOLDER / OUTPUT).write_text("fluxes\n")
        return b"NEO b49339750\n", b""

    job.execute = execute
    return job, calls


def _run(job, attempts_execution=1):
    job.full_process("bash mitim_shell_executor.sh", attempts_execution=attempts_execution,
                     check_files_in_folder={FOLDER: [OUTPUT]})


def test_refusal_is_rerun_until_success():
    job, calls = _job(fail_times=2)
    with _harness(tty=False) as (slept, log, asked):
        _run(job)
    out = log.getvalue()
    assert len(calls) == 3, calls
    assert out.count("SLURM refused 5 job step(s)") == 2, out
    assert "step-refusal re-run 1/3" in out and "step-refusal re-run 2/3" in out, out
    assert slept == [60, 120], slept  # in-place retrieve() adds no wait of its own
    assert (job.folder_local / FOLDER / OUTPUT).exists()
    print("PASS test_refusal_is_rerun_until_success")


def test_refusal_reruns_do_not_spend_attempts():
    # attempts_execution=2 (CGYRO/TGLF): 3 refused runs, then the 4th succeeds
    job, calls = _job(fail_times=3)
    with _harness(tty=False) as (slept, log, asked):
        _run(job, attempts_execution=2)
    assert len(calls) == 4, calls
    print("PASS test_refusal_reruns_do_not_spend_attempts")


def test_persistent_refusal_in_batch_raises_a_clear_error():
    job, calls = _job(fail_times=99)
    with _harness(tty=False) as (slept, log, asked):
        try:
            _run(job)
            raise AssertionError("full_process should have raised")
        except FARMINGtools.MissingOutputsError as e:
            msg = str(e)
    assert len(calls) == 1 + len(FARMINGtools.SRUN_STEP_REFUSAL_WAITS), calls
    assert slept == list(FARMINGtools.SRUN_STEP_REFUSAL_WAITS), slept
    assert "SLURM refused 20 job step(s)" in msg and "3 re-run(s)" in msg, msg
    assert "mitim_farming.err" in msg and str(job.folderExecution) in msg, msg
    assert "Unable to create step" in (job.folder_local / "mitim_farming.err").read_text()
    # still an InteractiveTerminalError, so EPEDbeat/TRANSP/SIMtools handlers keep working
    assert issubclass(FARMINGtools.MissingOutputsError, LOGtools.InteractiveTerminalError)
    print("PASS test_persistent_refusal_in_batch_raises_a_clear_error")


def test_slurm_snapshot_inside_allocation():
    '''Each re-run and the final give-up log what slurmctld says about the job; nothing outside an allocation.'''
    import os
    bindir = Path(tempfile.mkdtemp())
    for cmd in ("scontrol", "squeue"):
        body = ("echo JobId=$4 JobState=RUNNING Reason=None RunTime=03:32:00 TimeLimit=07:45:00 EndTime=2026-10-02T09:59:12"
                if cmd == "scontrol" else "echo STEPID STATE TIME NODELIST; echo 58888652.batch RUNNING 3:32:00 nid001533")
        (bindir / cmd).write_text(f"#!/bin/sh\necho FAKE-{cmd} \"$@\"\n{body}\n")
        (bindir / cmd).chmod(0o755)
    saved_path, saved_id = os.environ.get("PATH", ""), os.environ.get("SLURM_JOB_ID")
    os.environ["PATH"] = f"{bindir}:{saved_path}"
    try:
        os.environ["SLURM_JOB_ID"] = "58888652"
        job, calls = _job(fail_times=99)
        with _harness(tty=False) as (slept, log, asked):
            try:
                _run(job)
            except FARMINGtools.MissingOutputsError:
                pass
        snap = (job.folder_local / "mitim_slurm_snapshot.txt").read_text()
        assert snap.count("=====") == 2 * (len(FARMINGtools.SRUN_STEP_REFUSAL_WAITS) + 1), snap
        assert "before re-run 1" in snap and "giving up" in snap, snap
        assert "FAKE-scontrol show job 58888652" in snap and "FAKE-squeue -s -j 58888652" in snap, snap
        # the key fields also reach the (permanent) driver log
        out = log.getvalue()
        assert "SLURM snapshot (step refusal, before re-run 1): JobState=RUNNING Reason=None" in out, out
        assert "EndTime=2026-10-02T09:59:12" in out and "58888652.batch RUNNING" in out, out

        del os.environ["SLURM_JOB_ID"]
        job, calls = _job(fail_times=99)
        with _harness(tty=False) as (slept, log, asked):
            try:
                _run(job)
            except FARMINGtools.MissingOutputsError:
                pass
        assert not (job.folder_local / "mitim_slurm_snapshot.txt").exists()
    finally:
        os.environ["PATH"] = saved_path
        if saved_id is not None:
            os.environ["SLURM_JOB_ID"] = saved_id
        else:
            os.environ.pop("SLURM_JOB_ID", None)
    print("PASS test_slurm_snapshot_inside_allocation")


def test_missing_outputs_without_refusal_is_not_rerun():
    job, calls = _job(fail_times=0)
    job.execute = lambda command_str, **kwargs: (calls.append(command_str) or (b"", b"segfault\n"))
    with _harness(tty=False) as (slept, log, asked):
        try:
            _run(job)
            raise AssertionError("full_process should have raised")
        except FARMINGtools.MissingOutputsError as e:
            msg = str(e)
    assert len(calls) == 1, calls
    assert "SLURM refused" not in msg and "after 1 execution(s)" in msg, msg
    print("PASS test_missing_outputs_without_refusal_is_not_rerun")


def test_interactive_session_still_prompts():
    job, calls = _job(fail_times=99)
    with _harness(tty=True) as (slept, log, asked):
        _run(job)  # answered 'y': returns, as before
    assert len(asked) == 1, asked
    assert len(calls) == 1 + len(FARMINGtools.SRUN_STEP_REFUSAL_WAITS), calls
    print("PASS test_interactive_session_still_prompts")


def test_dispatch_blocking_carries_the_cause():
    tmp = Path(tempfile.mkdtemp())
    cause = FARMINGtools.MissingOutputsError("[MITIM] Dispatch in /scratch/x did not return ... SLURM refused 5 job step(s)")

    def run(**kwargs):
        raise cause

    sim = types.SimpleNamespace(simulation_job=types.SimpleNamespace(run=run))
    settings = types.SimpleNamespace(run_type=SIMtools.RunType.NORMAL, attempts_execution=1,
                                     helper_lostconnection=False, code="neo", tmpFolder=tmp, input_file="input.neo")
    try:
        SIMtools.mitim_simulation._dispatch_blocking(sim, None, [tmp / "gone_rho_folder"], settings)
        raise AssertionError("_dispatch_blocking should have raised")
    except RuntimeError as e:
        assert "SLURM refused 5 job step(s)" in str(e) and e.__cause__ is cause, e
    print("PASS test_dispatch_blocking_carries_the_cause")


def test_outputs_only_folder_is_not_repeated():
    # A failed retrieval can recreate the rho folder with partial outputs only (no input.neo):
    # repeating would re-send it and fail in NEO's parser, so it must stop with the clear error.
    tmp = Path(tempfile.mkdtemp())
    rho = tmp / "rho_0.5"
    rho.mkdir()
    (rho / "out.neo.run").write_text("")
    calls = []

    def run(**kwargs):
        calls.append(1)
        raise FARMINGtools.MissingOutputsError("[MITIM] Dispatch ... SLURM refused 5 job step(s)")

    sim = types.SimpleNamespace(simulation_job=types.SimpleNamespace(run=run))
    settings = types.SimpleNamespace(run_type=SIMtools.RunType.NORMAL, attempts_execution=1,
                                     helper_lostconnection=False, code="neo", tmpFolder=tmp, input_file="input.neo")
    try:
        SIMtools.mitim_simulation._dispatch_blocking(sim, None, [rho], settings)
        raise AssertionError("_dispatch_blocking should have raised")
    except RuntimeError as e:
        assert "staged" in str(e) and len(calls) == 1, (e, calls)
    print("PASS test_outputs_only_folder_is_not_repeated")


if __name__ == "__main__":
    folders_before = set(Path(tempfile.gettempdir()).glob("tmp*"))
    test_refusal_is_rerun_until_success()
    test_refusal_reruns_do_not_spend_attempts()
    test_persistent_refusal_in_batch_raises_a_clear_error()
    test_slurm_snapshot_inside_allocation()
    test_missing_outputs_without_refusal_is_not_rerun()
    test_interactive_session_still_prompts()
    test_dispatch_blocking_carries_the_cause()
    test_outputs_only_folder_is_not_repeated()
    for folder in set(Path(tempfile.gettempdir()).glob("tmp*")) - folders_before:
        shutil.rmtree(folder, ignore_errors=True)
    print("\nALL PASS")
