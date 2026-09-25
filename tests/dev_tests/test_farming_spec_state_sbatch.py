"""
test_farming_spec_state_sbatch.py
=================================
The three objects that carry the mitim_job I/O contract:

    RetrievalSpec   what retrieve() brings back. run()'s submit mode and check() used to
                    blank and restore five attributes on the job; they now hand retrieve()
                    a narrowed spec, so a status poll can no longer tar a folder or rename
                    a live CGYRO restart file on the remote.
    SlurmState /    every squeue row, typed. Array submissions print one row per element,
    SqueueRecord    so node_of() can name the node of the element that stalled instead of
                    the whole array's node list.
    SbatchScript    the sbatch builder. The text it writes must be byte-identical to what
                    the pre-refactor function wrote (checked here against commit 33212239).

Run as:

    python tests/dev_tests/test_farming_spec_state_sbatch.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
mitim_root = repo_root / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.misc_tools import FARMINGtools

REFERENCE_COMMIT = "33212239"

SQUEUE_ARRAY = (
    "             JOBID                                          PARTITION               NAME       USER      STATE       TIME TIME_LIMIT NODES NODELIST(REASON)\n"
    "    12345678_[3-5]                                              sched         cgyro_test    pablorf    PENDING       0:00   12:00:00     1 (Priority)\n"
    "        12345678_1                                              sched         cgyro_test    pablorf    RUNNING      10:32   12:00:00     1 node0731\n"
    "        12345678_2                                              sched         cgyro_test    pablorf    RUNNING       9:58   12:00:00     1 (null)\n"
)


@contextlib.contextmanager
def _quiet():
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        yield log


def _job_with_full_spec(folder):
    job = FARMINGtools.mitim_job(folder)
    job.prep(
        "echo hello",
        output_files=["mitim.out"],
        output_folders=["rho_0.55"],
        output_folders_selective={"rho_0.55": ["out.cgyro.info", "bin.cgyro.restart"]},
        output_file_fallbacks={"bin.cgyro.restart": "bin.cgyro.restart.old"},
        check_files_in_folder={"rho_0.55": ["out.cgyro.info"]},
    )
    return job


# ---------------------------------------------------------------------------
# RetrievalSpec
# ---------------------------------------------------------------------------


def test_only_returns_a_new_spec_with_nothing_else_set():
    spec = FARMINGtools.RetrievalSpec(
        files=["a.dat"],
        folders=["rho_0.55"],
        selective={"rho_0.55": ["bin.cgyro.restart"]},
        fallbacks={"bin.cgyro.restart": "bin.cgyro.restart.old"},
        check_in_folder={"rho_0.55": ["out.cgyro.info"]},
    )
    narrow = spec.only("squeue_output.dat", optional=["slurm_output.dat"])

    assert narrow is not spec
    assert narrow.files == ["squeue_output.dat"], narrow
    assert narrow.optional == ["slurm_output.dat"], narrow
    assert (narrow.folders, narrow.selective, narrow.fallbacks, narrow.check_in_folder) == ([], {}, {}, {})
    # the original is untouched: nothing to restore afterwards
    assert spec.files == ["a.dat"] and spec.folders == ["rho_0.55"], spec
    print("PASS test_only_returns_a_new_spec_with_nothing_else_set")


def test_legacy_attributes_are_views_on_the_spec():
    folder = Path(tempfile.mkdtemp())
    try:
        job = _job_with_full_spec(folder)
        assert job.output_files == ["mitim.out"]
        assert job.output_folders_selective["rho_0.55"] == ["out.cgyro.info", "bin.cgyro.restart"]
        assert job.check_files_in_folder == {"rho_0.55": ["out.cgyro.info"]}

        # SIMtools.load_submission_state assigns these directly on a re-attached job
        job.output_files = ["mitim.out", "extra.dat"]
        job.output_file_fallbacks = {"a": "b"}
        assert job.spec.files == ["mitim.out", "extra.dat"], job.spec
        assert job.spec.fallbacks == {"a": "b"}, job.spec

        # full_process edits the containers in place
        job.output_folders.append("rho_0.75")
        assert job.spec.folders == ["rho_0.55", "rho_0.75"], job.spec
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_legacy_attributes_are_views_on_the_spec")


def test_setstate_migrates_a_pre_spec_pickle():
    '''gk_object.pkl holds the mitim_job; older ones carry the five names in __dict__.'''
    job = FARMINGtools.mitim_job.__new__(FARMINGtools.mitim_job)
    job.__setstate__(
        {
            "jobid": "42",
            "output_files": ["out.cgyro.info"],
            "output_folders": ["rho_0.55"],
            "output_folders_selective": {"rho_0.55": ["out.cgyro.info"]},
            "output_file_fallbacks": {"bin.cgyro.restart": "bin.cgyro.restart.old"},
            "check_files_in_folder": {"rho_0.55": ["out.cgyro.info"]},
        }
    )
    assert job.jobid == "42"
    assert job.output_files == ["out.cgyro.info"], job.spec
    assert job.spec.fallbacks == {"bin.cgyro.restart": "bin.cgyro.restart.old"}, job.spec
    print("PASS test_setstate_migrates_a_pre_spec_pickle")


# ---------------------------------------------------------------------------
# retrieve() narrowing: run(waitYN=False) and check()
# ---------------------------------------------------------------------------


def test_submit_mode_run_retrieves_only_mitim_out():
    folder = Path(tempfile.mkdtemp())
    try:
        job = _job_with_full_spec(folder)
        job.machineSettings = {"machine": "local", "modules": None, "slurm": {}, "folderWork": str(folder)}
        job.folderExecution = str(folder)
        job.launchSlurm = False
        job.slurm_settings = {"job-name": "mitim_job"}

        seen = {}

        def fake_full_process(comm, **kwargs):
            seen["spec"] = kwargs["spec"]

        job.full_process = fake_full_process

        with _quiet():
            job.run(waitYN=False)

        assert seen["spec"].files == ["mitim.out"], seen["spec"]
        assert (seen["spec"].folders, seen["spec"].selective, seen["spec"].fallbacks) == ([], {}, {})
        # the job keeps the real output spec for the later fetch()
        assert job.output_folders == ["rho_0.55"], job.spec
        assert job.output_folders_selective["rho_0.55"] == ["out.cgyro.info", "bin.cgyro.restart"]
        assert job.output_file_fallbacks == {"bin.cgyro.restart": "bin.cgyro.restart.old"}

        with _quiet():
            job.run(waitYN=True)
        assert seen["spec"] is None, "a waiting run retrieves with the job's own spec"
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_submit_mode_run_retrieves_only_mitim_out")


def test_check_polls_with_a_narrowed_spec():
    folder = Path(tempfile.mkdtemp())
    try:
        job = _job_with_full_spec(folder)
        job.machineSettings = {"machine": "local", "modules": None, "slurm": {}, "folderWork": str(folder)}
        job.folderExecution = str(folder)
        job.jobid = "12345678"
        job.slurm_settings = {"job-name": "cgyro_test"}

        seen = {}

        def fake_execute(command, **kwargs):
            (folder / "squeue_output.dat").write_text(SQUEUE_ARRAY)
            return b"", b""

        def fake_retrieve(**kwargs):
            seen["spec"] = kwargs["spec"]
            return True

        job.execute = fake_execute
        job.retrieve = fake_retrieve

        with _quiet():
            job.check()

        spec = seen["spec"]
        assert spec.files == ["squeue_output.dat"], spec
        assert spec.optional == ["slurm_output.dat"], spec
        # nothing else: no folder is tarred and no primary/fallback rm/mv runs on a poll
        assert (spec.folders, spec.selective, spec.fallbacks) == ([], {}, {}), spec
        # and the job's own spec is intact afterwards
        assert job.output_folders == ["rho_0.55"], job.spec
        assert job.output_file_fallbacks == {"bin.cgyro.restart": "bin.cgyro.restart.old"}
        assert job.status == 0, job.status
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_check_polls_with_a_narrowed_spec")


def test_check_with_the_remote_folder_gone():
    '''A job whose remote folder was deleted is "not found" (status 2), not pending forever: the poll
    used to `cd` into the missing folder, retrieve nothing and assume PENDING on every cycle (engaging
    driver 23553772, 2026-09-23: 2 h 25 min re-attached to a dead array whose scratch was cleaned up).'''
    folder = Path(tempfile.mkdtemp())
    try:
        job = _job_with_full_spec(folder)
        job.machineSettings = {"machine": "local", "modules": None, "slurm": {}, "folderWork": str(folder)}
        job.folderExecution = str(folder / "scratch_deleted")
        job.jobid = "12345678"
        job.slurm_settings = {"job-name": "cgyro_test"}
        job.retrieve = lambda **kwargs: (_ for _ in ()).throw(AssertionError("nothing to retrieve from a deleted folder"))
        with _quiet() as log:
            job.check()
        assert job.status == 2, (job.status, log.getvalue())
        assert job.infoSLURM["STATE"] == FARMINGtools.SlurmState.ABSENT.value, job.infoSLURM
        assert "no longer exists" in log.getvalue(), log.getvalue()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_check_with_the_remote_folder_gone")


def test_retrieve_tars_exactly_what_the_spec_lists():
    '''End-to-end over a local "remote": the tar/copy/extract really runs.'''
    local = Path(tempfile.mkdtemp())
    remote = Path(tempfile.mkdtemp())
    try:
        (remote / "rho_0.55").mkdir()
        (remote / "mitim.out").write_text("Submitted batch job 999\n")
        (remote / "rho_0.55" / "out.cgyro.info").write_text("info\n")
        (remote / "rho_0.55" / "bin.cgyro.restart.old").write_text("restart\n")
        (remote / "rho_0.55" / "huge.bin").write_text("x" * 1000)

        job = _job_with_full_spec(local)
        job.machineSettings = {"machine": "local"}
        job.folderExecution = str(remote)
        job.run_in_place = False
        job.ssh = None

        executed = []

        def fake_execute(command, **kwargs):
            executed.append(command)
            subprocess.run(command, shell=True, capture_output=True)
            return b"", b""

        job.execute = fake_execute

        # narrowed: only the file, no folder, no fallback resolution
        with _quiet():
            assert job.retrieve(spec=job.spec.only("mitim.out")) is True
        assert (local / "mitim.out").exists()
        assert not (local / "rho_0.55").exists(), "a narrowed retrieval must not pull folders"
        assert not any("bin.cgyro.restart.old" in c for c in executed), executed
        assert (remote / "rho_0.55" / "bin.cgyro.restart.old").exists(), "the fallback was renamed on a narrowed pull"

        # full spec: selective folder contents + the primary/fallback rename
        executed.clear()
        with _quiet():
            assert job.retrieve() is True
        assert (local / "rho_0.55" / "out.cgyro.info").exists()
        assert (local / "rho_0.55" / "bin.cgyro.restart").exists(), "the fallback should have been renamed remotely"
        assert not (local / "rho_0.55" / "huge.bin").exists(), "only the selective patterns are tarred"
        assert any(c.startswith("if [ -f") for c in executed), executed
    finally:
        shutil.rmtree(local, ignore_errors=True)
        shutil.rmtree(remote, ignore_errors=True)
    print("PASS test_retrieve_tars_exactly_what_the_spec_lists")


# ---------------------------------------------------------------------------
# SlurmState / SqueueRecord / node_of
# ---------------------------------------------------------------------------


def test_squeue_parse_reads_every_row():
    records = FARMINGtools.SqueueRecord.parse(SQUEUE_ARRAY)
    assert len(records) == 3, records
    assert [r.jobid for r in records] == ["12345678_[3-5]", "12345678_1", "12345678_2"], records
    assert [r.state for r in records] == [
        FARMINGtools.SlurmState.PENDING,
        FARMINGtools.SlurmState.RUNNING,
        FARMINGtools.SlurmState.RUNNING,
    ], records
    assert records[1].nodelist == "node0731" and records[1].node == "node0731"
    assert records[2].nodelist == "(null)" and records[2].node is None
    assert records[0].node is None, "a pending reason is not a node"
    assert all(r.name == "cgyro_test" for r in records), records

    # header only (job gone), and empty output
    assert FARMINGtools.SqueueRecord.parse(SQUEUE_ARRAY.splitlines()[0] + "\n") == []
    assert FARMINGtools.SqueueRecord.parse("") == []
    print("PASS test_squeue_parse_reads_every_row")


def test_state_tokens():
    S = FARMINGtools.SlurmState
    assert S.from_token("NOT FOUND") is S.ABSENT
    assert S.from_token("CANCELLED+") is S.CANCELLED
    assert S.from_token("CANCELLED by 12345") is S.CANCELLED
    assert S.from_token("something_else") is S.UNKNOWN
    assert S.from_token(None) is S.UNKNOWN
    assert all(s.in_queue for s in (S.PENDING, S.RUNNING, S.COMPLETING, S.CONFIGURING, S.REQUEUED, S.SUSPENDED))
    assert not S.COMPLETED.in_queue and not S.ABSENT.in_queue
    assert all(s.terminal for s in (S.COMPLETED, S.COMPLETING, S.CANCELLED, S.FAILED, S.TIMEOUT,
                                    S.OUT_OF_MEMORY, S.BOOT_FAIL, S.NODE_FAIL, S.PREEMPTED, S.REVOKED, S.DEADLINE))
    assert not S.PENDING.terminal and not S.RUNNING.terminal
    print("PASS test_state_tokens")


def _interpret(folder, text):
    job = FARMINGtools.mitim_job(folder)
    job.jobid = "12345678"
    job.slurm_settings = {"job-name": "cgyro_test"}
    (folder / "squeue_output.dat").write_text(text)
    with _quiet():
        job.interpret_status()
    return job


def test_interpret_status_mapping():
    folder = Path(tempfile.mkdtemp())
    try:
        header = SQUEUE_ARRAY.splitlines()[0]
        row = "        12345678                                              sched         cgyro_test    pablorf {:>10s}      10:32   12:00:00     1 node0731"

        for state, status in [("PENDING", 0), ("RUNNING", 1), ("COMPLETING", 1), ("REQUEUED", 0)]:
            job = _interpret(folder, header + "\n" + row.format(state) + "\n")
            assert job.status == status, (state, job.status)
            assert job.infoSLURM["STATE"] == state, job.infoSLURM
            assert job.infoSLURM["NODELIST(REASON)"] == "node0731", job.infoSLURM
            assert job.jobid_found == "12345678", job.jobid_found

        # job no longer in the queue -> status 2, the one status the callers read as "done"
        job = _interpret(folder, header + "\n")
        assert job.status == 2 and job.infoSLURM["STATE"] == "NOT FOUND", (job.status, job.infoSLURM)
        assert job.records == [], job.records
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_interpret_status_mapping")


def test_node_of_an_array_element():
    folder = Path(tempfile.mkdtemp())
    try:
        job = _interpret(folder, SQUEUE_ARRAY)
        assert job.status == 0, job.status                     # first row is the pending one
        assert job.infoSLURM["STATE"] == "PENDING", job.infoSLURM
        assert job.node_of(1) == "node0731", job.records
        assert job.node_of(2) is None, "'(null)' means no node yet"
        assert job.node_of(4) is None, "still inside the compressed 12345678_[3-5] row"
        assert job.node_of(None) is None
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_node_of_an_array_element")


# ---------------------------------------------------------------------------
# SbatchScript: the written text must not move
# ---------------------------------------------------------------------------


def _load_reference_module():
    source = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{REFERENCE_COMMIT}:src/mitim_tools/misc_tools/FARMINGtools.py"],
        capture_output=True,
    )
    if source.returncode != 0:
        return None
    path = Path(tempfile.mkdtemp()) / "old_farming.py"
    path.write_bytes(source.stdout)
    spec = importlib.util.spec_from_file_location("old_farming", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = {
    "plain": dict(
        slurm_settings={"job-name": "mitim_run", "minutes": 90, "ntasks": 4, "cpus-per-task": 8},
        slurm_allocation={"partition": "sched_mit_psfc", "account": "psfc", "mem": "64GB"},
        label_log_files="",
        launchSlurm=True,
    ),
    "array_gpu": dict(
        slurm_settings={
            "job-name": "cgyro_array", "time": "12:00:00", "array": "0-5", "array_limit": 2,
            "nodes": 1, "gpus-per-node": 4, "exclusive": "user", "requeue": False,
        },
        slurm_allocation={"partition": "gpu", "qos": "high", "constraint": "a100", "exclude": "node0731", "email": "someone@mit.edu"},
        label_log_files="_rho1",
        if_array_relabel=True,
        append_mode=True,
        launchSlurm=True,
    ),
    "lock_file": dict(
        slurm_settings={"job-name": "mitim_lock", "time": "01:00:00"},
        slurm_allocation={"partition": "sched_mit_psfc", "exclusive": True},
        lock_file=True,
        lock_file_timeout_hours=12,
        wait_until_sbatch=False,
        launchSlurm=True,
    ),
    "bash_mode": dict(
        slurm_settings={"job-name": "mitim_bash"},
        slurm_allocation={},
        launchSlurm=False,
    ),
}

COMMON = dict(
    command=["cd /scratch/run", "./cgyro_call.sh"],
    folderExecution="/scratch/run with space",
    modules_remote="export GACODE_ROOT=/opt/gacode; . ${GACODE_ROOT}/shared/bin/gacode_setup",
    shellPreCommands=["echo pre"],
    shellPostCommands=["echo post"],
)


def test_sbatch_text_is_byte_identical_to_the_reference_commit():
    reference = _load_reference_module()
    if reference is None:
        print(f"SKIP test_sbatch_text_is_byte_identical (commit {REFERENCE_COMMIT} not available)")
        return

    for name, case in CASES.items():
        folders = {}
        texts = {}
        for tag, module in (("old", reference), ("new", FARMINGtools)):
            folder = Path(tempfile.mkdtemp())
            folders[tag] = folder
            kwargs = dict(COMMON)
            kwargs.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in case.items()})
            with _quiet():
                comm, fileSBATCH, fileSHELL = module.create_slurm_execution_files(folder_local=folder, **kwargs)
            texts[tag] = (
                comm.replace(str(folder), "<folder>"),
                Path(fileSBATCH).read_bytes(),
                Path(fileSHELL).read_bytes(),
                Path(fileSBATCH).name,
                Path(fileSHELL).name,
            )
        for folder in folders.values():
            shutil.rmtree(folder, ignore_errors=True)
        assert texts["old"][1] == texts["new"][1], f"{name}: mitim_bash.src changed\n--- old ---\n{texts['old'][1].decode()}\n--- new ---\n{texts['new'][1].decode()}"
        # Arrays launched with --wait also wait for every task to leave the queue (FARMINGtools._ARRAY_WAIT_BASH),
        # right after the sbatch line; everything else is unchanged
        shell_new = texts["new"][2].decode()
        if case.get("launchSlurm") and case.get("wait_until_sbatch", True) and case["slurm_settings"].get("array"):
            block = "\n".join(FARMINGtools._ARRAY_WAIT_BASH) + "\n"
            assert ".src\n" + block in shell_new, f"{name}: array wait block missing after the sbatch line"
            shell_new = shell_new.replace(block, "", 1)
        else:
            assert "_mitim_jobid" not in shell_new, f"{name}: array wait block in a non-array/no-wait launch"
        assert texts["old"][2].decode() == shell_new, f"{name}: mitim_shell_executor.sh changed"
        assert texts["old"][0] == texts["new"][0], f"{name}: launch command changed"
        assert texts["old"][3:] == texts["new"][3:], f"{name}: file names changed"
    print(f"PASS test_sbatch_text_is_byte_identical_to_the_reference_commit ({len(CASES)} parameter sets)")


def test_builder_does_not_write_into_the_caller_dicts():
    settings = {"job-name": "mitim_run", "minutes": 30}
    allocation = {"partition": "sched_mit_psfc"}
    folder = Path(tempfile.mkdtemp())
    try:
        with _quiet():
            FARMINGtools.create_slurm_execution_files(
                folder_local=folder, slurm_settings=settings, slurm_allocation=allocation, **COMMON
            )
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert settings == {"job-name": "mitim_run", "minutes": 30}, settings
    assert allocation == {"partition": "sched_mit_psfc"}, allocation
    print("PASS test_builder_does_not_write_into_the_caller_dicts")


if __name__ == "__main__":
    test_only_returns_a_new_spec_with_nothing_else_set()
    test_legacy_attributes_are_views_on_the_spec()
    test_setstate_migrates_a_pre_spec_pickle()
    test_submit_mode_run_retrieves_only_mitim_out()
    test_check_polls_with_a_narrowed_spec()
    test_check_with_the_remote_folder_gone()
    test_retrieve_tars_exactly_what_the_spec_lists()
    test_squeue_parse_reads_every_row()
    test_state_tokens()
    test_interpret_status_mapping()
    test_node_of_an_array_element()
    test_sbatch_text_is_byte_identical_to_the_reference_commit()
    test_builder_does_not_write_into_the_caller_dicts()
    print("\nALL PASS")
