"""
test_job_identity_by_folder.py
==============================
A job without a recorded jobid is found by its name AND the folder it was submitted from, never by
its name alone. PORTALS names every CGYRO array cgyro_ev<N>; on Perlmutter (2026-10-14) a driver with
no jobid polled `squeue -n cgyro_ev0` and followed another run's array, and `delete()` ran
`scancel -n <name>`, which cancels every job of that name of the user.

The generated shell runs for real against a fake `squeue` on PATH that holds two jobs named
cgyro_ev0, submitted from different folders (squeue %Z).

Run as:

    python tests/dev_tests/test_job_identity_by_folder.py
"""
from __future__ import annotations

import contextlib
import io
import os
import shlex
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
from mitim_tools.simulation_tools import SIMtools

NAME = "cgyro_ev0"
FORMAT = '-o "%.15i %.50P %.18j %.10u %.10T %.10M %.10l %.5D %R"'
HEADER = "             JOBID                                          PARTITION               NAME       USER      STATE       TIME TIME_LIMIT NODES NODELIST(REASON)\n"


def _row(jobid, state="RUNNING"):
    return f"        {jobid}_0                                              sched          {NAME}    pablorf    {state}      10:32   12:00:00     1 node0731\n"


@contextlib.contextmanager
def _quiet():
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        yield log


def _fake_squeue(bindir, jobs, fail=False):
    '''`squeue -h -n NAME -o "%F|%Z"` lists `jobs` {jobid: workdir}; `squeue -j IDS -o ...` prints their rows.'''
    lines = "\\n".join(f"{j}|{wd}" for j, wd in jobs.items())
    rows = "".join(f'    *{j}*) printf "%s" {shlex.quote(_row(j))} ;;\n' for j in jobs)
    script = f'''#!/bin/bash
{"echo 'slurm_load_jobs error: Socket timed out' >&2; exit 1" if fail else ""}
if [ "$1" = "-h" ]; then printf "{lines}\\n"; exit 0; fi
if [ "$1" = "-j" ]; then
  printf "%s" {shlex.quote(HEADER)}
  for id in $(echo "$2" | tr , ' '); do
    case "$id" in
{rows}    esac
  done
  exit 0
fi
echo "unexpected squeue call: $*" >&2; exit 2
'''
    path = bindir / "squeue"
    path.write_text(script)
    path.chmod(0o755)


def _job(folder, jobid=None):
    job = FARMINGtools.mitim_job(folder)
    job.prep("echo hello", output_files=["mitim.out"])
    job.machineSettings = {"machine": "local", "modules": None, "slurm": {}, "folderWork": str(folder)}
    job.folderExecution = str(folder)
    job.jobid = jobid
    job.slurm_settings = {"job-name": NAME}
    job.retrieve = lambda **kwargs: True
    return job


def _run_check(job, bindir):
    seen = {}

    def execute(command, **kwargs):
        seen["command"] = command
        out = subprocess.run(["bash", "-c", command], capture_output=True,
                             env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"})
        return out.stdout, out.stderr

    job.execute = execute
    with _quiet() as log:
        job.check()
    return seen["command"], log.getvalue()


def test_recorded_jobid_command_unchanged():
    '''With a recorded jobid the poll command is the one MITIM sent before this fix, byte for byte.'''
    folder = Path(tempfile.mkdtemp())
    try:
        job = _job(folder, jobid="12345678")
        seen = {}
        job.execute = lambda command, **kwargs: (seen.setdefault("command", command), (b"", b""))[1]
        with _quiet():
            job.check()
        q = shlex.quote(str(folder))
        squeue = (f'_squeue_err=$(squeue -j 12345678 {FORMAT} 2>&1 > squeue_output.dat); '
                  f'echo "{FARMINGtools.SqueueAnswer.TRAILER}$? $_squeue_err" >> squeue_output.dat')
        expected = f'if [ -d {q} ]; then cd {q} && {{ {squeue}; }}; else echo {job.REMOTE_FOLDER_GONE}; fi'
        assert seen["command"] == expected, seen["command"]
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_recorded_jobid_command_unchanged")


def test_no_jobid_adopts_only_the_job_of_this_folder():
    '''Two jobs named cgyro_ev0: only the one submitted from this folder is followed, also when the
    folder is reached through a symlink (squeue reports the physical path).'''
    root = Path(tempfile.mkdtemp()).resolve()
    try:
        bindir, mine, other = root / "bin", root / "run_E" / "scratch", root / "run_A" / "scratch"
        for d in (bindir, mine, other):
            d.mkdir(parents=True)
        _fake_squeue(bindir, {"59786869": str(other), "59788097": str(mine)})

        job = _job(mine)
        command, log = _run_check(job, bindir)
        assert job.jobid_found == "59788097_0", (job.jobid_found, log)
        assert job.status == 1, job.status
        assert "59786869" not in (mine / "squeue_output.dat").read_text()
        assert "following job(s) 59788097" in log, log

        link = root / "link_to_E"
        link.symlink_to(mine)
        job = _job(link)
        job.folder_local = mine
        command, log = _run_check(job, bindir)
        assert job.jobid_found == "59788097_0", (job.jobid_found, log)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_no_jobid_adopts_only_the_job_of_this_folder")


def test_no_match_is_not_found_and_silence_is_not():
    '''Only another folder's job of that name: not found (status 2). A lookup slurm did not answer:
    keep polling (status 0), never "finished".'''
    root = Path(tempfile.mkdtemp()).resolve()
    try:
        bindir, mine, other = root / "bin", root / "run_E", root / "run_A"
        for d in (bindir, mine, other):
            d.mkdir(parents=True)
        _fake_squeue(bindir, {"59786869": str(other)})
        job = _job(mine)
        _run_check(job, bindir)
        assert job.status == 2 and job.jobid_found is None, (job.status, job.jobid_found)

        _fake_squeue(bindir, {"59786869": str(other)}, fail=True)
        job = _job(mine)
        _, log = _run_check(job, bindir)
        assert job.status == 0, (job.status, log)
        assert "no answer" in log, log
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_no_match_is_not_found_and_silence_is_not")


def test_delete_never_cancels_by_name():
    '''delete() cancels the recorded jobid, or the jobs of this name from this folder; never `scancel -n`.'''
    folder = Path(tempfile.mkdtemp())
    try:
        for jobid in ("12345678", None):
            sim = SIMtools.mitim_simulation.__new__(SIMtools.mitim_simulation)
            job = _job(folder, jobid=jobid)
            job.jobid_found = None
            job.launchSlurm = True
            sent = {}
            job.prep = lambda command, **kwargs: sent.setdefault("command", command)
            job.run = lambda **kwargs: None
            sim.simulation_job = job
            with _quiet():
                sim.delete()
            assert "scancel -n" not in sent["command"], sent["command"]
            if jobid is not None:
                assert sent["command"] == "scancel 12345678", sent["command"]
            else:
                assert "%Z" in sent["command"] and 'scancel ${_ids//,/ }' in sent["command"], sent["command"]
            assert job.launchSlurm is True
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print("PASS test_delete_never_cancels_by_name")


def test_delete_lookup_cancels_only_this_folder():
    '''The shell delete() sends, run against the fake squeue, cancels only the job of this folder.'''
    root = Path(tempfile.mkdtemp()).resolve()
    try:
        bindir, mine, other = root / "bin", root / "run_E", root / "run_A"
        for d in (bindir, mine, other):
            d.mkdir(parents=True)
        _fake_squeue(bindir, {"59786869": str(other), "59788097": str(mine)})
        (bindir / "scancel").write_text(f'#!/bin/bash\necho "$@" >> {root}/scancel.log\n')
        (bindir / "scancel").chmod(0o755)
        job = _job(mine)
        job.jobid_found = None
        job.launchSlurm = True
        sent = {}
        job.prep = lambda command, **kwargs: sent.setdefault("command", command)
        job.run = lambda **kwargs: None
        sim = SIMtools.mitim_simulation.__new__(SIMtools.mitim_simulation)
        sim.simulation_job = job
        with _quiet():
            sim.delete()
        command = sent["command"]
        subprocess.run(["bash", "-c", f"cd {shlex.quote(str(mine))} && {command}"], check=True,
                       env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"})
        assert (root / "scancel.log").read_text().split() == ["59788097"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_delete_lookup_cancels_only_this_folder")


if __name__ == "__main__":
    test_recorded_jobid_command_unchanged()
    test_no_jobid_adopts_only_the_job_of_this_folder()
    test_no_match_is_not_found_and_silence_is_not()
    test_delete_never_cancels_by_name()
    test_delete_lookup_cancels_only_this_folder()
    print("\nALL PASS")
