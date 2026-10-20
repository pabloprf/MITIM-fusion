"""
test_srun_shim_fallback.py
==========================
The no-srun fallback for codes that opt in (NEO: run_specifications 'srun_shim_on_refusal').
Once a Perlmutter job has run past its --time-min, SLURM refuses every new step ("Unable to create
step ... already completing or completed") for the rest of the job, so the timed re-runs of
test_srun_step_refusal_retry.py cannot help. A local bash dispatch of such a code is instead re-run
once with FARMINGtools.SRUN_SHIM_SCRIPT as `srun`, first on PATH after the modules.

- The shim runs a 1-task launch directly (whatever the option spelling) and hands any other launch,
  with its arguments untouched, to the real srun.
- mitim_job.run() on a local in-place job: a refused dispatch is re-run without srun, without any
  wait, its outputs are kept, the driver log says the outputs come from a run WITHOUT srun, and the
  shim folder is removed from the user's folder.
- A job that did not opt in keeps the timed re-runs and never writes the shim.

Run as:

    python tests/dev_tests/test_srun_shim_fallback.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.misc_tools import CONFIGread, FARMINGtools, LOGtools

FOLDER = "rho_0.5"
OUTPUT = "out.neo.transport_flux"


def _bin(folder, name, body):
    path = folder / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)
    return path


def _real_srun(folder, refuse=True):
    '''A stand-in for the SLURM srun: refuses the step as Perlmutter does, or just reports its arguments.'''
    folder.mkdir(parents=True, exist_ok=True)
    body = ('echo "srun: error: Unable to create step for job 60016676: Job/step already completing or completed" >&2; exit 1'
            if refuse else 'echo "REAL srun $*"')
    return _bin(folder, "srun", body).parent


@contextlib.contextmanager
def _harness():
    """Batch session (no tty), verbose 3, sleeps recorded instead of slept."""
    slept = []
    saved = (FARMINGtools.time.sleep, CONFIGread.read_verbose_level, sys.stdin)
    FARMINGtools.time.sleep = lambda seconds: slept.append(seconds)
    CONFIGread.read_verbose_level = lambda: 3
    sys.stdin = io.StringIO()
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            yield slept, log
    finally:
        FARMINGtools.time.sleep, CONFIGread.read_verbose_level, sys.stdin = saved


def test_shim_routes_by_task_count():
    tmp = Path(tempfile.mkdtemp())
    real = _real_srun(tmp / "real", refuse=False)
    shim_dir = tmp / FARMINGtools.SRUN_SHIM_FOLDER
    shim_dir.mkdir()
    (shim_dir / "srun").write_text(FARMINGtools.SRUN_SHIM_SCRIPT)
    (shim_dir / "srun").chmod(0o755)
    prog = _bin(tmp, "prog", 'echo "PROG $* OMP=$OMP_NUM_THREADS"')
    env = dict(os.environ, PATH=f"{shim_dir}:{real}:{os.environ['PATH']}", OMP_NUM_THREADS="4")

    def srun(*args):
        res = subprocess.run(["srun", *args], capture_output=True, text=True, env=env)
        return res.stdout.strip(), res.stderr

    # the exact line of gacode's platform/exec/exec.PERLMUTTER_CPU for a 1-core NEO
    out, err = srun("--cpu_bind=cores", "-n", "1", "-c", "4", str(prog), "a", "b")
    assert out == "PROG a b OMP=4" and "without srun" in err, (out, err)
    for spelling in (["-n1"], ["--ntasks=1"], ["--ntasks", "1"], []):
        out, _ = srun(*spelling, str(prog))
        assert out == "PROG  OMP=4", (spelling, out)
    # more than one task: the real srun, arguments untouched, shim no longer on its PATH
    out, err = srun("--cpu_bind=cores", "-n", "4", "-c", "2", str(prog), "x")
    assert out == f"REAL srun --cpu_bind=cores -n 4 -c 2 {prog} x" and "without srun" not in err, (out, err)
    shutil.rmtree(tmp)
    print("PASS test_shim_routes_by_task_count")


def _local_job(opt_in):
    '''Local in-place job whose command launches a fake 1-task NEO through srun; the modules put a refusing srun on PATH.'''
    folder = Path(tempfile.mkdtemp())
    real = _real_srun(folder / "slurm_bin")
    (folder / FOLDER).mkdir()
    neo = _bin(folder, "fake_neo", f'echo "fluxes" > "$1/{OUTPUT}"')

    job = FARMINGtools.mitim_job(folder)
    job.machineSettings = {"machine": "local", "modules": f'export PATH={real}:"$PATH"', "slurm": {}}
    job.folderExecution = folder
    job.run_in_place = True
    job.launchSlurm = False
    job.slurm_settings = {}
    job.check_files_in_folder = {FOLDER: [OUTPUT]}
    job.srun_shim_on_refusal = opt_in
    job.prep(f"srun --cpu_bind=cores -n 1 -c 1 {neo} {folder / FOLDER}",
             output_folders=[FOLDER], check_files_in_folder={FOLDER: [OUTPUT]})
    return job, folder


def test_refused_dispatch_reruns_without_srun():
    job, folder = _local_job(opt_in=True)
    with _harness() as (slept, log):
        job.run(waitYN=True, attempts_execution=1)
    out = log.getvalue()
    assert (folder / FOLDER / OUTPUT).read_text() == "fluxes\n"
    assert slept == [], slept  # straight to the shim, no timed re-run
    assert "re-running the dispatch WITHOUT srun" in out and "come from a run WITHOUT srun" in out, out
    assert not (folder / FARMINGtools.SRUN_SHIM_FOLDER).exists()  # in place: nothing left in the user's folder
    # the PATH line sits after the modules, so the shim wins over the module's srun
    bash = (folder / "mitim_bash.src").read_text().splitlines()
    assert bash.index(job.machineSettings["modules"]) < next(i for i, line in enumerate(bash) if FARMINGtools.SRUN_SHIM_FOLDER in line), bash
    shutil.rmtree(folder)
    print("PASS test_refused_dispatch_reruns_without_srun")


def test_job_without_opt_in_keeps_timed_reruns():
    job, folder = _local_job(opt_in=False)
    with _harness() as (slept, log):
        try:
            job.run(waitYN=True, attempts_execution=1)
            raise AssertionError("run should have raised")
        except FARMINGtools.MissingOutputsError as e:
            msg = str(e)
    assert slept == list(FARMINGtools.SRUN_STEP_REFUSAL_WAITS), slept
    assert "without srun" not in msg and not (folder / FARMINGtools.SRUN_SHIM_FOLDER).exists(), msg
    assert issubclass(FARMINGtools.MissingOutputsError, LOGtools.InteractiveTerminalError)
    shutil.rmtree(folder)
    print("PASS test_job_without_opt_in_keeps_timed_reruns")


if __name__ == "__main__":
    test_shim_routes_by_task_count()
    test_refused_dispatch_reruns_without_srun()
    test_job_without_opt_in_keeps_timed_reruns()
    print("\nALL PASS")
