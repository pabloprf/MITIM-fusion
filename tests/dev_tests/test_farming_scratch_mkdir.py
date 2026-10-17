"""
test_farming_scratch_mkdir.py
=============================
mitim_job.create_scratch_folder raises, with the path and the mkdir stderr, when the scratch
folder cannot be created (e.g. "Disk quota exceeded"), instead of letting the run fail later
with a bare sftp FileNotFoundError. Remote execution is mocked; the local cases run mkdir for real.

Run as:

    python tests/dev_tests/test_farming_scratch_mkdir.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.misc_tools import FARMINGtools


def fake_job(folder, execute, remote=True):
    return types.SimpleNamespace(run_in_place=False, ssh=object() if remote else None, folderExecution=folder, execute=execute)


def create(job):
    with contextlib.redirect_stdout(io.StringIO()):
        return FARMINGtools.mitim_job.create_scratch_folder(job)


def raises(job, *pieces):
    try:
        create(job)
    except RuntimeError as e:
        for piece in pieces:
            assert piece in str(e), (piece, str(e))
        return
    raise AssertionError("create_scratch_folder did not raise")


def test_remote_failure_raises_with_stderr():
    folder = "/orcd/pool/003/pablorf/scratch/mitim_run"
    err = b"mkdir: cannot create directory '/orcd/pool/003/pablorf/scratch/mitim_run': Disk quota exceeded\n"
    raises(fake_job(folder, lambda cmd: (b"", err)), folder, "Disk quota exceeded", "remote")
    # stderr noise from a chatty bashrc does not count as a failure
    out, _ = create(fake_job(folder, lambda cmd: (b"MITIM_MKDIR_OK\n", b"module: loaded something\n")))
    assert b"MITIM_MKDIR_OK" in out
    # execute_remote returns (None, None) on a socket timeout
    raises(fake_job(folder, lambda cmd: (None, None)), folder, "timed out")
    print("PASS: remote mkdir failure / success with stderr noise / timeout")


def test_local_mkdir_for_real():
    execute = lambda cmd: FARMINGtools.run_subprocess([cmd], localRun=True)
    with tempfile.TemporaryDirectory() as tmp:
        ok = Path(tmp) / "a" / "b"
        create(fake_job(ok, execute, remote=False))
        assert ok.is_dir()
        blocker = Path(tmp) / "file"
        blocker.write_text("x")
        raises(fake_job(blocker / "sub", execute, remote=False), str(blocker / "sub"), "Not a directory")
    print("PASS: local mkdir created, and a path under a regular file raises with its stderr")


def test_run_in_place_skips():
    assert FARMINGtools.mitim_job.create_scratch_folder(types.SimpleNamespace(run_in_place=True)) == (None, None)
    print("PASS: run_in_place skips the scratch folder")


if __name__ == "__main__":
    test_remote_failure_raises_with_stderr()
    test_local_mkdir_for_real()
    test_run_in_place_skips()
    print("\nALL PASS")
