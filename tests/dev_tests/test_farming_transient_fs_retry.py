"""
test_farming_transient_fs_retry.py
==================================
mitim_job._pull_outputs redoes the tar/download/extract after a transient file-system error (EBUSY,
also when tarfile wraps it in ReadError) instead of raising the disk-full prompt, which kills a batch
driver (G_pt06 driver 26540033, 2026-10-22). After TRANSIENT_FS_RETRY_WAITS it falls back to the
prompt; any other error goes to the prompt at once. Local mode: tar runs for real, sleeps are recorded.

Run as:

    python tests/dev_tests/test_farming_transient_fs_retry.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import errno
import io
import subprocess
import sys
import tarfile
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.misc_tools import FARMINGtools


def fake_job(root):
    remote, local = Path(root) / "remote", Path(root) / "local"
    remote.mkdir()
    (remote / "out.cgyro.info").write_text("EXIT\n")
    return types.SimpleNamespace(
        ssh=None, folderExecution=remote, folder_local=local,
        execute=lambda cmd, **kw: subprocess.run(cmd, shell=True, check=True),
        _resolve_fallbacks_on_remote=lambda spec: None,
    )


@contextlib.contextmanager
def patched(fail_times, error):
    '''tarfile.open raises `error()` the first `fail_times` calls; sleeps are recorded; the prompt answers no.'''
    calls, waits, prompts = {"n": 0}, [], []
    real_open, real_sleep, real_print = tarfile.open, FARMINGtools.time.sleep, FARMINGtools.print

    def flaky_open(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise error()
        return real_open(*args, **kwargs)

    def answer_no(*args, typeMsg="", **kwargs):
        if typeMsg == "q":
            prompts.append(args[0])
            return False
        return real_print(*args, typeMsg=typeMsg, **kwargs)

    FARMINGtools.tarfile.open, FARMINGtools.time.sleep, FARMINGtools.print = flaky_open, waits.append, answer_no
    try:
        yield waits, prompts
    finally:
        FARMINGtools.tarfile.open, FARMINGtools.time.sleep, FARMINGtools.print = real_open, real_sleep, real_print


def ebusy_wrapped():
    '''As tarfile raises it: ReadError("not a gzip file") from OSError EBUSY.'''
    try:
        raise OSError(errno.EBUSY, "Device or resource busy")
    except OSError as e:
        try:
            raise tarfile.ReadError("not a gzip file") from e
        except tarfile.ReadError as wrapped:
            return wrapped


def pull(job):
    spec = FARMINGtools.RetrievalSpec(files=["out.cgyro.info"])
    with contextlib.redirect_stdout(io.StringIO()):
        return FARMINGtools.mitim_job._pull_outputs(job, spec, [])


def test_ebusy_is_retried_and_the_pull_succeeds():
    with tempfile.TemporaryDirectory() as d:
        job = fake_job(d)
        with patched(1, ebusy_wrapped) as (waits, prompts):
            assert pull(job) is True
        assert waits == [FARMINGtools.TRANSIENT_FS_RETRY_WAITS[0]] and prompts == []
        assert (job.folder_local / "out.cgyro.info").read_text() == "EXIT\n"
    print("PASS test_ebusy_is_retried_and_the_pull_succeeds")


def test_persistent_ebusy_falls_back_to_the_prompt():
    n = len(FARMINGtools.TRANSIENT_FS_RETRY_WAITS)
    with tempfile.TemporaryDirectory() as d:
        with patched(n + 1, ebusy_wrapped) as (waits, prompts):
            try:
                pull(fake_job(d))
            except tarfile.ReadError:
                pass
            else:
                raise AssertionError("persistent EBUSY did not raise")
        assert waits == list(FARMINGtools.TRANSIENT_FS_RETRY_WAITS) and len(prompts) == 1
    print("PASS test_persistent_ebusy_falls_back_to_the_prompt")


def test_other_errors_go_to_the_prompt_at_once():
    with tempfile.TemporaryDirectory() as d:
        with patched(1, lambda: tarfile.ReadError("not a gzip file")) as (waits, prompts):
            try:
                pull(fake_job(d))
            except tarfile.ReadError:
                pass
            else:
                raise AssertionError("a corrupt tarball did not raise")
        assert waits == [] and len(prompts) == 1
    assert FARMINGtools.transient_fs_errno(OSError(errno.ENOSPC, "No space left on device")) is None
    print("PASS test_other_errors_go_to_the_prompt_at_once")


if __name__ == "__main__":
    test_ebusy_is_retried_and_the_pull_succeeds()
    test_persistent_ebusy_falls_back_to_the_prompt()
    test_other_errors_go_to_the_prompt_at_once()
    print("\nALL TESTS PASSED")
