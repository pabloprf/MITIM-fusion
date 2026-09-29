"""
test_farming_retry_session.py
=============================
The shared transient-retry policy and the ssh/jump/sftp session lifecycle of mitim_job.

RetryPolicy is the single copy of the "retry a remote op across a VPN/DNS flap" loop that
connect_ssh, _sftp_transfer_with_retry and execute_remote used to carry each on their own,
with three different exception tuples (sftp's was missing socket.gaierror, the DNS failure
the machinery exists for). The session is connect/close with try/finally, closing whatever
is live before rebuilding so a reconnect does not orphan a transport + an sftp channel.

Run as:

    python tests/dev_tests/test_farming_retry_session.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import paramiko

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.misc_tools import FARMINGtools

REMOTE_SETTINGS = {
    "machine": "some_cluster",
    "user": "someone",
    "tunnel": None,
    "port": None,
    "identity": None,
}


@contextlib.contextmanager
def _no_sleep():
    """Silence the retry waits and collect them, so the tests run instantly."""
    slept = []
    original = FARMINGtools.time.sleep
    FARMINGtools.time.sleep = lambda seconds: slept.append(seconds)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            yield slept, log
    finally:
        FARMINGtools.time.sleep = original


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _job(machine_settings=REMOTE_SETTINGS):
    folder = Path(tempfile.mkdtemp())
    job = FARMINGtools.mitim_job(folder)
    job.machineSettings = dict(machine_settings)
    return job


def _fake_connect(job, sftp=None):
    """Stand in for define_jump/define_server: hand out a fresh set of clients."""
    def define_jump():
        job.jump_client = FakeClient()

    def define_server(**kwargs):
        job.ssh = FakeClient()
        job.sftp = sftp if sftp is not None else FakeClient()

    job.define_jump = define_jump
    job.define_server = define_server
    return job


def test_transient_error_is_retried_until_success():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise socket.gaierror("temporary failure in name resolution")
        return "done"

    policy = FARMINGtools.RetryPolicy(wait_seconds=0.5, attempts=5)
    with _no_sleep() as (slept, _):
        assert policy.run("fake op", flaky) == "done"
    assert len(calls) == 3, calls
    assert slept == [0.5, 0.5], slept
    print("PASS test_transient_error_is_retried_until_success")


def test_gaierror_is_transient_everywhere():
    '''The DNS flap must be retried on the sftp path too, not only on connect.'''
    assert socket.gaierror in FARMINGtools.RetryPolicy.TRANSIENT

    calls = []

    class FlakySftp:
        closed = False

        def get(self, *args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise socket.gaierror("temporary failure in name resolution")
            return "transferred"

        def close(self):
            pass

    flaky_sftp = FlakySftp()
    # the reconnect between attempts hands out a new sftp, as paramiko does
    job = _fake_connect(_job(), sftp=flaky_sftp)
    job.sftp = flaky_sftp
    job.connection_retry_settings = {"wait_seconds": 0, "attempts": 3}
    with _no_sleep():
        assert job._sftp_transfer_with_retry("get", "remote/file", "local/file") == "transferred"
    assert len(calls) == 2, calls
    print("PASS test_gaierror_is_transient_everywhere")


def test_attempts_exhausted_reraises():
    calls = []

    def always_fails():
        calls.append(1)
        raise EOFError("transport closed")

    policy = FARMINGtools.RetryPolicy(wait_seconds=1, attempts=3)
    with _no_sleep():
        try:
            policy.run("fake op", always_fails)
        except EOFError:
            pass
        else:
            raise AssertionError("exhausted attempts should re-raise")
    assert len(calls) == 3, calls
    print("PASS test_attempts_exhausted_reraises")


def test_non_transient_error_is_not_retried():
    calls = []

    def bad_input():
        calls.append(1)
        raise ValueError("this is a real failure")

    policy = FARMINGtools.RetryPolicy(wait_seconds=1, attempts=None)
    with _no_sleep():
        try:
            policy.run("fake op", bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-transient error must propagate")
    assert len(calls) == 1, calls
    print("PASS test_non_transient_error_is_not_retried")


def test_on_retry_runs_between_attempts_and_survives_its_own_failure():
    calls, reconnects = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise paramiko.ssh_exception.SSHException("transport closed")
        return "done"

    def reconnect():
        reconnects.append(1)
        if len(reconnects) == 1:
            raise OSError("cannot reach the host right now")

    policy = FARMINGtools.RetryPolicy(wait_seconds=0, attempts=None)
    with _no_sleep():
        assert policy.run("fake op", flaky, on_retry=reconnect) == "done"
    assert len(reconnects) == 2, reconnects
    print("PASS test_on_retry_runs_between_attempts_and_survives_its_own_failure")


def test_from_settings_validation():
    assert FARMINGtools.RetryPolicy.from_settings(None) == FARMINGtools.RetryPolicy(5.0, 3)
    assert FARMINGtools.RetryPolicy.from_settings({}) == FARMINGtools.RetryPolicy(5.0, 3)

    forever = FARMINGtools.RetryPolicy.from_settings({"wait_seconds": 30, "attempts": None})
    assert forever.attempts is None and forever.wait_seconds == 30.0, forever

    for bad in (-1, 0, 2.5, "3"):
        try:
            FARMINGtools.RetryPolicy.from_settings({"attempts": bad})
        except ValueError:
            continue
        raise AssertionError(f"attempts={bad!r} should be rejected")
    print("PASS test_from_settings_validation")


def test_policy_follows_late_assignment():
    '''SIMtools/transport_cgyro set connection_retry_settings after the job exists.'''
    job = _job()
    assert job.retry == FARMINGtools.RetryPolicy(5.0, 3), job.retry
    job.connection_retry_settings = {"wait_seconds": 12, "attempts": None}
    assert job.retry == FARMINGtools.RetryPolicy(12.0, None), job.retry
    print("PASS test_policy_follows_late_assignment")


def test_close_on_never_connected_job_does_not_raise():
    job = _job()
    with _no_sleep():
        job.close()
        job.close()
    assert (job.ssh, job.sftp, job.jump_client) == (None, None, None)
    print("PASS test_close_on_never_connected_job_does_not_raise")


def test_connect_closes_the_previous_clients():
    job = _fake_connect(_job())
    with _no_sleep():
        job.connect()
        first_ssh, first_sftp, first_jump = job.ssh, job.sftp, job.jump_client
        job.connect()
    assert first_ssh.closed and first_sftp.closed and first_jump.closed, "previous session leaked"
    assert job.ssh is not first_ssh and not job.ssh.closed, "new session must be live"
    print("PASS test_connect_closes_the_previous_clients")


def test_session_closes_even_when_the_body_raises():
    job = _fake_connect(_job())
    with _no_sleep():
        try:
            with job.session():
                clients = (job.ssh, job.sftp, job.jump_client)
                raise RuntimeError("retrieve blew up")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the session must not swallow the error")
    assert all(client.closed for client in clients), "session left the transport open"
    assert (job.ssh, job.sftp, job.jump_client) == (None, None, None)
    print("PASS test_session_closes_even_when_the_body_raises")


def test_session_closes_a_half_built_connection():
    '''ssh up, open_sftp raised: nothing must be left behind.'''
    job = _job()
    built = []

    def define_jump():
        job.jump_client = FakeClient()
        built.append(job.jump_client)

    def define_server(**kwargs):
        job.ssh = FakeClient()
        built.append(job.ssh)
        raise Exception("[MITIM] SFTPError: Your bashrc on the server likely contains print statements")

    job.define_jump, job.define_server = define_jump, define_server
    with _no_sleep():
        try:
            with job.session():
                raise AssertionError("connect should have failed")
        except Exception as e:
            assert "SFTPError" in str(e), e
    assert all(client.closed for client in built), "half-built session leaked"
    print("PASS test_session_closes_a_half_built_connection")


if __name__ == "__main__":
    folders_before = set(Path(tempfile.gettempdir()).glob("tmp*"))
    test_transient_error_is_retried_until_success()
    test_gaierror_is_transient_everywhere()
    test_attempts_exhausted_reraises()
    test_non_transient_error_is_not_retried()
    test_on_retry_runs_between_attempts_and_survives_its_own_failure()
    test_from_settings_validation()
    test_policy_follows_late_assignment()
    test_close_on_never_connected_job_does_not_raise()
    test_connect_closes_the_previous_clients()
    test_session_closes_even_when_the_body_raises()
    test_session_closes_a_half_built_connection()
    for folder in set(Path(tempfile.gettempdir()).glob("tmp*")) - folders_before:
        shutil.rmtree(folder, ignore_errors=True)
    print("\nALL PASS")
