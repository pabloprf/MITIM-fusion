"""
test_ssh_timeouts.py
====================
Every paramiko call of a remote session is bounded (FARMINGtools SSH_* / SFTP_IDLE_TIMEOUT), so a link
that stays up but stops answering raises a transient error that the retry policy turns into a reconnect,
instead of blocking the driver (a laptop driver hung 10-15 min in SSHClient.open_sftp on 2026-10-14).

Checks, with paramiko mocked (no network):
  1. connect() hands timeout / banner_timeout / auth_timeout to SSHClient.connect.
  2. The sftp session is opened with a channel-open timeout and its channel gets the idle timeout.
  3. A channel open that never answers ends with a transient error within its timeout, and the
     connect retry rebuilds the session and succeeds on the next attempt.

Run as:

    python tests/dev_tests/test_ssh_timeouts.py
"""

import contextlib
import io
import threading
import time
from pathlib import Path
from unittest import mock

import paramiko

from mitim_tools.misc_tools import FARMINGtools


class FakeChannel:
    def __init__(self):
        self.timeout, self.subsystem = None, None

    def settimeout(self, t):
        self.timeout = t

    def invoke_subsystem(self, name):
        self.subsystem = name


class FakeTransport:
    '''open_session that, while `hang` is set, behaves like paramiko's: waits for the server up to `timeout`.'''

    def __init__(self, hang):
        self.hang = hang
        self.opened = []

    def open_session(self, timeout=None):
        if self.hang:
            assert timeout is not None, "an unbounded channel open would block forever"
            threading.Event().wait(timeout)
            raise paramiko.ssh_exception.SSHException("Timeout opening channel.")
        channel = FakeChannel()
        self.opened.append((timeout, channel))
        return channel


class FakeSSHClient:
    instances = []
    hang_first = True

    def __init__(self):
        self.connect_kwargs = None
        hang = FakeSSHClient.hang_first and not FakeSSHClient.instances
        self.transport = FakeTransport(hang)
        FakeSSHClient.instances.append(self)

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, host, **kwargs):
        self.connect_kwargs = kwargs

    def get_transport(self):
        return self.transport

    def close(self):
        pass


def _job():
    job = FARMINGtools.mitim_job(Path("/tmp/mitim_ssh_timeouts_test"))
    job.machineSettings = {"machine": "perlmutter.example", "tunnel": None, "user": "me", "port": None, "identity": None}
    job.connection_retry_settings = {"wait_seconds": 0, "attempts": 3}
    return job


def _connect(job):
    FakeSSHClient.instances = []
    with mock.patch.object(FARMINGtools.paramiko, "SSHClient", FakeSSHClient), \
         mock.patch.object(FARMINGtools.paramiko, "SFTPClient", lambda channel: ("sftp", channel)), \
         mock.patch.object(FARMINGtools, "SSH_CHANNEL_OPEN_TIMEOUT", 1.0), \
         contextlib.redirect_stdout(io.StringIO()) as log:
        t0 = time.time()
        job.connect()
        return time.time() - t0, log.getvalue()


def test_connect_passes_timeouts():
    FakeSSHClient.hang_first = False
    job = _job()
    _connect(job)
    (client,) = FakeSSHClient.instances
    kw = client.connect_kwargs
    assert (kw["timeout"], kw["banner_timeout"], kw["auth_timeout"]) == (
        FARMINGtools.SSH_CONNECT_TIMEOUT, FARMINGtools.SSH_BANNER_TIMEOUT, FARMINGtools.SSH_AUTH_TIMEOUT), kw
    ((timeout, channel),) = client.transport.opened
    assert timeout == 1.0 and channel.timeout == FARMINGtools.SFTP_IDLE_TIMEOUT and channel.subsystem == "sftp"
    assert job.sftp == ("sftp", channel)
    print("PASS test_connect_passes_timeouts")


def test_hung_channel_open_is_retried():
    FakeSSHClient.hang_first = True
    job = _job()
    elapsed, log = _connect(job)
    assert len(FakeSSHClient.instances) == 2, "the hung session must be rebuilt once"
    assert job.sftp is not None and job.sftp[0] == "sftp"
    assert 0.9 <= elapsed < 10, f"the hang must end at the channel-open timeout, took {elapsed:.1f} s"
    assert "Timeout opening channel" in log, log
    print(f"PASS test_hung_channel_open_is_retried ({elapsed:.1f} s)")


if __name__ == "__main__":
    test_connect_passes_timeouts()
    test_hung_channel_open_is_retried()
    print("\nALL PASS")
