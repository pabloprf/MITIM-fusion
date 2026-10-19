"""
test_cgyro_read_batch.py
========================
A CGYRO read that fails offers to run `cgyro -t` in the folder (a 'q' question). In a SLURM driver
there is no terminal: LOGtools refuses to ask (InteractiveTerminalError), and the driver must then die
with the READ error, not with InteractiveTerminalError (G pt06 lost six chained links to that on
2026-10-13). At a terminal the question is asked as before.

Run as:

    python tests/dev_tests/test_cgyro_read_batch.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from mitim_tools.gacode_tools.utils import CGYROutils
from mitim_tools.misc_tools import CONFIGread, LOGtools


class _ReadError(ValueError):
    pass


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


@contextlib.contextmanager
def patched(tty, answer=None):
    '''A pygacode reader that always fails, verbose 2 (questions are asked), a stdin that is or is not a
    terminal, the answer a terminal user gives, and a record of the `cgyro -t` calls.'''
    calls = []
    keep = (CGYROutils.__dict__.get("cgyrodata_plot"), CGYROutils.subprocess.run, CONFIGread.read_verbose_level,
            sys.stdin, LOGtools.query_yes_no)

    def fail(path):
        raise _ReadError("cannot reshape array of size 16288 into shape (2,16,501)")

    CGYROutils.cgyrodata_plot = fail
    CGYROutils.subprocess.run = lambda cmd, **kw: calls.append(cmd)
    CONFIGread.read_verbose_level = lambda: 2
    sys.stdin = _Stdin(tty)
    if answer is not None:
        LOGtools.query_yes_no = lambda *a, **k: answer
    try:
        yield calls
    finally:
        if keep[0] is None:
            del CGYROutils.cgyrodata_plot
        else:
            CGYROutils.cgyrodata_plot = keep[0]
        CGYROutils.subprocess.run, CONFIGread.read_verbose_level, sys.stdin, LOGtools.query_yes_no = keep[1:]


def read(folder):
    out = CGYROutils.CGYROoutput.__new__(CGYROutils.CGYROoutput)
    with contextlib.redirect_stdout(io.StringIO()):
        return out.read_using_cgyroplot(Path(folder), None)


def test_no_terminal_raises_the_read_error():
    with tempfile.TemporaryDirectory() as tmp, patched(tty=False) as calls:
        try:
            read(tmp)
        except LOGtools.InteractiveTerminalError:
            raise AssertionError("InteractiveTerminalError escaped instead of the read error")
        except _ReadError as e:
            assert "cannot reshape" in str(e), e
            assert e.__suppress_context__, "the refused question must not be chained onto the read error"
        else:
            raise AssertionError("a failed read returned")
        assert calls == [], f"`cgyro -t` must not run without an answer: {calls}"
    print("PASS test_no_terminal_raises_the_read_error")


def test_terminal_still_asks():
    for answer, expected_calls in ((False, []), (True, [["cgyro", "-t"]])):
        with tempfile.TemporaryDirectory() as tmp, patched(tty=True, answer=answer) as calls:
            try:
                read(tmp)
            except _ReadError:
                pass
            else:
                raise AssertionError("a failed read returned")
            assert calls == expected_calls, (answer, calls)
    print("PASS test_terminal_still_asks")


if __name__ == "__main__":
    test_no_terminal_raises_the_read_error()
    test_terminal_still_asks()
