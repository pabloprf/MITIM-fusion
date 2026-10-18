"""
test_cgyro_restart_validate.py
==============================
A filesystem outage during a CGYRO restart write left an empty out.cgyro.tag and a bin.cgyro.restart with a
zeroed header; the next launch crashed and the driver discarded the radius. Before every launch,
templates/gk_control/cgyro_restart_validate.sh checks both and falls back to bin.cgyro.restart.old, the
watchdog's tag history (.mitim_tag_last / .mitim_tag_prev) or a tag rebuilt from out.cgyro.time.

The real launch markers (gk_control.watchdog.launch_markers) run in bash on fake radius folders (RESTART_STEP 100,
outputs every 0.5 a/cs, binaries with one record per output):
  - valid tag and blob, a warm start (blob, no tag) and a fresh run: untouched;
  - zeroed blob + empty tag + valid .old + history: .old back in place with .mitim_tag_last, binaries cut to it;
  - zeroed blob + its own tag landed + history: .old with .mitim_tag_prev (the entry older than that tag);
  - zeroed blob + empty tag + valid .old, no history: tag rebuilt RESTART_STEP before the newest restart point;
  - valid blob + empty tag + .old + history: the exact pair .old + .mitim_tag_last; without .old the tag is
    rebuilt one RESTART_STEP after .mitim_tag_last;
  - valid blob + empty tag, nothing to rebuild from: no tag (warm start at t=0);
  - no valid blob at all: tag removed, blob set aside (cold start);
  - report mode (interrupted-run probe): nothing changes, the resume time is printed, and the probe gives an
    empty-tag radius a resume time;
  - the watchdog rotates the tag history on every new valid tag and ignores a broken one (real bash, ~70 s).

Run as:

    python tests/dev_tests/test_cgyro_restart_validate.py
"""

from __future__ import annotations

import contextlib
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from mitim_tools.misc_tools import FARMINGtools
from mitim_tools.simulation_tools.gk_control import markers, watchdog

MAGIC, RS, DT_OUT, REC = 140974129, 100, 0.5, 64
INPUT = f"DELTA_T=0.01\nPRINT_STEP=50\nRESTART_STEP={RS}\nMAX_TIME=500.0\n"


def tag_text(i):
    return f"          {i}\n  {i * DT_OUT:.10E}\n  1.0000000000E-02\n"


def blob(valid=True, size=4096, magic=MAGIC):
    head = struct.pack("<ii", magic, 3) if valid else b""
    return head + b"\0" * (size - len(head))


def fake_radius(d, rows=350, tag=300, restart=True, old=False, last=None, prev=None, t0=None, ky_records=None):
    '''Radius folder: `rows` outputs, tag = output number (None: no tag, "": empty), blobs valid unless False.'''
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.cgyro").write_text(INPUT)
    (d / "out.cgyro.time").write_text("".join(f" {DT_OUT * i:.4E}  1.0E-03  1.0E-08  1.0E-02\n" for i in range(1, rows + 1)))
    (d / "bin.cgyro.ky_flux").write_bytes(b"\1" * REC * (ky_records or rows))
    if tag is not None:
        (d / "out.cgyro.tag").write_text("" if tag == "" else tag_text(tag))
    if restart is not None:
        (d / "bin.cgyro.restart").write_bytes(blob(restart))
    if old is not None and old is not False:
        (d / "bin.cgyro.restart.old").write_bytes(blob(old is True))
    for name, i in ((markers.TAG_LAST, last), (markers.TAG_PREV, prev)):
        if i is not None:
            (d / name).write_text(tag_text(i))
    if t0 is not None:
        (d / markers.T0).write_text(f"{t0}\n")
    return d


def launch(d):
    '''Run the launch markers of radius folder d in bash; return its stderr.'''
    marker_cmd, _cleanup = watchdog.launch_markers(str(d))
    out = subprocess.run(["bash", "-c", marker_cmd], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out
    return out.stderr


def report(d):
    out = subprocess.run(["bash", "-c", watchdog.restart_validate(str(d), apply=False)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stderr == "", out
    return out.stdout.strip()


def snapshot(d):
    return {f.name: f.read_bytes() for f in sorted(d.iterdir()) if f.name != markers.RUN_STARTED}


def tag_of(d):
    lines = (d / "out.cgyro.tag").read_text().split()
    return int(lines[0]), float(lines[1])


def is_valid_blob(path):
    return path.read_bytes()[:4] == struct.pack("<i", MAGIC)


def test_valid_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "valid": fake_radius(Path(tmp) / "valid", rows=300, tag=300, last=300, prev=200),
            "warm": fake_radius(Path(tmp) / "warm", rows=0, tag=None),
            "fresh": fake_radius(Path(tmp) / "fresh", rows=0, tag=None, restart=None),
        }
        for name, d in cases.items():
            before = snapshot(d)
            assert report(d) == ("1.5000000000E+02" if name == "valid" else "0"), (name, report(d))
            err = launch(d)
            after = snapshot(d)
            after.pop(markers.T0, None); after.pop(markers.T_END, None)
            assert after == before and "restart_validate" not in err, (name, err)
    print("PASS test_valid_untouched")


def test_zero_header_old_and_last():
    '''The outage hit the blob AND its tag write: .mitim_tag_last is the tag of .old.'''
    with tempfile.TemporaryDirectory() as tmp:
        d = fake_radius(Path(tmp) / "rho", rows=310, tag="", restart=False, old=True, last=200, prev=100)
        assert report(d) == f"{200 * DT_OUT:.10E}"
        err = launch(d)
        assert tag_of(d) == (200, 100.0) and is_valid_blob(d / "bin.cgyro.restart"), err
        assert not (d / "bin.cgyro.restart.old").exists() and not is_valid_blob(d / markers.RESTART_BAD)
        assert (d / markers.TAG_LAST).read_text() == tag_text(200) and not (d / markers.TAG_PREV).exists()
        assert (d / markers.T0).read_text().strip() == "100"
        assert (d / "bin.cgyro.ky_flux").stat().st_size == REC * 200, "binaries cut to the restored tag"
        assert "MITIM restart_validate: bin.cgyro.restart has no valid header" in err and ".mitim_tag_last: output 200" in err, err
    print("PASS test_zero_header_old_and_last")


def test_zero_header_own_tag_landed():
    '''The blob is zero but its tag (output 300) landed and the watchdog recorded it: .old is .mitim_tag_prev.'''
    with tempfile.TemporaryDirectory() as tmp:
        d = fake_radius(Path(tmp) / "rho", rows=310, tag=300, restart=False, old=True, last=300, prev=200)
        err = launch(d)
        assert tag_of(d) == (200, 100.0), err
        assert ".mitim_tag_prev: output 200" in err, err
        # same outage, the watchdog had not seen tag 300 yet: .mitim_tag_last (200) is older than it
        d = fake_radius(Path(tmp) / "rho2", rows=310, tag=300, restart=False, old=True, last=200, prev=100)
        err = launch(d)
        assert tag_of(d) == (200, 100.0) and ".mitim_tag_last: output 200" in err, err
    print("PASS test_zero_header_own_tag_landed")


def test_zero_header_old_no_history():
    '''No history: the bad write was the newest restart point after the launch start (.mitim_t0 = output 100).'''
    with tempfile.TemporaryDirectory() as tmp:
        d = fake_radius(Path(tmp) / "rho", rows=320, tag="", restart=False, old=True, t0=50.0)
        assert report(d) == f"{200 * DT_OUT:.4E}"
        err = launch(d)
        assert tag_of(d) == (200, 100.0) and is_valid_blob(d / "bin.cgyro.restart"), err
        assert (d / "out.cgyro.tag").read_text().count("\n") == 2
        assert (d / "bin.cgyro.ky_flux").stat().st_size == REC * 200
        assert "tag rebuilt from out.cgyro.time: output 200" in err, err
    print("PASS test_zero_header_old_no_history")


def test_empty_tag_valid_blob():
    with tempfile.TemporaryDirectory() as tmp:
        # the tag of the newest blob never landed: .old + .mitim_tag_last is the exact pair
        d = fake_radius(Path(tmp) / "pair", rows=310, tag="", old=True, last=200, prev=100)
        newest = (d / "bin.cgyro.restart").read_bytes()
        err = launch(d)
        assert tag_of(d) == (200, 100.0) and (d / markers.RESTART_BAD).read_bytes() == newest, err
        assert not (d / "bin.cgyro.restart.old").exists() and "continuing from bin.cgyro.restart.old" in err, err
        # no .old: the tag of the blob in place is one RESTART_STEP after .mitim_tag_last
        d = fake_radius(Path(tmp) / "rebuild", rows=310, tag="", last=200)
        err = launch(d)
        assert tag_of(d) == (300, 150.0) and (d / "bin.cgyro.ky_flux").stat().st_size == REC * 300, err
        assert "one RESTART_STEP 100 after .mitim_tag_last" in err, err
    print("PASS test_empty_tag_valid_blob")


def test_nothing_recoverable():
    with tempfile.TemporaryDirectory() as tmp:
        # valid blob, empty tag, no history and no out.cgyro.time: warm start at t=0
        d = fake_radius(Path(tmp) / "warm", rows=310, tag="")
        (d / "out.cgyro.time").unlink()
        assert report(d) == "0"
        err = launch(d)
        assert not (d / "out.cgyro.tag").exists() and is_valid_blob(d / "bin.cgyro.restart"), err
        assert "WARM START AT t=0" in err, err
        # no valid blob anywhere: cold start
        d = fake_radius(Path(tmp) / "cold", rows=310, tag=300, restart=False, old=False, last=300)
        assert report(d) == ""
        err = launch(d)
        assert not (d / "out.cgyro.tag").exists() and not (d / "bin.cgyro.restart").exists(), err
        assert (d / markers.RESTART_BAD).exists() and not (d / markers.TAG_LAST).exists(), err
        assert (d / "bin.cgyro.ky_flux").stat().st_size == REC * 310, "no tag: binaries untouched"
        assert "COLD START" in err, err
    print("PASS test_nothing_recoverable")


def test_report_mode_and_probe():
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "scratch"
        fake_radius(scratch / "base_cgyro/rho_0.5000", rows=310, tag="", restart=False, old=True, last=200, prev=100)
        fake_radius(scratch / "base_cgyro/rho_0.6000", rows=310, tag=300)
        fake_radius(scratch / "base_cgyro/rho_0.7000", rows=310, tag="", restart=False)
        before = {d.name: snapshot(d) for d in (scratch / "base_cgyro").iterdir()}

        def execute(cmd):
            out = subprocess.run(["bash", "-c", cmd], capture_output=True)
            return out.stdout, out.stderr
        job = types.SimpleNamespace(run_in_place=False, folderExecution=scratch, folder_local=scratch.parent,
                                    session=lambda log_file=None: contextlib.nullcontext(), execute=execute)
        rels = [f"base_cgyro/rho_0.{i}000" for i in (5, 6, 7)]
        found = FARMINGtools.mitim_job.probe_interrupted_runs(
            job, rels, ["bin.cgyro.restart", "out.cgyro.tag"], "input.cgyro", progress_file="out.cgyro.tag", progress_line=2,
            checksum_ignore_prefix=["MAX_TIME", "RESTART_STEP"], progress_cmd=watchdog.restart_validate("$d", apply=False))
        assert found[rels[0]][1] == f"{200 * DT_OUT:.10E}", found
        assert found[rels[1]][1] == f"{300 * DT_OUT:.10E}", found
        assert found[rels[2]][1] is None, "a radius that would cold-start keeps no resume time and is discarded"
        assert {d.name: snapshot(d) for d in (scratch / "base_cgyro").iterdir()} == before, "the probe changes nothing"
    print("PASS test_report_mode_and_probe")


def test_watchdog_tag_history():
    '''Fake CGYRO: tag 100, tag 200, then a tag write that the filesystem dropped (empty file).'''
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "rho"
        d.mkdir()
        tag = d / "out.cgyro.tag"
        cmd = (f"printf '{tag_text(100)}' > {tag}; sleep 23; printf '{tag_text(200)}' > {tag}; sleep 21; "
               f": > {tag}; sleep 21")
        subprocess.run(["bash", "-c", watchdog.Watchdog(str(d)).wrap(cmd)], timeout=120)
        assert (d / markers.TAG_LAST).read_text() == tag_text(200), (d / markers.TAG_LAST).read_text()
        assert (d / markers.TAG_PREV).read_text() == tag_text(100)
    print("PASS test_watchdog_tag_history")


if __name__ == "__main__":
    test_valid_untouched()
    test_zero_header_old_and_last()
    test_zero_header_own_tag_landed()
    test_zero_header_old_no_history()
    test_empty_tag_valid_blob()
    test_nothing_recoverable()
    test_report_mode_and_probe()
    test_watchdog_tag_history()
