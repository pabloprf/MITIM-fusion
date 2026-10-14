"""
test_cgyro_inplace_truncate.py
==============================
A CGYRO radius continued in its own folder (interrupted-run rescue or SLURM requeue) resumes at output
i_current (line 1 of out.cgyro.tag). Its per-output binaries must end at that output before CGYRO
starts, or a continuation that stops early leaves stale records the reader cannot reshape (G pt06 ev1
rho 0.8105: 509 records in bin.cgyro.ky_flux against 501 rows of out.cgyro.time).

The real launch markers (gk_control.watchdog.launch_markers, templates/cgyro_inplace_truncate.sh) run in
bash on fake radius folders:
  - 509 outputs, tag at 500: every time-series binary ends at 500 records; bin.cgyro.restart, bin.cgyro.geo
    and the ASCII files are untouched;
  - already inconsistent (501 rows, 509 records): the record count comes from bin.cgyro.ky_flux and
    out.cgyro.grids, and the binaries still end at 500;
  - a warm start (no out.cgyro.tag), binaries shorter than the tag, and a record count that cannot be
    determined: nothing is touched.

Run as:

    python tests/dev_tests/test_cgyro_inplace_truncate.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from mitim_tools.simulation_tools.gk_control import watchdog

N_N, N_SPECIES, N_FIELD = 4, 3, 2
# bytes per output record of each time-series binary (float32 ky_flux with 4 moments, complex kxky, ...)
RECORD = {
    "bin.cgyro.ky_flux": N_N * N_SPECIES * N_FIELD * 4 * 4,
    "bin.cgyro.ky_cflux": N_N * N_SPECIES * N_FIELD * 4 * 4,
    "bin.cgyro.kxky_phi": 2 * 4 * 16 * 3 * N_N,
    "bin.cgyro.freq": 2 * 4 * N_N,
}
STATIC = {"bin.cgyro.restart": 123457, "bin.cgyro.geo": 999}


def fake_radius(d, n_records, n_rows, tag=500, grids=True):
    d.mkdir(parents=True, exist_ok=True)
    for name, rec in RECORD.items():
        (d / name).write_bytes(b"\1" * rec * n_records)
    for name, size in STATIC.items():
        (d / name).write_bytes(b"\2" * size)
    (d / "out.cgyro.time").write_text("".join(f" {0.1 * i:.4E}  1.0E-03  1.0E-08  1.0E-02\n" for i in range(1, n_rows + 1)))
    if grids:
        (d / "out.cgyro.grids").write_text(f"{N_N}\n{N_SPECIES}\n{N_FIELD}\n184\n16\n")
    if tag is not None:
        (d / "out.cgyro.tag").write_text(f"          {tag}\n  5.0000000000E+01\n  1.0E-02\n")
    return d


def launch(d):
    '''Run the launch markers of radius folder d in bash; return its stderr.'''
    marker_cmd, _cleanup = watchdog.launch_markers(str(d))
    out = subprocess.run(["bash", "-c", marker_cmd], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out
    return out.stderr


def sizes(d):
    return {name: (d / name).stat().st_size for name in list(RECORD) + list(STATIC)}


def assert_records(d, n):
    for name, rec in RECORD.items():
        assert (d / name).stat().st_size == rec * n, (d.name, name, (d / name).stat().st_size / rec, n)
    for name, size in STATIC.items():
        assert (d / name).stat().st_size == size, (d.name, name)


def test_continuation_cut_to_restart_output():
    with tempfile.TemporaryDirectory() as tmp:
        d = fake_radius(Path(tmp) / "rho_0.8105", n_records=509, n_rows=509)
        rows = (d / "out.cgyro.time").read_text()
        err = launch(d)
        assert_records(d, 500)
        assert (d / "out.cgyro.time").read_text() == rows, "CGYRO rewinds the ASCII files itself"
        assert "cut from 509 to 500 records" in err, err
    print("PASS test_continuation_cut_to_restart_output")


def test_inconsistent_folder_uses_grids():
    with tempfile.TemporaryDirectory() as tmp:
        d = fake_radius(Path(tmp) / "rho_0.8105", n_records=509, n_rows=501)
        err = launch(d)
        assert_records(d, 500)
        assert "cut from 509 to 500 records" in err, err
    print("PASS test_inconsistent_folder_uses_grids")


def test_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "warm_start": fake_radius(Path(tmp) / "warm", n_records=509, n_rows=509, tag=None),
            "shorter_than_tag": fake_radius(Path(tmp) / "short", n_records=480, n_rows=480),
            "unknown_count": fake_radius(Path(tmp) / "unknown", n_records=509, n_rows=501, grids=False),
        }
        before = {k: sizes(d) for k, d in cases.items()}
        errs = {k: launch(d) for k, d in cases.items()}
        for k, d in cases.items():
            assert sizes(d) == before[k], (k, sizes(d), before[k])
        assert errs["warm_start"] == "" and errs["shorter_than_tag"] == "", errs
        assert "record count of the binaries unknown, none cut" in errs["unknown_count"], errs
    print("PASS test_left_alone")


if __name__ == "__main__":
    test_continuation_cut_to_restart_output()
    test_inconsistent_folder_uses_grids()
    test_left_alone()
