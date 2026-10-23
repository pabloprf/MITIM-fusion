"""
test_cgyro_read_short_ky_flux.py
================================
When bin.cgyro.ky_flux holds fewer records than out.cgyro.time has rows, CGYROoutput reads only the rows
ky_flux holds: in the private read directory it trims out.cgyro.time AND every other per-output binary
that holds one record per row (bin.cgyro.freq kept 500 records against a ky_flux of 493 and pygacode
failed to reshape it, D_pt03 SR ev3 r/a 0.9, 2026-10-22). The run folder itself is never touched.

Run as:

    python tests/dev_tests/test_cgyro_read_short_ky_flux.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools.utils.CGYROutils import CGYROoutput

N_N, N_SPECIES, N_FIELD, N_MOMENTS = 16, 4, 3, 4
KY_FLUX_RECORD = N_N * N_SPECIES * N_FIELD * N_MOMENTS * 4   # float32
FREQ_RECORD, PHI_RECORD = 2 * N_N * 4, 41472


def write_run(folder, suffix, nt, nt_flux):
    '''D_pt03 rho 0.8085 as it was read: out.cgyro.time, freq and kxky_phi at nt records, ky_flux at nt_flux.'''
    np.savetxt(folder / f"out.cgyro.time{suffix}", np.column_stack([np.arange(1, nt + 1), np.ones((nt, 3))]), fmt="%.4E")
    (folder / f"out.cgyro.grids{suffix}").write_text(f"{N_N}\n{N_SPECIES}\n{N_FIELD}\n")
    (folder / f"bin.cgyro.ky_flux{suffix}").write_bytes(b"\1" * KY_FLUX_RECORD * nt_flux)
    (folder / f"bin.cgyro.freq{suffix}").write_bytes(b"\2" * FREQ_RECORD * nt)
    (folder / f"bin.cgyro.kxky_phi{suffix}").write_bytes(b"\3" * PHI_RECORD * nt)
    (folder / f"bin.cgyro.restart{suffix}").write_bytes(b"\4" * 1000)


def staged_reader(folder, suffix):
    '''A CGYROoutput with the private read directory read_using_cgyroplot builds (links canonical -> suffixed).'''
    out = CGYROoutput.__new__(CGYROoutput)
    out._read_tmpdir = tempfile.mkdtemp(prefix=".mitim_read_test_", dir=folder)
    out.folder_read = Path(out._read_tmpdir)
    for f in folder.glob(f"*{suffix}"):
        os.symlink(f, out.folder_read / f.name[:-len(suffix)])
    return out


def test_every_per_output_binary_is_trimmed_to_ky_flux():
    suffix = "_0.8085"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_run(d, suffix, nt=500, nt_flux=493)
        out = staged_reader(d, suffix)
        with contextlib.redirect_stdout(io.StringIO()):
            out._reconcile_time_vector()
        r = out.folder_read
        assert len(np.loadtxt(r / "out.cgyro.time")) == 493
        assert (r / "bin.cgyro.freq").stat().st_size == FREQ_RECORD * 493 and not (r / "bin.cgyro.freq").is_symlink()
        assert (r / "bin.cgyro.kxky_phi").stat().st_size == PHI_RECORD * 493
        assert (r / "bin.cgyro.restart").is_symlink()                     # not a per-output binary
        # the run folder keeps its files as they were
        assert (d / f"bin.cgyro.freq{suffix}").stat().st_size == FREQ_RECORD * 500
        assert len(np.loadtxt(d / f"out.cgyro.time{suffix}")) == 500
    print("PASS test_every_per_output_binary_is_trimmed_to_ky_flux")


def test_consistent_run_is_read_through_its_links():
    suffix = "_0.7765"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_run(d, suffix, nt=500, nt_flux=500)
        out = staged_reader(d, suffix)
        out._reconcile_time_vector()
        assert all((out.folder_read / n).is_symlink() for n in ("out.cgyro.time", "bin.cgyro.freq", "bin.cgyro.kxky_phi"))
    print("PASS test_consistent_run_is_read_through_its_links")


if __name__ == "__main__":
    test_every_per_output_binary_is_trimmed_to_ky_flux()
    test_consistent_run_is_read_through_its_links()
    print("\nALL TESTS PASSED")
