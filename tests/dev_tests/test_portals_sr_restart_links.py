"""
test_portals_sr_restart_links.py
================================
The serial simple-relax promotion (Initialization/initialization_simple_relax/portals_sr_ev_<s>/ ->
Execution/Evaluation.<s>/) hard-links the CGYRO/GX restart blobs instead of copying them, so a
PORTALS-CGYRO run holds each start-up restart once on disk.

These tests lock down:
  - restarts become hard links (symlink when the filesystem refuses), everything else is a copy;
  - removing one name (pruning, --cold, a re-promotion) leaves the other intact;
  - a retrieval-style replace of the Execution name never writes through to the Initialization original;
  - prune_cgyro_restarts keeps both names of the kept evaluation and counts each file's data once.

Builds synthetic trees on disk -- no PORTALS run, no cluster.

Run as:

    python tests/dev_tests/test_portals_sr_restart_links.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_modules.portals.utils import PORTALSoptimization
from mitim_modules.portals.scripts import prune_portals

RHOS = ["0.5000", "0.7000"]
RESTART_KB = 10


def _touch(path, size_kb=1, fill=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fill * (size_kb * 1024))


def _build_sr_step(root, s):
    '''One serial simple-relax step folder, laid out as SIMtools leaves it (keep_files "all").'''
    tsf = root / "Initialization" / "initialization_simple_relax" / f"portals_sr_ev_{s}" / "transport_simulation_folder"
    _touch(tsf / "fluxes_turb.json")
    for rho in RHOS:
        _touch(tsf / "base_cgyro" / f"bin.cgyro.restart_{rho}", RESTART_KB)
        _touch(tsf / "base_cgyro" / f"bin.cgyro.restart_{rho}.old", RESTART_KB)
        _touch(tsf / "base_cgyro" / f"out.cgyro.time_{rho}")
        _touch(tsf / "base_gx" / f"gxplasma.restart.nc_{rho}", RESTART_KB)
    return tsf


def _promote(root, s):
    '''What initialization_simple_relax does for n_traj == 1.'''
    dst = root / "Execution" / f"Evaluation.{s}" / "transport_simulation_folder"
    shutil.copytree(_build_sr_step(root, s), dst, symlinks=True, copy_function=PORTALSoptimization._link_restarts_copy_rest)
    return root / "Initialization" / "initialization_simple_relax" / f"portals_sr_ev_{s}" / "transport_simulation_folder", dst


def _pairs(src, dst):
    return [(p, dst / p.relative_to(src)) for p in src.rglob("*") if p.is_file()]


def test_restarts_hard_linked_rest_copied():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = _promote(Path(tmp), 0)
        pairs = _pairs(src, dst)
        assert len(pairs) == 1 + 4 * len(RHOS), pairs
        for a, b in pairs:
            restart = a.name.startswith(PORTALSoptimization._RESTART_PREFIXES)
            assert not b.is_symlink(), b
            assert os.path.samefile(a, b) == restart, (a, b)
            assert a.stat().st_nlink == (2 if restart else 1), a
            assert a.read_bytes() == b.read_bytes()
    print("PASS test_restarts_hard_linked_rest_copied")


def test_symlink_fallback_when_hard_links_refused():
    original = os.link
    def refuse(*args, **kwargs):
        raise OSError(1, "Operation not permitted")
    os.link = refuse
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = _promote(Path(tmp), 0)
            for a, b in _pairs(src, dst):
                restart = a.name.startswith(PORTALSoptimization._RESTART_PREFIXES)
                assert b.is_symlink() == restart, b
                if restart:
                    assert not os.path.isabs(os.readlink(b)), os.readlink(b)
                    assert os.path.samefile(a, b)
    finally:
        os.link = original
    print("PASS test_symlink_fallback_when_hard_links_refused")


def test_removing_one_name_keeps_the_other():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, dst = _promote(root, 0)
        restart = f"base_cgyro/bin.cgyro.restart_{RHOS[0]}"
        data = (src / restart).read_bytes()
        shutil.rmtree(root / "Initialization")
        assert (dst / restart).read_bytes() == data
        assert (dst / restart).stat().st_nlink == 1
        # Re-promotion of the same step wipes the Execution copy first (rmtree) and links again
        shutil.rmtree(dst)
        src, dst = _promote(root, 0)
        assert os.path.samefile(src / restart, dst / restart)
        shutil.rmtree(root / "Execution")
        assert (src / restart).read_bytes() == data
    print("PASS test_removing_one_name_keeps_the_other")


def test_retrieval_replace_does_not_write_through():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, dst = _promote(root, 0)
        restart = f"base_cgyro/bin.cgyro.restart_{RHOS[0]}"
        original = (src / restart).read_bytes()
        # SIMtools retrieval: unlink the destination, then move the fetched file in
        fetched = root / "tmp_fetch" / "bin.cgyro.restart"
        _touch(fetched, RESTART_KB, fill=b"y")
        (dst / restart).unlink(missing_ok=True)
        fetched.replace(dst / restart)
        assert (src / restart).read_bytes() == original
        assert (dst / restart).read_bytes() != original
    print("PASS test_retrieval_replace_does_not_write_through")


def _serial_run(root, n_sr=3, n_bo=2):
    for s in range(n_sr):
        _promote(root, s)
    for i in range(n_sr, n_sr + n_bo):
        for rho in RHOS:
            _touch(root / "Execution" / f"Evaluation.{i}" / "transport_simulation_folder" / "base_cgyro" / f"bin.cgyro.restart_{rho}", RESTART_KB)


def _cgyro_restarts(root):
    return sorted(root.rglob("bin.cgyro.restart*"))


def test_prune_keeps_hard_linked_sr_result():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _serial_run(root)
        freed = prune_portals.prune_cgyro_restarts(root, keep_index=1, apply=False)
        # SR steps 0 and 2 (4 restart files each, data counted once over 2 names) + 2 BO evaluations (2 each)
        assert freed == (2 * 4 + 2 * 2) * RESTART_KB * 1024, freed
        prune_portals.prune_cgyro_restarts(root, keep_index=1, apply=True)
        kept = _cgyro_restarts(root)
        assert len(kept) == 8 and all(("Evaluation.1" in p.parts) or ("portals_sr_ev_1" in p.parts) for p in kept), kept
        assert all(p.stat().st_nlink == 2 for p in kept)
    print("PASS test_prune_keeps_hard_linked_sr_result")


def test_prune_bo_result_frees_sr_data_once():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _serial_run(root)
        freed = prune_portals.prune_cgyro_restarts(root, keep_index=4, apply=True)
        assert freed == (3 * 4 + 1 * 2) * RESTART_KB * 1024, freed
        kept = _cgyro_restarts(root)
        assert len(kept) == 2 and all("Evaluation.4" in p.parts for p in kept), kept
    print("PASS test_prune_bo_result_frees_sr_data_once")


if __name__ == "__main__":
    test_restarts_hard_linked_rest_copied()
    test_symlink_fallback_when_hard_links_refused()
    test_removing_one_name_keeps_the_other()
    test_retrieval_replace_does_not_write_through()
    test_prune_keeps_hard_linked_sr_result()
    test_prune_bo_result_frees_sr_data_once()
    print("\nAll PORTALS simple-relax restart-link tests passed")
