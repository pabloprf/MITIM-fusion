"""
test_portals_prune_restarts.py
==============================
Sanity tests for the opt-in PORTALS-CGYRO restart pruning
(transport.options.cgyro.prune_restarts_at_convergence and the mitim_prune_portals CLI,
both backed by prune_portals.prune_cgyro_restarts).

Pruning is irreversible, so these tests lock down the SURVIVAL set: the result evaluation's
restarts (including the simple-relax originals its symlinks point to) and every non-restart file.

Builds a synthetic run tree on disk -- no PORTALS run, no cluster. The result-index lookup
(resolve_result, which unpickles the run) is monkeypatched for the CLI tests.

Run as:

    python tests/dev_tests/test_portals_prune_restarts.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_modules.portals.scripts import prune_portals

RHOS = ["0.5000", "0.7000"]


def _touch(path, size_kb=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * (size_kb * 1024))


def _build_tree(root):
    '''
    Parallel simple-relax layout (n_traj = 2): portals_sr_ev_0 holds the real restarts in
    base_cgyro_plasma{0,1}; Evaluation.0/1 link to them. Evaluation.2/3 are BO iterations with
    real restarts (plus a .old) and an extra_cgyro point. Every evaluation carries non-restart files.
    '''
    sr = root / "Initialization" / "initialization_simple_relax" / "portals_sr_ev_0" / "transport_simulation_folder"
    for t in range(2):
        for rho in RHOS:
            _touch(sr / f"base_cgyro_plasma{t}" / f"bin.cgyro.restart_{rho}", 10)
            _touch(sr / f"base_cgyro_plasma{t}" / f"out.cgyro.time_{rho}")

    for i in range(4):
        tsf = root / "Execution" / f"Evaluation.{i}" / "transport_simulation_folder"
        _touch(tsf / "fluxes_turb.json")
        _touch(tsf / "base_cgyro" / f"input.cgyro_{RHOS[0]}")
        for rho in RHOS:
            dst = tsf / "base_cgyro" / f"bin.cgyro.restart_{rho}"
            if i < 2:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = sr / f"base_cgyro_plasma{i}" / f"bin.cgyro.restart_{rho}"
                dst.symlink_to(os.path.relpath(src, dst.parent))
            else:
                _touch(dst, 10)
                _touch(tsf / "base_cgyro" / f"bin.cgyro.restart_{rho}.old", 10)
        if i >= 2:
            _touch(tsf / "extra_cgyro" / "rho_0.7000" / "bin.cgyro.restart", 10)
            _touch(tsf / "extra_cgyro" / "rho_0.7000" / "out.cgyro.time")


def _restarts(root):
    return sorted(p for p in root.rglob("bin.cgyro.restart*"))


def _others(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("bin.cgyro.restart"))


def test_dry_run_deletes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_tree(root)
        before = _restarts(root)
        freed = prune_portals.prune_cgyro_restarts(root, keep_index=3, apply=False)
        assert _restarts(root) == before
        # Everything except Evaluation.3 (2 restarts + 2 .old + 1 extra, 10 KB each): 4 SR originals
        # + Evaluation.2 (5 files); symlinks of Evaluation.0/1 count 0 bytes
        assert freed >= (4 + 5) * 10 * 1024 and freed < (4 + 5) * 10 * 1024 + 4 * 1024, freed
    print("PASS test_dry_run_deletes_nothing")


def test_apply_keeps_only_result_bo_evaluation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_tree(root)
        others = _others(root)
        prune_portals.prune_cgyro_restarts(root, keep_index=3, apply=True)
        kept = _restarts(root)
        assert kept and all("Evaluation.3" in p.parts for p in kept), kept
        assert len(kept) == 5, kept
        assert _others(root) == others
    print("PASS test_apply_keeps_only_result_bo_evaluation")


def test_apply_keeps_symlinked_sr_result():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_tree(root)
        prune_portals.prune_cgyro_restarts(root, keep_index=1, apply=True)
        ev1 = root / "Execution" / "Evaluation.1" / "transport_simulation_folder" / "base_cgyro"
        for rho in RHOS:
            link = ev1 / f"bin.cgyro.restart_{rho}"
            assert link.is_symlink() and link.exists(), f"dangling or missing {link}"
            assert link.read_bytes()
        remaining = {p.resolve() for p in _restarts(root)}
        expected = {(ev1 / f"bin.cgyro.restart_{rho}").resolve() for rho in RHOS}
        assert remaining == expected, remaining
        # Evaluation.0's links are gone, not left dangling
        assert not list((root / "Execution" / "Evaluation.0").rglob("bin.cgyro.restart*"))
    print("PASS test_apply_keeps_symlinked_sr_result")


def _run_main(argv, index=3, converged=True):
    original_resolve, original_argv = prune_portals.resolve_result, sys.argv
    prune_portals.resolve_result = lambda folder, keep="best": (index, converged)
    sys.argv = ["mitim_prune_portals"] + argv
    try:
        prune_portals.main()
    finally:
        prune_portals.resolve_result, sys.argv = original_resolve, original_argv


def test_cli_dry_run_then_apply():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_tree(root)
        before = _restarts(root)
        _run_main([str(root)])
        assert _restarts(root) == before
        _run_main([str(root), "--apply"])
        kept = _restarts(root)
        assert len(kept) == 5 and all("Evaluation.3" in p.parts for p in kept), kept
    print("PASS test_cli_dry_run_then_apply")


def test_cli_refuses_unconverged_without_force():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_tree(root)
        before = _restarts(root)
        _run_main([str(root), "--apply"], converged=False)
        assert _restarts(root) == before
        _run_main([str(root), "--apply", "--force"], converged=False)
        assert len(_restarts(root)) == 5
    print("PASS test_cli_refuses_unconverged_without_force")


if __name__ == "__main__":
    test_dry_run_deletes_nothing()
    test_apply_keeps_only_result_bo_evaluation()
    test_apply_keeps_symlinked_sr_result()
    test_cli_dry_run_then_apply()
    test_cli_refuses_unconverged_without_force()
    print("\nAll PORTALS restart-prune tests passed")
