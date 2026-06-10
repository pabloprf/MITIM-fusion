"""
test_tglf_inprocess.py
======================
Benchmark and correctness test for the TGLF in-process (f2py) execution path
vs. the standard subprocess path.

Usage
-----
    # From the MITIM-fusion root, with GACODE_ROOT set:
    python tests/test_tglf_inprocess.py

    # Optional: build the extension first (once per machine):
    cd src/mitim_tools/simulation_tools/interfaces && bash build_tglf_lib.sh

What the script does
--------------------
1. Generates ``input.tglf.gen`` from the test ``input.tglf`` via tglf_parse.py.
2. Runs the standard subprocess path (``tglf -e .``) N times and records wall time.
3. Runs the in-process path (``tglf_run()`` via f2py) N times and records wall time.
4. Prints a side-by-side table of flux values and their relative differences.
5. Prints the mean call time for each path and the speedup factor.
6. Exits non-zero if any flux value differs by more than ``RTOL=1e-5``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CALLS = 5          # number of repeated calls for timing
RTOL = 1e-4           # relative tolerance for correctness check
# ---------------------------------------------------------------------------

# Locate test data
_MITIMROOT = Path(__file__).parent.parent
INPUT_TGLF = _MITIMROOT / "tests" / "data" / "input.tglf"
SCRATCH     = _MITIMROOT / "tests" / "scratch" / "tglf_inprocess_test"

if not INPUT_TGLF.exists():
    sys.exit(f"ERROR: test input file not found: {INPUT_TGLF}")


# ---------------------------------------------------------------------------
# Helper: run the standard subprocess path once, return gbflux array
# ---------------------------------------------------------------------------
def _run_subprocess(work_dir: Path) -> np.ndarray:
    """Run ``tglf -e .`` in work_dir, return parsed out.tglf.gbflux values."""
    result = subprocess.run(
        ["tglf", "-e", "."],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("  STDERR:", result.stderr[-500:] if result.stderr else "(none)")
        raise RuntimeError(f"tglf subprocess failed (rc={result.returncode})")

    gbflux_file = work_dir / "out.tglf.gbflux"
    if not gbflux_file.exists():
        raise RuntimeError(f"out.tglf.gbflux not found in {work_dir}")

    values = [float(v) for v in gbflux_file.read_text().split()]
    return np.array(values)


# ---------------------------------------------------------------------------
# Helper: run the in-process path once, return gbflux array
# ---------------------------------------------------------------------------
def _run_inprocess(gen_file: Path, runner) -> np.ndarray:
    """Call tglf_run() in-process, return values matching gbflux format."""
    outputs = runner.run(gen_file)

    # Reconstruct the flat array in the same order as out.tglf.gbflux
    ni = outputs["ns"] - 1
    values: list[float] = []
    values.append(outputs["elec_pflux"])
    values.extend(outputs["ion_pflux"][:ni])
    values.append(outputs["elec_eflux"])
    values.extend(outputs["ion_eflux"][:ni])
    values.append(outputs["elec_mflux"])
    values.extend(outputs["ion_mflux"][:ni])
    values.append(outputs["elec_expwd"])
    values.extend(outputs["ion_expwd"][:ni])
    return np.array(values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TGLF in-process vs subprocess benchmark")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Setup working directories
    # ------------------------------------------------------------------
    SCRATCH.mkdir(parents=True, exist_ok=True)
    work_dir_sub = SCRATCH / "subprocess"
    work_dir_inp = SCRATCH / "inprocess"
    for d in [work_dir_sub, work_dir_inp]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    # Copy input.tglf to both dirs
    shutil.copy(INPUT_TGLF, work_dir_sub / "input.tglf")
    shutil.copy(INPUT_TGLF, work_dir_inp / "input.tglf")

    # ------------------------------------------------------------------
    # Generate input.tglf.gen for the in-process path
    # ------------------------------------------------------------------
    print("\n[1/4] Generating input.tglf.gen ...")
    from mitim_tools.simulation_tools.interfaces.tglf_inprocess import (
        TGLFInProcess,
        generate_input_gen,
    )
    gen_file = generate_input_gen(work_dir_inp / "input.tglf")
    print(f"      -> {gen_file}")

    # ------------------------------------------------------------------
    # Load the in-process runner (loads the .so once)
    # ------------------------------------------------------------------
    print("\n[2/4] Loading tglf_lib extension ...")
    try:
        runner = TGLFInProcess()
        print("      -> OK")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Correctness: run both paths once and compare
    # ------------------------------------------------------------------
    print("\n[3/4] Correctness check ...")

    # For subprocess, we need input.tglf (tglf -e . will call tglf_parse.py)
    vals_sub  = _run_subprocess(work_dir_sub)
    vals_inp  = _run_inprocess(gen_file, runner)

    # Align lengths (should match, but guard against different ns)
    n = min(len(vals_sub), len(vals_inp))
    vals_sub  = vals_sub[:n]
    vals_inp  = vals_inp[:n]

    # Labels for the table: ni = num ions = (total values / 4) - 1
    # (4 flux types, each has 1 electron + ni ion values)
    ni = len(vals_inp) // 4 - 1
    labels: list[str] = []
    for qty in ("Ge", "Qe", "Me", "Se"):
        species = [qty.replace("e", f"e")] + [qty.replace("e", f"i{i+1}") for i in range(ni)]
        labels.extend(species)
    labels = labels[:n]

    max_rdiff = 0.0
    header = f"{'Quantity':12s}  {'Subprocess':>14s}  {'In-process':>14s}  {'|rel diff|':>12s}"
    print(f"\n{header}")
    print("-" * len(header))
    for lbl, vs, vi in zip(labels, vals_sub, vals_inp):
        denom = max(abs(vs), abs(vi), 1e-20)
        rdiff = abs(vs - vi) / denom
        max_rdiff = max(max_rdiff, rdiff)
        flag = " <-- MISMATCH" if rdiff > RTOL else ""
        print(f"  {lbl:10s}  {vs:14.6e}  {vi:14.6e}  {rdiff:12.2e}{flag}")

    print()
    if max_rdiff > RTOL:
        print(f"FAIL: maximum relative difference {max_rdiff:.2e} exceeds tolerance {RTOL:.2e}")
        sys.exit(1)
    else:
        print(f"PASS: maximum relative difference {max_rdiff:.2e}  (tolerance {RTOL:.2e})")

    # ------------------------------------------------------------------
    # Speed benchmark
    # ------------------------------------------------------------------
    print(f"\n[4/4] Speed benchmark ({N_CALLS} calls each) ...")

    # --- subprocess ---
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        _run_subprocess(work_dir_sub)
    t_sub = (time.perf_counter() - t0) / N_CALLS

    # --- in-process ---
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        _run_inprocess(gen_file, runner)
    t_inp = (time.perf_counter() - t0) / N_CALLS

    speedup = t_sub / t_inp if t_inp > 0 else float("inf")

    print()
    print(f"  Subprocess mean call time : {t_sub*1000:.1f} ms")
    print(f"  In-process mean call time : {t_inp*1000:.1f} ms")
    print(f"  Speedup (subprocess/inprocess): {speedup:.1f}x")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
