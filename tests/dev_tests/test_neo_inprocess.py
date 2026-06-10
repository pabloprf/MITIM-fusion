"""
test_neo_inprocess.py
=====================
Benchmark and correctness test for the NEO in-process (ctypes) execution path
vs. the standard subprocess path.

Usage
-----
    # From the MITIM-fusion root, with GACODE_ROOT set:
    pixi run -- python tests/test_neo_inprocess.py

    # Optional: build the extension first (once per machine):
    cd src/mitim_tools/simulation_tools/interfaces && bash build_neo_lib.sh

What the script does
--------------------
1. Generates ``input.neo.gen`` from a NEO regression ``input.neo`` via
   neo_parse.py.
2. Runs the standard subprocess path (``neo -e .``) N times and records
   wall time.
3. Runs the in-process path (``c_neo_run()`` via ctypes) N times and records
   wall time.
4. Prints a side-by-side table of flux values and their relative differences.
5. Prints the mean call time for each path and the speedup factor.
6. Exits non-zero if any flux value differs by more than ``RTOL=1e-4``.
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
N_CALLS = 3            # NEO is much heavier than TGLF — keep this small
RTOL    = 1e-4         # relative tolerance for correctness check
ATOL    = 1e-10        # absolute floor: values below this are treated as zero
                       # (NEO momentum flux can drop to ~1e-13 where the file's
                       # 5-digit printf is dominated by numerical noise)
# ---------------------------------------------------------------------------

# Locate test data — use a NEO regression input from the gacode tree.
_MITIMROOT = Path(__file__).parent.parent
GACODE_ROOT = os.environ.get("GACODE_ROOT")
if not GACODE_ROOT:
    sys.exit("ERROR: GACODE_ROOT is not set. Source your gacode_setup script.")

# Pick a small / fast NEO regression case.
INPUT_NEO = Path(GACODE_ROOT) / "neo" / "tools" / "input" / "reg02" / "input.neo"
SCRATCH    = _MITIMROOT / "tests" / "scratch" / "neo_inprocess_test"

if not INPUT_NEO.exists():
    sys.exit(f"ERROR: test input file not found: {INPUT_NEO}")


# ---------------------------------------------------------------------------
# Helper: read out.neo.transport_flux into a flat array (ordered)
# ---------------------------------------------------------------------------
def _parse_transport_flux(filepath: Path) -> dict:
    """
    Return a dict with 'dke', 'gv', 'tgyro' each → list of (Z, G, Q, M) rows
    in the order they appear in the file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"NEO output not found: {filepath}")

    sections: dict[str, list[list[float]]] = {}
    current = None
    with open(filepath) as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("#"):
                if "pflux_dke"     in s: current = "dke";   sections[current] = []
                elif "pflux_gv"    in s: current = "gv";    sections[current] = []
                elif "pflux_tgyro" in s: current = "tgyro"; sections[current] = []
                # Other comment lines (units, headers) leave `current` alone.
            elif current is not None and s and not s.startswith("("):
                vals = s.split()
                if len(vals) == 4:
                    sections[current].append([float(v) for v in vals])
    return sections


def _flatten(sections: dict) -> tuple[list[str], np.ndarray]:
    """Flatten parsed sections into labelled arrays for side-by-side comparison."""
    labels: list[str] = []
    values: list[float] = []
    for sec_name in ("dke", "gv", "tgyro"):
        rows = sections.get(sec_name, [])
        for i, (Z, G, Q, M) in enumerate(rows):
            labels.append(f"{sec_name}.G[{int(Z):+d}]")
            values.append(G)
            labels.append(f"{sec_name}.Q[{int(Z):+d}]")
            values.append(Q)
            labels.append(f"{sec_name}.M[{int(Z):+d}]")
            values.append(M)
    return labels, np.array(values)


# ---------------------------------------------------------------------------
# Helper: run the standard subprocess path once, return parsed sections
# ---------------------------------------------------------------------------
def _run_subprocess(work_dir: Path) -> dict:
    """Run ``neo -e .`` in work_dir, return parsed transport_flux sections."""
    result = subprocess.run(
        ["neo", "-e", "."],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("  STDERR:", result.stderr[-500:] if result.stderr else "(none)")
        raise RuntimeError(f"neo subprocess failed (rc={result.returncode})")

    return _parse_transport_flux(work_dir / "out.neo.transport_flux")


# ---------------------------------------------------------------------------
# Helper: run the in-process path once, return a "sections-like" dict from
# the in-memory output dict so we can compare apples-to-apples.
# ---------------------------------------------------------------------------
def _outputs_to_sections(out: dict, Z_per_species: list[float], gb: tuple) -> dict:
    """
    Convert NEOInProcess.run() output → same shape as the file parser,
    applying the GB normalization (pgb, egb, mgb) used by NEO when writing
    out.neo.transport_flux.  The interface variables themselves are in
    "norm" units (Gamma/Gamma_norm etc.), so we divide by the gb factors
    to match the file output exactly.
    """
    ns = out["ns"]
    Z = list(Z_per_species)[:ns]
    pgb, egb, mgb = gb
    sections = {"dke": [], "gv": [], "tgyro": []}
    for k in range(ns):
        sections["dke"].append([
            Z[k],
            out["pflux_dke"][k]    / pgb,
            out["efluxtot_dke"][k] / egb,
            out["mflux_dke"][k]    / mgb,
        ])
        sections["gv"].append([
            Z[k],
            out["pflux_gv"][k]    / pgb,
            out["efluxtot_gv"][k] / egb,
            out["mflux_gv"][k]    / mgb,
        ])
        sections["tgyro"].append([
            Z[k],
            (out["pflux_dke"][k]    + out["pflux_gv"][k])    / pgb,
            (out["efluxtot_dke"][k] + out["efluxtot_gv"][k]) / egb,
            (out["mflux_dke"][k]    + out["mflux_gv"][k])    / mgb,
        ])
    return sections


def _read_gb_from_gen(gen_file: Path) -> tuple:
    """
    Compute (pgb, egb, mgb) the same way NEO does when writing
    transport_flux: pgb = n_e * rho^2 * T_e^1.5, etc.  When AE_FLAG=1 the
    electron density/temperature come from DENS_AE/TEMP_AE.
    """
    params: dict[str, float] = {}
    with open(gen_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    params[parts[1]] = float(parts[0])
                except ValueError:
                    pass  # non-numeric (e.g. RBF_DIR=d3d_4)

    rho     = params.get("RHO_STAR", 1e-3)
    ae_flag = int(params.get("AE_FLAG", 0))
    if ae_flag == 1:
        dens_e = params.get("DENS_AE", 1.0)
        temp_e = params.get("TEMP_AE", 1.0)
    else:
        # find the species with Z = -1 (electrons)
        ns = int(params.get("N_SPECIES", 1))
        dens_e, temp_e = 1.0, 1.0
        for i in range(1, ns + 1):
            if int(params.get(f"Z_{i}", 0)) == -1:
                dens_e = params.get(f"DENS_{i}", 1.0)
                temp_e = params.get(f"TEMP_{i}", 1.0)
                break

    pgb = dens_e * rho**2 * temp_e**1.5
    egb = dens_e * rho**2 * temp_e**2.5
    mgb = dens_e * rho**2 * temp_e**2.0
    return (pgb, egb, mgb)


def _read_Z_from_gen(gen_file: Path, ns: int) -> list[float]:
    """
    The .gen file from neo_parse stores 'value KEY' per line. Walk it and
    pull Z_1..Z_ns so the in-process output can be labelled with the same
    species charges as the subprocess output for comparison.
    """
    z = {}
    with open(gen_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1].startswith("Z_") and parts[1][2:].isdigit():
                z[int(parts[1][2:])] = float(parts[0])
    return [z[i + 1] for i in range(ns) if (i + 1) in z]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("NEO in-process vs subprocess benchmark")
    print("=" * 70)
    print(f"  Test input: {INPUT_NEO}")

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

    shutil.copy(INPUT_NEO, work_dir_sub / "input.neo")
    shutil.copy(INPUT_NEO, work_dir_inp / "input.neo")

    # ------------------------------------------------------------------
    # Generate input.neo.gen for the in-process path
    # ------------------------------------------------------------------
    print("\n[1/4] Generating input.neo.gen ...")
    from mitim_tools.simulation_tools.interfaces.neo_inprocess import (
        NEOInProcess,
        generate_input_gen,
    )
    gen_file = generate_input_gen(work_dir_inp / "input.neo")
    print(f"      -> {gen_file}")

    # ------------------------------------------------------------------
    # Load the in-process runner (loads the .so once)
    # ------------------------------------------------------------------
    print("\n[2/4] Loading neo_lib extension ...")
    try:
        runner = NEOInProcess()
        print("      -> OK")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Correctness: run both paths once and compare
    # ------------------------------------------------------------------
    print("\n[3/4] Correctness check ...")

    sections_sub = _run_subprocess(work_dir_sub)
    out_inp      = runner.run(gen_file)

    if out_inp["error_status"] != 0:
        print(f"  WARNING: in-process error_status = {out_inp['error_status']}")

    Z_per_species = _read_Z_from_gen(gen_file, out_inp["ns"])
    gb_factors    = _read_gb_from_gen(gen_file)
    sections_inp  = _outputs_to_sections(out_inp, Z_per_species, gb_factors)

    labels_sub, vals_sub = _flatten(sections_sub)
    labels_inp, vals_inp = _flatten(sections_inp)

    # Align labels (the file parser picks up species in file order, in-process
    # uses the input dict order — both should match because they come from
    # the same .gen file).
    n = min(len(vals_sub), len(vals_inp))
    vals_sub = vals_sub[:n]
    vals_inp = vals_inp[:n]
    labels   = labels_sub[:n]

    max_rdiff = 0.0
    header = f"{'Quantity':16s}  {'Subprocess':>14s}  {'In-process':>14s}  {'|rel diff|':>12s}"
    print(f"\n{header}")
    print("-" * len(header))
    for lbl, vs, vi in zip(labels, vals_sub, vals_inp):
        denom = max(abs(vs), abs(vi), 1e-20)
        rdiff = abs(vs - vi) / denom
        # Skip values that are below the absolute floor — they are physically
        # zero and any difference is dominated by ASCII round-trip noise.
        if max(abs(vs), abs(vi)) < ATOL:
            flag = " (≈0, skipped)"
        elif rdiff > RTOL:
            flag = " <-- MISMATCH"
            max_rdiff = max(max_rdiff, rdiff)
        else:
            flag = ""
            max_rdiff = max(max_rdiff, rdiff)
        print(f"  {lbl:14s}  {vs:14.6e}  {vi:14.6e}  {rdiff:12.2e}{flag}")

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
        runner.run(gen_file)
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
