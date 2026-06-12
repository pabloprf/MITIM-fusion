"""
TGLF in-process workflow — runs TGLF via ctypes (no subprocess, no files).

Two differences from a standard TGLF workflow:
  1. `in_process=True` in the constructor.
  2. `prep()` does not require a folder — pass only the input.gacode.

All methods (`run`, `read`, `run_scan`, `runScanTurbulenceDrives`) produce
zero file I/O: no directories are created, no input.tglf or out.tglf.gbflux
files are written.  Scan methods run all cases in parallel across all
available CPU cores via threads (each thread loads its own private .so copy
for independent Fortran globals).

Prerequisites — build the shared library once per machine:

    cd src/mitim_tools/simulation_tools/interfaces
    bash build_tglf_lib.sh
"""

import os
import time
import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
rhos = [0.5, 0.7]

# ---------------------------------------------------------------------------
# Timing helper — collects (label, wall_seconds) into TIMINGS so we can print
# a summary table at the end.  Use as: with timed("label"): ...
# ---------------------------------------------------------------------------
from contextlib import contextmanager
from mitim_tools.misc_tools import IOtools

TIMINGS: list[tuple[str, float]] = []

@contextmanager
def timed(label: str):
    print(f"\n>>> [{label}] start")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        TIMINGS.append((label, dt))
        print(f"<<< [{label}] done in {dt:.2f}s")

# Echo the BLAS / OpenMP thread environment so we can correlate timings with
# whatever thread pinning is in effect (the in_process modules pin to 1 by
# default; values may have been overridden by the calling shell).
print(
    "[env] MKL_NUM_THREADS={}  OPENBLAS_NUM_THREADS={}  OMP_NUM_THREADS={}  "
    "MKL_DYNAMIC={}  cpu_count={}".format(
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_DYNAMIC"),
        os.cpu_count(),
    )
)

# ── single-point runs (zero file I/O) ───────────────────────────────────────

# in_process=True: no folder needed — prep() works with just the input file
with timed("single: prep"):
    tglf = TGLFtools.TGLF(rhos=rhos, in_process=True)
    tglf.prep(input_gacode)

with timed("single: run SAT1"):
    tglf.run("run1/", code_settings="SAT1", cold_start=cold_start, forceIfcold_start=True)
    tglf.read(label="SAT1")

with timed("single: run SAT1 EM"):
    tglf.run("run2/", code_settings="SAT1", cold_start=cold_start, forceIfcold_start=True,
             extraOptions={"USE_BPER": True, "USE_BPAR": True})
    tglf.read(label="SAT1 EM")

print(f"\nQe (SAT1):    {[f'{r.Qe:.4f}' for r in tglf.results['SAT1']['output']]}")
print(f"Qe (SAT1 EM): {[f'{r.Qe:.4f}' for r in tglf.results['SAT1 EM']['output']]}")

# ── runScanTurbulenceDrives: subprocess vs in-process comparison ─────────────

DRIVES_KWARGS = dict(
    code_settings="SAT1",
    resolutionPoints=3,
    variablesDrives=["RLTS_1", "RLTS_2", "RLNS_1"],
    cold_start=cold_start,
    forceIfcold_start=True,
)

# subprocess run
folder_sub = __mitimroot__ / "tests" / "scratch" / "tglf_drives_sub"
if cold_start and folder_sub.exists():
    IOtools.shutil_rmtree(folder_sub)

with timed("scanTurbulenceDrives: subprocess"):
    tglf_sub = TGLFtools.TGLF(rhos=rhos, in_process=False)
    tglf_sub.prep(input_gacode, folder_sub)
    tglf_sub.runScanTurbulenceDrives(subfolder="drives", **DRIVES_KWARGS)

# in-process run (zero file I/O — no folder needed at any step)
with timed("scanTurbulenceDrives: in-process"):
    tglf_ip = TGLFtools.TGLF(rhos=rhos, in_process=True)
    tglf_ip.prep(input_gacode)
    tglf_ip.runScanTurbulenceDrives(subfolder="drives", **DRIVES_KWARGS)

# comparison
shared_labels = sorted(k for k in tglf_sub.results if k in tglf_ip.results)
print(f"\n{'='*65}")
print(f"Comparing {len(shared_labels)} scan labels across {len(rhos)} rhos")
print(f"{'='*65}")

all_ok = True
for label in shared_labels:
    out_sub = tglf_sub.results[label]["output"]
    out_ip  = tglf_ip.results[label]["output"]
    for i, rho in enumerate(rhos):
        Qe_s, Qe_i = out_sub[i].Qe, out_ip[i].Qe
        Qi_s, Qi_i = out_sub[i].Qi, out_ip[i].Qi
        # The subprocess reads fluxes from out.tglf.gbflux (limited file precision
        # ~4-5 significant figures); in-process retains full double precision.
        # rtol=1e-3 (0.1 %) is tight enough to catch physics disagreements.
        ok = np.isclose(Qe_s, Qe_i, rtol=1e-3) and np.isclose(Qi_s, Qi_i, rtol=1e-3)
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}]  {label:<40s}  rho={rho:.2f}"
            f"  Qe: {Qe_s:.6f} vs {Qe_i:.6f}"
            f"  Qi: {Qi_s:.6f} vs {Qi_i:.6f}"
        )

print(f"\n{'='*65}")
print(f"Overall: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
print(f"{'='*65}")

# ── timing summary ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Timing summary")
print(f"{'='*65}")
total = 0.0
for label, dt in TIMINGS:
    print(f"  {label:<40s}  {dt:8.2f}s")
    total += dt
print(f"  {'-'*40}  {'-'*9}")
print(f"  {'TOTAL':<40s}  {total:8.2f}s")
print(f"{'='*65}")
