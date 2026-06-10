"""
NEO in-process workflow — runs NEO via ctypes (no subprocess, no files).

Two differences from a standard NEO workflow:
  1. `in_process=True` in the constructor.
  2. `prep()` does not require a folder — pass only the input.gacode.

All methods (`run`, `read`, `run_scan`) produce zero file I/O: no
directories are created, no input.neo or out.neo.* files are written.
Scan methods run all cases in parallel across all available CPU cores
via threads (each thread loads its own private .so copy for independent
Fortran globals).

Prerequisites — build the shared library once per machine:

    cd src/mitim_tools/simulation_tools/interfaces
    bash build_neo_lib.sh
"""

import os
import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
rhos = [0.8, 0.9]

# ── single-point runs (zero file I/O) ───────────────────────────────────────

# in_process=True: no folder needed — prep() works with just the input file
neo = NEOtools.NEO(rhos=rhos, in_process=True)
neo.prep(input_gacode)

neo.run("neo1/", code_settings="Sonic", cold_start=cold_start, forceIfcold_start=True)
neo.read(label="NEO default")

neo.run("neo2/", code_settings="Sonic", cold_start=cold_start, forceIfcold_start=True,
        extraOptions={"N_ENERGY": 5, "N_XI": 11, "N_THETA": 11})
neo.read(label="NEO low res")

neo.run("neo3/", code_settings="Sonic", cold_start=cold_start, forceIfcold_start=True,
        extraOptions={"N_ENERGY": 5, "N_XI": 11, "N_THETA": 11},
        multipliers={"DLNTDR_1": 1.5})
neo.read(label="NEO low res + 50% aLTi1")

print(f"\nQe (default):     {[f'{r.Qe:.4e}' for r in neo.results['NEO default']['output']]}")
print(f"Qe (low res):     {[f'{r.Qe:.4e}' for r in neo.results['NEO low res']['output']]}")
print(f"Qe (low res +50): {[f'{r.Qe:.4e}' for r in neo.results['NEO low res + 50% aLTi1']['output']]}")
print(f"Qi (default):     {[f'{r.Qi:.4e}' for r in neo.results['NEO default']['output']]}")
print(f"Qi (low res):     {[f'{r.Qi:.4e}' for r in neo.results['NEO low res']['output']]}")
print(f"Qi (low res +50): {[f'{r.Qi:.4e}' for r in neo.results['NEO low res + 50% aLTi1']['output']]}")

# ── run_scan: subprocess vs in-process comparison ──────────────────────────

SCAN_KWARGS = dict(
    cold_start=cold_start,
    forceIfcold_start=True,
    variable="DLNTDR_1",
    varUpDown=np.linspace(0.5, 1.5, 4).tolist(),
    code_settings="Sonic",
    extraOptions={"N_ENERGY": 5, "N_XI": 11, "N_THETA": 11},
)

# subprocess run
folder_sub = __mitimroot__ / "tests" / "scratch" / "neo_scan_sub"
if cold_start and folder_sub.exists():
    os.system(f"rm -r {folder_sub.resolve()}")

neo_sub = NEOtools.NEO(rhos=rhos, in_process=False)
neo_sub.prep(input_gacode, folder_sub)
neo_sub.run_scan(subfolder="scan_dltdr1", **SCAN_KWARGS)

# in-process run (zero file I/O — no folder needed at any step)
neo_ip = NEOtools.NEO(rhos=rhos, in_process=True)
neo_ip.prep(input_gacode)
neo_ip.run_scan(subfolder="scan_dltdr1", **SCAN_KWARGS)

# comparison
shared_labels = sorted(k for k in neo_sub.results if k in neo_ip.results)
print(f"\n{'=' * 75}")
print(f"Comparing {len(shared_labels)} scan labels across {len(rhos)} rhos")
print(f"{'=' * 75}")

# NEO momentum flux is genuinely tiny (≲1e-12 GB) for ITG-style cases, so
# rtol on Qe/Qi is the meaningful check; momentum is dominated by noise.
RTOL = 5e-3
all_ok = True
for label in shared_labels:
    out_sub = neo_sub.results[label]["output"]
    out_ip  = neo_ip.results[label]["output"]
    for i, rho in enumerate(rhos):
        Qe_s, Qe_i = out_sub[i].Qe, out_ip[i].Qe
        Qi_s, Qi_i = out_sub[i].Qi, out_ip[i].Qi
        ok = np.isclose(Qe_s, Qe_i, rtol=RTOL) and np.isclose(Qi_s, Qi_i, rtol=RTOL)
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}]  {label:<32s}  rho={rho:.2f}"
            f"  Qe: {Qe_s:.4e} vs {Qe_i:.4e}"
            f"  Qi: {Qi_s:.4e} vs {Qi_i:.4e}"
        )

print(f"\n{'=' * 75}")
print(f"Overall: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
print(f"{'=' * 75}")
