"""
DEV TEST: MINUET vs TRANSP current diffusion, SAWTEETH ON (SPARC PRD).

Same benchmark as minuet_vs_transp_01_no_sawteeth.py but with Porcelli
sawteeth armed on both sides with matched constants (MINUET's trigger and
Porcelli-reconnection defaults were adopted from TRANSP/MITIM: c=[1,3,1,0.4],
island fraction 0.63, current-sheet width 0.05). Bootstrap is Sauter in
both. The comparison covers the sawtooth limit cycle (period, crash
statistics) on top of the secular current diffusion.

cold_start=False resumes/reuses everything (MAESTRO checkpoints, MINUET
cache); set True to re-run from scratch (submits TRANSP again).

Run:  python tests/dev_tests/minuet_vs_transp_02_sawteeth.py
"""

from minuet_vs_transp import run_benchmark

run_benchmark(sawteeth=True, cold_start=False)
