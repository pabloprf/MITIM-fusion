"""
DEV TEST: MINUET vs TRANSP current diffusion, SAWTEETH OFF (SPARC PRD).

The clean benchmark: pure poloidal-field diffusion, no crashes on either
side (beat knob sawteeth=false parks t_sawtooth_on beyond time_end; no
MINUET trigger armed). Everything else matched — including the bootstrap
model (Sauter in both). See minuet_vs_transp.py for the full matching table
and the run mechanics.

cold_start=False resumes/reuses everything (MAESTRO checkpoints, MINUET
cache); set True to re-run from scratch (submits TRANSP again).

Run:  python tests/dev_tests/minuet_vs_transp_01_no_sawteeth.py
"""

from minuet_vs_transp import run_benchmark

run_benchmark(sawteeth=False, cold_start=False)
