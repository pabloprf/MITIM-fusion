"""
DEV TEST: MINUET vs TRANSP sawteeth, HIGH-RESOLUTION TRANSP variant.

Same benchmark as minuet_vs_transp_02_sawteeth.py, but with TRANSP's output
cadence and current-diffusion stepping properly resolved. The transp_soft
template deliberately runs coarse (dtOut_ms = dtCurrentDiffusion_ms = 100
-> sedit/stedit = dtmaxb = 0.1 s): the CDF then has ~150 frames over 10 s
(one event pair per crash + 0.1 s ticks) and the CD equation itself advances
in up-to-0.1 s steps -- so the measured "crash completes over ~40 ms"
behavior and the -dW dip shapes are quantized at TRANSP's internal stepping,
not resolved. This variant:

  - dtOut_ms = 10            -> sedit/stedit = 10 ms (~1000 frames, ~1 GB
                                CDF at nzones=200; ~40 points per sawtooth
                                cycle, ~4 across the crash completion)
  - dtCurrentDiffusion_ms = 2 -> dtmaxb = 2 ms (TRANSP's own default), so
                                the crash-cycle dynamics are genuinely
                                integrated, not stepped at 0.1 s
  - MINUET n_save = 1001     -> 10 ms saves, matching cadence

Runs in a separate scratch folder (tag ..._hires): the coarse cached runs
are untouched. Questions this resolves: is the 40-ms post-crash completion
physical or 0.1-s stepping? What is the true -dW reset/dip shape? How does
the first ~0.5 s (limit-cycle level locking) proceed?

Run:  python tests/dev_tests/minuet_vs_transp_03_sawteeth_hires.py
"""

from minuet_vs_transp import run_benchmark

run_benchmark(
    sawteeth=True,
    cold_start=False,
    transp_overrides={"dtOut_ms": 10.0, "dtCurrentDiffusion_ms": 2.0},
    tag_extra="_hires",
    n_save=1001,
)
