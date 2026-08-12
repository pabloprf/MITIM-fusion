# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **SOL / separatrix estimates collapsed into `mitim_state.calculate_sol()`**: always computes
    the legacy 2-point `Te_lcfs_estimate` (now DEPRECATED — its `Bp = eps*Bt/q95` is a rough averaged
    poloidal field, ~2.4x below the true outboard-midplane value; it will be removed in the future),
    the new `Te_lcfs_2pt` (same model with the exact `Bpol_omp` from the poloidal-flux gradient, also
    stored) and, optionally (`lengyel=True`), the extended-Lengyel model via `calculate_sol_lengyel()`
    (`Te_lcfs_lengyel`; optional `[lengyel]` extra — degrades gracefully to NaN if missing; `mode='seeded'` = the package's detachment-seeded driver, `mode='clean'` = unseeded upstream leg, pure conduction at the state's own Zeff). The clean mode is fully package-native
    (`Lengyel.run_forward()`: registered extended-lengyel/cfspopcon algorithms composed per
    `templates/input.lengyel_clean.controls.yaml` — no detachment solve, no seeding, no radas data
    needed; Brunner lambda_q convention, documented in the template header). Teaching
    script: `tests/capability_tests/profiles_02_sol_estimates.py`.

*   💥 **MAESTRO scan interpretation** (`mitim_modules.maestro.utils.MAESTROscan` + new
    `mitim_plot_maestro_scan` CLI): scan-level analysis of a folder of `case_*` MAESTRO runs —
    seed-spread violin panels of performance scalars (seed-only spread; deterministic scan inputs
    split into x-axis/color series, benchmark runs overlaid at interpolated positions), per-seed
    profile-spread figures with per-seed Pfus readouts, per-beat evolution traces, cumulative
    beat timing (wall time or CPU-hours) with a reference run's chain, and a per-case PDF report.

*   💥 **fGped with geqdsk initialization no longer requires explicit Ip/a**: when
    `initialization_type: geqdsk`, any of the two left `null` is read from the equilibrium file
    itself (|CURRENT| in MA; a = separatrix half-width) for the fGped -> neped conversion —
    removing namelist entries that were redundant with (and could silently disagree with) the
    geqdsk. Explicit values still take precedence; other initialization types are unchanged.


*   💥 **MAESTRO BetaN can be a confinement-quality string**: `profiles_initialization.parameters.BetaN`
    now accepts `"H98y2"` or `"H89p"` (optionally with a target, e.g. `"H98y2=1.1"`) instead of a fixed
    number: an achievable BetaN is estimated by inverting the corresponding tau_E scaling with the
    engineering parameters (loss power = Paux only, so deliberately on the low side), so engineering
    scans (e.g. in Ip) no longer need a per-case guess that can break initialization when unreachable.
    Also fixed a crash in the initializer pressure guess when neither profiles nor BetaN were provided
    (now falls back to 1.0 MPa with a warning), and de-duplicated the FiBE copy of that formula.

*   💥 **MAESTRO confinement beat: invertible isothermal-edge guard** (`sep_max_frac`): instead of
    flooring the H-servo at `Te_bc >= 1.2*Tesep`, the beat can now let `Te_bc` go arbitrarily low and
    cap the APPLIED separatrix Te/Ti at `sep_max_frac * Te_bc` (edge stays monotone, TRANSP-safe),
    with `Te_bc_min_Tesep_factor: null` disabling the dynamic floor. An H-target that demands an edge
    at/below the physical (e.g. Lengyel) Tesep then shows up as a result, not a rail.

*   💥 **EPED physics-based stability rule** (`postprocess_eped` rule `'W'`, exposed in `EPED.read()`
    and the MAESTRO eped-beat knob `stability_rule`): the pedestal can now be selected with the EPED1
    diamagnetic criterion gamma > C*omega_*pi(n)/2, with omega_*pi the HALF-maximum of the ion
    diamagnetic frequency across the barrier (Snyder PoP 2009 / NF 2011) so that C = 1 is EPED1 as
    published — the threshold grows ~linearly with toroidal mode number and the answer converges in
    the mode-set ceiling, unlike the flat gamma/omega_A > 0.03 cut in deeply ballooning-limited
    pedestals. `stability_threshold = None` now resolves per-rule (flat: 0.03; 'W': C = 1), a
    flat-like C warns, the companion gacode state is sanity-checked against the EPED scalars, and an
    optional `consecutive_heights` knob (default 1 = plain first crossing) can reject selections
    carried by isolated unconverged-ELITE spikes. EPED runs also keep per-height TOQ/ELITE work
    directories by default for post-mortems (`clean_intermediate_files=True` restores the old
    cleanup), and a launch whose num_heights x num_modes would overflow the EPED runner's silent
    1024-job table (excess ELITE jobs never run, gamma = -1 everywhere with exit code 0) now asks
    for confirmation at submission instead of failing undetectably.

*   💥 **MAESTRO transp beat: prescribed equilibrium and frozen-field (heating-only) mode**:
    `machine_initialization: null` now hands TRANSP the state's own nested flux surfaces as
    data (LEVGEO=8) — no TEQ seed machine, no shape morph, so any target shape works from t=0.
    The new `frozen_field: true` knob additionally pins q verbatim to the input (no GS solve,
    no current diffusion), turning the beat into a pure source calculator (TORIC/NUBEAM) for
    downstream beats. Validated end-to-end by `tests/dev_tests/test_transp_prescribed_eq.py`,
    which can execute a real transp+portals chain (`--full`).

*   💥 **MAESTRO transp beat can seed from ITER** (`machine_initialization: ITER`): TEQ warm-starts
    its first solve from a stored per-device equilibrium keyed to the tokamak label, with a tight
    (~1.3x) convergence basin — so reactor-scale targets (ARC-class) should morph from ITER rather
    than a ~7x walk from CMOD. The namelist comment now documents the pick-the-nearest-machine rule.

*   💥 **MAESTRO lengyel beat `mode: 'clean'`**: non-detached forward-conduction separatrix
    temperature (the package-native clean-Lengyel mode above) applied to the profiles WITHOUT
    touching densities/impurities — no detachment solve, no seeding, no radas. Gives BC-setting
    beats (sharpness/confinement) a physics-based Tsep scale instead of the namelist constant;
    PORTALS surrogate data stays reusable. Test chain: `tests/dev_tests/test_lengyel_clean_beat.py`.

*   💥 **Confinement/sharpness beats support Te_bc under-relaxation** (`relaxation` knob in both
    beats' `parameters_prepare`, default 1.0 = previous behavior): the applied boundary temperature
    is blended with the value applied by the previous confinement/sharpness beat (shared trans-beat
    memory `Te_bc_applied`, so mixed chains relax coherently), damping beat-to-beat oscillations of
    the BC servo. The first incarnation takes the full step; with `relaxation < 1` the target
    (H-factor / xi) converges across beat iterations and the applied effective xi is reported as
    `xi_eff`. The confinement beat also gained an isothermal-edge guard (`Te_bc_min_Tesep_factor`,
    default 1.2): the effective Te_bc floor is `max(bound, factor * Tesep)` of the incoming state,
    so the H-servo can never apply a sub-separatrix boundary temperature (which SIGFPEs TRANSP);
    pinned optima are flagged `Te_bc_at_floor` in the beat results instead of crashing the chain
    (a pin with H below the target — a Nelder-Mead bound-clipping artifact — is re-solved exactly
    by a bracketed root find). Both beats can now also run a measured-response servo
    (`servo_mode: response_fit` + `servo_*` knobs): every incarnation records the delivered
    (post-transport) H or xi at the previously applied Te_bc into a persistent trans-beat history,
    and the step comes from a local linear fit of that measured response (fallback secant → seeded
    step, trust-clamped) instead of a fixed relaxation of the frozen-shape solve — which is ~2.5×
    too stiff (delivered dlnH/dlnTe_bc ≈ 0.4 vs ~1.0 frozen), the cause of slow cross-beat
    convergence in confinement↔PORTALS chains. Default remains the previous relaxation behavior.
    Test chains: `tests/dev_tests/test_bc_relaxation.py`, `tests/dev_tests/test_bc_servo_response_fit.py`.

*   💥 **New PORTALS tab "Fluxes vs Gradients"** (`mitim_plot_portals --complete`, or
    `PORTALSanalyzer.plotFluxesVsGradients()`): an NxN matrix over the predicted channels
    scattering every evaluated flux against every evolved gradient, one color per radius,
    with 1-sigma transport-model error bars and a per-radius least-squares line. The diagonal
    panels (flux vs its own drive) expose the critical-gradient / stiffness behavior of the
    transport model across the whole run, with the vertical spread at fixed gradient showing
    the effect of everything else that moved (Ti/Te, nu_ei, beta_e, ...). Fluxes are gyro-Bohm
    normalized by default; `flux_type` selects turbulent (default), neoclassical or the sum.

*   💥 **MAESTRO graded pruning** (`maestro.prune_level`, 0-3, replacing the `keep_all_files`
    boolean): 0 keeps everything; 1 drops per-beat execution scratch nothing reads back (TRANSP
    `results/` CDF duplicate + PH.CDF, EPED per-height TOQ/ELITE dirs, PORTALS `Execution/`
    trees) with every plot tab intact; 2 also wipes `run_<name>/`; 3 adds the PORTALS output
    prune and the initializer prune (incl. the nested `initializer_eped/run_eped/` tree that
    the old cleanup never reached). Overridable per beat (`maestro.<beat>.prune_level`), and
    `mitim_prune_maestro --level N` applies any level post-hoc to a finished run, importing the
    same per-beat tables so the two cannot drift. `mitim_plot_maestro` degrades gracefully on
    pruned runs (placeholder tabs + an aggregated "skipped" report instead of failures).

### Bug Fixes

*   🐛 **Every NEO retrieval waited 60 s for a file NEO never writes**: `out.neo.rotation` was
    listed as a mandatory output, but NEO only produces it for the rotation models that solve
    for the poloidal potential (never with `ROTATION_MODEL=1`). Each retrieval therefore
    reported it missing, slept 60 s and re-pulled every output of every radius once more —
    ~224 times (~3.7 h of pure sleep) in a 14-beat MAESTRO chain, enough to push long chains
    past their wall clock. It is now declared optional: still retrieved whenever NEO does
    write it, its absence only warns.

*   🐛 **TRANSP beat wrote a negative ICRF antenna frequency for negative-`bcentr` states**:
    `frqicha` was derived from the signed field, so any state stored with the opposite sign
    convention (legitimate in gacode) got `frqicha < 0` in the deck; the resonance condition
    only involves |B| (field direction reaches TRANSP via `nlbccw`), so |bcentr| is now used.

*   🐛 **MAESTRO Transition tabs showed spurious flux-surface offsets between beats with
    different radial grids**: the equilibria overlay picked surfaces by nearest-grid-point
    snap, so two states with different rho grids drew rings at different surfaces — up to
    half a grid spacing (~5 mm) of fake geometry "drift". Surfaces (and the psi_N=0.995
    curve) are now interpolated at the requested coordinate; only real differences remain.

*   🐛 **MPI TRANSP (ICRF/NUBEAM parallel servers) crashed at startup when submitted via sbatch on
    hyperthreaded partitions** ("mpirun ... no available cpus in the allocation"): `--ntasks N` buys
    N hyperthreads = N/2 physical cores, but the container binds one rank per core. New
    `cpus_per_task` argument in `defineRunParameters()` (also reachable via `transp_run.run()` and
    the transp-beat run kwargs; default None = previous behavior) — set to 2 on such partitions.

*   🐛 **MAESTRO separatrix initializer delivered less auxiliary power than requested when
    the freegs profile correction ran**: sources were volume-normalized on the solved freegs
    flux surfaces, but the geometry was then overwritten with the analytic shaping guess without
    renormalizing — delivered/requested fell to ~0.96/0.88/0.83 at kappa_sep 1.5/1.735/1.97
    (elongation-dependent, R-independent). The aux channels are now renormalized against the
    written geometry. Also fixed: `BetaN: null` crashed the initializer pressure guess (presence
    test instead of a None check); the freegs-correction failure was swallowed silently — it
    now logs the exception; and the renormalization initially double-applied because
    `equilibrium_to_profiles` aliased the e/i aux channels to the same array (now copied,
    and the renormalization is non-in-place).

*   🐛 **NEO silent failure at extreme Ti/Te fixed and made self-describing**: with zero
    rotation, the Sonic preset's `ROTATION_MODEL=2` quasineutrality solve could fail to
    converge for Ti/Te ≲ 1e-2 (reachable by optimizer excursion candidates) and exit 0 with
    empty transport files; `to_neo` now auto-selects model 1 when w0≡0 (identical fluxes,
    robust), and the empty-output error now includes the reason NEO wrote to `out.neo.run`.

*   🐛 **MAESTRO robustness: TRANSP beats restart cleanly and failures are self-describing**:
    a restarted TRANSP beat used to stage the *previous attempt's outputs* into the new run
    (TRANSP tried to resume from the stale state and aborted at NSTEP 1) — staging is now a
    whitelist of the actual inputs (ufiles + namelist), so re-running a MAESTRO folder needs no
    manual cleanup. Batch runs no longer die with a blank "interactive response required":
    the TRDAT and CDF-retrieval paths raise with the diagnosed cause, and the failure classifier
    reports the true one (infra launch/binding failures only claimed when the run never advanced,
    MPI_ABORT and geometry-update categories added, the routine complaint preceding the trap
    takes precedence). A `plfhe4` namelist knob (fusion-product MC source-power gate) is also exposed.

*   🐛 **TGLF `processDominated` returned TEM values under the ETG keys**: `g_ETG_max`/
    `k_ETG_max`/`f_ETG_max` were copies of the TEM maxima (so `eta_ITGETG` always equalled
    `eta_ITGTEM`); the ETG-range values computed in the same function are now returned.

*   🐛 **TRANSP fast-model alphas no longer lost when NUBEAM is off**: the `nalpha=1`
    analytic fast-alpha model writes the alpha population under different CDF variables
    than NUBEAM (`NALPHA`/`UALPHPP`-`UALPHPA`/`PALE`-`PALI` vs `NFI`,`FDENS_4`/`UFIPP`-`UFIPA`/
    `PFE`-`PFI`); the CDF reader only knew the NUBEAM names and silently zero-filled, so
    NUBEAM-free runs (e.g. MAESTRO with `useNUBEAMforAlphas: false` and no ICRH/NBI) dropped
    the fast-alpha species AND its heating from the extracted state. The reader now falls
    back to the fast-model names (validated against a burning-plasma CDF: 275 MW alpha
    heating and a 1 MeV, 0.03e20 m^-3 alpha species recovered). NUBEAM runs are unaffected.

*   🐛 **MAESTRO initializer: BetaN auto-lowering when the seed profiles would break TRANSP**:
    the profile initializer matches a target BetaN by scanning the temperature gradient; at
    low density the target can be unreachable and the seed saturates above TRANSP's 100 keV
    input ceiling, killing the first TRANSP beat (TRDAT `CKDRNG` rejection). The BetaN target
    is now lowered by 25% and re-solved (repeatedly, with warnings) until the on-axis
    temperature is TRANSP-safe.

*   🐛 **MAESTRO EPED beat: teped-lowering retries now also fire on NaN returns**: full EPED can
    complete but return NaN when the marginal point falls outside the explored `TEPED_BOUND`
    window (e.g. low-shaping/low-Ip cases unstable already at the window floor); this bypassed the
    retry loop and killed the beat on the first attempt. Such returns now get the same
    floor-lowering retries as the exception path, and the final error reports the EPED inputs.

*   🐛 **MAESTRO frozen TRANSP boundary crash on the 2nd TRANSP beat**: reusing the frozen boundary
    disabled the MXH projection for *all* time slices, so the machine-initialization curve (different
    point count) made `write_ufiles` fail with a ragged-array `ValueError` in any chain with two or
    more TRANSP beats. The frozen curve is now tagged per time slice and used verbatim while other
    slices are still projected onto the common theta grid; also fixed swapped `delta/zeta/z0`
    arguments when building machine structures from an overridden boundary, and the freeze now
    stores the *plasma* boundary (last time slice) instead of the machine-initialization
    equilibrium (earliest slice) — previously later beats could inherit the startup machine's
    tiny boundary — with a loud guard refusing to freeze a curve inconsistent with the plasma
    minor radius.

### Changes for developers (internal execution)

*   🔎 **NEW CHANGE**, description

### Back-compatibility considerations and defaults

*   🔮 **`maestro.keep_all_files` is deprecated** in favor of `prune_level` (true -> 0, false -> 3).
    The boolean still works everywhere it did (YAML, `maestro(keep_all_files=...)`,
    `--no-keep-all-files`) with a deprecation notice; the default remains keep-everything.

*   🔮 **MAESTRO template PORTALS exploration ranges widened**: `portals_parameters.solution.
    exploration_ranges` in `namelist.maestro.yaml` now defaults to `ymax: 4.0`,
    `yminymax_atleast: [null, 4]` (previously inheriting the PORTALS defaults 3.0 / [0, 2]),
    matching what the ARC MAESTRO scans have been overriding successfully. Standalone PORTALS
    (`namelist.portals.yaml`) is unchanged.

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
