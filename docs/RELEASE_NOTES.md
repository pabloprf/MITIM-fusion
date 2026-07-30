# vX.Y.Z — TITLE

DESCRIPTION

### New Features

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

### Bug Fixes

*   🐛 **MAESTRO robustness: TRANSP beats restart cleanly and failures are self-describing**:
    a restarted TRANSP beat used to stage the *previous attempt's outputs* into the new run
    (TRANSP tried to resume from the stale state and aborted at NSTEP 1) — staging is now a
    whitelist of the actual inputs (ufiles + namelist), so re-running a MAESTRO folder needs no
    manual cleanup. Batch runs no longer die with a blank "interactive response required":
    the TRDAT and CDF-retrieval paths raise with the diagnosed cause, and the failure classifier
    reports the true one (infra launch/binding failures only claimed when the run never advanced,
    MPI_ABORT and geometry-update categories added, the routine complaint preceding the trap
    takes precedence). A `plfhe4` namelist knob (fusion-product MC source-power gate) is also exposed.

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

*   🔮 **MAESTRO template PORTALS exploration ranges widened**: `portals_parameters.solution.
    exploration_ranges` in `namelist.maestro.yaml` now defaults to `ymax: 4.0`,
    `yminymax_atleast: [null, 4]` (previously inheriting the PORTALS defaults 3.0 / [0, 2]),
    matching what the ARC MAESTRO scans have been overriding successfully. Standalone PORTALS
    (`namelist.portals.yaml`) is unchanged.

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
