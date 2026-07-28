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


### Bug Fixes

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

*   🔮 **`reuse_scan_ball` now selects the reuse region, not just on/off**: the TGLF option
    accepts `null` / `"box"` / `"ball"` instead of a boolean. The historical region — despite the
    name — is `box`: every input independently within ±delta, whose corners reach delta·sqrt(N) in
    relative L2 and are exactly what lets *multi-dimensional* combinations enter the sample cloud
    (next to a stiffness cliff those combinations can populate a whole second flux branch that the
    one-at-a-time scan never reaches). `ball` applies a relative L2 bound instead, consistent with
    the scan stencil but rejecting most combinations. `true`/`false` still work and map to
    `box`/`null`, so existing namelists are unaffected; defaults are unchanged (`null` standalone,
    `box` in MAESTRO).

*   🔮 **MAESTRO template PORTALS exploration ranges widened**: `portals_parameters.solution.
    exploration_ranges` in `namelist.maestro.yaml` now defaults to `ymax: 4.0`,
    `yminymax_atleast: [null, 4]` (previously inheriting the PORTALS defaults 3.0 / [0, 2]),
    matching what the ARC MAESTRO scans have been overriding successfully. Standalone PORTALS
    (`namelist.portals.yaml`) is unchanged.

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
