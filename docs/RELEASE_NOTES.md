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

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
