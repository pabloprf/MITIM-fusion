"""
Registry mapping MITIM `.derived` keys to fusio `plasma_io`'s own native
derived-quantity field names, for plasma_io-backed `mitim_state` instances.

Populated incrementally, one validated entry at a time (see
`test/fusio_integration/plasma_io_migration_plan.md`, "New phase
(2026-08-12)"), only for keys actually consumed by an in-scope PORTALS file.
A key with no entry here simply falls back to `mitim_state`'s own
`derive_quantities_full` computation (via `get_derived()`'s shadow-object
bootstrap) -- unmapped is the safe default, not an error.

No entry is added without first parity-testing it against
`derive_quantities_full`'s existing value for that key (reusing the
tolerance conventions already established in `test/fusio_integration/`).
Keys with a known, unresolved discrepancy (e.g. `aLni`, see the migration
plan doc) are deliberately kept out of this map rather than mapped anyway.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class FieldMapping:
    plasma_io_key: str
    factor: float = 1.0
    transform: Optional[Callable[[Any], Any]] = None

    def apply(self, native: Mapping[str, Any]) -> Any:
        value = native[self.plasma_io_key]
        if self.transform is not None:
            return self.transform(value)
        return value * self.factor


# {mitim_derived_key: FieldMapping}. Empty until Phase 3 of the
# self.profiles/self.derived removal work validates and adds entries
# module-by-module.
DERIVED_FIELD_MAP: dict[str, FieldMapping] = {}


# Native plasma_io fields required to compute geometry_factors_from_native()'s six outputs.
_GEOMETRY_FACTOR_NATIVE_KEYS = ('dvolume_dr', 'surface_area', 'mxh_field_inner', 'field_squared', 'field_unit')


def geometry_factors_from_native(native: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """
    Pre-population values for the six quantities PROFILEStools.gacode_state.derive_geometry()
    would otherwise get from calculateGeometricFactors() (volp_geo, surf_geo, gradr_geo,
    bp2_geo, bt2_geo, bt_geo), sourced from plasma_io's own native derived-quantity fields
    instead. Consumed by MITIMstate._compute_derived_from_plasma_io(), which stashes the
    result on a throwaway shadow instance's `_geometry_factor_prepop` attribute --
    derive_geometry() uses it instead of calling calculateGeometricFactors() when present,
    leaving everything else in that method (flux-surface reconstruction, kappa95/delta95/etc,
    B_ref) untouched.

    Returns None if any required native key is missing from `native` (e.g.
    plasma_io.compute_derived_quantities() hasn't been run on the source object yet) -- the
    caller falls back to the unmodified calculateGeometricFactors()-based computation in that
    case, matching this module's "unmapped is the safe default, not an error" convention.

    Why this exists: calculateGeometricFactors() operates on shape_cosN(-)/shape_sinN(-),
    which get_profiles() zero-fills whenever gacode_io.from_plasma() doesn't populate them
    from the source plasma_io object's own mxh_cos/mxh_sin fields. For a genuinely-shaped
    equilibrium this isn't just noise at the r/a=1 boundary -- it silently computes geometry
    for a zero-shape (circular/Miller-only) approximation across the *whole* profile (see
    test/fusio_integration/plasma_io_migration_plan.md, "mitim_plot_portals CLI check" entry,
    2026-08-13). Sourcing these six quantities from plasma_io's own native fields instead
    sidesteps that entirely, and does not depend on gacode_io.from_plasma()'s shape-coefficient
    extraction (separately found to be flaky/non-deterministic, not chased further -- same
    entry) at all.

    Mappings (all indexed on plasma_io's native grid, which get_profiles() confirmed is
    identical, point for point, to the grid derive_geometry() stores these under):
      - volp_geo <- dvolume_dr directly: both are dV/dr in m^2 on the same r_minor grid.
      - surf_geo <- surface_area directly: both are actual flux-surface area in m^2.
      - gradr_geo <- surf_geo / volp_geo, using derive_geometry()'s own documented relation
        ("Surf = V' <|grad r|>, Surf_GACODE = V'") -- self-consistent by construction, not an
        independent native lookup.
      - bt_geo <- mxh_field_inner's toroidal component (direction index 0), matching the
        already-validated B_ref = field_unit * mxh_field_inner[:, 0] convention established
        for TRANSFORMtools.plasma_io_to_powerstate() (see the migration plan's 2026-08-07
        entries) -- derive_geometry()'s own `B_ref = B_unit * bt_geo` line then reproduces
        that exact, already-validated B_ref automatically.
      - bp2_geo / bt2_geo <- field_squared's poloidal/toroidal components (direction indices
        1/0), divided by field_unit**2 to match calculateGeometricFactors()'s B_unit-normalized
        convention (derive_quantities_full()'s `bp2_exp = bp2_geo * B_unit**2`).
    """
    if not all(k in native for k in _GEOMETRY_FACTOR_NATIVE_KEYS):
        return None

    dvolume_dr = native['dvolume_dr']
    surface_area = native['surface_area']
    field_unit_sq = native['field_unit'] ** 2

    # Defensive: a handful of fusio's own geometry fields carry known, already-documented
    # edge-case NaN/inf potential (e.g. dvolume_dr's r_minor=0 floor exists precisely because
    # the raw finite-difference can misbehave there). A *large* non-finite fraction, though,
    # means the source data itself is unreliable for this object (confirmed: the stale,
    # pre-2026-08-07-fix arc_v3a_refined_plasma.nc/_v2/_v3 fixtures show ~97-98% NaN in
    # surface_area when recomputed against current code) -- silently zero-filling that much of
    # the array would produce worse volume integrals than just falling back to
    # calculateGeometricFactors(), so bail out to the safe default (None) instead.
    _max_nonfinite_fraction = 0.05
    _checks = (dvolume_dr, surface_area, native['mxh_field_inner'], native['field_squared'])
    if any(np.mean(~np.isfinite(arr)) > _max_nonfinite_fraction for arr in _checks):
        return None

    volp_geo = np.nan_to_num(dvolume_dr)
    surf_geo = np.nan_to_num(surface_area)
    gradr_geo = np.divide(surf_geo, volp_geo, out=np.zeros_like(surf_geo, dtype=float), where=(volp_geo != 0))
    bt_geo = np.nan_to_num(native['mxh_field_inner'][:, 0])
    bt2_geo = np.nan_to_num(native['field_squared'][:, 0] / field_unit_sq)
    bp2_geo = np.nan_to_num(native['field_squared'][:, 1] / field_unit_sq)

    return {
        'volp_geo': volp_geo,
        'surf_geo': surf_geo,
        'gradr_geo': gradr_geo,
        'bt_geo': bt_geo,
        'bp2_geo': bp2_geo,
        'bt2_geo': bt2_geo,
    }
