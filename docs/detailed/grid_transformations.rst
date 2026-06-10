Grid Transformations and Profile Flow
======================================

PORTALS optimizes plasma profiles by working in normalized-gradient space (:math:`a/L_T`, :math:`a/L_n`) on a
coarse radial grid of ~5 control points. Transport codes (TGLF, CGYRO, NEO) need full profiles on a fine
grid. This page documents every grid transformation, interpolation, and derivative computation that happens
along the way --- and the subtle numerical issues that arise from them.


Grid types
----------

MITIM uses several radial grids throughout the workflow. Understanding which grid is active at each stage
is essential for debugging discrepancies.

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Grid
     - Typical size
     - Description
   * - **Original**
     - ~30--120 pts
     - The user-provided ``input.gacode`` file. Profiles live on a :math:`\rho` (square root of normalized toroidal flux) coordinate.
   * - **Enhanced fine**
     - ~100+ pts
     - After ``improve_resolution_profiles()``. Uniform fill between control points, plus extra points at :math:`\pm 10^{-3}` and :math:`\pm 2 \times 10^{-3}` around each control point.
   * - **Coarse powerstate**
     - ~5 pts
     - The PORTALS prediction radii (e.g., :math:`\rho = 0.2, 0.4, 0.55, 0.7, 0.85`). These are the free-parameter locations for the optimization. A zero at the axis is prepended internally.
   * - **Fine targets** (optional)
     - variable
     - A higher-resolution grid used only inside ``calculateTargets()`` for more accurate volume integration of power sources (fusion, radiation, exchange).
   * - **Scalar extraction**
     - 1 pt per surface
     - What TGLF/CGYRO/NEO receives: local values at each flux surface, obtained by cubic-spline interpolation from the enhanced fine grid.


Workflow overview
-----------------

The following diagram traces the ``input.gacode`` data from the user's file all the way to what TGLF/CGYRO receives.
Grid changes and derivative computations are highlighted.

.. code-block:: text

   USER's input.gacode  (~30-120 pts, in rho)
   |
   +---> input.gacode_original  (untouched backup)
   |
   v
   profiles.correct(recalculate_ptot=True)
   |
   ===== powerstate.__init__() ==========================================
   |
   |  improve_resolution_profiles()
   |     ~40 pts --> ~100+ pts
   |     Cubic-spline interpolation of all quantities to new grid
   |     Adds extra points at +/-1e-3, +/-2e-3 of each control point
   |     ** Modifies profiles IN-PLACE **
   |
   |  gacode_to_powerstate()
   |     ~100+ pts --> ~5 coarse pts
   |     Cubic-spline interpolation of profiles to powerstate grid
   |     DERIVATIVE: parameterizers.piecewise_linear()
   |       Computes a/LT via derivation_into_Lx() on the ~100+ pt grid
   |       Samples at coarse control points
   |       Creates profile_constructor functions (fine/coarse/middle)
   |
   ===== PORTALSinit: First roundtrip ===================================
   |
   |  update_var(): a/LT (coarse) --> integration --> T (coarse)
   |     ** Profile now differs from original (roundtrip error) **
   |
   |  from_powerstate(): a/LT (coarse) --> profile_constructors_fine
   |     --> T on ~100+ pt grid
   |     Writes: input.gacode (parameterized), input.gacode_modified
   |
   ===== Each PORTALS evaluation ========================================
   |
   |  1. update_var(): insert new a/LT, integrate to T on coarse grid
   |  2. calculateProfileFunctions(): gyro-Bohm norms, collisionality, etc.
   |  3. calculateTargets(): power balance (optionally on fine targets grid)
   |  4. _produce_profiles():
   |       profile_constructors_fine: a/LT (coarse) --> T on ~100+ pt grid
   |       derive_quantities(): recompute aLTe, aLne, etc. on fine grid
   |       Write: input.gacode (~100+ pts)
   |
   |  5. Transport code input preparation:
   |       to_tglf() / to_cgyro() / to_neo():
   |         Cubic-spline interpolation in r/a space to exact flux surface
   |         --> RLTS, RLNS, DLNTDR, DLNNDR (scalar values per surface)
   |
   ======================================================================


Key operations
--------------

Derivative computation
^^^^^^^^^^^^^^^^^^^^^^

The normalized gradient scale length is defined as:

.. math::

   \frac{a}{L_X} = -\frac{a}{X} \frac{dX}{dr}

MITIM computes this via ``derivation_into_Lx(r, p)``, which uses a **3-point Lagrange interpolating
polynomial** derivative of :math:`-\ln(p)`. This is the same scheme as GACODE's ``bound_deriv`` in
``expro_util.f90``, ensuring that MITIM and GACODE (TGLF, CGYRO, NEO) compute identical gradient
values from the same ``input.gacode`` profiles.

The Lagrange polynomial derivative evaluates at each point :math:`r_i` using its two nearest
neighbors :math:`(r_{i-1}, r_i, r_{i+1})` for interior points, and a one-sided 3-point stencil
at the boundaries.

An alternative **central-difference** variant is available as ``derivation_into_Lx_central()``.
It computes:

.. math::

   z_i = -\frac{\ln p_{i+1} - \ln p_{i-1}}{r_{i+1} - r_{i-1}} \quad \text{(interior points)}

This is structurally more consistent with ``integration_Lx``'s trapezoidal log-space formula and
reduces the fine-grid roundtrip error by ~30%. However, it uses a different stencil than GACODE's
``bound_deriv``, which introduces small discrepancies (~0.01%) when transport codes recompute
gradients from the same ``input.gacode``. For this reason, the Lagrange method is the default.


Integration (gradient to profile)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``integration_Lx(x, z, f_bound)`` reconstructs a profile from its normalized gradient, adapted from
TGYRO's ``tgyro_profile_functions.f90``:

.. math::

   \frac{T_i}{T_{i+1}} = \exp\!\Big[\tfrac{1}{2}(z_i + z_{i+1})(r_{i+1} - r_i)\Big]

where :math:`z = a/L_T` and :math:`f_\text{bound}` is the boundary condition at the last grid point.
This trapezoidal rule in log-space is **exact** for piecewise-linear :math:`z(r)`, which is the
representation used after coarse-grid sampling and interpolation.


Interpolation coordinate: :math:`r/a` vs :math:`\rho`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When extracting local quantities at a specific flux surface (e.g., for TGLF input), MITIM interpolates
profiles using a cubic spline. The choice of **which radial coordinate** to use for the spline matters:

- **GACODE's approach**: ``expro_locsim`` in GACODE uses natural cubic splines in :math:`r/a` (``rmin``)
  space (see ``cub_spline1`` in ``expro_locsim.f90``).

- **MITIM's approach**: The ``to_tglf()``, ``to_cgyro()``, ``to_neo()``, and ``to_gx()`` functions now
  also interpolate in :math:`r/a` space, matching GACODE's convention. Input :math:`\rho` values are
  first converted to :math:`r/a` before interpolation.

**Why this matters**: The mapping :math:`\rho \to r/a` is nonlinear (it depends on the magnetic
equilibrium geometry). For a point that falls between grid nodes, a cubic spline in :math:`\rho` and
a cubic spline in :math:`r/a` will give slightly different interpolated values. Near the plasma edge
where the mapping is most nonlinear, the discrepancy can reach ~0.1--0.2%.

By using the same interpolation coordinate as GACODE, MITIM ensures that the gradient values written to
``input.tglf`` / ``input.cgyro`` match what GACODE's internal routines would compute from
``input.gacode`` via ``PROFILE_MODEL=2``.


Profile reconstruction and the trailing edge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When PORTALS reconstructs a full profile from coarse gradients (via ``profile_constructors_fine``),
the following steps are applied:

1. **Piecewise-linear gradient interpolation**: The gradient values at the coarse control points are
   linearly interpolated to the full fine grid.

2. **smoothAroundCoarsing** (controlled by ``transport.flatten_gradients_at_control_points`` in the PORTALS namelist):
   The gradient between control points is piecewise-linear, which creates a slope change (kink in
   gradient space) at each control point. If a code recomputes gradients from the reconstructed
   profile using a higher-order stencil (e.g., TGYRO's 3-point Lagrange), it will "see" the slope
   change through neighboring points and return a value that differs from the intended control point
   gradient.

   When enabled (default: ``true``), the gradient at :math:`\pm 1` neighboring fine-grid points
   around each control point is flattened to match the control point value. This creates a tiny
   plateau so that any derivative stencil returns the correct value regardless of its order.
   The extra points at :math:`\pm 10^{-3}` added by ``improve_resolution_profiles()`` exist
   specifically to provide these close neighbors; they are skipped when smoothing is disabled.

   When disabled (``false``), the piecewise-linear gradient is left as-is, the :math:`\pm 10^{-3}`
   extra grid points are not added, and the fine grid is more uniform. This may be preferable
   when the PORTALS evaluation does not rely on codes that recompute gradients from profiles.

3. **Integration with direct BC**: The gradient profile is integrated via ``integration_Lx`` with the
   boundary condition applied directly at the last control point. No extra points or rescaling are
   needed.

4. **Hermite trailing edge blending**: Beyond the last control point, the reconstructed profile must
   transition back to the original (unmodified) profile. This is done with a C1-continuous Hermite
   smoothstep blend (:math:`w(t) = 3t^2 - 2t^3`) over a small region (a few grid points). The
   reconstructed profile is linearly extrapolated from the junction using its last gradient, then
   blended with the original profile. This eliminates the derivative discontinuity (kink) that
   previously caused ~0.1% gradient discrepancies when transport codes recomputed gradients from
   the reconstructed profiles.


The roundtrip problem
---------------------

Converting a profile to gradients and back does not reproduce the original profile:

.. math::

   T \xrightarrow{\text{derivative}} \frac{a}{L_T} \xrightarrow{\text{coarsen to 5 pts}} \xrightarrow{\text{interpolate}} \xrightarrow{\text{integrate}} T' \neq T

The error has two sources:

1. **Derivative--integration mismatch**: No point-value derivative is the exact mathematical inverse of
   ``integration_Lx``. The integration uses interval averages :math:`\tfrac{1}{2}(z_i + z_{i+1})`,
   which cannot be uniquely inverted to point values :math:`z_i`.

2. **Coarsening**: Sampling the gradient at ~5 points and piecewise-linearly interpolating loses the
   original gradient shape between control points.

**Current policy**: PORTALS allows exactly one roundtrip pass at initialization (in ``PORTALSinit``).
This "bakes in" the roundtrip error but ensures the starting point is self-consistent with the
parameterization. Subsequent evaluations operate purely in gradient space and do not introduce
additional roundtrip degradation --- the profile is always reconstructed from the current gradients,
never re-differentiated.


GACODE consistency
------------------

GACODE's internal derivative (``bound_deriv`` in ``expro_util.f90``) and MITIM's
``derivation_into_Lx`` are both 2nd-order accurate but use different stencils. When GACODE
(e.g., CGYRO with ``PROFILE_MODEL=2``) reads the same ``input.gacode`` file and recomputes gradients,
small differences can arise:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Source
     - Typical magnitude
     - Cause
   * - Derivative stencil
     - < 0.01%
     - Both methods are 2nd-order on the same grid; differences are :math:`O(h^2)` where :math:`h` is the fine grid spacing.
   * - Interpolation coordinate
     - ~0.1--0.2% at edge
     - Fixed by interpolating in :math:`r/a` instead of :math:`\rho` (now implemented).
   * - Text format truncation
     - ~0.001%
     - ``input.gacode`` text format has ~7 significant digits. Derivatives of truncated values differ from derivatives of full-precision values.
   * - Trailing edge blending
     - negligible
     - The Hermite smoothstep blend ensures C1 continuity at the trailing edge junction.


Known caveats and future work
-----------------------------

- **Grid overdimensioning**: The :math:`\pm 10^{-3}` extra points added by
  ``improve_resolution_profiles()`` exist to support ``smoothAroundCoarsing``. They are automatically
  skipped when ``transport.flatten_gradients_at_control_points: false`` is set in the PORTALS namelist.

- **Multi-pass idempotency**: Achieving perfect stability (pass 2 = pass 1) through the derivative
  function alone is not possible because ``integration_Lx`` uses interval averages that cannot be
  uniquely inverted to point values. True idempotency would require changes at the parameterization
  level (e.g., self-consistent initialization via iterative gradient correction).
