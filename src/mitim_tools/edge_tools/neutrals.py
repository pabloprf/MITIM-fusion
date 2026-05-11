"""
neutrals.py
-----------
Main-ion (D⁰) neutral density solver for PORTALS-Edge.

All models share the public interface::

    model = <Model>(options)
    model.solve(powerstate, batch_idx=0)

After ``solve()``, the following keys are written into ``powerstate.plasma``:

    plasma['n0']              : torch.Tensor (batch, rmin)  [1e19 m⁻³]
        Main-ion (D⁰) ground-state neutral density profile (gridded on rmin
        [meters]).  This is the key read by ``analytical_model_edge._evaluate_particle_fluxes``
        and ``_evaluate_ionization_loss``.

    plasma['S_ion_main']      : torch.Tensor (batch, rmin)  [1e19 m⁻³ s⁻¹]
        Volumetric ionisation source:  S = n₀ × ν_iz (gridded on rmin [meters]).
        Directly usable in particle-flux target and ionisation power-loss
        calculations without needing to recompute the rate coefficient.

    plasma['nu_ioniz_main']   : torch.Tensor (batch, rmin)  [s⁻¹]
        Effective D⁰ ionisation frequency:  ν_iz = n_e × ⟨σv⟩_iz(n_e, T_e).
        Evaluated from ADAS H SCD data when Aurora is available; falls back
        to an analytic Lyman-α-weighted fit otherwise.

Physics
-------
Solver selection is governed by the Knudsen number estimated over the
outermost ``Kn_eval_fraction`` of the radial domain:

    Kn(r) = λ_mfp(r) / L_ne(r)

    λ_mfp = v_th / ν_iz          (ionisation-limited mean free path)
    L_ne  = |n_e / (dn_e/dr)|    (electron density scale length)
    v_th  = √(2 k_B T_i / m_D)  (neutral thermal speed ≈ ion thermal speed)

  Kn_edge ≤ Kn_thresh  →  **diffusive** solver (collisional sub-regime)

      1-D slab steady-state diffusion with ionisation sink:

          d/dx [ D_n(x) dn₀/dx ] − ν_iz(x) n₀ = 0

      D_n = v_th² / (3 ν_total)   with ν_total = ν_iz + ν_cx

      Solved as a tridiagonal linear system via ``scipy.linalg.solve_banded``.
      Boundary conditions:
          n₀[0]  = 0  (inner, Dirichlet: fully absorbed / ionised)
          n₀[-1] = 1  (outer, Dirichlet, rescaled to source_rate)

  Kn_edge  > Kn_thresh  →  **kinetic** (free-streaming) solver

      Mono-energetic streaming with ionisation attenuation (Beer-Lambert):

          n₀(r) ∝ exp(−τ(r))
          τ(r) = ∫_r^{r_LCFS} ν_iz(r') / v_th(r') dr'

In both cases the amplitude is fixed by steady-state particle balance:

    ∫ n₀(r) × ν_iz(r) dV = source_rate   [s⁻¹]

where  dV = volp(r) dr  in GACODE convention (volp = dV/dr_min [m²]).

Charge-exchange
---------------
CX (D⁰ + D⁺ → D⁺ + D⁰) does not net-remove D⁰ atoms from the neutral
population, so it does not appear as a sink in the density equation.  However
it increases the total D⁰-D⁺ collision frequency, reducing the effective
mean free path.  When ``include_cx`` is True, ν_cx is added to ν_total when
computing D_n, using a simple analytic rate fit for self-CX:

    ⟨σv⟩_cx ≈ 2e-14 × (T_i_eV / 1000)^0.3 cm³/s  (D+D resonant CX, rough fit)

ν_cx is NOT added to the Kn estimate or the density-equation sink.

Integration with STATEedge / powerstate_edge
--------------------------------------------
``calculateNeutrals()`` in ``powerstate_edge`` calls this solver.  It is
invoked between ``calculateChargeStates()`` and ``calculateTargets()``, after
kinetic profiles are fully reconstructed.

Notes
-----
* KN1D (IDL-based) is NOT called.  This is a pure-Python analytic model for
  use inside every PORTALS iteration.
* ADAS H SCD rates are loaded once and cached in ``_ADAS_CACHE`` at module
  level; subsequent calls reuse the same table.
* The 1-D slab approximation is used for the diffusion equation.  The
  cylindrical correction of order (dr/r)² ≈ 1 % in the pedestal region
  is negligible.
"""

import numpy as np
import torch
from scipy.linalg import solve_banded
from scipy.constants import m_p, e as q_e

from mitim_tools.misc_tools.LOGtools import printMsg as print

try:
    import aurora as _aurora_pkg
    _AURORA_AVAILABLE = True
except ImportError:
    _aurora_pkg = None
    _AURORA_AVAILABLE = False


# ---------------------------------------------------------------------------
# ADAS data cache — loaded once on first use, shared across all instances
# ---------------------------------------------------------------------------

_ADAS_CACHE: dict = {}


def _load_h_ioniz_rate():
    """Return ADAS H SCD ionisation rate table, loading once and caching."""
    if "H_scd" not in _ADAS_CACHE:
        if not _AURORA_AVAILABLE:
            _ADAS_CACHE["H_scd"] = None
        else:
            try:
                ad = _aurora_pkg.atomic.get_atom_data("H", ["scd"])
                _ADAS_CACHE["H_scd"] = ad["scd"]
            except Exception as exc:
                print(
                    f"[AnalyticNeutrals] Could not load ADAS H SCD data: {exc}.  "
                    f"Falling back to analytic fit.",
                    typeMsg="w",
                )
                _ADAS_CACHE["H_scd"] = None
    return _ADAS_CACHE["H_scd"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NeutralModel:
    """Abstract base — subclasses implement ``solve(powerstate, batch_idx)``."""

    def __init__(self, options: dict):
        self.options = options

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Populate ``powerstate.plasma`` with main-ion neutral density data.

        Must write:
          ``plasma['n0']``             (batch, rmin)  [1e19 m⁻³]
          ``plasma['S_ion_main']``     (batch, rmin)  [1e19 m⁻³ s⁻¹]
          ``plasma['nu_ioniz_main']``  (batch, rmin)  [s⁻¹]
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NullNeutrals
# ---------------------------------------------------------------------------

class NullNeutrals(NeutralModel):
    """Zero neutral density — no source.  Useful for testing or for runs
    where the main-ion recycling is handled externally."""

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        p     = powerstate.plasma
        batch = p["te"].shape[0]
        n_rho = p["te"].shape[1]
        _kw   = {"dtype": p["te"].dtype, "device": p["te"].device}

        if "n0" not in p or p["n0"].shape != (batch, n_rho):
            p["n0"]            = torch.zeros(batch, n_rho, **_kw)
            p["S_ion_main"]    = torch.zeros(batch, n_rho, **_kw)
            p["nu_ioniz_main"] = torch.zeros(batch, n_rho, **_kw)
            p["tau_n0"]        = torch.zeros(batch, n_rho, **_kw)


# ---------------------------------------------------------------------------
# AnalyticNeutrals — 1-D diffusive / kinetic steady-state solver
# ---------------------------------------------------------------------------

class AnalyticNeutrals(NeutralModel):
    """
    1-D steady-state D⁰ neutral density solver with automatic regime selection.

    Parameters
    ----------
    options : dict
        ``source_rate`` : float, default 1e21
            Number of D⁰ neutrals crossing the LCFS inward per second [s⁻¹].
            Sets the amplitude of n₀ via particle balance.
        ``mu_amu`` : float, default 2.014
            Atomic mass of the neutral species [amu].  2.014 for deuterium.
        ``Kn_thresh`` : float, default 0.3
            Knudsen number threshold for solver selection.  Above this value
            the kinetic (free-streaming) solver is used; at or below, the
            diffusive (collisional) solver is used.
        ``Kn_eval_fraction`` : float, default 0.3
            Fraction of the outer radial domain used to compute the
            representative Kn for solver selection.
        ``include_cx`` : bool, default True
            When True, an analytic charge-exchange rate is added to the
            total collision frequency when computing D_n.  CX is NOT added
            to the Knudsen number estimate or the ionisation sink.
        ``verbose`` : bool, default False
    """

    def __init__(self, options: dict):
        super().__init__(options)
        self.source_rate      = options.get("source_rate",      1e21)
        self.mu_amu           = options.get("mu_amu",           2.014)
        self.Kn_thresh        = options.get("Kn_thresh",        0.3)
        self.Kn_eval_fraction = options.get("Kn_eval_fraction", 0.3)
        self.include_cx       = options.get("include_cx",       True)
        self.verbose          = options.get("verbose",          False)

    # ------------------------------------------------------------------
    # Ionisation frequency
    # ------------------------------------------------------------------

    def _nu_ioniz(
        self, ne_cm3: np.ndarray, Te_eV: np.ndarray
    ) -> np.ndarray:
        """
        Return  ν_iz(r) [s⁻¹] = n_e × ⟨σv⟩_iz(n_e, T_e).

        Uses the ADAS H SCD effective ionisation rate coefficient when Aurora
        is available; otherwise falls back to the Voronov (1997) analytic fit
        for H ground-state ionisation (ADNDT 65, 1997, Table 1):

            U = 13.6 eV / T_e
            ⟨σv⟩_iz = 2.91×10⁻¹⁴ × U^0.39 × exp(−U) / (0.232 + U)  m³/s

        The Voronov fit is accurate to better than 10% for 5 ≤ T_e ≤ 5000 eV.
        Use ADAS for quantitative work.
        """
        scd = _load_h_ioniz_rate()
        if scd is not None:
            # interp_atom_prof: xprof = log10(ne [cm⁻³]), yprof = log10(Te [eV])
            # x_multiply=True → returns ne × S_cd [cm³/s × cm⁻³ = s⁻¹]
            # Input shape: (nt=1, nr)
            log_ne = np.log10(np.maximum(ne_cm3, 1e8))[np.newaxis, :]
            log_Te = np.log10(np.maximum(Te_eV,  0.1 ))[np.newaxis, :]
            # Returns (nt=1, nion=1, nr) for H (single ionisation stage)
            nu_grid = _aurora_pkg.atomic.interp_atom_prof(
                scd, log_ne, log_Te, x_multiply=True
            )
            return nu_grid[0, 0, :].copy()   # (nr,)  s⁻¹
        else:
            # Voronov (1997) fit: U = chi/T_e, chi = 13.6 eV (H ionisation energy)
            U = 13.6 / np.maximum(Te_eV, 0.1)
            sigma_v_m3 = (
                2.91e-14          # A = 2.91e-8 cm³/s converted to m³/s
                * U ** 0.39
                * np.exp(-U)
                / (0.232 + U)
            )  # m³/s  (P=0, so (1 + P√U) denominator is unity)
            return ne_cm3 * 1e6 * sigma_v_m3   # s⁻¹ (ne_cm3→m⁻³, σv in m³/s)

    # ------------------------------------------------------------------
    # Charge-exchange collision frequency (approximate, for D_n only)
    # ------------------------------------------------------------------

    def _nu_cx(
        self, ni_cm3: np.ndarray, Ti_eV: np.ndarray
    ) -> np.ndarray:
        """
        Analytic estimate of ν_cx = nᵢ × ⟨σv⟩_cx for D⁰ + D⁺ → D⁺ + D⁰.

        Fit to Freeman & Jones (1974) D resonant-CX cross-sections:
            ⟨σv⟩_cx ≈ 2e-14 × (T_i/1000)^0.3  cm³/s   (T_i in eV)

        Returns ν_cx [s⁻¹].
        """
        sigma_v_cx = 2e-14 * (np.maximum(Ti_eV, 1.0) / 1000.0) ** 0.3  # cm³/s
        return ni_cm3 * sigma_v_cx   # s⁻¹

    # ------------------------------------------------------------------
    # Thermal speed
    # ------------------------------------------------------------------

    def _v_th(self, Ti_eV: np.ndarray) -> np.ndarray:
        """Return v_th = √(2 k_B T_i / m_D) [m/s]."""
        return np.sqrt(
            2.0 * np.maximum(Ti_eV, 0.1) * q_e / (self.mu_amu * m_p)
        )

    # ------------------------------------------------------------------
    # Knudsen number
    # ------------------------------------------------------------------

    def _knudsen(
        self,
        r_m:    np.ndarray,
        ne_cm3: np.ndarray,
        Ti_eV:  np.ndarray,
        nu_iz:  np.ndarray,
    ) -> np.ndarray:
        """
        Kn(r) = λ_mfp(r) / L_ne(r).

        λ_mfp = v_th / ν_iz  (ionisation dominates for deeply penetrating
                               neutrals in the closed-flux region)
        L_ne  = |n_e / (∂n_e/∂r)|  (density gradient scale length)
        """
        v_th       = self._v_th(Ti_eV)
        lambda_mfp = v_th / np.maximum(nu_iz, 1.0)     # m

        dne_dr = np.gradient(ne_cm3, r_m)               # cm⁻³ m⁻¹
        L_ne   = np.where(
            np.abs(dne_dr) > 0,
            np.abs(ne_cm3 / np.maximum(np.abs(dne_dr), 1e-30)),
            1.0,
        )   # m  (cm⁻³ / (cm⁻³ m⁻¹) = m, consistent with λ_mfp)
        L_ne   = np.clip(L_ne, 1e-3, 1e3)

        return lambda_mfp / L_ne    # dimensionless

    # ------------------------------------------------------------------
    # Diffusive solver
    # ------------------------------------------------------------------

    def _solve_diffusive(
        self,
        r_m:    np.ndarray,
        nu_iz:  np.ndarray,
        nu_cx:  np.ndarray,
        Ti_eV:  np.ndarray,
        volp:   np.ndarray,
    ) -> np.ndarray:
        """
        Solve  d/dx [ D_n(x) dn₀/dx ] − ν_iz(x) n₀ = 0  (1-D slab).

        D_n = v_th² / (3 ν_total)  where  ν_total = ν_iz + ν_cx.

        Boundary conditions:
          n₀[0]  = 0   (inner Dirichlet — absorbing)
          n₀[-1] = 1   (outer Dirichlet — rescaled below)

        The shape solution is rescaled so that
          ∫ n₀ × ν_iz × dV = source_rate  [s⁻¹].

        Returns n₀ [m⁻³].
        """
        nr    = len(r_m)
        v_th  = self._v_th(Ti_eV)                             # (nr,) m/s
        nu_tot = nu_iz + nu_cx                                 # (nr,) s⁻¹
        D_n    = v_th**2 / (3.0 * np.maximum(nu_tot, 1e-10))  # (nr,) m²/s

        lo  = np.zeros(nr)
        di  = np.zeros(nr)
        hi  = np.zeros(nr)
        rhs = np.zeros(nr)

        # Inner Dirichlet: n₀[0] = 0
        di[0] = 1.0

        # Interior nodes
        for i in range(1, nr - 1):
            dr_up = r_m[i + 1] - r_m[i]
            dr_dn = r_m[i]     - r_m[i - 1]
            dr_c  = 0.5 * (dr_up + dr_dn)
            D_up  = 0.5 * (D_n[i] + D_n[i + 1])
            D_dn  = 0.5 * (D_n[i] + D_n[i - 1])
            hi[i] =  D_up / (dr_up * dr_c)
            lo[i] =  D_dn / (dr_dn * dr_c)
            di[i] = -(hi[i] + lo[i] + nu_iz[i])

        # Outer Dirichlet: n₀[-1] = 1 (rescaled after solve)
        di[-1]  = 1.0
        rhs[-1] = 1.0

        # scipy solve_banded format for (1, 1) banded matrix:
        #   ab[0, j] = superdiag at column j   (ab[0, 0] unused)
        #   ab[1, j] = diagonal  at column j
        #   ab[2, j] = subdiag   at column j   (ab[2, -1] unused)
        ab       = np.zeros((3, nr))
        ab[0, 1:] = hi[:-1]   # hi[i] connects row i → column i+1
        ab[1, :]  = di
        ab[2, :-1]= lo[1:]    # lo[i] connects row i → column i-1

        try:
            n0_shape = solve_banded((1, 1), ab, rhs)
        except Exception as exc:
            print(
                f"[AnalyticNeutrals] Tridiagonal diffusive solve failed: {exc}.  "
                f"Inserting zeros.",
                typeMsg="w",
            )
            return np.zeros(nr)

        n0_shape = np.maximum(n0_shape, 0.0)
        return self._rescale_to_source(n0_shape, nu_iz, volp, r_m)

    # ------------------------------------------------------------------
    # Kinetic (free-streaming) solver
    # ------------------------------------------------------------------

    def _solve_kinetic(
        self,
        r_m:   np.ndarray,
        nu_iz: np.ndarray,
        Ti_eV: np.ndarray,
        volp:  np.ndarray,
    ) -> np.ndarray:
        """
        Free-streaming model with ionisation Beer-Lambert attenuation:

            n₀(r) ∝ exp(−τ(r))
            τ(r) = ∫_{r}^{r_LCFS} ν_iz(r') / v_th(r') dr'

        τ is integrated inward from the LCFS by trapezoidal rule.
        Amplitude is set by particle balance (see ``_rescale_to_source``).

        Returns n₀ [m⁻³].
        """
        nr         = len(r_m)
        v_th       = self._v_th(Ti_eV)                 # (nr,) m/s
        nu_over_v  = nu_iz / np.maximum(v_th, 1.0)     # (nr,) m⁻¹

        # Cumulative optical depth from outer boundary inward
        tau = np.zeros(nr)
        for i in range(nr - 2, -1, -1):
            dr     = abs(r_m[i + 1] - r_m[i])   # abs guards against non-monotone r_m
            tau[i] = tau[i + 1] + 0.5 * (nu_over_v[i] + nu_over_v[i + 1]) * dr

        n0_shape = np.exp(-tau)   # dimensionless; n0_shape[-1] = 1.0 by construction
        return self._rescale_to_source(n0_shape, nu_iz, volp, r_m), tau

    # ------------------------------------------------------------------
    # Shared particle-balance rescaling
    # ------------------------------------------------------------------

    def _rescale_to_source(
        self,
        n0_shape: np.ndarray,
        nu_iz:    np.ndarray,
        volp:     np.ndarray,
        r_m:      np.ndarray,
    ) -> np.ndarray:
        """
        Scale a dimensionless shape profile so that
          ∫ n₀(r) × ν_iz(r) dV = source_rate  [s⁻¹].

        Unit analysis:
          A [m⁻³] × ∫ n₀_shape [-] × ν_iz [s⁻¹] × volp [m²] dr [m] = source_rate [s⁻¹]
          → integral [m³/s], A = source_rate [s⁻¹] / integral [m³/s] = m⁻³  ✓

        Returns A × n₀_shape  in [m⁻³].
        """
        integral = np.trapz(n0_shape * nu_iz * volp, r_m)   # m³/s
        if integral <= 0.0:
            if self.verbose:
                print(
                    "[AnalyticNeutrals] Particle balance integral = 0 — "
                    "inserting zeros.",
                    typeMsg="w",
                )
            return np.zeros_like(n0_shape)

        A = self.source_rate / integral   # m⁻³
        return n0_shape * A               # m⁻³

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Compute the D⁰ neutral density and ionisation source for batch element
        *batch_idx* and write results into ``powerstate.plasma``.
        """
        b     = batch_idx
        p     = powerstate.plasma
        batch = p["te"].shape[0]
        n_rho = p["te"].shape[1]
        dfT   = p["te"]
        _kw   = {"dtype": dfT.dtype, "device": dfT.device}

        # --- Extract 1-D profiles ------------------------------------------------
        rmin_m  = p["rmin"][b, :].cpu().numpy()              # m
        ne_raw  = p["ne"][b, :].cpu().numpy()                # expected: 1e19 m⁻³
        # Guard against silently wrong units: typical pedestal ne is 0.1–50 × 1e19 m⁻³
        if ne_raw.max() < 1e-3 or ne_raw.max() > 1e4:
            print(
                f"[AnalyticNeutrals] batch {b}: p['ne'] max = {ne_raw.max():.3e} — "
                f"expected units are 1e19 m\u207b\u00b3 (typical range 0.1\u201350).  "
                f"Check unit convention before proceeding.",
                typeMsg="w",
            )
        ne_cm3  = ne_raw * 1e13                              # 1e19 m⁻³ → cm⁻³
        Te_eV   = p["te"][b,  :].cpu().numpy() * 1e3         # keV → eV
        Ti_eV   = (
            p["ti"][b, :].cpu().numpy() * 1e3
            if p["ti"].dim() >= 2
            else Te_eV.copy()
        )   # keV → eV
        volp    = p["volp"][b, :].cpu().numpy()               # m² (dV/dr_min)

        ne_cm3 = np.maximum(ne_cm3, 1e8)
        Te_eV  = np.maximum(Te_eV,  0.1)
        Ti_eV  = np.maximum(Ti_eV,  0.1)

        # Main ion density for CX; fall back to ne if ni not available
        if "ni" in p and p["ni"].dim() == 3:
            ni_cm3 = p["ni"][b, :, 0].cpu().numpy() * 1e13
        else:
            ni_cm3 = ne_cm3.copy()
        ni_cm3 = np.maximum(ni_cm3, 1e8)

        # --- Rate coefficients ---------------------------------------------------
        nu_iz = self._nu_ioniz(ne_cm3, Te_eV)      # (nr,) s⁻¹
        nu_cx = self._nu_cx(ni_cm3, Ti_eV) if self.include_cx else np.zeros_like(nu_iz)

        # --- Knudsen number and solver selection ---------------------------------
        Kn      = self._knudsen(rmin_m, ne_cm3, Ti_eV, nu_iz)
        n_outer = max(1, int(self.Kn_eval_fraction * n_rho))
        Kn_edge = float(np.median(Kn[-n_outer:]))

        use_kinetic = Kn_edge > self.Kn_thresh
        if self.verbose:
            solver_tag = "kinetic (free-streaming)" if use_kinetic else "diffusive (collisional)"
            print(
                f"[AnalyticNeutrals] batch {b}: "
                f"Kn_edge = {Kn_edge:.3f} (threshold = {self.Kn_thresh}) "
                f"→ {solver_tag} solver.",
                typeMsg="i",
            )

        # --- Solve ---------------------------------------------------------------
        if use_kinetic:
            # _solve_kinetic returns (n0, tau) — reuse tau to avoid duplication
            n0_m3, tau_n0 = self._solve_kinetic(rmin_m, nu_iz, Ti_eV, volp)
        else:
            n0_m3 = self._solve_diffusive(rmin_m, nu_iz, nu_cx, Ti_eV, volp)
            # Compute optical depth separately for the diffusive case
            v_th_arr  = self._v_th(Ti_eV)                            # (nr,) m/s
            nu_over_v = nu_iz / np.maximum(v_th_arr, 1.0)           # (nr,) m⁻¹
            tau_n0    = np.zeros(n_rho)
            for i in range(n_rho - 2, -1, -1):
                dr        = abs(rmin_m[i + 1] - rmin_m[i])
                tau_n0[i] = tau_n0[i + 1] + 0.5 * (nu_over_v[i] + nu_over_v[i + 1]) * dr

        n0_m3    = np.maximum(n0_m3, 0.0)

        S_ion_m3 = n0_m3 * nu_iz     # m⁻³ s⁻¹

        # --- Convert to powerstate units: 1e19 m⁻³ (as ne, ni) ------------------
        n0_1e19    = n0_m3  * 1e-19
        S_ion_1e19 = S_ion_m3 * 1e-19

        # n0/ne trace-neutral validity guard
        n0_over_ne = n0_m3 / np.maximum(ne_cm3 * 1e6, 1.0)  # both in m⁻³
        if n0_over_ne.max() > 0.1:
            print(
                f"[AnalyticNeutrals] batch {b}: max(n0/ne) = {n0_over_ne.max():.2f} "
                f"exceeds 0.1 — trace-neutral approximation may be breaking down.",
                typeMsg="w",
            )

        # Sanity check: integrated source should match source_rate within
        # floating-point precision.
        if self.verbose:
            check = float(np.trapz(S_ion_m3 * volp, rmin_m))
            print(
                f"[AnalyticNeutrals] batch {b}: "
                f"∫Sion dV = {check:.3e} s⁻¹  (target = {self.source_rate:.3e} s⁻¹).",
                typeMsg="i",
            )

        # --- Write to plasma -----------------------------------------------------
        if "n0" not in p or p["n0"].shape != (batch, n_rho):
            p["n0"]            = torch.zeros(batch, n_rho, **_kw)
            p["S_ion_main"]    = torch.zeros(batch, n_rho, **_kw)
            p["nu_ioniz_main"] = torch.zeros(batch, n_rho, **_kw)
            p["tau_n0"]        = torch.zeros(batch, n_rho, **_kw)

        p["n0"][b]            = torch.from_numpy(n0_1e19).to(dfT)
        p["S_ion_main"][b]    = torch.from_numpy(S_ion_1e19).to(dfT)
        p["nu_ioniz_main"][b] = torch.from_numpy(nu_iz).to(dfT)
        p["tau_n0"][b]        = torch.from_numpy(tau_n0.astype(np.float64)).to(dfT)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NEUTRAL_MODELS: dict = {
    "Null":             NullNeutrals,
    "none":             NullNeutrals,
    "Analytic":         AnalyticNeutrals,
    "AnalyticNeutrals": AnalyticNeutrals,
}


def build_neutral_model(name: str, options: dict) -> NeutralModel:
    """
    Instantiate a neutral model by name.

    Parameters
    ----------
    name : str
        One of the keys in ``NEUTRAL_MODELS``.
    options : dict
        Model-specific options forwarded to the class constructor.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in NEUTRAL_MODELS:
        raise KeyError(
            f"Unknown neutral model '{name}'.  "
            f"Available: {list(NEUTRAL_MODELS.keys())}"
        )
    return NEUTRAL_MODELS[name](options)
