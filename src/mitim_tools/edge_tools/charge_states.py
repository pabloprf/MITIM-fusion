"""
charge_states.py
----------------
Aurora-based charge-state solvers for PORTALS-Edge.

All models share the public interface::

    model = <Model>(options)
    model.solve(powerstate, batch_idx=0)

After ``solve()``, the following keys are written into ``powerstate.plasma``:

    plasma['nz_all']         : torch.Tensor (batch, rho_pts, nZ+1)  [1e19 m⁻³]
        Charge-state density profiles for charge states 0 (neutral) through Z.
        Index 0 corresponds to the neutral (ground-state) population, index k to
        the k-th ionization stage.

    plasma['qrad_aurora']    : torch.Tensor (batch, rho_pts)         [W cm⁻³]
        Total impurity radiation power density (line + continuum, summed over
        charge states) on the powerstate radial grid.

    plasma['nu_scd_imp']     : torch.Tensor (batch, rho_pts, nZ+1)   [s⁻¹]
        Effective impurity ionisation frequency per charge state,
        ``nu_scd = n_e × S_cd``  where *S_cd* [cm³ s⁻¹] is the ADAS rate coefficient.

    plasma['nu_acd_imp']     : torch.Tensor (batch, rho_pts, nZ+1)   [s⁻¹]
        Effective impurity recombination frequency per charge state,
        ``nu_acd = n_e × α_cd``.

``plasma['n0']`` is **not** written by these models.  It is reserved for the main-ion
neutral density (D⁰ in a deuterium plasma) produced by a dedicated neutrals module
(e.g. KN1D or a 1-D analytic neutral model).  Access the impurity neutral population
via ``plasma['nz_all'][..., 0]``.

``AuroraChargeStates`` also writes (when ``update_ni_charge_balance=True``):

    plasma['ni']             : torch.Tensor (batch, rho_pts, n_species)  [1e19 m⁻³]
        Main-ion density at species index ``main_ion_species_index`` updated via
        quasi-neutrality:  n_main = n_e − Σ_{z=1}^{Z_imp} z × n_z(r).

All Aurora calculations use CGS units internally (ne in cm⁻³, Te in eV, radii in cm).

Notes
-----
* Only the analytically-solved steady state  (``run_aurora_steady_analytic``)
  is supported here — no time-dependent transport.
* The powerstate ``roa`` coordinate is used as a proxy for Aurora's ``rhop``
  (poloidal flux radius) during profile interpolation.  For configurations with
  significant Shafranov shift (large ε), a proper mapping via the g-file is
  preferable.
* Results are interpolated back onto the powerstate ``rmin`` radial grid via
  linear interpolation from Aurora's internal rvol_grid.
"""

import sys
import numpy as np
import torch
from pathlib import Path

from mitim_tools.misc_tools.LOGtools import printMsg as print

# Aurora is an optional dependency.  Failures are caught at instantiation time
# so that the rest of MITIM can be imported even when Aurora is unavailable.
try:
    import aurora as _aurora_pkg
    _AURORA_AVAILABLE = True
except ImportError:
    _aurora_pkg = None
    _AURORA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ChargeStateModel:
    """Abstract base — subclasses implement ``solve(powerstate, batch_idx)``."""

    def __init__(self, options: dict):
        self.options = options

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Populate ``powerstate.plasma`` with charge-state data.

        Must set at minimum:
          ``plasma['nz_all']``   (batch, rho_pts, nZ+1)  [1e19 m⁻³]
          ``plasma['n0']``       (batch, rho_pts)          [1e19 m⁻³]
          ``plasma['qrad_aurora']``  (batch, rho_pts)      [W cm⁻³]
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NullChargeStates — no-op for early development / unit testing
# ---------------------------------------------------------------------------

class NullChargeStates(ChargeStateModel):
    """
    Sets all charge-state arrays to zero.  Useful for testing the downstream
    code paths before Aurora is available or configured.
    """

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        p = powerstate.plasma
        batch = p["te"].shape[0]
        n_rho = p["te"].shape[1]

        zero2d = p["te"] * 0.0    # (batch, rho)
        zero3d = zero2d.unsqueeze(-1)  # (batch, rho, 1) → expanded below

        # Two charge states (neutral + fully stripped) as a minimal placeholder
        p["nz_all"]      = zero3d.expand(batch, n_rho, 2).clone()
        p["qrad_aurora"] = zero2d.clone()
        p["nu_scd_imp"]  = zero3d.expand(batch, n_rho, 2).clone()
        p["nu_acd_imp"]  = zero3d.expand(batch, n_rho, 2).clone()


# ---------------------------------------------------------------------------
# AuroraChargeStates — steady-state Aurora transport
# ---------------------------------------------------------------------------

class AuroraChargeStates(ChargeStateModel):
    """
    Steady-state impurity charge-state distribution and radiation using Aurora.

    Parameters
    ----------
    options : dict
        Configuration for the Aurora run.  Relevant keys:

        ``imp`` : str, default ``"C"``
            Impurity element symbol (e.g. "C", "W", "Ar").
        ``main_element`` : str, default ``"D"``
            Background hydrogenic species.
        ``D_z_m2_s`` : float, default 0.1
            Spatially-uniform diffusion coefficient [m²/s].
        ``V_z_m_s`` : float, default -0.5
            Amplitude of the linear-ramp pinch velocity [m/s].
            The radial profile is  V(r) = V_z_m_s × r/rvol_lcfs.
        ``source_rate`` : float, default 1e21
            Total impurity injection rate [s⁻¹].  Aurora's steady-state solver
            uses this directly; the absolute density profile is set by the
            balance of source, diffusion, and pinch — no post-hoc rescaling is
            applied.
        ``cxr_flag`` : bool, default False
            Whether to enable charge-exchange recombination.
        ``max_dilution_fraction`` : float, default 1.0
            Maximum allowed peak impurity charge fraction
            ``max_r[ sum_z z*nz_z(r) / ne(r) ]`` before source_rate is
            reduced. Set to 1.0 (100%) to allow realistic impurity transport
            up to the quasi-neutrality limit.
            The solver rescales ``source_rate`` exactly (one step is sufficient
            due to linearity of the steady-state solve) and re-runs Aurora to
            verify. Up to ``max_source_rate_iters`` iterations are performed.
        ``max_source_rate_iters`` : int, default 5
            Maximum number of source-rate relaxation iterations.
        ``update_ni_charge_balance`` : bool, default True
            If True, update ``plasma['ni']`` for the main ion after each Aurora
            run using quasi-neutrality.
        ``main_ion_species_index`` : int, default 0
            Species index in ``plasma['ni'][b, :, idx]`` that corresponds to the
            main ion (e.g. deuterium).
        ``verbose`` : bool, default False
    """

    def __init__(self, options: dict):
        super().__init__(options)
        if not _AURORA_AVAILABLE:
            raise ImportError(
                "Aurora is not installed.  "
                "Install it from /aurora or via `pip install aurora-fusion`."
            )
        self.imp             = options.get("imp",             "C")
        self.main_element    = options.get("main_element",    "D")
        self.D0              = options.get("D_z_m2_s",       0.1)
        self.V0              = options.get("V_z_m_s",       -0.5)
        self.source_rate     = options.get("source_rate",     1e21)
        self.cxr_flag        = options.get("cxr_flag",        False)
        self.max_dilution_fraction  = options.get("max_dilution_fraction",   1.0)
        self.max_source_rate_iters  = options.get("max_source_rate_iters",   5)
        self.update_ni              = options.get("update_ni_charge_balance", True)
        self.main_ion_species_index = options.get("main_ion_species_index",   0)
        self.verbose         = options.get("verbose",         False)

    # ------------------------------------------------------------------

    def _build_namelist(self, powerstate, b: int, source_rate: float = None) -> dict:
        """Construct a minimal Aurora namelist from powerstate tensors."""
        from aurora.default_nml import load_default_namelist
        nml = load_default_namelist()

        p = powerstate.plasma

        def _lcfs_scalar(tensor):
            if not isinstance(tensor, torch.Tensor):
                return float(tensor)
            if tensor.dim() == 0:
                return float(tensor.item())
            if tensor.dim() == 1:
                return float(tensor[-1].item())
            return float(tensor[b, -1].item())

        # Geometry
        rmin_lcfs_m = _lcfs_scalar(p["rmin"])
        eps_lcfs    = _lcfs_scalar(p["eps"])
        rvol_lcfs_cm  = rmin_lcfs_m * 100.0           # volume-avg minor radius at LCFS [cm]
        Raxis_cm      = rmin_lcfs_m / max(eps_lcfs, 1e-6) * 100.0

        nml["rvol_lcfs"]    = rvol_lcfs_cm
        nml["Raxis_cm"]     = Raxis_cm
        nml["imp"]          = self.imp
        nml["main_element"] = self.main_element
        nml["source_rate"]  = self.source_rate if source_rate is None else source_rate
        nml["recycling_flag"] = False
        nml["cxr_flag"]     = self.cxr_flag

        # Build kinetic profiles on the powerstate roa grid (proxy for rhop)
        roa_1d  = p["roa"][b, :].cpu().numpy()           # proxy for rhop (0 → 1)
        ne_cm3  = p["ne"] [b, :].cpu().numpy() * 1e13   # 1e19 m⁻³ → cm⁻³
        Te_eV   = p["te"] [b, :].cpu().numpy() * 1e3    # keV → eV
        if p["ti"].dim() >= 2:
            Ti_eV = p["ti"][b, :].cpu().numpy() * 1e3
        else:
            Ti_eV = Te_eV.copy()

        # Ensure minimum physical values
        ne_cm3 = np.maximum(ne_cm3, 1e8)
        Te_eV  = np.maximum(Te_eV,  1.0)
        Ti_eV  = np.maximum(Ti_eV,  1.0)

        nml["kin_profs"] = {
            "ne": {"fun": "interpa", "times": [1.0],
                   "vals": ne_cm3,  "rhop": roa_1d},
            "Te": {"fun": "interpa", "times": [1.0],
                   "vals": Te_eV,   "rhop": roa_1d},
            "Ti": {"fun": "interpa", "times": [1.0],
                   "vals": Ti_eV,   "rhop": roa_1d},
        }

        if self.cxr_flag:
            # Use pre-computed main-ion neutral density when available (from calculateNeutrals)
            if "n0" in p and p["n0"].abs().max().item() > 1e-30:
                n0_1e19 = p["n0"][b, :].cpu().numpy()
                n0_cm3  = n0_1e19 * 1e13          # 1e19 m⁻³ → cm⁻³
                nml["kin_profs"]["n0"] = {
                    "fun": "interpa", "times": [1.0],
                    "vals": n0_cm3,   "rhop": roa_1d,
                }
            else:
                print(
                    f"[AuroraChargeStates] cxr_flag=True but no usable plasma['n0'] is available for batch {b}. "
                    "Disabling CX for this Aurora call instead of fabricating a neutral profile.",
                    typeMsg="w",
                )
                nml["cxr_flag"] = False

        # Use one steady-state time point
        nml["timing"] = {
            "dt_increase":    np.array([1.005, 1.0]),
            "dt_start":       np.array([1e-5, 0.001]),
            "steps_per_cycle": np.array([1, 1]),
            "times":          np.array([0.0, 0.1]),
        }

        return nml

    def _transport_coefficients(self, asim) -> tuple:
        """Return (D_z, V_z) arrays on Aurora's rvol_grid."""
        nr       = len(asim.rvol_grid)
        nZ_plus1 = asim.Z_imp + 1         # includes neutral

        D_z = self.D0 * 1e4 * np.ones((nr, nZ_plus1))   # m²/s → cm²/s
        # Linearly increasing inward pinch from axis to LCFS
        v_ramp   = self.V0 * 100.0 * asim.rvol_grid / max(asim.rvol_lcfs, 1e-6)  # m/s → cm/s
        V_z = v_ramp[:, None] * np.ones((1, nZ_plus1))

        return D_z, V_z

    # ------------------------------------------------------------------

    def _run_aurora_with_relaxation(self, powerstate, b: int):
        """
        Run Aurora with adaptive source-rate reduction.

        The steady-state analytic solver is a direct linear solve, so
        ``nz_steady ∝ source_rate`` exactly.  One rescaling step is therefore
        mathematically sufficient to hit any dilution target; subsequent
        iterations serve only as verification passes.

        Failure modes
        -------------
        *Negative densities* — indicates a near-singular transport matrix
        (check D_z / V_z); **cannot** be fixed by source_rate scaling since
        the sign is scale-invariant.  Returns ``None`` → zeros are written.

        *Peak dilution > max_dilution_fraction* — trace-impurity limit
        exceeded.  source_rate is rescaled by ``max_dilution / peak_dilution``
        and Aurora is re-run.  A warning is emitted with the scale factor.

        Returns
        -------
        tuple (asim, nz_on_ps, rmin_clip) or None on hard failure.
          asim       : aurora_sim object (needed for rate extraction)
          nz_on_ps   : ndarray (nZ+1, n_rho) in cm⁻³ on powerstate grid
          rmin_clip  : ndarray (n_rho,) powerstate minor radii clipped to
                       Aurora grid [cm]
        """
        p      = powerstate.plasma
        n_rho  = p["te"].shape[1]
        ne_cm3 = p["ne"][b, :].cpu().numpy() * 1e13   # cm⁻³, for dilution check
        roa_1d = p["roa"][b, :].cpu().numpy()          # proxy for rhop

        source_rate = self.source_rate

        for iteration in range(self.max_source_rate_iters + 1):
            nml = self._build_namelist(powerstate, b, source_rate=source_rate)
            try:
                asim = _aurora_pkg.core.aurora_sim(nml)
                D_z, V_z = self._transport_coefficients(asim)
                _, nz_steady = asim.run_aurora_steady_analytic(D_z, V_z)
            except Exception as exc:
                print(
                    f"[AuroraChargeStates] Aurora run failed for batch {b} "
                    f"(iter {iteration}, source_rate={source_rate:.3e}): {exc}.  "
                    f"Inserting zeros.",
                    typeMsg="w",
                )
                return None

            # Hard failure: negative densities — cannot be fixed by source_rate scaling.
            nz_abs_max = np.abs(nz_steady).max()
            if nz_steady.min() < -1e-3 * max(nz_abs_max, 1e-30):
                print(
                    f"[AuroraChargeStates] Negative charge-state densities detected "
                    f"(batch {b}, iter {iteration}, min={nz_steady.min():.3e} cm⁻³). "
                    f"This indicates a near-singular transport matrix; inspect D_z/V_z. "
                    f"Inserting zeros.",
                    typeMsg="w",
                )
                return None

            # User-facing dilution check: max_r [ sum_z z*nz_z(r) / ne(r) ]
            # Map ne onto Aurora's rhop_grid using the same roa proxy.
            rhop_aurora = asim.rvol_grid / max(asim.rvol_lcfs, 1e-10)
            ne_aurora   = np.interp(rhop_aurora, roa_1d, ne_cm3).clip(1e8)

            # Aurora can return nz_steady on one fewer radial cell than rvol_grid.
            n_common = min(nz_steady.shape[1], ne_aurora.shape[0])
            nz_use = nz_steady[:, :n_common]
            ne_use = ne_aurora[:n_common]

            nz_positive = np.maximum(nz_use, 0.0)  # shape (nZ+1, nr)
            Z_vec = np.arange(nz_use.shape[0], dtype=nz_use.dtype)
            peak_dilution = ((nz_positive * Z_vec[:, None]).sum(axis=0) / ne_use).max()

            if peak_dilution > 1.0 + 1e-6:
                print(
                    f"[AuroraChargeStates] Peak impurity charge fraction {peak_dilution:.4f} exceeds 1.0 "
                    f"for batch {b} (iter {iteration}). Charge balance will drive the main-ion density negative; "
                    "inspect source_rate / transport coefficients.",
                    typeMsg="w",
                )

            # Avoid noisy repeated "rescale by 1.0" passes from tiny FP overages.
            if peak_dilution <= self.max_dilution_fraction * (1.0 + 1e-6):
                break  # acceptable solution — exit loop

            # Exact rescale (one step is sufficient due to linearity)
            scale = self.max_dilution_fraction / peak_dilution
            new_source_rate = source_rate * scale
            # If scale is numerically unity, no meaningful correction is left.
            if np.isclose(scale, 1.0, rtol=1e-6, atol=1e-10):
                break

            print(
                f"[AuroraChargeStates] Peak impurity charge fraction {peak_dilution:.4f} exceeds "
                f"limit {self.max_dilution_fraction:.4f} for batch {b} "
                f"(iter {iteration}).  Rescaling source_rate by {scale:.3e}: "
                f"{source_rate:.3e} → {new_source_rate:.3e} s⁻¹.",
                typeMsg="w",
            )
            source_rate = new_source_rate

            if iteration == self.max_source_rate_iters:
                print(
                    f"[AuroraChargeStates] Source-rate relaxation did not verify "
                    f"within {self.max_source_rate_iters} iterations for batch {b}.  "
                    f"Using last result (peak dilution={peak_dilution:.4f}).",
                    typeMsg="w",
                )

        # Interpolate steady-state densities onto the powerstate rmin grid
        rvol_aurora = asim.rvol_grid
        n_grid = min(rvol_aurora.shape[0], nz_steady.shape[1])
        rvol_use = rvol_aurora[:n_grid]
        nz_use = nz_steady[:, :n_grid]

        rmin_ps_cm  = p["rmin"][b, :].cpu().numpy() * 100.0  # m → cm
        rmin_clip   = np.clip(rmin_ps_cm, rvol_use[0], rvol_use[-1])

        nZ_plus1 = nz_use.shape[0]
        nz_on_ps = np.zeros((nZ_plus1, n_rho))   # cm⁻³
        for z in range(nZ_plus1):
            nz_on_ps[z, :] = np.interp(rmin_clip, rvol_use, nz_use[z, :])

        return asim, nz_on_ps, rmin_clip

    # ------------------------------------------------------------------

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Run Aurora (with source-rate relaxation if needed) for batch element
        *batch_idx* and store results in ``powerstate.plasma``.
        """
        b      = batch_idx
        p      = powerstate.plasma
        batch  = p["te"].shape[0]
        n_rho  = p["te"].shape[1]
        dfT    = p["te"]

        result = self._run_aurora_with_relaxation(powerstate, b)
        if result is None:
            self._write_zeros(powerstate, b)
            return

        asim, nz_on_ps, rmin_clip = result
        nZ_plus1    = nz_on_ps.shape[0]
        rvol_aurora = asim.rvol_grid

        # Convert cm⁻³ → 1e19 m⁻³  (1 cm⁻³ = 1e-13 × 1e19 m⁻³)
        nz_1e19 = nz_on_ps * 1e-13

        # Total radiation via Aurora's compute_rad
        ne_cm3 = p["ne"][b, :].cpu().numpy() * 1e13               # cm⁻³
        ne_arr = ne_cm3[np.newaxis, :]                              # (1, n_rho)
        Te_arr = (p["te"][b, :].cpu().numpy() * 1e3)[np.newaxis, :]  # (1, n_rho) eV
        nz_arr = nz_on_ps[np.newaxis, :, :]                        # (1, nZ+1, n_rho) cm⁻³

        try:
            rad_res   = _aurora_pkg.radiation.compute_rad(
                self.imp, nz_arr, ne_arr, Te_arr, prad_flag=True)
            qrad_wcm3 = rad_res["tot"][0, :]
        except Exception as exc:
            if self.verbose:
                print(f"[AuroraChargeStates] compute_rad failed: {exc}", typeMsg="w")
            qrad_wcm3 = np.zeros(n_rho)

        nz_tensor = torch.from_numpy(nz_1e19.T).to(dfT)    # (n_rho, nZ+1)
        qrad_t    = torch.from_numpy(qrad_wcm3).to(dfT)     # (n_rho,)

        # Atomic rate frequencies: Sne_rates / Rne_rates shape (nr_aurora, nZ+1, nt)
        nu_scd_a = asim.Sne_rates[:, :, 0]
        nu_acd_a = asim.Rne_rates[:, :, 0]
        _ir = lambda arr: np.column_stack([
            np.interp(rmin_clip, rvol_aurora, arr[:, z]) for z in range(nZ_plus1)
        ])
        nu_scd_t = torch.from_numpy(_ir(nu_scd_a)).to(dfT)
        nu_acd_t = torch.from_numpy(_ir(nu_acd_a)).to(dfT)

        # Initialise or resize batch-aware output arrays
        _kw = {"dtype": dfT.dtype, "device": dfT.device}
        if ("nz_all" not in p
                or p["nz_all"].dim() != 3
                or p["nz_all"].shape[0] != batch
                or p["nz_all"].shape[1] != n_rho
                or p["nz_all"].shape[2] != nZ_plus1):
            p["nz_all"]      = torch.zeros(batch, n_rho, nZ_plus1, **_kw)
            p["qrad_aurora"] = torch.zeros(batch, n_rho, **_kw)
            p["nu_scd_imp"]  = torch.zeros(batch, n_rho, nZ_plus1, **_kw)
            p["nu_acd_imp"]  = torch.zeros(batch, n_rho, nZ_plus1, **_kw)

        p["nz_all"][b]      = nz_tensor
        p["qrad_aurora"][b] = qrad_t
        p["nu_scd_imp"][b]  = nu_scd_t
        p["nu_acd_imp"][b]  = nu_acd_t

        # Charge-balance update of main ion density
        if self.update_ni:
            self._update_ni_charge_balance(powerstate, b, nz_tensor)

        # Keep Zeff consistent with the latest ni / ne state.
        self._update_zeff(powerstate, b)

    # ------------------------------------------------------------------

    def _write_zeros(self, powerstate, b: int) -> None:
        p     = powerstate.plasma
        batch = p["te"].shape[0]
        n_rho = p["te"].shape[1]
        _kw   = {"dtype": p["te"].dtype, "device": p["te"].device}
        if "nz_all" not in p or p["nz_all"].shape[0] != batch:
            p["nz_all"]      = torch.zeros(batch, n_rho, 2, **_kw)
            p["qrad_aurora"] = torch.zeros(batch, n_rho, **_kw)
            p["nu_scd_imp"]  = torch.zeros(batch, n_rho, 2, **_kw)
            p["nu_acd_imp"]  = torch.zeros(batch, n_rho, 2, **_kw)

    # ------------------------------------------------------------------

    def _update_ni_charge_balance(self, powerstate, b: int, nz_tensor) -> None:
        """
        Update ``plasma['ni']`` for the main ion using quasi-neutrality:

            n_main(r) = n_e(r) − Σ_{z=1}^{Z_imp} z · n_z(r)

        where the sum covers all ionisation stages (neutral z=0 contributes
        zero to the electron inventory).

        Warns if the result is negative anywhere, which would indicate that
        the impurity charge density locally exceeds the electron density
        and the trace-impurity approximation has broken down.

        Parameters
        ----------
        nz_tensor : torch.Tensor, shape (n_rho, nZ+1)  [1e19 m⁻³]
            Charge-state densities on the powerstate radial grid.
        """
        p = powerstate.plasma
        if "ni" not in p:
            return

        nZ_plus1 = nz_tensor.shape[-1]
        # z-weight vector [0, 1, 2, ..., Z_imp]  shape (nZ+1,)
        Z_vec = torch.arange(nZ_plus1, dtype=nz_tensor.dtype, device=nz_tensor.device)

        # Impurity contribution to electron density  [1e19 m⁻³]: Σ_z  z · nz_z
        imp_charge_dens = (nz_tensor * Z_vec).sum(dim=-1)   # (n_rho,)

        ni_main_new = p["ne"][b, :] - imp_charge_dens        # (n_rho,)

        ni_min = ni_main_new.min().item()
        if ni_min < 0.0:
            print(
                f"[AuroraChargeStates] Charge balance gives ni_main < 0 "
                f"(min={ni_min:.3e} ×10¹⁹ m⁻³) for batch {b}.  "
                f"Impurity charge density exceeds n_e — trace approximation "
                f"has broken down.  Clamping to 0.",
                typeMsg="w",
            )

        idx = self.main_ion_species_index
        ni  = p["ni"]
        if ni.dim() == 3 and idx < ni.shape[2]:
            ni[b, :, idx] = ni_main_new.clamp(min=0.0)
        elif ni.dim() == 2:
            ni[b, :]      = ni_main_new.clamp(min=0.0)

    def _update_zeff(self, powerstate, b: int) -> None:
        """
        Update ``plasma['Zeff']`` for batch element ``b`` using:

            Zeff(r) = sum_i[ n_i(r) * Z_i^2 ] / n_e(r)

        where thermal-ion densities and charges are taken from ``plasma['ni']``
        and ``plasma['ions_set_Zi']`` respectively.
        """
        p = powerstate.plasma
        if ("ni" not in p) or ("ne" not in p) or ("ions_set_Zi" not in p):
            return

        ni = p["ni"]
        ne = p["ne"]
        Zi = p["ions_set_Zi"]

        if ni.dim() != 3 or ne.dim() != 2:
            return

        if Zi.dim() == 1 and Zi.shape[0] == ni.shape[-1]:
            Zi_b = Zi
        elif Zi.dim() == 2 and Zi.shape[0] == ni.shape[0] and Zi.shape[1] == ni.shape[-1]:
            Zi_b = Zi[b, :]
        else:
            if self.verbose:
                print(
                    f"[AuroraChargeStates] Cannot update Zeff: ions_set_Zi shape "
                    f"{tuple(Zi.shape)} incompatible with ni shape {tuple(ni.shape)}.",
                    typeMsg="w",
                )
            return

        Zeff_b = (ni[b, :, :] * (Zi_b.unsqueeze(0) ** 2)).sum(dim=-1) / ne[b, :].clamp(min=1e-30)

        _kw = {"dtype": ne.dtype, "device": ne.device}
        if ("Zeff" not in p) or (p["Zeff"].shape != ne.shape):
            p["Zeff"] = torch.zeros_like(ne, **_kw)
        p["Zeff"][b, :] = Zeff_b


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CHARGE_STATE_MODELS: dict = {
    "Null":            NullChargeStates,
    "none":            NullChargeStates,
    "Aurora":          AuroraChargeStates,
    "AuroraSteady":    AuroraChargeStates,
}


def build_charge_state_model(name: str, options: dict) -> ChargeStateModel:
    """
    Instantiate a charge-state model by name.

    Parameters
    ----------
    name : str
        One of the keys in ``CHARGE_STATE_MODELS``.
    options : dict
        Model-specific options forwarded to the class constructor.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in CHARGE_STATE_MODELS:
        raise KeyError(
            f"Unknown charge-state model '{name}'.  "
            f"Available: {list(CHARGE_STATE_MODELS.keys())}"
        )
    return CHARGE_STATE_MODELS[name](options)
