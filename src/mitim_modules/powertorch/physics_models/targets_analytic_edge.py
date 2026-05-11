"""
targets_analytic_edge.py
------------------------
Edge-specific target model that extends ``analytical_model`` with the following
physics additions relevant for the pedestal / SOL:

  - Fine/coarse-grid interpolation of edge-specific plasma keys
  - Aurora impurity radiation replacing the TGYRO Chebyshev contribution
    for the tracked impurity (avoids double-counting)
  - Aurora H/D main-ion neutral radiation
  - Particle flux targets with ionization sources (D⁰ + impurity net ionization)
  - Ionization power loss in the electron energy channel

Usage
-----
Pass as ``target_options['evaluator'] = analytical_model_edge`` when constructing a
``powerstate_edge`` object.

The three edge-specific methods read the following keys populated by
``powerstate_edge.calculateChargeStates()`` (which runs before ``calculateTargets()``
in the overridden ``calculate()`` sequence):

  plasma['nz_all']         (batch, rho, nZ+1)   [1e19 m⁻³]
      Impurity charge-state density profiles; index 0 = impurity neutral (Z=0).
  plasma['qrad_aurora']    (batch, rho)          [W cm⁻³]
      Total impurity radiation (line + continuum) from Aurora.
  plasma['nu_scd_imp']     (batch, rho, nZ+1)   [s⁻¹]
      Effective impurity ionisation frequency per charge state (= n_e × S_cd).
  plasma['nu_acd_imp']     (batch, rho, nZ+1)   [s⁻¹]
      Effective impurity recombination frequency per charge state (= n_e × α_cd).

Edge keys consumed
------------------
  plasma['nz_all']         (batch, rho, nZ+1)   [1e19 m⁻³]
  plasma['qrad_aurora']    (batch, rho)          [W cm⁻³]
  plasma['nu_scd_imp']     (batch, rho, nZ+1)   [s⁻¹]
  plasma['nu_acd_imp']     (batch, rho, nZ+1)   [s⁻¹]
  plasma['n0']             (batch, rho)          [1e19 m⁻³]
  plasma['S_ion_main']     (batch, rho)          [1e19 m⁻³ s⁻¹]
  plasma['tau_n0']         (batch, rho)          [dimensionless]
    plasma['qpar_main']     (batch, rho)          [1e20 m⁻³ s⁻¹]
    plasma['qpar_imp']      (batch, rho)          [1e20 m⁻³ s⁻¹]
    plasma['qpar_wall']     (batch, rho)          [1e20 m⁻³ s⁻¹]
    plasma['qpar_Z']        (batch, rho)          [1e20 m⁻³ s⁻¹]
"""

from mitim_tools.misc_tools import PLASMAtools
import numpy as np
import torch

from mitim_modules.powertorch.physics_models.targets_analytic import analytical_model
from mitim_tools.misc_tools.LOGtools import printMsg as print


# Edge-specific plasma keys that require grid interpolation
_EDGE_KEYS_2D = ("n0", "tau_n0", "S_ion_main", "nu_ioniz_main", "qrad_aurora", "qiziz_loss")
_EDGE_KEYS_3D = ("nz_all", "nu_scd_imp", "nu_acd_imp")


def _ensure_1e19_units(x: "torch.Tensor", name: str, threshold: float = 1e16) -> "torch.Tensor":
    """
    Ensure internal MITIM normalization (1e19-based units) for density-like arrays.

    Some upstream paths can provide SI values (m^-3 or m^-3 s^-1) while this
    model assumes 1e19-based storage. If values are clearly SI-scale, convert
    by 1e-19 to prevent O(1e19) flux blow-ups.
    """
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return x

    vmax = x.detach().abs().max().item()
    if vmax > threshold:
        print(
            f"[analytical_model_edge] {name} appears to be in SI units "
            f"(max={vmax:.3e}); converting to 1e19-based units.",
            typeMsg="w",
        )
        return x * 1e-19

    return x


def _edge_postprocessing(powerstate, integrated_targets=None, force_zero_particle_flux=False, relative_error_assumed=1.0):
    """
    Shared postprocessing for edge target models.

    Plugs in fixed targets with integrated source contributions, computes convective fluxes,
    assigns standard errors, and produces GB-normalised outputs.
    """
    p = powerstate.plasma
    P = integrated_targets

    # **************************************************************************************************
    # Combine edge targets with integrated source contributions
    # **************************************************************************************************

    # P may contain 2 blocks (QeMWm2, QiMWm2 only — legacy / no qpar evolution)
    # or 4 blocks (QeMWm2, QiMWm2, Ge, GZ — edge flux_integrate always cats all four).
    # Use the batch size to determine the block stride.
    batch = p["te"].shape[0]
    n_blocks = P.shape[0] // batch  # 2 (legacy) or 4 (edge with ge/gz blocks)

    p["QeMWm2"]  = p["Qe_edgetargets"] + P[         : batch,     :]  # MW/m^2
    p["QiMWm2"]  = p["Qi_edgetargets"] + P[batch    : 2 * batch, :]  # MW/m^2

    if n_blocks >= 4:
        p["Ge1E20m2"] = p["Ge_edgetargets"] + P[2 * batch : 3 * batch, :]  # 1E20/s/m^2
        p["GZ1E20m2"] = p["GZ_edgetargets"] + P[3 * batch :,            :]  # 1E20/s/m^2
    else:
        p["Ge1E20m2"] = p["Ge_edgetargets"]   # 1E20/s/m^2
        p["GZ1E20m2"] = p["GZ_edgetargets"]   # 1E20/s/m^2

    p["MtJm2"] = p["Mt_edgetargets"]  # J/m^2  (no integrated source contribution)

    if force_zero_particle_flux:
        p["Ge1E20m2"] = p["Ge1E20m2"] * 0

    # Convective fluxes
    p["Ce"] = PLASMAtools.convective_flux(p["te"], p["Ge1E20m2"])  # MW/m^2
    p["CZ"] = PLASMAtools.convective_flux(p["te"], p["GZ1E20m2"])  # MW/m^2

    # **************************************************************************************************
    # Error
    # **************************************************************************************************

    variables_to_error = ["QeMWm2", "QiMWm2", "Ce", "CZ", "MtJm2", "Ge1E20m2", "GZ1E20m2"]

    for i in variables_to_error:
        p[i + "_stds"] = abs(p[i]) * relative_error_assumed / 100

	# **************************************************************************************************
	# GB Normalized (Note: This is useful for mitim surrogate variables of targets)
	# **************************************************************************************************

    p["QeGB"] = p["QeMWm2"]   / p["Qgb"]
    p["QiGB"] = p["QiMWm2"]   / p["Qgb"]
    p["GeGB"] = p["Ge1E20m2"] / p["Ggb"]
    p["GZGB"] = p["GZ1E20m2"] / p["Ggb"]
    p["CeGB"] = p["Ce"]       / p["Qgb"]
    p["CZGB"] = p["CZ"]       / p["Qgb"]
    p["MtGB"] = p["MtJm2"]    / p["Pgb"]


class analytical_model_edge(analytical_model):
    """
    Edge-specific subclass of ``analytical_model``.

    Execution order in ``evaluate()``
    ----------------------------------
    1.  ``_evaluate_radiation()`` override (this class):
          - Zeros the TGYRO Chebyshev contribution for the Aurora-tracked impurity,
            calls base ``_evaluate_radiation()``, restores coefficients, adds
            ``qrad_aurora``, and adds Aurora H/D neutral radiation.
    2.  ``_evaluate_particle_fluxes()`` — D⁰ ionisation source + impurity net
        electron source (trace-regime only).
    3.  ``_evaluate_ionization_loss()`` — ionisation energy cost from qie.
    """

    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def flux_integrate(self):
        """
		**************************************************************************************************
		Calculate integral of all targets, and then sum aux.
		Reason why I do it this convoluted way is to make it faster in mitim, not to run the volume integral all the time.
		Run once for all the batch and also for electrons and ions
		(in MW/m^2)
		**************************************************************************************************
		"""

        qe = torch.zeros_like(self.powerstate.plasma["te"])
        qi = torch.zeros_like(self.powerstate.plasma["te"])
        ge = torch.zeros_like(self.powerstate.plasma["te"])
        gz = torch.zeros_like(self.powerstate.plasma["te"])
        
        if "qie" in self.powerstate.target_options['options']['targets_evolve']:
            qe += -self.powerstate.plasma["qie"]
            qi +=  self.powerstate.plasma["qie"]
            if "qiz" in self.powerstate.plasma:
                # Ionization energy is an electron-channel sink, not an e-i exchange term.
                qe -= self.powerstate.plasma["qiz"]

        if "qfus" in self.powerstate.target_options['options']['targets_evolve']:
            qe +=  self.powerstate.plasma["qfuse"]
            qi +=  self.powerstate.plasma["qfusi"]

        if "qrad" in self.powerstate.target_options['options']['targets_evolve']:
            qe -=  self.powerstate.plasma["qrad"]

        if "qpar" in self.powerstate.target_options['options']['targets_evolve']:
            ge += self.powerstate.plasma["qpar_wall"]
            gz += self.powerstate.plasma["qpar_Z"]

        q = torch.cat((qe, qi, ge, gz)).to(qe)
        self.P = self.powerstate.from_density_to_flux(q, force_dim=q.shape[0])

    # ------------------------------------------------------------------
    # evaluate() — extend base with edge terms
    # ------------------------------------------------------------------

    def evaluate(self):
        """
        Extend base ``analytical_model.evaluate()`` with edge-specific physics.

        The call order is:
             1. ``qie`` / ``qfus`` / ``qrad`` are evaluated exactly as in the
                 base model (with edge radiation overrides).
             2. ``_evaluate_particle_fluxes()`` populates local source densities
                 ``qpar_main``, ``qpar_imp``, ``qpar_wall`` and ``qpar_Z``
                 [1e20 m^-3 s^-1].
             3. ``_evaluate_ionization_loss()`` subtracts ionisation energy cost
                 from ``qie`` when that channel is evolved.
        """

        if "qie" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_energy_exchange()
            self._evaluate_ionization_loss()

        if "qfus" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_alpha_heating()

        if "qrad" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_radiation()

        if "qpar" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_particle_fluxes()


    # ------------------------------------------------------------------
    # Override: _evaluate_radiation()
    # ------------------------------------------------------------------

    def _evaluate_radiation(self):
        """
        Compute edge radiation only (Aurora impurity + Aurora H/D neutrals).

        This intentionally skips ``super()._evaluate_radiation()`` if edge modules are active
        and builds all radiation channels directly on the active grid (the current
        ``plasma['te']`` shape) to avoid fine/coarse length mismatches.
        """
        p = self.powerstate.plasma
        p["qrad_bremms"] = p["te"] * 0.0
        p["qrad_line"] = p["te"] * 0.0
        p["qrad_sync"] = p["te"] * 0.0
        p["qrad"] = p["te"] * 0.0

        has_aurora_rad = (
            "qrad_aurora" in p
            and p["qrad_aurora"].abs().max().item() > 1e-30
        )
        has_neutrals = (
            "n0" in p and p["n0"].abs().max().item() > 1e-30
        )

        if has_aurora_rad:
            qrad_aurora = p["qrad_aurora"]
            if qrad_aurora.shape != p["qrad"].shape:
                if (
                    qrad_aurora.dim() == 2
                    and p["qrad"].dim() == 2
                    and qrad_aurora.shape[0] == p["qrad"].shape[0]
                ):
                    src_x = np.linspace(0.0, 1.0, qrad_aurora.shape[-1])
                    dst_x = np.linspace(0.0, 1.0, p["qrad"].shape[-1])
                    qrad_np = qrad_aurora.detach().cpu().numpy()
                    qrad_aurora = torch.from_numpy(
                        np.stack([np.interp(dst_x, src_x, qrad_np[i, :]) for i in range(qrad_np.shape[0])], axis=0)
                    ).to(p["qrad"])
                else:
                    print(
                        f"[analytical_model_edge] qrad_aurora shape {qrad_aurora.shape} "
                        f"incompatible with qrad shape {p['qrad'].shape}; skipping.",
                        typeMsg="w",
                    )
                    qrad_aurora = None

            if qrad_aurora is not None:
                qrad_aurora = qrad_aurora.to(p["qrad"])
                p["qrad"] = p["qrad"] + qrad_aurora

        if has_neutrals:
            # Add Aurora H/D main-ion neutral line radiation
            self._add_aurora_H_radiation()

        # Add with legacy method if needed (bremss + line + sync)
        if not has_aurora_rad:
            super()._evaluate_radiation()

    # ------------------------------------------------------------------
    # Aurora H radiation
    # ------------------------------------------------------------------

    def _add_aurora_H_radiation(self):
        """
        Compute and add line radiation from main-ion (D/H) neutrals using
        Aurora's ``compute_rad``.

        Constructs ``nz_H[t=0, z, r]`` from:
          - ``plasma['n0']``  → D⁰ ground-state population (z=0)
          - ``plasma['ni'][:,:,0]`` → D⁺ main-ion population (z=1)

        Calls ``aurora.radiation.compute_rad("H", ...)`` and adds the
        resulting total radiated power to ``plasma['qrad']``.
        """
        try:
            import aurora as _aurora_pkg
        except ImportError:
            return

        p = self.powerstate.plasma
        if "n0" not in p or p["n0"].abs().max().item() < 1e-30:
            return

        for b in range(p["te"].shape[0]):
            ne_cm3 = p["ne"][b, :].cpu().numpy() * 1e13    # cm⁻³
            Te_eV  = p["te"][b, :].cpu().numpy() * 1e3     # eV
            n0_cm3 = p["n0"][b, :].cpu().numpy() * 1e13    # cm⁻³

            if "ni" in p and p["ni"].dim() == 3:
                ni_D_cm3 = p["ni"][b, :, 0].cpu().numpy() * 1e13
            else:
                ni_D_cm3 = ne_cm3.copy()

            # nz_H shape: (nt=1, nZ+1=2, n_rho)  — [D⁰, D⁺]
            nz_H   = np.array([n0_cm3, ni_D_cm3])[np.newaxis, :, :]   # (1, 2, n_rho)
            ne_arr = ne_cm3[np.newaxis, :]                             # (1, n_rho)
            Te_arr = Te_eV[np.newaxis, :]                              # (1, n_rho)

            try:
                rad_res  = _aurora_pkg.radiation.compute_rad(
                    "H", nz_H, ne_arr, Te_arr, prad_flag=True)
                qrad_H_t = torch.from_numpy(
                    rad_res["tot"][0, :].copy()).to(p["qrad"])
            except Exception as exc:
                if self.powerstate.target_options["options"].get("verbose", False):
                    print(
                        f"[analytical_model_edge] Aurora H radiation compute_rad failed for batch {b}: {exc}",
                        typeMsg="w",
                    )
                return   # fail gracefully; H ADAS data may not be available

            if qrad_H_t.shape[-1] != p["qrad"].shape[-1]:
                src_x = np.linspace(0.0, 1.0, qrad_H_t.shape[-1])
                dst_x = np.linspace(0.0, 1.0, p["qrad"].shape[-1])
                qrad_H_t = torch.from_numpy(
                    np.interp(dst_x, src_x, qrad_H_t.detach().cpu().numpy())
                ).to(p["qrad"])

            p["qrad"][b] = p["qrad"][b] + qrad_H_t

    # ------------------------------------------------------------------
    # particle flux sources from neutral ionization
    # ------------------------------------------------------------------

    def _evaluate_particle_fluxes(self):
        """
        Populate local particle source densities ``qpar_main``, ``qpar_imp``,
        ``qpar_wall`` and ``qpar_Z``.

        Two contributions:
        1.  **D⁰ ionisation** — from ``plasma['S_ion_main']`` [1e19 m⁻³ s⁻¹]
            (ADAS-based, pre-computed by ``calculateNeutrals()``).
        2.  **Impurity net ionisation (electrons)** —
            ``Σ(scd·nz) - Σ(acd·nz)`` over charge states, applied only when
            peak charge-weighted dilution ``max(Σ z·nz / ne) < 0.1``
            (trace-impurity regime).
        3.  **Fully ionised impurity source** — net source for the fully
            stripped stage only, used for ``GZ`` coupling.

        Unit accounting
        ---------------
        ``qpar_*`` are stored as [1e20 m⁻³ s⁻¹] by multiplying source rates
        [1e19 m⁻³ s⁻¹] by 0.1. They are converted to fluxes in ``flux_integrate``.
        """
        p = self.powerstate.plasma

        p["qpar_main"] = p["te"] * 0.0
        p["qpar_imp"] = p["te"] * 0.0
        p["qpar_wall"] = p["te"] * 0.0
        p["qpar_Z"] = p["te"] * 0.0

        # ── 1. D⁰ ionisation source ─────────────────────────────────────────
        if "n0" in p and p["n0"].abs().max().item() > 1e-30:
            n0_1e19 = _ensure_1e19_units(p["n0"], "n0")
            ne_1e19 = _ensure_1e19_units(p["ne"], "ne")

            if "S_ion_main" in p:
                S_ion = _ensure_1e19_units(p["S_ion_main"], "S_ion_main")
            else:
                Te_eV   = p["te"] * 1e3
                # Fallback to Voronov (1997) H ionization fit, consistent with neutrals.py.
                U = 13.6 / Te_eV.clamp(0.1)
                sigma_v = 2.91e-14 * (U ** 0.39) * torch.exp(-U) / (0.232 + U)  # m^3/s
                # n0/ne in [1e19 m^-3] => product is [1e38 m^-6]; multiply by sigma_v and
                # by 1e-19 to return [1e19 m^-3 s^-1].
                S_ion = n0_1e19 * ne_1e19 * sigma_v * 1e-19
                print(
                    "[analytical_model_edge] Using fallback S_ion estimate because S_ion_main is missing.",
                    typeMsg="w",
                )

            p["qpar_main"] = p["qpar_main"] + S_ion * 0.1

        # ── 2. Impurity net electron source (trace regime only) ─────────────
        if (
            "nz_all"     in p
            and "nu_scd_imp" in p
            and "nu_acd_imp" in p
        ):
            nz  = p["nz_all"]       # (batch, rho, nZ+1)  [1e19 m⁻³]
            scd = p["nu_scd_imp"]   # (batch, rho, nZ+1)  [s⁻¹]
            acd = p["nu_acd_imp"]   # (batch, rho, nZ+1)  [s⁻¹]
            ne  = p["ne"]           # (batch, rho)         [1e19 m⁻³]

            nz = _ensure_1e19_units(nz, "nz_all")
            ne = _ensure_1e19_units(ne, "ne")

            nZ_plus1 = nz.shape[-1]
            Z_vec = torch.arange(nZ_plus1, dtype=nz.dtype, device=nz.device)
            charge_dens = (nz * Z_vec).sum(dim=-1)
            dilution = charge_dens / ne.clamp(min=1e-30)
            trace_mask = dilution.max(dim=1).values < 0.1

            # nu_scd_imp / nu_acd_imp already include ne multiplication (s^-1).
            S_imp_iz  = (scd[:, :, :-1] * nz[:, :, :-1]).sum(dim=-1)
            S_imp_rec = (acd[:, :,  1:] * nz[:, :,  1:]).sum(dim=-1)
            S_imp_net = S_imp_iz - S_imp_rec

            if trace_mask.any():
                p["qpar_imp"][trace_mask, :] = p["qpar_imp"][trace_mask, :] + S_imp_net[trace_mask, :] * 0.1

                # Fully stripped impurity stage (charge Z):
                # source from ionisation into Z minus recombination out of Z.
                # scd[..., -2] drives (Z-1 -> Z), acd[..., -1] drives (Z -> Z-1).
                if nZ_plus1 >= 2:
                    S_Z = scd[:, :, -2] * nz[:, :, -2] - acd[:, :, -1] * nz[:, :, -1]
                    p["qpar_Z"][trace_mask, :] = p["qpar_Z"][trace_mask, :] + S_Z[trace_mask, :] * 0.1

        # Electron-wall source is main-ion plus impurity electron source.
        p["qpar_wall"] = p["qpar_main"] + p["qpar_imp"]

    # ------------------------------------------------------------------
    # ionization power loss
    # ------------------------------------------------------------------

    def _evaluate_ionization_loss(self):
        """
        ionisation power subtracted from qe.

        Each D⁰ ionisation event costs ~40 eV (13.6 eV ionisation potential
        plus ~26 eV of prior excitation radiation losses).

        Uses ``plasma['S_ion_main']`` when available; falls back to the
        analytic rate when only ``plasma['n0']`` is present.

        If ``plasma['n0']`` is absent, this is a no-op.
        """
        p = self.powerstate.plasma
        p["qiz"] = torch.zeros_like(p["te"])
        if "n0" not in p or p["n0"].abs().max().item() < 1e-30:
            return

        if "S_ion_main" in p:
            S_ion = _ensure_1e19_units(p["S_ion_main"], "S_ion_main")
        else:
            raise NotImplementedError("Ionisation loss evaluation requires S_ion_main")

        E_ion_eff_J = 40.0 * 1.60218e-19
        # Numerically equivalent to MW/m^3; kept in the same units as qie/qrad arrays.
        Q_ion_MWm3 = S_ion * 1e19 * E_ion_eff_J * 1e-6

        p["qiz"] = Q_ion_MWm3

    def postprocessing(self, force_zero_particle_flux=False, relative_error_assumed=1.0):
        _edge_postprocessing(
            self.powerstate,
            integrated_targets=self.P,
            force_zero_particle_flux=force_zero_particle_flux,
            relative_error_assumed=relative_error_assumed,
        )


class analytical_model_legacy_edge_compat(analytical_model):
    """
    Compatibility adapter for running legacy analytical targets with powerstate_edge.

    This preserves legacy analytical evaluate/flux_integrate physics and only
    patches postprocessing shape assumptions that can break after edge-domain
    trimming.
    """

    def postprocessing(self, force_zero_particle_flux=False, relative_error_assumed=1.0):
        _edge_postprocessing(
            self.powerstate,
            integrated_targets=self.P,
            force_zero_particle_flux=force_zero_particle_flux,
            relative_error_assumed=relative_error_assumed,
        )