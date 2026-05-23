"""
boundary.py
-----------
Separatrix boundary-condition models for PORTALS-Edge.

All models share the same public interface::

    model = <Model>(options)
    model.get_boundary_conditions(powerstate, batch_idx=0)
    bc_dict = model.bc_dict   # {var: [value, rho_location]}

The returned ``bc_dict`` is consumed by
``powerstate_edge.calculateBoundaryConditions()`` which writes the values into
``self._lcfs_bc`` and subsequently enforces them at the end of each
``modify()`` call.

Variable name conventions (match powerstate.plasma keys)
---------------------------------------------------------
ne      : electron density           [1e19 m⁻³]
te      : electron temperature       [keV]
ti      : ion temperature            [keV]
ni      : main-ion density           [1e19 m⁻³]
aLne    : e-density gradient scale   [dimensionless] = a/L_ne
aLte    : e-temp gradient scale      [dimensionless] = a/L_Te
aLti    : ion-temp gradient scale    [dimensionless] = a/L_Ti
aLni    : ion-density gradient scale [dimensionless] = a/L_ni

The ``rho_location`` value of 1 means the variable is pinned at rho = 1.0.
Sub-unity values (not yet used) would pin an inner surface.
"""

import numpy as np
import torch
from scipy.constants import e as e_J
from scipy.optimize import brentq, fsolve

from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.PLASMAtools import md_u as _MD_U

try:
    from mitim_tools.edge_tools.equilibrium import load_equilibrium
except Exception:
    load_equilibrium = None

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BoundaryConditions:
    """
    Abstract base that extracts LCFS scalars from a powerstate batch element.

    Parameters
    ----------
    options : dict
        Model-specific options.  Common keys:
          ``verbose`` (bool, default False)
          ``Zeff``  (float, default None → taken from plasma['Zeff'])
          ``mi_ref_u`` (float, default md_u)  reference ion mass in atomic units
          ``Lpar``  (float, default None → π·q·R₀)  parallel connection length [m]
          ``fmom``  (float, default 0.5)  momentum-loss factor in TFTP
    """

    def __init__(self, options: dict):
        self.verbose = options.get("verbose", False)
        self._Zeff_override = options.get("Zeff", None)
        self.mi_ref = options.get("mi_ref_u", _MD_U)
        self._Lpar_override = options.get("Lpar", None)
        self.fmom = options.get("fmom", 0.5)
        self.bc_dict: dict = {}

    def _extract_lcfs_scalars(self, powerstate, b: int = 0) -> None:
        """
        Extract all LCFS scalars and store them as instance attributes in
        physical units.

        Quantities are extracted from the current ``powerstate.plasma`` batch
        element so each member of a large batch gets unique BC inputs.
        Geometry fallbacks still come from immutable gacode-derived arrays.
        """
        d = powerstate.profiles.derived
        p_raw = powerstate.profiles.profiles
        p = powerstate.plasma

        rho_t = p.get("rho", None)
        rho_ref = None
        if isinstance(rho_t, torch.Tensor):
            if rho_t.dim() == 2:
                rho_ref = rho_t[b, :].detach().cpu().numpy()
            elif rho_t.dim() == 1:
                rho_ref = rho_t.detach().cpu().numpy()

        def _plasma_at_lcfs(tensor, species_idx=None):
            """Interpolate a plasma tensor to rho = 1 (clips to last grid point)."""
            if not isinstance(tensor, torch.Tensor):
                return float(tensor)
            if tensor.dim() == 0:
                return float(tensor.item())
            if tensor.dim() == 1:
                if rho_ref is not None and tensor.shape[0] == len(rho_ref):
                    return float(np.interp(1.0, rho_ref, tensor.detach().cpu().numpy()))
                idx = b if tensor.shape[0] > b else -1
                return float(tensor[idx].item())
            if tensor.dim() == 2:
                arr = tensor[b, :].detach().cpu().numpy()
                return float(np.interp(1.0, rho_ref, arr)) if rho_ref is not None else float(arr[-1])
            if tensor.dim() == 3:
                s = 0 if species_idx is None else int(species_idx)
                arr = tensor[b, :, s].detach().cpu().numpy()
                return float(np.interp(1.0, rho_ref, arr)) if rho_ref is not None else float(arr[-1])
            return float(tensor.reshape(-1)[-1].item())

        def _derived_lcfs(arr, species_idx=None):
            """Interpolate a derived/profile array to rho = 1 on gacode grid."""
            p_raw = powerstate.profiles.profiles
            rho_gacode = np.asarray(p_raw["rho(-)"], dtype=float)
            a = np.asarray(arr, dtype=float)
            if a.ndim == 2:
                a = a[:, 0 if species_idx is None else int(species_idx)]
            return float(np.interp(1.0, rho_gacode, a))

        # Profile quantities at rho = 1 (from current batch state)
        self.ne = _plasma_at_lcfs(p["ne"]) * 1e19   # m⁻³
        self.te = _plasma_at_lcfs(p["te"]) * 1e3    # eV
        self.ti = _plasma_at_lcfs(p["ti"], species_idx=0) * 1e3
        self.ni = _plasma_at_lcfs(p["ni"], species_idx=0) * 1e19

        # Zeff
        if self._Zeff_override is not None:
            self.Zeff = float(self._Zeff_override)
        elif "Zeff" in p:
            self.Zeff = _plasma_at_lcfs(p["Zeff"])
        elif "Zeff" in d:
            self.Zeff = _derived_lcfs(d["Zeff"])
        else:
            self.Zeff = 1.0

        # Equilibrium geometry
        self.r = _derived_lcfs(d["r"])              # minor radius at LCFS [m]
        self.a   = float(d["a"])           # minor radius [m]  (scalar = LCFS value)
        eps_lcfs = float(d["eps"])         # a / R0           (scalar)
        self.aspect_ratio = 1/eps_lcfs
        # R0 via the minor-radius profile at the LCFS (last gacode point = a = rmin[-1])
        self.R0  = d['R_LF'][-1]

        self.BT    = abs(_derived_lcfs(d["B_ref"]))
        self.q = _derived_lcfs(powerstate.profiles.profiles["q(-)"])
        self.Bp    = abs(self.R0**(-1)*np.gradient(p_raw["polflux(Wb/radian)"],d['r']))[-1]#self.BT * eps_lcfs / self.q_cyl if self.q_cyl > 0 else 1e-6

        self.kappa = float(d["kappa995"])
        self.delta = float(d["delta995"])
        self.kappa_hat = np.sqrt((1 + self.kappa**2*(1+2*self.delta**2-1.2*self.delta**3))/2)
        self.q_cyl = self.kappa_hat*np.abs(self.BT)/(self.Bp/eps_lcfs)


        if self._Lpar_override is not None:
            self.Lpar = float(self._Lpar_override)
        else:
            self.Lpar = np.pi * abs(self.q_cyl) * self.R0

        self.shear = _derived_lcfs(d["s_hat"]) if "s_hat" in d else 1.0

        # nuei is recomputed from te/ne in the iterative two-fluid models;
        # keep at 0 here to avoid reading the (potentially outdated) plasma tensor.
        self.nuei = 0.0

        # Derived plasma parameters at rho = 1
        ne_20 = self.ne * 1e-20
        self.LogLam = float(
            PLASMAtools.loglam(
                torch.tensor(self.te * 1e-3),
                torch.tensor(ne_20),
            ).item()
        )
        self.rho_s = PLASMAtools.rho_s(self.te * 1e-3, self.mi_ref, self.BT)
        self.c_s   = PLASMAtools.c_s(self.te * 1e-3, self.mi_ref)

        from scipy.constants import u as _u_kg
        me_kg = 9.1094e-31
        self.me_over_mi = me_kg / (self.mi_ref * _u_kg)

        # Gradient scale lengths at rho = 1
        self.aLne = _plasma_at_lcfs(p["aLne"]) if "aLne" in p else _derived_lcfs(d["aLne"])
        self.aLte = _plasma_at_lcfs(p["aLte"]) if "aLte" in p else _derived_lcfs(d["aLTe"])
        self.aLti = _plasma_at_lcfs(p["aLti"], species_idx=0) if "aLti" in p else _derived_lcfs(d["aLTi"], species_idx=0)
        if "aLni0" in p:
            self.aLni = _plasma_at_lcfs(p["aLni0"])
        elif "aLni" in p:
            self.aLni = _plasma_at_lcfs(p["aLni"], species_idx=0)
        else:
            self.aLni = _derived_lcfs(d["aLni"], species_idx=0)

        # volp at LCFS from the geometry (gacode-derived, not solver-evolved)
        volp_lcfs = float(d["volp_geo"][-1])
        self.A    = volp_lcfs

        if "QeMWm2" in p:
            _Qe_key = "QeMWm2"
        elif "QeMWm2_edgetargets" in p:
            _Qe_key = "QeMWm2_edgetargets"
        else:
            _Qe_key = "QeMWm2_fixedtargets"

        if "QiMWm2" in p:
            _Qi_key = "QiMWm2"
        elif "QiMWm2_edgetargets" in p:
            _Qi_key = "QiMWm2_edgetargets"
        else:
            _Qi_key = "QiMWm2_fixedtargets"
        self.Pe = _plasma_at_lcfs(p[_Qe_key]) * volp_lcfs
        self.Pi = _plasma_at_lcfs(p[_Qi_key]) * volp_lcfs

    def get_boundary_conditions(self, powerstate, batch_idx: int = 0) -> None:
        self._extract_lcfs_scalars(powerstate, batch_idx)


# ---------------------------------------------------------------------------
# Fixed
# ---------------------------------------------------------------------------

class Fixed(BoundaryConditions):
    """
    Freeze the LCFS values for the lifetime of the run.

    Values are taken from ``bc_model_options`` when provided, with fallback to
    the current powerstate for any keys that are omitted. The resulting values
    are stored permanently so ``bc_dict`` stays fixed across iterations.
    """

    def __init__(self, options: dict):
        super().__init__(options)
        self._bc_model_options = dict(options)
        self._fixed_values: dict | None = None

    def _read_fixed_values(self, powerstate) -> dict:
        """
        Extract fixed LCFS scalars from ``bc_model_options`` with fallback.

        Any key supplied in ``bc_model_options`` overrides the value read from
        the plasma state. Keys not supplied fall back to the current LCFS value
        from the powerstate so existing workflows remain usable.
        """
        d = powerstate.profiles.derived
        p = powerstate.profiles.profiles
        roa = np.asarray(d["roa"], dtype=float)

        def _at_lcfs(arr, species_idx=None):
            a = np.asarray(arr, dtype=float)
            if a.ndim == 2:
                a = a[:, 0 if species_idx is None else int(species_idx)]
            return float(np.interp(1.0, roa, a))

        def _from_options_or_profiles(key: str, default_value):
            if key in self._bc_model_options:
                return float(self._bc_model_options[key])
            return float(default_value)

        return {
            "ne":   _from_options_or_profiles("ne",   _at_lcfs(p["ne(10^19/m^3)"])),
            "te":   _from_options_or_profiles("te",   _at_lcfs(p["te(keV)"])),
            "ti":   _from_options_or_profiles("ti",   _at_lcfs(p["ti(keV)"],          species_idx=0)),
            "ni":   _from_options_or_profiles("ni",   _at_lcfs(p["ni(10^19/m^3)"],    species_idx=0)),
            "aLne": _from_options_or_profiles("aLne", _at_lcfs(d["aLne"])),
            "aLte": _from_options_or_profiles("aLte", _at_lcfs(d["aLTe"])),
            "aLti": _from_options_or_profiles("aLti", _at_lcfs(d["aLTi"],             species_idx=0)),
            "aLni": _from_options_or_profiles("aLni", _at_lcfs(d["aLni"],             species_idx=0)),
        }

    def get_boundary_conditions(self, powerstate, batch_idx: int = 0) -> None:
        if self._fixed_values is None:
            self._fixed_values = self._read_fixed_values(powerstate)
        self.bc_dict = {k: [v, 1.0] for k, v in self._fixed_values.items()}


# ---------------------------------------------------------------------------
# TwoFluidTwoPoint_PeretSSF
# ---------------------------------------------------------------------------

class TwoFluidTwoPoint_PeretSSF(BoundaryConditions):
    """
    Iterative LCFS boundary condition solver coupling the Peret 2025 SSF
    turbulent flux-decay-length model with the two-fluid two-point model (TFTP).
    """

    def __init__(self, options: dict):
        super().__init__(options)
        self.max_iter = options.get("max_iter", 100)
        self.tol = options.get("tol", 1e-3)
        self.damping = options.get("damping", 0.5)
        self.G0 = options.get("G0", 1.25)
        self.shear_ref = options.get("shear_ref", 5.0) # (+) for LSN, (-) for USN
        self.f_Delta = options.get("f_Delta", 1.0)
        self.Lambda = options.get("Lambda", 3.0)
        self._gfile_path = options.get("gfile_path", options.get("gfile", None))
        self._eq = None
        self._eq_n_points = int(options.get("eq_n_points", 512))
        self._eq_eddy_width_m = float(options.get("eq_eddy_width_m", 0.005))
        self.ne_target = options.get("ne_target", 3.0)
        self.Te_target = options.get("Te_target", 0.005)
        self.Ti_target = options.get("Ti_target", None)
        self.fq_e = options.get("fq_e", 0.2)
        self.fq_i = options.get("fq_i", 0.2)
        self.fmom = options.get("fmom", 0.8)

    def get_boundary_conditions(self, powerstate, batch_idx: int = 0) -> None:
        super().get_boundary_conditions(powerstate, batch_idx)
        self._solve_lcfs_iterative()

    def _load_equilibrium(self):
        """Load and cache the EquilibriumSSF object from the g-file."""
        if load_equilibrium is None:
            if self.verbose and self._gfile_path is not None:
                print("Warning: equilibrium loader unavailable; using fallback G0/alpha_s.", typeMsg="w")
            self._gfile_path = None
            self._eq = None
            return None
        if self._gfile_path is None:
            return None
        if self._eq is not None:
            return self._eq
        try:
            self._eq = load_equilibrium(
                self._gfile_path,
                n_points=self._eq_n_points,
                eddy_width_m=self._eq_eddy_width_m,
            )
        except Exception as exc:
            print(f"Warning: equilibrium load failed ({exc}); using fallback G0/alpha_s.")
            self._gfile_path = None  # disable further attempts
            self._eq = None
        return self._eq

    def _solve_peret_SSF(self):
        me_over_mi = self.me_over_mi
        ti = self.ti
        te = self.te
        Te_J = te * e_J
        Ti_J = ti * e_J
        ne = self.ne
        rho_s = self.rho_s
        c_s = self.c_s
        R0 = self.R0
        Lpar = self.Lpar
        a = self.a
        G0 = self.G0
        f_Delta = self.f_Delta
        Lambda = self.Lambda

        # --- G0 and alpha_s from equilibrium or fallback ---
        eq = self._load_equilibrium()
        if eq is not None:
            G0      = eq.G0
            alpha_s = eq.alpha_s
        else:
            # Heuristic: alpha_s scales inversely with magnetic shear;
            # shear_ref = 5.0 gives alpha_s = -0.5 for moderate shear
            alpha_s = -self.shear_ref / max(abs(self.shear), 0.1)

        # Sheath power exhaust factor
        #gamma_0 = (c_s / R0) * np.sqrt(rho_s / R0)
        gamma_0 = 2.5*ti/te - 0.5*np.log(2*np.pi*me_over_mi*(1+ti/te))
        gamma = 2*gamma_0/3 # max(1+1e-6,

        # Get plasma parameters from state
        rho_s = self.rho_s
        c_s = self.c_s
        R0 = self.R0
        Lpar = self.Lpar
        a = self.a

        # instability curvature drive due to magnetic topology
        g = G0*rho_s/R0

        # Solve nonlinear equation (11)
        def residual(lp):
            # lambda_T from lambda_p analytically
            sqrt_gamma = np.sqrt(gamma)
            lT = sqrt_gamma / (sqrt_gamma - 1) * lp

            beta = f_Delta * (1/Lambda + lp/lT)
            alpha_ExB = -0.43 * beta * Lambda * (rho_s/lp)**1.5 / g**0.5

            lhs = lp / rho_s
            rhs = 3.9 * g**(3/11) * (2*rho_s/Lpar)**(-6/11) * \
                gamma**(-4/11) / (1 + (alpha_s + alpha_ExB)**2)**(9/11)

            return lhs - rhs

        # Usually two non-zero roots, start near the smaller one
        res = fsolve(residual,10*rho_s,full_output=True, xtol=1e-6)
        if res[2] != 1:
            raise RuntimeError("fsolve did not converge")
        lambda_p = res[0][0]
        sqrt_gamma = np.sqrt(gamma)
        lambda_n = sqrt_gamma * lambda_p
        lambda_T = sqrt_gamma / (sqrt_gamma - 1.0) * lambda_p

        beta = f_Delta*(1/Lambda + lambda_p/lambda_T)
        alpha_ExB = -0.43*beta*Lambda*(rho_s/lambda_p)**1.5/g**0.5

        # Perpendicular turbulent fluxes, Eqn 7-8
        q0 = 41*g**0.75*(2*rho_s/Lpar)**-0.5*(rho_s/lambda_p)**(7/4)/\
            (1+(alpha_s+alpha_ExB)**2)**(9/4)
        Gamma_e = q0*(lambda_p/lambda_n)*ne*c_s # eqn 7
        q_e = q0* ne * (Te_J + Ti_J)*c_s # eqn 8, W/m^2
        lambda_qe = (2/7)*lambda_T # Spitzer-Harm, m

        return a / lambda_p, a / lambda_n, a / lambda_T, Gamma_e, q_e, lambda_qe

    def _solve_two_fluid_two_point(self, lambda_q: float):
        Ti_target = self.Ti_target if self.Ti_target is not None else self.Te_target
        ne_t = self.ne_target * 1e19
        Te_t = self.Te_target * 1e3
        Ti_t = Ti_target * 1e3

        kappa_0e = 2600.0 / (0.672 + 0.076 * self.Zeff ** 0.5 + 0.252 * self.Zeff)
        kappa_0i = kappa_0e * self.me_over_mi ** 0.5

        Apar_sol = 4.0 * np.pi * self.R0 * lambda_q * (self.Bp / self.BT)

        qpar_e = self.Pe * 1e6 / Apar_sol
        qpar_i = self.Pi * 1e6 / Apar_sol

        Te_u = max(Te_t, (Te_t ** 3.5 + 3.5 * self.fq_e * qpar_e * self.Lpar / kappa_0e) ** (2.0 / 7.0))
        Ti_u = max(Te_u, (Ti_t ** 3.5 + 3.5 * self.fq_i * qpar_i * self.Lpar / kappa_0i) ** (2.0 / 7.0))
        ne_u = (2.0 * ne_t * (Te_t + Ti_t)) / (self.fmom * (Te_u + Ti_u / self.Zeff))

        return ne_u, Te_u, Ti_u, qpar_e, qpar_i

    def _solve_lcfs_iterative(self):
        eq = self._load_equilibrium()
        if eq is not None:
            # Prefer g-file-derived SSF coefficients when available.
            self.G0 = float(eq.G0)
            self.alpha_s = float(eq.alpha_s)

        converged = False
        for iteration in range(self.max_iter):
            ne_old, te_old, ti_old = self.ne, self.te, self.ti

            self.rho_s = PLASMAtools.rho_s(self.te * 1e-3, self.mi_ref, self.BT)
            self.c_s = PLASMAtools.c_s(self.te * 1e-3, self.mi_ref)

            try:
                _, aLne, aLTe, _, _, lambda_q = self._solve_peret_SSF()
            except Exception as exc:
                if self.verbose:
                    print(f"    [PeretSSF] iteration {iteration}: {exc}", typeMsg="w")
                break

            try:
                ne_new, Te_new, Ti_new, _, _ = self._solve_two_fluid_two_point(lambda_q)
            except Exception as exc:
                if self.verbose:
                    print(f"    [TFTP] iteration {iteration}: {exc}", typeMsg="w")
                break

            d = self.damping
            self.ne = (1 - d) * self.ne + d * ne_new
            self.te = (1 - d) * self.te + d * Te_new
            self.ti = (1 - d) * self.ti + d * Ti_new
            self.ni = self.ne / self.Zeff

            rel = max(
                abs((self.ne - ne_old) / max(ne_old, 1e-30)),
                abs((self.te - te_old) / max(te_old, 1e-30)),
                abs((self.ti - ti_old) / max(ti_old, 1e-30)),
            )
            if rel < self.tol:
                converged = True
                break

        if not converged and self.verbose:
            print(
                f"    [BoundaryConditions] PeretSSF+TFTP did not converge "
                f"after {self.max_iter} iterations",
                typeMsg="w",
            )

        _, self.aLne, self.aLte, _, _, _ = self._solve_peret_SSF()
        self.aLti = (self.te / self.ti) * self.aLte
        self.converged = converged

        self._build_bc_dict()

    def _build_bc_dict(self):
        self.bc_dict = {
            "ne": [self.ne * 1e-19, 1.0],
            "te": [self.te * 1e-3, 1.0],
            "ti": [self.ti * 1e-3, 1.0],
            "ni": [self.ni * 1e-19, 1.0],
            "aLne": [self.aLne, 1.0],
            "aLte": [self.aLte, 1.0],
            "aLti": [self.aLti, 1.0],
            "aLni": [self.aLne, 1.0],
        }


# ---------------------------------------------------------------------------
# Tftp_SepOS  (Eich–Manz SepOS + TFTP)
# ---------------------------------------------------------------------------

class Tftp_SepOS(BoundaryConditions):
    """
    Two-fluid two-point model coupled with the Eich–Manz SepOS scale-length
    scaling for the upstream (LCFS) heat-flux width.
    """

    def __init__(self, options: dict):
        super().__init__(options)
        self.max_iter = options.get("max_iter", 100)
        self.tol = options.get("tol", 1e-3)
        self.damping = options.get("damping", 0.5)
        self.ne_target = options.get("ne_target", 4.0)
        self.Te_target = options.get("Te_target", 0.005)
        self.Ti_target = options.get("Ti_target", None)

    def get_boundary_conditions(self, powerstate, batch_idx: int = 0) -> None:
        super().get_boundary_conditions(powerstate, batch_idx)
        self._solve_lcfs_iterative()

    def _get_SepOS_aLy(self):
        me_over_mi = self.me_over_mi
        ti = self.ti
        te = self.te
        ne = self.ne
        rho_s = self.rho_s
        c_s = self.c_s
        R0 = self.R0
        a = self.a
        rho_s_pol = rho_s * (self.BT / self.Bp)
        nuei_cgs = 2.91e-6 * (ne * 1e-6) * self.LogLam / self.te ** 1.5
        alpha_t = (1 + ti / te) * me_over_mi * nuei_cgs * self.q_cyl ** 2 * R0 / c_s

        gamma_0 = 2.5 * ti / te - 0.5 * np.log10(2 * np.pi * me_over_mi * (1 + ti / te))
        _ = max(1 + 1e-6, 2 * gamma_0 / 3)

        lambda_n = 2.9 * (1 + 10.4 * alpha_t ** 2.5) * rho_s_pol
        lambda_T = 2.1 * (1 + 2.1 * alpha_t ** 1.7) * rho_s_pol

        aLne = a * max(0.0, 1.0 / lambda_n)
        aLte = a * max(0.0, 1.0 / lambda_T)
        lambda_q = (2.0 / 7.0) * lambda_T

        return aLne, aLte, lambda_q

    def _solve_two_fluid_two_point(self, lambda_q: float):
        Ti_target = self.Ti_target if self.Ti_target is not None else self.Te_target
        ne_t = self.ne_target * 1e19
        Te_t = self.Te_target * 1e3
        Ti_t = Ti_target * 1e3

        kappa_0e = 2600.0 / (0.672 + 0.076 * self.Zeff ** 0.5 + 0.252 * self.Zeff)
        kappa_0i = kappa_0e * self.me_over_mi ** 0.5
        Apar_sol = 4.0 * np.pi * self.R0 * lambda_q * (self.Bp / self.BT)
        qpar_e = self.Pe * 1e6 / Apar_sol
        qpar_i = self.Pi * 1e6 / Apar_sol

        Te_u = max(Te_t, (Te_t ** 3.5 + 3.5 * self.fmom * qpar_e * self.Lpar / kappa_0e) ** (2.0 / 7.0))
        Ti_u = max(Te_u, (Ti_t ** 3.5 + 3.5 * self.fmom * qpar_i * self.Lpar / kappa_0i) ** (2.0 / 7.0))
        ne_u = (2.0 * ne_t * (Te_t + Ti_t)) / (self.fmom * (Te_u + Ti_u / self.Zeff))

        return ne_u, Te_u, Ti_u, qpar_e, qpar_i

    def _solve_lcfs_iterative(self):
        converged = False
        for iteration in range(self.max_iter):
            ne_old, te_old, ti_old = self.ne, self.te, self.ti

            self.rho_s = PLASMAtools.rho_s(self.te * 1e-3, self.mi_ref, self.BT)
            self.c_s = PLASMAtools.c_s(self.te * 1e-3, self.mi_ref)

            try:
                _, _, lambda_q = self._get_SepOS_aLy()
            except Exception as exc:
                if self.verbose:
                    print(f"    [SepOS] iteration {iteration}: {exc}", typeMsg="w")
                break

            try:
                ne_new, Te_new, Ti_new, _, _ = self._solve_two_fluid_two_point(lambda_q)
            except Exception as exc:
                if self.verbose:
                    print(f"    [TFTP] iteration {iteration}: {exc}", typeMsg="w")
                break

            d = self.damping
            self.ne = (1 - d) * self.ne + d * ne_new
            self.te = (1 - d) * self.te + d * Te_new
            self.ti = (1 - d) * self.ti + d * Ti_new
            self.ni = self.ne / self.Zeff

            rel = max(
                abs((self.ne - ne_old) / max(ne_old, 1e-30)),
                abs((self.te - te_old) / max(te_old, 1e-30)),
                abs((self.ti - ti_old) / max(ti_old, 1e-30)),
            )
            if rel < self.tol:
                converged = True
                break

        if not converged and self.verbose:
            print(
                f"    [BoundaryConditions] SepOS+TFTP did not converge "
                f"after {self.max_iter} iterations",
                typeMsg="w",
            )

        self.aLne, self.aLte, self.lambda_q = self._get_SepOS_aLy()
        self.aLti = (self.te / self.ti) * self.aLte
        self.aLni = self.aLne
        self.converged = converged

        self.bc_dict = {
            "ne": [self.ne * 1e-19, 1.0],
            "te": [self.te * 1e-3, 1.0],
            "ti": [self.ti * 1e-3, 1.0],
            "ni": [self.ni * 1e-19, 1.0],
            "aLne": [self.aLne, 1.0],
            "aLte": [self.aLte, 1.0],
            "aLti": [self.aLti, 1.0],
            "aLni": [self.aLni, 1.0],
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BOUNDARY_CONDITION_MODELS: dict = {
    "Fixed": Fixed,
    "FixedInitial": Fixed,
    "TwoFluid_PeretSSF": TwoFluidTwoPoint_PeretSSF,
    "TwoFluidTwoPoint_PeretSSF": TwoFluidTwoPoint_PeretSSF,
    "TwoFluid_EichManz": Tftp_SepOS,
    "Tftp_SepOS": Tftp_SepOS,
}


def build_bc_model(name: str, options: dict) -> BoundaryConditions:
    """
    Instantiate a boundary-condition model by name.
    """
    if name not in BOUNDARY_CONDITION_MODELS:
        raise KeyError(
            f"Unknown BC model '{name}'.  "
            f"Available: {list(BOUNDARY_CONDITION_MODELS.keys())}"
        )
    return BOUNDARY_CONDITION_MODELS[name](options)
