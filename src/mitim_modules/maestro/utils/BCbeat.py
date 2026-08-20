import copy
import shutil
import numpy as np
import torch
from scipy.optimize import minimize, brentq
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from mitim_modules.powertorch.utils import CALCtools
from IPython import embed

# Mapping from the namelist scaling name to the (H-factor, tau-scaling) keys in profiles.derived
_SCALING_MAP = {
    "H98y2": ("H98", "tau98y2"),   # IPB98(y,2) thermal energy confinement scaling
    "H89p":  ("H89", "tau89p"),    # ITER89-P L-mode scaling
}


def relax_bc(maestro_instance, Te_bc_new, relaxation):
    """
    Under-relax the boundary-condition temperature against the value applied by
    the previous bc beat incarnation:

        Te_bc = Te_bc_prev + relaxation * (Te_bc_new - Te_bc_prev)

    Te_bc_prev is read from parameters_trans_beat['Te_bc_applied'], a single key
    shared by all bc-beat methods (they set the same physical actuator), so
    mixed chains relax coherently. Full step when relaxation=1.0 or on the
    first incarnation (no previous value stored).
    """
    prev = maestro_instance.parameters_trans_beat.get("Te_bc_applied")
    if relaxation < 1.0 and prev is not None:
        Te_bc = prev + relaxation * (Te_bc_new - prev)
        print(
            f"\t- BC relaxation ({relaxation:.2f}): previous {prev:.4f} keV, "
            f"target {Te_bc_new:.4f} keV -> applied {Te_bc:.4f} keV",
            typeMsg="i",
        )
        return Te_bc
    return Te_bc_new


def record_bc_response(maestro_instance, kind, delivered_value, extra=None):
    """
    Record the MEASURED response of the chain to the boundary condition applied by the
    previous bc-beat incarnation: the pair (Te_bc that was applied, value of the
    controlled quantity that the state came back with after the intervening beats).

    'kind' labels the controlled quantity (the H-factor name for method 'confinement',
    'xi' for method 'sharpness') so a mixed chain keeps separate response curves on the
    same shared actuator. The pair is skipped when there is no previous applied Te_bc
    (first incarnation: nothing was actuated yet, so nothing was measured).

    Pairs carry the 'railed' flag of the applied value (bounds/floor pin): a railed
    actuation did not go where the servo asked, so the resulting pair is not a valid
    sample of the response curve and servo_step drops it.

    History lives in parameters_trans_beat['bc_response_history'] as plain floats/bools,
    so MAESTRO's per-beat JSON snapshot persists it across checkpoint restarts for free.
    """

    tb = maestro_instance.parameters_trans_beat
    Te_bc_prev = tb.get("Te_bc_applied")
    if Te_bc_prev is None:
        return None

    record = {
        "kind":   kind,
        "beat":   int(maestro_instance.counter_current),
        "Te_bc":  float(Te_bc_prev),
        "value":  float(delivered_value),
        "railed": bool(tb.get("Te_bc_applied_railed", False)),
    }
    if extra is not None:
        record.update(extra)

    tb.setdefault("bc_response_history", []).append(record)

    print(
        f"\t- BC response recorded: {kind} = {record['value']:.4f} delivered at "
        f"Te_bc = {record['Te_bc']:.4f} keV"
        + (" (railed: excluded from fits)" if record["railed"] else "")
        + f" [{len(tb['bc_response_history'])} pairs in history]",
        typeMsg="i",
    )

    return record


def servo_step(maestro_instance, kind, target, Te_bc_target_frozen, bounds_eff, *,
               fit_window=3, alpha_band=(0.10, 2.0), trust_factor=1.5, seed_gain=2.5):
    """
    Response-fit BC servo: step the boundary condition using the DELIVERED response
    measured over previous cycles instead of trusting the frozen-shape solve.

    The frozen-shape solve has, by construction, alpha = dln(value)/dln(Te_bc) ~ 1
    (T(rho) scales with Te_bc at frozen a/L). The response actually delivered once the
    downstream beats have moved the state is much softer -- measured median alpha = 0.40
    (IQR 0.26-0.60, 5-95% [0.04, 1.33]) -- so the frozen step is ~2.5x too stiff and a
    fixed relaxation factor, which cannot see that, converges at only ~0.7/cycle.

    Over the ~x1.44 Te_bc window the loop explores, curvature of the response is not
    identifiable, so the model is a LOCAL LINEAR fit of the delivered value vs Te_bc
    over the last <= fit_window measured pairs (it beats both a secant tail and a
    quadratic, which invents false roots inside the window).

    Rung ladder, first one that produces an acceptable step wins:
      'fit'    : >=2 usable pairs spanning >2% in Te_bc; least squares value = a + b*Te_bc
      'secant' : slope from the last two pairs (fit degenerate or rejected)
      'seed'   : one pair only / everything above degenerate; step seed_gain x the frozen
                 solve's own step (compensating its ~2.5x over-stiffness)
      'full'   : no previous applied Te_bc (first incarnation) -> the frozen target itself

    A slope is accepted only if b > 0 and alpha = b*Te_last/value_last lies in alpha_band;
    the step is then the linear crossing Te_last + (target - value_last)/b. All rungs but
    'full' are clamped to [Te_prev/trust_factor, Te_prev*trust_factor] (the fit is never
    extrapolated far beyond the explored range) and then to bounds_eff.

    Returns (Te_bc_applied, diag).
    """

    tb = maestro_instance.parameters_trans_beat
    Te_prev = tb.get("Te_bc_applied")
    pairs = [
        h for h in tb.get("bc_response_history", []) if h["kind"] == kind and not h["railed"]
    ][-fit_window:]

    x = np.array([float(h["Te_bc"]) for h in pairs])
    y = np.array([float(h["value"]) for h in pairs])

    diag = {"rung": "full", "n_pairs": len(pairs), "slope": None, "alpha": None,
            "trust_clamped": False, "bounds_clamped": False}

    def _step_from_slope(b):
        """Crossing of the linear response with the target, if the slope is credible."""
        if b <= 0.0:
            return None, None
        alpha = b * x[-1] / y[-1]
        if not alpha_band[0] <= alpha <= alpha_band[1]:
            return None, alpha
        return x[-1] + (target - y[-1]) / b, alpha

    Te_star = None
    if len(pairs) >= 2:
        if x.max() / x.min() > 1.02:
            b = float(np.polyfit(x, y, 1)[0])
            Te_star, alpha = _step_from_slope(b)
            if Te_star is not None:
                diag.update(rung="fit", slope=b, alpha=alpha)
        if Te_star is None and max(x[-2:]) / min(x[-2:]) > 1.02:
            b = float((y[-1] - y[-2]) / (x[-1] - x[-2]))
            Te_star, alpha = _step_from_slope(b)
            if Te_star is not None:
                diag.update(rung="secant", slope=b, alpha=alpha)

    if Te_star is None and Te_prev is not None:
        Te_star = Te_prev + seed_gain * (Te_bc_target_frozen - Te_prev)
        diag["rung"] = "seed"

    if Te_star is None:
        Te_star = Te_bc_target_frozen
        diag["rung"] = "full"

    if diag["rung"] != "full":
        Te_trust = min(max(Te_star, Te_prev / trust_factor), Te_prev * trust_factor)
        diag["trust_clamped"] = Te_trust != Te_star
        Te_star = Te_trust

    Te_bc = min(max(Te_star, bounds_eff[0]), bounds_eff[1])
    diag["bounds_clamped"] = Te_bc != Te_star

    print(
        f"\t- BC servo (response_fit, {kind}): rung={diag['rung']}, n_pairs={diag['n_pairs']}"
        + (f", slope={diag['slope']:.4f}, alpha={diag['alpha']:.3f}" if diag["alpha"] is not None else "")
        + f", frozen target {Te_bc_target_frozen:.4f} keV -> applied {Te_bc:.4f} keV"
        + (" [trust-clamped]" if diag["trust_clamped"] else "")
        + (" [bounds-clamped]" if diag["bounds_clamped"] else ""),
        typeMsg="i",
    )

    return float(Te_bc), diag


def _recompute_alpha_power(profiles):
    """
    Recompute qfuse/qfusi in *profiles* (in-place, also returns it) from the current
    kinetic profiles, using the analytic powerstate targets on the profile's own fine
    grid — the same transport-free recipe as TRANSFORMtools.powerstate_to_gacode_powers.
    The recomputed sources enter qHeat, so tauE and the H-factor see the alpha-power
    response to the boundary-condition change. No-op physics for non-DT plasmas
    (the analytic model returns zero fusion without thermal D+T).
    Radiation and exchange sources are deliberately left untouched: they do not enter
    the H-factor (qHeat does not subtract radiation) and recomputing them here would
    swap the upstream beat's radiation model in the passed-forward state.
    """
    from mitim_tools.misc_tools import LOGtools

    # Only qfus: refreshing radiation/exchange here would swap the upstream beat's
    # radiation model and they do not enter qHeat (see the note above).
    with LOGtools.HiddenPrints():
        profiles.recompute_targets(targets=["qfus"])

    return profiles


# --------------------------------------------------------------------------------------------
# Per-method parameter sets (prepare() validation)
# --------------------------------------------------------------------------------------------
# The bc beat determines Te_bc by one of two routes:
#   - a CLOSED-FORM solve (method 'sharpness': Te_bc = Tsep/(1 - xi*C); method 'betap':
#     invert the prescribed edge poloidal-beta gradient for the BC thermal pressure)
#   - the GENERIC ITERATIVE metric-matching solver (method 'confinement': scan Te_bc,
#     applying the BC at each trial and re-deriving the metric, until metric == target)
# A future method whose target needs the full modified state slots into the iterative
# route with its own metric evaluation — see _run_confinement for the pattern.
#
# Namelist shape: common knobs (_COMMON_DEFAULTS) sit at the TOP level of
# parameters_prepare; each method's knobs (_METHOD_DEFAULTS[m]) live in a
# '<m>_parameters' sub-dict (confinement_parameters, sharpness_parameters,
# betap_parameters). Only the selected method's sub-dict is consumed; the others may
# coexist (typo-checked but ignored), so base_module inheritance and method switching
# need no pruning.

_COMMON_DEFAULTS = dict(
    x_bc=0.90,
    bc_coordinate="rho",
    tite=1.0,
    density_treatment="bc",
    relaxation=1.0,
    servo_mode="relaxation",
    servo_fit_window=3,
    servo_alpha_band=(0.10, 2.0),
    servo_trust_factor=1.5,
    servo_seed_gain=2.5,
    update_bc_based_on_portals=False,
)

_METHOD_DEFAULTS = {
    "sharpness": dict(
        sharpness=1.0,
        sharpness_coordinate="psin",
    ),
    "confinement": dict(
        confinement_scaling="H98y2",
        confinement=1.0,
        edge_shape="linear",
        alpha_power_feedback=False,
        Te_bc_bounds=(0.05, 10.0),
        Te_bc_min_Tesep_factor=1.2,
        sep_max_frac=None,
    ),
    "betap": dict(
        betap_prime=2.0,
    ),
}

BC_METHODS = tuple(_METHOD_DEFAULTS.keys())


class bc_beat(beat):
    """
    Boundary-condition (bc) beat: sets the temperature boundary condition at rho_bc
    by one of several methods, sharing the same BC-application machinery (core scaled
    preserving a/L, analytical edge down to the separatrix, shared Te_bc_applied
    trans-beat memory and relaxation/response-fit servo):

    method 'sharpness' — from the sharpness parameter xi defined in
    Rodriguez-Fernandez et al. (L-mode paper):

        xi = |dT/dpsi_n|_edge  /  |dT/dpsi_n|_core_at_bc

    where the edge gradient goes linearly in psi_n from T_bc to T_sep and the core
    gradient at rho_bc is taken from the current profiles (PORTALS output). Given xi,
    T_sep (from profiles), and the core gradient, T_bc follows in closed form:

        C    = (1 - c_bc) * aLT_bc * d(r/a)/dc|_bc
        T_bc = T_sep / (1 - xi * C)

    method 'betap' — from a prescribed magnitude of the edge poloidal-beta gradient
    (two-point finite difference in psi_n between the BC and the separatrix):

        beta_p(psin) = 2*mu0*p_th(psin) / Bpa^2,   Bpa = mu0*Ip/L_pol  (engineering norm)
        betap' = [beta_p(x_bc) - beta_p(sep)] / (1 - psin_bc)     (prescribed positive)

    with p_th the THERMAL pressure only. Inverted in closed form:

        p_bc  = (Bpa^2/(2*mu0)) * betap' * (1 - psin_bc) + p_sep
        Te_bc = p_bc / [ne_bc * (1 + f_i*tite)],   f_i = sum_i(thermal) ni/ne at x_bc

    Density is a spectator: ne_bc in the inversion is the ne that will actually stand
    at x_bc after application (neped_20 under density_treatment 'bc', the incoming
    state's value under 'keep'), so the delivered betap' matches the target under both.
    The edge is LINEAR IN THERMAL PRESSURE (not the shared linear-in-Te edge): Te is
    derived pointwise from the pressure line and the standing density, so
    d(beta_p)/dpsin is constant along the edge (Te comes out convex-up).

    method 'confinement' — such that the plasma state matches a prescribed
    confinement level (H-factor). The H-factor cannot be inverted analytically for
    T_bc (it depends on the volume-integrated thermal stored energy of the full
    modified profiles), so T_bc is found by minimization:

        find Te_bc such that  ((H - H_target)/H_target)^2  is minimized

    where at each trial Te_bc the boundary condition is applied with the same profile
    machinery and the H-factor is re-derived.

    NOTE (assumption, method 'confinement'): by default, the auxiliary, fusion and
    radiation source profiles stored in input.gacode are NOT recomputed during the
    Te_bc scan, so the total heating power entering both tauE and the scaling law
    stays frozen; the H-factor responds through the thermal stored energy Wthr.
    Source self-consistency is recovered by the subsequent PORTALS/TRANSP beat in
    the MAESTRO chain. With alpha_power_feedback=True, qfuse/qfusi are recomputed
    analytically (powerstate targets) at every trial Te_bc and in the final output
    state, so the H-factor accounts for the alpha-heating response — relevant for
    burning plasmas, where freezing the sources biases H high and underestimates
    the Te_bc needed for a given target.

    neped_20 (a MAESTRO trans-beat parameter) is reinterpreted here as ne_bc,
    i.e. the electron density at the boundary condition location rho_bc.
    """

    def __init__(self, maestro_instance, method=None, folder_name=None, legacy=False):
        if method not in BC_METHODS:
            raise ValueError(
                f"[MITIM] bc beat requires parameters_prepare 'method' in {list(BC_METHODS)}, got {method!r}"
            )
        self.method = method

        # Read-side backward compatibility: a pre-refactor run folder (run_sharpness/,
        # run_confinement/, <method>_results.npy, input.gacode.<method>) carries the same
        # artifacts under their old names. legacy=True maps this beat onto them so
        # plotting/checking old runs keeps working; LAUNCHING with the old beat_types
        # still raises in MAESTROmain.define_beat.
        self.legacy = legacy
        if legacy:
            print(
                f"\t- reading legacy '{method}' beat folder (pre-'bc' refactor); "
                f"new runs use run_bc_{method}",
                typeMsg="w",
            )
        self._results_file = f"{method}_results.npy" if legacy else "bc_results.npy"
        self._state_file   = f"input.gacode.{method}" if legacy else "input.gacode.bc"

        super().__init__(maestro_instance,
                         beat_name=method if legacy else f"bc_{method}",
                         folder_name=folder_name)

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(self, method=None, **params):
        """
        Common knobs sit at the top level of parameters_prepare; method-specific
        knobs live in a per-method sub-dict named '<method>_parameters'
        (confinement_parameters, sharpness_parameters; a future betap method adds
        betap_parameters symmetrically). Both sub-dicts may coexist in a block —
        only the selected method's sub-dict is consumed, the other is ignored
        (its keys are still checked for typos), so base_module inheritance and
        method switching work without pruning. A method-specific knob placed at
        the TOP level raises with a pointer to the right sub-dict.

        Parameters (common, top level of parameters_prepare)
        ----------------------------------------------------
        method : str
            Which bc method this beat runs ('sharpness' or 'confinement'). Must
            match the method the beat was defined with (it reaches both places
            from the same parameters_prepare block).
        x_bc : float
            Location of the boundary condition in the coordinate given by
            bc_coordinate (default 0.90).
        bc_coordinate : str
            Coordinate system for x_bc: 'rho' (rho_tor, default), 'roa' (r/a),
            or 'psin' (normalized poloidal flux).
        tite : float
            Ti / Te ratio at the boundary condition (default 1.0).
        density_treatment : str
            'bc' (default): core ne rescaled to ne_bc (= neped_20) preserving
            a/Lne, edge replaced, ion densities rescaled to keep ni/ne ratios.
            'keep': ne and all ion densities left untouched; only Te/Ti are
            modified. The neped_20 passed to subsequent beats is then the
            actual ne at rho_bc read from the profiles.
        relaxation : float
            Under-relaxation factor for Te_bc across beat incarnations (see
            relax_bc): applied Te_bc = previous + relaxation * (new - previous),
            with the previous value read from the shared trans-beat parameter
            'Te_bc_applied' (written by every bc-beat method). Default 1.0 =
            full step. With relaxation < 1 the target is only approached across
            beat iterations, not within one.
        servo_mode : str
            How the applied Te_bc is derived from the frozen-shape target.
            'relaxation' (default): the under-relaxation above. 'response_fit':
            step from a local linear fit of the DELIVERED quantity measured at
            the previously applied Te_bc values (see servo_step), falling back
            to secant and then to a seeded step when the fit is degenerate.
            The response-fit statistics were calibrated on the confinement
            H-factor response (median alpha = 0.40, IQR 0.26-0.60); for the
            sharpness method it is EXPERIMENTAL (xi response uncharacterized).
        servo_fit_window : int
            response_fit: how many of the most recent usable (non-railed) pairs
            enter the fit (default 3).
        servo_alpha_band : (float, float)
            response_fit: acceptance band on the fitted sensitivity
            alpha = dln(value)/dln(Te_bc); outside it the rung falls back to
            secant and then to the seeded step. Default (0.10, 2.0).
        servo_trust_factor : float
            response_fit: maximum multiplicative change of Te_bc per cycle
            (default 1.5).
        servo_seed_gain : float
            response_fit: gain applied to the frozen solve's own step when only
            one measured pair exists (default 2.5 = the measured over-stiffness
            1.0/0.40 of the frozen-shape solve). Capped by servo_trust_factor.
        update_bc_based_on_portals : bool
            If True, override x_bc with the outermost radial location used by
            the previous PORTALS beat (stored in parameters_trans_beat as
            predicted_rho[-1] or predicted_roa[-1]). bc_coordinate is updated
            automatically (sharpness_coordinate is never changed). Default False.

        Parameters (sharpness_parameters sub-dict; consumed when method='sharpness')
        ----------------------------------------------------------------------------
        sharpness : float
            Prescribed sharpness parameter xi (default 1.0).
        sharpness_coordinate : str
            Coordinate system in which the sharpness parameter xi (the gradient
            ratio) is defined: 'rho' (rho_tor), 'roa' (r/a), or 'psin' (default).
            This is independent of bc_coordinate.

        Parameters (betap_parameters sub-dict; consumed when method='betap')
        --------------------------------------------------------------------
        betap_prime : float
            Prescribed magnitude of the edge poloidal-beta gradient (positive;
            default 2.0):  betap' = [beta_p(x_bc) - beta_p(sep)]/(1 - psin_bc),
            with beta_p = 2*mu0*p_th/Bpa^2 (thermal pressure only) and the
            engineering norm Bpa = mu0*Ip/L_pol. Inverted in closed form for
            Te_bc; density is a spectator (never set by this method). The edge
            is linear in THERMAL PRESSURE (Te derived pointwise, convex-up), so
            d(beta_p)/dpsin is constant across the edge, equal to -betap'.

        Parameters (confinement_parameters sub-dict; consumed when method='confinement')
        --------------------------------------------------------------------------------
        confinement_scaling : str
            Confinement scaling law whose H-factor is matched: 'H98y2'
            (IPB98(y,2), default) or 'H89p' (ITER89-P).
        confinement : float
            Target H-factor value (default 1.0).
        edge_shape : str
            Shape of the Te/Ti/ne profiles in the edge region (rho > rho_bc):
            'linear' (default) interpolates linearly in psi_n from the BC value
            to the separatrix value; 'tanh' uses the pedestal tanh functional
            form of the eped_initializer (FunctionalForms.pedestal_tanh in r/a),
            also anchored at the BC and separatrix values. (The sharpness
            method always uses 'linear': its xi definition assumes the linear
            edge gradient.)
        alpha_power_feedback : bool
            If True, recompute the fusion source profiles (qfuse/qfusi) from the
            trial profiles at every minimization step (analytic powerstate
            targets), so the H-factor sees the alpha-power response to the BC
            change. The recomputed sources are also written into the beat output
            state. The baseline H is recomputed with the same model for a
            consistent comparison. Default False (sources frozen during scan).
        Te_bc_bounds : (float, float)
            Bounds on Te_bc in keV during the minimization (default (0.05, 10.0)).
        Te_bc_min_Tesep_factor : float
            Dynamic lower-bound guard: the effective floor is
            max(Te_bc_bounds[0], factor * Te_sep), with Te_sep read from the
            incoming state (last profile point). A Te_bc at or below Te_sep
            makes the rho_bc->separatrix edge isothermal/inverted and TRANSP
            dies on it (SIGFPE), so the floor adapts to whatever separatrix
            temperature an earlier beat (e.g. lengyel) set. Values <= 1
            effectively disable the margin. Default 1.2. If the optimum pins
            at this floor the H target is unreachable above the guard —
            flagged as 'Te_bc_at_floor' in the beat results. A floor pin with
            H BELOW the target is a Nelder-Mead bound-clipping artifact (the
            crossing is bracketed above) and is re-solved by brentq.
            None disables the dynamic floor entirely (only Te_bc_bounds[0]
            remains) — pair with sep_max_frac so the edge stays monotone.
        sep_max_frac : float or None
            Inverts the isothermal-edge guard: instead of flooring Te_bc at
            1.2*Tesep, let Te_bc go as low as the servo wants and cap the
            APPLIED separatrix temperatures in the written state at
            sep_max_frac * bc value (Te and Ti; forwarded to _apply_bc). The
            physical Tesep (e.g. from the lengyel beat) is untouched upstream
            and stays available to analysis; a case whose Te_bc lands at/below
            it is then a physics result (sharpness <= 0), not a rail. Typical
            value 0.8. Default None = old behavior.
        """

        if method is not None and method != self.method:
            raise ValueError(
                f"[MITIM] bc beat was defined with method '{self.method}' but prepare() received "
                f"method '{method}' — parameters_prepare is inconsistent"
            )

        # ---- pull out the per-method sub-dicts ('<method>_parameters') ----
        # Every sub-dict present is key-checked against its own method's knobs (typo
        # guard), but only the selected method's sub-dict is consumed — the other is
        # ignored, so a block can carry both and be switched/inherited freely.
        method_params = {}
        for m, defaults in _METHOD_DEFAULTS.items():
            sub = params.pop(f"{m}_parameters", None)
            if sub is None:
                continue
            if not isinstance(sub, dict):
                raise ValueError(
                    f"[MITIM] bc beat: '{m}_parameters' must be a dict of {m}-specific knobs, "
                    f"got {type(sub).__name__}"
                )
            unknown_sub = set(sub) - set(defaults)
            if unknown_sub:
                raise ValueError(
                    f"[MITIM] bc beat: unknown key(s) in '{m}_parameters': "
                    f"{sorted(unknown_sub)} (valid: {sorted(defaults)})"
                )
            if m == self.method:
                method_params = dict(sub)

        # ---- top level: only common knobs allowed ----
        # (beat-block namelist keys are not validated anywhere else, so raise here)
        unknown = set(params) - set(_COMMON_DEFAULTS)
        if unknown:
            msgs = []
            for key in sorted(unknown):
                owners = [m for m, d in _METHOD_DEFAULTS.items() if key in d]
                if owners:
                    msgs.append(f"'{key}' (method-specific: place it inside '{owners[0]}_parameters')")
                else:
                    msgs.append(f"'{key}' (not a bc-beat parameter)")
            raise ValueError(
                f"[MITIM] bc beat (method '{self.method}') got invalid top-level "
                f"parameters_prepare key(s): {', '.join(msgs)}"
            )

        resolved = {**_COMMON_DEFAULTS, **params, **_METHOD_DEFAULTS[self.method], **method_params}

        # ---- common validation ----
        if resolved["bc_coordinate"] not in ("rho", "roa", "psin"):
            raise ValueError(
                f"bc_coordinate must be 'rho', 'roa', or 'psin', got '{resolved['bc_coordinate']}'"
            )
        if resolved["density_treatment"] not in ("bc", "keep"):
            raise ValueError(
                f"density_treatment must be 'bc' or 'keep', got '{resolved['density_treatment']}'"
            )
        if not 0.0 < resolved["relaxation"] <= 1.0:
            raise ValueError(f"relaxation must be in (0, 1], got {resolved['relaxation']}")
        if resolved["servo_mode"] not in ("relaxation", "response_fit"):
            raise ValueError(
                f"servo_mode must be 'relaxation' or 'response_fit', got '{resolved['servo_mode']}'"
            )

        # ---- method-specific validation ----
        if self.method == "sharpness":
            if resolved["sharpness_coordinate"] not in ("rho", "roa", "psin"):
                raise ValueError(
                    f"sharpness_coordinate must be 'rho', 'roa', or 'psin', got '{resolved['sharpness_coordinate']}'"
                )
        elif self.method == "confinement":
            if resolved["confinement_scaling"] not in _SCALING_MAP:
                raise ValueError(
                    f"confinement_scaling must be one of {list(_SCALING_MAP.keys())}, got '{resolved['confinement_scaling']}'"
                )
            if resolved["edge_shape"] not in ("linear", "tanh"):
                raise ValueError(
                    f"edge_shape must be 'linear' or 'tanh', got '{resolved['edge_shape']}'"
                )
        elif self.method == "betap":
            if not resolved["betap_prime"] > 0.0:
                raise ValueError(
                    f"betap_prime must be positive (magnitude of the falling edge beta_p gradient), "
                    f"got {resolved['betap_prime']}"
                )

        # ---- store ----
        self.x_bc = resolved["x_bc"]
        self.bc_coordinate = resolved["bc_coordinate"]
        self.tite = resolved["tite"]
        self.density_treatment = resolved["density_treatment"]
        self.relaxation = resolved["relaxation"]
        self.servo_mode = resolved["servo_mode"]
        self.servo_fit_window = resolved["servo_fit_window"]
        self.servo_alpha_band = tuple(resolved["servo_alpha_band"])
        self.servo_trust_factor = resolved["servo_trust_factor"]
        self.servo_seed_gain = resolved["servo_seed_gain"]
        self.update_bc_based_on_portals = resolved["update_bc_based_on_portals"]

        if self.method == "sharpness":
            self.sharpness = resolved["sharpness"]
            self.sharpness_coordinate = resolved["sharpness_coordinate"]
            print(
                f"\t- BC beat (sharpness): x_bc={self.x_bc} ({self.bc_coordinate}), "
                f"sharpness_coord={self.sharpness_coordinate}, xi={self.sharpness}, Ti/Te={self.tite}, "
                f"density_treatment={self.density_treatment}, relaxation={self.relaxation}, "
                f"servo_mode={self.servo_mode}",
                typeMsg="i",
            )
        elif self.method == "confinement":
            self.confinement_scaling = resolved["confinement_scaling"]
            self.confinement = resolved["confinement"]
            self.edge_shape = resolved["edge_shape"]
            self.alpha_power_feedback = resolved["alpha_power_feedback"]
            self.Te_bc_bounds = tuple(resolved["Te_bc_bounds"])
            self.Te_bc_min_Tesep_factor = resolved["Te_bc_min_Tesep_factor"]
            self.sep_max_frac = resolved["sep_max_frac"]
            print(
                f"\t- BC beat (confinement): x_bc={self.x_bc} ({self.bc_coordinate}), "
                f"target {self.confinement_scaling}={self.confinement}, Ti/Te={self.tite}, "
                f"edge_shape={self.edge_shape}, density_treatment={self.density_treatment}, "
                f"alpha_power_feedback={self.alpha_power_feedback}, relaxation={self.relaxation}, "
                f"servo_mode={self.servo_mode}",
                typeMsg="i",
            )
        elif self.method == "betap":
            self.betap_prime = resolved["betap_prime"]
            print(
                f"\t- BC beat (betap): x_bc={self.x_bc} ({self.bc_coordinate}), "
                f"betap_prime={self.betap_prime}, Ti/Te={self.tite}, "
                f"density_treatment={self.density_treatment}, relaxation={self.relaxation}, "
                f"servo_mode={self.servo_mode}",
                typeMsg="i",
            )

        self._portals_rho_bc = None   # (value, coordinate) set by _inform() if update_bc_based_on_portals
        self.neped_20 = None   # resolved in _inform() from plasma/parameters or previous beat

        self._inform()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, **kwargs):

        # Copy current input.gacode to working folder
        shutil.copy2(self.initialize.folder / "input.gacode", self.folder / "input.gacode")

        # ------------------------------------------------------------------
        # Compute T_bc (per method) and apply to profiles
        # ------------------------------------------------------------------

        bc_results = self._run()

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------

        np.save(self.folder / self._results_file, bc_results)

        self.rho_bc_rho = bc_results["rho_bc_rho"]   # store for _inform_save

    # ------------------------------------------------------------------
    # _run (core physics)
    # ------------------------------------------------------------------

    def _run(self):

        profiles = copy.deepcopy(self.profiles_current)
        profiles.derive_quantities(rederiveGeometry=False)

        # With alpha feedback on, make the baseline sources consistent with the same
        # analytic model used during the scan, so H_initial and the trial H values
        # are directly comparable (stored qfus may come from a different model)
        if self.method == "confinement" and self.alpha_power_feedback:
            print("\t- Alpha power feedback ON: recomputing baseline qfuse/qfusi from profiles")
            Pfus_stored = float(profiles.derived["Pfus"])
            profiles = _recompute_alpha_power(profiles)
            print(
                f"\t\t- Pfus: {Pfus_stored:.2f} MW (stored sources) -> "
                f"{float(profiles.derived['Pfus']):.2f} MW (analytic recomputation)"
            )

        rho        = profiles.profiles["rho(-)"]
        psi_pol_n  = profiles.derived["psi_pol_n"]
        roa        = profiles.derived["roa"]
        Te         = profiles.profiles["te(keV)"]
        ne         = profiles.profiles["ne(10^19/m^3)"]

        # ------------------------------------------------------------------
        # 1. Convert rho_bc to rho_tor
        # ------------------------------------------------------------------

        if self._portals_rho_bc is not None:
            # Location comes from the last PORTALS beat
            _val, _coord = self._portals_rho_bc
            rho_bc_rho = _convert_bc_location(_val, _coord, rho, roa, psi_pol_n)
            print(f"\t- BC location overridden from PORTALS: {_val:.4f} ({_coord}) -> rho_tor={rho_bc_rho:.4f}")
        else:
            rho_bc_rho = _convert_bc_location(
                self.x_bc, self.bc_coordinate, rho, roa, psi_pol_n
            )
        psin_bc = float(np.interp(rho_bc_rho, rho, psi_pol_n))

        print(
            f"\t- BC location: x_bc={self.x_bc} ({self.bc_coordinate}) "
            f"-> rho_tor={rho_bc_rho:.4f}, psi_n={psin_bc:.4f}"
        )

        # ------------------------------------------------------------------
        # 2. Determine ne_bc
        # ------------------------------------------------------------------

        if self.density_treatment == "keep":
            # Density untouched: ne_bc reported (and passed forward as neped_20) is the
            # actual ne at rho_bc from the profiles, not any inherited target
            ne_bc_1e19 = float(np.interp(rho_bc_rho, rho, ne))
            ne_bc_20   = ne_bc_1e19 * 0.1   # convert 10^19 -> 10^20
            if self.neped_20 is not None:
                print(
                    f"\t- density_treatment='keep': ne/ni profiles left untouched; ignoring "
                    f"neped_20={self.neped_20:.3f}, reporting actual ne at rho_bc: {ne_bc_20:.3f} 10^20 m^-3",
                    typeMsg="i",
                )
        elif self.neped_20 is None:
            # Fall back: use ne from profiles at rho_bc
            ne_bc_1e19 = float(np.interp(rho_bc_rho, rho, ne))
            ne_bc_20   = ne_bc_1e19 * 0.1   # convert 10^19 -> 10^20
            print(
                f"\t- neped_20 not provided, using ne at rho_bc from profiles: "
                f"{ne_bc_20:.3f} 10^20 m^-3"
            )
        else:
            ne_bc_20   = self.neped_20
            ne_bc_1e19 = ne_bc_20 * 10.0   # 10^20 -> 10^19

        print(f"\t- ne_bc = {ne_bc_20:.3f} (10^20 m^-3)")

        # ------------------------------------------------------------------
        # 3. Method-specific Te_bc determination
        # ------------------------------------------------------------------

        if self.method == "sharpness":
            return self._run_sharpness(profiles, rho, psi_pol_n, roa, Te, ne,
                                       rho_bc_rho, psin_bc, ne_bc_1e19, ne_bc_20)
        elif self.method == "confinement":
            return self._run_confinement(profiles, rho, Te,
                                         rho_bc_rho, psin_bc, ne_bc_1e19, ne_bc_20)
        elif self.method == "betap":
            return self._run_betap(profiles, rho, psi_pol_n, Te, ne,
                                   rho_bc_rho, psin_bc, ne_bc_1e19, ne_bc_20)

    # ------------------------------------------------------------------
    # method 'sharpness': closed-form solve
    # ------------------------------------------------------------------

    def _run_sharpness(self, profiles, rho, psi_pol_n, roa, Te, ne,
                       rho_bc_rho, psin_bc, ne_bc_1e19, ne_bc_20):

        # ------------------------------------------------------------------
        # Compute T_bc from sharpness formula (in the coordinate c selected
        # by sharpness_coordinate: rho, roa or psin — all equal 1 at the
        # separatrix)
        #
        #    xi = (T_bc - T_sep) / [(1 - c_bc) * aLT_bc * T_bc * droa_dcoord_bc]
        #       => T_bc = T_sep / [1 - xi * C]
        #    where C = (1 - c_bc) * aLT_bc * droa_dcoord_bc
        # ------------------------------------------------------------------

        Te_sep = float(Te[-1])
        ne_sep_1e19 = float(ne[-1])

        # aLT = -d(ln Te)/d(r/a)   (positive for Te decreasing outward)
        aLT_Te = CALCtools.derivation_into_Lx(
            torch.from_numpy(roa), torch.from_numpy(Te), array=False
        ).numpy()
        aLT_Te_bc = float(np.interp(rho_bc_rho, rho, aLT_Te))

        # Sharpness factor  C = (1 - c_bc) * aLT * d(roa)/dc, where c is the
        # coordinate selected by sharpness_coordinate (rho, roa and psin are all
        # 1 at the separatrix, so 1-c_bc is the distance to it in that coordinate)
        if self.sharpness_coordinate == "psin":
            coord, c_bc = psi_pol_n, psin_bc
        elif self.sharpness_coordinate == "rho":
            coord, c_bc = rho, rho_bc_rho
        else:  # roa
            coord, c_bc = roa, float(np.interp(rho_bc_rho, rho, roa))
        droa_dcoord    = np.gradient(roa, coord)
        droa_dcoord_bc = float(np.interp(rho_bc_rho, rho, droa_dcoord))

        C = (1.0 - c_bc) * aLT_Te_bc * droa_dcoord_bc

        # Delivered sharpness of the INCOMING state: the same xi definition evaluated at
        # the Te_bc the state actually came back with. This IS the response to the BC the
        # previous incarnation applied (the actuator is held at the BC, so the response
        # arrives through the core gradient, i.e. through C). Recorded before the C clamp
        # so the measurement reflects the state, not the guarded solve.
        Te_bc_current = float(np.interp(rho_bc_rho, rho, Te))
        xi_delivered = (1.0 - Te_sep / Te_bc_current) / C
        record_bc_response(self.maestro_instance, "xi", xi_delivered)

        if C >= 1.0 / self.sharpness:
            print(
                f"\t- WARNING: sharpness formula denominator non-positive "
                f"(xi*C={self.sharpness*C:.3f} >= 1). Clamping C.",
                typeMsg="w",
            )
            # Clamp to avoid division by zero / negative T_bc
            C = 0.99 / max(self.sharpness, 1e-6)

        Te_bc_target = Te_sep / (1.0 - self.sharpness * C)

        if self.servo_mode == "response_fit":
            # No Te_bc bound concept in this method, so the servo only sees its trust clamp
            Te_bc, servo_diag = servo_step(
                self.maestro_instance, "xi", self.sharpness, Te_bc_target, (0.0, np.inf),
                fit_window=self.servo_fit_window,
                alpha_band=self.servo_alpha_band,
                trust_factor=self.servo_trust_factor,
                seed_gain=self.servo_seed_gain,
            )
        else:
            servo_diag = None
            Te_bc = relax_bc(self.maestro_instance, Te_bc_target, self.relaxation)
        Ti_bc = Te_bc * self.tite

        # Effective sharpness actually applied (== prescribed xi when no relaxation acted)
        xi_eff = (1.0 - Te_sep / Te_bc) / C

        print(f"\t- Sharpness C={C:.4f}, xi={self.sharpness:.3f}" +
              (f" (xi_eff applied after relaxation: {xi_eff:.3f})" if abs(xi_eff - self.sharpness) > 1e-6 else ""))
        print(f"\t- T_sep={Te_sep:.4f} keV")
        print(f"\t- Te_bc={Te_bc:.4f} keV,  Ti_bc={Ti_bc:.4f} keV")

        # ------------------------------------------------------------------
        # Modify profiles using the sharpness boundary condition
        # ------------------------------------------------------------------

        profiles_out = _apply_bc(
            profiles,
            rho_bc_rho,
            psin_bc,
            Te_bc,
            Ti_bc,
            ne_bc_1e19,
            density_treatment=self.density_treatment,
        )

        # ------------------------------------------------------------------
        # Store
        # ------------------------------------------------------------------

        bc_results = {
            "method":          "sharpness",
            "x_bc":            self.x_bc,
            "bc_coordinate":   self.bc_coordinate,
            "rho_bc_rho":      rho_bc_rho,
            "psin_bc":         psin_bc,
            "sharpness":       self.sharpness,
            "sharpness_coord": self.sharpness_coordinate,
            "xi_eff":          xi_eff,
            "xi_delivered":    xi_delivered,
            "relaxation":      self.relaxation,
            "servo_mode":      self.servo_mode,
            # This method has no Te_bc bounds, so the applied value can only be trust-clamped
            "Te_bc_applied_railed": bool(servo_diag["bounds_clamped"]) if servo_diag is not None else False,
            "C":              C,
            "Te_bc":          Te_bc,
            "Te_bc_target":   Te_bc_target,
            "Ti_bc":          Ti_bc,
            "Te_sep":         Te_sep,
            "ne_bc_20":       ne_bc_20,
            "neped_20":       ne_bc_20,   # keep standard key name for compatibility
            "ne_sep_1e19":    ne_sep_1e19,
            "aLT_Te_bc":      aLT_Te_bc,
            "droa_dcoord_bc": droa_dcoord_bc,
            "tite":           self.tite,
            "density_treatment": self.density_treatment,
        }

        if servo_diag is not None:
            bc_results.update({
                "servo_rung":            servo_diag["rung"],
                "servo_n_pairs":         servo_diag["n_pairs"],
                "servo_alpha":           servo_diag["alpha"],
                "servo_slope":           servo_diag["slope"],
                "servo_trust_clamped":   servo_diag["trust_clamped"],
                "servo_bounds_clamped":  servo_diag["bounds_clamped"],
            })

        for key, val in bc_results.items():
            print(f"\t\t- {key}: {val}")

        # Write intermediate result
        profiles_out.write_state(file=self.folder / self._state_file)

        self.profiles_output = profiles_out

        return bc_results

    # ------------------------------------------------------------------
    # method 'betap': closed-form solve from the edge poloidal-beta gradient
    # ------------------------------------------------------------------

    def _run_betap(self, profiles, rho, psi_pol_n, Te, ne,
                   rho_bc_rho, psin_bc, ne_bc_1e19, ne_bc_20):

        # ------------------------------------------------------------------
        # Compute Te_bc from the prescribed edge poloidal-beta gradient
        #
        #    beta_p(psin) = 2*mu0*p_th(psin) / Bpa^2,   Bpa = mu0*Ip/L_pol
        #    betap' = [beta_p(x_bc) - beta_p(sep)] / (1 - psin_bc)   (prescribed POSITIVE:
        #             magnitude of the falling edge gradient, two-point finite difference)
        #      => p_bc  = (Bpa^2/(2*mu0)) * betap' * (1 - psin_bc) + p_sep
        #         Te_bc = p_bc / [ne_bc * (1 + f_i*tite)]
        #
        # p_th is THERMAL pressure only; density is a SPECTATOR (never set here): ne_bc in
        # the inversion is the ne that will actually stand at the BC point after application.
        #
        # Literature mapping (grounds the default betap' ~ 2): our quantity relates to the
        # ballooning parameter as
        #    alpha_MHD = betap' * R * q^2 * (Bpa/B0)^2 * (dpsin/dr)
        # giving alpha_MHD ~ 1 (DIII-D-class) to ~2-3.6 (ARC-class, q95-driven) at betap'=2 —
        # above the spontaneous-barrier threshold alpha ~ 0.25-0.5 of Rogers, Drake & Zeiler,
        # PRL 81 (1998) 4396, and an order-unity fraction of the separatrix ideal-ballooning
        # limit alpha_c = kappa^1.2*(1+1.5*delta) ~ 2.4 of Eich & Manz, NF 61 (2021) 086017
        # (review: Manz, Eich & Grover, Rev. Mod. Plasma Phys. 9 (2025) 5). Caveats: this is
        # an x_bc->sep AVERAGE (locally larger at the separatrix, where SepOS uses lambda_p),
        # and fixed betap' is NOT fixed alpha_MHD across machines (q95^2*(Bpa/B0)^2 factor).
        # Do not confuse alpha_MHD with SepOS alpha_t (no pressure gradient in the latter).
        # ------------------------------------------------------------------

        e_J = 1.602176634e-19   # elementary charge [C]: n(1e19 m^-3)*T(keV) -> Pa via 1e19*1e3*e
        mu0 = 4.0e-7 * np.pi    # [T m/A] (pre-2019-SI value; fractional difference ~1e-10)

        Bpa, L_pol_m, Ip_A = _betap_normalization(profiles)
        coef = Bpa**2 / (2.0 * mu0)   # [Pa]: p_bc - p_sep = coef * betap' * (1 - psin_bc)

        # Grid-point convention: _apply_bc lands Te_bc EXACTLY on the grid point nearest
        # rho_bc_rho, so every inversion input is taken at that point (psin included) —
        # the written state then delivers the target betap' exactly, under both density
        # treatments. psin_bc (interpolated, shared preamble) is stored for reference only.
        ibc = int(np.argmin(np.abs(rho - rho_bc_rho)))
        psin_g = float(psi_pol_n[ibc])

        p_th = _thermal_pressure_Pa(profiles)   # [Pa]
        p_sep = float(p_th[-1])                 # invariant under _apply_bc (separatrix held fixed)
        Te_sep = float(Te[-1])
        ne_sep_1e19 = float(ne[-1])

        # Thermal-ion fraction f_i = sum_i(thermal) ni/ne at the BC grid point: invariant
        # under 'bc' (ni/ne ratios preserved by _apply_bc) and untouched under 'keep'
        ni_th = np.zeros_like(ne)
        for sp in range(len(profiles.Species)):
            if profiles.Species[sp]["S"] != "fast":
                ni_th += profiles.profiles["ni(10^19/m^3)"][:, sp]
        f_i = float(ni_th[ibc] / ne[ibc])

        # ne standing at the BC point after application: neped_20 under 'bc' (core rescaled
        # to it exactly at ibc), the incoming grid value under 'keep'
        ne_used_1e19 = ne_bc_1e19 if self.density_treatment == "bc" else float(ne[ibc])

        # Delivered betap' of the INCOMING state: the response to the BC the previous
        # incarnation applied (measured response curve for the relaxation/response_fit servo)
        betap_prime_delivered = (float(p_th[ibc]) - p_sep) / (coef * (1.0 - psin_g))
        record_bc_response(self.maestro_instance, "betap", betap_prime_delivered)

        # Closed-form inversion (all thermal ions share Ti = tite*Te_bc after application)
        p_bc_target = coef * self.betap_prime * (1.0 - psin_g) + p_sep
        Te_bc_target = p_bc_target / (ne_used_1e19 * 1e19 * (1.0 + f_i * self.tite) * 1e3 * e_J)

        # Guard (xi*C-clamp analog): an inversion at/below the incoming separatrix temperature
        # would make the edge isothermal/inverted (TRANSP SIGFPEs on it). Near-impossible with
        # a positive target and nsep < nped, but clamp at 1.2*Tesep and flag if it happens.
        Te_bc_floor = 1.2 * Te_sep
        Te_bc_at_floor = Te_bc_target <= Te_bc_floor
        if Te_bc_at_floor:
            print(
                f"\t- WARNING: betap inversion yields Te_bc={Te_bc_target*1e3:.1f} eV at/below "
                f"1.2 x Tesep={Te_sep*1e3:.1f} eV. Clamping to {Te_bc_floor*1e3:.1f} eV (railed).",
                typeMsg="w",
            )
            Te_bc_target = Te_bc_floor

        if self.servo_mode == "response_fit":
            # No Te_bc bound concept in this method, so the servo only sees its trust clamp
            Te_bc, servo_diag = servo_step(
                self.maestro_instance, "betap", self.betap_prime, Te_bc_target, (0.0, np.inf),
                fit_window=self.servo_fit_window,
                alpha_band=self.servo_alpha_band,
                trust_factor=self.servo_trust_factor,
                seed_gain=self.servo_seed_gain,
            )
        else:
            servo_diag = None
            Te_bc = relax_bc(self.maestro_instance, Te_bc_target, self.relaxation)
        Ti_bc = Te_bc * self.tite

        # Effective betap' actually applied (== prescribed when no relaxation/clamp acted)
        p_bc_applied = ne_used_1e19 * 1e19 * (1.0 + f_i * self.tite) * Te_bc * 1e3 * e_J
        betap_prime_eff = (p_bc_applied - p_sep) / (coef * (1.0 - psin_g))

        print(f"\t- Betap norm: Ip={Ip_A*1e-6:.3f} MA, L_pol={L_pol_m:.3f} m -> Bpa={Bpa:.4f} T")
        print(f"\t- betap'={self.betap_prime:.3f}" +
              (f" (betap'_eff applied after relaxation: {betap_prime_eff:.3f})"
               if abs(betap_prime_eff - self.betap_prime) > 1e-6 else ""))
        print(f"\t- p_sep={p_sep:.1f} Pa, p_bc={p_bc_applied:.1f} Pa, T_sep={Te_sep:.4f} keV")
        print(f"\t- Te_bc={Te_bc:.4f} keV,  Ti_bc={Ti_bc:.4f} keV")

        # ------------------------------------------------------------------
        # Modify profiles: shared machinery first (core a/L-preserving rescale + the
        # standard edge), then rewrite the edge PRESSURE-LINEAR ("option 2"): overwrite
        # Te/Ti on the edge interior so p_th follows the straight line from
        # (psin_bc_grid, p_bc_applied) to (1, p_sep) — d(beta_p)/dpsin is then CONSTANT
        # = -betap'_eff along the whole edge (equal to the secant), instead of the
        # sagging product-of-linears the shared linear-in-Te edge produces.
        # ------------------------------------------------------------------

        profiles_out = _apply_bc(
            profiles,
            rho_bc_rho,
            psin_bc,
            Te_bc,
            Ti_bc,
            ne_bc_1e19,
            density_treatment=self.density_treatment,
        )
        profiles_out = _rewrite_edge_pressure_linear(
            profiles_out, rho_bc_rho, self.tite, p_bc_applied, p_sep,
        )

        # ------------------------------------------------------------------
        # Store
        # ------------------------------------------------------------------

        bc_results = {
            "method":            "betap",
            "x_bc":              self.x_bc,
            "bc_coordinate":     self.bc_coordinate,
            "rho_bc_rho":        rho_bc_rho,
            "psin_bc":           psin_bc,
            "psin_bc_grid":      psin_g,
            "betap_prime":       self.betap_prime,
            "betap_prime_eff":   betap_prime_eff,
            "betap_prime_delivered": betap_prime_delivered,
            "relaxation":        self.relaxation,
            "servo_mode":        self.servo_mode,
            # Railed if the floor clamp or the servo bounds acted (the actuator did not
            # go where the solve asked); pairs formed from it are excluded from servo fits
            "Te_bc_applied_railed": bool(servo_diag["bounds_clamped"]) if servo_diag is not None else bool(Te_bc_at_floor),
            "Te_bc_at_floor":    bool(Te_bc_at_floor),
            "Ip_MA":             Ip_A * 1e-6,
            "L_pol_m":           L_pol_m,
            "Bpa_T":             Bpa,
            "p_sep_Pa":          p_sep,
            "p_bc_target_Pa":    p_bc_target,
            "p_bc_applied_Pa":   p_bc_applied,
            "ne_bc_used_1e19":   ne_used_1e19,
            "f_i_bc":            f_i,
            "Te_bc":             Te_bc,
            "Te_bc_target":      Te_bc_target,
            "Ti_bc":             Ti_bc,
            "Te_sep":            Te_sep,
            "ne_bc_20":          ne_bc_20,
            "neped_20":          ne_bc_20,   # keep standard key name for compatibility
            "ne_sep_1e19":       ne_sep_1e19,
            "tite":              self.tite,
            "density_treatment": self.density_treatment,
        }

        if servo_diag is not None:
            bc_results.update({
                "servo_rung":            servo_diag["rung"],
                "servo_n_pairs":         servo_diag["n_pairs"],
                "servo_alpha":           servo_diag["alpha"],
                "servo_slope":           servo_diag["slope"],
                "servo_trust_clamped":   servo_diag["trust_clamped"],
                "servo_bounds_clamped":  servo_diag["bounds_clamped"],
            })

        for key, val in bc_results.items():
            print(f"\t\t- {key}: {val}")

        # Write intermediate result
        profiles_out.write_state(file=self.folder / self._state_file)

        self.profiles_output = profiles_out

        return bc_results

    # ------------------------------------------------------------------
    # method 'confinement': generic iterative metric-matching solve
    # (the template for a future method whose target quantity needs the full
    #  modified state: apply the BC at each trial Te_bc, re-derive, iterate)
    # ------------------------------------------------------------------

    def _run_confinement(self, profiles, rho, Te,
                         rho_bc_rho, psin_bc, ne_bc_1e19, ne_bc_20):

        H_key, tau_key = _SCALING_MAP[self.confinement_scaling]

        # ------------------------------------------------------------------
        # Minimize over Te_bc to match the target H-factor
        # (same spirit as the eped_initializer a/LT matching of BetaN)
        # ------------------------------------------------------------------

        H_initial = float(profiles.derived[H_key])

        # H of the INCOMING state = the H actually DELIVERED at the Te_bc the previous
        # incarnation applied (after the intervening beats moved the state). Recorded in
        # both servo modes: it is the measured response curve, diagnostic gold either way.
        record_bc_response(self.maestro_instance, self.confinement_scaling, H_initial)

        Te_bc_guess = float(np.interp(rho_bc_rho, rho, Te))

        # Isothermal-edge guard: never let the scan probe at/below the separatrix
        # temperature of the incoming state (TRANSP SIGFPEs on a flat/inverted edge).
        # With sep_max_frac the guard is inverted (the APPLIED Tesep follows Te_bc
        # down inside _apply_bc), so no dynamic floor is needed.
        Te_sep = float(Te[-1])
        if self.Te_bc_min_Tesep_factor is None:
            Te_bc_floor = self.Te_bc_bounds[0]
        else:
            Te_bc_floor = max(self.Te_bc_bounds[0], self.Te_bc_min_Tesep_factor * Te_sep)
        Te_bc_bounds_eff = (Te_bc_floor, self.Te_bc_bounds[1])
        if Te_bc_floor > self.Te_bc_bounds[0]:
            print(
                f"\t- Te_bc floor raised {self.Te_bc_bounds[0]:.3f} -> {Te_bc_floor:.3f} keV "
                f"({self.Te_bc_min_Tesep_factor:.2f} x Tesep = {Te_sep*1e3:.1f} eV)"
            )
        if self.sep_max_frac is not None:
            print(
                f"\t- Applied-Tesep cap active: Tesep_applied = min(Tesep, "
                f"{self.sep_max_frac:.2f} x Te_bc)  (incoming Tesep = {Te_sep*1e3:.1f} eV)"
            )
        Te_bc_guess = max(Te_bc_guess, Te_bc_floor)

        print(
            f"\t- Optimizing Te_bc to match {self.confinement_scaling} = {self.confinement:.3f} "
            f"(initial {self.confinement_scaling} = {H_initial:.3f}, "
            f"Te_bc guess = {Te_bc_guess:.4f} keV, bounds = {Te_bc_bounds_eff} keV)"
        )

        history = []   # (Te_bc, H, residual) per evaluation

        def _H_at(Te_bc_trial):
            p_mod = _apply_bc(
                profiles, rho_bc_rho, psin_bc,
                Te_bc_trial, Te_bc_trial * self.tite, ne_bc_1e19,
                edge_shape=self.edge_shape,
                density_treatment=self.density_treatment,
                sep_max_frac=self.sep_max_frac,
            )
            if self.alpha_power_feedback:
                p_mod = _recompute_alpha_power(p_mod)
            H_trial = float(p_mod.derived[H_key])
            res = ((H_trial - self.confinement) / self.confinement) ** 2
            history.append((Te_bc_trial, H_trial, res))
            return H_trial

        def _residual(Te_bc_arr):
            H_trial = _H_at(float(Te_bc_arr[0]))
            return ((H_trial - self.confinement) / self.confinement) ** 2

        opt = minimize(
            _residual,
            [Te_bc_guess],
            method="Nelder-Mead",
            tol=1e-4,
            bounds=[Te_bc_bounds_eff],
        )
        Te_bc_target = float(opt.x[0])
        Te_bc_at_floor = Te_bc_target <= Te_bc_floor * 1.001

        # Nelder-Mead with bounds can collapse its simplex onto the clipped floor and
        # declare convergence there even when the H(Te_bc) crossing sits above it (H is
        # monotone increasing in Te_bc under the BC rescale). A floor pin with H BELOW
        # the target is that artifact — the root is bracketed, so solve it directly.
        if Te_bc_at_floor and _H_at(Te_bc_floor) < self.confinement:
            for Te_hi in (Te_bc_guess, self.Te_bc_bounds[1]):
                if Te_hi > Te_bc_floor and _H_at(Te_hi) > self.confinement:
                    Te_bc_target = float(brentq(
                        lambda te: _H_at(te) - self.confinement,
                        Te_bc_floor, Te_hi, xtol=1e-4,
                    ))
                    Te_bc_at_floor = False
                    print(
                        f"\t- Floor pin rejected (H at floor below target): bracketed root find "
                        f"in [{Te_bc_floor:.4f}, {Te_hi:.4f}] keV -> Te_bc = {Te_bc_target:.4f} keV",
                        typeMsg="i",
                    )
                    break

        if self.servo_mode == "response_fit":
            Te_bc, servo_diag = servo_step(
                self.maestro_instance, self.confinement_scaling, self.confinement,
                Te_bc_target, Te_bc_bounds_eff,
                fit_window=self.servo_fit_window,
                alpha_band=self.servo_alpha_band,
                trust_factor=self.servo_trust_factor,
                seed_gain=self.servo_seed_gain,
            )
        else:
            servo_diag = None
            Te_bc = relax_bc(self.maestro_instance, Te_bc_target, self.relaxation)
        Ti_bc = Te_bc * self.tite

        # Pin flag of the APPLIED value (Te_bc_at_floor above flags the frozen-solve target
        # instead): a railed actuation is not a valid sample of the response curve, so the
        # next incarnation must exclude the pair it forms
        Te_bc_applied_railed = bool(
            (Te_bc <= Te_bc_floor * 1.001) or (Te_bc >= self.Te_bc_bounds[1] * 0.999)
        )

        # ------------------------------------------------------------------
        # Apply the optimal boundary condition
        # ------------------------------------------------------------------

        profiles_out = _apply_bc(
            profiles, rho_bc_rho, psin_bc, Te_bc, Ti_bc, ne_bc_1e19,
            edge_shape=self.edge_shape,
            density_treatment=self.density_treatment,
            sep_max_frac=self.sep_max_frac,
        )
        if self.alpha_power_feedback:
            # Recomputed sources also travel in the beat output state, so the next
            # beat receives a power balance consistent with the H-factor reported here
            profiles_out = _recompute_alpha_power(profiles_out)

        H_achieved   = float(profiles_out.derived[H_key])
        tauE         = float(profiles_out.derived["tauE"])
        tau_scaling  = float(profiles_out.derived[tau_key])

        mismatch = abs(H_achieved - self.confinement) / self.confinement
        was_relaxed = Te_bc != Te_bc_target
        # Both servo modes deliberately depart from the frozen-solve target; only the wording differs
        step_note = "servo-stepped" if self.servo_mode == "response_fit" else "relaxed"
        print(
            f"\t- Optimization finished after {len(history)} evaluations: "
            f"Te_bc = {Te_bc:.4f} keV, Ti_bc = {Ti_bc:.4f} keV"
            + (f" ({step_note} from target {Te_bc_target:.4f} keV)" if was_relaxed else "")
        )
        print(
            f"\t- {self.confinement_scaling}: achieved {H_achieved:.4f} "
            f"(target {self.confinement:.4f}), tauE = {tauE:.4f} s, "
            f"{tau_key} = {tau_scaling:.4f} s"
        )
        if self.alpha_power_feedback:
            print(
                f"\t- Alpha power feedback: Pfus {float(profiles.derived['Pfus']):.2f} MW (initial) -> "
                f"{float(profiles_out.derived['Pfus']):.2f} MW (at final Te_bc)"
            )
        if mismatch > 0.01:
            if was_relaxed:
                # Expected under relaxation: the target is approached across beat
                # iterations, not within one — informational, not a warning
                print(
                    f"\t- H-factor mismatch of {mismatch*100:.1f}% at the {step_note} Te_bc "
                    + ("(expected with servo_mode=response_fit; converges across beats)"
                       if self.servo_mode == "response_fit" else
                       f"(expected with relaxation={self.relaxation}; converges across beats)"),
                    typeMsg="i",
                )
            else:
                print(
                    f"\t- WARNING: H-factor mismatch of {mismatch*100:.1f}% remains after optimization "
                    f"(target may be unreachable within Te_bc bounds {Te_bc_bounds_eff} keV)",
                    typeMsg="w",
                )
        if Te_bc_at_floor:
            print(
                f"\t- Te_bc pinned at the floor {Te_bc_floor:.4f} keV: {self.confinement_scaling} = "
                f"{self.confinement:.3f} unattainable above the Tesep guard (best achievable {H_achieved:.4f})",
                typeMsg="w",
            )

        # ------------------------------------------------------------------
        # Store
        # ------------------------------------------------------------------

        history = np.array(history)   # (#evals, 3)

        # Inputs to the tau scaling (PLASMAtools.tau98y2/tau89p signature): the
        # engineering ones are fixed under the BC change; <ne>_vol and Ptot=qHeat
        # (and Wthr through tauE) can move with it. Stored so the plot reports
        # exactly what the scan used (relevant when alpha_power_feedback
        # recomputed the baseline sources).
        scaling_params = {
            "Ip_MA":    abs(float(profiles.profiles["current(MA)"][-1])),
            "Rgeo_m":   float(profiles.derived["Rgeo"]),
            "epsilon":  float(profiles.derived["a"] / profiles.derived["Rgeo"]),
            "kappa_a":  float(profiles.derived["kappa_a"]),
            "B0_T":     float(profiles.derived["B0"]),
            "mbg_amu":  float(profiles.derived["mbg_main"]),
        }

        bc_results = {
            "method":              "confinement",
            "x_bc":                self.x_bc,
            "bc_coordinate":       self.bc_coordinate,
            "rho_bc_rho":          rho_bc_rho,
            "psin_bc":             psin_bc,
            "confinement_scaling": self.confinement_scaling,
            "H_target":            self.confinement,
            "H_initial":           H_initial,
            "H_achieved":          H_achieved,
            "tauE":                tauE,
            "tau_scaling":         tau_scaling,
            "tauE_initial":        float(profiles.derived["tauE"]),
            "tau_scaling_initial": float(profiles.derived[tau_key]),
            "Wthr_initial_MJ":     float(profiles.derived["Wthr"]),
            "Wthr_achieved_MJ":    float(profiles_out.derived["Wthr"]),
            "qHeat_initial_MW":    float(profiles.derived["qHeat"]),
            "qHeat_achieved_MW":   float(profiles_out.derived["qHeat"]),
            "ne_vol20_initial":    float(profiles.derived["ne_vol20"]),
            "ne_vol20_achieved":   float(profiles_out.derived["ne_vol20"]),
            "scaling_params":      scaling_params,
            "Te_bc":               Te_bc,
            "Te_bc_target":        Te_bc_target,
            "relaxation":          self.relaxation,
            "servo_mode":          self.servo_mode,
            "Te_bc_applied_railed": Te_bc_applied_railed,
            "Ti_bc":               Ti_bc,
            "Te_bc_guess":         Te_bc_guess,
            "Te_bc_bounds":        self.Te_bc_bounds,
            "Te_bc_bounds_eff":    Te_bc_bounds_eff,
            "Te_bc_at_floor":      Te_bc_at_floor,
            "Te_sep_keV":          Te_sep,
            "ne_bc_20":            ne_bc_20,
            "neped_20":            ne_bc_20,   # keep standard key name for compatibility
            "tite":                self.tite,
            "edge_shape":          self.edge_shape,
            "density_treatment":   self.density_treatment,
            "alpha_power_feedback": self.alpha_power_feedback,
            "Pfus_initial":        float(profiles.derived["Pfus"]),
            "Pfus_achieved":       float(profiles_out.derived["Pfus"]),
            "history_Te_bc":       history[:, 0],
            "history_H":           history[:, 1],
            "history_residual":    history[:, 2],
        }

        if servo_diag is not None:
            bc_results.update({
                "servo_rung":           servo_diag["rung"],
                "servo_n_pairs":        servo_diag["n_pairs"],
                "servo_alpha":          servo_diag["alpha"],
                "servo_slope":          servo_diag["slope"],
                "servo_trust_clamped":  servo_diag["trust_clamped"],
                "servo_bounds_clamped": servo_diag["bounds_clamped"],
            })

        for key, val in bc_results.items():
            if not key.startswith("history_"):
                print(f"\t\t- {key}: {val}")

        # Write intermediate result
        profiles_out.write_state(file=self.folder / self._state_file)

        self.profiles_output = profiles_out

        return bc_results

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self, **kwargs):

        # On a re-invocation after a prior keep_all_files: false cleanup wiped
        # self.folder, the run artifacts are gone and folder_output already holds
        # bc_results.npy + input.gacode from the prior run — do not wipe it
        # (the wipe-first flow would destroy the persisted results and then crash
        # on the missing copy source). Same guard as the TRANSP/EPED/PORTALS beats.
        if not (
            (self.folder / self._results_file).exists()
            and (self.folder / self._state_file).exists()
        ):
            self.profiles_output = PROFILEStools.gacode_state(self.folder_output / "input.gacode")
            return

        # Clear old output
        for item in self.folder_output.glob("*"):
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                IOtools.shutil_rmtree(item)

        # Copy results
        shutil.copy2(
            self.folder / self._results_file,
            self.folder_output / self._results_file,
        )

        # Write profiles to output folder
        self.profiles_output = PROFILEStools.gacode_state(
            self.folder / self._state_file
        )
        self.profiles_output.write_state(file=self.folder_output / "input.gacode")

    # ------------------------------------------------------------------
    # merge_parameters
    # ------------------------------------------------------------------

    def merge_parameters(self):
        # The bc beat does not change the grid or engineering parameters,
        # so no special merging is required (same as EPED beat).
        pass

    # ------------------------------------------------------------------
    # grab_output
    # ------------------------------------------------------------------

    def grab_output(self, **kwargs):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:
            loaded_results = np.load(
                self.folder_output / self._results_file, allow_pickle=True
            ).item()
            profiles = PROFILEStools.gacode_state(self.folder_output / "input.gacode")
        else:
            loaded_results = None
            profiles = None

        return loaded_results, profiles

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------

    def plot(self, fn=None, counter=0, full_plot=True):

        if fn is None:
            fn = GUItools.FigureNotebook(f"BC ({self.method})")

        loaded_results, profiles_after = self.grab_output()

        profiles_before = self.incoming_profiles()
        if profiles_before is not None:
            profiles_before.derive_quantities(rederiveGeometry=False)

        if loaded_results is not None and profiles_after is not None and profiles_before is not None:
            profiles_after.derive_quantities(rederiveGeometry=False)
            if self.method == "sharpness":
                _plot_sharpness_bc(fn, loaded_results, profiles_before, profiles_after, counter)
                _plot_bc_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter,
                                         label="Sharpness")
            elif self.method == "confinement":
                _plot_confinement_bc(fn, loaded_results, profiles_before, profiles_after, counter)
                _plot_bc_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter,
                                         label="Confinement")
            elif self.method == "betap":
                _plot_betap_bc(fn, loaded_results, profiles_before, profiles_after, counter)
                _plot_bc_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter,
                                         label="Betap")
        else:
            # Fallback: nothing to show (never ran yet, or the inputs were pruned)
            label = self.method.capitalize()
            fig = fn.add_figure(label=label, tab_color=counter)
            fig.add_subplot(111).text(0.5, 0.5, f"No {self.method} bc results available",
                                      ha="center", va="center", transform=fig.transFigure)

        return f"\t\t- Plotting of bc beat ({self.method}) done"

    # ------------------------------------------------------------------
    # _inform / _inform_save
    # ------------------------------------------------------------------

    def _inform(self):
        """Receive parameters from previous beats or from the plasma/parameters namelist."""

        # 0. Grab the last PORTALS prediction radius if requested (stored separately;
        #    sharpness_coordinate is NOT changed — it governs the derivative, not the location)
        if self.update_bc_based_on_portals:
            tb = self.maestro_instance.parameters_trans_beat
            if tb.get("predicted_rho") is not None:
                self._portals_rho_bc = (float(tb["predicted_rho"][-1]), "rho")
                print(f"\t\t- update_bc_based_on_portals: BC location will use predicted_rho[-1] = {self._portals_rho_bc[0]:.4f} (rho)")
            elif tb.get("predicted_roa") is not None:
                self._portals_rho_bc = (float(tb["predicted_roa"][-1]), "roa")
                print(f"\t\t- update_bc_based_on_portals: BC location will use predicted_roa[-1] = {self._portals_rho_bc[0]:.4f} (roa)")
            else:
                print("\t\t- update_bc_based_on_portals=True but no predicted_rho/roa found in trans-beat parameters; keeping x_bc as specified", typeMsg="w")

        # 1. neped_20 from a previous EPED or bc beat (highest priority)
        if "neped_20" in self.maestro_instance.parameters_trans_beat:
            self.neped_20 = self.maestro_instance.parameters_trans_beat["neped_20"]
            print(f"\t\t- Using neped_20 from previous beat: {self.neped_20:.3f}")

        # 2. Fall back to plasma/parameters section of the namelist
        elif self.neped_20 is None:
            try:
                self.neped_20 = self.maestro_instance.maestro_namelist["plasma"]["parameters"]["neped_20"]
                print(f"\t\t- Using neped_20 from namelist plasma/parameters: {self.neped_20:.3f}")
            except (KeyError, TypeError):
                pass  # will fall back to reading from profiles at rho_bc in _run()

    def _inform_save(self, bc_output=None):
        """Save parameters for subsequent beats."""

        if bc_output is None:
            bc_output, _ = self.grab_output()

        if bc_output is None:
            return

        # Keep neped_20 (= ne_bc) available to subsequent beats
        self.maestro_instance.parameters_trans_beat["neped_20"] = bc_output[
            "neped_20"
        ]

        # rhotop is understood by PORTALS to set the last radial prediction point
        self.maestro_instance.parameters_trans_beat["rhotop"] = bc_output[
            "rho_bc_rho"
        ]

        # Applied BC temperature: memory for the relax_bc under-relaxation / response_fit
        # servo of the next bc beat (shared key across all bc methods). The rail flag
        # travels with it: the next incarnation pairs the delivered response with this
        # Te_bc and must know whether the actuator went where it was asked.
        self.maestro_instance.parameters_trans_beat["Te_bc_applied"] = bc_output[
            "Te_bc"
        ]
        self.maestro_instance.parameters_trans_beat["Te_bc_applied_railed"] = bc_output.get(
            "Te_bc_applied_railed", False
        )

        print(
            f"\t\t- neped_20={bc_output['neped_20']:.3f}, "
            f"rhotop={bc_output['rho_bc_rho']:.3f} and "
            f"Te_bc_applied={bc_output['Te_bc']:.4f} saved for future beats"
        )


# ============================================================================
# Helper functions
# ============================================================================


def _thermal_pressure_Pa(p):
    """
    Thermal pressure profile [Pa]: ne*Te + sum over THERMAL ions of ni*Ti (fast species
    excluded). Units: profiles carry n in 10^19 m^-3 and T in keV, so
    p[Pa] = n*1e19 * T*1e3 * 1.602176634e-19.
    """
    e_J = 1.602176634e-19
    p_th = p.profiles["ne(10^19/m^3)"] * p.profiles["te(keV)"]
    for sp in range(len(p.Species)):
        if p.Species[sp]["S"] != "fast":
            p_th = p_th + p.profiles["ni(10^19/m^3)"][:, sp] * p.profiles["ti(keV)"][:, sp]
    return p_th * 1e19 * 1e3 * e_J


def _betap_normalization(p):
    """
    Engineering poloidal-field normalization for beta_p:  Bpa = mu0*Ip/L_pol  [T],
    with Ip the total plasma current (|profiles['current(MA)'][-1]|, a flat profile in
    input.gacode, converted MA -> A) and L_pol the poloidal perimeter of the LCFS
    (closed-loop arc length of derived['R_surface'][0][-1], ['Z_surface'][0][-1]).
    Returns (Bpa [T], L_pol [m], Ip [A]). Re-derives geometry if the surface contours
    are not present in derived.
    """
    mu0 = 4.0e-7 * np.pi
    Ip_A = abs(float(p.profiles["current(MA)"][-1])) * 1e6
    if "R_surface" not in p.derived:
        p.derive_quantities(rederiveGeometry=True)
    R = np.asarray(p.derived["R_surface"][0][-1])
    Z = np.asarray(p.derived["Z_surface"][0][-1])
    L_pol = float(np.sum(np.hypot(np.diff(np.append(R, R[0])), np.diff(np.append(Z, Z[0])))))
    return mu0 * Ip_A / L_pol, L_pol, Ip_A


def _rewrite_edge_pressure_linear(p, rho_bc_rho, tite, p_bc_Pa, p_sep_Pa):
    """
    Betap-only edge rewrite ("option 2"): overwrite Te/Ti on the EDGE INTERIOR grid
    points so the thermal pressure follows the straight line (the ibc->separatrix
    secant) from (psin_bc_grid, p_bc) to (1, p_sep):

        Te(psin) = p_lin(psin) / [(ne + tite*sum_i(thermal) ni) * 1e19 * 1e3 * e],
        Ti = tite*Te for thermal ions (fast untouched, same handling as _apply_bc)

    which is exact since p_th = Te*(ne + tite*ni_th) when Ti = tite*Te: the local
    d(beta_p)/dpsin is then constant along the edge, equal to the two-point betap'.
    The densities used are the ones standing AFTER the density treatment, so the
    rewrite is exact under both 'bc' and 'keep'.

    Offset convention preserved: the point at i_edge = ibc+1 keeps its core-extended
    value (the BC point's a/L stencil reads core-shaped values on both sides, as with
    the shared edge), and the separatrix point is untouched (its own Te/Ti already
    give exactly p_sep). Both endpoints of the metric are pinned, so the delivered
    two-point betap' is unaffected; only the interior points move onto the secant.
    The one-cell segment i_edge -> i_edge+1 carries the (tiny) kink, exactly where
    the shared edge puts its slope discontinuity.
    """
    e_J = 1.602176634e-19

    rho  = p.profiles["rho(-)"]
    psin = p.derived["psi_pol_n"]
    ibc = int(np.argmin(np.abs(rho - rho_bc_rho)))
    i_edge = min(ibc + 1, len(rho) - 2)
    psin_g = float(psin[ibc])

    j0, j1 = i_edge + 1, len(rho) - 1   # rewrite j0..j1-1; separatrix (j1) untouched
    if j0 >= j1:
        return p

    p_lin = p_bc_Pa + (p_sep_Pa - p_bc_Pa) * (psin[j0:j1] - psin_g) / (1.0 - psin_g)

    ni_th = np.zeros_like(p.profiles["ne(10^19/m^3)"])
    for sp in range(len(p.Species)):
        if p.Species[sp]["S"] != "fast":
            ni_th += p.profiles["ni(10^19/m^3)"][:, sp]
    denom = (p.profiles["ne(10^19/m^3)"][j0:j1] + tite * ni_th[j0:j1]) * 1e19 * 1e3 * e_J

    TiTimain_orig = p.profiles["ti(keV)"] / p.profiles["ti(keV)"][:, [0]]

    Te_edge = p_lin / denom
    p.profiles["te(keV)"][j0:j1] = Te_edge
    p.profiles["ti(keV)"][j0:j1, 0] = tite * Te_edge

    p.makeAllThermalIonsHaveSameTemp()
    for sp in range(len(p.Species)):
        if p.Species[sp]["S"] == "fast":
            p.profiles["ti(keV)"][:, sp] = p.profiles["ti(keV)"][:, 0] * TiTimain_orig[:, sp]

    p.derive_quantities(rederiveGeometry=False)
    from mitim_tools.misc_tools import LOGtools
    with LOGtools.HiddenPrints():
        p.selfconsistentPTOT()

    return p


def _convert_bc_location(rho_bc, coordinate, rho, roa, psi_pol_n):
    """
    Convert the boundary condition location to rho_tor.

    Parameters
    ----------
    rho_bc      : float   – BC location in the given coordinate
    coordinate  : str     – 'rho', 'roa', or 'psin'
    rho         : array   – rho_tor grid
    roa         : array   – r/a grid
    psi_pol_n   : array   – normalised poloidal flux grid
    """
    if coordinate == "rho":
        return float(rho_bc)
    elif coordinate == "roa":
        return float(np.interp(rho_bc, roa, rho))
    elif coordinate == "psin":
        return float(np.interp(rho_bc, psi_pol_n, rho))
    else:
        raise ValueError(f"Unknown coordinate: {coordinate}")


def _apply_bc(profiles, rho_bc_rho, psin_bc, Te_bc, Ti_bc, ne_bc_1e19, edge_shape="linear",
              density_treatment="bc", sep_max_frac=None):
    """
    Modify *profiles* in-place (returns modified copy) so that:

    - Edge region, per edge_shape:
        'linear': Te, Ti, ne interpolated linearly in psi_n down to the separatrix.
        'tanh':   Te, Ti, ne follow the pedestal tanh of FunctionalForms.pedestal_tanh
                  in r/a (the same functional form the eped_initializer uses).
                  NOTE: the sharpness method's xi definition assumes the linear edge
                  gradient, so it always uses 'linear'; 'tanh' is for the confinement
                  method, whose matching criterion (H-factor) is integral.
        The analytical edge starts ONE grid point past the BC (anchored at the
        core-extended value at ibc+1): the BC point's 3-point derivative stencil
        (MATHtools.deriv / GACODE bound_deriv) then reads core-shaped values on
        both sides, so the a/L gradients that PORTALS reads at its last control
        point are preserved exactly. The slope discontinuity sits at ibc+1,
        outside the PORTALS prediction grid. (Anchoring the edge at ibc itself
        polluted the BC-point gradient by ~2x and propagated into the next
        PORTALS beat's starting DVs.)
    - Core region (rho <= rho_bc, extended through ibc+1):
        profiles multiplied by y_bc / y_old(ibc) — the exact way to change the
        BC value while preserving the normalised gradients a/L of the current
        transport solution (log-derivatives are invariant under scaling; no
        derive-then-integrate roundtrip error).
    - Density, per density_treatment:
        'bc':   ne treated like the temperatures (core rescaled to ne_bc_1e19
                preserving a/Lne, edge replaced) and all ion densities rescaled
                to keep ni/ne ratios.
        'keep': ne and all ion densities left exactly as in the input profiles;
                only Te/Ti are modified (ne_bc_1e19 is ignored).

    Parameters
    ----------
    profiles    : gacode_state (copied inside)
    rho_bc_rho  : float   – BC location in rho_tor
    psin_bc     : float   – BC location in psi_n
    Te_bc       : float   – Te at BC  [keV]
    Ti_bc       : float   – Ti at BC  [keV]
    ne_bc_1e19  : float   – ne at BC  [10^19 m^-3] (unused if density_treatment='keep')
    edge_shape  : str     – 'linear' (default) or 'tanh'
    density_treatment : str – 'bc' (default) or 'keep'
    sep_max_frac : float or None
        If set, cap the APPLIED separatrix temperatures at sep_max_frac * y_bc
        (Te and Ti only; ne untouched): y_sep_applied = min(y_sep, sep_max_frac*y_bc).
        This keeps the edge monotone-decreasing when the BC is pushed at/below the
        incoming separatrix temperature (TRANSP SIGFPEs on a flat/inverted edge),
        replacing the confinement method's old Te_bc >= 1.2*Tesep floor. The physical
        (e.g. Lengyel) Tesep remains recorded upstream; only the state written here
        is modified. None (default) = old behavior, separatrix values untouched.
    """

    if edge_shape not in ("linear", "tanh"):
        raise ValueError(f"edge_shape must be 'linear' or 'tanh', got '{edge_shape}'")
    if density_treatment not in ("bc", "keep"):
        raise ValueError(f"density_treatment must be 'bc' or 'keep', got '{density_treatment}'")

    p = copy.deepcopy(profiles)

    rho        = p.profiles["rho(-)"]
    psi_pol_n  = p.derived["psi_pol_n"]
    roa        = p.derived["roa"]

    ibc = int(np.argmin(np.abs(rho - rho_bc_rho)))

    # The analytical edge starts one grid point past the BC (see docstring);
    # guard the degenerate case of a BC at the very edge of the grid.
    i_edge = min(ibc + 1, len(rho) - 2)

    # ---- edge grids ----
    # Anchored at grid values at i_edge to ensure exact continuity with the
    # extended core.
    psin_edge      = psi_pol_n[i_edge:]
    psin_anchor    = psi_pol_n[i_edge]
    roa_edge       = roa[i_edge:]
    roa_anchor     = roa[i_edge]

    def _linear_edge(y_anchor, y_sep):
        """Linear interpolation in psi_n from the anchor (at i_edge) to separatrix."""
        return y_anchor + (y_sep - y_anchor) * (psin_edge - psin_anchor) / (1.0 - psin_anchor)

    def _tanh_edge(y_anchor, y_sep):
        """Pedestal tanh in r/a from the anchor (at i_edge) to separatrix, as in the initializer."""
        from mitim_tools.popcon_tools import FunctionalForms
        _, y = FunctionalForms.pedestal_tanh(y_anchor, y_sep, 1.0 - roa_anchor, x=roa_edge)
        return y

    _edge = _tanh_edge if edge_shape == "tanh" else _linear_edge

    def _scale_core(y, y_bc):
        """Exact a/L-preserving core: multiply by y_bc/y(ibc) through i_edge."""
        y_new = y.copy()
        y_new[:i_edge + 1] = y[:i_edge + 1] * (y_bc / y[ibc])
        return y_new

    Te_sep     = float(p.profiles["te(keV)"][-1])
    Ti_sep     = float(p.profiles["ti(keV)"][-1, 0])
    ne_sep     = float(p.profiles["ne(10^19/m^3)"][-1])

    if sep_max_frac is not None:
        Te_sep = min(Te_sep, sep_max_frac * Te_bc)
        Ti_sep = min(Ti_sep, sep_max_frac * Ti_bc)

    # ---- build new full profiles ----
    Te_new  = _scale_core(p.profiles["te(keV)"],       Te_bc)
    Ti_new  = _scale_core(p.profiles["ti(keV)"][:, 0], Ti_bc)

    # Replace edge with the selected shape, anchored at the core-extended value at i_edge
    Te_new[i_edge:]  = _edge(Te_new[i_edge], Te_sep)
    Ti_new[i_edge:]  = _edge(Ti_new[i_edge], Ti_sep)

    if density_treatment == "bc":
        ne_new          = _scale_core(p.profiles["ne(10^19/m^3)"], ne_bc_1e19)
        ne_new[i_edge:] = _edge(ne_new[i_edge], ne_sep)
        # Ratios of ion species to electron density (before modification)
        nine_orig = p.profiles["ni(10^19/m^3)"] / p.profiles["ne(10^19/m^3)"][:, None]

    # ---- store modified profiles ----
    TiTimain_orig = p.profiles["ti(keV)"] / p.profiles["ti(keV)"][:, [0]]

    p.profiles["te(keV)"]          = Te_new
    p.profiles["ti(keV)"][:, 0]   = Ti_new

    if density_treatment == "bc":
        p.profiles["ne(10^19/m^3)"]   = ne_new
        # Keep ion-to-electron density ratios
        for i in range(p.profiles["ni(10^19/m^3)"].shape[-1]):
            p.profiles["ni(10^19/m^3)"][:, i] = ne_new * nine_orig[:, i]

    # Make all thermal ions share the same temperature profile
    p.makeAllThermalIonsHaveSameTemp()

    # Restore relative fast-ion temperatures
    for sp in range(len(p.Species)):
        if p.Species[sp]["S"] == "fast":
            p.profiles["ti(keV)"][:, sp] = p.profiles["ti(keV)"][:, 0] * TiTimain_orig[:, sp]

    p.derive_quantities(rederiveGeometry=False)

    # The rescale changed n,T (up to orders of magnitude at extreme BCs) but nothing
    # above touches the ptot(Pa) column; make the written state self-consistent
    # (downstream PORTALS/TRANSP recompute or ignore it, but direct readers of the
    # beat_results input.gacode would otherwise get pre-BC pressure silently).
    # Silenced: the confinement method calls this per Nelder-Mead trial (same
    # HiddenPrints pattern as its _recompute_alpha_power)
    from mitim_tools.misc_tools import LOGtools
    with LOGtools.HiddenPrints():
        p.selfconsistentPTOT()

    return p


# ============================================================================
# Plotting helpers — method 'sharpness'
# ============================================================================


def _plot_sharpness_bc(fn, loaded_results, profiles_before, profiles_after, counter):
    """
    3-panel sharpness figure.

    Panel 0 – Profiles vs psi_n  (Te/Ti left axis [keV], ne right axis [1e20 m^-3])
    Panel 1 – |dT/d(psi_n)| vs psi_n  (Te solid, Ti dashed; key xi visualisation)
    Panel 2 – a/LQ vs roa  (Te solid, Ti dashed, ne dotted; all dimensionless)

    Color: before = blue, after / xi run = red, xi=1 reference = gray dashed.
    Y-axes run from 0 to the maximum value found up to x_bc + 0.025 (in the
    respective coordinate), giving a stable, informative scale.
    """

    FS      = 13    # axis-label / title font size
    FS_tick = 11    # tick label font size
    FS_leg  = 10    # legend font size
    FS_ann  = 11    # annotation font size
    MARGIN  = 0.01  # how far past x_bc to evaluate the y-axis ceiling

    # ------------------------------------------------------------------
    # Unpack stored results
    # ------------------------------------------------------------------
    xi         = loaded_results["sharpness"]
    C          = loaded_results["C"]
    Te_bc      = loaded_results["Te_bc"]
    Ti_bc      = loaded_results["Ti_bc"]
    Te_sep     = loaded_results["Te_sep"]
    ne_bc_20   = loaded_results["ne_bc_20"]
    rho_bc_rho = loaded_results["rho_bc_rho"]
    tite       = loaded_results["tite"]

    Te_bc_xi1 = Te_sep / (1.0 - 1.0 * C)
    Ti_bc_xi1 = Te_bc_xi1 * tite

    # ------------------------------------------------------------------
    # Profile arrays
    # ------------------------------------------------------------------
    psin_b = profiles_before.derived["psi_pol_n"]
    roa_b  = profiles_before.derived["roa"]
    Te_b   = profiles_before.profiles["te(keV)"]
    Ti_b   = profiles_before.profiles["ti(keV)"][:, 0]
    ne_b   = profiles_before.profiles["ne(10^19/m^3)"] * 0.1

    psin_a = profiles_after.derived["psi_pol_n"]
    roa_a  = profiles_after.derived["roa"]
    Te_a   = profiles_after.profiles["te(keV)"]
    Ti_a   = profiles_after.profiles["ti(keV)"][:, 0]
    ne_a   = profiles_after.profiles["ne(10^19/m^3)"] * 0.1

    Ti_sep = float(profiles_before.profiles["ti(keV)"][-1, 0])

    ibc       = int(np.argmin(np.abs(profiles_after.profiles["rho(-)"] - rho_bc_rho)))
    psin_bc_g = psin_a[ibc]
    roa_bc    = roa_a[ibc]

    psin_edge   = psin_a[ibc:]
    _lin = lambda y_bc, y_sep: (
        y_bc + (y_sep - y_bc) * (psin_edge - psin_bc_g) / (1.0 - psin_bc_g)
    )
    Te_edge_xi1 = _lin(Te_bc_xi1, Te_sep)
    Ti_edge_xi1 = _lin(Ti_bc_xi1, Ti_sep)

    # ------------------------------------------------------------------
    # Intermediate profile: before-core extended with constant dT/dpsi_n
    # slope at rho_bc into the edge region.  Shows where T_sep would land
    # if the core gradient just continued outward.
    # ------------------------------------------------------------------
    _dTe_dpsin = np.gradient(Te_b, psin_b)
    _dTi_dpsin = np.gradient(Ti_b, psin_b)
    _dne_dpsin = np.gradient(ne_b, psin_b)
    slope_Te = float(_dTe_dpsin[ibc])   # negative (T decreasing outward)
    slope_Ti = float(_dTi_dpsin[ibc])
    slope_ne = float(_dne_dpsin[ibc])

    Te_bc_b = float(Te_b[ibc])
    Ti_bc_b = float(Ti_b[ibc])
    ne_bc_b = float(ne_b[ibc])

    Te_int_edge = Te_bc_b + slope_Te * (psin_edge - psin_bc_g)
    Ti_int_edge = Ti_bc_b + slope_Ti * (psin_edge - psin_bc_g)
    ne_int_edge = ne_bc_b + slope_ne * (psin_edge - psin_bc_g)

    # Full intermediate arrays (core = before, edge = extrapolated)
    Te_int = np.concatenate([Te_b[:ibc], Te_int_edge])
    Ti_int = np.concatenate([Ti_b[:ibc], Ti_int_edge])
    ne_int = np.concatenate([ne_b[:ibc], ne_int_edge])
    psin_int = np.concatenate([psin_b[:ibc], psin_edge])
    roa_int  = np.concatenate([roa_b[:ibc],  roa_a[ibc:]])

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------
    def _dpsin(psin, y):
        return np.abs(np.gradient(y, psin))

    def _aL(roa, y):
        return CALCtools.derivation_into_Lx(
            torch.from_numpy(roa),
            torch.from_numpy(np.where(y > 0, y, 1e-30)),
            array=False,
        ).numpy()

    dTe_b, dTi_b, dne_b = _dpsin(psin_b, Te_b), _dpsin(psin_b, Ti_b), _dpsin(psin_b, ne_b)
    dTe_a, dTi_a, dne_a = _dpsin(psin_a, Te_a), _dpsin(psin_a, Ti_a), _dpsin(psin_a, ne_a)
    dTe_xi1 = np.abs(np.gradient(Te_edge_xi1, psin_edge))
    dTi_xi1 = np.abs(np.gradient(Ti_edge_xi1, psin_edge))
    dTe_int = _dpsin(psin_int, Te_int)
    dTi_int = _dpsin(psin_int, Ti_int)
    dne_int = _dpsin(psin_int, ne_int)

    aLTe_b, aLTi_b = _aL(roa_b, Te_b), _aL(roa_b, Ti_b)
    aLne_b         = _aL(roa_b, ne_b)
    aLTe_a, aLTi_a = _aL(roa_a, Te_a), _aL(roa_a, Ti_a)
    aLne_a         = _aL(roa_a, ne_a)
    aLTe_int = _aL(roa_int, Te_int)
    aLTi_int = _aL(roa_int, Ti_int)
    aLne_int = _aL(roa_int, ne_int)

    # Scalar gradient values that define xi (for annotation on Te psin-derivative panel)
    grad_edge_Te  = (Te_bc     - Te_sep) / (1.0 - psin_bc_g)
    grad_core_Te  = grad_edge_Te / xi
    grad_edge_xi1 = (Te_bc_xi1 - Te_sep) / (1.0 - psin_bc_g)

    # ------------------------------------------------------------------
    # Y-axis ceiling helper: max of all supplied arrays up to x_cut
    # ------------------------------------------------------------------
    def _ymax(x_cut, *pairs):
        """
        pairs: (x_array, y_array) tuples.
        Returns 1.15 × max of all y values where x <= x_cut.
        """
        vals = []
        for x, y in pairs:
            mask = x <= x_cut
            if mask.any():
                vals.append(float(np.nanmax(np.abs(y[mask]))))
        return max(vals) * 1.15 if vals else 1.0

    psin_cut = psin_bc_g + MARGIN
    roa_cut  = roa_bc    + MARGIN

    # ------------------------------------------------------------------
    # Style constants
    # ------------------------------------------------------------------
    cb, ca, cxi, cint = "royalblue", "crimson", "gray", "darkorange"
    lw, lw_xi = 1.8, 1.3
    ls_xi     = "--"
    ls_int    = "-."

    def _vbc_psin(ax):
        ax.axvline(psin_bc_g, color="k", ls=":", lw=1.0, zorder=0)

    def _vbc_roa(ax):
        ax.axvline(roa_bc, color="k", ls=":", lw=1.0, zorder=0)

    def _style(ax, xlabel, ylabel, title, xlim, ylim_top):
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_title(title,   fontsize=FS)
        ax.set_xlim(xlim)
        ax.set_ylim(0, ylim_top)
        GRAPHICStools.addDenseAxis(ax)
        ax.tick_params(labelsize=FS_tick)

    def _legend_lines(ax):
        """Compact legend: before / intermediate / after / xi=1."""
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=cb,   lw=lw,    ls="-",    label="before"),
            Line2D([0], [0], color=cint, lw=lw,    ls=ls_int, label="intermediate"),
            Line2D([0], [0], color=ca,   lw=lw,    ls="-",    label=rf"$\xi={xi:.2f}$"),
            Line2D([0], [0], color=cxi,  lw=lw_xi, ls=ls_xi,  label=r"$\xi=1$"),
        ]
        ax.legend(handles=handles, prop={"size": FS_leg}, loc="upper right")

    # ------------------------------------------------------------------
    # Per-species ymax
    # ------------------------------------------------------------------
    ymax_Te = _ymax(psin_cut,
                    (psin_b, Te_b), (psin_a, Te_a),
                    (psin_edge, Te_edge_xi1), (psin_int, Te_int))
    ymax_Ti = _ymax(psin_cut,
                    (psin_b, Ti_b), (psin_a, Ti_a),
                    (psin_edge, Ti_edge_xi1), (psin_int, Ti_int))
    ymax_ne = _ymax(psin_cut,
                    (psin_b, ne_b), (psin_a, ne_a), (psin_int, ne_int))

    ymax_dTe = _ymax(psin_cut,
                     (psin_b, dTe_b), (psin_a, dTe_a),
                     (psin_edge, dTe_xi1), (psin_int, dTe_int))
    ymax_dTi = _ymax(psin_cut,
                     (psin_b, dTi_b), (psin_a, dTi_a),
                     (psin_edge, dTi_xi1), (psin_int, dTi_int))
    ymax_dne = _ymax(psin_cut,
                     (psin_b, dne_b), (psin_a, dne_a), (psin_int, dne_int))

    ymax_aLTe = _ymax(roa_cut,
                      (roa_b, aLTe_b), (roa_a, aLTe_a), (roa_int, aLTe_int))
    ymax_aLTi = _ymax(roa_cut,
                      (roa_b, aLTi_b), (roa_a, aLTi_a), (roa_int, aLTi_int))
    ymax_aLne = _ymax(roa_cut,
                      (roa_b, aLne_b), (roa_a, aLne_a), (roa_int, aLne_int))

    # ------------------------------------------------------------------
    # Figure: 3 rows × 3 cols
    #   Row 0: Te, Ti, ne profiles vs psi_n
    #   Row 1: |dTe/dpsin|, |dTi/dpsin|, |dne/dpsin| vs psi_n
    #   Row 2: a/L_Te, a/L_Ti, a/L_ne vs r/a
    # ------------------------------------------------------------------
    fig = fn.add_figure(label="Sharpness", tab_color=counter)
    gs  = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.45)
    axTe  = fig.add_subplot(gs[0, 0])
    axTi  = fig.add_subplot(gs[0, 1])
    axne  = fig.add_subplot(gs[0, 2])
    axdTe = fig.add_subplot(gs[1, 0])
    axdTi = fig.add_subplot(gs[1, 1])
    axdne = fig.add_subplot(gs[1, 2])
    axaLe = fig.add_subplot(gs[2, 0])
    axaLi = fig.add_subplot(gs[2, 1])
    axaLn = fig.add_subplot(gs[2, 2])

    # =====================================================================
    # ROW 0 — profiles vs psi_n
    # =====================================================================

    # --- Te ---
    axTe.plot(psin_b,    Te_b,         color=cb,   lw=lw,    ls="-")
    axTe.plot(psin_int,  Te_int,        color=cint, lw=lw,    ls=ls_int)
    axTe.plot(psin_a,    Te_a,          color=ca,   lw=lw,    ls="-")
    axTe.plot(psin_edge, Te_edge_xi1,   color=cxi,  lw=lw_xi, ls=ls_xi)
    _vbc_psin(axTe)
    _style(axTe, r"$\psi_n$", r"$T_e$ (keV)", r"$T_e$ profile", [0, 1], ymax_Te)
    _legend_lines(axTe)

    # --- Ti ---
    axTi.plot(psin_b,    Ti_b,         color=cb,   lw=lw,    ls="-")
    axTi.plot(psin_int,  Ti_int,        color=cint, lw=lw,    ls=ls_int)
    axTi.plot(psin_a,    Ti_a,          color=ca,   lw=lw,    ls="-")
    axTi.plot(psin_edge, Ti_edge_xi1,   color=cxi,  lw=lw_xi, ls=ls_xi)
    _vbc_psin(axTi)
    _style(axTi, r"$\psi_n$", r"$T_i$ (keV)", r"$T_i$ profile", [0, 1], ymax_Ti)
    _legend_lines(axTi)

    # --- ne ---
    axne.plot(psin_b,   ne_b,   color=cb,   lw=lw,    ls="-")
    axne.plot(psin_int, ne_int, color=cint, lw=lw,    ls=ls_int)
    axne.plot(psin_a,   ne_a,   color=ca,   lw=lw,    ls="-")
    _vbc_psin(axne)
    _style(axne, r"$\psi_n$", r"$n_e$ ($10^{20}$ m$^{-3}$)", r"$n_e$ profile",
           [0, 1], ymax_ne)
    _legend_lines(axne)

    # =====================================================================
    # ROW 1 — |dQ/dpsi_n| vs psi_n
    # =====================================================================

    # --- |dTe/dpsin| ---
    axdTe.plot(psin_b,    dTe_b,   color=cb,   lw=lw,    ls="-")
    axdTe.plot(psin_int,  dTe_int, color=cint, lw=lw,    ls=ls_int)
    axdTe.plot(psin_a,    dTe_a,   color=ca,   lw=lw,    ls="-")
    axdTe.plot(psin_edge, dTe_xi1, color=cxi,  lw=lw_xi, ls=ls_xi)
    axdTe.axhline(grad_edge_Te,  color=ca,  ls="-.", lw=1.0, zorder=0)
    axdTe.axhline(grad_core_Te,  color=ca,  ls=":",  lw=1.0, zorder=0)
    axdTe.axhline(grad_edge_xi1, color=cxi, ls=ls_xi, lw=1.0, zorder=0)
    x_arr = min(psin_bc_g + 0.05, 0.93)
    if abs(grad_edge_Te - grad_core_Te) > 0.01:
        axdTe.annotate(
            "", xy=(x_arr, grad_edge_Te), xytext=(x_arr, grad_core_Te),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
        )
        axdTe.text(
            x_arr + 0.02, 0.5 * (grad_edge_Te + grad_core_Te),
            rf"$\xi={xi:.2f}$", fontsize=FS_ann, color="dimgray", va="center",
        )
    _vbc_psin(axdTe)
    _style(axdTe, r"$\psi_n$", r"$|dT_e/d\psi_n|$ (keV)",
           r"$T_e$ gradient in $\psi_n$", [0, 1], ymax_dTe)
    _legend_lines(axdTe)

    # --- |dTi/dpsin| ---
    axdTi.plot(psin_b,    dTi_b,   color=cb,   lw=lw,    ls="-")
    axdTi.plot(psin_int,  dTi_int, color=cint, lw=lw,    ls=ls_int)
    axdTi.plot(psin_a,    dTi_a,   color=ca,   lw=lw,    ls="-")
    axdTi.plot(psin_edge, dTi_xi1, color=cxi,  lw=lw_xi, ls=ls_xi)
    _vbc_psin(axdTi)
    _style(axdTi, r"$\psi_n$", r"$|dT_i/d\psi_n|$ (keV)",
           r"$T_i$ gradient in $\psi_n$", [0, 1], ymax_dTi)
    _legend_lines(axdTi)

    # --- |dne/dpsin| ---
    axdne.plot(psin_b,   dne_b,   color=cb,   lw=lw,    ls="-")
    axdne.plot(psin_int, dne_int, color=cint, lw=lw,    ls=ls_int)
    axdne.plot(psin_a,   dne_a,   color=ca,   lw=lw,    ls="-")
    _vbc_psin(axdne)
    _style(axdne, r"$\psi_n$", r"$|dn_e/d\psi_n|$ ($10^{20}$ m$^{-3}$)",
           r"$n_e$ gradient in $\psi_n$", [0, 1], ymax_dne)
    _legend_lines(axdne)

    # =====================================================================
    # ROW 2 — a/L gradients vs r/a
    # =====================================================================

    # --- a/L_Te ---
    axaLe.plot(roa_b,   aLTe_b,   color=cb,   lw=lw,    ls="-")
    axaLe.plot(roa_int, aLTe_int, color=cint, lw=lw,    ls=ls_int)
    axaLe.plot(roa_a,   aLTe_a,   color=ca,   lw=lw,    ls="-")
    axaLe.axhline(loaded_results["aLT_Te_bc"], color="k", ls="-.", lw=1.0,
                  label=rf"bc: {loaded_results['aLT_Te_bc']:.2f}")
    _vbc_roa(axaLe)
    _style(axaLe, r"$r/a$", r"$a/L_{T_e}$", r"$a/L_{T_e}$ vs $r/a$",
           [0, 1], ymax_aLTe)
    axaLe.legend(prop={"size": FS_leg}, loc="upper left")

    # xi annotation: arrow between edge and core gradient levels
    x_arr = min(roa_bc + 0.05, 0.93)
    _grad_core_roa = float(np.interp(rho_bc_rho,
                                     profiles_before.profiles["rho(-)"], aLTe_b))
    _grad_edge_roa = float(np.interp(rho_bc_rho,
                                     profiles_after.profiles["rho(-)"],  aLTe_a))
    if abs(_grad_edge_roa - _grad_core_roa) > 0.05:
        axaLe.annotate(
            "", xy=(x_arr, _grad_edge_roa), xytext=(x_arr, _grad_core_roa),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
        )
        axaLe.text(
            x_arr + 0.02, 0.5 * (_grad_edge_roa + _grad_core_roa),
            rf"$\xi={xi:.2f}$", fontsize=FS_ann, color="dimgray", va="center",
        )

    # --- a/L_Ti ---
    axaLi.plot(roa_b,   aLTi_b,   color=cb,   lw=lw,    ls="-")
    axaLi.plot(roa_int, aLTi_int, color=cint, lw=lw,    ls=ls_int)
    axaLi.plot(roa_a,   aLTi_a,   color=ca,   lw=lw,    ls="-")
    _vbc_roa(axaLi)
    _style(axaLi, r"$r/a$", r"$a/L_{T_i}$", r"$a/L_{T_i}$ vs $r/a$",
           [0, 1], ymax_aLTi)
    _legend_lines(axaLi)

    # --- a/L_ne ---
    axaLn.plot(roa_b,   aLne_b,   color=cb,   lw=lw,    ls="-")
    axaLn.plot(roa_int, aLne_int, color=cint, lw=lw,    ls=ls_int)
    axaLn.plot(roa_a,   aLne_a,   color=ca,   lw=lw,    ls="-")
    _vbc_roa(axaLn)
    _style(axaLn, r"$r/a$", r"$a/L_{n_e}$", r"$a/L_{n_e}$ vs $r/a$",
           [0, 1], ymax_aLne)
    _legend_lines(axaLn)

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------
    xi_note = "" if abs(xi - 1.0) < 1e-3 else r",  gray = $\xi=1$ ref"
    fig.suptitle(
        rf"BC beat (sharpness)  |  $\xi={xi:.2f}$,  $\psi_{{n,bc}}={psin_bc_g:.3f}$"
        rf"  ($\rho_N={rho_bc_rho:.3f}$),  $T_{{e,bc}}={Te_bc:.3f}$ keV{xi_note}",
        fontsize=FS,
    )


def _plot_bc_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter,
                             label="BC"):
    """
    Full Te, Ti, ne profiles (rows) plotted against each of the three coordinate
    systems rho_tor, r/a, psi_n (columns), before (blue) and after (red) the
    boundary condition. The BC location is marked by a vertical dashed line in
    whichever coordinate each column uses. `label` names the figure tab (shared
    by all bc methods, which apply the same BC machinery).
    """

    FS, FS_tick, FS_leg = 13, 11, 10
    cb, ca = "royalblue", "crimson"
    lw = 1.8

    rho_bc_rho = loaded_results["rho_bc_rho"]
    psin_bc    = loaded_results["psin_bc"]

    # Coordinate arrays for each state (geometry is unchanged by the BC, so before/after
    # share roa/psin, but read each from its own state to be safe).
    def _coords(p):
        return {"rho": p.profiles["rho(-)"],
                "roa": p.derived["roa"],
                "psin": p.derived["psi_pol_n"]}

    # Profile values for each state (ne in 10^20 m^-3 to match the other bc tab).
    def _vals(p):
        return {"Te": p.profiles["te(keV)"],
                "Ti": p.profiles["ti(keV)"][:, 0],
                "ne": p.profiles["ne(10^19/m^3)"] * 0.1}

    cb_x, ca_x = _coords(profiles_before), _coords(profiles_after)
    vb, va     = _vals(profiles_before),   _vals(profiles_after)

    # BC location in each coordinate (rho from results; roa interpolated; psin from results)
    roa_bc = float(np.interp(rho_bc_rho, ca_x["rho"], ca_x["roa"]))
    bc_loc = {"rho": rho_bc_rho, "roa": roa_bc, "psin": psin_bc}

    rows = [("Te", r"$T_e$ (keV)"),
            ("Ti", r"$T_i$ (keV)"),
            ("ne", r"$n_e$ ($10^{20}$ m$^{-3}$)")]
    cols = [("rho",  r"$\rho_{tor}$"),
            ("roa",  r"$r/a$"),
            ("psin", r"$\psi_n$")]

    fig = fn.add_figure(label=f"{label} - Profiles", tab_color=counter)
    gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    col_top_ax = {}   # top axis of each column, so the panels below share its x-axis
    for ir, (rk, rlab) in enumerate(rows):
        for ic, (ck, clab) in enumerate(cols):
            # Link x within a column: zoom/pan on any panel drives the whole column.
            ax = fig.add_subplot(gs[ir, ic], sharex=col_top_ax.get(ic))
            if ir == 0:
                col_top_ax[ic] = ax
            # markers expose the actual grid points underneath the lines
            ax.plot(cb_x[ck], vb[rk], '-o', color=cb, lw=lw, ms=4.0, mew=0, label="before")
            ax.plot(ca_x[ck], va[rk], '-o', color=ca, lw=lw, ms=4.0, mew=0, label="after")
            ax.axvline(bc_loc[ck], color="k", ls="--", lw=1.0, label="BC")
            ax.set_ylabel(rlab, fontsize=FS)
            ax.set_xlim([0, 1])
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=FS_tick)
            GRAPHICStools.addDenseAxis(ax)
            if ir == 0:                       # column header = coordinate of this column
                ax.set_title(clab, fontsize=FS)
            if ir == len(rows) - 1:           # x-axis label only on the bottom row
                ax.set_xlabel(clab, fontsize=FS)
            if ir == 0 and ic == 0:
                ax.legend(prop={"size": FS_leg}, loc="best")

    fig.suptitle(
        rf"{label} profiles  |  BC at $\rho_{{tor}}={bc_loc['rho']:.3f}$, "
        rf"$r/a={bc_loc['roa']:.3f}$, $\psi_n={bc_loc['psin']:.3f}$",
        fontsize=FS,
    )


# ============================================================================
# Plotting helpers — method 'confinement'
# ============================================================================


def _h_factor_params_text(loaded_results, profiles_before, profiles_after):
    """
    Build the text block listing the inputs to the tau-scaling / H-factor calculation:
    fixed engineering parameters, and the variable quantities initial -> final.
    Values are read from loaded_results (what the scan actually used); for results
    files predating these keys, they are recomputed from the before/after states
    (equivalent except for the alpha_power_feedback baseline, which old files
    never used anyway).
    """
    scaling  = loaded_results["confinement_scaling"]
    _, tau_key = _SCALING_MAP[scaling]

    if "scaling_params" in loaded_results:
        sp = loaded_results["scaling_params"]
        var = {
            "ne":   (loaded_results["ne_vol20_initial"], loaded_results["ne_vol20_achieved"]),
            "Wthr": (loaded_results["Wthr_initial_MJ"],  loaded_results["Wthr_achieved_MJ"]),
            "Ptot": (loaded_results["qHeat_initial_MW"], loaded_results["qHeat_achieved_MW"]),
            "tauE": (loaded_results["tauE_initial"],     loaded_results["tauE"]),
            "tauS": (loaded_results["tau_scaling_initial"], loaded_results["tau_scaling"]),
        }
    else:
        db, da = profiles_before.derived, profiles_after.derived
        sp = {
            "Ip_MA":   abs(float(profiles_before.profiles["current(MA)"][-1])),
            "Rgeo_m":  float(db["Rgeo"]),
            "epsilon": float(db["a"] / db["Rgeo"]),
            "kappa_a": float(db["kappa_a"]),
            "B0_T":    float(db["B0"]),
            "mbg_amu": float(db["mbg_main"]),
        }
        var = {
            "ne":   (float(db["ne_vol20"]), float(da["ne_vol20"])),
            "Wthr": (float(db["Wthr"]),     float(da["Wthr"])),
            "Ptot": (float(db["qHeat"]),    float(da["qHeat"])),
            "tauE": (float(db["tauE"]),     float(da["tauE"])),
            "tauS": (float(db[tau_key]),    float(da[tau_key])),
        }

    tau_lab = {"tau98y2": r"$\tau_{98y2}$", "tau89p": r"$\tau_{89p}$"}.get(tau_key, tau_key)

    lines = [
        f"{scaling} inputs",
        "",
        "Fixed (engineering):",
        rf"  $I_p$        = {sp['Ip_MA']:.2f} MA",
        rf"  $R_{{geo}}$   = {sp['Rgeo_m']:.2f} m",
        rf"  $\epsilon$        = {sp['epsilon']:.3f}",
        rf"  $\kappa_a$       = {sp['kappa_a']:.2f}",
        rf"  $B_0$       = {sp['B0_T']:.2f} T",
        rf"  $M_{{main}}$  = {sp['mbg_amu']:.2f} amu",
        "",
        "Variable (initial $\\to$ final):",
        rf"  $\langle n_e\rangle_{{vol}}$ = {var['ne'][0]:.2f} $\to$ {var['ne'][1]:.2f} $10^{{20}}m^{{-3}}$",
        rf"  $W_{{thr}}$  = {var['Wthr'][0]:.2f} $\to$ {var['Wthr'][1]:.2f} MJ",
        rf"  $P_{{tot}}$  = {var['Ptot'][0]:.2f} $\to$ {var['Ptot'][1]:.2f} MW",
        rf"  $\tau_E$    = {var['tauE'][0]:.3f} $\to$ {var['tauE'][1]:.3f} s",
        rf"  {tau_lab} = {var['tauS'][0]:.3f} $\to$ {var['tauS'][1]:.3f} s",
        rf"  {scaling}  = {loaded_results['H_initial']:.3f} $\to$ {loaded_results['H_achieved']:.3f}",
    ]

    return "\n".join(lines)


def _plot_confinement_bc(fn, loaded_results, profiles_before, profiles_after, counter):
    """
    Main confinement figure (3 rows x 3 plot cols + info column).

    Row 0 — Te, Ti, ne profiles vs r/a, before (blue) and after (red), BC marked.
    Row 1 — a/L_Te, a/L_Ti, a/L_ne gradients vs rho_N, before/after, BC marked.
            The region beyond the BC is drawn at 0.5 alpha and excluded from
            the y-limits (the analytical-edge gradients are much larger and
            would otherwise hide the core structure).
    Row 2 — Optimization diagnostics:
        - H-factor vs evaluation number (target dashed)
        - Te_bc vs evaluation number
        - H-factor vs Te_bc trajectory (target crosshair, final point starred)
    Right column — inputs to the H-factor calculation: fixed engineering
    parameters and the variable quantities (stored energy, heating power, ...)
    from initial to final state.
    """

    FS      = 13    # axis-label / title font size
    FS_tick = 11    # tick label font size
    FS_leg  = 10    # legend font size
    MARGIN  = 0.01  # how far past the BC to evaluate the y-axis ceiling

    # ------------------------------------------------------------------
    # Unpack stored results
    # ------------------------------------------------------------------
    scaling    = loaded_results["confinement_scaling"]
    H_target   = loaded_results["H_target"]
    H_initial  = loaded_results["H_initial"]
    H_achieved = loaded_results["H_achieved"]
    Te_bc      = loaded_results["Te_bc"]
    rho_bc_rho = loaded_results["rho_bc_rho"]
    hist_Te    = loaded_results["history_Te_bc"]
    hist_H     = loaded_results["history_H"]

    # ------------------------------------------------------------------
    # Profile arrays
    # ------------------------------------------------------------------
    roa_b  = profiles_before.derived["roa"]
    Te_b   = profiles_before.profiles["te(keV)"]
    Ti_b   = profiles_before.profiles["ti(keV)"][:, 0]
    ne_b   = profiles_before.profiles["ne(10^19/m^3)"] * 0.1

    roa_a  = profiles_after.derived["roa"]
    Te_a   = profiles_after.profiles["te(keV)"]
    Ti_a   = profiles_after.profiles["ti(keV)"][:, 0]
    ne_a   = profiles_after.profiles["ne(10^19/m^3)"] * 0.1

    # Normalized inverse gradient scale lengths a/Lx (main ion for Ti, matching Ti[:,0])
    aLTe_b, aLTi_b, aLne_b = profiles_before.derived["aLTe"], profiles_before.derived["aLTi"][:, 0], profiles_before.derived["aLne"]
    aLTe_a, aLTi_a, aLne_a = profiles_after.derived["aLTe"],  profiles_after.derived["aLTi"][:, 0],  profiles_after.derived["aLne"]

    rho_b  = profiles_before.profiles["rho(-)"]
    rho_a  = profiles_after.profiles["rho(-)"]

    ibc    = int(np.argmin(np.abs(profiles_after.profiles["rho(-)"] - rho_bc_rho)))
    roa_bc = roa_a[ibc]

    # ------------------------------------------------------------------
    # Y-axis ceiling helper: max of all supplied arrays up to x_cut
    # ------------------------------------------------------------------
    def _ymax(x_cut, *pairs):
        vals = []
        for x, y in pairs:
            mask = x <= x_cut
            if mask.any():
                vals.append(float(np.nanmax(np.abs(y[mask]))))
        return max(vals) * 1.15 if vals else 1.0

    roa_cut = roa_bc + MARGIN

    ymax_Te = _ymax(roa_cut, (roa_b, Te_b), (roa_a, Te_a))
    ymax_Ti = _ymax(roa_cut, (roa_b, Ti_b), (roa_a, Ti_a))
    ymax_ne = _ymax(roa_cut, (roa_b, ne_b), (roa_a, ne_a))

    # Gradient panels: y-range from the core region ONLY (strictly up to the BC,
    # no margin) — the analytical-edge gradients beyond it are much larger
    ymax_gTe = _ymax(rho_bc_rho, (rho_b, aLTe_b), (rho_a, aLTe_a))
    ymax_gTi = _ymax(rho_bc_rho, (rho_b, aLTi_b), (rho_a, aLTi_a))
    ymax_gne = _ymax(rho_bc_rho, (rho_b, aLne_b), (rho_a, aLne_a))

    # ------------------------------------------------------------------
    # Style constants
    # ------------------------------------------------------------------
    cb, ca, ct = "royalblue", "crimson", "gray"
    lw = 1.8

    def _style(ax, xlabel, ylabel, title, xlim=None, ylim_top=None):
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_title(title,   fontsize=FS)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim_top is not None:
            ax.set_ylim(0, ylim_top)
        GRAPHICStools.addDenseAxis(ax)
        ax.tick_params(labelsize=FS_tick)

    def _vbc(ax):
        ax.axvline(roa_bc, color="k", ls=":", lw=1.0, zorder=0)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = fn.add_figure(label="Confinement", tab_color=counter)
    gs  = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.40, width_ratios=[1, 1, 1, 0.55])
    axTe = fig.add_subplot(gs[0, 0])
    axTi = fig.add_subplot(gs[0, 1])
    axne = fig.add_subplot(gs[0, 2])
    axgTe = fig.add_subplot(gs[1, 0])
    axgTi = fig.add_subplot(gs[1, 1])
    axgne = fig.add_subplot(gs[1, 2])
    axHe = fig.add_subplot(gs[2, 0])
    axTb = fig.add_subplot(gs[2, 1])
    axHT = fig.add_subplot(gs[2, 2])
    axIn = fig.add_subplot(gs[:, 3])
    axIn.set_axis_off()

    # =====================================================================
    # ROW 0 — profiles vs r/a
    # =====================================================================

    axTe.plot(roa_b, Te_b, color=cb, lw=lw, ls="-", label="before")
    axTe.plot(roa_a, Te_a, color=ca, lw=lw, ls="-", label="after")
    _vbc(axTe)
    _style(axTe, r"$r/a$", r"$T_e$ (keV)", r"$T_e$ profile", [0, 1], ymax_Te)
    axTe.legend(prop={"size": FS_leg}, loc="upper right")

    axTi.plot(roa_b, Ti_b, color=cb, lw=lw, ls="-", label="before")
    axTi.plot(roa_a, Ti_a, color=ca, lw=lw, ls="-", label="after")
    _vbc(axTi)
    _style(axTi, r"$r/a$", r"$T_i$ (keV)", r"$T_i$ profile", [0, 1], ymax_Ti)
    axTi.legend(prop={"size": FS_leg}, loc="upper right")

    axne.plot(roa_b, ne_b, color=cb, lw=lw, ls="-", label="before")
    axne.plot(roa_a, ne_a, color=ca, lw=lw, ls="-", label="after")
    _vbc(axne)
    _style(axne, r"$r/a$", r"$n_e$ ($10^{20}$ m$^{-3}$)", r"$n_e$ profile", [0, 1], ymax_ne)
    axne.legend(prop={"size": FS_leg}, loc="upper right")

    # =====================================================================
    # ROW 1 — normalized inverse gradient scale lengths a/Lx vs rho_N
    # =====================================================================
    # Markers expose the grid points: the analytical edge starts one grid point
    # past the BC, so without markers that single cell renders as a jump "at"
    # the BC line even though the value AT the BC point is exactly preserved.
    # The edge region (beyond the BC point) is drawn at 0.5 alpha and excluded
    # from the y-limits, so the panels focus on the core structure.

    def _plot_grad(ax, y_b, y_a, ylab, ymax, legend=False):
        ax.plot(rho_b[:ibc + 1], y_b[:ibc + 1], "-o", color=cb, lw=lw, ms=3.5, mew=0, label="before")
        ax.plot(rho_a[:ibc + 1], y_a[:ibc + 1], "-o", color=ca, lw=lw, ms=3.5, mew=0, label="after")
        ax.plot(rho_b[ibc:], y_b[ibc:], "-o", color=cb, lw=lw, ms=3.5, mew=0, alpha=0.5)
        ax.plot(rho_a[ibc:], y_a[ibc:], "-o", color=ca, lw=lw, ms=3.5, mew=0, alpha=0.5)
        ax.axvline(rho_bc_rho, color="k", ls=":", lw=1.0, zorder=0)
        _style(ax, r"$\rho_N$", ylab, ylab + " profile", [0, 1], ymax)
        if legend:
            ax.legend(prop={"size": FS_leg}, loc="upper left")

    _plot_grad(axgTe, aLTe_b, aLTe_a, r"$a/L_{T_e}$", ymax_gTe, legend=True)
    _plot_grad(axgTi, aLTi_b, aLTi_a, r"$a/L_{T_i}$", ymax_gTi)
    _plot_grad(axgne, aLne_b, aLne_a, r"$a/L_{n_e}$", ymax_gne)

    # =====================================================================
    # ROW 2 — optimization diagnostics
    # =====================================================================

    evals = np.arange(1, len(hist_H) + 1)

    # --- H vs evaluation ---
    axHe.plot(evals, hist_H, "-o", color=ca, lw=1.2, ms=4.0, mew=0)
    axHe.axhline(H_target, color=ct, ls="--", lw=1.3, label=f"target = {H_target:.3f}")
    axHe.axhline(H_initial, color=cb, ls=":", lw=1.3, label=f"initial = {H_initial:.3f}")
    _style(axHe, "evaluation #", scaling, f"{scaling} convergence")
    axHe.legend(prop={"size": FS_leg}, loc="best")

    # --- Te_bc vs evaluation ---
    axTb.plot(evals, hist_Te, "-o", color=ca, lw=1.2, ms=4.0, mew=0)
    axTb.axhline(Te_bc, color="k", ls="--", lw=1.0, label=f"final = {Te_bc:.3f} keV")
    _style(axTb, "evaluation #", r"$T_{e,bc}$ (keV)", r"$T_{e,bc}$ trajectory")
    axTb.legend(prop={"size": FS_leg}, loc="best")

    # --- H vs Te_bc ---
    isort = np.argsort(hist_Te)
    axHT.plot(hist_Te[isort], hist_H[isort], "-o", color=ca, lw=1.0, ms=4.0, mew=0, alpha=0.7)
    axHT.axhline(H_target, color=ct, ls="--", lw=1.3)
    axHT.axvline(Te_bc, color="k", ls=":", lw=1.0)
    axHT.plot([Te_bc], [H_achieved], "*", color="k", ms=14, zorder=5,
              label=f"$T_{{e,bc}}$={Te_bc:.3f} keV")
    _style(axHT, r"$T_{e,bc}$ (keV)", scaling, f"{scaling} vs $T_{{e,bc}}$")
    axHT.legend(prop={"size": FS_leg}, loc="best")

    # =====================================================================
    # INFO COLUMN — inputs to the H-factor calculation
    # =====================================================================

    axIn.text(
        0.0, 1.0, _h_factor_params_text(loaded_results, profiles_before, profiles_after),
        transform=axIn.transAxes, ha="left", va="top",
        fontsize=10, linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="lightgray"),
    )

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------
    edge_shape = loaded_results.get("edge_shape", "linear")
    alpha_note = ""
    if loaded_results.get("alpha_power_feedback", False):
        alpha_note = (
            rf",  $\alpha$ feedback: $P_{{fus}}$ {loaded_results['Pfus_initial']:.1f}"
            rf"$\to${loaded_results['Pfus_achieved']:.1f} MW"
        )
    fig.suptitle(
        rf"BC beat (confinement)  |  {scaling}: {H_initial:.3f} $\to$ {H_achieved:.3f} "
        rf"(target {H_target:.3f}),  $T_{{e,bc}}={Te_bc:.3f}$ keV at $\rho_N={rho_bc_rho:.3f}$,  "
        rf"edge: {edge_shape}{alpha_note}",
        fontsize=FS,
    )


# ============================================================================
# Plotting helpers — method 'betap'
# ============================================================================


def _plot_betap_bc(fn, loaded_results, profiles_before, profiles_after, counter):
    """
    Betap figure (2 rows x 3 plot cols + info column).

    Row 0 — Te, Ti, ne profiles vs psi_n, before (blue) and after (red), BC marked.
    Row 1 — thermal pressure p_th vs psi_n; beta_p vs psi_n with the BC->separatrix
            secant chords whose slopes ARE the two-point betap' (delivered before,
            applied after); and -d(beta_p)/dpsin over the edge neighborhood with the
            delivered/applied betap' levels (the "after" curve sits flat ON the target
            level across the edge with the pressure-linear edge).
    Right column — the normalization and inversion quantities (Ip, L_pol, Bpa,
    p_sep/p_bc, ne_bc used, f_i) and target/delivered/applied betap'.
    """

    FS, FS_tick, FS_leg = 13, 11, 10
    MARGIN = 0.01

    betap_target    = loaded_results["betap_prime"]
    betap_eff       = loaded_results["betap_prime_eff"]
    betap_delivered = loaded_results["betap_prime_delivered"]
    Te_bc      = loaded_results["Te_bc"]
    rho_bc_rho = loaded_results["rho_bc_rho"]
    psin_g     = loaded_results["psin_bc_grid"]
    Bpa        = loaded_results["Bpa_T"]
    mu0        = 4.0e-7 * np.pi
    coef       = Bpa**2 / (2.0 * mu0)   # [Pa]

    psin_b = profiles_before.derived["psi_pol_n"]
    psin_a = profiles_after.derived["psi_pol_n"]
    Te_b, Te_a = profiles_before.profiles["te(keV)"], profiles_after.profiles["te(keV)"]
    Ti_b, Ti_a = profiles_before.profiles["ti(keV)"][:, 0], profiles_after.profiles["ti(keV)"][:, 0]
    ne_b = profiles_before.profiles["ne(10^19/m^3)"] * 0.1
    ne_a = profiles_after.profiles["ne(10^19/m^3)"] * 0.1

    # Thermal pressure and beta_p of each state (same formula the beat used)
    p_b = _thermal_pressure_Pa(profiles_before)
    p_a = _thermal_pressure_Pa(profiles_after)
    bp_b, bp_a = p_b / coef, p_a / coef

    ibc = int(np.argmin(np.abs(profiles_after.profiles["rho(-)"] - rho_bc_rho)))
    psin_bc_g = psin_a[ibc]

    def _ymax(x_cut, *pairs):
        vals = []
        for x, y in pairs:
            mask = x <= x_cut
            if mask.any():
                vals.append(float(np.nanmax(np.abs(y[mask]))))
        return max(vals) * 1.15 if vals else 1.0

    psin_cut = psin_bc_g + MARGIN
    cb, ca = "royalblue", "crimson"
    lw = 1.8

    def _style(ax, xlabel, ylabel, title, ylim_top=None):
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_title(title, fontsize=FS)
        ax.set_xlim([0, 1])
        if ylim_top is not None:
            ax.set_ylim(0, ylim_top)
        GRAPHICStools.addDenseAxis(ax)
        ax.tick_params(labelsize=FS_tick)

    def _vbc(ax):
        ax.axvline(psin_bc_g, color="k", ls=":", lw=1.0, zorder=0)

    fig = fn.add_figure(label="Betap", tab_color=counter)
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.45, width_ratios=[1, 1, 1, 0.6])
    axTe = fig.add_subplot(gs[0, 0])
    axTi = fig.add_subplot(gs[0, 1])
    axne = fig.add_subplot(gs[0, 2])
    axp  = fig.add_subplot(gs[1, 0])
    axbp = fig.add_subplot(gs[1, 1])
    axdbp = fig.add_subplot(gs[1, 2])
    axIn = fig.add_subplot(gs[:, 3])
    axIn.set_axis_off()

    for ax, yb, ya, lab in ((axTe, Te_b, Te_a, r"$T_e$ (keV)"),
                            (axTi, Ti_b, Ti_a, r"$T_i$ (keV)"),
                            (axne, ne_b, ne_a, r"$n_e$ ($10^{20}$ m$^{-3}$)")):
        ax.plot(psin_b, yb, color=cb, lw=lw, label="before")
        ax.plot(psin_a, ya, color=ca, lw=lw, label="after")
        _vbc(ax)
        _style(ax, r"$\psi_n$", lab, lab.split(" ")[0] + " profile",
               _ymax(psin_cut, (psin_b, yb), (psin_a, ya)))
        ax.legend(prop={"size": FS_leg}, loc="upper right")

    axp.plot(psin_b, p_b * 1e-3, color=cb, lw=lw, label="before")
    axp.plot(psin_a, p_a * 1e-3, color=ca, lw=lw, label="after")
    _vbc(axp)
    _style(axp, r"$\psi_n$", r"$p_{th}$ (kPa)", "Thermal pressure",
           _ymax(psin_cut, (psin_b, p_b * 1e-3), (psin_a, p_a * 1e-3)))
    axp.legend(prop={"size": FS_leg}, loc="upper right")

    # beta_p with the BC->separatrix secant chords (slope magnitude = two-point betap')
    axbp.plot(psin_b, bp_b, color=cb, lw=lw, label="before")
    axbp.plot(psin_a, bp_a, color=ca, lw=lw, label="after")
    axbp.plot([psin_g, 1.0], [bp_b[ibc], bp_b[-1]], color=cb, ls="--", lw=1.2,
              label=rf"delivered $\beta_p'$={betap_delivered:.2f}")
    axbp.plot([psin_g, 1.0], [bp_a[ibc], bp_a[-1]], color=ca, ls="--", lw=1.2,
              label=rf"applied $\beta_p'$={betap_eff:.2f}")
    _vbc(axbp)
    _style(axbp, r"$\psi_n$", r"$\beta_p$",
           r"$\beta_p=2\mu_0 p_{th}/B_{pa}^2$,  $B_{pa}=\mu_0 I_p/L_{pol}$",
           _ymax(psin_cut, (psin_b, bp_b), (psin_a, bp_a)))
    axbp.title.set_fontsize(FS - 2)   # full convention on one line, slightly smaller
    axbp.legend(prop={"size": FS_leg}, loc="upper right")

    # The controlled quantity itself: -d(beta_p)/dpsin. With the pressure-linear edge
    # the "after" curve sits flat ON the target level across the whole edge; the dashed
    # levels are drawn over the edge only, so they read as the secant values. X-range is
    # restricted to the edge neighborhood (the core derivative is not what this beat
    # controls) and the y-scale is set by the edge levels, not core/separatrix spikes.
    dbp_b = -np.gradient(bp_b, psin_b)
    dbp_a = -np.gradient(bp_a, psin_a)
    x0_dbp = max(0.7, psin_g - 0.15)
    axdbp.plot(psin_b, dbp_b, color=cb, lw=lw, label="before")
    axdbp.plot(psin_a, dbp_a, color=ca, lw=lw, label="after")
    axdbp.plot([psin_g, 1.0], [betap_delivered, betap_delivered], color=cb, ls="--", lw=1.2,
               label=rf"delivered $\beta_p'$={betap_delivered:.2f}")
    axdbp.plot([psin_g, 1.0], [betap_eff, betap_eff], color=ca, ls="--", lw=1.2,
               label=rf"applied $\beta_p'$={betap_eff:.2f}")
    _vbc(axdbp)
    axdbp.set_xlabel(r"$\psi_n$", fontsize=FS)
    axdbp.set_ylabel(r"$-d\beta_p/d\psi_n$", fontsize=FS)
    axdbp.set_title("Edge $\\beta_p$ gradient", fontsize=FS)
    axdbp.set_xlim([x0_dbp, 1.0])
    ymax_levels = max(abs(betap_eff), abs(betap_delivered))
    ymax_dbp = min(
        1.15 * max(float(np.nanmax(np.abs(dbp_b[psin_b >= x0_dbp]))),
                   float(np.nanmax(np.abs(dbp_a[psin_a >= x0_dbp]))), ymax_levels),
        3.0 * ymax_levels,   # clip: don't let core/separatrix spikes hide the edge levels
    )
    axdbp.set_ylim(0, ymax_dbp)
    GRAPHICStools.addDenseAxis(axdbp)
    axdbp.tick_params(labelsize=FS_tick)
    axdbp.legend(prop={"size": FS_leg}, loc="upper left")

    axIn.text(
        0.0, 1.0,
        "\n".join([
            "betap inversion",
            "",
            rf"  $I_p$    = {loaded_results['Ip_MA']:.2f} MA",
            rf"  $L_{{pol}}$ = {loaded_results['L_pol_m']:.2f} m",
            rf"  $B_{{pa}}=\mu_0 I_p/L_{{pol}}$ = {Bpa:.3f} T",
            rf"  $p_{{sep}}$ = {loaded_results['p_sep_Pa']*1e-3:.2f} kPa",
            rf"  $p_{{bc}}$  = {loaded_results['p_bc_applied_Pa']*1e-3:.2f} kPa",
            rf"  $n_{{e,bc}}$ = {loaded_results['ne_bc_used_1e19']*0.1:.2f} $10^{{20}}m^{{-3}}$",
            rf"  $f_i$   = {loaded_results['f_i_bc']:.3f}",
            "",
            rf"  $\beta_p'$ target    = {betap_target:.3f}",
            rf"  $\beta_p'$ delivered = {betap_delivered:.3f}",
            rf"  $\beta_p'$ applied   = {betap_eff:.3f}",
        ]),
        transform=axIn.transAxes, ha="left", va="top", fontsize=10, linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="lightgray"),
    )

    fig.suptitle(
        rf"BC beat (betap)  |  $\beta_p'={betap_target:.2f}$,  $\psi_{{n,bc}}={psin_bc_g:.3f}$"
        rf"  ($\rho_N={rho_bc_rho:.3f}$),  $T_{{e,bc}}={Te_bc:.3f}$ keV",
        fontsize=FS,
    )
