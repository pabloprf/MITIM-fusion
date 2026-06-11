import copy
import shutil
import numpy as np
from scipy.optimize import minimize
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from mitim_modules.maestro.utils.SHARPNESSbeat import (
    _convert_bc_location,
    _apply_sharpness_bc,
    _plot_sharpness_profiles_coords,
)
from IPython import embed

# Mapping from the namelist scaling name to the (H-factor, tau-scaling) keys in profiles.derived
_SCALING_MAP = {
    "H98y2": ("H98", "tau98y2"),   # IPB98(y,2) thermal energy confinement scaling
    "H89p":  ("H89", "tau89p"),    # ITER89-P L-mode scaling
}


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
    from mitim_modules.powertorch import STATEtools
    from mitim_modules.powertorch.physics_models import targets_analytic

    extra_points = 2   # same grid trick as powerstate_to_gacode_powers
    rhoy = profiles.profiles["rho(-)"][1:-extra_points]

    with LOGtools.HiddenPrints():
        state = STATEtools.powerstate(
            profiles,
            evolution_options={"rhoPredicted": rhoy},
            target_options={
                "evaluator": targets_analytic.analytical_model,
                "options": {"targets_evolve": ["qfus"], "target_evaluator_method": "powerstate"},
            },
            transport_options={"evaluator": None, "options": {}},
            increase_profile_resol=False,
        )
        state.calculateProfileFunctions()
        state.calculateTargets()

    for key, gkey in (("qfuse", "qfuse(MW/m^3)"), ("qfusi", "qfusi(MW/m^3)")):
        profiles.profiles[gkey][:-extra_points] = state.plasma[key][0, :].cpu().numpy()

    profiles.derive_quantities(rederiveGeometry=False)

    return profiles


class confinement_beat(beat):
    """
    Confinement beat: sets the temperature boundary condition at rho_bc such that
    the plasma state matches a prescribed confinement level (H-factor).

    The H-factor cannot be inverted analytically for T_bc (it depends on the
    volume-integrated thermal stored energy of the full modified profiles), so
    T_bc is found by minimization, in the same spirit as the eped_initializer
    matching of BetaN through a/LT:

        find Te_bc such that  ((H - H_target)/H_target)^2  is minimized

    where at each trial Te_bc the boundary condition is applied with the same
    profile machinery as the sharpness beat (core scaled preserving a/LT shape,
    edge from T_bc to T_sep with the shape selected by edge_shape: linear in
    psi_n, or the initializer's pedestal tanh in r/a) and the H-factor is
    re-derived.

    NOTE (assumption): by default, the auxiliary, fusion and radiation source
    profiles stored in input.gacode are NOT recomputed during the Te_bc scan, so
    the total heating power entering both tauE and the scaling law stays frozen;
    the H-factor responds through the thermal stored energy Wthr. Source
    self-consistency is recovered by the subsequent PORTALS/TRANSP beat in the
    MAESTRO chain. With alpha_power_feedback=True, qfuse/qfusi are recomputed
    analytically (powerstate targets) at every trial Te_bc and in the final
    output state, so the H-factor accounts for the alpha-heating response —
    relevant for burning plasmas, where freezing the sources biases H high and
    underestimates the Te_bc needed for a given target.

    neped_20 (a MAESTRO trans-beat parameter) is reinterpreted here as ne_bc,
    i.e. the electron density at the boundary condition location rho_bc
    (same convention as the sharpness beat).
    """

    def __init__(self, maestro_instance, folder_name=None):
        super().__init__(maestro_instance, beat_name="confinement", folder_name=folder_name)

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        x_bc=0.90,
        bc_coordinate="rho",            # coordinate for x_bc: 'rho', 'roa', or 'psin'
        confinement_scaling="H98y2",    # which H-factor to match: 'H98y2' or 'H89p'
        confinement=1.0,                # target H-factor value
        tite=1.0,
        edge_shape="linear",            # edge profile shape outside rho_bc: 'linear' or 'tanh'
        density_treatment="bc",         # 'bc': rescale ne to ne_bc (current behavior); 'keep': leave ne/ni untouched
        alpha_power_feedback=False,     # recompute qfuse/qfusi at each trial Te_bc (alpha-heating response)
        Te_bc_bounds=(0.05, 10.0),      # bounds on Te_bc (keV) for the minimization
        update_bc_based_on_portals=False,  # if True, override x_bc with last PORTALS prediction radius
        **kwargs,
    ):
        """
        Parameters
        ----------
        x_bc : float
            Location of the boundary condition in the coordinate given by
            bc_coordinate (default 0.90).
        bc_coordinate : str
            Coordinate system for x_bc: 'rho' (rho_tor, default), 'roa' (r/a),
            or 'psin' (normalized poloidal flux).
        confinement_scaling : str
            Confinement scaling law whose H-factor is matched: 'H98y2'
            (IPB98(y,2), default) or 'H89p' (ITER89-P).
        confinement : float
            Target H-factor value (default 1.0).
        tite : float
            Ti / Te ratio at the boundary condition (default 1.0).
        edge_shape : str
            Shape of the Te/Ti/ne profiles in the edge region (rho > rho_bc):
            'linear' (default) interpolates linearly in psi_n from the BC value
            to the separatrix value; 'tanh' uses the pedestal tanh functional
            form of the eped_initializer (FunctionalForms.pedestal_tanh in r/a),
            also anchored at the BC and separatrix values.
        density_treatment : str
            'bc' (default): core ne rescaled to ne_bc (= neped_20) preserving
            a/Lne, edge replaced, ion densities rescaled to keep ni/ne ratios.
            'keep': ne and all ion densities left untouched; only Te/Ti are
            modified — the H-factor then responds to the BC purely through the
            temperatures (note H98 carries <ne>^0.41, so 'bc' lets density move
            H independently of Te_bc). The neped_20 passed to subsequent beats
            is then the actual ne at rho_bc read from the profiles.
        alpha_power_feedback : bool
            If True, recompute the fusion source profiles (qfuse/qfusi) from the
            trial profiles at every minimization step (analytic powerstate
            targets), so the H-factor sees the alpha-power response to the BC
            change. The recomputed sources are also written into the beat output
            state. The baseline H is recomputed with the same model for a
            consistent comparison. Default False (sources frozen during scan).
        Te_bc_bounds : (float, float)
            Bounds on Te_bc in keV during the minimization (default (0.05, 10.0)).
        update_bc_based_on_portals : bool
            If True, override x_bc with the outermost radial location used by
            the previous PORTALS beat (stored in parameters_trans_beat as
            predicted_rho[-1] or predicted_roa[-1]).  bc_coordinate is updated
            automatically.  Default False.
        """

        if bc_coordinate not in ("rho", "roa", "psin"):
            raise ValueError(
                f"bc_coordinate must be 'rho', 'roa', or 'psin', got '{bc_coordinate}'"
            )
        if confinement_scaling not in _SCALING_MAP:
            raise ValueError(
                f"confinement_scaling must be one of {list(_SCALING_MAP.keys())}, got '{confinement_scaling}'"
            )
        if edge_shape not in ("linear", "tanh"):
            raise ValueError(
                f"edge_shape must be 'linear' or 'tanh', got '{edge_shape}'"
            )
        if density_treatment not in ("bc", "keep"):
            raise ValueError(
                f"density_treatment must be 'bc' or 'keep', got '{density_treatment}'"
            )

        self.x_bc = x_bc
        self.bc_coordinate = bc_coordinate
        self.confinement_scaling = confinement_scaling
        self.confinement = confinement
        self.tite = tite
        self.edge_shape = edge_shape
        self.density_treatment = density_treatment
        self.alpha_power_feedback = alpha_power_feedback
        self.Te_bc_bounds = tuple(Te_bc_bounds)
        self.update_bc_based_on_portals = update_bc_based_on_portals
        self._portals_rho_bc = None   # (value, coordinate) set by _inform() if update_bc_based_on_portals
        self.neped_20 = None   # resolved in _inform() from plasma/parameters or previous beat

        print(
            f"\t- Confinement beat: x_bc={x_bc} ({bc_coordinate}), "
            f"target {confinement_scaling}={confinement}, Ti/Te={tite}, edge_shape={edge_shape}, "
            f"density_treatment={density_treatment}, alpha_power_feedback={alpha_power_feedback}",
            typeMsg="i",
        )

        self._inform()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, **kwargs):

        # Copy current input.gacode to working folder
        shutil.copy2(self.initialize.folder / "input.gacode", self.folder / "input.gacode")

        # ------------------------------------------------------------------
        # Find Te_bc that matches the target H-factor and apply to profiles
        # ------------------------------------------------------------------

        confinement_results = self._run()

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------

        np.save(self.folder / "confinement_results.npy", confinement_results)

        self.rho_bc_rho = confinement_results["rho_bc_rho"]   # store for _inform_save

    # ------------------------------------------------------------------
    # _run (core physics)
    # ------------------------------------------------------------------

    def _run(self):

        profiles = copy.deepcopy(self.profiles_current)
        profiles.derive_quantities(rederiveGeometry=False)

        # With alpha feedback on, make the baseline sources consistent with the same
        # analytic model used during the scan, so H_initial and the trial H values
        # are directly comparable (stored qfus may come from a different model)
        if self.alpha_power_feedback:
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

        H_key, tau_key = _SCALING_MAP[self.confinement_scaling]

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
        # 3. Minimize over Te_bc to match the target H-factor
        #    (same spirit as the eped_initializer a/LT matching of BetaN)
        # ------------------------------------------------------------------

        H_initial = float(profiles.derived[H_key])
        Te_bc_guess = float(np.interp(rho_bc_rho, rho, Te))

        print(
            f"\t- Optimizing Te_bc to match {self.confinement_scaling} = {self.confinement:.3f} "
            f"(initial {self.confinement_scaling} = {H_initial:.3f}, "
            f"Te_bc guess = {Te_bc_guess:.4f} keV, bounds = {self.Te_bc_bounds} keV)"
        )

        history = []   # (Te_bc, H, residual) per evaluation

        def _residual(Te_bc_arr):
            Te_bc_trial = float(Te_bc_arr[0])
            p_mod = _apply_sharpness_bc(
                profiles, rho_bc_rho, psin_bc,
                Te_bc_trial, Te_bc_trial * self.tite, ne_bc_1e19,
                edge_shape=self.edge_shape,
                density_treatment=self.density_treatment,
            )
            if self.alpha_power_feedback:
                p_mod = _recompute_alpha_power(p_mod)
            H_trial = float(p_mod.derived[H_key])
            res = ((H_trial - self.confinement) / self.confinement) ** 2
            history.append((Te_bc_trial, H_trial, res))
            return res

        opt = minimize(
            _residual,
            [Te_bc_guess],
            method="Nelder-Mead",
            tol=1e-4,
            bounds=[self.Te_bc_bounds],
        )
        Te_bc = float(opt.x[0])
        Ti_bc = Te_bc * self.tite

        # ------------------------------------------------------------------
        # 4. Apply the optimal boundary condition
        # ------------------------------------------------------------------

        profiles_out = _apply_sharpness_bc(
            profiles, rho_bc_rho, psin_bc, Te_bc, Ti_bc, ne_bc_1e19,
            edge_shape=self.edge_shape,
            density_treatment=self.density_treatment,
        )
        if self.alpha_power_feedback:
            # Recomputed sources also travel in the beat output state, so the next
            # beat receives a power balance consistent with the H-factor reported here
            profiles_out = _recompute_alpha_power(profiles_out)

        H_achieved   = float(profiles_out.derived[H_key])
        tauE         = float(profiles_out.derived["tauE"])
        tau_scaling  = float(profiles_out.derived[tau_key])

        mismatch = abs(H_achieved - self.confinement) / self.confinement
        print(
            f"\t- Optimization finished after {len(history)} evaluations: "
            f"Te_bc = {Te_bc:.4f} keV, Ti_bc = {Ti_bc:.4f} keV"
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
            print(
                f"\t- WARNING: H-factor mismatch of {mismatch*100:.1f}% remains after optimization "
                f"(target may be unreachable within Te_bc bounds {self.Te_bc_bounds} keV)",
                typeMsg="w",
            )

        # ------------------------------------------------------------------
        # 5. Store
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

        confinement_results = {
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
            "Ti_bc":               Ti_bc,
            "Te_bc_guess":         Te_bc_guess,
            "Te_bc_bounds":        self.Te_bc_bounds,
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

        for key, val in confinement_results.items():
            if not key.startswith("history_"):
                print(f"\t\t- {key}: {val}")

        # Write intermediate result
        profiles_out.write_state(file=self.folder / "input.gacode.confinement")

        self.profiles_output = profiles_out

        return confinement_results

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self, **kwargs):

        # On a re-invocation after a prior keep_all_files: false cleanup wiped
        # self.folder, the run artifacts are gone and folder_output already holds
        # confinement_results.npy + input.gacode from the prior run — do not wipe it
        # (the wipe-first flow would destroy the persisted results and then crash
        # on the missing copy source). Same guard as the TRANSP/EPED/PORTALS beats.
        if not (
            (self.folder / "confinement_results.npy").exists()
            and (self.folder / "input.gacode.confinement").exists()
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
            self.folder / "confinement_results.npy",
            self.folder_output / "confinement_results.npy",
        )

        # Write profiles to output folder
        self.profiles_output = PROFILEStools.gacode_state(
            self.folder / "input.gacode.confinement"
        )
        self.profiles_output.write_state(file=self.folder_output / "input.gacode")

    # ------------------------------------------------------------------
    # merge_parameters
    # ------------------------------------------------------------------

    def merge_parameters(self):
        # Confinement beat does not change the grid or engineering parameters,
        # so no special merging is required (same as EPED and sharpness beats).
        pass

    # ------------------------------------------------------------------
    # grab_output
    # ------------------------------------------------------------------

    def grab_output(self, **kwargs):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:
            loaded_results = np.load(
                self.folder_output / "confinement_results.npy", allow_pickle=True
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
            fn = GUItools.FigureNotebook("Confinement")

        loaded_results, profiles_after = self.grab_output()
        # "before" profiles = the input this beat received. self.folder/input.gacode is
        # wiped under keep_all_files: false; fall back to the initializer copy (the same
        # input.gacode the beat ran on), which survives the cleanup.
        before_file = self.folder / "input.gacode"
        if not before_file.exists():
            cands = sorted(self.folder_beat.glob("initializer_*/input.gacode"))
            if cands:
                before_file = cands[0]
        profiles_before = PROFILEStools.gacode_state(before_file)
        profiles_before.derive_quantities(rederiveGeometry=False)

        if loaded_results is not None and profiles_after is not None:
            profiles_after.derive_quantities(rederiveGeometry=False)
            _plot_confinement_beat(fn, loaded_results, profiles_before, profiles_after, counter)
            _plot_sharpness_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter,
                                            label="Confinement")
        else:
            # Fallback: nothing to show yet
            fig = fn.add_figure(label="Confinement", tab_color=counter)
            fig.add_subplot(111).text(0.5, 0.5, "No confinement results available",
                                      ha="center", va="center", transform=fig.transFigure)

        return "\t\t- Plotting of confinement beat done"

    # ------------------------------------------------------------------
    # _inform / _inform_save
    # ------------------------------------------------------------------

    def _inform(self):
        """Receive parameters from previous beats or from the plasma/parameters namelist."""

        # 0. Grab the last PORTALS prediction radius if requested
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

        # 1. neped_20 from a previous EPED or sharpness/confinement beat (highest priority)
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

    def _inform_save(self, confinement_output=None):
        """Save parameters for subsequent beats."""

        if confinement_output is None:
            confinement_output, _ = self.grab_output()

        if confinement_output is None:
            return

        # Keep neped_20 (= ne_bc) available to subsequent beats
        self.maestro_instance.parameters_trans_beat["neped_20"] = confinement_output[
            "neped_20"
        ]

        # rhotop is understood by PORTALS to set the last radial prediction point
        self.maestro_instance.parameters_trans_beat["rhotop"] = confinement_output[
            "rho_bc_rho"
        ]

        print(
            f"\t\t- neped_20={confinement_output['neped_20']:.3f} and "
            f"rhotop={confinement_output['rho_bc_rho']:.3f} saved for future beats"
        )


# ============================================================================
# Plotting helpers
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


def _plot_confinement_beat(fn, loaded_results, profiles_before, profiles_after, counter):
    """
    Main confinement beat figure (3 rows x 3 plot cols + info column).

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
        rf"Confinement beat  |  {scaling}: {H_initial:.3f} $\to$ {H_achieved:.3f} "
        rf"(target {H_target:.3f}),  $T_{{e,bc}}={Te_bc:.3f}$ keV at $\rho_N={rho_bc_rho:.3f}$,  "
        rf"edge: {edge_shape}{alpha_note}",
        fontsize=FS,
    )
