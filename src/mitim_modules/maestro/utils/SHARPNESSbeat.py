import copy
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from mitim_modules.powertorch.utils import CALCtools
from IPython import embed

# Cubic-spline extrapolation helper (same as used in EPEDbeat)
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function


class sharpness_beat(beat):
    """
    Sharpness beat: sets the temperature boundary condition at rho_bc based on
    the sharpness parameter xi defined in Rodriguez-Fernandez et al. (L-mode paper).

    xi = |dT/dpsi_n|_edge  /  |dT/dpsi_n|_core_at_bc

    where:
      - the edge gradient goes linearly in psi_n from T_bc to T_sep
      - the core gradient at rho_bc is taken from the current profiles (PORTALS output)

    Given xi, T_sep (from profiles), and the core gradient, T_bc is found via:

        C   = (1 - psin_bc) * aLT_bc * d(r/a)/d(psin)|_bc
        T_bc = T_sep / (1 - xi * C)

    neped_20 (a MAESTRO trans-beat parameter) is reinterpreted here as ne_bc,
    i.e. the electron density at the boundary condition location rho_bc.
    """

    def __init__(self, maestro_instance, folder_name=None):
        super().__init__(maestro_instance, beat_name="sharpness", folder_name=folder_name)

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        x_bc=0.90,
        bc_coordinate="rho",           # coordinate for x_bc: 'rho', 'roa', or 'psin'
        sharpness=1.0,
        sharpness_coordinate="psin",   # coordinate for xi derivative: 'rho', 'roa', or 'psin'
        tite=1.0,
        density_treatment="bc",        # 'bc': rescale ne to ne_bc (current behavior); 'keep': leave ne/ni untouched
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
        sharpness : float
            Prescribed sharpness parameter xi (default 1.0).
        sharpness_coordinate : str
            Coordinate system in which the sharpness parameter xi (the gradient
            ratio) is defined: 'rho' (rho_tor), 'roa' (r/a), or 'psin' (default).
            This is independent of bc_coordinate.
        tite : float
            Ti / Te ratio at the boundary condition (default 1.0).
        density_treatment : str
            'bc' (default): core ne rescaled to ne_bc (= neped_20) preserving
            a/Lne, edge replaced, ion densities rescaled to keep ni/ne ratios.
            'keep': ne and all ion densities left untouched; only Te/Ti are
            modified. The neped_20 passed to subsequent beats is then the
            actual ne at rho_bc read from the profiles.
        update_bc_based_on_portals : bool
            If True, override x_bc with the outermost radial location used by
            the previous PORTALS beat (stored in parameters_trans_beat as
            predicted_rho[-1] or predicted_roa[-1]).  bc_coordinate is updated
            automatically; sharpness_coordinate is never changed.  Default False.
        """

        if bc_coordinate not in ("rho", "roa", "psin"):
            raise ValueError(
                f"bc_coordinate must be 'rho', 'roa', or 'psin', got '{bc_coordinate}'"
            )
        if sharpness_coordinate not in ("rho", "roa", "psin"):
            raise ValueError(
                f"sharpness_coordinate must be 'rho', 'roa', or 'psin', got '{sharpness_coordinate}'"
            )
        if density_treatment not in ("bc", "keep"):
            raise ValueError(
                f"density_treatment must be 'bc' or 'keep', got '{density_treatment}'"
            )

        self.x_bc = x_bc
        self.bc_coordinate = bc_coordinate
        self.sharpness = sharpness
        self.sharpness_coordinate = sharpness_coordinate
        self.tite = tite
        self.density_treatment = density_treatment
        self.update_bc_based_on_portals = update_bc_based_on_portals
        self._portals_rho_bc = None   # (value, coordinate) set by _inform() if update_bc_based_on_portals
        self.neped_20 = None   # resolved in _inform() from plasma/parameters or previous beat

        print(
            f"\t- Sharpness beat: x_bc={x_bc} ({bc_coordinate}), sharpness_coord={sharpness_coordinate}, "
            f"xi={sharpness}, Ti/Te={tite}, density_treatment={density_treatment}",
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
        # Compute T_bc and apply to profiles
        # ------------------------------------------------------------------

        sharpness_results = self._run()

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------

        np.save(self.folder / "sharpness_results.npy", sharpness_results)

        self.rho_bc_rho = sharpness_results["rho_bc_rho"]   # store for _inform_save

    # ------------------------------------------------------------------
    # _run (core physics)
    # ------------------------------------------------------------------

    def _run(self):

        profiles = copy.deepcopy(self.profiles_current)
        profiles.derive_quantities(rederiveGeometry=False)

        rho        = profiles.profiles["rho(-)"]
        psi_pol_n  = profiles.derived["psi_pol_n"]
        roa        = profiles.derived["roa"]
        Te         = profiles.profiles["te(keV)"]
        Ti_main    = profiles.profiles["ti(keV)"][:, 0]
        ne         = profiles.profiles["ne(10^19/m^3)"]

        # ------------------------------------------------------------------
        # 1. Convert rho_bc to rho_tor
        # ------------------------------------------------------------------

        if self._portals_rho_bc is not None:
            # Location comes from the last PORTALS beat; sharpness_coordinate is unaffected
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
        # 3. Compute T_bc from sharpness formula (in the coordinate c selected
        #    by sharpness_coordinate: rho, roa or psin — all equal 1 at the
        #    separatrix)
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

        if C >= 1.0 / self.sharpness:
            print(
                f"\t- WARNING: sharpness formula denominator non-positive "
                f"(xi*C={self.sharpness*C:.3f} >= 1). Clamping C.",
                typeMsg="w",
            )
            # Clamp to avoid division by zero / negative T_bc
            C = 0.99 / max(self.sharpness, 1e-6)

        Te_bc = Te_sep / (1.0 - self.sharpness * C)
        Ti_bc = Te_bc * self.tite

        print(f"\t- Sharpness C={C:.4f}, xi={self.sharpness:.3f}")
        print(f"\t- T_sep={Te_sep:.4f} keV")
        print(f"\t- Te_bc={Te_bc:.4f} keV,  Ti_bc={Ti_bc:.4f} keV")

        # ------------------------------------------------------------------
        # 4. Modify profiles using the sharpness boundary condition
        # ------------------------------------------------------------------

        profiles_out = _apply_sharpness_bc(
            profiles,
            rho_bc_rho,
            psin_bc,
            Te_bc,
            Ti_bc,
            ne_bc_1e19,
            density_treatment=self.density_treatment,
        )

        # ------------------------------------------------------------------
        # 5. Store
        # ------------------------------------------------------------------

        sharpness_results = {
            "x_bc":            self.x_bc,
            "bc_coordinate":   self.bc_coordinate,
            "rho_bc_rho":      rho_bc_rho,
            "psin_bc":         psin_bc,
            "sharpness":       self.sharpness,
            "sharpness_coord": self.sharpness_coordinate,
            "C":              C,
            "Te_bc":          Te_bc,
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

        for key, val in sharpness_results.items():
            print(f"\t\t- {key}: {val}")

        # Write intermediate result
        profiles_out.write_state(file=self.folder / "input.gacode.sharpness")

        self.profiles_output = profiles_out

        return sharpness_results

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self, **kwargs):

        # On a re-invocation after a prior keep_all_files: false cleanup wiped
        # self.folder, the run artifacts are gone and folder_output already holds
        # sharpness_results.npy + input.gacode from the prior run — do not wipe it
        # (the wipe-first flow would destroy the persisted results and then crash
        # on the missing copy source). Same guard as the TRANSP/EPED/PORTALS beats.
        if not (
            (self.folder / "sharpness_results.npy").exists()
            and (self.folder / "input.gacode.sharpness").exists()
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
            self.folder / "sharpness_results.npy",
            self.folder_output / "sharpness_results.npy",
        )

        # Write profiles to output folder
        self.profiles_output = PROFILEStools.gacode_state(
            self.folder / "input.gacode.sharpness"
        )
        self.profiles_output.write_state(file=self.folder_output / "input.gacode")

    # ------------------------------------------------------------------
    # merge_parameters
    # ------------------------------------------------------------------

    def merge_parameters(self):
        # Sharpness beat does not change the grid or engineering parameters,
        # so no special merging is required (same as EPED beat).
        pass

    # ------------------------------------------------------------------
    # grab_output
    # ------------------------------------------------------------------

    def grab_output(self, **kwargs):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:
            loaded_results = np.load(
                self.folder_output / "sharpness_results.npy", allow_pickle=True
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
            fn = GUItools.FigureNotebook("Sharpness")

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
            _plot_sharpness_beat(fn, loaded_results, profiles_before, profiles_after, counter)
            _plot_sharpness_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter)
        else:
            # Fallback: nothing to show yet
            fig = fn.add_figure(label="Sharpness", tab_color=counter)
            fig.add_subplot(111).text(0.5, 0.5, "No sharpness results available",
                                      ha="center", va="center", transform=fig.transFigure)

        return "\t\t- Plotting of sharpness beat done"

    # ------------------------------------------------------------------
    # _inform / _inform_save
    # ------------------------------------------------------------------

    def _inform(self):
        """Receive parameters from previous beats or from the plasma/parameters namelist."""

        # 0. Grab the last PORTALS prediction radius if requested (stored separately;
        #    sharpness_coordinate is NOT changed — it governs the derivative, not the location)
        if self.update_bc_based_on_portals:
            tb = self.maestro_instance.parameters_trans_beat
            if "predicted_rho" in tb:
                self._portals_rho_bc = (float(tb["predicted_rho"][-1]), "rho")
                print(f"\t\t- update_bc_based_on_portals: BC location will use predicted_rho[-1] = {self._portals_rho_bc[0]:.4f} (rho)")
            elif "predicted_roa" in tb:
                self._portals_rho_bc = (float(tb["predicted_roa"][-1]), "roa")
                print(f"\t\t- update_bc_based_on_portals: BC location will use predicted_roa[-1] = {self._portals_rho_bc[0]:.4f} (roa)")
            else:
                print("\t\t- update_bc_based_on_portals=True but no predicted_rho/roa found in trans-beat parameters; keeping x_bc as specified", typeMsg="w")

        # 1. neped_20 from a previous EPED or sharpness beat (highest priority)
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

    def _inform_save(self, sharpness_output=None):
        """Save parameters for subsequent beats."""

        if sharpness_output is None:
            sharpness_output, _ = self.grab_output()

        if sharpness_output is None:
            return

        # Keep neped_20 (= ne_bc) available to subsequent beats
        self.maestro_instance.parameters_trans_beat["neped_20"] = sharpness_output[
            "neped_20"
        ]

        # rhotop is understood by PORTALS to set the last radial prediction point
        self.maestro_instance.parameters_trans_beat["rhotop"] = sharpness_output[
            "rho_bc_rho"
        ]

        print(
            f"\t\t- neped_20={sharpness_output['neped_20']:.3f} and "
            f"rhotop={sharpness_output['rho_bc_rho']:.3f} saved for future beats"
        )


# ============================================================================
# Helper functions
# ============================================================================


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


def _apply_sharpness_bc(profiles, rho_bc_rho, psin_bc, Te_bc, Ti_bc, ne_bc_1e19, edge_shape="linear",
                        density_treatment="bc"):
    """
    Modify *profiles* in-place (returns modified copy) so that:

    - Edge region, per edge_shape:
        'linear': Te, Ti, ne interpolated linearly in psi_n down to the separatrix.
        'tanh':   Te, Ti, ne follow the pedestal tanh of FunctionalForms.pedestal_tanh
                  in r/a (the same functional form the eped_initializer uses).
                  NOTE: the sharpness beat's xi definition assumes the linear edge
                  gradient, so it always uses 'linear'; 'tanh' is for the confinement
                  beat, whose matching criterion (H-factor) is integral.
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

    return p


def _plot_sharpness_beat(fn, loaded_results, profiles_before, profiles_after, counter):
    """
    3-panel sharpness beat figure.

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
        rf"Sharpness beat  |  $\xi={xi:.2f}$,  $\psi_{{n,bc}}={psin_bc_g:.3f}$"
        rf"  ($\rho_N={rho_bc_rho:.3f}$),  $T_{{e,bc}}={Te_bc:.3f}$ keV{xi_note}",
        fontsize=FS,
    )


def _plot_sharpness_profiles_coords(fn, loaded_results, profiles_before, profiles_after, counter,
                                    label="Sharpness"):
    """
    Full Te, Ti, ne profiles (rows) plotted against each of the three coordinate
    systems rho_tor, r/a, psi_n (columns), before (blue) and after (red) the
    boundary condition. The BC location is marked by a vertical dashed line in
    whichever coordinate each column uses. `label` names the figure tab (also
    reused by the confinement beat, which applies the same BC machinery).
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

    # Profile values for each state (ne in 10^20 m^-3 to match the other sharpness tab).
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
