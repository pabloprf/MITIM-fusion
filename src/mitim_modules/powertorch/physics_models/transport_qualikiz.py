import numpy as np
from mitim_tools.qualikiz_tools import QLKtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as _interp


class qualikiz_model:

    def evaluate_turbulence(self):
        self._evaluate_qualikiz()

    # ------------------------------------------------------------------
    # Single-plasma QuaLiKiz dispatch
    # ------------------------------------------------------------------

    def _evaluate_qualikiz(self, pass_info=True):

        options_key = getattr(self, "_active_turb_options_key", None) or "qlk"
        simulation_options = self.transport_evaluator_options[options_key]
        cold_start = self.cold_start

        percent_error = simulation_options.get("percent_error", 10.0)
        allocation    = simulation_options.get("allocation", {"resources_per_call": 1, "minutes": 60})
        attempts      = simulation_options.get("attempts_execution", 1)

        # Side-aware: postproc may give the turbulence side a different species
        # list than the neoclassical side.
        ion_OI_position_in_ion_list = self._impurity_position_transport_for("turb")

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        qlk = QLKtools.QuaLiKiz(rhos=rho_locations)

        qlk.prep(
            self._profiles_transport_for("turb"),
            self.folder,
            cold_start=cold_start,
            forceIfcold_start=True,
        )

        # Subfolder mirrors the TGLF convention so named multi-fidelity configs
        # ('qlk1'/'qlk2') don't collide on disk.
        subfolder_name = f"base_{options_key}"
        qlk.run(
            subfolder_name,
            cold_start=cold_start,
            forceIfcold_start=True,
            extra_name=self.name,
            allocation=allocation,
            attempts_execution=attempts,
            **simulation_options.get("run", {}),
        )

        qlk.read(label="base", **simulation_options.get("read", {}))

        nrho = len(rho_locations)
        Qe = np.zeros(nrho)
        Qi = np.zeros(nrho)
        Ge = np.zeros(nrho)
        GZ = np.zeros(nrho)
        Mt = np.zeros(nrho)
        S  = np.zeros(nrho)   # Qie turbulent exchange not provided by QuaLiKiz

        for i, out in enumerate(qlk.results["base"]["output"]):
            # efe_GB: electron heat flux, dims [dimx] → scalar after isel
            Qe[i] = float(out["efe_GB"].values)
            # pfe_GB: electron particle flux, dims [dimx] → scalar
            Ge[i] = float(out["pfe_GB"].values)
            # efi_GB: ion heat flux, dims [dimx, nions] → sum over all ion species
            Qi[i] = float(out["efi_GB"].values.sum())
            # pfi_GB: ion particle flux, dims [dimx, nions] → pick impurity species
            GZ[i] = float(out["pfi_GB"].values[ion_OI_position_in_ion_list])
            # vfi_GB: ion toroidal angular momentum flux, dims [dimx, nions].
            # Only written when phys_meth >= 1 (STANDARD/ROTATION presets); fall
            # back to 0 (momentum not predicted) when absent.
            try:
                Mt[i] = float(out["vfi_GB"].values.sum())
            except KeyError:
                pass

        # QuaLiKiz normalises its GB outputs with B0 (on-axis field), but MITIM's
        # internal GB convention uses B_unit.  Since Q_GB ∝ ρ_s² ∝ B⁻², the
        # conversion factor from QuaLiKiz GB → MITIM GB is (B_unit/B0)² at each
        # radius.  This applies equally to all flux channels.
        p = self._profiles_transport_for("turb")
        B0 = float(np.abs(p.profiles["bcentr(T)"][-1]))
        B_unit_at_rhos = _interp(
            np.array(rho_locations), p.profiles["rho(-)"], p.derived["B_unit"]
        )
        b_correction = (B_unit_at_rhos / B0) ** 2
        Qe *= b_correction
        Qi *= b_correction
        Ge *= b_correction
        GZ *= b_correction
        Mt *= b_correction

        Flux_mean = np.array([Qe, Qi, Ge, GZ, Mt, S])
        Flux_std  = np.abs(Flux_mean) * percent_error / 100.0

        if pass_info:
            self.QeGB_turb       = Flux_mean[0]
            self.QeGB_turb_stds  = Flux_std[0]
            self.QiGB_turb       = Flux_mean[1]
            self.QiGB_turb_stds  = Flux_std[1]
            self.GeGB_turb       = Flux_mean[2]
            self.GeGB_turb_stds  = Flux_std[2]
            self.GZGB_turb       = Flux_mean[3]
            self.GZGB_turb_stds  = Flux_std[3]
            self.MtGB_turb       = Flux_mean[4]
            self.MtGB_turb_stds  = Flux_std[4]
            self.QieGB_turb      = Flux_mean[5]
            self.QieGB_turb_stds = Flux_std[5]

        return qlk
