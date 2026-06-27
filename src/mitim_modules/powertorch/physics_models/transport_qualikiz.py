import numpy as np
from mitim_tools.qualikiz_tools import QLKtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


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
            # Read physical-unit (SI) outputs; normalise below using the
            # powerstate's Qgb/Ggb/Pgb (which use B_unit), so no explicit
            # B0→B_unit correction is needed.
            Qe[i] = float(out["efe_SI"].values)
            Ge[i] = float(out["pfe_SI"].values)
            Qi[i] = float(out["efi_SI"].values.sum())
            GZ[i] = float(out["pfi_SI"].values[ion_OI_position_in_ion_list])
            try:
                Mt[i] = float(out["vfi_SI"].values.sum())
            except KeyError:
                pass

        # Normalise SI fluxes to MITIM GB units.
        # efe/efi_SI [W/m²]  → divide by Qgb [MW/m²] × 1e6
        # pfe/pfi_SI [m⁻²s⁻¹] → divide by Ggb [1e20 m⁻²s⁻¹] × 1e20
        # vfi_SI [J/m²]      → divide by Pgb [J/m²]
        Qgb = self.powerstate.plasma["Qgb"][0, 1:].cpu().numpy()
        Ggb = self.powerstate.plasma["Ggb"][0, 1:].cpu().numpy()
        Pgb = self.powerstate.plasma["Pgb"][0, 1:].cpu().numpy()
        Qe /= Qgb * 1e6
        Qi /= Qgb * 1e6
        Ge /= Ggb * 1e20
        GZ /= Ggb * 1e20
        Mt /= Pgb

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
