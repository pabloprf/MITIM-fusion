import copy
import numpy as np
from mitim_tools.gacode_tools import QLGYROtools
from mitim_modules.powertorch.physics_models.transport_cgyro import (
    gyrokinetic_model,
    logic_to_wait,
    post_checks,
    pre_checks,
)


class qlgyro_model(gyrokinetic_model):
    def evaluate_turbulence(self):
        simulation_options = self.transport_evaluator_options["qlgyro"]

        if simulation_options.get("run_base_tglf", True):
            simulation_options_tglf = self.transport_evaluator_options["tglf"]
            simulation_options_tglf["use_scan_trick_for_stds"] = None
            self._evaluate_tglf(pass_info=False)

        cold_start = self.cold_start
        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        qlgyro = QLGYROtools.QLGYRO(rhos=rho_locations)
        _ = qlgyro.prep(
            self.powerstate.profiles_transport,
            self.folder,
        )

        subfolder_name = "base_qlgyro"
        run_options = copy.deepcopy(simulation_options["run"])
        run_options.setdefault("run_type", "normal")
        run_options.setdefault("code_settings", "Linear")
        run_type = run_options["run_type"]

        _ = qlgyro.run(
            subfolder_name,
            cold_start=cold_start,
            forceIfcold_start=True,
            **run_options,
        )

        if run_type in ["normal", "submit"]:
            if run_type == "submit":
                qlgyro.check(every_n_minutes=10)
                qlgyro.fetch()

            qlgyro.read(
                label=subfolder_name,
                **simulation_options.get("read", {}),
            )

            outputs = qlgyro.results[subfolder_name]["output"]
            percent_error = simulation_options.get("percent_error", 0.0)
            impurity_position = self.powerstate.impurityPosition_transport

            self.QeGB_turb = np.array([output.Qe_mean for output in outputs])
            self.QeGB_turb_stds = np.abs(self.QeGB_turb) * percent_error / 100.0

            self.QiGB_turb = np.array([output.Qi_mean for output in outputs])
            self.QiGB_turb_stds = np.abs(self.QiGB_turb) * percent_error / 100.0

            self.GeGB_turb = np.array([output.Ge_mean for output in outputs])
            self.GeGB_turb_stds = np.abs(self.GeGB_turb) * percent_error / 100.0

            self.GZGB_turb = np.array([
                output.Gamma_i[impurity_position] if impurity_position < len(output.Gamma_i) else 0.0
                for output in outputs
            ])
            self.GZGB_turb_stds = np.abs(self.GZGB_turb) * percent_error / 100.0

            self.MtGB_turb = np.array([output.Mt_mean for output in outputs])
            self.MtGB_turb_stds = np.abs(self.MtGB_turb) * percent_error / 100.0

            self.QieGB_turb = np.array([output.Qie_mean for output in outputs])
            self.QieGB_turb_stds = np.abs(self.QieGB_turb) * percent_error / 100.0

            self.model_results = qlgyro.results[subfolder_name]

        elif run_type == "prep":
            self._write_json_from_variables_turb = False
            self.powerstate.profiles_transport.write_state(self.folder / subfolder_name / "input.gacode")

            pre_checks(self)

            file_path = self.folder / "fluxes_turb.json"

            attempts = 0
            all_good = post_checks(self) if file_path.exists() else False
            while (file_path.exists() is False) or (not all_good):
                if attempts > 0:
                    print("\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg="i")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg="i")
                    print(" MITIM could not find the file... looping back", typeMsg="i")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg="i")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg="i")
                logic_to_wait(self.folder, self.folder / subfolder_name)
                attempts += 1

                if file_path.exists():
                    all_good = post_checks(self)

        if "Qi_stable_criterion" in simulation_options:
            self._stable_correction(simulation_options)