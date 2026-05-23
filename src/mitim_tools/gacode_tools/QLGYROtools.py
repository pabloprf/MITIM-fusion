from pathlib import Path
import numpy as np
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools.CGYROtools import CGYROinput
from mitim_tools.gacode_tools.utils import GACODEdefaults
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools import CONFIGread, IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


class QLGYRO(SIMtools.mitim_simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def code_call(folder, p, n=1, nomp=1, additional_command="", **kwargs):
            return f"qlgyro -e {folder} -n {n} -nomp {nomp} {additional_command}"

        def code_slurm_settings(name, minutes, total_cores_required, cores_per_code_call, type_of_submission, array_list=None, **kwargs_slurm):
            slurm_settings = {
                "name": name,
                "minutes": minutes,
            }

            machine_settings = CONFIGread.machineSettings(code="qlgyro")

            if type_of_submission == "slurm_standard":
                slurm_settings["ntasks"] = total_cores_required // cores_per_code_call
                if machine_settings["gpus_per_node"] > 0:
                    slurm_settings["gpuspertask"] = cores_per_code_call
                else:
                    slurm_settings["cpuspertask"] = cores_per_code_call
            elif type_of_submission == "slurm_array":
                slurm_settings["ntasks"] = 1
                if machine_settings["gpus_per_node"] > 0:
                    slurm_settings["gpuspertask"] = cores_per_code_call
                else:
                    slurm_settings["cpuspertask"] = cores_per_code_call
                slurm_settings["job_array"] = ",".join(array_list)

            return slurm_settings

        self.run_specifications = {
            "code": "qlgyro",
            "input_file": "input.cgyro",
            "code_call": code_call,
            "code_slurm_settings": code_slurm_settings,
            "control_function": GACODEdefaults.addCGYROcontrol,
            "controls_file": "input.cgyro.controls",
            "state_converter": "to_cgyro",
            "input_class": CGYROinput,
            "complete_variation": None,
            "default_cores": 16,
            "output_class": QLGYROoutput,
        }

        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t QLGYRO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles_minimal = [
            "out.qlgyro.gbflux",
            "out.qlgyro.status",
            "out.qlgyro.units",
        ]

        self.ResultsFiles = self.ResultsFiles_minimal + [
            "out.qlgyro.run",
            "out.qlgyro.version",
            "out.qlgyro.ky_spectrum",
            "out.qlgyro.eigenvalue_spectrum",
            "out.qlgyro.QL_weight_spectrum",
            "out.qlgyro.field_spectrum",
            "out.qlgyro.flux_spectrum",
            "out.qlgyro.sat_geo_spectrum",
            "out.qlgyro.kxrms_spectrum",
            "out.qlgyro.taskmapping",
        ]

        self.qlgyro_input_files = {}

    def prep(self, mitim_state, FolderGACODE, cold_start=False, forceIfcold_start=False):
        cdf = super().prep(
            mitim_state,
            FolderGACODE,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
        )

        qlgyro_inputs_folder = self.FolderGACODE / "qlgyro_inputs"
        for rho in self.rhos:
            qlgyro_controls = GACODEdefaults.addQLGYROcontrol("default")
            qlgyro_controls["GAMMA_E"] = self.inputs_files[rho].plasma.get("GAMMA_E", qlgyro_controls["GAMMA_E"])

            qlgyro_input = QLGYROinput.initialize_in_memory(qlgyro_controls)

            qlgyro_file = qlgyro_inputs_folder / f"rho_{rho:.4f}" / "input.qlgyro"
            qlgyro_file.parent.mkdir(parents=True, exist_ok=True)
            qlgyro_input.file = qlgyro_file
            qlgyro_input.write_state()

            self.qlgyro_input_files[rho] = qlgyro_file

        return cdf

    def _run_prepare(self, subfolder_simulation, additional_files_to_send=None, **kwargs):
        merged_files = {} if additional_files_to_send is None else {rho: list(files) for rho, files in additional_files_to_send.items()}

        for rho in self.rhos:
            merged_files.setdefault(rho, [])
            if rho in self.qlgyro_input_files:
                merged_files[rho].append(self.qlgyro_input_files[rho])

        return super()._run_prepare(
            subfolder_simulation,
            additional_files_to_send=merged_files,
            **kwargs,
        )


class QLGYROoutput(SIMtools.GACODEoutput):
    def __init__(self, folder, suffix=None, **kwargs):
        super().__init__()

        self.folder = Path(folder)
        self.suffix = suffix or ""

        self.inputFile = None
        self.input_qlgyro = None

        input_cgyro_file = self.folder / f"input.cgyro{self.suffix}"
        if input_cgyro_file.exists():
            self.inputFile = input_cgyro_file.read_text()

        rho_label = self.suffix[1:] if self.suffix.startswith("_") else self.suffix
        if rho_label:
            qlgyro_input_file = self.folder / "qlgyro_inputs" / f"rho_{rho_label}" / "input.qlgyro"
            if qlgyro_input_file.exists():
                self.input_qlgyro = qlgyro_input_file.read_text()

        parsed_input = SIMtools.buildDictFromInput(self.inputFile) if self.inputFile else {}
        n_species = int(parsed_input.get("N_SPECIES", 0))

        gbflux_file = self.folder / f"out.qlgyro.gbflux{self.suffix}"
        if not gbflux_file.exists():
            raise FileNotFoundError(f"Could not find {gbflux_file}")

        gbflux = np.fromstring(gbflux_file.read_text(), sep=" ")
        if n_species == 0:
            if gbflux.size % 4 != 0:
                raise ValueError(f"Unexpected QLGYRO gbflux length {gbflux.size} in {gbflux_file}")
            n_species = gbflux.size // 4

        expected_length = 4 * n_species
        if gbflux.size != expected_length:
            raise ValueError(f"Expected {expected_length} entries in {gbflux_file}, found {gbflux.size}")

        gamma = gbflux[0:n_species]
        heat = gbflux[n_species:2 * n_species]
        momentum = gbflux[2 * n_species:3 * n_species]
        exchange = gbflux[3 * n_species:4 * n_species]

        self.Gamma_e = float(gamma[0])
        self.Gamma_i = np.array(gamma[1:])
        self.Qe = float(heat[0])
        self.Qi_species = np.array(heat[1:])
        self.Pi_e = float(momentum[0])
        self.Pi_i = np.array(momentum[1:])
        self.Se = float(exchange[0])
        self.Si = np.array(exchange[1:])

        self.Ge_mean = self.Gamma_e
        self.Qe_mean = self.Qe
        self.Qi_mean = float(np.sum(self.Qi_species))
        self.Mt_mean = float(np.sum(self.Pi_i))
        self.Qie_mean = self.Se

        self.Ge_std = 0.0
        self.Qe_std = 0.0
        self.Qi_std = 0.0
        self.Mt_std = 0.0
        self.Qie_std = 0.0

        status_file = self.folder / f"out.qlgyro.status{self.suffix}"
        self.status = status_file.read_text() if status_file.exists() else ""
        if self.status and "unconverged" in self.status.lower():
            print(f"\t- QLGYRO status reports unconverged points in {IOtools.clipstr(status_file)}", typeMsg="w")


class QLGYROinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file=__mitimroot__ / "templates" / "input.qlgyro.controls",
            code="QLGYRO",
        )