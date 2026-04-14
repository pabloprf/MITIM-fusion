import json
import numpy as np
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class gyrokinetic_model:

    def _evaluate_gyrokinetic_model(self, code = 'cgyro', gk_object = None):
        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options
        # ------------------------------------------------------------------------------------------------------------------------

        simulation_options = self.transport_evaluator_options[code]
        cold_start = self.cold_start

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]        
        run_type = SIMtools._normalize_run_type(simulation_options["run"]["run_type"])
        keep_gk_files = simulation_options.get("keep_files", 'all')

        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare object
        # ------------------------------------------------------------------------------------------------------------------------
               
        subfolder_name = f"base_{code}"
            
        # <><><><><><>
        # If the way to store data is in pickle, try first to read the stored pickled in the folder (e.g. for SR stage)
        # <><><><><><>
        gk_object_unpickled = False
        if keep_gk_files in ['pickle']:
            try:
                pickle_file = self.folder / f"{subfolder_name}" / "gk_object.pkl"
                gk_object = SIMtools.restore_class_pickle(pickle_file)
                gk_object_unpickled = True
                print('\t- Pickle file with GK object information has been restored successfully', typeMsg='i')
            except Exception as e:
                gk_object_unpickled = False
                print('\t- Pickle file could not be read, with error:', typeMsg='w')
                print(e)
                
        # <><><><><><>
        # Standard run
        # <><><><><><>
        if not gk_object_unpickled:
            gk_object = gk_object(rhos=rho_locations)

            _ = gk_object.prep(
                self.powerstate.profiles_transport,
                self.folder,
                )

            _ = gk_object.run(
                subfolder_name,
                cold_start=cold_start,
                forceIfcold_start=True,
                only_minimal_files=keep_gk_files in ['none', 'pickle'],
                **simulation_options["run"]
                )
        
        if run_type in ['normal', 'submit', 'send']:
            
            if not gk_object_unpickled:
                
                if run_type in ['submit']:
                    gk_object.check(every_n_minutes=10)
                    gk_object.fetch()

                gk_object.read(
                    label=subfolder_name,
                    minimal=True,  # In case I pickle, I don't want to be extra heavy
                    **simulation_options["read"]
                    )
                
                # Special case to keep only the pickle file but remove all heavy files
                if keep_gk_files in ['pickle']:
                    
                    # Remove results files in subfolder
                    for file in gk_object.output_files_simulation["complete"]:
                        
                        for rho in gk_object.rhos:
                            fileN = f"{file}_{rho:.4f}"
                        
                            (self.folder / f"{subfolder_name}" / fileN).unlink(missing_ok=True)
                    
                    # Save the gk_object as pickle
                    gk_object.save_pickle(pickle_file)
        
            # ------------------------------------------------------------------------------------------------------------------------
            # Pass the information to what power_transport expects
            # ------------------------------------------------------------------------------------------------------------------------

            self.QeGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Qe_mean for i in range(len(rho_locations))])
            self.QeGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Qe_std for i in range(len(rho_locations))])
                    
            self.QiGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Qi_mean for i in range(len(rho_locations))])
            self.QiGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Qi_std for i in range(len(rho_locations))])
                    
            self.GeGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Ge_mean for i in range(len(rho_locations))])
            self.GeGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Ge_std for i in range(len(rho_locations))]) 
            
            self.GZGB_turb = self.QeGB_turb*0.0 #TODO     
            self.GZGB_turb_stds = self.QeGB_turb*0.0 #TODO          

            self.MtGB_turb = self.QeGB_turb*0.0 #TODO     
            self.MtGB_turb_stds = self.QeGB_turb*0.0 #TODO     

            self.QieGB_turb = self.QeGB_turb*0.0 #TODO     
            self.QieGB_turb_stds = self.QeGB_turb*0.0 #TODO     

        elif run_type == 'prep':
            
            # Prevent writing the json file from variables, as we will wait for the user to run CGYRO externally and provide the json themselves
            self._write_json_from_variables_turb = False
            
            # Wait until the user has placed the json file in the right folder
            
            self.powerstate.profiles_transport.write_state(self.folder / subfolder_name / "input.gacode")
            
            pre_checks(self)

            file_path = self.folder / 'fluxes_turb.json'

            attempts = 0
            all_good = post_checks(self) if file_path.exists() else False
            while (file_path.exists() is False) or (not all_good):
                if attempts > 0:
                    print(f"\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f" MITIM could not find the file... looping back", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                logic_to_wait(self.folder, self.folder / subfolder_name)
                attempts += 1

                if file_path.exists():
                    all_good = post_checks(self)

    def _stable_correction(self, simulation_options_all):

        print(f"\n- Checking if any radius has Qi below the stability criterion to apply a stable correction if needed...", typeMsg='i')

        simulation_options = simulation_options_all["cgyro"]

        Qi_stable_criterion = simulation_options["Qi_stable_criterion"]
        Qi_stable_percent_error = simulation_options["Qi_stable_percent_error"]

        # Check if Qi in MW/m2 < Qi_stable_criterion
        QiMWm2 = self.powerstate.plasma['QiMWm2_tr_turb']
        QiMWm2_stds = self.powerstate.plasma['QiMWm2_tr_turb_stds']

        # Handle both single-plasma (1D: nrho) and batched (2D: N x nrho) arrays
        is_batched = np.ndim(QiMWm2) >= 2

        if is_batched:
            N = QiMWm2.shape[0]
            nrho = QiMWm2.shape[1]
            for b in range(N):
                QiMWm2_target_b = self.powerstate.plasma['QiMWm2'][b, 1:].cpu().numpy()
                for i in range(nrho):
                    if QiMWm2[b, i] < Qi_stable_criterion:
                        print(f"\n\t- Qi considered stable at plasma #{b}, radius #{i}: {QiMWm2[b, i]:.2e} MW/m^2 in CGYRO simulation < {Qi_stable_criterion:.2e} MW/m^2 criterion (see namelist)", typeMsg='q')
                        Qi_std = QiMWm2_target_b[i] * Qi_stable_percent_error / 100
                        print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target value as standard deviation: sigma = {Qi_std:.2e} MW/m^2 instead of {QiMWm2_stds[b, i]:.2e} MW/m^2", typeMsg='i')
                        QiMWm2_stds[b, i] = Qi_std
        else:
            QiMWm2_target = self.powerstate.plasma['QiMWm2'][0, 1:].cpu().numpy()
            for i in range(len(QiMWm2)):
                if QiMWm2[i] < Qi_stable_criterion:
                    print(f"\n\t- Qi considered stable at radius #{i}: {QiMWm2[i]:.2e} MW/m^2 in CGYRO simulation < {Qi_stable_criterion:.2e} MW/m^2 criterion (see namelist)", typeMsg='q')
                    Qi_std = QiMWm2_target[i] * Qi_stable_percent_error / 100
                    print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target value as standard deviation: sigma = {Qi_std:.2e} MW/m^2 instead of {QiMWm2_stds[i]:.2e} MW/m^2", typeMsg='i')
                    QiMWm2_stds[i] = Qi_std

class cgyro_model(gyrokinetic_model):

    def evaluate_turbulence(self):

        if self.transport_evaluator_options["cgyro"].get("run_base_tglf", True):
            # Run base TGLF, to keep track of discrepancies! ---------------------------------------------
            simulation_options_tglf = self.transport_evaluator_options["tglf"]
            simulation_options_tglf["use_scan_trick_for_stds"] = None
            self._evaluate_tglf(pass_info = False)
            # --------------------------------------------------------------------------------------------

        self._evaluate_gyrokinetic_model(code = 'cgyro', gk_object = CGYROtools.CGYRO)

    # ----------------------------------------------------------------------------------------
    # Multi-plasma CGYRO — fan a list of profile states through run_over_plasmas so that every
    # (plasma, rho) work unit is dispatched concurrently by the existing FARMINGtools pipeline.
    # Used by power_transport._evaluate_batched() when the powerstate carries batch_size > 1.
    # ----------------------------------------------------------------------------------------
    def evaluate_turbulence_batched(self, list_of_states, pass_info=True):

        # Run base TGLF for diagnostics, same as single-plasma path (pass_info=False
        # so TGLF results are computed for comparison but don't overwrite the flux arrays)
        if self.transport_evaluator_options["cgyro"].get("run_base_tglf", True):
            from mitim_modules.powertorch.physics_models.transport_tglf import tglf_model
            simulation_options_tglf = self.transport_evaluator_options["tglf"]
            simulation_options_tglf["use_scan_trick_for_stds"] = None
            tglf_model._evaluate_tglf_batched(self, list_of_states, pass_info=False)

        simulation_options = self.transport_evaluator_options["cgyro"]
        cold_start = self.cold_start

        run_type = SIMtools._normalize_run_type(simulation_options["run"].get("run_type", "normal"))
        if run_type == "prep":
            raise NotImplementedError(
                "run_type='prep' (interactive external CGYRO run) is not supported in "
                "batched mode. Use single-plasma evaluation or run_type='normal'/'submit'."
            )

        keep_gk_files = simulation_options.get("keep_files", "all")

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        N = len(list_of_states)
        nrho = len(rho_locations)

        # Try to restore from pickle if keep_files == 'pickle' (mirrors single-plasma path)
        cgyro_unpickled = False
        pickle_file = self.folder / "base_cgyro" / "gk_object_batched.pkl"
        if keep_gk_files in ["pickle"]:
            try:
                cgyro = SIMtools.restore_class_pickle(pickle_file)
                cgyro_unpickled = True
                plasma_labels = {p: f"base_cgyro_plasma{p}" for p in range(N)}
                print("\t- Pickle file with batched GK object information has been restored successfully", typeMsg="i")
            except Exception as e:
                cgyro_unpickled = False
                print("\t- Pickle file could not be read, with error:", typeMsg="w")
                print(e)

        if not cgyro_unpickled:
            cgyro = CGYROtools.CGYRO(rhos=rho_locations)

            _ = cgyro.prep(
                list_of_states[0],
                self.folder,
                cold_start=cold_start,
            )

            # run_over_plasmas calls _run_prepare directly (not CGYRO.run()), so
            # preprocess_options must be set on the object beforehand for
            # _run_prepare -> _apply_cgyro_preprocessing to pick it up.
            cgyro._preprocess_options = simulation_options["run"].get("preprocess_options")

            # Filter simulation_options["run"] to only keys that run_over_plasmas accepts;
            # CGYRO-specific keys (preprocess_options) are handled above.
            _run_over_plasmas_keys = {
                "code_settings", "extraOptions", "multipliers", "minimum_delta_abs",
                "ApplyCorrections", "Quasineutral", "launchSlurm", "allocation",
                "run_type", "additional_files_to_send", "helper_lostconnection",
            }
            run_kwargs = {k: v for k, v in simulation_options["run"].items() if k in _run_over_plasmas_keys}

            plasma_labels = cgyro.run_over_plasmas(
                list_of_states,
                base_subfolder="base_cgyro",
                cold_start=cold_start,
                forceIfcold_start=True,
                extra_name=self.name,
                attempts_execution=2,
                only_minimal_files=keep_gk_files in ["none", "pickle"],
                **run_kwargs,
            )
        Qe_batch     = np.zeros((N, nrho))
        Qe_std_batch = np.zeros((N, nrho))
        Qi_batch     = np.zeros((N, nrho))
        Qi_std_batch = np.zeros((N, nrho))
        Ge_batch     = np.zeros((N, nrho))
        Ge_std_batch = np.zeros((N, nrho))
        GZ_batch     = np.zeros((N, nrho))
        GZ_std_batch = np.zeros((N, nrho))
        Mt_batch     = np.zeros((N, nrho))
        Mt_std_batch = np.zeros((N, nrho))
        S_batch      = np.zeros((N, nrho))
        S_std_batch  = np.zeros((N, nrho))

        for p, label in plasma_labels.items():
            if not cgyro_unpickled:
                cgyro.read_plasma(
                    p,
                    label=label,
                    minimal=True,
                    **simulation_options["read"],
                )
            outputs = cgyro.results[label]["output"]

            Qe_batch[p, :]     = np.array([outputs[i].Qe_mean for i in range(nrho)])
            Qe_std_batch[p, :] = np.array([outputs[i].Qe_std for i in range(nrho)])
            Qi_batch[p, :]     = np.array([outputs[i].Qi_mean for i in range(nrho)])
            Qi_std_batch[p, :] = np.array([outputs[i].Qi_std for i in range(nrho)])
            Ge_batch[p, :]     = np.array([outputs[i].Ge_mean for i in range(nrho)])
            Ge_std_batch[p, :] = np.array([outputs[i].Ge_std for i in range(nrho)])

            # GZ, Mt, Qie not yet available from CGYRO — zero as in single-plasma path
            GZ_batch[p, :]     = 0.0  # TODO
            GZ_std_batch[p, :] = 0.0  # TODO
            Mt_batch[p, :]     = 0.0  # TODO
            Mt_std_batch[p, :] = 0.0  # TODO
            S_batch[p, :]      = 0.0  # TODO
            S_std_batch[p, :]  = 0.0  # TODO

        # Save pickle and remove heavy files (mirrors single-plasma path)
        if keep_gk_files in ["pickle"] and not cgyro_unpickled:
            for p, label in plasma_labels.items():
                for file in cgyro.output_files_simulation["complete"]:
                    for rho in cgyro.rhos:
                        fileN = f"{file}_{rho:.4f}"
                        (self.folder / label / fileN).unlink(missing_ok=True)
            pickle_file.parent.mkdir(parents=True, exist_ok=True)
            cgyro.save_pickle(pickle_file)

        if pass_info:
            self.QeGB_turb      = Qe_batch
            self.QeGB_turb_stds = Qe_std_batch

            self.QiGB_turb      = Qi_batch
            self.QiGB_turb_stds = Qi_std_batch

            self.GeGB_turb      = Ge_batch
            self.GeGB_turb_stds = Ge_std_batch

            self.GZGB_turb      = GZ_batch
            self.GZGB_turb_stds = GZ_std_batch

            self.MtGB_turb      = Mt_batch
            self.MtGB_turb_stds = Mt_std_batch

            self.QieGB_turb      = S_batch
            self.QieGB_turb_stds = S_std_batch

        return cgyro


def pre_checks(self):
    
    plasma = self.powerstate.plasma

    txt = "\nFluxes to be matched by turbulence ( Target - Neoclassical ):"

    # Print gradients
    for var, varn in zip(
        ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
        ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
    ):
        txt += f"\n{var}   = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]:.6f}   "

    # Print target fluxes
    for var, varn in zip(
        ["Qe (GB)", "Qi (GB)", "Ge (GB)", "GZ (GB)", "Mt (GB)"],
        ["QeGB", "QiGB", "GeGB", "GZGB", "MtGB"],
    ):
        txt += f"\n{var}  = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]-self.__dict__[f'{varn}_neoc'][j]:.4e}   "

    print(txt)

def logic_to_wait(folder, subfolder):
    print(f"\n**** Simulation inputs prepared. Please, run it from the simulation setup in folder:\n", typeMsg='i')
    print(f"\t {subfolder}\n", typeMsg='i')
    print(f"**** When finished, the fluxes_turb.json file should be placed in:\n", typeMsg='i')
    print(f"\t {folder}/fluxes_turb.json\n", typeMsg='i')
    while not print(f"**** When you have done that, please say yes", typeMsg='q'):
        pass

def post_checks(self, rtol = 1e-3):
    
    with open(self.folder / 'fluxes_turb.json', 'r') as f:
        json_dict = json.load(f)
        
    additional_info_from_json = json_dict.get('additional_info', {})
    
    all_good = True
    
    if len(additional_info_from_json) == 0:
        print(f"\t- No additional info found in fluxes_turb.json to be compared with", typeMsg='i')
        
    else:
        print(f"\t- Additional info found in fluxes_turb.json:", typeMsg='i')
        for k, v in additional_info_from_json.items():
            vP = self.powerstate.plasma[k].cpu().numpy()[0,1:]
            
            crit = not np.allclose(v, vP, rtol=rtol)

            print(f"\t   {k} from JSON      : {[round(float(i),4) for i in v]}", typeMsg='' if not crit else 'i')
            print(f"\t   {k} from POWERSTATE: {[round(float(i),4) for i in vP]}", typeMsg='' if not crit else 'i')

            if crit:
                all_good = print(f"{k} does not match with a relative tolerance of {rtol*100.0:.3f}%, max rel difference: {np.max(np.abs(v - vP) / np.maximum(np.abs(v), np.abs(vP)))*100.0:.3f}%", typeMsg='q')

    return all_good

def write_json_CGYRO(roa, fluxes_mean, fluxes_stds, additional_info = None, file = 'fluxes_turb.json'):
    '''
    *********************
    Helper to write JSON
    *********************
        roa
            Must be an array: [0.25, 0.35, ...]
        fluxes_mean
            Must be a dictionary with the fields and arrays:
                'QeMWm2': [0.1, 0.2, ...],
                'QiMWm2': ...,
                'Ge1E20m2': ...,
                'GZ1E20m2': ...,
                'MtJm2': ...,
                'QieMWm3': ..
            or, alternatively (or complementary), in GB units:
                'QeGB': [0.1, 0.2, ...],
                'QiGB': ...,
                'GeGB': ...,
                'GZGB': ...,
                'MtGB': ...,
                'QieGB': ..
        fluxes_stds
            Exact same structure as fluxes_mean
        additional_info
            A dictionary with any additional information to include in the JSON and compare to powerstate,
            for example (and recommended):
                'aLte': [0.2, 0.5, ...],
                'aLti': [0.3, 0.6, ...],
                'aLne': [0.3, 0.6, ...],
                'Qgb': [0.4, 0.7, ...],
                'rho': [0.2, 0.5, ...],
    '''
    
    if additional_info is None:
        additional_info = {}

    with open(file, 'w') as f:

        additional_info_extended = additional_info | {'roa': roa.tolist() if not isinstance(roa, list) else roa}

        json_dict = {
            'fluxes_mean': fluxes_mean,
            'fluxes_stds': fluxes_stds,
            'additional_info': additional_info_extended
        }

        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.generic,)):
                return obj.item()
            else:
                return obj
            
        json.dump(convert_numpy(json_dict), f, indent=4)
