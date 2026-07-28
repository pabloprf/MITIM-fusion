from pathlib import Path
import numpy as np
import pandas as pd
from mitim_tools.misc_tools import IOtools, LOGtools
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class tglf_model:

    def evaluate_turbulence(self):
        self._evaluate_tglf()

    def evaluate_turbulence_batched(self, list_of_states):
        self._evaluate_tglf_batched(list_of_states)

    # ----------------------------------------------------------------------------------------
    # Multi-plasma TGLF — fan a list of profile states through run_over_plasmas so that every
    # (plasma, rho) work unit is dispatched concurrently by the existing FARMINGtools pipeline.
    # Used by power_transport._evaluate_batched() when the powerstate carries batch_size > 1.
    # ----------------------------------------------------------------------------------------
    def _evaluate_tglf_batched(self, list_of_states, pass_info=True, options_key=None):

        # options_key lets cgyro_model reach in for the base-TGLF diagnostic and pin the
        # read to options["tglf"] even when the active turbulence fidelity is cgyro.
        options_key = options_key or getattr(self, "_active_turb_options_key", None) or "tglf"
        simulation_options = self.transport_evaluator_options[options_key]
        cold_start = self.cold_start

        Qi_includes_fast = simulation_options["Qi_includes_fast"]
        use_tglf_scan_trick = simulation_options["use_scan_trick_for_stds"]
        cores_per_tglf_instance = simulation_options["cores_per_tglf_instance"]
        keep_tglf_files = simulation_options["keep_files"]
        percent_error = simulation_options["percent_error"]
        uncertainty_statistics = simulation_options.get("uncertainty_statistics", "gaussian")
        in_process = self.powerstate.transport_options.get("in_process", False)

        # Side-aware (turbulence): see evaluate_turbulence() comment.
        ion_OI_position_in_ion_list = self._impurity_position_transport_for("turb")

        reuse_scan_ball_region = resolve_reuse_scan_ball(simulation_options)
        reuse_scan_ball_file = self.powerstate.transport_options['folder'] / 'Outputs' / 'tglf_ball.npz' if reuse_scan_ball_region is not None else None

        # All per-plasma powerstates are tiled from the same rho grid, so the rho locations
        # come from batch 0 of the (already batched) self.powerstate exactly as the single-
        # plasma path does — not from list_of_states, whose gacode_state objects use their
        # own rho grid and do not need to agree with predicted_rho.
        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        tglf = TGLFtools.TGLF(rhos=rho_locations, in_process=in_process)

        # Bootstrap FolderGACODE / NormalizationSets for the first plasma; run_over_plasmas
        # re-preps once per plasma and snapshots per-plasma state on self.results_per_plasma
        # before the parallel dispatch kicks in.
        _ = tglf.prep(
            list_of_states[0],
            self.folder,
            cold_start=cold_start,
        )

        plasma_labels = tglf.run_over_plasmas(
            list_of_states,
            base_subfolder=f"base_{options_key}",
            cold_start=cold_start,
            forceIfcold_start=True,
            extra_name=self.name,
            allocation={
                "resources_per_call": cores_per_tglf_instance,
                "minutes": 2,
            },
            attempts_execution=2,
            only_minimal_files=keep_tglf_files in ["none"],
            **simulation_options["run"],
        )

        N = len(plasma_labels)
        nrho = len(rho_locations)
        Qe_batch = np.zeros((N, nrho))
        Qi_batch = np.zeros((N, nrho))
        Ge_batch = np.zeros((N, nrho))
        GZ_batch = np.zeros((N, nrho))
        Mt_batch = np.zeros((N, nrho))
        S_batch = np.zeros((N, nrho))

        for p, label in plasma_labels.items():
            tglf.read_plasma(
                p,
                label=label,
                require_all_files=False,
                **simulation_options["read"],
            )
            outputs = tglf.results[label]["output"]

            Qe = np.array([outputs[i].Qe for i in range(nrho)])
            Qi = np.array([outputs[i].Qi for i in range(nrho)])
            Ge = np.array([outputs[i].Ge for i in range(nrho)])
            GZ = np.array([outputs[i].GiAll[ion_OI_position_in_ion_list] for i in range(nrho)])
            Mt = np.array([outputs[i].Mt for i in range(nrho)])
            S = np.array([outputs[i].Se for i in range(nrho)])

            if Qi_includes_fast:
                Qifast = np.array([outputs[i].Qifast for i in range(nrho)])
                if Qifast.sum() != 0.0:
                    print(f"\t- Qi includes fast ions for plasma {p}, adding their contribution")
                    Qi = Qi + Qifast

            Qe_batch[p, :] = Qe
            Qi_batch[p, :] = Qi
            Ge_batch[p, :] = Ge
            GZ_batch[p, :] = GZ
            Mt_batch[p, :] = Mt
            S_batch[p, :] = S

            # Warnings reuse the single-plasma helper by temporarily pointing tglf at the
            # per-plasma profiles/inputs that read_plasma() already cached.
            self._raise_warnings(tglf, rho_locations, Qi_includes_fast)

        Flux_base = np.stack([Qe_batch, Qi_batch, Ge_batch, GZ_batch, Mt_batch, S_batch], axis=0)

        # ------------------------------------------------------------------------------------------------------------------------
        # Evaluate TGLF uncertainty — same choice as the single-plasma path:
        # either ad-hoc percent_error or the scan-based uncertainty model.
        # ------------------------------------------------------------------------------------------------------------------------

        if use_tglf_scan_trick is None:
            Flux_value = Flux_base
            Flux_std = np.abs(Flux_value) * percent_error / 100.0
            Flux_extras = None
        else:
            # All N plasmas' scan work is accumulated into one code_executor and dispatched
            # in a single _run call, so N × ~65 scan jobs run concurrently.
            Flux_base_per_plasma = {
                p: np.array([Qe_batch[p], Qi_batch[p], Ge_batch[p],
                             GZ_batch[p], Mt_batch[p], S_batch[p]])
                for p in plasma_labels
            }
            run_kwargs = {k: v for k, v in simulation_options["run"].items() if k != "code_settings"}
            Flux_value, Flux_std, Flux_extras = _run_tglf_uncertainty_model_batched(
                tglf,
                rho_locations,
                self.powerstate.predicted_channels,
                Flux_base_per_plasma=Flux_base_per_plasma,
                plasma_labels=plasma_labels,
                ion_OI_position_in_ion_list=ion_OI_position_in_ion_list,
                delta=use_tglf_scan_trick,
                cold_start=cold_start,
                extra_name=self.name,
                cores_per_tglf_instance=cores_per_tglf_instance,
                Qi_includes_fast=Qi_includes_fast,
                only_minimal_files=keep_tglf_files in ["none", "base"],
                reuse_scan_ball_file=reuse_scan_ball_file,
                reuse_scan_ball_region=reuse_scan_ball_region,
                code_settings=simulation_options["run"].get("code_settings"),
                statistics=uncertainty_statistics,
                **run_kwargs,
            )

        if pass_info:
            self.QeGB_turb      = Flux_value[0]
            self.QeGB_turb_stds = Flux_std[0]

            self.QiGB_turb      = Flux_value[1]
            self.QiGB_turb_stds = Flux_std[1]

            self.GeGB_turb      = Flux_value[2]
            self.GeGB_turb_stds = Flux_std[2]

            self.GZGB_turb      = Flux_value[3]
            self.GZGB_turb_stds = Flux_std[3]

            self.MtGB_turb      = Flux_value[4]
            self.MtGB_turb_stds = Flux_std[4]

            self.QieGB_turb      = Flux_value[5]
            self.QieGB_turb_stds = Flux_std[5]

            # Asymmetric-statistics sidecar (semi-deviations symmetric to the std, and no
            # samples, when the scan trick is off)
            for i, var in enumerate(["Qe", "Qi", "Ge", "GZ", "Mt", "Qie"]):
                setattr(self, f"{var}GB_turb_stds_minus", Flux_extras["stds_minus"][i] if Flux_extras is not None else Flux_std[i])
                setattr(self, f"{var}GB_turb_stds_plus",  Flux_extras["stds_plus"][i]  if Flux_extras is not None else Flux_std[i])
                setattr(self, f"{var}GB_turb_samples",    Flux_extras["samples"][i]    if Flux_extras is not None else None)
            self.turb_samples_from_ball = Flux_extras["samples_from_ball"] if Flux_extras is not None else None

        return tglf

    # Have it separate such that I can call it from the CGYRO class but without the decorator
    def _evaluate_tglf(self, pass_info = True, options_key=None):

        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options
        # ------------------------------------------------------------------------------------------------------------------------

        # options_key lets cgyro_model reach in for the base-TGLF diagnostic and pin the
        # read to options["tglf"] even when the active turbulence fidelity is cgyro.
        options_key = options_key or getattr(self, "_active_turb_options_key", None) or "tglf"
        simulation_options = self.transport_evaluator_options[options_key]
        cold_start = self.cold_start

        Qi_includes_fast = simulation_options["Qi_includes_fast"]
        use_tglf_scan_trick = simulation_options["use_scan_trick_for_stds"]
        reuse_scan_ball_region = resolve_reuse_scan_ball(simulation_options)
        reuse_scan_ball_file = self.powerstate.transport_options['folder'] / 'Outputs' / 'tglf_ball.npz' if reuse_scan_ball_region is not None else None
        cores_per_tglf_instance = simulation_options["cores_per_tglf_instance"]
        keep_tglf_files = simulation_options["keep_files"]
        percent_error = simulation_options["percent_error"]
        uncertainty_statistics = simulation_options.get("uncertainty_statistics", "gaussian")
        # If True, TGLF runs in-process via ctypes (libtglf_serial.so) — no
        # subprocess fork, no folder / file I/O.  See namelist.portals.yaml.
        in_process = self.powerstate.transport_options.get("in_process", False)

        # Grab impurity from powerstate ( because it may have been modified in produce_profiles() )
        # [ion1,ion2,ion3,...], so if I want ion3, I need to do ion_OI_position_in_ion_list = 2
        # Side-aware: under per-model postproc the turbulence side may have a
        # different species list (and impurity position) than the neoclassical side.
        ion_OI_position_in_ion_list = self._impurity_position_transport_for("turb")

        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare TGLF object
        # ------------------------------------------------------------------------------------------------------------------------

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]

        tglf = TGLFtools.TGLF(rhos=rho_locations, in_process=in_process)

        _ = tglf.prep(
            self._profiles_transport_for("turb"),
            self.folder,
            cold_start = cold_start,
            )
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Run TGLF (base)
        # ------------------------------------------------------------------------------------------------------------------------

        # Subfolder name tracks the instance (so named multi-fidelity configs like
        # 'tglf1' / 'tglf2' don't collide on disk). Single-fidelity keeps writing to
        # 'base_tglf' exactly as before.
        subfolder_name = f"base_{options_key}"
        tglf.run(
            subfolder_name,
            ApplyCorrections=False,
            cold_start= cold_start,
            forceIfcold_start=True,
            extra_name= self.name,
            allocation={
                "resources_per_call": cores_per_tglf_instance,
                "minutes": 2,
                },
            attempts_execution=2,
            only_minimal_files=keep_tglf_files in ['none'],
            **simulation_options["run"]
        )
    
        tglf.read(
            label='base',
            require_all_files=False,
            **simulation_options["read"])
        
        # Grab values
        Qe = np.array([tglf.results['base']['output'][i].Qe for i in range(len(rho_locations))])
        Qi = np.array([tglf.results['base']['output'][i].Qi for i in range(len(rho_locations))])
        Ge = np.array([tglf.results['base']['output'][i].Ge for i in range(len(rho_locations))])
        GZ = np.array([tglf.results['base']['output'][i].GiAll[ion_OI_position_in_ion_list] for i in range(len(rho_locations))])
        Mt = np.array([tglf.results['base']['output'][i].Mt for i in range(len(rho_locations))])
        S = np.array([tglf.results['base']['output'][i].Se for i in range(len(rho_locations))])

        if Qi_includes_fast:

            Qifast = np.array([tglf.results['base']['output'][i].Qifast for i in range(len(rho_locations))])

            if Qifast.sum() != 0.0:
                print(f"\t- Qi includes fast ions, adding their contribution")
                Qi += Qifast
                
        Flux_base = np.array([Qe, Qi, Ge, GZ, Mt, S])
              
        # ------------------------------------------------------------------------------------------------------------------------
        # Evaluate TGLF uncertainty
        # ------------------------------------------------------------------------------------------------------------------------
                
        if use_tglf_scan_trick is None:
            
            # *******************************************************************
            # Just apply an ad-hoc percent error to the results
            # *******************************************************************
            
            Flux_value = Flux_base
            Flux_std = abs(Flux_value)*percent_error/100.0
            Flux_extras = None

        else:

            # *******************************************************************
            # Run TGLF with scans to estimate the uncertainty
            # *******************************************************************

            Flux_value, Flux_std, Flux_extras = _run_tglf_uncertainty_model(
                tglf,
                rho_locations, 
                self.powerstate.predicted_channels, 
                Flux_base = Flux_base,
                ion_OI_position_in_ion_list=ion_OI_position_in_ion_list, 
                delta = use_tglf_scan_trick,
                cold_start=cold_start,
                extra_name=self.name,
                cores_per_tglf_instance=cores_per_tglf_instance,
                Qi_includes_fast=Qi_includes_fast,
                only_minimal_files=keep_tglf_files in ['none', 'base'],
                reuse_scan_ball_file=reuse_scan_ball_file,
                reuse_scan_ball_region=reuse_scan_ball_region,
                statistics=uncertainty_statistics,
                **simulation_options["run"]
                )

        self._raise_warnings(tglf, rho_locations, Qi_includes_fast)

        # ------------------------------------------------------------------------------------------------------------------------
        # Pass the information to what power_transport expects
        # ------------------------------------------------------------------------------------------------------------------------
        
        if pass_info:
            
            self.QeGB_turb = Flux_value[0]
            self.QeGB_turb_stds = Flux_std[0]
                    
            self.QiGB_turb = Flux_value[1]
            self.QiGB_turb_stds = Flux_std[1]
                    
            self.GeGB_turb = Flux_value[2]
            self.GeGB_turb_stds = Flux_std[2]        
            
            self.GZGB_turb = Flux_value[3]           
            self.GZGB_turb_stds = Flux_std[3]       

            self.MtGB_turb = Flux_value[4]
            self.MtGB_turb_stds = Flux_std[4]

            self.QieGB_turb = Flux_value[5]
            self.QieGB_turb_stds = Flux_std[5]

            # Asymmetric-statistics sidecar (semi-deviations symmetric to the std, and no
            # samples, when the scan trick is off)
            for i, var in enumerate(["Qe", "Qi", "Ge", "GZ", "Mt", "Qie"]):
                setattr(self, f"{var}GB_turb_stds_minus", Flux_extras["stds_minus"][i] if Flux_extras is not None else Flux_std[i])
                setattr(self, f"{var}GB_turb_stds_plus",  Flux_extras["stds_plus"][i]  if Flux_extras is not None else Flux_std[i])
                setattr(self, f"{var}GB_turb_samples",    Flux_extras["samples"][i]    if Flux_extras is not None else None)
            self.turb_samples_from_ball = Flux_extras["samples_from_ball"] if Flux_extras is not None else None

        return tglf

    def _raise_warnings(self, tglf, rho_locations, Qi_includes_fast):

        for i in range(len(tglf.profiles.Species)):
            gacode_type = tglf.profiles.Species[i]['S']
            for rho in rho_locations:
                try:
                    tglf_type = tglf.inputs_files[rho].ions_info[i+2]['type']
                except KeyError:
                    print(f"\t\t\t* Could not determine ion type from TGLF inputs because ion {i+2} was not there for {rho =}, skipping consistency check", typeMsg="w")
                    continue
                
                if gacode_type[:5] != tglf_type[:5]:
                    print(f"\t- For location {rho=:.2f}, ion specie #{i+1} ({tglf.profiles.Species[i]['N']}) is considered '{gacode_type}' by gacode but '{tglf_type}' by TGLF. Make sure this is consistent with your use case", typeMsg="w")
        
                    if tglf_type == 'fast':
        
                        if Qi_includes_fast:
                            print(f"\t\t\t* The fast ion considered by TGLF was summed into the Qi", typeMsg="i")
                        else:
                            print(f"\t\t\t* The fast ion considered by TGLF was NOT summed into the Qi", typeMsg="i")
                            
                    else:
                        print(f"\t\t\t* The thermal ion considered by TGLF was summed into the Qi", typeMsg="i")

def _build_uncertainty_scan_setup(predicted_channels, ion_OI_position_in_total_padded_list, minimum_abs_gradient):
    '''
    Map each predicted channel to the TGLF input variable scanned to estimate flux
    uncertainties, plus the value-derived secondary inputs (TAUS/XNUE/BETAE).

    Note that the scanned variable is only the REPRESENTATIVE of the group that the
    corresponding powerstate DV actually moves: completeVariation_TGLF (wired as the
    TGLF run_specifications['complete_variation'] and applied inside _prepare_scan)
    co-varies with the same multiplier:
        RLTS_2      -> RLTS of all thermal ions
        RLNS_1      -> RLNS of all thermal ions (quasineutral with electrons)
        TAUS_2      -> TAUS of all thermal ions
        VEXB_SHEAR  -> VPAR_SHEAR of electrons + all thermal ions (both ~ d(w0)/dr)

    The minimum_delta_abs floor (same numerical value as for the gradients, in their
    respective TGLF units) avoids no-op scan members around zero values (e.g. aLn ~ 0.0
    or zero-rotation shears, which would otherwise report artificially zero stds). The
    floor is registered for every species position (TGLF supports up to 6) so the
    co-varied keys produced by completeVariation_TGLF are floored consistently with
    their representative.
    '''

    variables_to_scan = []
    for i in predicted_channels:
        if i == 'te': variables_to_scan.append('RLTS_1')
        if i == 'ti': variables_to_scan.append('RLTS_2')
        if i == 'ne': variables_to_scan.append('RLNS_1')
        if i == 'nZ': variables_to_scan.append(f'RLNS_{ion_OI_position_in_total_padded_list}')
        if i == 'w0': variables_to_scan.append('VEXB_SHEAR')

    #TODO: Only if that parameter is changing at that location
    if 'te' in predicted_channels or 'ti' in predicted_channels:
        variables_to_scan.append('TAUS_2')
    if 'te' in predicted_channels or 'ne' in predicted_channels:
        variables_to_scan.append('XNUE')
    if 'te' in predicted_channels or 'ne' in predicted_channels:
        variables_to_scan.append('BETAE')

    max_species_tglf = 6
    minimum_delta_abs = {}
    for ikey in variables_to_scan:
        if 'RL' in ikey:
            base = ikey.rsplit('_', 1)[0]   # RLTS / RLNS
            for j in range(1, max_species_tglf + 1):
                minimum_delta_abs[f'{base}_{j}'] = minimum_abs_gradient
        if ikey == 'VEXB_SHEAR':
            minimum_delta_abs['VEXB_SHEAR'] = minimum_abs_gradient
            for j in range(1, max_species_tglf + 1):
                minimum_delta_abs[f'VPAR_SHEAR_{j}'] = minimum_abs_gradient

    return variables_to_scan, minimum_delta_abs


def _run_tglf_uncertainty_model(
    tglf,
    rho_locations,
    predicted_channels,
    Flux_base = None,
    code_settings=None,
    extraOptions=None,
    ion_OI_position_in_ion_list=1,
    delta=0.02,
    minimum_abs_gradient=0.005, # This is 0.5% of aLx=1.0, to avoid extremely small scans when, for example, having aLn ~ 0.0
    cold_start=False,
    extra_name="",
    remove_folders_out = False,
    cores_per_tglf_instance = 4, # e.g. 4 core per radius, since this is going to launch ~ Nr=5 x (Nv=6 x Nd=2 + 1) = 65 TGLFs at once
    Qi_includes_fast=False,
    only_minimal_files=True,    # Since I only care about fluxes here, do not retrieve all the files
    reuse_scan_ball_file=None,      # If not None, it will reuse previous evaluations within the delta ball (to capture combinations)
    reuse_scan_ball_region='box',   # 'box' | 'ball'; see resolve_reuse_scan_ball()
    print_logs = False,
    subfolder_name = 'turb_drives',  # Base subfolder for the scan; per-plasma callers override to avoid collisions
    statistics="gaussian",          # Reduction of scan samples to (value, std): 'gaussian' or 'asymmetric' (see _aggregate_scan_fluxes)
    ):

    print(f"\t- Running TGLF standalone scans ({delta = }) to determine relative errors")

    ion_OI_position_in_total_padded_list = ion_OI_position_in_ion_list + 2 # Because in input.tglf 1 is electrons, and 2 is first ion, etc

    # Prepare scan (see _build_uncertainty_scan_setup for the channel -> variable
    # mapping, the completeVariation_TGLF co-variation and the zero-value floors)
    variables_to_scan, minimum_delta_abs = _build_uncertainty_scan_setup(
        predicted_channels, ion_OI_position_in_total_padded_list, minimum_abs_gradient
    )

    relative_scan = [1-delta, 1+delta]

    name = subfolder_name

    tglf.rhos = rho_locations # To avoid the case in which TGYRO was run with an extra rho point

    # Estimate job minutes based on cases and cores (mostly IO I think at this moment, otherwise it should be independent on cases)
    num_cases = len(rho_locations) * len(variables_to_scan) * len(relative_scan)
    if cores_per_tglf_instance == 1:
        minutes = 10 * (num_cases / 60) # Ad-hoc formula
    else:
        minutes = 1 * (num_cases / 60) # Ad-hoc formula

    # Enforce minimum minutes
    minutes = max(2, minutes)

    if print_logs:
        print_context = IOtools.nullcontext()
    else:
        print(f"\n\t- Running TGLF scans for turbulence drives (logs are hidden to avoid cluttering the log file)...\n", typeMsg="i")
        print_context = LOGtools.HiddenPrints(show_if_contains=["* Executing", "-------------- Running process", "- TGLF will be executed", "[in-process] Submitting"])

    with print_context:
        tglf.runScanTurbulenceDrives(
                        subfolder = name,
                        variablesDrives = variables_to_scan,
                        varUpDown     = relative_scan,
                        minimum_delta_abs = minimum_delta_abs,
                        code_settings = code_settings,
                        extraOptions = extraOptions,
                        ApplyCorrections = False,
                        add_baseline_to = 'none',
                        cold_start=cold_start,
                        forceIfcold_start=True,
                        allocation={
                            "resources_per_call": cores_per_tglf_instance,
                            "minutes": minutes,
                                     },
                        extra_name = f'{extra_name}_{name}',
                        ion_OI_position_in_total_padded_list=ion_OI_position_in_total_padded_list,
                        attempts_execution=2,
                        only_minimal_files=only_minimal_files,
                        )

    # Remove folders because they are heavy to carry many throughout
    if remove_folders_out:
        IOtools.shutil_rmtree(tglf.FolderGACODE)

    return _aggregate_scan_fluxes(
        tglf, name, variables_to_scan, relative_scan, rho_locations,
        Flux_base=Flux_base,
        ion_OI_position_in_ion_list=ion_OI_position_in_ion_list,
        Qi_includes_fast=Qi_includes_fast,
        reuse_scan_ball_file=reuse_scan_ball_file,
        reuse_scan_ball_region=reuse_scan_ball_region,
        delta=delta,
        statistics=statistics,
    )


def resolve_reuse_scan_ball(simulation_options):
    '''
    Region used to decide which previously-evaluated points may be reused as extra samples of
    the current cloud (see _ball_workflow):

        null / None   no reuse; the cloud is only this evaluation's one-at-a-time scan
        'box'         every input independently within +-delta (L-infinity). Corners reach
                      delta*sqrt(N) in relative L2, which is what lets MULTI-dimensional
                      combinations in -- the reason the reuse exists. Historical behaviour.
        'ball'        relative L2 deviation within delta. Consistent with the scan stencil,
                      whose points sit at exactly delta, but admits little beyond near-axis
                      points: two inputs at 0.75*delta each are already outside.

    Deprecated: the boolean form true/false maps to 'box'/None (the historical region was the
    per-dimension box, despite the name).
    '''
    region = simulation_options.get("reuse_scan_ball", None)

    if isinstance(region, bool):
        if region:
            print("\t- 'reuse_scan_ball: true' is deprecated, interpreting as 'box' "
                  "(the historical region is the per-dimension box, not an L2 ball)", typeMsg="w")
        region = "box" if region else None

    if region not in (None, "box", "ball"):
        raise ValueError(f"reuse_scan_ball must be null, 'box' or 'ball', not {region!r}")

    return region


def calculate_median_semistds(Q):
    '''
    Median and RMS semi-deviations per side about the median, over axis=1 of Q (nrho, n_samples).
    Exact ties with the median are excluded from both sides; an empty side (e.g. all samples
    equal) returns 0. This is THE semi-deviation convention of the 'asymmetric'
    uncertainty_statistics — plotting and any other consumer must use this function
    rather than reimplementing it.
    '''
    med = np.nanmedian(Q, axis=1)
    dev = Q - med[:, np.newaxis]

    def semi_deviation(mask):
        cnt = mask.sum(axis=1)
        s2 = np.where(mask, dev**2, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        return np.sqrt(np.where(cnt > 0, s2, 0.0))

    finite = ~np.isnan(dev)
    return med, semi_deviation((dev < 0) & finite), semi_deviation((dev > 0) & finite)


def _aggregate_scan_fluxes(
    tglf, scan_subfolder_name, variables_to_scan, relative_scan, rho_locations,
    Flux_base=None,
    ion_OI_position_in_ion_list=1,
    Qi_includes_fast=False,
    reuse_scan_ball_file=None,
    reuse_scan_ball_region='box',
    delta=0.02,
    statistics="gaussian",
):
    '''
    Shared aggregation logic: collect per-variable scan results from tglf.scans[...]
    into (nrho, n_scan_cases) flux arrays, optionally prepend the base flux, then
    reduce across the scan axis. Used by both the single-plasma
    _run_tglf_uncertainty_model and the batched _run_tglf_uncertainty_model_batched.

    statistics selects what feeds the (value, std) consumed downstream (GP noise, Ricci):
        gaussian:   value = nanmean, std = nanstd (legacy behavior, bit-for-bit)
        asymmetric: value = nanmedian, std = sqrt((sigma_minus^2 + sigma_plus^2)/2), with
                    RMS semi-deviations about the median per side — robust to the one-sided
                    outliers produced when a scan member crosses a critical-gradient jump.
    In BOTH modes the per-side semi-deviations and the raw samples are returned in
    Flux_extras, so the asymmetric information is available regardless of the trunk choice.
    '''

    Qe = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan) ))
    Qi = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan) ))
    Ge = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan) ))
    GZ = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan) ))
    Mt = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan) ))
    S = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan) ))

    cont = 0
    for vari in variables_to_scan:
        jump = tglf.scans[f'{scan_subfolder_name}_{vari}']['Qe'].shape[-1]

        Qe[:,cont:cont+jump] = tglf.scans[f'{scan_subfolder_name}_{vari}']['Qe_gb']
        Qi[:,cont:cont+jump] = tglf.scans[f'{scan_subfolder_name}_{vari}']['Qi_gb'] + (0 if not Qi_includes_fast else tglf.scans[f'{scan_subfolder_name}_{vari}']['Qifast_gb'])
        Ge[:,cont:cont+jump] = tglf.scans[f'{scan_subfolder_name}_{vari}']['Ge_gb']
        GZ[:,cont:cont+jump] = tglf.scans[f'{scan_subfolder_name}_{vari}']['Gi_gb']
        Mt[:,cont:cont+jump] = tglf.scans[f'{scan_subfolder_name}_{vari}']['Mt_gb']
        S[:,cont:cont+jump] = tglf.scans[f'{scan_subfolder_name}_{vari}']['S_gb']

        cont += jump

    if Qi_includes_fast:
        print(f"\t- Qi includes fast ions, adding their contribution")

    if Flux_base is not None:
        Qe = np.append(np.atleast_2d(Flux_base[0]).T, Qe, axis=1)
        Qi = np.append(np.atleast_2d(Flux_base[1]).T, Qi, axis=1)
        Ge = np.append(np.atleast_2d(Flux_base[2]).T, Ge, axis=1)
        GZ = np.append(np.atleast_2d(Flux_base[3]).T, GZ, axis=1)
        Mt = np.append(np.atleast_2d(Flux_base[4]).T, Mt, axis=1)
        S = np.append(np.atleast_2d(Flux_base[5]).T, S, axis=1)

    samples_from_ball = np.zeros(Qe.shape[1], dtype=bool)
    if reuse_scan_ball_file is not None:
        Qe, Qi, Ge, GZ, Mt, S, samples_from_ball = _ball_workflow(reuse_scan_ball_file, variables_to_scan, rho_locations, tglf, ion_OI_position_in_ion_list, Qi_includes_fast, Qe, Qi, Ge, GZ, Mt, S, delta_ball=delta, region=reuse_scan_ball_region)

    def calculate_mean_std(Q):
        return np.nanmean(Q, axis=1), np.nanstd(Q, axis=1)

    Flux_value, Flux_std, Flux_std_minus, Flux_std_plus, Flux_samples = [], [], [], [], []
    for Q in [Qe, Qi, Ge, GZ, Mt, S]:
        med, s_minus, s_plus = calculate_median_semistds(Q)
        if statistics == "asymmetric":
            point, std = med, np.sqrt(0.5 * (s_minus**2 + s_plus**2))
        else:
            point, std = calculate_mean_std(Q)
        Flux_value.append(point)
        Flux_std.append(std)
        Flux_std_minus.append(s_minus)
        Flux_std_plus.append(s_plus)
        Flux_samples.append(Q)

    Flux_extras = {
        "statistics": statistics,
        "stds_minus": Flux_std_minus,
        "stds_plus": Flux_std_plus,
        "samples": Flux_samples,
        # Per-sample provenance (False = this evaluation's one-at-a-time scan, True = reused
        # from the delta ball). Diagnostics only: the reduction treats both identically, since
        # ball points are, by construction, inside the same delta ball.
        "samples_from_ball": samples_from_ball,
    }

    return Flux_value, Flux_std, Flux_extras


def _run_tglf_uncertainty_model_batched(
    tglf,
    rho_locations,
    predicted_channels,
    Flux_base_per_plasma,
    plasma_labels,
    ion_OI_position_in_ion_list=1,
    delta=0.02,
    minimum_abs_gradient=0.005,
    cold_start=False,
    extra_name="",
    cores_per_tglf_instance=4,
    Qi_includes_fast=False,
    only_minimal_files=True,
    reuse_scan_ball_file=None,
    reuse_scan_ball_region='box',
    print_logs=False,
    code_settings=None,
    extraOptions=None,
    statistics="gaussian",
    **kwargs_run,
):
    '''
    Multi-plasma version of _run_tglf_uncertainty_model. Instead of calling
    runScanTurbulenceDrives once (which dispatches ~65 scan jobs for one plasma),
    this function accumulates scan work from ALL N plasmas into a single
    code_executor dict and dispatches ONE _run call — so all N × ~65 scan jobs
    run concurrently through FARMINGtools.
    '''

    print(f"\t- Running batched TGLF scans ({delta = }) for {len(plasma_labels)} plasmas in parallel")

    ion_OI_position_in_total_padded_list = ion_OI_position_in_ion_list + 2

    # Same scan setup as the single-plasma path
    variables_to_scan, minimum_delta_abs = _build_uncertainty_scan_setup(
        predicted_channels, ion_OI_position_in_total_padded_list, minimum_abs_gradient
    )

    relative_scan = [1 - delta, 1 + delta]

    N = len(plasma_labels)
    num_cases_total = N * len(rho_locations) * len(variables_to_scan) * len(relative_scan)
    if cores_per_tglf_instance == 1:
        minutes = 10 * (num_cases_total / 60)
    else:
        minutes = 1 * (num_cases_total / 60)
    minutes = max(2, minutes)

    tglf.rhos = rho_locations

    # varUpDown dict (no baseline added — same as add_baseline_to='none' in the single-plasma path)
    varUpDown_dict = {variable: np.array(relative_scan) for variable in variables_to_scan}

    # ---- Phase 1: Accumulate scan work for ALL plasmas ----
    code_executor, code_executor_full, folders_per_plasma = {}, {}, {}

    if print_logs:
        print_context = IOtools.nullcontext()
    else:
        print(f"\n\t- Preparing TGLF scans for {N} plasmas x {len(variables_to_scan)} variables x {len(relative_scan)} deltas x {len(rho_locations)} rhos = {num_cases_total} total jobs...\n", typeMsg="i")
        print_context = LOGtools.HiddenPrints(show_if_contains=["* Executing", "-------------- Running process", "- TGLF will be executed", "[in-process] Submitting"])

    scan_kwargs = dict(
        code_settings=code_settings,
        extraOptions=extraOptions or {},
        ApplyCorrections=False,
        cold_start=cold_start,
        forceIfcold_start=True,
        allocation={"resources_per_call": cores_per_tglf_instance, "minutes": minutes},
        only_minimal_files=only_minimal_files,
        **kwargs_run,
    )

    with print_context:
        for p, label in plasma_labels.items():
            tglf.read_plasma(p, label=label, require_all_files=False)
            tglf.variablesDrives = variables_to_scan
            folders_per_plasma[p] = {}

            for cont_var, variable in enumerate(variables_to_scan):
                scan_name = f"turb_drives_plasma{p}_{variable}"
                scan_kwargs["forceIfcold_start"] = (p > 0 or cont_var > 0) or scan_kwargs.get("forceIfcold_start", True)

                ce0, cef0, fld0, _ = tglf._prepare_scan(
                    scan_name,
                    variable=variable,
                    varUpDown=varUpDown_dict[variable],
                    minimum_delta_abs=minimum_delta_abs,
                    **scan_kwargs,
                )
                code_executor |= ce0
                code_executor_full |= cef0
                folders_per_plasma[p][variable] = fld0

        # ---- Phase 2: ONE dispatch of all N × vars × mults × rhos ----
        tglf._run(
            code_executor,
            code_executor_full=code_executor_full,
            extra_name=f"{extra_name}_turb_drives_batched",
            **scan_kwargs,
        )

    # ---- Phase 3: Read results per plasma and aggregate ----
    Flux_value_list, Flux_std_list, Flux_extras_list = [], [], []
    for p, label in plasma_labels.items():
        tglf.read_plasma(p, label=label, require_all_files=False)

        cont_folder = 0
        for variable in variables_to_scan:
            scan_name = f"turb_drives_plasma{p}_{variable}"
            folders = folders_per_plasma[p][variable]
            for mult_idx, mult in enumerate(varUpDown_dict[variable]):
                tglf.read(
                    label=f"{tglf.subfolder_scan}_{variable}_{mult}",
                    folder=folders[mult_idx],
                    cold_startWF=False,
                    require_all_files=not only_minimal_files,
                )
                cont_folder += 1

            tglf.read_scan(
                label=scan_name,
                variable=variable,
                ion_OI_position_in_total_padded_list=ion_OI_position_in_total_padded_list,
            )

        Fvalue_p, Fstd_p, Fextras_p = _aggregate_scan_fluxes(
            tglf,
            f"turb_drives_plasma{p}",
            variables_to_scan,
            relative_scan,
            rho_locations,
            Flux_base=Flux_base_per_plasma.get(p),
            ion_OI_position_in_ion_list=ion_OI_position_in_ion_list,
            Qi_includes_fast=Qi_includes_fast,
            reuse_scan_ball_file=reuse_scan_ball_file,
            reuse_scan_ball_region=reuse_scan_ball_region,
            delta=delta,
            statistics=statistics,
        )
        Flux_value_list.append(np.array(Fvalue_p))
        Flux_std_list.append(np.array(Fstd_p))
        Flux_extras_list.append(Fextras_p)

    # Stack: list of (6, nrho) -> (6, N, nrho); samples (6, nrho, n_samples) -> (6, N, nrho, n_samples)
    Flux_value = np.stack(Flux_value_list, axis=1)
    Flux_std  = np.stack(Flux_std_list,  axis=1)
    Flux_extras = {
        "statistics": statistics,
        "stds_minus": np.stack([np.array(e["stds_minus"]) for e in Flux_extras_list], axis=1),
        "stds_plus":  np.stack([np.array(e["stds_plus"])  for e in Flux_extras_list], axis=1),
        "samples":    np.stack([np.array(e["samples"])    for e in Flux_extras_list], axis=1),
    }

    return Flux_value, Flux_std, Flux_extras


def _ball_workflow(file, variables_to_scan, rho_locations, tglf, ion_OI_position_in_ion_list, Qi_includes_fast, Qe_orig, Qi_orig, Ge_orig, GZ_orig, Mt_orig, S_orig, delta_ball=0.02, region='box'):
    '''
    Workflow to reuse previous TGLF evaluations within a delta ball to capture combinations
    around the current base case.
    '''
    
    # Grab all inputs and outputs of the current run 
    input_params_keys = variables_to_scan
    input_params = np.zeros((len(rho_locations), len(input_params_keys), len(tglf.results)))
    
    input_params_base = np.zeros((len(rho_locations), len(input_params_keys)))
    
    output_params_keys = ['Qe', 'Qi', 'Ge', 'Gi', 'Mt', 'S']
    output_params = np.zeros((len(rho_locations), len(output_params_keys), len(tglf.results)))
    
    for i, key in enumerate(tglf.results.keys()):
        for irho in range(len(rho_locations)):
            
            # Grab all inputs in array with shape (Nr, Ninputs, Ncases)
            for ikey in range(len(input_params_keys)):
                input_params[irho, ikey, i] = tglf.results[key]['parsed'][irho][input_params_keys[ikey]]
            
            # Grab base inputs in array with shape (Nr, Ninputs)
            if key == 'base':
                for ikey in range(len(input_params_keys)):
                    input_params_base[irho, ikey] = tglf.results[key]['parsed'][irho][input_params_keys[ikey]]
              
            # Grab all outputs in array with shape (Nr, Noutputs, Ncases)
            output_params[irho, 0, i] = tglf.results[key]['output'][irho].Qe
            output_params[irho, 1, i] = tglf.results[key]['output'][irho].Qi + (0 if not Qi_includes_fast else tglf.results[key]['output'][irho].Qifast)
            output_params[irho, 2, i] = tglf.results[key]['output'][irho].Ge
            output_params[irho, 3, i] = tglf.results[key]['output'][irho].GiAll[ion_OI_position_in_ion_list]
            output_params[irho, 4, i] = tglf.results[key]['output'][irho].Mt
            output_params[irho, 5, i] = tglf.results[key]['output'][irho].Se
    
    # --------------------------------------------------------------------------------------------------------
    # Read previous ball and append
    # --------------------------------------------------------------------------------------------------------

    rho_ball = np.array([])
    input_ball = np.array([])
    output_ball = np.array([])

    if Path(file).exists():

        print(f"\t- Reusing previous TGLF scan evaluations within the delta ball to capture combinations")

        # Grab ball contents
        with np.load(file) as data:
            rho_ball = data['rho']
            input_ball = data['input_params']
            output_ball = data['output_params']
            input_keys_ball = data['input_keys'] if 'input_keys' in data else None

        # The npz stores the inputs positionally: if the scanned-variable set changed
        # since the ball was written (e.g. different predicted channels on a restart),
        # the stored inputs would be silently misinterpreted -- discard the ball instead.
        # (input_keys was added later: files without it are only checked by count)
        if (input_keys_ball is not None and list(input_keys_ball) != list(input_params_keys)) or \
           (input_keys_ball is None and input_ball.shape[1] != len(input_params_keys)):
            print(
                f"\t\t- Ball file was written with different scan variables "
                f"({list(input_keys_ball) if input_keys_ball is not None else input_ball.shape[1]} vs {list(input_params_keys)}); "
                "discarding it and starting a fresh ball", typeMsg="w",
            )
            rho_ball = np.array([])
            input_ball = np.array([])
            output_ball = np.array([])

    n_orig = Qe_orig.shape[1]

    if rho_ball.shape[0] != 0:

        precision_check = 1E-5 # I needed to add a small number to avoid numerical issues because TGLF input files have limited precision
            
        # Get the indeces of the points within the delta ball (condition in which all inputs are within the delta of the base case for that specific radius)
        indices_to_grab = {}
        for irho in range(len(rho_locations)):
            indices_to_grab[irho] = []
            inputs_base = input_params_base[irho, :]
            for icase in range(input_ball.shape[-1]):
                inputs_case = input_ball[irho, :, icase]
                
                # Membership in the reuse region (see resolve_reuse_scan_ball):
                #   'box'  every input independently within +-delta_ball (L-infinity)
                #   'ball' relative L2 deviation within delta_ball
                if region == 'ball':
                    scale = np.where(np.abs(inputs_base) > 0, np.abs(inputs_base), np.nan)
                    rel = (inputs_case - inputs_base) / scale
                    # An input that is zero at the base only contributes if the case matches it
                    rel = np.where(np.isnan(rel), np.where(np.abs(inputs_case - inputs_base) <= precision_check, 0.0, np.inf), rel)
                    is_this_within_ball = np.linalg.norm(rel) <= delta_ball + precision_check
                else:
                    is_this_within_ball = True
                    for ikey in range(len(input_params_keys)):
                        val_current = inputs_base[ikey]
                        val_ball = inputs_case[ikey]

                        # I need to have all inputs within the delta ball
                        is_this_within_ball = is_this_within_ball and ( abs(val_current-val_ball) <= abs(val_current*delta_ball) + precision_check )
                    
                if is_this_within_ball:
                    indices_to_grab[irho].append(icase)

            print(f"\t\t- Out of {input_ball.shape[-1]} points in file, found {len(indices_to_grab[irho])} at location {irho} within the delta {region} ({delta_ball*100}%)", typeMsg="i" if len(indices_to_grab[irho]) > 0 else "")
        
        # Make an output_ball_select array equivalent to output_ball but only with the points within the delta ball (rest make them NaN)
        output_ball_select = np.full_like(output_ball, np.nan)
        for irho in range(len(rho_locations)):
            for icase in indices_to_grab[irho]:
                output_ball_select[irho, :, icase] = output_ball[irho, :, icase]

        # Append those points to the current run (these will always have shape (Nr, Ncases+original) but those cases that were not in the ball will be NaN)
        # The reason to do it this way is that I want to keep it as a uniform shape to be able to calculate stds later, and I would risk otherwise having different shapes per radius
        Qe = np.append(Qe_orig, output_ball_select[:, 0, :], axis=1)
        Qi = np.append(Qi_orig, output_ball_select[:, 1, :], axis=1)
        Ge = np.append(Ge_orig, output_ball_select[:, 2, :], axis=1)
        GZ = np.append(GZ_orig, output_ball_select[:, 3, :], axis=1)
        Mt = np.append(Mt_orig, output_ball_select[:, 4, :], axis=1)
        S = np.append(S_orig, output_ball_select[:, 5, :], axis=1)
        print(f"\t\t>>> Flux arrays have shape {Qe.shape} after appending ball points (NaNs are added to those locations and cases that did not fall within delta ball)")
        
        # Remove repeat points (for example when transitioning from simple relaxation initialization to full optimization)
        def remove_duplicate_cases(*arrays):
            """Remove duplicate cases (columns) from arrays of shape (rho_size, cases_size)
            
            Returns:
                tuple: (unique_arrays, duplicate_arrays) where each is a tuple of arrays
            """

            # Stack all arrays to create a combined signature for each case
            combined = np.vstack(arrays)  # Shape: (total_channels, cases_size)

            # Find unique cases, handling NaN values properly (Use pandas for robust duplicate detection with NaN support)
            df = pd.DataFrame(combined.T)  # Transpose so each row is a case
            unique_indices = df.drop_duplicates().index.values
            nan_indices = np.where(np.all(np.isnan(combined), axis=0))[0]
            unique_notnan_indeces = [idx for idx in unique_indices if idx not in nan_indices]            
            all_indices = np.arange(combined.shape[1])
            duplicate_indices = np.setdiff1d(all_indices, unique_notnan_indeces)
            
            print(f"\t\t* Removed {len(duplicate_indices)} duplicate / all-nan cases, keeping {len(unique_notnan_indeces)} unique cases", typeMsg="i")
            
            # Return arrays with unique cases and duplicate cases
            unique_arrays = tuple(arr[:, unique_notnan_indeces] for arr in arrays)
            duplicate_arrays = tuple(arr[:, duplicate_indices] for arr in arrays)
            
            return unique_arrays, duplicate_arrays, np.array(unique_notnan_indeces, dtype=int)
        
        unique_results, duplicate_results, kept_indices = remove_duplicate_cases(Qe, Qi, Ge, GZ, Mt, S)
        
        Qe, Qi, Ge, GZ, Mt, S = unique_results

        # Provenance of each surviving column: False = this evaluation's one-at-a-time scan,
        # True = reused from the delta ball. Diagnostics only (plots), never used in the
        # statistics -- ball points are legitimate members of the local cloud.
        from_ball = kept_indices >= n_orig
        
        print(f"\t\t>>> Flux arrays have shape {Qe.shape} after finding unique points")

    else:

        Qe, Qi, Ge, GZ, Mt, S = Qe_orig, Qi_orig, Ge_orig, GZ_orig, Mt_orig, S_orig
        from_ball = np.zeros(n_orig, dtype=bool)

    # --------------------------------------------------------------------------------------------------------
    # Save new ball
    # --------------------------------------------------------------------------------------------------------
    
    # Append to the values read from previous ball
    if rho_ball.shape[0] != 0:
        input_params = np.append(input_ball, input_params, axis=2)
        output_params = np.append(output_ball, output_params, axis=2)

    # Save the new ball (input_keys records the positional meaning of input_params,
    # so a future run with different scan variables can detect the mismatch)
    np.savez(file, rho=rho_locations, input_params=input_params, output_params=output_params,
             input_keys=np.array(input_params_keys))
    print(f"\t- Saved updated ball with {input_params.shape[-1]} points to {IOtools.clipstr(file)}", typeMsg="i")
    
    return Qe, Qi, Ge, GZ, Mt, S, from_ball
