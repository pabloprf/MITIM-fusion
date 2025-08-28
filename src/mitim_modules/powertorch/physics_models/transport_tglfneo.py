
import shutil
import numpy as np
from mitim_tools.misc_tools import IOtools
from functools import partial
from mitim_tools.gacode_tools import TGLFtools, NEOtools
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class tglfneo_model(TRANSPORTtools.power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        self._produce_profiles()
        
    # ************************************************************************************
    # Private functions for the evaluation
    # ************************************************************************************
    
    @IOtools.hook_method(after=partial(TRANSPORTtools.write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self):        
        self._evaluate_tglf()

    # Have it separate such that I can call it from the CGYRO class but without the decorator
    def _evaluate_tglf(self):
        
        transport_evaluator_options = self.powerstate.transport_options["options"]
        cold_start = self.powerstate.transport_options["cold_start"]
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options from powerstate
        # ------------------------------------------------------------------------------------------------------------------------

        simulation_options_tglf = transport_evaluator_options["tglf"]
        
        Qi_includes_fast = simulation_options_tglf["Qi_includes_fast"]
        launchMODELviaSlurm = simulation_options_tglf["launchEvaluationsAsSlurmJobs"]
        use_tglf_scan_trick = simulation_options_tglf["use_scan_trick_for_stds"]
        cores_per_tglf_instance = simulation_options_tglf["cores_per_tglf_instance"]
        keep_tglf_files = simulation_options_tglf["keep_files"]
        percent_error = simulation_options_tglf["percent_error"]

        # Grab impurity from powerstate ( because it may have been modified in produce_profiles() )
        impurityPosition = self.powerstate.impurityPosition_transport
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare TGLF object
        # ------------------------------------------------------------------------------------------------------------------------
        
        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        tglf = TGLFtools.TGLF(rhos=rho_locations)

        _ = tglf.prep(
            self.powerstate.profiles_transport,
            self.folder,
            cold_start = cold_start,
            )
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Run TGLF (base)
        # ------------------------------------------------------------------------------------------------------------------------

        tglf.run(
            'base_tglf',
            ApplyCorrections=False,
            launchSlurm= launchMODELviaSlurm,
            cold_start= cold_start,
            forceIfcold_start=True,
            extra_name= self.name,
            slurm_setup={
                "cores": cores_per_tglf_instance,      
                "minutes": 2,
                },
            attempts_execution=2,
            only_minimal_files=keep_tglf_files in ['minimal'],
            **simulation_options_tglf["run"]
        )
    
        tglf.read(
            label='base',
            require_all_files=False,
            **simulation_options_tglf["read"])
        
        # Grab values
        Qe = np.array([tglf.results['base']['TGLFout'][i].Qe for i in range(len(rho_locations))])
        Qi = np.array([tglf.results['base']['TGLFout'][i].Qi for i in range(len(rho_locations))])
        Ge = np.array([tglf.results['base']['TGLFout'][i].Ge for i in range(len(rho_locations))])
        GZ = np.array([tglf.results['base']['TGLFout'][i].GiAll[impurityPosition] for i in range(len(rho_locations))])
        Mt = np.array([tglf.results['base']['TGLFout'][i].Mt for i in range(len(rho_locations))])
        S = np.array([tglf.results['base']['TGLFout'][i].Se for i in range(len(rho_locations))])

        if Qi_includes_fast:
            
            Qifast = [tglf.results['base']['TGLFout'][i].Qifast for i in range(len(rho_locations))]
            
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
            
            Flux_mean = Flux_base
            Flux_std = abs(Flux_mean)*percent_error/100.0

        else:
            
            # *******************************************************************
            # Run TGLF with scans to estimate the uncertainty
            # *******************************************************************
            
            Flux_mean, Flux_std = _run_tglf_uncertainty_model(
                tglf,
                rho_locations, 
                self.powerstate.predicted_channels, 
                Flux_base = Flux_base,
                impurityPosition=impurityPosition, 
                delta = use_tglf_scan_trick,
                cold_start=cold_start,
                extra_name=self.name,
                cores_per_tglf_instance=cores_per_tglf_instance,
                launchMODELviaSlurm=launchMODELviaSlurm,
                Qi_includes_fast=Qi_includes_fast,
                only_minimal_files=keep_tglf_files in ['minimal', 'base'],
                **simulation_options_tglf["run"]
                )

        self._raise_warnings(tglf, rho_locations, Qi_includes_fast)

        # ------------------------------------------------------------------------------------------------------------------------
        # Pass the information to what power_transport expects
        # ------------------------------------------------------------------------------------------------------------------------
        
        self.QeGB_turb = Flux_mean[0]
        self.QeGB_turb_stds = Flux_std[0]
                
        self.QiGB_turb = Flux_mean[1]
        self.QiGB_turb_stds = Flux_std[1]
                
        self.GeGB_turb = Flux_mean[2]
        self.GeGB_turb_stds = Flux_std[2]        
        
        self.GZGB_turb = Flux_mean[3]           
        self.GZGB_turb_stds = Flux_std[3]       

        self.MtGB_turb = Flux_mean[4]
        self.MtGB_turb_stds = Flux_std[4] 

        self.QieGB_turb = Flux_mean[5]
        self.QieGB_turb_stds = Flux_std[5]

        return tglf

    @IOtools.hook_method(after=partial(TRANSPORTtools.write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))
    def evaluate_neoclassical(self):
        
        transport_evaluator_options = self.powerstate.transport_options["options"]
        cold_start = self.powerstate.transport_options["cold_start"]
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options from powerstate
        # ------------------------------------------------------------------------------------------------------------------------
        
        simulation_options_neo = transport_evaluator_options["neo"]
        percent_error = simulation_options_neo["percent_error"]
        impurityPosition = self.powerstate.impurityPosition_transport
                
        # ------------------------------------------------------------------------------------------------------------------------        
        # Run
        # ------------------------------------------------------------------------------------------------------------------------
        
        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        neo = NEOtools.NEO(rhos=rho_locations)

        _ = neo.prep(
            self.powerstate.profiles_transport,
            self.folder,
            cold_start = cold_start,
            )
        
        neo.run(
            'base_neo',
            cold_start=cold_start,
            forceIfcold_start=True,
            **simulation_options_neo["run"]
        )
    
        neo.read(
            label='base',
            **simulation_options_neo["read"])
        
        Qe = np.array([neo.results['base']['NEOout'][i].Qe for i in range(len(rho_locations))])
        Qi = np.array([neo.results['base']['NEOout'][i].Qi for i in range(len(rho_locations))])
        Ge = np.array([neo.results['base']['NEOout'][i].Ge for i in range(len(rho_locations))])
        GZ = np.array([neo.results['base']['NEOout'][i].GiAll[impurityPosition-1] for i in range(len(rho_locations))])
        Mt = np.array([neo.results['base']['NEOout'][i].Mt for i in range(len(rho_locations))])
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Pass the information to what power_transport expects
        # ------------------------------------------------------------------------------------------------------------------------
        
        self.QeGB_neoc = Qe
        self.QiGB_neoc = Qi
        self.GeGB_neoc = Ge
        self.GZGB_neoc = GZ
        self.MtGB_neoc = Mt
        
        # Uncertainties is just a percent of the value
        self.QeGB_neoc_stds = abs(Qe) * percent_error/100.0
        self.QiGB_neoc_stds = abs(Qi) * percent_error/100.0
        self.GeGB_neoc_stds = abs(Ge) * percent_error/100.0
        self.GZGB_neoc_stds = abs(GZ) * percent_error/100.0
        self.MtGB_neoc_stds = abs(Mt) * percent_error/100.0

        # No neoclassical exchange
        self.QieGB_neoc = Qe * 0.0
        self.QieGB_neoc_stds = Qe * 0.0

        return neo


    def _raise_warnings(self, tglf, rho_locations, Qi_includes_fast):

        for i in range(len(tglf.profiles.Species)):
            gacode_type = tglf.profiles.Species[i]['S']
            for rho in rho_locations:
                tglf_type = tglf.inputs_files[rho].ions_info[i+2]['type']
                
                if gacode_type[:5] != tglf_type[:5]:
                    print(f"\t- For location {rho=:.2f}, ion specie #{i+1} ({tglf.profiles.Species[i]['N']}) is considered '{gacode_type}' by gacode but '{tglf_type}' by TGLF. Make sure this is consistent with your use case", typeMsg="w")
        
                    if tglf_type == 'fast':
        
                        if Qi_includes_fast:
                            print(f"\t\t\t* The fast ion considered by TGLF was summed into the Qi", typeMsg="i")
                        else:
                            print(f"\t\t\t* The fast ion considered by TGLF was NOT summed into the Qi", typeMsg="i")
                            
                    else:
                        print(f"\t\t\t* The thermal ion considered by TGLF was summed into the Qi", typeMsg="i")


def _run_tglf_uncertainty_model(
    tglf,
    rho_locations, 
    predicted_channels, 
    Flux_base = None,
    code_settings=None,
    extraOptions=None,
    impurityPosition=1,
    delta=0.02, 
    minimum_abs_gradient=0.005, # This is 0.5% of aLx=1.0, to avoid extremely small scans when, for example, having aLn ~ 0.0
    cold_start=False, 
    extra_name="", 
    remove_folders_out = False,
    cores_per_tglf_instance = 4, # e.g. 4 core per radius, since this is going to launch ~ Nr=5 x (Nv=6 x Nd=2 + 1) = 65 TGLFs at once
    launchMODELviaSlurm=False,
    Qi_includes_fast=False,
    only_minimal_files=True,    # Since I only care about fluxes here, do not retrieve all the files
    ):

    print(f"\t- Running TGLF standalone scans ({delta = }) to determine relative errors")

    # Prepare scan 
    variables_to_scan = []
    for i in predicted_channels:
        if i == 'te': variables_to_scan.append('RLTS_1')
        if i == 'ti': variables_to_scan.append('RLTS_2')
        if i == 'ne': variables_to_scan.append('RLNS_1')
        if i == 'nZ': variables_to_scan.append(f'RLNS_{impurityPosition+2}')
        if i == 'w0': variables_to_scan.append('VEXB_SHEAR') #TODO: is this correct? or VPAR_SHEAR?

    #TODO: Only if that parameter is changing at that location
    if 'te' in predicted_channels or 'ti' in predicted_channels:
        variables_to_scan.append('TAUS_2')
    if 'te' in predicted_channels or 'ne' in predicted_channels:
        variables_to_scan.append('XNUE')
    if 'te' in predicted_channels or 'ne' in predicted_channels:
        variables_to_scan.append('BETAE')
    
    relative_scan = [1-delta, 1+delta]

    # Enforce at least "minimum_abs_gradient" in gradient, to avoid zero gradient situations
    minimum_delta_abs = {}
    for ikey in variables_to_scan:
        if 'RL' in ikey:
            minimum_delta_abs[ikey] = minimum_abs_gradient

    name = 'turb_drives'

    tglf.rhos = rho_locations # To avoid the case in which TGYRO was run with an extra rho point

    # Estimate job minutes based on cases and cores (mostly IO I think at this moment, otherwise it should be independent on cases)
    num_cases = len(rho_locations) * len(variables_to_scan) * len(relative_scan)
    if cores_per_tglf_instance == 1:
        minutes = 10 * (num_cases / 60) # Ad-hoc formula
    else:
        minutes = 1 * (num_cases / 60) # Ad-hoc formula

    # Enforce minimum minutes
    minutes = max(2, minutes)

    tglf.runScanTurbulenceDrives(	
                    subfolder = name,
                    variablesDrives = variables_to_scan,
                    varUpDown     = relative_scan,
                    minimum_delta_abs = minimum_delta_abs,
                    TGLFsettings = code_settings,
                    extraOptions = extraOptions,
                    ApplyCorrections = False,
                    add_baseline_to = 'none',
                    cold_start=cold_start,
                    forceIfcold_start=True,
                    slurm_setup={
                        "cores": cores_per_tglf_instance,      
                        "minutes": minutes,
                                 },
                    extra_name = f'{extra_name}_{name}',
                    positionIon=impurityPosition+2,
                    attempts_execution=2, 
                    only_minimal_files=only_minimal_files,
                    launchSlurm=launchMODELviaSlurm,
                    )

    # Remove folders because they are heavy to carry many throughout
    if remove_folders_out:
        IOtools.shutil_rmtree(tglf.FolderGACODE)

    Qe = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))
    Qi = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))
    Qifast = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))
    Ge = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))
    GZ = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))
    Mt = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))
    S = np.zeros((len(rho_locations), len(variables_to_scan)*len(relative_scan)+1 ))

    cont = 0
    for vari in variables_to_scan:
        jump = tglf.scans[f'{name}_{vari}']['Qe'].shape[-1]

        Qe[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qe_gb']
        Qi[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qi_gb']
        Qifast[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qifast_gb']
        Ge[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Ge_gb']
        GZ[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Gi_gb']
        Mt[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Mt_gb']
        S[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['S_gb']
        cont += jump
        
    if Qi_includes_fast:
        print(f"\t- Qi includes fast ions, adding their contribution")
        Qi += Qifast

    # Add the base that was calculated earlier
    if Flux_base is not None:
        Qe = np.append(np.atleast_2d(Flux_base[0]).T, Qe, axis=1)
        Qi = np.append(np.atleast_2d(Flux_base[1]).T, Qi, axis=1)
        Ge = np.append(np.atleast_2d(Flux_base[2]).T, Ge, axis=1)
        GZ = np.append(np.atleast_2d(Flux_base[3]).T, GZ, axis=1)
        Mt = np.append(np.atleast_2d(Flux_base[4]).T, Mt, axis=1)
        S = np.append(np.atleast_2d(Flux_base[5]).T, S, axis=1)

    # Calculate the standard deviation of the scans, that's going to be the reported stds

    def calculate_mean_std(Q):
        # Assumes Q is [radii, points], with [radii, 0] being the baseline

        Qm = np.mean(Q, axis=1)
        Qstd = np.std(Q, axis=1)

        # Qm = Q[:,0]
        # Qstd = np.std(Q, axis=1)

        # Qstd    = ( Q.max(axis=1)-Q.min(axis=1) )/2 /2  # Such that the range is 2*std
        # Qm      = Q.min(axis=1) + Qstd*2                # Mean is at the middle of the range

        return  Qm, Qstd

    Qe_point, Qe_std = calculate_mean_std(Qe)
    Qi_point, Qi_std = calculate_mean_std(Qi)
    Ge_point, Ge_std = calculate_mean_std(Ge)
    GZ_point, GZ_std = calculate_mean_std(GZ)
    Mt_point, Mt_std = calculate_mean_std(Mt)
    S_point, S_std = calculate_mean_std(S)

    #TODO: Careful with fast particles
    Flux_mean = [Qe_point, Qi_point, Ge_point, GZ_point, Mt_point, S_point]
    Flux_std  = [Qe_std, Qi_std, Ge_std, GZ_std, Mt_std, S_std]

    return Flux_mean, Flux_std
