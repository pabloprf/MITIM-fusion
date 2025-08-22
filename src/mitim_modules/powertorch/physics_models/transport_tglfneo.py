
import shutil
import numpy as np
from mitim_tools.misc_tools import IOtools
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

    def evaluate_turbulence(self):

        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options from powerstate
        # ------------------------------------------------------------------------------------------------------------------------

        transport_evaluator_options = self.powerstate.transport_options["transport_evaluator_options"]
        
        TGLFsettings = transport_evaluator_options["MODELparameters"]["transport_model"]["TGLFsettings"]
        extraOptions = transport_evaluator_options["MODELparameters"]["transport_model"]["extraOptionsTGLF"]
        
        Qi_includes_fast = transport_evaluator_options.get("Qi_includes_fast",False)
        launchMODELviaSlurm = transport_evaluator_options.get("launchMODELviaSlurm", False)
        cold_start = transport_evaluator_options.get("cold_start", False)
        provideTurbulentExchange = transport_evaluator_options.get("TurbulentExchange", False)
        percentError = transport_evaluator_options.get("percentError", [5, 1, 0.5])
        use_tglf_scan_trick = transport_evaluator_options.get("use_tglf_scan_trick", None)
        cores_per_tglf_instance = transport_evaluator_options.get("extra_params", {}).get('PORTALSparameters', {}).get("cores_per_tglf_instance", 1)
        
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
        # Run TGLF
        # ------------------------------------------------------------------------------------------------------------------------
        
        if use_tglf_scan_trick is None:
            
                # *******************************************************************
                # Just run TGLF once and apply an ad-hoc percent error to the results
                # *******************************************************************
            
                tglf.run(
                    'base_tglf',
                    Settings=TGLFsettings,
                    extraOptions=extraOptions,
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
                    only_minimal_files=True,
                )
            
                tglf.read(label='base',require_all_files=False)
                
                Qe = np.array([tglf.results['base']['TGLFout'][i].Qe_unn for i in range(len(rho_locations))])
                Qi = np.array([tglf.results['base']['TGLFout'][i].Qi_unn for i in range(len(rho_locations))])
                Ge = np.array([tglf.results['base']['TGLFout'][i].Ge_unn for i in range(len(rho_locations))])
                GZ = np.array([tglf.results['base']['TGLFout'][i].GiAll_unn[impurityPosition] for i in range(len(rho_locations))])
                Mt = np.array([tglf.results['base']['TGLFout'][i].Mt_unn for i in range(len(rho_locations))])
                S = np.array([tglf.results['base']['TGLFout'][i].Se_unn for i in range(len(rho_locations))])

                if Qi_includes_fast:
                    
                    Qifast = [tglf.results['base']['TGLFout'][i].Qifast_unn for i in range(len(rho_locations))]
                    
                    if Qifast.sum() != 0.0:
                        print(f"\t- Qi includes fast ions, adding their contribution")
                        Qi += Qifast
                
                Flux_mean = np.array([Qe, Qi, Ge, GZ, Mt, S])
                Flux_std = abs(Flux_mean)*percentError[0]/100.0

        else:
            
            # *******************************************************************
            # Run TGLF with scans to estimate the uncertainty
            # *******************************************************************
            
            Flux_base, Flux_mean, Flux_std = _run_tglf_uncertainty_model(
                tglf,
                rho_locations, 
                self.powerstate.ProfilesPredicted, 
                TGLFsettings=TGLFsettings,
                extraOptionsTGLF=extraOptions,
                impurityPosition=impurityPosition, 
                delta = use_tglf_scan_trick,
                cold_start=cold_start,
                extra_name=self.name,
                cores_per_tglf_instance=cores_per_tglf_instance,
                launchMODELviaSlurm=launchMODELviaSlurm,
                Qi_includes_fast=Qi_includes_fast,
                )

        self._raise_warnings(tglf, rho_locations, Qi_includes_fast)

        # ------------------------------------------------------------------------------------------------------------------------
        # Pass the information to POWERSTATE
        # ------------------------------------------------------------------------------------------------------------------------
        
        self.powerstate.plasma["QeMWm2_tr_turb"] = Flux_mean[0]
        self.powerstate.plasma["QeMWm2_tr_turb_stds"] = Flux_std[0]
                
        self.powerstate.plasma["QiMWm2_tr_turb"] = Flux_mean[1]
        self.powerstate.plasma["QiMWm2_tr_turb_stds"] = Flux_std[1]
                
        self.powerstate.plasma["Ge1E20m2_tr_turb"] = Flux_mean[2]
        self.powerstate.plasma["Ge1E20m2_tr_turb_stds"] = Flux_std[2]        
        
        self.powerstate.plasma["GZ1E20m2_tr_turb"] = Flux_mean[3]           
        self.powerstate.plasma["GZ1E20m2_tr_turb_stds"] = Flux_std[3]       

        self.powerstate.plasma["MtJm2_tr_turb"] = Flux_mean[4]
        self.powerstate.plasma["MtJm2_tr_turb_stds"] = Flux_std[4] 

        if provideTurbulentExchange:
            self.powerstate.plasma["QieMWm3_tr_turb"] = Flux_mean[5]
            self.powerstate.plasma["QieMWm3_tr_turb_stds"] = Flux_std[5]
        else:
            self.powerstate.plasma["QieMWm3_tr_turb"] = Flux_mean[5] * 0.0
            self.powerstate.plasma["QieMWm3_tr_turb_stds"] = Flux_std[5] * 0.0

        return tglf

    def evaluate_neoclassical(self):
        
        # Options
        
        transport_evaluator_options = self.powerstate.transport_options["transport_evaluator_options"]
        
        cold_start = transport_evaluator_options.get("cold_start", False)
        percentError = transport_evaluator_options.get("percentError", [5, 1, 0.5])
        impurityPosition = self.powerstate.impurityPosition_transport
                
        # Run
        
        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        neo = NEOtools.NEO(rhos=rho_locations)

        _ = neo.prep(
            self.powerstate.profiles_transport,
            self.folder,
            cold_start = cold_start,
            )
        
        neo.run('base_neo')
    
        neo.read(label='base')
        
        Qe = np.array([neo.results['base']['NEOout'][i].Qe_unn for i in range(len(rho_locations))])
        Qi = np.array([neo.results['base']['NEOout'][i].Qi_unn for i in range(len(rho_locations))])
        Ge = np.array([neo.results['base']['NEOout'][i].Ge_unn for i in range(len(rho_locations))])
        GZ = np.array([neo.results['base']['NEOout'][i].GiAll_unn[impurityPosition] for i in range(len(rho_locations))])
        Mt = np.array([neo.results['base']['NEOout'][i].Mt_unn for i in range(len(rho_locations))])
        
        self.powerstate.plasma["QeMWm2_tr_neoc"] = Qe
        self.powerstate.plasma["QiMWm2_tr_neoc"] = Qi
        self.powerstate.plasma["Ge1E20m2_tr_neoc"] = Ge
        self.powerstate.plasma["GZ1E20m2_tr_neoc"] = GZ
        self.powerstate.plasma["MtJm2_tr_neoc"] = Mt
        
        self.powerstate.plasma["QeMWm2_tr_neoc_stds"] = abs(Qe) * percentError[1]/100.0
        self.powerstate.plasma["QiMWm2_tr_neoc_stds"] = abs(Qi) * percentError[1]/100.0
        self.powerstate.plasma["Ge1E20m2_tr_neoc_stds"] = abs(Ge) * percentError[1]/100.0
        self.powerstate.plasma["GZ1E20m2_tr_neoc_stds"] = abs(GZ) * percentError[1]/100.0
        self.powerstate.plasma["MtJm2_tr_neoc_stds"] = abs(Mt) * percentError[1]/100.0

        self.powerstate.plasma["QieMWm3_tr_neoc"] = Qe * 0.0
        self.powerstate.plasma["QieMWm3_tr_neoc_stds"] = Qe * 0.0

        return neo
                
    def _profiles_to_store(self):

        if "extra_params" in self.powerstate.transport_options["transport_evaluator_options"] and "folder" in self.powerstate.transport_options["transport_evaluator_options"]["extra_params"]:
            whereFolder = IOtools.expandPath(self.powerstate.transport_options["transport_evaluator_options"]["extra_params"]["folder"] / "Outputs" / "portals_profiles")
            if not whereFolder.exists():
                IOtools.askNewFolder(whereFolder)

            fil = whereFolder / f"input.gacode.{self.evaluation_number}"
            shutil.copy2(self.file_profs, fil)
            shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
            print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
        else:
            print("\t- Could not move files", typeMsg="w")

    def _raise_warnings(self, tglf, rho_locations, Qi_includes_fast):

        for i in range(len(tglf.profiles.Species)):
            gacode_type = tglf.profiles.Species[i]['S']
            for rho in rho_locations:
                tglf_type = tglf.inputs_files[0.25].ions_info[i+2]['type']
                
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
    ProfilesPredicted, 
    TGLFsettings=None,
    extraOptionsTGLF=None,
    impurityPosition=1,
    delta=0.02, 
    minimum_abs_gradient=0.005, # This is 0.5% of aLx=1.0, to avoid extremely small scans when, for example, having aLn ~ 0.0
    cold_start=False, 
    extra_name="", 
    remove_folders_out = False,
    cores_per_tglf_instance = 4, # e.g. 4 core per radius, since this is going to launch ~ Nr=5 x (Nv=6 x Nd=2 + 1) = 65 TGLFs at once
    launchMODELviaSlurm=False,
    Qi_includes_fast=False,
    ):

    print(f"\t- Running TGLF standalone scans ({delta = }) to determine relative errors")

    # Prepare scan 
    variables_to_scan = []
    for i in ProfilesPredicted:
        if i == 'te': variables_to_scan.append('RLTS_1')
        if i == 'ti': variables_to_scan.append('RLTS_2')
        if i == 'ne': variables_to_scan.append('RLNS_1')
        if i == 'nZ': variables_to_scan.append(f'RLNS_{impurityPosition+2}')
        if i == 'w0': variables_to_scan.append('VEXB_SHEAR') #TODO: is this correct? or VPAR_SHEAR?

    #TODO: Only if that parameter is changing at that location
    if 'te' in ProfilesPredicted or 'ti' in ProfilesPredicted:
        variables_to_scan.append('TAUS_2')
    if 'te' in ProfilesPredicted or 'ne' in ProfilesPredicted:
        variables_to_scan.append('XNUE')
    if 'te' in ProfilesPredicted or 'ne' in ProfilesPredicted:
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
                    TGLFsettings = TGLFsettings,
                    extraOptions = extraOptionsTGLF,
                    ApplyCorrections = False,
                    add_baseline_to = 'first',
                    cold_start=cold_start,
                    forceIfcold_start=True,
                    slurm_setup={
                        "cores": cores_per_tglf_instance,      
                        "minutes": minutes,
                                 },
                    extra_name = f'{extra_name}_{name}',
                    positionIon=impurityPosition+2,
                    attempts_execution=2, 
                    only_minimal_files=True,    # Since I only care about fluxes here, do not retrieve all the files
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

        Qe[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qe']
        Qi[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qi']
        Qifast[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qifast']
        Ge[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Ge']
        GZ[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Gi']
        Mt[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Mt']
        S[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['S']
        cont += jump
        
    if Qi_includes_fast:
        print(f"\t- Qi includes fast ions, adding their contribution")
        Qi += Qifast

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

    Flux_base = [Qe[:,0], Qi[:,0], Ge[:,0], GZ[:,0], Mt[:,0], S[:,0]]
    Flux_mean = [Qe_point, Qi_point, Ge_point, GZ_point, Mt_point, S_point]
    Flux_std  = [Qe_std, Qi_std, Ge_std, GZ_std, Mt_std, S_std]

    return Flux_base, Flux_mean, Flux_std
