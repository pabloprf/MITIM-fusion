import torch
import copy
import shutil
import numpy as np
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import TGYROtools
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class tgyro_model(TRANSPORTtools.power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        self._produce_profiles()

    # ************************************************************************************
    # Private functions for TGLF and NEO evaluations
    # ************************************************************************************

    # Do nothing here
    def evaluate_neoclassical(self):
        pass

    # Evaluate both
    def evaluate_turbulence(self):
        
        tgyro = self._evaluate_tglf_neo()
        self._postprocess_tgyro(tgyro, "tglf_neo")
        
        TRANSPORTtools.write_json(self, file_name = 'fluxes_neoc.json', suffix= 'neoc')
        TRANSPORTtools.write_json(self, file_name = 'fluxes_turb.json', suffix= 'turb')

    def _evaluate_tglf_neo(self):

        transport = self.powerstate.transport_options["transport_evaluator_options"]["transport_parameters"]
        cold_start = self.powerstate.transport_options["transport_evaluator_options"]["cold_start"]

        Qi_includes_fast = transport["Qi_includes_fast"]
        launchMODELviaSlurm = transport["launchMODELviaSlurm"]
        percent_error = transport["percent_error"]
        cores_per_tglf_instance = transport["cores_per_tglf_instance"]
        use_tglf_scan_trick = transport["use_scan_trick_for_stds"]

        # Grab impurity from powerstate ( because it may have been modified in produce_profiles() )
        impurityPosition = self.powerstate.impurityPosition_transport #transport_evaluator_options.get("impurityPosition", 1)

        # ------------------------------------------------------------------------------------------------------------------------
        # tglf_neo_original: Run TGYRO workflow - TGLF + NEO in subfolder tglf_neo_original (original as in... without stds or merging)
        # ------------------------------------------------------------------------------------------------------------------------

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]

        tgyro = TGYROtools.TGYRO(cdf=dummyCDF(self.folder, self.folder))
        tgyro.prep(self.folder, profilesclass_custom=self.powerstate.profiles_transport)

        if launchMODELviaSlurm:
            print("\t- Launching TGYRO evaluation as a batch job")
        else:
            print("\t- Launching TGYRO evaluation as a terminal job")

        tgyro.run(
            subFolderTGYRO="tglf_neo_original",
            cold_start=cold_start,
            forceIfcold_start=True,
            special_radii=rho_locations,
            iterations=0,
            PredictionSet=[
                int("te" in self.powerstate.predicted_channels),
                int("ti" in self.powerstate.predicted_channels),
                int("ne" in self.powerstate.predicted_channels),
            ],
            TGLFsettings=transport["transport_evaluator_options"]["TGLFsettings"],
            extraOptionsTGLF=transport["transport_evaluator_options"]["extraOptionsTGLF"],
            TGYRO_physics_options=portals_parameters["model_parameters"]["Physics_options"],
            launchSlurm=launchMODELviaSlurm,
            minutesJob=5,
            forcedName=self.name,
        )

        tgyro.read(label="tglf_neo_original")

        # Copy one with evaluated targets
        self.file_profs_targets = tgyro.FolderTGYRO / "input.gacode.new"

        # ------------------------------------------------------------------------------------------------------------------------
        # tglf_neo: Write TGLF, NEO and TARGET errors in tgyro files as well
        # ------------------------------------------------------------------------------------------------------------------------

        # Copy original TGYRO folder
        if (self.folder / "tglf_neo").exists():
            IOtools.shutil_rmtree(self.folder / "tglf_neo")
        shutil.copytree(self.folder / "tglf_neo_original", self.folder / "tglf_neo")

        # Add errors and merge fluxes as we would do if this was a CGYRO run
        curateTGYROfiles(
            tgyro,
            "tglf_neo_original",
            rho_locations,
            self.powerstate.predicted_channels,
            self.folder / "tglf_neo",
            percent_error,
            impurityPosition=impurityPosition,
            Qi_includes_fast=Qi_includes_fast,
            use_tglf_scan_trick = use_tglf_scan_trick,
            cold_start=cold_start,
            extra_name = self.name,
            cores_per_tglf_instance=cores_per_tglf_instance
        )

        # Read again to capture errors
        tgyro.read(label="tglf_neo", folder=self.folder / "tglf_neo")

        return tgyro

    def _postprocess_tgyro(self, tgyro, label):

        portals_parameters = self.powerstate.transport_options["transport_evaluator_options"]["portals_parameters"]
        
        Qi_includes_fast = portals_parameters["model_parameters"]["transport_parameters"]["Qi_includes_fast"]
        UseFineGridTargets = portals_parameters["main_parmaters"]["UseFineGridTargets"]
        OriginalFimp = portals_parameters["main_parmaters"]["OriginalFimp"]
        forceZeroParticleFlux = portals_parameters["model_parameters"]["target_parameters"]["target_evaluator_options"]["forceZeroParticleFlux"]

        # Grab impurity from powerstate ( because it may have been modified in produce_profiles() )
        impurityPosition = self.powerstate.impurityPosition_transport

        # Produce right quantities (TGYRO -> powerstate.plasma)
        tgyro_to_powerstate(
            self,
            tgyro.results[label],
            Qi_includes_fast=Qi_includes_fast,
            impurityPosition=impurityPosition,
            UseFineGridTargets=UseFineGridTargets,
            OriginalFimp=OriginalFimp,
            forceZeroParticleFlux=forceZeroParticleFlux,
            provideTargets=self.powerstate.target_options['target_evaluator_options']["target_evaluator_method"] == "tgyro",
        )

        tgyro.results["use"] = tgyro.results[label]

        # Copy profiles to share
        self._profiles_to_store()

        # ------------------------------------------------------------------------------------------------------------------------
        # Results class that can be used for further plotting and analysis in PORTALS
        # ------------------------------------------------------------------------------------------------------------------------

        self.model_results = copy.deepcopy(tgyro.results["use"]) # Pass the TGYRO results class that should be use for plotting and analysis

        self.model_results.extra_analysis = {}
        for ikey in tgyro.results:
            if ikey != "use":
                self.model_results.extra_analysis[ikey] = tgyro.results[ikey]

    def _profiles_to_store(self):

        if "folder" in self.powerstate.transport_options["transport_evaluator_options"]:
            whereFolder = IOtools.expandPath(self.powerstate.transport_options["folder"] / "Outputs" / "portals_profiles")
            if not whereFolder.exists():
                IOtools.askNewFolder(whereFolder)

            fil = whereFolder / f"input.gacode.{self.evaluation_number}"
            shutil.copy2(self.file_profs, fil)
            shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
            shutil.copy2(self.file_profs_targets, fil.parent / f"{fil.name}.new_fromtgyro_modified")
            print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
        else:
            print("\t- Could not move files", typeMsg="w")

def tglf_scan_trick(
    tglf,
    rho_locations, 
    predicted_channels, 
    impurityPosition=1,
    Qi_includes_fast=False,  
    delta=0.02, 
    minimum_abs_gradient=0.005, # This is 0.5% of aLx=1.0, to avoid extremely small scans when, for example, having aLn ~ 0.0
    cold_start=False, 
    extra_name="", 
    remove_folders_out = False,
    cores_per_tglf_instance = 4 # e.g. 4 core per radius, since this is going to launch ~ Nr=5 x (Nv=6 x Nd=2 + 1) = 65 TGLFs at once
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
                    TGLFsettings = None,
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
# **************************************************************************************************
# Functions
# **************************************************************************************************

def curateTGYROfiles(
    tgyroObject,
    label,
    rho_locations,
    predicted_channels,
    folder,
    percent_error,
    impurityPosition=1,
    Qi_includes_fast=False,
    use_tglf_scan_trick=None,
    cold_start=False,
    extra_name="",
    cores_per_tglf_instance = 4,
    check_coincidence_thr=1E-2, 
    ):

    tgyro = tgyroObject.results[label]
    
    # Determine NEO and Target errors
    relativeErrorNEO = percent_error[1] / 100.0
    relativeErrorTAR = percent_error[2] / 100.0

    # Grab fluxes from TGYRO
    Qe = tgyro.Qe_sim_turb[0, 1:]
    Qi = tgyro.QiIons_sim_turb[0, 1:] if Qi_includes_fast else tgyro.QiIons_sim_turb_thr[0, 1:]
    Ge = tgyro.Ge_sim_turb[0, 1:]
    GZ = tgyro.Gi_sim_turb[impurityPosition, 0, 1:]
    Mt = tgyro.Mt_sim_turb[0, 1:]
    Pexch = tgyro.EXe_sim_turb[0, 1:]
    
    # Determine TGLF standard deviations
    if use_tglf_scan_trick is not None:

        # Grab TGLF object
        tglfObject = tgyroObject.grab_tglf_objects(fromlabel=label, subfolder = 'tglf_explorations')

        # Run TGLF scan trick
        Flux_base, Flux_mean, Flux_std = tglf_scan_trick(
            tglfObject,
            rho_locations, 
            predicted_channels, 
            impurityPosition=impurityPosition, 
            Qi_includes_fast=Qi_includes_fast, 
            delta = use_tglf_scan_trick,
            cold_start=cold_start,
            extra_name=extra_name,
            cores_per_tglf_instance=cores_per_tglf_instance
            )

        Qe, Qi, Ge, GZ, Mt, Pexch = Flux_mean
        QeE, QiE, GeE, GZE, MtE, PexchE = Flux_std

        # ----------------------------------------------------
        # Do a check that TGLF scans are consistent with TGYRO

        Qe_base, Qi_base, Ge_base, GZ_base, Mt_base, S_base = Flux_base

        # Grab fluxes from TGYRO
        Qe_tgyro = tgyro.Qe_sim_turb[0, 1:]
        Qi_tgyro = tgyro.QiIons_sim_turb[0, 1:] if Qi_includes_fast else tgyro.QiIons_sim_turb_thr[0, 1:]
        Ge_tgyro = tgyro.Ge_sim_turb[0, 1:]
        GZ_tgyro = tgyro.Gi_sim_turb[impurityPosition, 0, 1:]
        Mt_tgyro = tgyro.Mt_sim_turb[0, 1:]
        Pexch_tgyro = tgyro.EXe_sim_turb[0, 1:]

        Qe_err = np.abs( (Qe_base - Qe_tgyro) / Qe_tgyro ) if 'te' in predicted_channels else np.zeros_like(Qe_base)
        Qi_err = np.abs( (Qi_base - Qi_tgyro) / Qi_tgyro ) if 'ti' in predicted_channels else np.zeros_like(Qi_base)
        Ge_err = np.abs( (Ge_base - Ge_tgyro) / Ge_tgyro ) if 'ne' in predicted_channels else np.zeros_like(Ge_base)
        GZ_err = np.abs( (GZ_base - GZ_tgyro) / GZ_tgyro ) if 'nZ' in predicted_channels else np.zeros_like(GZ_base)
        Mt_err = np.abs( (Mt_base - Mt_tgyro) / Mt_tgyro ) if 'w0' in predicted_channels else np.zeros_like(Mt_base)
        Pexch_err = np.abs( (Pexch - Pexch_tgyro) / Pexch_tgyro )

        F_err = np.concatenate((Qe_err, Qi_err, Ge_err, GZ_err, Mt_err, Pexch_err))
        if F_err.max() > check_coincidence_thr:
            print(f"\t- TGLF scans are not consistent with TGYRO, maximum error = {F_err.max()*100:.2f}%, in quantity:",typeMsg="w")
            if ('te' in predicted_channels) and Qe_err.max() > check_coincidence_thr:
                print('\t\t* Qe:',Qe_err)
            if ('ti' in predicted_channels) and Qi_err.max() > check_coincidence_thr:
                print('\t\t* Qi:',Qi_err)
            if ('ne' in predicted_channels) and Ge_err.max() > check_coincidence_thr:
                print('\t\t* Ge:',Ge_err)
            if ('nZ' in predicted_channels) and GZ_err.max() > check_coincidence_thr:
                print('\t\t* GZ:',GZ_err)
            if ('w0' in predicted_channels) and Mt_err.max() > check_coincidence_thr:
                print('\t\t* Mt:',Mt_err)
            if Pexch_err.max() > check_coincidence_thr:
                print('\t\t* Pexch:',Pexch_err)
        else:
            print(f"\t- TGLF scans are consistent with TGYRO, maximum error = {F_err.max()*100:.2f}%")
        # ----------------------------------------------------

        min_relative_error = 0.01 # To avoid problems with gpytorch, 1% error minimum

        QeE = QeE.clip(abs(Qe)*min_relative_error)
        QiE = QiE.clip(abs(Qi)*min_relative_error)
        GeE = GeE.clip(abs(Ge)*min_relative_error)
        GZE = GZE.clip(abs(GZ)*min_relative_error)
        MtE = MtE.clip(abs(Mt)*min_relative_error)
        PexchE = PexchE.clip(abs(Pexch)*min_relative_error)

    else:

        # --------------------------------------------------------------
        # If simply a percentage error provided
        # --------------------------------------------------------------

        relativeErrorTGLF = [percent_error[0] / 100.0]*len(rho_locations)
    
        QeE = abs(Qe) * relativeErrorTGLF
        QiE = abs(Qi) * relativeErrorTGLF
        GeE = abs(Ge) * relativeErrorTGLF
        GZE = abs(GZ) * relativeErrorTGLF
        MtE = abs(Mt) * relativeErrorTGLF
        PexchE = abs(Pexch) * relativeErrorTGLF

    # **************************************************************************************************************************
    # Neo
    # **************************************************************************************************************************

    Qe_tr_neoc = tgyro.Qe_sim_neo[0, 1:]
    if Qi_includes_fast:
        Qi_tr_neoc = tgyro.QiIons_sim_neo[0, 1:]
    else:
        Qi_tr_neoc = tgyro.QiIons_sim_neo_thr[0, 1:]
    Ge_tr_neoc = tgyro.Ge_sim_neo[0, 1:]
    GZ_tr_neoc = tgyro.Gi_sim_neo[impurityPosition, 0, 1:]
    Mt_tr_neoc = tgyro.Mt_sim_neo[0, 1:]

    Qe_tr_neocE = abs(tgyro.Qe_sim_neo[0, 1:]) * relativeErrorNEO
    if Qi_includes_fast:
        Qi_tr_neocE = abs(tgyro.QiIons_sim_neo[0, 1:]) * relativeErrorNEO
    else:
        Qi_tr_neocE = abs(tgyro.QiIons_sim_neo_thr[0, 1:]) * relativeErrorNEO
    Ge_tr_neocE = abs(tgyro.Ge_sim_neo[0, 1:]) * relativeErrorNEO
    GZ_tr_neocE = abs(tgyro.Gi_sim_neo[impurityPosition, 0, 1:]) * relativeErrorNEO
    Mt_tr_neocE = abs(tgyro.Mt_sim_neo[0, 1:]) * relativeErrorNEO

    # Merge

    modifyFLUX(
        tgyro,
        folder,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        Qe_tr_neoc=Qe_tr_neoc,
        Qi_tr_neoc=Qi_tr_neoc,
        Ge_tr_neoc=Ge_tr_neoc,
        GZ_tr_neoc=GZ_tr_neoc,
        Mt_tr_neoc=Mt_tr_neoc,
        impurityPosition=impurityPosition,
    )

    modifyFLUX(
        tgyro,
        folder,
        QeE,
        QiE,
        GeE,
        GZE,
        MtE,
        PexchE,
        Qe_tr_neoc=Qe_tr_neocE,
        Qi_tr_neoc=Qi_tr_neocE,
        Ge_tr_neoc=Ge_tr_neocE,
        GZ_tr_neoc=GZ_tr_neocE,
        Mt_tr_neoc=Mt_tr_neocE,
        impurityPosition=impurityPosition,
        special_label="_stds",
    )

    # **************************************************************************************************************************
    # Targets
    # **************************************************************************************************************************

    QeTargetE = abs(tgyro.Qe_tar[0, 1:]) * relativeErrorTAR
    QiTargetE = abs(tgyro.Qi_tar[0, 1:]) * relativeErrorTAR
    GeTargetE = abs(tgyro.Ge_tar[0, 1:]) * relativeErrorTAR
    GZTargetE = GeTargetE * 0.0
    MtTargetE = abs(tgyro.Mt_tar[0, 1:]) * relativeErrorTAR

    modifyEVO(
        tgyro,
        folder,
        QeTargetE * 0.0,
        QiTargetE * 0.0,
        GeTargetE * 0.0,
        GZTargetE * 0.0,
        MtTargetE * 0.0,
        impurityPosition=impurityPosition,
        positionMod=1,
        special_label="_stds",
    )
    modifyEVO(
        tgyro,
        folder,
        QeTargetE,
        QiTargetE,
        GeTargetE,
        GZTargetE,
        MtTargetE,
        impurityPosition=impurityPosition,
        positionMod=2,
        special_label="_stds",
    )

def dummyCDF(GeneralFolder, FolderEvaluation):
    """
    This routine creates path to a dummy CDF file in FolderEvaluation, with the name "simulation_evaluation.CDF"

    GeneralFolder, e.g.    ~/runs_portals/run10/
    FolderEvaluation, e.g. ~/runs_portals/run10000/Execution/Evaluation.0/transport_simulation_folder/
    """

    # ------- Name construction for scratch folders in parallel ----------------

    GeneralFolder = IOtools.expandPath(GeneralFolder, ensurePathValid=True)

    a, subname = IOtools.reducePathLevel(GeneralFolder, level=1, isItFile=False)

    FolderEvaluation = IOtools.expandPath(FolderEvaluation)

    name = FolderEvaluation.name.split(".")[-1]  # 0   (evaluation #)

    if name == "":
        name = "0"

    cdf = FolderEvaluation / f"{subname}_ev{name}.CDF"

    return cdf

def modifyResults(
    Qe,
    Qi,
    Ge,
    GZ,
    Mt,
    Pexch,
    QeE,
    QiE,
    GeE,
    GZE,
    MtE,
    PexchE,
    tgyro,
    folder_tgyro,
    minErrorPercent=5.0,
    Qi_stable_criterion=0.0025,
    impurityPosition=3,
    OriginalFimp=1.0,
):
    """
    All in real units, with dimensions of (rho) from axis to edge
    """

    # If a plasma is very close to stable... do something about error
    if minErrorPercent is not None:
        (
            Qe_target,
            Qi_target,
            Ge_target_special,
            GZ_target_special,
            Mt_target,
        ) = defineReferenceFluxes(
            tgyro,
            impurityPosition=impurityPosition,
        )

        Qe_min = Qe_target * (minErrorPercent / 100.0)
        Qi_min = Qi_target * (minErrorPercent / 100.0)
        Ge_min = Ge_target_special * (minErrorPercent / 100.0)
        GZ_min = GZ_target_special * (minErrorPercent / 100.0)
        Mt_min = Mt_target * (minErrorPercent / 100.0)

        for i in range(Qe.shape[0]):
            if Qi[i] < Qi_stable_criterion:
                print(
                    f"\t- Based on 'Qi_stable_criterion', plasma considered stable (Qi = {Qi[i]:.2e} < {Qi_stable_criterion:.2e} MW/m2) at position #{i}, using minimum errors of {minErrorPercent}% of targets",
                    typeMsg="w",
                )
                QeE[i] = Qe_min[i]
                print(f"\t\t* QeE = {QeE[i]}")
                QiE[i] = Qi_min[i]
                print(f"\t\t* QiE = {QiE[i]}")
                GeE[i] = Ge_min[i]
                print(f"\t\t* GeE = {GeE[i]}")
                GZE[i] = GZ_min[i]
                print(f"\t\t* GZE = {GZE[i]}")
                MtE[i] = Mt_min[i]
                print(f"\t\t* MtE = {MtE[i]}")

    # Heat fluxes
    QeTot = Qe + tgyro.Qe_sim_neo[0, 1:]
    QiTot = Qi + tgyro.QiIons_sim_neo_thr[0, 1:]

    # Particle fluxes
    PeTot = Ge + tgyro.Ge_sim_neo[0, 1:]
    PZTot = GZ + tgyro.Gi_sim_neo[impurityPosition, 0, 1:]

    # Momentum fluxes
    MtTot = Mt + tgyro.Mt_sim_neo[0, 1:]

    # ************************************************************************************
    # **** Modify complete folder (Division of ion fluxes will be wrong, since I put everything in first ion)
    # ************************************************************************************

    # 1. Modify out.tgyro.evo files (which contain turb+neo summed together)

    print(f"\t- Modifying TGYRO out.tgyro.evo files in {IOtools.clipstr(folder_tgyro)}")
    modifyEVO(
        tgyro,
        folder_tgyro,
        QeTot,
        QiTot,
        PeTot,
        PZTot,
        MtTot,
        impurityPosition=impurityPosition,
    )

    # 2. Modify out.tgyro.flux files (which contain turb and neo separated)

    print(f"\t- Modifying TGYRO out.tgyro.flux files in {folder_tgyro}")
    modifyFLUX(
        tgyro,
        folder_tgyro,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        impurityPosition=impurityPosition,
    )

    # 3. Modify files for errors

    print(f"\t- Modifying TGYRO out.tgyro.flux_stds in {folder_tgyro}")
    modifyFLUX(
        tgyro,
        folder_tgyro,
        QeE,
        QiE,
        GeE,
        GZE,
        MtE,
        PexchE,
        impurityPosition=impurityPosition,
        special_label="_stds",
    )


def modifyEVO(
    tgyro,
    folder,
    QeT,
    QiT,
    GeT,
    GZT,
    MtT,
    impurityPosition=3,
    positionMod=1,
    special_label=None,
):
    QeTGB = QeT / tgyro.Q_GB[-1, 1:]
    QiTGB = QiT / tgyro.Q_GB[-1, 1:]
    GeTGB = GeT / tgyro.Gamma_GB[-1, 1:]
    GZTGB = GZT / tgyro.Gamma_GB[-1, 1:]
    MtTGB = MtT / tgyro.Pi_GB[-1, 1:]

    modTGYROfile(folder / "out.tgyro.evo_te", QeTGB, pos=positionMod, fileN_suffix=special_label)
    modTGYROfile(folder / "out.tgyro.evo_ti", QiTGB, pos=positionMod, fileN_suffix=special_label)
    modTGYROfile(folder / "out.tgyro.evo_ne", GeTGB, pos=positionMod, fileN_suffix=special_label)
    modTGYROfile(folder / "out.tgyro.evo_er", MtTGB, pos=positionMod, fileN_suffix=special_label)

    for i in range(tgyro.Qi_sim_turb.shape[0]):
        if i == impurityPosition:
            var = GZTGB
        else:
            var = GZTGB * 0.0
        modTGYROfile(
            folder / f"out.tgyro.evo_n{i+1}",
            var,
            pos=positionMod,
            fileN_suffix=special_label,
        )


def modifyFLUX(
    tgyro,
    folder,
    Qe,
    Qi,
    Ge,
    GZ,
    Mt,
    S,
    Qe_tr_neoc=None,
    Qi_tr_neoc=None,
    Ge_tr_neoc=None,
    GZ_tr_neoc=None,
    Mt_tr_neoc=None,
    impurityPosition=3,
    special_label=None,
):
    folder = IOtools.expandPath(folder)

    QeGB = Qe / tgyro.Q_GB[-1, 1:]
    QiGB = Qi / tgyro.Q_GB[-1, 1:]
    GeGB = Ge / tgyro.Gamma_GB[-1, 1:]
    GZGB = GZ / tgyro.Gamma_GB[-1, 1:]
    MtGB = Mt / tgyro.Pi_GB[-1, 1:]
    SGB = S / tgyro.S_GB[-1, 1:]

    # ******************************************************************************************
    # Electrons
    # ******************************************************************************************

    # Particle flux: Update

    modTGYROfile(folder / "out.tgyro.flux_e", GeGB, pos=2, fileN_suffix=special_label)
    if Ge_tr_neoc is not None:
        GeGB_neo = Ge_tr_neoc / tgyro.Gamma_GB[-1, 1:]
        modTGYROfile(folder / "out.tgyro.flux_e", GeGB_neo, pos=1, fileN_suffix=special_label)

    # Energy flux: Update

    modTGYROfile(folder / "out.tgyro.flux_e", QeGB, pos=4, fileN_suffix=special_label)
    if Qe_tr_neoc is not None:
        QeGB_neo = Qe_tr_neoc / tgyro.Q_GB[-1, 1:]
        modTGYROfile(folder / "out.tgyro.flux_e", QeGB_neo, pos=3, fileN_suffix=special_label)

    # Rotation: Remove (it will be sum to the first ion)

    modTGYROfile(folder / "out.tgyro.flux_e", GeGB * 0.0, pos=6, fileN_suffix=special_label)
    modTGYROfile(folder / "out.tgyro.flux_e", GeGB * 0.0, pos=5, fileN_suffix=special_label)

    # Energy exchange

    modTGYROfile(folder / "out.tgyro.flux_e", SGB, pos=7, fileN_suffix=special_label)

    # SMW  = S  # S is MW/m^3
    # modTGYROfile(f'{folder}/out.tgyro.power_e',SMW,pos=8,fileN_suffix=special_label)
    # print('\t\t- Modified turbulent energy exchange in out.tgyro.power_e')

    # ******************************************************************************************
    # Ions
    # ******************************************************************************************

    # Energy flux: Update

    modTGYROfile(folder / "out.tgyro.flux_i1", QiGB, pos=4, fileN_suffix=special_label)

    if Qi_tr_neoc is not None:
        QiGB_neo = Qi_tr_neoc / tgyro.Q_GB[-1, 1:]
        modTGYROfile(folder / "out.tgyro.flux_i1", QiGB_neo, pos=3, fileN_suffix=special_label)

    # Particle flux: Make ion particle fluxes zero, because I don't want to mistake TGLF with CGYRO when looking at tgyro results

    for i in range(tgyro.Qi_sim_turb.shape[0]):
        if tgyro.profiles.Species[i]["S"] == "therm":
            var = QiGB * 0.0
            modTGYROfile(folder / f"out.tgyro.flux_i{i+1}",var,pos=2,fileN_suffix=special_label,)  # Gi_turb
            modTGYROfile(folder / f"out.tgyro.evo_n{i+1}", var, pos=1, fileN_suffix=special_label)  # Gi (Gi_sim)

            if i != impurityPosition:
                modTGYROfile(folder / f"out.tgyro.flux_i{i+1}",var,pos=1,fileN_suffix=special_label)  # Gi_neo

    # Rotation: Update

    modTGYROfile(folder / "out.tgyro.flux_i1", MtGB, pos=6, fileN_suffix=special_label)

    if Mt_tr_neoc is not None:
        MtGB_neo = Mt_tr_neoc / tgyro.Pi_GB[-1, 1:]
        modTGYROfile(folder / "out.tgyro.flux_i1", MtGB_neo, pos=5, fileN_suffix=special_label)

    # Energy exchange: Remove (it will be the electrons one)

    modTGYROfile(folder / "out.tgyro.flux_i1", SGB * 0.0, pos=7, fileN_suffix=special_label)

    # ******************************************************************************************
    # Impurities
    # ******************************************************************************************

    # Remove everything from all the rest of non-first ions (except the particles for the impurity chosen)

    for i in range(tgyro.Qi_sim_turb.shape[0] - 1):
        if tgyro.profiles.Species[i + 1]["S"] == "therm":
            var = QiGB * 0.0
            for pos in [3, 4, 5, 6, 7]:
                modTGYROfile(folder / f"out.tgyro.flux_i{i+2}",var,pos=pos,fileN_suffix=special_label)
            for pos in [1, 2]:
                if i + 2 != impurityPosition:
                    modTGYROfile(folder / f"out.tgyro.flux_i{i+2}",var,pos=pos,fileN_suffix=special_label)

    modTGYROfile(folder / f"out.tgyro.flux_i{impurityPosition+1}",GZGB,pos=2,fileN_suffix=special_label)
    if GZ_tr_neoc is not None:
        GZGB_neo = GZ_tr_neoc / tgyro.Gamma_GB[-1, 1:]
        modTGYROfile(folder / f"out.tgyro.flux_i{impurityPosition+1}",GZGB_neo,pos=1,fileN_suffix=special_label)


def modTGYROfile(file, var, pos=0, fileN_suffix=None):
    fileN = file if fileN_suffix is None else file.parent / f"{file.name}{fileN_suffix}"

    if not fileN.exists():
        shutil.copy2(file, fileN)

    with open(fileN, "r") as f:
        lines = f.readlines()

    with open(fileN, "w") as f:
        f.write(lines[0])
        f.write(lines[1])
        f.write(lines[2])
        for i in range(var.shape[0]):
            new_s = [float(k) for k in lines[3 + i].split()]
            new_s[pos] = var[i]

            line_new = " "
            for k in range(len(new_s)):
                line_new += f'{"" if k==0 else "   "}{new_s[k]:.6e}'
            f.write(line_new + "\n")

def defineReferenceFluxes(
    tgyro, factor_tauptauE=5, impurityPosition=3
):
    Qe_target = abs(tgyro.Qe_tar[0, 1:])
    Qi_target = abs(tgyro.Qi_tar[0, 1:])
    Mt_target = abs(tgyro.Mt_tar[0, 1:])

    # For particle fluxes, since the targets are often zero... it's more complicated
    QeMW_target = abs(tgyro.Qe_tarMW[0, 1:])
    QiMW_target = abs(tgyro.Qi_tarMW[0, 1:])
    We, Wi, Ne, NZ = tgyro.profiles.deriveContentByVolumes(
        rhos=tgyro.rho[0, 1:], impurityPosition=impurityPosition
    )

    tau_special = (
        (We + Wi) / (QeMW_target + QiMW_target) * factor_tauptauE
    )  # tau_p in seconds
    Ge_target_special = (Ne / tau_special) / tgyro.dvoldr[0, 1:]  # (1E20/seconds/m^2)

    Ge_target_special = PLASMAtools.convective_flux(
        tgyro.Te[0, 1:], Ge_target_special
    )  # (1E20/seconds/m^2)

    GZ_target_special = Ge_target_special * NZ / Ne

    return Qe_target, Qi_target, Ge_target_special, GZ_target_special, Mt_target



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# This is where the definitions for the summation variables happen for mitim and PORTALSplot
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def tgyro_to_powerstate(
    self,
    TGYROresults,
    forceZeroParticleFlux=False,
    Qi_includes_fast=False,
    impurityPosition=1,
    UseFineGridTargets=False,
    OriginalFimp=1.0,
    provideTargets=False
    ):
    """
    This function is used to extract the TGYRO results and store them in the powerstate object, from numpy arrays to torch tensors.
    """

    if "tgyro_stds" not in TGYROresults.__dict__:
        TGYROresults.tgyro_stds = False

    if UseFineGridTargets:
        TGYROresults.useFineGridTargets(impurityPosition=impurityPosition)

    nr = self.powerstate.plasma['rho'].shape[-1]
    if self.powerstate.plasma['rho'].shape[-1] != TGYROresults.rho.shape[-1]:
        print('\t- TGYRO was run with an extra point in the grid, treating it carefully now')

    # **********************************
    # *********** Electron Energy Fluxes
    # **********************************

    self.QeGB_turb = TGYROresults.QeGB_sim_turb[0, 1:nr]
    self.QeGB_neoc = TGYROresults.QeGB_sim_neo[0, 1:nr]

    self.QeGB_turb_stds = TGYROresults.QeGB_sim_turb_stds[0, 1:nr]
    self.QeGB_neoc_stds = TGYROresults.QeGB_sim_neo_stds[0, 1:nr]

    # **********************************
    # *********** Ion Energy Fluxes
    # **********************************

    if Qi_includes_fast:

        self.QiGB_turb = TGYROresults.QiIons_sim_turb[0, 1:nr]
        self.QiGB_neoc = TGYROresults.QiIons_sim_neo[0, 1:nr]
        
        self.QiGB_turb_stds = TGYROresults.QiGBIons_sim_turb_stds[0, 1:nr]
        self.QiGB_neoc_stds = TGYROresults.QiGBIons_sim_neo_stds[0, 1:nr]

    else:

        self.QiGB_turb = TGYROresults.QiGBIons_sim_turb_thr[0, 1:nr]
        self.QiGB_neoc = TGYROresults.QiGBIons_sim_neo_thr[0, 1:nr]

        self.QiGB_turb_stds = TGYROresults.QiGBIons_sim_turb_thr_stds[0, 1:nr]
        self.QiGB_neoc_stds = TGYROresults.QiGBIons_sim_neo_thr_stds[0, 1:nr]

    # **********************************
    # *********** Momentum Fluxes
    # **********************************

    self.MtGB_turb = TGYROresults.MtGB_sim_turb[0, 1:nr] # So far, let's include fast in momentum
    self.MtGB_neoc = TGYROresults.MtGB_sim_neo[0, 1:nr]

    self.MtGB_turb_stds = TGYROresults.MtGB_sim_turb_stds[0, 1:nr]
    self.MtGB_neoc_stds = TGYROresults.MtGB_sim_neo_stds[0, 1:nr]
    
    # **********************************
    # *********** Particle Fluxes
    # **********************************

    # Store raw fluxes for better plotting later
    self.GeGB_turb = TGYROresults.GeGB_sim_turb[0, 1:nr]
    self.GeGB_neoc = TGYROresults.GeGB_sim_neo[0, 1:nr]

    self.GeGB_turb_stds = TGYROresults.GeGB_sim_turb_stds[0, 1:nr]
    self.GeGB_neoc_stds = TGYROresults.GeGB_sim_neo_stds[0, 1:nr]
        
    # **********************************
    # *********** Impurity Fluxes
    # **********************************

    # Store raw fluxes for better plotting later
    self.GZGB_turb = TGYROresults.GiGB_sim_turb[impurityPosition, 0, 1:nr]
    self.GZGB_neoc = TGYROresults.GiGB_sim_neo[impurityPosition, 0, 1:nr]
    
    self.GZGB_turb_stds = TGYROresults.GiGB_sim_turb_stds[impurityPosition, 0, 1:nr]
    self.GZGB_neoc_stds = TGYROresults.GiGB_sim_neo_stds[impurityPosition, 0, 1:nr]

    # **********************************
    # *********** Energy Exchange
    # **********************************

    self.QieGB_turb = TGYROresults.EXeGB_sim_turb[0, 1:nr]
    self.QieGB_turb_stds = TGYROresults.EXeGB_sim_turb_stds[0, 1:nr]

    self.QieGB_neoc = self.QeGB_turb * 0.0
    self.QieGB_neoc_stds = self.QeGB_turb_stds * 0.0

    # **********************************
    # *********** Targets
    # ***********************************

    if provideTargets:
        self.powerstate.plasma["QeMWm2"] = torch.Tensor(TGYROresults.Qe_tar[0, 1:nr]).to(self.powerstate.dfT)
        self.powerstate.plasma["QeMWm2_stds"] = torch.Tensor(TGYROresults.Qe_tar_stds[0, 1:nr]).to(self.powerstate.dfT) if TGYROresults.tgyro_stds else None
        self.powerstate.plasma["QiMWm2"] = torch.Tensor(TGYROresults.Qi_tar[0, 1:nr]).to(self.powerstate.dfT)
        self.powerstate.plasma["QiMWm2_stds"] = torch.Tensor(TGYROresults.Qi_tar_stds[0, 1:nr]).to(self.powerstate.dfT) if TGYROresults.tgyro_stds else None
        self.powerstate.plasma["MtJm2"] = torch.Tensor(TGYROresults.Mt_tar[0, 1:nr]).to(self.powerstate.dfT)
        self.powerstate.plasma["MtJm2_stds"] = torch.Tensor(TGYROresults.Mt_tar_stds[0, 1:nr]).to(self.powerstate.dfT) if TGYROresults.tgyro_stds else None
        self.powerstate.plasma["Ge1E20m2"] = torch.Tensor(TGYROresults.Ge_tar[0, 1:nr]).to(self.powerstate.dfT)
        self.powerstate.plasma["Ge1E20m2_stds"] = torch.Tensor(TGYROresults.Ge_tar_stds[0, 1:nr]).to(self.powerstate.dfT) if TGYROresults.tgyro_stds else None
        self.powerstate.plasma["Ce"] = torch.Tensor(TGYROresults.Ce_tar[0, 1:nr]).to(self.powerstate.dfT)
        self.powerstate.plasma["Ce_stds"] = torch.Tensor(TGYROresults.Ce_tar_stds[0, 1:nr]).to(self.powerstate.dfT) if TGYROresults.tgyro_stds else None
        self.powerstate.plasma["GZ1E20m2"] = torch.Tensor(TGYROresults.Gi_tar[impurityPosition, 0, 1:nr]).to(self.powerstate.dfT) 
        self.powerstate.plasma["GZ1E20m2_stds"] = torch.Tensor(TGYROresults.Gi_tar_stds[impurityPosition, 0, 1:nr]).to(self.powerstate.dfT)  if TGYROresults.tgyro_stds else None
        self.powerstate.plasma["CZ"] = torch.Tensor(TGYROresults.Ci_tar[impurityPosition, 0, 1:nr]).to(self.powerstate.dfT) / OriginalFimp
        self.powerstate.plasma["CZ_stds"] = torch.Tensor(TGYROresults.Ci_tar_stds[impurityPosition, 0, 1:nr]).to(self.powerstate.dfT) / OriginalFimp if TGYROresults.tgyro_stds else None

        if forceZeroParticleFlux:
            self.powerstate.plasma["Ce"] = self.powerstate.plasma["Ce"] * 0.0
            self.powerstate.plasma["Ce_stds"] = self.powerstate.plasma["Ce_stds"] * 0.0

