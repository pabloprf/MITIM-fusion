import copy
import shutil
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGYROtools
from mitim_modules.portals.utils import PORTALScgyro
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ----------------------------------------------------------------------------------------------------
# FULL TGYRO
# ----------------------------------------------------------------------------------------------------

class tgyro_model(TRANSPORTtools.power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        self._produce_profiles()

    def evaluate(self):

        tgyro = self._evaluate_tglf_neo()

        self._postprocess_results(tgyro, "tglf_neo")

    # ************************************************************************************
    # Private functions for TGLF and NEO evaluations
    # ************************************************************************************

    def _profiles_to_store(self):

        if "extra_params" in self.powerstate.TransportOptions["ModelOptions"] and "folder" in self.powerstate.TransportOptions["ModelOptions"]["extra_params"]:
            whereFolder = IOtools.expandPath(self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"] / "Outputs" / "portals_profiles")
            if not whereFolder.exists():
                IOtools.askNewFolder(whereFolder)

            fil = whereFolder / f"input.gacode.{self.evaluation_number}"
            shutil.copy2(self.file_profs, fil)
            shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
            shutil.copy2(self.file_profs_targets, fil.parent / f"{fil.name}.new")
            print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
        else:
            print("\t- Could not move files", typeMsg="w")


    def _postprocess_results(self, tgyro, label):

        ModelOptions = self.powerstate.TransportOptions["ModelOptions"]

        includeFast = ModelOptions.get("includeFastInQi",False)
        useConvectiveFluxes = ModelOptions.get("useConvectiveFluxes", True)
        UseFineGridTargets = ModelOptions.get("UseFineGridTargets", False)
        provideTurbulentExchange = ModelOptions.get("TurbulentExchange", False)
        OriginalFimp = ModelOptions.get("OriginalFimp", 1.0)
        forceZeroParticleFlux = ModelOptions.get("forceZeroParticleFlux", False)

        # Grab impurity from powerstate ( because it may have been modified in produce_profiles() )
        impurityPosition = self.powerstate.impurityPosition_transport #ModelOptions.get("impurityPosition", 1)

        # Produce right quantities (TGYRO -> powerstate.plasma)
        self.powerstate = tgyro.results[label].TGYROmodeledVariables(
            self.powerstate,
            useConvectiveFluxes=useConvectiveFluxes,
            includeFast=includeFast,
            impurityPosition=impurityPosition,
            UseFineGridTargets=UseFineGridTargets,
            OriginalFimp=OriginalFimp,
            forceZeroParticleFlux=forceZeroParticleFlux,
            provideTurbulentExchange=provideTurbulentExchange,
            provideTargets=self.powerstate.TargetOptions['ModelOptions']['TargetCalc'] == "tgyro",
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

    def _evaluate_tglf_neo(self):

        # ------------------------------------------------------------------------------------------------------------------------
        # Model Options
        # ------------------------------------------------------------------------------------------------------------------------

        ModelOptions = self.powerstate.TransportOptions["ModelOptions"]

        MODELparameters = ModelOptions.get("MODELparameters",None)
        includeFast = ModelOptions.get("includeFastInQi",False)
        launchMODELviaSlurm = ModelOptions.get("launchMODELviaSlurm", False)
        cold_start = ModelOptions.get("cold_start", False)
        provideTurbulentExchange = ModelOptions.get("TurbulentExchange", False)
        percentError = ModelOptions.get("percentError", [5, 1, 0.5])
        use_tglf_scan_trick = ModelOptions.get("use_tglf_scan_trick", None)
        cores_per_tglf_instance = ModelOptions.get("extra_params", {}).get('PORTALSparameters', {}).get("cores_per_tglf_instance", 1)

        # Grab impurity from powerstate ( because it may have been modified in produce_profiles() )
        impurityPosition = self.powerstate.impurityPosition_transport #ModelOptions.get("impurityPosition", 1)

        # ------------------------------------------------------------------------------------------------------------------------
        # tglf_neo_original: Run TGYRO workflow - TGLF + NEO in subfolder tglf_neo_original (original as in... without stds or merging)
        # ------------------------------------------------------------------------------------------------------------------------

        RadiisToRun = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]

        tgyro = TGYROtools.TGYRO(cdf=dummyCDF(self.folder, self.folder))
        tgyro.prep(self.folder, profilesclass_custom=self.profiles_transport)

        if launchMODELviaSlurm:
            print("\t- Launching TGYRO evaluation as a batch job")
        else:
            print("\t- Launching TGYRO evaluation as a terminal job")

        tgyro.run(
            subFolderTGYRO="tglf_neo_original",
            cold_start=cold_start,
            forceIfcold_start=True,
            special_radii=RadiisToRun,
            iterations=0,
            PredictionSet=[
                int("te" in self.powerstate.ProfilesPredicted),
                int("ti" in self.powerstate.ProfilesPredicted),
                int("ne" in self.powerstate.ProfilesPredicted),
            ],
            TGLFsettings=MODELparameters["transport_model"]["TGLFsettings"],
            extraOptionsTGLF=MODELparameters["transport_model"]["extraOptionsTGLF"],
            TGYRO_physics_options=MODELparameters["Physics_options"],
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
            RadiisToRun,
            self.powerstate.ProfilesPredicted,
            self.folder / "tglf_neo",
            percentError,
            impurityPosition=impurityPosition,
            includeFast=includeFast,
            provideTurbulentExchange=provideTurbulentExchange,
            use_tglf_scan_trick = use_tglf_scan_trick,
            cold_start=cold_start,
            extra_name = self.name,
            cores_per_tglf_instance=cores_per_tglf_instance
        )

        # Read again to capture errors
        tgyro.read(label="tglf_neo", folder=self.folder / "tglf_neo")

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Run TGLF standalone --> In preparation for the transition
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # from mitim_tools.gacode_tools import TGLFtools
        # tglf = TGLFtools.TGLF(rhos=RadiisToRun)
        # _ = tglf.prep(
        #     self.folder / 'stds',
        #     inputgacode=self.file_profs,
        #     recalculatePTOT=False, # Use what's in the input.gacode, which is what PORTALS TGYRO does
        #     cold_start=cold_start)

        # tglf.run(
        #     subFolderTGLF="tglf_neo_original",
        #     TGLFsettings=MODELparameters["transport_model"]["TGLFsettings"],
        #     cold_start=cold_start,
        #     forceIfcold_start=True,
        #     extraOptions=MODELparameters["transport_model"]["extraOptionsTGLF"],
        #     launchSlurm=launchMODELviaSlurm,
        #     slurm_setup={"cores": 4, "minutes": 1},
        # )

        # tglf.read(label="tglf_neo_original")

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        return tgyro

def tglf_scan_trick(
    fluxesTGYRO, 
    tgyro, 
    label, 
    RadiisToRun, 
    ProfilesPredicted, 
    impurityPosition=1, includeFast=False,  
    delta=0.02, 
    cold_start=False, 
    check_coincidence_thr=1E-2, 
    extra_name="", 
    remove_folders_out = False,
    cores_per_tglf_instance = 4 # e.g. 4 core per radius, since this is going to launch ~ Nr=5 x (Nv=6 x Nd=2 + 1) = 65 TGLFs at once
    ):

    print(f"\t- Running TGLF standalone scans ({delta = }) to determine relative errors")

    # Grab fluxes from TGYRO
    Qe_tgyro, Qi_tgyro, Ge_tgyro, GZ_tgyro, Mt_tgyro, Pexch_tgyro = fluxesTGYRO

    # ------------------------------------------------------------------------------------------------------------------------
    # TGLF scans
    # ------------------------------------------------------------------------------------------------------------------------

    # Prepare scan 

    tglf = tgyro.grab_tglf_objects(fromlabel=label, subfolder = 'tglf_explorations')

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

    name = 'turb_drives'

    tglf.rhos = RadiisToRun # To avoid the case in which TGYRO was run with an extra rho point

    # Estimate job minutes based on cases and cores (mostly IO I think at this moment, otherwise it should be independent on cases)
    num_cases = len(RadiisToRun) * len(variables_to_scan) * len(relative_scan)
    if cores_per_tglf_instance == 1:
        minutes = 10 * (num_cases / 60) # Ad-hoc formula
    else:
        minutes = 1 * (num_cases / 60) # Ad-hoc formula

    # Enforce minimum minutes
    minutes = max(2, minutes)

    tglf.runScanTurbulenceDrives(	
                    subFolderTGLF = name,
                    variablesDrives = variables_to_scan,
                    varUpDown     = relative_scan,
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

    Qe = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))
    Qi = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))
    Ge = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))
    GZ = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))

    cont = 0
    for vari in variables_to_scan:
        jump = tglf.scans[f'{name}_{vari}']['Qe'].shape[-1]

        Qe[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qe']
        Qi[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qi']
        Ge[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Ge']
        GZ[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Gi']
        cont += jump

    # ----------------------------------------------------
    # Do a check that TGLF scans are consistent with TGYRO
    Qe_err = np.abs( (Qe[:,0] - Qe_tgyro) / Qe_tgyro ) if 'te' in ProfilesPredicted else np.zeros_like(Qe[:,0])
    Qi_err = np.abs( (Qi[:,0] - Qi_tgyro) / Qi_tgyro ) if 'ti' in ProfilesPredicted else np.zeros_like(Qi[:,0])
    Ge_err = np.abs( (Ge[:,0] - Ge_tgyro) / Ge_tgyro ) if 'ne' in ProfilesPredicted else np.zeros_like(Ge[:,0])
    GZ_err = np.abs( (GZ[:,0] - GZ_tgyro) / GZ_tgyro ) if 'nZ' in ProfilesPredicted else np.zeros_like(GZ[:,0])

    F_err = np.concatenate((Qe_err, Qi_err, Ge_err, GZ_err))
    if F_err.max() > check_coincidence_thr:
        print(f"\t- TGLF scans are not consistent with TGYRO, maximum error = {F_err.max()*100:.2f}%",typeMsg="w")
        if 'te' in ProfilesPredicted:
            print('\t\t* Qe:',Qe_err)
        if 'ti' in ProfilesPredicted:
            print('\t\t* Qi:',Qi_err)
        if 'ne' in ProfilesPredicted:
            print('\t\t* Ge:',Ge_err)
        if 'nZ' in ProfilesPredicted:
            print('\t\t* GZ:',GZ_err)
    else:
        print(f"\t- TGLF scans are consistent with TGYRO, maximum error = {F_err.max()*100:.2f}%")
    # ----------------------------------------------------

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

    #TODO: Implement Mt and Pexch
    Mt_point, Pexch_point = Mt_tgyro, Pexch_tgyro
    Mt_std, Pexch_std = abs(Mt_point) * 0.1, abs(Pexch_point) * 0.1

    #TODO: Careful with fast particles

    return Qe_point, Qi_point, Ge_point, GZ_point, Mt_point, Pexch_point, Qe_std, Qi_std, Ge_std, GZ_std, Mt_std, Pexch_std

# **************************************************************************************************
# Functions
# **************************************************************************************************

def curateTGYROfiles(
    tgyroObject,
    label,
    RadiisToRun,
    ProfilesPredicted,
    folder,
    percentError,
    provideTurbulentExchange=False,
    impurityPosition=1,
    includeFast=False,
    use_tglf_scan_trick=None,
    cold_start=False,
    extra_name="",
    cores_per_tglf_instance = 4
    ):

    tgyro = tgyroObject.results[label]
    
    # Determine NEO and Target errors
    relativeErrorNEO = percentError[1] / 100.0
    relativeErrorTAR = percentError[2] / 100.0

    # **************************************************************************************************************************
    # TGLF
    # **************************************************************************************************************************
    
    # Grab fluxes
    Qe = tgyro.Qe_sim_turb[0, 1:]
    Qi = tgyro.QiIons_sim_turb[0, 1:] if includeFast else tgyro.QiIons_sim_turb_thr[0, 1:]
    Ge = tgyro.Ge_sim_turb[0, 1:]
    GZ = tgyro.Gi_sim_turb[impurityPosition, 0, 1:]
    Mt = tgyro.Mt_sim_turb[0, 1:]
    Pexch = tgyro.EXe_sim_turb[0, 1:]
    
    # Determine TGLF standard deviations
    if use_tglf_scan_trick is not None:

        if provideTurbulentExchange:
            print("> Turbulent exchange not implemented yet in TGLF scans", typeMsg="w") #TODO

        # --------------------------------------------------------------
        # If using the scan trick
        # --------------------------------------------------------------

        Qe, Qi, Ge, GZ, Mt, Pexch, QeE, QiE, GeE, GZE, MtE, PexchE = tglf_scan_trick(
            [Qe, Qi, Ge, GZ, Mt, Pexch],
            tgyroObject,
            label, 
            RadiisToRun, 
            ProfilesPredicted, 
            impurityPosition=impurityPosition, 
            includeFast=includeFast, 
            delta = use_tglf_scan_trick,
            cold_start=cold_start,
            extra_name=extra_name,
            cores_per_tglf_instance=cores_per_tglf_instance
            )

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

        relativeErrorTGLF = [percentError[0] / 100.0]*len(RadiisToRun)
    
        QeE = abs(Qe) * relativeErrorTGLF
        QiE = abs(Qi) * relativeErrorTGLF
        GeE = abs(Ge) * relativeErrorTGLF
        GZE = abs(GZ) * relativeErrorTGLF
        MtE = abs(Mt) * relativeErrorTGLF
        PexchE = abs(Pexch) * relativeErrorTGLF

    # **************************************************************************************************************************
    # Neo
    # **************************************************************************************************************************

    QeNeo = tgyro.Qe_sim_neo[0, 1:]
    if includeFast:
        QiNeo = tgyro.QiIons_sim_neo[0, 1:]
    else:
        QiNeo = tgyro.QiIons_sim_neo_thr[0, 1:]
    GeNeo = tgyro.Ge_sim_neo[0, 1:]
    GZNeo = tgyro.Gi_sim_neo[impurityPosition, 0, 1:]
    MtNeo = tgyro.Mt_sim_neo[0, 1:]

    QeNeoE = abs(tgyro.Qe_sim_neo[0, 1:]) * relativeErrorNEO
    if includeFast:
        QiNeoE = abs(tgyro.QiIons_sim_neo[0, 1:]) * relativeErrorNEO
    else:
        QiNeoE = abs(tgyro.QiIons_sim_neo_thr[0, 1:]) * relativeErrorNEO
    GeNeoE = abs(tgyro.Ge_sim_neo[0, 1:]) * relativeErrorNEO
    GZNeoE = abs(tgyro.Gi_sim_neo[impurityPosition, 0, 1:]) * relativeErrorNEO
    MtNeoE = abs(tgyro.Mt_sim_neo[0, 1:]) * relativeErrorNEO

    # Merge

    PORTALScgyro.modifyFLUX(
        tgyro,
        folder,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        QeNeo=QeNeo,
        QiNeo=QiNeo,
        GeNeo=GeNeo,
        GZNeo=GZNeo,
        MtNeo=MtNeo,
        impurityPosition=impurityPosition,
    )

    PORTALScgyro.modifyFLUX(
        tgyro,
        folder,
        QeE,
        QiE,
        GeE,
        GZE,
        MtE,
        PexchE,
        QeNeo=QeNeoE,
        QiNeo=QiNeoE,
        GeNeo=GeNeoE,
        GZNeo=GZNeoE,
        MtNeo=MtNeoE,
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

    PORTALScgyro.modifyEVO(
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
    PORTALScgyro.modifyEVO(
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


def profilesToShare(self):
    if "extra_params" in self.powerstate.TransportOptions["ModelOptions"] and "folder" in self.powerstate.TransportOptions["ModelOptions"]["extra_params"]:
        whereFolder = IOtools.expandPath(self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"] / "Outputs" / "portals_profiles")
        if not whereFolder.exists():
            IOtools.askNewFolder(whereFolder)

        fil = whereFolder / f"input.gacode.{self.evaluation_number}"
        shutil.copy2(self.file_profs, fil)
        shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
        shutil.copy2(self.file_profs_targets, fil.parent / f"{fil.name}.new_fromtgyro_modified")
        print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
    else:
        print("\t- Could not move files", typeMsg="w")


def cgyro_trick(self,FolderEvaluation_TGYRO):

    with open(FolderEvaluation_TGYRO / "mitim_flag", "w") as f:
        f.write("0")

    # **************************************************************************************************************************
    # Print Information
    # **************************************************************************************************************************

    txt = "\nFluxes to be matched by CGYRO ( TARGETS - NEO ):"

    for var, varn in zip(
        ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
        ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
    ):
        txt += f"\n{var}        = "
        for j in range(self.powerstate.plasma["rho"].shape[1] - 1):
            txt += f"{self.powerstate.plasma[varn][0,j+1]:.6f}   "

    for var, varn in zip(
        ["Qe (MW/m^2)", "Qi (MW/m^2)", "Ce (MW/m^2)", "CZ (MW/m^2)", "Mt (J/m^2) "],
        ["Pe", "Pi", "Ce", "CZ", "Mt"],
    ):
        txt += f"\n{var}  = "
        for j in range(self.powerstate.plasma["rho"].shape[1] - 1):
            txt += f"{self.powerstate.plasma[varn][0,j+1]-self.powerstate.plasma[f'{varn}_tr_neo'][0,j+1]:.4e}   "

    print(txt)

    # Copy profiles so that later it is easy to grab all the input.gacodes that were evaluated
    profilesToShare(self)

    # **************************************************************************************************************************
    # Evaluate CGYRO
    # **************************************************************************************************************************

    PORTALScgyro.evaluateCGYRO(
        self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["PORTALSparameters"],
        self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"],
        self.evaluation_number,
        FolderEvaluation_TGYRO,
        self.file_profs,
        self.powerstate.plasma["roa"][0,1:],
        self.powerstate.ProfilesPredicted,
    )

    # **************************************************************************************************************************
    # EXTRA
    # **************************************************************************************************************************

    # Make tensors
    for i in ["Pe_tr_turb", "Pi_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "Mt_tr_turb"]:
        try:
            self.powerstate.plasma[i] = torch.from_numpy(self.powerstate.plasma[i]).to(self.powerstate.dfT).unsqueeze(0)
        except:
            pass

    # Write a flag indicating this was performed, to avoid an issue that... the script crashes when it has copied tglf_neo, without cgyro_trick modification
    with open(FolderEvaluation_TGYRO / "mitim_flag", "w") as f:
        f.write("1")

def dummyCDF(GeneralFolder, FolderEvaluation):
    """
    This routine creates path to a dummy CDF file in FolderEvaluation, with the name "simulation_evaluation.CDF"

    GeneralFolder, e.g.    ~/runs_portals/run10/
    FolderEvaluation, e.g. ~/runs_portals/run10000/Execution/Evaluation.0/model_complete/
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
