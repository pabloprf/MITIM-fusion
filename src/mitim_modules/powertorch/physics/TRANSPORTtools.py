import copy, os, torch
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_tools.gacode_tools import TGYROtools
from mitim_modules.portals.aux import PORTALScgyro
from mitim_modules.powertorch.aux import TRANSFORMtools

from mitim_tools.misc_tools.IOtools import printMsg as print

# ------------------------------------------------------------------
# SIMPLE
# ------------------------------------------------------------------


def diffusion_model(self, ModelOptions, nameRun="test"):
    Pe_tr = PLASMAtools.conduction(
        self.plasma["ne"],
        self.plasma["te"],
        ModelOptions["chi_e"],
        self.plasma["aLte"],
        self.plasma["a"],
    )
    Pi_tr = PLASMAtools.conduction(
        self.plasma["ni"].sum(axis=-1),
        self.plasma["ti"],
        ModelOptions["chi_i"],
        self.plasma["aLti"],
        self.plasma["a"],
    )

    self.plasma["Pe_tr_turb"] = Pe_tr[:, 1:] * 2 / 3
    self.plasma["Pi_tr_turb"] = Pi_tr[:, 1:] * 2 / 3

    self.plasma["Pe_tr_neo"] = Pe_tr[:, 1:] * 1 / 3
    self.plasma["Pi_tr_neo"] = Pi_tr[:, 1:] * 1 / 3

    self.plasma["Pe_tr"] = self.plasma["Pe_tr_turb"] + self.plasma["Pe_tr_neo"]
    self.plasma["Pi_tr"] = self.plasma["Pi_tr_turb"] + self.plasma["Pi_tr_neo"]

    self.plasma["Pe"] = self.plasma["Pe"][:, 1:]  # This should be fixed later
    self.plasma["Pi"] = self.plasma["Pi"][:, 1:]  # This should be fixed later

    self.plasma["Ce_tr_turb"] = self.plasma["Pe_tr"] * 0.0
    self.plasma["Ce_tr_neo"] = self.plasma["Pe_tr"] * 0.0
    self.plasma["Ce_tr"] = self.plasma["Pe_tr"] * 0.0
    self.plasma["Ce"] = self.plasma["Pe"] * 0.0


# ------------------------------------------------------------------
# SURROGATE
# ------------------------------------------------------------------


def surrogate_model(self, ModelOptions, nameRun="test"):
    """
    flux_fun as given in ModelOptions must produce Q and Qtargets in order of te,ti,ne
    """

    Q, QT = ModelOptions["flux_fun"](self.Xcurrent[0])

    numeach = self.plasma["rho"].shape[1] - 1

    for c, i in enumerate(self.ProfilesPredicted):
        if i == "te":
            self.plasma["Pe_tr"] = Q[:, numeach * c : numeach * (c + 1)]
        if i == "ti":
            self.plasma["Pi_tr"] = Q[:, numeach * c : numeach * (c + 1)]
        if i == "ne":
            self.plasma["Ce_tr"] = Q[:, numeach * c : numeach * (c + 1)]
        if i == "nZ":
            self.plasma["CZ_tr"] = Q[:, numeach * c : numeach * (c + 1)]

    for c2, i in enumerate(self.ProfilesPredicted):
        if i == "te":
            self.plasma["Pe"] = QT[:, numeach * c2 : numeach * (c2 + 1)]
        if i == "ti":
            self.plasma["Pi"] = QT[:, numeach * c2 : numeach * (c2 + 1)]
        if i == "ne":
            self.plasma["Ce"] = QT[:, numeach * c2 : numeach * (c2 + 1)]
        if i == "nZ":
            self.plasma["CZ"] = QT[:, numeach * c2 : numeach * (c2 + 1)]


# ------------------------------------------------------------------
# FULL TGYRO
# ------------------------------------------------------------------


def tgyro_model(
    self,
    ModelOptions,
    name="test",
    folder="~/scratch/",
    provideTargets=True,
    TypeTransport="tglf_neo-tgyro",
    extra_params={},
):
    # ******************************************* Parameters that are needed ***************************************************

    TGYROparameters = ModelOptions["TGYROparameters"]
    TGLFparameters = ModelOptions["TGLFparameters"]
    includeFast = ModelOptions["includeFastInQi"]
    impurityPosition = ModelOptions["impurityPosition"]
    useConvectiveFluxes = ModelOptions["useConvectiveFluxes"]
    ProfilesPredicted = TGYROparameters["ProfilesPredicted"]
    UseFineGridTargets = ModelOptions["UseFineGridTargets"]

    launchTGYROviaSlurm = ModelOptions.get("launchTGYROviaSlurm", False)
    restart = ModelOptions.get("restart", False)
    provideTurbulentExchange = ModelOptions.get("TurbulentExchange", False)
    profiles_postprocessing_fun = ModelOptions.get("profiles_postprocessing_fun", None)
    OriginalFimp = ModelOptions.get("OriginalFimp", 1.0)
    forceZeroParticleFlux = ModelOptions.get("forceZeroParticleFlux", False)
    percentError = ModelOptions.get("percentError", [5, 1, 0.5])
    FolderEvaluation_TGYRO = IOtools.expandPath(folder)

    # ------------------------------------------------------------------------------------------------------------------------
    # Prepare case
    # ------------------------------------------------------------------------------------------------------------------------

    labels_results = []

    ProfilesTGYRO = [
        int("te" in self.ProfilesPredicted),
        int("ti" in self.ProfilesPredicted),
        int("ne" in self.ProfilesPredicted),
    ]

    # Write this updated profiles class (with parameterized profiles)
    self.file_profs = f"{FolderEvaluation_TGYRO}/input.gacode"
    profiles = self.insertProfiles(
        self.profiles,
        writeFile=self.file_profs,
        applyCorrections=TGYROparameters["applyCorrections"],
    )

    # copy for future modifications
    self.file_profs_mod = f"{self.file_profs}_modified"
    os.system(f"cp {self.file_profs} {self.file_profs_mod}")

    # ------------------------------------------------------------------------------------------------------------------------
    # 1. tglf_neo_original: Run TGYRO workflow - TGLF + NEO in subfolder tglf_neo_original (original as in... without stds or merging)
    # ------------------------------------------------------------------------------------------------------------------------

    self.tgyro_current = TGYROtools.TGYRO(cdf=dummyCDF(folder, FolderEvaluation_TGYRO))
    self.tgyro_current.prep(FolderEvaluation_TGYRO, profilesclass_custom=profiles)

    RadiisToRun = [
        self.plasma["rho"][0, 1:][i].item()
        for i in range(len(self.plasma["rho"][0, 1:]))
    ]

    if launchTGYROviaSlurm:
        print("\t- Launching TGYRO evaluation as a batch job")
    else:
        print("\t- Launching TGYRO evaluation as a terminal job")

    self.tgyro_current.run(
        subFolderTGYRO=f"tglf_neo_original/",
        restart=restart,
        forceIfRestart=True,
        special_radii=RadiisToRun,
        iterations=0,
        PredictionSet=ProfilesTGYRO,
        TGLFsettings=TGLFparameters["TGLFsettings"],
        extraOptionsTGLF=TGLFparameters["extraOptionsTGLF"],
        TGYRO_physics_options=TGYROparameters["TGYRO_physics_options"],
        launchSlurm=launchTGYROviaSlurm,
        minutesJob=5,
        forcedName=name,
    )

    self.tgyro_current.read(label=f"tglf_neo_original")

    # Copy one with evaluated targets
    self.file_profs_targets = f"{self.tgyro_current.FolderTGYRO}/input.gacode.new"

    # ------------------------------------------------------------------------------------------------------------------------
    # 2. tglf_neo: Write TGLF, NEO and TARGET errors in tgyro files as well
    # ------------------------------------------------------------------------------------------------------------------------

    # Copy original TGYRO folder
    if os.path.exists(f"{FolderEvaluation_TGYRO}/tglf_neo/"):
        os.system(f"{FolderEvaluation_TGYRO}/tglf_neo/")
    os.system(
        f"cp -r {FolderEvaluation_TGYRO}/tglf_neo_original {FolderEvaluation_TGYRO}/tglf_neo"
    )

    # Add errors and merge fluxes as we would do if this was a CGYRO run
    curateTGYROfiles(
        self.tgyro_current.results[f"tglf_neo_original"],
        f"{FolderEvaluation_TGYRO}/tglf_neo/",
        percentError,
        impurityPosition=impurityPosition,
        includeFast=includeFast,
    )

    # Read again to capture errors
    self.tgyro_current.read(
        label=f"tglf_neo", folder=f"{FolderEvaluation_TGYRO}/tglf_neo/"
    )
    labels_results.append("tglf_neo")

    # Produce right quantities

    TGYROresults = self.tgyro_current.results["tglf_neo"]

    portals_variables = TGYROresults.TGYROmodeledVariables(
        useConvectiveFluxes=useConvectiveFluxes,
        includeFast=includeFast,
        impurityPosition=impurityPosition,
        UseFineGridTargets=UseFineGridTargets,
        OriginalFimp=OriginalFimp,
        forceZeroParticleFlux=forceZeroParticleFlux,
        dfT=self.dfT,
    )

    # ------------------------------------------------------------------------------------------------------------------------
    # 3. cgyro_neo: Trick to fake a tgyro output to reflect CGYRO
    # ------------------------------------------------------------------------------------------------------------------------

    if TypeTransport == "cgyro_neo-tgyro":
        portals_variables_orig = copy.deepcopy(portals_variables)

        print(
            "\t- Checking whether cgyro_neo folder exists and it was written correctly via cgyro_trick..."
        )

        correctly_run = os.path.exists(f"{FolderEvaluation_TGYRO}/cgyro_neo")
        if correctly_run:
            print("\t\t- Folder exists, but was cgyro_trick run?")
            with open(f"{FolderEvaluation_TGYRO}/cgyro_neo/mitim_flag", "r") as f:
                correctly_run = bool(float(f.readline()))

        if correctly_run:
            print("\t\t\t* Yes, it was", typeMsg="w")
        else:
            print("\t\t\t* No, it was not, repating process", typeMsg="i")

            # Copy tglf_neo results
            os.system(
                f"cp -r {FolderEvaluation_TGYRO}/tglf_neo {FolderEvaluation_TGYRO}/cgyro_neo"
            )

            # CGYRO writter
            cgyro_trick(
                self,
                f"{FolderEvaluation_TGYRO}/cgyro_neo",
                portals_variables=portals_variables,
                profiles_postprocessing_fun=profiles_postprocessing_fun,
                extra_params=extra_params,
                name=name,
            )

        # Read TGYRO files and construct portals variables

        self.tgyro_current.read(
            label="cgyro_neo", folder=f"{FolderEvaluation_TGYRO}/cgyro_neo"
        )  # Re-read TGYRO to store
        TGYROresults = self.tgyro_current.results["cgyro_neo"]
        labels_results.append("cgyro_neo")

        portals_variables = TGYROresults.TGYROmodeledVariables(
            useConvectiveFluxes=useConvectiveFluxes,
            includeFast=includeFast,
            impurityPosition=impurityPosition,
            UseFineGridTargets=UseFineGridTargets,
            OriginalFimp=OriginalFimp,
            forceZeroParticleFlux=forceZeroParticleFlux,
            dfT=self.dfT,
        )

        print(f"\t- Checking model modifications:")
        for r in ["Qe_turb", "Qi_turb", "Ge_turb", "GZ_turb", "Mt_turb", "PexchTurb"]:
            print(
                f"\t\t{r}(tglf)  = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(portals_variables_orig[r][0][1:],portals_variables_orig[r+'_stds'][0][1:]) ])}"
            )
            print(
                f"\t\t{r}(cgyro) = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(portals_variables[r][0][1:],portals_variables[r+'_stds'][0][1:]) ])}"
            )

        # **
        self.tgyro_current.results["use"] = self.tgyro_current.results["cgyro_neo"]

    else:
        # copy profiles too!
        profilesToShare(self, extra_params)

        # **
        self.tgyro_current.results["use"] = self.tgyro_current.results["tglf_neo"]

    labels_results.append("use")

    # --------------------------------------------------------------------------------------------------------------------------------
    # TURBULENCE and NEOCLASSICAL
    # --------------------------------------------------------------------------------------------------------------------------------

    iteration = 0
    tuple_rho_indeces = ()
    for rho in self.tgyro_current.rhosToSimulate:
        tuple_rho_indeces += (np.argmin(np.abs(rho - TGYROresults.rho)),)

    mapper = {
        "Pe_tr_turb": "Qe_turb",
        "Pi_tr_turb": "Qi_turb",
        "Ce_tr_turb": "Ge_turb",
        "CZ_tr_turb": "GZ_turb",
        "Mt_tr_turb": "Mt_turb",
        "Pe_tr_neo": "Qe_neo",
        "Pi_tr_neo": "Qi_neo",
        "Ce_tr_neo": "Ge_neo",
        "CZ_tr_neo": "GZ_neo",
        "Mt_tr_neo": "Mt_neo",
    }

    if provideTurbulentExchange:
        mapper.update(
            {"PexchTurb": "PexchTurb"}
        )  # I need to do this outside of provideTargets because powerstate cannot compute this

    if provideTargets:
        mapper.update(
            {
                "Pe": "Qe",
                "Pi": "Qi",
                "Ce": "Ge",
                "CZ": "GZ",
                "Mt": "Mt",
            }
        )
    else:
        self.plasma["Pe"] = self.plasma["Pe"][:, 1:]
        self.plasma["Pi"] = self.plasma["Pi"][:, 1:]
        self.plasma["Ce"] = self.plasma["Ce"][:, 1:]
        self.plasma["CZ"] = self.plasma["CZ"][:, 1:]
        self.plasma["Mt"] = self.plasma["Mt"][:, 1:]

        percentErrorTarget = percentError[2] / 100.0

        self.plasma["Pe_stds"] = abs(self.plasma["Pe"]) * percentErrorTarget
        self.plasma["Pi_stds"] = abs(self.plasma["Pi"]) * percentErrorTarget
        self.plasma["Ce_stds"] = abs(self.plasma["Ce"]) * percentErrorTarget
        self.plasma["CZ_stds"] = abs(self.plasma["CZ"]) * percentErrorTarget
        self.plasma["Mt_stds"] = abs(self.plasma["Mt"]) * percentErrorTarget

    for ikey in mapper:
        self.plasma[ikey] = (
            torch.from_numpy(
                portals_variables[mapper[ikey]][iteration, tuple_rho_indeces]
            )
            .to(self.dfT)
            .unsqueeze(0)
        )
        self.plasma[ikey + "_stds"] = (
            torch.from_numpy(
                portals_variables[mapper[ikey] + "_stds"][iteration, tuple_rho_indeces]
            )
            .to(self.dfT)
            .unsqueeze(0)
        )

    # ------------------------------------------------------------------------------------------------------------------------
    # Sum here, after modifications
    # ------------------------------------------------------------------------------------------------------------------------

    self.plasma["Pe_tr"] = self.plasma["Pe_tr_turb"] + self.plasma["Pe_tr_neo"]
    self.plasma["Pi_tr"] = self.plasma["Pi_tr_turb"] + self.plasma["Pi_tr_neo"]
    self.plasma["Ce_tr"] = self.plasma["Ce_tr_turb"] + self.plasma["Ce_tr_neo"]
    self.plasma["CZ_tr"] = self.plasma["CZ_tr_turb"] + self.plasma["CZ_tr_neo"]
    self.plasma["Mt_tr"] = self.plasma["Mt_tr_turb"] + self.plasma["Mt_tr_neo"]

    # ------------------------------------------------------------------------------------------------------------------------
    # For consistency, modify input.gacode.new with the targets used in PORTALS (i.e. sometimes with POWESTATE calculations)
    # ------------------------------------------------------------------------------------------------------------------------

    for lab in labels_results:
        print(
            f"\t- Inserting PORTALS powers into {IOtools.clipstr(self.tgyro_current.results[lab].profiles_final.file)}",
        )
        TRANSFORMtools.insertPowersNew(
            self.tgyro_current.results[lab].profiles_final, state=self
        )
        self.tgyro_current.results[lab].profiles_final.writeCurrentStatus(
            file=self.tgyro_current.results[lab].profiles_final.file
        )

    return TGYROresults


def curateTGYROfiles(
    tgyro, folder, percentError, impurityPosition=1, includeFast=False
):
    # TGLF ---------------------------------------------------------------------------------------------------------

    Qe = tgyro.Qe_sim_turb[0, 1:]
    if includeFast:
        Qi = tgyro.QiIons_sim_turb[0, 1:]
    else:
        Qi = tgyro.QiIons_sim_turb_thr[0, 1:]
    Ge = tgyro.Ge_sim_turb[0, 1:]
    GZ = tgyro.Gi_sim_turb[impurityPosition - 1, 0, 1:]
    Mt = tgyro.Mt_sim_turb[0, 1:]
    Pexch = tgyro.EXe_sim_turb[0, 1:]

    percentErrorTGLF = percentError[0] / 100.0

    QeE = abs(tgyro.Qe_sim_turb[0, 1:]) * percentErrorTGLF
    if includeFast:
        QiE = abs(tgyro.QiIons_sim_turb[0, 1:]) * percentErrorTGLF
    else:
        QiE = abs(tgyro.QiIons_sim_turb_thr[0, 1:]) * percentErrorTGLF
    GeE = abs(tgyro.Ge_sim_turb[0, 1:]) * percentErrorTGLF
    GZE = abs(tgyro.Gi_sim_turb[impurityPosition - 1, 0, 1:]) * percentErrorTGLF
    MtE = abs(tgyro.Mt_sim_turb[0, 1:]) * percentErrorTGLF
    PexchE = abs(tgyro.EXe_sim_turb[0, 1:]) * percentErrorTGLF

    # Neo ----------------------------------------------------------------------------------------------------------

    QeNeo = tgyro.Qe_sim_neo[0, 1:]
    if includeFast:
        QiNeo = tgyro.QiIons_sim_neo[0, 1:]
    else:
        QiNeo = tgyro.QiIons_sim_neo_thr[0, 1:]
    GeNeo = tgyro.Ge_sim_neo[0, 1:]
    GZNeo = tgyro.Gi_sim_neo[impurityPosition - 1, 0, 1:]
    MtNeo = tgyro.Mt_sim_neo[0, 1:]

    percentErrorNeo = percentError[1] / 100.0

    QeNeoE = abs(tgyro.Qe_sim_neo[0, 1:]) * percentErrorNeo
    if includeFast:
        QiNeoE = abs(tgyro.QiIons_sim_neo[0, 1:]) * percentErrorNeo
    else:
        QiNeoE = abs(tgyro.QiIons_sim_neo_thr[0, 1:]) * percentErrorNeo
    GeNeoE = abs(tgyro.Ge_sim_neo[0, 1:]) * percentErrorNeo
    GZNeoE = abs(tgyro.Gi_sim_neo[impurityPosition - 1, 0, 1:]) * percentErrorNeo
    MtNeoE = abs(tgyro.Mt_sim_neo[0, 1:]) * percentErrorNeo

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

    # Targets -------------------------------------------------------------------------------------------------------

    percentErrorTarget = percentError[2] / 100.0

    QeTargetE = abs(tgyro.Qe_tar[0, 1:]) * percentErrorTarget
    QiTargetE = abs(tgyro.Qi_tar[0, 1:]) * percentErrorTarget
    GeTargetE = abs(tgyro.Ge_tar[0, 1:]) * percentErrorTarget
    GZTargetE = GeTargetE * 0.0
    MtTargetE = abs(tgyro.Mt_tar[0, 1:]) * percentErrorTarget

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


def profilesToShare(self, extra_params):
    if "folder" in extra_params:
        whereFolder = IOtools.expandPath(
            extra_params["folder"] + "/Outputs/ProfilesEvaluated/"
        )
        if not os.path.exists(whereFolder):
            IOtools.askNewFolder(whereFolder)

        fil = f"{whereFolder}/input.gacode.{extra_params['numPORTALS']}"
        os.system(f"cp {self.file_profs_mod} {fil}")
        os.system(f"cp {self.file_profs} {fil}_unmodified")
        os.system(f"cp {self.file_profs_targets} {fil}_unmodified.new")
        print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
    else:
        print("\t- Could not move files", typeMsg="w")


def cgyro_trick(
    self,
    FolderEvaluation_TGYRO,
    portals_variables=None,
    profiles_postprocessing_fun=None,
    extra_params={},
    name="",
):
    with open(f"{FolderEvaluation_TGYRO}/mitim_flag", "w") as f:
        f.write("0")

    # **************************************************************************************************************************
    # Print Information
    # **************************************************************************************************************************

    if portals_variables is not None:
        txt = "\nFluxes to be matched by CGYRO ( TARGETS - NEO ):"

        for var, varn in zip(
            ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
            ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
        ):
            txt += f"\n{var}        = "
            for j in range(self.plasma["rho"].shape[1] - 1):
                txt += f"{self.plasma[varn][0,j+1]:.6f}   "

        for var, varn in zip(
            ["Qe (MW/m^2)", "Qi (MW/m^2)", "Ce (MW/m^2)", "CZ (MW/m^2)", "Mt (J/m^2) "],
            ["Qe", "Qi", "Ge", "GZ", "Mt"],
        ):
            txt += f"\n{var}  = "
            for j in range(self.plasma["rho"].shape[1] - 1):
                txt += f"{portals_variables[varn][0,j+1]-portals_variables[f'{varn}_neo'][0,j+1]:.4e}   "

        print(txt)

    # **************************************************************************************************************************
    # Modification to input.gacode (e.g. lump impurities)
    # **************************************************************************************************************************

    if profiles_postprocessing_fun is not None:
        print(
            f"\t- Modifying input.gacode.modified to run transport calculations based on {profiles_postprocessing_fun}",
            typeMsg="i",
        )
        profiles = profiles_postprocessing_fun(self.file_profs_mod)

    # Copy profiles so that later it is easy to grab all the input.gacodes that were evaluated
    profilesToShare(self, extra_params)

    # **************************************************************************************************************************
    # Evaluate CGYRO
    # **************************************************************************************************************************

    PORTALScgyro.evaluateCGYRO(
        extra_params["PORTALSparameters"],
        extra_params["folder"],
        extra_params["numPORTALS"],
        FolderEvaluation_TGYRO,
        self.file_profs,
        rad=self.plasma["rho"].shape[1] - 1,
    )

    # **************************************************************************************************************************
    # EXTRA
    # **************************************************************************************************************************

    # Make tensors
    for i in ["Pe_tr_turb", "Pi_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "Mt_tr_turb"]:
        try:
            self.plasma[i] = torch.from_numpy(self.plasma[i]).to(self.dfT).unsqueeze(0)
        except:
            pass

    # Write a flag indicating this was performed, to avoid an issue that... the script crashes when it has copied tglf_neo, without cgyro_trick modification
    with open(f"{FolderEvaluation_TGYRO}/mitim_flag", "w") as f:
        f.write("1")


def full_targets(TGYROresults, tuple_rho_indeces, iteration=0):
    """
    in MW
    """

    Pfuse = TGYROresults.Qe_tarMW_fus[iteration, tuple_rho_indeces]
    Pfusi = TGYROresults.Qi_tarMW_fus[iteration, tuple_rho_indeces]
    Pie = TGYROresults.Qe_tarMW_exch[iteration, tuple_rho_indeces]
    Prad_bremms = -TGYROresults.Qe_tarMW_brem[iteration, tuple_rho_indeces]
    Prad_sync = -TGYROresults.Qe_tarMW_sync[iteration, tuple_rho_indeces]
    Prad_line = -TGYROresults.Qe_tarMW_line[iteration, tuple_rho_indeces]

    return Pfuse, Pfusi, Pie, Prad_bremms, Prad_sync, Prad_line


def dummyCDF(GeneralFolder, FolderEvaluation):
    """
    This routine creates path to a dummy CDF file in FolderEvaluation, with the name "simulation_evaluation.CDF"

    GeneralFolder, e.g.    ~/runs_portals/run10/
    FolderEvaluation, e.g. ~/runs_portals/run10000/Execution/Evaluation.0//model_complete/
    """

    # ------- Name construction for scratch folders in parallel ----------------

    GeneralFolder = IOtools.expandPath(GeneralFolder, ensurePathValid=True)

    subname = GeneralFolder.split("/")[-1]  # run10 (simulation)
    if len(subname) == 0:
        subname = GeneralFolder.split("/")[-2]

    name = FolderEvaluation.split(".")[-1].split("/")[0]  # 0 	(evaluation #)

    cdf = f"{FolderEvaluation}/{subname}_ev{name}.CDF"

    return cdf
