import copy
import os
import torch
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_tools.gacode_tools import TGYROtools
from mitim_modules.portals.aux import PORTALScgyro
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed


class power_transport:
    '''
    Default class for power transport models, change "evaluate" method to implement a new model and produce_profiles if the model requires written input.gacode written

    Notes:
        - After evaluation, the self.model_results attribute will contain the results of the model, which can be used for plotting and analysis
        - model results can have .plot() method that can grab kwargs or be similar to TGYRO plot

    '''
    def __init__(self, powerstate, name = "test", folder = "~/scratch/", evaluation_number = 0):

        self.name = name
        self.folder = IOtools.expandPath(folder)
        self.evaluation_number = evaluation_number
        self.powerstate = powerstate

        # Allowed fluxes in powerstate so far
        self.quantities = ['Pe', 'Pi', 'Ce', 'CZ', 'Mt']

        # Each flux has a turbulent and neoclassical component
        self.variables = [f'{i}_tr_turb' for i in self.quantities] + [f'{i}_tr_neo' for i in self.quantities]

        # Each flux component has a standard deviation
        self.variables += [f'{i}_stds' for i in self.variables]

        # There is also turbulent exchange
        self.variables += ['PexchTurb', 'PexchTurb_stds']

        # And total transport flux
        self.variables += [f'{i}_tr' for i in self.quantities]

        # Model results is None by default, but can be assigned in evaluate
        self.model_results = None

        # Assign zeros to transport ones if not evaluated
        for i in self.variables:
            self.powerstate.plasma[i] = self.powerstate.plasma["te"] * 0.0

        # There is also target components
        self.variables += [f'{i}' for i in self.quantities] + [f'{i}_stds' for i in self.quantities]

        # ----------------------------------------------------------------------------------------
        # labels for plotting
        # ----------------------------------------------------------------------------------------

        self.powerstate.labelsFluxes = {
            "te": "$Q_e$ ($MW/m^2$)",
            "ti": "$Q_i$ ($MW/m^2$)",
            "ne": (
                "$Q_{conv}$ ($MW/m^2$)"
                if self.powerstate.TransportOptions["ModelOptions"].get("useConvectiveFluxes", True)
                else "$\\Gamma_e$ ($10^{20}/s/m^2$)"
            ),
            "nZ": (
                "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)"
                if self.powerstate.TransportOptions["ModelOptions"].get("useConvectiveFluxes", True)
                else "$\\Gamma_Z$ $\\cdot f_{Z,0}$ ($10^{20}/s/m^2$)"
            ),
            "w0": "$M_T$ ($J/m^2$)",
        }

    def _produce_profiles(self,deriveQuantities=True):

        self.applyCorrections = (
            self.powerstate.TransportOptions["ModelOptions"]
            .get("MODELparameters", {})
            .get("applyCorrections", {})
        )

        # Write this updated profiles class (with parameterized profiles and target powers)
        self.file_profs = f"{IOtools.expandPath(self.folder)}/input.gacode"
        self.powerstate.profiles = self.powerstate.insertProfiles(
            self.powerstate.profiles,
            writeFile=self.file_profs,
            applyCorrections=self.applyCorrections,
            rederive_profiles = deriveQuantities,    # Derive quantities so that it's ready for analysis and plotting later
            insertPowers = deriveQuantities,         # Insert powers so that Q, Pfus and all that it's consistent when read later
        )

    def clean(self):

        # Insert powers again in case they come from TGYRO instead of powerstate previous step
        if self.powerstate.TargetCalc == "tgyro":
            self.powerstate.profiles = self.powerstate.insertProfiles(
                self.powerstate.profiles,
                writeFile=self.file_profs,
                applyCorrections=self.applyCorrections,
                insertPowers=True,      # So that later I can read it fully with the powers, fusion, etc
            )

    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    def evaluate(self):
        print("Nothing to evaluate", typeMsg="w")
        pass

# ----------------------------------------------------------------------------------------------------
# FULL TGYRO
# ----------------------------------------------------------------------------------------------------

class tgyro_model(power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        self._produce_profiles()

    def evaluate(self):

        # After producing the profiles, copy for future modifications
        self.file_profs_mod = f"{self.file_profs}_modified"
        os.system(f"cp {self.file_profs} {self.file_profs_mod}")

        # ------------------------------------------------------------------------------------------------------------------------
        # Model Options
        # ------------------------------------------------------------------------------------------------------------------------

        MODELparameters = self.powerstate.TransportOptions["ModelOptions"].get("MODELparameters",None)
        includeFast = self.powerstate.TransportOptions["ModelOptions"].get("includeFastInQi",False)
        impurityPosition = self.powerstate.TransportOptions["ModelOptions"].get("impurityPosition", 1)
        useConvectiveFluxes = self.powerstate.TransportOptions["ModelOptions"].get("useConvectiveFluxes", True)
        UseFineGridTargets = self.powerstate.TransportOptions["ModelOptions"].get("UseFineGridTargets", False)
        launchMODELviaSlurm = self.powerstate.TransportOptions["ModelOptions"].get("launchMODELviaSlurm", False)
        restart = self.powerstate.TransportOptions["ModelOptions"].get("restart", False)
        provideTurbulentExchange = self.powerstate.TransportOptions["ModelOptions"].get("TurbulentExchange", False)
        profiles_postprocessing_fun = self.powerstate.TransportOptions["ModelOptions"].get("profiles_postprocessing_fun", None)
        OriginalFimp = self.powerstate.TransportOptions["ModelOptions"].get("OriginalFimp", 1.0)
        forceZeroParticleFlux = self.powerstate.TransportOptions["ModelOptions"].get("forceZeroParticleFlux", False)
        percentError = self.powerstate.TransportOptions["ModelOptions"].get("percentError", [5, 1, 0.5])

        # ------------------------------------------------------------------------------------------------------------------------
        # 1. tglf_neo_original: Run TGYRO workflow - TGLF + NEO in subfolder tglf_neo_original (original as in... without stds or merging)
        # ------------------------------------------------------------------------------------------------------------------------

        RadiisToRun = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        tgyro = TGYROtools.TGYRO(cdf=dummyCDF(self.folder, self.folder))
        tgyro.prep(self.folder, profilesclass_custom=self.powerstate.profiles)

        if launchMODELviaSlurm:
            print("\t- Launching TGYRO evaluation as a batch job")
        else:
            print("\t- Launching TGYRO evaluation as a terminal job")

        tgyro.run(
            subFolderTGYRO="tglf_neo_original/",
            restart=restart,
            forceIfRestart=True,
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
        self.file_profs_targets = f"{tgyro.FolderTGYRO}/input.gacode.new"

        # ------------------------------------------------------------------------------------------------------------------------
        # 2. tglf_neo: Write TGLF, NEO and TARGET errors in tgyro files as well
        # ------------------------------------------------------------------------------------------------------------------------

        # Copy original TGYRO folder
        if os.path.exists(f"{self.folder}/tglf_neo/"):
            os.system(f"{self.folder}/tglf_neo/")
        os.system(
            f"cp -r {self.folder}/tglf_neo_original {self.folder}/tglf_neo"
        )

        # Add errors and merge fluxes as we would do if this was a CGYRO run
        curateTGYROfiles(
            tgyro.results["tglf_neo_original"],
            f"{self.folder}/tglf_neo/",
            percentError,
            impurityPosition=impurityPosition,
            includeFast=includeFast,
        )

        # Read again to capture errors
        tgyro.read(label="tglf_neo", folder=f"{self.folder}/tglf_neo/")

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Run TGLF standalone --> In preparation for the transition
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # from mitim_tools.gacode_tools import TGLFtools
        # tglf = TGLFtools.TGLF(rhos=RadiisToRun)
        # _ = tglf.prep(
        #     self.folder+'/stds/',
        #     inputgacode=self.file_profs,
        #     recalculatePTOT=False, # Use what's in the input.gacode, which is what PORTALS TGYRO does
        #     restart=restart)

        # tglf.run(
        #     subFolderTGLF="tglf_neo_original/",
        #     TGLFsettings=MODELparameters["transport_model"]["TGLFsettings"],
        #     restart=restart,
        #     forceIfRestart=True,
        #     extraOptions=MODELparameters["transport_model"]["extraOptionsTGLF"],
        #     launchSlurm=launchMODELviaSlurm,
        #     slurm_setup={"cores": 4, "minutes": 1},
        # )

        # tglf.read(label="tglf_neo_original")

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # ------------------------------------------------------------------------------------------------------------------------
        # 3. tglf_neo: Populate powerstate with the TGYRO results
        # ------------------------------------------------------------------------------------------------------------------------

        # Produce right quantities (TGYRO -> powerstate.plasma)
        self.powerstate = tgyro.results["tglf_neo"].TGYROmodeledVariables(
            self.powerstate,
            useConvectiveFluxes=useConvectiveFluxes,
            includeFast=includeFast,
            impurityPosition=impurityPosition,
            UseFineGridTargets=UseFineGridTargets,
            OriginalFimp=OriginalFimp,
            forceZeroParticleFlux=forceZeroParticleFlux,
            provideTurbulentExchange=provideTurbulentExchange,
            provideTargets=self.powerstate.TargetCalc == "tgyro",
        )

        # ------------------------------------------------------------------------------------------------------------------------
        # 4. cgyro_neo: Trick to fake a tgyro output to reflect CGYRO
        # ------------------------------------------------------------------------------------------------------------------------

        if MODELparameters['transport_model']['turbulence'] == 'CGYRO':

            print(
                "\t- Checking whether cgyro_neo folder exists and it was written correctly via cgyro_trick..."
            )

            correctly_run = os.path.exists(f"{self.folder}/cgyro_neo")
            if correctly_run:
                print("\t\t- Folder exists, but was cgyro_trick run?")
                with open(f"{self.folder}/cgyro_neo/mitim_flag", "r") as f:
                    correctly_run = bool(float(f.readline()))

            if correctly_run:
                print("\t\t\t* Yes, it was", typeMsg="w")
            else:
                print("\t\t\t* No, it was not, repating process", typeMsg="i")

                # Copy tglf_neo results
                os.system(
                    f"cp -r {self.folder}/tglf_neo {self.folder}/cgyro_neo"
                )

                # CGYRO writter
                cgyro_trick(
                    self,
                    f"{self.folder}/cgyro_neo",
                    profiles_postprocessing_fun=profiles_postprocessing_fun,
                    name=self.name,
                )

            # Read TGYRO files and construct portals variables

            tgyro.read(label="cgyro_neo", folder=f"{self.folder}/cgyro_neo") 

            powerstate_orig = copy.deepcopy(self.powerstate)

            self.powerstate = tgyro.results["cgyro_neo"].TGYROmodeledVariables(
                self.powerstate,
                useConvectiveFluxes=useConvectiveFluxes,
                includeFast=includeFast,
                impurityPosition=impurityPosition,
                UseFineGridTargets=UseFineGridTargets,
                OriginalFimp=OriginalFimp,
                forceZeroParticleFlux=forceZeroParticleFlux,
                provideTurbulentExchange=provideTurbulentExchange,
                provideTargets=self.powerstate.TargetCalc == "tgyro",
            )

            print("\t- Checking model modifications:")
            for r in ["Pe_tr_turb", "Pi_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "Mt_tr_turb"]: #, "PexchTurb"]: #TO FIX
                print(
                    f"\t\t{r}(tglf)  = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(powerstate_orig.plasma[r][0][1:],powerstate_orig.plasma[r+'_stds'][0][1:]) ])}"
                )
                print(
                    f"\t\t{r}(cgyro) = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(self.powerstate.plasma[r][0][1:],self.powerstate.plasma[r+'_stds'][0][1:]) ])}"
                )

            # **
            tgyro.results["use"] = tgyro.results["cgyro_neo"]

        else:
            # copy profiles too!
            profilesToShare(self)

            # **
            tgyro.results["use"] = tgyro.results["tglf_neo"]

        # ------------------------------------------------------------------------------------------------------------------------
        # Results class that can be used for further plotting and analysis in PORTALS
        # ------------------------------------------------------------------------------------------------------------------------

        self.model_results = copy.deepcopy(tgyro.results["use"]) # Pass the TGYRO results class that should be use for plotting and analysis

        self.model_results.extra_analysis = {}
        for ikey in tgyro.results:
            if ikey != "use":
                self.model_results.extra_analysis[ikey] = tgyro.results[ikey]

# ------------------------------------------------------------------
# SIMPLE Diffusion
# ------------------------------------------------------------------

class diffusion_model(power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

        # Ensure that the provided diffusivities include the zero location
        self.chi_e = self.powerstate.TransportOptions["ModelOptions"]["chi_e"]
        self.chi_i = self.powerstate.TransportOptions["ModelOptions"]["chi_i"]

        if self.chi_e.shape[0] < self.powerstate.plasma['rho'].shape[-1]:
            self.chi_e = torch.cat((torch.zeros(1), self.chi_e))

        if self.chi_i.shape[0] < self.powerstate.plasma['rho'].shape[-1]:
            self.chi_i = torch.cat((torch.zeros(1), self.chi_i))

    def produce_profiles(self):
        pass

    def evaluate(self):

        # Make sure the chis are applied to all the points in the batch
        Pe_tr = PLASMAtools.conduction(
            self.powerstate.plasma["ne"],
            self.powerstate.plasma["te"],
            self.chi_e.repeat(self.powerstate.plasma['rho'].shape[0],1),
            self.powerstate.plasma["aLte"],
            self.powerstate.plasma["a"].unsqueeze(-1),
        )
        Pi_tr = PLASMAtools.conduction(
            self.powerstate.plasma["ni"].sum(axis=-1),
            self.powerstate.plasma["ti"],
            self.chi_i.repeat(self.powerstate.plasma['rho'].shape[0],1),
            self.powerstate.plasma["aLti"],
            self.powerstate.plasma["a"].unsqueeze(-1),
        )

        self.powerstate.plasma["Pe_tr_turb"] = Pe_tr * 2 / 3
        self.powerstate.plasma["Pi_tr_turb"] = Pi_tr * 2 / 3

        self.powerstate.plasma["Pe_tr_neo"] = Pe_tr * 1 / 3
        self.powerstate.plasma["Pi_tr_neo"] = Pi_tr * 1 / 3

        self.powerstate.plasma["Pe_tr"] = self.powerstate.plasma["Pe_tr_turb"] + self.powerstate.plasma["Pe_tr_neo"]
        self.powerstate.plasma["Pi_tr"] = self.powerstate.plasma["Pi_tr_turb"] + self.powerstate.plasma["Pi_tr_neo"]

# ------------------------------------------------------------------
# SURROGATE
# ------------------------------------------------------------------

class surrogate_model(power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        pass

    def evaluate(self):

        """
        flux_fun as given in ModelOptions must produce Q and Qtargets in order of te,ti,ne
        """

        X = torch.cat((self.powerstate.plasma['aLte'][:,1:],self.powerstate.plasma['aLti'][:,1:],self.powerstate.plasma['aLne'][:,1:]),axis=1)

        _, Q, _, _ = self.powerstate.TransportOptions["ModelOptions"]["flux_fun"](X) #self.Xcurrent[0])

        numeach = self.powerstate.plasma["rho"].shape[1] - 1

        quantities = {
            "te": "Pe",
            "ti": "Pi",
            "ne": "Ce",
            "nZ": "CZ",
            "w0": "Mt",
        }

        for c, i in enumerate(self.powerstate.ProfilesPredicted):
            self.powerstate.plasma[f"{quantities[i]}_tr"] = torch.cat((torch.tensor([[0.0]]),Q[:, numeach * c : numeach * (c + 1)]),dim=1)

# **************************************************************************************************
# Functions
# **************************************************************************************************

def curateTGYROfiles(tgyro, folder, percentError, impurityPosition=1, includeFast=False):
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


def profilesToShare(self):
    if "folder" in self.powerstate.TransportOptions["ModelOptions"]["extra_params"]:
        whereFolder = IOtools.expandPath(
            self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"] + "/Outputs/portals_profiles/"
        )
        if not os.path.exists(whereFolder):
            IOtools.askNewFolder(whereFolder)

        fil = f"{whereFolder}/input.gacode.{self.evaluation_number}"
        os.system(f"cp {self.file_profs_mod} {fil}")
        os.system(f"cp {self.file_profs} {fil}_unmodified")
        os.system(f"cp {self.file_profs_targets} {fil}_unmodified.new")
        print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
    else:
        print("\t- Could not move files", typeMsg="w")


def cgyro_trick(
    self,
    FolderEvaluation_TGYRO,
    profiles_postprocessing_fun=None,
    name="",
):

    with open(f"{FolderEvaluation_TGYRO}/mitim_flag", "w") as f:
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
        rad=self.powerstate.plasma["rho"].shape[1] - 1,
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
    with open(f"{FolderEvaluation_TGYRO}/mitim_flag", "w") as f:
        f.write("1")

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
