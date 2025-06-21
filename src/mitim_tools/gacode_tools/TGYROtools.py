import shutil
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mitim_tools.misc_tools import (
    IOtools,
    GRAPHICStools,
    PLASMAtools,
)
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.gacode_tools.utils import GACODEinterpret, GACODEdefaults, GACODErun
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
Same philosophy as the TGLFtools
	tgyro = TGYRO(cdf='12345A02.CDF',time=4.2)
	tgyro.prep('./tgyro/')
	tgyro.run(subFolderTGYRO='tgyro1/')

*NOTE*
The goal of the prep stage is to have an input.gacode class ready to go, but a custom one
that was modified as you like can be passed instead, not using the cdf file.

*NOTE ON NORMALIZATION*
Reading results from TGYRO will create a tgyro output class that also reads the out.tgyro.gyrobohm
for the normalization:
	QGB, GammaGB
	c_s
"""


class TGYRO:
    def __init__(self, cdf=None, time=100.0, avTime=0.0):
        """
        cdf is not required if later I provide the input.gacode directly. However, it is best to give a "dummy location"
        so that the name can be grabbed and be used as the nameJob
        """

        self.LocationCDF = cdf
        if cdf is not None:
            _, self.nameRunid = IOtools.getLocInfo(self.LocationCDF)
        else:
            self.nameRunid = "mitim"

        self.time, self.avTime = time, avTime

        # Main, unmutable files
        self.outputFiles = [
            "out.tgyro.run",
            "out.tgyro.control",
            "out.tgyro.residual",
            "out.tgyro.gyrobohm",
            "out.tgyro.profile",
            "out.tgyro.profile_e",
            "out.tgyro.evo_er",
            "out.tgyro.evo_ne",
            "out.tgyro.evo_te",
            "out.tgyro.evo_ti",
            "out.tgyro.iterate",
            "out.tgyro.geometry.1",
            "out.tgyro.geometry.2",
            "out.tgyro.flux_e",
            "out.tgyro.alpha",
            "out.tgyro.nu_rho",
            "out.tgyro.power_e",
            "out.tgyro.power_i",
            "out.tgyro.prec",
            "input.tgyro.gen",
            "input.gacode.new",
            "input.gacode",
        ]

        self.results = {}
        self.tglf = {}

    def prep(
        self,
        FolderGACODE,
        profilesclass_custom=None,
        cold_start=False,
        forceIfcold_start=False,
        subfolder="prep_tgyro",
        BtIp_dirs=[0, 0],
        gridsTRXPL=[151, 101, 101],
        remove_tmp=False,
        includeGEQ=True,
        correctPROFILES=False,
    ):
        """
        Run workflow to prepare TGYRO, but pass the cold_start flag. Checks will happen inside.
        Include all species, including fast,

        Goal of this step is to get a input.gacode class stored in self.profiles

        - includeGEQ = True runs profiles_gen with the -g option, thus using a GACODE representation of the flux surfaces
                rather than the geometric parameters in the .cdf plasmastate file.

        - profilesclass_custom is used when one wants to modify manually the profiles class (e.g. mitim)

        if correctPROFILES:
                - Renames LUMPED impurity into a proper one
                - etc

        """

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)
        self.FolderGACODE_tmp = self.FolderGACODE / subfolder

        # Define name
        _, self.nameRuns_default = IOtools.reducePathLevel(self.FolderGACODE, level=1)
        # -----------

        if (not self.FolderGACODE.exists()) or cold_start:
            print(
                f"\t- Folder {FolderGACODE} does not exist, or cold_start has been requested... creating folder to store"
            )
            IOtools.askNewFolder(
                self.FolderGACODE, force=forceIfcold_start or (not cold_start)
            )

        if (not self.FolderGACODE_tmp.exists()) or cold_start:
            IOtools.askNewFolder(
                self.FolderGACODE_tmp, force=forceIfcold_start or (not cold_start)
            )

        if profilesclass_custom is None:
            print("> Preparation of TGYRO run (i.e. input.gacode generator)")
            produceInputs_TGYROworkflow(
                self.time,
                self.FolderGACODE,
                self.FolderGACODE_tmp,
                self.LocationCDF,
                avTime=self.avTime,
                forceEntireWorkflow=cold_start,
                BtIp_dirs=BtIp_dirs,
                gridsTRXPL=gridsTRXPL,
                includeGEQ=includeGEQ,
            )

            self.file_input_profiles = self.FolderGACODE / "input.gacode"
            from mitim_tools.gacode_tools import PROFILEStools
            self.profiles = PROFILEStools.gacode_state(self.file_input_profiles)

            if correctPROFILES:
                self.profiles.correct(write=True)

        else:
            print("\t- Custom input.gacode class used")

            self.profiles = profilesclass_custom

        if remove_tmp:
            print(
                "\t~~ Remove intermediate TGYRO files to avoid consuming too much memory",
                typeMsg="i",
            )
            IOtools.shutil_rmtree(self.FolderGACODE_tmp)

    def run(
        self,
        subFolderTGYRO="tgyro1",
        cold_start=False,
        vectorRange=[0.2, 0.8, 10],
        special_radii=None,
        iterations=0,
        PredictionSet=[1, 1, 0],
        TGLFsettings=0,
        extraOptionsTGLF={},
        forceIfcold_start=False,
        minutesJob=30,
        launchSlurm=True,
        TGYRO_physics_options={},
        TGYRO_solver_options={},
        TGYROcontinue=None,
        rhosCopy=None,
        forcedName=None,
        modify_inputgacodenew=True,
    ):
        """
        vectorRange defines the range to run TGYRO.
                [0] -> from
                [1] -> to (B.C.)
                [2] -> number of points in between

        Instead of vectorRange, I can provide special_radii
        """

        if rhosCopy is None:
            rhosCopy = []

        # ------------------------------------------------------------------------
        # Defaults
        # ------------------------------------------------------------------------
        """
		onlyThermal=True does not include the fast particles (end of file)
		Note: running with onlyThermal will result in a input.gacode.new that has 0.0 in the columns of the fast particles
		"""
        onlyThermal = TGYRO_physics_options.get("onlyThermal", False)
        limitSpecies = TGYRO_physics_options.get("limitSpecies", 100)  # limitSpecies is not to consider the last ions. In there are 6 ions in input.gacode, and limitSpecies=4, the last 2 won't be considered
        quasineutrality = TGYRO_physics_options.get("quasineutrality", [])

        if (
            (iterations == 0)
            and ("tgyro_method" in TGYRO_solver_options)
            and (TGYRO_solver_options["tgyro_method"] != 1)
        ):
            print(f"\t- Zero iteration must be run with method 1, changing it from {TGYRO_solver_options['tgyro_method']} to 1",typeMsg="w",)
            TGYRO_solver_options["tgyro_method"] = 1

        # ------------------------------------------------------------------------

        print("\n> Run TGYRO")

        special_radii_mod = copy.deepcopy(special_radii)

        # -------
        if special_radii_mod is not None:
            # make sure it's an array
            special_radii_mod = np.array(special_radii_mod)
            vectorRange = [
                special_radii_mod[0],
                special_radii_mod[-1],
                len(special_radii_mod),
            ]

        if vectorRange[2] == 1:
            extra_add = vectorRange[0] + 1e-3
            print(f" -> TGYRO cannot run with only one point ({vectorRange[0]}), adding {extra_add}",typeMsg="w",)
            vectorRange[2], vectorRange[1] = 2, extra_add
            if special_radii_mod is not None:
                special_radii_mod = np.append(special_radii_mod, [extra_add])

        self.rhosToSimulate = (
            special_radii_mod
            if (special_radii_mod is not None)
            else np.linspace(vectorRange[0], vectorRange[1], vectorRange[2])
        )
        # -------

        Tepred, Tipred, nepred = PredictionSet

        self.FolderTGYRO = IOtools.expandPath(self.FolderGACODE / subFolderTGYRO)
        self.FolderTGYRO_tmp = (
            self.FolderTGYRO / "tmp_tgyro_run"
        )  # Folder to run TGYRO on (or to retrieve the raw outputs from a cluster)

        inputclass_TGYRO = TGYROinput(
            input_profiles=self.profiles,
            onlyThermal=onlyThermal,
            limitSpecies=limitSpecies,
        )
        self.loc_n_ion = int(inputclass_TGYRO.loc_n_ion)

        """
		Name of the job (will also define the folder in the cluster)

		Notes:
			- To avoid colliding, I use the nameRunid from the .CDF or dummy, and the subfolder for scans.
				In mitim, I need to provide a dummy CDF with the simulation+evaluation name, so that the name is: 	tgyro_run10000_2_tgyro1
				In TGYRO, I don't need to provide a dummy CDF as long as I use a different folder name. e.g.:		tgyro_12345A01_foldername
		"""

        if forcedName is None:
            nameJob = f"tgyro_{self.nameRunid}_{subFolderTGYRO.strip('/')}"
        else:
            nameJob = forcedName

        # -----------------------------------
        # ------ Print important info
        # -----------------------------------

        if onlyThermal:
            if len(quasineutrality) < 1:
                print("\t- TGYRO will be run by removing fast particles (only thermal ones), without correcting for quasineutrality",typeMsg="i",)
            else:
                print(f"\t- TGYRO will be run by removing fast particles (only thermal ones), with quasineutrality={quasineutrality}",typeMsg="i",)
        else:
            if len(quasineutrality) < 1:
                print("\t- TGYRO will be run with all species (fast and thermal), without correcting for quasineutrality",typeMsg="i",)
            else:
                print(f"\t- TGYRO will be run with all species (fast and thermal), with quasineutrality={quasineutrality}",typeMsg="i",)

        # -----------------------------------
        # ------ Complete required outputs
        # -----------------------------------

        for i in range(self.loc_n_ion):
            self.outputFiles.append(f"out.tgyro.flux_i{i+1}")
            self.outputFiles.append(f"out.tgyro.profile_i{i+1}")
            self.outputFiles.append(f"out.tgyro.evo_n{i+1}")

        # Retrieve the input files to TGLF
        for i in range(len(self.rhosToSimulate)):
            self.outputFiles.append(f"input.tglf.new{i+1}")

        # ----------------------------------------------------------------
        # Do I need to run TGYRO?
        # ----------------------------------------------------------------

        inputFiles_TGYRO = []

        # Do the required files exist?
        exists = not cold_start

        txt_nonexist = ""
        if exists:
            for j in self.outputFiles:
                file = self.FolderTGYRO / f"{j}"
                existThis = file.exists()
                if not existThis:
                    txt_nonexist += f"\t\t- {IOtools.clipstr(file)}\n"
                exists = exists and existThis

            if not exists:
                print("\t- Some of the required output files did not exist, running TGYRO")

        # ----------------------------------------------------------------
        # ----------------------------------------------------------------

        if not exists:
            IOtools.askNewFolder(self.FolderTGYRO, force=forceIfcold_start)

            # ----------------------------------------------------------------
            # cold_started
            # ----------------------------------------------------------------
            if TGYROcontinue is not None:
                print("\n\n~~ Option to cold_start from previous TGYRO iteration selected\n\n")

                self.FolderTGYRO_cold_started = self.FolderGACODE / TGYROcontinue

                for i in self.outputFiles:
                    inputFiles_TGYRO.append(self.FolderTGYRO_cold_started / f"{i}")

        """ -----------------------------------
		 	Create TGLF file (only control info, no species)
				If TGLFsettings = None, use what TGYRO generated.
				If TGLFsettings = 0, minimal
		 	-----------------------------------
		"""

        if not self.FolderTGYRO_tmp.exists():
            IOtools.askNewFolder(self.FolderTGYRO_tmp)

        print(f"\t\t- Creating only-controls input.tglf file in {IOtools.clipstr(str(self.FolderTGYRO_tmp.resolve()))}input.tglf")
        inputclass_TGLF = TGLFtools.TGLFinput()
        inputclass_TGLF = GACODErun.modifyInputs(
            inputclass_TGLF,
            Settings=TGLFsettings,
            extraOptions=extraOptionsTGLF,
            addControlFunction=GACODEdefaults.addTGLFcontrol,
            NS=self.loc_n_ion + 1,
        )
        inputclass_TGLF.writeCurrentStatus(file=self.FolderTGYRO_tmp / "input.tglf")

        # -----------------------------------
        # ------ Write input profiles
        # -----------------------------------

        print(f"\t\t- Using input.profiles from {IOtools.clipstr(self.profiles.file)}")
        fil = "input.gacode"

        if self.profiles.profiles['rho(-)'][0] > 0.0:
            print("\t\t- input.gacode had a finite first rho, which is not allowed. Setting it to 0.0", typeMsg="i")
            self.profiles.profiles['rho(-)'][0] = 0.0

        # Make sure it has a Zeff column
        if "z_eff(-)" not in self.profiles.profiles:
            self.profiles.profiles["z_eff(-)"] = self.profiles.derived["Zeff"]

        self.profiles.writeCurrentStatus(file=self.FolderTGYRO_tmp / f"{fil}")

        # -----------------------------------
        # ------ Create TGYRO file
        # -----------------------------------

        inputclass_TGYRO = modifyInputToTGYRO(
            inputclass_TGYRO,
            vectorRange,
            iterations,
            TGYRO_physics_options,
            TGYRO_solver_options,
            Tepred=Tepred,
            Tipred=Tipred,
            nepred=nepred,
            TGYROcontinue=TGYROcontinue is not None,
            special_radii=special_radii_mod,
        )

        inputclass_TGYRO.writeCurrentStatus(file=self.FolderTGYRO_tmp / "input.tgyro")

        # -----------------------------------
        # ------ Check density for problems
        # -----------------------------------

        threshold = 1e-10
        fi = self.profiles.derived["fi"]
        ix1 = np.argmin(np.abs(self.profiles.profiles["rho(-)"] - vectorRange[0]))
        ix2 = np.argmin(np.abs(self.profiles.profiles["rho(-)"] - vectorRange[1]))
        fi[ix1 : ix2 + 1].min(axis=0)
        minn = np.where(fi[ix1 : ix2 + 1].min(axis=0) < threshold)[0]

        minn_true = []
        for i in minn:
            if (self.profiles.Species[i]["S"] != "fast") or (not TGYRO_physics_options.get("onlyThermal", False)):
                minn_true.append(i)

        if len(minn_true) > 0:
            print(f"* Ions in positions {minn_true} have a relative density lower than {threshold}, which can cause problems. Continue (c)?",typeMsg="q",)

        # -----------------------------------
        # ------ Run
        # -----------------------------------

        nparallel = len(self.rhosToSimulate)

        if not exists:
            # Run TGYRO and store data in self.FolderTGYRO_tmp (this folder will also have all the inputs, that's why I'm keeping it)
            GACODErun.runTGYRO(
                self.FolderTGYRO_tmp,
                nameRunid=nameJob,
                outputFiles=self.outputFiles,
                inputFiles=inputFiles_TGYRO,
                launchSlurm=launchSlurm,
                nameJob=nameJob,
                nparallel=nparallel,
                minutes=minutesJob,
            )

            """
			------------------------------------------------------------------------------------------------------------------------
				Make modifications to output
				****************************
				This is because regardless of the TypeTarget used, TGYRO will replace
				all quantities always. If I'm not doing radiation calculation, I don't
				want it be changed!
			------------------------------------------------------------------------------------------------------------------------
			"""
            if modify_inputgacodenew:
                print("\t- It was requested that input.gacode.new is modified according to what TypeTarget was",typeMsg="i",)

                inputgacode_new = PROFILEStools.PROFILES_GACODE(self.FolderTGYRO_tmp / "input.gacode.new")

                if TGYRO_physics_options["TypeTarget"] < 3:
                    for ikey in [
                        "qbrem(MW/m^3)",
                        "qsync(MW/m^3)",
                        "qline(MW/m^3)",
                        "qfuse(MW/m^3)",
                        "qfusi(MW/m^3)",
                    ]:
                        print(f"\t- Replacing {ikey} from input.gacode.new to have the same as input.gacode")
                        if ikey in self.profiles.profiles:
                            inputgacode_new.profiles[ikey] = self.profiles.profiles[ikey]
                        else:
                            inputgacode_new.profiles[ikey] = inputgacode_new.profiles["rho(-)"] * 0.0

                if TGYRO_physics_options["TypeTarget"] < 2:
                    for ikey in ["qei(MW/m^3)"]:
                        print(f"\t- Replacing {ikey} from input.gacode.new to have the same as input.gacode")
                        if ikey in self.profiles.profiles:
                            inputgacode_new.profiles[ikey] = self.profiles.profiles[ikey]
                        else:
                            inputgacode_new.profiles[ikey] = inputgacode_new.profiles["rho(-)"] * 0.0

                inputgacode_new.writeCurrentStatus()
            # ------------------------------------------------------------------------------------------------------------------------

            # Copy those files that I'm interested in, plus the extra file, into the main folder
            for file in self.outputFiles + ["input.tgyro"]:
                shutil.copy2(self.FolderTGYRO_tmp / f"{file}", self.FolderTGYRO / f"{file}")

            # Rename the input.tglf.news to the actual rho they where at
            for cont, i in enumerate(self.rhosToSimulate):
                shutil.copy2(self.FolderTGYRO / f"input.tglf.new{cont + 1}", self.FolderTGYRO / f"input.tglf_{i:.4f}")

            # If I have run TGYRO with the goal of generating inputs, move them to the GACODE folder
            for rho in rhosCopy:
                shutil.copy2(self.FolderTGYRO / f"input.tglf_{rho:.4f}", self.FolderGACODE)

            # Remove temporary folder
            try: 
                IOtools.shutil_rmtree(self.FolderTGYRO_tmp)
            except OSError:
                print(f"\t- Could not remove {self.FolderTGYRO_tmp}. Trying again.", typeMsg="w")
                IOtools.shutil_rmtree(self.FolderTGYRO_tmp)

        else:
            print(" ~~~ Not running TGYRO", typeMsg="i")

    def read(self, label="tgyro1", folder=None, file_input_profiles=None):
        # If no specified folder, check the last one
        if not isinstance(folder, (str, Path)):
            folder = self.FolderTGYRO
        else:
            folder = IOtools.expandPath(folder)

        if not isinstance(file_input_profiles, (str, Path)):
            if "profiles" not in self.__dict__:
                prof = None
            else:
                prof = self.profiles
        else:
            from mitim_tools.gacode_tools import PROFILEStools
            prof = PROFILEStools.gacode_state(file_input_profiles)

        self.results[label] = TGYROoutput(folder, profiles=prof)

    def converge(
        self,
        factor_reduction=1e-2,
        cold_start=False,
        vectorRange=[0.2, 0.8, 10],
        special_radii=None,
        PredictionSet=[1, 1, 0],
        TGLFsettings=0,
        extraOptionsTGLF={},
        forceIfcold_start=False,
        launchSlurm=True,
        TGYRO_physics_options={},
        TGYRO_solver_options={},
    ):
        """
        As in run but without solver or iterations
        """

        """
		Step 1
		-------------
		Start with a high number of method-6 iterations with default steps
		"""

        iterations = 50
        minutesJob = 60
        solver = {"tgyro_method": 6, "step_max": 0.2, "relax_param": 0.2}
        self.run(
            TGYROcontinue=None,
            subFolderTGYRO=f"{self.nameRuns_default}_1",
            iterations=iterations,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            special_radii=special_radii,
            vectorRange=vectorRange,
            PredictionSet=PredictionSet,
            minutesJob=minutesJob,
            launchSlurm=launchSlurm,
            TGLFsettings=TGLFsettings,
            extraOptionsTGLF=extraOptionsTGLF,
            TGYRO_solver_options=solver,
            TGYRO_physics_options=TGYRO_physics_options,
        )
        self.read(label=f"{self.nameRuns_default}_1")

        # iterations       = 20
        # minutesJob 		 = 60
        # solver           =  { 'tgyro_method': 1, 'step_max':1.0,  'relax_param': 2.0, 'step_jac':  0.1}
        # self.run(TGYROcontinue = None,
        # 	subFolderTGYRO=f'{self.nameRuns_default}_1',iterations=iterations,cold_start=cold_start,forceIfcold_start=forceIfcold_start,
        # 	special_radii=special_radii,vectorRange=vectorRange,PredictionSet=PredictionSet,
        # 	minutesJob=minutesJob,launchSlurm = launchSlurm,
        # 	TGLFsettings = TGLFsettings, extraOptionsTGLF = extraOptionsTGLF,
        # 	TGYRO_solver_options=solver,TGYRO_physics_options = TGYRO_physics_options)
        # self.read(label = f'{self.nameRuns_default}_1')

        """
		Step 2
		---------------
		If not converged
		"""

        # iterations       = 50
        # minutesJob 		 = 60
        # solver           = { 'tgyro_method': 1, 'step_max':1.0,  'relax_param': 2.0, 'step_jac':  0.1}
        # tgyro.run(TGYROcontinue = 'run1',
        # 	subFolderTGYRO='run2',iterations=iterations,cold_start=cold_start,forceIfcold_start=forceIfcold_start,
        # 	special_radii=rhos,PredictionSet=PredictionSet,
        # 	minutesJob=minutesJob,launchSlurm = launchSlurm,
        # 	TGLFsettings = TGLFsettings, extraOptionsTGLF = extraOptionsTGLF,
        # 	TGYRO_solver_options=solver,TGYRO_physics_options = physics_options)
        # tgyro.read(label = 'run2')

        """
		Final step
		---------------
		Provide final as conv
		"""
        self.read(label="conv")

    def grab_tglf_objects(self, subfolder="tglf_runs", fromlabel="tgyro1", rhos=None):

        if rhos is None:
            rhos = self.rhosToSimulate

        # Create class of inputs to TGLF
        inputsTGLF = {}
        for cont, rho in enumerate(rhos):
            fileN = self.FolderTGYRO / f"input.tglf_{rho:.4f}"
            inputclass = TGLFtools.TGLFinput(file=fileN)
            inputsTGLF[rho] = inputclass

        tglf = TGLFtools.TGLF(rhos=rhos)
        tglf.prep(
            self.FolderGACODE / subfolder,
            specificInputs=inputsTGLF,
            inputgacode=self.FolderTGYRO / "input.gacode",
            tgyro_results=self.results[fromlabel],
        )

        return tglf


    def runTGLF(self, fromlabel="tgyro1", rhos=None, cold_start=False):
        """
        This runs TGLF at the final point: Using the out.local.dumps for running TGLF, and using input.gacode.new for normalization

        Doesn't work with specific inputs now
        """

        self.tglf[fromlabel] = self.grab_tglf_objects(fromlabel=fromlabel, rhos=rhos)

        """
		Run TGLF the same way as TGYRO did:		No TGLFsettings and no corrections for species, but what is in the file
		"""

        label = f"{self.nameRuns_default}_tglf1"

        self.tglf[fromlabel].run(
            subFolderTGLF=f"{label}",
            TGLFsettings=None,
            ApplyCorrections=False,
            cold_start=cold_start,
        )
        self.tglf[fromlabel].read(label=label)

    def runTGLFsensitivities(self, fromlabel="tgyro1", rho=0.5, cold_start=False):
        """
        This runs TGLF scans at the final point

        NEEDS TO BE WORKED OUT.
        Does't work with specific Inputs now
        """

        # Create class of inputs to TGLF
        inputsTGLF = {
            rho: TGLFtools.TGLFinput(file=(self.FolderTGYRO / f"input.tglf_{rho:.4f}"))
        }

        self.tglf[fromlabel] = TGLFtools.TGLF(rhos=[rho])
        self.tglf[fromlabel].prep(
            "tglf_runs",
            specificInputs=inputsTGLF,
            inputgacode=self.FolderTGYRO / "input.gacode.new",
            tgyro_results=self.results[fromlabel],
        )

        self.tglf[fromlabel].runScanTurbulenceDrives(
            subFolderTGLF=f"{self.nameRuns_default}_tglf",
            TGLFsettings=None,
            ApplyCorrections=False,
            cold_start=cold_start,
            specificInputs=inputsTGLF,
        )

        # self.tglf[fromlabel].plotScanTurbulenceDrives(self,label='scan1',figs=None)

    def run_tglf_scan(
        self,
        rhos=[0.4, 0.6],
        onlyThermal=False,
        quasineutrality=None,
        subFolderTGYRO="tgyro_dummy",
        cold_start=False,
        label="tgyro1",
        donotrun=False,
        recalculatePTOT=True,
    ):
        """
        onlyThermal will remove from the TGYRO run the fast species, so the resulting input.tglf files will not have
                        that species.
        quasineutrality will adjust the ions densities so that it fulfills quasineutrality. If quasineutrality=[], then
                        don't do anything, so if onlyThermal=True, then the plasma won't be quasineutral in the input.tglf
        """

        if quasineutrality is None:
            quasineutrality = []

        print(
            "\t- Entering in TGLF scans module (running a dummy zero-iteration TGYRO)..."
        )
        # print(
        #     "\t\t* Note: It is recommended not to use input.gacode.new since that will be messed up in cases with pedestal",
        #     typeMsg="w",
        # )

        minutesJob = 5

        # ---- Set up to only generate files without no much calcs
        TGLFsettings = 0
        TGYRO_physics_options = {
            "TypeTarget": 1,  # Do not do anything with targets
            "TurbulentExchange": 0,  # Do not calculate turbulent exchange
            "InputType": 1,  # Use exact profiles
            "GradientsType": 0,  # Do not recompute the gradients
            "onlyThermal": onlyThermal,
            "quasineutrality": quasineutrality,
            "neoclassical": 0,  # Do not run or check NEOTGYRO canno
            "PtotType": int(not recalculatePTOT), # Recalculate Ptot or use what's there
        }
        # ------------------------------------------------------------

        if not donotrun:
            self.run(
                subFolderTGYRO=subFolderTGYRO,
                TGLFsettings=TGLFsettings,
                TGYRO_physics_options=TGYRO_physics_options,
                iterations=0,
                cold_start=cold_start,
                forceIfcold_start=True,
                minutesJob=minutesJob,
                rhosCopy=rhos,
                special_radii=rhos,
            )
        else:  # special_radii=rhos
            print("\t\t- No need to run dummy iteration of TGYRO", typeMsg="i")

        try:
            self.read(
                label=label,
                folder=IOtools.expandPath(self.FolderGACODE / subFolderTGYRO),
            )
            res = self.results[label]
        except:
            print("\t* TGYRO output could not be read", typeMsg="w")
            res = None

        return res

    def plot(self, fn=None, labels=["tgyro1"], doNotShow=False, fn_color=None):
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fn = FigureNotebook("TGYRO Output Notebook", geometry="1800x900")
        else:
            self.fn = fn

        # **** Plot the TGYRO output for all the labels

        maxChar = 8
        for label in labels:
            if len(label) > maxChar:
                lab = label[-maxChar:]
            else:
                lab = label
            self.results[label].plot(fn=self.fn, label=" .." + lab)

        # **** Final real

        fig1 = self.fn.add_figure(tab_color=fn_color, label="Final Comp.")
        grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.3)
        ax00 = fig1.add_subplot(grid[0, 0])
        ax01 = fig1.add_subplot(grid[0, 1])
        ax02 = fig1.add_subplot(grid[0, 2])
        ax10 = fig1.add_subplot(grid[1, 0])
        ax11 = fig1.add_subplot(grid[1, 1])
        ax12 = fig1.add_subplot(grid[1, 2])
        ax20 = fig1.add_subplot(grid[2, 0])
        ax21 = fig1.add_subplot(grid[2, 1])
        ax22 = fig1.add_subplot(grid[2, 2])
        ax03 = fig1.add_subplot(grid[0, 3])
        ax13 = fig1.add_subplot(grid[1, 3])
        ax23 = fig1.add_subplot(grid[2, 3])

        colors = GRAPHICStools.listColors()

        ax = ax00
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1],
                res.Te[-1],
                "o-",
                c=colors[cont],
                label=label,
                markersize=3,
                lw=0.5,
            )
        roa = res.profiles.profiles["rmin(m)"] / res.profiles.profiles["rmin(m)"][-1]
        Te = res.profiles.profiles["te(keV)"]
        ax.plot(roa, Te, "-", label="profiles", lw=0.2, c="g")  # colors[cont+1])
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("T (keV)")
        # ax.set_ylim(bottom=0)
        ax.set_title("Electron Temperature")
        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax, axE = ax10, ax20
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1][1:],
                res.Qe_tar[-1][1:],
                "--o",
                c=colors[cont],
                label=label,
                markersize=3,
            )
            ax.plot(
                res.roa[-1][1:], res.Qe_sim[-1][1:], "-o", c=colors[cont], markersize=3
            )
            axE.plot(
                res.roa[-1][1:],
                res.Qe_res[-1][1:],
                "-o",
                c=colors[cont],
                label=label,
                markersize=3,
            )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q_e$ ($MW/m^2$)")
        ax.set_title("Electron energy flux")
        axE.set_xlabel("$r/a$")
        axE.set_xlim([0, 1])
        axE.set_ylabel("Residual")
        # axE.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax01
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1],
                res.Ti[-1, 0],
                "o-",
                label=label,
                c=colors[cont],
                markersize=3,
                lw=0.5,
            )
        roa = res.profiles.profiles["rmin(m)"] / res.profiles.profiles["rmin(m)"][-1]
        Te = res.profiles.profiles["ti(keV)"][:, 0]
        ax.plot(roa, Te, "-", lw=0.2, c="g")  # colors[cont+1])
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$T_e$ (keV)")
        # ax.set_ylim(bottom=0)
        ax.set_title("Ion Temperature")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax, axE = ax11, ax21
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1][1:],
                res.Qi_tar[-1][1:],
                "--o",
                c=colors[cont],
                label=label,
                markersize=3,
            )
            ax.plot(
                res.roa[-1][1:], res.Qi_sim[-1][1:], "-o", c=colors[cont], markersize=3
            )
            axE.plot(
                res.roa[-1][1:],
                res.Qi_res[-1][1:],
                "-o",
                c=colors[cont],
                label=label,
                markersize=3,
            )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q_i$ ($MW/m^2$)")
        ax.set_ylim(bottom=0)
        ax.set_title("Ion energy flux")
        axE.set_xlabel("$r/a$")
        axE.set_xlim([0, 1])
        axE.set_ylabel("Residual")
        # axE.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax02
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1],
                res.ne[-1],
                "o-",
                c=colors[cont],
                label=label,
                markersize=3,
                lw=0.4,
            )
        roa = res.profiles.profiles["rmin(m)"] / res.profiles.profiles["rmin(m)"][-1]
        Te = res.profiles.profiles["ne(10^19/m^3)"] * 1e-1
        ax.plot(roa, Te, "-", lw=0.2, c="g")  # colors[cont])
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        # ax.set_ylim(bottom=0)
        ax.set_title("Electron Density")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax, axE = ax12, ax22
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1][1:],
                res.Ge_tar[-1][1:],
                "--o",
                c=colors[cont],
                label=label,
                markersize=3,
            )
            ax.plot(
                res.roa[-1][1:], res.Ge_sim[-1][1:], "-o", c=colors[cont], markersize=3
            )
            axE.plot(
                res.roa[-1][1:],
                res.Ge_res[-1][1:],
                "-o",
                c=colors[cont],
                label=label,
                markersize=3,
            )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ ($10^{20}/m^2/s$)")
        ax.axhline(y=0, ls="--", c="k", lw=0.5)
        ax.set_title("Particle flux")
        axE.set_xlabel("$r/a$")
        axE.set_xlim([0, 1])
        axE.set_ylabel("Residual")
        # axE.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        # Together

        ax = ax03
        axt = ax03.twinx()
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(
                res.roa[-1], res.Te[-1], "o-", c="r", label=label, markersize=3, lw=0.5
            )
            ax.plot(
                res.roa[-1],
                res.Ti[-1, 0],
                "o-",
                c="b",
                label=label,
                markersize=3,
                lw=0.5,
            )
            axt.plot(
                res.roa[-1], res.ne[-1], "o-", c="m", label=label, markersize=3, lw=0.5
            )
        roa = res.profiles.profiles["rmin(m)"] / res.profiles.profiles["rmin(m)"][-1]
        ax.plot(
            roa, res.profiles.profiles["te(keV)"], "-", label="profiles", lw=0.2, c="r"
        )
        ax.plot(
            roa,
            res.profiles.profiles["ti(keV)"][:, 0],
            "--",
            label="profiles",
            lw=0.2,
            c="b",
        )
        axt.plot(
            roa,
            res.profiles.profiles["ne(10^19/m^3)"] * 1e-1,
            "-.",
            label="profiles",
            lw=0.2,
            c="m",
        )

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$T$ (keV)")
        axt.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        ax.set_title("All together")
        # ax.legend(prop={'size':6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax, axE = ax13, ax23
        for cont, label in enumerate(labels):
            res = self.results[label]
            ax.plot(res.roa[-1], res.Qe_tar[-1], "--o", c="r", label="Qe", markersize=3)
            ax.plot(res.roa[-1], res.Qe_sim[-1], "-o", c="r", markersize=3)
            axE.plot(res.roa[-1], res.Qe_res[-1], "-o", c="r", markersize=3)

            ax.plot(res.roa[-1], res.Qi_tar[-1], "--o", c="b", label="Qi", markersize=3)
            ax.plot(res.roa[-1], res.Qi_sim[-1], "-o", c="b", markersize=3)
            axE.plot(res.roa[-1], res.Qi_res[-1], "-o", c="b", markersize=3)

            ax.plot(
                res.roa[-1], res.Ce_tar[-1], "--o", c="m", label="Qconv", markersize=3
            )
            ax.plot(res.roa[-1], res.Ce_sim[-1], "-o", c="m", markersize=3)
            axE.plot(res.roa[-1], np.abs(res.Ce_res[-1]), "-o", c="m", markersize=3)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q$ ($MW/m^2$)")
        ax.set_title("Energy fluxes")
        axE.set_xlabel("$r/a$")
        axE.set_xlim([0, 1])
        axE.set_ylabel("|Residual|")
        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        figProf_1 = self.fn.add_figure(tab_color=fn_color, label="GACODE-Prof.")
        figProf_2 = self.fn.add_figure(tab_color=fn_color, label="GACODE-Power")
        figProf_3 = self.fn.add_figure(tab_color=fn_color, label="GACODE-Geom.")
        figProf_4 = self.fn.add_figure(tab_color=fn_color, label="GACODE-Grad.")
        figProf_6 = self.fn.add_figure(tab_color=fn_color, label="GACODE-Other")
        figProf_7 = self.fn.add_figure(tab_color=fn_color, label="GACODE-Impurities")

        grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        axsProf_1 = [
            figProf_1.add_subplot(grid[0, 0]),
            figProf_1.add_subplot(grid[1, 0]),
            figProf_1.add_subplot(grid[2, 0]),
            figProf_1.add_subplot(grid[0, 1]),
            figProf_1.add_subplot(grid[1, 1]),
            figProf_1.add_subplot(grid[2, 1]),
            figProf_1.add_subplot(grid[0, 2]),
            figProf_1.add_subplot(grid[1, 2]),
            figProf_1.add_subplot(grid[2, 2]),
        ]

        grid = plt.GridSpec(3, 2, hspace=0.3, wspace=0.3)
        axsProf_2 = [
            figProf_2.add_subplot(grid[0, 0]),
            figProf_2.add_subplot(grid[0, 1]),
            figProf_2.add_subplot(grid[1, 0]),
            figProf_2.add_subplot(grid[1, 1]),
            figProf_2.add_subplot(grid[2, 0]),
            figProf_2.add_subplot(grid[2, 1]),
        ]
        grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.3)
        ax00c = figProf_3.add_subplot(grid[0, 0])
        axsProf_3 = [
            ax00c,
            figProf_3.add_subplot(grid[1, 0]),
            figProf_3.add_subplot(grid[2, 0]),
            figProf_3.add_subplot(grid[0, 1]),
            figProf_3.add_subplot(grid[1, 1]),
            figProf_3.add_subplot(grid[2, 1]),
            figProf_3.add_subplot(grid[0, 2]),
            figProf_3.add_subplot(grid[1, 2]),
            figProf_3.add_subplot(grid[2, 2]),
            figProf_3.add_subplot(grid[0, 3]),
            figProf_3.add_subplot(grid[1, 3]),
            figProf_3.add_subplot(grid[2, 3]),
        ]

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        axsProf_4 = [
            figProf_4.add_subplot(grid[0, 0]),
            figProf_4.add_subplot(grid[0, 1]),
            figProf_4.add_subplot(grid[0, 2]),
            figProf_4.add_subplot(grid[1, 0]),
            figProf_4.add_subplot(grid[1, 1]),
            figProf_4.add_subplot(grid[1, 2]),
        ]

        grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
        axsProf_6 = [
            figProf_6.add_subplot(grid[0, 0]),
            figProf_6.add_subplot(grid[:, 1]),
            figProf_6.add_subplot(grid[0, 2]),
            figProf_6.add_subplot(grid[1, 0]),
            figProf_6.add_subplot(grid[1, 2]),
            figProf_6.add_subplot(grid[0, 3]),
            figProf_6.add_subplot(grid[1, 3]),
        ]

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        axsImps = [
            figProf_7.add_subplot(grid[0, 0]),
            figProf_7.add_subplot(grid[0, 1]),
            figProf_7.add_subplot(grid[0, 2]),
            figProf_7.add_subplot(grid[1, 0]),
            figProf_7.add_subplot(grid[1, 1]),
            figProf_7.add_subplot(grid[1, 2]),
        ]

        # **** Final real

        if "profiles" in self.results[label].__dict__:
            try:
                self.results[label].profiles.plot(
                    axs1=axsProf_1,
                    axs2=axsProf_2,
                    axs3=axsProf_3,
                    axs4=axsProf_4,
                    axs6=axsProf_6,
                    axsImps=axsImps,
                    color="b",
                    legYN=True,
                    extralab="original, ",
                )
            except:
                print("Could not plot profiles", typeMsg="w")

        if "profiles_final" in self.results[label].__dict__:
            try:
                self.results[label].profiles_final.plot(
                    axs1=axsProf_1,
                    axs2=axsProf_2,
                    axs3=axsProf_3,
                    axs4=axsProf_4,
                    axs6=axsProf_6,
                    axsImps=axsImps,
                    color="r",
                    legYN=True,
                    extralab="final, ",
                )
            except:
                print("Could not plot profiles_final", typeMsg="w")

            try:
                figFlows = self.fn.add_figure(
                    tab_color=fn_color, label="GACODE-FlowsFin"
                )
                self.results[label].plotBalance(fig=figFlows)
            except:
                print("Could not plot final flows", typeMsg="w")

        for label in labels:
            if label in self.tglf:
                self.tglf[label].plot(
                    fn=self.fn,
                    labels=[f"{self.nameRuns_default}_tglf1"],
                    fontsizeLeg=5,
                    normalizations=None,
                    extratitle="TGLF_" + label,
                    plotGACODE=False,
                )


class TGYROoutput:
    def __init__(self, FolderTGYRO, profiles=None):
        self.FolderTGYRO = FolderTGYRO

        if (profiles is None) and (FolderTGYRO / "input.gacode").exists():
            from mitim_tools.gacode_tools import PROFILEStools
            profiles = PROFILEStools.gacode_state(FolderTGYRO / f"input.gacode", calculateDerived=False)

        self.profiles = profiles

        self.readResidual()
        self.readGeo()
        self.readFluxes()
        self.readNormalization()
        self.unnormalize_quantities()
        self.readNu()
        self.readProfiles()

        calculateDerived = True
        try:
            self.profiles_final = PROFILEStools.gacode_state(self.FolderTGYRO / "input.gacode.new",calculateDerived=calculateDerived,)
        except:
            self.profiles_final = None

        self.derived()
        self.readprocess()

        self.readResidualManuals()

    def maskp(self, var):
        return var[self.norepeats, :]

    def readResidual(self):
        file = self.FolderTGYRO / f"out.tgyro.residual"
        with open(file, "r") as f:
            aux = f.readlines()

        self.residual, self.iterations, self.calls_solver = [], [], []
        for i in range(len(aux)):
            if "ITERATION" in aux[i]:
                ff = aux[i].replace("*", " ").split()
                self.iterations.append(int(ff[2]))
                self.residual.append(float(ff[3]))
                self.calls_solver.append(int(ff[5].replace("]", " ")))
        self.iterations = np.array(self.iterations)
        self.residual = np.array(self.residual)
        self.calls_solver = np.array(self.calls_solver)

        # calls_solvers for cold_starting cases is not straight=forward
        calls_solver_corrected = copy.deepcopy(self.calls_solver)
        for i in range(len(calls_solver_corrected) - 1):
            if (calls_solver_corrected[i + 1] - calls_solver_corrected[i]) < 0:
                calls_solver_corrected[i + 1 :] += self.calls_solver[i]
        self.calls_solver = calls_solver_corrected

        """
		For some reason that I don't understand, TGYRO repeats some values randomly in the files (problem with IO), so I need to define
		a mask to apply for every quantity that I read
		"""

        self.norepeats = []
        cont = 0
        for i in range(self.iterations.shape[0]):
            if self.iterations[i] == cont:
                self.norepeats.append(i)
                cont += 1

        self.iterations = self.maskp(np.transpose(np.atleast_2d(self.iterations)))[
            :, -1
        ]
        self.calls_solver = self.maskp(np.transpose(np.atleast_2d(self.calls_solver)))[
            :, -1
        ]
        self.residual = self.maskp(np.transpose(np.atleast_2d(self.residual)))[:, -1]

    def readFluxes(self):
        file = self.FolderTGYRO / f"out.tgyro.run"
        with open(file, "r") as f:
            aux = f.readlines()
        for i in aux:
            if "LOC_TE_FEEDBACK_FLAG" in i:
                self.Te_predicted = i.split()[-1] == "ON"
            if "LOC_TI_FEEDBACK_FLAG" in i:
                self.Ti_predicted = i.split()[-1] == "ON"
            if "LOC_ER_FEEDBACK_FLAG" in i:
                self.Er_predicted = i.split()[-1] == "ON"
            if "TGYRO_DEN_METHOD0" in i:
                self.ne_predicted = i.split()[-1] != "fixed"

        file = self.FolderTGYRO / f"out.tgyro.evo_te"
        if file.exists():
            self.roa, self.QeGB_sim, self.QeGB_tar = GACODEinterpret.readGeneral(
                file, numcols=3, maskfun=self.maskp
            )
        else:
            self.QeGB_sim, self.QeGB_tar = None, None

        file = self.FolderTGYRO / f"out.tgyro.evo_ti"
        if file.exists():
            self.roa, self.QiGB_sim, self.QiGB_tar = GACODEinterpret.readGeneral(
                file, numcols=3, maskfun=self.maskp
            )
        else:
            self.QiGB_sim, self.QiGB_tar = None, None

        file = self.FolderTGYRO / f"out.tgyro.evo_ne"
        if file.exists():
            self.roa, self.GeGB_sim, self.GeGB_tar = GACODEinterpret.readGeneral(
                file, numcols=3, maskfun=self.maskp
            )
        else:
            self.GeGB_sim, self.GeGB_tar = None, None

        file = self.FolderTGYRO / f"out.tgyro.evo_er"
        if file.exists():
            self.roa, self.MtGB_sim, self.MtGB_tar = GACODEinterpret.readGeneral(
                file, numcols=3, maskfun=self.maskp
            )
        else:
            self.MtGB_sim, self.MtGB_tar = None, None

        self.GiGB_sim, self.GiGB_tar = [], []

        for i in range(10):
            file = self.FolderTGYRO / f"out.tgyro.evo_n{i + 1}"
            if file.exists():
                _, GiGB_sim, GiGB_tar = GACODEinterpret.readGeneral(
                    file, numcols=3, maskfun=self.maskp
                )

                self.GiGB_sim.append(GiGB_sim)
                self.GiGB_tar.append(GiGB_tar)

            else:
                break

        self.GiGB_sim = np.array(self.GiGB_sim)
        self.GiGB_tar = np.array(self.GiGB_tar)

        [self.num_iterations, self.radii] = self.roa.shape

        self.iterations = np.arange(0, self.num_iterations)

        if self.QeGB_sim is None:
            self.QeGB_sim = self.QeGB_tar = np.zeros((self.num_iterations, self.radii))
        if self.QiGB_sim is None:
            self.QiGB_sim = self.QiGB_tar = np.zeros((self.num_iterations, self.radii))
        if self.GeGB_sim is None:
            self.GeGB_sim = self.GeGB_tar = np.zeros((self.num_iterations, self.radii))
        if self.MtGB_sim is None:
            self.MtGB_sim = self.MtGB_tar = np.zeros((self.num_iterations, self.radii))

        self.QeGB_res = np.abs(self.QeGB_sim - self.QeGB_tar)
        self.QiGB_res = np.abs(self.QiGB_sim - self.QiGB_tar)
        self.GeGB_res = np.abs(self.GeGB_sim - self.GeGB_tar)
        self.GiGB_res = np.abs(self.GiGB_sim - self.GiGB_tar)
        self.MtGB_res = np.abs(self.MtGB_sim - self.MtGB_tar)

        # ***************************************************************
        # Fluxes
        # ***************************************************************

        file = self.FolderTGYRO / f"out.tgyro.flux_e"
        (
            _,
            self.GeGB_sim_neo,
            self.GeGB_sim_turb,
            self.QeGB_sim_neo,
            self.QeGB_sim_turb,
            self.MeGB_sim_neo,
            self.MeGB_sim_turb,
            self.EXeGB_sim_turb,
        ) = GACODEinterpret.readGeneral(file, numcols=8, maskfun=self.maskp)

        self.EXeGB_sim = self.EXeGB_sim_turb

        (
            self.GiGB_sim_neo,
            self.GiGB_sim_turb,
            self.QiGB_sim_neo,
            self.QiGB_sim_turb,
            self.MiGB_sim_neo,
            self.MiGB_sim_turb,
            self.EXiGB_sim_turb,
        ) = ([], [], [], [], [], [], [])

        self.EXiGB_sim = self.EXiGB_sim_turb

        for i in range(10):
            file = self.FolderTGYRO / f"out.tgyro.flux_i{i + 1}"
            if file.exists():
                (
                    _,
                    GiGB_sim_neo,
                    GiGB_sim_turb,
                    QiGB_sim_neo,
                    QiGB_sim_turb,
                    MiGB_sim_neo,
                    MiGB_sim_turb,
                    EXiGB_sim_turb,
                ) = GACODEinterpret.readGeneral(file, numcols=8, maskfun=self.maskp)

                self.GiGB_sim_neo.append(GiGB_sim_neo)
                self.GiGB_sim_turb.append(GiGB_sim_turb)
                self.QiGB_sim_turb.append(QiGB_sim_turb)
                self.QiGB_sim_neo.append(QiGB_sim_neo)
                self.MiGB_sim_neo.append(MiGB_sim_neo)
                self.MiGB_sim_turb.append(MiGB_sim_turb)
                self.EXiGB_sim_turb.append(EXiGB_sim_turb)

            else:
                break

        self.GiGB_sim_neo = np.array(self.GiGB_sim_neo)
        self.GiGB_sim_turb = np.array(self.GiGB_sim_turb)
        self.QiGB_sim_neo = np.array(self.QiGB_sim_neo)
        self.QiGB_sim_turb = np.array(self.QiGB_sim_turb)
        self.MiGB_sim_neo = np.array(self.MiGB_sim_neo)
        self.MiGB_sim_turb = np.array(self.MiGB_sim_turb)
        self.EXiGB_sim_turb = np.array(self.EXiGB_sim_turb)

        # Total momentum, sum over all species
        self.MtGB_sim_neo = self.MiGB_sim_neo.sum(axis=0) + self.MeGB_sim_neo
        self.MtGB_sim_turb = self.MiGB_sim_turb.sum(axis=0) + self.MeGB_sim_turb

        # ***************************************************************
        # Errors - Constructed outside of TGYRO call (e.g. powerstate)
        # ***************************************************************

        if not (self.FolderTGYRO / "out.tgyro.flux_e_stds").exists():
            self.tgyro_stds = False

        else:
            print("\t- Errors in TGYRO fluxes and targets found, adding to class")
            self.tgyro_stds = True

            file = self.FolderTGYRO / "out.tgyro.flux_e_stds"
            (
                _,
                self.GeGB_sim_neo_stds,
                self.GeGB_sim_turb_stds,
                self.QeGB_sim_neo_stds,
                self.QeGB_sim_turb_stds,
                self.MeGB_sim_neo_stds,
                self.MeGB_sim_turb_stds,
                self.EXeGB_sim_turb_stds,
            ) = GACODEinterpret.readGeneral(file, numcols=8, maskfun=self.maskp)

            self.EXeGB_sim_stds = self.EXeGB_sim_turb_stds

            (
                self.GiGB_sim_neo_stds,
                self.GiGB_sim_turb_stds,
                self.QiGB_sim_neo_stds,
                self.QiGB_sim_turb_stds,
                self.MiGB_sim_neo_stds,
                self.MiGB_sim_turb_stds,
                self.EXiGB_sim_turb_stds,
            ) = ([], [], [], [], [], [], [])

            self.EXiGB_sim_stds = self.EXiGB_sim_turb_stds

            for i in range(10):
                file = self.FolderTGYRO / f"out.tgyro.flux_i{i + 1}_stds"
                if file.exists():
                    (
                        _,
                        GiGB_sim_neo,
                        GiGB_sim_turb,
                        QiGB_sim_neo,
                        QiGB_sim_turb,
                        MiGB_sim_neo,
                        MiGB_sim_turb,
                        EXiGB_sim_turb,
                    ) = GACODEinterpret.readGeneral(file, numcols=8, maskfun=self.maskp)

                    self.GiGB_sim_neo_stds.append(GiGB_sim_neo)
                    self.GiGB_sim_turb_stds.append(GiGB_sim_turb)
                    self.QiGB_sim_turb_stds.append(QiGB_sim_turb)
                    self.QiGB_sim_neo_stds.append(QiGB_sim_neo)
                    self.MiGB_sim_neo_stds.append(MiGB_sim_neo)
                    self.MiGB_sim_turb_stds.append(MiGB_sim_turb)
                    self.EXiGB_sim_turb_stds.append(EXiGB_sim_turb)

                else:
                    break

            self.GiGB_sim_neo_stds = np.array(self.GiGB_sim_neo_stds)
            self.GiGB_sim_turb_stds = np.array(self.GiGB_sim_turb_stds)
            self.QiGB_sim_neo_stds = np.array(self.QiGB_sim_neo_stds)
            self.QiGB_sim_turb_stds = np.array(self.QiGB_sim_turb_stds)
            self.MiGB_sim_neo_stds = np.array(self.MiGB_sim_neo_stds)
            self.MiGB_sim_turb_stds = np.array(self.MiGB_sim_turb_stds)
            self.EXiGB_sim_turb_stds = np.array(self.EXiGB_sim_turb_stds)

            # Total momentum, sum over all species
            self.MtGB_sim_neo_stds = (
                self.MiGB_sim_neo_stds.sum(axis=0) + self.MeGB_sim_neo_stds
            )
            self.MtGB_sim_turb_stds = (
                self.MiGB_sim_turb_stds.sum(axis=0) + self.MeGB_sim_turb_stds
            )

            # Targets
            file = self.FolderTGYRO / f"out.tgyro.evo_te_stds"
            if file.exists():
                _, _, self.QeGB_tar_stds = GACODEinterpret.readGeneral(
                    file, numcols=3, maskfun=self.maskp
                )
            else:
                self.QeGB_tar_stds = None

            file = self.FolderTGYRO / f"out.tgyro.evo_ti_stds"
            if file.exists():
                _, _, self.QiGB_tar_stds = GACODEinterpret.readGeneral(
                    file, numcols=3, maskfun=self.maskp
                )
            else:
                self.QiGB_tar_stds = None

            file = self.FolderTGYRO / f"out.tgyro.evo_ne_stds"
            if file.exists():
                _, _, self.GeGB_tar_stds = GACODEinterpret.readGeneral(
                    file, numcols=3, maskfun=self.maskp
                )
            else:
                self.GeGB_tar_stds = None

            file = self.FolderTGYRO / f"out.tgyro.evo_er_stds"
            if file.exists():
                _, _, self.MtGB_tar_stds = GACODEinterpret.readGeneral(
                    file, numcols=3, maskfun=self.maskp
                )
            else:
                self.MtGB_tar_stds = None

            self.GiGB_tar_stds = []
            for i in range(10):
                file = self.FolderTGYRO / f"out.tgyro.evo_n{i + 1}_stds"
                if file.exists():
                    _, _, GiGB_tar = GACODEinterpret.readGeneral(
                        file, numcols=3, maskfun=self.maskp
                    )
                    self.GiGB_tar_stds.append(GiGB_tar)
                else:
                    break
            self.GiGB_tar_stds = np.array(self.GiGB_tar)

        # ***************************************************************
        # Powers
        # ***************************************************************

        file = self.FolderTGYRO / f"out.tgyro.power_e"
        try:
            (
                _,
                self.Qe_tarMW_fus,
                self.Qe_tarMW_aux,
                Qe_tarMW_ohm,
                self.Qe_tarMW_brem,
                self.Qe_tarMW_sync,
                self.Qe_tarMW_line,
                self.Qe_tarMW_exch,
                self.Qe_tarMW_expwd,
                self.Qe_tarMW_tot,
            ) = GACODEinterpret.readGeneral(file, numcols=10, maskfun=self.maskp)

            # ADD OHMIC TO AUX
            self.Qe_tarMW_aux += Qe_tarMW_ohm
            # ----------------

        except:
            print(
                "\t* TGYRO power file seems to be old (before 10/05/2022), Ohmic power is included in Aux",
                typeMsg="w",
            )
            (
                _,
                self.Qe_tarMW_fus,
                self.Qe_tarMW_aux,
                self.Qe_tarMW_brem,
                self.Qe_tarMW_sync,
                self.Qe_tarMW_line,
                self.Qe_tarMW_exch,
                self.Qe_tarMW_expwd,
                self.Qe_tarMW_tot,
            ) = GACODEinterpret.readGeneral(file, numcols=9, maskfun=self.maskp)

        self.Qe_tarMW_rad = self.Qe_tarMW_brem + self.Qe_tarMW_sync + self.Qe_tarMW_line

        file = self.FolderTGYRO / f"out.tgyro.power_i"
        (
            _,
            self.Qi_tarMW_fus,
            self.Qi_tarMW_aux,
            self.Qi_tarMW_exch,
            self.Qi_tarMW_expwd,
            self.Qi_tarMW_tot,
        ) = GACODEinterpret.readGeneral(file, numcols=6, maskfun=self.maskp)

    def readNormalization(self):
        file = self.FolderTGYRO / f"out.tgyro.gyrobohm"

        (
            roa,
            self.Chi_GB,
            self.Q_GB,
            self.Gamma_GB,
            self.Pi_GB,
            self.S_GB,
            self.c_s,
        ) = GACODEinterpret.readGeneral(file, numcols=7, maskfun=self.maskp)

        # Q is MW/m^2
        # Pi in J/m^2
        # S in MW/m^3

        # Convert Gamma to 1E20
        self.Gamma_GB = self.Gamma_GB * 1e-1

    def readResidualManuals(self):
        # Residual is always given in GB, so here I do myself
        tots = int(self.Te_predicted) + int(self.Ti_predicted) + int(self.ne_predicted)

        self.QeGB_res_mean = np.mean(self.QeGB_res[:, 1:], axis=1)
        self.QiGB_res_mean = np.mean(self.QiGB_res[:, 1:], axis=1)
        self.GeGB_res_mean = np.mean(self.GeGB_res[:, 1:], axis=1)
        self.CeGB_res_mean = np.mean(self.CeGB_res[:, 1:], axis=1)
        G = self.GeGB_res_mean
        self.residual_manual = (
            self.QeGB_res_mean * int(self.Te_predicted)
            + self.QiGB_res_mean * int(self.Ti_predicted)
            + G * int(self.ne_predicted)
        ) / tots

        self.QeGB_res_max = np.max(self.QeGB_res[:, 1:], axis=1)
        self.QiGB_res_max = np.max(self.QiGB_res[:, 1:], axis=1)
        self.GeGB_res_max = np.max(self.GeGB_res[:, 1:], axis=1)
        self.CeGB_res_max = np.max(self.CeGB_res[:, 1:], axis=1)
        G = self.GeGB_res_max
        self.max_manual = np.max(
            np.transpose(
                np.array(
                    [
                        self.QeGB_res_max * int(self.Te_predicted),
                        self.QiGB_res_max * int(self.Ti_predicted),
                        G * int(self.ne_predicted),
                    ]
                )
            ),
            axis=1,
        )

        self.Qe_res_mean = np.mean(self.Qe_res[:, 1:], axis=1)
        self.Qi_res_mean = np.mean(self.Qi_res[:, 1:], axis=1)
        self.Ge_res_mean = np.mean(self.Ge_res[:, 1:], axis=1)
        self.Ce_res_mean = np.mean(self.Ce_res[:, 1:], axis=1)
        G = self.Ge_res_mean
        self.residual_manual_real = (
            self.Qe_res_mean * int(self.Te_predicted)
            + self.Qi_res_mean * int(self.Ti_predicted)
            + G * int(self.ne_predicted)
        ) / tots

        self.Qe_res_max = np.max(self.Qe_res[:, 1:], axis=1)
        self.Qi_res_max = np.max(self.Qi_res[:, 1:], axis=1)
        self.Ge_res_max = np.max(self.Ge_res[:, 1:], axis=1)
        self.Ce_res_max = np.max(self.Ce_res[:, 1:], axis=1)
        G = self.Ge_res_max
        self.max_manual_real = np.max(
            np.transpose(
                np.array(
                    [
                        self.Qe_res_max * int(self.Te_predicted),
                        self.Qi_res_max * int(self.Ti_predicted),
                        G * int(self.ne_predicted),
                    ]
                )
            ),
            axis=1,
        )

    def readProfiles(self):
        file = self.FolderTGYRO / f"out.tgyro.profile_e"
        (
            roa,
            self.ne,
            self.aLne,
            self.Te,
            self.aLte,
            self.betae_unit,
        ) = GACODEinterpret.readGeneral(file, numcols=6, maskfun=self.maskp)
        self.ne = self.ne * 1e6 * 1e-20

        self.ni, self.aLni, self.Ti, self.aLti, self.betai_unit = [], [], [], [], []

        cont = 1
        file = self.FolderTGYRO / f"out.tgyro.profile_i{cont}"
        while file.exists():
            _, ni, aLni, Ti, aLti, betai_unit = GACODEinterpret.readGeneral(
                file, numcols=6, maskfun=self.maskp
            )
            cont += 1
            file = self.FolderTGYRO / f"out.tgyro.profile_i{cont}"

            self.ni.append(ni)
            self.aLni.append(aLni)
            self.Ti.append(Ti)
            self.aLti.append(aLti)
            self.betai_unit.append(betai_unit)

        """
		These variables will have (iteration,specie,radius)
		"""
        self.ni = np.transpose(np.array(self.ni), axes=(1, 0, 2)) * 1e6 * 1e-20
        self.aLni = np.transpose(np.array(self.aLni), axes=(1, 0, 2))
        self.Ti = np.transpose(np.array(self.Ti), axes=(1, 0, 2))
        self.aLti = np.transpose(np.array(self.aLti), axes=(1, 0, 2))
        self.betai_unit = np.transpose(np.array(self.betai_unit), axes=(1, 0, 2))

    def readGeo(self):
        file1 = self.FolderTGYRO / f"out.tgyro.geometry.1"
        file2 = self.FolderTGYRO / f"out.tgyro.geometry.2"

        (
            _,
            self.rho,
            self.q,
            self.s,
            self.kappa,
            self.s_kappa,
            self.delta,
            self.s_delta,
            self.shift,
            self.rmajoa,
            self.bunit,
        ) = GACODEinterpret.readGeneral(file1, numcols=11, maskfun=self.maskp)
        (
            _,
            self.zmagoa,
            self.dzmag,
            self.zeta,
            self.s_zeta,
            self.volume,
            self.dvoldr,
            self.gradr,
            self.rmin,
        ) = GACODEinterpret.readGeneral(file2, numcols=9, maskfun=self.maskp)

    def readNu(self):
        file1 = self.FolderTGYRO / f"out.tgyro.nu_rho"
        (
            roa,
            self.nui,
            self.nue,
            self.nu_star,
            self.nu_exch,
            self.rhoia,
            self.rhosa,
            self.fracae,
        ) = GACODEinterpret.readGeneral(file1, numcols=8, maskfun=self.maskp)

    def readprocess(self):
        file = self.FolderTGYRO / f"input.tgyro.gen"
        with open(file, "r") as f:
            aux = f.readlines()

        self.inputs = {}
        for i in aux:
            if len(i.split()) > 1:
                self.inputs[i.split()[1]] = i.split()[0]
            else:
                break

        file = self.FolderTGYRO / f"out.tgyro.iterate"
        with open(file, "r") as f:
            aux = f.readlines()

        # -----

        self.calls_dict = {}
        for i in aux:
            if "ITERATION" in i:
                m = []
                self.calls_dict[i] = m  # [int(i.split()[-1])]
            else:
                m.append([float(j) for j in i.split()])

        nr = self.rho.shape[1] - 1
        nprof = int(self.Te_predicted) + int(self.Ti_predicted) + int(self.ne_predicted)

        if self.inputs["TGYRO_ITERATION_METHOD"] == "1":
            ev_profs, ev_extras = nprof, nr
        elif self.inputs["TGYRO_ITERATION_METHOD"] == "6":
            ev_profs, ev_extras = 1, 0
        else:
            print(
                f"\t- TGYRO_ITERATION_METHOD={self.inputs['TGYRO_ITERATION_METHOD']} logic not implemented yet, assuming same as 1",
                typeMsg="w",
            )
            ev_profs, ev_extras = nprof, nr

        ev_calls = ev_profs * nr + ev_extras

        num_evs = nr
        self.calls_solver_manual = [num_evs]
        for i in self.calls_dict:
            num_evs += ev_calls
            for j in range(len(self.calls_dict[i]) - 1):
                num_evs += ev_extras
            self.calls_solver_manual.append(num_evs)
        self.calls_solver_manual = np.array(self.calls_solver_manual)

        # It should be the same as calls_solver, if GACODE works like this..

    def unnormalize_quantities(self):
        # ----------------------------------------
        # Total simulation, target and residual
        # ----------------------------------------

        self.Qe_res, self.Qe_sim, self.Qe_tar = self.Q_GB * (
            self.QeGB_res,
            self.QeGB_sim,
            self.QeGB_tar,
        )
        self.Qi_res, self.Qi_sim, self.Qi_tar = self.Q_GB * (
            self.QiGB_res,
            self.QiGB_sim,
            self.QiGB_tar,
        )
        self.Ge_res, self.Ge_sim, self.Ge_tar = self.Gamma_GB * (
            self.GeGB_res,
            self.GeGB_sim,
            self.GeGB_tar,
        )
        self.Gi_res, self.Gi_sim, self.Gi_tar = self.Gamma_GB * (
            self.GiGB_res,
            self.GiGB_sim,
            self.GiGB_tar,
        )
        self.Mt_res, self.Mt_sim, self.Mt_tar = self.Pi_GB * (
            self.MtGB_res,
            self.MtGB_sim,
            self.MtGB_tar,
        )

        # Convert to total (MW/m^2 -> MW)

        self.Qe_simMW = self.Qe_sim * self.dvoldr
        self.Qe_tarMW = self.Qe_tar * self.dvoldr

        self.Qi_simMW = self.Qi_sim * self.dvoldr
        self.Qi_tarMW = self.Qi_tar * self.dvoldr

        self.Ge_simABS = self.Ge_sim * self.dvoldr
        self.Ge_tarABS = self.Ge_tar * self.dvoldr

        self.Mt_simABS = self.Mt_sim * self.dvoldr
        self.Mt_tarABS = self.Mt_tar * self.dvoldr

        # ----------------------------------------
        # Target Errors
        # ----------------------------------------
        if self.tgyro_stds:
            self.Qe_tar_stds = self.Q_GB * self.QeGB_tar_stds
            self.Qi_tar_stds = self.Q_GB * self.QiGB_tar_stds
            self.Ge_tar_stds = self.Gamma_GB * self.GeGB_tar_stds
            self.Gi_tar_stds = self.Gamma_GB * self.GiGB_tar_stds
            self.Mt_tar_stds = self.Pi_GB * self.MtGB_tar_stds

        # ----------------------------------------
        # Fluxes
        # ----------------------------------------

        self.Qe_sim_neo, self.Qe_sim_turb = self.Q_GB * (
            self.QeGB_sim_neo,
            self.QeGB_sim_turb,
        )
        self.Qi_sim_neo, self.Qi_sim_turb = self.Q_GB * (
            self.QiGB_sim_neo,
            self.QiGB_sim_turb,
        )
        self.Ge_sim_neo, self.Ge_sim_turb = self.Gamma_GB * (
            self.GeGB_sim_neo,
            self.GeGB_sim_turb,
        )
        self.Gi_sim_neo, self.Gi_sim_turb = self.Gamma_GB * (
            self.GiGB_sim_neo,
            self.GiGB_sim_turb,
        )
        self.Me_sim_neo, self.Me_sim_turb = self.Pi_GB * (
            self.MeGB_sim_neo,
            self.MeGB_sim_turb,
        )
        self.Mi_sim_neo, self.Mi_sim_turb = self.Pi_GB * (
            self.MiGB_sim_neo,
            self.MiGB_sim_turb,
        )
        self.EXe_sim, self.EXe_sim_turb = self.S_GB * (
            self.EXeGB_sim,
            self.EXeGB_sim_turb,
        )
        self.EXi_sim, self.EXi_sim_turb = self.S_GB * (
            self.EXiGB_sim,
            self.EXiGB_sim_turb,
        )
        self.Mt_sim_neo, self.Mt_sim_turb = self.Pi_GB * (
            self.MtGB_sim_neo,
            self.MtGB_sim_turb,
        )

        # Sum thermal ions
        self.QiIons_sim_turb = self.Qe_res * 0.0
        self.QiIons_sim_neo = self.Qe_res * 0.0
        self.QiIons_sim_turb_thr = self.Qe_res * 0.0
        self.QiIons_sim_neo_thr = self.Qe_res * 0.0
        for i in range(self.Qi_sim_turb.shape[0]):
            self.QiIons_sim_neo += self.Qi_sim_neo[i, :, :]
            self.QiIons_sim_turb += self.Qi_sim_turb[i, :, :]
            if self.profiles.Species[i]["S"] == "therm":
                self.QiIons_sim_neo_thr += self.Qi_sim_neo[i, :, :]
                self.QiIons_sim_turb_thr += self.Qi_sim_turb[i, :, :]

        (
            self.QiGBIons_sim_turb,
            self.QiGBIons_sim_neo,
            self.QiGBIons_sim_turb_thr,
            self.QiGBIons_sim_neo_thr,
        ) = (
            self.QiIons_sim_turb,
            self.QiIons_sim_neo,
            self.QiIons_sim_turb_thr,
            self.QiIons_sim_neo_thr,
        ) / self.Q_GB

        # ----------------------------------------
        # Fluxes Errors
        # ----------------------------------------

        if self.tgyro_stds:
            self.Qe_sim_neo_stds, self.Qe_sim_turb_stds = self.Q_GB * (
                self.QeGB_sim_neo_stds,
                self.QeGB_sim_turb_stds,
            )
            self.Qi_sim_neo_stds, self.Qi_sim_turb_stds = self.Q_GB * (
                self.QiGB_sim_neo_stds,
                self.QiGB_sim_turb_stds,
            )
            self.Ge_sim_neo_stds, self.Ge_sim_turb_stds = self.Gamma_GB * (
                self.GeGB_sim_neo_stds,
                self.GeGB_sim_turb_stds,
            )
            self.Gi_sim_neo_stds, self.Gi_sim_turb_stds = self.Gamma_GB * (
                self.GiGB_sim_neo_stds,
                self.GiGB_sim_turb_stds,
            )
            self.Me_sim_neo_stds, self.Me_sim_turb_stds = self.Pi_GB * (
                self.MeGB_sim_neo_stds,
                self.MeGB_sim_turb_stds,
            )
            self.Mi_sim_neo_stds, self.Mi_sim_turb_stds = self.Pi_GB * (
                self.MiGB_sim_neo_stds,
                self.MiGB_sim_turb_stds,
            )
            self.EXe_sim_stds, self.EXe_sim_turb_stds = self.S_GB * (
                self.EXeGB_sim_stds,
                self.EXeGB_sim_turb_stds,
            )
            self.EXi_sim_stds, self.EXi_sim_turb_stds = self.S_GB * (
                self.EXiGB_sim_stds,
                self.EXiGB_sim_turb_stds,
            )
            self.Mt_sim_neo_stds, self.Mt_sim_turb_stds = self.Pi_GB * (
                self.MtGB_sim_neo_stds,
                self.MtGB_sim_turb_stds,
            )

            # Sum thermal ions
            self.QiIons_sim_turb_stds = self.Qe_res * 0.0
            self.QiIons_sim_neo_stds = self.Qe_res * 0.0
            self.QiIons_sim_turb_thr_stds = self.Qe_res * 0.0
            self.QiIons_sim_neo_thr_stds = self.Qe_res * 0.0
            for i in range(self.Qi_sim_turb_stds.shape[0]):
                self.QiIons_sim_neo_stds += self.Qi_sim_neo_stds[i, :, :]
                self.QiIons_sim_turb_stds += self.Qi_sim_turb_stds[i, :, :]
                if self.profiles.Species[i]["S"] == "therm":
                    self.QiIons_sim_neo_thr_stds += self.Qi_sim_neo_stds[i, :, :]
                    self.QiIons_sim_turb_thr_stds += self.Qi_sim_turb_stds[i, :, :]

            (
                self.QiGBIons_sim_turb_stds,
                self.QiGBIons_sim_neo_stds,
                self.QiGBIons_sim_turb_thr_stds,
                self.QiGBIons_sim_neo_thr_stds,
            ) = (
                self.QiIons_sim_turb_stds,
                self.QiIons_sim_neo_stds,
                self.QiIons_sim_turb_thr_stds,
                self.QiIons_sim_neo_thr_stds,
            ) / self.Q_GB

    def derived(self):
        # "Convective" flux
        self.Ce_sim = PLASMAtools.convective_flux(self.Te, self.Ge_sim)
        self.Ce_sim_neo = PLASMAtools.convective_flux(self.Te, self.Ge_sim_neo)
        self.Ce_sim_turb = PLASMAtools.convective_flux(self.Te, self.Ge_sim_turb)
        self.Ce_tar = PLASMAtools.convective_flux(self.Te, self.Ge_tar)

        self.CeGB_sim = self.Ce_sim / self.Q_GB
        self.CeGB_tar = self.Ce_tar / self.Q_GB

        self.Ce_simMW = self.Ce_sim * self.dvoldr  # Convert to total (MW/m^2 -> MW)
        self.Ce_tarMW = self.Ce_tar * self.dvoldr  # Convert to total (MW/m^2 -> MW)

        self.CeGB_res = np.abs(self.CeGB_sim - self.CeGB_tar)
        self.Ce_res = np.abs(self.Ce_sim - self.Ce_tar)

        # "Convective" flux for ions
        self.Ci_sim = PLASMAtools.convective_flux(self.Te, self.Gi_sim)
        self.Ci_sim_neo = PLASMAtools.convective_flux(self.Te, self.Gi_sim_neo)
        self.Ci_sim_turb = PLASMAtools.convective_flux(self.Te, self.Gi_sim_turb)
        self.Ci_tar = PLASMAtools.convective_flux(self.Te, self.Gi_tar)

        self.CiGB_sim = self.Ci_sim / self.Q_GB
        self.CiGB_tar = self.Ci_tar / self.Q_GB

        self.Ci_simMW = self.Ci_sim * self.dvoldr  # Convert to total (MW/m^2 -> MW)
        self.Ci_tarMW = self.Ci_tar * self.dvoldr  # Convert to total (MW/m^2 -> MW)

        self.CiGB_res = np.abs(self.CiGB_sim - self.CiGB_tar)
        self.Ci_res = np.abs(self.Ci_sim - self.Ci_tar)

        if self.tgyro_stds:
            self.Ce_sim_turb_stds = PLASMAtools.convective_flux(
                self.Te, self.Ge_sim_turb_stds
            )
            self.Ci_sim_turb_stds = PLASMAtools.convective_flux(
                self.Te, self.Gi_sim_turb_stds
            )
            self.Ce_sim_neo_stds = PLASMAtools.convective_flux(
                self.Te, self.Ge_sim_neo_stds
            )
            self.Ci_sim_neo_stds = PLASMAtools.convective_flux(
                self.Te, self.Gi_sim_neo_stds
            )
            self.Ce_tar_stds = PLASMAtools.convective_flux(self.Te, self.Ge_tar_stds)
            self.Ci_tar_stds = PLASMAtools.convective_flux(self.Te, self.Gi_tar_stds)
        # Calculations from last point of TGYRO

        self.P_alphaT_tgyro = self.Qe_tarMW_fus[:, -1] + self.Qi_tarMW_fus[:, -1]
        self.P_fusT_tgyro = self.P_alphaT_tgyro * 5.0
        self.P_inT_tgyro = self.Qi_tarMW_aux[:, -1] + self.Qe_tarMW_aux[:, -1]

        self.Q = self.P_fusT_tgyro / self.P_inT_tgyro

        self.Pheat_tgyro = self.P_inT_tgyro + self.P_alphaT_tgyro
        # ALL IONS
        (
            self.We_tgyro,
            self.Wi_tgyro,
            self.Ne_tgyro,
            self.Ni_tgyro,
        ) = PLASMAtools.calculateContent(
            self.rmin * 1e-2, self.Te, self.Ti, self.ne, self.ni, self.dvoldr
        )
        self.W_tgyro = self.We_tgyro + self.Wi_tgyro

        """
		Notes:
			Total fusion power
			------------------
			Obtained from the predicted alpha power in TGYRO.
			For robustness, this uses the last point predicted with TGYRO.
			This works well if the fusion produced in the outer plasma (beyond the TGYRO B.C.) is negligible, which is roughly correct.

			Total input power
			------------------
			Input power used in Q calculation is not evolved during a TGYRO run (because it is auxiliary+ohmic), so the first thing
			to try is to grab it from input.gacode (the new one better, although it should be the same!)
			If no input.gacode was provided to the TGYRO class, I use the last point of the TGYRO auxiliary heating vector, but keep in mind
			that this is not accuracy if substantial ammounts of heating occur beyong the TGYRO B.C. 
			Extrapolation has been found to not be very robust.

			Fusion Gain
			------------------
			Note: This is only valid in the converged case???????????????
		"""

        if (self.profiles_final is not None) and ("derived" in self.profiles_final.__dict__):
            prof = self.profiles_final
        elif (self.profiles is not None) and ("derived" in self.profiles.__dict__):
            prof = self.profiles
        else:
            prof = None

        if prof is not None:
            self.P_inT = np.array([prof.derived["qIn"]])
        else:
            self.P_inT = self.P_inT_tgyro

        self.Q_better = self.P_fusT_tgyro / self.P_inT

        if (self.profiles_final is not None) and ("derived" in self.profiles_final.__dict__):
            self.Q_best = self.profiles_final.derived["Q"]

        """
		Notes:
			Stored Energy / Confinement Time calculations
			------------------
			Fusion power is last point simulated, which means that in cases with a lot of fusion power in the edge, this is not accurate
			Total heating power is auxiliary + fusion (case of radiation NOT substracted).
		"""

        self.tauE = self.W_tgyro / self.Pheat_tgyro  # seconds

        # H-factors
        self.h98y2 = np.zeros(self.roa.shape[0])
        self.h89p = np.zeros(self.roa.shape[0])

        """
		Volume average
		"""
        self.ne_avol = PLASMAtools.calculateVolumeAverage(
            self.rmin * 1e-2, self.ne, self.dvoldr
        )

    def useFineGridTargets(self, impurityPosition=1):
        print("\t\t\t* Recalculating targets on the fine grid of input.gacode.new")

        if self.profiles_final is None:
            print(
                "\t\t- input.gacode.new was not loaded! I could not do it", typeMsg="w"
            )
            return

        rho_fine = self.profiles_final.profiles["rho(-)"]

        for i in range(self.rho.shape[0]):
            rho_coarse = self.rho[i, :]

            self.Qe_tar[i, :] = np.interp(
                rho_coarse, rho_fine, self.profiles_final.derived["qe_MWm2"]
            )
            self.Qi_tar[i, :] = np.interp(
                rho_coarse, rho_fine, self.profiles_final.derived["qi_MWm2"]
            )
            self.Ge_tar[i, :] = np.interp(
                rho_coarse, rho_fine, self.profiles_final.derived["ge_10E20m2"]
            )
            self.Ce_tar[i, :] = np.interp(
                rho_coarse, rho_fine, self.profiles_final.derived["ce_MWm2"]
            )
            # Profiles do not include ion fluxes
            for j in range(self.Gi_tar.shape[0]):
                self.Gi_tar[j, i, :], self.Ci_tar[j, i, :] = self.Ce_tar[i, :] * 0.0, self.Ce_tar[i, :] * 0.0

            self.Mt_tar[i, :] = np.interp(
                rho_coarse, rho_fine, self.profiles_final.derived["mt_Jm2"]
            )

        # Also the volumetric power
        self.Qe_tarMW = self.Qe_tar * self.dvoldr
        self.Qi_tarMW = self.Qi_tar * self.dvoldr
        self.Ge_tarMW = self.Ge_tar * self.dvoldr
        self.Ce_tarMW = self.Ce_tar * self.dvoldr

    def plot(self, fn=None, label="", prelabel="", fn_color=None):
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fn = FigureNotebook("TGYRO Output Notebook", geometry="1800x900")

        else:
            self.fn = fn

        # ------------------------------------------------------------------------------
        # Summary 1
        # ------------------------------------------------------------------------------

        fig1 = self.fn.add_figure(
            tab_color=fn_color, label=prelabel + "Overview" + label
        )

        grid = plt.GridSpec(3, 4, hspace=0.45, wspace=0.3)
        ax00 = fig1.add_subplot(grid[:, 0])

        ax01 = fig1.add_subplot(grid[0, 1])
        ax11 = fig1.add_subplot(grid[1, 1])
        ax21 = fig1.add_subplot(grid[2, 1])

        ax02 = fig1.add_subplot(grid[0, 2])
        ax12 = fig1.add_subplot(grid[1, 2])
        ax22 = fig1.add_subplot(grid[2, 2])

        ax03 = fig1.add_subplot(grid[0, 3])
        ax13 = fig1.add_subplot(grid[1, 3])
        ax23 = fig1.add_subplot(grid[2, 3])

        ax = ax00
        ax.plot(self.iterations, self.residual, "-s", color="b", markersize=5)
        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_ylabel("Residual")
        ax.set_yscale("log")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax01
        if self.num_iterations > 1:
            ax.plot(
                self.roa[0],
                self.Te[0],
                "-o",
                c="blue",
                label=f"#{0}",
                markersize=3,
            )
        ax.plot(
            self.roa[-1],
            self.Te[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )

        # from profiles
        roa = self.profiles.profiles["rmin(m)"] / self.profiles.profiles["rmin(m)"][-1]
        Te = self.profiles.profiles["te(keV)"]
        ax.plot(roa, Te, "-", c="green", label="input", lw=0.5)

        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            Te = self.profiles_final.profiles["te(keV)"]
            ax.plot(roa, Te, "-", c="k", label="output", lw=0.5)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("T (keV)")
        ax.set_title("Electron Temperature")
        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax11
        ax.plot(
            self.roa[0],
            self.aLte[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.aLte[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )

        #
        roa = self.profiles.profiles["rmin(m)"] / self.profiles.profiles["rmin(m)"][-1]
        aLTe = self.profiles.derived["aLTe"]
        ax.plot(roa, aLTe, "-", c="green", label="input", lw=0.5)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            aLTe = self.profiles_final.derived["aLTe"]
            ax.plot(roa, aLTe, "-", c="k", label="output", lw=0.5)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_T$")
        ax.set_title("Electron Temperature Gradient")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax21
        ax.plot(
            self.roa[0][1:],
            self.QeGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.QeGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QeGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QeGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q (GB)")
        ax.set_yscale("log")
        ax.set_title("Electron energy flux")

        # -- for legend
        (l1,) = ax.plot(
            self.roa[0][1:], self.Qe_sim[0][1:], "-o", c="k", markersize=3, lw=0.5
        )
        (l2,) = ax.plot(
            self.roa[0][1:], self.Qe_tar[0][1:], "--o", c="k", markersize=3, lw=0.5
        )

        ax.legend([l1, l2], ["$Q_{tr}$", "$Q_{tar}$"], loc="best", prop={"size": 7})
        l1.set_visible(False)
        l2.set_visible(False)
        # -------------

        GRAPHICStools.addDenseAxis(ax)

        ax = ax02
        ax.plot(
            self.roa[0],
            self.Ti[0, 0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Ti[-1, 0],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )

        roa = self.profiles.profiles["rmin(m)"] / self.profiles.profiles["rmin(m)"][-1]
        Ti = self.profiles.profiles["ti(keV)"][:, 0]
        ax.plot(roa, Ti, "-", c="green", label="input", lw=0.5)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            Te = self.profiles_final.profiles["ti(keV)"][:, 0]
            ax.plot(roa, Te, "-", c="k", label="output", lw=0.5)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("T (keV)")
        ax.set_title("Ion Temperature")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax12
        ax.plot(
            self.roa[0],
            self.aLti[0, 0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.aLti[-1, 0],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )
        aLTi = self.profiles.derived["aLTi"][:, 0]
        ax.plot(roa, aLTi, "-", c="green", label="input", lw=0.5)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            aLTi = self.profiles_final.derived["aLTi"][:, 0]
            ax.plot(roa, aLTi, "-", c="k", label="output", lw=0.5)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_T$")
        ax.set_title("Ion Temperature Gradient")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax22
        ax.plot(
            self.roa[0][1:],
            self.QiGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.QiGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QiGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QiGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q (GB)")
        ax.set_yscale("log")
        ax.set_title("Ion energy flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax03
        ax.plot(
            self.roa[0],
            self.ne[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.ne[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )

        roa = self.profiles.profiles["rmin(m)"] / self.profiles.profiles["rmin(m)"][-1]
        ne = self.profiles.profiles["ne(10^19/m^3)"] * 1e-1
        ax.plot(roa, ne, "-", c="green", label="input", lw=0.5)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            ne = self.profiles_final.profiles["ne(10^19/m^3)"] * 1e-1
            ax.plot(roa, ne, "-", c="k", label="output", lw=0.5)

        ax.plot(self.roa[0], self.ni[0, 0], "--o", c="blue", markersize=3)
        ax.plot(self.roa[-1], self.ni[-1, 0], "--o", c="red", markersize=3)
        ne = self.profiles.profiles["ni(10^19/m^3)"][:, 0] * 1e-1
        ax.plot(roa, ne, "-", c="magenta", label="input", lw=0.5)

        # -- for legend
        (l1,) = ax.plot(self.roa[0], self.ne[0], "-o", c="k", markersize=3, lw=0.5)
        (l2,) = ax.plot(self.roa[0], self.ni[0, 0], "--o", c="k", markersize=3, lw=0.5)

        ax.legend([l1, l2], ["$n_e$", "$n_{i,1}$"], loc="best", prop={"size": 7})
        l1.set_visible(False)
        l2.set_visible(False)
        # -------------

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("n ($10^{20}m^{-3}$)")
        ax.set_title("Electron Density")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax13
        ax.plot(
            self.roa[0],
            self.aLne[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.aLne[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )

        aLne = self.profiles.derived["aLne"]
        ax.plot(roa, aLne, "-", c="green", label="input", lw=0.5)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            aLne = self.profiles_final.derived["aLne"]
            ax.plot(roa, aLne, "-", c="k", label="output", lw=0.5)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_n$")
        ax.set_title("Density Gradient")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax23
        ax.plot(
            self.roa[0][1:],
            self.GeGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.GeGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.GeGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.GeGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ (GB)")
        # ax.set_yscale('log')
        ax.set_title("Particle flux")

        ax.axhline(y=0, lw=0.5, ls="--", c="k")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        # ------------------------------------------------------------------------------
        # Summary 2
        # ------------------------------------------------------------------------------

        fig1 = self.fn.add_figure(tab_color=fn_color, label=prelabel + "Match" + label)

        grid = plt.GridSpec(4, 4, hspace=0.45, wspace=0.3)
        ax00 = fig1.add_subplot(grid[0, 0])
        ax10 = fig1.add_subplot(grid[1, 0])
        ax20 = fig1.add_subplot(grid[2, 0])
        ax30 = fig1.add_subplot(grid[3, 0])

        ax01 = fig1.add_subplot(grid[0, 1])
        ax11 = fig1.add_subplot(grid[1, 1])  # )
        ax21 = fig1.add_subplot(grid[2, 1])  # )
        ax31 = fig1.add_subplot(grid[3, 1])  # )

        ax02 = fig1.add_subplot(grid[0, 2])  # )
        ax12 = fig1.add_subplot(grid[1, 2])  # )
        ax22 = fig1.add_subplot(grid[2, 2])  # )
        ax32 = fig1.add_subplot(grid[3, 2])  # )

        ax03 = fig1.add_subplot(grid[0, 3])  # )
        ax13 = fig1.add_subplot(grid[1, 3])  # )
        ax23 = fig1.add_subplot(grid[2, 3])  # )
        ax33 = fig1.add_subplot(grid[3, 3])  # )

        # Electrons
        ax = ax00
        ax.plot(
            self.roa[0][1:],
            self.QeGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.QeGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QeGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QeGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q (GB)")
        ax.set_yscale("log")
        ax.set_title("Electron GB energy flux")

        # -- for legend
        (l1,) = ax.plot(
            self.roa[0][1:], self.Qe_sim[0][1:], "-o", c="k", markersize=3, lw=0.5
        )
        (l2,) = ax.plot(
            self.roa[0][1:], self.Qe_tar[0][1:], "--o", c="k", markersize=3, lw=0.5
        )

        ax.legend([l1, l2], ["$Q_{tr}$", "$Q_{tar}$"], loc="best", prop={"size": 7})
        l1.set_visible(False)
        l2.set_visible(False)

        # -------------

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax01
        ax.plot(
            self.roa[0][1:],
            self.QeGB_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QeGB_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q (GB)")
        ax.set_yscale("log")
        ax.set_title("Electron GB energy residual")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax02
        ax.plot(
            self.roa[0][1:],
            self.Qe_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.Qe_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Qe_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Qe_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q ($MW/m^2$)")
        # ax.set_yscale('log')
        ax.set_title("Electron energy flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax03
        ax.plot(
            self.roa[0][1:],
            self.Qe_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Qe_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q ($MW/m^2$)")
        # ax.set_yscale('log')
        ax.set_title("Electron energy residual")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # Ions
        ax = ax10
        ax.plot(
            self.roa[0][1:],
            self.QiGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.QiGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QiGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QiGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q (GB)")
        ax.set_yscale("log")
        ax.set_title("Ion GB energy flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax11
        ax.plot(
            self.roa[0][1:],
            self.QiGB_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.QiGB_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q (GB)")
        ax.set_yscale("log")
        ax.set_title("Ion GB energy residual")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax12
        ax.plot(
            self.roa[0][1:],
            self.Qi_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.Qi_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Qi_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Qi_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q ($MW/m^2$)")
        # ax.set_yscale('log')
        ax.set_title("Ion energy flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax13
        ax.plot(
            self.roa[0][1:],
            self.Qi_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Qi_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q ($MW/m^2$)")
        # ax.set_yscale('log')
        ax.set_title("Ion energy residual")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # Particles
        ax = ax20
        ax.plot(
            self.roa[0][1:],
            self.GeGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.GeGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.GeGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.GeGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ (GB)")
        # ax.set_yscale('log')
        ax.set_title("Electron GB particle flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax21
        ax.plot(
            self.roa[0][1:],
            self.GeGB_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.GeGB_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ (GB)")
        # ax.set_yscale('log')
        ax.set_title("Electron GB particle residual")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax22
        ax.plot(
            self.roa[0][1:],
            self.Ge_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.Ge_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Ge_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Ge_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ ($10^{20}/m^2/s$)")
        # ax.set_yscale('log')
        ax.set_title("Electron particle flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax23
        ax.plot(
            self.roa[0][1:],
            self.Ge_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Ge_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xticklabels([])  # ax.set_xlabel('$r/a$');
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ ($10^{20}/m^2/s$)")
        # ax.set_yscale('log')
        ax.set_title("Electron particle residual")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        # Convective
        ax = ax30
        ax.plot(
            self.roa[0][1:],
            self.CeGB_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.CeGB_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.CeGB_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.CeGB_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q_{conv}$ (GB)")
        # ax.set_yscale('log')
        ax.set_title("Electron GB convective flux (w/factor)")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax31
        ax.plot(
            self.roa[0][1:],
            self.CeGB_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.CeGB_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q_{conv}$ (GB)")
        # ax.set_yscale('log')
        ax.set_title("Electron GB convective residual (w/factor)")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax32
        ax.plot(
            self.roa[0][1:],
            self.Ce_tar[0][1:],
            "--o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[0][1:],
            self.Ce_sim[0][1:],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Ce_tar[-1][1:],
            "--o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Ce_sim[-1][1:],
            "-o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q ($MW/m^2$)")  # ax.set_yscale('log')
        ax.set_title("Electron convective flux")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax33
        ax.plot(
            self.roa[0][1:],
            self.Ce_res[0][1:],
            "-.o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1][1:],
            self.Ce_res[-1][1:],
            "-.o",
            c="red",
            label=f"#{0}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Q ($MW/m^2$)")  # ax.set_yscale('log')
        ax.set_title("Electron convective residual")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # ------------------------------------------------------------------------------
        # Convergence
        # ------------------------------------------------------------------------------

        fig1 = self.fn.add_figure(
            tab_color=fn_color, label=prelabel + "Convergence" + label
        )

        try:
            self.plotConvergence(fig1=fig1)
        except:
            print(
                "Could not plot convergence... maybe because several tgyro methods used?",
                typeMsg="w",
            )

        # ------------------------------------------------------------------------------
        # Fluxes
        # ------------------------------------------------------------------------------

        fig1 = self.fn.add_figure(tab_color=fn_color, label=prelabel + "Fluxes" + label)

        grid = plt.GridSpec(4, self.Qi_sim_turb.shape[0] + 3, hspace=0.3, wspace=0.3)

        # *****************
        # ******* Heat Flux
        # *****************

        ax = axE = fig1.add_subplot(grid[0, 0])
        ax.plot(
            self.roa[0], self.Qe_sim_neo[-1], "-o", c="b", label="NEO", markersize=3
        )
        ax.plot(
            self.roa[0], self.Qe_sim_turb[-1], "-o", c="m", label="TURB", markersize=3
        )
        ax.plot(self.roa[0], self.Qe_sim[-1], "-o", c="r", label="TOT", markersize=3)
        tot = self.Qe_sim_neo[-1] + self.Qe_sim_turb[-1]
        ax.plot(self.roa[0], tot, "--", c="y", label="sum", markersize=2)
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.set_ylabel("Heat: $Q$ ($MW/m^2$)")
        ax.legend(prop={"size": 6}, loc="best")
        ax.set_title("Electron")
        GRAPHICStools.addDenseAxis(ax)

        for i in range(self.Qi_sim_turb.shape[0]):
            if i == 0:
                ax = ax0 = fig1.add_subplot(grid[0, i + 1], sharex=axE, sharey=axE)
            else:
                ax = fig1.add_subplot(grid[0, i + 1], sharex=axE, sharey=axE)
            ax.plot(
                self.roa[0],
                self.Qi_sim_neo[i, -1],
                "-o",
                c="b",
                label="NEO",
                markersize=3,
            )
            ax.plot(
                self.roa[0],
                self.Qi_sim_turb[i, -1],
                "-o",
                c="m",
                label="TURB",
                markersize=3,
            )
            # ax.plot(self.roa[0],self.Qi_sim[i,-1],'-o',c='r',label='TOT',markersize=3)
            tot = self.Qi_sim_neo[i, -1] + self.Qi_sim_turb[i, -1]
            ax.plot(self.roa[0], tot, "-o", c="r", label="sum", markersize=3)
            ax.set_xlim([0, 1])
            # ax.set_xlabel('$r/a$')
            ax.axhline(y=0, lw=0.5, c="k", ls="--")
            ax.legend(prop={"size": 6}, loc="best")
            ax.set_title(
                f"Ion #{i+1} ({self.profiles.Species[i]['N']}{self.profiles.Species[i]['Z']:.0f},{self.profiles.Species[i]['A']:.0f})"
            )

            GRAPHICStools.addDenseAxis(ax)

        ax = ax = fig1.add_subplot(grid[0, -2], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], self.Qi_sim[-1], "-o", c="r", label="TOT", markersize=3)
        ax.plot(
            self.roa[0],
            self.QiIons_sim_neo_thr[-1],
            "-o",
            c="b",
            label="sum NEO (thr)",
            markersize=3,
        )
        ax.plot(
            self.roa[0],
            self.QiIons_sim_turb_thr[-1],
            "-o",
            c="m",
            label="sum TURB (thr)",
            markersize=3,
        )
        ax.plot(
            self.roa[0],
            self.QiIons_sim_turb_thr[-1] + self.QiIons_sim_neo_thr[-1],
            "--",
            c="y",
            label="sum (thr)",
            markersize=3,
        )
        ax.plot(
            self.roa[0],
            self.QiIons_sim_turb[-1] + self.QiIons_sim_neo[-1],
            "--",
            c="c",
            label="sum (tot)",
            markersize=3,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")
        ax.set_title("Ions")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax = fig1.add_subplot(grid[0, -1], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], self.Qe_sim[-1], "-o", c="b", label="e-", markersize=3)
        ax.plot(self.roa[0], self.Qi_sim[-1], "-o", c="m", label="i+", markersize=3)
        ax.plot(
            self.roa[0],
            self.Qe_sim[-1] + self.Qi_sim[-1],
            "-o",
            c="r",
            label="(e-) + (i+)",
            markersize=3,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")
        ax.set_title("All species")

        GRAPHICStools.addDenseAxis(ax)

        # *********************
        # ******* Particle Flux
        # *********************

        ax = axE = fig1.add_subplot(grid[1, 0])
        ax.plot(
            self.roa[0], self.Ge_sim_neo[-1], "-o", c="b", label="NEO", markersize=3
        )
        ax.plot(
            self.roa[0], self.Ge_sim_turb[-1], "-o", c="m", label="TURB", markersize=3
        )
        ax.plot(self.roa[0], self.Ge_sim[-1], "-o", c="r", label="TOT", markersize=3)
        tot = self.Ge_sim_neo[-1] + self.Ge_sim_turb[-1]
        ax.plot(self.roa[0], tot, "--", c="y", label="sum", markersize=3)
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.set_ylabel("Particle: $\\Gamma$ ($10^{20}/m^2/s$)")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        totT_turb = np.zeros(len(self.roa[0]))
        totT_neo = np.zeros(len(self.roa[0]))
        totT_turb_thr = np.zeros(len(self.roa[0]))
        totT_neo_thr = np.zeros(len(self.roa[0]))

        totT_Z = np.zeros(len(self.roa[0]))

        for i in range(self.Gi_sim_turb.shape[0]):
            if i == 0:
                ax = ax0 = fig1.add_subplot(grid[1, i + 1], sharex=axE, sharey=axE)
            else:
                ax = fig1.add_subplot(grid[1, i + 1], sharex=axE, sharey=axE)
            ax.plot(
                self.roa[0],
                self.Gi_sim_neo[i, -1],
                "-o",
                c="b",
                label="NEO",
                markersize=3,
            )
            ax.plot(
                self.roa[0],
                self.Gi_sim_turb[i, -1],
                "-o",
                c="m",
                label="TURB",
                markersize=3,
            )
            ax.plot(
                self.roa[0], self.Gi_sim[i, -1], "-o", c="r", label="TOT", markersize=3
            )
            tot = self.Gi_sim_neo[i, -1] + self.Gi_sim_turb[i, -1]
            ax.plot(self.roa[0], tot, "--", c="y", label="sum", markersize=3)
            ax.set_xlim([0, 1])
            # ax.set_xlabel('$r/a$')
            ax.axhline(y=0, lw=0.5, c="k", ls="--")
            ax.legend(prop={"size": 6}, loc="best")

            totT_neo += self.Gi_sim_neo[i, -1]
            totT_turb += self.Gi_sim_turb[i, -1]
            if self.profiles.Species[i]["S"] == "therm":
                totT_neo_thr += self.Gi_sim_neo[i, -1]
                totT_turb_thr += self.Gi_sim_turb[i, -1]

            totT_Z += self.Gi_sim_neo[i, -1] + self.Gi_sim_turb[i, -1]

            GRAPHICStools.addDenseAxis(ax)

        ax = fig1.add_subplot(grid[1, -2], sharex=ax0, sharey=ax0)
        ax.plot(
            self.roa[0], totT_neo_thr, "-o", c="b", label="sum NEO (thr)", markersize=3
        )
        ax.plot(
            self.roa[0],
            totT_turb_thr,
            "-o",
            c="m",
            label="sum TURB (thr)",
            markersize=3,
        )
        ax.plot(
            self.roa[0],
            totT_turb_thr + totT_neo_thr,
            "--",
            c="y",
            label="sum (thr)",
            markersize=3,
        )
        ax.plot(
            self.roa[0],
            totT_turb + totT_neo,
            "--",
            c="c",
            label="sum (tot)",
            markersize=3,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        ax = fig1.add_subplot(grid[1, -1], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], self.Ge_sim[-1], "-o", c="b", label="e-", markersize=3)
        ax.plot(self.roa[0], totT_Z, "-o", c="m", label="Z$\\cdot$i+", markersize=3)
        ax.plot(
            self.roa[0],
            self.Ge_sim[-1] - totT_Z,
            "-o",
            c="r",
            label="(e-) - (Z$\\cdot$i+)",
            markersize=3,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        # *********************
        # ******* Momentum Flux
        # *********************

        self.Me_sim_neo = self.roa * 0.0
        self.Me_sim_turb = self.roa * 0.0
        self.Mi_sim_neo = self.Gi_sim_neo * 0.0
        self.Mi_sim_turb = self.Gi_sim_neo * 0.0
        self.EXi_sim = self.Gi_sim_neo * 0.0

        ax = axE = fig1.add_subplot(grid[2, 0])
        ax.plot(
            self.roa[0], self.Me_sim_neo[-1], "-o", c="b", label="NEO", markersize=3
        )
        ax.plot(
            self.roa[0], self.Me_sim_turb[-1], "-o", c="m", label="TURB", markersize=3
        )
        tot = Me = self.Me_sim_neo[-1] + self.Me_sim_turb[-1]
        ax.plot(self.roa[0], tot, "--", c="y", label="sum", markersize=2)
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.set_ylabel("Momentum: $\\Pi$ ($J/m^2$)")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        Mi = np.zeros(len(self.roa[0]))
        for i in range(self.Qi_sim_turb.shape[0]):
            if i == 0:
                ax = ax0 = fig1.add_subplot(grid[2, i + 1], sharex=axE, sharey=axE)
            else:
                ax = fig1.add_subplot(grid[2, i + 1], sharex=axE, sharey=axE)
            ax.plot(
                self.roa[0],
                self.Mi_sim_neo[i, -1],
                "-o",
                c="b",
                label="NEO",
                markersize=3,
            )
            ax.plot(
                self.roa[0],
                self.Mi_sim_turb[i, -1],
                "-o",
                c="m",
                label="TURB",
                markersize=3,
            )
            tot = self.Mi_sim_neo[i, -1] + self.Mi_sim_turb[i, -1]
            Mi += tot
            ax.plot(self.roa[0], tot, "-o", c="r", label="sum", markersize=3)
            ax.set_xlim([0, 1])
            # ax.set_xlabel('$r/a$')
            ax.axhline(y=0, lw=0.5, c="k", ls="--")
            # ax.set_ylabel('$Q_{i}$ ($MW/m^2$)')
            ax.legend(prop={"size": 6}, loc="best")

            GRAPHICStools.addDenseAxis(ax)

        ax = ax = fig1.add_subplot(grid[2, -2], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], self.Mt_sim[-1], "-o", c="r", label="TOT", markersize=3)
        ax.plot(
            self.roa[0], self.Mt_sim_neo[-1], "-o", c="b", label="sum NEO", markersize=3
        )
        ax.plot(
            self.roa[0],
            self.Mt_sim_turb[-1],
            "-o",
            c="m",
            label="sum TURB",
            markersize=3,
        )
        ax.plot(self.roa[0], Mi, "--", c="y", label="sum", markersize=3)
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        ax = fig1.add_subplot(grid[2, -1], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], Me, "-o", c="b", label="e-", markersize=3)
        ax.plot(self.roa[0], self.Mt_sim[-1], "-o", c="m", label="i+", markersize=3)
        ax.plot(
            self.roa[0],
            Me + self.Mt_sim[-1],
            "-o",
            c="r",
            label="(e-) + (i+)",
            markersize=3,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$r/a$')
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        # ***********************
        # ******* Energy Exchange
        # ***********************

        ax = axE = fig1.add_subplot(grid[3, 0])
        ax.plot(self.roa[0], self.EXe_sim[-1], "-o", c="m", markersize=3)
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.set_ylabel("Exchange: $S$ ($MW/m^3$)")

        GRAPHICStools.addDenseAxis(ax)

        Si = np.zeros(len(self.roa[0]))
        for i in range(self.Qi_sim_turb.shape[0]):
            if i == 0:
                ax = ax0 = fig1.add_subplot(grid[3, i + 1], sharex=axE, sharey=axE)
            else:
                ax = fig1.add_subplot(grid[3, i + 1], sharex=axE, sharey=axE)
            ax.plot(self.roa[0], self.EXi_sim[i, -1], "-o", c="m", markersize=3)
            ax.set_xlabel("$r/a$")
            ax.set_xlim([0, 1])
            ax.axhline(y=0, lw=0.5, c="k", ls="--")
            ax.legend(prop={"size": 6}, loc="best")

            GRAPHICStools.addDenseAxis(ax)

            Si += self.EXi_sim[i, -1]

        ax = ax = fig1.add_subplot(grid[3, -2], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], Si, "-o", c="m", label="sum TURB", markersize=3)
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        ax = fig1.add_subplot(grid[3, -1], sharex=ax0, sharey=ax0)
        ax.plot(self.roa[0], self.EXe_sim[-1], "-o", c="b", label="e-", markersize=3)
        ax.plot(self.roa[0], Si, "-o", c="m", label="i+", markersize=3)
        ax.plot(
            self.roa[0],
            self.EXe_sim[-1] + Si,
            "-o",
            c="r",
            label="(e-) + (i+)",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.axhline(y=0, lw=0.5, c="k", ls="--")
        ax.legend(prop={"size": 6}, loc="best")

        GRAPHICStools.addDenseAxis(ax)

        # ------------------------------------------------------------------------------
        # Powers
        # ------------------------------------------------------------------------------

        fig1 = self.fn.add_figure(tab_color=fn_color, label=prelabel + "Powers" + label)

        grid = plt.GridSpec(2, 5, hspace=0.2, wspace=0.4)
        ax00 = fig1.add_subplot(grid[0, 0])
        ax10 = fig1.add_subplot(grid[1, 0])

        ax01 = fig1.add_subplot(grid[0, 1])
        ax11 = fig1.add_subplot(grid[1, 1])

        ax02 = fig1.add_subplot(grid[0, 2])
        ax12 = fig1.add_subplot(grid[1, 2])

        ax03 = fig1.add_subplot(grid[0, 3])
        ax13 = fig1.add_subplot(grid[1, 3])

        ax04 = fig1.add_subplot(grid[0, 4])
        ax14 = fig1.add_subplot(grid[1, 4])

        ax = ax00
        ax.plot(
            self.roa[0],
            self.Qe_tarMW_fus[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qe_tarMW_fus[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{fus,e}$ (MW)")
        ax.legend(prop={"size": 6}, loc="best")

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qe_fus_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)

        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qe_fus_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax01
        ax.plot(
            self.roa[0],
            self.Qe_tarMW_aux[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qe_tarMW_aux[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{aux,e}$ (MW)")
        # ax.set_title('Fusion Power')

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qe_aux_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)

        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qe_aux_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax02
        sign = -1
        ax.plot(
            self.roa[0],
            sign * self.Qe_tarMW_brem[0],
            "-o",
            c="blue",
            label="Brem",
            markersize=1,
            lw=0.5,
        )
        ax.plot(
            self.roa[-1],
            sign * self.Qe_tarMW_brem[-1],
            "-o",
            c="red",
            markersize=1,
            lw=0.5,
        )
        ax.plot(
            self.roa[0],
            sign * self.Qe_tarMW_sync[0],
            "-.o",
            c="blue",
            label="Sync",
            markersize=1,
            lw=0.5,
        )
        ax.plot(
            self.roa[-1],
            sign * self.Qe_tarMW_sync[-1],
            "-.o",
            c="red",
            markersize=1,
            lw=0.5,
        )
        ax.plot(
            self.roa[0],
            sign * self.Qe_tarMW_line[0],
            "--o",
            c="blue",
            label="Line",
            markersize=1,
            lw=0.5,
        )
        ax.plot(
            self.roa[-1],
            sign * self.Qe_tarMW_line[-1],
            "--o",
            c="red",
            markersize=1,
            lw=0.5,
        )
        Pa = self.Qe_tarMW_rad
        ax.plot(self.roa[0], sign * Pa[0], "-o", c="blue", markersize=3, label="Total")
        ax.plot(self.roa[-1], sign * Pa[-1], "-o", c="red", markersize=3)
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{rad}$ (MW)")

        # ax.set_title('Fusion Power')

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = -1 * sign * self.profiles.derived["qe_rad_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)

        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qe_rad_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax03
        ax.plot(
            self.roa[0],
            self.Qe_tarMW_exch[0],
            "--o",
            c="blue",
            label="exch",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qe_tarMW_exch[-1],
            "--o",
            c="red",
            markersize=3,
        )
        ax.plot(
            self.roa[0],
            self.Qe_tarMW_expwd[0],
            "-.o",
            c="blue",
            label="expwd",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qe_tarMW_expwd[-1],
            "-.o",
            c="red",
            markersize=3,
        )

        ax.plot(
            self.roa[0],
            self.Qe_tarMW_exch[0] + self.Qe_tarMW_expwd[0],
            "-o",
            c="blue",
            label="total",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qe_tarMW_exch[-1] + self.Qe_tarMW_expwd[-1],
            "-o",
            c="red",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{exch,e}$ (MW)")

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = -self.profiles.derived["qe_exc_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = -self.profiles_final.derived["qe_exc_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        ax.legend(prop={"size": 6})
        # ax.set_title('Fusion Power')

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax04
        ax.plot(
            self.roa[0],
            self.Qe_tarMW_tot[0],
            "-o",
            c="blue",
            label="Total",
            markersize=3,
        )
        ax.plot(self.roa[-1], self.Qe_tarMW_tot[-1], "-o", c="red", markersize=3)

        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{tot,e}$ (MW)")

        P = (
            self.Qe_tarMW_fus
            + self.Qe_tarMW_aux
            + (self.Qe_tarMW_brem + self.Qe_tarMW_sync + self.Qe_tarMW_line)
            + (self.Qe_tarMW_exch + self.Qe_tarMW_expwd)
        )
        ax.plot(self.roa[0], P[0], "--", c="blue", markersize=3, label="sum")
        ax.plot(self.roa[-1], P[-1], "--", c="red", markersize=3)

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qe_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qe_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax10
        ax.plot(
            self.roa[0],
            self.Qi_tarMW_fus[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qi_tarMW_fus[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{fus,i}$ (MW)")

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qi_fus_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qi_fus_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax11
        ax.plot(
            self.roa[0],
            self.Qi_tarMW_aux[0],
            "-o",
            c="blue",
            label=f"#{0}",
            markersize=3,
        )
        ax.plot(
            self.roa[-1],
            self.Qi_tarMW_aux[-1],
            "-o",
            c="red",
            label=f"#{self.iterations[-1]}",
            markersize=3,
        )
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{aux,i}$ (MW)")

        ax.plot(
            self.roa[-1],
            self.Qi_tarMW_aux[-1] + self.Qe_tarMW_aux[-1],
            "--o",
            lw=0.5,
            c="c",
            markersize=1,
            label=f"#{self.iterations[-1]} i+e",
        )

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qi_aux_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qi_aux_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

            ax.plot(
                roa,
                self.profiles_final.derived["qi_aux_MW"]
                + self.profiles_final.derived["qe_aux_MW"],
                "-.",
                c="y",
                lw=0.5,
                label="profiles_new (miller) i+e",
            )

        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax12
        Pa = self.Qi_tarMW_fus + self.Qe_tarMW_fus
        ax.plot(self.roa[0], 5 * Pa[0], "-o", c="blue", markersize=3)
        ax.plot(self.roa[-1], 5 * Pa[-1], "-o", c="red", markersize=3)
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{out} = 5*(P_{fus,e}$+$P_{fus,i})$ (MW)")

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = (
                self.profiles.derived["qe_fus_MW"]
                + self.profiles.derived["qi_fus_MW"]
            )
            ax.plot(roa, 5 * P, "--", c="green", label="profiles (miller)", lw=1.0)
        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles_final.profiles["rmin(m)"][-1]
            )
            P = (
                self.profiles_final.derived["qe_fus_MW"]
                + self.profiles_final.derived["qi_fus_MW"]
            )
            ax.plot(roa, 5 * P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        ax.legend(prop={"size": 6})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        ax = ax13
        ax.plot(
            self.roa[0],
            self.Qi_tarMW_exch[0],
            "--o",
            c="blue",
            label="exch",
            markersize=3,
        )
        ax.plot(self.roa[-1], self.Qi_tarMW_exch[-1], "--o", c="red", markersize=3)
        ax.plot(
            self.roa[0],
            self.Qi_tarMW_expwd[0],
            "-.o",
            c="blue",
            label="expwd",
            markersize=3,
        )
        ax.plot(self.roa[-1], self.Qi_tarMW_expwd[-1], "-.o", c="red", markersize=3)
        Pa = self.Qi_tarMW_exch + self.Qi_tarMW_expwd
        ax.plot(self.roa[0], Pa[0], "-o", c="blue", label="total", markersize=3)
        ax.plot(self.roa[-1], Pa[-1], "-o", c="red", markersize=3)
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P_{exch,i}$ (MW)")

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qe_exc_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)

        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qe_exc_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax14
        Pa = self.Qi_tarMW_tot
        ax.plot(self.roa[0], Pa[0], "-o", c="blue", label="Total", markersize=3)
        ax.plot(self.roa[-1], Pa[-1], "-o", c="red", markersize=3)
        ax.set_ylabel("$P_{tot,i}$ (MW)")

        P = (
            self.Qi_tarMW_fus
            + self.Qi_tarMW_aux
            + self.Qi_tarMW_exch
            + self.Qi_tarMW_expwd
        )
        ax.plot(self.roa[0], P[0], "--", c="blue", markersize=3, label="sum")
        ax.plot(self.roa[-1], P[-1], "--", c="red", markersize=3)

        if self.profiles is not None:
            roa = (
                self.profiles.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles.derived["qi_MW"]
            ax.plot(roa, P, "--", c="green", label="profiles (miller)", lw=1.0)

        if self.profiles_final is not None:
            roa = (
                self.profiles_final.profiles["rmin(m)"]
                / self.profiles.profiles["rmin(m)"][-1]
            )
            P = self.profiles_final.derived["qi_MW"]
            ax.plot(roa, P, "--", c="k", label="profiles_new (miller)", lw=1.0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # ------------------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------------------

        fig1 = self.fn.add_figure(
            tab_color=fn_color, label=prelabel + "Perform." + label
        )
        grid = plt.GridSpec(2, 2, hspace=0.45, wspace=0.3)

        # Fusion Gain
        ax = fig1.add_subplot(grid[0, 0])
        ax.plot(
            self.iterations,
            self.Q,
            "-s",
            color="b",
            markersize=3,
            label="(approx.) Pfus from TGYRO, Paux from TGYRO",
        )
        ax.plot(
            self.iterations,
            self.Q_better,
            "-^",
            color="r",
            markersize=2,
            label="(approx.) Pfus from TGYRO, Paux from last PROFILES",
        )

        ity, Q = [], []
        if self.profiles is not None:
            ity.append(self.iterations[0])
            Q.append(self.profiles.derived["Q"])

            if self.profiles_final is not None:
                ity.append(np.max([self.iterations[-1], 0.1]))
                Q.append(self.Q_best)

        ax.plot(
            ity,
            Q,
            "*--",
            lw=1.0,
            c="green",
            markersize=10,
            label="(correct) Pfus from PROFILES, Paux from PROFILES",
        )

        Q_mod = Q[-1] * (self.Q[-1] / self.Q_better[-1])
        ax.plot(
            [ity[-1]],
            [Q_mod],
            "v--",
            lw=0.5,
            c="cyan",
            markersize=5,
            label="(checker) Pfus from last PROFILES, Paux from last TGYRO",
        )

        # ax.plot(self.iterations,self.Q_better*(Q/self.Q_better[0]),'--o',lw=0.2,color='b',markersize=1,label='Corrrected Q')

        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_title("Fusion Gain Q")
        ax.set_ylabel("$Q_{plasma}$")
        # ax.set_ylim(bottom=0)
        ax.legend(prop={"size": 7})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # Confinement time
        ax = fig1.add_subplot(grid[0, 1])
        ax.plot(
            self.iterations,
            self.tauE,
            "-s",
            color="b",
            markersize=3,
            label="(approx.) W from TGYRO, Pheat from TGYRO",
        )

        ity, tau = [], []
        if self.profiles is not None:
            ity.append(self.iterations[0])
            tau.append(self.profiles.derived["tauE"])

            if self.profiles_final is not None:
                ity.append(np.max([self.iterations[-1], 0.1]))
                tau.append(self.profiles_final.derived["tauE"])

        ax.plot(
            ity,
            tau,
            "*--",
            lw=1.0,
            c="g",
            markersize=10,
            label="(correct) W from PROFILES, Pheat from PROFILES",
        )

        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_title("Energy Confinement Time")
        ax.set_ylabel("$\\tau_E$ (s)")
        ax.legend(prop={"size": 7})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # H-factors
        ax = fig1.add_subplot(grid[1, 0])
        # ax.plot(self.iterations,self.h98y2,'-s',color='b',markersize=3,label='TGYRO H98y2 (*approx*)')
        # ax.plot(self.iterations,self.h89p,'-s',color='r',markersize=3,label='TGYRO H89p (*approx*)')

        ity, h98, h89 = [], [], []
        if self.profiles is not None:
            ity.append(self.iterations[0])
            h98.append(self.profiles.derived["H98"])
            h89.append(self.profiles.derived["H89"])

            if self.profiles_final is not None:
                ity.append(np.max([self.iterations[-1], 0.1]))
                h98.append(self.profiles_final.derived["H98"])
                h89.append(self.profiles_final.derived["H89"])

        ax.plot(
            ity,
            h98,
            "*--",
            lw=0.5,
            c="b",
            markersize=10,
            label="(correct) H98y2 from PROFILES",
        )
        ax.plot(
            ity,
            h89,
            "*--",
            lw=0.5,
            c="r",
            markersize=10,
            label="(correct) H89p  from PROFILES",
        )

        ax.axhline(y=1.0, c="k", lw=0.5, ls="--")

        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_ylabel("H")
        ax.legend(prop={"size": 7})

        ax.set_title("H-factors")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        # Density
        ax = fig1.add_subplot(grid[1, 1])
        ax.plot(
            self.iterations,
            self.ne_avol,
            "-s",
            color="b",
            markersize=5,
            label="(approx.) from TGYRO",
        )

        ity, fG, ne_avol = [], [], []
        if self.profiles is not None:
            ity.append(self.iterations[0])
            fG.append(self.profiles.derived["fG"])
            ne_avol.append(self.profiles.derived["ne_vol20"])

            if self.profiles_final is not None:
                ity.append(np.max([self.iterations[-1], 0.1]))
                fG.append(self.profiles_final.derived["fG"])
                ne_avol.append(self.profiles_final.derived["ne_vol20"])

        ax.plot(
            ity,
            ne_avol,
            "*--",
            lw=0.5,
            c="g",
            markersize=10,
            label="(correct) from PROFILES",
        )
        # ax.plot(ity,fG,'*--',lw=0.5,c='b',markersize=10,label='PROFILES fG')

        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        # ax.set_ylabel(r'Greenwald Fraction'); ax.set_ylim([0,1.2])
        ax.set_title("Volume average density")
        ax.set_ylabel("$<n_e>$ ($10^{20}m^{-3}$)")
        # ax.set_ylim(bottom=0)
        ax.legend(prop={"size": 7})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        if self.profiles_final is not None:
            self.profiles_final.printInfo(label=f" {prelabel}Final PROFILES{label} ")

            # ------------------------------------------------------------------------------
            # Final
            # ------------------------------------------------------------------------------

            fig1 = self.fn.add_figure(
                tab_color=fn_color, label=prelabel + "Flows" + label
            )
            self.plotBalance(fig=fig1)

    """
		Note that input.gacode and TGYRO may differ in Pfus, Prad, Exch, etc because TGYRO has internal calculations.
		I believe those are passed to the output of tgyro, even though the tgyro option is static.
		If it has run only with dynamic exchange, the differences in the exch,i and tot,i plots must be the same.
	"""

    def plotBalance(self, fig=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        axs = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[0, 1]),
        ]
        axs.append(fig.add_subplot(grid[0, 2]))
        axs.append(fig.add_subplot(grid[1, 1]))
        axs.append(fig.add_subplot(grid[1, 2]))

        self.profiles_final.plotBalance(
            axs=axs, limits=[self.roa[-1, 1], self.roa[-1, -1]]
        )

        axs[2].plot(
            self.roa[-1],
            -self.Qe_simMW[-1],
            "-.s",
            c="g",
            lw=2,
            markersize=5,
            label="$P_{e,tr}$",
        )
        axs[2].plot(
            self.roa[-1],
            -self.Ce_simMW[-1],
            "-.s",
            c="k",
            lw=1,
            markersize=2,
            label="($P_{e,conv,tr}$)",
        )
        axs[3].plot(
            self.roa[-1],
            -self.Qi_simMW[-1],
            "-.s",
            c="g",
            lw=2,
            markersize=5,
            label="$P_{i,tr}$",
        )
        axs[4].plot(
            self.roa[-1],
            self.Ge_simABS[-1],
            "-.s",
            c="g",
            lw=2,
            markersize=5,
            label="$\\Gamma_{e,tr}$",
        )
        axs[5].plot(
            self.roa[-1],
            -(self.Qi_simMW[-1] + self.Qe_simMW[-1]),
            "-.s",
            c="g",
            lw=2,
            markersize=5,
            label="$P_{tr}$",
        )

    def plotConvergence(self, fig1=None):
        grid = plt.GridSpec(2, 2, hspace=0.45, wspace=0.3)
        ax00 = fig1.add_subplot(grid[0, 0])
        ax10 = fig1.add_subplot(grid[1, 0])
        ax01 = fig1.add_subplot(grid[0, 1])
        ax11 = fig1.add_subplot(grid[1, 1])

        ax = ax00
        ax.plot(self.iterations, self.residual, "-s", color="b", markersize=5)
        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_ylabel("Residual (GB)")
        ax.set_yscale("log")
        whichticks = ax.get_xticks()
        _ = GRAPHICStools.addXaxis(
            ax,
            self.iterations,
            self.calls_solver,
            label="Calls to transport solver",
            whichticks=whichticks,
        )

        GRAPHICStools.addDenseAxis(ax)
        # GRAPHICStools.autoscale_y(ax)

        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=False, extraPad=0, loc="center left", size=6
        )
        # GRAPHICStools.addLegendApart(ax2,ratio=0.9,withleg=False,extraPad=0,loc='center left',size=6)

        ax = ax10
        colsE = (
            GRAPHICStools.listColors()
        )  # GRAPHICStools.colorTableFade(self.radii-1,startcolor='b',endcolor='b',alphalims=[0.3,1.0])
        colsI = (
            GRAPHICStools.listColors()
        )  # GRAPHICStools.colorTableFade(self.radii-1,startcolor='r',endcolor='r',alphalims=[0.3,1.0])
        for i in range(self.radii - 1):
            label = f"r/a={self.roa[0, i + 1]:.4f}"

            if self.Te_predicted:
                if i == 0:
                    label = f"$Qe$, r/a={self.roa[0, i + 1]:.4f}"
                ax.plot(
                    self.iterations,
                    self.QeGB_res[:, i + 1],
                    "-o",
                    c=colsE[i],
                    label=label,
                    markersize=1,
                    lw=0.5,
                )
                label = ""
            if self.Ti_predicted:
                if i == 0:
                    label = f"$Qi$, r/a={self.roa[0, i + 1]:.4f}"
                ax.plot(
                    self.iterations,
                    self.QiGB_res[:, i + 1],
                    "--o",
                    c=colsI[i],
                    label=label,
                    markersize=1,
                    lw=0.5,
                )
                label = ""

            if self.ne_predicted:
                if i == 0:
                    label = f"$\\Gamma_e$, r/a={self.roa[0, i + 1]:.4f}"
                ax.plot(
                    self.iterations,
                    self.GeGB_res[:, i + 1],
                    "-.o",
                    c=colsE[i],
                    label=label,
                    markersize=1,
                    lw=0.5,
                )
                label = ""

        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_ylabel("Individual Residuals (GB)")
        ax.set_yscale("log")
        # ax.legend(loc='best',prop={'size':5})
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, extraPad=0, loc="center left", size=6
        )
        # ax2 = GRAPHICStools.addXaxis(ax,self.iterations,self.calls_solver,label='Calls to transport solver',whichticks=whichticks)

        GRAPHICStools.addDenseAxis(ax)
        # GRAPHICStools.autoscale_y(ax)

        ax = ax01
        ax.plot(
            self.iterations, self.residual_manual_real, "-s", color="b", markersize=5
        )
        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_ylabel("Residual (real)")
        ax.set_yscale("log")

        GRAPHICStools.addDenseAxis(ax)
        # GRAPHICStools.autoscale_y(ax)

        _ = GRAPHICStools.addXaxis(
            ax,
            self.iterations,
            self.calls_solver,
            label="Calls to transport solver",
            whichticks=whichticks,
        )

        ax = ax11
        for i in range(self.radii - 1):
            if self.Te_predicted:
                ax.plot(
                    self.iterations,
                    self.Qe_res[:, i + 1],
                    "-o",
                    c=colsE[i],
                    markersize=2,
                    lw=0.5,
                )
            if self.Ti_predicted:
                ax.plot(
                    self.iterations,
                    self.Qi_res[:, i + 1],
                    "--o",
                    c=colsI[i],
                    markersize=2,
                    lw=0.5,
                )
            if self.ne_predicted:
                ax.plot(
                    self.iterations,
                    self.Ge_res[:, i + 1],
                    "-.o",
                    c=colsE[i],
                    markersize=1,
                    lw=0.5,
                )
        ax.set_xlabel("Iterations")
        ax.set_xlim(left=0)
        ax.set_ylabel("Individual Residuals (real)")
        ax.set_yscale("log")

        # ax2 = GRAPHICStools.addXaxis(ax,self.iterations,self.calls_solver,label='Calls to transport solver',whichticks=whichticks)

        GRAPHICStools.addDenseAxis(ax)
        # GRAPHICStools.autoscale_y(ax)


def plotAll(TGYROoutputs, labels=None, fn=None):
    if fn is None:
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        fn = FigureNotebook("TGYRO Output Notebook", geometry="1800x900")

    if labels is None:
        labels = [f" {i}" for i in np.arange(1, len(TGYROoutputs) + 1, 1)]

    for i, TGYROoutput in enumerate(TGYROoutputs):
        TGYROoutput.plot(fn=fn, label=labels[i])

    return fn


class TGYROinput:
    def __init__(self, input_profiles, file=None, onlyThermal=False, limitSpecies=100):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None

        if self.file is not None and self.file.exists():
            with open(self.file, "r") as f:
                lines = f.readlines()
            self.file_txt = "".join(lines)
        else:
            self.file_txt = ""

        self.input_dict = GACODErun.buildDictFromInput(self.file_txt)

        # Species
        self.species = input_profiles.Species

        self.onlyThermal = onlyThermal

        self.limitSpecies = limitSpecies

        # Get number of ions
        _, spec = GACODEdefaults.addTGYROspecies(
            self.species, onlyThermal=self.onlyThermal, limitSpecies=self.limitSpecies
        )
        self.loc_n_ion = spec["LOC_N_ION"]

    def writeCurrentStatus(self, file=None):
        print("\t- Writting TGYRO input file")

        if file is None:
            file = self.file

        _, spec = GACODEdefaults.addTGYROspecies(
            self.species, onlyThermal=self.onlyThermal, limitSpecies=self.limitSpecies
        )

        with open(file, "w") as f:
            f.write("#---------------------------------------\n")
            f.write("# file modified by MITIM (Rodriguez-Fernandez, 2020)\n")
            f.write("#---------------------------------------\n\n\n")

            f.write("# Control parameters:\n\n")
            for ikey in self.input_dict:
                if self.input_dict[ikey] is not None:
                    f.write(f"{ikey} = {self.input_dict[ikey]}\n")
                else:
                    f.write(f"{ikey}\n")

            f.write("\n\n# Species:\n")
            for ikey in spec:
                f.write(f"{ikey} = {spec[ikey]}\n")

        print(f"\t\t~ File {IOtools.clipstr(file)} written")


def print_options(physics_options, solver_options):
    if len(physics_options) > 0:
        print("\t- TGYRO physics options:")
        for i in physics_options:
            print(f"\t\t{i:25}: {physics_options[i]}")

    if len(solver_options) > 0:
        print("\t- TGYRO solver options:")
        for i in solver_options:
            print(f"\t\t{i:25}: {solver_options[i]}")


def modifyInputToTGYRO(
    inputTGYRO,
    vectorRange,
    iterations,
    TGYRO_physics_options,
    solver_options,
    Tepred=1,
    Tipred=1,
    nepred=1,
    TGYROcontinue=False,
    special_radii=None,
):
    (
        TGYROoptions,
        TGYRO_physics_options_new,
        solver_options_new,
    ) = GACODEdefaults.addTGYROcontrol(
        howmany=vectorRange[2],
        fromRho=vectorRange[0],
        ToRho=vectorRange[1],
        num_it=iterations,
        Tepred=Tepred,
        Tipred=Tipred,
        nepred=nepred,
        physics_options=TGYRO_physics_options,
        solver_options=solver_options,
        cold_start=TGYROcontinue,
        special_radii=special_radii,
    )

    # Print after adding the defaults, to see what's really running
    print_options(TGYRO_physics_options_new, solver_options_new)

    # ~~~~~~~~~~ Change with presets
    for ikey in TGYROoptions:
        inputTGYRO.input_dict[ikey] = TGYROoptions[ikey]

    return inputTGYRO


def produceInputs_TGYROworkflow(
    time,
    finalFolder,
    tmpFolder,
    LocationCDF,
    avTime=0.0,
    forceEntireWorkflow=False,
    sendState=True,
    includeGEQ=True,
    BtIp_dirs=[0, 0],
    gridsTRXPL=[151, 101, 101],
):
    folderWork = IOtools.expandPath(tmpFolder)
    try:
        nameRunid = LocationCDF.stem
    except:
        # This is in case I have given a None to the location of the cdf because I just want
        # to prep() from .cdf and .geq directly
        LocationCDF = tmpFolder
        nameRunid = "10001"

    # ---------------------------------------------------------------------------------------------------------------
    # cold_starting options
    # ---------------------------------------------------------------------------------------------------------------

    files_to_look = ["input.gacode"]

    print(f"\t- Info to be stored in ..{finalFolder}")

    file_to_look = files_to_look[0]
    print("\t- Testing... do TGYRO files already exist?")
    lookfile = finalFolder / f"{file_to_look}"
    if lookfile.exists():
        ProfilesGenerated, StateGenerated = True, True
        print(f"\t\t+++++++ {IOtools.clipstr(lookfile)} already generated")
    else:
        print(
            f"\t\t+++++++ {IOtools.clipstr(lookfile)} file not found"
        )
        ProfilesGenerated = False
        if (finalFolder / "10001.cdf").exists():
            StateGenerated = True
            print(
                "\t\t+++++++ .cdf plasma-state file already generated, I need to run profiles_gen"
            )
            # Copying them to tmp folder in case there are not there and cold_starting from .cdf and .geq in final folder
            shutil.copy2(finalFolder / "10001.cdf", folderWork)
            shutil.copy2(finalFolder / "10001.geq", folderWork)
        else:
            print(f"\t\t+++++++ {finalFolder / '10001.cdf'} file not found")
            StateGenerated = False
            print(
                "\t\t+++++++ Workflow needs to be cold_started completely, no files found"
            )

    if not ProfilesGenerated or forceEntireWorkflow:
        FullWorkflow = True
    else:
        FullWorkflow = False

    # ---------------------------------------------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------------------------------------------

    if FullWorkflow:
        print("> Preparing TGYRO run...")
        LocationNML, dummy = GACODErun.findNamelist(
            LocationCDF, folderWork=folderWork, nameRunid=nameRunid
        )

        if dummy:
            print(
                "\t- TRXPL will use default directions of Bt and Ip because TR.DAT was not found",
                typeMsg="w",
            )

        GACODErun.prepareTGYRO(
            LocationCDF,
            LocationNML,
            time,
            avTime=avTime,
            BtIp_dirs=BtIp_dirs,
            folderWork=folderWork,
            StateGenerated=StateGenerated,
            nameRunid=nameRunid,
            sendState=sendState,
            gridsTRXPL=gridsTRXPL,
            includeGEQ=includeGEQ,
        )

        # ---------------------------------------------------------------------------------------------------------------
        # Pass files
        # ---------------------------------------------------------------------------------------------------------------

        shutil.copy2(tmpFolder / "10001.cdf", finalFolder)
        shutil.copy2(tmpFolder / "10001.geq", finalFolder)
        for file_to_look in files_to_look:
            for item in tmpFolder.glob(f"{file_to_look}*"):
                shutil.copy2(item, finalFolder)

    else:
        print(
            "\t\t- TGYRO run already prepared, not entering in preparation (TRXPL+PROFILES_GEN) routines",
            typeMsg="i",
        )
