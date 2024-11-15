import copy
import pickle
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools import TGYROtools, PROFILEStools
from mitim_tools.misc_tools import (
    IOtools,
    GRAPHICStools,
    PLASMAtools,
    GUItools,
)
from mitim_tools.gacode_tools.utils import (
    NORMtools,
    GACODEinterpret,
    GACODEdefaults,
    GACODEplotting,
    GACODErun,
)
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

mi_D = 2.01355


class TGLF:
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
        cdf=None,  # Option1: Start from CDF file (TRANSP) - Path to file
        time=100.0,  # 		   Time to extract CDF file
        avTime=0.0,  # 		   Averaging window to extract CDF file
        alreadyRun=None,  # Option2: Do more stuff with a class that has already been created and store
    ):
        """
        TGLF class that manages the run and the results.

        The philosophy of this is that a single 'tglf' class will handle the tglf simulation and results
        at one time slice but with possibility of several radii at once.

        It can also handle different TGLF settings, running them one by one, storing results in folders and then
        grabbing them.

        Scans can also be run. At several radii at once if wanted.

        *Note*
        The 'run' command does not require label. When performing a 'read', the results extracted from
        the specified folder will be saved with the label indicated in 'read', in the "results" or "scans"
        dictionaries. Plotting then can happen with more than one label of the same category.

        *Note*
        The 'run' command uses input.tglf from the specified folder, but one can change the TGLFsettings presets,
        extraOptions and multipliers. The modified inputs is not rewritten in the actual folder, it is only written
        in the tmp folder on which the simulation takes place.

        *Note*
        After a 'prep' command, the class can be detached from the file system, as it stores the input tglf file
        to run later with different options. It also stores the Normalizations, since the runs are expected
        to only change dimensionless parameteres.

        **************************************
        ***** Example use for standalone *****
        **************************************

            # Initialize class, by specifying where the inputs to TGLF come from (TRANSP cdf)
            tglf = TGLF(cdf='~/testTGLF/12345B12.CDF',time=1.45,avTime=0.1,rhos=[0.4,0.6])

            # Prepare TGLF (this will create input.tglf in the specified folder)
            cdf = tglf.prep('~/testTGLF/')

            # Run standalone TGLF (this will find the input.tglf in the previous folder,
            # and then copy to this specify TGLF run, and run it there)
            tglf.run(subFolderTGLF='tglf1/',TGLFsettings=1,extraOptions={'NS':3})

            # Read results
            tglf.read(label='run1',folder='~/testTGLF/tglf1/')

            # Plot
            plt.ion();	tglf.plot(labels=['run1'])

        *********************************
        ***** Example use for scans *****
        *********************************

            # Initialize class, by specifying where the inputs to TGLF come from (TRANSP cdf)
            tglf = TGLF(cdf='~/testTGLF/12345B12.CDF',time=1.45,avTime=0.1,rhos=[0.4,0.6])

            # Prepare TGLF (this will create input.tglf in the specified folder)
            cdf = tglf.prep('~/testTGLF/')

            # Run
            tglf.runScan('scan1/',TGLFsettings=1,varUpDown=np.linspace(0.5,2.0,20),variable='RLTS_2')

            # Read scan
            tglf.readScan(label='scan1',variable='RLTS_2')

            # Plot
            plt.ion(); tglf.plotScan(labels=['scan1'],variableLabel='RLTS_2')

        ****************************
        ***** Special analysis *****
        ****************************

            Following the prep phase, we can run "runAnalysis()" and select among the different options:
                - Chi_inc
                - D and V for trace impurity
            Then, plotAnalysis() with the right option for different labels too

        ****************************
        ***** Do more stuff with a class that has already been created and store
        ****************************

            tglf = TGLF(alreadyRun=previousClass)
            tglf.FolderGACODE = '~/testTGLF/'

            ** Modify the class as wish, and do run,read, etc **
            ** Because normalizations are stored in the prep phase, that's all ready **
        """
        print(
            "\n-----------------------------------------------------------------------------------------"
        )
        print("\t\t\t TGLF class module")
        print(
            "-----------------------------------------------------------------------------------------\n"
        )

        if alreadyRun is not None:
            # For the case in which I have run TGLF somewhere else, not using to plot and modify the class
            self.__class__ = alreadyRun.__class__
            self.__dict__ = alreadyRun.__dict__
            print("* Readying previously-run TGLF class", typeMsg="i")
        else:
            self.ResultsFiles = [
                "out.tglf.run",
                "out.tglf.gbflux",
                "out.tglf.eigenvalue_spectrum",
                "out.tglf.sum_flux_spectrum",
                "out.tglf.ky_spectrum",
                "out.tglf.temperature_spectrum",
                "out.tglf.density_spectrum",
                "out.tglf.intensity_spectrum",
                "out.tglf.nete_crossphase_spectrum",
                "out.tglf.nsts_crossphase_spectrum",
                "out.tglf.width_spectrum",
                "out.tglf.version",
                "out.tglf.scalar_saturation_parameters",
                "out.tglf.spectral_shift_spectrum",
                "out.tglf.ave_p0_spectrum",
                "out.tglf.field_spectrum",
                "out.tglf.QL_flux_spectrum",
                "input.tglf.gen",
            ]

            self.ResultsFiles_WF = [
                "out.tglf.run",
                "out.tglf.wavefunction",
            ]

            self.LocationCDF = cdf
            if self.LocationCDF is not None:
                _, self.nameRunid = IOtools.getLocInfo(self.LocationCDF)
            else:
                self.nameRunid = "0"
            self.time, self.avTime = time, avTime
            self.rhos = np.array(rhos)

            (
                self.results,
                self.scans,
                self.tgyro,
                self.ky_single,
            ) = ({}, {}, None, None)

            self.NormalizationSets = {
                "TRANSP": None,
                "PROFILES": None,
                "TGYRO": None,
                "EXP": None,
                "input_gacode": None,
                "SELECTED": None,
            }

    def prepare_for_save_TGLF(self):
        """
        This is a function that will be called when saving the class as pickle.
        It will delete things that are not easily pickled.
        """

        if "fn" in self.__dict__:
            del self.fn  # otherwise it cannot deepcopy

        tglf_copy = copy.deepcopy(self)

        del tglf_copy.convolution_fun_fluct
        for label in tglf_copy.results:
            if "convolution_fun_fluct" in tglf_copy.results[label]:
                tglf_copy.results[label]["convolution_fun_fluct"] = None

        return tglf_copy

    def save_pkl(self, file):
        print(f"> Saving tglf class as pickle file: {IOtools.clipstr(file)}")

        # Prepare
        tglf_copy = self.prepare_for_save_TGLF()

        # Write
        with open(file, "wb") as handle:
            pickle.dump(tglf_copy, handle, protocol=4)

    def prep(
        self,
        FolderGACODE,  # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,  # If True, do not use what it potentially inside the folder, run again
        onlyThermal_TGYRO=False,  # Ignore fast particles in TGYRO
        recalculatePTOT=True, # Recalculate PTOT in TGYRO
        cdf_open=None,  # Grab normalizations from CDF file that is open as transp_output class
        inputgacode=None,  # *NOTE BELOW*
        specificInputs=None,  # *NOTE BELOW*
        tgyro_results=None,  # *NOTE BELOW*
        forceIfcold_start=False,  # Extra flag
        ):
        """
        * Note on inputgacode, specificInputs and tgyro_results:
                If I don't want to prepare, I can provide inputgacode and specificInputs, but I have to make sure they are consistent with one another!
                Optionally, I can give tgyro_results for further info in such a case
        """

        print("> Preparation of TGLF run")

        # PROFILES class.

        profiles = (
            PROFILEStools.PROFILES_GACODE(inputgacode)
            if inputgacode is not None
            else None
        )

        # TGYRO class. It checks existence and creates input.profiles/input.gacode

        self.tgyro = TGYROtools.TGYRO(
            cdf=self.LocationCDF, time=self.time, avTime=self.avTime
        )
        self.tgyro.prep(
            FolderGACODE,
            cold_start=cold_start,
            remove_tmp=True,
            subfolder="tmp_tgyro_prep",
            profilesclass_custom=profiles,
            forceIfcold_start=forceIfcold_start,
        )

        self.FolderGACODE, self.FolderGACODE_tmp = (
            self.tgyro.FolderGACODE,
            self.tgyro.FolderGACODE_tmp,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize by preparing a tgyro class and running for -1 iterations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if specificInputs is None:
            print("\t- Testing... do TGLF files already exist?")

            exists = not cold_start
            for j in self.rhos:
                fii = self.FolderGACODE / f"input.tglf_{j:.4f}"
                if fii.exists():
                    print(f"\t\t- Testing {fii}")
                    inp = TGLFinput(fii)
                    exists = exists and not inp.onlyControl
                else:
                    print(
                        f"\t\t- Running scans because it does not exist file {IOtools.clipstr(fii)}"
                    )
                    exists = False
            if exists:
                print(
                    "\t\t- All input files to TGLF exist, not running scans",
                    typeMsg="i",
                )

            """
			Sometimes, if I'm running TGLF only from input.tglf file, I may not need to run the entire TGYRO workflow
			just be able to plot normalizations.
			"""
            if not exists:
                donotrun = False
            else:
                donotrun = True

            self.tgyro_results = self.tgyro.run_tglf_scan(
                rhos=self.rhos,
                cold_start=not exists,
                onlyThermal=onlyThermal_TGYRO,
                recalculatePTOT=recalculatePTOT,
                donotrun=donotrun,
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create the array of input classes
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            print(
                "\t- Creating dictionary with all input files generated by TGLF_scans"
            )

            self.inputsTGLF = {}
            for cont, rho in enumerate(self.rhos):
                fileN = self.FolderGACODE / f"input.tglf_{rho:.4f}"
                inputclass = TGLFinput(file=fileN)
                self.inputsTGLF[rho] = inputclass

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize by taking directly the inputs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else:
            self.inputsTGLF = specificInputs
            self.tgyro_results = tgyro_results

        """
		~~~~~ Create Normalizations ~~~~~
			- Only input.gacode needed
			- I can also give TRANSP CDF for complement. It is used in prep anyway, so good to store here
				and have the values for plotting the experimental fluxes.
			- I can also give TGYRO class for complement. It is used in prep anyway, so good to store here
				for plotting and check grid conversions.

		Note about the TGLF normalization:
			What matters is what's the mass used to normalized the MASS_X.
			If TGYRO was used to generate the input.tglf file, then the normalization mass is deuterium and all
			must be normalized to deuterium
		"""

        print("> Setting up normalizations")

        print(
            "\t- Using mass of deuterium to normalize things (not necesarily the first ion)",
            typeMsg="w",
        )
        self.tgyro.profiles.deriveQuantities(mi_ref=mi_D)

        self.NormalizationSets, cdf = NORMtools.normalizations(
            self.tgyro.profiles,
            LocationCDF=self.LocationCDF,
            time=self.time,
            avTime=self.avTime,
            cdf_open=cdf_open,
            tgyro=self.tgyro_results,
        )

        return cdf

    def prep_direct_tglf(
        self,
        FolderGACODE,  # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,  # If True, do not use what it potentially inside the folder, run again
        onlyThermal_TGYRO=False,  # Ignore fast particles in TGYRO
        recalculatePTOT=True, # Recalculate PTOT in TGYRO
        cdf_open=None,  # Grab normalizations from CDF file that is open as transp_output class
        inputgacode=None,  # *NOTE BELOW*
        specificInputs=None,  # *NOTE BELOW*
        tgyro_results=None,  # *NOTE BELOW*
        forceIfcold_start=False,  # Extra flag
        ):
        """
        * Note on inputgacode, specificInputs and tgyro_results:
                If I don't want to prepare, I can provide inputgacode and specificInputs, but I have to make sure they are consistent with one another!
                Optionally, I can give tgyro_results for further info in such a case
        """

        print("> Preparation of TGLF run")

        # PROFILES class.

        self.profiles = (
            PROFILEStools.PROFILES_GACODE(inputgacode)
            if inputgacode is not None
            else None
        )

        if self.profiles is None:
            
            # TGYRO class. It checks existence and creates input.profiles/input.gacode

            self.tgyro = TGYROtools.TGYRO(
                cdf=self.LocationCDF, time=self.time, avTime=self.avTime
            )
            self.tgyro.prep(
                FolderGACODE,
                cold_start=cold_start,
                remove_tmp=True,
                subfolder="tmp_tgyro_prep",
                profilesclass_custom=self.profiles,
                forceIfcold_start=forceIfcold_start,
            )

            self.profiles = self.tgyro.profiles

        self.profiles.deriveQuantities(mi_ref=mi_D)

        self.profiles.correct(options={'recompute_ptot':recalculatePTOT,'removeFast':onlyThermal_TGYRO})

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize by preparing a tgyro class and running for -1 iterations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if specificInputs is None:

            self.inputsTGLF = self.profiles.to_tglf(rhos=self.rhos)

            for rho in self.inputsTGLF:
                self.inputsTGLF[rho] = TGLFinput.initialize_in_memory(self.inputsTGLF[rho])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize by taking directly the inputs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else:
            self.inputsTGLF = specificInputs
        
        self.tgyro_results  = tgyro_results

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)
        
        if cold_start or not self.FolderGACODE.exists():
            IOtools.askNewFolder(self.FolderGACODE, force=forceIfcold_start)

        for rho in self.inputsTGLF:
            self.inputsTGLF[rho].file = self.FolderGACODE / f'input.tglf_{rho:.4f}'
            self.inputsTGLF[rho].writeCurrentStatus()

        """
		~~~~~ Create Normalizations ~~~~~
			- Only input.gacode needed
			- I can also give TRANSP CDF for complement. It is used in prep anyway, so good to store here
				and have the values for plotting the experimental fluxes.
			- I can also give TGYRO class for complement. It is used in prep anyway, so good to store here
				for plotting and check grid conversions.

		Note about the TGLF normalization:
			What matters is what's the mass used to normalized the MASS_X.
			If TGYRO was used to generate the input.tglf file, then the normalization mass is deuterium and all
			must be normalized to deuterium
		"""

        print("> Setting up normalizations")

        print(
            "\t- Using mass of deuterium to normalize things (not necesarily the first ion)",
            typeMsg="w",
        )
        self.profiles.deriveQuantities(mi_ref=mi_D)

        self.NormalizationSets, cdf = NORMtools.normalizations(
            self.profiles,
            LocationCDF=self.LocationCDF,
            time=self.time,
            avTime=self.avTime,
            cdf_open=cdf_open,
            tgyro=self.tgyro_results,
        )

        return cdf



    def prep_from_tglf(
        self,
        FolderGACODE,  # Main folder where all caculations happen (runs will be in subfolders)
        input_tglf_file,  # input.tglf file to start with
        input_gacode=None,
    ):
        print("> Preparation of TGLF class directly from input.tglf")

        # Main folder where things are
        self.FolderGACODE = IOtools.expandPath(FolderGACODE)

        # Main folder where things are
        self.NormalizationSets, _ = NORMtools.normalizations(
            PROFILEStools.PROFILES_GACODE(input_gacode)
            if input_gacode is not None
            else None
        )

        # input_tglf_file
        inputclass = TGLFinput(file=input_tglf_file)

        roa = inputclass.geom["RMIN_LOC"]
        print(f"\t- This file correspond to r/a={roa} according to RMIN_LOC")

        if self.NormalizationSets["input_gacode"] is not None:
            rho = np.interp(
                roa,
                self.NormalizationSets["input_gacode"].derived["roa"],
                self.NormalizationSets["input_gacode"].profiles["rho(-)"],
            )
            print(f"\t\t- rho={rho:.4f}, using input.gacode for conversion")
        else:
            print(
                "\t\t- No input.gacode for conversion, assuming rho=r/a, EXTREME CAUTION PLEASE",
                typeMsg="w",
            )
            rho = roa

        self.rhos = [rho]

        self.inputsTGLF = {self.rhos[0]: inputclass}

    def run(
        self,
        subFolderTGLF,  # 'tglf1/',
        TGLFsettings=None,
        extraOptions={},
        multipliers={},
        runWaveForms=None,  # e.g. runWaveForms = [0.3,1.0]
        forceClosestUnstableWF=True,  # Look at the growth rate spectrum and run exactly the ky of the closest unstable
        ApplyCorrections=True,  # Removing ions with too low density and that are fast species
        Quasineutral=False,  # Ensures quasineutrality. By default is False because I may want to run the file directly
        launchSlurm=True,
        cold_start=False,
        forceIfcold_start=False,
        extra_name="exe",
        anticipate_problems=True,
        slurm_setup={
            "cores": 4,
            "minutes": 5,
        },  # Cores per TGLF call (so, when running nR radii -> nR*4)
    ):

        if runWaveForms is None: runWaveForms = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare inputs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        tglf_executor, tglf_executor_full, folderlast = self._prepare_run_radii(
            subFolderTGLF,
            tglf_executor={},
            tglf_executor_full={},
            TGLFsettings=TGLFsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            runWaveForms=runWaveForms,
            forceClosestUnstableWF=forceClosestUnstableWF,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
            anticipate_problems=anticipate_problems,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run TGLF
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self._run(
            tglf_executor,
            tglf_executor_full=tglf_executor_full,
            TGLFsettings=TGLFsettings,
            runWaveForms=runWaveForms,
            forceClosestUnstableWF=forceClosestUnstableWF,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
        )

        self.FolderTGLFlast = folderlast

    def _run(self, tglf_executor, tglf_executor_full={}, **kwargs_TGLFrun):
        """
        extraOptions and multipliers are not being grabbed from kwargs_TGLFrun, but from tglf_executor for WF
        """

        print("\n> Run TGLF")

        c = 0
        for subFolderTGLF in tglf_executor:
            c += len(tglf_executor[subFolderTGLF])

        if c > 0:
            GACODErun.runTGLF(
                self.FolderGACODE,
                tglf_executor,
                filesToRetrieve=self.ResultsFiles,
                minutes=(
                    kwargs_TGLFrun["slurm_setup"]["minutes"]
                    if "slurm_setup" in kwargs_TGLFrun
                    and "minutes" in kwargs_TGLFrun["slurm_setup"]
                    else 5
                ),
                cores_tglf=(
                    kwargs_TGLFrun["slurm_setup"]["cores"]
                    if "slurm_setup" in kwargs_TGLFrun
                    and "cores" in kwargs_TGLFrun["slurm_setup"]
                    else 4
                ),
                name=f"tglf_{self.nameRunid}{kwargs_TGLFrun['extra_name'] if 'extra_name' in kwargs_TGLFrun else ''}",
                launchSlurm=(
                    kwargs_TGLFrun["launchSlurm"]
                    if "launchSlurm" in kwargs_TGLFrun
                    else True
                ),
            )
        else:
            print(
                "\t- TGLF not run because all results files found (please ensure consistency!)",
                typeMsg="i",
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Waveform if requested
        #  Cannot be in parallel to the previous run, because it needs the results of unstable ky
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if "runWaveForms" in kwargs_TGLFrun and len(kwargs_TGLFrun["runWaveForms"]) > 0:
            self._run_wf(
                kwargs_TGLFrun["runWaveForms"], tglf_executor_full, **kwargs_TGLFrun
            )

    def _prepare_run_radii(
        self,
        subFolderTGLF,  # 'tglf1/',
        rhos=None,
        tglf_executor={},
        tglf_executor_full={},
        TGLFsettings=None,
        extraOptions={},
        multipliers={},
        ApplyCorrections=True,  # Removing ions with too low density and that are fast species
        Quasineutral=False,  # Ensures quasineutrality. By default is False because I may want to run the file directly
        launchSlurm=True,
        cold_start=False,
        forceIfcold_start=False,
        anticipate_problems=True,
        extra_name="exe",
        slurm_setup={
            "cores": 4,
            "minutes": 5,
        },  # Cores per TGLF call (so, when running nR radii -> nR*4)
        **kwargs):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare for run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if rhos is None:
            rhos = self.rhos

        inputs = copy.deepcopy(self.inputsTGLF)
        FolderTGLF = self.FolderGACODE / subFolderTGLF

        ResultsFiles_new = []
        for i in self.ResultsFiles:
            if "mitim.out" not in i:
                ResultsFiles_new.append(i)
        self.ResultsFiles = ResultsFiles_new

        # Do I need to run all radii?
        rhosEvaluate = cold_start_checker(
            rhos,
            self.ResultsFiles,
            FolderTGLF,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
        )

        if len(rhosEvaluate) == len(rhos):
            # All radii need to be evaluated
            IOtools.askNewFolder(FolderTGLF, force=forceIfcold_start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Change this specific run of TGLF
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        (
            latest_inputsFileTGLF,
            latest_inputsFileTGLFDict,
        ) = changeANDwrite_TGLF(
            rhos,
            inputs,
            FolderTGLF,
            TGLFsettings=TGLFsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
        )

        tglf_executor_full[subFolderTGLF] = {}
        tglf_executor[subFolderTGLF] = {}
        for irho in self.rhos:
            tglf_executor_full[subFolderTGLF][irho] = {
                "folder": FolderTGLF,
                "dictionary": latest_inputsFileTGLFDict[irho],
                "inputs": latest_inputsFileTGLF[irho],
                "extraOptions": extraOptions,
                "multipliers": multipliers,
            }
            if irho in rhosEvaluate:
                tglf_executor[subFolderTGLF][irho] = tglf_executor_full[subFolderTGLF][
                    irho
                ]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stop if I expect problems
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if anticipate_problems:
            anticipate_problems_func(
                latest_inputsFileTGLFDict, rhosEvaluate, slurm_setup, launchSlurm
            )

        return tglf_executor, tglf_executor_full, FolderTGLF

    def _run_wf(self, kys, tglf_executor, **kwargs_TGLFrun):
        """
        extraOptions and multipliers are not being grabbed from kwargs_TGLFrun, but from tglf_executor
        """

        if "runWaveForms" in kwargs_TGLFrun:
            del kwargs_TGLFrun["runWaveForms"]

        # Grab these from tglf_executor
        if "extraOptions" in kwargs_TGLFrun:
            del kwargs_TGLFrun["extraOptions"]
        if "multipliers" in kwargs_TGLFrun:
            del kwargs_TGLFrun["multipliers"]

        self.ky_single = kys
        ResultsFiles = copy.deepcopy(self.ResultsFiles)
        self.ResultsFiles = copy.deepcopy(self.ResultsFiles_WF)

        self.FoldersTGLF_WF = {}
        if self.ky_single is not None:

            tglf_executorWF = {}
            for ky_single0 in self.ky_single:
                print(f"> Running TGLF waveform analysis, ky~{ky_single0}")

                self.FoldersTGLF_WF[f"ky{ky_single0}"] = {}
                for subFolderTGLF in tglf_executor:

                    ky_single_orig = copy.deepcopy(ky_single0)

                    FolderTGLF_old = tglf_executor[subFolderTGLF][
                        list(tglf_executor[subFolderTGLF].keys())[0]
                    ]["folder"]

                    self.ky_single = None
                    self.read(label=f"ky{ky_single0}", folder=FolderTGLF_old, cold_startWF = False)
                    self.ky_single = kys

                    self.FoldersTGLF_WF[f"ky{ky_single0}"][
                        FolderTGLF_old
                    ] = FolderTGLF_old / f"ky{ky_single0}"

                    ky_singles = []
                    for i, ir in enumerate(self.rhos):
                        # -------- Get the closest unstable mode to the one requested
                        if (
                            kwargs_TGLFrun["forceClosestUnstableWF"]
                            if "forceClosestUnstableWF" in kwargs_TGLFrun
                            else True
                        ):
                            # Only unstable ones
                            kys_n = []
                            for j in range(
                                len(self.results[f"ky{ky_single0}"]["TGLFout"][i].ky)
                            ):
                                if (
                                    self.results[f"ky{ky_single0}"]["TGLFout"][i].g[
                                        0, j
                                    ]
                                    > 0.0
                                ):
                                    kys_n.append(
                                        self.results[f"ky{ky_single0}"]["TGLFout"][
                                            i
                                        ].ky[j]
                                    )
                            kys_n = np.array(kys_n)
                            # ----

                            closest_ky = kys_n[
                                np.argmin(np.abs(kys_n - ky_single_orig))
                            ]
                            print(
                                f"\t- rho = {ir:.3f}, requested ky={ky_single_orig:.3f}, & closest unstable ky based on previous run: ky={closest_ky:.3f}",
                                typeMsg="i",
                            )
                            ky_single = closest_ky
                        else:
                            ky_single = ky_single0

                        ky_singles.append(ky_single)
                        # ------------------------------------------------------------

                    kwargs_TGLFrun0 = copy.deepcopy(kwargs_TGLFrun)
                    if "extraOptions" in kwargs_TGLFrun:
                        extraOptions_WF = copy.deepcopy(kwargs_TGLFrun["extraOptions"])
                        del kwargs_TGLFrun0["extraOptions"]

                    else:
                        extraOptions_WF = {}

                    extraOptions_WF = copy.deepcopy(tglf_executor[subFolderTGLF][
                        list(tglf_executor[subFolderTGLF].keys())[0]
                    ]["extraOptions"])
                    multipliers_WF = copy.deepcopy(tglf_executor[subFolderTGLF][
                        list(tglf_executor[subFolderTGLF].keys())[0]
                    ]["multipliers"])

                    extraOptions_WF["USE_TRANSPORT_MODEL"] = "F"
                    extraOptions_WF["WRITE_WAVEFUNCTION_FLAG"] = 1
                    extraOptions_WF["KY"] = ky_singles
                    extraOptions_WF["VEXB_SHEAR"] = (
                        0.0  # See email from G. Staebler on 05/16/2021
                    )

                    tglf_executorWF, _, _ = self._prepare_run_radii(
                        (FolderTGLF_old / f"ky{ky_single0}").relative_to(FolderTGLF_old.parent),
                        tglf_executor=tglf_executorWF,
                        extraOptions=extraOptions_WF,
                        multipliers=multipliers_WF,
                        **kwargs_TGLFrun0,
                    )

            # Run them all
            self._run(
                tglf_executorWF,
                runWaveForms=[],
                **kwargs_TGLFrun0,
            )

        # Recover previous stuff
        self.ResultsFiles_WF = copy.deepcopy(self.ResultsFiles)
        self.ResultsFiles = ResultsFiles
        # -----------

    def read(
        self,
        label="tglf1",
        folder=None,  # If None, search in the previously run folder
        suffix=None,  # If None, search with my standard _0.55 suffixes corresponding to rho of this TGLF class
        d_perp_cm=None,  # It can be a dictionary with rhos. If None provided, use the last one employed
        cold_startWF = True, # If this is a "complete" read, I will assign a None
    ):
        print("> Reading TGLF results")

        if d_perp_cm is not None:
            if isinstance(d_perp_cm, float):
                self.d_perp_dict = {}
                for rho in self.rhos:
                    self.d_perp_dict[rho] = d_perp_cm
            else:
                self.d_perp_dict = d_perp_cm

            # ---- Check d_perp
            keys_match = [i in self.d_perp_dict for i in self.rhos]
            if not np.all(keys_match):
                raise Exception("d_perp_cm provided does not have all radii requested")
            # --------

        else:
            if "d_perp_dict" not in self.__dict__:
                self.d_perp_dict = None
            else:
                if self.d_perp_dict is not None:
                    print(
                        "\t- Using d_perp stored from prevous reading:",
                        self.d_perp_dict,
                        typeMsg="w",
                    )

        # If no specified folder, check the last one
        if folder is None:
            folder = self.FolderTGLFlast


        # -----------------------------------------
        # ~~~~~~~ Read results
        # -----------------------------------------

        # ------------ Fluctuation stuff -----------
        self.updateConvolution()
        # ------------------------------------------

        self.results[label] = readTGLFresults(
            folder,
            self.NormalizationSets,
            self.rhos,
            convolution_fun_fluct=self.convolution_fun_fluct,
            factorTot_to_Perp=self.factorTot_to_Perp,
            suffix=suffix,
        )

        self.results[label]["convolution_fun_fluct"] = self.convolution_fun_fluct
        self.results[label]["DRMAJDX_LOC"] = self.DRMAJDX_LOC

        # -----------------------------------------
        # Store the input.gacode used
        # -----------------------------------------

        self.results[label]["profiles"] = (
            self.NormalizationSets["input_gacode"]
            if (self.NormalizationSets["input_gacode"] is not None)
            else None
        )

        # ------------------------------------------------------------
        # Waveform
        # ------------------------------------------------------------

        self.results[label]["wavefunction"] = {}
        if self.ky_single is not None:
            for ky_single0 in self.ky_single:
                if f"ky{ky_single0}" not in self.FoldersTGLF_WF:
                    continue

                if folder not in self.FoldersTGLF_WF[f'ky{ky_single0}']:
                    print(f"\t - Results not found for ky={ky_single0}, likely due to a cold_start with no wavefunction option", typeMsg="w")
                    continue

                self.results[label]["wavefunction"][f"ky{ky_single0}"] = {}
                for ir in self.rhos:
                    suffix0 = f"_{ir:.4f}" if suffix is None else suffix

                    self.results[label]["wavefunction"][f"ky{ky_single0}"][ir] = (
                        GACODEinterpret.Waveform_read(
                            self.FoldersTGLF_WF[f'ky{ky_single0}'][folder] / f"out.tglf.wavefunction{suffix0}",
                            self.FoldersTGLF_WF[f'ky{ky_single0}'][folder] / f"out.tglf.run{suffix0}",
                        )
                    )

        # After read, go back to no waveforms in case I want to read another case without it
        if cold_startWF:
            self.ky_single = None

    def plot(
        self,
        fn=None,
        labels=["tglf1"],
        fontsizeLeg=7,
        normalizations=None,  # I allow here to give a different normalization because that's use to plot the experimental flux
        extratitle="",
        forceXposition=None,  # can be used to force the use of e.g. the first rho position simulated (This is useful when I combined different runs that had different positions but I want to plot them together.)
        colors=None,  # colors must have in just one list the [labels*rhos]
        plotGACODE=True,
        plotNormalizations=True,
        labels_legend=None,
        title_legend=None,
        fn_color=None,
        WarnMeAboutPlots=True,
    ):
        total_plots = len(self.rhos) * len(labels)

        if WarnMeAboutPlots and (total_plots >= 10):
            surePlot = print(
                f">> TGLF module will plot {total_plots} individual TGLF per figure, which might be too much and lead to long plotting time or seg faults",
                typeMsg="q",
            )
            if not surePlot:
                print("Good choice! leaving module")
                return

        for label in labels:
            plotGACODE = (
                plotGACODE
                and ("profiles" in self.results[label])
                and (self.results[label]["profiles"] is not None)
            )
        plotNormalizations = plotNormalizations and plotGACODE


        # Grab all possibilities of wavefunctions
        ky_single_stored = {}
        ky_single_stored_unique = []
        for contLab, label in enumerate(labels):
            if "wavefunction" in self.results[label]:
                ky_single_stored[label] = [float(k.split('ky')[-1]) for k in self.results[label]["wavefunction"].keys()]
                ky_single_stored_unique += ky_single_stored[label] 
            else:
                ky_single_stored[label] = None
        ky_single_stored_unique = np.unique(ky_single_stored_unique)
        # --


        if labels_legend is None:
            labels_legend = labels

        if colors is None:
            colors = []
            for contrho in range(len(self.rhos)):
                for contLab in range(len(labels)):
                    colors.append(
                        GRAPHICStools.listColors()[contLab + contrho * len(labels)]
                    )

        if normalizations is None:
            normalizations = {}
            for label in labels:
                normalizations[label] = self.NormalizationSets

        max_num_species = 0
        max_fields = []
        successful_normalization = True
        for label in labels:
            for irho in range(len(self.rhos)):
                successful_normalization = (
                    successful_normalization
                    and self.results[label]["TGLFout"][irho].unnormalization_successful
                )
                max_num_species = np.max(
                    [max_num_species, self.results[label]["TGLFout"][irho].num_species]
                )

                for il in self.results[label]["TGLFout"][irho].fields:
                    if il not in max_fields:
                        max_fields.append(il)

        if fn is None:
            self.fn = GUItools.FigureNotebook(
                "TGLF MITIM Notebook", geometry="1700x900", vertical=True
            )
        else:
            self.fn = fn

        # *** TGLF Figures
        fig1 = self.fn.add_figure(label=f"{extratitle}Summary", tab_color=fn_color)
        fig2 = self.fn.add_figure(label=f"{extratitle}Contributors", tab_color=fn_color)
        figFluctuations = self.fn.add_figure(
            label=f"{extratitle}Spectra", tab_color=fn_color
        )
        figFields1 = self.fn.add_figure(
            label=f"{extratitle}Fields: Phi", tab_color=fn_color
        )
        if "a_par" in max_fields:
            figFields2 = self.fn.add_figure(
                label=f"{extratitle}Fields: A_par", tab_color=fn_color
            )
        if "a_per" in max_fields:
            figFields3 = self.fn.add_figure(
                label=f"{extratitle}Fields: A_per", tab_color=fn_color
            )
        figO = self.fn.add_figure(
            label=f"{extratitle}Model Details", tab_color=fn_color
        )

        figsWF = {}
        for ky_single0 in ky_single_stored_unique:
            figsWF[ky_single0] = self.fn.add_figure(
                label=f"{extratitle}WF @ ky~{ky_single0}", tab_color=fn_color
            )
        fig5 = self.fn.add_figure(label=f"{extratitle}Input Plasma", tab_color=fn_color)
        fig7 = self.fn.add_figure(
            label=f"{extratitle}Input Controls", tab_color=fn_color
        )

        # *** Postprocess Figures
        if successful_normalization:
            figS = self.fn.add_figure(label=f"{extratitle}Simple", tab_color=fn_color)
            figF = self.fn.add_figure(
                label=f"{extratitle}Fluctuations", tab_color=fn_color
            )
            fig4 = (
                self.fn.add_figure(
                    label=f"{extratitle}Normalization", tab_color=fn_color
                )
                if plotNormalizations
                else None
            )
            fig3 = self.fn.add_figure(
                label=f"{extratitle}Exp. Fluxes", tab_color=fn_color
            )

        # *** GACODE Figures

        if plotGACODE:
            figProf_1 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Prof.", tab_color=fn_color
            )
            figProf_2 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Power", tab_color=fn_color
            )
            figProf_3 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Geom.", tab_color=fn_color
            )
            figProf_4 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Grad.", tab_color=fn_color
            )
            figProf_5 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Flows", tab_color=fn_color
            )
            figProf_6 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Other", tab_color=fn_color
            )
            figProf_7 = self.fn.add_figure(
                label=f"{extratitle}GACODE-Imp.", tab_color=fn_color
            )

        grid = plt.GridSpec(4, 3, hspace=0.7, wspace=0.2)
        axsTGLF1 = np.empty((4, 3), dtype=plt.Axes)

        axsTGLF1[0, 0] = fig1.add_subplot(grid[0, 0])
        axsTGLF1[1, 0] = fig1.add_subplot(grid[1, 0], sharex=axsTGLF1[0, 0])
        axsTGLF1[2, 0] = fig1.add_subplot(grid[2, 0], sharex=axsTGLF1[0, 0])
        axsTGLF1[3, 0] = fig1.add_subplot(grid[3, 0])
        axsTGLF1[0, 1] = fig1.add_subplot(grid[0, 1], sharex=axsTGLF1[0, 0])
        axsTGLF1[1, 1] = fig1.add_subplot(grid[1, 1], sharex=axsTGLF1[0, 0])
        axsTGLF1[2, 1] = fig1.add_subplot(grid[2, 1], sharex=axsTGLF1[0, 0])
        axsTGLF1[3, 1] = fig1.add_subplot(grid[3, 1])
        axsTGLF1[0, 2] = fig1.add_subplot(grid[0, 2], sharex=axsTGLF1[0, 0])
        axsTGLF1[1, 2] = fig1.add_subplot(grid[1, 2], sharex=axsTGLF1[0, 0])
        axsTGLF1[2, 2] = fig1.add_subplot(grid[2, 2], sharex=axsTGLF1[0, 0])
        axsTGLF1[3, 2] = fig1.add_subplot(grid[3, 2])

        grid = plt.GridSpec(3, 5, hspace=0.6, wspace=0.3)
        axsTGLF2 = np.empty((3, 4), dtype=plt.Axes)

        axsTGLF2[0, 0] = fig2.add_subplot(grid[0, 0])
        axsTGLF2[1, 0] = fig2.add_subplot(grid[1, 0], sharex=axsTGLF2[0, 0])
        axsTGLF2[2, 0] = fig2.add_subplot(grid[2, 0], sharex=axsTGLF2[0, 0])
        axsTGLF2[0, 1] = fig2.add_subplot(grid[0, 1], sharex=axsTGLF2[0, 0])
        axsTGLF2[1, 1] = fig2.add_subplot(grid[1, 1], sharex=axsTGLF2[0, 0])
        axsTGLF2[2, 1] = fig2.add_subplot(grid[2, 1], sharex=axsTGLF2[0, 0])
        axsTGLF2[0, 2] = fig2.add_subplot(grid[0, 2])
        axsTGLF2[1, 2] = fig2.add_subplot(grid[1, 2])
        axsTGLF2[2, 2] = fig2.add_subplot(grid[2, 2])
        axsTGLF2[0, 3] = fig2.add_subplot(grid[0, 3:], sharex=axsTGLF2[0, 0])
        axsTGLF2[1, 3] = fig2.add_subplot(grid[1, 3:], sharex=axsTGLF2[0, 0])
        axsTGLF2[2, 3] = fig2.add_subplot(grid[2, 3:], sharex=axsTGLF2[0, 0])

        grid = plt.GridSpec(2, 2, hspace=0.6, wspace=0.3)
        axsTGLF3 = np.empty((2, 2), dtype=plt.Axes)

        axsTGLF3[0, 0] = figO.add_subplot(grid[0, 0])
        axsTGLF3[1, 0] = figO.add_subplot(grid[1, 0], sharex=axsTGLF3[0, 0])
        axsTGLF3[0, 1] = figO.add_subplot(grid[0, 1], sharex=axsTGLF3[0, 0])
        axsTGLF3[1, 1] = figO.add_subplot(grid[1, 1], sharex=axsTGLF3[0, 0])

        grid = plt.GridSpec(5, max_num_species, hspace=0.9, wspace=0.3)

        axsTGLF_flucts = np.empty((5, max_num_species), dtype=plt.Axes)
        for i in range(max_num_species):
            for j in range(5):
                axsTGLF_flucts[j, i] = figFluctuations.add_subplot(grid[j, i])

        grid = plt.GridSpec(2, max_num_species + 2, hspace=0.6, wspace=0.3)

        axsTGLF_fields1 = np.empty((2, max_num_species + 2), dtype=plt.Axes)
        axsTGLF_fields1[0, 0] = figFields1.add_subplot(grid[0, 0])
        axsTGLF_fields1[1, 0] = figFields1.add_subplot(
            grid[1, 0], sharex=axsTGLF_fields1[0, 0]
        )

        if "a_par" in max_fields:
            axsTGLF_fields2 = np.empty((2, max_num_species + 2), dtype=plt.Axes)
            axsTGLF_fields2[0, 0] = figFields2.add_subplot(grid[0, 0])
            axsTGLF_fields2[1, 0] = figFields2.add_subplot(
                grid[1, 0], sharex=axsTGLF_fields2[0, 0]
            )
        if "a_per" in max_fields:
            axsTGLF_fields3 = np.empty((2, max_num_species + 2), dtype=plt.Axes)
            axsTGLF_fields3[0, 0] = figFields3.add_subplot(grid[0, 0])
            axsTGLF_fields3[1, 0] = figFields3.add_subplot(
                grid[1, 0], sharex=axsTGLF_fields3[0, 0]
            )

        for i in range(max_num_species + 1):
            axsTGLF_fields1[0, i + 1] = figFields1.add_subplot(
                grid[0, i + 1], sharex=axsTGLF_fields1[0, 0]
            )
            axsTGLF_fields1[1, i + 1] = figFields1.add_subplot(
                grid[1, i + 1], sharex=axsTGLF_fields1[0, 0]
            )

            if "a_par" in max_fields:
                axsTGLF_fields2[0, i + 1] = figFields2.add_subplot(
                    grid[0, i + 1], sharex=axsTGLF_fields2[0, 0]
                )
                axsTGLF_fields2[1, i + 1] = figFields2.add_subplot(
                    grid[1, i + 1], sharex=axsTGLF_fields2[0, 0]
                )

            if "a_per" in max_fields:
                axsTGLF_fields3[0, i + 1] = figFields3.add_subplot(
                    grid[0, i + 1], sharex=axsTGLF_fields3[0, 0]
                )
                axsTGLF_fields3[1, i + 1] = figFields3.add_subplot(
                    grid[1, i + 1], sharex=axsTGLF_fields3[0, 0]
                )

        grid = plt.GridSpec(4, 2, hspace=0.5, wspace=0.3)
        axs5 = [
            fig5.add_subplot(grid[0, 0]),
            fig5.add_subplot(grid[0, 1]),
            fig5.add_subplot(grid[1, 0]),
            fig5.add_subplot(grid[1, 1]),
        ]
        axs5.append(axs5[-1].twinx())

        axs6 = [fig5.add_subplot(grid[2, :]), fig5.add_subplot(grid[3, :])]

        grid = plt.GridSpec(2, 1, hspace=0.5, wspace=0.5)
        axs7 = [fig7.add_subplot(grid[0]), fig7.add_subplot(grid[1])]

        if successful_normalization:
            grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.2)
            axT2 = fig3.add_subplot(grid[0, 0])
            axT2_2 = fig3.add_subplot(grid[0, 1], sharex=axT2)
            axT2_3 = fig3.add_subplot(grid[1, 0], sharex=axT2)
            axT2_4 = fig3.add_subplot(grid[1, 1], sharex=axT2)

            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
            axS00 = figS.add_subplot(grid[0, 0])
            axS01 = figS.add_subplot(grid[0, 1])
            axS10 = figS.add_subplot(grid[1, 0])
            axS11 = figS.add_subplot(grid[1, 1], sharex=axS01, sharey=axS01)
            axS20 = figS.add_subplot(grid[0, 2])
            axS21 = figS.add_subplot(grid[1, 2], sharex=axS01)

            grid = plt.GridSpec(3, 3, hspace=0.5, wspace=0.3)
            axFluc00 = figF.add_subplot(grid[0, 0])
            axFluc10e = figF.add_subplot(grid[1, 0], sharex=axFluc00)
            axFluc10 = figF.add_subplot(grid[2, 0])
            axFluc01 = figF.add_subplot(grid[0, 1], sharex=axFluc00)
            axFluc11e = figF.add_subplot(grid[1, 1], sharex=axFluc00)
            axFluc11 = figF.add_subplot(grid[2, 1])
            axFluc02 = figF.add_subplot(grid[0, 2], sharex=axFluc00)
            axFluc12e = figF.add_subplot(grid[1, 2], sharex=axFluc00)
            axFluc12 = figF.add_subplot(grid[2, 2])

            axFluc00Sym = axFluc00.twinx()
            axFluc01Sym = axFluc01.twinx()
            axFluc02Sym = axFluc02.twinx()

        if plotGACODE:
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

            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
            axsProf_2 = [
                figProf_2.add_subplot(grid[0, 0]),
                figProf_2.add_subplot(grid[0, 1]),
                figProf_2.add_subplot(grid[1, 0]),
                figProf_2.add_subplot(grid[1, 1]),
                figProf_2.add_subplot(grid[0, 2]),
                figProf_2.add_subplot(grid[1, 2]),
            ]
            grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.5)
            ax00c = figProf_3.add_subplot(grid[0, 0])
            axsProf_3 = [
                ax00c,
                figProf_3.add_subplot(grid[1, 0], sharex=ax00c),
                figProf_3.add_subplot(grid[2, 0], sharex=ax00c),
                figProf_3.add_subplot(grid[0, 1], sharex=ax00c),
                figProf_3.add_subplot(grid[1, 1], sharex=ax00c),
                figProf_3.add_subplot(grid[2, 1], sharex=ax00c),
                figProf_3.add_subplot(grid[0, 2], sharex=ax00c),
                figProf_3.add_subplot(grid[1, 2], sharex=ax00c),
                figProf_3.add_subplot(grid[2, 2], sharex=ax00c),
                figProf_3.add_subplot(grid[0, 3], sharex=ax00c),
                figProf_3.add_subplot(grid[1, 3], sharex=ax00c),
                figProf_3.add_subplot(grid[2, 3], sharex=ax00c),
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

            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

            axsProf_5 = [
                figProf_5.add_subplot(grid[0, 0]),
                figProf_5.add_subplot(grid[1, 0]),
                figProf_5.add_subplot(grid[0, 1]),
                figProf_5.add_subplot(grid[0, 2]),
                figProf_5.add_subplot(grid[1, 1]),
                figProf_5.add_subplot(grid[1, 2]),
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
            axsProf_7 = [
                figProf_7.add_subplot(grid[0, 0]),
                figProf_7.add_subplot(grid[0, 1]),
                figProf_7.add_subplot(grid[0, 2]),
                figProf_7.add_subplot(grid[1, 0]),
                figProf_7.add_subplot(grid[1, 1]),
                figProf_7.add_subplot(grid[1, 2]),
            ]

        grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.6)

        if plotNormalizations:
            axsNorm = [
                fig4.add_subplot(grid[0, 0]),
                fig4.add_subplot(grid[0, 1]),
                fig4.add_subplot(grid[0, 2]),
                fig4.add_subplot(grid[0, 3]),
                fig4.add_subplot(grid[1, 0]),
                fig4.add_subplot(grid[1, 1]),
                fig4.add_subplot(grid[1, 2]),
                fig4.add_subplot(grid[1, 3]),
            ]

            axsNorm_twins = []
            for ax in axsNorm:
                axsNorm_twins.append(ax.twinx())
        else:
            axsNorm = axsNorm_twins = None

        # ************************************************************************************
        # Plotting Loop
        # ************************************************************************************

        cont = 0
        for contLab, (label, label_legend) in enumerate(zip(labels, labels_legend)):
            # Rewrite self.rhos for this specific label to the forced position
            if forceXposition is not None:
                self.rhos = np.zeros(len(forceXposition))
                for lo in range(len(forceXposition)):
                    self.rhos[lo] = self.results[label]["x"][forceXposition[lo]]
            # -----------------------------------------------------------------------------------------------

            labZX = f"{labels_legend[contLab]} " if len(labels_legend) > 1 else ""

            for irho_cont in range(len(self.rhos)):
                full_label = (
                    f"{labZX}$\\rho_N={self.rhos[irho_cont]:.4f}$"
                    if len(self.rhos) > 1
                    else labZX
                )
                irho = np.where(self.results[label]["x"] == self.rhos[irho_cont])[0][0]

                # --------------------------------
                # Plot Raw TGLF (normalized)
                # --------------------------------
                self.results[label]["TGLFout"][irho].plotTGLF_Summary(
                    c=colors[cont], label=full_label, axs=axsTGLF1, irho_cont=irho_cont
                )
                self.results[label]["TGLFout"][irho].plotTGLF_Contributors(
                    c=colors[cont],
                    label=full_label,
                    axs=axsTGLF2,
                    fontsizeLeg=fontsizeLeg,
                    title_legend=title_legend,
                    cont=cont,
                )
                self.results[label]["TGLFout"][irho].plotTGLF_Model(
                    axs=axsTGLF3, c=colors[cont], label=full_label
                )

                self.results[label]["TGLFout"][irho].plotTGLF_Fluctuations(
                    axs=axsTGLF_flucts,
                    c=colors[cont],
                    label=full_label,
                    fontsizeLeg=fontsizeLeg,
                    title_legend=title_legend,
                    cont=cont,
                )

                self.results[label]["TGLFout"][irho].plotTGLF_Field(
                    quantity="phi",
                    c=colors[cont],
                    label=full_label,
                    axs=axsTGLF_fields1,
                    fontsizeLeg=fontsizeLeg,
                    title_legend=title_legend,
                    cont=cont,
                )

                if "a_par" in max_fields:
                    self.results[label]["TGLFout"][irho].plotTGLF_Field(
                        quantity="a_par",
                        c=colors[cont],
                        label=full_label,
                        axs=axsTGLF_fields2,
                        fontsizeLeg=fontsizeLeg,
                        title_legend=title_legend,
                        cont=cont,
                    )
                if "a_per" in max_fields:
                    self.results[label]["TGLFout"][irho].plotTGLF_Field(
                        quantity="a_per",
                        c=colors[cont],
                        label=full_label,
                        axs=axsTGLF_fields3,
                        fontsizeLeg=fontsizeLeg,
                        title_legend=title_legend,
                        cont=cont,
                    )

                # --------------------------------
                # TGLF inputs
                # --------------------------------

                self.results[label]["inputclasses"][irho].plotSpecies(
                    axs=axs5, color=colors[cont], legends=contLab == 0
                )
                self.results[label]["inputclasses"][irho].plotPlasma(
                    axs=axs6, color=colors[cont], legends=contLab == 0
                )
                self.results[label]["inputclasses"][irho].plotControls(
                    axs=axs7,
                    color=colors[cont],
                    markersize=2 * len(self.rhos) * len(labels) - cont,
                )

                # --------------------------------
                # "Simple" Plot
                # --------------------------------

                if successful_normalization:
                    GACODEplotting.plotTGLFspectrum(
                        [axS00, axS10],
                        self.results[label]["TGLFout"][irho].ky,
                        self.results[label]["TGLFout"][irho].g[0, :],
                        freq=self.results[label]["TGLFout"][irho].f[0, :],
                        coeff=0.0,
                        c=colors[cont],
                        ls="-",
                        lw=1,
                        label=full_label,
                        markersize=20,
                        alpha=1.0,
                        titles=["Growth Rate", "Real Frequency"],
                        removeLow=1e-4,
                        ylabel=True,
                    )

                cont += 1

            # --------------------------------
            # Profiles
            # --------------------------------

            colorLab = colors[len(self.rhos) * contLab : len(self.rhos) * (contLab + 1)]

            Qe, Qi, Ge = [], [], []
            QeGB, QiGB, GeGB = [], [], []
            TeF, neTe = [], []
            roas = []
            for irho_cont in range(len(self.rhos)):
                irho = np.where(self.results[label]["x"] == self.rhos[irho_cont])[0][0]

                if self.results[label]["TGLFout"][irho].unnormalization_successful:
                    Qe.append(self.results[label]["TGLFout"][irho].Qe_unn)
                    Qi.append(self.results[label]["TGLFout"][irho].Qi_unn)
                    Ge.append(self.results[label]["TGLFout"][irho].Ge_unn)
                    TeF.append(
                        self.results[label]["TGLFout"][irho].AmplitudeSpectrum_Te_level
                    )
                    neTe.append(self.results[label]["TGLFout"][irho].neTeSpectrum_level)

                roas.append(self.results[label]["TGLFout"][irho].roa)

                QeGB.append(self.results[label]["TGLFout"][irho].Qe)
                QiGB.append(self.results[label]["TGLFout"][irho].Qi)
                GeGB.append(self.results[label]["TGLFout"][irho].Ge)

            if self.results[label]["TGLFout"][irho].unnormalization_successful:
                axT2.plot(self.rhos, Qe, "-o", c=colorLab[0], lw=2, label=full_label)
                axS01.plot(self.rhos, Qe, "-o", c=colorLab[0], lw=2, label=full_label)

                axT2_2.plot(self.rhos, Qi, "-o", c=colorLab[0], lw=2, label=full_label)
                axS11.plot(self.rhos, Qi, "-o", c=colorLab[0], lw=2, label=full_label)
                axT2_3.plot(self.rhos, Ge, "-o", c=colorLab[0], lw=2, label=label)

                axT2_4.plot(
                    self.rhos,
                    QiGB,
                    "-o",
                    c=colorLab[0],
                    lw=2,
                    label="Qi, " + label,
                    markersize=3,
                )
                axT2_4.plot(
                    self.rhos,
                    QeGB,
                    "--s",
                    c=colorLab[0],
                    lw=2,
                    label="Qe, " + label,
                    markersize=3,
                )
                axT2_4.plot(
                    self.rhos,
                    GeGB,
                    "-.^",
                    c=colorLab[0],
                    lw=2,
                    label="Ge, " + label,
                    markersize=3,
                )

                axS20.plot(self.rhos, TeF, "-o", c=colorLab[0], lw=2, label=label)
                axS21.plot(self.rhos, neTe, "-o", c=colorLab[0], lw=2, label=label)

            axsTGLF1[3, 0].plot(roas, QeGB, "-", c=colorLab[0], lw=1)
            axsTGLF1[3, 1].plot(roas, QiGB, "-", c=colorLab[0], lw=1)
            axsTGLF1[3, 2].plot(roas, GeGB, "-", c=colorLab[0], lw=1)

            axsTGLF2[0, 2].plot(roas, QeGB, "-", c=colorLab[0], lw=1)
            axsTGLF2[1, 2].plot(roas, QiGB, "-", c=colorLab[0], lw=1)
            axsTGLF2[2, 2].plot(roas, GeGB, "-", c=colorLab[0], lw=1)

            # --------------------------------
            # Normalizations
            # --------------------------------

            if (
                successful_normalization
                and (normalizations[label] is not None)
                and (normalizations[label]["EXP"] is not None)
            ):
                normalization = normalizations[label]

                xp = normalization["EXP"]["rho"]

                yp = normalization["EXP"]["exp_Qe"]

                axT2.plot(xp, yp, color=colorLab[0], lw=0.5)
                axS01.plot(xp, yp, color=colorLab[0], lw=0.5)
                if "exp_Qe_error" in normalization["EXP"]:
                    GRAPHICStools.fillGraph(
                        axS01,
                        xp,
                        yp - np.array(normalization["EXP"]["exp_Qe_error"]),
                        y_up=yp + np.array(normalization["EXP"]["exp_Qe_error"]),
                        alpha=0.2,
                        color=colorLab[0],
                        label="",
                    )

                yp = normalization["EXP"]["exp_Qi"]

                axT2_2.plot(xp, yp, color=colorLab[0], lw=0.5)
                axS11.plot(xp, yp, color=colorLab[0], lw=0.5)
                if "exp_Qi_error" in normalization["EXP"]:
                    GRAPHICStools.fillGraph(
                        axS11,
                        xp,
                        yp - np.array(normalization["EXP"]["exp_Qi_error"]),
                        y_up=yp + np.array(normalization["EXP"]["exp_Qi_error"]),
                        alpha=0.2,
                        color=colorLab[0],
                        label="",
                    )

                if "exp_TeFluct" in normalization["EXP"]:
                    xp0 = normalization["EXP"]["exp_TeFluct_rho"]
                    yp = np.array(normalization["EXP"]["exp_TeFluct"])
                    yp_up = np.array(normalization["EXP"]["exp_TeFluct_error"])
                    yp_do = np.array(normalization["EXP"]["exp_TeFluct_error"])

                    axS20.plot(xp0, yp, "-*", color=colorLab[0], lw=0.5)

                    if len(xp0) == 1:
                        axS20.errorbar(
                            xp0,
                            yp,
                            c=colorLab[0],
                            yerr=np.atleast_2d([yp_do, yp_up]),
                            capsize=5.0,
                            fmt="none",
                        )
                    else:
                        GRAPHICStools.fillGraph(
                            axS20,
                            xp0,
                            yp + yp_do,
                            y_up=yp + yp_up,
                            alpha=0.2,
                            color=colorLab[0],
                            label="",
                        )

                if "exp_neTe" in normalization["EXP"]:
                    xp0 = normalization["EXP"]["exp_neTe_rho"]
                    yp = np.array(normalization["EXP"]["exp_neTe"])
                    yp_up = np.array(normalization["EXP"]["exp_neTe_error"])
                    yp_do = np.array(normalization["EXP"]["exp_neTe_error"])

                    axS21.plot(xp0, yp, "-*", color=colorLab[0], lw=0.5)

                    if len(xp0) == 1:
                        axS21.errorbar(
                            xp0,
                            yp,
                            c=colorLab[0],
                            yerr=np.atleast_2d([yp_do, yp_up]),
                            capsize=5.0,
                            fmt="none",
                        )
                    else:
                        GRAPHICStools.fillGraph(
                            axS21,
                            xp0,
                            yp + yp_do,
                            y_up=yp + yp_up,
                            alpha=0.2,
                            color=colorLab[0],
                            label="",
                        )

                yp = normalization["EXP"]["exp_Ge"]
                axT2_3.plot(xp, yp, color=colorLab[0], lw=0.5)

                yp = normalization["EXP"]["exp_Qi_gb"]
                axT2_4.plot(xp, yp, "-", color=colorLab[0], lw=0.5)

                yp = normalization["EXP"]["exp_Qe_gb"]
                axT2_4.plot(xp, yp, "--", color=colorLab[0], lw=0.5)

                yp = normalization["EXP"]["exp_Ge_gb"]
                axT2_4.plot(xp, yp, "-.", color=colorLab[0], lw=0.5)

                for irho_cont in range(len(self.rhos)):
                    QeInterp = np.interp(
                        self.rhos[irho_cont],
                        normalization["EXP"]["rho"],
                        normalization["EXP"]["exp_Qe"],
                    )
                    QiInterp = np.interp(
                        self.rhos[irho_cont],
                        normalization["EXP"]["rho"],
                        normalization["EXP"]["exp_Qi"],
                    )

        if len(self.rhos) > 1 or len(labels) > 1:
            axsTGLF1[1, 0].legend(
                loc="lower right", fontsize=fontsizeLeg, title=title_legend
            )

        if successful_normalization:
            if len(self.rhos) > 1 or len(labels) > 1:
                axS10.legend(loc="best", fontsize=fontsizeLeg * 1.5, title=title_legend)

            ax = axT2
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$Q_e$ ($MW/m^2$)")
            ax.set_ylim(bottom=0)
            ax.set_title("Electron heat flux")
            ax.legend(loc="best")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)

            # Simple
            ax = axS01
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$Q_e$ ($MW/m^2$)")
            ax.set_ylim(bottom=0)
            ax.set_title("Electron heat flux")
            ax.legend(loc="best", fontsize=fontsizeLeg * 1.5, title=title_legend)
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)

            # ---

            ax = axT2_2
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$Q_i$ ($MW/m^2$)")
            ax.set_ylim(bottom=0)
            ax.set_title("Ion heat flux")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)

            # Simple
            ax = axS11
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$Q_i$ ($MW/m^2$)")
            ax.set_ylim(bottom=0)
            ax.set_title("Ion heat flux")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)
            # ---

            # Simple
            ax = axS20
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$\\delta T_e/T_e$ %")
            ax.set_ylim(bottom=0)
            ax.set_title("Temperature fluctuations")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)
            # ---

            # Simple
            ax = axS21
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("Angle (degrees)")
            ax.set_title("$n_eT_e$ phase angle")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)
            # ---

            ax = axT2_3
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$\\Gamma_e$ ($1E20/s/m^2$)")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)

            ax = axT2_4
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("Transport Flux (GB)")
            ax.axhline(y=0, ls="--", c="k", lw=1)
            ax.set_title("GB Fluxes")
            ax.set_yscale("log")
            ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)

            # FLUCTS

            cont = 0
            for contLab, (label, label_legend) in enumerate(zip(labels, labels_legend)):
                labZX = f"{labels_legend[contLab]} " if len(labels_legend) > 1 else ""

                normalization = normalizations[label]["SELECTED"]

                for irho_cont in range(len(self.rhos)):
                    full_label = (
                        f"{labZX}$\\rho_N={self.rhos[irho_cont]:.4f}$"
                        if len(self.rhos) > 1
                        else labZX
                    )

                    irho = np.where(self.results[label]["x"] == self.rhos[irho_cont])[
                        0
                    ][0]

                    x = normalization["rho"]
                    rho_s = (
                        normalization["rho_s"][
                            np.argmin(np.abs(x - self.rhos[irho_cont]))
                        ]
                        * 100
                    )
                    a = normalization["rmin"][-1] * 100
                    rhosa = rho_s / a

                    kys = self.results[label]["TGLFout"][irho].ky / rho_s

                    xP = np.linspace(0, kys[-1], 1000)

                    if ("convolution_fun_fluct" in self.results[label]) and (
                        self.results[label]["convolution_fun_fluct"] is not None
                    ):
                        try:
                            yP = self.results[label]["convolution_fun_fluct"](
                                xP * rho_s, rho_s=rho_s, rho_eval=self.rhos[irho_cont]
                            )
                        except:
                            print(
                                "Could not plot convolution. Likely because of combination of runs with different evaluated rhos",
                                typeMsg="w",
                            )
                            yP = np.ones(len(xP))
                    else:
                        yP = np.ones(len(xP))

                    ax = axFluc00
                    fluct = self.results[label]["TGLFout"][irho].AmplitudeSpectrum_Te
                    ylabel = "$A_{T_e}(k_y)$"
                    GACODEplotting.plotTGLFfluctuations(
                        ax,
                        kys,
                        fluct,
                        markerType="o",
                        c=colors[cont],
                        ls="-",
                        lw=1,
                        label=full_label,
                        markersize=20,
                        alpha=1.0,
                        title="Temperature Fluctuation Amplitude",
                        ylabel=ylabel,
                    )

                    # kysPlot, fluctPlot 	= [0,kys[0]], [0,fluct[0]]
                    kysPlot, fluctPlot = kys, fluct
                    ax.plot(kysPlot, fluctPlot, "--", lw=0.3, color=colors[cont])

                    axFluc00Sym.plot(xP, yP, ls="-.", lw=0.5, color=colors[cont])

                    ax = axFluc10e
                    fluct = self.results[label]["TGLFout"][
                        irho
                    ].AmplitudeSpectrum_Te * np.interp(kys, xP, yP)
                    ylabel = "$A_{T_e}(k_y)$*W"
                    GACODEplotting.plotTGLFfluctuations(
                        ax,
                        kys,
                        fluct,
                        markerType="o",
                        c=colors[cont],
                        ls="-",
                        lw=1,
                        label=full_label,
                        markersize=20,
                        alpha=1.0,
                        title="Convoluted",
                        ylabel=ylabel,
                    )
                    # kysPlot, fluctPlot 	= [0,kys[0]], [0,fluct[0]]
                    kysPlot, fluctPlot = kys, fluct
                    ax.plot(kysPlot, fluctPlot, "--", lw=0.3, color=colors[cont])

                    ax = axFluc01
                    fluct = self.results[label]["TGLFout"][irho].AmplitudeSpectrum_ne
                    ylabel = "$A_{n_e}(k_y)$"
                    GACODEplotting.plotTGLFfluctuations(
                        ax,
                        kys,
                        fluct,
                        markerType="o",
                        c=colors[cont],
                        ls="-",
                        lw=1,
                        label=full_label,
                        markersize=20,
                        alpha=1.0,
                        title="Density Fluctuation Amplitude",
                        ylabel=ylabel,
                    )

                    # kysPlot, fluctPlot 	= [0,kys[0]], [0,fluct[0]]
                    kysPlot, fluctPlot = kys, fluct
                    ax.plot(kysPlot, fluctPlot, "--", lw=0.3, color=colors[cont])

                    axFluc01Sym.plot(xP, yP, ls="-.", lw=0.5, color=colors[cont])

                    ax = axFluc11e
                    fluct = self.results[label]["TGLFout"][
                        irho
                    ].AmplitudeSpectrum_ne * np.interp(kys, xP, yP)
                    ylabel = "$A_{n_e}(k_y)$*W"
                    GACODEplotting.plotTGLFfluctuations(
                        ax,
                        kys,
                        fluct,
                        markerType="o",
                        c=colors[cont],
                        ls="-",
                        lw=1,
                        label=full_label,
                        markersize=20,
                        alpha=1.0,
                        title="Convoluted",
                        ylabel=ylabel,
                    )

                    # kysPlot, fluctPlot 	= [0,kys[0]], [0,fluct[0]]
                    kysPlot, fluctPlot = kys, fluct
                    ax.plot(kysPlot, fluctPlot, "--", lw=0.3, color=colors[cont])

                    # ---
                    for inmode in range(
                        self.results[label]["TGLFout"][irho].num_nmodes
                    ):
                        ax = axFluc02
                        fluct = self.results[label]["TGLFout"][irho].neTeSpectrum[
                            inmode, :
                        ]
                        ylabel = "$n_eT_e(k_y)$"
                        GACODEplotting.plotTGLFfluctuations(
                            ax,
                            kys,
                            fluct,
                            markerType="o",
                            c=colors[cont],
                            ls="-",
                            lw=1 if inmode == 0 else 0.5,
                            label=full_label,
                            markersize=20,
                            alpha=1.0,
                            title="ne-Te cross-phase",
                            ylabel=ylabel,
                        )

                        # kysPlot, fluctPlot 	= [0,kys[0]], [0,fluct[0]]
                        kysPlot, fluctPlot = kys, fluct
                        ax.plot(
                            kysPlot,
                            fluctPlot,
                            "--",
                            lw=0.3 if inmode == 0 else 0.1,
                            color=colors[cont],
                        )

                        axFluc02Sym.plot(
                            xP,
                            yP,
                            ls="-.",
                            lw=0.5 if inmode == 0 else 0.3,
                            color=colors[cont],
                        )

                        ax = axFluc12e
                        fluct = self.results[label]["TGLFout"][irho].neTeSpectrum[
                            inmode, :
                        ] * np.interp(kys, xP, yP)
                        ylabel = "$n_eT_e(k_y)$*W"
                        GACODEplotting.plotTGLFfluctuations(
                            ax,
                            kys,
                            fluct,
                            markerType="o",
                            c=colors[cont],
                            ls="-",
                            lw=1 if inmode == 0 else 0.5,
                            label=full_label,
                            markersize=20,
                            alpha=1.0,
                            title="Convoluted",
                            ylabel=ylabel,
                        )

                        # kysPlot, fluctPlot 	= [0,kys[0]], [0,fluct[0]]
                        kysPlot, fluctPlot = kys, fluct
                        ax.plot(
                            kysPlot,
                            fluctPlot,
                            "--",
                            lw=0.3 if inmode == 0 else 0.1,
                            color=colors[cont],
                        )

                    cont += 1

            lims = [0, 1.5 / rho_s]
            ax = axFluc00
            ax.set_xlabel("$k_{\\perp}$ ($cm^{-1}$)")
            ax.set_ylim(bottom=0)
            ax.legend(loc="best", fontsize=fontsizeLeg)
            ax.set_xscale("linear")
            ax.set_xlim(lims)
            GRAPHICStools.addDenseAxis(ax)
            ax = axFluc01
            ax.set_xlabel("$k_{\\perp}$ ($cm^{-1}$)")
            ax.set_ylim(bottom=0)
            # ax.legend(loc='best');
            ax.set_xscale("linear")
            ax.set_xlim([0, 3])
            GRAPHICStools.addDenseAxis(ax)

            ax = axFluc10e
            ax.set_xlabel("$k_{\\perp}$ ($cm^{-1}$)")
            ax.set_ylim(bottom=0)
            ax.set_xscale("linear")
            ax.set_xlim(lims)
            GRAPHICStools.addDenseAxis(ax)
            ax = axFluc11e
            ax.set_xlabel("$k_{\\perp}$ ($cm^{-1}$)")
            ax.set_ylim(bottom=0)
            ax.set_xscale("linear")
            ax.set_xlim(lims)
            GRAPHICStools.addDenseAxis(ax)

            axFluc00Sym.set_ylabel("Convolution")
            GRAPHICStools.addDenseAxis(axFluc00Sym)
            axFluc01Sym.set_ylabel("Convolution")
            GRAPHICStools.addDenseAxis(axFluc01Sym)
            axFluc00Sym.set_ylim([0, 1])
            axFluc01Sym.set_ylim([0, 1])

            cont = 0
            T, TL, N, NT, C = [], [], [], [], []

            for contLab, (label, label_legend) in enumerate(zip(labels, labels_legend)):
                labZX = f"{labels_legend[contLab]} " if len(labels_legend) > 1 else ""

                for irho_cont in range(len(self.rhos)):
                    irho = np.where(self.results[label]["x"] == self.rhos[irho_cont])[
                        0
                    ][0]

                    T.append(
                        self.results[label]["TGLFout"][irho].AmplitudeSpectrum_Te_level
                    )
                    N.append(
                        self.results[label]["TGLFout"][irho].AmplitudeSpectrum_ne_level
                    )
                    NT.append(self.results[label]["TGLFout"][irho].neTeSpectrum_level)
                    TL.append(f"{labZX}$\\rho_N={self.rhos[irho_cont]:.4f}$")
                    C.append(colors[cont])
                    cont += 1

            ax = axFluc10
            ax.plot(TL, T, "-", c="k")
            ax.scatter(TL, T, color=C)
            ax.set_ylabel("$\\delta T_e/T_e$ (%)")
            ax.set_ylim(bottom=0)
            GRAPHICStools.addDenseAxis(ax)

            for tick in ax.get_xticklabels():
                tick.set_rotation(20)

            ax = axFluc11
            ax.plot(TL, N, "-", c="k")
            ax.scatter(TL, N, color=C)
            ax.set_ylabel("$\\delta n_e/n_e$ (%)")
            ax.set_ylim(bottom=0)
            GRAPHICStools.addDenseAxis(ax)

            for tick in ax.get_xticklabels():
                tick.set_rotation(20)

            ax = axFluc12
            ax.plot(TL, NT, "-", c="k")
            ax.scatter(TL, NT, color=C)
            ax.set_ylabel("$n_eT_e$ angle (degrees)")
            GRAPHICStools.addDenseAxis(ax)

            for tick in ax.get_xticklabels():
                tick.set_rotation(20)

            # --------------------------------
            # Normalization
            # --------------------------------
            if axsNorm is not None:
                for contLab, label in enumerate(labels):
                    normalization = normalizations[label]

                    colorsPlot = GRAPHICStools.listColors()[
                        contLab * 3 : contLab * 3 + 3
                    ]
                    NORMtools.plotNormalizations(
                        NormalizationSets=normalizations[label],
                        axs=axsNorm,
                        ax_twins=axsNorm_twins,
                        colors=colorsPlot,
                        extralab=label + " ",
                    )

        # --------------------------------
        # Wavefunction?
        # --------------------------------

        if len(ky_single_stored_unique)>0:
            includeSubdominant = False  # True if len(labels)<5 else False

            addLegend = True

            grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)

            for kycont, ky_single0 in enumerate(ky_single_stored_unique):
                figWF = figsWF[ky_single0]

                ax00 = figWF.add_subplot(grid[0, 0])
                ax01 = figWF.add_subplot(grid[0, 1])
                ax02 = figWF.add_subplot(grid[0, 2])
                ax03 = figWF.add_subplot(grid[0, 3])
                ax10 = figWF.add_subplot(grid[1, 0])
                ax11 = figWF.add_subplot(grid[1, 1])
                ax12 = figWF.add_subplot(grid[1, 2])
                ax13 = figWF.add_subplot(grid[1, 3])

                cont = 0
                for contLab, label in enumerate(labels):

                    if ky_single0 not in ky_single_stored[label]:
                        continue

                    labZX = (
                        f"{labels_legend[contLab]} " if len(labels_legend) > 1 else ""
                    )

                    for irho_cont in range(len(self.rhos)):
                        full_label = (
                            f"{labZX}$\\rho_N={self.rhos[irho_cont]:.4f}$"
                            if len(self.rhos) > 1
                            else labZX
                        )

                        irho = np.where(
                            self.results[label]["x"] == self.rhos[irho_cont]
                        )[0][0]

                        wf = self.results[label]["wavefunction"][f"ky{ky_single0}"][
                            self.rhos[irho_cont]
                        ]
                        theta = wf["theta"] / np.pi

                        markers = GRAPHICStools.listmarkers()

                        # ES
                        max0, min0 = GACODEplotting.plotWaveform(
                            [ax01, ax11],
                            theta,
                            wf["RE(phi)"],
                            wf["IM(phi)"],
                            color=colors[cont],
                            label=full_label,
                            includeSubdominant=includeSubdominant,
                        )

                        # BPER
                        max0, min0 = GACODEplotting.plotWaveform(
                            [ax02, ax12],
                            theta,
                            wf["RE(Bper)"],
                            wf["IM(Bper)"],
                            color=colors[cont],
                            label=full_label,
                            includeSubdominant=includeSubdominant,
                        )

                        # BPAR
                        max0, min0 = GACODEplotting.plotWaveform(
                            [ax03, ax13],
                            theta,
                            wf["RE(Bpar)"],
                            wf["IM(Bpar)"],
                            color=colors[cont],
                            label=full_label,
                            includeSubdominant=includeSubdominant,
                        )

                        # Eigenvalue

                        ax00.axvline(
                            x=ky_single_stored_unique[kycont],
                            ls="--",
                            lw=0.5,
                            c=colors[cont],
                            label="Requested" if cont == 0 else "",
                        )
                        ax10.axvline(
                            x=ky_single_stored_unique[kycont],
                            ls="--",
                            lw=0.5,
                            c=colors[cont],
                            label="Requested" if cont == 0 else "",
                        )

                        ax00.plot(
                            [wf["ky"][0]],
                            [wf["gamma"][0]],
                            markers[0],
                            markersize=7,
                            color=colors[cont],
                            label=f"{labZX}$\\rho_N={self.rhos[irho_cont]:.4f}$",
                        )
                        ax10.plot(
                            [wf["ky"][0]],
                            [wf["freq"][0]],
                            markers[0],
                            markersize=7,
                            color=colors[cont],
                        )

                        # all eigenvalues
                        ax00.plot(
                            self.results[label]["TGLFout"][irho_cont].ky,
                            self.results[label]["TGLFout"][irho_cont].g[0],
                            "-s",
                            markersize=3,
                            color=colors[cont],
                        )
                        ax10.plot(
                            self.results[label]["TGLFout"][irho_cont].ky,
                            self.results[label]["TGLFout"][irho_cont].f[0],
                            "-s",
                            markersize=3,
                            color=colors[cont],
                        )

                        if includeSubdominant:
                            for i in range(wf["ky"].shape[0] - 1):
                                ax00.plot(
                                    [wf["ky"][1 + i]],
                                    [wf["gamma"][1 + i]],
                                    markers[i + 1],
                                    markersize=2,
                                    color=colors[cont],
                                    label=f"{labZX}$\rho_N={self.rhos[irho_cont]:.4f}$ (mode {i+2})",
                                )
                                ax10.plot(
                                    [wf["ky"][1 + i]],
                                    [wf["freq"][1 + i]],
                                    markers[i + 1],
                                    markersize=2,
                                    color=colors[cont],
                                )

                            # all eigenvalues
                            ax00.plot(
                                self.results[label]["TGLFout"][irho_cont].ky,
                                self.results[label]["TGLFout"][irho_cont].g[i + 1],
                                "-s",
                                markersize=1,
                                color=colors[cont],
                            )
                            ax10.plot(
                                self.results[label]["TGLFout"][irho_cont].ky,
                                self.results[label]["TGLFout"][irho_cont].f[i + 1],
                                "-s",
                                markersize=1,
                                color=colors[cont],
                            )

                        cont += 1

                for ax in [ax03, ax01, ax02, ax13, ax11, ax12]:
                    ax.axhline(y=0, ls="-.", lw=0.5, c="g")
                    ax.axvline(x=0, ls="-.", lw=0.5, c="g")
                    ax.set_xlim([-3, 3])
                    ax.set_xlabel("Poloidal angle $\\theta$ ($\\pi$)")

                ax = ax00
                ax.set_xlabel("$k_\\theta \\rho_s$")
                ax.set_ylabel("$\\gamma$ ($c_s/a$)")
                GRAPHICStools.addDenseAxis(ax)
                if addLegend:
                    GRAPHICStools.addLegendApart(
                        ax, size=6, ratio=0.6, title=title_legend
                    )
                ax.set_title("Growth Rate")
                ax.set_xlim([ky_single_stored_unique[kycont] - 2.0, ky_single_stored_unique[kycont] + 2])
                # ax.set_yscale('log')

                ax = ax10
                ax.set_xlabel("$k_\\theta \\rho_s$")
                ax.set_ylabel("$\\omega$ ($c_s/a$)")
                GRAPHICStools.addDenseAxis(ax)
                if addLegend:
                    GRAPHICStools.addLegendApart(ax, size=6, ratio=0.6, withleg=False)
                ax.set_title("Real Frequency")
                ax.set_xlim([ky_single_stored_unique[kycont] - 2.0, ky_single_stored_unique[kycont] + 2])

                ax = ax01
                ax.set_xlabel("Poloidal angle $\\theta$ ($\\pi$)")
                ax.set_ylabel("Electric potential $\\delta\\phi$")
                ax.set_title("Real component $\\delta\\phi$")
                GRAPHICStools.addDenseAxis(ax)

                ax = ax11
                ax.set_ylabel("Electric potential $\\delta\\phi$")
                ax.set_title("Imaginary component $\\delta\\phi$")
                GRAPHICStools.addDenseAxis(ax)

                ax = ax02
                ax.set_ylabel("Magnetic potential $\\delta A_{\\parallel}$")
                ax.set_title(
                    "Real component $\\delta A_{\\parallel}$ ($\\delta B_{\\perp}$)"
                )
                GRAPHICStools.addDenseAxis(ax)

                ax = ax12
                ax.set_ylabel("Magnetic potential $\\delta A_{\\parallel}$")
                ax.set_title(
                    "Imaginary component $\\delta A_{\\parallel}$ ($\\delta B_{\\perp}$)"
                )
                GRAPHICStools.addDenseAxis(ax)

                ax = ax03

                ax.set_ylabel("Magnetic potential $\\delta A_{\\perp}$")
                ax.set_title(
                    "Real component $\\delta A_{\\perp}$ ($\\delta B_{\\parallel}$)"
                )
                GRAPHICStools.addDenseAxis(ax)

                ax = ax13
                ax.set_ylabel("Magnetic potential $\\delta A_{\\perp}$")
                ax.set_title(
                    "Imaginary component $\\delta A_{\\perp}$ ($\\delta B_{\\parallel}$)"
                )
                GRAPHICStools.addDenseAxis(ax)

        # --------------------------------
        # Profiles
        # --------------------------------

        for contLab, label in enumerate(labels):
            colorLab = colors[len(self.rhos) * contLab : len(self.rhos) * (contLab + 1)]
            if plotGACODE:
                self.results[label]["profiles"].plot(
                    axs1=axsProf_1,
                    axs2=axsProf_2,
                    axs3=axsProf_3,
                    axs4=axsProf_4,
                    axsFlows=axsProf_5,
                    axs6=axsProf_6,
                    axsImps=axsProf_7,
                    color=colorLab[0],
                    legYN=contLab == 0,
                )

    # ~~~~~~~~~~~~~~ Scan options

    def runScan(
        self,
        subFolderTGLF,  # 'scan1',
        multipliers={},
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        relativeChanges=True,
        **kwargs_TGLFrun,
    ):

        # -------------------------------------
        # Add baseline
        # -------------------------------------
        if (1.0 not in varUpDown) and relativeChanges:
            print(
                "\n* Since variations vector did not include base case, I am adding it",
                typeMsg="i",
            )
            varUpDown_new = []
            added = False
            for i in varUpDown:
                if i > 1.0 and not added:
                    varUpDown_new.append(1.0)
                    added = True
                varUpDown_new.append(i)
        else:
            varUpDown_new = varUpDown


        tglf_executor, tglf_executor_full, folders, varUpDown_new = self._prepare_scan(
            subFolderTGLF,
            multipliers=multipliers,
            variable=variable,
            varUpDown=varUpDown_new,
            relativeChanges=relativeChanges,
            **kwargs_TGLFrun,
        )

        # Run them all
        self._run(
            tglf_executor,
            tglf_executor_full=tglf_executor_full,
            **kwargs_TGLFrun,
        )

        # Read results
        for cont_mult, mult in enumerate(varUpDown_new):
            name = f"{variable}_{mult}"
            self.read(
                label=f"{self.subFolderTGLF_scan}_{name}", folder=folders[cont_mult], cold_startWF = False
            )

    def _prepare_scan(
        self,
        subFolderTGLF,  # 'scan1',
        multipliers={},
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        relativeChanges=True,
        **kwargs_TGLFrun,
    ):
        """
        Multipliers will be modified by adding the scaning variables, but I don't want to modify the original
        multipliers, as they may be passed to the next scan

        Set relativeChanges=False if varUpDown contains the exact values to change, not multipleiers
        """
        multipliers_mod = copy.deepcopy(multipliers)

        self.subFolderTGLF_scan = subFolderTGLF

        if relativeChanges:
            for i in range(len(varUpDown)):
                varUpDown[i] = round(varUpDown[i], 6)

        print(f"\n- Proceeding to scan {variable}:")
        tglf_executor = {}
        tglf_executor_full = {}
        folders = []
        for cont_mult, mult in enumerate(varUpDown):
            mult = round(mult, 6)

            if relativeChanges:
                print(
                    f"\n + Multiplier: {mult} -----------------------------------------------------------------------------------------------------------"
                )
            else:
                print(
                    f"\n + Value: {mult} ----------------------------------------------------------------------------------------------------------------"
                )

            multipliers_mod[variable] = mult
            name = f"{variable}_{mult}"

            species = self.inputsTGLF[self.rhos[0]]  # Any rho will do

            multipliers_mod = completeVariation(multipliers_mod, species)

            if not relativeChanges:
                for ikey in multipliers_mod:
                    kwargs_TGLFrun["extraOptions"][ikey] = multipliers_mod[ikey]
                multipliers_mod = {}

            # Force ensure quasineutrality if the
            if variable in ["AS_3", "AS_4", "AS_5", "AS_6"]:
                kwargs_TGLFrun["Quasineutral"] = True

            # Only ask the cold_start in the first round
            kwargs_TGLFrun["forceIfcold_start"] = cont_mult > 0 or (
                "forceIfcold_start" in kwargs_TGLFrun and kwargs_TGLFrun["forceIfcold_start"]
            )

            tglf_executor, tglf_executor_full, folderlast = self._prepare_run_radii(
                f"{self.subFolderTGLF_scan}_{name}",
                tglf_executor=tglf_executor,
                tglf_executor_full=tglf_executor_full,
                multipliers=multipliers_mod,
                **kwargs_TGLFrun,
            )

            folders.append(copy.deepcopy(folderlast))

        return tglf_executor, tglf_executor_full, folders, varUpDown

    def readScan(
        self, label="scan1", subFolderTGLF=None, variable="RLTS_1", positionIon=2
    ):

        if subFolderTGLF is None:
            subFolderTGLF = self.subFolderTGLF_scan

        self.scans[label] = {}
        self.scans[label]["variable"] = variable
        self.scans[label]["positionBase"] = None
        self.scans[label]["unnormalization_successful"] = True
        self.scans[label]["results_tags"] = []

        # ----
        x, Qe, Qi, Ge, Gi, ky, g, f, eta1, eta2, itg, tem, etg = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        Qe_gb, Qi_gb, Ge_gb = [], [], []
        etalow_g, etalow_f, etalow_k = [], [], []
        cont = 0
        for ikey in self.results:
            isThisTheRightReadResults = (subFolderTGLF in ikey) and (
                variable
                == "_".join(ikey.split("_")[:-1]).split(subFolderTGLF + "_")[-1]
            )

            if isThisTheRightReadResults:

                self.scans[label]["results_tags"].append(ikey)

                x0, Qe0, Qi0, Ge0, Gi0, ky0, g0, f0, eta10, eta20, itg0, tem0, etg0 = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                Qe_gb0, Qi_gb0, Ge_gb0 = [], [], []
                etalow_g0, etalow_f0, etalow_k0 = [], [], []
                for irho_cont in range(len(self.rhos)):
                    irho = np.where(self.results[ikey]["x"] == self.rhos[irho_cont])[0][
                        0
                    ]

                    # Unnormalized
                    x0.append(self.results[ikey]["parsed"][irho][variable])
                    Qe_gb0.append(self.results[ikey]["TGLFout"][irho].Qe)
                    Qi_gb0.append(self.results[ikey]["TGLFout"][irho].Qi)
                    Ge_gb0.append(self.results[ikey]["TGLFout"][irho].Ge)
                    ky0.append(self.results[ikey]["TGLFout"][irho].ky)
                    g0.append(self.results[ikey]["TGLFout"][irho].g)
                    f0.append(self.results[ikey]["TGLFout"][irho].f)
                    eta10.append(
                        self.results[ikey]["TGLFout"][irho].etas["metrics"][
                            "eta_ITGTEM"
                        ]
                    )
                    eta20.append(
                        self.results[ikey]["TGLFout"][irho].etas["metrics"][
                            "eta_ITGETG"
                        ]
                    )
                    etalow_g0.append(
                        self.results[ikey]["TGLFout"][irho].etas["metrics"][
                            "g_lowk_max"
                        ]
                    )
                    etalow_k0.append(
                        self.results[ikey]["TGLFout"][irho].etas["metrics"][
                            "k_lowk_max"
                        ]
                    )
                    etalow_f0.append(
                        self.results[ikey]["TGLFout"][irho].etas["metrics"][
                            "f_lowk_max"
                        ]
                    )
                    itg0.append(
                        self.results[ikey]["TGLFout"][irho].etas["ITG"]["g_max"]
                    )
                    tem0.append(
                        self.results[ikey]["TGLFout"][irho].etas["TEM"]["g_max"]
                    )
                    etg0.append(
                        self.results[ikey]["TGLFout"][irho].etas["ETG"]["g_max"]
                    )

                    if self.results[ikey]["TGLFout"][irho].unnormalization_successful:
                        Qe0.append(self.results[ikey]["TGLFout"][irho].Qe_unn)
                        Qi0.append(self.results[ikey]["TGLFout"][irho].Qi_unn)
                        Ge0.append(self.results[ikey]["TGLFout"][irho].Ge_unn)
                        Gi0.append(
                            self.results[ikey]["TGLFout"][irho].GiAll_unn[
                                positionIon - 2
                            ]
                        )  # minus 2 because first is electrons and python starts at 0
                    else:
                        self.scans[label]["unnormalization_successful"] = False

                x.append(x0)
                Qe.append(Qe0)
                Qi.append(Qi0)
                Ge.append(Ge0)
                Qe_gb.append(Qe_gb0)
                Qi_gb.append(Qi_gb0)
                Ge_gb.append(Ge_gb0)
                Gi.append(Gi0)
                ky.append(ky0)
                g.append(g0)
                f.append(f0)
                eta1.append(eta10)
                eta2.append(eta20)
                etalow_g.append(etalow_g0)
                etalow_f.append(etalow_f0)
                etalow_k.append(etalow_k0)
                itg.append(itg0)
                tem.append(tem0)
                etg.append(etg0)

                if float(ikey.split('_')[-1]) == 1.0:
                    self.scans[label]["positionBase"] = cont
                cont += 1

        self.scans[label]["x"] = np.array(self.rhos)
        self.scans[label]["xV"] = np.atleast_2d(np.transpose(x))
        self.scans[label]["Qe_gb"] = np.atleast_2d(np.transpose(Qe_gb))
        self.scans[label]["Qi_gb"] = np.atleast_2d(np.transpose(Qi_gb))
        self.scans[label]["Ge_gb"] = np.atleast_2d(np.transpose(Ge_gb))
        self.scans[label]["Qe"] = np.atleast_2d(np.transpose(Qe))
        self.scans[label]["Qi"] = np.atleast_2d(np.transpose(Qi))
        self.scans[label]["Ge"] = np.atleast_2d(np.transpose(Ge))
        self.scans[label]["Gi"] = np.atleast_2d(np.transpose(Gi))
        self.scans[label]["eta1"] = np.atleast_2d(np.transpose(eta1))
        self.scans[label]["eta2"] = np.atleast_2d(np.transpose(eta2))
        self.scans[label]["itg"] = np.atleast_2d(np.transpose(itg))
        self.scans[label]["tem"] = np.atleast_2d(np.transpose(tem))
        self.scans[label]["etg"] = np.atleast_2d(np.transpose(etg))
        self.scans[label]["g_lowk_max"] = np.atleast_2d(np.transpose(etalow_g))
        self.scans[label]["f_lowk_max"] = np.atleast_2d(np.transpose(etalow_f))
        self.scans[label]["k_lowk_max"] = np.atleast_2d(np.transpose(etalow_k))
        self.scans[label]["ky"] = np.array(ky)
        self.scans[label]["g"] = np.array(g)
        self.scans[label]["f"] = np.array(f)
        if len(self.scans[label]["ky"].shape) == 2:
            self.scans[label]["ky"] = self.scans[label]["ky"].reshape(
                (1, self.scans[label]["ky"].shape[0], self.scans[label]["ky"].shape[1])
            )
            self.scans[label]["g"] = self.scans[label]["g"].reshape(
                (1, self.scans[label]["g"].shape[0], self.scans[label]["g"].shape[1])
            )
            self.scans[label]["f"] = self.scans[label]["f"].reshape(
                (1, self.scans[label]["f"].shape[0], self.scans[label]["f"].shape[1])
            )
        else:
            self.scans[label]["ky"] = np.transpose(
                self.scans[label]["ky"], axes=[1, 0, 2]
            )
            self.scans[label]["g"] = np.transpose(
                self.scans[label]["g"], axes=[1, 0, 2, 3]
            )
            self.scans[label]["f"] = np.transpose(
                self.scans[label]["f"], axes=[1, 0, 2, 3]
            )

    def plotScan(
        self,
        labels=["scan1"],
        figs=None,
        variableLabel="X",
        plotExperiment=True,
        normalizations=None,
        relativeX=False,
        forceXposition=None,
        plotTGLFs=True,
    ):
        unnormalization_successful = True
        for label in labels:
            unnormalization_successful = (
                unnormalization_successful
                and self.scans[label]["unnormalization_successful"]
            )

        if figs is None:
            self.fn = GUItools.FigureNotebook(
                "TGLF Scan MITIM Notebook", geometry="1500x900", vertical=True
            )
            if unnormalization_successful:
                fig1 = self.fn.add_figure(label="Fluxes")
            fig1e = self.fn.add_figure(label="Fluxes (GB)")
            fig2 = self.fn.add_figure(label="Linear Stability")
        else:
            [fig1, fig1e, fig2] = figs

        colors = GRAPHICStools.listColors()
        fontsizeLeg = 7

        colorsLines = GRAPHICStools.listColors()[5:]

        if unnormalization_successful:
            grid = plt.GridSpec(1, 3, hspace=0.3, wspace=0.3)
            ax1_00 = fig1.add_subplot(grid[0, 0])
            ax1_10 = fig1.add_subplot(grid[0, 1], sharex=ax1_00)
            ax1_20 = fig1.add_subplot(grid[0, 2], sharex=ax1_00)

        grid = plt.GridSpec(1, 3, hspace=0.3, wspace=0.3)
        ax1_00e = fig1e.add_subplot(grid[0, 0])
        ax1_10e = fig1e.add_subplot(grid[0, 1], sharex=ax1_00e)
        ax1_20e = fig1e.add_subplot(grid[0, 2], sharex=ax1_00e)

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        ax2_00 = fig2.add_subplot(grid[0, 0])
        ax2_10 = fig2.add_subplot(grid[1, 0], sharex=ax2_00)
        ax2_01 = fig2.add_subplot(grid[0, 1])
        ax2_11 = fig2.add_subplot(grid[1, 1], sharex=ax2_01)

        if normalizations is None:
            normalizations = {}
            for label in labels:
                normalizations[label] = self.NormalizationSets

        doIhaveBase = True
        for label in labels:
            if self.scans[label]["positionBase"] is None:
                doIhaveBase = False

        if (not doIhaveBase) and relativeX:
            print('\t\t- No base case (1.0) found for all scans to be plotted, I cannot plot relative')
            relativeX = False

        if relativeX:
            variableLabel += " (%)"

        ms = 20
        cont = 0
        colorsChosen = []
        for contLab, label in enumerate(labels):
            # Rewrite self.rhos for this specific label to the forced position
            if forceXposition is not None:
                self.rhos = np.zeros(len(forceXposition))
                for lo in range(len(forceXposition)):
                    self.rhos[lo] = self.scans[label]["x"][forceXposition[lo]]
            # -----------------------------------------------------------------------------------------------

            positionBase = self.scans[label]["positionBase"]

            x = self.scans[label]["xV"]
            if relativeX:
                xbase = x[:, positionBase : positionBase + 1]
                x = (x - xbase) / xbase * 100.0

            Qe, Qi, Ge = (
                self.scans[label]["Qe"],
                self.scans[label]["Qi"],
                self.scans[label]["Ge"],
            )
            Qe_gb, Qi_gb, Ge_gb = (
                self.scans[label]["Qe_gb"],
                self.scans[label]["Qi_gb"],
                self.scans[label]["Ge_gb"],
            )
            eta1, eta2 = self.scans[label]["eta1"], self.scans[label]["eta2"]
            itg, tem, etg = (
                self.scans[label]["itg"],
                self.scans[label]["tem"],
                self.scans[label]["etg"],
            )
            ky, g, f = (
                self.scans[label]["ky"],
                self.scans[label]["g"],
                self.scans[label]["f"],
            )

            normalization = normalizations[label]["EXP"]

            colorsChosen0 = []
            for irho_cont in range(len(self.rhos)):
                irho = np.where(self.scans[label]["x"] == self.rhos[irho_cont])[0][0]

                if len(labels) > 1:
                    labZX = f"{label} "
                else:
                    labZX = ""

                # If the scans included more than one rho or more than one label, then use alpha fading per each
                if len(self.rhos) + len(labels) > 2:
                    scan_colors = [
                        colors[cont],
                        colors[cont],
                    ]  # [colors[cont*2+1],colors[cont*2]]
                    colorsC, _ = GRAPHICStools.colorTableFade(
                        ky.shape[1],
                        startcolor=scan_colors[1],
                        endcolor=scan_colors[0],
                        alphalims=[0.2, 1.0],
                    )
                    colorLine = colors[cont]

                # If the scans were a single one, then use fading in between two colors
                else:
                    scan_colors = [colors[cont], colors[cont + 1]]
                    colorsC, _ = GRAPHICStools.colorTableFade(
                        ky.shape[1],
                        startcolor=scan_colors[1],
                        endcolor=scan_colors[0],
                        alphalims=[1.0, 1.0],
                    )
                    colorLine = colorsLines[contLab]

                colorsChosen0.append(colorsC)

                if unnormalization_successful:
                    ax = ax1_00
                    ax.plot(
                        x[irho],
                        Qe[irho],
                        "-",
                        c=colorLine,
                        lw=1.0,
                        label=labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$",
                    )
                    ax.scatter(x[irho], Qe[irho], marker="o", facecolor=colorsC, s=ms)
                    if positionBase is not None:
                        ax.axvline(
                            x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                        )
                    if (plotExperiment) and (normalization is not None):
                        exp_x = normalization["rho"]
                        exp_y = normalization["exp_Qe"]
                        ix = np.argmin(np.abs(exp_x - self.rhos[irho_cont]))
                        yy = exp_y[ix]
                        ax.axhline(y=yy, ls="--", c=scan_colors[0], lw=1.0)

                ax = ax1_00e
                ax.plot(
                    x[irho],
                    Qe_gb[irho],
                    "-",
                    c=colorLine,
                    lw=1.0,
                    label=labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$",
                )
                ax.scatter(x[irho], Qe_gb[irho], marker="o", facecolor=colorsC, s=ms)
                if positionBase is not None:
                    ax.axvline(
                        x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                    )
                if (plotExperiment) and (normalization is not None):
                    exp_x = normalization["rho"]
                    exp_y = normalization["exp_Qe_gb"]
                    ix = np.argmin(np.abs(exp_x - self.rhos[irho_cont]))
                    yy = exp_y[ix]
                    ax.axhline(y=yy, ls="--", c=scan_colors[0], lw=1.0)

                if unnormalization_successful:
                    ax = ax1_10
                    ax.plot(
                        x[irho],
                        Qi[irho],
                        "-",
                        c=colorLine,
                        lw=1.0,
                        label=labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$",
                    )
                    ax.scatter(x[irho], Qi[irho], marker="o", facecolor=colorsC, s=ms)
                    if positionBase is not None:
                        ax.axvline(
                            x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                        )
                    if (plotExperiment) and (normalization is not None):
                        exp_x = normalization["rho"]
                        exp_y = normalization["exp_Qi"]
                        ix = np.argmin(np.abs(exp_x - self.rhos[irho_cont]))
                        yy = exp_y[ix]
                        ax.axhline(y=yy, ls="--", c=scan_colors[0], lw=1.0)

                ax = ax1_10e
                ax.plot(
                    x[irho],
                    Qi_gb[irho],
                    "-",
                    c=colorLine,
                    lw=1.0,
                    label=labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$",
                )
                ax.scatter(x[irho], Qi_gb[irho], marker="o", facecolor=colorsC, s=ms)
                if positionBase is not None:
                    ax.axvline(
                        x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                    )
                if (plotExperiment) and (normalization is not None):
                    exp_x = normalization["rho"]
                    exp_y = normalization["exp_Qi_gb"]
                    ix = np.argmin(np.abs(exp_x - self.rhos[irho_cont]))
                    yy = exp_y[ix]
                    ax.axhline(y=yy, ls="--", c=scan_colors[0], lw=1.0)

                if unnormalization_successful:
                    ax = ax1_20
                    ax.plot(
                        x[irho],
                        Ge[irho],
                        "-",
                        c=colorLine,
                        lw=1.0,
                        label=labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$",
                    )
                    ax.scatter(x[irho], Ge[irho], marker="o", facecolor=colorsC, s=ms)
                    if positionBase is not None:
                        ax.axvline(
                            x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                        )
                    if (plotExperiment) and (normalization is not None):
                        exp_x = normalization["rho"]
                        exp_y = normalization["exp_Ge"]
                        ix = np.argmin(np.abs(exp_x - self.rhos[irho_cont]))
                        yy = exp_y[ix]
                        ax.axhline(y=yy, ls="--", c=scan_colors[0], lw=1.0)

                ax = ax1_20e
                ax.plot(
                    x[irho],
                    Ge_gb[irho],
                    "-",
                    c=colorLine,
                    lw=1.0,
                    label=labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$",
                )
                ax.scatter(x[irho], Ge_gb[irho], marker="o", facecolor=colorsC, s=ms)
                if positionBase is not None:
                    ax.axvline(
                        x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                    )

                if (plotExperiment) and (normalization is not None):
                    exp_x = normalization["rho"]
                    exp_y = normalization["exp_Ge_gb"]
                    ix = np.argmin(np.abs(exp_x - self.rhos[irho_cont]))
                    yy = exp_y[ix]
                    ax.axhline(y=yy, ls="--", c=scan_colors[0], lw=1.0)

                axs = [ax2_00, ax2_10]

                for ivar in range(ky.shape[1]):
                    if ivar == ky.shape[1] - 1:
                        labb = labZX + f"$\\rho_N={self.rhos[irho_cont]:.4f}$"
                    else:
                        labb = ""

                    nmode = 0
                    GACODEplotting.plotTGLFspectrum(
                        axs,
                        ky[irho][ivar],
                        g[irho][ivar][nmode],
                        freq=f[irho][ivar][nmode],
                        coeff=0.0,
                        c=colorsC[ivar],
                        ls="-",
                        lw=1,
                        label=labb,
                        markersize=20,
                        alpha=colorsC[ivar][-1],
                        titles=["Growth rates", "Real frequency"],
                        removeLow=1e-4,
                        ylabel=True,
                    )

                ax = ax2_11
                if cont == 0:
                    labb = (
                        labZX
                        + "$\\eta_{ITG/TEM}$,"
                        + f" $\\rho_N={self.rhos[irho_cont]:.4f}$"
                    )
                else:
                    labb = ""
                ax.plot(x[irho], eta1[irho], "-", c=colorLine, lw=0.5)
                ax.scatter(
                    x[irho], eta1[irho], marker="o", facecolor=colorsC, s=ms, label=labb
                )

                if cont == 0:
                    labb = (
                        labZX
                        + "$\\eta_{ITG/ETG}$,"
                        + f" $\\rho_N={self.rhos[irho_cont]:.4f}$"
                    )
                else:
                    labb = ""
                ax.plot(x[irho], eta2[irho], "-", c=colorLine, lw=0.5)
                ax.scatter(
                    x[irho], eta2[irho], marker="s", facecolor=colorsC, s=ms, label=labb
                )

                if positionBase is not None:
                    ax.axvline(
                        x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                    )

                ax = ax2_01
                if cont == 0:
                    labb = (
                        labZX
                        + "$\\gamma_{ITG}$,"
                        + f" $\\rho_N={self.rhos[irho_cont]:.4f}$"
                    )
                else:
                    labb = ""

                ax.plot(x[irho], itg[irho], "-", c=colorLine, lw=0.5)
                ax.scatter(
                    x[irho], itg[irho], marker="o", facecolor=colorsC, s=ms, label=labb
                )

                if cont == 0:
                    labb = (
                        labZX
                        + "$\\gamma_{TEM}$,"
                        + f" $\\rho_N={self.rhos[irho_cont]:.4f}$"
                    )
                else:
                    labb = ""

                ax.plot(x[irho], tem[irho], "-", c=colorLine, lw=0.5)
                ax.scatter(
                    x[irho], tem[irho], marker="s", facecolor=colorsC, s=ms, label=labb
                )

                if cont == 0:
                    labb = (
                        labZX
                        + "$\\gamma_{ETG}$,"
                        + f" $\\rho_N={self.rhos[irho_cont]:.4f}$"
                    )
                else:
                    labb = ""

                ax.plot(x[irho], etg[irho], "-", c=colorLine, lw=0.5)
                ax.scatter(
                    x[irho], etg[irho], marker="x", facecolor=colorsC, s=ms, label=labb
                )

                if positionBase is not None:
                    ax.axvline(
                        x=x[irho][positionBase], ls="--", c=scan_colors[0], lw=1.0
                    )

                cont += 1

            colorsChosen.append(colorsChosen0)

        if unnormalization_successful:
            ax = ax1_00
            ax.set_xlabel(variableLabel)
            ax.set_ylabel("$Q_e$ ($MW/m^2$)")
            ax.legend(loc="best", fontsize=fontsizeLeg)
            ax.set_ylim(bottom=0)
            ax.set_title("Electron heat flux")
            GRAPHICStools.addDenseAxis(ax)

            ax = ax1_10
            ax.set_xlabel(variableLabel)
            ax.set_ylabel("$Q_i$ ($MW/m^2$)")
            ax.legend(loc="best", fontsize=fontsizeLeg)
            ax.set_ylim(bottom=0)
            ax.set_title("Ion heat flux")
            GRAPHICStools.addDenseAxis(ax)

            ax = ax1_20
            ax.set_xlabel(variableLabel)
            ax.set_ylabel("$\\Gamma_e$ ($1E20/s/m^2$)")
            ax.legend(loc="best", fontsize=fontsizeLeg)
            ax.axhline(y=0, ls="-.", c="k", lw=1)
            ax.set_title("Electron particle flux")
            GRAPHICStools.addDenseAxis(ax)

        ax = ax1_00e
        ax.set_xlabel(variableLabel)
        ax.set_ylabel("$Q_e$ (GB)")
        ax.legend(loc="best", fontsize=fontsizeLeg)
        ax.set_ylim(bottom=0)
        ax.set_title("Electron heat flux")
        GRAPHICStools.addDenseAxis(ax)

        ax = ax1_10e
        ax.set_xlabel(variableLabel)
        ax.set_ylabel("$Q_i$ (GB)")
        # ax.legend(loc='best')
        ax.set_ylim(bottom=0)
        ax.set_title("Ion heat flux")
        GRAPHICStools.addDenseAxis(ax)

        ax = ax1_20e
        ax.set_xlabel(variableLabel)
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        # ax.legend(loc='best')
        ax.axhline(y=0, ls="--", c="k", lw=1)
        ax.set_title("Electron particle flux")
        GRAPHICStools.addDenseAxis(ax)

        ax = ax2_11
        ax.set_xlabel(variableLabel)
        ax.set_ylabel("$\\eta_{a/b}=max(\\gamma_{a})/max(\\gamma_{b})$")
        ax.legend(loc="best", fontsize=fontsizeLeg)
        ax.set_ylim(bottom=0)
        ax.set_title("Dominant turbulence")
        ax.axhline(y=1.0, ls="--", c="k", lw=0.5)
        GRAPHICStools.addDenseAxis(ax)

        ax = ax2_01
        ax.set_xlabel(variableLabel)
        ax.set_ylabel("$\\gamma$ ($c_s/a$)")
        ax.legend(loc="best", fontsize=fontsizeLeg)
        ax.set_ylim(bottom=0)
        ax.set_title("Largest growth rate")
        GRAPHICStools.addDenseAxis(ax)

        ax = ax2_00
        ax.set_xlim([1e-2, 3e1])
        ax.legend(loc="best", fontsize=fontsizeLeg)
        GRAPHICStools.addDenseAxis(ax)

        ax = ax2_10
        GRAPHICStools.addDenseAxis(ax)

        # --------------------------------------------------------
        # Plot full TGLFs
        # --------------------------------------------------------

        total_plots = 0
        for ikey in labels:
            total_plots += self.scans[ikey]["Qe_gb"].size

        if plotTGLFs and (
            total_plots < 10
            or (
                print(
                    f">> TGLFscan module wants to *also* plot {total_plots} individual TGLF, are you sure you want to do this?",
                    typeMsg="q",
                )
            )
        ):
            for contLabel, ikey in enumerate(labels):

                labelsExtraPlot = self.scans[ikey]["results_tags"]
                labels_legend = [lab.split("_")[-1] for lab in labelsExtraPlot]

                colorsC = np.transpose(
                    np.array(colorsChosen[contLabel]), axes=(1, 0, 2)
                )
                colorsC_tuple = []
                for i in range(colorsC.shape[0]):
                    for j in range(colorsC.shape[1]):
                        colorsC_tuple.append(tuple(colorsC[i, j, :]))

                self.plot(
                    extratitle=f"{ikey} - ",
                    labels=labelsExtraPlot,
                    labels_legend=labels_legend,
                    title_legend=ikey,
                    fn=self.fn,
                    colors=colorsC_tuple,
                    plotNormalizations=False,
                    plotGACODE=False,
                    fn_color=contLabel,
                    WarnMeAboutPlots=False,  # Do not warn me because I have already accepted/declined
                )

    # ~~~~~~~~~~~~~~ Extra complete analysis options

    def runScanTurbulenceDrives(
        self,
        subFolderTGLF="drives1",
        varUpDown = None,           # This setting supercedes the resolutionPoints and variation
        resolutionPoints=5,
        variation=0.5,
        add_baseline_to = 'all', # 'all' or 'first' or 'none'
        add_also_baseline_to_first = True,
        variablesDrives=["RLTS_1", "RLTS_2", "RLNS_1", "XNUE", "TAUS_2"],
        **kwargs_TGLFrun,
    ):

        self.variablesDrives = variablesDrives

        if varUpDown is None:
            varUpDown = np.linspace(1 - variation, 1 + variation, resolutionPoints)

        varUpDown_dict = {}
        for i,variable in enumerate(self.variablesDrives):
            if add_baseline_to == 'all' or (add_baseline_to == 'first' and i == 0):
                varUpDown_dict[variable] = np.append(1.0, varUpDown)
            else:
                varUpDown_dict[variable] = varUpDown

        # ------------------------------------------
        # Prepare all scans
        # ------------------------------------------

        tglf_executor, tglf_executor_full, folders = {}, {}, []
        for cont, variable in enumerate(self.variablesDrives):
            # Only ask the cold_start in the first round
            kwargs_TGLFrun["forceIfcold_start"] = cont > 0 or (
                "forceIfcold_start" in kwargs_TGLFrun and kwargs_TGLFrun["forceIfcold_start"]
            )

            scan_name = f"{subFolderTGLF}_{variable}"  # e.g. turbDrives_RLTS_1

            tglf_executor0, tglf_executor_full0, folders0, _ = self._prepare_scan(
                scan_name,
                variable=variable,
                varUpDown=varUpDown_dict[variable],
                **kwargs_TGLFrun,
            )

            tglf_executor = tglf_executor | tglf_executor0
            tglf_executor_full = tglf_executor_full | tglf_executor_full0
            folders += folders0

        # ------------------------------------------
        # Run them all
        # ------------------------------------------

        self._run(
            tglf_executor,
            tglf_executor_full=tglf_executor_full,
            **kwargs_TGLFrun,
        )

        # ------------------------------------------
        # Read results
        # ------------------------------------------

        cont = 0
        for variable in self.variablesDrives:
            for mult in varUpDown_dict[variable]:
                name = f"{variable}_{mult}"
                self.read(
                    label=f"{self.subFolderTGLF_scan}_{name}", folder=folders[cont], cold_startWF = False
                )
                cont += 1

            scan_name = f"{subFolderTGLF}_{variable}"  # e.g. turbDrives_RLTS_1

            self.readScan(label=scan_name, variable=variable)

    def plotScanTurbulenceDrives(
        self, label="drives1", figs=None, **kwargs_TGLFscanPlot
    ):
        labels = []
        for variable in self.variablesDrives:
            labels.append(f"{label}_{variable}")

        if figs is None:
            self.fn = GUItools.FigureNotebook(
                "TGLF Drives MITIM Notebook", geometry="1500x900", vertical=True
            )
            if self.NormalizationSets["SELECTED"] is not None:
                fig1 = self.fn.add_figure(label="Fluxes - Relative")
            else:
                fig1 = None
            fig2 = self.fn.add_figure(label="Fluxes (GB) - Relative")
            fig3 = self.fn.add_figure(label="Linear Stability - Relative")
            figs1 = [fig1, fig2, fig3]
            if self.NormalizationSets["SELECTED"] is not None:
                fig1 = self.fn.add_figure(label="Fluxes")
            else:
                fig1 = None
            fig2 = self.fn.add_figure(label="Fluxes (GB)")
            fig3 = self.fn.add_figure(label="Linear Stability")
            figs2 = [fig1, fig2, fig3]
        else:
            figs1, figs2 = None, None

        kwargs_TGLFscanPlot.pop("figs", None)

        self.plotScan(
            labels=labels,
            figs=figs1,
            variableLabel="X",
            relativeX=True,
            **kwargs_TGLFscanPlot,
        )

        kwargs_TGLFscanPlot["plotTGLFs"] = False
        self.plotScan(
            labels=labels, figs=figs2, variableLabel="X", **kwargs_TGLFscanPlot
        )

    def runAnalysis(
        self,
        subFolderTGLF="analysis1",
        label="analysis1",
        analysisType="chi_e",
        trace=[50.0, 174.0],
        **kwargs_TGLFrun,
    ):
        if self.NormalizationSets["SELECTED"] is None:
            raise Exception(
                "MITIM Exception: No normalizations provided, but runAnalysis will require it!"
            )

        # ------------------------------------------
        # Electron thermal incremental diffusivity
        # ------------------------------------------
        if (
            analysisType == "chi_e"
            or analysisType == "chi_i"
            or analysisType == "chi_ei"
        ):
            if analysisType == "chi_e":
                print(
                    "*** Running analysis of electron thermal incremental diffusivity"
                )
                self.variable, self.variable_y = "RLTS_1", "Qe"

            elif analysisType == "chi_i":
                print("*** Running analysis of ion thermal incremental diffusivity")
                self.variable, self.variable_y = "RLTS_2", "Qi"

            elif analysisType == "chi_ei":
                print(
                    "*** Running analysis of cross thermal incremental diffusivity (ion temperature gradient on electron heat flux)"
                )
                self.variable, self.variable_y = "RLTS_2", "Qe"

            variation = 0.10
            # Run 5 below, 5 above, and base case
            varUpDown = np.append(
                np.linspace(1 - variation / 2, 1, 6),
                np.linspace(1, 1 + variation / 2, 6)[1:],
            )

            self.runScan(
                subFolderTGLF,
                varUpDown=varUpDown,
                variable=self.variable,
                **kwargs_TGLFrun,
            )

            self.readScan(label=label, variable=self.variable)

            if analysisType == "chi_e":
                Te_prof = self.NormalizationSets["SELECTED"]["Te_keV"]
                ne_prof = self.NormalizationSets["SELECTED"]["ne_20"]

            elif analysisType == "chi_i":
                Te_prof = self.NormalizationSets["SELECTED"]["Ti_keV"]
                ne_prof = self.NormalizationSets["SELECTED"]["ni_20"]

            elif analysisType == "chi_ei":
                Te_prof = self.NormalizationSets["SELECTED"]["Ti_keV"]
                ne_prof = self.NormalizationSets["SELECTED"]["ne_20"]

            rho = self.NormalizationSets["SELECTED"]["rho"]
            a = self.NormalizationSets["SELECTED"]["rmin"][-1]

            x = self.scans[label]["xV"]
            yV = self.scans[label][self.variable_y]

            self.scans[label]["chi_inc"] = []
            self.scans[label]["chi_eff"] = []
            self.scans[label]["chi_pb"] = []
            self.scans[label]["Vpinch"] = []
            self.scans[label]["x_grid"] = []
            self.scans[label]["y_grid"] = []
            for irho_cont in range(len(self.rhos)):
                irho = np.where(self.scans[label]["x"] == self.rhos[irho_cont])[0][0]

                Te_keV = Te_prof[np.argmin(np.abs(rho - self.rhos[irho_cont]))]
                ne_20 = ne_prof[np.argmin(np.abs(rho - self.rhos[irho_cont]))]
                a_m = a

                aLTe_base = x[irho][self.scans[label]["positionBase"]]

                Chi_inc, x_grid, y_grid, Chi_pb, Vpinch, Chi_eff = PLASMAtools.chi_inc(
                    x[irho], yV[irho], Te_keV, a_m, ne_20, aLTe_base
                )
                self.scans[label]["chi_inc"].append(Chi_inc)
                self.scans[label]["chi_pb"].append(Chi_pb)
                self.scans[label]["chi_eff"].append(Chi_eff)
                self.scans[label]["Vpinch"].append(Vpinch)
                self.scans[label]["x_grid"].append(x_grid)
                self.scans[label]["y_grid"].append(y_grid)

            self.scans[label]["var_x"] = self.variable
            self.scans[label]["var_y"] = self.variable_y

        # ------------------------------------------
        # Impurity D and V
        # ------------------------------------------
        elif analysisType == "Z":
            if ("ApplyCorrections" not in kwargs_TGLFrun) or (
                kwargs_TGLFrun["ApplyCorrections"]
            ):
                print(
                    "\t- Forcing ApplyCorrections=False because otherwise the species orderingin TGLF file might be messed up",
                    typeMsg="w",
                )
                kwargs_TGLFrun["ApplyCorrections"] = False

            varUpDown = np.linspace(0.5, 1.5, 3)

            fimp, Z, A = 1e-6, trace[0], trace[1]

            print(
                f"*** Running D and V analysis for trace ({fimp:.1e}) specie with Z={trace[0]:.1f}, A={trace[1]:.1f}"
            )

            self.inputsTGLF_orig = copy.deepcopy(self.inputsTGLF)

            # ------------------------
            # Add trace impurity
            # ------------------------

            for irho in self.inputsTGLF:
                position = self.inputsTGLF[irho].addTraceSpecie(Z, A, AS=fimp)

            self.variable = f"RLNS_{position}"

            self.runScan(
                subFolderTGLF,
                varUpDown=varUpDown,
                variable=self.variable,
                **kwargs_TGLFrun,
            )

            self.readScan(label=label, variable=self.variable, positionIon=position)

            x = self.scans[label]["xV"]
            yV = self.scans[label]["Gi"]
            self.variable_y = "Gi"

            self.scans[label]["DZ"] = []
            self.scans[label]["VZ"] = []
            self.scans[label]["VoD"] = []
            self.scans[label]["x_grid"] = []
            self.scans[label]["y_grid"] = []
            for irho_cont in range(len(self.rhos)):
                irho = np.where(self.scans[label]["x"] == self.rhos[irho_cont])[0][0]

                rho = self.NormalizationSets["SELECTED"]["rho"]
                a_m = self.NormalizationSets["SELECTED"]["rmin"][-1]
                ni_prof = self.NormalizationSets["SELECTED"]["ne_20"] * fimp

                ni_20 = ni_prof[np.argmin(np.abs(rho - self.rhos[irho_cont]))]

                D, V, x_grid, y_grid = PLASMAtools.DVanalysis(
                    x[irho], yV[irho], a_m, ni_20
                )
                self.scans[label]["DZ"].append(D)
                self.scans[label]["VZ"].append(V)

                self.scans[label]["var_x"] = self.variable
                self.scans[label]["var_y"] = self.variable_y
                self.scans[label]["x_grid"].append(x_grid)
                self.scans[label]["y_grid"].append(y_grid)

                self.scans[label]["VoD"].append(V / D)

            # Back to original (not trace)
            self.inputsTGLF = self.inputsTGLF_orig

    def plotAnalysis(self, labels=["analysis1"], analysisType="chi_e", figs=None):
        if figs is None:
            self.fn = GUItools.FigureNotebook(
                "TGLF Analysis MITIM Notebook", geometry="1500x900"
            )
            fig1 = self.fn.add_figure(label="Analysis")
            fig2 = self.fn.add_figure(label="Fluxes")
            fig2e = self.fn.add_figure(label="Fluxes (GB)")
            fig3 = self.fn.add_figure(label="Linear Stability")

        if analysisType == "chi_e":
            variableLabel = "RLTS_1"
        elif analysisType == "chi_ei":
            variableLabel = "RLTS_2"
        elif analysisType == "Z":
            variableLabel = self.variable
        self.plotScan(
            labels=labels, figs=[fig2, fig2e, fig3], variableLabel=variableLabel
        )

        colors = GRAPHICStools.listColors()
        fontsizeLeg = 15
        colorsLines = colors  # [10:]

        if (
            analysisType == "chi_e"
            or analysisType == "chi_i"
            or analysisType == "chi_ei"
        ):
            grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
            ax00 = fig1.add_subplot(grid[0, 0])
            ax01 = fig1.add_subplot(grid[0, 1])
            ax10 = fig1.add_subplot(grid[1, 0])
            ax11 = fig1.add_subplot(grid[1, 1])

            cont = 0
            x_max = -np.inf
            y_max = -np.inf
            x_min = np.inf
            y_min = np.inf
            for contLab, label in enumerate(labels):
                if len(labels) > 1:
                    labZX = f"{label} "
                else:
                    labZX = ""

                for irho_cont in range(len(self.rhos)):
                    irho = np.where(self.scans[label]["x"] == self.rhos[irho_cont])[0][
                        0
                    ]

                    labelA = (
                        labZX
                        + "$\\rho_N={0:.4f}$, $\\chi^{{inc}}={1:.2f}$ $m^2/s$".format(
                            self.rhos[irho_cont], self.scans[label]["chi_inc"][irho]
                        )
                    )

                    xV = self.scans[label]["xV"][irho]
                    Qe = self.scans[label][self.scans[label]["var_y"]][irho]
                    xgrid = self.scans[label]["x_grid"][irho]
                    ygrid = np.array(self.scans[label]["y_grid"][irho])
                    xba = self.scans[label]["xV"][irho][
                        self.scans[label]["positionBase"]
                    ]
                    yba = np.interp(xba, xV, Qe)

                    ax = ax00
                    ax.plot(xV, Qe, "o-", c=colors[cont], label=labelA)
                    ix1 = np.argmin(np.abs(xgrid - np.min(xV) * 0.8))
                    ix2 = np.argmin(np.abs(xgrid - np.max(xV) * 1.2))
                    ax.plot(
                        xgrid[ix1:ix2], ygrid[ix1:ix2], "--", lw=0.5, c=colors[cont]
                    )
                    ax.axvline(x=xba, ls="--", c=colors[cont], lw=1.0)
                    ax.plot([0, xba], [0, yba], "-.", lw=0.2, c=colors[cont])

                    ax = ax10
                    ax.plot(xV / xba, Qe / yba, "o-", c=colors[cont], label=labelA)
                    ax.plot(xgrid / xba, ygrid / yba, "--", lw=0.5, c=colors[cont])

                    cont += 1

                    x_max = np.max([x_max, np.max(xV)])
                    x_min = np.min([x_min, np.min(xV)])
                    y_max = np.max([y_max, np.max(Qe)])
                    y_min = np.min([y_min, np.min(Qe)])

                ax = ax01
                colorLab = colors[
                    len(self.rhos) * contLab : len(self.rhos) * (contLab + 1)
                ]

                rho = self.rhos
                chi_inc = self.scans[label]["chi_inc"]
                chi_pb = self.scans[label]["chi_pb"]
                chi_eff = self.scans[label]["chi_eff"]
                Vpinch = self.scans[label]["Vpinch"]

                if contLab == 0:
                    labb = labZX + "$\\chi^{inc}$"
                else:
                    labb = ""
                ax.plot(rho, chi_inc, c=colorsLines[contLab], ls="-", lw=2, label=labb)
                ax.scatter(rho, chi_inc, facecolor=colorLab, s=20, marker="o")
                if contLab == 0:
                    labb = labZX + "$\\chi^{eff}$"
                else:
                    labb = ""
                ax.plot(rho, chi_eff, c=colorsLines[contLab], ls="--", lw=2, label=labb)
                ax.scatter(rho, chi_eff, facecolor=colorLab, s=20, marker="s")

                ax = ax11

                if contLab == 0:
                    labb = labZX + "$\\chi^{PB}$"
                else:
                    labb = ""
                ax.plot(rho, chi_pb, c=colorsLines[contLab], ls="-", lw=2, label=labb)
                ax.scatter(rho, chi_pb, facecolor=colorLab, s=20, marker="o")
                if contLab == 0:
                    labb = labZX + "$V^{PB}$"
                else:
                    labb = ""
                ax.plot(rho, Vpinch, c=colorsLines[contLab], ls="--", lw=2, label=labb)
                ax.scatter(rho, Vpinch, facecolor=colorLab, s=20, marker="s")

            ax = ax00
            ax.set_xlabel(
                f"Inv. Norm. Gradient Scale Length: {self.scans[label]['var_x']}"
            )
            ax.set_ylabel(f"Transport flux {self.scans[label]['var_y']}")
            ax.set_title("Incremental diffusivity calculations")
            # if len(labels) > 1 or len(self.rhos) > 1:
            ax.legend(fontsize=8, loc="best")
            GRAPHICStools.addDenseAxis(ax)

            ax.set_xlim([x_min * 0.9, x_max * 1.1])
            ax.set_ylim([y_min * 0.9, y_max * 1.1])

            ax = ax10
            ax.set_xlabel(
                "RELATIVE Inv. Norm. Gradient Scale Length: {0}".format(
                    self.scans[label]["var_x"]
                )
            )
            ax.set_ylabel(f"RELATIVE Transport flux {self.scans[label]['var_y']}")
            ax.axvline(x=1.0, ls="-", c="k", lw=0.3)
            ax.axhline(y=1.0, ls="-", c="k", lw=0.3)

            ax.set_xlim([0.8, 1.2])
            ax.set_ylim([0.5, 2.0])
            GRAPHICStools.addDenseAxis(ax)

            ax = ax01
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$\\chi^{inc}$ ($m^2/s$)")
            ax.set_xlim([0, 1])
            # ax.set_ylim(bottom=0)
            ax.set_title("Diffusivity profiles (effective and incremental)")
            ax.legend(fontsize=10, loc="best")
            ax.axhline(y=0.0, ls="-", c="k", lw=0.3)
            GRAPHICStools.addDenseAxis(ax)

            ax = ax11
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$\\chi^{PB}$ ($m^2/s$), $V^{PB}$ ($m/s$)")
            ax.set_xlim([0, 1])
            ax.set_title("Phenomenological (diffusivity + pinch)")
            ax.legend(fontsize=10, loc="best")
            ax.axhline(y=0.0, ls="-", c="k", lw=0.3)
            GRAPHICStools.addDenseAxis(ax)

        elif analysisType == "Z":
            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
            ax00 = fig1.add_subplot(grid[:, 0])
            ax01 = fig1.add_subplot(grid[0, 1])
            ax11 = fig1.add_subplot(grid[1, 1])
            ax02 = fig1.add_subplot(grid[0, 2])
            ax12 = fig1.add_subplot(grid[1, 2])

            cont = 0
            for contLab, label in enumerate(labels):
                if len(labels) > 1:
                    labZX = f"{label} "
                else:
                    labZX = ""

                colorLab = colors[
                    len(self.rhos) * contLab : len(self.rhos) * (contLab + 1)
                ]

                for irho_cont in range(len(self.rhos)):
                    irho = np.where(self.scans[label]["x"] == self.rhos[irho_cont])[0][
                        0
                    ]

                    col = colorLab[irho]

                    labelA = (
                        labZX
                        + "$\\rho_N={0:.4f}$, $D={1:.2f}$ $m^2/s$, $V={2:.2f}$ $m/s$".format(
                            self.rhos[irho_cont],
                            self.scans[label]["DZ"][irho],
                            self.scans[label]["VZ"][irho],
                        )
                    )

                    ax = ax00
                    ax.plot(
                        self.scans[label]["xV"][irho],
                        np.array(self.scans[label]["Gi"][irho]),
                        "o-",
                        c=col,
                        label=labelA,
                    )
                    ax.plot(
                        self.scans[label]["x_grid"][irho],
                        self.scans[label]["y_grid"][irho],
                        "--",
                        lw=0.5,
                        c=col,
                    )
                    # ax.axvline(x=self.scans[label]['xV'][irho][self.scans[label]['positionBase']],ls='--',c=col,lw=1.)

                    cont += 1

                ax = ax01
                rho = self.rhos
                y = self.scans[label]["DZ"]
                ax.plot(rho, y, c=colorsLines[contLab], ls="-", lw=2)
                ax.scatter(rho, y, facecolor=colorLab, s=20, marker="o")

                ax = ax11

                y = self.scans[label]["VZ"]
                ax.plot(rho, y, c=colorsLines[contLab], ls="-", lw=2)
                ax.scatter(rho, y, facecolor=colorLab, s=20, marker="o")

                ax = ax02

                a_m = self.NormalizationSets["SELECTED"]["rmin"][-1]
                y = -a_m * np.array(self.scans[label]["VoD"])
                ax.plot(rho, y, c=colorsLines[contLab], ls="-", lw=2)
                ax.scatter(rho, y, facecolor=colorLab, s=20, marker="o")

                ax = ax12

                rho_mod = np.append([0], rho)
                aLn = np.append([0], y)
                import torch
                from mitim_modules.powertorch.physics import CALCtools

                BC = 1.0
                T = CALCtools.integrateGradient(
                    torch.from_numpy(rho_mod).unsqueeze(0),
                    torch.from_numpy(aLn).unsqueeze(0),
                    BC,
                )[0]
                ax.plot(rho_mod, T, c=colorsLines[contLab], ls="-", lw=2)
                ax.scatter(rho_mod, T, s=20, marker="o")

            ax = ax00
            ax.axhline(y=0, ls="--", c="k", lw=0.5)
            ax.axvline(x=0, ls="--", c="k", lw=0.5)
            ax.set_xlabel(
                f"Inv. Norm. Gradient Scale Length: {self.scans[label]['var_x']}"
            )
            ax.set_ylabel(f"Transport flux {self.scans[label]['var_y']}")
            ax.set_title("D and V calculations")
            ax.legend(fontsize=7, loc="best")
            GRAPHICStools.addDenseAxis(ax)

            ax = ax11
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("V ($m/s$)")
            ax.set_xlim([0, 1])
            ax.axhline(y=0, lw=1, ls="--", c="k")
            ax.set_title("V pinch profiles")
            GRAPHICStools.addDenseAxis(ax)

            ax = ax01
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("D ($m^2/s$)")
            ax.set_xlim([0, 1])
            ax.axhline(y=0, lw=1, ls="--", c="k")
            ax.set_title("D diffusivity profiles")
            GRAPHICStools.addDenseAxis(ax)

            ax = ax02
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$a\\cdot V/D = a/L_{nZ}$")
            ax.set_xlim([0, 1])
            ax.axhline(y=0, lw=1, ls="--", c="k")
            ax.set_title("Gradients at zero-flux condition")
            GRAPHICStools.addDenseAxis(ax)

            ax = ax12
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("Relative $n_Z$")
            ax.set_xlim([0, 1])
            ax.set_title(f"Integrated profile using BC={BC}")
            GRAPHICStools.addDenseAxis(ax)

    def updateConvolution(self):
        self.DRMAJDX_LOC = {}
        if "latest_inputsFileTGLFDict" not in self.__dict__:
            for i in self.rhos:
                self.DRMAJDX_LOC[i] = 0.0
            print(
                "\t- [convolution] Using DRMAJDX_LOC =0 because no input file was stored",
                typeMsg="i",
            )
        else:
            for i in self.latest_inputsFileTGLFDict:
                if "DRMAJDX_LOC" in self.latest_inputsFileTGLFDict[i].geom:
                    self.DRMAJDX_LOC[i] = self.latest_inputsFileTGLFDict[i].geom[
                        "DRMAJDX_LOC"
                    ]
                else:
                    self.DRMAJDX_LOC[i] = 0.0
                    print(
                        " ~~ Using DRMAJDX_LOC=0 because stored input to tglf had no DRMAJDX_LOC",
                        typeMsg="w",
                    )

        self.convolution_fun_fluct, self.factorTot_to_Perp = None, 1.0
        if self.d_perp_dict is not None:
            (
                self.convolution_fun_fluct,
                self.factorTot_to_Perp,
            ) = GACODEdefaults.convolution_CECE(self.d_perp_dict, dRdx=self.DRMAJDX_LOC)


def completeVariation(setVariations, species):
    ions_info = species.ions_info

    setVariations_new = copy.deepcopy(setVariations)

    for variable in setVariations:
        if variable == "RLTS_2":
            print(
                " \t- Varying temperature gradients of all thermal ions equally during scan",
                typeMsg="i",
            )
            for i in ions_info["thermal_list_extras"]:
                setVariations_new[f"RLTS_{i}"] = setVariations[variable]
        if variable == "RLNS_1":
            print(
                " \t- Varying density gradients of all thermal ions equally to electrons during scan",
                typeMsg="i",
            )
            setVariations_new["RLNS_2"] = setVariations[variable]
            for i in ions_info["thermal_list_extras"]:
                setVariations_new[f"RLNS_{i}"] = setVariations[variable]
        if variable == "TAUS_2":
            print(
                " \t- Varying temperatures of all thermal ions equally during scan",
                typeMsg="i",
            )
            for i in ions_info["thermal_list_extras"]:
                setVariations_new[f"TAUS_{i}"] = setVariations[variable]
        if variable == "VEXB_SHEAR":
            print(
                " \t- Varying all rotation shears of thermal ions equal during scan",
                typeMsg="i",
            )
            setVariations_new["VPAR_SHEAR_1"] = setVariations[variable]
            for i in ions_info["thermal_list"]:
                setVariations_new[f"VPAR_SHEAR_{i}"] = setVariations[variable]

    return setVariations_new


# ~~~~~~~~~ Input class


def changeANDwrite_TGLF(
    rhos,
    inputs0,
    FolderTGLF,
    TGLFsettings=None,
    extraOptions={},
    multipliers={},
    ApplyCorrections=True,
    Quasineutral=False,
):
    """
    Received inputs classes and gives text.
    ApplyCorrections refer to removing ions with too low density and that are fast species
    """

    inputs = copy.deepcopy(inputs0)

    modInputTGLF = {}
    ns_max = []
    for i, rho in enumerate(rhos):
        print(f"\t- Changing input file for rho={rho:.4f}")
        NS = inputs[rho].plasma["NS"]
        inputTGLF_rho = GACODErun.modifyInputs(
            inputs[rho],
            Settings=TGLFsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            position_change=i,
            addControlFunction=GACODEdefaults.addTGLFcontrol,
            NS=NS,
        )

        newfile = FolderTGLF / f"input.tglf_{rho:.4f}"

        if TGLFsettings is not None:
            # Apply corrections
            if ApplyCorrections:
                print("\t- Applying corrections")
                inputTGLF_rho.removeLowDensitySpecie()
                inputTGLF_rho.removeFast()

            # Ensure that plasma to run is quasineutral
            if Quasineutral:
                inputTGLF_rho.ensureQuasineutrality()
        else:
            print('\t- Not applying corrections nor quasineutrality because "TGLFsettings" is None')

        inputTGLF_rho.writeCurrentStatus(file=newfile)

        modInputTGLF[rho] = inputTGLF_rho

        ns_max.append(inputs[rho].plasma["NS"])

    # Convert back to a string because that's how runTGLFproduction operates
    inputFileTGLF = inputToVariable(FolderTGLF, rhos)

    if (np.diff(ns_max) > 0).any():
        print(
            "> Each radial location has its own number of species... probably because of removal of fast or low density...",
            typeMsg="w",
        )
        print(
            "\t * Reading of TGLF results will fail... consider doing something before launching run",
            typeMsg="q",
        )

    return inputFileTGLF, modInputTGLF


def reduceToControls(dict_all):
    controls, plasma, geom = {}, {}, {}
    for ikey in dict_all:
        if ikey in [
            "VEXB",
            "VEXB_SHEAR",
            "BETAE",
            "XNUE",
            "ZEFF",
            "DEBYE",
            "SIGN_BT",
            "SIGN_IT",
            "NS",
        ]:
            plasma[ikey] = dict_all[ikey]

        elif (len(ikey.split("_")) > 1) and (ikey.split("_")[-1] in ["SA", "LOC"]):
            geom[ikey] = dict_all[ikey]

        else:
            controls[ikey] = dict_all[ikey]

    return controls, plasma, geom


class TGLFinput:
    def __init__(self, file=None):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None

        if self.file is not None and self.file.exists():
            with open(self.file, "r") as f:
                lines = f.readlines()
            file_txt = "".join(lines)
        else:
            file_txt = ""
        input_dict = GACODErun.buildDictFromInput(file_txt)

        self.process(input_dict)

    @classmethod
    def initialize_in_memory(cls, input_dict):
        instance = cls()
        instance.process(input_dict)
        return instance

    def process(self, input_dict):

        # Get number of recorded species
        self.num_recorded = 0
        if "NS" in input_dict:
            self.num_recorded = int(input_dict["NS"])

        # Species -----------
        self.species = {}
        species_vars = [
            "ZS",
            "MASS",
            "RLNS",
            "RLTS",
            "TAUS",
            "AS",
            "VPAR",
            "VPAR_SHEAR",
            "VNS_SHEAR",
            "VTS_SHEAR",
        ]
        controls_all = copy.deepcopy(input_dict)

        for i in range(self.num_recorded):
            specie = {}
            for var in species_vars:
                varod = f"{var}_{i+1}"
                if varod in input_dict:
                    specie[var] = input_dict[varod]
                    del controls_all[varod]
                else:
                    specie[var] = 0.0
            self.species[i + 1] = specie

        self.controls, self.plasma, self.geom = reduceToControls(controls_all)
        self.processSpecies()

    def processSpecies(self, MinMultiplierToBeFast=2.0):
        """
        if MinMultiplierToBeFast = 2, species with 2x the first ion temperature are considered fast

        Note that species are ordered as they are indicated in the TGLF input file:

                This means that species[1] = electrons, species[2] = first ion, etc.

        """

        if 1 in self.species:
            # Define maximum Ti/Te to be considered thermal species
            TiTe_main = self.species[2]["TAUS"]  # First ion is 2 (e.g. RLTS_2)
            thrTemperatureRatio = MinMultiplierToBeFast * TiTe_main

            # By convention
            self.ions_info = {
                1: {"name": "electrons", "type": "thermal"},
                2: {"name": "main", "type": "thermal"},
            }

            thermal_indeces = [1, 2]
            for i in range(len(self.species) - 2):
                TiTe = self.species[3 + i]["TAUS"]
                if TiTe < thrTemperatureRatio:
                    self.ions_info[3 + i] = {"type": "thermal"}
                    thermal_indeces.append(3 + i)
                else:
                    self.ions_info[3 + i] = {"type": "fast"}

            self.ions_info["thermal_list"] = thermal_indeces
            self.ions_info["thermal_list_extras"] = thermal_indeces[
                2:
            ]  # remove electrons and mains

            self.onlyControl = False
        else:
            print(
                "\t- No species in this input.tglf (it is either a controls-only file or there was a problem generating it)"
            )
            self.onlyControl = True

    def isThePlasmaDT(self):
        """
        First two ions are D and T?
        """

        m2, m3 = self.species[2]["MASS"], self.species[3]["MASS"]
        if m2 > m3:
            mrat = m2 / m3
        else:
            mrat = m3 / m2

        return np.abs(mrat - 1.5) < 0.01

    def removeFast(self):
        self.processSpecies()
        i = 1
        while i <= len(self.species):
            if self.ions_info[i]["type"] == "fast":
                self.removeSpecie(
                    indexSpecie=i, reason=" because it is not a thermal specie"
                )
            else:
                i += 1

    def removeLowDensitySpecie(self, minRelativeDensity=1e-8):
        self.processSpecies()
        i = 1
        while i <= len(self.species):
            if self.species[i]["AS"] < minRelativeDensity:
                self.removeSpecie(
                    indexSpecie=i,
                    reason=f" because too low density (nk/ne={self.species[i]['AS']})",
                )
            else:
                i += 1

    def removeSpecie(self, indexSpecie=None, ZS=0, MASS=0, reduceNS=True, reason=""):
        if indexSpecie is None:
            indexSpecie = identifySpecie(self.species, {"ZS": ZS, "MASS": MASS})

        species_new, ions_info_new = {}, {}
        for ikey in self.species:
            if ikey < indexSpecie:
                species_new[ikey] = self.species[ikey]
                ions_info_new[ikey] = self.ions_info[ikey]
            elif ikey > indexSpecie:
                species_new[ikey - 1] = self.species[ikey]
                ions_info_new[ikey - 1] = self.ions_info[ikey]
        print(f"\t\t\t* Species {indexSpecie} removed{reason}", typeMsg="w")
        self.species = species_new
        self.ions_info = ions_info_new

        if reduceNS:
            self.plasma["NS"] -= 1
            if "NMODES" in self.controls:
                self.controls["NMODES"] -= 1
            print(
                f"\t\t\t* Total species to run TGLF with reduced to {self.plasma['NS']}"
            )

        self.num_recorded -= 1

        self.processSpecies()

    def ensureQuasineutrality(self):
        if self.isThePlasmaDT():
            speciesMod = [2, 3]
        else:
            speciesMod = [2]

        diff = self.calcualteQuasineutralityError()
        print(f"\t- Oiriginal quasineutrality error: {diff:.1e}", typeMsg="i")
        print(
            f"\t- Modifying species {speciesMod} to ensure quasineutrality",
            typeMsg="i",
        )
        for i in speciesMod:
            self.species[i]["AS"] -= diff / self.species[i]["ZS"] / len(speciesMod)
        self.processSpecies()
        print(
            "\t- New quasineutrality error: {0:.1e}".format(
                self.calcualteQuasineutralityError()
            ),
            typeMsg="i",
        )

    def calcualteQuasineutralityError(self):
        fiZi = 0
        for i in self.species:
            fiZi += self.species[i]["ZS"] * self.species[i]["AS"]

        return fiZi

    def addTraceSpecie(
        self, ZS, MASS, AS=1e-6, position=None, increaseNS=True, positionCopy=2
        ):
        """
        Here provide ZS and MASS already normalized
        """

        if position is None:
            position = self.num_recorded + 1

        if position > 6:
            print(
                " ***** Exceeded maximum (6) ions in TGLF call, removing last ion",
                typeMsg="w",
            )
            position = 6

        specie = {"ZS": ZS, "MASS": MASS, "AS": AS}

        for ivar in [
            "RLNS",
            "RLTS",
            "TAUS",
            "VPAR",
            "VPAR_SHEAR",
            "VNS_SHEAR",
            "VTS_SHEAR",
        ]:
            specie[ivar] = self.species[positionCopy][ivar]

        self.species[position] = specie
        print(f" ~~ Specie Z = {ZS}, A = {MASS} added with index {position}")

        self.plasma["NS"] += 1
        self.num_recorded += 1

        self.processSpecies()

        return position

    def writeCurrentStatus(self, file=None):
        print("\t- Writting TGLF input file")

        maxSpeciesTGLF = 6  # TGLF cannot handle more than 6 species

        if file is None:
            file = self.file

        with open(file, "w") as f:
            f.write(
                "#-------------------------------------------------------------------------\n"
            )
            f.write(
                "# TGLF input file modified by MITIM framework (Rodriguez-Fernandez, 2020)\n"
            )
            f.write(
                "#-------------------------------------------------------------------------"
            )

            f.write("\n\n# Control parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.controls:
                var = self.controls[ikey]
                f.write(f"{ikey.ljust(23)} = {var}\n")

            f.write("\n\n# Geometry parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.geom:
                var = self.geom[ikey]
                f.write(f"{ikey.ljust(23)} = {var}\n")

            f.write("\n\n# Plasma parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.plasma:
                if ikey == "NS":
                    var = np.min([self.plasma[ikey], maxSpeciesTGLF])
                else:
                    var = self.plasma[ikey]
                f.write(f"{ikey.ljust(23)} = {var}\n")

            f.write("\n\n# Species\n")
            f.write("# -------\n")
            for ikey in self.species:
                if ikey > maxSpeciesTGLF:
                    print(
                        "\t- Maximum number of species in TGLF reached, not considering after {0} species".format(
                            maxSpeciesTGLF
                        ),
                        typeMsg="w",
                    )
                    break
                if ikey == 1:
                    extralab = " (electrons)"
                elif self.ions_info[ikey]["type"] == "fast":
                    extralab = " (fast ion)"
                else:
                    extralab = " (thermal ion)"
                f.write(f"\n# Specie #{ikey}{extralab}\n")
                for ivar in self.species[ikey]:
                    ikar = f"{ivar}_{ikey}"
                    f.write(f"{ikar.ljust(12)} = {self.species[ikey][ivar]}\n")

        print(f"\t\t~ File {IOtools.clipstr(file)} written")

    def plotSpecies(self, axs=None, color="b", legends=True):
        if axs is None:
            plt.ion()
            fig = plt.figure()

            grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.5)

            axs = [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[1, 1]),
            ]

            axs.append(axs[-1].twinx())

        ax = axs[0]

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["RLTS"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "RLTS"
        else:
            label = ""
        ax.plot(x, y, "-o", lw=1, color=color, label=label)

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["RLNS"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "RLNS"
        else:
            label = ""
        ax.plot(x, y, "--*", lw=1, color=color, label=label)
        # ax.set_xticklabels(x,rotation=90,fontsize=8)
        # GRAPHICStools.addXaxis(ax,old_tick_locations,new_tick_locations,label='',whichticks=None)

        ax.legend(loc="best", prop={"size": 7})
        ax.set_title("Normalized gradients")
        # ax.set_xlabel('Specie')
        # GRAPHICStools.addLegendApart(ax,ratio=0.9,withleg=True,extraPad=0,size=7,loc='center left')

        ax = axs[1]

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["TAUS"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "TAUS"
        else:
            label = ""
        ax.plot(x, y, "-o", lw=1, color=color, label=label)

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["AS"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "AS"
        else:
            label = ""
        ax.plot(x, y, "--o", lw=1, color=color, label=label)
        # ax.set_xticklabels(x,rotation=90,fontsize=8)

        ax.legend(loc="best", prop={"size": 7})
        # GRAPHICStools.addLegendApart(ax,ratio=0.9,withleg=True,extraPad=0,size=7,loc='center left')
        ax.set_title("Temperature & Density")
        # ax.set_xlabel('Specie')

        ax = axs[2]

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["ZS"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "ZS"
        else:
            label = ""
        ax.plot(x, y, "-o", lw=1, color=color, label=label)

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["MASS"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "MASS"
        else:
            label = ""
        ax.plot(x, y, "--o", lw=1, color=color, label=label)

        ax.legend(loc="best", prop={"size": 7})
        # GRAPHICStools.addLegendApart(ax,ratio=0.9,withleg=True,extraPad=0,size=7,loc='center left')
        ax.set_title("Mass & Charge")
        ax.set_xlabel("Specie")
        # ax.set_xticklabels(x,rotation=90,fontsize=8)

        ax = axs[3]

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["VPAR_SHEAR"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "VPAR_SHEAR"
        else:
            label = ""
        ax.plot(x, y, "-o", lw=1, color=color, label=label)

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["VNS_SHEAR"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "VNS_SHEAR"
        else:
            label = ""
        ax.plot(x, y, "--^", lw=1, color=color, label=label)

        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["VTS_SHEAR"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "VTS_SHEAR"
        else:
            label = ""
        ax.plot(x, y, "-.*", lw=1, color=color, label=label)

        ax1 = axs[4]
        x, y = [], []
        for i in self.species:
            x.append(i)
            y.append(self.species[i]["VPAR"])
        x, y = np.array(x), np.array(y)
        if legends:
            label = "VPAR"
        else:
            label = ""
        ax1.plot(x, y, "-s", lw=1, color=color, label=label)
        # ax.set_xticklabels(x,rotation=90,fontsize=8)

        ax.legend(loc="upper left", prop={"size": 7})
        ax1.legend(loc="upper right", prop={"size": 7})
        # GRAPHICStools.addLegendApart(ax,ratio=0.9,withleg=True,extraPad=0,size=7,loc='center left')
        # GRAPHICStools.addLegendApart(ax1,ratio=0.9,withleg=True,extraPad=0,size=7,loc='center left')
        ax.set_title("Velocities and Shears")
        ax.set_xlabel("Specie")

    def plotPlasma(self, axs=None, color="b", legends=True):
        if axs is None:
            plt.ion()
            fig = plt.figure()

            grid = plt.GridSpec(2, 1, hspace=0.3, wspace=0.5)

            axs = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]

        ax = axs[0]

        x, y = [], []
        for i in self.plasma:
            x.append(i)
            y.append(self.plasma[i])
        x, y = np.array(x), np.array(y)
        ax.plot(x, y, "-o", lw=1, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90, fontsize=6)
        ax.set_ylabel("PLASMA")

        ax = axs[1]

        x, y = [], []
        for i in self.geom:
            x.append(i)
            y.append(self.geom[i])
        x, y = np.array(x), np.array(y)
        ax.plot(x, y, "-o", lw=1, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90, fontsize=6)
        ax.set_ylabel("GEOMETRY")

    def plotControls(self, axs=None, color="b", markersize=5):
        if axs is None:
            plt.ion()
            fig = plt.figure()

            grid = plt.GridSpec(2, 1, hspace=0.5, wspace=0.5)
            axs = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]

        x1, y1 = [], []
        x2, y2 = [], []
        cont = 0
        x, y = x1, y1

        dicts = [self.controls]  # ,self.geom,self.plasma]

        for dictT in dicts:
            for i in dictT:
                if cont == 30:
                    x, y = x2, y2
                    cont = 0
                x.append(i)
                if type(dictT[i]) == str:
                    y_val0 = dictT[i].split()[0]
                    if y_val0.lower() == "cgyro":
                        y_val = 5
                    elif y_val0.lower() == "gyro":
                        y_val = 0
                    else:
                        y_val = np.nan
                else:
                    y_val = dictT[i]
                y.append(y_val)
                cont += 1

        ax = axs[0]
        ax.plot(x1, y1, "o", color=color, markersize=markersize)
        ax.set_xticks(x1)
        ax.set_xticklabels(x1, rotation=90, fontsize=6)
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        ax.axhline(y=1, ls="--", lw=0.5, c="k")
        ax.set_ylabel("Control Value")

        ax = axs[1]
        ax.plot(x2, y2, "o", color=color, markersize=markersize)
        ax.set_xticks(x2)
        ax.set_xticklabels(x2, rotation=90, fontsize=6)
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        ax.axhline(y=1, ls="--", lw=0.5, c="k")
        ax.set_ylabel("Control Value")


def identifySpecie(dict_species, dict_find):
    found_index = None
    for ikey in dict_species:
        found = True
        for ikey_find in dict_find:
            if dict_find[ikey_find] != dict_species[ikey][ikey_find]:
                found = False and found
        if found:
            print(f" ~~ Species found with index {ikey}")
            found_index = ikey
            break

    return found_index


# From file to dict


def inputToVariable(finalFolder, rhos):
    """
    Entire text file to variable
    """

    inputFilesTGLF = {}
    for cont, rho in enumerate(rhos):
        fileN = finalFolder / f"input.tglf_{rho:.4f}"

        with open(fileN, "r") as f:
            lines = f.readlines()
        inputFilesTGLF[rho] = "".join(lines)

    return inputFilesTGLF


# ~~~~~~~~~~~~~ Functions to handle results


def readTGLFresults(
    FolderGACODE_tmp,
    NormalizationSets,
    rhos,
    convolution_fun_fluct=None,
    factorTot_to_Perp=1.0,
    suffix=None,
):
    TGLFstd_TGLFout, inputclasses, parsed = [], [], []

    for rho in rhos:
        # Read full folder
        TGLFout = TGLFoutput(
            FolderGACODE_tmp, suffix=f"_{rho:.4f}" if suffix is None else suffix
        )

        # Unnormalize
        TGLFout.unnormalize(
            NormalizationSets["SELECTED"],
            rho=rho,
            convolution_fun_fluct=convolution_fun_fluct,
            factorTot_to_Perp=factorTot_to_Perp,
        )

        TGLFstd_TGLFout.append(TGLFout)
        inputclasses.append(TGLFout.inputclass)

        parse = GACODErun.buildDictFromInput(TGLFout.inputFileTGLF)
        parsed.append(parse)

    results = {
        "inputclasses": inputclasses,
        "parsed": parsed,
        "TGLFout": TGLFstd_TGLFout,
        "x": np.array(rhos),
    }

    return results


class TGLFoutput:
    def __init__(self, FolderGACODE, suffix=""):
        self.FolderGACODE, self.suffix = FolderGACODE, suffix

        if suffix == "":
            print(
                f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} without suffix"
            )
        else:
            print(
                f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} with suffix {suffix}"
            )

        self.inputclass = TGLFinput(file=self.FolderGACODE / f"input.tglf{self.suffix}")
        self.roa = self.inputclass.geom["RMIN_LOC"]

        self.read()
        self.postprocess()

        print(
            f"\t- TGLF was run with {self.num_species} species, {self.num_nmodes} modes, {self.num_fields} field(s) ({', '.join(self.fields)}), {self.num_ky} wavenumbers",
        )

    def postprocess(self):
        coeff, klow = 0.0, 0.8
        self.etas = processGrowthRates(
            self.ky,
            self.g[0, :],
            self.f[0, :],
            self.g[1:, :].transpose((1, 0)),
            self.f[1:, :].transpose((1, 0)),
            klow=klow,
            coeff=coeff,
        )

        self.QeES = np.sum(self.SumFlux_Qe_phi)
        self.QeEM = np.sum(self.SumFlux_Qe_a)
        self.QiES = np.sum(self.SumFlux_Qi_phi)
        self.QiEM = np.sum(self.SumFlux_Qi_a)
        self.GeES = np.sum(self.SumFlux_Ge_phi)
        self.GeEM = np.sum(self.SumFlux_Ge_a)

    def read(self):
        # --------------------------------------------------------------------------------
        # Ions to include? e.g. IncludeExtraIonsInQi = [2,3,4] -> This will sum to ion 1
        # --------------------------------------------------------------------------------

        IncludeExtraIonsInQi = (
            [i - 1 for i in self.inputclass.ions_info["thermal_list_extras"]]
            if self.inputclass is not None
            else []
        )
        self.ions_included = (1,) + tuple(IncludeExtraIonsInQi)

        # ------------------------------------------------------------------------
        # Fluxes
        # ------------------------------------------------------------------------
        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.gbflux" + self.suffix),
            blocks=1,
            columns=1,
            numky=None,
        )

        data = np.reshape(data, (4, data.shape[1] // 4))  # [quantiy=G,Q,P,S; specie]

        self.Ge = data[0, 0]
        self.Qe = data[1, 0]

        self.GiAll = data[0, 1:]
        self.QiAll = data[1, 1:]

        print(
            f"\t\t- For Qi, summing contributions from ions {self.ions_included} (#0 is e-)",
            typeMsg="i",
        )
        self.Gi = data[0, self.ions_included].sum()
        self.Qi = data[1, self.ions_included].sum()

        # ------------------------------------------------------------------------
        # Growth rates
        # ------------------------------------------------------------------------

        # Wavenumber grid
        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.ky_spectrum" + self.suffix),
            blocks=1,
            columns=1,
            numky=None,
        )
        self.ky = data[0, :, 0]

        self.num_ky = self.ky.shape[0]

        # Linear stability
        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.eigenvalue_spectrum" + self.suffix),
            blocks=1,
            columns=None,
            numky=self.num_ky,
        )

        self.num_nmodes = int(data.shape[-1] / 2)

        # Using my convention of [quantity=g,f,nmode,ky]
        self.Eigenvalues = (
            data[0, :, :]
            .reshape((self.num_ky, self.num_nmodes, 2))
            .transpose((2, 1, 0))
        )

        self.g = self.Eigenvalues[0, :, :]
        self.f = self.Eigenvalues[1, :, :]

        # ------------------------------------------------------------------------
        # TGLF model
        # ------------------------------------------------------------------------

        width = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.width_spectrum" + self.suffix),
            blocks=1,
            columns=1,
            numky=None,
        )
        spectral_shift = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.spectral_shift_spectrum" + self.suffix),
            blocks=1,
            columns=1,
            numky=None,
        )
        ave_p0 = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.ave_p0_spectrum" + self.suffix),
            blocks=1,
            columns=1,
            numky=None,
        )

        self.tglf_model = {
            "width": width[0, :, 0],
            "spectral_shift": spectral_shift[0, :, 0],
            "ave_p0": ave_p0[0, :, 0],
        }

        # ------------------------------------------------------------------------
        # Fluctuation Spectrum
        # ------------------------------------------------------------------------

        dataT = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.temperature_spectrum" + self.suffix),
            blocks=1,
            columns=None,
            numky=self.num_ky,
        )
        datan = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.density_spectrum" + self.suffix),
            blocks=1,
            columns=None,
            numky=self.num_ky,
        )

        # Using my convention of [quantity=(T,n,nT),species,ky]
        self.AmplitudeSpectrum = np.append(dataT, datan, axis=0).transpose((0, 2, 1))

        self.num_species = self.AmplitudeSpectrum.shape[1]

        self.AmplitudeSpectrum_Te = self.AmplitudeSpectrum[0, 0, :]
        self.AmplitudeSpectrum_Ti = self.AmplitudeSpectrum[0, 1:, :]

        self.AmplitudeSpectrum_ne = self.AmplitudeSpectrum[1, 0, :]
        self.AmplitudeSpectrum_ni = self.AmplitudeSpectrum[1, 1:, :]

        # ------------------------------------------------------------------------
        # Cross Phase Spectrum
        # ------------------------------------------------------------------------

        datanT = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.nsts_crossphase_spectrum" + self.suffix),
            blocks=self.num_species,
            columns=None,
            numky=self.num_ky,
        )

        # Using my convention of [species,nmode,ky]
        self.nTSpectrum = datanT.transpose((0, 2, 1)) * 180 / (np.pi)

        self.neTeSpectrum = self.nTSpectrum[0, :, :]
        self.niTiSpectrum = self.nTSpectrum[1:, :, :]

        # ----------------------------------------------------------------------------------------
        # Field Spectrum (gyro-bohm normalized field fluctuation intensity spectra per mode)
        # ----------------------------------------------------------------------------------------

        # phi*nmodes, apar*nmods, aper*nmodes
        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.field_spectrum" + self.suffix),
            blocks=1,
            columns=None,
            numky=self.num_ky,
        )

        # Using my convention of [quantity,nmode,ky]
        self.FieldSpectrum = np.zeros((4, self.num_nmodes, self.num_ky))
        for i in range(self.num_nmodes):
            self.FieldSpectrum[0, i] = data[0, :, 4 * i + 0]
            self.FieldSpectrum[1, i] = data[0, :, 4 * i + 1]
            self.FieldSpectrum[2, i] = data[0, :, 4 * i + 2]
            self.FieldSpectrum[3, i] = data[0, :, 4 * i + 3]
        # *************************************************

        self.v_spectrum = self.FieldSpectrum[0, :, :]
        self.phi_spectrum = self.FieldSpectrum[1, :, :]
        self.a_par_spectrum = self.FieldSpectrum[2, :, :]
        self.a_per_spectrum = self.FieldSpectrum[3, :, :]

        with open(
            self.FolderGACODE / ("out.tglf.field_spectrum" + self.suffix), "r"
        ) as f:
            aux = f.readlines()
        self.fields = ["phi"]
        if aux[4].split()[0].split("_")[-1] == "yes":
            self.fields.append("a_par")
        if aux[5].split()[0].split("_")[-1] == "yes":
            self.fields.append("a_per")

        # Because if APAR is False and APER is True, it still conserves the spot for APAR, but not the opposite
        self.num_fields = len(self.fields)

        # ------------------------------------------------------------------------
        # Flux Spectrum  -   SumFluxSpectrum [quantity,species,field,ky]
        # ------------------------------------------------------------------------

        # data [specie*field, ky, quantity]
        num_quantities = (
            5  # particle flux,energy flux,toroidal stress,parallel stress,exchange
        )
        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.sum_flux_spectrum" + self.suffix),
            blocks=None,
            columns=num_quantities,
            numky=self.num_ky,
        )

        # self.num_fields = int(data.shape[0]/self.num_species)

        # Re-arrange to separa specie and field [species, field, ky, quantity]
        data_re = np.reshape(
            data, (self.num_species, self.num_fields, self.num_ky, num_quantities)
        )

        # Using my convention of [quantity,species,field,ky]
        self.SumFluxSpectrum = data_re.transpose((3, 0, 1, 2))
        # *************************************************

        self.SumFlux_Qe_phi = self.SumFluxSpectrum[1, 0, 0, :]
        self.SumFlux_Ge_phi = self.SumFluxSpectrum[0, 0, 0, :]
        self.SumFlux_QiAll_phi = self.SumFluxSpectrum[1, 1:, 0, :]

        contF = 0
        if "a_par" in self.fields:
            self.SumFlux_Qe_a_par = self.SumFluxSpectrum[1, 0, 1 + contF, :]
            self.SumFlux_Ge_a_par = self.SumFluxSpectrum[0, 0, 1 + contF, :]
            self.SumFlux_QiAll_a_par = self.SumFluxSpectrum[1, 1:, 1 + contF, :]
            contF += 1
        else:
            self.SumFlux_Qe_a_par = self.SumFlux_Qe_phi * 0.0
            self.SumFlux_Ge_a_par = self.SumFlux_Ge_phi * 0.0
            self.SumFlux_QiAll_a_par = self.SumFlux_QiAll_phi * 0.0

        if "a_per" in self.fields:
            self.SumFlux_Qe_a_per = self.SumFluxSpectrum[1, 0, 1 + contF, :]
            self.SumFlux_Ge_a_per = self.SumFluxSpectrum[0, 0, 1 + contF, :]
            self.SumFlux_QiAll_a_per = self.SumFluxSpectrum[1, 1:, 1 + contF, :]
            contF += 1
        else:
            self.SumFlux_Qe_a_per = self.SumFlux_Qe_phi * 0.0
            self.SumFlux_Ge_a_per = self.SumFlux_Ge_phi * 0.0
            self.SumFlux_QiAll_a_per = self.SumFlux_QiAll_phi * 0.0

        self.SumFlux_Qe_a = self.SumFlux_Qe_a_par + self.SumFlux_Qe_a_per
        self.SumFlux_Ge_a = self.SumFlux_Ge_a_par + self.SumFlux_Ge_a_per
        self.SumFlux_QiAll_a = self.SumFlux_QiAll_a_par + self.SumFlux_QiAll_a_per

        self.SumFlux_Qe = self.SumFlux_Qe_phi + self.SumFlux_Qe_a
        self.SumFlux_Ge = self.SumFlux_Ge_phi + self.SumFlux_Ge_a
        self.SumFlux_QiAll = self.SumFlux_QiAll_phi + self.SumFlux_QiAll_a

        # Sum ion contributions

        print(
            f"\t\t- For Qi spectrum, summing contributions from ions {self.ions_included} (#0 is e-)",
            typeMsg="i",
        )
        sum_ions = tuple([i - 1 for i in self.ions_included])

        self.SumFlux_Qi_phi = self.SumFlux_QiAll_phi[sum_ions, :].sum(axis=0)
        self.SumFlux_Qi_a = self.SumFlux_QiAll_a[sum_ions, :].sum(axis=0)
        self.SumFlux_Qi = self.SumFlux_Qi_phi + self.SumFlux_Qi_a

        # ------------------------------------------------------------------------
        # QL Flux Spectrum (QL weights per mode)
        # ------------------------------------------------------------------------

        # particle flux,energy flux,toroidal stress,parallel stress,exchange (?)
        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.QL_flux_spectrum" + self.suffix),
            blocks=self.num_species * self.num_fields * self.num_nmodes,
            columns=None,
            numky=self.num_ky,
        )

        # Re-arrange to separa specie and field

        data_re = np.reshape(
            data,
            (
                self.num_species,
                self.num_fields,
                self.num_nmodes,
                self.num_ky,
                num_quantities,
            ),
            order="C",
        )

        # Using my convention of [quantity,species,field,nmode,ky]
        self.QLFluxSpectrum = data_re.transpose((4, 0, 1, 2, 3))
        # *************************************************

        self.QLFluxSpectrum_Ge_phi = self.QLFluxSpectrum[0, 0, 0, :, :]
        self.QLFluxSpectrum_Qe_phi = self.QLFluxSpectrum[1, 0, 0, :, :]

        self.QLFluxSpectrum_GiAll_phi = self.QLFluxSpectrum[0, 1:, 0, :, :]
        self.QLFluxSpectrum_Gi_phi = self.QLFluxSpectrum_GiAll_phi[sum_ions, :].sum(
            axis=0
        )  # Sum over ions
        self.QLFluxSpectrum_QiAll_phi = self.QLFluxSpectrum[1, 1:, 0, :, :]
        self.QLFluxSpectrum_Qi_phi = self.QLFluxSpectrum_QiAll_phi[sum_ions, :].sum(
            axis=0
        )  # Sum over ions

        contF = 1
        if "a_par" in self.fields:
            self.QLFluxSpectrum_Ge_a_par = self.QLFluxSpectrum[0, 0, 1, :, :]
            self.QLFluxSpectrum_Qe_a_par = self.QLFluxSpectrum[1, 0, 1, :, :]

            self.QLFluxSpectrum_GiAll_a_par = self.QLFluxSpectrum[0, 1:, 1, :, :]
            self.QLFluxSpectrum_Gi_a_par = self.QLFluxSpectrum_GiAll_a_par[
                sum_ions, :
            ].sum(
                axis=0
            )  # Sum over ions
            self.QLFluxSpectrum_QiAll_a_par = self.QLFluxSpectrum[1, 1:, 1, :, :]
            self.QLFluxSpectrum_Qi_a_par = self.QLFluxSpectrum_QiAll_a_par[
                sum_ions, :
            ].sum(
                axis=0
            )  # Sum over ions
            contF += 1
        else:
            self.QLFluxSpectrum_Ge_a_par = self.QLFluxSpectrum_Ge_phi * 0.0
            self.QLFluxSpectrum_Qe_a_par = self.QLFluxSpectrum_Qe_phi * 0.0

            self.QLFluxSpectrum_GiAll_a_par = self.QLFluxSpectrum_QiAll_phi * 0.0
            self.QLFluxSpectrum_Gi_a_par = self.QLFluxSpectrum_Qi_phi * 0.0
            self.QLFluxSpectrum_QiAll_a_par = self.QLFluxSpectrum_QiAll_phi * 0.0
            self.QLFluxSpectrum_Qi_a_par = self.QLFluxSpectrum_Qi_phi * 0.0

        if "a_per" in self.fields:
            self.QLFluxSpectrum_Ge_a_per = self.QLFluxSpectrum[0, 0, 2, :, :]
            self.QLFluxSpectrum_Qe_a_per = self.QLFluxSpectrum[1, 0, 2, :, :]

            self.QLFluxSpectrum_GiAll_a_per = self.QLFluxSpectrum[0, 1:, 2, :, :]
            self.QLFluxSpectrum_Gi_a_per = self.QLFluxSpectrum_GiAll_a_per[
                sum_ions, :
            ].sum(
                axis=0
            )  # Sum over ions
            self.QLFluxSpectrum_QiAll_a_per = self.QLFluxSpectrum[1, 1:, 2, :, :]
            self.QLFluxSpectrum_Qi_a_per = self.QLFluxSpectrum_QiAll_a_per[
                sum_ions, :
            ].sum(
                axis=0
            )  # Sum over ions
            contF += 1
        else:
            self.QLFluxSpectrum_Ge_a_per = self.QLFluxSpectrum_Ge_phi * 0.0
            self.QLFluxSpectrum_Qe_a_per = self.QLFluxSpectrum_Qe_phi * 0.0

            self.QLFluxSpectrum_GiAll_a_per = self.QLFluxSpectrum_GiAll_phi * 0.0
            self.QLFluxSpectrum_Gi_a_per = self.QLFluxSpectrum_Gi_phi * 0.0
            self.QLFluxSpectrum_QiAll_a_per = self.QLFluxSpectrum_QiAll_phi * 0.0
            self.QLFluxSpectrum_Qi_a_per = self.QLFluxSpectrum_Qi_phi * 0.0

        # ------------------------------------------------------------------------
        # Intensity Spectrum
        # ------------------------------------------------------------------------

        data = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.tglf.intensity_spectrum" + self.suffix),
            blocks=1,
            columns=None,
            numky=self.num_ky,
        )

        data_re = np.reshape(
            data[0, :, :],
            (self.num_species, self.num_nmodes, self.num_ky, 4),
            order="C",
        )

        # Using my convention of [quantity=(density,temperature,parallel velocity,parallel energy),species,nmode,ky]
        self.IntensitySpectrum = data_re.transpose((3, 0, 1, 2))

        self.IntensitySpectrum_ne = self.IntensitySpectrum[0, 0, :, :]
        self.IntensitySpectrum_Te = self.IntensitySpectrum[1, 0, :, :]

        self.IntensitySpectrum_ni = self.IntensitySpectrum[0, 1:, :, :]
        self.IntensitySpectrum_Ti = self.IntensitySpectrum[1, 1:, :, :]

        # ------------------------------------------------------------------------
        # TGLF input file
        # ------------------------------------------------------------------------

        with open(self.FolderGACODE / ("input.tglf" + self.suffix), "r") as fi:
            lines = fi.readlines()
        self.inputFileTGLF = "".join(lines)

    def unnormalize(
        self, normalization, rho=None, convolution_fun_fluct=None, factorTot_to_Perp=1.0
    ):
        if normalization is not None:
            rho_x = normalization["rho"]
            roa_x = normalization["roa"]
            q_gb = normalization["q_gb"]
            g_gb = normalization["g_gb"]
            rho_s = normalization["rho_s"]
            a = normalization["rmin"][-1]

            # ------------------------------------
            # Usage of normalization quantities
            # ------------------------------------

            if rho is None:
                ir = np.argmin(np.abs(roa_x - self.roa))
                rho_eval = rho_x[ir]
            else:
                ir = np.argmin(np.abs(rho_x - rho))
                rho_eval = rho

            self.Qe_unn = self.Qe * q_gb[ir]
            self.Qi_unn = self.Qi * q_gb[ir]
            self.Ge_unn = self.Ge * g_gb[ir]
            self.GiAll_unn = self.GiAll * g_gb[ir]

            self.AmplitudeSpectrum_Te_level = GACODErun.obtainFluctuationLevel(
                self.ky,
                self.AmplitudeSpectrum_Te,
                rho_s[ir],
                a,
                convolution_fun_fluct=convolution_fun_fluct,
                rho_eval=rho_eval,
                factorTot_to_Perp=factorTot_to_Perp,
            )
            self.AmplitudeSpectrum_ne_level = GACODErun.obtainFluctuationLevel(
                self.ky,
                self.AmplitudeSpectrum_ne,
                rho_s[ir],
                a,
                convolution_fun_fluct=convolution_fun_fluct,
                rho_eval=rho_eval,
                factorTot_to_Perp=factorTot_to_Perp,
            )
            self.neTeSpectrum_level = GACODErun.obtainNTphase(
                self.ky,
                self.neTeSpectrum[0, :],
                rho_s[ir],
                a,
                convolution_fun_fluct=convolution_fun_fluct,
                rho_eval=rho_eval,
                factorTot_to_Perp=factorTot_to_Perp,
            )

            self.unnormalization_successful = True

        else:
            self.unnormalization_successful = False

    def to_xarray(self):
        coord_rename_mapper = {
            'num_ky': 'nky',
            'num_nmodes': 'nmodes',
            'num_species': 'nspecies',
            'num_fields': 'nfields',
        }
        data_rename_mapper = {
            'AmplitudeSpectrum': 'amplitude',
            'Eigenvalues': 'eigenvalue',
            'FieldSpectrum': 'field',
            'IntensitySpectrum': 'intensity',
            'nTSpectrum': 'crossphase',
            'QLFluxSpectrum': 'ql_flux',
            'SumFluxSpectrum': 'sum_flux',
        }
        data_dim_mapper = {
            'amplitude': ['nspecies', 'nky'],
            'eigenvalue': ['nmodes', 'nky'],
            'field': ['nmodes', 'nky'],
            'intensity': ['nspecies', 'nmodes', 'nky'],
            'crossphase': ['nspecies', 'nmodes', 'nky'],
            'ql_flux': ['nspecies', 'nfields', 'nmodes', 'nky'],
            'sum_flux': ['nspecies', 'nfields', 'nky'],
        }
        data_type_mapper = {
            'amplitude': ['density', 'temperature'],
            'eigenvalue': ['imaginary', 'real'],
            'field': ['density', 'temperature', 'parallel_velocity', 'parallel_energy'],
            'intensity': ['density', 'temperature', 'parallel_velocity', 'parallel_energy'],
            'ql_flux': ['particle', 'energy', 'toroidal_stress', 'parallel_stress', 'exchange'],
            'sum_flux': ['particle', 'energy', 'toroidal_stress', 'parallel_stress', 'exchange'],
        }
        coord_dict = {'nruns': [0]}
        dvars_dict = {}
        attrs_dict = {}
        for attr, key in coord_rename_mapper.items():
            if hasattr(self, f'{attr}'):
                dim_length = getattr(self, f'{attr}')
                coord_dict[f'{key}'] = np.arange(dim_length)
        for attr, key in data_rename_mapper.items():
            if hasattr(self, f'{attr}'):
                vals = getattr(self, f'{attr}')
                data_type_dict = {}
                if f'{key}' in data_type_mapper:
                    if vals.shape[0] == len(data_type_mapper[f'{key}']):
                        tkeys = [f'{key}_{tkey}' for tkey in data_type_mapper[f'{key}']]
                        tvals = np.split(vals, vals.shape[0], axis=0)
                        for tkey, tval in zip(tkeys, tvals):
                            data_type_dict[f'{tkey}'] = np.squeeze(tval, axis=0)
                else:
                    data_type_dict[f'{key}'] = vals
                dims = ['nruns']
                if f'{key}' in data_dim_mapper:
                    dims += data_dim_mapper[f'{key}']
                for dkey in data_type_dict:
                    dvars_dict[f'{dkey}'] = (dims, np.expand_dims(data_type_dict[f'{dkey}'], axis=0))
        if dvars_dict:
            input_dims = ['nruns']
            dvars_dict['ky'] = (input_dims + ['nky'], np.expand_dims(self.ky, axis=0))
            if hasattr(self.inputclass, 'plasma'):
                input_dict = getattr(self.inputclass, 'plasma')
                for key in input_dict:
                    nkey = key.lower()
                    dvars_dict[f'{nkey}'] = (input_dims, np.expand_dims(input_dict[f'{key}'], axis=0))
            if hasattr(self.inputclass, 'species'):
                input_dict = getattr(self.inputclass, 'species')
                species_dict = {}
                for ii in range(self.num_species):
                    tglf_ion_num = ii + 1
                    if tglf_ion_num in input_dict:
                        for key in input_dict[tglf_ion_num]:
                            nkey = key.lower()
                            if nkey not in species_dict:
                                species_dict[f'{nkey}'] = np.full((self.num_species, ), np.nan)
                            species_dict[f'{nkey}'][ii] = input_dict[tglf_ion_num][f'{key}']
                for nkey in species_dict:
                    dvars_dict[f'{nkey}'] = (input_dims + ['nspecies'], np.expand_dims(species_dict[f'{nkey}'], axis=0))
            if hasattr(self.inputclass, 'geom'):
                input_dict = getattr(self.inputclass, 'geom')
                for key in input_dict:
                    nkey = key.lower()
                    dvars_dict[f'{nkey}'] = (input_dims, np.expand_dims(input_dict[f'{key}'], axis=0))
            attrs_dict['field_names'] = self.fields
            if hasattr(self.inputclass, 'controls'):
                input_dict = getattr(self.inputclass, 'controls')
                for key in input_dict:
                    nkey = key.lower()
                    val = input_dict[f'{key}']
                    if isinstance(val, bool):
                        val = 'T' if val else 'F'
                    attrs_dict[f'{nkey}'] = val
        return xr.Dataset(data_vars=dvars_dict, coords=coord_dict, attrs=attrs_dict)

    def plotTGLF_Summary(self, c="b", label="", axs=None, irho_cont=0):
        removeLow = 1e-6  # If Growth Rate below this, remove from list for better logarithmic plotting

        if axs is None:
            plt.ion()
            fig1 = plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(4, 3, hspace=0.7, wspace=0.2)

            axs = np.empty((4, 3), dtype=plt.Axes)

            axs[0, 0] = fig1.add_subplot(grid[0, 0])
            axs[1, 0] = fig1.add_subplot(grid[1, 0], sharex=axs[0, 0])
            axs[2, 0] = fig1.add_subplot(grid[2, 0], sharex=axs[0, 0])
            axs[3, 0] = fig1.add_subplot(grid[3, 0])
            axs[0, 1] = fig1.add_subplot(grid[0, 1], sharex=axs[0, 0])
            axs[1, 1] = fig1.add_subplot(grid[1, 1], sharex=axs[0, 0])
            axs[2, 1] = fig1.add_subplot(grid[2, 1], sharex=axs[0, 0])
            axs[3, 1] = fig1.add_subplot(grid[3, 1])
            axs[0, 2] = fig1.add_subplot(grid[0, 2], sharex=axs[0, 0])
            axs[1, 2] = fig1.add_subplot(grid[1, 2], sharex=axs[0, 0])
            axs[2, 2] = fig1.add_subplot(grid[2, 2], sharex=axs[0, 0])
            axs[3, 2] = fig1.add_subplot(grid[3, 2])

        # ***** Growth Rate and Frequency

        GACODEplotting.plotTGLFspectrum(
            [axs[0, 0], axs[1, 0]],
            self.ky,
            self.g[0, :],
            freq=self.f[0, :],
            coeff=0.0,
            c=c,
            ls="-",
            lw=1,
            label=label,
            markersize=20,
            alpha=1.0,
            titles=["Growth Rate", "Real Frequency"],
            removeLow=removeLow,
            ylabel=True,
        )

        try:
            gammaExB = np.abs(
                self.inputsTGLF[self.rhos[irho_cont]].plasma["VEXB_SHEAR"]
            )
            if gammaExB > 1e-5:
                GRAPHICStools.drawLineWithTxt(
                    axs[0, 0],
                    gammaExB,
                    label="$\\gamma_{ExB}$" if irho_cont == 0 else "",
                    orientation="horizontal",
                    color=c,
                    lw=1,
                    ls="--",
                    alpha=1.0,
                    fontsize=10,
                    fromtop=0.7,
                    fontweight="normal",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    separation=0,
                )
        except:
            pass  # print('\t\t- Could not analyze ExB shear to plot')

        # ***** Zonal Flow Mixing

        ax = axs[2, 0]
        GACODEplotting.plotTGLFspectrum(
            ax,
            self.ky,
            self.g[0, :],
            freq=None,
            coeff=1.0,
            c=c,
            ls="-",
            lw=1,
            label=label,
            markersize=20,
            alpha=1.0,
            titles=["Zonal Flow Mixing, $\\gamma/k$"],
            removeLow=removeLow,
            ylabel=True,
        )

        # ***** Temperature and Density Fluctuations

        ax = axs[0, 1]

        GACODEplotting.plot_spec(
            ax,
            self.ky,
            self.AmplitudeSpectrum_Te,
            markers="o",
            c=c,
            lw=1,
            ls="-",
            label=label,
            alpha=1.0,
            markersize=20,
            facecolor=c,
        )

        ax.set_title("Spectrum: $\\delta T_e$ Amplitude")
        ax.set_ylabel("$A_{T_e}(k_y)$")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1, 1]

        GACODEplotting.plot_spec(
            ax,
            self.ky,
            self.AmplitudeSpectrum_ne,
            markers="o",
            c=c,
            lw=1,
            ls="-",
            label=label,
            alpha=1.0,
            markersize=20,
            facecolor=c,
        )

        ax.set_title("Spectrum: $\\delta n_e$ Amplitude")
        ax.set_ylabel("$A_{n_e}(k_y)$")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[2, 1]

        GACODEplotting.plot_spec(
            ax,
            self.ky,
            self.neTeSpectrum[0, :],
            markers="o",
            c=c,
            lw=1,
            ls="-",
            label=label,
            alpha=1.0,
            markersize=20,
            facecolor=c,
        )

        ax.set_title("Spectrum: $n_e-T_e$ cross-phase angle")
        ax.set_ylabel("$\\alpha_{n_e,T_e}$ (degrees)")
        GRAPHICStools.addDenseAxis(ax)

        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylim([-180, 180])

        # ***** Flux Spectra

        ax = axs[0, 2]

        ax.plot(self.ky, self.SumFlux_Qe, "-o", color=c, lw=1.0, markersize=2)

        ax.set_title("Spectrum: $Q_e$")
        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$Q_{e,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1, 2]

        ax.plot(self.ky, self.SumFlux_Qi, "-o", color=c, lw=1.0, markersize=2)

        ax.set_title("Spectrum: $Q_i$")
        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$Q_{i,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[2, 2]

        ax.plot(self.ky, self.SumFlux_Ge, "-o", color=c, lw=1.0, markersize=2)

        ax.set_title("Spectrum: $\\Gamma_e$")
        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$\\Gamma_{e,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)

        # ***** Flux Values

        ax = axs[3, 0]

        ax.plot([self.roa], [self.Qe], "-o", c=c, markersize=5)

        ax.set_xlabel("$r/a$ (RMIN_LOC)")
        ax.set_ylabel("$Q_e/Q_{GB}$")
        ax.set_xlim([0, 1])
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Profile: $Q_e$ (GB)")

        ax = axs[3, 1]

        ax.plot([self.roa], [self.Qi], "-o", c=c, markersize=5)

        ax.set_xlabel("$r/a$ (RMIN_LOC)")
        ax.set_ylabel("$Q_i/Q_{GB}$")
        ax.set_xlim([0, 1])
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Profile: $Q_i$ (GB)")

        ax = axs[3, 2]

        ax.plot([self.roa], [self.Ge], "-o", c=c, markersize=5)

        ax.set_xlabel("$r/a$ (RMIN_LOC)")
        ax.set_ylabel("$\\Gamma/\\Gamma_{GB}$")
        ax.set_xlim([0, 1])
        ax.axhline(y=0, ls="--", lw=1, c="k")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Profile: $\\Gamma_e$ (GB)")

    def plotTGLF_Contributors(
        self, c="b", label="", axs=None, fontsizeLeg=4, title_legend=None, cont=0
    ):
        removeLow = 1e-6  # If Growth Rate below this, remove from list for better logarithmic plotting

        if axs is None:
            plt.ion()
            fig2 = plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(3, 5, hspace=0.6, wspace=0.3)
            axs = np.empty((3, 4), dtype=plt.Axes)

            axs[0, 0] = fig2.add_subplot(grid[0, 0])
            axs[1, 0] = fig2.add_subplot(grid[1, 0], sharex=axs[0, 0])
            axs[2, 0] = fig2.add_subplot(grid[2, 0], sharex=axs[0, 0])
            axs[0, 1] = fig2.add_subplot(grid[0, 1], sharex=axs[0, 0])
            axs[1, 1] = fig2.add_subplot(grid[1, 1], sharex=axs[0, 0])
            axs[2, 1] = fig2.add_subplot(grid[2, 1], sharex=axs[0, 0])
            axs[0, 2] = fig2.add_subplot(grid[0, 2])
            axs[1, 2] = fig2.add_subplot(grid[1, 2])
            axs[2, 2] = fig2.add_subplot(grid[2, 2])
            axs[0, 3] = fig2.add_subplot(grid[0, 3:], sharex=axs[0, 0])
            axs[1, 3] = fig2.add_subplot(grid[1, 3:], sharex=axs[0, 0])
            axs[2, 3] = fig2.add_subplot(grid[2, 3:], sharex=axs[0, 0])

        # ***** EM+ES Flux Spectra

        ax = axs[0, 0]
        ax.plot(
            self.ky,
            self.SumFlux_Qe_phi,
            "-o",
            color=c,
            lw=0.5,
            markersize=2,
            label="ES" if cont == 0 else "",
        )
        ax.plot(
            self.ky,
            self.SumFlux_Qe_a,
            "-s",
            color=c,
            lw=0.3,
            markersize=2,
            label="EM" if cont == 0 else "",
        )

        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$Q_{e,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Spectrum (ES+EM): $Q_e$")
        ax.legend(loc="best", fontsize=fontsizeLeg * 1.5, title=title_legend)

        ax = axs[1, 0]
        ax.plot(
            self.ky,
            self.SumFlux_Qi_phi,
            "-o",
            color=c,
            lw=0.5,
            markersize=2,
            label="ES" if cont == 0 else "",
        )
        ax.plot(
            self.ky,
            self.SumFlux_Qi_a,
            "-s",
            color=c,
            lw=0.3,
            markersize=2,
            label="EM" if cont == 0 else "",
        )

        ax.set_title("Spectrum (ES+EM): $Q_i$")
        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$Q_{i,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[2, 0]
        ax.plot(
            self.ky,
            self.SumFlux_Ge_phi,
            "-o",
            color=c,
            lw=0.5,
            markersize=2,
            label="ES" if cont == 0 else "",
        )
        ax.plot(
            self.ky,
            self.SumFlux_Ge_a,
            "-s",
            color=c,
            lw=0.3,
            markersize=2,
            label="EM" if cont == 0 else "",
        )

        ax.set_title("Spectrum (ES+EM): $\\Gamma_e$")
        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$\\Gamma_{e,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)

        # ***** EM+ES Flux radial

        ax = axs[0, 2]
        ax.plot(
            [self.roa],
            [self.QeES],
            "-o",
            c=c,
            markersize=5,
            label="ES" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.QeEM],
            "-s",
            c=c,
            markersize=5,
            label="EM" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.Qe],
            "-*",
            c=c,
            markersize=5,
            label="ES+EM" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.QeES + self.QeEM],
            "-v",
            c="k",
            markersize=1,
            label="check" if cont == 0 else "",
        )

        ax.set_xlim([0, 1.0])
        ax.set_xlabel("$r/a$ (RMIN_LOC)")
        ax.set_ylabel("$Q_e$ (GB)")
        ax.set_title("Profile (ES+EM): $Q_e$ (GB)")
        ax.legend(loc="best", fontsize=fontsizeLeg * 1.5, title=title_legend)
        ax.axhline(y=0, ls="--", c="k", lw=1)
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1, 2]
        ax.plot(
            [self.roa],
            [self.QiES],
            "-o",
            c=c,
            markersize=5,
            label="ES" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.QiEM],
            "-s",
            c=c,
            markersize=5,
            label="EM" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.Qi],
            "-*",
            c=c,
            markersize=5,
            label="ES+EM" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.QiES + self.QiEM],
            "-v",
            c="k",
            markersize=1,
            label="check" if cont == 0 else "",
        )

        ax.set_xlim([0, 1.0])
        ax.set_xlabel("$r/a$ (RMIN_LOC)")
        ax.set_ylabel("$Q_i$ (GB)")
        ax.set_title("Profile (ES+EM): $Q_i$ (GB)")
        ax.axhline(y=0, ls="--", c="k", lw=1)
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[2, 2]

        ax.plot(
            [self.roa],
            [self.GeES],
            "-o",
            c=c,
            markersize=5,
            label="ES" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.GeEM],
            "-s",
            c=c,
            markersize=5,
            label="EM" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.Ge],
            "-*",
            c=c,
            markersize=5,
            label="ES+EM" if cont == 0 else "",
        )
        ax.plot(
            [self.roa],
            [self.GeES + self.GeEM],
            "-v",
            c="k",
            markersize=1,
            label="check" if cont == 0 else "",
        )

        ax.set_xlim([0, 1.0])
        ax.set_xlabel("$r/a$ (RMIN_LOC)")
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        ax.set_title("Profile (ES+EM): $\\Gamma_e$ (GB)")
        ax.axhline(y=0, ls="--", c="k", lw=1)
        GRAPHICStools.addDenseAxis(ax)

        # ***** Cumulative Spectra

        ax = axs[0, 1]
        ax.plot(
            self.ky, np.cumsum(self.SumFlux_Qe), "-o", color=c, lw=1.0, markersize=2
        )
        ax.axhline(y=self.Qe, ls="--", lw=0.5, c=c)
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Spectrum (Cumulative): $Q_e$")

        ax = axs[1, 1]
        ax.plot(
            self.ky, np.cumsum(self.SumFlux_Qi), "-o", color=c, lw=1.0, markersize=2
        )
        ax.axhline(y=self.Qi, ls="--", lw=0.5, c=c)
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Spectrum (Cumulative): $Q_i$")

        ax = axs[2, 1]
        ax.plot(
            self.ky, np.cumsum(self.SumFlux_Ge), "-o", color=c, lw=1.0, markersize=2
        )
        ax.axhline(y=self.Ge, ls="--", lw=0.5, c=c)
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Spectrum (Cumulative): $\\Gamma_e$")
        ax.axhline(y=0, ls="--", lw=0.5, c="k")

        # ***** Growth Rates and Subdominant

        GACODEplotting.plotTGLFspectrum(
            [axs[0, 3], axs[1, 3]],
            self.ky,
            self.g[0, :],
            freq=self.f[0, :],
            coeff=0.0,
            c=c,
            ls="-",
            lw=1,
            label=f"{label}, nmode = 1",
            markersize=20,
            alpha=1.0,
            titles=["Growth Rate (all modes)", "Real Frequency (all modes)"],
            removeLow=removeLow,
            ylabel=True,
        )

        for subdom in range(self.num_nmodes - 1):
            GACODEplotting.plotTGLFspectrum(
                [axs[0, 3], axs[1, 3]],
                self.ky,
                self.g[subdom + 1, :],
                freq=self.f[subdom + 1, :],
                coeff=0.0,
                c=c,
                ls="-",
                lw=0.5,
                markersize=10,
                alpha=0.5,
                titles=None,
                removeLow=removeLow,
                ylabel=False,
                label=f"{label}, nmode = {2+subdom}",
            )

        GRAPHICStools.addLegendApart(
            axs[0, 3], size=fontsizeLeg, ratio=0.9, title=title_legend
        )
        GRAPHICStools.addLegendApart(
            axs[1, 3], size=fontsizeLeg, ratio=0.9, title=title_legend, withleg=False
        )
        GRAPHICStools.addLegendApart(
            axs[2, 3], size=fontsizeLeg, ratio=0.9, title=title_legend, withleg=False
        )

        # ***** All ions

        typeline = GRAPHICStools.listmarkersLS()

        ax = axs[2, 3]
        for ion in range(self.SumFlux_QiAll.shape[0]):
            ax.plot(
                self.ky,
                self.SumFlux_QiAll[ion, :],
                typeline[ion],
                color=c,
                lw=0.5,
                markersize=2,
                label=f"ion {ion+1}" if cont == 0 else "",
            )

        ax.axhline(y=0, ls="--", lw=1, c="k")
        ax.set_ylabel("$Q_{i,ky}$")
        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Spectrum (each ion): $Q_i$")
        ax.legend(loc="best", fontsize=fontsizeLeg * 1.5, title=title_legend)

    def plotTGLF_Fluctuations(
        self, c="b", label="", axs=None, fontsizeLeg=4, title_legend="", cont=0
    ):
        if axs is None:
            plt.ion()
            fig1 = plt.figure(figsize=(8, 8))

            max_num_species = self.num_species

            grid = plt.GridSpec(5, max_num_species, hspace=0.9, wspace=0.3)

            axs = np.empty((5, max_num_species), dtype=plt.Axes)
            for i in range(max_num_species):
                for j in range(5):
                    axs[j, i] = fig1.add_subplot(grid[j, i])

        typeline = GRAPHICStools.listmarkersLS()

        ax = axs[0, 0]
        ax.plot(self.ky, self.AmplitudeSpectrum_ne, "-o", color=c, lw=1.0, markersize=3)

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Amplitude Spectrum: $n_e$")

        ax = axs[1, 0]
        ax.plot(self.ky, self.AmplitudeSpectrum_Te, "-o", color=c, lw=1.0, markersize=3)

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Amplitude Spectrum: $T_e$")

        ax = axs[2, 0]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                self.neTeSpectrum[i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
                label=f"{label}, nmode = {i+1}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("$n_e-T_e$ Spectrum")
        ax.legend(loc="best", fontsize=fontsizeLeg, title=title_legend)

        ax = axs[3, 0]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                self.IntensitySpectrum_ne[i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
                label=f"{label}, nmode = {i+1}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Intensity Spectrum: $n_e$")

        ax = axs[4, 0]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                self.IntensitySpectrum_Te[i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
                label=f"{label}, nmode = {i+1}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Intensity Spectrum: $T_e$")

        for ion in range(self.num_species - 1):
            ax = axs[0, ion + 1]
            ax.plot(
                self.ky,
                self.AmplitudeSpectrum_ni[ion, :],
                "-o",
                color=c,
                lw=1.0,
                markersize=3,
            )

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"Amplitude Spectrum: $n_{{i}}$ (ion {ion+1})")

            ax = axs[1, ion + 1]
            ax.plot(
                self.ky,
                self.AmplitudeSpectrum_Ti[ion, :],
                "-o",
                color=c,
                lw=1.0,
                markersize=3,
            )

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"Amplitude Spectrum: $T_{{i}}$ (ion {ion+1})")

            ax = axs[2, ion + 1]
            for i in range(self.num_nmodes):
                ax.plot(
                    self.ky,
                    self.niTiSpectrum[ion, i, :],
                    typeline[i],
                    color=c,
                    lw=1.0 if i == 0 else 0.5,
                    markersize=3 if i == 0 else 2,
                    label=f"{label}, nmode = {i+1}",
                )

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"$n_i-T_i$ Spectrum (ion {ion+1})")

            ax = axs[3, ion + 1]
            for i in range(self.num_nmodes):
                ax.plot(
                    self.ky,
                    self.IntensitySpectrum_ni[ion, i, :],
                    typeline[i],
                    color=c,
                    lw=1.0 if i == 0 else 0.5,
                    markersize=3 if i == 0 else 2,
                    label=f"{label}, nmode = {i+1}",
                )

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"Intensity Spectrum: $n_{{i}}$ (ion {ion+1})")

            ax = axs[4, ion + 1]
            for i in range(self.num_nmodes):
                ax.plot(
                    self.ky,
                    self.IntensitySpectrum_Ti[ion, i, :],
                    typeline[i],
                    color=c,
                    lw=1.0 if i == 0 else 0.5,
                    markersize=3 if i == 0 else 2,
                    label=f"{label}, nmode = {i+1}",
                )

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"Intensity Spectrum: $T_{{i}}$ (ion {ion+1})")

    def plotTGLF_Field(
        self,
        quantity="phi",
        c="b",
        label="",
        axs=None,
        fontsizeLeg=4,
        title_legend="",
        cont=0,
    ):
        if quantity == "phi":
            v = [
                self.phi_spectrum,
                "$\\delta \\phi$",
                self.QLFluxSpectrum_Qe_phi,
                self.QLFluxSpectrum_Ge_phi,
                self.QLFluxSpectrum_QiAll_phi,
                self.SumFlux_Qe_phi,
                self.SumFlux_Ge_phi,
                self.SumFlux_QiAll_phi,
            ]
        elif quantity == "a_par":
            v = [
                self.a_par_spectrum,
                "$\\delta A_{\\parallel}$",
                self.QLFluxSpectrum_Qe_a_par,
                self.QLFluxSpectrum_Ge_a_par,
                self.QLFluxSpectrum_QiAll_a_par,
                self.SumFlux_Qe_a_par,
                self.SumFlux_Ge_a_par,
                self.SumFlux_QiAll_a_par,
            ]
        elif quantity == "a_per":
            v = [
                self.a_per_spectrum,
                "$\\delta A_{\\perp}$",
                self.QLFluxSpectrum_Qe_a_per,
                self.QLFluxSpectrum_Ge_a_per,
                self.QLFluxSpectrum_QiAll_a_per,
                self.SumFlux_Qe_a_per,
                self.SumFlux_Ge_a_per,
                self.SumFlux_QiAll_a_per,
            ]

        if axs is None:
            plt.ion()
            fig1 = plt.figure(figsize=(8, 8))

            max_num_species = self.num_species

            grid = plt.GridSpec(2, max_num_species + 2, hspace=0.6, wspace=0.3)
            axs = np.empty((2, max_num_species + 2), dtype=plt.Axes)
            axs[0, 0] = fig1.add_subplot(grid[0, 0])
            axs[1, 0] = fig1.add_subplot(grid[1, 0])
            for i in range(max_num_species + 1):
                axs[0, i + 1] = fig1.add_subplot(grid[0, i + 1], sharex=axs[0, 0])
                axs[1, i + 1] = fig1.add_subplot(grid[1, i + 1], sharex=axs[0, 0])

        # ****************************************************************
        # Growth Rate and Fields
        # ****************************************************************

        typeline = GRAPHICStools.listmarkersLS()

        ax = axs[0, 0]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                self.g[i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
                label=f"{label}, nmode = {i+1}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title(f"Growth Rate Spectrum")
        ax.legend(loc="best", fontsize=fontsizeLeg, title=title_legend)

        ax = axs[1, 0]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                v[0][i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
                label=f"{label}, nmode = {i+1}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title(f"Field Spectrum: {v[1]}")

        # ****************************************************************
        # QL
        # ****************************************************************

        ax = axs[0, 1]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                v[2][i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("QL Spectrum: $Q_e$")

        ax = axs[0, 2]
        for i in range(self.num_nmodes):
            ax.plot(
                self.ky,
                v[3][i, :],
                typeline[i],
                color=c,
                lw=1.0 if i == 0 else 0.5,
                markersize=3 if i == 0 else 2,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("QL Spectrum: $\\Gamma_e$")

        for ion in range(self.num_species - 1):
            ax = axs[0, 3 + ion]
            for i in range(self.num_nmodes):
                ax.plot(
                    self.ky,
                    v[4][ion, i, :],
                    typeline[i],
                    color=c,
                    lw=1.0 if i == 0 else 0.5,
                    markersize=3 if i == 0 else 2,
                )

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"QL Spectrum: $Q_i$ (ion {ion+1})")

        # ****************************************************************
        # Sum Flux
        # ****************************************************************

        ax = axs[1, 1]
        ax.plot(self.ky, v[5], "-o", color=c, lw=1.0, markersize=3)

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Sum Flux Spectrum: $Q_e$")

        ax = axs[1, 2]
        ax.plot(self.ky, v[6], "-o", color=c, lw=1.0, markersize=3)

        ax.set_xscale("log")
        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Sum Flux Spectrum: $\\Gamma_e$")

        for ion in range(self.num_species - 1):
            ax = axs[1, 3 + ion]
            ax.plot(self.ky, v[7][ion, :], "-o", color=c, lw=1.0, markersize=3)

            ax.set_xscale("log")
            ax.set_xlabel("$k_{\\theta}\\rho_s$")
            GRAPHICStools.addDenseAxis(ax)
            ax.set_title(f"Sum Flux Spectrum: $Q_i$ (ion {ion+1})")

    def plotTGLF_Model(self, c="b", label="", axs=None):
        if axs is None:
            plt.ion()
            fig3 = plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(2, 2, hspace=0.6, wspace=0.3)
            axs = np.empty((2, 2), dtype=plt.Axes)

            axs[0, 0] = fig3.add_subplot(grid[0, 0])
            axs[1, 0] = fig3.add_subplot(grid[1, 0], sharex=axs[0, 0])
            axs[0, 1] = fig3.add_subplot(grid[0, 1], sharex=axs[0, 0])
            axs[1, 1] = fig3.add_subplot(grid[1, 1], sharex=axs[0, 0])

        ax = axs[0, 0]
        ax.plot(
            self.ky,
            self.tglf_model["width"],
            "-o",
            color=c,
            lw=2,
            markersize=4,
            label=label,
        )

        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        ax.set_ylabel("$\\theta_W$")
        ax.set_xscale("log")
        for i in [0.3, 1.65]:
            ax.axhline(y=i, lw=0.5, ls="--", c="k")
        ax.set_title("Gaussian Width Spectrum")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1, 0]
        ax.plot(
            self.ky,
            self.tglf_model["spectral_shift"],
            "-o",
            color=c,
            lw=2,
            markersize=4,
            label=label,
        )

        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        ax.set_ylabel("$kx_e$")
        ax.set_xscale("log")
        ax.set_title("Spectral Shift Spectrum, $<phi|kx/ky|phi>/<phi|phi>$")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[0, 1]
        ax.plot(
            self.ky,
            self.tglf_model["ave_p0"],
            "-o",
            color=c,
            lw=2,
            markersize=4,
            label=label,
        )

        ax.set_xlabel("$k_{\\theta}\\rho_s$")
        ax.set_ylabel("$<p_0>$")
        ax.set_xscale("log")
        ax.set_title("SAT0 normalization")
        GRAPHICStools.addDenseAxis(ax)


def processGrowthRates(k, g, f, gs, fs, klow=0.8, coeff=0):
    linearDict = {}

    subdom = 1

    # Calculate maximum growth rate (and the associated frequency and mode number) for each mode type

    dict_g = processDominated(k, g, f, krange=[0, klow], coeff=coeff)
    if gs is not None and gs.shape[1] > 0:
        dict_gs = processDominated(
            k, gs[:, subdom - 1], fs[:, subdom - 1], krange=[0, klow], coeff=coeff
        )

    # If for each type there's no a dominant mode, use subdominant

    g_ITG_max, f_ITG_max, k_ITG_max = (
        dict_g["g_ITG_max"],
        dict_g["f_ITG_max"],
        dict_g["k_ITG_max"],
    )
    if g_ITG_max == 0 and gs is not None and gs.shape[1] > 0:
        g_ITG_max, f_ITG_max, k_ITG_max = (
            dict_gs["g_ITG_max"],
            dict_gs["f_ITG_max"],
            dict_gs["k_ITG_max"],
        )

    g_ETG_max, f_ETG_max, k_ETG_max = (
        dict_g["g_ETG_max"],
        dict_g["f_ETG_max"],
        dict_g["k_ETG_max"],
    )
    if g_ETG_max == 0 and gs is not None and gs.shape[1] > 0:
        g_ETG_max, f_ETG_max, k_ETG_max = (
            dict_gs["g_ETG_max"],
            dict_gs["f_ETG_max"],
            dict_gs["k_ETG_max"],
        )

    g_TEM_max, f_TEM_max, k_TEM_max = (
        dict_g["g_TEM_max"],
        dict_g["f_TEM_max"],
        dict_g["k_TEM_max"],
    )
    if g_TEM_max == 0 and gs is not None and gs.shape[1] > 0:
        g_TEM_max, f_TEM_max, k_TEM_max = (
            dict_gs["g_TEM_max"],
            dict_gs["f_TEM_max"],
            dict_gs["k_TEM_max"],
        )

    # Store

    linearDict["ITG"] = {"g_max": g_ITG_max, "f_max": f_ITG_max, "k_max": k_ITG_max}
    linearDict["ETG"] = {"g_max": g_ETG_max, "f_max": f_ETG_max, "k_max": k_ETG_max}
    linearDict["TEM"] = {"g_max": g_TEM_max, "f_max": f_TEM_max, "k_max": k_TEM_max}

    # Calculate some eta metrics

    if g_ETG_max == 0:
        eta_ITGETG = 100
    else:
        eta_ITGETG = g_ITG_max / g_ETG_max

    if g_TEM_max == 0:
        eta_ITGTEM = 100
    else:
        eta_ITGTEM = g_ITG_max / g_TEM_max

    linearDict["metrics"] = {
        "eta_ITGETG": eta_ITGETG,
        "eta_ITGTEM": eta_ITGTEM,
        "g_lowk_max": dict_g["g_lowk_max"],
        "f_lowk_max": dict_g["f_lowk_max"],
        "k_lowk_max": dict_g["k_lowk_max"],
    }

    return linearDict


def processDominated(k, g, f, krange=[0.0, 0.8], coeff=0):
    # ky range to consider ITG and TEM modes
    ilow = np.argmin(np.abs(k - krange[0]))
    ihigh = np.argmin(np.abs(k - krange[1]))

    # ---------------------------------------------
    # ------- Separate contribution from each type
    # ---------------------------------------------

    # ~~~~~~~ ITG
    k_ITG, g_ITG, f_ITG = [], [], []
    for i in range(len(k)):
        if i <= ihigh and i >= ilow and f[i] < 0.0:
            k_ITG.append(k[i])
            g_ITG.append(g[i])
            f_ITG.append(f[i])
    k_ITG, g_ITG, f_ITG = np.array(k_ITG), np.array(g_ITG), np.array(f_ITG)

    # ~~~~~~~ ETG
    k_ETG, g_ETG, f_ETG = [], [], []
    for i in range(len(k)):
        if i > ihigh and f[i] > 0.0:
            k_ETG.append(k[i])
            g_ETG.append(g[i])
            f_ETG.append(f[i])
    k_ETG, g_ETG, f_ETG = np.array(k_ETG), np.array(g_ETG), np.array(f_ETG)

    # ~~~~~~~ TEM
    k_TEM, g_TEM, f_TEM = [], [], []
    for i in range(len(k)):
        if i <= ihigh and i >= ilow and f[i] > 0.0:
            k_TEM.append(k[i])
            g_TEM.append(g[i])
            f_TEM.append(f[i])
    k_TEM, g_TEM, f_TEM = np.array(k_TEM), np.array(g_TEM), np.array(f_TEM)

    # ---------------------------------------------
    # ------- Calculate the maximum of each type in that ky range
    # ---------------------------------------------

    # coeff will determine if it's just growth rate, zonal flow mixing of mixing length maxima

    if len(g_ITG) > 0:
        g_ITG_max, k_ITG_max, f_ITG_max = (
            np.max(g_ITG / k_ITG**coeff),
            k_ITG[np.argmax(g_ITG / k_ITG**coeff)],
            f_ITG[np.argmax(g_ITG / k_ITG**coeff)],
        )
    else:
        g_ITG_max, k_ITG_max, f_ITG_max = 0, np.nan, np.nan
    if len(g_ETG) > 0:
        g_ETG_max, k_ETG_max, f_ETG_max = (
            np.max(g_ETG / k_ETG**coeff),
            k_ETG[np.argmax(g_ETG / k_ETG**coeff)],
            f_ETG[np.argmax(g_ETG / k_ETG**coeff)],
        )
    else:
        g_ETG_max, k_ETG_max, f_ETG_max = 0, np.nan, np.nan
    if len(g_TEM) > 0:
        g_TEM_max, k_TEM_max, f_TEM_max = (
            np.max(g_TEM / k_TEM**coeff),
            k_TEM[np.argmax(g_TEM / k_TEM**coeff)],
            f_TEM[np.argmax(g_TEM / k_TEM**coeff)],
        )
    else:
        g_TEM_max, k_TEM_max, f_TEM_max = 0, np.nan, np.nan

    # Overall max at low k
    g_all = np.array([g_ITG_max, g_TEM_max])
    f_all = np.array([f_ITG_max, f_TEM_max])
    k_all = np.array([k_ITG_max, k_TEM_max])

    g_max, k_max, f_max = (
        np.max(g_all),
        k_all[np.argmax(g_all)],
        f_all[np.argmax(g_all)],
    )

    return {
        "g_ITG_max": g_ITG_max,
        "g_ETG_max": g_TEM_max,
        "g_TEM_max": g_TEM_max,
        "g_lowk_max": g_max,
        "k_ITG_max": k_ITG_max,
        "k_ETG_max": k_TEM_max,
        "k_TEM_max": k_TEM_max,
        "k_lowk_max": k_max,
        "f_ITG_max": f_ITG_max,
        "f_ETG_max": f_TEM_max,
        "f_TEM_max": f_TEM_max,
        "f_lowk_max": f_max,
    }


def compare_tglf_gen(file1, file2):
    with open(file1, "r") as f:
        aux1 = f.readlines()
    with open(file2, "r") as f:
        aux2 = f.readlines()

    d1 = {}
    for i in range(len(aux1)):
        d1[aux1[i].split()[1]] = aux1[i].split()[0]
    d2 = {}
    for i in range(len(aux2)):
        d2[aux2[i].split()[1]] = aux2[i].split()[0]

    for ikey in d1:
        if d1[ikey] in ["t", "T", ".true.", "True", "true", "t", "GYRO"]:
            d1[ikey] = 1.0
        elif d1[ikey] in ["f", "F", ".false.", "False", "false", "f"]:
            d1[ikey] = 0.0
        else:
            d1[ikey] = float(d1[ikey])
    for ikey in d2:
        if d2[ikey] in ["t", "T", ".true.", "True", "true", "t", "GYRO"]:
            d2[ikey] = 1.0
        elif d2[ikey] in ["f", "F", ".false.", "False", "false", "f"]:
            d2[ikey] = 0.0
        else:
            d2[ikey] = float(d2[ikey])

    d = {}
    for ikey in d1:
        d[ikey] = [d1[ikey], d2[ikey]]

    for ikey in d:
        if np.abs(d[ikey][0] - d[ikey][1]) > 1e-4:
            print(ikey, d[ikey])


# ~~~~~~~~~~~~~ Functions to handle run


def createCombinedRuns(tglfs=(), new_names=(), results_names=(), isItScan=False):
    """
    Two tglf classes can be combined to plot together.
    It is recommended that they have the same rhos
    """

    tglf = copy.deepcopy(tglfs[0])

    if not isItScan:
        for i in range(len(tglfs)):
            tglf.results[new_names[i]] = tglfs[i].results[results_names[i]]
    else:
        for i in range(len(tglfs)):
            tglf.scans[new_names[i]] = tglfs[i].scans[results_names[i]]

    try:
        for i in range(len(tglfs)):
            tglf.tgyro.results[new_names[i]] = tglfs[
                i
            ].tgyro_results  # tgyro.results['tgyro1']
    except:
        pass

    normalizations = {}
    for i in range(len(tglfs)):
        normalizations[new_names[i]] = tglfs[i].NormalizationSets

    return tglf, normalizations


def cold_start_checker(
    rhos,
    ResultsFiles,
    FolderTGLF,
    cold_start=False,
    forceIfcold_start=False,
):
    """
    This function checks if the TGLF inputs are already in the folder. If they are, it returns True
    """
    if cold_start:
        rhosEvaluate = rhos
    else:
        rhosEvaluate = []
        for ir in rhos:
            existsRho = True
            for j in ResultsFiles:
                ffi = FolderTGLF / f"{j}_{ir:.4f}"
                existsThis = ffi.exists()
                existsRho = existsRho and existsThis
                if not existsThis:
                    print(f"\t* {ffi} does not exist")
            if not existsRho:
                rhosEvaluate.append(ir)

    if len(rhosEvaluate) < len(rhos) and len(rhosEvaluate) > 0:
        print(
            "~ Not all radii are found, but not removing folder and running only those that are needed",
            typeMsg="i",
        )

    return rhosEvaluate


def anticipate_problems_func(
    latest_inputsFileTGLFDict, rhosEvaluate, slurm_setup, launchSlurm
):

    # -----------------------------------
    # ------ Check density for problems
    # -----------------------------------

    threshold = 1e-10

    minn = []
    for irho in latest_inputsFileTGLFDict:
        for cont, ip in enumerate(latest_inputsFileTGLFDict[irho].species):
            if (cont <= latest_inputsFileTGLFDict[irho].plasma["NS"]) and (
                latest_inputsFileTGLFDict[irho].species[ip]["AS"] < threshold
            ):
                minn.append([irho, ip])

    if len(minn) > 0:
        print(
            f"* Ions in positions [rho,pos] {minn} have a relative density lower than {threshold}, which can cause problems",
            typeMsg="q",
        )

    # -----------------------------------
    # ------ Check cores problem
    # -----------------------------------

    expected_allocated_cores = int(len(rhosEvaluate) * slurm_setup["cores"])

    warning = 32 * 2

    if launchSlurm:
        print(
            f'\t- Slurm job will be submitted with {expected_allocated_cores} cores ({len(rhosEvaluate)} radii x {slurm_setup["cores"]} cores/radius)',
            typeMsg="" if expected_allocated_cores < warning else "q",
        )
