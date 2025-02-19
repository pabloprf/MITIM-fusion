import copy
import torch
import numpy as np
import pandas as pd
import dill as pickle_dill
import matplotlib.pyplot as plt
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import IOtools, PLASMAtools, GRAPHICStools
from mitim_tools.gacode_tools import TGLFtools, TGYROtools, PROFILEStools
from mitim_tools.gacode_tools.utils import PORTALSinteraction
from mitim_modules.portals.utils import PORTALSplot
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.utils import POWERplot
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class PORTALSanalyzer:
    # ****************************************************************************
    # INITIALIZATION
    # ****************************************************************************

    def __init__(self, opt_fun, folderAnalysis=None):
        print("\n************************************")
        print("* Initializing PORTALS analyzer...")
        print("************************************")

        self.opt_fun = opt_fun

        self.folder = (
            IOtools.expandPath(folderAnalysis)
            if folderAnalysis is not None
            else self.opt_fun.folder / "Analysis"
        )

        self.folder.mkdir(parents=True, exist_ok=True)

        self.fn = None

        # Preparation
        print("- Grabbing model")
        self.step = self.opt_fun.mitim_model.steps[-1]
        self.gp = self.step.GP["combined_model"]

        self.powerstate = self.opt_fun.mitim_model.optimization_object.surrogate_parameters["powerstate"]

        # Read dictionaries
        with open(self.opt_fun.mitim_model.optimization_object.optimization_extra, "rb") as f:
            self.mitim_runs = pickle_dill.load(f)

        self.prep_metrics()

    @classmethod
    def from_folder(cls, folder, folderRemote=None, folderAnalysis=None):
        print(f"\n...Opening PORTALS class from folder {IOtools.clipstr(folder)}")

        if folder.exists() or folderRemote is not None:

            opt_fun = STRATEGYtools.opt_evaluator(folder)

            try:

                opt_fun.read_optimization_results(analysis_level=4, folderRemote=folderRemote)
                step = opt_fun.mitim_model.steps[-1]    # To trigger potential exception

                return cls(opt_fun, folderAnalysis=folderAnalysis)

            except (FileNotFoundError, AttributeError, IndexError) as e:
                print("\t- Could not read optimization results due to error:", typeMsg="w")
                print(f"\t\t{e}")
                print("\t- Trying to read PORTALS initialization...", typeMsg="i")

                opt_fun_ini = PORTALSinitializer(folder)

                # Attach to it what I could read from the original
                opt_fun_ini.opt_fun_full = opt_fun

                return opt_fun_ini
        else:
            print(
                "\t- Folder does not exist, are you sure you are on the right path?",
                typeMsg="w",
            )

    @classmethod
    def merge_instances(cls, instances, folderAnalysis=None, base_index=0):
        print(
            "\t- Merging PORTALSanalyzer instances by tricking evaluations counter",
            typeMsg="w",
        )

        merged_mitim_runs = {}
        cont = 0
        for instance in instances:
            for key in range(0, instance.ilast + 1):
                merged_mitim_runs[cont + key] = instance.mitim_runs[key]
            
            cont += instance.ilast + 1

        base_instance = instances[base_index]

        merged_instance = cls(base_instance.opt_fun, folderAnalysis)
        merged_instance.mitim_runs = merged_mitim_runs
        merged_instance.mitim_runs["profiles_original"] = base_instance.mitim_runs[
            "profiles_original"
        ]
        merged_instance.mitim_runs["profiles_modified"] = base_instance.mitim_runs[
            "profiles_modified"
        ]

        merged_instance.prep_metrics(ilast=cont - 1)

        return merged_instance

    @classmethod
    def merge_from_folders(cls, folders, folderAnalysis=None, base_index=0):
        instances = [cls.from_folder(folder) for folder in folders]
        return cls.merge_instances(
            instances, folderAnalysis=folderAnalysis, base_index=base_index
        )

    # ****************************************************************************
    # PREPARATION
    # ****************************************************************************

    def prep_metrics(self, calculateRicci={"d0": 2.0, "l": 1.0}, ilast=None):
        print("- Interpreting PORTALS results")

        # What's the last iteration?
        if ilast is None:
            # self.opt_fun.mitim_model.train_Y.shape[0]
            for ikey in self.mitim_runs:
                if not isinstance(self.mitim_runs[ikey], dict):
                    break
            self.ilast = ikey - 1
        else:
            self.ilast = ilast

        # Store indeces
        self.ibest = self.opt_fun.res.best_absolute_index
        self.i0 = 0

        if self.ilast == self.ibest:
            self.iextra = None
        else:
            self.iextra = self.ilast

        if self.mitim_runs[0] is None:
            print("* Issue with reading mitim_run 0, likely due to a cold_start of PORTALS simulation that took values from optimization_data.csv but did not generate powerstates", typeMsg="w")
            print("* This issue should be fixed in the future, have you contacted P. Rodriguez-Fernandez for help?", typeMsg="q")

        # Store setup of TGYRO run
        self.rhos   = self.mitim_runs[0]['powerstate'].plasma['rho'][0,1:].cpu().numpy()
        self.roa    = self.mitim_runs[0]['powerstate'].plasma['roa'][0,1:].cpu().numpy()

        self.PORTALSparameters = self.opt_fun.mitim_model.optimization_object.PORTALSparameters
        self.MODELparameters = self.opt_fun.mitim_model.optimization_object.MODELparameters

        # Useful flags
        self.ProfilesPredicted = self.MODELparameters["ProfilesPredicted"]

        self.runWithImpurity = self.powerstate.impurityPosition if "nZ" in self.ProfilesPredicted else None

        self.runWithRotation = "w0" in self.ProfilesPredicted
        self.includeFast = self.PORTALSparameters["includeFastInQi"]
        self.useConvectiveFluxes = self.PORTALSparameters["useConvectiveFluxes"]
        self.forceZeroParticleFlux = self.PORTALSparameters["forceZeroParticleFlux"]

        # Profiles and tgyro results
        print("\t- Reading profiles and tgyros for each evaluation")

        self.powerstates = []
        for i in range(self.ilast + 1):
            self.powerstates.append(self.mitim_runs[i]["powerstate"])

        # runWithImpurity_transport is stored after powerstate has run transport
        self.runWithImpurity_transport = self.powerstates[0].impurityPosition_transport if "nZ" in self.ProfilesPredicted else None


        if len(self.powerstates) <= self.ibest:
            print("\t- PORTALS was read after new residual was computed but before pickle was written!",typeMsg="w")
            self.ibest -= 1
            self.iextra = None

        self.profiles_next = None
        x_train_num = self.step.train_X.shape[0]
        file = self.opt_fun.folder / "Execution" / f"Evaluation.{x_train_num}" / "model_complete" / "input.gacode_unmodified"
        if file.exists():
            print("\t\t- Reading next profile to evaluate (from folder)")
            self.profiles_next = PROFILEStools.PROFILES_GACODE(file, calculateDerived=False)

            file = self.opt_fun.folder / "Execution" / f"Evaluation.{x_train_num}" / "model_complete" / "input.gacode.new"
            if file.exists():
                self.profiles_next_new = PROFILEStools.PROFILES_GACODE(
                    file, calculateDerived=False
                )
                self.profiles_next_new.printInfo(label="NEXT")
            else:
                self.profiles_next_new = self.profiles_next
                self.profiles_next_new.deriveQuantities()
        else:
            print("\t\t- Could not read next profile to evaluate (from folder)")

        print("\t- Processing metrics")

        self.evaluations, self.resM = [], []
        self.FusionGain, self.tauE, self.FusionPower = [], [], []
        self.resTe, self.resTi, self.resne, self.resnZ, self.resw0 = [], [], [], [], []
        if calculateRicci is not None:
            self.qR_Ricci, self.chiR_Ricci, self.points_Ricci = [], [], []
        else:
            self.qR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None

        for i, power in enumerate(self.powerstates):
            print(f"\t\t- Processing evaluation {i}/{len(self.powerstates)-1}")

            if 'Q' not in power.profiles.derived:
                power.profiles.deriveQuantities()

            self.evaluations.append(i)
            self.FusionGain.append(power.profiles.derived["Q"])
            self.FusionPower.append(power.profiles.derived["Pfus"])
            self.tauE.append(power.profiles.derived["tauE"])

            # ------------------------------------------------
            # Residual definitions
            # ------------------------------------------------

            _, _, source, res = PORTALSinteraction.calculate_residuals(
                power,
                self.PORTALSparameters,
            )

            # Make sense of tensor "source" which are defining the entire predictive set in
            Qe_resR = np.zeros(self.rhos.shape[0])
            Qi_resR = np.zeros(self.rhos.shape[0])
            Ge_resR = np.zeros(self.rhos.shape[0])
            GZ_resR = np.zeros(self.rhos.shape[0])
            Mt_resR = np.zeros(self.rhos.shape[0])
            cont = 0
            for prof in self.MODELparameters["ProfilesPredicted"]:
                for ix in range(self.rhos.shape[0]):
                    if prof == "te":
                        Qe_resR[ix] = source[0, cont].abs()
                    if prof == "ti":
                        Qi_resR[ix] = source[0, cont].abs()
                    if prof == "ne":
                        Ge_resR[ix] = source[0, cont].abs()
                    if prof == "nZ":
                        GZ_resR[ix] = source[0, cont].abs()
                    if prof == "w0":
                        Mt_resR[ix] = source[0, cont].abs()

                    cont += 1

            res = -res.item()

            self.resTe.append(Qe_resR)
            self.resTi.append(Qi_resR)
            self.resne.append(Ge_resR)
            self.resnZ.append(GZ_resR)
            self.resw0.append(Mt_resR)
            self.resM.append(res)

            # Ricci Metrics
            if calculateRicci is not None:
                try:
                    (
                        y1,
                        y2,
                        y1_std,
                        y2_std,
                    ) = PORTALSinteraction.calculate_residuals_distributions(
                        power,
                        self.PORTALSparameters,
                    )

                    QR, chiR = PLASMAtools.RicciMetric(
                        y1,
                        y2,
                        y1_std,
                        y2_std,
                        d0=calculateRicci["d0"],
                        l=calculateRicci["l"],
                    )

                    self.qR_Ricci.append(QR[0])
                    self.chiR_Ricci.append(chiR[0])
                    self.points_Ricci.append(
                        [
                            y1.cpu().numpy()[0, :],
                            y2.cpu().numpy()[0, :],
                            y1_std.cpu().numpy()[0, :],
                            y2_std.cpu().numpy()[0, :],
                        ]
                    )
                except:
                    print("\t- Could not calculate Ricci metric", typeMsg="w")
                    calculateRicci = None
                    self.qR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None

        self.labelsFluxes = power.labelsFluxes

        self.FusionGain = np.array(self.FusionGain)
        self.FusionPower = np.array(self.FusionPower)
        self.tauE = np.array(self.tauE)
        self.resM = np.array(self.resM)
        self.evaluations = np.array(self.evaluations)
        self.resTe, self.resTi, self.resne, self.resnZ, self.resw0 = (
            np.array(self.resTe),
            np.array(self.resTi),
            np.array(self.resne),
            np.array(self.resnZ),
            np.array(self.resw0),
        )

        if calculateRicci is not None:
            self.chiR_Ricci = np.array(self.chiR_Ricci)
            self.qR_Ricci = np.array(self.qR_Ricci)
            self.points_Ricci = np.array(self.points_Ricci)

        # Normalized L1 norms
        self.resTeM = np.abs(self.resTe).mean(axis=1)
        self.resTiM = np.abs(self.resTi).mean(axis=1)
        self.resneM = np.abs(self.resne).mean(axis=1)
        self.resnZM = np.abs(self.resnZ).mean(axis=1)
        self.resw0M = np.abs(self.resw0).mean(axis=1)

        self.resCheck = (
            self.resTeM + self.resTiM + self.resneM + self.resnZM + self.resw0M
        ) / len(self.MODELparameters["ProfilesPredicted"])

        # ---------------------------------------------------------------------------------------------------------------------
        # Jacobian
        # ---------------------------------------------------------------------------------------------------------------------

        DeltaQ1 = []
        for i in self.MODELparameters["ProfilesPredicted"]:
            if i == "te":
                DeltaQ1.append(-self.resTe)
            if i == "ti":
                DeltaQ1.append(-self.resTi)
            if i == "ne":
                DeltaQ1.append(-self.resne)
        DeltaQ1 = np.array(DeltaQ1)
        self.DeltaQ = DeltaQ1[0, :, :]
        for i in range(DeltaQ1.shape[0] - 1):
            self.DeltaQ = np.append(self.DeltaQ, DeltaQ1[i + 1, :, :], axis=1)

        self.aLTn_perc = None
        # try:	self.aLTn_perc  = calcLinearizedModel(self.opt_fun.mitim_model,self.DeltaQ,numChannels=self.numChannels,numRadius=self.numRadius,sepers=[self.i0, self.ibest])
        # except:	print('\t- Jacobian calculation failed',typeMsg='w')

        self.DVdistMetric_x = self.opt_fun.res.DVdistMetric_x
        self.DVdistMetric_y = self.opt_fun.res.DVdistMetric_y

    # ****************************************************************************
    # PLOTTING
    # ****************************************************************************

    def plotPORTALS(self, tabs_colors_common = None):
        if self.fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fn = FigureNotebook("PORTALS Summary", geometry="1700x1000")

        fig = self.fn.add_figure(label="PROFILES Ranges", tab_color=0 if tabs_colors_common is None else tabs_colors_common)
        self.plotRanges(fig=fig)

        self.plotSummary(fn=self.fn, fn_color=1 if tabs_colors_common is None else tabs_colors_common)

        fig = self.fn.add_figure(label="PORTALS Metrics", tab_color=2 if tabs_colors_common is None else tabs_colors_common)
        self.plotMetrics(fig=fig)

        fig = self.fn.add_figure(label="PORTALS Expected", tab_color=3 if tabs_colors_common is None else tabs_colors_common)
        self.plotExpected(fig=fig)

        fig = self.fn.add_figure(label="PORTALS Simulation", tab_color=4 if tabs_colors_common is None else tabs_colors_common)
        _, _ = self.plotModelComparison(fig=fig)

    def plotMetrics(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotMetrics(self, **kwargs)

    def plotExpected(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotExpected(self, **kwargs)

    def plotSummary(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotSummary(self, **kwargs)

    def plotRanges(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotRanges(self, **kwargs)

    def plotModelComparison(self, UseThisTGLFfull=None, **kwargs):
        UseTGLFfull_x = None

        if UseThisTGLFfull is not None:
            """
            UseThisTGLFfull should be a tuple (folder,label) where to read the
            results of running the method runTGLFfull() below.
            Note that it could be [None, label]
            """
            folder, label = UseThisTGLFfull
            if folder is None:
                folder = self.folder / "tglf_full"
            self.tglf_full = TGLFtools.TGLF(rhos=self.rhos)
            for ev in range(self.ilast + 1):
                self.tglf_full.read(
                    folder=folder / f"Evaluation.{ev}" / f"tglf_{label}", label=f"ev{ev}"
                )
            UseTGLFfull_x = label

        if self.mitim_runs[0]["powerstate"].model_results is not None:
            return PORTALSplot.PORTALSanalyzer_plotModelComparison(
                self, UseTGLFfull_x=UseTGLFfull_x, **kwargs
            )
        else:
            print("- No model results available to plot model comparisons", typeMsg="w")
            return None, None

    # ****************************************************************************
    # UTILITIES to extract aspects of PORTALS
    # ****************************************************************************

    def extractProfiles(self, evaluation=None, modified_profiles=False):
        '''
        modified_profiles: if True, it will extract the profiles that supposedly has been modified by the model (e.g. lumping, etc)
        '''
        if evaluation is None:
            evaluation = self.ibest
        elif evaluation < 0:
            evaluation = self.ilast

        powerstate = self.mitim_runs[evaluation]["powerstate"]

        try:
            p0 =  powerstate.profiles if not modified_profiles else powerstate.model_results.profiles
        except TypeError:
            raise Exception(f"[MITIM] Could not extract profiles from evaluation {evaluation}, are you sure you have the right index?")

        p = copy.deepcopy(p0)

        return p

    def extractModels(self, step=-1):
        if step < 0:
            step = len(self.opt_fun.mitim_model.steps) - 1

        gps = self.opt_fun.mitim_model.steps[step].GP["individual_models"]

        # Make dictionary
        models = {}
        for gp in gps:
            models[gp.output] = gp

        # PRINTING
        print(
            f"""
****************************************************************************************************
> MITIM has extracted {len(models)} GP models as a dictionary (only returned variable), to proceed:
    1. Look at the dictionary keys to see which models are available:
                models.keys()
    2. Select one model and print its information (e.g. variable labels and order):
                m = models['QeTurb_1']
                m.printInfo()
    3. Trained points are stored as m.x, m.y, m.yvar, and you can make predictions with:
                x_test = m.x
                mean, upper, lower = m(x_test)
    4. Extract samples from the GP with:
                x_test = m.x
                samples = m(x_test,samples=100)
****************************************************************************************************
""",
            typeMsg="i",
        )

        return wrapped_model_portals(models)

    def extractPORTALS(self, evaluation=None, folder=None, modified_profiles=False):
        if evaluation is None:
            evaluation = self.ibest
        elif evaluation < 0:
            evaluation = self.ilast

        if folder is None:
            folder = self.folder / f"portals_step{evaluation}"

        folder = IOtools.expandPath(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # Original class
        portals_fun_original = self.opt_fun.mitim_model.optimization_object

        # Start from the profiles of that step
        fileGACODE = folder / "input.gacode_transferred"
        p = self.extractProfiles(evaluation=evaluation, modified_profiles=modified_profiles)
        p.writeCurrentStatus(file=fileGACODE)

        # New class
        from mitim_modules.portals.PORTALSmain import portals

        portals_fun = portals(folder)

        # Transfer settings
        portals_fun.PORTALSparameters = portals_fun_original.PORTALSparameters
        portals_fun.MODELparameters = portals_fun_original.MODELparameters

        # PRINTING
        print(
            f"""
****************************************************************************************************
> MITIM has extracted PORTALS class to run in {IOtools.clipstr(folder)}, to proceed:
    1. Modify any parameter as required
                portals_fun.PORTALSparameters, portals_fun.MODELparameters, portals_fun.optimization_options
    2. Take the class portals_fun (arg #0) and prepare it with fileGACODE (arg #1) and folder (arg #2) with:
                portals_fun.prep(fileGACODE,folder)
    3. Run PORTALS with:
                mitim_bo = STRATEGYtools.MITIM_BO(portals_fun);     mitim_bo.run()
****************************************************************************************************
""",
            typeMsg="i",
        )

        return portals_fun, fileGACODE, folder

    def extractTGYRO(self, folder=None, cold_start=False, evaluation=0, modified_profiles=False):
        if evaluation is None:
            evaluation = self.ibest
        elif evaluation < 0:
            evaluation = self.ilast

        if folder is None:
            folder = self.folder / f"tgyro_step{evaluation}"

        folder = IOtools.expandPath(folder)
        folder.mkdir(parents=True, exist_ok=True)

        print(f"> Extracting and preparing TGYRO in {IOtools.clipstr(folder)}")

        profiles = self.extractProfiles(evaluation=evaluation, modified_profiles=modified_profiles)

        tgyro = TGYROtools.TGYRO()
        tgyro.prep(
            folder, profilesclass_custom=profiles, cold_start=cold_start, forceIfcold_start=True
        )

        TGLFsettings = self.MODELparameters["transport_model"]["TGLFsettings"]
        extraOptionsTGLF = self.MODELparameters["transport_model"]["extraOptionsTGLF"]
        PredictionSet = [
            int("te" in self.MODELparameters["ProfilesPredicted"]),
            int("ti" in self.MODELparameters["ProfilesPredicted"]),
            int("ne" in self.MODELparameters["ProfilesPredicted"]),
        ]

        return tgyro, self.rhos, PredictionSet, TGLFsettings, extraOptionsTGLF

    def extractTGLF(self, folder=None, positions=None, evaluation=None, cold_start=False, modified_profiles=False):
        if evaluation is None:
            evaluation = self.ibest
        elif evaluation < 0:
            evaluation = self.ilast

        """
        NOTE on radial location extraction:
        Two possible options for the rho locations to use:
            1. self.MODELparameters["RhoLocations"] -> the ones PORTALS sent to TGYRO
            2. self.rhos (came from TGYRO's t.rho[0, 1:]) -> the ones written by the TGYRO run (clipped to 7 decimal places)
        Because we want here to run TGLF *exactly* as TGYRO did, we use the first option.
        #TODO: This should be fixed in the future, we should never send to TGYRO more than 7 decimal places of any variable
        """
        rhos_considered = self.MODELparameters["RhoLocations"]

        if positions is None:
            rhos = rhos_considered
        else:
            rhos = []
            for i in positions:
                rhos.append(rhos_considered[i])

        if folder is None:
            folder = self.folder / f"tglf_ev{evaluation}"

        folder = IOtools.expandPath(folder)

        folder.mkdir(parents=True, exist_ok=True)

        print(f"> Extracting and preparing TGLF in {IOtools.clipstr(folder)} from evaluation #{evaluation}")

        inputgacode = folder / f"input.gacode.start"
        p = self.extractProfiles(evaluation=evaluation,modified_profiles=modified_profiles)
        p.writeCurrentStatus(file=inputgacode)

        tglf = TGLFtools.TGLF(rhos=rhos)
        _ = tglf.prep(folder, cold_start=cold_start, inputgacode=inputgacode)

        TGLFsettings = self.MODELparameters["transport_model"]["TGLFsettings"]
        extraOptions = self.MODELparameters["transport_model"]["extraOptionsTGLF"]

        return tglf, TGLFsettings, extraOptions

    # ****************************************************************************
    # UTILITIES for post-analysis
    # ****************************************************************************

    def runTGLFfull(
        self,
        folder=None,
        cold_start=False,
        label="default",
        tglf_object=None,
        onlyBest=False,
        **kwargsTGLF,
    ):
        """
        This runs TGLF for all evaluations, all radii.
        This is convenient if I want to re=run TGLF with different settings, e.g. different TGLFsettings,
        that you can provide as keyword arguments.
        """

        if folder is None:
            folder = self.folder / "tglf_full"

        folder.mkdir(parents=True, exist_ok=True)

        if onlyBest:
            ranges = [self.ibest]
        else:
            ranges = range(self.ilast + 1)

        for ev in ranges:
            tglf, TGLFsettings, extraOptions = self.extractTGLF(
                folder=folder / f"Evaluation.{ev}", evaluation=ev, cold_start=cold_start
            )

            kwargsTGLF_this = copy.deepcopy(kwargsTGLF)

            if "TGLFsettings" not in kwargsTGLF_this:
                kwargsTGLF_this["TGLFsettings"] = TGLFsettings
            if "extraOptions" not in kwargsTGLF_this:
                kwargsTGLF_this["extraOptions"] = extraOptions

            tglf.run(subFolderTGLF=f"tglf_{label}", cold_start=cold_start, **kwargsTGLF_this)

        # Read all previously run cases into a single class
        if tglf_object is None:
            tglf_object = copy.deepcopy(tglf)

        for ev in ranges:
            tglf_object.read(
                folder=folder / f"Evaluation.{ev}" / f"tglf_{label}",
                label=f"{label}_ev{ev}",
            )

        return tglf_object

    def runCases(self, onlyBest=False, cold_start=False, fn=None):
        from mitim_modules.portals.PORTALSmain import runModelEvaluator

        variations_best = self.opt_fun.res.best_absolute_full["x"]
        variations_original = self.opt_fun.res.evaluations[0]["x"]

        if not onlyBest:
            print("\t- Running original case")
            FolderEvaluation = self.folder / f"final_analysis_original"
            if not FolderEvaluation.exists():
                IOtools.askNewFolder(FolderEvaluation, force=True)

            dictDVs = {}
            for i in variations_best:
                dictDVs[i] = {"value": variations_original[i]}

            # Run
            a, b = IOtools.reducePathLevel(self.folder, level=1)
            name0 = f"portals_{b}_ev{0}"  # e.g. portals_jet37_ev0

            tgyroO, powerstateO, _ = runModelEvaluator(
                self.opt_fun.mitim_model.optimization_object,
                FolderEvaluation,
                dictDVs,
                name0,
                cold_start=cold_start,
            )

        print(f"\t- Running best case #{self.opt_fun.res.best_absolute_index}")
        FolderEvaluation = self.folder / "Outputs" / "final_analysis_best"
        if not FolderEvaluation.exists():
            IOtools.askNewFolder(FolderEvaluation, force=True)

        dictDVs = {}
        for i in variations_best:
            dictDVs[i] = {"value": variations_best[i]}

        # Run
        a, b = IOtools.reducePathLevel(self.folder, level=1)
        name = f"portals_{b}_ev{self.res.best_absolute_index}"  # e.g. portals_jet37_ev0
        tgyroB, powerstateB, _ = runModelEvaluator(
            self.opt_fun.mitim_model.optimization_object,
            FolderEvaluation,
            dictDVs,
            name,
            cold_start=cold_start,
        )

        # Plot
        if fn is not None:
            if not onlyBest:
                tgyroO.plot(fn=fn, labels=[name0])
            tgyroB.plot(fn=fn, labels=[name])


# ****************************************************************************
# Helpers
# ****************************************************************************


class wrapped_model_portals:
    def __init__(self, gpdict):
        self._models = {}
        self._targets = {}
        self._input_variables = []
        self._output_variables = []
        self._training_inputs = {}
        self._training_outputs = {}
        if isinstance(gpdict, dict):
            for key in gpdict:
                if 'Tar' in key:
                    self._targets[key] = gpdict[key]
                else:
                    self._models[key] = gpdict[key]
        for key in self._models:
            if hasattr(self._models[key], 'variables'):
                for var in self._models[key].variables:
                    if var not in self._input_variables:
                        self._input_variables.append(var)
                if key not in self._output_variables:
                    self._output_variables.append(key)
        for key in self._models:
            if hasattr(self._models[key], 'gpmodel'):
                if hasattr(self._models[key].gpmodel, 'train_X_usedToTrain'):
                    xtrain = self._models[key].gpmodel.train_X_usedToTrain.detach().cpu().numpy()
                    if len(xtrain.shape) < 2:
                        xtrain = np.atleast_2d(xtrain)
                    if xtrain.shape[1] != len(self._input_variables):
                        xtrain = xtrain.T
                    self._training_inputs[key] = pd.DataFrame(xtrain, columns=self._input_variables)
                if hasattr(self._models[key].gpmodel, 'train_Y_usedToTrain'):
                    ytrain = self._models[key].gpmodel.train_Y_usedToTrain.detach().cpu().numpy()
                    if len(ytrain.shape) < 2:
                        ytrain = np.atleast_2d(ytrain)
                    if ytrain.shape[1] != 1:
                        ytrain = ytrain.T
                    self._training_outputs[key] = pd.DataFrame(ytrain, columns=[key])

    @property
    def models(self):
        return self._models

    @property
    def targets(self):
        return self._targets

    @property
    def input_variables(self):
        return copy.deepcopy(self._input_variables)

    @property
    def output_variables(self):
        return copy.deepcopy(self._output_variables)

    @property
    def training_inputs(self):
        return copy.deepcopy(self._training_inputs)

    @property
    def training_outputs(self):
        return copy.deepcopy(self._training_outputs)

    def printInfo(self, detailed=False):
        print(f"> Models for {self.output_variables} created")
        print(
            f"\t- Requires {len(self.input_variables)} variables to evaluate: {self.input_variables}"
        )
        if detailed:
            for key, model in self._models.items():
                model.printInfo()

    def evalModel(self, x, key):
        numpy_provided = False
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            numpy_provided = True

        mean, upper, lower, _ = self._models[key].predict(
            x, produceFundamental=True
        )

        mean_out = mean[..., 0].detach()
        upper_out = upper[..., 0].detach()
        lower_out = lower[..., 0].detach()
        if numpy_provided:
            mean_out = mean_out.cpu().numpy()
            upper_out = upper_out.cpu().numpy()
            lower_out = lower_out.cpu().numpy()

        return mean_out, upper_out, lower_out

    def sampleModel(self, x, samples, key):
        numpy_provided = False
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            numpy_provided = True

        _, _, _, samples = self._models[key].predict(
            x, produceFundamental=True, nSamples=samples
        )

        samples_out = samples[..., 0].detach()
        if numpy_provided:
            samples_out = samples_out.cpu().numpy()

        return samples_out

    def predict(self, x, outputs=None):
        y = {}
        targets = outputs if isinstance(outputs, (list, tuple)) else list(self._models.keys())
        for ytag, model in self._models.items():
            if ytag in targets:
                inp = copy.deepcopy(x)
                if isinstance(x, pd.DataFrame):
                    inp = x.loc[:, model.variables].to_numpy()
                y[f'{ytag}'], y[f'{ytag}_upper'], y[f'{ytag}_lower'] = self.evalModel(inp, ytag)
        return pd.DataFrame(data=y)

    def sample(self, x, samples, outputs=None):
        y = {}
        targets = outputs if isinstance(outputs, (list, tuple)) else list(self._models.keys())
        for ytag, model in self._models.items():
            if ytag in targets:
                inp = copy.deepcopy(x)
                if isinstance(x, pd.DataFrame):
                    inp = x.loc[:, model.variables].to_numpy()
                y[f'{ytag}'] = self.sampleModel(inp, samples, ytag)
        return pd.DataFrame(data=y)

    def __call__(self, x, samples=None, outputs=None):
        out = None
        if samples is None:
            out = self.predict(x, outputs=outputs)
        else:
            out = self.sample(x, samples=samples, outputs=outputs)
        return out

    def generateScan(self, output, iteration=-1, scan_range=0.5, scan_resolution=3):
        scan_list = []
        if output in self._training_inputs:
            base = self._training_inputs[output]
            idx = base.index.values[iteration]
            for var in base:
                scan_length = int(scan_resolution) if isinstance(scan_resolution, (float, int)) else 3
                scan_data = {}
                if scan_length > 1:
                    scan_width = float(scan_range) if isinstance(scan_range, (float, int)) else 0.5
                    scan_data = {key: np.array([base.loc[idx, key]] * scan_length) for key in base}
                    scan_data[var] = (1.0 + np.linspace(-1.0, 1.0, scan_length) * scan_width) * scan_data[var]
                else:
                    scan_data = base.loc[idx, :].to_dict()
                for key in scan_data:
                    scan_data[key] = np.atleast_1d(scan_data[key])
                scan_list.append(pd.DataFrame(data=scan_data))
        return pd.concat(scan_list, axis=0, ignore_index=True)


def calcLinearizedModel(
    mitim_model, DeltaQ, posBase=-1, numChannels=3, numRadius=4, sepers=[]
):
    """
    posBase = 1 is aLTi, 0 is aLTe, if the order is [a/LTe,aLTi]
    -1 is diagonal
    -2 is

    NOTE for PRF: THIS ONLY WORKS FOR TURBULENCE, nOT NEO!
    """

    trainx = mitim_model.steps[-1].GP["combined_model"].train_X.cpu().numpy()

    istep, aLTn_est, aLTn_base = 0, [], []
    for i in range(trainx.shape[0]):
        if i >= mitim_model.optimization_options["initialization_options"]["initial_training"]:
            istep += 1

        # Jacobian
        J = (
            mitim_model.steps[istep]
            .GP["combined_model"]
            .localBehavior(trainx[i, :], plotYN=False)
        )

        J = 1e-3 * J[: trainx.shape[1], : trainx.shape[1]]  # Only turbulence

        print(f"\t- Reading Jacobian for step {istep}")

        Q = DeltaQ[i, :]

        if posBase < 0:
            # All channels together ------------------------------------------------
            mult = torch.Tensor()
            for i in range(12):
                if posBase == -1:
                    a = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Diagonal
                elif posBase == -2:
                    a = torch.Tensor(
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                    )  # Block diagonal
                a = torch.roll(a, i)
                mult = torch.cat((mult, a.unsqueeze(0)), dim=0)

            J_reduced = J * mult
            aLTn = (J_reduced.inverse().cpu().numpy()).dot(Q)
            aLTn_base0 = trainx[i, :]
            # ------------------------------------------------------------------------

        else:
            # Channel per channel, only ion temperature gradient ------------------------
            J_mod = []
            aLTn_base0 = []
            cont = 0
            for c in range(numChannels):
                for r in range(numRadius):
                    J_mod.append(J[cont, posBase * numRadius + r].cpu().numpy())
                    aLTn_base0.append(trainx[i, posBase * numRadius + r])
                    cont += 1
            J_mod = np.array(J_mod)
            aLTn_base0 = np.array(aLTn_base0)
            aLTn = Q / J_mod
            # ------------------------------------------------------------------------

        aLTn_base.append(aLTn_base0)
        aLTn_est.append(aLTn)

    aLTn_est = np.array(aLTn_est)
    aLTn_base = np.array(aLTn_base)

    aLTn_perc = [
        np.abs(i / j) * 100.0 if i is not None else None
        for i, j in zip(aLTn_est, aLTn_base)
    ]

    return aLTn_perc


class PORTALSinitializer:
    def __init__(self, folder):
        self.folder = IOtools.expandPath(folder)

        # Read powerstates
        self.powerstates = []
        self.profiles = []
        for i in range(100):
            try:
                prof = PROFILEStools.PROFILES_GACODE(
                    self.folder / "Outputs" / "portals_profiles" / f"input.gacode.{i}"
                )
            except FileNotFoundError:
                break
            self.profiles.append(prof)

        for i in range(100):
            try:
                p = STATEtools.read_saved_state(
                    self.folder / "Initialization" / "initialization_simple_relax" / f"portals_sr_{IOtools.reducePathLevel(self.folder)[1]}_ev_{i}" / "powerstate.pkl"
                )
            except FileNotFoundError:
                break

            p.profiles.deriveQuantities()
            self.powerstates.append(p)

        self.fn = None

    def plotMetrics(self, extra_lab="", **kwargs):

        if len(self.powerstates) == 0:
            print("- No powerstates available to plot metrics", typeMsg="w")
            return

        # Prepare figure --------------------------------------------------
        if 'fig' in kwargs and kwargs['fig'] is not None:
            print('Using provided figure, assuming I only want a summary')
            figMain = kwargs['fig']
            figG = None
        else:
            if self.fn is None:
                from mitim_tools.misc_tools.GUItools import FigureNotebook
                self.fn = FigureNotebook("PowerState", geometry="1800x900")
            figMain = self.fn.add_figure(label=f"{extra_lab} - PowerState")
            figG = self.fn.add_figure(label=f"{extra_lab} - Sequence")
        # -----------------------------------------------------------------

        axs, axsM = STATEtools.add_axes_powerstate_plot(figMain, num_kp=np.max([3,len(self.powerstates[-1].ProfilesPredicted)]))

        colors = GRAPHICStools.listColors()
        axsGrads_extra = []
        cont = 0
        for i in range(np.max([3,len(self.powerstates[-1].ProfilesPredicted)])):
            axsGrads_extra.append(axs[cont])
            axsGrads_extra.append(axs[cont+1])
            cont += 4

        # ---------------------------------------------------------------------------------
        # POWERPLOT
        # ---------------------------------------------------------------------------------

        if len(self.powerstates) > 0:
            for i in range(len(self.powerstates)):
                self.powerstates[i].plot(axs=axs, c=colors[i], label=f"#{i}")

                # Add profiles too
                self.powerstates[i].profiles.plotGradients(
                    axsGrads_extra,
                    color=colors[i],
                    plotImpurity=self.powerstates[-1].impurityPosition if 'nZ' in self.powerstates[-1].ProfilesPredicted else None,
                    plotRotation='w0' in self.powerstates[0].ProfilesPredicted,
                    ls='-',
                    lw=0.5,
                    lastRho=self.powerstates[0].plasma["rho"][-1, -1].item(),
                    label='',
                )

            axs[0].legend(prop={"size": 8})

        # Add next profile
        if len(self.profiles) > len(self.powerstates):
            self.profiles[-1].plotGradients(
                axsGrads_extra,
                color=colors[i+1],
                plotImpurity=self.powerstates[-1].impurityPosition_transport if 'nZ' in self.powerstates[-1].ProfilesPredicted else None,
                plotRotation='w0' in self.powerstates[0].ProfilesPredicted,
                ls='-',
                lw=1.0,
                lastRho=self.powerstates[0].plasma["rho"][-1, -1].item(),
                label=f"next ({len(self.profiles)-len(self.powerstates)})",
            )

        # Metrics
        POWERplot.plot_metrics_powerstates(
            axsM,
            self.powerstates,
            profiles = self.profiles[-1] if len(self.profiles) > len(self.powerstates) else None,
            profiles_color=colors[i+1],
        )

        # GRADIENTS
        if figG is not None:
            if len(self.powerstates) > 0:
                grid = plt.GridSpec(2, 5, hspace=0.3, wspace=0.3)
                axsGrads = []
                for j in range(5):
                    for i in range(2):
                        axsGrads.append(figG.add_subplot(grid[i, j]))
                for i, p in enumerate(self.powerstates):
                    p.profiles.plotGradients(
                        axsGrads,
                        color=colors[i],
                        plotImpurity=p.impurityPosition if 'nZ' in p.ProfilesPredicted else None,
                        plotRotation='w0' in p.ProfilesPredicted,
                        lastRho=self.powerstates[0].plasma["rho"][-1, -1].item(),
                        label=f"profile #{i}",
                    )


                if len(self.profiles) > len(self.powerstates):
                    prof = self.profiles[-1]
                    prof.plotGradients(
                        axsGrads,
                        color=colors[i+1],
                        plotImpurity=p.impurityPosition_transport if 'nZ' in p.ProfilesPredicted else None,
                        plotRotation='w0' in p.ProfilesPredicted,
                        lastRho=p.plasma["rho"][-1, -1].item(),
                        label="next",
                    )

                axs[0].legend(prop={"size": 8})
                axsGrads[0].legend(prop={"size": 8})
