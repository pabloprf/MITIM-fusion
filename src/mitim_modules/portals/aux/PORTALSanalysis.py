import os
import numpy as np
import dill as pickle_dill
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import IOtools,PLASMAtools
from mitim_tools.gacode_tools import TGLFtools,TGYROtools,PROFILEStools
from mitim_tools.gacode_tools.aux import PORTALSinteraction
from mitim_modules.portals.aux import PORTALSplot

class PORTALSanalyzer:

    # ****************************************************************************
    # INITIALIZATION
    # ****************************************************************************

    def __init__(self, opt_fun, folderAnalysis=None):

        print('\n************************************')
        print("* Initializing PORTALS analyzer...")
        print('************************************')

        self.opt_fun = opt_fun
        self.prep(folderAnalysis)
    
    @classmethod
    def from_folder(cls,folder,folderRemote=None, folderAnalysis=None):

        print(f"\n...opening PORTALS class from folder {IOtools.clipstr(folder)}")

        opt_fun = STRATEGYtools.FUNmain(folder)
        opt_fun.read_optimization_results(analysis_level=4, plotYN=False, folderRemote=folderRemote)

        return cls(opt_fun, folderAnalysis=folderAnalysis)

    # ****************************************************************************
    # PREPARATION
    # ****************************************************************************

    def prep(self,folderAnalysis,calculateRicci  = {"d0": 2.0, "l": 1.0}):
        with open(self.opt_fun.prfs_model.mainFunction.MITIMextra, "rb") as f:
            self.mitim_runs = pickle_dill.load(f)

        for ikey in self.mitim_runs:
            if not isinstance(self.mitim_runs[ikey],dict):
                break
        self.ilast = ikey - 1

        self.ibest = self.opt_fun.res.best_absolute_index
        self.rhos = self.mitim_runs[0]['tgyro'].results['tglf_neo'].rho[0,1:]
        self.roa = self.mitim_runs[0]['tgyro'].results['tglf_neo'].roa[0,1:]
        
        self.PORTALSparameters = self.opt_fun.prfs_model.mainFunction.PORTALSparameters
        self.TGYROparameters = self.opt_fun.prfs_model.mainFunction.TGYROparameters
        self.TGLFparameters = self.opt_fun.prfs_model.mainFunction.TGLFparameters

        self.folder = folderAnalysis if folderAnalysis is not None else f'{self.opt_fun.folder}/Analysis/'

        if not os.path.exists(self.folder): os.system(f'mkdir {self.folder}')

        self.runWithImpurity = (
            self.PORTALSparameters["ImpurityOfInterest"]
            if "nZ" in self.TGYROparameters["ProfilesPredicted"]
            else None
        )
        self.runWithRotation = (
            "w0" in self.TGYROparameters["ProfilesPredicted"]
        )

        print("- Interpreting results...")

        includeFast = self.opt_fun.prfs_model.mainFunction.PORTALSparameters["includeFastInQi"]
        impurityPosition = self.opt_fun.prfs_model.mainFunction.PORTALSparameters[
            "ImpurityOfInterest"
        ]
        self.useConvectiveFluxes = self.opt_fun.prfs_model.mainFunction.PORTALSparameters[
            "useConvectiveFluxes"
        ]

        self.numChannels = len(
            self.opt_fun.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]
        )
        self.numRadius = len(
            self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"]
        )
        self.numBest = self.opt_fun.res.best_absolute_index
        self.numOrig = 0
        self.numExtra = self.opt_fun.prfs_model.train_Y.shape[0]-1

        self.sepers = [self.numOrig, self.numBest]

        if (self.numExtra is not None) and (self.numExtra == self.numBest):
            self.numExtra = None

        self.posZ = self.opt_fun.prfs_model.mainFunction.PORTALSparameters["ImpurityOfInterest"] - 1

        # Profiles and tgyro results
        print("\t- Reading profiles and tgyros for each evaluation")
        self.profiles, self.tgyros = [], []
        for i in range(self.opt_fun.prfs_model.train_Y.shape[0]):
            # print(f'\t- Reading TGYRO results and PROFILES for evaluation {i}/{self.opt_fun.prfs_model.train_Y.shape[0]-1}')
            if self.mitim_runs is not None:
                # print('\t\t* Reading from self.mitim_runs',typeMsg='i')
                self.tgyros.append(self.mitim_runs[i]["tgyro"].results["use"])
                self.profiles.append(
                    self.mitim_runs[i]["tgyro"].results["use"].profiles_final
                )
            elif self.opt_fun.folder is not None:
                # print('\t\t* Reading from scratch from folders (will surely take longer)',typeMsg='i')
                folderEvaluation = f"{self.opt_fun.folder}/Execution/Evaluation.{i}/"
                self.profiles.append(
                    PROFILEStools.PROFILES_GACODE(
                        folderEvaluation + "/model_complete/input.gacode.new"
                    )
                )
                self.tgyros.append(
                    TGYROtools.TGYROoutput(
                        folderEvaluation + "model_complete/", profiles=self.profiles[i]
                    )
                )
            else:
                print("Neither MITIMextra nor folder were provided",typeMsg='w')

        if len(self.profiles) <= self.numBest:
            print(
                "\t- PORTALS was read after new residual was computed but before pickle was written!",
                typeMsg="w",
            )
            self.numBest -= 1
            self.numExtra = None

        # Create some metrics

        print("\t- Processing metrics")

        self.evaluations, self.resM = [], []
        self.FusionGain, self.tauE, self.FusionPower = [], [], []
        self.resTe, self.resTi, self.resne, self.resnZ, self.resw0 = [], [], [], [], []
        if calculateRicci is not None:
            self.QR_Ricci, self.chiR_Ricci, self.points_Ricci = [], [], []
        else:
            self.QR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None
        
        for i, (p, t) in enumerate(zip(self.profiles, self.tgyros)):

            print(f"\t\t- Processing evaluation {i}/{len(self.profiles)-1}")
            
            self.evaluations.append(i)
            self.FusionGain.append(p.derived["Q"])
            self.FusionPower.append(p.derived["Pfus"])
            self.tauE.append(p.derived["tauE"])

            # ------------------------------------------------
            # Residual definitions
            # ------------------------------------------------

            powerstate = self.opt_fun.prfs_model.mainFunction.powerstate

            try:
                OriginalFimp = powerstate.TransportOptions["ModelOptions"][
                    "OriginalFimp"
                ]
            except:
                OriginalFimp = 1.0

            portals_variables = t.TGYROmodeledVariables(
                useConvectiveFluxes=self.useConvectiveFluxes,
                includeFast=includeFast,
                impurityPosition=impurityPosition,
                ProfilesPredicted=self.opt_fun.prfs_model.mainFunction.TGYROparameters[
                    "ProfilesPredicted"
                ],
                UseFineGridTargets=self.opt_fun.prfs_model.mainFunction.PORTALSparameters[
                    "fineTargetsResolution"
                ],
                OriginalFimp=OriginalFimp,
                forceZeroParticleFlux=self.opt_fun.prfs_model.mainFunction.PORTALSparameters[
                    "forceZeroParticleFlux"
                ],
            )

            if (
                len(powerstate.plasma["volp"].shape) > 1
                and powerstate.plasma["volp"].shape[1] > 1
            ):
                powerstate.unrepeat(do_fine=False)
                powerstate.repeat(do_fine=False)

            _, _, source, res = PORTALSinteraction.calculatePseudos(
                portals_variables["var_dict"],
                self.opt_fun.prfs_model.mainFunction.PORTALSparameters,
                self.opt_fun.prfs_model.mainFunction.TGYROparameters,
                powerstate,
            )

            # Make sense of tensor "source" which are defining the entire predictive set in
            Qe_resR = np.zeros(
                len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            Qi_resR = np.zeros(
                len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            Ge_resR = np.zeros(
                len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            GZ_resR = np.zeros(
                len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            Mt_resR = np.zeros(
                len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            cont = 0
            for prof in self.opt_fun.prfs_model.mainFunction.TGYROparameters[
                "ProfilesPredicted"
            ]:
                for ix in range(
                    len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
                ):
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
                    ) = PORTALSinteraction.calculatePseudos_distributions(
                        portals_variables["var_dict"],
                        self.opt_fun.prfs_model.mainFunction.PORTALSparameters,
                        self.opt_fun.prfs_model.mainFunction.TGYROparameters,
                        powerstate,
                    )

                    QR, chiR = PLASMAtools.RicciMetric(
                        y1,
                        y2,
                        y1_std,
                        y2_std,
                        d0=calculateRicci["d0"],
                        l=calculateRicci["l"],
                    )
                    self.QR_Ricci.append(QR[0])
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
                    self.QR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None

        self.labelsFluxes = portals_variables["labels"]

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
            self.QR_Ricci = np.array(self.QR_Ricci)
            self.points_Ricci = np.array(self.points_Ricci)

        # Normalized L1 norms
        self.resTeM = np.abs(self.resTe).mean(axis=1)
        self.resTiM = np.abs(self.resTi).mean(axis=1)
        self.resneM = np.abs(self.resne).mean(axis=1)
        self.resnZM = np.abs(self.resnZ).mean(axis=1)
        self.resw0M = np.abs(self.resw0).mean(axis=1)

        self.resCheck = (
            self.resTeM + self.resTiM + self.resneM + self.resnZM + self.resw0M
        ) / len(self.opt_fun.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"])

        # ---------------------------------------------------------------------------------------------------------------------
        # Jacobian
        # ---------------------------------------------------------------------------------------------------------------------

        DeltaQ1 = []
        for i in self.opt_fun.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
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

        self.aLTn_perc = aLTi_perc = None
        # try:	self.aLTn_perc  = aLTi_perc  = calcLinearizedModel(self.opt_fun.prfs_model,self.DeltaQ,numChannels=self.numChannels,numRadius=self.numRadius,sepers=self.sepers)
        # except:	print('\t- Jacobian calculation failed',typeMsg='w')

        self.DVdistMetric_x = self.opt_fun.res.DVdistMetric_x
        self.DVdistMetric_y = self.opt_fun.res.DVdistMetric_y


    # ****************************************************************************
    # PLOTTING
    # ****************************************************************************

    def plotPORTALS(self,fn=None):

        if fn is None:
            plt.ioff()
            from mitim_tools.misc_tools.GUItools import FigureNotebook
            fn = FigureNotebook(0, "PORTALS Summary", geometry="1700x1000")
            fnprov = False
        else:
            fnprov = True

        fig = fn.add_figure(label="PROFILES Ranges")
        self.plotRanges(fig=fig)

        self.plotSummary(fn)

        fig = fn.add_figure(label="PORTALS Metrics")
        self.plotMetrics(fig=fig)

        fig = fn.add_figure(label="PORTALS Expected")
        self.plotExpected(fig=fig)

        if not fnprov:
            fn.show()

    def plotMetrics(self,fig=None,indexToMaximize=None,plotAllFluxes=False,index_extra=None,file_save=None):
        PORTALSplot.PORTALSanalyzer_plotMetrics(self,fig=fig,indexToMaximize=indexToMaximize,plotAllFluxes=plotAllFluxes,index_extra=index_extra,file_save=file_save)

    def plotExpected(self,fig=None,stds = 2, max_plot_points=4,plotNext=True):
        PORTALSplot.PORTALSanalyzer_plotExpected(self,fig=fig,stds=stds,max_plot_points=max_plot_points,plotNext=plotNext)

    def plotSummary(self,fn):
        PORTALSplot.PORTALSanalyzer_plotSummary(self,fn)

    def plotRanges(self,fig=None):
        PORTALSplot.PORTALSanalyzer_plotRanges(self,fig=fig)

    def plotModelComparison(self,axs = None,GB=True,radial_label=True):
        PORTALSplot.PORTALSanalyzer_plotModelComparison(self,axs=axs,GB=GB,radial_label=radial_label)
        
    # ****************************************************************************
    # ADDITIONAL UTILITIES
    # ****************************************************************************

    def extractPROFILES(self, based_on_last=False, true_original=True):

        if based_on_last:
            i = self.ilast
        else:
            i = self.ibest

        p_new = self.mitim_runs[i]['tgyro'].results['tglf_neo'].profiles_final

        if true_original:
            p_orig = self.mitim_runs["profiles_original"]
        else:
            p_orig =  self.mitim_runs[0]['tgyro'].results['tglf_neo'].profiles_final

        return p_orig, p_new

    def extractTGYRO_init(self,folder=None,restart=False):

        if folder is None:
            folder = f'{self.folder}/tgyro_step0/' 

        folder = IOtools.expandPath(folder)
        if not os.path.exists(folder): os.system(f'mkdir {folder}')

        print(f"> Extracting and preparing TGLF in {IOtools.clipstr(folder)}")

        profiles = self.mitim_runs[0]["tgyro"].profiles

        tgyro = TGYROtools.TGYRO()
        tgyro.prep(folder, profilesclass_custom=profiles, restart=restart, forceIfRestart=True)

        TGLFsettings = self.TGLFparameters["TGLFsettings"]
        extraOptionsTGLF = self.TGLFparameters["extraOptionsTGLF"]
        PredictionSet=[
                    int("te" in self.TGYROparameters["ProfilesPredicted"]),
                    int("ti" in self.TGYROparameters["ProfilesPredicted"]),
                    int("ne" in self.TGYROparameters["ProfilesPredicted"]),
                ]


        return tgyro,self.rhos,PredictionSet,TGLFsettings,extraOptionsTGLF


    def extractTGLF(self,folder=None,positions=None,step=-1,restart=False):        

        if step < 0:
            step = self.ibest

        if positions is None:
            rhos = self.rhos
        else:
            rhos = []
            for i in positions:
                rhos.append(self.rhos[i])

        if folder is None:
            folder = f'{self.folder}/tglf_step{step}/' 

        folder = IOtools.expandPath(folder)

        if not os.path.exists(folder): os.system(f'mkdir {folder}')

        print(f"> Extracting and preparing TGLF in {IOtools.clipstr(folder)} from step #{step}")

        inputgacode = f"{folder}/input.gacode.start"
        self.mitim_runs[step]['tgyro'].profiles.writeCurrentStatus(file=inputgacode)

        tglf = TGLFtools.TGLF(rhos=rhos)
        _ = tglf.prep(folder, restart=restart, inputgacode=inputgacode)

        TGLFsettings = self.TGLFparameters["TGLFsettings"]
        extraOptions = self.TGLFparameters["extraOptionsTGLF"]

        return tglf, TGLFsettings, extraOptions

    def runTGLF(self,folder=None,positions=None,step=-1,restart=False):

        tglf, TGLFsettings, extraOptions = self.extractTGLF(folder=folder,positions=positions,step=step,restart=restart)

        tglf.run(
            subFolderTGLF="tglf_standalone/",
            TGLFsettings=TGLFsettings,
            extraOptions=extraOptions,
            restart=restart,
        )

        tglf.read(label='tglf_standalone')

        tglf.plotRun(labels=['tglf_standalone'])

    def runCases(self,onlyBest=False,restart=False,fn=None):

        from mitim_modules.portals.PORTALSmain import runModelEvaluator

        variations_best = self.opt_fun.res.best_absolute_full["x"]
        variations_original = self.opt_fun.res.evaluations[0]["x"]

        if not onlyBest:
            print("\t- Running original case")
            FolderEvaluation = f"{self.folder}/final_analysis_original/"
            if not os.path.exists(FolderEvaluation):
                IOtools.askNewFolder(FolderEvaluation, force=True)

            dictDVs = {}
            for i in variations_best:
                dictDVs[i] = {"value": variations_original[i]}

            # Run
            a, b = IOtools.reducePathLevel(self.folder, level=1)
            name0 = f"portals_{b}_ev{0}"  # e.g. portals_jet37_ev0

            resultsO, tgyroO, powerstateO, _ = runModelEvaluator(
                self.opt_fun.prfs_model.mainFunction, FolderEvaluation, 0, dictDVs, name0, restart=restart
            )

        print(f"\t- Running best case #{self.opt_fun.res.best_absolute_index}")
        FolderEvaluation = f"{self.folder}/Outputs/final_analysis_best/"
        if not os.path.exists(FolderEvaluation):
            IOtools.askNewFolder(FolderEvaluation, force=True)

        dictDVs = {}
        for i in variations_best:
            dictDVs[i] = {"value": variations_best[i]}

        # Run
        a, b = IOtools.reducePathLevel(self.folder, level=1)
        name = f"portals_{b}_ev{self.res.best_absolute_index}"  # e.g. portals_jet37_ev0
        resultsB, tgyroB, powerstateB, _ = runModelEvaluator(
            self.opt_fun.prfs_model.mainFunction,
            FolderEvaluation,
            self.res.best_absolute_index,
            dictDVs,
            name,
            restart=restart,
        )

        # Plot
        if fn is not None:
            if not onlyBest:
                tgyroO.plotRun(fn=fn, labels=[name0])
            tgyroB.plotRun(fn=fn, labels=[name])

def fix_pickledstate(state_to_mod, powerstate_to_add):
    """
    If I have modified the source code of powerstate, it won't be able to load an old PORTALS
    What you can do here is to insert an updated class. You can do this by running a bit the code restarting, to get portals_fun
    and then apply this:

    E.g.:
            fix_pickledstate('sparc_cgyro1/Outputs/MITIMstate.pkl',portals_fun.surrogate_parameters['powerstate'])
    """

    with open(state_to_mod, "rb") as f:
        aux = pickle_dill.load(f)

    # Powerstate is stored at the highest level
    aux.surrogate_parameters["powerstate"] = powerstate_to_add
    aux.mainFunction.powerstate = powerstate_to_add

    for i in range(len(aux.steps)):
        # Surrogate parameters are used in stepSettings at each step
        aux.steps[i].stepSettings["surrogate_parameters"][
            "powerstate"
        ] = powerstate_to_add

        # Surrogate parameters are passed to each individual surrogate model, at each step, in surrogate_parameters_extended
        for j in range(len(aux.steps[i].GP["individual_models"])):
            aux.steps[i].GP["individual_models"][j].surrogate_parameters_step[
                "powerstate"
            ] = powerstate_to_add

    with open(state_to_mod, "wb") as f:
        pickle_dill.dump(aux, f)