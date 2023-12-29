import os
import torch
import numpy as np
import dill as pickle_dill
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import IOtools,PLASMAtools
from mitim_tools.gacode_tools import TGLFtools,TGYROtools,PROFILEStools
from mitim_tools.gacode_tools.aux import PORTALSinteraction
from mitim_modules.portals.aux import PORTALSplot

from mitim_tools.misc_tools.IOtools import printMsg as print

class PORTALSanalyzer:

    # ****************************************************************************
    # INITIALIZATION
    # ****************************************************************************

    def __init__(self, opt_fun, folderAnalysis=None):

        print('\n************************************')
        print("* Initializing PORTALS analyzer...")
        print('************************************')

        self.opt_fun = opt_fun
        self.prep()
    
        self.folder = folderAnalysis if folderAnalysis is not None else f'{self.opt_fun.folder}/Analysis/'
        if not os.path.exists(self.folder): os.system(f'mkdir {self.folder}')

    @classmethod
    def from_folder(cls,folder,folderRemote=None, folderAnalysis=None):

        print(f"\n...opening PORTALS class from folder {IOtools.clipstr(folder)}")

        opt_fun = STRATEGYtools.FUNmain(folder)
        opt_fun.read_optimization_results(analysis_level=4, plotYN=False, folderRemote=folderRemote)

        return cls(opt_fun, folderAnalysis=folderAnalysis)

    # ****************************************************************************
    # PREPARATION
    # ****************************************************************************

    def prep(self):

        print("- Grabbing model")
        self.step =  self.opt_fun.prfs_model.steps[-1]
        self.gp = self.step.GP["combined_model"]

        self.powerstate = self.opt_fun.prfs_model.mainFunction.surrogate_parameters["powerstate"]

        print("- Interpreting PORTALS results")

        # Read dictionaries
        with open(self.opt_fun.prfs_model.mainFunction.MITIMextra, "rb") as f:
            self.mitim_runs = pickle_dill.load(f)

        # What's the last iteration?
        # self.opt_fun.prfs_model.train_Y.shape[0]
        for ikey in self.mitim_runs:
            if not isinstance(self.mitim_runs[ikey],dict):
                break

        # Store indeces
        self.ibest = self.opt_fun.res.best_absolute_index
        self.i0 = 0
        self.ilast = ikey - 1
        
        if self.ilast == self.ibest:
            self.iextra = None
        else:                        
            self.iextra = self.ilast
        
        # Store setup of TGYRO run
        self.rhos = self.mitim_runs[0]['tgyro'].results['tglf_neo'].rho[0,1:]
        self.roa = self.mitim_runs[0]['tgyro'].results['tglf_neo'].roa[0,1:]
        
        self.PORTALSparameters = self.opt_fun.prfs_model.mainFunction.PORTALSparameters
        self.TGYROparameters = self.opt_fun.prfs_model.mainFunction.TGYROparameters
        self.TGLFparameters = self.opt_fun.prfs_model.mainFunction.TGLFparameters

        # Useful flags
        self.ProfilesPredicted = self.TGYROparameters["ProfilesPredicted"]

        self.runWithImpurity = (
            self.PORTALSparameters["ImpurityOfInterest"] -1
            if "nZ" in self.ProfilesPredicted
            else None
        )

        self.runWithRotation = "w0" in self.ProfilesPredicted
        self.includeFast = self.PORTALSparameters["includeFastInQi"]
        self.useConvectiveFluxes = self.PORTALSparameters["useConvectiveFluxes"]

        # Profiles and tgyro results
        print("\t- Reading profiles and tgyros for each evaluation")
       
        if self.mitim_runs is None:
            print('\t\t* Reading from scratch from folders',typeMsg='i')
        
        self.profiles, self.tgyros = [], []
        for i in range(self.ilast+1):
            if self.mitim_runs is not None:
               t = self.mitim_runs[i]["tgyro"].results["use"]
               p = t.profiles_final
            else:
                p = PROFILEStools.PROFILES_GACODE(f"{self.opt_fun.folder}/Execution/Evaluation.{i}/model_complete/input.gacode.new")
                t = TGYROtools.TGYROoutput(f"{self.opt_fun.folder}/Execution/Evaluation.{i}/model_complete/", profiles=p)

            self.tgyros.append(t)
            self.profiles.append(p)

        if len(self.profiles) <= self.ibest:
            print(
                "\t- PORTALS was read after new residual was computed but before pickle was written!",
                typeMsg="w",
            )
            self.ibest -= 1
            self.iextra = None

        self.profiles_next = None
        x_train_num = self.step.train_X.shape[0]
        file = f"{self.opt_fun.folder}/Execution/Evaluation.{x_train_num}/model_complete/input.gacode"
        if os.path.exists(file):
            print("\t\t- Reading next profile to evaluate (from folder)")
            self.profiles_next = PROFILEStools.PROFILES_GACODE(file, calculateDerived=False)

            file = f"{self.opt_fun.folder}/Execution/Evaluation.{x_train_num}/model_complete/input.gacode.new"
            if os.path.exists(file):
                self.profiles_next_new = PROFILEStools.PROFILES_GACODE(file, calculateDerived=False)
                self.profiles_next_new.printInfo(label="NEXT")
            else:
                self.profiles_next_new = self.profiles_next
                self.profiles_next_new.deriveQuantities()
        else:
            print("\t\t- Could not read next profile to evaluate (from folder)")

        # Create some metrics
        prep_metrics(self)

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

        self.plotSummary(fn=fn)

        fig = fn.add_figure(label="PORTALS Metrics")
        self.plotMetrics(fig=fig)

        fig = fn.add_figure(label="PORTALS Expected")
        self.plotExpected(fig=fig)

        if not fnprov:
            fn.show()

    def plotMetrics(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotMetrics(self, **kwargs)
    def plotExpected(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotExpected(self, **kwargs)
    def plotSummary(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotSummary(self, **kwargs)
    def plotRanges(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotRanges(self, **kwargs)
    def plotModelComparison(self, **kwargs):
        PORTALSplot.PORTALSanalyzer_plotModelComparison(self, **kwargs)
    
    # ****************************************************************************
    # ADDITIONAL UTILITIES
    # ****************************************************************************

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

def prep_metrics(self,calculateRicci  = {"d0": 2.0, "l": 1.0}): 
    
    print("\t- Processing metrics")

    self.evaluations, self.resM = [], []
    self.FusionGain, self.tauE, self.FusionPower = [], [], []
    self.resTe, self.resTi, self.resne, self.resnZ, self.resw0 = [], [], [], [], []
    if calculateRicci is not None:
        self.qR_Ricci, self.chiR_Ricci, self.points_Ricci = [], [], []
    else:
        self.qR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None
    
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

        impurityPosition = self.runWithImpurity+1 if self.runWithImpurity is not None else 1

        portals_variables = t.TGYROmodeledVariables(
            useConvectiveFluxes=self.useConvectiveFluxes,
            includeFast=self.includeFast,
            impurityPosition=impurityPosition,
            UseFineGridTargets=self.PORTALSparameters["fineTargetsResolution"],
            OriginalFimp=OriginalFimp,
            forceZeroParticleFlux=self.PORTALSparameters["forceZeroParticleFlux"],
        )

        if (
            len(powerstate.plasma["volp"].shape) > 1
            and powerstate.plasma["volp"].shape[1] > 1
        ):
            powerstate.unrepeat(do_fine=False)
            powerstate.repeat(do_fine=False)

        _, _, source, res = PORTALSinteraction.calculatePseudos(
            portals_variables["var_dict"],
            self.PORTALSparameters,
            self.TGYROparameters,
            powerstate,
        )

        # Make sense of tensor "source" which are defining the entire predictive set in
        Qe_resR = np.zeros(self.rhos.shape[0])
        Qi_resR = np.zeros(self.rhos.shape[0])
        Ge_resR = np.zeros(self.rhos.shape[0])
        GZ_resR = np.zeros(self.rhos.shape[0])
        Mt_resR = np.zeros(self.rhos.shape[0])
        cont = 0
        for prof in self.TGYROparameters[
            "ProfilesPredicted"
        ]:
            for ix in range(
                self.rhos.shape[0]
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
                    self.PORTALSparameters,
                    self.TGYROparameters,
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
    ) / len(self.TGYROparameters["ProfilesPredicted"])

    # ---------------------------------------------------------------------------------------------------------------------
    # Jacobian
    # ---------------------------------------------------------------------------------------------------------------------

    DeltaQ1 = []
    for i in self.TGYROparameters["ProfilesPredicted"]:
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
    # try:	self.aLTn_perc  = calcLinearizedModel(self.opt_fun.prfs_model,self.DeltaQ,numChannels=self.numChannels,numRadius=self.numRadius,sepers=[self.i0, self.ibest])
    # except:	print('\t- Jacobian calculation failed',typeMsg='w')

    self.DVdistMetric_x = self.opt_fun.res.DVdistMetric_x
    self.DVdistMetric_y = self.opt_fun.res.DVdistMetric_y


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

def calcLinearizedModel(
    prfs_model, DeltaQ, posBase=-1, numChannels=3, numRadius=4, sepers=[]
):
    """
    posBase = 1 is aLTi, 0 is aLTe, if the order is [a/LTe,aLTi]
    -1 is diagonal
    -2 is

    NOTE for PRF: THIS ONLY WORKS FOR TURBULENCE, nOT NEO!
    """

    trainx = prfs_model.steps[-1].GP["combined_model"].train_X.cpu().numpy()

    istep, aLTn_est, aLTn_base = 0, [], []
    for i in range(trainx.shape[0]):
        if i >= prfs_model.Optim["initialPoints"]:
            istep += 1

        # Jacobian
        J = (
            prfs_model.steps[istep]
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
