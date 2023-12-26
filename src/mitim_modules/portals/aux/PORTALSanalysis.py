import os
import dill as pickle_dill
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import GRAPHICStools,IOtools
from mitim_tools.gacode_tools import TGLFtools,TGYROtools,PROFILEStools
from mitim_modules.portals.aux import PORTALSplot

'''
Set of tools to read PORTALS pickle file and do fun stuff with it
'''

class PORTALSanalyzer:
    def __init__(self, opt_fun, folderAnalysis=None):

        print("* Initializing PORTALS analyzer...")

        self.opt_fun = opt_fun
        self.prep(folderAnalysis)
    
    @classmethod
    def from_folder(cls,folder,folderRemote=None, folderAnalysis=None):

        print(f"...opening PORTALS class from folder {IOtools.clipstr(folder)}")

        opt_fun = STRATEGYtools.FUNmain(folder)
        opt_fun.read_optimization_results(analysis_level=4, plotYN=False, folderRemote=folderRemote)

        return cls(opt_fun, folderAnalysis=folderAnalysis)

    def prep(self,folderAnalysis):
        with open(self.opt_fun.prfs_model.mainFunction.MITIMextra, "rb") as f:
            self.mitim_runs = pickle_dill.load(f)

        for ikey in self.mitim_runs:
            if type(self.mitim_runs[ikey]) != dict:
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
        
        self.portals_plot = PORTALSplot.PORTALSresults(
            self.opt_fun.prfs_model,
            self.opt_fun.res,
            MITIMextra_dict=self.mitim_runs,
            indecesPlot=[self.opt_fun.res.best_absolute_index, 0, -1],
        )

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

        print("- Plotting PORTALS Metrics")

        if index_extra is not None:
             self.portals_plot.numExtra = index_extra
        
        if fig is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

        PORTALSplot.plotConvergencePORTALS(
            self.portals_plot,
            fig=fig,
            indexToMaximize=indexToMaximize,
            plotAllFluxes=plotAllFluxes,
        )

        # Save plot
        if file_save is not None:
            plt.savefig(file_save, transparent=True, dpi=300)

    def plotExpected(self,fig=None,stds = 2, max_plot_points=4,plotNext=True):

        print("- Plotting PORTALS Expected")

        if fig is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

        # ----------------------------------------------------------------------
        # Plot
        # ----------------------------------------------------------------------

        trained_points = self.opt_fun.prfs_model.steps[-1].train_X.shape[0]
        indexBest = self.opt_fun.res.best_absolute_index

        # Best point
        plotPoints = [indexBest]
        labelAssigned = [f"#{indexBest} (best)"]

        # Last point
        if (trained_points - 1) != indexBest:
            plotPoints.append(trained_points - 1)
            labelAssigned.append(f"#{trained_points-1} (last)")

        # Last ones
        i = 0
        while len(plotPoints) < max_plot_points:
            if (trained_points - 2 - i) < 1:
                break
            if (trained_points - 2 - i) != indexBest:
                plotPoints.append(trained_points - 2 - i)
                labelAssigned.append(f"#{trained_points-2-i}")
            i += 1

        # First point
        if 0 not in plotPoints:
            if len(plotPoints) == max_plot_points:
                plotPoints[-1] = 0
                labelAssigned[-1] = "#0 (base)"
            else:
                plotPoints.append(0)
                labelAssigned.append("#0 (base)")

        PORTALSplot.plotExpected(
            self.opt_fun.prfs_model,
            self.mitim_runs,
            folder=self.opt_fun.folder,
            fig=fig,
            plotPoints=plotPoints,
            plotNext=plotNext,
            labelAssigned=labelAssigned,
            labelsFluxes=self.portals_plot.labelsFluxes,
            stds=stds,
        )

    def plotSummary(self,fn):

        print("- Plotting PORTALS summary of TGYRO and PROFILES classes")

        # -------------------------------------------------------
        # Plot TGYROs
        # -------------------------------------------------------

        # It may have changed
        indecesPlot = [
            self.portals_plot.numBest,
            self.portals_plot.numOrig,
            self.portals_plot.numExtra,
        ]

        self.portals_plot.tgyros[indecesPlot[1]].plot(
            fn=fn, prelabel=f"({indecesPlot[1]}) TGYRO - "
        )
        if indecesPlot[0] < len(self.portals_plot.tgyros):
            self.portals_plot.tgyros[indecesPlot[0]].plot(
                fn=fn, prelabel=f"({indecesPlot[0]}) TGYRO - "
            )

        # -------------------------------------------------------
        # Plot PROFILES
        # -------------------------------------------------------

        figs = [
            fn.add_figure(label="PROFILES - Profiles"),
            fn.add_figure(label="PROFILES - Powers"),
            fn.add_figure(label="PROFILES - Geometry"),
            fn.add_figure(label="PROFILES - Gradients"),
            fn.add_figure(label="PROFILES - Flows"),
            fn.add_figure(label="PROFILES - Other"),
            fn.add_figure(label="PROFILES - Impurities"),
        ]

        if indecesPlot[0] < len(self.portals_plot.profiles):
            PROFILEStools.plotAll(
                [
                    self.portals_plot.profiles[indecesPlot[1]],
                    self.portals_plot.profiles[indecesPlot[0]],
                ],
                figs=figs,
                extralabs=[f"{indecesPlot[1]}", f"{indecesPlot[0]}"],
            )


        # -------------------------------------------------------
        # Plot Comparison
        # -------------------------------------------------------

        profile_original = self.mitim_runs[0]['tgyro'].results['tglf_neo'].profiles
        profile_best = self.mitim_runs[self.ibest]['tgyro'].results['tglf_neo'].profiles

        profile_original_unCorrected = self.mitim_runs["profiles_original_un"]
        profile_original_0 = self.mitim_runs["profiles_original"]

        fig4 = fn.add_figure(label="PROFILES Comparison")
        grid = plt.GridSpec(
            2,
            np.max(
                [3, len(self.TGYROparameters["ProfilesPredicted"])]
            ),
            hspace=0.3,
            wspace=0.3,
        )
        axs4 = [
            fig4.add_subplot(grid[0, 0]),
            fig4.add_subplot(grid[1, 0]),
            fig4.add_subplot(grid[0, 1]),
            fig4.add_subplot(grid[1, 1]),
            fig4.add_subplot(grid[0, 2]),
            fig4.add_subplot(grid[1, 2]),
        ]

        cont = 1
        if  self.runWithImpurity:
            axs4.append(fig4.add_subplot(grid[0, 2 + cont]))
            axs4.append(fig4.add_subplot(grid[1, 2 + cont]))
            cont += 1
        if  self.runWithRotation:
            axs4.append(fig4.add_subplot(grid[0, 2 + cont]))
            axs4.append(fig4.add_subplot(grid[1, 2 + cont]))

        colors = GRAPHICStools.listColors()

        for i, (profiles, label, alpha) in enumerate(
            zip(
                [
                    profile_original_unCorrected,
                    profile_original_0,
                    profile_original,
                    profile_best,
                ],
                ["Original", "Corrected", "Initial", "Final"],
                [0.2, 1.0, 1.0, 1.0],
            )
        ):
            profiles.plotGradients(
                axs4,
                color=colors[i],
                label=label,
                lastRho=self.TGYROparameters["RhoLocations"][-1],
                alpha=alpha,
                useRoa=True,
                RhoLocationsPlot=self.TGYROparameters["RhoLocations"],
                plotImpurity=self.runWithImpurity,
                plotRotation=self.runWithRotation,
            )

        axs4[0].legend(loc="best")

    def plotRanges(self,fig=None):

        if fig is None:
            plt.ion(); fig = plt.figure()
    
        pps = np.max(
            [3, len(self.TGYROparameters["ProfilesPredicted"])]
        )  # Because plotGradients require at least Te, Ti, ne
        grid = plt.GridSpec(2, pps, hspace=0.3, wspace=0.3)
        axsR = []
        for i in range(pps):
            axsR.append(fig.add_subplot(grid[0, i]))
            axsR.append(fig.add_subplot(grid[1, i]))

        PORTALSplot.produceInfoRanges(
            self.opt_fun.prfs_model.mainFunction,
            self.opt_fun.prfs_model.bounds_orig,
            axsR=axsR,
            color="k",
            lw=0.2,
            alpha=0.05,
            label="original",
        )
        PORTALSplot.produceInfoRanges(
            self.opt_fun.prfs_model.mainFunction,
            self.opt_fun.prfs_model.bounds,
            axsR=axsR,
            color="c",
            lw=0.2,
            alpha=0.05,
            label="final",
        )

        p = self.mitim_runs[0]['tgyro'].results['tglf_neo'].profiles
        p.plotGradients(
            axsR,
            color="b",
            lastRho=self.TGYROparameters["RhoLocations"][-1],
            ms=0,
            lw=1.0,
            label="#0",
            ls="-o" if self.opt_fun.prfs_model.avoidPoints else "--o",
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
        )

        for ikey in self.mitim_runs:
            if type(self.mitim_runs[ikey]) != dict:
                break

            p = self.mitim_runs[ikey]['tgyro'].results['tglf_neo'].profiles
            p.plotGradients(
                axsR,
                color="r",
                lastRho=self.TGYROparameters["RhoLocations"][-1],
                ms=0,
                lw=0.3,
                ls="-o" if self.opt_fun.prfs_model.avoidPoints else "-.o",
                plotImpurity=self.runWithImpurity,
                plotRotation=self.runWithRotation,
            )

        p.plotGradients(
            axsR,
            color="g",
            lastRho=self.TGYROparameters["RhoLocations"][-1],
            ms=0,
            lw=1.0,
            label=f"#{self.opt_fun.res.best_absolute_index} (best)",
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
        )

        axsR[0].legend(loc="best")

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


    def plotModelComparison(self,axs = None,GB=True,radial_label=True):

        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(ncols=3,figsize=(12,6))

        self.plotModelComparison_quantity(axs[0],
                                     quantity=f'Qe{"GB" if GB else ""}_sim_turb',
                                     quantity_stds=f'Qe{"GB" if GB else ""}_sim_turb_stds',
                                     labely = '$Q_e^{GB}$' if GB else '$Q_e$',
                                     title = f"Electron energy flux {'(GB)' if GB else '($MW/m^2$)'}",
                                     typeScale='log' if GB else 'linear',
                                     radial_label = radial_label)

        self.plotModelComparison_quantity(axs[1],
                                     quantity=f'Qi{"GB" if GB else ""}Ions_sim_turb_thr',
                                     quantity_stds=f'Qi{"GB" if GB else ""}Ions_sim_turb_thr_stds',
                                     labely = '$Q_i^{GB}$' if GB else '$Q_i$',
                                     title =f"Ion energy flux {'(GB)' if GB else '($MW/m^2$)'}",
                                     typeScale='log' if GB else 'linear',
                                     radial_label = radial_label)

        self.plotModelComparison_quantity(axs[2],
                                     quantity=f'Ge{"GB" if GB else ""}_sim_turb',
                                     quantity_stds=f'Ge{"GB" if GB else ""}_sim_turb_stds',
                                     labely = '$\\Gamma_e^{GB}$' if GB else '$\\Gamma_e$',
                                     title = f"Electron particle flux {'(GB)' if GB else '($MW/m^2$)'}",
                                     typeScale='linear',
                                     radial_label = radial_label)

        plt.tight_layout()

        return axs

    def plotModelComparison_quantity(self,ax,
                                     quantity='QeGB_sim_turb',
                                     quantity_stds='QeGB_sim_turb_stds',
                                     labely = '',
                                     title = '',
                                     typeScale='linear',
                                     radial_label = True):

        F_tglf = []
        F_cgyro = []
        F_tglf_stds = []
        F_cgyro_stds = []
        for i in range(len(self.mitim_runs)-2):
            try:
                F_tglf.append(self.mitim_runs[i]['tgyro'].results['tglf_neo'].__dict__[quantity][0,1:])
                F_tglf_stds.append(self.mitim_runs[i]['tgyro'].results['tglf_neo'].__dict__[quantity_stds][0,1:])
                F_cgyro.append(self.mitim_runs[i]['tgyro'].results['cgyro_neo'].__dict__[quantity][0,1:])
                F_cgyro_stds.append(self.mitim_runs[i]['tgyro'].results['cgyro_neo'].__dict__[quantity_stds][0,1:])
            except TypeError:
                break   
        F_tglf  = np.array(F_tglf)
        F_cgyro = np.array(F_cgyro)
        F_tglf_stds  = np.array(F_tglf_stds)
        F_cgyro_stds = np.array(F_cgyro_stds)

        colors = GRAPHICStools.listColors()

        for ir in range(F_tglf.shape[1]):
            ax.errorbar(
                F_tglf[:,ir],
                F_cgyro[:,ir],
                yerr=F_cgyro_stds[:,ir],
                c=colors[ir],
                markersize=2,
                capsize=2,
                fmt="s",
                elinewidth=1.0,
                capthick=1.0,
                label=f"$r/a={self.roa[ir]:.2f}$" if radial_label else ""
            )

        minFlux = np.min([F_tglf.min(),F_cgyro.min()])
        maxFlux = np.max([F_tglf.max(),F_cgyro.max()])

        if radial_label: ax.plot([minFlux,maxFlux],[minFlux,maxFlux],'--',color='k')
        if typeScale == 'log':
            ax.set_xscale('log')
            ax.set_yscale('log')
        elif typeScale == 'symlog':
            ax.set_xscale('symlog')#,linthresh=1E-2)
            ax.set_yscale('symlog')#,linthresh=1E-2)
        ax.set_xlabel(f'{labely} (TGLF)')
        ax.set_ylabel(f'{labely} (CGYRO)')
        ax.set_title(title)
        GRAPHICStools.addDenseAxis(ax)

        ax.legend()

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

        embed()

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

