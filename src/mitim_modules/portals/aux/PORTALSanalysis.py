from genericpath import exists
import os
import dill as pickle_dill
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import GRAPHICStools,IOtools
from mitim_tools.gacode_tools import TGLFtools,TGYROtools
from mitim_modules.portals.aux import PORTALSplot

'''
Set of tools to read PORTALS pickle file and do fun stuff with it
'''

class PORTALSanalyzer:
    def __init__(self, folder,folderRemote=None,folderAnalysis=None):

        print(f"> Opening PORTALS class in {IOtools.clipstr(folder)}")

        self.opt_fun = STRATEGYtools.FUNmain(folder)
        self.opt_fun.read_optimization_results(analysis_level=4, plotYN=False, folderRemote=folderRemote)

        with open(self.opt_fun.prfs_model.mainFunction.MITIMextra, "rb") as f:
            self.mitim_runs = pickle_dill.load(f)

        for ikey in self.mitim_runs:
            if type(self.mitim_runs[ikey]) != dict:
                break
        self.ilast = ikey - 1

        self.ibest = self.opt_fun.res.best_absolute_index
        self.rhos = self.mitim_runs[0]['tgyro'].results['tglf_neo'].rho[0,1:]
        self.roa = self.mitim_runs[0]['tgyro'].results['tglf_neo'].roa[0,1:]

        self.folder = folderAnalysis if folderAnalysis is not None else f'{self.opt_fun.folder}/Analysis/'

        if not os.path.exists(self.folder): os.system(f'mkdir {self.folder}')

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

    def plotMetrics(self,indexToMaximize=None,plotAllFluxes=False,index_extra=None,file_save=None):

        # Interpret results
        self.portals_plot = PORTALSplot.PORTALSresults(
             self.opt_fun.folder, # TO FIX: With remote folder this may fail, fix
            self.opt_fun.prfs_model,
            self.opt_fun.res,
            MITIMextra_dict=self.mitim_runs,
            indecesPlot=[self.opt_fun.res.best_absolute_index, 0, index_extra],
        )

        # Plot
        plt.ion()
        fig = plt.figure(figsize=(18, 9))
        PORTALSplot.plotConvergencePORTALS(
            self.portals_plot,
            fig=fig,
            indexToMaximize=indexToMaximize,
            plotAllFluxes=plotAllFluxes,
        )

        if file_save is not None:
            plt.savefig(file_save, transparent=True, dpi=300)

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
                F_cgyro.append(self.mitim_runs[i]['tgyro'].results['cgyro_neo'].__dict__[quantity][0,1:])
                F_tglf_stds.append(self.mitim_runs[i]['tgyro'].results['tglf_neo'].__dict__[quantity_stds][0,1:])
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

        TGLFsettings = self.opt_fun.prfs_model.mainFunction.TGLFparameters["TGLFsettings"]
        extraOptionsTGLF = self.opt_fun.prfs_model.mainFunction.TGLFparameters["extraOptionsTGLF"]
        PredictionSet=[
                    int("te" in self.opt_fun.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]),
                    int("ti" in self.opt_fun.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]),
                    int("ne" in self.opt_fun.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]),
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

        TGLFsettings = self.opt_fun.prfs_model.mainFunction.TGLFparameters["TGLFsettings"]
        extraOptions = self.opt_fun.prfs_model.mainFunction.TGLFparameters["extraOptionsTGLF"]

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
