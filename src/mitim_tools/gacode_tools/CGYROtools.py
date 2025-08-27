import shutil
import datetime
import time
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools.utils import GACODEdefaults, GACODErun, CGYROutils
from mitim_tools.misc_tools import IOtools, GRAPHICStools, FARMINGtools, CONFIGread
from mitim_tools.gacode_tools.utils import GACODEplotting
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class CGYRO(GACODErun.gacode_simulation):
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        def code_call(folder, p, n = 1, nomp = 1, additional_command="", **kwargs):
            return f"cgyro -e {folder} -n {n} -nomp {nomp} {additional_command} -p {p}"

        def code_slurm_settings(name, minutes, total_cores_required, cores_per_code_call, type_of_submission, array_list=None):

            slurm_settings = {
                "name": name,
                "minutes": minutes,
            }

            # Gather if this is a GPU enabled machine
            machineSettings = CONFIGread.machineSettings(code='cgyro')

            if type_of_submission == "slurm_standard":
                
                slurm_settings['ntasks'] = total_cores_required
                
                if machineSettings['gpus_per_node'] > 0:
                    slurm_settings['gpuspertask'] = cores_per_code_call
                else:
                    slurm_settings['cpuspertask'] = cores_per_code_call

            elif type_of_submission == "slurm_array":

                slurm_settings['ntasks'] = 1
                if machineSettings['gpus_per_node'] > 0:
                    slurm_settings['gpuspertask'] = cores_per_code_call
                else:
                    slurm_settings['cpuspertask'] = cores_per_code_call
                slurm_settings['job_array'] = ",".join(array_list)

            return slurm_settings

        self.run_specifications = {
            'code': 'cgyro',
            'input_file': 'input.cgyro',
            'code_call': code_call,
            'code_slurm_settings': code_slurm_settings,
            'control_function': GACODEdefaults.addCGYROcontrol,
            'controls_file': 'input.cgyro.controls',
            'state_converter': 'to_cgyro',
            'input_class': CGYROinput,
            'complete_variation': None,
            'default_cores': 16,  # Default cores to use in the simulation
            'output_class': CGYROutils.CGYROout,
            'output_store': 'CGYROout'
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t CGYRO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles = self.ResultsFiles_minimal = [
            "bin.cgyro.geo",
            "bin.cgyro.kxky_e",
            "bin.cgyro.kxky_n",
            "bin.cgyro.kxky_phi",
            "bin.cgyro.kxky_apar", 
            "bin.cgyro.kxky_bpar", 
            "bin.cgyro.kxky_v",
            "bin.cgyro.ky_cflux",
            "bin.cgyro.ky_flux",
            "bin.cgyro.phib",
            "bin.cgyro.aparb",
            "bin.cgyro.bparb",
            "bin.cgyro.restart",
            "input.cgyro",
            "input.cgyro.gen",
            "mitim.out",
            "out.cgyro.egrid",
            "out.cgyro.equilibrium",
            "out.cgyro.freq",
            "out.cgyro.grids",
            "out.cgyro.hosts",
            "out.cgyro.info",
            "out.cgyro.memory",
            "out.cgyro.mpi",
            "out.cgyro.prec",
            "out.cgyro.rotation",
            "out.cgyro.startups",
            "out.cgyro.tag",
            "out.cgyro.time",
            "out.cgyro.timing",
            "out.cgyro.version",
        ]

        self.output_files_test = [
            "out.cgyro.equilibrium",
            "out.cgyro.info",
            "out.cgyro.mpi",
            "input.cgyro.gen",
            "out.cgyro.egrid",
            "out.cgyro.grids",
            "out.cgyro.memory",
            "out.cgyro.rotation",
        ]

    # Re-defined to make specific arguments explicit
    def read(
        self,
        tmin = 0.0, 
        minimal = False, 
        last_tmin_for_linear = True,
        **kwargs
    ):
    
        super().read(
            tmin = tmin,
            minimal = minimal,
            last_tmin_for_linear = last_tmin_for_linear,
            **kwargs)

    def _labelize(self, labels):

        # If it has radii, we need to correct the labels
        self.results_all = copy.deepcopy(self.results)
        self.results = {}
        labels_with_rho = []
        for label in labels:
            for i,rho in enumerate(self.rhos):
                labels_with_rho.append(f"{label}_{rho}")
                self.results[f'{label}_{rho}'] = self.results_all[label]['CGYROout'][i]
        labels = labels_with_rho
        # ------------------------------------------------
        
        # If it has scans, we need to correct the labels
        labels_corrected = []
        for i in range(len(labels)):
            if isinstance(self.results[labels[i]], CGYROutils.CGYROlinear_scan):    
                for scan_label in self.results[labels[i]].labels:
                    labels_corrected.append(scan_label)
            else:
                labels_corrected.append(labels[i])
        labels = labels_corrected
        # ------------------------------------------------
        
        return labels

    def plot(
        self,
        labels=[""],
        fn=None,
        include_2D=True,
        common_colorbar=True):
        
        labels = self._labelize(labels)
    
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook
            self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")
        else:
            self.fn = fn

        fig = self.fn.add_figure(label="Fluxes (time)")
        axsFluxes_t = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Fluxes (ky)")
        axsFluxes_ky = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Intensities (time)")
        axsIntensities = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (ky)")
        axsIntensities_ky = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (kx)")
        axsIntensities_kx = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Cross-phases (ky)")
        axsCrossPhases = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Turbulence (linear)")
        axsTurbulence = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
      
        create_ballooning = False
        for label in labels:
            if 'phi_ballooning' in self.results[label].__dict__:
                create_ballooning = True
            
        if create_ballooning:

            fig = self.fn.add_figure(label="Ballooning")
            axsBallooning = fig.subplot_mosaic(
                """
                135
                246
                """
                )
        else:
            axsBallooning = None
        
        
        if include_2D:
            axs2D = []
            for i in range(len(labels)):
                fig = self.fn.add_figure(label="Turbulence (2D), " + labels[i])
                
                mosaic = _2D_mosaic(4) # Plot 4 times by default
                
                axs2D.append(fig.subplot_mosaic(mosaic))
        
        fig = self.fn.add_figure(label="Inputs")
        axsInputs = fig.subplot_mosaic(
            """
            A
            B
            """
        )

        
        colors = GRAPHICStools.listColors()

        colorbars_all = []  # Store all colorbars for later use
        for j in range(len(labels)):
            
            self.plot_fluxes(
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            self.plot_fluxes_ky(
                axs=axsFluxes_ky,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            self.plot_intensities_ky(
                axs=axsIntensities_ky,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,
            )
            self.plot_intensities(
                axs=axsIntensities,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            self.plot_intensities_kx(
                axs=axsIntensities_kx,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            self.plot_turbulence(
                axs=axsTurbulence,
                label=labels[j],
                c=colors[j],
            )
            self.plot_cross_phases(
                axs=axsCrossPhases,
                label=labels[j],
                c=colors[j],
            )
            if create_ballooning:
                self.plot_ballooning(
                    axs=axsBallooning,
                    label=labels[j],
                    c=colors[j],
                )
            
            if include_2D:
                
                colorbars = self.plot_2D(
                    axs=axs2D[j],
                    label=labels[j],
                )
                
                colorbars_all.append(colorbars)
            
            self.plot_inputs(
                ax=axsInputs["A"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
                normalization_label= labels[0],  # Normalize to the first label
                only_plot_differences=len(labels) > 1,  # Only plot differences if there are multiple labels
            )
            
            self.plot_inputs(
                ax=axsInputs["B"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
            )
            
        axsInputs["A"].axhline(
            1.0,
            color="k",
            ls="--",
            lw=2.0
        )
        
        GRAPHICStools.adjust_subplots(axs=axsInputs, vertical=0.4, horizontal=0.3)
        
        # Modify the colorbars to have a common range
        if include_2D and common_colorbar and len(colorbars_all) > 0:
            for var in ['phi', 'n', 'e']:
                min_val = np.inf
                max_val = -np.inf
                for ilabel in range(len(colorbars_all)):
                    cb = colorbars_all[ilabel][0][var]
                    vals = cb.mappable.get_clim()
                    min_val = min(min_val, vals[0])
                    max_val = max(max_val, vals[1])
                
                for ilabel in range(len(colorbars_all)):
                    for it in range(len(colorbars_all[ilabel])):
                        cb = colorbars_all[ilabel][it][var]
                        cb.mappable.set_clim(min_val, max_val)
                        cb.update_ticks()
                        #cb.set_label(f"{var} (common range)")

        self.results = self.results_all

    def _plot_trace(self, ax, label, variable, c="b", lw=1, ls="-", label_plot='', meanstd=True, var_meanstd= None):
        
        t = self.results[label].t
        
        if not isinstance(variable, str):
            z = variable
            if var_meanstd is not None:
                z_mean = var_meanstd[0]
                z_std = var_meanstd[1]
            
        else:
            z = self.results[label].__dict__[variable]
            if meanstd and (f'{variable}_mean' in self.results[label].__dict__):
                z_mean = self.results[label].__dict__[variable + '_mean']
                z_std = self.results[label].__dict__[variable + '_std']
            else:
                z_mean = None
                z_std = None
        
        ax.plot(
            t,
            z,
            ls=ls,
            lw=lw,
            c=c,
            label=label_plot,
        )
        
        if meanstd and z_std>0.0:
            GRAPHICStools.fillGraph(
                ax,
                t[t>self.results[label].tmin],
                z_mean,
                y_down=z_mean
                - z_std,
                y_up=z_mean
                + z_std,
                alpha=0.1,
                color=c,
                lw=0.5,
                islwOnlyMean=True,
                label=label_plot + f" $\\mathbf{{{z_mean:.2f} \\pm {z_std:.2f}}}$ (1$\\sigma$)",
            )

    def plot_inputs(self, ax = None, label="", c="b", ms = 10, normalization_label=None, only_plot_differences=False):
        
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(18, 9))

        rel_tol = 1e-2

        legadded = False
        for i, ikey in enumerate(self.results[label].params1D):
            
            z = self.results[label].params1D[ikey]
            
            if normalization_label is not None:
                z0 = self.results[normalization_label].params1D[ikey]
                zp = z/z0 if z0 != 0 else 0
                label_plot = f"{label} / {normalization_label}"
            else:
                label_plot = label
                zp = z

            if (not only_plot_differences) or (not np.isclose(z, z0, rtol=rel_tol)):
                ax.plot(ikey,zp,'o',markersize=ms,color=c,label=label_plot if not legadded else '')
                legadded = True

        if normalization_label is not None:
            if only_plot_differences:
                ylabel = f"Parameters (DIFFERENT by {rel_tol*100:.2f}%) relative to {normalization_label}"
            else:
                ylabel = f"Parameters relative to {normalization_label}"
        else:
            ylabel = "Parameters"

        ax.set_xlabel("Parameter")
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel(ylabel)
        GRAPHICStools.addDenseAxis(ax)
        if legadded:
            ax.legend(loc='best')

    def plot_fluxes(self, axs=None, label="", c="b", lw=1, plotLegend=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
				AB
                CD
				"""
            )

        ls = GRAPHICStools.listLS()

        # Electron energy flux
        ax = axs["A"]
        self._plot_trace(ax,label,"Qe",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
        self._plot_trace(ax,label,"Qe_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
        
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron energy flux')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Electron particle flux
        ax = axs["B"]
        self._plot_trace(ax,label,"Ge",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
        self._plot_trace(ax,label,"Ge_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
        
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle flux')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Ion energy fluxes
        ax = axs["C"]
        self._plot_trace(ax,label,"Qi",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
        self._plot_trace(ax,label,"Qi_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
        
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Ion species energy fluxes
        ax = axs["D"]
        for j, i in enumerate(self.results[label].ions_flags):
            self._plot_trace(ax,label,self.results[label].Qi_all[j],c=c,lw=lw,ls=ls[j],label_plot=f"{label}, {self.results[label].all_names[i]}", meanstd=False)
            
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes (separate species)')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_fluxes_ky(self, axs=None, label="", c="b", lw=1, plotLegend=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )
            
        ls = GRAPHICStools.listLS()

        # Electron energy flux
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].Qe_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Qe_ky_mean-self.results[label].Qe_ky_std, self.results[label].Qe_ky_mean+self.results[label].Qe_ky_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$Q_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron energy flux vs. $k_\\theta\\rho_s$')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Electron particle flux
        ax = axs["B"]
        ax.plot(self.results[label].ky, self.results[label].Ge_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Ge_ky_mean-self.results[label].Ge_ky_std, self.results[label].Ge_ky_mean+self.results[label].Ge_ky_std, color=c, alpha=0.2)
    
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle flux vs. $k_\\theta\\rho_s$')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Ion energy flux
        ax = axs["C"]
        ax.plot(self.results[label].ky, self.results[label].Qi_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Qi_ky_mean-self.results[label].Qi_ky_std, self.results[label].Qi_ky_mean+self.results[label].Qi_ky_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes vs. $k_\\theta\\rho_s$')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

        # Ion species energy fluxes
        ax = axs["D"]
        for j, i in enumerate(self.results[label].ions_flags):
            ax.plot(self.results[label].ky, self.results[label].Qi_all_ky_mean[j],ls[j]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[i]}")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes vs. $k_\\theta\\rho_s$(separate species)')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities(self, axs = None, label= "cgyro1", c="b", addText=True):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
            
        ax = axs["A"]
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta \\phi/\\phi_0$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].t, self.results[label].apar_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}, $A_\\parallel$")
            ax.plot(self.results[label].t, self.results[label].bpar_rms_sumnr_sumn*100.0, '--', c=c, lw=2, label=f"{label}, $B_\\parallel$")
            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity fluctuations')
        

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["C"]
        ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_e/n_{e,0}/n_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Density intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["D"]
        ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_e/T_{e,0}/T_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Temperature intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))




        ax = axs["E"]
        ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Density intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["F"]
        ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/T_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Temperature intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        ax = axs["G"]
        for ion in self.results[label].ions_flags:
            ax.plot(self.results[label].t, self.results[label].ni_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Density intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)


        ax = axs["H"]
        for ion in self.results[label].ions_flags:
            ax.plot(self.results[label].t, self.results[label].Ti_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Temperature intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)


        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_ky(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].phi_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].phi_rms_sumnr_mean-self.results[label].phi_rms_sumnr_std, self.results[label].phi_rms_sumnr_mean+self.results[label].phi_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta\phi/\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].ky, self.results[label].apar_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].apar_rms_sumnr_mean-self.results[label].apar_rms_sumnr_std, self.results[label].apar_rms_sumnr_mean+self.results[label].apar_rms_sumnr_std, color=c, alpha=0.2)
            ax.plot(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean, '--', markersize=5, color=c, label=label+', $B_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean-self.results[label].bpar_rms_sumnr_std, self.results[label].bpar_rms_sumnr_mean+self.results[label].bpar_rms_sumnr_std, color=c, alpha=0.2)

            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta F_\parallel/F_{\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs. $k_\\theta\\rho_s$')
        
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["C"]
        ax.plot(self.results[label].ky, self.results[label].ne_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].ne_rms_sumnr_mean-self.results[label].ne_rms_sumnr_std, self.results[label].ne_rms_sumnr_mean+self.results[label].ne_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Electron temperature intensity
        ax = axs["D"]
        ax.plot(self.results[label].ky, self.results[label].Te_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Te_rms_sumnr_mean-self.results[label].Te_rms_sumnr_std, self.results[label].Te_rms_sumnr_mean+self.results[label].Te_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["E"]
        ax.plot(self.results[label].ky, self.results[label].ni_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].ni_rms_sumnr_mean-self.results[label].ni_rms_sumnr_std, self.results[label].ni_rms_sumnr_mean+self.results[label].ni_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion particle intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Ion temperature intensity
        ax = axs["F"]
        ax.plot(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean-self.results[label].Ti_rms_sumnr_std, self.results[label].Ti_rms_sumnr_mean+self.results[label].Ti_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion temperature intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["G"]
        for ion in self.results[label].ions_flags:
            ax.plot(self.results[label].ky, self.results[label].ni_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")


        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) particle intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        
        # Ion temperature intensity
        ax = axs["H"]
        for ion in self.results[label].ions_flags:
            ax.plot(self.results[label].ky, self.results[label].Ti_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")


        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) temperature intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_kx(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta \\phi/\\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs kx')
        ax.legend(loc='best', prop={'size': 8},)
        ax.set_yscale('log')
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["C"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].kx, self.results[label].apar_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].bpar_rms_sumn_mean, '--', markersize=1.0, lw=1.0, color=c, label=label+', $B_\\parallel$ (mean)')

            ax.legend(loc='best', prop={'size': 8},)


        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs kx')
        ax.set_yscale('log')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["B"]
        ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
        ax.plot(self.results[label].kx, self.results[label].ne_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
        ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs kx')
        ax.legend(loc='best', prop={'size': 8},)
        ax.set_yscale('log')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        # Electron temperature intensity
        ax = axs["D"]
        ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
        ax.plot(self.results[label].kx, self.results[label].Te_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
        ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs kx')
        ax.legend(loc='best', prop={'size': 8},)
        ax.set_yscale('log')
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_turbulence(self, axs = None, label= "cgyro1", c="b", kys = None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Is no kys provided, select just 3: first, last and middle
        if kys is None:
            ikys = [0]
            if len(self.results[label].ky) > 1:
                ikys.append(-1)
            if len(self.results[label].ky) > 2:
                ikys.append(len(self.results[label].ky) // 2)
                
            ikys = np.unique(ikys)            
        else:
            ikys = [self.results[label].ky.index(ky) for ky in kys if ky in self.results[label].ky]    

        # Growth rate as function of time
        ax = axs["A"]
        for i,ky in enumerate(ikys):
            self._plot_trace(
                ax,
                label,
                self.results[label].g[ky, :],
                c=c,
                ls = GRAPHICStools.listLS()[i],
                lw=1,
                label_plot=f"$k_{{\\theta}}\\rho_s={np.abs(self.results[label].ky[ky]):.2f}$",
                var_meanstd = [self.results[label].g_mean[ky], self.results[label].g_std[ky]],
            )
            
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\gamma$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Growth rate vs time')
        ax.legend(loc='best', prop={'size': 8},)

        # Frequency as function of time
        ax = axs["B"]
        for i,ky in enumerate(ikys):
            self._plot_trace(
                ax,
                label,
                self.results[label].f[ky, :],
                c=c,
                ls = GRAPHICStools.listLS()[i],
                lw=1,
                label_plot=f"$k_{{\\theta}}\\rho_s={np.abs(self.results[label].ky[ky]):.2f}$",
                var_meanstd = [self.results[label].f_mean[ky], self.results[label].f_std[ky]],
            )
            
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\omega$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Real Frequency vs time')
        ax.legend(loc='best', prop={'size': 8},)

        # Mean+Std Growth rate as function of ky
        ax = axs["C"]
        ax.errorbar(self.results[label].ky, self.results[label].g_mean, yerr=self.results[label].g_std, fmt='-o', markersize=5, color=c, label=label+' (mean+std)')
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\gamma$ (norm.)")
        ax.set_title('Saturated Growth Rate')
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best', prop={'size': 8},)
        
        # Mean+Std Frequency as function of ky
        ax = axs["D"]
        ax.errorbar(self.results[label].ky, self.results[label].f_mean, yerr=self.results[label].f_std, fmt='-o', markersize=5, color=c, label=label+' (mean+std)')
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\omega$ (norm.)")
        ax.set_title('Saturated Real Frequency')
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best', prop={'size': 8},)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_cross_phases(self, axs = None, label= "cgyro1", c="b"):

        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
        m = GRAPHICStools.listmarkers()
            
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].neTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
        ax.fill_between(self.results[label].ky, self.results[label].neTe_kx0_mean-self.results[label].neTe_kx0_std, self.results[label].neTe_kx0_mean+self.results[label].neTe_kx0_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_e-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_e-T_e$ cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)


        ax = axs["B"]
        ax.plot(self.results[label].ky, self.results[label].niTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
        ax.fill_between(self.results[label].ky, self.results[label].niTi_kx0_mean-self.results[label].niTi_kx0_std, self.results[label].niTi_kx0_mean+self.results[label].niTi_kx0_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_i-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_i-T_i$ cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)

        ax = axs["C"]
        ax.plot(self.results[label].ky, self.results[label].phine_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
        ax.fill_between(self.results[label].ky, self.results[label].phine_kx0_mean-self.results[label].phine_kx0_std, self.results[label].phine_kx0_mean+self.results[label].phine_kx0_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_e$ cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)

        ax = axs["D"]
        ax.plot(self.results[label].ky, self.results[label].phini_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
        ax.fill_between(self.results[label].ky, self.results[label].phini_kx0_mean-self.results[label].phini_kx0_std, self.results[label].phini_kx0_mean+self.results[label].phini_kx0_std, color=c, alpha=0.2)
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)


        ax = axs["E"]
        ax.plot(self.results[label].ky, self.results[label].phiTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
        ax.fill_between(self.results[label].ky, self.results[label].phiTe_kx0_mean-self.results[label].phiTe_kx0_std, self.results[label].phiTe_kx0_mean+self.results[label].phiTe_kx0_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_e$ cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)
        

        ax = axs["F"]
        ax.plot(self.results[label].ky, self.results[label].phiTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
        ax.fill_between(self.results[label].ky, self.results[label].phiTi_kx0_mean-self.results[label].phiTi_kx0_std, self.results[label].phiTi_kx0_mean+self.results[label].phiTi_kx0_std, color=c, alpha=0.2)
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)
        
        
        ax = axs["G"]
        for ion in self.results[label].ions_flags:
            ax.plot(self.results[label].ky, self.results[label].phiTi_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ (all) cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)
        

        ax = axs["H"]
        for ion in self.results[label].ions_flags:
            ax.plot(self.results[label].ky, self.results[label].phini_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
        
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ (all) cross-phase ($k_x=0$)')
        ax.legend(loc='best', prop={'size': 8},)
        
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_ballooning(self, time = None, label="cgyro1", c="b", axs=None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                135
                246
                """
            )

        if time is None:
            time = np.min([self.results[label].tmin, self.results[label].tmax_fluct])
        
        it = np.argmin(np.abs(self.results[label].t - time))

        colorsC, _ = GRAPHICStools.colorTableFade(
            len(self.results[label].ky),
            startcolor=c,
            endcolor=c,
            alphalims=[1.0, 0.4],
        )

        ax = axs['1']
        for ky in range(len(self.results[label].ky)):
            for var, axsT in zip(
                ["phi_ballooning", "apar_ballooning", "bpar_ballooning"],
                [[axs['1'], axs['2']], [axs['3'], axs['4']], [axs['5'], axs['6']]],
            ):

                f = self.results[label].__dict__[var][:, it]
                y1 = np.real(f)
                y2 = np.imag(f)
                x = self.results[label].theta_ballooning / np.pi

                # Normalize
                y1_max = np.max(np.abs(y1))
                y2_max = np.max(np.abs(y2))
                y1 /= y1_max
                y2 /= y2_max

                ax = axsT[0]
                ax.plot(
                    x,
                    y1,
                    color=colorsC[ky],
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y1_max:.2e})",
                )
                ax = axsT[1]
                ax.plot(
                    x, 
                    y2, 
                    color=colorsC[ky], 
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y2_max:.2e})",
                )


        ax = axs['1']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta\\phi$)")
        ax.set_title("$\\delta\\phi$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([-2 * np.pi, 2 * np.pi])

        ax = axs['3']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta A\\parallel$)")
        ax.set_title("$\\delta A\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['5']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta B\\parallel$)")
        ax.set_title("$\\delta B\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['2']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta\\phi$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['4']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta A\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['6']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta B\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)


        for ax in [axs['1'], axs['3'], axs['5'], axs['2'], axs['4'], axs['6']]:
            ax.axvline(x=0, lw=0.5, ls="--", c="k")
            ax.axhline(y=0, lw=0.5, ls="--", c="k")
            
            
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_2D(self, label="cgyro1", axs=None, times = None):
    
        if times is None:
            times = []
            
            number_times = len(axs)//3 if axs is not None else 4

            try:
                times = [self.results[label].t[-1-i*10] for i in range(number_times)]
            except IndexError:
                 times = [self.results[label].t[-1-i*1] for i in range(number_times)]

        if axs is None:

            mosaic = _2D_mosaic(len(times))

            plt.ion()
            fig = plt.figure(figsize=(18, 9))
            axs = fig.subplot_mosaic(mosaic)

        # Pre-calculate global min/max for each field type across all times
        phi_values = []
        n_values = []
        e_values = []
        
        for time in times:
            it = np.argmin(np.abs(self.results[label].t - time))
            
            # Get phi values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)
            phi_values.append(fp)
            
            # Get n values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)
            n_values.append(fp)
            
            # Get e values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)
            e_values.append(fp)
        
        # Calculate global ranges
        phi_max = np.max([np.max(np.abs(fp)) for fp in phi_values])
        phi_min, phi_max = -phi_max, +phi_max
        
        n_max = np.max([np.max(np.abs(fp)) for fp in n_values])
        n_min, n_max = -n_max, +n_max
        
        e_max = np.max([np.max(np.abs(fp)) for fp in e_values])
        e_min, e_max = -e_max, +e_max

        colorbars = []  # Store colorbar references
        # Now plot with consistent colorbar ranges
        for time_i, time in enumerate(times):
            
            print(f"\t- Plotting 2D turbulence for {label} at time {time}")
            
            it = np.argmin(np.abs(self.results[label].t - time))
            
            cfig = axs[str(time_i+1)].get_figure()
            
            # Phi plot
            ax = axs[str(time_i+1)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)

            cs1 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(phi_min,phi_max,(phi_max-phi_min)/256),cmap=plt.get_cmap('jet'))
            cphi = cfig.colorbar(cs1, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta\\phi/\\phi_0$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # N plot
            ax = axs[str(time_i+1+len(times))]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)

            cs2 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(n_min,n_max,(n_max-n_min)/256),cmap=plt.get_cmap('jet'))
            cn = cfig.colorbar(cs2, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta n_e/n_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # E plot
            ax = axs[str(time_i+1+len(times)*2)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)

            cs3 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(e_min,e_max,(e_max-e_min)/256),cmap=plt.get_cmap('jet'))
            ce = cfig.colorbar(cs3, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta E_e/E_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            # Store the colorbar objects with their associated contour plots
            colorbars.append({
                'phi': cphi,
                'n': cn,
                'e': ce
            })

        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.4, horizontal=0.3)

        return colorbars
        
    def _to_real_space(self, variable = 'kxky_phi', species = None, label="cgyro1", theta_plot = 0, it = -1):
        
        # from pygacode
        def maptoreal_fft(nr,nn,nx,ny,c):

            d = np.zeros([nx,nn],dtype=complex)
            for i in range(nr):
                p = i-nr//2
                if -p < 0:
                    k = -p+nx
                else:
                    k = -p
                d[k,0:nn] = np.conj(c[i,0:nn])
            f = np.fft.irfft2(d,s=[nx,ny],norm='forward')*0.5

            # Correct for half-sum
            f = 2*f

            return f

        # Real space
        nr = self.results[label].cgyrodata.n_radial
        nn = self.results[label].cgyrodata.n_n
        craw = self.results[label].cgyrodata.__dict__[variable]
        
        itheta = np.argmin(np.abs(self.results[label].theta_stored-theta_plot))
        if species is None:
            c = craw[:,itheta,:,it]
        else:
            c = craw[:,itheta,species,:,it]

        nx = self.results[label].cgyrodata.__dict__[variable].shape[0]
        ny = nx
        
        # Arrays
        x = np.arange(nx)*2*np.pi/nx
        y = np.arange(ny)*2*np.pi/ny
        f = maptoreal_fft(nr,nn,nx,ny,c)
        
        # Physical maxima
        ky1 = self.results[label].cgyrodata.ky[1] if len(self.results[label].cgyrodata.ky) > 1 else self.results[label].cgyrodata.ky[0]
        xmax = self.results[label].cgyrodata.length
        ymax = (2*np.pi)/np.abs(ky1)
        xp = x/(2*np.pi)*xmax
        yp = y/(2*np.pi)*ymax

        # Periodic extensions
        xp = np.append(xp,xmax)
        yp = np.append(yp,ymax)
        fp = np.zeros([nx+1,ny+1])
        fp[0:nx,0:ny] = f[:,:]
        fp[-1,:] = fp[0,:]
        fp[:,-1] = fp[:,0]
        
        return xp, yp, fp
        
    def plot_quick_linear(self, labels=["cgyro1"], fig=None):
 
        colors = GRAPHICStools.listColors()

        if fig is None:
            fig = plt.figure(figsize=(15,9))

        axs = fig.subplot_mosaic(
            """
            12
            34
            """
        )
            
        def _plot_linear_stability(axs, labels, label_base,col_lin ='b', start_cont=0):

            for cont, label in enumerate(labels):
                c = self.results[label]
                baseColor = colors[cont+start_cont+1]
                colorsC, _ = GRAPHICStools.colorTableFade(
                    len(c.ky),
                    startcolor=baseColor,
                    endcolor=baseColor,
                    alphalims=[1.0, 0.4],
                )

                ax = axs['1']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.g[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                    )

                ax = axs['2']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.f[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                    )

            GACODEplotting.plotTGLFspectrum(
                [axs['3'], axs['4']],
                self.results[label_base].ky,
                self.results[label_base].g_mean,
                freq=self.results[label_base].f_mean,
                coeff=0.0,
                c=col_lin,
                ls="-",
                lw=1,
                label="",
                facecolors=colors,
                markersize=50,
                alpha=1.0,
                titles=["Growth Rate", "Real Frequency"],
                removeLow=1e-4,
                ylabel=True,
            )
            
            return cont
       

        
        labels = self._labelize(labels)
        
        # Make it linear object for nice plotting
        labels = self._kyfy(labels)
        
        co = -1
        for i,label0 in enumerate(labels):
            if isinstance(self.results[label0], CGYROutils.CGYROlinear_scan):
                co = _plot_linear_stability(axs, self.results[label0].labels, label0, start_cont=co, col_lin=colors[i])
            else:
                co = _plot_linear_stability(axs, [label0], label0, start_cont=co, col_lin=colors[i])

        ax = axs['1']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$\\gamma$ $(c_s/a)$")
        ax.set_title("Growth Rate")
        ax.set_xlim(left=0)
        ax.legend()
        ax = axs['2']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.set_ylabel("$\\omega$ $(c_s/a)$")
        ax.set_title("Real Frequency")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_xlim(left=0)

    def _kyfy(self,labels_original):
        '''
        This function transforms the original labels into the linear scan
        e.g. from labels:
            ['scan1_KY_0.3_0.5',
            'scan1_KY_0.3_0.7',
            'scan1_KY_0.4_0.5',
            'scan1_KY_0.4_0.7']
        to:
            ['scan1_0.5',
            'scan1_0.7']
        where these are the CGYROlinear_scan object
        '''
    
        labelsD = {}
        for label in labels_original:
            parts = label.split('_')
            if len(parts) >= 4 and parts[1] == "KY":
                # Extract the base name (scan1), middle value (0.3/0.4), and last value (0.5/0.7)
                base_name = parts[0]
                middle_value = float(parts[2])
                last_value = parts[3]
                
                # Create the new key format: base_name + "_" + last_value
                new_key = f"{base_name}_{last_value}"
                
                # Add the middle value to the list for this key
                if new_key not in labelsD:
                    labelsD[new_key] = []
                labelsD[new_key].append(label)
        
        labels = []
        for label in labelsD:
            self.results[label] = CGYROutils.CGYROlinear_scan(labelsD[label], self.results
            )
            labels.append(label)

        return labels


    # def prep(self, folder, inputgacode_file):

    #     # Prepare main folder with input.gacode
    #     self.folder = IOtools.expandPath(folder)

    #     self.folder.mkdir(parents=True, exist_ok=True)

    #     self.inputgacode_file = self.folder / "input.gacode"
    #     if IOtools.expandPath(inputgacode_file) != self.inputgacode_file:
    #         shutil.copy2(IOtools.expandPath(inputgacode_file), self.inputgacode_file)

    def _prerun(
        self,
        subfolder,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
    ):

        self.folder = self.FolderGACODE / Path(subfolder)

        self.folder.mkdir(parents=True, exist_ok=True)

        input_cgyro_file = self.folder / "input.cgyro"
        inputCGYRO = CGYROinput(file=input_cgyro_file)

        inputgacode_file_this = self.folder / "input.gacode"
        self.profiles.write_state(inputgacode_file_this)

        ResultsFiles_new = []
        for i in self.ResultsFiles:
            if "mitim.out" not in i:
                ResultsFiles_new.append(i)
        self.ResultsFiles = ResultsFiles_new

        inputCGYRO = GACODErun.modifyInputs(
            inputCGYRO,
            code_settings=CGYROsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            addControlFunction=GACODEdefaults.addCGYROcontrol,
            rmin=roa,
            controls_file = 'input.cgyro.controls'
        )

        inputCGYRO.write_state()

        return input_cgyro_file, inputgacode_file_this


    def run_test(
        self,
        subfolder,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
        **kwargs
    ):
        
        if 'scan_param' in kwargs:
            print("\t- Cannot run CGYRO tests with scan_param, running just the base",typeMsg="i")
        
        input_cgyro_file, inputgacode_file_this = self._prerun(
            subfolder,
            roa=roa,
            CGYROsettings=CGYROsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
        )

        self.cgyro_job = FARMINGtools.mitim_job(self.folder)

        name = f'mitim_cgyro_{subfolder}_{roa:.6f}_test'

        self.cgyro_job.define_machine(
            "cgyro",
            name,
            slurm_settings={
                "name": name,
                "minutes": 5,
                "cpuspertask": 1,
                "ntasks": 1,
            },
        )

        CGYROcommand = "cgyro -t ."

        self.cgyro_job.prep(
            CGYROcommand,
            input_files=[input_cgyro_file, inputgacode_file_this],
            output_files=self.output_files_test,
        )

        self.cgyro_job.run()

    def run1(self,subfolder,test_run=False,**kwargs):

        if test_run:
            self.run_test(subfolder,**kwargs)
        else:
            self.run_full(subfolder,**kwargs)

    def run_full(
        self,
        subfolder,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
        scan_param = None,      # {'variable': 'KY', 'values': [0.2,0.3,0.4]}
        enforce_equality = None,  # e.g. {'DLNTDR_SCALE_2': 'DLNTDR_SCALE_1', 'DLNTDR_SCALE_3': 'DLNTDR_SCALE_1'}
        minutes = 5,
        n = 16,
        nomp = 1,
        cpuspertask=None, # if None, will default to 1
        queue=None, #if blank will default to the one in settings
        mem=None, # in MB
        submit_via_qsub=True, #TODO fix this, works only at NERSC? no scans?
        clean_folder_going_in=True, # Make sure the scratch folder is removed before running (unless I want a restart!)
        submit_run=True, # False if I just want to check and fetch the job that was already submitted (e.g. via qsub or slurm)
    ):
        
        input_cgyro_file, inputgacode_file_this = self._prerun(
            subfolder,
            roa=roa,
            CGYROsettings=CGYROsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
        )

        self.cgyro_job = FARMINGtools.mitim_job(self.folder)

        name = f'mitim_cgyro_{subfolder}_{roa:.6f}'

        if scan_param is not None and submit_via_qsub:
            raise Exception(" <MITIM> Cannot use scan_param with submit_via_qsub=True, because it requires a different job for each value of the scan parameter.")

        if submit_via_qsub:

            self.cgyro_job.define_machine(
                "cgyro",
                name,
                launchSlurm=False,
            )

            subfolder = "scan0"
            queue = "-queue " + self.cgyro_job.machineSettings['slurm']['partition'] if "partition" in self.cgyro_job.machineSettings['slurm'] else ""

            if mem is not None:
                CGYROcommand = f'gacode_qsub -e {subfolder} -n {n} -nomp {nomp} -mem {mem} {queue} -w 0:{minutes}:00 -s'
            else:
                CGYROcommand = f'gacode_qsub -e {subfolder} -n {n} -nomp {nomp} {queue} -w 0:{minutes}:00 -s'

            if "account" in self.cgyro_job.machineSettings["slurm"] and self.cgyro_job.machineSettings["slurm"]["account"] is not None:
                CGYROcommand += f" -repo {self.cgyro_job.machineSettings['slurm']['account']}"

            self.slurm_output = "batch.out"

            # ---
            folder_run = self.folder / subfolder
            folder_run.mkdir(parents=True, exist_ok=True)

            # Copy the input.cgyro in the subfolder
            input_cgyro_file_this = folder_run / "input.cgyro"
            shutil.copy2(input_cgyro_file, input_cgyro_file_this)

            # Copy the input.gacode file in the subfolder
            inputgacode_file_this = folder_run / "input.gacode"
            shutil.copy2(self.inputgacode_file, inputgacode_file_this)

            # Prepare the input and output folders
            input_folders = [folder_run]
            output_folders = [subfolder]

        else:

            if scan_param is None:
                job_array = None
                folder = 'scan0'
                scan_param = {'variable': None, 'values': [0]}  # Dummy scan parameter to avoid issues with the code below
            else:
                # Array
                job_array = ''
                for i,value in enumerate(scan_param['values']):
                    if job_array != '':
                        job_array += ','
                    job_array += str(i)

                folder = 'scan"$SLURM_ARRAY_TASK_ID"'

            # Machine
            self.cgyro_job.define_machine(
                "cgyro",
                name,
                slurm_settings={
                    "name": name,
                    "minutes": minutes,
                    "ntasks": n,
                    "job_array": job_array,
                    # Validate n and nomp before assigning cpuspertask
                },
            )

            if cpuspertask is not None:
                if not isinstance(cpuspertask, int):
                    raise TypeError(" <MITIM> cpuspertask must be an integer")
                self.cgyro_job.slurm_settings["cpuspertask"] = cpuspertask
            

            if queue is not None:
                self.cgyro_job.machineSettings['slurm']['partition'] = queue

            if mem is not None:
                self.cgyro_job.slurm_settings['mem'] = mem

            # if not self.cgyro_job.launchSlurm:
            #     raise Exception(" <MITIM> Cannot run CGYRO scans without slurm")

            # Command to run cgyro
            CGYROcommand = f'cgyro -e {folder} -n {n} -nomp {nomp} -p {self.cgyro_job.folderExecution}'

            # Scans
            input_folders = []
            output_folders = []
            for i,value in enumerate(scan_param['values']):
                subfolder = f"scan{i}"
                folder_run = self.folder / subfolder
                folder_run.mkdir(parents=True, exist_ok=True)

                # Copy the input.cgyro in the subfolder
                input_cgyro_file_this = folder_run / "input.cgyro"
                shutil.copy2(input_cgyro_file, input_cgyro_file_this)

                # Modify the input.cgyro file with the scan parameter
                extraOptions_this = extraOptions.copy()
                if scan_param['variable'] is not None:
                    extraOptions_this[scan_param['variable']] = value


                # If there is an enforce_equality, apply it
                if enforce_equality is not None:
                    for key in enforce_equality:
                        extraOptions_this[key] = extraOptions_this[enforce_equality[key]]

                inputCGYRO = CGYROinput(file=input_cgyro_file_this)
                input_cgyro_file_this = GACODErun.modifyInputs(
                    inputCGYRO,
                    code_settings=CGYROsettings,
                    extraOptions=extraOptions_this,
                    multipliers=multipliers,
                    addControlFunction=GACODEdefaults.addCGYROcontrol,
                    rmin=roa,
                    control_file = 'input.cgyro.controls'
                )

                input_cgyro_file_this.write_state()

                # Copy the input.gacode file in the subfolder
                inputgacode_file_this = folder_run / "input.gacode"
                shutil.copy2(self.inputgacode_file, inputgacode_file_this)

                # Prepare the input and output folders
                input_folders.append(folder_run)
                output_folders.append(subfolder)

            self.slurm_output = "slurm_output.dat"

        # First submit the job with gacode_qsub, which will submit the cgyro job via slurm, with name 
        self.cgyro_job.prep(
            CGYROcommand,
            input_folders = input_folders,
            output_folders=output_folders,
            )

        if submit_run:
            self.cgyro_job.run(
                waitYN=False,
                check_if_files_received=False,
                removeScratchFolders=False,
                removeScratchFolders_goingIn=clean_folder_going_in, 
                )

        # Prepare how to search for the job without waiting for it
        name_default_submission_qsub = Path(self.cgyro_job.folderExecution).name

        self.cgyro_job.launchSlurm = True
        self.cgyro_job.slurm_settings['name'] = name_default_submission_qsub


    def check(self, every_n_minutes=5):

        if self.cgyro_job.launchSlurm:
            print("- Checker job status")

            while True:
                self.cgyro_job.check(file_output = self.slurm_output)
                print(f'\t- Current status (as of  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}): {self.cgyro_job.status} ({self.cgyro_job.infoSLURM["STATE"]})')
                if self.cgyro_job.status == 2:
                    break
                else:
                    print(f"\t- Waiting {every_n_minutes} minutes")
                    time.sleep(every_n_minutes * 60)
        else:
            print("- Not checking status because this was run command line (not slurm)")

        print("\n\t* Job considered finished",typeMsg="i")

    def fetch(self):
        """
        For a job that has been submitted but not waited for, once it is done, get the results
        """

        print("\n\n\t- Fetching results")

        if self.cgyro_job.launchSlurm:
            self.cgyro_job.connect()
            self.cgyro_job.retrieve()
            self.cgyro_job.close()
        else:
            print("- Not retrieving results because this was run command line (not slurm)")

    def delete(self):

        print("\n\n\t- Deleting job")

        self.cgyro_job.launchSlurm = False

        self.cgyro_job.prep(
            f"scancel -n {self.cgyro_job.slurm_settings['name']}",
            label_log_files="_finish",
        )

        self.cgyro_job.run()

    def read1(self, label="cgyro1", folder=None, tmin = 0.0, minimal = False, last_tmin_for_linear = True):

        folder = IOtools.expandPath(folder) if folder is not None else self.folder

        folders = sorted(list((folder).glob("scan*")), key=lambda f: int(''.join(filter(str.isdigit, f.name))))
        
        if len(folders) == 0:
            folders = [folder]
            attach_name = False
        else:
            print(f"\t- Found {len(folders)} scan folders in {folder.resolve()}:")
            for f in folders:
                print(f"\t\t- {f.name}")
            attach_name = True

        data = {}
        labels = []
        for folder in folders:
            
            if attach_name:
                label1 = f"{label}_{folder.name}"
            else:
                label1 = label

            data[label1] = CGYROutils.CGYROout(folder, tmin=tmin, minimal=minimal, last_tmin_for_linear=last_tmin_for_linear)
            labels.append(label1)

        self.results.update(data)
        
        if attach_name:
            self.results[label] = CGYROutils.CGYROlinear_scan(labels, data)

class CGYROinput(GACODErun.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.cgyro.controls",
            code="CGYRO",
            n_species='N_SPECIES',
        )

def _2D_mosaic(n_times):

    num_cols = n_times

    # Create the mosaic layout dynamically
    mosaic = []
    counter = 1
    for _ in range(3):
        row = []
        for _ in range(num_cols):
            row.append(str(counter))
            counter += 1
        mosaic.append(row)
        
    return mosaic