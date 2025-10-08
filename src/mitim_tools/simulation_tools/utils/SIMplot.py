import copy
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle_dill
from mitim_tools.misc_tools import GRAPHICStools, IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class GKplotting:
    def _correct_rhos_labels(self, labels):
        # If it has radii, we need to correct the labels
        self.results_all = copy.deepcopy(self.results)
        self.results = {}
        labels_with_rho = []
        for label in labels:
            for i,rho in enumerate(self.rhos):
                labels_with_rho.append(f"{label}_{rho}")
                self.results[f'{label}_{rho}'] = self.results_all[label]['output'][i]
        labels = labels_with_rho
        # ------------------------------------------------
        
        return labels
    
    def _plot_trace(self, ax, object_or_label, variable, c="b", lw=1, ls="-", label_plot='', meanstd=True, var_meanstd= None):

        if isinstance(object_or_label, str):
            object_grab = self.results[object_or_label]
        else:
            object_grab = object_or_label

        t = object_grab.t
        
        if not isinstance(variable, str):
            z = variable
            if var_meanstd is not None:
                z_mean = var_meanstd[0]
                z_std = var_meanstd[1]
            
        else:
            z = object_grab.__dict__[variable]
            if meanstd and (f'{variable}_mean' in object_grab.__dict__):
                z_mean = object_grab.__dict__[variable + '_mean']
                z_std = object_grab.__dict__[variable + '_std']
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
                t[t>object_grab.tmin],
                z_mean,
                y_down=z_mean
                - z_std,
                y_up=z_mean
                + z_std,
                alpha=0.1,
                color=c,
                lw=0.5,
                islwOnlyMean=True,
                label=label_plot + f" $\\mathbf{{{z_mean:.3f} \\pm {z_std:.3f}}}$ (1$\\sigma$)",
            )
            
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
        
        if "Qe_EM" in self.results[label].__dict__:
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
        if "Ge_EM" in self.results[label].__dict__:
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
        if "Qi_EM" in self.results[label].__dict__:
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

        positive_f_mask = self.results[label].f_mean>0.0

        # Mean+Std Growth rate as function of ky
        ax = axs["C"]
        ax.errorbar(self.results[label].ky, self.results[label].g_mean, yerr=self.results[label].g_std, fmt='-', markersize=5, color=c, label=label+' (mean+std)')
        # filled circle for positive frequency, empty square for negative frequency
        ax.plot(self.results[label].ky[positive_f_mask], self.results[label].g_mean[positive_f_mask], 'o', color=c)
        ax.plot(self.results[label].ky[~positive_f_mask], self.results[label].g_mean[~positive_f_mask], 's', mfc='none', color=c)
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\gamma$ (norm.)")
        ax.set_title('Saturated Growth Rate')
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Mean+Std Frequency as function of ky
        ax = axs["D"]
        ax.errorbar(self.results[label].ky, self.results[label].f_mean, yerr=self.results[label].f_std, fmt='-o', markersize=5, color=c, label=label+' (mean+std)')
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\omega$ (norm.)")
        ax.set_title('Saturated Real Frequency')
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)
        
    def save_pickle(self, file):
        
        print('...Pickling GX class...')
    
        with open(file, "wb") as handle:
            pickle_dill.dump(self, handle, protocol=4)
            
def restore_class_pickle(file):
    
    return IOtools.unpickle_mitim(file)