import copy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class GKplotting:
    def _correct_rhos_labels(self, labels):
        # If it has radii, we need to correct the labels
        self.results_all = copy.deepcopy(self.results)
        results = {}
        labels_with_rho = []
        for label in labels:
            for i,rho in enumerate(self.rhos):
                labels_with_rho.append(f"{label}_{rho}")
                results[f'{label}_{rho}'] = self.results_all[label]['output'][i]
        labels = labels_with_rho
        print(f"\t- Corrected labels for rhos: {labels}", typeMsg='i')
        self.results = results
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

        # Track (mean, std) per trace for the factor-based y-clamp in
        # _finalize_flux_axis. The clamp uses mean+2sigma (upper) and mean-2sigma
        # (lower) so the uncertainty bars on each trace stay inside the view.
        # Prefer the driver-side scalars (z_mean / z_std from *_mean / *_std
        # attributes, which come from proper autocorrelation-aware time-averaging)
        # and fall back to a raw np.mean within tmin (std=0) when the driver hasn't
        # populated those. Stored on the axis so it accumulates across multiple
        # plot_fluxes() calls (one per rho-label).
        try:
            m = s = None
            if z_mean is not None:
                m = float(z_mean)
                s = float(z_std) if z_std is not None else 0.0
            else:
                tmin = getattr(object_grab, "tmin", None)
                if tmin is not None:
                    z_arr = np.asarray(z)
                    t_arr = np.asarray(t)
                    mask = t_arr > tmin
                    if mask.any():
                        m = float(np.mean(z_arr[mask]))
                        s = 0.0
            if m is not None and np.isfinite(m) and np.isfinite(s):
                if not hasattr(ax, "_mitim_trace_avgs"):
                    ax._mitim_trace_avgs = []
                ax._mitim_trace_avgs.append((m, s))
        except Exception:
            pass

        if meanstd and z_std is not None and z_std > 0.0:
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
            
    def _finalize_flux_axis(self, ax, legend_title=None, factor=2.5):
        """Render the legend outside the axes so it doesn't overlap the traces,
        and clamp the y-axis to factor-scaled mean+/-2sigma bounds so transient
        peaks don't dominate the view while the per-trace uncertainty bars stay
        inside the frame. `_mitim_trace_avgs` is populated by `_plot_trace` as a
        list of (mean, std) tuples during each trace draw.

            ymax = factor * max(mean + 2*std)
            ymin = min(0, factor * min(mean - 2*std))

        Rationale: transient spikes get cropped (mean-driven ceiling) but each
        trace's mean +/- 2sigma indicator still lands inside the view.
        """
        GRAPHICStools.addLegendApart(ax, loc='upper left', size=8, ratio=0.7)

        pairs = getattr(ax, "_mitim_trace_avgs", None)
        if pairs:
            uppers = [m + 2.0 * s for (m, s) in pairs]
            lowers = [m - 2.0 * s for (m, s) in pairs]
            ymax = factor * max(uppers)
            ymin = min(0.0, factor * min(lowers))
            # Guard against degenerate / NaN cases — skip the clamp rather than
            # collapse the axis to a single point.
            if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin:
                ax.set_ylim(ymin, ymax)

    def _annotate_unavailable(self, ax, what):
        ax.text(0.5, 0.5, f"{what}\nnot available in this output", transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='gray')
        ax.set_xticks([]); ax.set_yticks([])

    def plot_fluxes(self, axs=None, label="", c="b", lw=1, plotLegend=True, factor=2.5):

        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
				ABEG
                CDFH
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
            self._finalize_flux_axis(ax, factor=factor)

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
            self._finalize_flux_axis(ax, factor=factor)

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
            self._finalize_flux_axis(ax, factor=factor)

        # Ion species energy fluxes
        ax = axs["D"]
        for j, i in enumerate(self.results[label].ions_flags):
            self._plot_trace(ax,label,self.results[label].Qi_all[j],c=c,lw=lw,ls=ls[j],label_plot=f"{label}, {self.results[label].all_names[i]}", meanstd=False)

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes (separate species)')
        if plotLegend:
            self._finalize_flux_axis(ax, factor=factor)

        # Ion particle fluxes (total)
        if "E" in axs:
            ax = axs["E"]
            if "Gi" in self.results[label].__dict__:
                self._plot_trace(ax,label,"Gi",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
                if "Gi_EM" in self.results[label].__dict__:
                    self._plot_trace(ax,label,"Gi_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
                ax.set_xlabel("$t$ ($a/c_s$)")
                ax.set_ylabel("$\\Gamma_i$ (GB)")
                GRAPHICStools.addDenseAxis(ax)
                if plotLegend:
                    self._finalize_flux_axis(ax, factor=factor)
            else:
                self._annotate_unavailable(ax, "Ion particle flux")
            ax.set_title('Ion particle fluxes')

        # Ion species particle fluxes
        if "F" in axs:
            ax = axs["F"]
            if "Gi_all" in self.results[label].__dict__:
                for j, i in enumerate(self.results[label].ions_flags):
                    self._plot_trace(ax,label,self.results[label].Gi_all[j],c=c,lw=lw,ls=ls[j],label_plot=f"{label}, {self.results[label].all_names[i]}", meanstd=False)
                ax.set_xlabel("$t$ ($a/c_s$)")
                ax.set_ylabel("$\\Gamma_i$ (GB)")
                GRAPHICStools.addDenseAxis(ax)
                if plotLegend:
                    self._finalize_flux_axis(ax, factor=factor)
            else:
                self._annotate_unavailable(ax, "Per-species particle fluxes")
            ax.set_title('Ion particle fluxes (separate species)')

        # Momentum flux (all species)
        if "G" in axs:
            ax = axs["G"]
            if "Mt" in self.results[label].__dict__:
                self._plot_trace(ax,label,"Mt",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
                if "Mt_EM" in self.results[label].__dict__:
                    self._plot_trace(ax,label,"Mt_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
                ax.set_xlabel("$t$ ($a/c_s$)")
                ax.set_ylabel("$\\Pi$ (GB)")
                GRAPHICStools.addDenseAxis(ax)
                ax.axhline(0.0, color='k', ls='--', lw=0.5)
                if plotLegend:
                    self._finalize_flux_axis(ax, factor=factor)
            else:
                self._annotate_unavailable(ax, "Momentum flux")
            ax.set_title('Momentum flux (all species)')

        # Turbulent energy exchange
        if "H" in axs:
            ax = axs["H"]
            if "Se" in self.results[label].__dict__:
                self._plot_trace(ax,label,"Se",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, electrons")
                if "Si" in self.results[label].__dict__:
                    self._plot_trace(ax,label,"Si",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, ions (sum)", meanstd=False)
                ax.set_xlabel("$t$ ($a/c_s$)")
                ax.set_ylabel("$S$ (GB)")
                GRAPHICStools.addDenseAxis(ax)
                ax.axhline(0.0, color='k', ls='--', lw=0.5)
                if plotLegend:
                    self._finalize_flux_axis(ax, factor=factor)
            else:
                self._annotate_unavailable(ax, "Turbulent exchange\n(needs CGYRO output with n_flux=4)")
            ax.set_title('Turbulent energy exchange')

        # horizontal=0.9 (vs the default 0.3) gives each column enough slack for
        # the addLegendApart extrusion on subplots A/C; without this bump the
        # legend text spills across the gutter onto subplots B/D.
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.9)


    def plot_fluxes_ky(self, axs=None, label="", c="b", lw=1, plotLegend=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACE
                BDF
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

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes vs. $k_\\theta\\rho_s$(separate species)')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Ion species particle fluxes
        if "E" in axs:
            ax = axs["E"]
            if "Gi_all_ky_mean" in self.results[label].__dict__:
                for j, i in enumerate(self.results[label].ions_flags):
                    ax.plot(self.results[label].ky, self.results[label].Gi_all_ky_mean[j],ls[j]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[i]}")
                ax.set_xlabel("$k_{\\theta} \\rho_s$")
                ax.set_ylabel("$\\Gamma_i$ (GB)")
                GRAPHICStools.addDenseAxis(ax)
                ax.axhline(0.0, color='k', ls='--', lw=1)
                if plotLegend:
                    ax.legend(loc='best', prop={'size': 8},)
            else:
                self._annotate_unavailable(ax, "Per-species particle fluxes")
            ax.set_title('Ion particle fluxes vs. $k_\\theta\\rho_s$ (separate species)')

        # Momentum flux
        if "F" in axs:
            ax = axs["F"]
            if "Mt_ky_mean" in self.results[label].__dict__:
                ax.plot(self.results[label].ky, self.results[label].Mt_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
                ax.fill_between(self.results[label].ky, self.results[label].Mt_ky_mean-self.results[label].Mt_ky_std, self.results[label].Mt_ky_mean+self.results[label].Mt_ky_std, color=c, alpha=0.2)
                ax.set_xlabel("$k_{\\theta} \\rho_s$")
                ax.set_ylabel("$\\Pi$ (GB)")
                GRAPHICStools.addDenseAxis(ax)
                ax.axhline(0.0, color='k', ls='--', lw=1)
                if plotLegend:
                    ax.legend(loc='best', prop={'size': 8},)
            else:
                self._annotate_unavailable(ax, "Momentum flux")
            ax.set_title('Momentum flux vs. $k_\\theta\\rho_s$')

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
        