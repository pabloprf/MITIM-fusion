from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed
from mitim_tools.plasmastate_tools.utils import state_plotting

def plot(self, axs, axsRes, figs=None, c="r", label="powerstate", batch_num=0, compare_to_state=None, c_orig="b", show_stds=False):
    
    # -----------------------------------------------------------------------------------------------------------
    # ---- Plot profiles object
    # -----------------------------------------------------------------------------------------------------------

    if figs is not None:

        # Insert profiles with the latest powerstate
        profiles_new = self.from_powerstate(insert_highres_powers=True)

        # Plot the inserted profiles together with the original ones
        _ = state_plotting.plotAll([self.profiles, profiles_new], figs=figs)

    # -----------------------------------------------------------------------------------------------------------
    # ---- Plot plasma state
    # -----------------------------------------------------------------------------------------------------------

    set_plots = [ ]

    if "te" in self.predicted_channels:
        set_plots.append(
            [   'te', 'aLte', 'QeMWm2_tr', 'QeMWm2',
                'Electron Temperature','$T_e$ (keV)','$a/LT_e$','$Q_e$ (GB)','$Q_e$ ($MW/m^2$)',
                1.0,"Qgb"])
    if "ti" in self.predicted_channels:
        set_plots.append(
            [   'ti', 'aLti', 'QiMWm2_tr', 'QiMWm2',
                'Ion Temperature','$T_i$ (keV)','$a/LT_i$','$Q_i$ (GB)','$Q_i$ ($MW/m^2$)',
                1.0,"Qgb"])
    if "ne" in self.predicted_channels:

        # If this model provides the raw particle flux, go for it
        if 'Ge1E20m2_tr' in self.plasma:
            set_plots.append(
                [   'ne', 'aLne', 'Ge1E20m2_tr', 'Ge1E20m2',
                    'Electron Density','$n_e$ ($10^{20}m^{-3}$)','$a/Ln_e$','$\\Gamma_e$ (GB)','$\\Gamma_e$ ($10^{20}m^{-3}/s$)',
                    1E-1,"Ggb"])
        else:
            set_plots.append(
                [   'ne', 'aLne', 'Ce_tr', 'Ce',
                    'Electron Density','$n_e$ ($10^{20}m^{-3}$)','$a/Ln_e$','$Q_{conv,e}$ (GB)','$Q_{conv,e}$ ($MW/m^2$)',
                    1E-1,"Qgb"])

    if "nZ" in self.predicted_channels:

        # If this model provides the raw particle flux, go for it
        if 'GZ1E20m2_tr' in self.plasma:
            set_plots.append(
                [   'nZ', 'aLnZ', 'GZ1E20m2_tr', 'GZ1E20m2',
                    'Impurity Density','$n_Z$ ($10^{20}m^{-3}$)','$a/Ln_Z$','$\\Gamma_Z$ (GB)','$\\Gamma_Z$ ($10^{20}m^{-3}/s$)',
                    1E-1,"Ggb"])
        else:
            set_plots.append(
                [   'nZ', 'aLnZ', 'CZ_tr', 'CZ',
                    'Impurity Density','$n_Z$ ($10^{20}m^{-3}$)','$a/Ln_Z$','$\\widehat{Q}_{conv,Z}$ (GB)','$\\widehat{Q}_{conv,Z}$ ($MW/m^2$)',
                    1E-1,"Qgb"])

    if "w0" in self.predicted_channels:
        set_plots.append(
            [   'w0', 'aLw0', 'MtJm2_tr', 'MtJm2',
                'Rotation','$\\omega_0$ ($krad/s$)','$-d\\omega_0/dr$ ($krad/s/cm$)','$\\Pi$ (GB)','$\\Pi$ ($J/m^2$)',
                1E-3,"Pgb"])

    cont = 0
    for set_plot in set_plots:
            
            if compare_to_state is not None:
                plot_kp(
                    compare_to_state.plasma,
                    axs[cont], axs[cont+1], axs[cont+2], axs[cont+3],
                    *set_plot,
                    c_orig, 'original', batch_num=batch_num)

            plot_kp(
                self.plasma,
                axs[cont], axs[cont+1], axs[cont+2], axs[cont+3],
                *set_plot,
                c, label, batch_num=batch_num, show_stds=show_stds)

            if  cont == 0:
                axs[cont].legend()

            cont += 4

    # -----------------------------------------------------------------------------------------------------------
    # ---- Plot flux matching
    # -----------------------------------------------------------------------------------------------------------

    # Nice LaTeX labels per predicted channel
    _nice_labels = {
        'te': ('$a/L_{T_e}$',  '$|\\Delta Q_e|$'),
        'ti': ('$a/L_{T_i}$',  '$|\\Delta Q_i|$'),
        'ne': ('$a/L_{n_e}$',  '$|\\Delta \\Gamma_e|$'),
        'nZ': ('$a/L_{n_Z}$',  '$|\\Delta \\Gamma_Z|$'),
        'w0': ('$|d\\omega_0/dr|$', '$|\\Delta \\Pi|$'),
    }

    if self.FluxMatch_Yopt.shape[0] > 0:
        ax = axsRes[0]
        ax.plot(self.FluxMatch_Yopt.mean(axis=1), "-o", color=c, markersize=2)

        # Stopping criterion
        if getattr(self, 'FluxMatch_tol', None) is not None:
            ax.axhline(y=self.FluxMatch_tol, color='k', linestyle='--', lw=1.2, label=f'Tolerance Criterion')
            ax.legend(fontsize=10, frameon=False)

        # Oscillation-check iterations
        for it in getattr(self, 'FluxMatch_osc_iters', []):
            ax.axvline(x=it, color='0.6', linestyle=':', lw=0.8)

        ax.set_ylabel("Mean flux residual")
        ax.set_xlim(left=0)
        ax.set_yscale("log")

        colors = GRAPHICStools.listColors()
        
        lw = 0.5

        cont = 0
        for i, ch in enumerate(self.predicted_channels):
            aL_label, res_label = _nice_labels.get(ch, (self.labelsFM[i][0], f'$|\\Delta${self.labelsFM[i][1]}$|$'))

            # Plot gradient evolution
            ax = axsRes[1+cont]
            for j in range(self.plasma['rho'].shape[-1]-1):

                position_in_batch = i * ( self.plasma['rho'].shape[-1] -1 ) + j

                ax.plot(self.FluxMatch_Xopt[:,position_in_batch], "-o", color=colors[j], lw=lw, label=f"$r/a={self.plasma['roa'][batch_num,j+1]:.2f}$", markersize=0.5)
                if self.bounds_current is not None:
                    for u in [0,1]:
                        ax.axhline(y=self.bounds_current[u,position_in_batch], color=colors[j], linestyle='-.', lw=0.2)

            ax.set_ylabel(aL_label)

            for it in getattr(self, 'FluxMatch_osc_iters', []):
                ax.axvline(x=it, color='0.6', linestyle=':', lw=0.8)

            if i == len(self.predicted_channels)-1:
                GRAPHICStools.addLegendApart(ax, ratio=1.0, extraPad=0.05, size=9)

            # Plot residual evolution
            ax = axsRes[1+cont+1]
            for j in range(self.plasma['rho'].shape[-1]-1):

                position_in_batch = i * ( self.plasma['rho'].shape[-1] -1 ) + j

                ax.plot(self.FluxMatch_Yopt[:,position_in_batch], "-o", color=colors[j], lw=lw, markersize=1)

            # if getattr(self, 'FluxMatch_tol', None) is not None:
            #     ax.axhline(y=self.FluxMatch_tol, color='k', linestyle='--', lw=1.2)
            for it in getattr(self, 'FluxMatch_osc_iters', []):
                ax.axvline(x=it, color='0.6', linestyle=':', lw=0.8)

            ax.set_ylabel(res_label)
            ax.set_yscale("log")

            # Plot relaxation parameter evolution
            ax = axsRes[1+cont+2]
            if self.FluxMatch_relax.numel() > 0:
                for j in range(self.plasma['rho'].shape[-1]-1):

                    position_in_batch = i * ( self.plasma['rho'].shape[-1] -1 ) + j

                    ax.plot(self.FluxMatch_relax[:,position_in_batch], "-o", color=colors[j], lw=lw, markersize=0.5)

            for it in getattr(self, 'FluxMatch_osc_iters', []):
                ax.axvline(x=it, color='0.6', linestyle=':', lw=0.8)

            ax.set_ylabel("Relaxation param., $\\eta$")
            ax.set_yscale("log")

            cont += 3

        for ax in axsRes:
            ax.set_xlabel("Iteration")
            ax.set_xlim(left=0)
            #GRAPHICStools.addDenseAxis(ax)
        
def plot_kp(plasma, ax, ax_aL, ax_Fgb, ax_F, key, key_aL, key_Ftr, key_Ftar, title, ylabel, ylabel_aL, ylabel_Fgb, ylabel_F, multiplier_profile, labelGB, c, label, batch_num=0, show_stds=False):

    ax.set_title(title)
    ax.plot(
        plasma["rho"][batch_num,:],
        plasma[key][batch_num,:]*multiplier_profile,
        "-o",
        color=c,
        label=label,
        markersize=3,
        lw=1.0,
    )
    ax.set_xlim([0, 1])
    ax.set_ylabel(ylabel)
    # ax.set_ylim(bottom=0)
    
    ax_aL.plot(
        plasma["rho"][batch_num,:],
        plasma[key_aL][batch_num,:],
        "-o",
        color=c,
        label=label,
        markersize=3,
        lw=1.0,
    )
    ax_aL.set_xlim([0, 1])
    ax_aL.set_ylabel(ylabel_aL)
    # ax_aL.set_ylim(bottom=0)
    
    ax_Fgb.plot(
        plasma["rho"][batch_num,1:],
        plasma[key_Ftr][batch_num,1:] / plasma[labelGB][batch_num,1:],
        "-o",
        color=c,
        markersize=3,
        lw=1.0,
    )
    ax_Fgb.plot(
        plasma["rho"][batch_num,1:],
        plasma[key_Ftar][batch_num,1:] / plasma[labelGB][batch_num,1:],
        "--*",
        color=c,
        markersize=3,
        lw=1.0,
    )
    ax_Fgb.set_xlim([0, 1])
    ax_Fgb.set_xlabel('$\\rho$')
    ax_Fgb.set_ylabel(ylabel_Fgb)
    # Heat fluxes (Qe, Qi) are physically positive -> log. Particle fluxes
    # (Ge, GZ) and momentum flux (Mt) can be negative under inward pinch /
    # counter-rotation regimes, so log would drop those points. Use symlog
    # with a small linthresh so near-zero values don't blow up the axis.
    if key in ('te', 'ti'):
        ax_Fgb.set_yscale("log")
    else:
        ax_Fgb.set_yscale("symlog", linthresh=1e-2)
    
    ax_F.plot(
        plasma["rho"][batch_num,1:],
        plasma[key_Ftr][batch_num,1:],
        "-o",
        color=c,
        markersize=3,
        lw=1.0,
    )
    ax_F.plot(
        plasma["rho"][batch_num,1:], plasma[key_Ftar][batch_num,1:], "--*", color=c, markersize=3, lw=1.0
    )
    ax_F.set_xlim([0, 1])
    ax_F.set_xlabel('$\\rho$')
    ax_F.set_ylabel(ylabel_F)
    # ax_F.set_ylim(bottom=0)

    # Optional per-evaluation uncertainty on the transport flux. The turbulent
    # and neoclassical contributions carry independent stds so we combine them
    # in quadrature; fall back gracefully for channels (convective Ce/CZ) that
    # don't expose both halves as _tr_turb_stds / _tr_neoc_stds.
    if show_stds:
        turb_std = plasma.get(f"{key_Ftr}_turb_stds")
        neoc_std = plasma.get(f"{key_Ftr}_neoc_stds")
        if turb_std is not None and neoc_std is not None:
            import torch
            if isinstance(turb_std, torch.Tensor) or isinstance(neoc_std, torch.Tensor):
                std_all = (turb_std**2 + neoc_std**2).sqrt()
            else:
                import numpy as _np
                std_all = _np.sqrt(turb_std**2 + neoc_std**2)
            rho = plasma["rho"][batch_num, 1:]
            std_row = std_all[batch_num, 1:]
            ax_Fgb.errorbar(
                rho,
                plasma[key_Ftr][batch_num, 1:] / plasma[labelGB][batch_num, 1:],
                yerr=std_row / plasma[labelGB][batch_num, 1:],
                fmt='none', ecolor=c, elinewidth=0.6, capsize=2, alpha=0.7, zorder=2,
            )
            ax_F.errorbar(
                rho,
                plasma[key_Ftr][batch_num, 1:],
                yerr=std_row,
                fmt='none', ecolor=c, elinewidth=0.6, capsize=2, alpha=0.7, zorder=2,
            )

    for ax in [ax, ax_aL, ax_Fgb, ax_F]:
        GRAPHICStools.addDenseAxis(ax)


def plot_metrics_powerstates(axsM, powerstates, profiles=None, profiles_color='b', n_trajectories=1):

    _TRAJ_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple',
                    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    n_ps = len(powerstates)
    n_traj = n_trajectories

    # --- Residual panel ---
    ax = axsM[0]
    if n_traj > 1:
        for t in range(n_traj):
            xs, ys = [], []
            for i in range(n_ps):
                if i % n_traj == t:
                    xs.append(i)
                    ys.append(powerstates[i].plasma['residual'].item())
            ax.plot(xs, ys, '-s', color=_TRAJ_COLORS[t % len(_TRAJ_COLORS)],
                    lw=1, ms=4, label=f'T{t}')
        ax.legend(prop={"size": 7})
    else:
        x, y = [], []
        for h in range(n_ps):
            x.append(h)
            y.append(powerstates[h].plasma['residual'].item())
        ax.plot(x, y, '-s', color='b', lw=1, ms=5)
    ax.set_yscale('log')
    ax.set_ylabel('Mean Residual')
    ax.set_xlim([0, n_ps + 1])
    GRAPHICStools.addDenseAxis(ax)

    # --- Fusion power panel ---
    ax = axsM[1]
    if n_traj > 1:
        for t in range(n_traj):
            xs, ys = [], []
            for i in range(n_ps):
                if i % n_traj == t:
                    xs.append(i)
                    Pfus = powerstates[i].from_density_to_flux(
                        (powerstates[i].plasma["qfuse"] + powerstates[i].plasma["qfusi"]) * 5.0
                    ) * powerstates[i].plasma["volp"]
                    ys.append(Pfus[..., -1].item())
            ax.plot(xs, ys, '-s', color=_TRAJ_COLORS[t % len(_TRAJ_COLORS)],
                    lw=1, ms=4, label=f'T{t}')
    else:
        x, y = [], []
        for h in range(n_ps):
            x.append(h)
            Pfus = powerstates[h].from_density_to_flux(
                (powerstates[h].plasma["qfuse"] + powerstates[h].plasma["qfusi"]) * 5.0
            ) * powerstates[h].plasma["volp"]
            y.append(Pfus[..., -1].item())
        if profiles is not None:
            x.append(h + 1)
            y.append(profiles.derived["Pfus"])
        ax.plot(x, y, '-s', color='b', lw=1, ms=5)
        if profiles is not None:
            ax.plot(x[-1], y[-1], 's', color=profiles_color, ms=5)
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Fusion Power (MW)')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, n_ps + 1])
