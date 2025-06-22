from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed
from mitim_tools.plasmastate_tools.utils import state_plotting

def plot(self, axs, axsRes, figs=None, c="r", label="powerstate",batch_num=0, compare_to_state=None, c_orig = "b"):
    
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

    if "te" in self.ProfilesPredicted:
        set_plots.append(
            [   'te', 'aLte', 'QeMWm2_tr', 'QeMWm2',
                'Electron Temperature','$T_e$ (keV)','$a/LT_e$','$Q_e$ (GB)','$Q_e$ ($MW/m^2$)',
                1.0,"Qgb"])
    if "ti" in self.ProfilesPredicted:
        set_plots.append(
            [   'ti', 'aLti', 'QiMWm2_tr', 'QiMWm2',
                'Ion Temperature','$T_i$ (keV)','$a/LT_i$','$Q_i$ (GB)','$Q_i$ ($MW/m^2$)',
                1.0,"Qgb"])
    if "ne" in self.ProfilesPredicted:

        # If this model provides the raw particle flux, go for it
        if 'Ge1E20sm2_tr' in self.plasma:
            set_plots.append(
                [   'ne', 'aLne', 'Ge1E20sm2_tr', 'Ge1E20sm2',
                    'Electron Density','$n_e$ ($10^{20}m^{-3}$)','$a/Ln_e$','$\\Gamma_e$ (GB)','$\\Gamma_e$ ($10^{20}m^{-3}/s$)',
                    1E-1,"Ggb"])
        else:
            if self.useConvectiveFluxes:
                set_plots.append(
                    [   'ne', 'aLne', 'Ce_tr', 'Ce',
                        'Electron Density','$n_e$ ($10^{20}m^{-3}$)','$a/Ln_e$','$Q_{conv,e}$ (GB)','$Q_{conv,e}$ ($MW/m^2$)',
                        1E-1,"Qgb"])
            else:
                set_plots.append(
                    [   'ne', 'aLne', 'Ce_tr', 'Ce',
                        'Electron Density','$n_e$ ($10^{20}m^{-3}$)','$a/Ln_e$','$\\Gamma_e$ (GB)','$\\Gamma_e$ ($10^{20}m^{-3}/s$)',
                        1E-1,"Ggb"])

    if "nZ" in self.ProfilesPredicted:

        # If this model provides the raw particle flux, go for it
        if 'CZ_raw_tr' in self.plasma:
            set_plots.append(
                [   'nZ', 'aLnZ', 'CZ_raw_tr', 'CZ_raw',
                    'Impurity Density','$n_Z$ ($10^{20}m^{-3}$)','$a/Ln_Z$','$\\Gamma_Z$ (GB)','$\\Gamma_Z$ ($10^{20}m^{-3}/s$)',
                    1E-1,"Ggb"])
        else:
            if self.useConvectiveFluxes:
                set_plots.append(
                    [   'nZ', 'aLnZ', 'CZ_tr', 'CZ',
                        'Impurity Density','$n_Z$ ($10^{20}m^{-3}$)','$a/Ln_Z$','$\\widehat{Q}_{conv,Z}$ (GB)','$\\widehat{Q}_{conv,Z}$ ($MW/m^2$)',
                        1E-1,"Qgb"])
            else:
                set_plots.append(
                    [   'nZ', 'aLnZ', 'CZ_tr', 'CZ',
                        'Impurity Density','$n_Z$ ($10^{20}m^{-3}$)','$a/Ln_Z$','$\\Gamma_Z$ (GB)','$\\Gamma_Z$ ($10^{20}m^{-3}/s$)',
                        1E-1,"Ggb"])

    if "w0" in self.ProfilesPredicted:
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
                c, label, batch_num=batch_num)

            if  cont == 0:
                axs[cont].legend()

            cont += 4

    # -----------------------------------------------------------------------------------------------------------
    # ---- Plot flux matching
    # -----------------------------------------------------------------------------------------------------------

    if self.FluxMatch_Yopt.shape[0] > 0:
        ax = axsRes[0]
        ax.plot(self.FluxMatch_Yopt.mean(axis=1),"-o",color=c,markersize=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean residual")
        ax.set_xlim(left=0)
        ax.set_yscale("log")

        colors = GRAPHICStools.listColors()

        cont = 0
        for i in range(len(self.ProfilesPredicted)):

            # Plot gradient evolution
            ax = axsRes[1+cont]
            for j in range(self.plasma['rho'].shape[-1]-1):    

                position_in_batch = i * ( self.plasma['rho'].shape[-1] -1 ) + j

                ax.plot(self.FluxMatch_Xopt[:,position_in_batch], "-o", color=colors[j], lw=1.0, label = f"r/a = {self.plasma['roa'][batch_num,j+1]:.2f}",markersize=0.5)
                if self.bounds_current is not None:
                    for u in [0,1]:
                        ax.axhline(y=self.bounds_current[u,position_in_batch], color=colors[j], linestyle='-.', lw=0.2)

            ax.set_ylabel(self.labelsFM[i][0])
            
            if i == len(self.ProfilesPredicted)-1:
                GRAPHICStools.addLegendApart(ax, ratio=1.0,extraPad=0.05, size=9)

            # Plot residual evolution
            ax = axsRes[1+cont+1]
            for j in range(self.plasma['rho'].shape[-1]-1):    

                position_in_batch = i * ( self.plasma['rho'].shape[-1] -1 ) + j

                ax.plot(self.FluxMatch_Yopt[:,position_in_batch], "-o", color=colors[j], lw=1.0,markersize=1)

            ax.set_ylabel(f'{self.labelsFM[i][1]} residual')
            ax.set_yscale("log")

            cont += 2

        for ax in axsRes:
            ax.set_xlabel("Iteration")
            ax.set_xlim(left=0)
            GRAPHICStools.addDenseAxis(ax)
        
def plot_kp(plasma,ax, ax_aL, ax_Fgb, ax_F, key, key_aL, key_Ftr, key_Ftar, title, ylabel, ylabel_aL, ylabel_Fgb, ylabel_F, multiplier_profile,labelGB, c, label, batch_num=0):

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
    ax_Fgb.set_yscale("log")
    
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

    for ax in [ax, ax_aL, ax_Fgb, ax_F]:
        GRAPHICStools.addDenseAxis(ax)


def plot_metrics_powerstates(axsM, powerstates, profiles=None, profiles_color='b'):

    ax = axsM[0]
    x , y = [], []
    for h in range(len(powerstates)):
        x.append(h)
        y.append(powerstates[h].plasma['residual'].item())
        
    ax.plot(x,y,'-s', color='b', lw=1, ms=5)
    ax.set_yscale('log')
    #ax.set_xlabel('Evaluation')
    ax.set_ylabel('Mean Residual')
    ax.set_xlim([0,len(powerstates)+1])
    GRAPHICStools.addDenseAxis(ax)

    ax = axsM[1]
    x , y = [], []
    for h in range(len(powerstates)):
        x.append(h)
        Pfus = powerstates[h].volume_integrate(
            (powerstates[h].plasma["qfuse"] + powerstates[h].plasma["qfusi"]) * 5.0
            ) * powerstates[h].plasma["volp"]
        y.append(Pfus[..., -1].item())

    if profiles is not None:
        x.append(h+1)
        y.append(profiles.derived["Pfus"])
    ax.plot(x,y,'-s', color='b', lw=1, ms=5)
    if profiles is not None:
            ax.plot(x[-1],y[-1],'s', color=profiles_color, ms=5)

    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Fusion Power (MW)')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom=0)
    ax.set_xlim([0,len(powerstates)+1])
