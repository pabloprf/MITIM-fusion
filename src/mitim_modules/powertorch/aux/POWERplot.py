from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

def plot(self, axs, axsRes, figs=None, c="r", label="",batch_num=0, compare_to_orig=None, c_orig = 'b'):
    profiles_new = self.insertProfiles(self.profiles, insertPowers=True)
    if figs is not None:
        _ = PROFILEStools.plotAll([self.profiles, profiles_new], figs=figs)

    # ---- Plot flux matching
    
    if self.FluxMatch_Yopt.shape[0] > 0:
        ax = axsRes[0]
        ax.plot(self.FluxMatch_Yopt.mean(axis=1),"-o",color=c,markersize=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean residual")
        ax.set_xlim(left=0)
        ax.set_yscale("log")

        ax = axsRes[1]
        ax.plot(self.FluxMatch_Yopt, "-", color=c, lw=0.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual (channel)")
        ax.set_xlim(left=0)
        ax.set_yscale("log")

    for ax in axsRes:
        GRAPHICStools.addDenseAxis(ax)

    # ---------------------------------------------------
    # ---- Plot plasma state
    # ---------------------------------------------------

    set_plots = [ ]

    if "te" in self.ProfilesPredicted:
        set_plots.append(
            [   'te', 'aLte', 'Pe_tr', 'Pe',
                'Electron Temperature','$T_e$ (keV)','$a/LT_e$','$Q_e$ (GB)','$Q_e$ ($MW/m^2$)',
                1.0])
    if "ti" in self.ProfilesPredicted:
        set_plots.append(
            [   'ti', 'aLti', 'Pi_tr', 'Pi',
                'Ion Temperature','$T_i$ (keV)','$a/LT_i$','$Q_i$ (GB)','$Q_i$ ($MW/m^2$)',
                1.0])
    if "ne" in self.ProfilesPredicted:
        set_plots.append(
            [   'ne', 'aLne', 'Ce_tr', 'Ce',
                'Electron Density','$n_e$ ($10^{20}m^{-3}$)','$a/Ln_e$','$Q_{conv,e}$ (GB)','$Q_{conv,e}$ ($MW/m^2$)',
                1E-1])
    if "nZ" in self.ProfilesPredicted:
        set_plots.append(
            [   'nZ', 'aLnZ', 'CZ_tr', 'CZ',
                'Impurity Density','$n_Z$ ($10^{20}m^{-3}$)','$a/Ln_Z$','$Q_{conv,Z}$ (GB)','$Q_{conv,Z}$ ($MW/m^2$)',
                1E-1])
    if "w0" in self.ProfilesPredicted:
        set_plots.append(
            [   'w0', 'aLw0', 'Mt_tr', 'Mt',
                'Rotation','$\omega_0$ ($krad/s$)','$-d\omega_0/dr$ ($krad/s/cm$)','$\Pi$ (GB)','$\Pi$ ($J/m^2$)',
                1.0])

    cont = 0
    for set_plot in set_plots:
            
            if compare_to_orig is not None:
                plot_kp(
                    compare_to_orig.plasma,
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

def plot_kp(plasma,ax, ax_aL, ax_Fgb, ax_F, key, key_aL, key_Ftr, key_Ftar, title, ylabel, ylabel_aL, ylabel_Fgb, ylabel_F, multiplier_profile, c, label, batch_num=0):

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
        plasma[key_Ftr][batch_num,1:] / plasma["Qgb"][batch_num,1:],
        "-o",
        color=c,
        markersize=3,
        lw=1.0,
    )
    ax_Fgb.plot(
        plasma["rho"][batch_num,1:],
        plasma[key_Ftar][batch_num,1:] / plasma["Qgb"][batch_num,1:],
        "--*",
        color=c,
        markersize=3,
        lw=1.0,
    )
    ax_Fgb.set_xlim([0, 1])
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
    ax_F.set_ylabel(ylabel_F)
    # ax_F.set_ylim(bottom=0)

    for ax in [ax, ax_aL, ax_Fgb, ax_F]:
        GRAPHICStools.addDenseAxis(ax)
