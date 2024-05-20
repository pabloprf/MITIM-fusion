import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print


def plot(self, axs, axsRes, figs=None, c="r", label=""):
    profiles_new = self.insertProfiles(self.profiles, insertPowers=True)
    if figs is not None:
        fn = PROFILEStools.plotAll([self.profiles, profiles_new], figs=figs)

    # plotPlasma(self,self.FluxMatch_plasma_orig,axs,color='b')
    plotPlasma(self, self.plasma, axs, color=c, label=label)

    ax = axsRes
    if self.FluxMatch_Yopt.shape[0] > 0:
        ax.plot(self.FluxMatch_Yopt, "--*", color=c, markersize=1, lw=0.2)
        ax.plot(
            self.FluxMatch_Yopt.mean(axis=1),
            "-o",
            label=f"Mean {label}",
            color=c,
            markersize=3,
            lw=2.0,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Flux difference")
        ax.set_xlim(left=0)
        ax.legend(prop={"size": 6})
        ax.set_yscale("log")

    for ax in axs:
        GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.addDenseAxis(axsRes)


def plotPlasma(powerstate, plasma, axs, color="b", label=""):
    axs[0].set_title("Electron Temperature")
    axs[1].set_title("Ion Temperature")
    axs[2].set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
    # ax.set_ylim(bottom=0)
    axs[3].set_title("Impurity Density")
    axs[4].set_title("Rotation")

    if "te" in powerstate.ProfilesPredicted:
        ax = axs[0]
        ax.plot(
            plasma["rho"][0],
            plasma["te"][0],
            "-o",
            color=color,
            label=label,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$T_e$ (keV)")
        # ax.set_ylim(bottom=0)

        ax = axs[5]
        ax.plot(
            plasma["rho"][0],
            plasma["aLte"][0],
            "-o",
            color=color,
            label=label,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$a/LT_e$")
        # ax.set_ylim(bottom=0)

    if "ti" in powerstate.ProfilesPredicted:
        ax = axs[1]
        ax.plot(
            plasma["rho"][0], plasma["ti"][0], "-o", color=color, markersize=3, lw=1.0
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$T_i$ (keV)")
        # ax.set_ylim(bottom=0)

        ax = axs[6]
        ax.plot(
            plasma["rho"][0],
            plasma["aLti"][0],
            "-o",
            color=color,
            label=label,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$a/LT_i$")
        # ax.set_ylim(bottom=0)

    if "ne" in powerstate.ProfilesPredicted:
        ax = axs[2]
        ax.plot(
            plasma["rho"][0],
            plasma["ne"][0] * 1e-1,
            "-o",
            color=color,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');

        ax = axs[7]
        ax.plot(
            plasma["rho"][0],
            plasma["aLne"][0],
            "-o",
            color=color,
            label=label,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$a/Ln_e$")
        # ax.set_ylim(bottom=0)

    if "nZ" in powerstate.ProfilesPredicted:
        ax = axs[3]
        ax.plot(
            plasma["rho"][0],
            plasma["nZ"][0] * 1e-1,
            "-o",
            color=color,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
        # ax.set_ylim(bottom=0)

        ax = axs[8]
        ax.plot(
            plasma["rho"][0],
            plasma["aLnZ"][0],
            "-o",
            color=color,
            label=label,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$a/Ln_Z$")
        # ax.set_ylim(bottom=0)

    if "w0" in powerstate.ProfilesPredicted:
        ax = axs[4]
        ax.plot(
            plasma["rho"][0],
            plasma["w0"][0] * 1e-3,
            "-o",
            color=color,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$\\omega_0$ ($krad/s$)")
        # ax.set_ylim(bottom=0)

        ax = axs[9]
        ax.plot(
            plasma["rho"][0],
            plasma["aLw0"][0],
            "-o",
            color=color,
            label=label,
            markersize=3,
            lw=1.0,
        )
        ax.set_xlim([0, 1])
        # ax.set_xlabel('$\\rho_N$');
        ax.set_ylabel("$-d\\omega_0/dr$ ($krad/s/cm$)")
        # ax.set_ylim(bottom=0)

    ax = axs[10]
    ax.plot(
        plasma["rho"][0],
        plasma["Pe_tr"][0] / plasma["Qgb"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["Pe"][0] / plasma["Qgb"][0],
        "--*",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.set_xlim([0, 1])
    # ax.set_xlabel('$\\rho_N$');
    ax.set_ylabel("$Q_e$ (GB)")
    # ax.set_ylim(bottom=0)
    ax.set_yscale("log")

    ax = axs[11]
    ax.plot(
        plasma["rho"][0],
        plasma["Pi_tr"][0] / plasma["Qgb"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["Pi"][0] / plasma["Qgb"][0],
        "--*",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.set_xlim([0, 1])
    # ax.set_xlabel('$\\rho_N$');
    ax.set_ylabel("$Q_i$ (GB)")
    # ax.set_ylim(bottom=0)
    ax.set_yscale("log")

    Ggb = (
        plasma["Qgb"][0] if powerstate.useConvectiveFluxes else plasma["Ggb"][0]
    )

    ax = axs[12]
    ax.plot(
        plasma["rho"][0],
        plasma["Ce_tr_turb"][0] / Ggb,
        "-o",
        color=color,
        markersize=4,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["Ce_tr"][0] / Ggb,
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["Ce"][0] / Ggb,
        "--*",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.set_xlim([0, 1])
    # ax.set_xlabel('$\\rho_N$');
    ax.set_ylabel("$Q_{conv,e}$ (GB, w/factor)")
    ax.axhline(y=0, ls="--", c="k")

    ax = axs[13]
    ax.plot(
        plasma["rho"][0],
        plasma["CZ_tr_turb"][0] / Ggb,
        "-o",
        color=color,
        markersize=4,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["CZ_tr"][0] / Ggb,
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["CZ"][0] / Ggb,
        "--*",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.set_xlim([0, 1])
    # ax.set_xlabel('$\\rho_N$');
    ax.set_ylabel("$Q_{conv,Z}$ (GB, w/factor)")
    ax.axhline(y=0, ls="--", c="k")

    ax = axs[14]
    ax.plot(
        plasma["rho"][0],
        plasma["Mt_tr"][0] / plasma["Pgb"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0],
        plasma["Mt"][0] / plasma["Pgb"][0],
        "--*",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.set_xlim([0, 1])
    # ax.set_xlabel('$\\rho_N$');
    ax.set_ylabel("$\\Pi$ (GB)")
    # ax.set_ylim(bottom=0)
    # ax.set_yscale('log')

    ax = axs[15]
    ax.plot(
        plasma["rho"][0],
        plasma["Pe_tr"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0], plasma["Pe"][0], "--*", color=color, markersize=3, lw=1.0
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$Q_e$ ($MW/m^2$)")
    # ax.set_ylim(bottom=0)

    ax = axs[16]
    ax.plot(
        plasma["rho"][0],
        plasma["Pi_tr"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0], plasma["Pi"][0], "--*", color=color, markersize=3, lw=1.0
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$Q_i$ ($MW/m^2$)")
    # ax.set_ylim(bottom=0)

    ax = axs[17]
    ax.plot(
        plasma["rho"][0],
        plasma["Ce_tr"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0], plasma["Ce"][0], "--*", color=color, markersize=3, lw=1.0
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$Q_{conv,e}$ ($MW/m^2$)")
    ax.axhline(y=0, ls="--", c="k")

    ax = axs[18]
    ax.plot(
        plasma["rho"][0],
        plasma["CZ_tr"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0], plasma["CZ"][0], "--*", color=color, markersize=3, lw=1.0
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$Q_{conv,Z}$ ($MW/m^2$)")
    ax.axhline(y=0, ls="--", c="k")

    ax = axs[19]
    ax.plot(
        plasma["rho"][0],
        plasma["Mt_tr"][0],
        "-o",
        color=color,
        markersize=3,
        lw=1.0,
    )
    ax.plot(
        plasma["rho"][0], plasma["Mt"][0], "--*", color=color, markersize=3, lw=1.0
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$\\Pi$ ($J/m^2$)")
    # ax.set_ylim(bottom=0)
