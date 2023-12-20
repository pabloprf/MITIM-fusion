import copy
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from mitim_tools.misc_tools.IOtools import printMsg as print


def normalizations(
    profiles, LocationCDF=None, time=None, avTime=None, cdf_open=None, tgyro=None
):
    """
    Philosophy is that input.gacode is always required since we need to know the first ion.
    TRANSP CDF is optional for plotting purposes. I interpolate into input.gacode grid

    Here is where interpolation takes place. All in same rho grid. PROFILES, TRANSP and TGYRO
    """

    norm_select = "PROFILES"  # Use this normalization throughout

    # -----------------
    # input.gacode
    # -----------------
    Norm_profiles, rho, roa, mi_ref = normalizations_profiles(profiles)

    # -----------------
    # TRANSP .CDF
    # -----------------
    if LocationCDF is not None:
        Norm_transp, cdf = normalizations_transp(
            LocationCDF, mi_ref, time, avTime, cdf_open=cdf_open, interp_rho=rho
        )
    else:
        Norm_transp, cdf = None, None

    # -----------------
    # TGYRO
    # -----------------
    if tgyro is not None:
        Norm_tgyro = normalizations_tgyro(tgyro, rho, roa)
    else:
        Norm_tgyro = None

    # -----------------
    # ~~~~ Combination
    # -----------------
    Norms = {
        "TRANSP": Norm_transp,
        "PROFILES": Norm_profiles,
        "TGYRO": Norm_tgyro,
        "EXP": None,
        "input_gacode": profiles,
    }

    Norms["SELECTED"] = Norms[norm_select]

    # ------------------------
    # ~~~~ Experimental fluxes
    # ------------------------
    if Norms[norm_select] is not None:
        Norms["EXP"] = {
            "rho": Norms[norm_select]["rho"],
            "exp_Qe": Norms[norm_select]["exp_Qe"],
            "exp_Qi": Norms[norm_select]["exp_Qi"],
            "exp_Ge": Norms[norm_select]["exp_Ge"],
            "exp_Qe_gb": Norms[norm_select]["exp_Qe"] / Norms[norm_select]["q_gb"],
            "exp_Qi_gb": Norms[norm_select]["exp_Qi"] / Norms[norm_select]["q_gb"],
            "exp_Ge_gb": Norms[norm_select]["exp_Ge"] / Norms[norm_select]["g_gb"],
        }
    else:
        print("\t\t\t- Normalization set is empty", typeMsg="w")

    return Norms, cdf


def normalizations_tgyro(tgyro, rho, roa):
    """
    Grab normalizations of TGYRO frmo the last iteration
    """
    iteration = -1

    # Complete in future

    x_tgyro = np.interp(tgyro.roa[iteration], roa, rho)

    Set_norm = {
        "rho": rho,
        "q_gb": np.interp(rho, x_tgyro, tgyro.Q_GB[iteration]),
        "g_gb": np.interp(rho, x_tgyro, tgyro.Gamma_GB[iteration]),
        "c_s": np.interp(rho, x_tgyro, tgyro.c_s[iteration]),
    }

    return Set_norm


def normalizations_profiles(profiles):
    if profiles is not None:
        Set_norm = {
            "rho": profiles.profiles["rho(-)"],
            "roa": profiles.derived["roa"],
            "rmin": np.abs(profiles.profiles["rmin(m)"]),
            "q_gb": np.abs(profiles.derived["q_gb"]),
            "g_gb": np.abs(profiles.derived["g_gb"]),
            "exp_Qe": np.abs(profiles.derived["qe"]),
            "exp_Qi": np.abs(profiles.derived["qi"]),
            "exp_Ge": np.abs(profiles.derived["ge"]),
            "B_unit": np.abs(profiles.derived["B_unit"]),
            "rho_s": np.abs(profiles.derived["rho_s"]),
            "c_s": np.abs(profiles.derived["c_s"]),
            "Te_keV": np.abs(
                profiles.profiles[
                    "te(keV)" if "te(keV)" in profiles.profiles else "Te(keV)"
                ]
            ),
            "ne_20": np.abs(profiles.profiles["ne(10^19/m^3)"]) * 1e-1,
            "Ti_keV": np.abs(profiles.profiles["ti(keV)"][:, 0]),
            "ni_20": np.abs(profiles.derived["ni_thrAll"]) * 1e-1,
            "exp_Qe": profiles.derived["qe_MWmiller"]
            / profiles.derived["surfGACODE_miller"],  # This is the same as qe_MWm2
            "exp_Qi": profiles.derived["qi_MWmiller"]
            / profiles.derived["surfGACODE_miller"],
            "exp_Ge": profiles.derived["ge_10E20miller"]
            / profiles.derived["surfGACODE_miller"],
            "mi_ref": profiles.derived["mi_ref"],
        }

        return Set_norm, Set_norm["rho"], Set_norm["roa"], Set_norm["mi_ref"]

    else:
        print("\t\t\t- Cannot read normalization", typeMsg="w")

        return None, None, None, None


def normalizations_transp(
    LocationCDF, mi_ref, time, avTime, cdf_open=None, interp_rho=None
):
    # Normalization
    if cdf_open is None:
        print("\t- Opening TRANSP CDF file to grab normalizations")
        from mitim_tools.transp_tools.CDFtools import CDFreactor

        try:
            cdf = CDFreactor(LocationCDF)
        except:
            print("~! Could not be opened", typeMsg="w")
            cdf = None
    else:
        cdf = cdf_open

    it1 = np.argmin(np.abs(cdf.t - (time - avTime)))
    it2 = np.argmin(np.abs(cdf.t - (time + avTime)))

    if it1 == it2:
        it1 -= 1

    print(f"\t- Using mass={mi_ref}u, as read in profiles class")

    cdf.getTGLFparameters(mi_u=mi_ref)
    # ------------------------------------------------------------

    x = cdf.x[it1:it2, :].mean(axis=0)
    xb = cdf.xb[it1:it2, :].mean(axis=0)

    # First write everything in boundary grid
    rho = xb
    Set_norm = {
        "rho": copy.deepcopy(rho),
        "rmin": cdf.rmin[it1:it2, :].mean(axis=0),
        "q_gb": cdf.TGLF_Qgb[it1:it2, :].mean(axis=0),
        "g_gb": cdf.TGLF_Ggb[it1:it2, :].mean(axis=0),
        "exp_Qe": cdf.qe_obs_GACODE[it1:it2, :].mean(axis=0),  # MW/m^2
        "exp_Qi": cdf.qi_obs_GACODE[it1:it2, :].mean(axis=0),  # MW/m^2
        "exp_Ge": cdf.Ge_obs_GACODE[it1:it2, :].mean(axis=0),  # 10^{20}/s/m^2
        "B_unit": cdf.TGLF_Bunit[it1:it2, :].mean(axis=0),
        "rho_s": np.interp(rho, x, cdf.TGLF_rhos[it1:it2, :].mean(axis=0)),
        "c_s": np.interp(rho, x, cdf.TGLF_cs[it1:it2, :].mean(axis=0)),
        "Te_keV": np.interp(rho, x, cdf.TGLF_Te[it1:it2, :].mean(axis=0)),
        "ne_20": np.interp(rho, x, cdf.TGLF_ne[it1:it2, :].mean(axis=0)),
        "Ti_keV": np.interp(rho, x, cdf.TGLF_Ti[it1:it2, :].mean(axis=0)),
        "ni_20": np.interp(rho, x, cdf.TGLF_ni[it1:it2, :].mean(axis=0)),
    }

    if interp_rho is not None:
        print("\t- Interpolating into external grid")
        for i in Set_norm:
            Set_norm[i] = np.interp(interp_rho, rho, Set_norm[i])

    return Set_norm, cdf


def plotNormQuantity(
    NormalizationSets,
    ax=None,
    axE=None,
    var="q_gb",
    colors=["b", "r", "g"],
    legYN=True,
    label="$Q$ ($MW/m^2$)",
    multiplier=1.0,
    extralab="",
):
    limE = 20  # 50

    if ax is None:
        fig, ax = plt.subplots()
        axE = ax.twinx()

    # Plot PROFILES
    x = NormalizationSets["PROFILES"]["rho"]
    zProfiles = np.abs(NormalizationSets["PROFILES"][var]) * multiplier
    ax.plot(
        x,
        zProfiles,
        "-s",
        c=colors[0],
        lw=2,
        label=extralab + "(input.gacode)",
        markersize=1,
    )

    for cont, i in enumerate(["TRANSP", "TGYRO"]):
        if NormalizationSets[i] is not None and var in NormalizationSets[i]:
            z = np.abs(NormalizationSets[i][var]) * multiplier
            ax.plot(
                x,
                z,
                "-s",
                c=colors[cont + 1],
                lw=1,
                label=f"{extralab}({i})",
                markersize=1,
            )
            err = np.abs(z - zProfiles) / zProfiles * 100
            axE.plot(x, err, c=colors[cont + 1], lw=0.5, ls="--")

    if legYN:
        ax.legend(loc="best", fontsize=6)
    ax.set_xlabel("$\\rho$")
    if label is not None:
        ax.set_ylabel(label)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    axE.set_ylabel("Relative error (%)")
    axE.set_ylim([0, limE])


def plotNormalizations(
    NormalizationSets=None,
    axs=None,
    ax_twins=None,
    colors=["b", "r", "g"],
    legYN=True,
    extralab="",
):
    if NormalizationSets is not None:
        if axs is None:
            fig4 = fn.add_figure(label="Normalization")
            grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.8)

            ax00 = fig4.add_subplot(grid[0, 0])
            ax01 = fig4.add_subplot(grid[0, 1])
            ax02 = fig4.add_subplot(grid[0, 2])
            ax03 = fig4.add_subplot(grid[0, 3])
            ax10 = fig4.add_subplot(grid[1, 0])
            ax11 = fig4.add_subplot(grid[1, 1])
            ax12 = fig4.add_subplot(grid[1, 2])
            ax13 = fig4.add_subplot(grid[1, 3])

        else:
            [ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13] = axs

        if ax_twins is None:
            ax00twin = ax00.twinx()
            ax01twin = ax01.twinx()
            ax02twin = ax02.twinx()
            ax03twin = ax03.twinx()
            ax10twin = ax10.twinx()
            ax11twin = ax11.twinx()
            ax12twin = ax12.twinx()
            ax13twin = ax13.twinx()
        else:
            [
                ax00twin,
                ax01twin,
                ax02twin,
                ax03twin,
                ax10twin,
                ax11twin,
                ax12twin,
                ax13twin,
            ] = ax_twins

        ax = ax00
        ax1 = ax00twin
        plotNormQuantity(
            NormalizationSets,
            var="q_gb",
            label="$Q$ ($MW/m^2$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
        )

        ax = ax10
        ax1 = ax10twin
        plotNormQuantity(
            NormalizationSets,
            var="g_gb",
            label="$\\Gamma$ ($10^{20}/s/m^{2}$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
        )

        ax = ax01
        ax1 = ax01twin
        plotNormQuantity(
            NormalizationSets,
            var="rho_s",
            label="$\\rho_s$ ($mm$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            multiplier=1e3,
            extralab=extralab,
        )

        ax = ax11
        ax1 = ax11twin
        plotNormQuantity(
            NormalizationSets,
            var="B_unit",
            label="$B_{unit}$ ($T$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
        )

        ax = ax02
        ax1 = ax02twin
        plotNormQuantity(
            NormalizationSets,
            var="Te_keV",
            label="$T_{e}$ ($keV$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
        )
        plotNormQuantity(
            NormalizationSets,
            var="ne_20",
            label=None,
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=False,
            extralab=extralab,
        )
        ax.set_ylabel("$T_{e}$ ($keV$), $n_{e}$ ($10^{20}m^{-3}$)")

        ax = ax12
        ax1 = ax12twin
        plotNormQuantity(
            NormalizationSets,
            var="c_s",
            label="$c_{s}$ ($km/s$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
            multiplier=1e-3,
        )

        ax = ax03
        ax1 = ax03twin
        plotNormQuantity(
            NormalizationSets,
            var="exp_Qe",
            label="Exp. $Q_{e}$ ($MW/m^2$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
        )

        ax = ax13
        ax1 = ax13twin
        plotNormQuantity(
            NormalizationSets,
            var="exp_Qi",
            label="Exp. $Q_{i}$ ($MW/m^2$)",
            ax=ax,
            axE=ax1,
            colors=colors,
            legYN=legYN,
            extralab=extralab,
        )
