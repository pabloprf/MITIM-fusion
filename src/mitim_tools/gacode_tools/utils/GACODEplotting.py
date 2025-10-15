import copy
import numpy as np
from IPython import embed

from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print


def plotTGLFspectrum(
    axs,
    kys,
    grates,
    freq=None,
    coeff=0.0,
    removeLow=1e-4,
    markerTypes=["o", "^"],
    facecolors=None,
    c="r",
    ls="-",
    lw=1,
    label="",
    markersize=20,
    alpha=1.0,
    titles=["", "", "", "", "", ""],
    ylabel=True,
    limity=False,
    thr_symlog=1e-2,
):
    """
    Coeff is used: grates/kys**coeff
    """

    # -------------------------------------
    # Remove values with growth rate lower than removeLow
    # -------------------------------------

    if removeLow is not None:
        kys_n, grates_n, freq_n = [], [], []
        for i in range(len(kys)):
            if np.abs(grates[i]) > removeLow:
                kys_n.append(kys[i])
                grates_n.append(grates[i])
                if freq is not None:
                    freq_n.append(freq[i])
        kys = np.array(kys_n)
        grates = np.array(grates_n)
        if freq is not None:
            freq = np.array(freq_n)

    if freq is None or type(axs) != list:
        ax = axs
    else:
        ax, axF = axs[0], axs[1]

    markers = []
    for i in range(len(kys)):
        if freq is not None and freq[i] < 0:
            markers.append(markerTypes[0])
        else:
            markers.append(markerTypes[1])

    if facecolors is None:
        facecolors = [c] * len(kys)

    plot_spec(
        ax,
        kys,
        grates,
        markers=markers,
        coeff=coeff,
        c=c,
        lw=lw,
        ls=ls,
        label=label,
        alpha=alpha,
        markersize=markersize,
        facecolor=facecolors,
    )

    GRAPHICStools.addDenseAxis(ax)

    if titles is not None:
        ax.set_title(titles[0])
    if ylabel:
        ax.set_ylabel(decorateLabel("$\\gamma$", coeff))

    if coeff == 0:
        ax.set_yscale("log")
    elif limity:
        ax.set_ylim(bottom=0)

    freq_coeff = 0 # The real frequencies should not be normalized by ky

    if freq is not None and type(axs) == list:
        plot_spec(
            axF,
            kys,
            freq,
            markers=markers,
            coeff=freq_coeff,
            c=c,
            lw=lw,
            label=label,
            alpha=alpha,
            markersize=markersize,
            facecolor=facecolors,
        )

        if ylabel:
            axF.set_ylabel(decorateLabel("$\\omega$", freq_coeff))

        if freq_coeff == 0:
            axF.set_yscale("symlog", linthresh=thr_symlog)
        elif limity:
            axF.set_ylim(bottom=0)

        axF.axhline(y=0, lw=1, ls="--", c="k")

        if titles is not None:
            axF.set_title(titles[1])

        GRAPHICStools.addDenseAxis(axF)


def decorateLabel(label, coeff):
    if coeff == 0:
        label += " ($c_s/a$)"
    elif coeff == 1:
        label += "/$k_\\theta\\rho_s$ ($c_s/a$)"
    elif coeff == 2:
        label += "/$k^2_\\theta\\rho^2_s$ ($c_s/a$)"

    return label


def plotWaveform(
    axs,
    theta,
    real,
    imag,
    color="b",
    label="",
    includeSubdominant=True,
    typeline=[
        "-o",
        "-s",
        "-^",
        "-v",
        "--o",
        "--s",
        "--^",
        "--v",
        "-.o",
        "-.s",
        "-.^",
        "-.v",
    ],
):
    ax0 = axs[0]
    ax1 = axs[1]

    markersize, lw = 3, 0.5

    # Dominant
    if real.shape[0] > 0 and np.abs(np.sum(real[0])) > 1e-10:
        ax0.plot(
            theta,
            real[0],
            typeline[0],
            c=color,
            lw=lw,
            markersize=markersize,
            label=label + " (mode 1)",
        )
    if imag.shape[0] > 0 and np.abs(np.sum(imag[0])) > 1e-10:
        ax1.plot(
            theta,
            imag[0],
            typeline[0],
            c=color,
            lw=lw,
            markersize=markersize,
            label=label + " (mode 1)",
        )

    markersize, lw = 2, 0.25

    # Subdominant
    if includeSubdominant:
        for imode in range(real.shape[0] - 1):
            if np.abs(np.sum(real[imode + 1])) > 1e-10:
                ax0.plot(
                    theta,
                    real[imode + 1],
                    typeline[1 + imode],
                    c=color,
                    lw=lw,
                    markersize=markersize,
                    label=label + f" (mode {imode + 2})",
                )
            if np.abs(np.sum(imag[imode + 1])) > 1e-10:
                ax1.plot(
                    theta,
                    imag[imode + 1],
                    typeline[1 + imode],
                    c=color,
                    lw=lw,
                    markersize=markersize,
                    label=label + f" (mode {imode + 2})",
                )

    try:
        maxx = np.nanmax([np.nanmax(real), np.nanmax(imag)])
        minn = np.nanmin([np.nanmin(real), np.nanmin(imag)])
    except:
        maxx = -np.inf
        minn = np.inf

    return maxx, minn


def plot_spec(
    ax,
    ky,
    quant,
    markers="s",
    coeff=0,
    c="b",
    ls="-",
    lw=1,
    label="",
    alpha=None,
    markersize=20,
    facecolor="b",
):
    if type(facecolor) in [str, tuple]:
        facecolor = [facecolor] * len(ky)

    if alpha == 1.0:
        alpha = None

    if type(markers) != list:
        m = copy.deepcopy(markers)
        markers = []
        for i in range(len(ky)):
            markers.append(m)

    ax.plot(ky, quant / ky**coeff, c=c, ls=ls, lw=lw, label=label, alpha=alpha)
    for i in range(len(ky)):
        ax.scatter(
            [ky[i]],
            [quant[i] / ky[i] ** coeff],
            s=markersize,
            facecolors=facecolor[i],
            marker=markers[i],
            alpha=alpha,
        )

    ax.set_xlabel("$k_{\\theta}\\rho_s$")
    ax.set_xscale("log")


def plotTGLFfluctuations(
    ax,
    kys,
    fluct,
    markerType="o",
    c="r",
    ls="-",
    lw=1,
    label="",
    markersize=20,
    alpha=1.0,
    title="",
    ylabel="",
):
    plot_spec(
        ax,
        kys,
        fluct,
        markers=markerType,
        c=c,
        lw=lw,
        ls=ls,
        label=label,
        alpha=alpha,
        markersize=markersize,
        facecolor=c,
    )

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    GRAPHICStools.addDenseAxis(ax)

    return ax
