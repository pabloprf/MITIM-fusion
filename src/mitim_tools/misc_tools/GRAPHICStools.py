import socket
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from IPython import embed
from mitim_tools.misc_tools import IOtools


def aroundZeroLims(zlims):
    z = np.max(np.abs(zlims))
    zlims = [-z, z]

    return zlims


def convert_to_hex_soft(color):
    """
    Color can be an integer, in that case it'll grab that position in the color_hex dictionary
    """

    # chatGPT created this dictionary to convert to hex colors
    color_hex = OrderedDict(
        {
            "b": "#CCCCFF",  # very light blue
            "m": "#FFCCFF",  # very light magenta
            "r": "#FFCCCC",  # very light red
            "t": "#99CCCC",  # very light teal
            "y": "#FFFFCC",  # very light yellow
            "g": "#CCFFCC",  # very light green instead of green
            "c": "#CCFFFF",  # very light cyan instead of cyan
            "n": "#666699",  # lighter navy
            "o": "#CCCC99",  # very light olive instead of olive
            "p": "#FFCCFF",  # very light purple instead of purple
            "s": "#E0E0E0",  # lighter silver
            "a": "#FFA07A",  # light salmon
            "h": "#ADFF2F",  # green yellow
            "i": "#FFB6C1",  # light pink
            "j": "#FFD700",  # gold
            "q": "#20B2AA",  # light sea green
            "u": "#87CEFA",  # light sky blue
            "v": "#778899",  # light slate gray
            "x": "#B0C4DE",  # light steel blue
            "z": "#F08080",  # light coral
            "k": "#666666",  # medium gray instead of black
            "w": "#FAFAFA",  # very light gray instead of white
        }
    )

    if isinstance(color, int):
        # If color is an integer, get the color at that index
        color_keys = list(color_hex.keys())
        if color < len(color_keys):
            return color_hex[color_keys[color]]
        else:
            return None
    elif (color is not None) and (color in color_hex):
        return color_hex[color]
    else:
        return None


def plotRange(
    t,
    x,
    z,
    ax=None,
    it1=0,
    it2=-1,
    howmany=None,
    itBig=[],
    lw=0.2,
    colors=["r", "b"],
    colorsBig=["r", "b"],
    legend=True,
    alpha=1.0,
):
    if ax is None:
        fig, ax = plt.subplots()

    if howmany is None:
        howmany = np.argmin(np.abs(t - t[it2])) - np.argmin(np.abs(t - t[it1]))

    cols, _ = colorTableFade(howmany, startcolor=colors[0], endcolor=colors[1])
    tarr = [int(i) for i in np.linspace(it1, it2, howmany)]
    for i, k in enumerate(tarr):
        if (i == 0 or i == len(tarr) - 1) and legend:
            label = f"$t={t[k]:.3f}s$"
        else:
            label = ""
        ax.plot(x[k], z[k], lw=lw, label=label, c=cols[i], alpha=alpha)

    for i, k in enumerate(itBig):
        label = f"$t={t[k]:.3f}s$" if legend else ""
        ax.plot(x[k], z[k], lw=3, label=label, c=colorsBig[i], alpha=alpha)

    return cols


def ylabel_formatter10(x, y):
    return f"${x:.0e}".replace("e", "\\cdot 10^{") + "}$"


def addScientificY(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)


def plotMultiVariate(
    y,
    axs=None,
    marker="s",
    markersize=4,
    color="b",
    label="",
    axislabels=None,
    bounds=None,
    boundsThis=None,
    alpha=1.0,
    labelsize=8,
):
    if not IOtools.islistarray(y):
        y = np.array([y])
    y = np.atleast_2d(y)

    numOFs = y.shape[1]
    num_plots = int(np.ceil(numOFs / 2))

    if axs is None:
        if num_plots > 1:
            axs = producePlotsGrid(num_plots)
        else:
            fig, axs = plt.subplots()
            axs = [axs]

    for i in range(num_plots):
        ax = axs[i]
        try:
            ax.plot(
                y[:, i * 2],
                y[:, i * 2 + 1],
                marker,
                markersize=markersize,
                color=color,
                label=label,
                alpha=alpha,
            )
            if axislabels is not None:
                ax.set_xlabel(axislabels[i * 2], fontsize=labelsize)
                ax.set_ylabel(axislabels[i * 2 + 1], fontsize=labelsize)
            if bounds is not None:
                ax.axvline(x=bounds[i * 2][0], ls="--", lw=0.5, c="g")
                ax.axvline(x=bounds[i * 2][1], ls="--", lw=0.5, c="g")
                ax.axhline(y=bounds[i * 2 + 1][0], ls="--", lw=0.5, c="g")
                ax.axhline(y=bounds[i * 2 + 1][1], ls="--", lw=0.5, c="g")
            if boundsThis is not None:
                ax.axvline(x=boundsThis[i * 2, 0], ls="--", lw=0.5, c="orange")
                ax.axvline(x=boundsThis[i * 2, 1], ls="--", lw=0.5, c="orange")
                ax.axhline(y=boundsThis[i * 2 + 1, 0], ls="--", lw=0.5, c="orange")
                ax.axhline(y=boundsThis[i * 2 + 1, 1], ls="--", lw=0.5, c="orange")
        except:
            ax.plot(
                y[:, i * 2],
                y[:, i * 2],
                marker,
                markersize=markersize,
                color=color,
                label=label,
                alpha=alpha,
            )
            if axislabels is not None:
                ax.set_xlabel(axislabels[i * 2], fontsize=labelsize)
                ax.set_ylabel(axislabels[i * 2], fontsize=labelsize)
            if bounds is not None:
                ax.axvline(x=bounds[i * 2][0], ls="--", lw=0.5, c=color)
                ax.axvline(x=bounds[i * 2][1], ls="--", lw=0.5, c=color)
                ax.axhline(y=bounds[i * 2][0], ls="--", lw=0.5, c=color)
                ax.axhline(y=bounds[i * 2][1], ls="--", lw=0.5, c=color)
            if boundsThis is not None:
                ax.axvline(x=boundsThis[i * 2, 0], ls="--", lw=0.5, c="orange")
                ax.axvline(x=boundsThis[i * 2, 1], ls="--", lw=0.5, c="orange")
                ax.axhline(y=boundsThis[i * 2, 0], ls="--", lw=0.5, c="orange")
                ax.axhline(y=boundsThis[i * 2, 1], ls="--", lw=0.5, c="orange")


def plotViolin(
    data, labels=None, ax=None, colors=None, plotQuartiles=False, vertical=False
):
    """
    data is (batch,num_points)
    """

    if ax is None:
        fig, ax = plt.subplots()

    labe_x = np.arange(1, len(labels) + 1)

    if labels is None:
        labels = labe_x

    if colors is None:
        colors = ["b"] * len(data)

    parts = ax.violinplot(
        data,
        labe_x,
        points=100,
        widths=1 / len(data),
        showextrema=False,
        showmeans=False,
        vert=vertical,
    )  # ,color=color)

    if vertical:
        for i in range(len(data)):
            ax.plot(
                [labe_x[i]] * len(data[i]),
                data[i],
                "o",
                markersize=1,
                color=colors[i],
                alpha=0.2,
            )
    else:
        for i in range(len(data)):
            ax.plot(
                data[i],
                [labe_x[i]] * len(data[i]),
                "o",
                markersize=1,
                color=colors[i],
                alpha=0.2,
            )

    # From https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(colors[i])
        pc.set_alpha(0.3)

    if plotQuartiles:

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)

        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array(
            [
                adjacent_values(sorted_array, q1, q3)
                for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
            ]
        )
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        ax.scatter(labe_x, medians, marker="o", color="white", s=30, zorder=3)
        ax.vlines(labe_x, quartile1, quartile3, color="k", linestyle="-", lw=5)
        ax.vlines(labe_x, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)

    # ------

    if vertical:
        for i in range(len(data)):
            ax.plot([labe_x[i]], [np.mean(data[i])], "o", c=colors[i], markersize=3)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    else:
        for i in range(len(data)):
            ax.plot([np.mean(data[i])], [labe_x[i]], "o", c=colors[i], markersize=3)
        ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels)


def addMinor(ax, length=4, n=None):
    ax.minorticks_on()
    # Which ones to show?
    if ax.xaxis.get_scale() != "log":
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=n))
    # Thick
    ax.tick_params(which="minor", length=length)


def adjustYlimToYtick(ax):
    ax.set_ylim([ax.get_yticks()[0], ax.get_yticks()[-1]])


def ensurefloatsticks(ax):
    ticks = ax.get_yticks().astype(np.float32)
    tickLabels = map(str, ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tickLabels)

    ticks = ax.get_xticks().astype(np.float32)
    tickLabels = map(str, ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickLabels)


def addDenseAxis(ax, grid=True, n=10, axTwinx=None, ensure_floats=False):
    if ensure_floats:
        ensurefloatsticks(ax)

    addMinor(ax, n=n)
    ax.tick_params(
        which="major", length=5, bottom=True, top=True, left=True, right=True
    )
    ax.tick_params(
        which="minor", length=3, bottom=True, top=True, left=True, right=True
    )

    if grid:
        ax.grid(axis="x", color="k", alpha=1.0, linewidth=0.2, linestyle="-")
        ax.grid(axis="y", color="k", alpha=1.0, linewidth=0.2, linestyle="-")

    if axTwinx is not None:
        adjustYlimToYtick(ax)

        minY = axTwinx.get_yticks()[0]
        maxY = axTwinx.get_yticks()[-1]
        numticks = len(ax.get_yticks())

        newticks = np.linspace(minY, maxY, numticks)

        axTwinx.set_yticks(newticks)

        adjustYlimToYtick(axTwinx)

        if ensure_floats:
            ensurefloatsticks(axTwinx)


def output_figure_papers(name, fig=None, dpi=445):
    if name is not None:
        if fig is None:
            plt.savefig(name + ".svg", transparent=True)
            plt.savefig(name + ".jpeg", transparent=True, dpi=dpi)
            plt.savefig(name + ".png", transparent=True, dpi=dpi)
            plt.savefig(name + "_white.png", transparent=False, dpi=dpi)
        else:
            fig.savefig(name + ".svg", transparent=True)
            fig.savefig(name + ".jpeg", transparent=True, dpi=dpi)
            fig.savefig(name + ".png", transparent=True, dpi=dpi)
            fig.savefig(name + "_white.png", transparent=False, dpi=dpi)



def prep_figure_papers(size=15, slower_but_latex=False):
    plt.rc("font", family="serif", serif="Times", size=size)
    plt.rc("xtick.minor", size=size)
    plt.rc("legend", fontsize=size)  # *0.8)
    if slower_but_latex:
        plt.rc("text", usetex=True)

    # Had to a
    # plt.rcParams['axes.linewidth'] = 0.2
    # plt.rcParams['font.size'] = 4
    # plt.rcParams['legend.fontsize'] = 5
    # plt.rcParams['xtick.major.size'] = 2
    # plt.rcParams['xtick.major.width'] = 0.2
    # plt.rcParams['ytick.major.size'] = 2
    # plt.rcParams['ytick.major.width'] = 0.2


def makePlotInvisible(ax):
    ax.axis("off")


def addSubplotLetter(
    ax,
    txt="a)",
    position=[0.04, 0.94],
    fs=15,
    c="k",
    relative=True,
    fc_box="w",
    alpha_box=1.0,
    va="center",
    ha="center",
):
    if relative:
        ax.text(
            position[0],
            position[1],
            txt,
            color=c,
            weight="regular",
            bbox=dict(facecolor=fc_box, alpha=alpha_box, boxstyle="round"),
            fontsize=fs,
            horizontalalignment=ha,
            verticalalignment=va,
            transform=ax.transAxes,
        )
    else:
        ax.text(
            position[0],
            position[1],
            txt,
            color=c,
            weight="regular",
            bbox=dict(facecolor=fc_box, alpha=alpha_box, boxstyle="round"),
            fontsize=fs,
            horizontalalignment=ha,
            verticalalignment=va,
        )


def drawLineWithTxt(
    ax,
    pos,
    label="",
    orientation="vertical",
    color="k",
    lw=1,
    ls="--",
    alpha=1.0,
    fontsize=10,
    fromtop=0.85,
    fontweight="normal",
    verticalalignment="bottom",
    horizontalalignment="left",
    separation=0,
    box=False,
    explicit_rotation=None,
):
    bbox = dict(facecolor="white", alpha=0.5) if box else None  # dict()

    if orientation == "vertical":
        ax.axvline(x=pos, c=color, lw=lw, alpha=alpha, ls=ls)
        ax.text(
            pos + separation,
            fromtop * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0],
            label,
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=90 if explicit_rotation is None else explicit_rotation,
            bbox=bbox,
        )

    elif orientation == "horizontal":
        ax.axhline(y=pos, c=color, lw=lw, alpha=alpha, ls=ls)
        ax.text(
            fromtop * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0],
            pos + separation,
            label,
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=0 if explicit_rotation is None else explicit_rotation,
        )


def drawSpaningWithTxt(
    ax,
    posU,
    posD,
    labelU="",
    labelD="",
    orientation="horizontal",
    lw=1,
    ls="--",
    color="k",
    alpha=1.0,
    fontsize=10,
    fromtop=0.85,
    centralLine=False,
    extra=0,
):
    if orientation == "horizontal":
        ax.axhspan(posD, posU, alpha=alpha, color=color)
        ax.text(
            fromtop * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0],
            posU + extra,
            labelU,
            color=color,
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.text(
            fromtop * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0],
            posD - extra,
            labelD,
            color=color,
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="center",
        )

        if centralLine:
            ax.axhline(y=np.mean([posU, posD]), c=color, ls=ls, lw=lw)


def producePlotsGrid(
    num_plots, fig=None, hspace=0.6, wspace=0.6, sharex=False, sharey=False
):
    if fig is None:
        fig = plt.figure()

    s = int(np.ceil(np.sqrt(num_plots)))

    sc = int(s)
    sr = int(s)

    # Case in which last raw is completely empty
    if sc * (sr - 1) == num_plots:
        sr -= 1
    # ---

    grid = plt.GridSpec(nrows=sr, ncols=sc, hspace=hspace, wspace=wspace)
    axs = []
    cont = 0
    for i in range(sr):
        for j in range(sc):
            cont += 1
            if cont <= num_plots:
                if cont == 1 or (not sharex and not sharey):
                    thisfig = fig.add_subplot(grid[i, j])
                else:
                    if sharex and sharey:
                        thisfig = fig.add_subplot(
                            grid[i, j], sharex=axs[0], sharey=axs[0]
                        )
                    elif sharex:
                        thisfig = fig.add_subplot(grid[i, j], sharex=axs[0])
                    elif sharey:
                        thisfig = fig.add_subplot(grid[i, j], sharey=axs[0])
                axs.append(thisfig)

    return axs


def plotMatrix(
    M,
    ax=None,
    xlabels=None,
    ylabels=None,
    expo=False,
    title="",
    fontsize=None,
    fontcolor="white",
    cmap="viridis",
    fontsize_title=15,
    symmetrice=False,
    rangeText=1e-10,
):
    M = M.float()

    if fontsize is None:
        if M.shape[0] < 4 and M.shape[1] < 4:
            fontsize = 12
        else:
            fontsize = 10

    if ax is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 9))

    Min = M.min()
    Max = M.max()
    if symmetrice:
        Mm = np.max([np.abs(Min), np.abs(Max)])
        vmin = -Mm
        vmax = Mm
    else:
        vmin = Min
        vmax = Max

    ax.matshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    font = {
        "family": "serif",
        "color": fontcolor,
        "weight": "bold",
        "size": fontsize,
    }

    for (i, j), z in np.ndenumerate(M):
        if np.abs(z) > rangeText:
            if expo:
                ax.text(j, i, f"{z:0.1e}", ha="center", va="center", fontdict=font)
            else:
                ax.text(j, i, f"{z:0.2f}", ha="center", va="center", fontdict=font)

    if xlabels is not None:
        ax.set_xticks(np.arange(0, M.shape[1]))
        ax.set_xticklabels(xlabels, rotation=90)
    if ylabels is not None:
        ax.set_yticks(np.arange(0, M.shape[0]))
        ax.set_yticklabels(ylabels)

    ax.set_title(title, fontsize=fontsize_title)


def addSecondaryXaxis(ax1, funOperation, new_tick_locations=None, label=""):
    ax1.set_xlim([ax1.get_xticks()[0], ax1.get_xticks()[-1]])

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    if new_tick_locations is None:
        new_tick_locations = ax1.get_xticks()

    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(funOperation(new_tick_locations))
    ax2.set_xlabel(label)

    return ax2


def fillGraph(
    ax,
    x,
    y,
    y_down=None,
    y_up=None,
    alpha=0.3,
    color="r",
    label="",
    lw=0,
    islwOnlyMean=False,
    ms=None,
    ls="-",
):
    if y_up is not None:
        l = ax.fill_between(x, y, y_up, facecolor=color, alpha=alpha, label=label)
    if y_down is not None:
        l = ax.fill_between(
            x,
            y_down,
            y,
            facecolor=color,
            alpha=alpha,
            label=label if y_up is None else "",
        )
    if y_up is None and y_down is None:
        l = ax.fill_between(x, y * 0, y, facecolor=color, alpha=alpha, label=label)

    if lw > 0:
        ax.plot(x, y, ls, c=color, lw=lw, markersize=ms)
        if not islwOnlyMean:
            if y_up is not None:
                ax.plot(x, y_up, ls, c=color, lw=lw, markersize=ms)
            if y_down is not None:
                ax.plot(x, y_down, ls, c=color, lw=lw, markersize=ms)

    return l


def listColors():
    col = [
        "b",
        "r",
        "m",
        "orange",
        "c",
        "k",
        "g",
        "y",
        "chocolate",
        "olive",
        "fuchsia",
        "slategrey",
        "tomato",
        "peru",
        "khaki",
        "papayawhip",
        "saddlebrown",
        "powderblue",
        "dimgrey",
        "indianred",
    ]
    for i in range(5):
        col.extend(col)

    return col


def listLS():
    ls = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]

    return ls


def listmarkers():
    return [
        "o",
        "s",
        "^",
        "v",
        "+",
        "<",
        ">",
        "o",
        "s",
        "^",
        "v",
        "+",
        "<",
        ">",
        "o",
        "s",
        "^",
        "v",
        "+",
        "<",
        ">",
    ]


def listmarkersLS():
    ls = []
    for j in listmarkers():
        for i in listLS():
            ls.append(f"{i}{j}")

    return ls


def addLegendApart(
    ax,
    elements=None,
    ratio=0.7,
    withleg=True,
    extraPad=0,
    size=None,
    loc="upper left",
    title=None,
):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * ratio, box.height])
    prop = {}
    if size is not None:
        prop["size"] = size
    if withleg:
        if elements is None:
            ax.legend(
                loc=loc, bbox_to_anchor=(1 + extraPad, 1.0), prop=prop, title=title
            )
        else:
            ax.legend(
                elements,
                loc=loc,
                bbox_to_anchor=(1 + extraPad, 1.0),
                prop=prop,
                title=title,
            )


def addColorbarSubplot(
    ax,
    cs,
    fig=None,
    barfmt="%3.1f",
    title="",
    fontsize=10,
    fontsizeTitle=None,
    ylabel="",
    ticks=[],
    ticklabels=None,
    orientation="horizontal",
    drawedges=False,
    force_position=None,
    padCB="10%",
    sizeC="3%",
):
    """
    Note that the figure that I apply this to has to be the last one open. Otherwise the axes are messed up.
    To solve this, pass the figure as kwarg
    """

    # cs is contour

    if fontsizeTitle is None:
        fontsizeTitle = fontsize

    divider = make_axes_locatable(ax)

    if orientation == "horizontal":
        cax = divider.append_axes("top", size=sizeC, pad=padCB)
    elif orientation == "vertical":
        cax = divider.append_axes("right", size=sizeC, pad=padCB)
    elif orientation == "full":
        cax = ax
        orientation = "horizontal"
    else:
        cax = divider.append_axes(orientation, size=sizeC, pad=padCB)
        if orientation in ["top", "bottom"]:
            orientation = "horizontal"
        else:
            orientation = "vertical"

    if fig is None:
        cbar = plt.colorbar(
            cs,
            format=barfmt,
            cax=cax,
            ax=ax,
            orientation=orientation,
            drawedges=drawedges,
        )  # ,boundaries=[0,20])
    else:
        cbar = fig.colorbar(
            cs,
            format=barfmt,
            cax=cax,
            ax=ax,
            orientation=orientation,
            drawedges=drawedges,
        )  # ,boundaries=[0,20])
    cbar.ax.set_ylabel(ylabel)
    cbar.ax.set_title(title, fontsize=fontsizeTitle)

    cbar.ax.tick_params(labelsize=fontsize)

    if len(ticks) == 0:
        ticks = cbar.ax.get_xticks()

    if len(ticks) > 0:
        cbar.set_ticks(ticks)

    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)

    return cbar


def addXaxis(ax, old_tick_locations, new_tick_locations, label="", whichticks=None):
    """
    old is the position of the old axis to map the new ones
    """

    def tick_function(x):
        return np.interp(x, old_tick_locations, new_tick_locations)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    if whichticks is None:
        whichticks = old_tick_locations
    ax2.set_xticks(whichticks)
    ax2.set_xticklabels(tick_function(whichticks))

    ax2.set_xlabel(label)

    return ax2


def add_subplot_axes(ax, rect, axisbg="w"):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], facecolor=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.5
    # y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def set_relative_position(ax, parent_ax, rel_coords):
    """
    Adjust the position of ax using coordinates relative to parent_ax.

    Created by ChatGPT-4 on 2023-10-03

    Parameters:
    - ax: The axis to be repositioned.
    - parent_ax: The parent axis.
    - rel_coords: A list of relative coordinates [rel_left, rel_bottom, rel_right, rel_top].
    """
    parent_pos = parent_ax.get_position()
    abs_coords = [
        parent_pos.x0 + rel_coords[0] * parent_pos.width,
        parent_pos.y0 + rel_coords[1] * parent_pos.height,
        (rel_coords[2] - rel_coords[0]) * parent_pos.width,
        (rel_coords[3] - rel_coords[1]) * parent_pos.height,
    ]
    ax.set_position(abs_coords)


def plotTimeFade(ax, t, x, z, color1="r", color2="b", alpha=1.0):
    cols, _ = colorTableFade(
        len(t), startcolor=color1, endcolor=color2, alphalims=[alpha, alpha]
    )

    for i in range(len(t)):
        ax.plot(x, z[i], c=cols[i])


def gradientSPAN(
    ax,
    x1,
    x2,
    color="b",
    color2=None,
    startingalpha=1.0,
    endingalpha=0.0,
    orientation="vertical",
    label="",
):
    cols, _ = colorTableFade(
        100,
        startcolor=color,
        endcolor=color if color2 is None else color2,
        alphalims=[startingalpha, endingalpha],
    )

    x = np.linspace(x1, x2, 100)

    if (startingalpha == endingalpha) and (color2 is None):
        if orientation == "vertical":
            ax.axvspan(
                x[0],
                x[-1],
                facecolor=color,
                alpha=startingalpha,
                edgecolor="none",
                label=label,
            )
        elif orientation == "horizontal":
            ax.axhspan(
                x[0],
                x[-1],
                facecolor=color,
                alpha=startingalpha,
                edgecolor="none",
                label=label,
            )
    else:
        for i in range(len(x) - 1):
            fraction = i / len(x)
            pos = x[i]
            posnext = x[i + 1]

            if orientation == "vertical":
                ax.axvspan(
                    pos, posnext, facecolor=cols[i], edgecolor="none", label=label
                )
            elif orientation == "horizontal":
                ax.axhspan(
                    pos, posnext, facecolor=cols[i], edgecolor="none", label=label
                )


def autoscale_y(ax, margin=0.01, b=None, bottomy=None):
    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        try:
            y_displayed = yd[((xd > lo) & (xd < hi))]
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed) - margin * h
            top = np.max(y_displayed) + margin * h
        except:
            bot, top = -np.inf, np.inf

        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    if b is not None:
        bot = b

    try:
        if (
            top != bot
            and (ax.get_yscale() != "log" or bot > 0)
            and (not np.isinf(top))
            and (not np.isinf(bot))
        ):
            ax.set_ylim(bot, top)
        if (
            bottomy is not None
            and (ax.get_yscale() != "log" or bottomy > 0)
            and (not np.isinf(top))
            and (not np.isinf(bot))
        ):
            if bottomy != ax.get_ylim()[1]:
                ax.set_ylim(bottom=bottomy)
    except:
        print("could not adjust y-axis - PRF")


def colorTableFade(num, startcolor="b", endcolor="r", alphalims=[1.0, 1.0]):
    if startcolor is not None:
        if endcolor is None:
            endcolor = startcolor
        cm1 = mcol.LinearSegmentedColormap.from_list(
            "MyCmapName", [endcolor, startcolor]
        )
    else:
        cm1 = "gist_rainbow"

    cnorm = mcol.Normalize(vmin=0, vmax=num, clip=True)
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    cols = []
    for i in range(num):
        cols.append(cpick.to_rgba(num - i))

    alphas = np.linspace(alphalims[0], alphalims[1], num)

    cn = [(col[0], col[1], col[2], alpha) for col, alpha in zip(cols, alphas)]

    return cn, cpick


def createAnimation(
    fig, FunctionToAnimate, framesCalc, FramesPerSecond, BITrate, MovieFile, DPIs
):
    plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"
    if "mfews" in socket.gethostname():
        plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
    ani = animation.FuncAnimation(
        fig, FunctionToAnimate, frames=framesCalc, repeat=True
    )

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=FramesPerSecond, metadata=dict(artist="PRF"), bitrate=BITrate)
    ani.save(writer=writer, filename=MovieFile, dpi=DPIs)


def animageFunction(
    plottingFunction,
    axs,
    fig,
    MovieFile,
    HowManyFrames,
    framePS=50,
    BITrate=1200,
    DPIs=150,
):
    if type(axs) not in [np.ndarray, list]:
        axs = [axs]

    def animate(i):
        for j in range(len(axs)):
            axs[j].clear()
        plottingFunction(axs, i)
        print(f"\t~~ Frame {i + 1}/{HowManyFrames}")

    print(" --> Creating animation")
    createAnimation(fig, animate, HowManyFrames, framePS, BITrate, MovieFile, DPIs)


def reduceVariable(var, howmanytimes, t=None, trange=[0, 100]):
    if t is not None:
        var = var[np.argmin(np.abs(t - trange[0])) : np.argmin(np.abs(t - trange[1]))]

    howmanytimes = np.min([howmanytimes, var.shape[0]])
    positionsPlot = np.arange(
        0, var.shape[0], int(round(var.shape[0] / (howmanytimes - 1)))
    )

    return var[positionsPlot]


# chatgpt
def drawArrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    txt="",
    alpha=1.0,
    linewidth=1.0,
    colorArrow="k",
    arrowstyle="->",
):
    """
    Draw an arrow on the given ax from (x1, y1) to (x2, y2), with specified transparency and line width.

    Parameters:
    - ax: Axes object to draw the arrow on.
    - x1, y1: Starting coordinates of the arrow.
    - x2, y2: Ending coordinates of the arrow.
    - txt: Text label for the arrow.
    - alpha: Transparency of the arrow and text. Default is 1.0 (opaque).
    - linewidth: Line width of the arrow. Default is 1.0.
    """
    ax.annotate(
        txt,
        xy=(x2, y2),
        xycoords="data",
        xytext=(x1, y1),
        textcoords="data",
        arrowprops=dict(
            arrowstyle=arrowstyle,
            connectionstyle="arc3",
            linewidth=linewidth,
            color=colorArrow,
        ),
        alpha=alpha,
    )


def drawArrowAcrossMosaic(
    ax, fig, start_ax, end_ax, start_pos, end_pos, txt="", alpha=1.0, linewidth=1.0
):
    """
    Draw an arrow in a figure with subplot mosaic layout, spanning across multiple subplots.

    Parameters:
    - ax: An Axes object from the mosaic on which to draw the arrow.
    - fig: Figure object containing the subplots.
    - start_ax, end_ax: Axes objects where the arrow starts and ends.
    - start_pos, end_pos: Starting and ending positions (x, y) within the start and end axes.
    - txt: Text label for the arrow.
    - alpha: Transparency of the arrow. Default is 1.0 (opaque).
    - linewidth: Line width of the arrow. Default is 1.0.
    """
    # Transform start and end positions to figure coordinates
    transFigure = fig.transFigure.inverted()
    coord1 = transFigure.transform(start_ax.transData.transform(start_pos))
    coord2 = transFigure.transform(end_ax.transData.transform(end_pos))

    # Draw the arrow using one of the axes
    ax.annotate(
        txt,
        xy=coord2,
        xycoords="figure fraction",
        xytext=coord1,
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=linewidth),
        alpha=alpha,
    )


def diagram_plotModule(
    ax,
    lab,
    positionText,
    c="k",
    typeBox="round",
    noLab=False,
    arrowsOut=[None],
    carrowsOut=None,
    arrowsIn={},
    carrowsIn=None,
):
    cont = 0

    for alab in arrowsOut:
        if cont < 1 and not noLab:
            alphaC = 1.0
        else:
            alphaC = 0.0

        if carrowsOut is None:
            ca = c
        else:
            ca = carrowsOut[cont]

        x0 = positionText[0]
        y0 = positionText[1]

        x0arrow = arrowsOut[alab][0]
        y0arrow = arrowsOut[alab][1]

        if x0arrow > x0:
            relx = 1.0
        elif x0arrow < x0:
            relx = 0.0
        else:
            relx = 0.5

        if y0arrow > y0:
            rely = 1.0
        elif y0arrow < y0:
            rely = 0.0
        else:
            rely = 0.5

        # relx = 1.0
        # rely = 0.5
        ann = ax.annotate(
            lab,
            xy=(x0arrow, y0arrow),
            xytext=(x0, y0),
            color="k",
            alpha=alphaC,
            size=15,
            va="center",
            ha="center",
            bbox=dict(boxstyle=typeBox + ",pad=1", fc=c, alpha=0.1),
            arrowprops=dict(arrowstyle="simple", relpos=(relx, rely), fc=ca, ec=ca),
        )

        # Calculate where to put label of arrow
        x1 = 0.5 * (positionText[0] + arrowsOut[alab][0])
        y1 = 0.5 * (positionText[1] + arrowsOut[alab][1]) + 0.03

        if arrowsOut[alab] is not None:
            ax.annotate(
                alab,
                xy=(x1, y1),
                color=ca,
                fontweight="bold",
                size=12,
                va="center",
                ha="center",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=ca, alpha=1.0),
            )
        cont += 1

    # In arrrows are recursive
    if carrowsIn is None:
        ca = c
    else:
        ca = carrowsIn[cont]
    cont = 0
    for alab in arrowsIn:
        x2 = arrowsIn[alab][0]
        y2 = arrowsIn[alab][1]
        arrowsIn[alab][0] = x0 - 0.03
        arrowsIn[alab][1] = y0
        diagram_plotModule(
            ax, "", [x2, y2], noLab=True, arrowsOut=arrowsIn, carrowsOut=ca
        )
        cont += 1


def compareExpSim(
    t_exp,
    z_exp,
    t_sim,
    z_sim,
    x_exp=None,
    x_sim=None,
    ax=None,
    z_exp_err=None,
    lab_exp="",
    lab_sim="",
    title="",
    ylabel="",
    plotError=False,
    lw=3,
    alphaE=1.0,
    xrrange=None,
    yrrange=None,
    inRhoPol=None,
):
    colorsE = ["b", "c", "y", "m"]
    colorsS = ["r", "g", "k", "orange"]

    # 1D
    if x_exp is None:
        if ax is None:
            fig, ax = plt.subplots()

        for j in range(len(z_sim)):
            ax.plot(
                t_sim,
                z_sim[j],
                lw=lw,
                label=f"Sim {lab_sim[j]}",
                c=colorsS[0 + j],
            )

        for i in range(len(z_exp)):
            try:
                ax.plot(
                    t_exp,
                    z_exp[i],
                    lw=lw,
                    label=f"Exp {lab_exp[i]}",
                    c=colorsE[1 + j + i],
                )
            except:
                embed()

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")

        if len(lab_exp) > 0:
            ax.legend().set_draggable(True)

        if plotError:
            z_sim1 = np.interp(t_exp, t_sim, z_sim)
            error = np.abs(z_sim1 - z_exp) / z_exp * 100.0

            ax1 = ax.twinx()
            ax1.plot(t_exp, error, ls="--", c="y")
            ax1.set_ylim([0, 100])
            ax1.set_ylabel("Difference (%)")

    # 2D
    else:
        if ax is None:
            fig, ax = plt.subplots()

        for j in range(len(z_sim)):
            ax.plot(x_sim, z_sim[j], lw=lw, c=colorsS[0 + j])

        for i in range(len(z_exp)):
            if inRhoPol is not None:
                xx = np.interp(x_exp[i], inRhoPol["rho_pol"], inRhoPol["rho_tor"])
            else:
                xx = x_exp[i]

            if z_exp_err is None:
                ax.plot(
                    xx,
                    z_exp[i],
                    lw=3,
                    label=f"Exp {lab_exp[i]}",
                    c=colorsE[1 + i + j],
                    alpha=alphaE,
                )
            else:
                try:
                    ax.errorbar(
                        xx,
                        z_exp[i],
                        c=colorsE[1 + i + j],
                        yerr=z_exp_err[i],
                        capsize=5.0,
                        fmt="none",
                        alpha=alphaE,
                        label=f"Exp {lab_exp[i]}",
                    )
                    ax.scatter(xx, z_exp[i], 5, c="k")
                except:
                    embed()

        for j in range(len(z_sim)):
            ax.plot(
                x_sim,
                z_sim[j],
                lw=lw,
                label=f"Sim {lab_sim[j]}",
                c=colorsS[0 + j],
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("$\\rho_N$")

        if len(lab_exp) > 0:
            ax.legend().set_draggable(True)

        ax.set_ylim(bottom=0)

    if xrrange is not None:
        ax.set_xlim(xrrange)
    if yrrange is not None:
        if isinstance(yrrange, bool):
            ax.set_ylim(bottom=0)
        else:
            ax.set_xlim(yrrange)


def plotLScontour(
    fig,
    ax,
    x,
    y,
    z,
    zF=1,
    xlabel="",
    ylabel="$k_\\theta\\rho_s$",
    zlabel="$\\gamma$ ($c_s/a$)",
    division=0,
    xlims=None,
    ylims=None,
    logy=True,
    zlims=None,
    size=15,
):
    from mitim_tools.misc_tools import MATHtools

    # Divide if needed
    if division > 0:
        z = MATHtools.divideZero(z, y**division) * np.sign(zF)
    # if division>0:z 		= z / y**division * np.sign(zF)

    if xlims is not None:
        i1 = np.argmin(np.abs(x[:, 0] - xlims[0]))
        i2 = np.argmin(np.abs(x[:, 0] - xlims[1]))
    else:
        i1 = 0
        i2 = -1

    # ~~~~~~~ Plot
    if zlims is None:
        am = np.amax(np.abs(z[i1:i2, :]))
        ming = -am
        maxg = am
    else:
        ming = zlims[0]
        maxg = zlims[1]
    levels = np.linspace(ming, maxg, 100)
    colticks = np.linspace(ming, maxg, 5)

    ff = ax.contourf(x, y, z, levels=levels, cmap="seismic", extend="both")
    clb = fig.colorbar(ff, ticks=colticks, orientation="vertical")  # ,extend='both')

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, size=size)
    ax.set_ylabel(ylabel, size=size)

    ax.set_ylim([5e-2, 3e1])

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if division == 0:
        clb.set_label(zlabel, size=size)
    elif division == 1:
        clb.set_label(zlabel + " /$\\mathrm{k_\\theta\\rho_s}$", size=size)
    elif division == 2:
        clb.set_label(zlabel + " /$\\mathrm{k^2_\\theta\\rho^2_s}$", size=size)
