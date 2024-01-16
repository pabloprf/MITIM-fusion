import os
import copy
import torch
import sys
import dill as pickle_dill
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from collections import OrderedDict
from IPython import embed

from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools, MATHtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.aux import TESTtools

from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()


# ----------------------------------------------------------------------------------------------------
# Tools not to clutter SURROGATEtools.py
# ----------------------------------------------------------------------------------------------------


def plot_surrogate_model(
    self,
    axs=None,
    x_next=None,
    x_best=None,
    y_best=None,
    yVar_best=None,
    extralab="",
    plotFundamental=True,
    stds=2,
    nSamples1Dmodels=10,
    pointsEvaluate=50,
    labelEvaluated="",
):
    """
    Notes:
            - PlotFundamental = True plots exactly what the surrogate was trained on (e.g. a/LTe, nuei to fit QiGB)
                                                    False plots optimization variables, so that means in untransformed space (e.g. a/LTe_1, aLTe_2 to fit Qi)
    """

    """
	------------------------------------------------------------------------------------------------------------
	Prepare Figures
	------------------------------------------------------------------------------------------------------------
	"""

    # Input dimensions determine the type of plot. If it's fundamental, I have to account for transformed input dimensions instead
    dimX = self.gpmodel.ard_num_dims if plotFundamental else self.train_X.shape[-1]

    if axs is None:
        plt.ion()
        fig = plt.figure(figsize=(6, 9))

        if dimX == 1:
            grid = plt.GridSpec(nrows=5, ncols=1, hspace=0.6, wspace=0.4)
            ax = fig.add_subplot(grid[:4])
            ax2 = None
            axL = fig.add_subplot(grid[-1])
        elif dimX == 2:
            grid = plt.GridSpec(nrows=3, ncols=1, hspace=0.4, wspace=0.4)
            ax = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[1, 0])
            axL = fig.add_subplot(grid[2, 0])
        else:
            grid = plt.GridSpec(nrows=3, ncols=1, hspace=0.4, wspace=0.4)
            ax = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[1, 0])
            axL = fig.add_subplot(grid[2, 0])

    else:
        [ax, ax2, axL] = axs

    if plotFundamental:
        ax.set_title(self.output_transformed)
    else:
        ax.set_title(self.output)

    newLabels = self.variables if plotFundamental else [ikey for ikey in self.bounds]
    if newLabels is None or self.gpmodel.ard_num_dims > len(newLabels):
        newLabels = [
            ikey for ikey in self.bounds
        ]  # For cases where I actually did not transform even if physicsInformed exitsts (constant)

    """
	------------------------------------------------------------------------------------------------------------
	Prepare variables to plot
		- without "tr" is the raw variables throughout here
		- with "tr" is transformed variables if plotFundamental, otherwise it's the same as without
	------------------------------------------------------------------------------------------------------------
	"""

    trainX = self.train_X
    trainY, trainYvar = self.train_Y, self.train_Yvar

    y_next_pred = self.predict(x_next)[0].detach() if x_next is not None else None

    if plotFundamental:
        trainXtr = self.gpmodel.input_transform.tf1(trainX)
        trainYtr, trainYvar_tr = self.gpmodel.outcome_transform.tf1(
            trainX, trainY, trainYvar
        )

        xtr_next = (
            self.gpmodel.input_transform.tf1(x_next) if x_next is not None else None
        )
        ytr_next_pred, _ = (
            self.gpmodel.outcome_transform.tf1(x_next, y_next_pred, y_next_pred)
            if x_next is not None
            else (None, None)
        )

        xtr_best = (
            self.gpmodel.input_transform.tf1(x_best) if x_best is not None else None
        )
        ytr_best, yVartr_best = (
            self.gpmodel.outcome_transform.tf1(x_best, y_best, yVar_best)
            if y_best is not None
            else (None, None)
        )

        # Add additional points added from file
        trainXtr = torch.cat((self.train_X_added, trainXtr), axis=0)
        trainYtr = torch.cat((self.train_Y_added, trainYtr), axis=0)
        trainYvar_tr = torch.cat((self.train_Yvar_added, trainYvar_tr), axis=0)

    else:
        trainXtr = trainX
        trainYtr, trainYvar_tr = trainY, trainYvar

        xtr_next = x_next
        ytr_next_pred = y_next_pred

        xtr_best = x_best
        ytr_best, yVartr_best = y_best, yVar_best

    """
	------------------------------------------------------------------------------------------------------------
	Plot model
	------------------------------------------------------------------------------------------------------------
	"""

    # If a lot of training points, reduce markersize and alpha
    markersize, amult = (5, 1.0) if (trainX.shape[0] < 10) else (2, 0.5)

    # **********************************
    # 1D models
    # **********************************
    if dimX == 1:
        # Note: 95% confidence is 2*std (my standard for plotting, i.e. stds=2) -> confidene is +/- 2*np.sqrt(trainYvar)
        confidence_region = stds * trainYvar_tr[:, 0] ** 0.5

        # Plot training data
        ax.errorbar(
            trainXtr[:, 0],
            trainYtr[:, 0],
            yerr=confidence_region,
            c="k",
            alpha=1.0 * amult,
            label=extralab + "Observed Data",
            markersize=markersize,
            capsize=3.0,
            fmt="s",
            elinewidth=0.5,
            capthick=0.5,
        )

        # Develop x-grid
        if plotFundamental:
            minX, maxX = trainXtr.min(), trainXtr.max()
        else:
            minX, maxX = self.bounds[list(self.bounds)[0]]

        xgrid = torch.linspace(minX, maxX, pointsEvaluate).to(self.dfT).unsqueeze(1)

        # Predict
        mean, upper, lower, samples = self.predict(
            xgrid, nSamples=nSamples1Dmodels, produceFundamental=plotFundamental
        )
        mean, upper, lower, samples = (
            mean.detach(),
            upper.detach(),
            lower.detach(),
            samples.detach(),
        )

        # Plot
        contour = plot1D(
            ax,
            xgrid,
            mean,
            upper=upper * stds / 2.0,
            lower=lower * stds / 2.0,
            extralab=extralab,
        )

        if samples is not None:
            for q in range(samples.shape[0] - 1):
                ax.plot(xgrid, samples[q, :, 0], lw=0.1, c="g")
            ax.plot(
                xgrid,
                samples[-1, :, 0],
                lw=0.1,
                c="g",
                label=f"{nSamples1Dmodels} random samples",
            )

        # ----------------------

        # Plot next
        if xtr_next is not None:
            for c, i in enumerate(xtr_next[:, 0]):
                if c == 0:
                    lw = 1
                else:
                    lw = 0.5
                ax.axvline(x=i, ls="-.", c="c", lw=lw, label="Next")
            for c, i in enumerate(ytr_next_pred[:, 0]):
                if c == 0:
                    lw = 1
                else:
                    lw = 0.5
                ax.axhline(y=i, ls="-.", c="c", lw=lw)

        ax.set_xlabel(newLabels[0])

        ax.legend(loc="best")

        GRAPHICStools.addDenseAxis(ax)

    elif dimX == 2:  # TO FIX
        # Plot training data
        ax.plot(trainXtr[:, 0], trainXtr[:, 1], "ks", markersize=markersize)
        ax2.plot(trainXtr[:, 0], trainXtr[:, 1], "ks", markersize=markersize)

        """
		----------------------
		Plot model
		"""

        # Develop x-grid

        # Develop x-grid
        if plotFundamental:
            minX1, maxX1 = trainXtr[:, 0].min(), trainXtr[:, 0].max()
            minX2, maxX2 = trainXtr[:, 1].min(), trainXtr[:, 1].max()
        else:
            minX1, maxX1 = self.bounds[list(self.bounds)[0]]
            minX2, maxX2 = self.bounds[list(self.bounds)[1]]

        xgrid0 = torch.cat(
            (
                torch.linspace(minX1, maxX1, pointsEvaluate).to(self.dfT).unsqueeze(1),
                torch.linspace(minX2, maxX2, pointsEvaluate).to(self.dfT).unsqueeze(1),
            ),
            dim=1,
        )
        xgrid = torch.from_numpy(MATHtools.create2Dmesh(xgrid0[:, 0], xgrid0[:, 1])).to(
            self.dfT
        )

        # Predict
        mean, upper, lower, _ = self.predict(xgrid, produceFundamental=plotFundamental)

        # Plot
        zM = mean.detach().reshape(xgrid0.shape[0], xgrid0.shape[0])

        contour, vmaxvmin2 = plot2D(ax, xgrid0[:, 0], xgrid0[:, 1], zM, levels=None)

        # Plot uncertainty
        uncert = ((upper - lower) * stds / 2.0) * 0.5
        z = uncert.detach().reshape(xgrid0.shape[0], xgrid0.shape[0])
        contourUnc, _ = plot2D(ax2, xgrid0[:, 0], xgrid0[:, 1], z, cmap="viridis")

        """
		----------------------
		Plot next
		"""

        if x_next is not None:
            for axj in [ax, ax2]:
                axj.plot(
                    xtr_next[:, 0],
                    xtr_next[:, 1],
                    "*",
                    c="w",
                    markersize=5,
                    label="Next",
                )

        ax.set_xlabel(newLabels[0])
        ax.set_ylabel(newLabels[1])
        # ax.set_xlabel(newLabels[1]); ax.set_ylabel(newLabels[0]) # Check this, but I think it's right...

        ax2.set_title("Uncertainty")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.addDenseAxis(ax2)

    # **********************************
    # >2D models -> Sensitivities
    # **********************************
    else:
        numP, percent = (
            (pointsEvaluate, 0.5) if plotFundamental else (pointsEvaluate, 1.0)
        )

        # ----------------------------------
        # Senstitivies around x_best or last
        # ----------------------------------

        if x_best is not None:
            ToPlotAround_x, yEvaluated, yVarEvaluated, labelWhich = (
                x_best[0, :],
                ytr_best,
                yVartr_best,
                "best",
            )
            isCeterProvidedTransformed = False
        elif trainX.shape[0] > 0:
            ToPlotAround_x, yEvaluated, yVarEvaluated, labelWhich = (
                trainX[-1, :],
                trainYtr[-1, :],
                trainYvar_tr[-1, :],
                "last",
            )
            isCeterProvidedTransformed = False
        else:
            ToPlotAround_x, yEvaluated, yVarEvaluated, labelWhich = (
                trainXtr[-1, :],
                trainYtr[-1, :],
                trainYvar_tr[-1, :],
                "last (tr)",
            )
            isCeterProvidedTransformed = True

        self.plotSensitivities(
            ToPlotAround_x,
            isCeterProvidedTransformed=isCeterProvidedTransformed,
            y_center=yEvaluated,
            yVar_center=yVarEvaluated,
            trainX=trainXtr,
            plotFundamental=plotFundamental,
            ax=ax,
            extralab=extralab,
            legendYN=7,
            labelWhich=labelWhich,
            labels=newLabels,
            numP=numP,
            percent=percent,
            stds=stds,
            labelEvaluated=labelEvaluated,
        )

        # ---------------------------------------
        # Senstitivies around next point or first
        # ---------------------------------------
        if ax2 is not None:
            if x_next is not None:
                ToPlotAround_x, yEvaluated, yVarEvaluated, labelWhich = (
                    x_next[0, :],
                    None,
                    None,
                    "next",
                )
                isCeterProvidedTransformed = False
            elif trainX.shape[0] > 0:
                ToPlotAround_x, yEvaluated, yVarEvaluated, labelWhich = (
                    trainX[0, :],
                    trainYtr[self.train_X_added.shape[0] if plotFundamental else 0, :],
                    trainYvar_tr[
                        self.train_X_added.shape[0] if plotFundamental else 0, :
                    ],
                    "first",
                )
                isCeterProvidedTransformed = False
            else:
                ToPlotAround_x, yEvaluated, yVarEvaluated, labelWhich = (
                    trainXtr[0, :],
                    trainYtr[0, :],
                    trainYvar_tr[0, :],
                    "first (tr)",
                )
                isCeterProvidedTransformed = True

            self.plotSensitivities(
                ToPlotAround_x,
                isCeterProvidedTransformed=isCeterProvidedTransformed,
                y_center=yEvaluated,
                yVar_center=yVarEvaluated,
                ax=ax2,
                extralab=extralab,
                labelWhich=labelWhich,
                labels=newLabels,
                plotFundamental=plotFundamental,
                trainX=trainXtr,
                numP=numP,
                percent=percent,
                stds=stds,
                labelEvaluated=labelEvaluated,
            )

        # --------------------------------
        # Plot next, as an intuitive guide
        # --------------------------------

        if y_next_pred is not None:
            for c, i in enumerate(ytr_next_pred[:, 0]):
                ax.axhline(y=i, ls="-.", c="c", lw=1 if c == 0 else 0.5)
                if ax2 is not None:
                    ax2.axhline(y=i, ls="-.", c="c", lw=1 if c == 0 else 0.5)

    """
	------------------------------------------------------------------------------------------------------------
	Plot Losses
	------------------------------------------------------------------------------------------------------------
	"""

    if (self.losses is not None) and (axL is not None):
        losses_all, loss_ini, loss_final = (
            self.losses["losses"],
            self.losses["loss_ini"],
            self.losses["loss_final"],
        )

        if losses_all is not None:
            y, x = losses_all, np.arange(len(losses_all))
            fin, lin = x[-1], "o"
            axL.plot(x, y, "-s", markersize=2)
        else:
            fin, lin = 1, "-o"

        axL.set_ylabel("Loss (-MLL)")
        axL.set_xlabel("Iterations")
        axL.set_xlim(left=0)
        axL.plot([0, fin], [loss_ini, loss_final], lin, markersize=10)

        GRAPHICStools.addDenseAxis(axL)


def plotSensitivities_surrogate_model(
    self,
    x_center0,
    isCeterProvidedTransformed=False,
    y_center=None,
    yVar_center=None,
    ax=None,
    extralab="",
    alpha=0.1,
    legendYN=None,
    labelWhich="",
    plotRelativeX=True,
    labels=None,
    plotFundamental=True,
    trainX=None,
    numP=50,
    percent=0.5,
    stds=2,
    labelEvaluated="",
):
    """
    x_center0 should be given raw, untransformed, even if I choose plotFundamental
    """

    colors = GRAPHICStools.listColors()

    x_center = (
        self.gpmodel.input_transform.tf1(x_center0)
        if (plotFundamental and (not isCeterProvidedTransformed))
        else x_center0
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Define xgrid
    # ~~~~~~~~~~~~~~~~~~~~~~~

    xgrid = torch.repeat_interleave(x_center.unsqueeze(0), numP, dim=0)

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Prepare figure
    # ~~~~~~~~~~~~~~~~~~~~~~~

    if ax is None:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for i in range(xgrid.shape[1]):
        if labels is not None:
            case = labels[i]

        xgrid_new = xgrid.clone()

        # -----------------------------------------------------------------
        # Define ranges to predict

        centerPoint = xgrid[0, i]

        # Range defined as the training range or with bounds (to deal with cases where I read values from file)
        if plotFundamental:
            boundsRange, labelx = trainX[:, i].max() - trainX[:, i].min(), "train"
        else:
            boundsRange, labelx = (
                self.bounds[list(self.bounds.keys())[i]].max()
                - self.bounds[list(self.bounds.keys())[i]].min(),
                "bounds",
            )

        minRange = centerPoint - boundsRange * percent
        maxRange = centerPoint + boundsRange * percent

        xgrid_new[:, i] = torch.linspace(minRange, maxRange, numP)

        if plotRelativeX:
            extracted_x_norm = (xgrid_new - centerPoint) / boundsRange * 100
        else:
            extracted_x_norm = xgrid_new

        # -----------------------------------------------------------------
        mean, upper, lower, _ = self.predict(
            xgrid_new, produceFundamental=plotFundamental
        )
        mean, upper, lower = (
            mean[:, 0].detach(),
            upper[:, 0].detach(),
            lower[:, 0].detach(),
        )

        minVariation = 1e-10

        # print(f'Variable {i} scanned from {minRange} to {maxRange}')
        # Do not plot if this variable did not affect the value of the surrogate, unless it's a fixed one
        if (
            (minVariation is None)
            or ((mean.max() - mean.min()).abs().item() > minVariation)
            or (self.FixedValue)
        ):
            contour = plot1D(
                ax,
                extracted_x_norm[:, i],
                mean,
                upper=upper * stds / 2.0,
                lower=lower * stds / 2.0,
                color=colors[i],
                extralab=case,
                alpha=alpha,
                legendConf=False,
                ls="-",
                lw=1.0,
            )

    # Point evaluation
    if y_center is not None:
        confidence_region = (
            stds * yVar_center**0.5 if yVar_center is not None else None
        )
        ax.errorbar(
            [0],
            y_center,
            yerr=confidence_region,
            c="k",
            markersize=5,
            capsize=3.0,
            fmt="s",
            elinewidth=0.5,
            capthick=0.5,
            label=labelEvaluated,
        )

    if plotRelativeX:
        ax.set_xlabel(f"% of {labelx} range around {labelWhich}")
    else:
        ax.set_ylabel("X")

    if legendYN is not None:
        ax.legend(loc="best", fontsize=legendYN)

    GRAPHICStools.addDenseAxis(ax)


def plotTraining_surrogate_model(
    self, axs=None, relative_to=-1, figIndex_inner=0, stds=2.0, legYN=True
):
    colors = GRAPHICStools.listColors()

    if axs is None:
        plt.ion()
        fig = plt.figure(figsize=(6, 9))

        grid = plt.GridSpec(nrows=3, ncols=1, hspace=0.4, wspace=0.4)
        ax0 = fig.add_subplot(grid[0, 0])
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[2, 0])

    else:
        [ax0, ax1, ax2] = axs

    newLabels = self.variables

    train_X = self.gpmodel.input_transform.tf1(self.train_X)
    trainYtr, trainYvar_tr = self.gpmodel.outcome_transform.tf1(
        self.train_X, self.train_Y, self.train_Yvar
    )

    # Add additional points added from file
    train_X = torch.cat((self.train_X_added, train_X), axis=0)
    trainYtr = torch.cat((self.train_Y_added, trainYtr), axis=0)
    trainYvar_tr = torch.cat((self.train_Yvar_added, trainYvar_tr), axis=0)

    if newLabels is None or len(newLabels) < train_X.shape[1]:
        newLabels = [ikey for ikey in self.bounds]

    x = np.arange(0, train_X.shape[0])

    for j in range(train_X.shape[1]):
        ax0.plot(
            x,
            train_X[:, j],
            "-s",
            markersize=1,
            lw=0.5,
            color=colors[j],
            label=newLabels[j],
        )
        ax1.plot(
            x,
            train_X[:, j] / train_X[relative_to, j],
            "-s",
            markersize=1,
            lw=0.5,
            color=colors[j],
            label=newLabels[j],
        )

    ax2.plot(x, trainYtr[:, 0], "-s", markersize=3, color="b", label="train")
    ax2.errorbar(
        x,
        trainYtr[:, 0],
        c="b",
        yerr=stds * trainYvar_tr[:, 0] ** 0.5,
        capsize=5.0,
        fmt="none",
    )  # 2*std, confidence bounds

    mean, upper, lower, _ = self.predict(train_X, produceFundamental=True)
    mean = mean[:, 0].detach().cpu().numpy()
    ax2.plot(x, mean, "-s", color="r", lw=0.5, markersize=3, label="model")
    ax2.errorbar(
        x,
        mean,
        c="r",
        yerr=[
            mean - lower[:, 0].detach().cpu().numpy() * stds / 2.0,
            upper[:, 0].detach().cpu().numpy() * stds / 2.0 - mean,
        ],
        capsize=3.0,
        fmt="none",
        alpha=1.0,
        lw=0.5,
        markersize=3,
    )  # 2*std, confidence bounds

    ax0.set_title(self.output_transformed)
    ax0.legend()
    GRAPHICStools.addDenseAxis(ax0)
    ax0.set_xlim(left=0)

    if figIndex_inner == 0:
        ax0.set_ylabel("Absolute DVs")
        ax1.set_ylabel("Relative DVs to last")
        ax2.set_ylabel("OF absolute value")

    ax1.set_ylim([0.9, 1.1])
    GRAPHICStools.addDenseAxis(ax1)
    ax1.set_xlim(left=0)

    ax2.set_xlabel("Evaluation")
    GRAPHICStools.addDenseAxis(ax2)
    ax2.set_xlim(left=0)

    if legYN:
        ax2.legend(loc="best", prop={"size": 4})


def localBehavior_surrogate_model(
    self, x, outputs=None, plotYN=True, ax=None, prefix="", rangeZeroth=1e-6
):
    """
    x (dim), no batch
    J =
            dQe1/dze1   dQe1/dze2   dQe1/dzi1   dQe1/dzi2
            dQe2/dze1   dQe2/dze2   dQe2/dzi1   dQe2/dzi2
            dQi1/dze1   dQi1/dze2   dQi1/dzi1   dQi1/dzi2
            dQi2/dze1   dQi2/dze2   dQi2/dzi1   dQi2/dzi2

    - I can work for invidual models but also ModelList
    """

    X = torch.tensor(x, requires_grad=True).to(self.dfT)

    # Option 1: Get Jacobian my constructing the matrix manually

    # mean = self.predict(X.unsqueeze(0))[0].squeeze(0)

    # J = torch.zeros((mean.shape[0],X.shape[0]))
    # for i in range(mean.shape[0]):	J[i,:] = torch.autograd.grad(mean[i],X,retain_graph=True)[0]  # dQe1/dze1  dQe1/dze2   dQe1/dzi1   dQe1/dzi2

    # Option 2: Use jacobian function

    def ev(X):
        return self.predict(X.unsqueeze(0))[0].squeeze(0)

    J = torch.autograd.functional.jacobian(
        ev, X, strict=False, vectorize=True
    )  # vectorize makes it faster, but could have troubles

    # -----------------------------------------------------------------------------

    if plotYN:
        xlabels = [i for i in self.bounds.keys()]
        ylabels = outputs if outputs is not None else [self.output]

        titletxt = "["
        for ix in x:
            titletxt += f"{ix:.2f}, "
        titletxt = titletxt[:-2] + "]"
        lim = 80
        if len(titletxt) > lim:
            titletxt = titletxt[:lim] + "..."

        GRAPHICStools.plotMatrix(
            J,
            ax=ax,
            xlabels=xlabels,
            ylabels=ylabels,
            title=f"{prefix}$\\vec{{x}}$ = {titletxt}",
            fontsize_title=5,
            fontsize=8,
            expo=True,
            symmetrice=True,
            cmap="seismic",
            fontcolor="black",
            rangeText=rangeZeroth,
        )

    return J


def localBehavior_scan_surrogate_model(
    self, x, numP=50, dimension_label=None, plotYN=True, axs=None, c="b", xrange=0.5
):
    """
    This works only for individual models
    """

    xlabels = [i for i in self.bounds.keys()]
    x_dim_chosen = (
        0
        if dimension_label is None
        else np.where(np.array(xlabels) == dimension_label)[0][0]
    )

    def ev(X):
        return self.predict(X.unsqueeze(0))[0].squeeze(0)

    vari = np.linspace(-xrange, xrange, numP)

    Jx = np.zeros(numP)
    J = np.zeros(numP)
    Y = np.zeros(numP)
    for i, val in enumerate(vari):
        X_mod = x.clone()
        X_mod[x_dim_chosen] = x[x_dim_chosen] * (1 + val)

        X = torch.tensor(X_mod, requires_grad=True).to(self.dfT)

        Jx[i] = X[x_dim_chosen]
        Y[i] = ev(X)[0]
        J[i] = torch.autograd.functional.jacobian(ev, X, strict=False)[0, x_dim_chosen]

    if plotYN:
        if axs is None:
            fig, axs = plt.subplots(nrows=2, figsize=(6, 9))

        xlabel = xlabels[x_dim_chosen]

        ax = axs[0]
        ax.plot(Jx, Y, "-o", color=c, lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{self.output}")

        ax = axs[1]
        ax.plot(Jx, J, "-o", color=c, lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"d({self.output})/d({xlabel})")
        ax.set_title("Scan of local Jacobian")


# ----------------------------------------------------------------------------------------------------


def retrieveResults(
    folderWork,
    folderRemote=None,
    analysis_level=0,
    doNotShow=False,
    plotYN=True,
    pointsEvaluateEachGPdimension=50,
):
    # ----------------------------------------------------------------------------------------------------------------
    # Grab remote results optimization
    # ----------------------------------------------------------------------------------------------------------------

    if folderRemote is not None:
        [machine, folderRemote0] = folderRemote.split(":")
        if "-" in machine:
            [machine, port] = machine.split("-")
            port = "-P " + port
        else:
            port = ""

        username = IOtools.expandPath("$USER")

        print(" - Grabbing remote")
        if not os.path.exists(folderWork):
            os.system(f"mkdir {folderWork}")

        os.system(
            f"scp -TO -r {port} {username}@{machine}:{folderRemote0}/Outputs {folderWork}"
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Viewing workflow
    # ----------------------------------------------------------------------------------------------------------------

    print("\t\t--> Opening MITIMstate.pkl")
    prfs_model = STRATEGYtools.read_from_scratch(f"{folderWork}/Outputs/MITIMstate.pkl")

    if "timeStamp" in prfs_model.__dict__:
        print(f"\t\t\t- Time stamp of MITIMstate.pkl: {prfs_model.timeStamp}")
    else:
        print("\t\t\t- Time stamp of MITIMstate.pkl not found")

    # ---------------- Read ResultsOptimization
    fileOutputs = folderWork + "/Outputs/ResultsOptimization.out"
    res = ResultsOptimization(file=fileOutputs)
    res.readClass(prfs_model)
    res.read()

    # ---------------- Read Logger
    log = LogFile(folderWork + "/Outputs/MITIM.log")
    try:
        log.interpret()
    except:
        print("Could not read log", typeMsg="w")
        log = None

    # ---------------- Read Tabular
    if analysis_level >= 0:
        _, dataTabular = readTabularLines(folderWork + "/Outputs/TabularData.dat")
        _, dataETabular = readTabularLines(folderWork + "/Outputs/TabularDataStds.dat")
    else:
        dataETabular = dataTabular = None

    fn = None
    # If pickle, plot all the strategy info
    if analysis_level > 0:
        # Store here the store
        try:
            with open(f"{folderWork}/Outputs/MITIMextra.pkl", "rb") as handle:
                prfs_model.dictStore = pickle_dill.load(handle)
        except:
            print("Could not load MITIMextra", typeMsg="w")
            prfs_model.dictStore = None
        # -------------------

        prfs_model.ResultsOptimization = res
        prfs_model.logFile = log
        if plotYN:
            fn = prfs_model.plot(
                doNotShow=doNotShow,
                pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
            )

    # If no pickle, plot only the contents of ResultsOptimization
    else:
        if plotYN:
            fn = res.plot(doNotShow=doNotShow, log=log)
        prfs_model = None

    return fn, res, prfs_model, log, dataTabular, dataETabular


class LogFile:
    def __init__(self, file):
        self.file = file

    def activate(self, writeAlsoTerminal=True):
        sys.stdout = IOtools.Logger(
            logFile=self.file, writeAlsoTerminal=writeAlsoTerminal
        )

    def interpret(self):
        with open(self.file, "r") as f:
            lines = f.readlines()

        self.steps = {}
        for line in lines:
            if "Starting MITIM Optimization" in line:
                try:
                    self.steps["start"] = IOtools.getTimeFromString(
                        line.split(",")[0].strip()
                    )
                except:
                    self.steps["start"] = IOtools.getTimeFromString(
                        " ".join(line.split(",")[0].strip().split()[-2:])
                    )
                self.steps["steps"] = {}
            if "MITIM Step" in line:
                aft = line.split("Step")[-1]
                ikey = int(aft.split()[0])
                time_str = aft.split("(")[-1].split(")")[0]
                self.steps["steps"][ikey] = {
                    "start": IOtools.getTimeFromString(time_str),
                    "optimization": {},
                }
            if "Posterior Optimization" in line:
                time_str = line.split(",")[-1][:-2].strip()
                self.steps["steps"][ikey]["optimization"] = {
                    "start": IOtools.getTimeFromString(time_str),
                    "steps": {},
                }
                cont = 0
            if "Optimization stage " in line:
                aft = line.split("Step")[-1]
                time_str = aft.split("(")[-1].split(")")[0]
                self.steps["steps"][ikey]["optimization"]["steps"][cont] = {
                    "name": line.split()[4],
                    "start": IOtools.getTimeFromString(time_str),
                }
                cont += 1

        self.process()

    def process(self):
        for step in self.steps["steps"]:
            time_start = self.steps["steps"][step]["start"]

            if "start" not in self.steps["steps"][step]["optimization"]:
                break
            time_end = self.steps["steps"][step]["optimization"]["start"]
            timeF = IOtools.getTimeDifference(
                time_start, newTime=time_end, niceText=False
            )
            self.steps["steps"][step]["fitting"] = timeF

            if step + 1 in self.steps["steps"]:
                time_end = self.steps["steps"][step + 1]["start"]
                time = IOtools.getTimeDifference(
                    time_start, newTime=time_end, niceText=False
                )
                self.steps["steps"][step]["time_s"] = time

            for opt_step in self.steps["steps"][step]["optimization"]["steps"]:
                time_start = self.steps["steps"][step]["optimization"]["steps"][
                    opt_step
                ]["start"]

                if opt_step + 1 in self.steps["steps"][step]["optimization"]["steps"]:
                    time_end = self.steps["steps"][step]["optimization"]["steps"][
                        opt_step + 1
                    ]["start"]
                    time = IOtools.getTimeDifference(
                        time_start, newTime=time_end, niceText=False
                    )
                    self.steps["steps"][step]["optimization"]["steps"][opt_step][
                        "time_s"
                    ] = time
                else:
                    if step + 1 in self.steps["steps"]:
                        time_end = time_end = self.steps["steps"][step + 1]["start"]
                        time = IOtools.getTimeDifference(
                            time_start, newTime=time_end, niceText=False
                        )
                        self.steps["steps"][step]["optimization"]["steps"][opt_step][
                            "time_s"
                        ] = time

        self.points = [
            0,
            IOtools.getTimeDifference(
                self.steps["start"],
                newTime=self.steps["steps"][0]["start"],
                niceText=False,
            ),
        ]
        self.types = ["b"]

        for step in self.steps["steps"]:
            if "fitting" in self.steps["steps"][step]:
                self.points.append(
                    self.steps["steps"][step]["fitting"] + self.points[-1]
                )
                self.types.append("r")

            if "steps" not in self.steps["steps"][step]["optimization"]:
                break
            for opt_step in self.steps["steps"][step]["optimization"]["steps"]:
                if (
                    "time_s"
                    in self.steps["steps"][step]["optimization"]["steps"][opt_step]
                ):
                    self.points.append(
                        self.steps["steps"][step]["optimization"]["steps"][opt_step][
                            "time_s"
                        ]
                        + self.points[-1]
                    )
                    self.types.append("g")

        self.points = np.array(self.points)

        self.its = np.linspace(0, len(self.points) - 1, len(self.points))

    def plot(
        self,
        axs=None,
        factor=60.0,
        fullCumulative=False,
        ls="-",
        lab="",
        marker="o",
        color="b",
    ):
        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(ncols=2)

        ax = axs[0]
        subtractor = 0
        totals = {"ini": 0.0, "fit": 0.0, "opt": 0.0}

        for i in range(len(self.points) - 1):
            if self.types[i] == "r":
                ax.axvline(x=self.its[i], ls="--", c="k", lw=0.5)
                if not fullCumulative:
                    subtractor = self.points[i]

            ps = [
                (self.points[i] - subtractor) / factor,
                (self.points[i + 1] - subtractor) / factor,
            ]

            if self.types[i] == "b":
                totals["ini"] += ps[1] - ps[0]
            if self.types[i] == "g":
                totals["opt"] += ps[1] - ps[0]
            if self.types[i] == "r":
                totals["fit"] += ps[1] - ps[0]

            if i == 0:
                labb = lab
            else:
                labb = ""

            ax.plot(
                [self.its[i], self.its[i + 1]],
                ps,
                marker + ls,
                c=self.types[i],
                label=labb,
            )

        if factor == 60.0:
            label = "minutes"
            ax.axhline(y=60, ls="-.", lw=0.2)
        elif factor == 3600.0:
            label = "hours"
            ax.axhline(y=1, ls="-.", lw=0.2)
        else:
            label = "seconds"

        # ax.set_xlabel('Workflow Steps')
        ax.set_ylabel(f"Cumulated Time ({label})")
        # ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="b", lw=2),
            Line2D([0], [0], color="r", lw=2),
            Line2D([0], [0], color="g", lw=2),
        ]

        legs = [
            "Initialization + Evaluation",
            "Evaluation + Fitting",
            "Optimization",
            "Total",
        ]
        ax.legend(custom_lines, legs)

        ax = axs[1]
        ax.bar(
            legs,
            [
                totals["ini"],
                totals["fit"],
                totals["opt"],
                totals["ini"] + totals["fit"] + totals["opt"],
            ],
            1 / 3,
            alpha=0.5,
            label=lab,
            color=color,
        )  # , label=equil_names[i],color=colors[i],align='edge')

        # ax.set_xlabel('Workflow')
        ax.set_ylabel(f"Cumulated Time ({label})")


class TabularData:
    def __init__(
        self,
        inputs,
        outputs,
        file="Outputs/TabularData.dat",
        forceNew=False,
        interface="HIGH_FIDELITY",
        uniqueNumbering=False,
    ):
        # If start from scratch, overwrite the tabular, otherwise there's risk of error if not all OFs coincide, that's why forceNew

        self.file = file
        self.nameFile = self.file.split("/")[-1]
        self.inputs = inputs
        self.outputs = outputs

        STR_Tabular = self.produceLine(varsX=None)

        if (not os.path.exists(self.file)) or forceNew:
            with open(self.file, "w") as f:
                f.write(STR_Tabular + "\n")

        self.updateFromFile(interface=interface, uniqueNumbering=uniqueNumbering)

    def updateFromFile(self, interface="HIGH_FIDELITY", uniqueNumbering=False):
        self.OriginalLines, self.data = readTabularLines(
            self.file, interface=interface, uniqueNumbering=uniqueNumbering
        )

    def produceLine(self, varsX=None):
        if varsX is None:
            varsX0 = ["#", "interface"]
            for i in self.inputs:
                varsX0.append(i)
            for i in self.outputs:
                varsX0.append(i)
        else:
            varsX0 = [
                f"{varsX[i]:.16f}" if IOtools.isfloat(varsX[i]) else str(varsX[i])
                for i in range(len(varsX))
            ]

        str_line = f"{varsX0[0]:>4}"
        for i in range(len(varsX0) - 1):
            str_line += f"{varsX0[i+1]:>30s}"

        return str_line + "\n"

    def grabFromTabular(self, points=[0, 1, 2, 3, 4, 5]):
        print(
            f"\t* Reading points from tabular file ({self.nameFile})",
            verbose=verbose_level,
        )

        X = np.zeros((len(points), len(self.inputs)))
        Y = np.zeros((len(points), len(self.outputs)))
        for i in range(X.shape[0]):
            aux = []
            if points[i] in self.data:
                for j in self.data[points[i]]:
                    aux.append(self.data[points[i]][j])
            else:
                aux = np.ones(100) * np.nan
            for j in range(X.shape[1]):
                X[i, j] = aux[j]
            for k in range(Y.shape[1]):
                Y[i, k] = aux[j + k + 1]

        return X, Y

    def grabFromTabular_Specific(self, x, printStuff=True):
        coincidentPoint = None
        for ipoint in self.data:
            coincident = True
            for iDV, ikey in enumerate(self.inputs):
                if self.data[ipoint][ikey] != round(x[iDV], 16):
                    # print(self.data[ipoint][ikey],x[iDV])
                    coincident = coincident and False
            if coincident:
                coincidentPoint = ipoint
                if printStuff:
                    print(
                        f"\t* Element required coincided with element {coincidentPoint} from table ({self.nameFile})",
                        verbose=verbose_level,
                    )
                break

        if coincidentPoint is None:
            if printStuff:
                print(
                    f"\t* Element required could not be found in table ({self.nameFile})",
                    typeMsg="w",
                )
            y = np.ones(len(self.outputs)) * np.nan
        else:
            y = []
            for iout in self.outputs:
                if self.data[ipoint][iout] is not None:
                    y.append(self.data[ipoint][iout])
                else:
                    y.append(np.nan)

        return np.array(y), coincidentPoint

    def updatePoints(self, X, Y=[]):
        dataOrig = copy.deepcopy(self.data)

        for ipoint in range(len(X)):
            self.data[ipoint] = OrderedDict()
            for iDV in range(len(self.inputs)):
                self.data[ipoint][self.inputs[iDV]] = round(X[ipoint][iDV], 16)
            for iout in range(len(self.outputs)):
                try:
                    self.data[ipoint][self.outputs[iout]] = round(Y[ipoint][iout], 16)
                except:
                    # If there is a point there evaluated, do not change it
                    try:
                        self.data[ipoint][self.outputs[iout]] = dataOrig[ipoint][
                            self.outputs[iout]
                        ]
                    except:
                        self.data[ipoint][self.outputs[iout]] = np.nan

        self.updateFile()

    def updateFile(self, source_interface=None):
        if source_interface is None:
            source_interface = ["HIGH_FIDELITY"] * len(self.data)

        str_header = self.produceLine(varsX=None)

        str_points = ""
        for ipoint, source_interface0 in zip(self.data, source_interface):
            xy = [ipoint, source_interface0]
            for iDV in self.inputs:
                xy.append(self.data[ipoint][iDV])
            for iout in self.outputs:
                xy.append(self.data[ipoint][iout])
            str_p = self.produceLine(xy)
            str_points += str_p

        # I need to check it has not been opened
        with open(self.file, "w") as f:
            f.write(str_header + str_points)
        print(f"\t* Table file updated ({self.nameFile})", verbose=verbose_level)

    def removePointsAfter(self, fromPoint):
        dataOrig = copy.deepcopy(self.data)
        for ipoint in dataOrig:
            if int(ipoint) > fromPoint:
                del self.data[ipoint]

        self.updateFile()


def modTabulars(inputs, outputs, folder, folder_new, new_dv_with_old_value):
    for fi in ["TabularData.dat", "TabularDataStds.dat"]:
        fileTabular = IOtools.expandPath(f"{folder}/{fi}")
        newFile = IOtools.expandPath(f"{folder_new}/{fi}")

        # -------------------------------------------------------------------------------------
        # Read Old
        # -------------------------------------------------------------------------------------

        TabularData_d = TabularData(inputs, outputs, file=fileTabular)

        # -------------------------------------------------------------------------------------
        # Modify
        # -------------------------------------------------------------------------------------

        new_inputs = copy.deepcopy(inputs)
        new_inputs.append(new_dv_with_old_value[0])

        TabularData_d.inputs = new_inputs

        for i in TabularData_d.data:
            TabularData_d.data[i][new_dv_with_old_value[0]] = new_dv_with_old_value[1]

        TabularData_d.file = newFile

        TabularData_d.updateFile()


class ResultsOptimization:
    def __init__(self, file="Outputs/ResultsOptimization.out"):
        self.file = file
        self.predictedSofar = 0

    def readClass(self, PRF_BOclass):
        self.PRF_BO = PRF_BOclass

    def initialize(self, PRF_BOclass):
        self.readClass(PRF_BOclass)
        self.createHeaders()
        self.save()

    def save(self):
        with open(self.file, "w") as f:
            f.write(self.lines)
        print("\t* ResultsOptimization updated", verbose=verbose_level)

    def read(self):
        print(f"\t\t--> Opening {IOtools.clipstr(self.file)}")

        with open(self.file, "r") as f:
            lines = f.readlines()

        self.evaluations = gatherEv(lines, var=" Evaluation ")

        self.predictions = gatherEv(lines, var=" Prediction ")

        if len(self.evaluations) > 0:
            self.DV_labels = [i for i in self.evaluations[0]["x"].keys()]
            self.OF_labels = [i for i in self.evaluations[0]["y"].keys()]
        else:
            print(" --> MITIM has not run first batch yet")

        self.getFeatures(lines)

        # Grab best case based on mean
        try:
            (
                self.best_absolute,
                self.best_absolute_index,
                self.best_absolute_full,
            ) = self.getBest()
        except:
            print("\t- Problem retrieving best evaluation", typeMsg="w")
            self.best_absolute = (
                self.best_absolute_index
            ) = self.best_absolute_full = None

    def addLines(self, lines):
        self.lines = self.OriginalLines + lines
        self.save()

    def addPoints(
        self,
        includePoints,
        executed=True,
        predicted=True,
        timingString=None,
        Best=False,
        Name="",
        forceWrite=False,
        addheader=True,
    ):
        linesBatch = f"\n\n* {Name}"

        if addheader:
            linesBatch += "\n\n~~~~~~~~~~~~~~~~~~~~"
            if Best:
                linesBatch += "\nPredicted optimum point as of this iteration"
            else:
                linesBatch += (
                    "\nRunning high-fidelity evaluations for {0} points...".format(
                        includePoints[1] - includePoints[0]
                    )
                )
                if timingString is not None:
                    linesBatch += f" (took total of {timingString})"
            linesBatch += "\n~~~~~~~~~~~~~~~~~~~~"

        for i in range(includePoints[0], includePoints[1]):
            linesBatch += self.produceDVlines(position=i, Best=Best)

            if executed:
                lin, l2, yE = self.produceOFlines(position=i, predicted=False)
            else:
                lin, l2, yE = self.produceOFlines(
                    position=i, predicted=False, zero=True
                )
            linesBatch += lin
            linesBatch += f"\n\t\t\tL2-norm = {l2:.5f}"

            if predicted:
                position = i - includePoints[0]
                lin, l2, yP = self.produceOFlines(position=position, predicted=True)
            else:
                lin, l2, yP = self.produceOFlines(position=i, predicted=True, zero=True)
            linesBatch += lin
            linesBatch += f"\n\t\t\tL2-norm = {l2:.5f}"

            ydifference = np.abs((yE - yP) / yE) * 100.0
            lin, l2, _ = self.produceOFlines(
                position=i, ydifference=np.atleast_2d(ydifference)
            )
            linesBatch += lin
            linesBatch += f"\n\t\t\tL2-norm = {l2:.5f}"

        if not executed:
            self.OriginalLines = copy.deepcopy(self.lines)

        self.lines = self.OriginalLines + linesBatch

        if Best or forceWrite:
            self.OriginalLines = self.lines
        self.save()

    def produceDVlines(self, position, Best=False):
        if Best:
            strn = "Prediction"
            positionShow = self.predictedSofar + 1
            self.predictedSofar = positionShow
        else:
            strn = "Evaluation"
            positionShow = position

        xb = ""
        for cont, j in enumerate(self.PRF_BO.bounds):
            xb += f"\t\t\t{j} = {self.PRF_BO.train_X[position,cont]:.6e}\n"
        xb = xb[:-1]

        return f"\n\n {strn} {positionShow}\n\t x :\n{xb}"

    def produceOFlines(self, position=0, predicted=False, ydifference=None, zero=False):
        maxNum = int(1e6)  # 1000

        if ydifference is not None:
            if not zero:
                y = yl = yu = ydifference
            else:
                y = yl = yu = np.ones((maxNum, len(self.PRF_BO.outputs))) * np.nan
            stry = " (rel diff, %)"
            position = 0
        else:
            if predicted:
                if not zero:
                    y = np.atleast_2d(self.PRF_BO.y_next_pred.cpu())
                    yl = np.atleast_2d(self.PRF_BO.y_next_pred_l.cpu())
                    yu = np.atleast_2d(self.PRF_BO.y_next_pred_u.cpu())
                else:
                    y = yl = yu = np.ones((maxNum, len(self.PRF_BO.outputs))) * np.nan
                stry = " (model)"
            else:
                if not zero:
                    y = self.PRF_BO.train_Y
                    yl = (
                        self.PRF_BO.train_Y - 2 * self.PRF_BO.train_Ystd
                    )  # -2*sigma, to imitate the predicted one
                    yu = (
                        self.PRF_BO.train_Y + 2 * self.PRF_BO.train_Ystd
                    )  # +2*sigma, to imitate the predicted one
                else:
                    y = yl = yu = np.ones((maxNum, len(self.PRF_BO.outputs))) * np.nan
                stry = ""

        xb = ""
        allY = []
        for cont, j in enumerate(self.PRF_BO.outputs):
            xb += f"\t\t\t{j} = {y[position,cont]:.6e}"
            allY.append(y[position, cont])

            extra = f" [{yl[position,cont]:.6e},{yu[position,cont]:.6e}]\n"

            xb += extra

        xb = xb[:-1]

        l2 = np.linalg.norm(y[position, :])

        return f"\n\t y{stry} :\n{xb}", l2, np.array(allY)

    def createHeaders(self):
        if self.PRF_BO.restartYN:
            txtR = "\n* Restarting capability requested, looking into previous TabularData.dat"
        else:
            txtR = ""

        txtIn = ""
        for i in self.PRF_BO.bounds:
            txtIn += "\t{0} from {1:.5f} to {2:.5f}\n".format(
                i, self.PRF_BO.bounds[i][0], self.PRF_BO.bounds[i][1]
            )
        txtOut = ""
        for cont, i in enumerate(self.PRF_BO.outputs):
            txtOut += f"\t{i}"

        STR_header = f"""
MITIM version 0.2 (P. Rodriguez-Fernandez, 2020)
Workflow start time: {IOtools.getStringFromTime()} 
\t"""

        if self.PRF_BO.Optim["BaselineDV"] is None:
            STR_base = ""
        else:
            txtBase = ""
            for cont, i in enumerate(self.PRF_BO.bounds):
                txtBase += f"\t{i} = {self.PRF_BO.Optim['BaselineDV'][cont]:.5f}\n"
            STR_base = f"""
* Baseline point (added as Evaluation.0 to initial batch)
{txtBase}
"""

        STR_inputs = """
---------------------------------------------------------------------------
 MITIM Optimization Namelist
---------------------------------------------------------------------------
{5}
* {3} Input/Design Variables (DVs)
{0}
* {4} Objective Functions (OFs)
{1}
{6}
* Writing Tabular file with high-fidelity evaluation results in {2}
	""".format(
            txtIn,
            txtOut,
            self.PRF_BO.ResultsOptimization.file,
            len(self.PRF_BO.bounds),
            len(self.PRF_BO.outputs),
            STR_base,
            txtR,
        )

        STR_exec = """
---------------------------------------------------------------------------
 Executing MITIM workflow
---------------------------------------------------------------------------"""

        self.lines = STR_header + STR_inputs + STR_exec

    def getBest(self, rangeT=None):
        if rangeT is None:
            evaluations = self.evaluations
        else:
            evaluations = self.evaluations[rangeT[0] : rangeT[1]]

        xe, ofcaldif, ofcalMdif, res = plotAndGrab(
            None,
            None,
            None,
            None,
            evaluations,
            self.OF_labels,
            None,
            self.PRF_BO.lambdaSingleObjective,
            alpha=0.0,
            alphaM=0.0,
            lab=False,
            OF_labels_complete=self.PRF_BO.stepSettings["name_objectives"],
        )

        best_absolute_index = np.nanargmin(res)
        best_absolute = res[best_absolute_index]

        if rangeT is not None:
            best_absolute_index += rangeT[0]
        best_absolute_full = self.evaluations[best_absolute_index]

        return best_absolute, best_absolute_index, best_absolute_full

    def getFeatures(self, lines):
        self.number_found_points = []
        self.numEvals = []
        for i in range(len(lines)):
            if "Running high-fidelity evaluations for " in lines[i]:
                addImediate = int(lines[i].split("points...")[0].split()[-1])
                for jline in lines[i:]:
                    if "* " in jline:
                        if "after trust region" in jline:  # trust region operation
                            addImediate += int(
                                jline.split("comprised of ")[-1].split("points")[0]
                            )
                        break
                self.numEvals.append(addImediate)

                # Grab how many evaluations this had
                self.number_found_points.append(int(lines[i].split()[4]))

        self.DVbounds = {}
        for i in range(len(lines)):
            if "Input/Design Variables (DVs)" in lines[i]:
                for j, val in enumerate(self.DV_labels):
                    bounds = lines[i + 1 + j].split("from")[1].split("to")
                    self.DVbounds[val] = [float(bounds[0]), float(bounds[1])]

        self.gatherOptima()

        # Gather base (may not be evaluation #0 if I have restarted from Tabular !!!!)
        self.DVbase = {}
        cont_bug = 0
        for i in range(len(lines)):
            if "Baseline point" in lines[i]:
                for j, val in enumerate(self.DV_labels):
                    try:
                        val0 = lines[i + 1 + j + cont_bug].split("=")[1]
                    except:
                        cont_bug += 1
                        val0 = lines[i + 1 + j + cont_bug].split("=")[1]
                    self.DVbase[val] = float(val0)

        # Calculate how DVs change from one iteration to another
        xT = []
        for k in range(len(self.evaluations)):
            xT0 = []
            for j in self.evaluations[k]["x"]:
                xT0.append(self.evaluations[k]["x"][j])
            xT.append(xT0)
        xT = np.array(xT)
        self.DVdistMetric_x, self.DVdistMetric_y = TESTtools.DVdistanceMetric(xT)

    def gatherOptima(self, basedOnBest=True):
        """
        if basedOnBest = False:
                Positions in the array that correspond to first point on the optimized set.
                This is not necesarily the best point in the array, as the evaluation may be better
                for other points, even though they are ordered in predicted set.

        if basedOnBest = True:
                Best in each optimization loop. So, if each optimization provides 5 points, this will search
                within that range. Not TURBO
        """

        self.iterationPositions = np.cumsum(self.numEvals)

        # -----------------------------------------------------------------
        # Best positions in each optimization loop
        # -----------------------------------------------------------------

        self.best_indeces = []
        for i in range(len(self.number_found_points) - 1):
            try:
                rangeT = [
                    self.iterationPositions[i],
                    self.iterationPositions[i] + self.number_found_points[i + 1],
                ]
                best_absolute, best_absolute_index, best_absolute_full = self.getBest(
                    rangeT=rangeT
                )
                self.best_indeces.append(best_absolute_index)
            except:
                break

        # Ensure they are the same size
        diff = len(self.iterationPositions) - len(self.best_indeces)
        for i in range(diff):
            self.best_indeces.append(self.iterationPositions[-diff + i])

        self.best_indeces = np.array(self.best_indeces)

        # -----------------------------------------------------------------
        # Decide where to evaluate
        # -----------------------------------------------------------------

        if not basedOnBest:
            self.optimaPositions = np.append([0], self.iterationPositions)
        else:
            self.optimaPositions = np.append([0], self.best_indeces)
            # print('\t- Results file processed based on best at each iteration')

        # -----------------------------------------------------------------
        # Grab values at the positions indicated previously
        # -----------------------------------------------------------------

        if len(self.evaluations) > 0:
            self.optima = [self.evaluations[0]]
            for i in range(len(self.optimaPositions) - 1):
                try:
                    self.optima.append(self.evaluations[self.optimaPositions[i + 1]])
                except:
                    break

    def plot(
        self, fn=None, doNotShow=True, separateOFs=False, log=None, tab_color=None
    ):
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fn = FigureNotebook("Calibration", geometry="1600x1000")
        else:
            self.fn = fn

        fig1 = self.fn.add_figure(label="Complete", tab_color=tab_color)
        fig1e = self.fn.add_figure(label="Complete (rel.)", tab_color=tab_color)
        fig2 = self.fn.add_figure(label="Metrics", tab_color=tab_color)
        fig3 = self.fn.add_figure(label="Deviations", tab_color=tab_color)
        fig3b = self.fn.add_figure(label="Separate", tab_color=tab_color)
        fig3c = self.fn.add_figure(label="Together", tab_color=tab_color)
        fig3cE = self.fn.add_figure(label="Together All", tab_color=tab_color)
        fig4 = self.fn.add_figure(label="Improvement", tab_color=tab_color)
        if log is not None:
            figTimes = self.fn.add_figure(label="Times", tab_color=tab_color)
            grid = plt.GridSpec(1, 2, hspace=0.3, wspace=0.3)
            axsTimes = [figTimes.add_subplot(grid[0]), figTimes.add_subplot(grid[1])]

        _ = self.plotComplete(
            fig=fig1,
            separateDVs=False,
            separateOFs=separateOFs,
            normalizeToFirstOF=False,
            normalizeDVtoBase=False,
            onlyFinals=False,
        )
        _ = self.plotComplete(
            fig=fig1e,
            separateDVs=False,
            separateOFs=separateOFs,
            normalizeToFirstOF=True,
            normalizeDVtoBase=True,
            onlyFinals=False,
        )
        self.plotMetrics(fig2)
        self.plotCalibrations(figs=[fig3, fig3b, fig3c, fig3cE], tab_color=tab_color)

        grid = plt.GridSpec(1, 3, hspace=0.3, wspace=0.3)
        ax0 = fig4.add_subplot(grid[0, 0])
        GRAPHICStools.addDenseAxis(ax0)
        ax1 = fig4.add_subplot(grid[0, 1], sharex=ax0)
        GRAPHICStools.addDenseAxis(ax1)
        ax2 = fig4.add_subplot(grid[0, 2], sharex=ax0)
        GRAPHICStools.addDenseAxis(ax2)
        _, _ = self.plotImprovement(axs=[ax0, ax1, ax2])

        if log is not None:
            log.plot(axs=[axsTimes[0], axsTimes[1]])

        return self.fn

    def plotDVs(
        self,
        axs=None,
        plotPred=True,
        separateDVs=True,
        addBounds=True,
        normalizeDVtoBase=False,
        onlyFinals=False,
    ):
        colors = GRAPHICStools.listColors()

        if axs is None:
            plt.ion()
            if separateDVs:
                fig, axs = plt.subplots(nrows=self.DV_labels, sharex=True)
            else:
                fig, ax = plt.subplots()
        else:
            ax = axs

        for cont, i in enumerate(self.DV_labels):
            if separateDVs:
                ax = axs[cont]

            # All
            x, y = [], []
            for k in range(len(self.evaluations)):
                x.append(k)
                y.append(self.evaluations[k]["x"][i])
            x, y = np.array(x), np.array(y)

            try:
                y0 = self.DVbase[i]
            except:
                y0 = y[0]

            if normalizeDVtoBase:
                y = (y - y0) / y0 * 100

            if not onlyFinals:
                ax.plot(x, y, "-*", alpha=0.3, c=colors[cont])

            if addBounds:
                if normalizeDVtoBase:
                    try:
                        yl, yu = (self.DVbounds[i][0] - y0) / y0 * 100, (
                            self.DVbounds[i][1] - y0
                        ) / y0 * 100
                    except:
                        print(f"\t- Could not plot normalized DV {i}", typeMsg="w")
                        continue
                else:
                    yl, yu = self.DVbounds[i][0], self.DVbounds[i][1]
                ax.axhline(y=yl, ls="--", c=colors[cont], lw=1)
                ax.axhline(y=yu, ls="--", c=colors[cont], lw=1)

            maxev = x[-1]

            # Optima
            x, y = [], []
            for k in range(len(self.optima)):
                x.append(self.optimaPositions[k])
                y.append(self.optima[k]["x"][i])
            x, y = np.array(x), np.array(y)

            if normalizeDVtoBase:
                y = (y - y0) / y0 * 100

            ax.plot(x, y, "-s", label=i, lw=3, c=colors[cont])

            if separateDVs:
                lab = "Design Variable"
            else:
                lab = "Design Variables"

            if normalizeDVtoBase:
                lab += " (%)"

            ax.set_ylabel(lab)

            if separateDVs:
                GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8)
            ax.set_xlim([0, maxev + 1])
            if not onlyFinals:
                for k in self.optimaPositions:
                    ax.axvline(x=k, lw=0.5, ls="--", c="k")

        if not separateDVs:
            GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8)

        ax.set_xlabel("Iterations")

    def plotOFs(
        self,
        mult=1.0,
        multConfidence=1.0,
        axs=None,
        separateOFs=False,
        plotPred=True,
        normalizeToFirst=False,
        plotModel=True,
        onlyFinals=False,
        onlyThis=None,
        leg=True,
        colorsS=None,
    ):
        colors = GRAPHICStools.listColors() if colorsS is None else [colorsS[0]] * 100
        colorsM = GRAPHICStools.listColors() if colorsS is None else [colorsS[1]] * 100

        if axs is None:
            plt.ion()
            if separateOFs:
                fig, axs = plt.subplots(nrows=self.OF_labels, sharex=True)
            else:
                fig, ax = plt.subplots()
        else:
            ax = axs

        for cont, i in enumerate(self.OF_labels):
            if separateOFs:
                ax = axs[cont]
                GRAPHICStools.addDenseAxis(ax)

            if onlyThis is not None and cont != onlyThis:
                continue
            ilb = i
            # All
            x, y, plotYN = [], [], True
            for k in range(len(self.evaluations)):
                x.append(k)
                yO = self.evaluations[k]["y"][i]

                isNan = False

                yS = yO

                y.append(yS)

            x, y = np.array(x), np.array(y)

            y0 = y[0]

            if normalizeToFirst:
                y = (y - y0) / y0 * 100

            if not onlyFinals:
                ax.plot(
                    x, y[:, 1] * mult, "-*", c=colors[cont], alpha=0.3, markersize=3
                )
                yerr = (
                    np.array([np.abs(y[:, 1] - y[:, 0]), np.abs(y[:, 1] - y[:, 2])])
                    * mult
                    * multConfidence
                )
                ax.errorbar(
                    x,
                    y[:, 1] * mult,
                    c=colors[cont],
                    yerr=yerr,
                    capsize=3.0,
                    fmt="none",
                )

            ymax = np.nanmax(y * mult)

            # Optima
            x, y = [], []
            for k in range(len(self.optima)):
                x.append(self.optimaPositions[k])

                yO = self.optima[k]["y"][i]

                isNan = False

                yS = yO

                y.append(yS)

            x, y = np.array(x), np.array(y)

            if normalizeToFirst:
                y = (y - y0) / y0 * 100

            # MAIN PLOTTING
            ax.plot(
                x,
                y[:, 1] * mult,
                "-s",
                label=i.split("VarPRF_")[-1],
                lw=2,
                c=colors[cont],
                markersize=3,
            )
            yerr = (
                np.array([np.abs(y[:, 1] - y[:, 0]), np.abs(y[:, 1] - y[:, 2])])
                * mult
                * multConfidence
            )
            ax.errorbar(
                x, y[:, 1] * mult, c=colors[cont], yerr=yerr, capsize=3.0, fmt="none"
            )

            # if not onlyFinals:
            # 	for k in self.optimaPositions: ax.axvline(x=k,lw=0.5,ls='--',c='k')

            if plotModel:
                # Model
                x, y = [], []
                for k in range(len(self.optima)):
                    if self.optima[k]["ym"][i] is not None:
                        x.append(self.optimaPositions[k])

                        yO = self.optima[k]["ym"][i]

                        isNan = False

                        yS = yO

                        y.append(yS)

                x, y = np.array(x), np.array(y)

                if len(y) > 0:
                    if normalizeToFirst:
                        y = (y - y0) / y0 * 100
                    ax.plot(
                        x,
                        y[:, 1] * mult,
                        "--o",
                        c=colorsM[cont],
                        label=i + " (model)",
                        lw=0.5,
                        markersize=2,
                    )
                    yerr = (
                        np.array([np.abs(y[:, 1] - y[:, 0]), np.abs(y[:, 1] - y[:, 2])])
                        * mult
                        * multConfidence
                    )

                    ax.errorbar(
                        x,
                        y[:, 1] * mult,
                        c=colorsM[cont],
                        yerr=yerr,
                        capsize=3.0,
                        fmt="none",
                    )

                # Model all
                x, y = [], []
                for k in range(len(self.evaluations)):
                    if self.evaluations[k]["ym"][i] is not None:
                        x.append(k)

                        yO = self.evaluations[k]["ym"][i]

                        isNan = False

                        yS = yO

                        y.append(yS)

                x, y = np.array(x), np.array(y)

                if len(y) > 0:
                    if normalizeToFirst:
                        y = (y - y0) / y0 * 100
                    if not onlyFinals:
                        ax.plot(
                            x,
                            y[:, 1] * mult,
                            "--o",
                            c=colorsM[cont],
                            alpha=0.3,
                            markersize=2,
                            lw=0.3,
                        )
                    if not onlyFinals:
                        ax.errorbar(
                            x,
                            y[:, 1] * mult,
                            c=colorsM[cont],
                            yerr=np.array(
                                [np.abs(y[:, 1] - y[:, 0]), np.abs(y[:, 1] - y[:, 2])]
                            )
                            * mult
                            * multConfidence,
                            capsize=3.0,
                            fmt="none",
                            alpha=0.3,
                        )

            if separateOFs:
                GRAPHICStools.addLegendApart(ax, ratio=0.8, size=6)

        ax.set_xlabel("Iterations")
        if normalizeToFirst:
            ax.set_ylabel("Objective Functions (%)")
            if np.abs(ymax) > 25.0:
                ax.axhline(y=25, lw=0.5, ls="--", c="k")
                ax.axhline(y=-25, lw=0.5, ls="--", c="k")
            elif np.abs(ymax) > 10.0:
                ax.axhline(y=10, lw=0.5, ls="--", c="k")
                ax.axhline(y=-10, lw=0.5, ls="--", c="k")
            if np.abs(ymax) > 50.0:
                ax.axhline(y=50, lw=0.5, ls="--", c="k")
                ax.axhline(y=-50, lw=0.5, ls="--", c="k")
            if np.abs(ymax) > 100.0:
                ax.axhline(y=100, lw=0.5, ls="--", c="k")
                ax.axhline(y=-100, lw=0.5, ls="--", c="k")
        else:
            ax.set_ylabel("Objective Functions")
        if leg:
            if not separateOFs:
                GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8)
        else:
            ax.set_ylabel(ilb)

    def plotComplete(
        self,
        fig,
        multOFs=1.0,
        separateDVs=True,
        separateOFs=False,
        normalizeToFirstOF=False,
        normalizeDVtoBase=False,
        plotModel=True,
        onlyFinals=False,
        addBounds=True,
    ):
        colors = GRAPHICStools.listColors()

        if len(self.DV_labels) > 5:
            # print('Forcing together DVs')
            separateDVs = False

        if separateDVs:
            numD = len(self.DV_labels)
        else:
            numD = 1
        if separateOFs:
            numO = len(self.OF_labels)
        else:
            numO = 1

        grid = plt.GridSpec(nrows=numD + numO, ncols=1, hspace=0.4, wspace=0.4)
        if not separateDVs:
            axDV = fig.add_subplot(grid[0])
        else:
            axDV = []
            for i in range(len(self.DV_labels)):
                axDV.append(fig.add_subplot(grid[i]))

        if not separateOFs:
            axOF = fig.add_subplot(grid[numD], sharex=axDV)
        else:
            axOF = []
            for i in range(len(self.OF_labels)):
                axOF.append(fig.add_subplot(grid[i + numD]))

        self.plotDVs(
            axs=axDV,
            separateDVs=separateDVs,
            normalizeDVtoBase=normalizeDVtoBase,
            onlyFinals=onlyFinals,
            addBounds=addBounds,
        )
        self.plotOFs(
            axs=axOF,
            mult=multOFs,
            separateOFs=separateOFs,
            normalizeToFirst=normalizeToFirstOF,
            plotModel=plotModel,
            onlyFinals=onlyFinals,
        )

        return np.append(axDV, axOF)

    def plotGoodness(self, ax=None, axDiff=None):
        colors = GRAPHICStools.listColors()

        if ax is None:
            plt.ion()
            fig, axs = plt.subplots(nrows=2)
            ax = axs[0]
            axDiff = axs[1]

        for cont, i in enumerate(self.OF_labels):
            if i is None:
                i = ""

            # Model all
            x, y, yl, yu, yActual = [], [], [], [], []
            for k in range(len(self.evaluations)):
                if self.evaluations[k]["ym"][i] is not None and not IOtools.isAnyNan(
                    self.evaluations[k]["ym"][i]
                ):
                    x.append(k)
                    y.append(self.evaluations[k]["ym"][i])
                    yActual.append(self.evaluations[k]["y"][i])
            x, y, yActual = np.array(x), np.array(y), np.array(yActual)

            try:
                ydifference = (
                    np.abs(yActual[:, 1] - y[:, 1]) / np.abs(yActual[:, 1]) * 100.0
                )
            except:
                continue

            if len(y) > 0:
                ax.plot(
                    x, yActual[:, 1], "-*", c=colors[cont], label=i + " (evaluated)"
                )
                ax.plot(
                    x,
                    y[:, 1],
                    "--o",
                    c=colors[cont],
                    alpha=0.3,
                    label=i + " (predicted)",
                )

                yerr = np.array([np.abs(y[:, 1] - y[:, 0]), np.abs(y[:, 1] - y[:, 2])])

                ax.errorbar(
                    x,
                    y[:, 1],
                    c=colors[cont],
                    yerr=yerr,
                    capsize=5.0,
                    fmt="none",
                    alpha=0.3,
                )

                sc = "-*"

                axDiff.plot(x, ydifference, sc, c=colors[cont], label=i)

        for k in self.optimaPositions:
            if k > 0:
                ax.axvline(x=k, lw=0.5, ls="--", c="k")
                axDiff.axvline(x=k, lw=0.5, ls="--", c="k")

        axDiff.set_xlabel("Iterations")
        ax.set_ylabel("Objective Functions")
        axDiff.set_ylabel("Relative Difference (%)")
        GRAPHICStools.addLegendApart(ax, ratio=0.8, size=6, withleg=False)
        GRAPHICStools.addLegendApart(axDiff, ratio=0.8, size=6, loc="center left")
        # axDiff.set_ylim([0,200])
        axDiff.set_yscale("log")

        axDiff.axhline(y=100, ls="--", c="k", lw=2)
        axDiff.axhline(y=50, ls="--", c="k", lw=1)
        axDiff.axhline(y=10, ls="--", c="k", lw=1)
        axDiff.axhline(y=1, ls="--", c="k", lw=1)

    def plotMetrics(self, fig):
        grid = plt.GridSpec(nrows=2, ncols=1, hspace=0.4, wspace=0.4)

        ax1 = fig.add_subplot(grid[0])
        ax2 = fig.add_subplot(grid[1], sharex=ax1)

        self.plotGoodness(ax=ax1, axDiff=ax2)

        return ax1

    def plotCalibrations(self, figs=None, tab_color=None):
        if figs is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fnCals = FigureNotebook("Calibration", geometry="1600x1000")
            fig3 = self.fnCals.add_figure(label="Deviations", tab_color=tab_color)
            fig3b = self.fnCals.add_figure(label="Separate", tab_color=tab_color)
            fig3c = self.fnCals.add_figure(label="Together", tab_color=tab_color)
            fig3c = self.fnCals.add_figure(label="Together All", tab_color=tab_color)
        else:
            [fig3, fig3b, fig3c, fig3cE] = figs

        # ---------------- Plot stuff

        colors = GRAPHICStools.listColors()

        grid = plt.GridSpec(4, 2, hspace=0.3, wspace=0.3)
        ax00 = fig3.add_subplot(grid[0, 0])
        GRAPHICStools.addDenseAxis(ax00)
        ax10 = fig3.add_subplot(grid[1, 0], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax10)
        ax20 = fig3.add_subplot(grid[3, 0], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax20)
        ax01 = fig3.add_subplot(grid[0, 1], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax01)
        ax11 = fig3.add_subplot(grid[1, 1], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax11)
        ax21 = fig3.add_subplot(grid[3, 1], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax21)

        ax20e = fig3.add_subplot(grid[2, 0], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax20e)
        ax21e = fig3.add_subplot(grid[2, 1], sharex=ax00)
        GRAPHICStools.addDenseAxis(ax21e)

        axs = np.array([[ax00, ax01], [ax10, ax11], [ax20, ax21], [ax20e, ax21e]])

        ax = axs[1, 0]
        axl = axs[1, 1]
        axR = axs[3, 0]
        axRl = axs[3, 1]
        x, yT, yTM, res = plotAndGrab(
            ax,
            axl,
            axR,
            axRl,
            self.optima,
            self.OF_labels,
            colors,
            self.PRF_BO.lambdaSingleObjective,
            OF_labels_complete=self.PRF_BO.stepSettings["name_objectives"],
        )
        xe, yTe, yTMe, resM = plotAndGrab(
            ax,
            axl,
            axR,
            axRl,
            self.evaluations,
            self.OF_labels,
            colors,
            self.PRF_BO.lambdaSingleObjective,
            alpha=0.0,
            alphaM=0.0,
            lab=False,
            OF_labels_complete=self.PRF_BO.stepSettings["name_objectives"],
        )

        # Compress evaluations to optima grid (assuming)
        xen = np.interp(xe, self.optimaPositions, np.arange(0, yTM.shape[0] + 1))

        axl.set_yscale("log")

        ms = 3

        # Note: Remember that it is possible that the model mean is not the same as during optimizaiton because of the resolution of the ResultsOptimization points

        ax = axs[2, 0]
        ax.plot(x, yT.mean(axis=1), "-s", label="mean", c="r", markersize=ms)
        ax.plot(x, yT.max(axis=1), "-s", label="max", c="b", markersize=ms)
        ax.plot(x, yTM.mean(axis=1), "--o", c="r", markersize=3)
        ax.plot(x, yTM.max(axis=1), "--o", c="b", markersize=3)

        ax.plot(xen, yTe.mean(axis=1), "-s", c="r", alpha=0.1, markersize=1, lw=0.5)
        ax.plot(xen, yTe.max(axis=1), "-s", c="b", alpha=0.2, markersize=1, lw=0.5)
        ax.plot(xen, yTMe.mean(axis=1), "--o", c="r", alpha=0.1, markersize=1, lw=0.5)
        ax.plot(xen, yTMe.max(axis=1), "--o", c="c", alpha=0.2, markersize=1, lw=0.5)

        best = np.nanmin(yTe.max(axis=1))
        bestM = np.nanmin(yTMe.max(axis=1))
        ax.axhline(y=best, lw=0.1, c="b", ls="-.")
        ax.axhline(y=bestM, lw=0.1, c="r", ls="-.")  # model

        bestmean = np.nanmin(yTe.mean(axis=1))
        # print('Best: ',best,' (max) ',bestmean,' (mean) ', self.best_absolute_index, '(index used)')

        ax = axs[2, 1]
        ax.plot(x, yT.mean(axis=1), "-s", label="mean", c="r", markersize=ms)
        ax.plot(x, yT.max(axis=1), "-s", label="max", c="b", markersize=ms)
        ax.plot(x, yTM.mean(axis=1), "--o", c="r", markersize=3)
        ax.plot(x, yTM.max(axis=1), "--o", c="b", markersize=3)
        ax.set_yscale("log")

        ax.plot(xen, yTe.mean(axis=1), "-s", c="r", alpha=0.1, markersize=1, lw=0.5)
        ax.plot(xen, yTe.max(axis=1), "-s", c="b", alpha=0.2, markersize=1, lw=0.5)
        ax.plot(xen, yTMe.mean(axis=1), "--o", c="r", alpha=0.1, markersize=1, lw=0.5)
        ax.plot(xen, yTMe.max(axis=1), "--o", c="c", alpha=0.2, markersize=1, lw=0.5)

        ax.axhline(y=best, lw=0.1, c="b", ls="-.")
        ax.axhline(y=bestM, lw=0.1, c="r", ls="-.")

        ax = axs[0, 0]
        ax2 = axs[0, 1]
        for cont, ii in enumerate(self.DV_labels):
            y = []
            y2 = []
            x = []
            for i in range(len(self.optima)):
                y2.append(self.optima[i]["x"][ii])
                y.append(self.optima[i]["x"][ii] - self.optima[0]["x"][ii])
                x.append(i)
            y = np.array(y)
            x = np.array(x)
            y2 = np.array(y2)
            ax.plot(x, y, "-s", label=ii, c=colors[cont], markersize=ms)
            ax2.plot(x, y2, "-s", label=ii, c=colors[cont], markersize=ms)

        ax = axs[0, 0]
        ax.set_ylabel("DV values")

        ax = axs[0, 1]
        ax.set_ylabel("DV deviation from base")
        GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8)

        ax = axs[1, 0]
        ax.set_ylabel("Deviation")
        ax.set_ylim(bottom=0)

        ax = axs[3, 0]
        ax.set_ylabel("Relative Deviation (%)")
        ax.set_ylim([0, 100])
        for i in np.arange(10, 100, 10):
            ax.axhline(y=i, ls="--", lw=0.5, c="k")

        ax = axs[1, 1]
        # ax.legend(loc='best',fontsize=8);
        GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8, withleg=False)

        ax.set_ylabel("Deviation")
        # GRAPHICStools.addLegendApart(ax,ratio=1.0,size=8)
        ax = axs[3, 1]
        # ax.legend(loc='best',fontsize=8);
        GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8)
        ax.set_ylabel("Relative Deviation (%)")
        ax.set_yscale("log")

        ax = axs[2, 0]
        ax.set_xlabel("Strategy iterations")
        ax.set_ylabel("Deviation")

        ax = axs[2, 1]
        ax.set_ylabel("Deviation")
        ax.set_xlabel("Strategy iterations")
        GRAPHICStools.addLegendApart(ax, ratio=0.8, size=8, withleg=False)
        ax.legend(loc="best", fontsize=8)

        ax.set_xlim([0, np.max([len(self.optima), 10])])

        # -------------------
        # Plot Separate
        # -------------------
        axs = GRAPHICStools.producePlotsGrid(
            len(self.OF_labels), fig=fig3b, hspace=0.3, wspace=0.4, sharex=True
        )
        for i in range(len(self.OF_labels)):
            self.plotOFs(
                axs=axs[i],
                plotModel=True,
                onlyFinals=True,
                onlyThis=i,
                leg=False,
                colorsS=["b", "r"],
            )

        axs[0].set_xlim(left=0)

        # -------------------
        # Plot Together
        # -------------------
        self.plotCalsTogheter(fig=fig3c, onlyFinals=True, plotModel=False)
        self.plotCalsTogheter(fig=fig3cE, onlyFinals=False, plotModel=True)

    def plotCalsTogheter(
        self,
        fig=None,
        mult=1.0,
        multConfidence=1.0,
        ylabels=None,
        plotModel=True,
        onlyFinals=True,
        legs=["opt", "tar"],
    ):
        if fig is None:
            plt.ion()
            fig = plt.figure(figsize=(13, 9))

        if onlyFinals:
            x, of, cal, ofM, calM = plotAndGrab(
                None,
                None,
                None,
                None,
                self.optima,
                self.OF_labels,
                None,
                self.PRF_BO.lambdaSingleObjective,
                alpha=0.0,
                alphaM=0.0,
                lab=False,
                retrievecomplete=True,
            )
        else:
            x, of, cal, ofM, calM = plotAndGrab(
                None,
                None,
                None,
                None,
                self.evaluations,
                self.OF_labels,
                None,
                self.PRF_BO.lambdaSingleObjective,
                alpha=0.0,
                alphaM=0.0,
                lab=False,
                retrievecomplete=True,
            )

        axs = GRAPHICStools.producePlotsGrid(
            of.shape[1], fig=fig, hspace=0.3, wspace=0.4, sharex=True
        )

        for i in range(of.shape[1]):
            ax = axs[i]

            ax.plot(x, of[:, i], "-o", color="b", label="OF", lw=0.7, markersize=3)
            ax.plot(x, cal[:, i], "-o", color="r", label="CAL", lw=0.7, markersize=3)

            if plotModel:
                ax.plot(
                    x,
                    ofM[:, i],
                    "--*",
                    color="c",
                    label="OF model",
                    lw=0.5,
                    markersize=2,
                )
                ax.plot(
                    x,
                    calM[:, i],
                    "--*",
                    color="m",
                    label="CAL model",
                    lw=0.5,
                    markersize=2,
                )

            if self.PRF_BO.stepSettings["name_objectives"] is not None:
                ax.set_ylabel(self.PRF_BO.stepSettings["name_objectives"][i])
            else:
                ax.set_ylabel(f"Objective Function #{i+1}")

            if onlyFinals:
                ax.set_xlabel("BO iteration")
            else:
                ax.set_xlabel("Evaluation")

        axs[0].set_xlim(left=0)
        axs[0].legend(loc="best", prop={"size": 5})

        for i in range(len(axs)):
            GRAPHICStools.addDenseAxis(axs[i])

    def plotImprovement(
        self,
        axs=None,
        color="b",
        legYN=True,
        extralab="",
        plotAllVlines=True,
        plotAllmembers=False,
        iterationsMultiplier=1.0,
        iterationsOffset=0.0,
        plotMeanMax=[True, True],
        cumulative=True,
        forceName=None,
    ):
        if axs is None:
            plt.ion()
            fig = plt.figure()
            grid = plt.GridSpec(3, 1, hspace=0.3, wspace=0.3)
            ax0 = fig.add_subplot(grid[0, 0])
            ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
            ax2 = fig.add_subplot(grid[2, 0], sharex=ax0)
        else:
            [ax0, ax1, ax2] = axs

        xe, yTe, yTMe, _ = plotAndGrab(
            None,
            None,
            None,
            None,
            self.evaluations,
            self.OF_labels,
            None,
            self.PRF_BO.lambdaSingleObjective,
            alpha=0.0,
            alphaM=0.0,
            lab=False,
        )

        xe = xe * iterationsMultiplier + iterationsOffset

        bestMaxSoFar, bestMeanSoFar = np.inf, np.inf
        yMax, yMean = np.zeros(len(yTe)), np.zeros(len(yTe))
        yCummMax, yCummMean = np.zeros(len(yTe)), np.zeros(len(yTe))
        for i in range(len(yTe)):
            yMax[i] = yTe[i].max()
            yMean[i] = yTe[i].mean()
            bestMaxSoFar = np.min([yTe[i].max(), bestMaxSoFar])
            bestMeanSoFar = np.min([yTe[i].mean(), bestMeanSoFar])
            yCummMax[i] = bestMaxSoFar
            yCummMean[i] = bestMeanSoFar

        if cumulative:
            yPlotMax, yPlotMean = yCummMax, yCummMean
        else:
            yPlotMax, yPlotMean = yMax, yMean

        ax = ax0

        if plotAllmembers:
            if plotMeanMax[1]:
                ax.plot(xe, yMax, "--s", c=color, markersize=1, lw=0.3, alpha=0.3)
            if plotMeanMax[0]:
                ax.plot(xe, yMean, "-s", c=color, markersize=1, lw=0.3, alpha=0.3)

        if plotMeanMax[1]:
            ax.plot(
                xe,
                yPlotMax,
                "--s",
                c=color,
                label=forceName if forceName is not None else extralab + "max",
                markersize=3,
                lw=2.0,
            )
        if plotMeanMax[0]:
            ax.plot(
                xe,
                yPlotMean,
                "-s",
                c=color,
                label=forceName if forceName is not None else extralab + "mean",
                markersize=3,
                lw=1.0,
            )
        ax.set_ylabel("Best Residue (Normalized L1-norm)")
        ax.set_xlabel("High-Fidelity Evaluations")
        if legYN:
            ax.legend(loc="best", prop={"size": 5})
        ax.set_xlim(left=0)
        if np.min(yPlotMax) > 0:
            # ax.set_ylim(bottom=0)
            ax.axhline(y=0, c="k", lw=0.5, ls="-.")

        GRAPHICStools.drawLineWithTxt(
            ax,
            self.optimaPositions[1] - 0.5,
            label="training",
            orientation="vertical",
            color=color,
            lw=0.2,
            ls="-.",
            alpha=1.0,
            fontsize=10,
            fromtop=0.8,
            fontweight="normal",
            separation=0,
            verticalalignment="center",
            horizontalalignment="right",
        )
        GRAPHICStools.drawLineWithTxt(
            ax,
            self.optimaPositions[1] - 0.5,
            label="optimization",
            orientation="vertical",
            color=color,
            lw=0.2,
            ls="-.",
            alpha=1.0,
            fontsize=10,
            fromtop=0.8,
            fontweight="normal",
            separation=0,
            verticalalignment="center",
            horizontalalignment="left",
        )

        if plotAllVlines:
            for i in self.optimaPositions[2:]:
                ax.axvline(x=i - 0.5, ls="-.", lw=0.1, color=color)

        ax = ax1

        if plotAllmembers:
            if plotMeanMax[1]:
                ax.plot(xe, yMax, "--s", c=color, markersize=1, lw=0.3, alpha=0.3)
            if plotMeanMax[0]:
                ax.plot(xe, yMean, "-s", c=color, markersize=1, lw=0.3, alpha=0.3)

        if plotMeanMax[1]:
            ax.plot(
                xe,
                yPlotMax,
                "--s",
                c=color,
                label=forceName if forceName is not None else extralab + "max",
                markersize=2,
                lw=0.5,
            )
        if plotMeanMax[0]:
            ax.plot(
                xe,
                yPlotMean,
                "-s",
                c=color,
                label=forceName if forceName is not None else extralab + "mean",
                markersize=2,
                lw=0.5,
            )
        ax.set_ylabel("Best Residue (Normalized L1-norm)")
        ax.set_xlabel("High-Fidelity Evaluations")
        ax.set_yscale("log")

        GRAPHICStools.drawLineWithTxt(
            ax,
            self.optimaPositions[1] - 0.5,
            label="",
            orientation="vertical",
            color=color,
            lw=0.2,
            ls="-.",
            alpha=1.0,
            fontsize=10,
            fromtop=0.6,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
        )

        if plotAllVlines:
            for i in self.optimaPositions[2:]:
                ax.axvline(x=i - 0.5, ls="-.", lw=0.1, color=color)

        if legYN:
            ax.legend(loc="best", prop={"size": 5})

        # Minimum change in the variable with respect to all previous runs
        if ax2 is not None:
            ax = ax2
            ax.plot(
                self.DVdistMetric_x,
                self.DVdistMetric_y,
                "-s",
                c=color,
                label=extralab + "max",
                markersize=2,
                lw=0.5,
            )
            ax.set_ylabel("$\\Delta$DVs compared to previous (% max)")
            ax.set_xlabel("High-Fidelity Evaluations")
            ax.set_yscale("log")
            ax.set_ylim([1e-2, 1e2])

            for i in [100.0, 10.0, 1.0, 0.1]:
                ax.axhline(y=i, ls="-.", lw=0.5, color="k")

            if plotAllVlines:
                for i in self.optimaPositions[2:]:
                    ax.axvline(x=i - 0.5, ls="-.", lw=0.1, color=color)

        return xe, yPlotMean


def updateSinglePoint_Tabular(inputs, outputs, file, x, y, folderScratch="~/scratch/"):
    TabularData1 = TabularData(inputs, outputs, file=file)

    _, coincidentPoint = TabularData1.grabFromTabular_Specific(x, printStuff=False)
    for iDV in range(len(TabularData1.inputs)):
        try:
            TabularData1.data[coincidentPoint][TabularData1.inputs[iDV]] = x[iDV]
        except:
            embed()
    for iout in range(len(TabularData1.outputs)):
        TabularData1.data[coincidentPoint][TabularData1.outputs[iout]] = y[iout]
    TabularData1.updateFile()


def gatherEv(lines, var=" Evaluation "):
    evaluations = []
    for i in range(len(lines)):
        if var in lines[i]:
            results, cont = {"x": {}, "y": {}, "ym": {}}, 2

            while "y :" not in lines[i + cont]:
                results["x"][lines[i + cont].split()[0]] = float(
                    lines[i + cont].split()[2]
                )
                cont += 1
            cont += 1

            while "y (model) :" not in lines[i + cont]:
                if "L2-norm" not in lines[i + cont].split()[0]:
                    k = 0
                    yl = float(lines[i + cont + k].split("[")[-1].split(",")[0])
                    ym = float(lines[i + cont + k].split()[2])
                    yu = float(lines[i + cont + k].split("]")[0].split(",")[-1])
                    results["y"][lines[i + cont + k].split()[0]] = [yl, ym, yu]
                cont += 1
            cont += 1

            while "y (rel diff, %) :" not in lines[i + cont]:
                for k in range(len(results["y"])):
                    if "y (rel diff, %) :" in lines[i + cont + k]:
                        break
                    if "L2-norm" not in lines[i + cont + k].split()[0]:
                        yl = float(lines[i + cont + k].split("[")[-1].split(",")[0])
                        ym = float(lines[i + cont + k].split()[2])
                        yu = float(lines[i + cont + k].split("]")[0].split(",")[-1])
                        results["ym"][lines[i + cont + k].split()[0]] = [yl, ym, yu]
                cont += 1
            evaluations.append(results)

    return evaluations


def readTabularLines(fileTabular, interface="HIGH_FIDELITY", uniqueNumbering=False):
    """
    if uniqueNumbering: ignore the numbering on the TabularData, extract from 0
    """

    if uniqueNumbering:
        print(
            "\t\t- Careful: I am ignoring the numbering in the tabular file"
        )  # ,typeMsg='w')

    with open(fileTabular, "r") as f:
        aux = f.readlines()

    lines = {}
    number_added = -1
    for i in aux:
        if len(i) > 1:
            if "#" in i:
                header = i
            else:
                if i.split()[1] == interface:
                    if uniqueNumbering:
                        number_added += 1
                        num = number_added
                    else:
                        num = int(i.split()[0])
                    lines[num] = i

    # Organize in dictionary
    DataMITIM = {}
    for i in lines:
        DataMITIM[i] = OrderedDict()
        for cont, j in enumerate(header.split()[2:]):
            try:
                DataMITIM[i][j] = float(lines[i].split()[2 + cont])
            except:
                DataMITIM[i][j] = None

    return lines, DataMITIM


def plot1D(
    ax,
    testX,
    mean,
    upper=None,
    lower=None,
    color="b",
    extralab="",
    alpha=0.2,
    legendConf=True,
    ls="-",
    ms=1,
    lw=1.0,
):
    if legendConf:
        labelMean, labelConf = extralab + "Mean", extralab + "Confidence"
    else:
        labelMean, labelConf = extralab, ""

    contour = ax.plot(testX, mean, ls, c=color, label=labelMean, markersize=ms, lw=lw)
    if upper is not None:
        try:
            ax.fill_between(
                testX, lower, upper, alpha=alpha, label=labelConf, color=color
            )
        except:
            try:
                ax.fill_between(
                    testX[:, 0],
                    lower[:, 0],
                    upper[:, 0],
                    alpha=alpha,
                    label=labelConf,
                    color=color,
                )
            except:
                ax.fill_between(
                    testX[:, 0], lower, upper, alpha=alpha, label=labelConf, color=color
                )
    return contour


def plot2D(
    ax,
    x,
    y,
    z,
    bounds=None,
    levels=None,
    cmap="jet",
    vmaxvmin=None,
    flevels=200,
    cb_fs=8,
    fig=None,
):
    if vmaxvmin is not None:
        maxYplot = vmaxvmin
    else:
        maxYplot = [np.nanmin(z), np.nanmax(z)]

    contour = plotContour(ax, x, y, z, maxYplot, cmap=cmap, cb_fs=cb_fs, fig=fig)
    if levels is not None:
        CS = ax.contour(x, y, z, flevels, levels=levels, colors=["k"])
        ax.clabel(CS, inline=1, fontsize=10)

    if bounds is not None:
        xb = []
        for i in bounds:
            xb.append(i)
        ax.set_xlabel(xb[0])
        ax.set_ylabel(xb[1])

    return contour, maxYplot


def plotContour(ax, x, y, z, maxYplot, cmap="jet", flevels=200, cb_fs=8, fig=None):
    if maxYplot is None:
        maxYplot = [z.min(), z.max()]

    normi = mpl.colors.Normalize(vmin=maxYplot[0], vmax=maxYplot[1])

    try:
        contour = ax.contourf(x, y, z, flevels, cmap=cmap, norm=normi, extend="both")
    except:
        contour = None

    ax.set_ylim([y.min(), y.max()])
    ax.set_xlim([x.min(), x.max()])

    # Colorbar
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(z)
    m.set_clim(maxYplot[0], maxYplot[1])
    GRAPHICStools.addColorbarSubplot(
        ax,
        m,
        fig=fig,
        barfmt="%.1e",
        title="",
        fontsize=cb_fs,
        padCB="20%",
        ylabel="",
        ticks=[],
        orientation="bottom",
        drawedges=False,
    )

    return contour


def plotBO_Pareto(
    resultsSet,
    bounds,
    outputs,
    axs=None,
    fig=None,
    marker="*",
    label="",
    markersize=25,
    plotDVs=True,
    plotBest=True,
    plotLine=True,
    color=None,
    axExtras=None,
):
    if axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    positionsOptimize, positionsNonOptimize = (
        resultsSet["GlobalPositions"]["positionsOptimize"],
        resultsSet["GlobalPositions"]["positionsNonOptimize"],
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting Pareto front (in OFs space)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plotDVs:
        ax = axs[0]
    else:
        ax = axs

    howmany = resultsSet["x_best"].shape[0]
    if color is not None:
        colors, _ = GRAPHICStools.colorTableFade(
            howmany, startcolor=color[0], endcolor=color[1]
        )
    else:
        colors, _ = GRAPHICStools.colorTableFade(
            howmany, startcolor="b", endcolor="r"
        )  # np.random.RandomState(0).rand(howmany)

    # If 2 OFs
    if len(positionsOptimize) == 2:
        y = resultsSet["y_best"]

        yOrder = np.array([a for _, a in sorted(zip(y[:, positionsOptimize[0]], y))])

        ax.scatter(
            yOrder[:, positionsOptimize[0]],
            yOrder[:, positionsOptimize[1]],
            c=colors,
            marker=marker,
            label=label,
            s=markersize,
        )
        if plotBest:
            ax.scatter(
                [y[0, positionsOptimize[0]]],
                [y[0, positionsOptimize[1]]],
                c="r",
                marker=marker,
                s=50,
            )

        if plotLine:
            ax.plot(
                yOrder[:, positionsOptimize[0]],
                yOrder[:, positionsOptimize[1]],
                c=color,
            )
        ax.set_xlabel(outputs[positionsOptimize[0]])
        ax.set_ylabel(outputs[positionsOptimize[1]])
        ax.set_title("Pareto Front")

        if axExtras is not None and len(positionsNonOptimize) > 0:
            for cont, i in enumerate(positionsNonOptimize):
                axExtras[cont].scatter(
                    yOrder[:, positionsOptimize[0]],
                    yOrder[:, i],
                    c=colors,
                    marker=marker,
                    label=label,
                    s=markersize,
                )
                axExtras[cont].set_xlabel(outputs[positionsOptimize[0]])
                axExtras[cont].set_ylabel(outputs[i])

    # If 3 OFs
    if len(positionsOptimize) == 3:
        plotx, ploty, plotz = 1, 2, 0
        x, y, z = (
            resultsSet["y_best"][:, plotx],
            resultsSet["y_best"][:, ploty],
            resultsSet["y_best"][:, plotz],
        )

        [X, Y] = np.meshgrid(np.unique(x), np.unique(y))
        Z = griddata((x, y), z, (X, Y))

        _, vmaxvmin = plot2D(
            ax,
            fig,
            X,
            Y,
            Z,
            bounds=[outputs[plotx], outputs[ploty]],
            cax=None,
            levels=[],
        )
        ax.scatter(
            resultsSet["y_best"][:, plotx],
            resultsSet["y_best"][:, ploty],
            c=colors,
            marker=marker,
            label=label,
        )

    if not plotDVs:
        ax.legend().set_draggable(True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting Pareto front (in DVs space)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if plotDVs:
        dOrder = np.array(
            [
                a
                for _, a in sorted(
                    zip(y[:, positionsOptimize[0]], resultsSet["x_best"])
                )
            ]
        )

        axs[1].scatter(
            dOrder[:, 0],
            dOrder[:, 1],
            c=colors,
            marker=marker,
            label=label,
            s=markersize,
        )
        if plotBest:
            axs[1].scatter(
                [resultsSet["x_best"][0, 0]],
                [resultsSet["x_best"][0, 1]],
                c="r",
                marker=marker,
                s=markersize,
            )
        axs[1].set_title("Corresponding DVs")
        if bounds is not None:
            xb = []
            for i in bounds:
                xb.append(i)
            axs[1].set_xlabel(xb[0])
            axs[1].set_ylabel(xb[1])
        axs[1].legend().set_draggable(True)


# --------------- FoR GA


def plotGA_fronts(
    fronts, toolbox, NumGenerations=5, fig=None, selected=None, trained=None
):
    # ___________________________

    # if len(fronts) == 1: fronts = fronts[0]

    if NumGenerations == -1:
        generations = [-1]
    else:
        sep = int(len(fronts) / NumGenerations)
        try:
            generations = np.arange(0, len(fronts), sep).tolist()
        except:
            generations = [0]
        if generations[-1] < len(fronts) - 1:
            generations.append(len(fronts) - 1)

    plot_colors = GRAPHICStools.listColors()

    numDVs = len(fronts[0][0])
    numOFs = len(toolbox.evaluate(fronts[0][0]))

    reqDVs = int(np.ceil(numDVs / 2))
    reqOFs = int(np.ceil(numOFs / 2))

    if fig is None:
        fig, ax = plt.figure()

    grid = plt.GridSpec(nrows=2, ncols=np.max([reqDVs, reqOFs]), hspace=0.2, wspace=0.2)
    if len(fronts) == 1:
        generations = [0]

    axDVs = []
    for j in range(reqDVs):
        ax = fig.add_subplot(grid[0, j])
        axDVs.append(ax)

    axOFs = []
    for j in range(reqOFs):
        ax = fig.add_subplot(grid[1, j])
        axOFs.append(ax)

    parLasts = []
    for cont, i in enumerate(generations):
        inds = fronts[i]
        par = []
        for contg, ind in enumerate(inds):
            parr = toolbox.evaluate(ind)
            par.append(parr)
            print(
                "Evaluated {0} ({1} out of {2} in generation {3}({5}) out of {4}({6}))".format(
                    ind,
                    contg,
                    len(inds),
                    cont + 1,
                    len(generations),
                    i,
                    generations[-1],
                )
            )
        par = np.array(par)

        if numDVs > 1:
            for j in range(reqDVs):
                axDVs[j].scatter(
                    inds[:, 2 * j],
                    inds[:, 2 * j + 1],
                    color=plot_colors[cont],
                    label="Front " + str(i),
                )
        else:
            for j in range(reqDVs):
                axDVs[j].scatter(
                    inds[:, 2 * j],
                    inds[:, 2 * j],
                    color=plot_colors[cont],
                    label="Front " + str(i),
                )

        if numOFs > 1:
            for j in range(reqOFs):
                axOFs[j].scatter(
                    par[:, 2 * j],
                    par[:, 2 * j + 1],
                    color=plot_colors[cont],
                    label="Front " + str(i),
                )
        else:
            if len(par.shape) > 1:
                for j in range(reqOFs):
                    axOFs[j].scatter(
                        par[:, 2 * j],
                        par[:, 2 * j],
                        color=plot_colors[cont],
                        label="Front " + str(i),
                    )
            else:
                for j in range(reqOFs):
                    axOFs[j].scatter(
                        par[2 * j],
                        par[2 * j],
                        color=plot_colors[cont],
                        label="Front " + str(i),
                    )

        indLast = inds
        parLasts.append(par)

    for j in range(reqDVs):
        ax = axDVs[j]
        GRAPHICStools.addLegendApart(ax, ratio=0.8)
        ax.set_xlabel(f"$x_{2 * j + 1}$")
        ax.set_ylabel(f"$x_{2 * j + 2}$")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    for j in range(reqOFs):
        ax = axOFs[j]
        GRAPHICStools.addLegendApart(ax, ratio=0.8)
        ax.set_xlabel(f"$y_{2 * j + 1}$")
        ax.set_ylabel(f"$y_{2 * j + 2}$")

    if selected is not None:
        allparsel = plotSet(
            selected, toolbox, reqDVs, reqOFs, axDVs, axOFs, numOFs=numOFs
        )
    if trained is not None:
        _ = plotSet(
            trained, toolbox, reqDVs, reqOFs, axDVs, axOFs, numOFs=numOFs, color="k"
        )

    return parLasts, allparsel


def plotSet(
    selected, toolbox, reqDVs, reqOFs, axDVs, axOFs, numOFs=1, marker="*", color="g"
):
    allparsel = []
    for k in range(selected.shape[0]):
        ind = selected[k]
        parr = toolbox.evaluate(ind)
        allparsel.append(parr)
        for j in range(reqDVs):
            ax = axDVs[j]
            ax.scatter(ind[2 * j], ind[2 * j + 1], marker=marker, color=color)

        if numOFs > 1:
            for j in range(reqOFs):
                ax = axOFs[j]
                ax.scatter(parr[2 * j], parr[2 * j + 1], marker=marker, color=color)
        else:
            for j in range(reqOFs):
                ax = axOFs[j]
                ax.scatter(parr[2 * j], parr[2 * j], marker=marker, color=color)

    allparsel = np.array(allparsel)

    return allparsel


def plotGA_fitness(info, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4, 4))

    avg0 = info["avg"]
    min0 = info["min"]
    max0 = info["max"]

    ax.plot(info["gen"], np.abs((info["avg"] - avg0) / avg0) * 100, label="avg")
    ax.plot(info["gen"], np.abs((info["min"] - min0) / min0) * 100, label="min")
    ax.plot(info["gen"], np.abs((info["max"] - max0) / max0) * 100, label="max")

    ax.set_ylabel("Fitness (relative to start, %)")
    ax.set_xlabel("Generation")
    ax.legend()


def plotGA_results(
    fronts,
    info,
    toolbox,
    fn=None,
    NumGenerations=5,
    color="r",
    members=None,
    subname="",
    selected=None,
    trained=None,
):
    plot_colors = GRAPHICStools.listColors()

    if fn is None:
        fn = GUItools.FigureNotebook("MITIM GA Notebook", geometry="1500x1000")

    fig = fn.add_figure(label="Pareto Front" + subname)

    selected = np.array(selected)
    fronts = np.array(fronts)

    parLasts, allparsel = plotGA_fronts(
        fronts,
        toolbox,
        NumGenerations=NumGenerations,
        fig=fig,
        selected=selected,
        trained=trained,
    )

    if members is not None:
        members = np.array(members)
        fig = fn.add_figure(label="Population Members" + subname)
        parLastMs, _ = plotGA_fronts(
            members,
            toolbox,
            NumGenerations=NumGenerations,
            fig=fig,
            selected=selected,
            trained=trained,
        )

    # ------------
    figAnalysis = fn.add_figure(label="Analysis" + subname)
    grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.2, wspace=0.2)
    ax = figAnalysis.add_subplot(grid[0, 0])

    if members is not None:
        for cont, parLastM in enumerate(parLastMs):
            try:
                norms = np.linalg.norm(parLastM, axis=1)
            except:
                norms = np.array([np.linalg.norm(parLastM)])
            x = np.linspace(0, len(norms) - 1, len(norms))
            ax.plot(x, norms, "-s", c="b", label=f"Members Norm {cont + 1}")

    for cont, parLast in enumerate(parLasts):
        try:
            norms = np.linalg.norm(parLast, axis=1)
        except:
            norms = np.array([np.linalg.norm(parLast)])
        x = np.linspace(0, len(norms) - 1, len(norms))
        ax.plot(
            x,
            norms,
            "-*",
            c=plot_colors[cont],
            label=f"Pareto Norm {cont + 1}",
        )

    try:
        norms = np.linalg.norm(allparsel, axis=1)
        for j in norms:
            ax.axhline(y=j, c="g", label="Selected")
    except:
        pass

    ax.legend()

    ax = figAnalysis.add_subplot(grid[1, 0])
    plotGA_fitness(info, ax=ax)

    return fn


def plotGA_essential(GAOF, fn=None, NumGenerations=5, plotAllmembers=False, subname=""):
    if fn is None:
        fn = GUItools.FigureNotebook("MITIM GA Notebook", geometry="1500x1000")

    if plotAllmembers:
        members = GAOF["All_x"]
    else:
        members = None
    fn = plotGA_results(
        GAOF["Paretos_x"],
        GAOF["fitness"],
        GAOF["toolbox"],
        NumGenerations=NumGenerations,
        members=members,
        fn=fn,
        subname=subname,
        selected=GAOF["selected"],
        trained=GAOF["trained"],
    )

    return fn


def printConstraint(constraint_name, constraint, extralab=""):
    print(f"{extralab} * {constraint_name:55} = {constraint}")


def printParam(param_name, param, extralab=""):
    # try:
    # 	print(f'{extralab} * {param_name:42} = {param.item()}')
    # except:
    print(f"{extralab} * {param_name} = ")
    if len(param.shape) > 1:
        for i in range(param.shape[0]):
            printSingleRow(param[i], extralab=extralab + "  ")
    else:
        printSingleRow(param, extralab=extralab + "  ")


def printSingleRow(param, extralab=""):
    if len(param.shape) > 1:
        param = param.squeeze(0)

    if len(param.shape) == 0:
        param = torch.Tensor([param])

    txt = ""
    for j in range(param.shape[0]):
        txt += f"{param[j].item()}, "
    print(extralab, txt[:-2])


def plotAndGrab(
    ax,
    axl,
    axR,
    axRl,
    resu,
    OF_labels,
    colors,
    lambdaSingleObjective,
    alpha=1.0,
    alphaM=1.0,
    lw=1.0,
    lab=True,
    retrievecomplete=False,
    OF_labels_complete=None,
):
    x, yT, yTM = [], [], []

    for i in range(len(resu)):
        x.append(i)
        yT0, yTM0 = [], []
        for ii in OF_labels:
            yT0.append(resu[i]["y"][ii][1])
            yTM0.append(resu[i]["ym"][ii][1])
        yT.append(yT0)
        yTM.append(yTM0)

    yT, yTM, x = np.array(yT), np.array(yTM), np.array(x)

    of, cal, y = lambdaSingleObjective(torch.from_numpy(yT))
    ofM, calM, yM = lambdaSingleObjective(torch.from_numpy(yTM))

    of, cal, y = of.cpu().numpy(), cal.cpu().numpy(), -y.cpu().numpy()
    ofM, calM, yM = ofM.cpu().numpy(), calM.cpu().numpy(), -yM.cpu().numpy()

    ofcaldif = np.abs(of - cal)
    ofcalMdif = np.abs(ofM - calM)

    if ax is not None:
        for cont in range(ofcaldif.shape[1]):
            label = (
                OF_labels_complete[cont]
                if ((OF_labels_complete is not None) and lab)
                else ""
            )

            ax.plot(
                x,
                ofcaldif[:, cont],
                "-s",
                label=label,
                c=colors[cont],
                lw=lw,
                alpha=alpha,
                markersize=5,
            )
            ax.plot(
                x,
                ofcalMdif[:, cont],
                "--o",
                c=colors[cont],
                lw=lw,
                alpha=alphaM,
                markersize=3,
            )

            axl.plot(
                x,
                ofcaldif[:, cont],
                "-s",
                label=label,
                c=colors[cont],
                lw=lw,
                alpha=alpha,
                markersize=5,
            )
            axl.plot(
                x,
                ofcalMdif[:, cont],
                "--o",
                c=colors[cont],
                lw=lw,
                alpha=alphaM,
                markersize=3,
            )

            axR.plot(
                x,
                ofcaldif[:, cont] / np.abs(cal[:, cont]) * 100.0,
                "-s",
                label=label,
                c=colors[cont],
                lw=lw,
                alpha=alpha,
                markersize=5,
            )
            axR.plot(
                x,
                ofcalMdif[:, cont] / np.abs(calM[:, cont]) * 100.0,
                "--o",
                c=colors[cont],
                lw=lw,
                alpha=alphaM,
                markersize=3,
            )

            axRl.plot(
                x,
                ofcaldif[:, cont] / np.abs(cal[:, cont]) * 100.0,
                "-s",
                label=label,
                c=colors[cont],
                lw=lw,
                alpha=alpha,
                markersize=5,
            )
            axRl.plot(
                x,
                ofcalMdif[:, cont] / np.abs(calM[:, cont]) * 100.0,
                "--o",
                c=colors[cont],
                lw=lw,
                alpha=alphaM,
                markersize=3,
            )

    if not retrievecomplete:
        return x, ofcaldif, ofcalMdif, y
    else:
        return x, of, cal, ofM, calM
