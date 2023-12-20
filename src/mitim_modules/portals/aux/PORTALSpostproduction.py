import torch, os, copy
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.opt_tools import OPTtools, STEPtools, STRATEGYtools
from mitim_tools.opt_tools.aux import BOgraphics
from mitim_tools.misc_tools.IOtools import printMsg as print

"""
Technique to reutilize flux surrogates to predict new conditions
----------------------------------------------------------------
Notes:
	* So far only works if Te,Ti,ne
"""


def externalFluxMatch(
    portals_fun,
    FluxesFile,
    writeNextPoint=None,
    plotYN=True,
    RootOnly=False,
    startFrom=None,
):
    prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN=False, askQuestions=False)

    # ----------------------------------------------------
    # Perform operations
    # ----------------------------------------------------

    print(">>>>> Retrain targets", typeMsg="i")

    if startFrom is None:
        startFrom = prf_bo.Optim["BaselineDV"]  # Run a single target at base case
    X = torch.Tensor([i for i in startFrom]).unsqueeze(0)

    prf_bo.mainFunction.powerstate.repeat(batch_size=X.shape[0], includeDerived=False)
    prf_bo.mainFunction.powerstate.modify(X)
    prf_bo.mainFunction.powerstate.calculateProfileFunctions()
    prf_bo.mainFunction.powerstate.calculateTargets()

    Qe_tar = prf_bo.mainFunction.powerstate.plasma["Pe"][:, 1:]
    Qi_tar = prf_bo.mainFunction.powerstate.plasma["Pi"][:, 1:]
    Ge_tar = prf_bo.mainFunction.powerstate.plasma["Ce"][:, 1:]

    # Modify the Y, Yvar of only Y the targets!
    Y = torch.Tensor(size=(X.shape[0], len(prf_bo.outputs)))
    Yvar = torch.Tensor(size=(X.shape[0], len(prf_bo.outputs)))
    for i in range(len(prf_bo.outputs)):
        pos = int(prf_bo.outputs[i].split("_")[1]) - 1
        if "Tar" in prf_bo.outputs[i]:
            if "QeTar" in prf_bo.outputs[i]:
                Y[:, i] = Qe_tar[:, pos]
            if "QiTar" in prf_bo.outputs[i]:
                Y[:, i] = Qi_tar[:, pos]
            if (
                "GeTar" in prf_bo.outputs[i]
                and not prf_bo.mainFunction.PORTALSparameters["forceZeroParticleFlux"]
            ):
                Y[:, i] = Ge_tar[:, pos]
            Yvar[:, i] = (
                Y[:, i] * prf_bo.mainFunction.PORTALSparameters["percentError"][2] / 100
            ) ** 2

    step = STEPtools.OPTstep(
        X.numpy(),
        Y.numpy(),
        Yvar.numpy(),
        bounds=prf_bo.bounds,
        stepSettings=prf_bo.stepSettings,
        surrogate_parameters=prf_bo.surrogate_parameters,
        StrategyOptions=prf_bo.StrategyOptions,
        BOmetrics=prf_bo.BOmetrics,
    )

    # Re-fit Targets
    step.surrogateOptions["extrapointsFile"] = FluxesFile
    step.fit_step(fitWithTrainingDataIfContains="Tar")

    # ----------------------------------------------------
    # Perform flux matching
    # ----------------------------------------------------

    # Use full optimization

    step.stepSettings["Optim"]["minimumResidual"] = 1e-4
    step.stepSettings["Optim"]["relativePerformanceSurrogate"] = None

    step.BOmetrics["overall"]["indBest"] = -1
    step.BOmetrics["overall"]["Residual"] = [np.inf]

    if RootOnly:
        step.optimizers = "root_1"  # Use a one-shot optimization
    step.optimize(
        prf_bo.lambdaSingleObjective, seed=prf_bo.seed, forceAllPointsInBounds=True
    )

    x_opt = torch.from_numpy(step.InfoOptimization[-1]["info"]["x"])
    y_opt_residual = torch.from_numpy(step.InfoOptimization[-1]["info"]["y"])
    xGuesses = torch.from_numpy(step.InfoOptimization[0]["info"]["x_start"])
    obj = step.evaluators["residual_function"]
    acq = step.evaluators["acq_function"]

    # ----------------------------------------------------
    # Plot (for debugging?)
    # ----------------------------------------------------

    if plotYN:
        radii = len(prf_bo.mainFunction.TGYROparameters["RhoLocations"])

        roa = prf_bo.mainFunction.powerstate.plasma["roa"]

        yOut0, yOut_fun0, yOut_cal0, mean0 = obj(xGuesses, outputComponents=True)
        yOut, yOut_fun, yOut_cal, mean = obj(x_opt, outputComponents=True)

        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 8))

        ax = axs[0, 0]
        ax.plot(roa[0, 1:], xGuesses[0, :radii], "o-", c="b")
        ax.plot(roa[0, 1:], x_opt[0, :radii], "o-", c="r")
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{Te}$")
        # ax.set_ylim([0,10])
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[0, 1]
        ax.plot(roa[0, 1:], xGuesses[0, radii : radii * 2], "o-", c="b")
        ax.plot(roa[0, 1:], x_opt[0, radii : radii * 2], "o-", c="r")
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{Ti}$")
        # ax.set_ylim([0,10])
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[0, 2]
        ax.plot(roa[0, 1:], xGuesses[0, radii * 2 :], "o-", c="b")
        ax.plot(roa[0, 1:], x_opt[0, radii * 2 :], "o-", c="r")
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{ne}$")
        # ax.set_ylim([-1,5])
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1, 0]
        ax.plot(
            roa[0, 1:], yOut_fun0.detach()[0, :radii], "o-", c="b", label="Transport"
        )
        ax.plot(roa[0, 1:], yOut_cal0.detach()[0, :radii], "o--", c="b", label="Target")
        ax.plot(roa[0, 1:], yOut_fun.detach()[0, :radii], "o-", c="r")
        ax.plot(roa[0, 1:], yOut_cal.detach()[0, :radii], "o--", c="r")
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q_e$ ($MW/m^2$)")
        GRAPHICStools.addDenseAxis(ax)
        ax.legend()

        ax = axs[1, 1]
        ax.plot(roa[0, 1:], yOut_fun0.detach()[0, radii : radii * 2], "o-", c="b")
        ax.plot(roa[0, 1:], yOut_cal0.detach()[0, radii : radii * 2], "o--", c="b")
        ax.plot(roa[0, 1:], yOut_fun.detach()[0, radii : radii * 2], "o-", c="r")
        ax.plot(roa[0, 1:], yOut_cal.detach()[0, radii : radii * 2], "o--", c="r")
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$Q_i$ ($MW/m^2$)")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1, 2]
        ax.plot(roa[0, 1:], yOut_fun0.detach()[0, radii * 2 :], "o-", c="b")
        ax.plot(roa[0, 1:], yOut_cal0.detach()[0, radii * 2 :], "o--", c="b")
        ax.plot(roa[0, 1:], yOut_fun.detach()[0, radii * 2 :], "o-", c="r")
        ax.plot(roa[0, 1:], yOut_cal.detach()[0, radii * 2 :], "o--", c="r")
        ax.set_xlabel("$r/a$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$C_e$ ($MW/m^2$)")
        GRAPHICStools.addDenseAxis(ax)

        step.GP["individual_models"][0].plot()
        step.GP["individual_models"][2].plot()

        plt.show()

    # ----------------------------------------------------
    # Write In Table
    # ----------------------------------------------------

    X = x_opt[0, :].unsqueeze(0).numpy()

    if writeNextPoint is not None:
        inputs = []
        for i in prf_bo.bounds:
            inputs.append(i)
        TabularData = BOgraphics.TabularData(
            inputs,
            prf_bo.outputs,
            file=writeNextPoint + "TabularData.dat",
            forceNew=True,
        )

        TabularData.updatePoints(X)

        os.system(
            f"cp {writeNextPoint}/TabularData.dat {writeNextPoint}/TabularDataStds.dat"
        )

    return X, step, abs(y_opt_residual.item())
