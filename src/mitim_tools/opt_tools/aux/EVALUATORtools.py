import os
import numpy as np
from collections import OrderedDict
from IPython import embed
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.opt_tools.aux import BOgraphics

from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS THAT HANDLE PARELELIZATION AND MITIM-SPECIFIC FEATURES (ALLOWING GENERALIZATION TO ANY FUNCTION)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DTYPE = np.float64


def parallel_main(Params, cont):
    x = Params["x"][cont]
    numEval = Params["numEval"]
    folderExecution = Params["folderExecution"]
    bounds = Params["bounds"]
    outputs = Params["outputs"]
    TabularData = Params["TabularData"]
    TabularDataStds = Params["TabularDataStds"]
    restartYN = Params["restartYN"]
    mainFunction = Params["mainFunction"]

    lock = Params["lock"]

    return mitimRun(
        mainFunction,
        x,
        numEval + cont,
        folderExecution,
        bounds,
        outputs,
        TabularData,
        TabularDataStds,
        restartYN=restartYN,
        lock=lock,
    )


def fun(
    mainFunction,
    x,
    folderExecution,
    bounds,
    outputs,
    TabularData,
    TabularDataStds,
    parallel=1,
    restartYN=True,
    numEval=0,
):
    """
    parallel        = 1:          Just run the evaluations of "x" in series
    parallel        = >1:         Run in parallel batches of X evaluations
    parallel        = -1:         Run the entire batch in parallel

    mainFunction is a class with the method .run(paramsfile,resultsfile)

    restartYN: False if trying to find values first

    """

    if not os.path.exists(folderExecution + "/Execution/"):
        os.mkdir(folderExecution + "/Execution/")

    try:
        x = np.atleast_2d(x)
    except:
        x = np.atleast_2d(x.cpu())
    offset = numEval
    y = OrderedDict()
    if parallel < 0:
        parallel = len(x)

    Params = {}
    Params["x"] = x
    Params["numEval"] = numEval
    Params["folderExecution"] = folderExecution
    Params["bounds"] = bounds
    Params["outputs"] = outputs
    Params["TabularData"] = TabularData
    Params["TabularDataStds"] = TabularDataStds
    Params["restartYN"] = restartYN
    Params["mainFunction"] = mainFunction
    Params["lock"] = None

    if parallel == 1:
        res = FARMINGtools.SerialProcedure(parallel_main, Params, howmany=len(x))
        ynew, yEnew = np.array(res)
    else:
        res = FARMINGtools.ParallelProcedure(
            parallel_main, Params, parallel=parallel, howmany=len(x), array=False
        )

        # ----------------------------------------------------
        ynew, yEnew = [], []
        for i in range(len(res)):
            ynew.append(res[i][0])
            yEnew.append(res[i][1])
        ynew, yEnew = np.array(ynew), np.array(yEnew)
        # ----------------------------------------------------

    print("--> Evaluator function has finished!")

    vectorEvaluations = np.arange(offset, offset + len(x))

    numEval_new = vectorEvaluations[-1] + 1

    return ynew, yEnew, numEval_new


def mitimRun(
    mainFunction,
    x,
    numEval,
    folderExecution,
    bounds,
    outputs,
    TabularData,
    TabularDataStds,
    restartYN=True,
    lock=None,
):
    folderEvaluation = folderExecution + f"/Execution/Evaluation.{numEval}/"
    paramsfile = f"{folderEvaluation}/params.in.{numEval}"
    resultsfile = f"{folderEvaluation}/results.out.{numEval}"

    mainFunction.lock = lock

    if (not restartYN) and TabularData is not None:
        # Read result in Tabular Data
        print("--> Reading Table files...", verbose=verbose_level)
        y, _ = TabularData.grabFromTabular_Specific(x)
        yE, _ = TabularDataStds.grabFromTabular_Specific(x)

        if np.isnan(y).any() or np.isnan(yE).any():
            print(
                "--> Reading Tabular file failed or not evaluated yet for element {0}".format(
                    numEval
                ),
                typeMsg="w",
            )
            restartYN = True
        else:
            print(
                f"--> Reading Tabular file successful for element {numEval}",
                verbose=verbose_level,
            )

    if restartYN:
        # Create folder
        if not os.path.exists(folderEvaluation):
            os.mkdir(folderEvaluation)

        # Write params.in.X
        IOtools.writeparams(x, paramsfile, bounds, outputs, numEval)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute evaluation (function must take FolderInputs, Params, Results). It must write results.out.X
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(f"\n>> Running evaluation #{numEval}")
        mainFunction.run(paramsfile, resultsfile)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Read results.out.X

        try:
            y, yE = IOtools.readresults(resultsfile)
        except:
            print(
                "Could not read results file for {0}, printing error file to screen".format(
                    numEval
                )
            )
            # print(error)
            y, yE = np.array([np.nan]), np.array([np.nan])

    # Write results into Tabular Data, but make sure it is only one process at a time

    inputs = []
    for i in bounds:
        inputs.append(i)

    if TabularData is not None:
        if lock is not None:
            lock.acquire()
        BOgraphics.updateSinglePoint_Tabular(inputs, outputs, TabularData.file, x, y)
        if lock is not None:
            lock.release()

    if TabularDataStds is not None:
        if lock is not None:
            lock.acquire()
        BOgraphics.updateSinglePoint_Tabular(
            inputs, outputs, TabularDataStds.file, x, yE
        )
        if lock is not None:
            lock.release()

    try:
        y_txt = ""
        for i in range(y.shape[0]):
            y_txt += "\t\ty{0} = {1:.5f}, yE{0.5f} = {2:.5f}\n".format(i, y[i], yE[i])
        print(f"\n~ Evaluation.{numEval} result:\n", y_txt)
    except:
        pass

    return y, yE
