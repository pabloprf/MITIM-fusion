import torch
import numpy as np
from collections import OrderedDict
import pandas as pd
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

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
    optimization_data = Params["optimization_data"]
    cold_start = Params["cold_start"]
    optimization_object = Params["optimization_object"]

    lock = Params["lock"]

    return mitimRun(
        optimization_object,
        x,
        numEval + cont,
        folderExecution,
        bounds,
        outputs,
        optimization_data,
        cold_start=cold_start,
        lock=lock,
    )


def fun(
    optimization_object,
    x,
    folderExecution,
    bounds,
    outputs,
    optimization_data,
    parallel=1,
    cold_start=True,
    numEval=0,
):
    """
    parallel        = 1:          Just run the evaluations of "x" in series
    parallel        = >1:         Run in parallel batches of X evaluations
    parallel        = -1:         Run the entire batch in parallel

    optimization_object is a class with the method .run(paramsfile,resultsfile)

    cold_start: False if trying to find values first

    """

    (folderExecution / "Execution").mkdir(parents=True, exist_ok=True)

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
    Params["optimization_data"] = optimization_data
    Params["cold_start"] = cold_start
    Params["optimization_object"] = optimization_object
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
    optimization_object,
    x,
    numEval,
    folderExecution,
    bounds,
    outputs,
    optimization_data,
    cold_start=True,
    lock=None,
):
    folderEvaluation = folderExecution / "Execution" / f"Evaluation.{numEval}"
    paramsfile = folderEvaluation / f"params.in.{numEval}"
    resultsfile = folderEvaluation / f"results.out.{numEval}"

    optimization_object.lock = lock

    if (not cold_start) and optimization_data is not None:
        # Read result in Tabular Data
        print("--> Reading Table files...")
        y, yE, _ = optimization_data.grab_data_point(x)

        if pd.Series(y).isna().any() or pd.Series(yE).isna().any():
            print(
                f"--> Reading Tabular file failed or not evaluated yet for element {numEval}",
                typeMsg="i",
            )
            cold_start = True
        else:
            print(
                f"--> Reading Tabular file successful for element {numEval}",
            )

    if cold_start:
        # Create folder
        folderEvaluation.mkdir(parents=True, exist_ok=True)

        # Write params.in.X
        IOtools.writeparams(x, paramsfile, bounds, outputs, numEval)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute evaluation (function must take FolderInputs, Params, Results). It must write results.out.X
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(f"\n>> Running evaluation #{numEval}")
        optimization_object.run(paramsfile, resultsfile)

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

    if optimization_data is not None:
        if lock is not None:
            lock.acquire()
        _,_,objective = optimization_object.scalarized_objective(torch.from_numpy(y))
        optimization_data.update_data_point(x,y,yE,objective=objective.cpu().numpy())
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
