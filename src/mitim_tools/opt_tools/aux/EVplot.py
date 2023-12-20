import sys
import numpy as np
from IPython import embed

from mitim_tools.opt_tools.aux import BOgraphics


def grabBestSpecificParams(
    subfolders,
    evaluations,
    SpecificParams,
    superfolder="./",
    avoidExtraOFs=True,
    alreadyAssigned=False,
):
    if type(subfolders) == str:
        subfolders = [subfolders]

    # None evaluation to find best

    useDict = "evaluations"  #'optima'

    cont = 0
    for ff, ev in zip(subfolders, evaluations):
        res = BOgraphics.ResultsOptimization(
            f"{superfolder}/{ff}/Outputs/ResultsOptimization.out"
        )
        res.read()
        useDict_class = res.__dict__[useDict]

        x, yT, _ = BOgraphics.plotAndGrab(
            None,
            None,
            None,
            None,
            res.evaluations,
            res.calibrations,
            res.summations,
            res.OF_labels,
            None,
        )

        yMax = yT.max(axis=1)
        yMean = yT.mean(axis=1)

        if ev is None:
            if len(yMax) == 0:
                ev, yev, yev2, maxev = 0, 1e6, 1e6, 0
            else:
                ev, yev, yev2, maxev = (
                    x[np.nanargmin(yMean)],
                    yMax[np.nanargmin(yMax)],
                    yMean[np.nanargmin(yMax)],
                    x[-1],
                )
            print(
                '\t- For {0}, read "{5}", and best evaluation found at {1} (/{2}) with a residual of {3:.3f} (max) and {4:.3f} (mean)'.format(
                    ff, ev, maxev, yev, yev2, useDict
                )
            )
        else:
            print(f"\t- For {ff}, using user-specified evaluation {ev}")

        # First equilibrium
        for i in res.DV_labels:
            if "_delta" not in i:
                if i not in SpecificParams or (type(SpecificParams[i]) == float):
                    SpecificParams[i] = []

                if not alreadyAssigned:
                    SpecificParams[i].append(useDict_class[ev]["x"][i])
                else:
                    try:
                        SpecificParams[i][cont] = useDict_class[ev]["x"][i]
                    except:
                        SpecificParams[i].append(useDict_class[ev]["x"][i])

        # Second equilibrium, when double optimization is enabled
        for i in res.DV_labels:
            if "_delta" in i:
                j = i[:-6]
                SpecificParams[j].append(
                    useDict_class[ev]["x"][i] + useDict_class[ev]["x"][j]
                )

        cont += 1

    return SpecificParams, ev
