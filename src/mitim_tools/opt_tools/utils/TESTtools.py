import torch
import numpy as np
from IPython import embed
from mitim_tools.misc_tools.LOGtools import printMsg as print


def DVdistanceMetric(xT):
    yG = []
    xG = []
    for i in range(xT.shape[0] - 1):
        yA = []
        for j in range(xT[: i + 1, :].shape[0]):
            yA.append(np.abs((xT[i + 1, :] - xT[j, :]) / xT[j, :]) * 100.0)
        yA = np.array(yA)
        # Store the closest difference to an existing point per each dimension
        yG.append(yA.min(axis=0))

        xG.append(i + 1)

    yG = np.array(yG)

    # Calculate the maximum distance in a dimension
    yG_max = yG.max(axis=1) if len(yG) > 0 else yG

    return xG, yG_max


def checkSolutionIsWithinBounds(x, bounds, maxExtrapolation=[0.0, 0.0], clipper = 1E-6):
    mi = bounds[0, :]
    ma = bounds[1, :]

    # Hard limits
    maxb, minb = ma, mi

    # Allow extrapolation (added clip to avoid numerical issues of points very close to the boundary)
    minb = mi - np.max([maxExtrapolation[0],clipper]) * (ma - mi) 
    maxb = ma + np.max([maxExtrapolation[1],clipper]) * (ma - mi)
    insideBounds = (x <= maxb).all() and (x >= minb).all()

    return insideBounds


def testBatchCapabilities(GPs, combinations=[2, 100, 1000]):
    """
    This assesses the relative error in cases where y_Normalized> thrImportance
    It stops running if the error gets larger than thrPercent in those cases
    """

    for i in combinations:
        x = GPs.train_X[0:1, :].repeat(i, 1)

        y1 = GPs.predict(x)[0]
        y2 = GPs.predict(x[0:1, :])[0]

        y1 = y1.detach().mean(axis=0).unsqueeze(0).cpu().numpy()
        y2 = y2.detach().cpu().numpy()

        maxPercent, trouble, indeces = checkSame(
            y1, y2, labels=[f"{i} SAMPLES", "1 SAMPLE"]
        )


def testCombinationCapabilities(GPs, GP):
    x = GP.train_X

    # Combined
    y, _, _, _ = GP.predict(x)

    # Separated
    ys = torch.Tensor().to(x)
    for i in range(len(GPs)):
        y0, _, _, _ = GPs[i].predict(x)
        ys = torch.cat((ys, y0), axis=1)

    # Test
    y, ys = y.detach(), ys.detach()
    err = ((y - ys).abs() / ys * 100).cpu().numpy()

    if np.nanmax(err) > 1e-5:
        print(
            f"\t Max error of combination (check!): {np.nanmax(err):.2f}%", typeMsg="w"
        )
        embed()


def isOutlier(y0, y, stds_outside=5, stds_outside_checker=1):
    mean = y.mean()
    stds = y.std()

    yu = mean + stds_outside * stds
    yl = mean - stds_outside * stds

    outlier = ((y0 < yl) or (y0 > yu)) and (y.shape[0] > stds_outside_checker)

    return outlier


def lookForTrouble(x, y_res, z_res, evaluators, stepSettings, elimintateTroubles=False):
    """
    This is to check that each optimization workflow has calculated correctly the acquisition for each member.
    GPYtorch is not robust enough, check here just in case
    """

    y_res_joint = evaluators["acq_function"](x.unsqueeze(1)).detach()

    y_res_single = torch.Tensor().to(x)
    for i in range(x.shape[0]):
        y = evaluators["acq_function"](x[i].unsqueeze(0).unsqueeze(1)).detach()
        y_res_single = torch.cat((y_res_single, y), axis=0)

    perMax1, trouble1, indeces1 = checkSame(
        y_res, y_res_joint, z=z_res, labels=["OPTIMIZATION", "JOINT"]
    )
    perMax2, trouble2, indeces2 = checkSame(
        y_res, y_res_single, z=z_res, labels=["OPTIMIZATION", "SINGLE"]
    )
    perMax3, trouble3, indeces3 = checkSame(
        y_res_joint, y_res_single, z=z_res, labels=["JOINT", "SINGLE"]
    )

    if len(indeces2) > 0:
        numBad = len(indeces2)
        if elimintateTroubles:
            print(
                "\n\t- It has been requested to eliminate troubled points in positions:",
                indeces2,
            )
            x = np.delete(x, indeces2, axis=0)
            y_res = np.delete(y_res, indeces2, axis=0)
            z_res = np.delete(z_res, indeces2, axis=0)
        else:
            print(
                "\n\t- No action taken, but found troubled points in positions:",
                indeces2,
            )
    else:
        numBad = 0

    return x, y_res, z_res, numBad


def checkSame(
    y1o, y2o, z=None, thresholdTrigger=0.5, absoluteTrigger=1e-3, labels=["", ""]
):
    print(
        f"\t\t- Checking evaluation quality between {labels[0]} and {labels[1]}",
    )

    try:
        y1 = y1o.detach().cpu().numpy()
        y2 = y2o.detach().cpu().numpy()
    except:
        y1 = y1o
        y2 = y2o

    percents, absloutes, absloutes2 = [], [], []
    for i in range(y1.shape[0]):
        per = (np.abs(y1[i] - y2[i]) / y1[i]) * 100.0
        percents.append(per)
        absloutes.append(np.abs(y1[i]))
        absloutes2.append(np.abs(y2[i]))
    percents, absloutes, absloutes2 = (
        np.array(percents),
        np.array(absloutes),
        np.array(absloutes2),
    )

    aError = np.where((percents > thresholdTrigger))[0]
    a = np.where((percents > thresholdTrigger) & (absloutes2 > absoluteTrigger))[0]

    if len(a) == 0:
        if len(aError) == 0:
            print(
                "\t\t\t~ Evaluators provided error in all individuals less than {0:.1e}% (< {1:.1f}%)".format(
                    percents.max(), thresholdTrigger
                ),
            )
        else:
            print(
                "\t\t\t~ Evaluators provided error in all individuals of {0:.1e}% (> {1:.1f}%), but the absolute value is very low".format(
                    percents.max(), thresholdTrigger
                ),
            )
        trouble = False
    else:
        trouble = True
        print(
            "\t\t\t~ Evaluators provided error more than {0:.1f}% in following individuals:".format(
                thresholdTrigger
            ),
            typeMsg="w",
        )
        for i in a:
            if z is not None:
                extratxt = f". Evaluated by {identifyType(z[i])}"
            else:
                extratxt = ""
            try:
                print(
                    "\t\t\t #{0} with an evaluated value of {1:.1e} and error percent of {2:.1f}% (absolute evaluator: {3:.1e}){4}".format(
                        i, absloutes[i], percents[i], absloutes2[i], extratxt
                    ),
                    typeMsg="w",
                )
            except:
                print(
                    "\t\t\t #{0} with an evaluated value of {1} and error percent of {2}% (absolute evaluator: {3}){4}".format(
                        i, absloutes[i], percents[i], absloutes2[i], extratxt
                    ),
                    typeMsg="w",
                )

    return percents.max(), trouble, a


def summaryTypes(z_opt):
    types = ""
    for i in [0, 1, 2, 3, 4, 5]:
        types += f"{(z_opt == i).sum()} from {identifyType(i)}, "

    return types[:-2]


def identifyType(z):
    if z == 0.0:
        method = "Previous Iteration"
    elif z == 1.0:
        method = "Trained"
    elif z == 2.0:
        method = "Random"
    elif z == 3.0:
        method = "BOTORCH"
    elif z == 4.0:
        method = "GA"
    elif z == 5.0:
        method = "ROOT"

    return method
