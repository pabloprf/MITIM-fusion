import torch, pyDOE
import numpy as np
from mitim_tools.opt_tools.aux import BOgraphics
from IPython import embed


def LHS(samples, bounds, seed=0):
    """
    bounds must be tensor (2,dimDVs)
    samples is the number of points created, so lhs will be (samples,dimDVs)
    """

    if seed is not None:
        np.random.seed(seed)

    lhs = torch.from_numpy(pyDOE.lhs(bounds.shape[-1], samples=samples))

    for iDV in range(bounds.shape[-1]):
        lhs[:, iDV] = lhs[:, iDV] * (bounds[1, iDV] - bounds[0, iDV]) + bounds[0, iDV]

    return lhs


def readInitializationFile(file, initialPoints, labs):
    train_X = []
    _, datTab = BOgraphics.readTabularLines(file)
    for i in range(initialPoints):
        xi = []
        for lb in labs:
            xi.append(datTab[i][lb])
        train_X.append(xi)

    return np.array(train_X)
