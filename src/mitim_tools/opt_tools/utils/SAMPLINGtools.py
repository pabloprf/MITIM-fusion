import torch
import pyDOE
import numpy as np
import pandas as pd
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


def readInitializationFile(file, initial_training, labs):

    data = pd.read_csv(file)

    return data[labs].to_numpy()[:initial_training]
