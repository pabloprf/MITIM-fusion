import torch
try:
    import pyDOE
except:
    import pydoe as pyDOE
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

    # Adopt the dtype AND device of `bounds`: the scaling loop below assigns
    # bounds-derived values into lhs slices, which fails for a CPU lhs when
    # bounds live on GPU (and downstream acquisition calls need the model device).
    lhs = torch.from_numpy(pyDOE.lhs(bounds.shape[-1], samples=samples)).to(bounds)

    for iDV in range(bounds.shape[-1]):
        lhs[:, iDV] = lhs[:, iDV] * (bounds[1, iDV] - bounds[0, iDV]) + bounds[0, iDV]

    return lhs


def readInitializationFile(file, initial_training, labs):

    data = pd.read_csv(file)

    return data[labs].to_numpy()[:initial_training]
