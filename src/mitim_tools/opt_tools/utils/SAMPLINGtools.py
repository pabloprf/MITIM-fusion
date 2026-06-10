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
        # Honored by older pyDOE versions, which draw from the global np.random state
        np.random.seed(seed)

    # Newer pyDOE (e.g. 0.9.x) draws from numpy's default_rng and IGNORES the
    # global seed, so pass random_state explicitly to keep the `seed` contract;
    # older versions lack the kwarg (TypeError) and honor np.random.seed above.
    try:
        samples_np = pyDOE.lhs(bounds.shape[-1], samples=samples, random_state=seed)
    except TypeError:
        samples_np = pyDOE.lhs(bounds.shape[-1], samples=samples)

    # Adopt the dtype AND device of `bounds`: the scaling loop below assigns
    # bounds-derived values into lhs slices, which fails for a CPU lhs when
    # bounds live on GPU (and downstream acquisition calls need the model device).
    # `bounds` may arrive as a numpy array (e.g. boundsInitialization from the
    # LHS init path) — coerce so `.to()` always receives a tensor; this is a
    # no-op pass-through (no copy, keeps device/dtype) when it already is one.
    bounds = torch.as_tensor(bounds)
    lhs = torch.from_numpy(samples_np).to(bounds)

    for iDV in range(bounds.shape[-1]):
        lhs[:, iDV] = lhs[:, iDV] * (bounds[1, iDV] - bounds[0, iDV]) + bounds[0, iDV]

    return lhs


def readInitializationFile(file, initial_training, labs):

    data = pd.read_csv(file)

    return data[labs].to_numpy()[:initial_training]
