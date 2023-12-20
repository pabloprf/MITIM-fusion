import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.misc_tools import MATHtools
from mitim_tools.misc_tools.IOtools import printMsg as print


def doubleLinear_aLT(x, g1, g2, t, T1):
    """
    Must be (batch,dimx)
    """

    g1 = np.transpose(np.atleast_2d(g1))

    ix = np.argmin(np.abs(x[0, :] - t))

    # Edge
    xi = x[:, ix:]
    lnT = g1 * ((xi - 1) * (-2 * t + xi + 1) / (2 * (t - 1))) + g2 * (
        (xi - 1) * (1 - xi) / (2 * (t - 1))
    )
    Tedge = T1 * np.exp(lnT)

    # Core
    xi = x[:, : ix + 1]
    lnT = g2 * (t**2 - xi**2) / (2 * t)
    Tcore = T1 * np.e ** (0.5 * (1 - t) * (g1 + g2)) * np.exp(lnT)

    # Full
    T = np.hstack((Tcore[:, :-1], Tedge))

    return T


def calculate_simplified_volavg(x, T):
    x = np.atleast_2d(x)
    dVdr = 2 * x
    vol = CALCtools.integrateQuadPoly(torch.from_numpy(x), torch.ones(x.shape) * dVdr)

    Tvol = (
        CALCtools.integrateQuadPoly(torch.from_numpy(x), torch.from_numpy(T) * dVdr)[
            :, -1
        ]
        / vol[:, -1]
    ).numpy()

    return Tvol
