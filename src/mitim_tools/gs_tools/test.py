import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, MATHtools
from IPython import embed
from mitim_tools.gs_tools import GEQtools

g = GEQtools.MITIMgeqdsk('/Users/hallj/Documents/Files/Research/ARC Modeling/ASTRA-POPCON matching/astra.geqdsk')
Rb, Yb = g.get_MXH_coeff()