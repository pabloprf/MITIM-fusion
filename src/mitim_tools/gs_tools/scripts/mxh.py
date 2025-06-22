import argparse
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gs_tools import GEQtools

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
args = parser.parse_args()

file = IOtools.expandPath(args.file)

g = GEQtools.MITIMgeqdsk(file)

pc = {}
for coeffs_MXH in np.arange(3,9):
    pc[coeffs_MXH] = g.to_profiles(coeffs_MXH = coeffs_MXH)


plt.ion()
fig, ax = plt.subplots(ncols = len(pc), figsize = (12,5))
ff = np.linspace(0, 1, 11)

for i, (coeffs_MXH, p) in enumerate(pc.items()):
    p.plot_state_flux_surfaces(ax=ax[i], surfaces_rho=ff, color="b")
    g.plotFluxSurfaces(ax=ax[i], fluxes=ff, rhoPol=False, sqrt=True, color="r", plot1=False)
    ax[i].set_title(f'coeffs_MXH = {coeffs_MXH}')

