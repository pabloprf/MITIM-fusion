import sys
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools


"""
Quick way to compare eq in MXH3 parameterization
e.g.
		compare_MXH3.py input.gacode geq_file [.CDF] [time]
"""

file_input_gacode = sys.argv[1]
p = PROFILEStools.gacode_state(file_input_gacode)

file_geq = sys.argv[2]
g = GEQtools.MITIMgeqdsk(file_geq)


plt.ion()
fig, ax = plt.subplots()

ff = np.linspace(0, 1, 11)

g.plotFluxSurfaces(ax=ax, fluxes=ff, rhoPol=False, sqrt=True, color="r", plot1=False)

p.plot_state_flux_surfaces(ax=ax, surfaces_rho=ff, color="b")

ax.set_xlabel("R (m)")
ax.set_ylabel("Z (m)")
ax.set_aspect("equal")
