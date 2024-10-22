import os
from mitim_tools.gs_tools import GEQtools
from mitim_tools import __mitimroot__
from matplotlib import pyplot as plt

# Geometry 
R = 1.85
a = 0.57
kappa_sep = 1.97
delta_sep = 0.54
zeta_sep = 0.0
z0 = 0.0

# Plasma
p0_MPa = 2.0
Ip_MA = 8.7
B_T = 12.16

# Equilibrium
f = GEQtools.freegs_millerized(R, a, kappa_sep, delta_sep, zeta_sep, z0)
f.prep(p0_MPa, Ip_MA, B_T, resol_eq = 2**7+1)
f.solve()

# Derive flux surfaces information
f.derive()

# Plot
fig = plt.figure(figsize=(16,7))
axs = fig.subplot_mosaic(
    """
    A12
    A34
    """)

f.plot(axs=[axs['A'], axs['1'], axs['2'], axs['3'], axs['4']])

# Write geqdsk
os.makedirs(os.path.join(__mitimroot__, "tests/scratch/freegs_test/"), exist_ok=True)
f.write(__mitimroot__ + "/tests/scratch/freegs_test/mitim_freegs.geqdsk")

# Write inputs to run TRANSP
f.to_transp(folder = __mitimroot__ + "/tests/scratch/freegs_test/transp_input/")

plt.show()