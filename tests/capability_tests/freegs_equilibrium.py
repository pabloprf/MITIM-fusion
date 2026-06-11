"""
CAPABILITY: Free-boundary equilibrium with FreeGS from shape parameters
-----------------------------------------------------------------------
This script teaches how to build a Grad-Shafranov equilibrium from scratch —
no input files needed — using the FreeGS solver wrapped by MITIM, starting
from a Miller-like parametrization of the separatrix. This runs locally and
in seconds: it is how MAESTRO bootstraps an equilibrium before any transport
code is involved.

Key teaching points:
    1. freegs_millerized(R, a, kappa_sep, delta_sep, zeta_sep, z0) defines the
       target separatrix shape; prep() sets the plasma quantities (pressure on
       axis, plasma current, vacuum field) and the solver grid resolution;
       solve() finds the free-boundary equilibrium that matches the shape.
    2. derive() computes the flux-surface information (geometry coefficients,
       q profile, ...) needed by everything downstream.
    3. The resulting equilibrium can be exported: write() produces a standard
       geqdsk file, and to_transp() writes the input files to start a TRANSP
       run from it.
    4. Plotting here uses plain matplotlib axes (no FigureNotebook): MITIM
       plot methods generally accept user-provided axes/figures, so they can
       be embedded in custom figures.
"""

from matplotlib import pyplot as plt
from mitim_tools.gs_tools import GEQtools
from mitim_tools import __mitimroot__

# Working folder: the exported geqdsk and TRANSP inputs are written here
folder = __mitimroot__ / "tests" / "scratch" / "capability_freegs"
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Define the separatrix shape and the plasma, and solve the equilibrium
# ---------------------------------------------------------------------------------------------------------------------

# Geometry (Miller-like separatrix parametrization, SPARC-like values)
R = 1.85          # major radius (m)
a = 0.57          # minor radius (m)
kappa_sep = 1.97  # separatrix elongation
delta_sep = 0.54  # separatrix triangularity
zeta_sep = 0.0    # separatrix squareness
z0 = 0.0          # vertical position of the magnetic axis (m)

# Plasma
p0_MPa = 2.0      # pressure on axis (MPa)
Ip_MA = 8.7       # plasma current (MA)
B_T = 12.16       # vacuum toroidal field at R (T)

f = GEQtools.freegs_millerized(R, a, kappa_sep, delta_sep, zeta_sep, z0)

# resol_eq is the number of grid points per direction of the solver (2^n + 1)
f.prep(p0_MPa, Ip_MA, B_T, resol_eq=2**7 + 1)

# Solve the free-boundary Grad-Shafranov problem
f.solve()

# Compute the flux-surface information (geometry, q profile, ...)
f.derive()

# ---------------------------------------------------------------------------------------------------------------------
# 2. Plot (equilibrium cross-section + profiles, in user-provided axes)
# ---------------------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 7))
axs = fig.subplot_mosaic(
    """
    A12
    A34
    """
)
f.plot(axs=[axs["A"], axs["1"], axs["2"], axs["3"], axs["4"]])

# ---------------------------------------------------------------------------------------------------------------------
# 3. Export: geqdsk file and TRANSP inputs
# ---------------------------------------------------------------------------------------------------------------------

# Standard geqdsk, readable by any equilibrium tool (and by mitim_plot_eq)
f.write(folder / "mitim_freegs.geqdsk")

# Input files to start a TRANSP run from this equilibrium
f.to_transp(folder=folder / "transp_input")

plt.show()
