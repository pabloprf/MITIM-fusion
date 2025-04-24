import argparse
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.powertorch import STATEtools

"""
This code is useful to get an idea of how well POWERSTATE is calculating the targets, compared to the standard of TGYRO.
To run:
		compareRadialResolution.py --file input.gacode --rhos 0.30771427 0.48583669 0.67665776 0.75750399 0.84737053
"""

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, type=str)
parser.add_argument(
    "--rhos",
    required=False,
    default=[0.0000, 0.4327, 0.5511, 0.6997, 0.7759, 0.8587],
    nargs="+",
)
parser.add_argument("--res", required=False, type=int, default=100)
args = parser.parse_args()


inputgacode = IOtools.expandPath(args.file)
rho = np.array([float(i) for i in args.rhos])

profiles = PROFILEStools.PROFILES_GACODE(inputgacode)

markersize_coarse = 6
markersize_fine = 3

ls = "o-"

sC = STATEtools.powerstate(profiles,EvolutionOptions={"rhoPredicted": rho},)
sC.calculateProfileFunctions()
sC.calculateTargets()

# Full state
rho = np.linspace(rho[0], rho[-1], args.res)

sF = STATEtools.powerstate(profiles,EvolutionOptions={"rhoPredicted": rho})
sF.calculateProfileFunctions()
sF.calculateTargets()

plt.close("all")
plt.ion()
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 7))

ax = axs[0, 0]

s, lab = sF, "Fine "
ax.plot(
    s.plasma["rho"][0],
    s.plasma["te"][0],
    ls,
    lw=0.5,
    label=f"{lab}Te",
    markersize=markersize_fine,
)
ax.plot(
    s.plasma["rho"][0],
    s.plasma["ti"][0],
    ls,
    lw=0.5,
    label=f"{lab}Ti",
    markersize=markersize_fine,
)


s, lab = sC, "Coarse "
ax.plot(
    s.plasma["rho"][0],
    s.plasma["te"][0],
    ls,
    lw=0.5,
    label=f"{lab}Te",
    markersize=markersize_coarse,
)
ax.plot(
    s.plasma["rho"][0],
    s.plasma["ti"][0],
    ls,
    lw=0.5,
    label=f"{lab}Ti",
    markersize=markersize_coarse,
)

ax.set_xlabel("$\\rho_N$")
ax.set_ylabel("$keV$")
ax.legend()

ax = axs[1, 0]

s, lab = sF, "Fine "
ax.plot(
    s.plasma["rho"][0],
    s.plasma["ne"][0] * 1e-1,
    ls,
    lw=0.5,
    label=f"{lab}ne",
    markersize=markersize_fine,
)


s, lab = sC, "Coarse "
ax.plot(
    s.plasma["rho"][0],
    s.plasma["ne"][0] * 1e-1,
    ls,
    lw=0.5,
    label=f"{lab}ne",
    markersize=markersize_coarse,
)

ax.set_xlabel("$\\rho_N$")
ax.set_ylabel("$10^{20}m^{-3}$")
ax.legend()


ax = axs[1, 1]
varsS = ["qrad", "qie", "qfusi"]

s, lab = sF, "Fine "
for var in varsS:
    ax.plot(
        s.plasma["rho"][0],
        s.plasma[var][0],
        ls,
        lw=0.5,
        label=f"{lab}{var}",
        markersize=markersize_fine,
    )

s, lab = sC, "Coarse "
for var in varsS:
    ax.plot(
        s.plasma["rho"][0],
        s.plasma[var][0],
        ls,
        lw=0.5,
        label=f"{lab}{var}",
        markersize=markersize_coarse,
    )

ax.set_xlabel("$\\rho_N$")
ax.set_ylabel("$MW/m^3$")
ax.legend()

ax = axs[0, 1]
varsS = ["QeMWm2", "QiMWm2"]

s, lab = sF, "Fine "
for var in varsS:
    ax.plot(
        s.plasma["rho"][0],
        s.plasma[var][0],
        ls,
        lw=0.5,
        label=f"{lab}{var}",
        markersize=markersize_fine,
    )

s, lab = sC, "Coarse "
for var in varsS:
    ax.plot(
        s.plasma["rho"][0],
        s.plasma[var][0],
        ls,
        lw=0.5,
        label=f"{lab}{var}",
        markersize=markersize_coarse,
    )

ax.set_xlabel("$\\rho_N$")
ax.set_ylabel("$MW/m^2$")
ax.legend()
