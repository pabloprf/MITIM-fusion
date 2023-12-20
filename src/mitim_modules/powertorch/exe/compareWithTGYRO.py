import sys, torch
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGYROtools
from mitim_modules.powertorch import STATEtools

"""
This code is useful to get an idea of how well POWERSTATE is calculating the targets, compared to the standard of TGYRO.
To run:
		compareWithTGYRO.py tgyro_folder/
"""


folderTGYRO = IOtools.expandPath(sys.argv[1])

markersize = 5
ls = "o-"


# TGYRO
t = TGYROtools.TGYROoutput(folderTGYRO)
t.profiles.deriveQuantities()

t.useFineGridTargets()


# STATE
s = STATEtools.powerstate(t.profiles, t.rho[0])
s.calculateProfileFunctions()
# s.TargetType = 1
s.calculateTargets()
#

plt.ion()
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 7))

ax = axs[0, 0]

ax.plot(t.rho[0], t.Te[0], "s-", lw=0.5, label="TGYRO Te", markersize=markersize)
ax.plot(
    s.plasma["rho"][0],
    s.plasma["te"][0],
    ls,
    lw=0.5,
    label="STATE Te",
    markersize=markersize,
)
MaxError = (np.abs(t.Te[0] - s.plasma["te"][0].numpy()) / t.Te[0] * 100.0).max()
print(f"{MaxError = :.3f} %")

ax.plot(t.rho[0], t.Ti[0, 0], "s-", lw=0.5, label="TGYRO Ti", markersize=markersize)
ax.plot(
    s.plasma["rho"][0],
    s.plasma["ti"][0],
    ls,
    lw=0.5,
    label="STATE Ti",
    markersize=markersize,
)
MaxError = (np.abs(t.Ti[0, 0] - s.plasma["ti"][0].numpy()) / t.Ti[0, 0] * 100.0).max()
print(f"{MaxError = :.3f} %")

ax.set_ylabel("$keV$")
ax.legend()

ax = axs[1, 0]

ax.plot(t.rho[0], t.ne[0], "s-", lw=0.5, label="TGYRO ne", markersize=markersize)
ax.plot(
    s.plasma["rho"][0],
    s.plasma["ne"][0] * 1e-1,
    ls,
    lw=0.5,
    label="STATE ne",
    markersize=markersize,
)
MaxError = (np.abs(t.ne[0] - s.plasma["ne"][0].numpy() * 1e-1) / t.ne[0] * 100.0).max()
print(f"{MaxError = :.3f} %")

ax.set_ylabel("$10^{20}m^{-3}$")
ax.set_xlabel("$\\rho_N$")
ax.legend()

ax = axs[0, 1]

labels = ["Prad", "Pei", "Pfusi"]
tgyroQuantitys = [-t.Qe_tarMW_rad, -t.Qe_tarMW_exch, t.Qi_tarMW_fus]
stateQuantitys = [s.plasma["qrad"], s.plasma["qie"], s.plasma["qfusi"]]

for tgyroQuantity, stateQuantity, label in zip(tgyroQuantitys, stateQuantitys, labels):
    ax.plot(
        t.rho[0],
        tgyroQuantity[0],
        "s-",
        lw=0.5,
        label="TGYRO " + label,
        markersize=markersize,
    )
    P = s.volume_integrate(stateQuantity, dim=2) * s.plasma["volp"]
    ax.plot(
        s.plasma["rho"][0],
        P[0],
        ls,
        lw=0.5,
        label="STATE " + label,
        markersize=markersize,
    )
    MaxError = np.nanmax(
        np.abs(tgyroQuantity[0] - P[0].numpy()) / tgyroQuantity[0] * 100.0
    )
    print(f"{label} {MaxError = :.3f} %")

ax.set_ylabel("$MW$")
ax.set_xlabel("$\\rho_N$")
ax.legend()

ax = axs[1, 1]

labels = ["Prad_b", "Prad_l", "Prad_s"]
tgyroQuantitys = [-t.Qe_tarMW_brem, -t.Qe_tarMW_line, -t.Qe_tarMW_sync]
stateQuantitys = [s.plasma["qrad_bremms"], s.plasma["qrad_line"], s.plasma["qrad_sync"]]

for tgyroQuantity, stateQuantity, label in zip(tgyroQuantitys, stateQuantitys, labels):
    ax.plot(
        t.rho[0],
        tgyroQuantity[0],
        "s-",
        lw=0.5,
        label="TGYRO " + label,
        markersize=markersize,
    )
    P = s.volume_integrate(stateQuantity, dim=2) * s.plasma["volp"]
    ax.plot(
        s.plasma["rho"][0],
        P[0],
        ls,
        lw=0.5,
        label="STATE " + label,
        markersize=markersize,
    )
    MaxError = np.nanmax(
        np.abs(tgyroQuantity[0] - P[0].numpy()) / tgyroQuantity[0] * 100.0
    )
    print(f"{label} {MaxError = :.3f} %")

ax.set_ylabel("$MW$")
ax.set_xlabel("$\\rho_N$")
ax.legend()


ax = axs[0, 2]

ax.plot(t.rho[0], t.Qe_tar[0], "s-", lw=0.5, label="TGYRO Pe", markersize=markersize)
P = s.plasma["Pe"]
ax.plot(s.plasma["rho"][0], P[0], ls, lw=0.5, label="STATE Pe", markersize=markersize)
MaxError = np.nanmax(np.abs(t.Qe_tarMW[0] - P[0].numpy()) / t.Qe_tarMW[0] * 100.0)
print(f"{MaxError = :.3f} %")

ax.plot(t.rho[0], t.Qi_tar[0], "s-", lw=0.5, label="TGYRO Pi", markersize=markersize)
P = s.plasma["Pi"]
ax.plot(s.plasma["rho"][0], P[0], ls, lw=0.5, label="STATE Pi", markersize=markersize)
MaxError = np.nanmax(np.abs(t.Qi_tarMW[0] - P[0].numpy()) / t.Qi_tarMW[0] * 100.0)
print(f"{MaxError = :.3f} %")

ax.plot(t.rho[0], t.Ce_tar[0], "s-", lw=0.5, label="TGYRO Ce", markersize=markersize)
P = s.plasma["Ce"]
ax.plot(s.plasma["rho"][0], P[0], ls, lw=0.5, label="STATE Ce", markersize=markersize)
MaxError = np.nanmax(np.abs(t.Ce_tarMW[0] - P[0].numpy()) / t.Ce_tarMW[0] * 100.0)
print(f"{MaxError = :.3f} %")

ax.set_ylabel("$MW/m^2$")
ax.set_xlabel("$\\rho_N$")
ax.legend()

ax = axs[1, 2]

ax.plot(t.rho[0], t.Qe_tarMW[0], "s-", lw=0.5, label="TGYRO Pe", markersize=markersize)
P = s.plasma["Pe"] * s.plasma["volp"]
ax.plot(s.plasma["rho"][0], P[0], ls, lw=0.5, label="STATE Pe", markersize=markersize)
MaxError = np.nanmax(np.abs(t.Qe_tarMW[0] - P[0].numpy()) / t.Qe_tarMW[0] * 100.0)
print(f"{MaxError = :.3f} %")

ax.plot(t.rho[0], t.Qi_tarMW[0], "s-", lw=0.5, label="TGYRO Pi", markersize=markersize)
P = s.plasma["Pi"] * s.plasma["volp"]
ax.plot(s.plasma["rho"][0], P[0], ls, lw=0.5, label="STATE Pi", markersize=markersize)
MaxError = np.nanmax(np.abs(t.Qi_tarMW[0] - P[0].numpy()) / t.Qi_tarMW[0] * 100.0)
print(f"{MaxError = :.3f} %")

ax.plot(t.rho[0], t.Ce_tarMW[0], "s-", lw=0.5, label="TGYRO Ce", markersize=markersize)
P = s.plasma["Ce"] * s.plasma["volp"]
ax.plot(s.plasma["rho"][0], P[0], ls, lw=0.5, label="STATE Ce", markersize=markersize)
MaxError = np.nanmax(np.abs(t.Ce_tarMW[0] - P[0].numpy()) / t.Ce_tarMW[0] * 100.0)
print(f"{MaxError = :.3f} %")

ax.set_ylabel("$MW$")
ax.set_xlabel("$\\rho_N$")
ax.legend()
