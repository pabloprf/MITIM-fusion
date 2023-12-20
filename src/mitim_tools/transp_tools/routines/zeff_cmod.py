import sys
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.experiment_tools.CMODtools import getZeff_neo

"""
Routine to get zeff from zeff_neo
e.g.
		zeff_cmod.py 1120815026 1.0 0.1
"""

shotNumber = sys.argv[1]
time = float(sys.argv[2])
avtime = float(sys.argv[3])

zeff, t = getZeff_neo(shotNumber)

print("time")
print(t)
print("Zeff")
print(zeff)

t_interp = np.linspace(t[0], t[-1], 10000)
Zeff_interp = np.interp(t_interp, t, zeff)

it1 = np.argmin(np.abs(t_interp - (time - avtime)))
it2 = np.argmin(np.abs(t_interp - (time + avtime)))

z = np.mean(Zeff_interp[it1:it2])
stra = f"t = {time - avtime:.3f}-{time + avtime:.3f}s, Zeff = {z:.2f}"
print(stra)

plt.ion()
fig, ax = plt.subplots()
ax.plot(t_interp, Zeff_interp, "-s", markersize=3, c="b")
ax.plot(t, zeff, "-s", markersize=5, c="g")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$Z_{eff}$")

ax.axvspan(time - avtime, time + avtime, facecolor="r", alpha=0.2, edgecolor="none")
ax.axhline(y=z, c="r", ls="-", lw=1.0)
ax.set_title(stra)
