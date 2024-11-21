"""
Routine to merge Ti at a given location

use:
	mod_Ti.py 0.85 200(eV to increase) 1.0(plot) 0.1(avTime)

"""

import sys, os, copy, shutil
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools import UFILEStools, CDFtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

merge_location = float(sys.argv[1])  # 0.85
Ti_offset = float(sys.argv[2])  # 0.85
try:
    plotTime = float(sys.argv[3])
except:
    plotTime = 1.0
try:
    avTime = float(sys.argv[4])
except:
    avTime = 0.1


telab = "TEL"
tilab = "TIO"

telabfile = IOtools.expandPath(f"./MIT12345.{telab}")
tilabfile = IOtools.expandPath(f"./MIT12345.{tilab}")
tilaboldfile = IOtools.expandPath(f"./MIT12345.{tilab}_old")
tilabtestfile = IOtools.expandPath(f"./MIT12345.{tilab}_test")

timesplot = np.linspace(plotTime - avTime, plotTime + avTime, 100)

ufTe = UFILEStools.UFILEtransp()
ufTe.readUFILE(telabfile)

if tilabfile.exists():
    shutil.copy2(tilabfile, tilaboldfile)

ufTi = UFILEStools.UFILEtransp()
ufTi.readUFILE(tilabfile)

# Ti in Te time scale
newZ = []
for ix in range(ufTi.Variables["X"].shape[0]):
    newZ.append(
        np.interp(ufTe.Variables["Y"], ufTi.Variables["Y"], ufTi.Variables["Z"][ix, :])
    )

ufTi.Variables["Z"] = np.array(newZ) + Ti_offset
ufTi.Variables["Y"] = ufTe.Variables["Y"]


ufTi_new = copy.deepcopy(ufTi)
for it in range(ufTi.Variables["Y"].shape[0]):
    ixTi = np.argmin(np.abs(ufTi.Variables["X"] - merge_location))
    ixTe = np.argmin(np.abs(ufTe.Variables["X"] - merge_location))

    ufTi_new.Variables["Z"][ixTi:, it] = np.interp(
        ufTi.Variables["X"][ixTi:],
        ufTe.Variables["X"][ixTe:],
        ufTe.Variables["Z"][ixTe:, it],
    )

ufTi_new.Variables["Z"].transpose()
ufTi_new.writeUFILE(tilabtestfile)


ufTi_test = UFILEStools.UFILEtransp()
ufTi_test.readUFILE(tilabtestfile)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))

ax = axs[0, 0]
# ufTe.plotVar(ax=ax,axisSlice='Y',val=plotTime)
# ufTi.plotVar(ax=ax,axisSlice='Y',val=plotTime)
for t in timesplot:
    ufTe.plotVar(ax=ax, axisSlice="Y", val=t, lw=0.1, ms=0, alpha=0.4)
    ufTi.plotVar(ax=ax, axisSlice="Y", val=t, lw=0.1, ms=0, alpha=0.4)

it1 = np.argmin(np.abs(timesplot[0] - ufTi.Variables["Y"]))
it2 = np.argmin(np.abs(timesplot[-1] - ufTi.Variables["Y"]))
prof = []
for ix in range(len(ufTi.Variables["X"])):
    prof.append(
        CDFtools.timeAverage(
            ufTi.Variables["Y"][it1:it2], ufTi.Variables["Z"][ix, it1:it2]
        )
    )
ax.plot(ufTi.Variables["X"], prof)
profTe = []
for ix in range(len(ufTe.Variables["X"])):
    profTe.append(
        CDFtools.timeAverage(
            ufTe.Variables["Y"][it1:it2], ufTe.Variables["Z"][ix, it1:it2]
        )
    )
ax.plot(ufTe.Variables["X"], profTe)


ax = axs[0, 1]
ufTe.plotVar(ax=ax, axisSlice="X", axis="Y", val=merge_location + 0.1)
ufTi.plotVar(ax=ax, axisSlice="X", axis="Y", val=merge_location + 0.1)
ax.set_title(f"Position X={merge_location + 0.1:.2f}")

ax = axs[1, 0]
# ufTe.plotVar(ax=ax,axisSlice='Y',val=plotTime)
# ufTi_test.plotVar(ax=ax,axisSlice='Y',val=plotTime)
for t in timesplot:
    ufTe.plotVar(ax=ax, axisSlice="Y", val=t, lw=0.1, ms=0, alpha=0.4)
    ufTi_test.plotVar(ax=ax, axisSlice="Y", val=t, lw=0.1, ms=0, alpha=0.4)

it1 = np.argmin(np.abs(timesplot[0] - ufTi_test.Variables["Y"]))
it2 = np.argmin(np.abs(timesplot[-1] - ufTi_test.Variables["Y"]))
prof = []
for ix in range(len(ufTi_test.Variables["X"])):
    prof.append(
        CDFtools.timeAverage(
            ufTi_test.Variables["Y"][it1:it2], ufTi_test.Variables["Z"][ix, it1:it2]
        )
    )
ax.plot(ufTi_test.Variables["X"], prof)
ax.plot(ufTe.Variables["X"], profTe)

ax = axs[1, 1]
ufTe.plotVar(ax=ax, axisSlice="X", axis="Y", val=merge_location + 0.1)
ufTi_test.plotVar(ax=ax, axisSlice="X", axis="Y", val=merge_location + 0.1)

plt.show()

print("Do you want to change Ti? (insert exit)", typeMsg="q")
embed()

tilabtestfile.replace(tilabfile)
