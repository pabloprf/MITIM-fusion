import sys
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.experiment_tools import CMODtools

"""
Routine to compare cmod cases to MDS+

e.g.
	interpret_cmod.py 1160714032 1.15 0.05 89884C01.CDF

"""

# ---- Inputs ----------------
shotNumber = int(sys.argv[1])
time = float(sys.argv[2])
avTime = float(sys.argv[3])
runs = sys.argv[4:]
# ----------------------------

# ----- Reading data: Simulations
cs = []
lab = []
for file in runs:
    cs.append(CDFtools.transp_output(file))
    lab.append(file.split("/")[-1].split(".")[0])

# ----- Reading data: MDS+ experimental data
exp = CMODtools.CMODexperiment(shotNumber)

# ----- Plotting
colors = GRAPHICStools.listColors()
plt.ion()
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(17, 7))

ax = axs[0, 0]
axt = ax.twinx()
for cont, c in enumerate(cs):
    it1 = np.argmin(np.abs(c.t - time + avTime))
    it2 = np.argmin(np.abs(c.t - time - avTime))
    T = CDFtools.timeAverage(c.t[it1:it2], c.Te[it1:it2])
    if cont == 0:
        label = "Te"
    else:
        label = ""
    ax.plot(c.x_lw, T, ls="--", c=colors[cont], label=label)
    T = CDFtools.timeAverage(c.t[it1:it2], c.Ti[it1:it2])
    if cont == 0:
        label = "Ti"
    else:
        label = ""
    ax.plot(c.x_lw, T, ls="-", c=colors[cont], label=label)

    T = CDFtools.timeAverage(c.t[it1:it2], c.ne[it1:it2])
    axt.plot(c.x_lw, T, ls="-.", lw=0.5, c=colors[cont])

ax.set_xlabel("$\\rho_N$")
ax.set_ylabel("$T_e$, $T_i$ (keV)")
ax.set_title(f"Time Av. from {time - avTime:.3f}s to {time + avTime:.3f}s")
ax.legend()
axt.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")

ax = axs[0, 1]
ax.plot(exp.time, exp.neut, c="k")
for cont, c in enumerate(cs):
    ax.plot(c.t, c.neutrons, ls="-", c=colors[cont])
ax.set_xlabel("Time (s)")
ax.set_ylabel("neutron rate ($10^{20}/s$)")
ax.axvspan(time - avTime, time + avTime, facecolor="b", alpha=0.1, edgecolor="none")

ax = axs[0, 2]
ax.plot(exp.time, exp.Wexp, c="k")
for cont, c in enumerate(cs):
    if cont == 0:
        label = "Thermal"
    else:
        label = ""
    ax.plot(c.t, c.Wth, ls="--", c=colors[cont], label=label)
    if cont == 0:
        label = "Total"
    else:
        label = ""
    ax.plot(c.t, c.Wtot, ls="-", c=colors[cont], label=label)
ax.set_xlabel("Time (s)")
ax.set_ylabel("W (MJ)")
ax.axvspan(time - avTime, time + avTime, facecolor="b", alpha=0.1, edgecolor="none")
ax.legend()

ax = axs[1, 0]
ax.plot(exp.time, exp.q95, c="k", label="exp")
for cont, c in enumerate(cs):
    ax.plot(c.t, c.q95, ls="-", c=colors[cont], label=lab[cont])
ax.set_xlabel("Time (s)")
ax.set_ylabel("q95")
ax.legend()
ax.axvspan(time - avTime, time + avTime, facecolor="b", alpha=0.1, edgecolor="none")

ax = axs[1, 1]
ax.plot(exp.time, exp.Vsurf, c="k")
for cont, c in enumerate(cs):
    ax.plot(c.t, c.Vsurf, ls="-", c=colors[cont])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Vsurf")
ax.axvspan(time - avTime, time + avTime, facecolor="b", alpha=0.1, edgecolor="none")

ax = axs[1, 2]
ax.plot(exp.time, exp.Li, c="k")
for cont, c in enumerate(cs):
    if cont == 0:
        label = "Li1"
    else:
        label = ""
    ax.plot(c.t, c.Li1, ls="--", c=colors[cont], label=label)
    if cont == 0:
        label = "LiVdiff"
    else:
        label = ""
    ax.plot(c.t, c.LiVDIFF, ls="-", c=colors[cont], label=label)
    if cont == 0:
        label = "Li3"
    else:
        label = ""
    ax.plot(c.t, c.Li3, ls="-.", c=colors[cont], label=label)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Internal inductance, Li")
ax.legend()
ax.axvspan(time - avTime, time + avTime, facecolor="b", alpha=0.1, edgecolor="none")

GRAPHICStools.adjust_figure_layout(fig)
