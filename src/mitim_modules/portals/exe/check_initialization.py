import sys
import matplotlib.pyplot as plt
from mitim_modules.powertorch import STATEtools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.GUItools import FigureNotebook

"""
To check how the method-6 initializaiton is going...

E.g.
	run ~/MITIM/mitim_opt/mitim/exe/check_initialization.py ~/PRF/project_2022_mitim/jet/cgyro/jet_cgyro_D/
"""

folder = IOtools.expandPath(sys.argv[1])

# Read powerstates
ps = []
profs = []
for i in range(10):
    try:
        profs.append(
            PROFILEStools.PROFILES_GACODE(
                f"{folder}/Outputs/ProfilesEvaluated/input.gacode.{i}"
            )
        )
        p = STATEtools.read_saved_state(
            f"{folder}/Initialization/initialization_simple_relax/portals_{IOtools.reducePathLevel(folder)[1]}_ev{i}/powerstate.pkl"
        )
    except:
        break

    p.profiles.deriveQuantities()
    ps.append(p)

plt.ioff()
fn = FigureNotebook(0, "PowerState", geometry="1800x900")
figMain = fn.add_figure(label="PowerState")
figG = fn.add_figure(label="Sequence")

grid = plt.GridSpec(4, 6, hspace=0.3, wspace=0.4)
axs = [
    figMain.add_subplot(grid[0, 1]),
    figMain.add_subplot(grid[0, 2]),
    figMain.add_subplot(grid[0, 3]),
    figMain.add_subplot(grid[0, 4]),
    figMain.add_subplot(grid[0, 5]),
    figMain.add_subplot(grid[1, 1]),
    figMain.add_subplot(grid[1, 2]),
    figMain.add_subplot(grid[1, 3]),
    figMain.add_subplot(grid[1, 4]),
    figMain.add_subplot(grid[1, 5]),
    figMain.add_subplot(grid[2, 1]),
    figMain.add_subplot(grid[2, 2]),
    figMain.add_subplot(grid[2, 3]),
    figMain.add_subplot(grid[2, 4]),
    figMain.add_subplot(grid[2, 5]),
    figMain.add_subplot(grid[3, 1]),
    figMain.add_subplot(grid[3, 2]),
    figMain.add_subplot(grid[3, 3]),
    figMain.add_subplot(grid[3, 4]),
    figMain.add_subplot(grid[3, 5]),
]

axsRes = figMain.add_subplot(grid[:, 0])

colors = GRAPHICStools.listColors()

# POWERPLOT

for i in range(len(ps)):
    ps[i].plot(axs=axs, axsRes=axsRes, c=colors[i], label=f"#{i}")

axs[0].legend(prop={"size": 8})

axsRes.set_xlim([0, i])

# GRADIENTS

grid = plt.GridSpec(2, 5, hspace=0.3, wspace=0.3)
axsGrads = []
for j in range(5):
    for i in range(2):
        axsGrads.append(figG.add_subplot(grid[i, j]))
for i, p in enumerate(profs):
    p.plotGradients(
        axsGrads,
        color=colors[i],
        plotImpurity=3,
        plotRotation=True,
        lastRho=ps[0].plasma["rho"][-1, -1].item(),
    )

axsGrads_extra = [
    axs[0],
    axs[5],
    axs[1],
    axs[6],
    axs[2],
    axs[7],
    axs[3],
    axs[8],
    axs[4],
    axs[9],
]
for i, p in enumerate(profs):
    p.plotGradients(
        axsGrads_extra,
        color=colors[i],
        plotImpurity=3,
        plotRotation=True,
        lastRho=ps[0].plasma["rho"][-1, -1].item(),
        lw=0.5,
        ms=0,
    )

fn.show()
