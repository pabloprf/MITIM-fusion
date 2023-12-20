import netCDF4, fnmatch, sys, os, pickle
import matplotlib.pyplot as plt
from mitim_tools.im_tools.outputs import PLOTtools
from mitim_tools.im_tools.outputs.database import plotDB
import numpy as np
from matplotlib.pyplot import cm

from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools, GUItools

from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

"""
run ~/MITIM/mitim/im_tools/outputs/plotALL.py ./ 2 1 2
"""

# ------ Arguments

# Folder
fold = sys.argv[1]

# Keep CDFs?
keepCDFs = int(sys.argv[2])  # 0= Nothing, 1= Only last cdfs, 2= All

contS = 2

# tie plot
timePlot = None
avTime = 0.15  # 0.0

# ------ If folder has ssh, copy

if ":" in fold:
    ssh = fold.split(":")[0]
    fold = fold.split(":")[1]
else:
    ssh = None

# ---------------

# If argument given with commas, it's list
if "," in sys.argv[contS + 1]:
    whichOnes = [int(a) for a in sys.argv[contS + 1].split(",")]
    cont = 1
# If only one number, just that one
elif len(sys.argv) < contS + 2:
    whichOnes = [int(sys.argv[contS + 1])]
    cont = 1
# If not, I have given first and last
else:
    firstNumber = int(sys.argv[contS + 1])  # -1
    lastNumber = int(sys.argv[contS + 2]) + 1
    whichOnes = np.arange(firstNumber, lastNumber, 1)
    cont = 0

try:
    xTag = sys.argv[contS + 3 - cont]
except:
    xTag = "Bt"
try:
    yTag = sys.argv[contS + 4 - cont]
except:
    yTag = "Rmajor"
try:
    wTag = sys.argv[contS + 5 - cont]
except:
    wTag = None

# ---------------------------------------------------------------------------------------------------------------
# ------ Get inputs (Folder where all subfolders are, numbers of Evaluations)

mainF = os.path.abspath(fold)
colorsC = GRAPHICStools.listColors()  # cm.rainbow(np.linspace(0,1,len(whichOnes)))

mainFred = mainF.split("/")[-1]

# ------ Plot Summary of all runs, and convergence summary

plt.ioff()
fn = GUItools.FigureNotebook(0, "ALL", geometry="1700x900")
fig1 = fn.add_figure(label="Data")

axSummary = []
grid = plt.GridSpec(nrows=2, ncols=6, hspace=0.3, wspace=0.25)
for i in range(2):
    axSumm = []
    for j in range(6):
        axSumm.append(fig1.add_subplot(grid[i, j]))
    axSummary.append(axSumm)
axSummary = np.array(axSummary)

axMachines = None
# figMachines,axMachines = plt.subplots(figsize=(5,10))

Qtot, Costtot, shotsListFlags, cdfs, cdfs_i, cdfs_p = PLOTtools.plotSummaryScan(
    axSummary,
    whichOnes,
    mainF,
    colorsC,
    axMachines=axMachines,
    ssh=ssh,
    keepCDFs=keepCDFs,
)

# ------ Plot database scatters
cdfs_clean = []
for c in cdfs:
    if type(c) != str:
        cdfs_clean.append(c)

cdfs_p_clean = []
for c in cdfs_p:
    if type(c) != str:
        cdfs_p_clean.append(c)


# fn = GUItools.FigureNotebook(0,'SPARC scenarios',geometry='1700x900')

figs = plotDB(
    cdfs_clean,
    runIDs=whichOnes,
    fn=fn,
    extralab=" cdfs",
    timePlot=timePlot,
    avTime=avTime,
)
figs = plotDB(
    cdfs_p_clean,
    runIDs=whichOnes,
    fn=fn,
    extralab=" cdfs_p",
    timePlot=timePlot,
    avTime=avTime,
)

fn.show()
