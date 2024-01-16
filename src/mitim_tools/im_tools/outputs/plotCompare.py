import netCDF4, fnmatch, sys, os, pickle
import matplotlib.pyplot as plt
from mitim_tools.transp_tools.outputs.PLOTtools import *
import numpy as np
from matplotlib.pyplot import cm

from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools, GUItools

# plotCompare.py cdf1.CDF cdf2.CDF

contS = 2

# time plot
timePlot = None
avTime = 0.005

whichOnes = []
cdfs_ready = []
for i in range(len(sys.argv) - 1):
    cdfs_ready.append(sys.argv[i + 1])
    whichOnes.append(i)

# ---------------------------------------------------------------------------------------------------------------
# ------ Get inputs (Folder where all subfolders are, numbers of Evaluations)

mainF = ""
colorsC = GRAPHICStools.listColors()  # cm.rainbow(np.linspace(0,1,len(whichOnes)))

# ------ Plot Summary of all runs, and convergence summary


fn = GUItools.FigureNotebook("ALL", geometry="1700x900")
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

Qtot, Costtot, shotsListFlags, cdfs, cdfs_i, cdfs_p = plotSummaryScan(
    axSummary,
    whichOnes,
    mainF,
    colorsC,
    axMachines=axMachines,
    cdfs_ready=cdfs_ready,
    keepCDFs=2,
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

from analysis._2020_SPARCscenariosDIV.dataPlot import plotDB

# from analysis_tools.SPARCdivertor.dataPlot import plotDB

# fn = GUItools.FigureNotebook('SPARC scenarios',geometry='1700x900')

figs = plotDB(
    cdfs_clean,
    runIDs=whichOnes,
    fn=fn,
    extralab=" cdfs",
    timePlot=timePlot,
    avTime=avTime,
)
