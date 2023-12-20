import sys, pickle, os
import numpy as np
import matplotlib.pyplot as plt

from mitim_tools.opt_tools import BOtools, BOgraphics
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.mitim_tools.src import main

# Input parameters

folderExecution = IOtools.expandPath(sys.argv[1])
try:
    multOFs = float(sys.argv[2])
except:
    multOFs = 1.0

# Notebook

plt.ioff()
fn = GUItools.FigureNotebook(0, "MITIM BO Notebook", geometry="1000x1000")
fig1 = fn.add_figure(label="Optimization")
fig1b = fn.add_figure(label="Optimization (relative)")
fig2 = fn.add_figure(label="Model")
fig3 = fn.add_figure(label="Workflow Metrics")
fig4 = fn.add_figure(label="Evolution")
# Results Optimization - Main Summary

ResultsOptimization = BOgraphics.ResultsOptimization(
    file=folderExecution + "/Outputs/ResultsOptimization.out"
)
ResultsOptimization.read()

plotModel = True
axs = ResultsOptimization.plotComplete(
    fig=fig1,
    multOFs=multOFs,
    separateDVs=True,
    normalizeToFirstOF=False,
    normalizeToFirstDV=False,
    plotModel=plotModel,
)
axs = ResultsOptimization.plotComplete(
    fig=fig1b,
    multOFs=multOFs,
    separateDVs=False,
    normalizeToFirstOF=True,
    normalizeToFirstDV=True,
    plotModel=plotModel,
)
axs = ResultsOptimization.plotMetrics(fig=fig3)

axs = ResultsOptimization.plotComplete(
    fig=fig4,
    multOFs=multOFs,
    separateDVs=True,
    normalizeToFirstOF=False,
    normalizeToFirstDV=False,
    plotModel=plotModel,
    onlyFinals=True,
)
# axs = ResultsOptimization.plotEvolution(fig=fig4)

# Re-create model

allStates = IOtools.findExistingFiles(folderExecution + "/Outputs/", "pkl")

for i in range(len(allStates)):
    allStates[i] = allStates[i].split("MITIMstate_")[1].split(".pkl")[0]

if "final" in allStates:
    file = folderExecution + "/Outputs/MITIMstate_final.pkl"
else:
    iterations = []
    for i in allStates:
        iterations.append(int(i.split("iteration")[-1]))
    try:
        maxit = np.max(iterations)
        file = folderExecution + f"/Outputs/MITIMstate_iteration{maxit}.pkl"
    except:
        file = None


vmaxvmin = None  # [0.5,1.5]
levels = []  # [1.0,5.0,10.0,29.5,142]

if file is not None:
    print(f"Reading {file} as last state file")
    with open(file, "rb") as f:
        PRF_BO = pickle.load(f)
    PRF_BO.plotResults(fig=fig2, vmaxvmin=vmaxvmin, levels=levels)
