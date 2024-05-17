import sys, copy, torch, datetime, cProfile, argparse
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools

"""
e.g.
	~/MITIM/mitim_opt/opt_tools/exe/plot_optimizationl.py --folder run1/

"""

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--step", type=int, required=False, default=-1)

args = parser.parse_args()

# Inputs

folderWork = args.folder
step_num = args.step

# Read

opt_fun = STRATEGYtools.opt_evaluator(folderWork)
opt_fun.read_optimization_results(analysis_level=4)
strat = opt_fun.prfs_model

strat.plotSurrogateOptimization(boStep=step_num)
