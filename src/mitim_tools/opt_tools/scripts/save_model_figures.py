import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools

"""
Script to create a folder with all the model figures around last point

"""

# ***************** Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--save", type=str, required=True)
parser.add_argument("--step", type=int, required=False, default=-1)
args = parser.parse_args()

folderWork = IOtools.expandPath(args.folder)
step_num = args.step
save = Path(args.save)

# ***************** Read

opt_fun = STRATEGYtools.opt_evaluator(folderWork)
opt_fun.read_optimization_results(analysis_level=4)
strat = opt_fun.mitim_model
step = strat.steps[step_num]
gps = step.GP["individual_models"]

# ***************** Plot and Save

save.mkdir(parents=True, exist_ok=True)

for i in range(len(gps)):
    name_save = save / f"model{i+1}"

    print(f"- Plotting and saving {name_save}")

    plt.close("all")
    fig, ax = plt.subplots()
    gps[i].plot(axs=[ax, None, None])

    GRAPHICStools.output_figure_papers(name_save, fig=fig)
