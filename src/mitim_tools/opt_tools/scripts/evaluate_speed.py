import torch
import argparse
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools

"""
speed_tester.py --folder run1/ --num 1000 --name test1
"""


parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--num", type=int, required=False, default=10000)
parser.add_argument("--name", type=str, required=False, default="")
args = parser.parse_args()

folder = IOtools.expandPath(args.folder) + "/"
cases = args.num
name = args.name

opt_fun = STRATEGYtools.opt_evaluator(folder)
opt_fun.read_optimization_results(analysis_level=4)
step = opt_fun.prfs_model.steps[-1]

x = torch.rand(cases, step.train_X.shape[-1])

with IOtools.speeder(f"profiler{name}.prof") as s:
    with torch.no_grad():
        mean, upper, lower, _ = step.GP["combined_model"].predict(x)

print(
    f"\nIt took {s.timeDiff:.3f}s to run {x.shape[0]:.1e} parallel evaluations (i.e. {s.timeDiff*1E6/cases:.3f}micro-s/member) of {mean.shape[-1]} GPs with {x.shape[-1]} raw input dimensions"
)
