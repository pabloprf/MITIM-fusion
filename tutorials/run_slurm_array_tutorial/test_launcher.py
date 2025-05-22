import json
import os
from mitim_tools.opt_tools.scripts.slurm import run_slurm_array
from mitim_tools.opt_tools.scripts.slurm import run_slurm
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.CONFIGread import load_settings

#  You have to have a slurm partition specified for your local machine for this to work!!
partition = load_settings()['local']['slurm']['partition'] 
print(f"Using partition: {partition}")


# Settings for slurm job
cpus = 2
hours = 1
memory = '100GB'
folder = __mitimroot__ / "tutorials" / "run_slurm_array_tutorial" / "scratch"
script = f'python {folder}/../test_script.py {folder} '

# Input the array runs
array_input = [62, 63, 81]

# To use run_slurm_array
run_slurm_array(script, array_input, folder,partition, max_concurrent_jobs = 2, hours=hours,n=cpus,mem=memory)

# For comparison run_slurm
# run_slurm(f'python test_script.py {folder} 84', '/home/audreysa/test_script/scratch_run_slurm', partition, environment, hours=hours, n=cpus, mem=memory)
