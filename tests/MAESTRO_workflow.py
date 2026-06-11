import os
import torch
from mitim_tools import __mitimroot__
from mitim_modules.maestro.scripts import run_maestro
from mitim_tools.misc_tools import IOtools

cold_start = True

folder = __mitimroot__ / "tests" / "scratch" / "maestro_test"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if cold_start and os.path.exists(folder):
    IOtools.shutil_rmtree(folder)

# Let's not consume the entire computer resources when running test... limit threads
torch.set_num_threads(8)

folder.mkdir(parents=True, exist_ok=True)

m = run_maestro.run_maestro_local(template, 
                                  folder=folder, 
                                  terminal_outputs = True, 
                                  force_cold_start=cold_start,
                                  cpus = 8
                                  )

m.plot(num_beats = 4)