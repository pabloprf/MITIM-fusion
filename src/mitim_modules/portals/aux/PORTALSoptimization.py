import copy, torch, os
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print

"""
*********************************************************************************************************************
	Initialization
*********************************************************************************************************************
"""


def initialization_simple_relax(self):
    # ------------------------------------------------------------------------------------
    # Perform flux matched using powerstate
    # ------------------------------------------------------------------------------------

    powerstate = copy.deepcopy(self.mainFunction.powerstate)

    folderExecution = IOtools.expandPath(self.folderExecution, ensurePathValid=True)

    if not os.path.exists(f"{folderExecution}/Initialization/"):
        os.system(f"mkdir {folderExecution}/Initialization/")
    MainFolder = f"{folderExecution}/Initialization/initialization_simple_relax/"
    if not os.path.exists(MainFolder):
        os.system(f"mkdir {MainFolder}")

    a, b = IOtools.reducePathLevel(self.folderExecution, level=1)
    namingConvention = f"portals_{b}_ev"

    algorithmOptions = {
        "tol": 1e-6,
        "max_it": self.OriginalInitialPoints,
        "relax": 0.2,
        "dx_max": 0.2,
        "print_each": 1,
        "MainFolder": MainFolder,
        "storeValues": True,
        "namingConvention": namingConvention,
    }

    # Define input.gacode to do flux-matching with

    readFile = f"{MainFolder}/input.gacode"
    with open(readFile, "w") as f:
        f.writelines(self.mainFunction.file_in_lines_initial_input_gacode)

    from mitim_tools.gacode_tools.PROFILEStools import PROFILES_GACODE

    powerstate.profiles = PROFILES_GACODE(readFile, calculateDerived=False)

    # Trick to actually start from different gradients than those in the initial_input_gacode

    X = torch.from_numpy(self.Optim["BaselineDV"]).to(self.dfT).unsqueeze(0)
    powerstate.modify(X)

    # Flux matching process

    powerstate.findFluxMatchProfiles(
        algorithm="simple_relax",
        algorithmOptions=algorithmOptions,
    )
    Xopt = powerstate.FluxMatch_Xopt

    # -------------------------------------------------------------------------------------------
    # Once flux matching has been attained, copy those as if they were direct MITIM evaluations
    # -------------------------------------------------------------------------------------------

    if not os.path.exists(f"{self.folderExecution}/Execution/"):
        os.mkdir(f"{self.folderExecution}/Execution/")

    for i in range(self.OriginalInitialPoints):
        ff = f"{self.folderExecution}/Execution/Evaluation.{i}/"
        if not os.path.exists(ff):
            os.mkdir(ff)
        os.system(
            f"cp -r {MainFolder}/{namingConvention}{i}/model_complete {ff}/model_complete"
        )

    return Xopt.cpu().numpy()