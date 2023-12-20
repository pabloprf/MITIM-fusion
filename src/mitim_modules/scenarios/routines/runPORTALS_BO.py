"""
BO-MITIM (Bayesian Optimization for MITIM)

INPUTS:
	1) Execution folder (the parent folder for subsequent Outputs, Execution)
	Rest are 0/1 representing booleans:
	~2) Restart? (if yes, assumes that Tabular Data contains all info)
	~3) Run directly in command line?

EXAMPLE:
	Standard use:
		python3 runMITIM_BO.py
		(same as python3 runMITIM_BO.py . 1 1)
	Other options:
		- No Running from scratch in the folder:
			python3 runMITIM_BO.py . 0 1

NOTES:
	- Execution folder must contain "mitim.namelist"
"""

import sys
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import IOtools, FARMINGtools, GRAPHICStools

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ User Parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

forceMinimumDebug = 0

testing = False

# ---------------------------------------------------

# Input parameters to function

try:
    folderExecution = sys.argv[1]
except:
    folderExecution = "./"

try:
    restartYN = bool(int(sys.argv[2]))
except:
    restartYN = False

try:
    directRun = bool(int(sys.argv[3]))
except:
    directRun = True

# ---------------------------------------------------

if directRun:
    folderExecution = IOtools.expandPath(folderExecution) + "/"
    Optim = IOtools.readOptim_Complete(folderExecution + "mitim.namelist")

    # Function to optimize. Must generate resultsfile, receving paramsfile

    if not testing:
        from mitim_tools_tools.src import main

        mainFunction = main.runMITIM
    else:
        from testing import evaluator

        mainFunction = evaluator.runMITIM

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~ Running full workflow
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    SpecificParams = {
        "IsItSingle": True,
        "forceMinimumDebug": forceMinimumDebug,
        "automaticProcess": True,
    }

    PRF_BO = STRATEGYtools.PRF_BO(
        folderExecution, Optim, mainFunction, SpecificParams, restartYN=restartYN
    )
    _ = PRF_BO.run()

else:
    print(
        " ~~~~ MITIM BO optimization in {0} will be run in the background".format(
            folderExecution
        )
    )
    fileRun = IOtools.expandPath("$MITIM_PATH/run/runMITIM_BO.py")
    Command = f"python3 {fileRun} {folderExecution} 0 {int(restartYN)} 1"
    error, result = FARMINGtools.run_subprocess(Command, localRun=True)
    try:
        from IPython import embed

        embed()
    except:
        pass
