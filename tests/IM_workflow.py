"""
This example runs the complete IM workflow
To run: python3  $MITIM_PATH/tests/IM_workflow.py

Takes ~ for full process

"""

import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.im_tools import IMtools

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

FolderEvaluation = IOtools.expandPath("$MITIM_PATH/tests/scratch/im_test/")
IMnamelist = IOtools.expandPath("$MITIM_PATH/templates/im.namelist")

if restart and os.path.exists(FolderEvaluation):
    os.system(f"rm -r {FolderEvaluation}")

os.system(f"mkdir {FolderEvaluation}")

im = IMtools.runIMworkflow(
    IMnamelist,
    FolderEvaluation,
    "0",
    SpecificParams={
        "forceMinimumDebug": None,
    },
    modNamelist={
        "minWaitLook": 10,
        "ftimeOffsets": [0.101, 0.101, 0.101],
        "DoubleInInterpretiveMinLength": 0.101,
    },
)

# Plot final CDF
im.Reactor.plot()

# Required if running in non-interactive mode
im.Reactor.fn.show()