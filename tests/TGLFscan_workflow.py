import os
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/tglfscan_test/")
input_gacode = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")

if restart and os.path.exists(folder):
    os.system(f"rm -r {folder}")

tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(folder, inputgacode=input_gacode, restart=restart)

tglf.runScan(	subFolderTGLF = 'scan1/',
                TGLFsettings  = None,
                restart       = restart,
                runWaveForms  = [0.67, 10.0],
                variable      = 'RLTS_1',
                varUpDown 	  = np.linspace(0.5,1.5,3))
tglf.readScan(label='scan1',variable = 'RLTS_1')

tglf.plotScan(labels=['scan1'])
tglf.fn.show()
tglf.fn.close()

tglf.runScanTurbulenceDrives(	
                subFolderTGLF = 'turb_drives/',
                TGLFsettings  = None,
                restart       = restart)

tglf.plotScanTurbulenceDrives(label='turb_drives')
tglf.fn.show()
tglf.fn.close()
