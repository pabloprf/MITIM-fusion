import os
import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglfscan_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and os.path.exists(folder):
    os.system(f"rm -r {folder}")

tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(folder, inputgacode=input_gacode, cold_start=cold_start)

tglf.runScan(	subFolderTGLF = 'scan1/',
                TGLFsettings  = None,
                cold_start       = cold_start,
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
                cold_start       = cold_start)

tglf.plotScanTurbulenceDrives(label='turb_drives')
tglf.fn.show()
tglf.fn.close()
