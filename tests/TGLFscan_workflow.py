import os
import numpy as np
from mitim_tools.gacode_tools import TGLFtools, PROFILEStools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglfscan_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode,folder, cold_start=cold_start)

tglf.run_scan(	subfolder = 'scan1',
                TGLFsettings  = None,
                cold_start       = cold_start,
                runWaveForms  = [0.67, 10.0],
                variable      = 'RLTS_1',
                varUpDown 	  = np.linspace(0.5,1.5,4))
tglf.read_scan(label='scan1',variable = 'RLTS_1')

tglf.plot_scan(labels=['scan1'])
tglf.fn.show()
tglf.fn.close()

tglf.runScanTurbulenceDrives(	
                subfolder = 'turb_drives',
                TGLFsettings  = None,
                resolutionPoints=3,
                cold_start       = cold_start)

tglf.plotScanTurbulenceDrives(label='turb_drives')
tglf.fn.show()
tglf.fn.close()
