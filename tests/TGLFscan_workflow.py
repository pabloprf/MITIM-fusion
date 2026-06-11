import os
import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglfscan_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
npz_file = folder / "scan_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

tglf.run_scan(  subfolder = 'scan1',
                code_settings  = None,
                extraOptions = {"USE_BPER": [False, True]},
                cold_start       = cold_start,
                runWaveForms  = [0.67, 10.0],
                variable      = 'RLTS_1',
                varUpDown     = np.linspace(0.5,1.5,4))

tglf.read_scan(label='scan1', variable='RLTS_1', save_and_cleanup=npz_file)

tglf_loaded = TGLFtools.TGLF.from_npz(npz_file)
tglf_loaded.plot_scan(labels=['scan1'], plotTGLFs=False)
tglf_loaded.fn.show()
tglf_loaded.fn.close()

tglf.runScanTurbulenceDrives(
                subfolder = 'turb_drives',
                code_settings  = None,
                resolutionPoints=3,
                cold_start       = cold_start,
                save_and_cleanup = npz_file)

tglf_loaded2 = TGLFtools.TGLF.from_npz(npz_file)
tglf_loaded2.plotScanTurbulenceDrives(label='turb_drives', plotTGLFs=False)
tglf_loaded2.fn.show()
tglf_loaded2.fn.close()
