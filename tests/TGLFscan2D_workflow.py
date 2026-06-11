import os
import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder      = __mitimroot__ / "tests" / "scratch" / "tglfscan2d_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
npz_file    = folder / "scan2d_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

tglf.run_scan2d(
    subfolder   = "scan2d",
    variable1   = "RLTS_1",
    varUpDown1  = np.linspace(0.5, 1.5, 4),
    variable2   = "RLTS_2",
    varUpDown2  = np.linspace(0.5, 1.5, 4),
    code_settings = None,
    cold_start  = cold_start,
    save_and_cleanup = npz_file,
)

# Load from npz and aggregate + plot (raw folders have been cleaned up)
tglf_loaded = TGLFtools.TGLF.from_npz(npz_file)
tglf_loaded.read_scan2d(label="scan2d", ky_target=0.3)

tglf_loaded.plot_scan2d(label="scan2d")
tglf_loaded.fn.show()
tglf_loaded.fn.close()
