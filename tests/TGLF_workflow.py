import os
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

restart = True

if not os.path.exists(__mitimroot__ + "/tests/scratch/"):
    os.system("mkdir " + __mitimroot__ + "/tests/scratch/")

folder = __mitimroot__ + "/tests/scratch/tglf_test/"
input_tglf = __mitimroot__ + "/tests/data/input.tglf"

if restart and os.path.exists(folder):
    os.system(f"rm -r {folder}")

tglf = TGLFtools.TGLF()
tglf.prep_from_tglf(folder, input_tglf)
tglf.run(
    subFolderTGLF="run1/",
    TGLFsettings=None,
    restart=restart,
    forceIfRestart=True,
    extraOptions={"USE_BPER": True},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="EM")


gamma = tglf.results['EM']['TGLFout'][0].g 
ky = tglf.results['EM']['TGLFout'][0].ky
neTeSpectrum = tglf.results['EM']['TGLFout'][0].neTeSpectrum



import matplotlib.pyplot as plt
plt.ion(); fig, ax = plt.subplots()

ax.plot(ky, neTeSpectrum[0,:], 'o-')
ax.set_xscale('log')

# tglf.plot(labels=["EM","ES"])



# # Required if running in non-interactive mode
# tglf.fn.show()


import numpy as np
tglf.runScan(	subFolderTGLF = 'scan1/',
                TGLFsettings  = 5,
                restart       = False,
                variable      = 'RLTS_1',
                varUpDown 	  = np.array([0.5, 1.5, 2.5]))
tglf.readScan(label='scan1',variable = 'RLTS_1')