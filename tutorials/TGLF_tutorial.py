import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

# Select the input.gacode file to start from and the folder where everything will be stored
inputgacode_file = __mitimroot__+"/tests/data/input.gacode"
folder = __mitimroot__+"/tests/scratch/tglf_tut/"

# Initialize the TGLF class with the rho (square root of the normalized toroidal flux) values
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])

# Prepare the TGLF class
cdf = tglf.prep(folder, inputgacode=inputgacode_file, cold_start=False)

'''
***************************************************************************
Run standandalone TGLF
***************************************************************************
'''

# Run TGLF in subfolder
tglf.run(
    subFolderTGLF="yes_em_folder/",
    TGLFsettings=5,
    extraOptions={},
    cold_start=False
    )

# Read results of previous run and store them in results dictionary
tglf.read(label="yes_em")

# Run TGLF in a different subfolder with different settings
tglf.run(
    subFolderTGLF="no_em_folder/",
    TGLFsettings=5,
    extraOptions={"USE_BPER": False},
    cold_start=False,
)

# Read results of previous run and store them in results dictionary
tglf.read(label="no_em")

# Plot the two cases together
tglf.plot(labels=["yes_em", "no_em"])

'''
***************************************************************************
Run TGLF scan
***************************************************************************
'''

tglf.runScan(	subFolderTGLF = 'scan1/',
                TGLFsettings  = 5,
                cold_start       = False,
                variable      = 'RLTS_1',
                varUpDown 	  = np.linspace(0.5,1.5,3))
tglf.readScan(label='scan1',variable = 'RLTS_1')


tglf.runScan(	subFolderTGLF = 'scan2/',
                TGLFsettings  = 5,
                cold_start       = False,
                variable      = 'RLTS_2',
                varUpDown 	  = np.linspace(0.5,1.5,3))
tglf.readScan(label='scan2',variable = 'RLTS_2')


tglf.plotScan(labels=['scan1','scan2'])

'''
***************************************************************************
Automatic scan of turbulence drives
***************************************************************************
'''

tglf.runScanTurbulenceDrives(	
                subFolderTGLF = 'turb_drives/',
                TGLFsettings  = 5,
                cold_start       = False)

tglf.plotScanTurbulenceDrives(label='turb_drives')

'''
***************************************************************************
Automatic scan of turbulence drives
***************************************************************************
'''

tglf.runAnalysis(
            subFolderTGLF 	= 'chi_e/',
            analysisType  	= 'chi_e',
            TGLFsettings  	= 5,
            cold_start 		= False,
            label 			= 'chi_eu')

tglf.plotAnalysis(labels=['chi_eu'],analysisType='chi_e')

'''
***************************************************************************
Explore all available MITIM settings for TGLF (with waveforms)
***************************************************************************
'''

for i in[1,2,3,4,5,6]:
    tglf.run(
        subFolderTGLF = f'settings{i}/',
        runWaveForms  = [0.67],
        TGLFsettings  = i,
        cold_start       = False)
    tglf.read(label=f'settings{i}')

tglf.plot(labels=[f'settings{i}' for i in range(1,6)])
