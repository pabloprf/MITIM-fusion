from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.misc_tools import IOtools

# Select the input.gacode file to start from and the folder where everything will be stored
inputgacode_file = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")
folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/tglf_tut/")

# Initialize the TGLF class with the rho (square root of the normalized toroidal flux) values
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])

# Prepare the TGLF class
cdf = tglf.prep(folder, inputgacode=inputgacode_file, restart=False)

# Run TGLF in subfolder
tglf.run(subFolderTGLF="yes_em_folder/", TGLFsettings=5, extraOptions={}, restart=True)

# Read results of previous run and store them in results dictionary
tglf.read(label="yes_em")

# Run TGLF in a different subfolder with different settings
tglf.run(
    subFolderTGLF="no_em_folder/",
    TGLFsettings=5,
    extraOptions={"USE_BPER": False},
    restart=True,
)

# Read results of previous run and store them in results dictionary
tglf.read(label="no_em")

# Plot the two cases together
tglf.plotRun(labels=["yes_em", "no_em"])
