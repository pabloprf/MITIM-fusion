import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import CGYROtools

restart = True

gacode_file = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")
folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/cgyro_test/")

if restart and os.path.exists(folder):
    os.system(f"rm -r {folder}")

if not os.path.exists(folder):
    os.system(f"mkdir -p {folder}")

cgyro = CGYROtools.CGYRO()

cgyro.prep(folder,gacode_file)

cgyro.run(
    'linear',
    roa = 0.55,
    CGYROsettings=0,
    extraOptions={
        'KY':0.3
    })
cgyro.read(label="cgyro1")

cgyro.run(
    'linear',
    roa = 0.55,
    CGYROsettings=0,
    extraOptions={
        'KY':0.5
    })
cgyro.read(label="cgyro2")

cgyro.run(
    'linear',
    roa = 0.55,
    CGYROsettings=0,
    extraOptions={
        'KY':0.7
    })
cgyro.read(label="cgyro3")


cgyro.plotLS()