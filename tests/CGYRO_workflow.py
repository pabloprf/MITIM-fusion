import os
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__

cold_start = True

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "cgyro_test"

if cold_start and folder.exists():
    os.system(f"rm -r {folder}")

folder.mkdir(parents=True, exist_ok=True)

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
cgyro.fnLS.show()
