import os
from mitim_tools.gacode_tools.PROFILEStools import gacode_state
from mitim_tools.gyrokinetics_tools import GXtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "gx_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Only 1 ion species
p = gacode_state(input_gacode)
p.lumpSpecies(ions_list=[1,2,3,4])
#

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

gx = GXtools.GX(rhos=[0.5,0.7])
gx.prep(p, folder)

gx.run(
    'gx1/',
    cold_start=cold_start,
    extraOptions={
        't_max':5.0, # Short, I just want to test the run
    },
    )
gx.read('gx1')

gx.plot(labels=['gx1'])

gx.fn.show()
gx.fn.close()
