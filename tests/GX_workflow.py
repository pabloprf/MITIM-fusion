import os
from mitim_tools.gacode_tools.PROFILEStools import gacode_state
from mitim_tools.gyrokinetics_tools import GXtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "gx_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Reduce the ion species to just 1
p = gacode_state(input_gacode)
p.lumpIons()
# --------------------------------

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

gx = GXtools.GX(rhos=[0.5, 0.6])
gx.prep(p, folder)

gx.run(
    'gx1/',
    cold_start=cold_start,
    code_settings=0, # Linear
    extraOptions={
        't_max':10.0,   # Run up to 50.0 a/c_s
        'y0' :5.0,     # kymin = 1/y0 = 0.2
        'ny': 34,       # nky = 1 + (ny-1)/3 = 12 -> ky_range = 0.2 - 2.4
    },
    )
gx.read('gx1')

gx.plot(labels=['gx1'])

gx.fn.show()
gx.fn.close()
