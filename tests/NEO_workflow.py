import os
from mitim_tools.gacode_tools import NEOtools, PROFILEStools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "neo_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

neo = NEOtools.NEO(
    rhos=[0.55]
)
neo.prep_direct(PROFILEStools.gacode_state(input_gacode), folder, )

neo.run('neo1')
