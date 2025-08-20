import os
import numpy as np
from mitim_tools.gacode_tools import NEOtools, PROFILEStools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "neo_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

neo = NEOtools.NEO(
    rhos=np.linspace(0.1,0.95,20)
)
neo.prep_direct(input_gacode, folder)

neo.run('neo1/')
neo.read('neo1')

neo.run('neo2/', extraOptions={'N_XI': 17, 'N_THETA': 17})
neo.read('neo2')

neo.plot(labels=['neo1', 'neo2'])

neo.fn.show()
neo.fn.close()