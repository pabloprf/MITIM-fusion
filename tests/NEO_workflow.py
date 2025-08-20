import os
import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools import __mitimroot__
from torch import mul

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
neo.read('NEO')

neo.run('neo2/', extraOptions={'N_ENERGY':6,'N_XI': 17, 'N_THETA': 17})
neo.read('NEO high res')

neo.run('neo3/', extraOptions={'N_ENERGY':6,'N_XI': 17, 'N_THETA': 17}, multipliers={'DLNTDR_1': 1.25})
neo.read('NEO high res + 25% aLTe')

neo.plot(labels=['NEO', 'NEO high res', 'NEO high res + 25% aLTe'])

neo.fn.show()
neo.fn.close()