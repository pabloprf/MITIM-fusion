import os
import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "neo_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

neo = NEOtools.NEO(rhos=np.linspace(0.1,0.95,10))
neo.prep(input_gacode, folder)

neo.run('neo1/', cold_start=cold_start)
neo.read('NEO default')

neo.run('neo2/', cold_start=cold_start, extraOptions={'N_ENERGY':5,'N_XI': 11, 'N_THETA': 11})
neo.read('NEO low res')

neo.run('neo3/', cold_start=cold_start, extraOptions={'N_ENERGY':5,'N_XI': 11, 'N_THETA': 11}, multipliers={'DLNTDR_1': 1.5})
neo.read('NEO low res + 50% aLTe')

# neo.run_scan('scan1', cold_start=cold_start, variable='DLNTDR_1', varUpDown=np.linspace(0.5, 1.5, 4))

neo.plot(labels=['NEO default', 'NEO low res', 'NEO low res + 50% aLTe'])
neo.fn.show()
neo.fn.close()