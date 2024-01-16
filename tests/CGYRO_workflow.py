import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import CGYROtools


gacode_file = '/Users/pablorf/scratch/input.gacode' #IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")
folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/cgyro_test/")

if not os.path.exists(folder):
    os.system(f"mkdir -p {folder}")

cgyro = CGYROtools.CGYRO(gacode_file)

cgyro.prep(folder)

cgyro.run_test(name='run1')

cgyro.check(every_n_minutes=5)

cgyro.get()
