import os
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__

cold_start = True

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "cgyro_test"

if cold_start and folder.exists():
    os.system(f"rm -r {folder}")

folder.mkdir(parents=True, exist_ok=True)

cgyro = CGYROtools.CGYRO(rhos = [0.5, 0.7])

cgyro.prep(
    gacode_file,
    folder)

cgyro.run(
    'linear',
    code_settings=0,
    extraOptions={
        'KY':0.3,
        'MAX_TIME': 10.0, # Short, I just want to test the run
    },
    slurm_setup={'cores':4}
    #submit_via_qsub=False # NERSC: True #TODO change this
    )

# cgyro.check(every_n_minutes=1)
# cgyro.fetch()
# cgyro.delete()

cgyro.read(label="cgyro1")

cgyro.plot(labels=["cgyro1_0.5","cgyro1_0.7"])
cgyro.fn.show()
