import os
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__

cold_start = True

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "cgyro_test2"

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
        'KY':0.3,
        'MAX_TIME': 1E1, # Short, I just want to test the run
    },
    submit_via_qsub=True # NERSC: True #TODO change this
    )

cgyro.check(every_n_minutes=1)
cgyro.fetch()
cgyro.delete()

cgyro.read(label="cgyro1")

cgyro.plot_quick_linear(labels=["cgyro1"])
cgyro.fn.show()
