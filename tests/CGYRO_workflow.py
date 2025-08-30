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

cgyro.prep(gacode_file,folder)

# ---------------
# Standalone run
# ---------------

run_type = 'submit' # 'normal': submit and wait; 'submit': Just prepare and submit, do not wait [requies cgyro.check() and cgyro.fetch()]

cgyro.run(
    'linear',
    code_settings=0,
    extraOptions={
        'KY':0.5,
        'MAX_TIME': 10.0, # Short, I just want to test the run. Enough to get the restart file
    },
    slurm_setup={
        'cores':16, # Each CGYRO instance (each radius will have this number of cores or gpus)
        'minutes': 10,
        },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type=run_type, 
    )

if run_type == 'submit':
    cgyro.check(every_n_minutes=1)
    cgyro.fetch()

cgyro.read(label="cgyro1")
cgyro.plot(labels=["cgyro1"])

# # ---------------
# # Scan of KY
# # ---------------

# run_type = 'normal'

# cgyro.run_scan(
#     'scan1',
#     extraOptions={
#         'MAX_TIME': 10.0, # Short, I just want to test the run. Enough to get the restart file
#     },
#     variable='KY',
#     varUpDown=[0.3,0.4],
#     slurm_setup={
#         'cores':16
#         },
#     cold_start=cold_start,
#     forceIfcold_start=True,
#     run_type=run_type
#     )

# cgyro.plot(labels=["scan1_KY_0.3","scan1_KY_0.4"], fn = cgyro.fn)

# fig = cgyro.fn.add_figure(label="Quick linear")
# cgyro.plot_quick_linear(labels=["scan1_KY_0.3","scan1_KY_0.4"], fig = fig)

# cgyro.fn.show()
