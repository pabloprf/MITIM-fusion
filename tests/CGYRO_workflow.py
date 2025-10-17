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

# ---------------------------------------------------------------------------
# Linear: standalone run and using the submit/check/fetch functionality
# ---------------------------------------------------------------------------

run_type = 'submit' # 'normal': submit and wait; 'submit': Just prepare and submit, do not wait [requies cgyro.check() and cgyro.fetch()]

cgyro.run(
    'linear',
    code_settings="Linear",
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

cgyro.check(every_n_minutes=1)
cgyro.fetch()

# Read results
cgyro.read(label="cgyro1")

# ---------------------------------------------------------------------------
# Linear: scan of KY and using the normal run functionality
# ---------------------------------------------------------------------------

run_type = 'normal'

cgyro.run_scan(
    'scan1',
    code_settings="Linear",
    extraOptions={
        'MAX_TIME': 10.0, # Short, I just want to test the run. Enough to get the restart file
    },
    variable='KY',
    varUpDown=[0.3,0.4],
    relativeChanges=False,
    slurm_setup={
        'cores':16,
        'minutes': 10,
        },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type=run_type
    )

# Read scan results
cgyro.read_linear_scan(label="scan1", variable='KY', store_as_label="scan1_rho0", irho=0)
cgyro.read_linear_scan(label="scan1", variable='KY', store_as_label="scan1_rho1", irho=1)

# ---------------------------------------------------------------------------
# Nonlinear (super coarse run to be fast!)
# ---------------------------------------------------------------------------

run_type = 'normal'

cgyro.run(
    'Nonlinear',
    code_settings="Nonlinear",
    extraOptions={
        'MAX_TIME': 10.0, # Short, I just want to test the run. Enough to get the restart file
        'KY': 0.1,
        'N_TOROIDAL': 12,
        'TOROIDALS_PER_PROC': 3,
        'BOX_SIZE': 3,
        'N_RADIAL': 48,
        'N_XI': 8,
        'N_THETA': 8,
        'N_ENERGY': 4,
        'COLLISION_MODEL': 5,
        'ROTATION_MODEL': 1,
    },
    slurm_setup={
        'cores':16, # Each CGYRO instance (each radius will have this number of cores or gpus)
        'minutes': 10,
        },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type=run_type, 
    )

cgyro.read(label="cgyro2")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Plot results of the linear runs, as normal ones
cgyro.plot(labels=["cgyro1"])
cgyro.plot(labels=["scan1_KY_0.3","scan1_KY_0.4"], fn = cgyro.fn)

# Special plot type: quick linear plot
fig = cgyro.fn.add_figure(label="Quick linear", tab_color=1)
cgyro.plot_quick_linear(labels=["scan1_rho0", "scan1_rho1"], fig = fig)

# Plot results of the nonlinear run
cgyro.plot(labels=["cgyro2"], fn = cgyro.fn)

cgyro.fn.show()
