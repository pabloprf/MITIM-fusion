import os
from mitim_tools.gacode_tools.PROFILEStools import gacode_state
from mitim_tools.simulation_tools.physics import GXtools
from mitim_tools.misc_tools import GUItools
from mitim_tools import __mitimroot__

cold_start = True
save_figures = True # if True, do not show the plot to screen, save to subfolder instead (good to test in non-interactive HPC)

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
    code_settings="Linear Tokamak",
    extraOptions={
        't_max':5.0,    # Run up to 5 a/c_s (should take ~1min using 8 A100s)
        'y0' :10.0,     # kymin = 1/y0 = 0.1
        'ny': 34,       # nky = 1 + (ny-1)/3 = 12 -> ky_range = 0.1 - 1.2
    },
    allocation = {
        "resources_per_call": 4,    # Each of the two radii with 4 GPUs
        "minutes": 10,
        }
    )
gx.read('gx1')

fn = GUItools.FigureNotebook("GX", geometry="1600x1000", show= not save_figures)

gx.plot(labels=['gx1'], fn = fn)

if not save_figures:
    gx.fn.show()
    gx.fn.close()
else:
    gx.fn.save(f'{folder}/figs_gx/')