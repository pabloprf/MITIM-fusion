import os
import numpy as np
from mitim_tools.gacode_tools.PROFILEStools import gacode_state
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools.physics import GXtools
from mitim_tools.misc_tools import IOtools, GUItools, GRAPHICStools
from mitim_tools import __mitimroot__

cold_start = True
save_figures = False # if True, do not show the plot to screen, save to subfolder instead (good to test in non-interactive HPC)

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "linear_gk_tutorial"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prepare case
# ---------------------------------------------------------------------------

p = gacode_state(gacode_file)
p.lumpIons()

rho = 0.7

# ---------------------------------------------------------------------------
# Run linear CGYRO
# ---------------------------------------------------------------------------

cgyro = CGYROtools.CGYRO(rhos = [rho])

cgyro.prep(p, folder / "cgyro")

run_type = 'normal'

cgyro.run_scan(
    'scan1',
    code_settings="Linear",
    extraOptions={
        'MAX_TIME': 50.0,
    },
    variable='KY',
    varUpDown=np.linspace(0.1,2.3,24),
    relativeChanges=False,
    allocation={
        'resources_per_call': 16,
        'minutes': 30,
        },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type=run_type
    )

cgyro.read_linear_scan(label="scan1", variable='KY', store_as_label="scan1", irho=0)

# ---------------------------------------------------------------------------
# Run linear GX
# ---------------------------------------------------------------------------

gx = GXtools.GX(rhos=[rho])
gx.prep(p, folder)

gx.run(
    'gx1/',
    cold_start=cold_start,
    code_settings="Linear Tokamak",
    extraOptions={
        't_max':50.0,
        'y0' :10.0,      # kymin = 1/y0 = 0.1
        'ny': 70,        # nky = 1 + (ny-1)/3 = 24 -> ky_range = 0.0 - 2.3
    },
    allocation = {
        "resources_per_call": 4,
        "minutes": 30,
        }
    )

gx.read('gx1')

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

fn = GUItools.FigureNotebook("GK", geometry="1600x1000", show= not save_figures)

# Combined

fig = fn.add_figure(label='Comparison')
axs = fig.subplot_mosaic(
        """
        fg
        """
    )

ax = axs['f']
ax.plot(np.abs(cgyro.results['scan1'].ky), cgyro.results['scan1'].f_mean, 'o-', label='CGYRO')
ax.plot(np.abs(gx.results['gx1']['output'][0].ky), gx.results['gx1']['output'][0].f_mean, 'o-', label='GX')
ax.set_xlabel('$k_y \\rho_s$')
ax.set_ylabel('Real frequency ($a/c_s$)')
ax.axhline(0, color='k', ls='--')
ax.legend()
GRAPHICStools.addDenseAxis(ax)

ax = axs['g']
ax.plot(np.abs(cgyro.results['scan1'].ky), cgyro.results['scan1'].g_mean, 'o-', label='CGYRO')
ax.plot(np.abs(gx.results['gx1']['output'][0].ky), gx.results['gx1']['output'][0].g_mean, 'o-', label='GX')
ax.set_xlabel('$k_y \\rho_s$')
ax.set_ylabel('Growth rate ($a/c_s$)')
ax.legend()
GRAPHICStools.addDenseAxis(ax)

# CGYRO

fig = fn.add_figure(label='CGYRO')
cgyro.plot_quick_linear(labels=["scan1"], fig = fig)

# GX

gx.plot(labels=['gx1'], fn = fn)


# Show or Save

if not save_figures:
    fn.show()
    fn.close()
else:
    fn.save(f'{folder}/figs_gk/')
