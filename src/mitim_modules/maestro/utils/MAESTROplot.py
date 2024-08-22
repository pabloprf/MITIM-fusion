import os
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.transp_tools import CDFtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat

def plotMAESTRO(folder, num_beats = 2, only_beats = None, full_plot = True):

    # Find beat results from folders
    folder_beats = f'{folder}/Beats/'
    beats = sorted(os.listdir(folder_beats))

    beat_types = [] 
    for beat in range(len(beats)):
        if 'run_transp' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('transp')
        elif 'run_portals' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('portals')
            
    beat_initializer = None
    if 'initializer_freegs' in os.listdir(f'{folder_beats}/{beats[0]}'):
        beat_initializer = 'freegs'

    # Create "dummy" maestro by only defining the beats
    from mitim_modules.maestro.MAESTROmain import maestro
    m = maestro(folder, terminal_outputs = False)
    for i,beat in enumerate(beat_types):
        m.define_beat(beat, initializer = beat_initializer if i == 0 else None)

    # Plot
    m.plot(num_beats=num_beats, only_beats = only_beats, full_plot = full_plot)

    return m

def plot_results(self, fn):

    # Collect PORTALS profiles and TRANSP cdfs
    ps = []
    ps_nums = []
    cdfs = []
    cdfs_nums = []
    for i,beat in enumerate(self.beats.values()):
        if isinstance(beat, portals_beat) and ('input.gacode' in os.listdir(beat.folder_output)):
            ps.append(PROFILEStools.PROFILES_GACODE(f'{beat.folder_output}/input.gacode'))
            ps_nums.append(i+1)

        if isinstance(beat, transp_beat):
            output_file = IOtools.findFileByExtension(beat.folder_output, '.CDF', agnostic_to_case=True, provide_full_path=True)
            if output_file is not None:
                cdfs.append(CDFtools.transp_output(output_file))
                cdfs_nums.append(i+1)
        
    # Collect geqdsk
    g = GEQtools.MITIMgeqdsk(f'{self.beats[1].initializer.folder}/freegs.geqdsk')

    # Plot profiles
    figs = PROFILEStools.add_figures(fn,fnlab_pre = "MAESTRO - ")
    log_file = f'{self.folder_logs}/plot_maestro.log' if (not self.terminal_outputs) else None
    with IOtools.conditional_log_to_file(log_file=log_file):
        PROFILEStools.plotAll(ps, extralabs=[f'beat_{i}' for i in range(len(ps))], figs=figs)

    for p in ps:
        p.printInfo()

    # Plot equilibria
    fig = fn.add_figure(label="MAESTRO Equilibria", tab_color=2)
    axs = fig.subplot_mosaic(
        """
        ABCD
        """
    )

    rho = np.linspace(0, 1, 11)

    colors = GRAPHICStools.listColors()

    ax = axs['A']

    g.plotFluxSurfaces(ax=ax, fluxes=rho, rhoPol=False, sqrt=True, color=colors[0], label='FreeGS')

    cont = 1
    for i in range(len(ps)):
        ps[i].plotGeometry(ax=ax, surfaces_rho=rho, label=f'PORTALS beat #{ps_nums[i]}', color=colors[cont])
        cont += 1
        ax.set_xlim([ps[0].derived['R_surface'].min()*0.95, ps[0].derived['R_surface'].max()*1.05])
    for i in range(len(cdfs)):
        it = cdfs[i].ind_saw - 1
        cdfs[i].plotGeometry(ax=ax, time=cdfs[i].t[it], rhoS=rho, label=f'TRANSP beat #{cdfs_nums[i]}',
            plotComplete=False, plotStructures=False, color=colors[cont], plotSurfs=True)
        cont += 1

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)

    # Plot Te profiles
    ax = axs['B']
    cont = 0
    for i in range(len(ps)):
        ax.plot(ps[i].profiles['rho(-)'], ps[i].profiles['te(keV)'], '-o', markersize=3, label=f'PORTALS beat #{ps_nums[i]}', color=colors[cont])
        cont += 1
    for i in range(len(cdfs)):
        it = cdfs[i].ind_saw - 1
        ax.plot(cdfs[i].x_lw, cdfs[i].Te[it], '--s', markersize=3, label=f'TRANSP beat #{cdfs_nums[i]}', color=colors[cont])
        cont += 1

    ax.set_xlabel("$\\rho_N$"); ax.set_xlim(0,1)
    ax.set_ylabel("$T_e$ (keV)"); ax.set_ylim(bottom = 0)
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)

    # Plot Ti profiles
    ax = axs['C']
    cont = 0
    for i in range(len(ps)):
        ax.plot(ps[i].profiles['rho(-)'], ps[i].profiles['ti(keV)'][:,0], '-o', markersize=3, label=f'PORTALS beat #{ps_nums[i]}', color=colors[cont])
        cont += 1
    for i in range(len(cdfs)):
        it = cdfs[i].ind_saw - 1
        ax.plot(cdfs[i].x_lw, cdfs[i].Ti[it], '--s', markersize=3, label=f'TRANSP beat #{cdfs_nums[i]}', color=colors[cont])
        cont += 1

    ax.set_xlabel("$\\rho_N$"); ax.set_xlim(0,1)
    ax.set_ylabel("$T_i$ (keV)"); ax.set_ylim(bottom = 0)
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)

    # Plot ne profiles
    ax = axs['D']
    cont = 0
    for i in range(len(ps)):
        ax.plot(ps[i].profiles['rho(-)'], ps[i].profiles['ne(10^19/m^3)']*1E-1, '-o', markersize=3, label=f'PORTALS beat #{ps_nums[i]}', color=colors[cont])
        cont += 1
    for i in range(len(cdfs)):
        it = cdfs[i].ind_saw - 1
        ax.plot(cdfs[i].x_lw, cdfs[i].ne[it], '--s', markersize=3, label=f'TRANSP beat #{cdfs_nums[i]}', color=colors[cont])
        cont += 1

    ax.set_xlabel("$\\rho_N$"); ax.set_xlim(0,1)
    ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)"); ax.set_ylim(bottom = 0)
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)