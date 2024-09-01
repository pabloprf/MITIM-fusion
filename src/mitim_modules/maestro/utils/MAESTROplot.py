import os
import numpy as np
from collections import OrderedDict
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat
from mitim_modules.maestro.utils.EPEDbeat import eped_beat

MARKERSIZE = 1
LW = 1.0

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
        elif 'run_eped' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('eped')

    # First initializer
    beat_initializer = None
    if 'initializer_freegs' in os.listdir(f'{folder_beats}/{beats[0]}'):
        beat_initializer = 'freegs'
    elif 'initializer_geqdsk' in os.listdir(f'{folder_beats}/{beats[0]}'):
        beat_initializer = 'geqdsk'
    elif 'initializer_profiles' in os.listdir(f'{folder_beats}/{beats[0]}'):
        beat_initializer = 'profiles'

    # Create "dummy" maestro by only defining the beats
    from mitim_modules.maestro.MAESTROmain import maestro
    m = maestro(folder, terminal_outputs = True)
    for i,beat in enumerate(beat_types):
        m.define_beat(beat, initializer = beat_initializer if i == 0 else None)

    # Plot
    m.plot(num_beats=num_beats, only_beats = only_beats, full_plot = full_plot)

    return m

def plot_results(self, fn):

    # ********************************************************************************************************
    # Collect info
    # ********************************************************************************************************

    # Collect initialization
    ini = {'geqdsk': None, 'profiles': PROFILEStools.PROFILES_GACODE(f'{self.beats[1].initialize.folder}/input.gacode')}
    if os.path.exists(f'{self.beats[1].initialize.folder}/input.geqdsk'):
        ini['geqdsk'] = GEQtools.MITIMgeqdsk(f'{self.beats[1].initialize.folder}/input.geqdsk')

    # Collect PORTALS profiles and TRANSP cdfs translated to profiles
    objs = OrderedDict()

    objs['Initial input.gacode'] = ini['profiles']

    for i,beat in enumerate(self.beats.values()):

        _, profs = beat.grab_output()

        if isinstance(beat, transp_beat):
            key = f'TRANSP beat #{i+1}'
        elif isinstance(beat, portals_beat):
            key = f'PORTALS beat #{i+1}'
        elif isinstance(beat, eped_beat):
            key = f'EPED beat #{i+1}'
        
        objs[key] = profs

    # ********************************************************************************************************
    # Plot profiles
    # ********************************************************************************************************
    ps, ps_lab = [], []
    for label in objs:
        if isinstance(objs[label], PROFILEStools.PROFILES_GACODE):
            ps.append(objs[label])
            ps_lab.append(label)

    if len(ps) > 0:
        # Plot profiles
        figs = PROFILEStools.add_figures(fn,fnlab_pre = 'MAESTRO - ')
        log_file = f'{self.folder_logs}/plot_maestro.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file):
            PROFILEStools.plotAll(ps, extralabs=ps_lab, figs=figs)

    for p,pl in zip(ps,ps_lab):
        p.printInfo(label = pl)

    keys = list(objs.keys())
    lw, ms = 1, 0

    # ********************************************************************************************************
    # Plot initialization (geqdsk to input.gacode)
    # ********************************************************************************************************

    fig = fn.add_figure(label='MAESTRO init', tab_color=2)
    axs = fig.subplot_mosaic(
        """
        ABCDH
        AEFGI
        """
    )
    axs = [ ax for ax in axs.values() ]

    if ini['geqdsk'] is not None:
        plot_g_quantities(ini['geqdsk'], axs, color = 'b', lw = lw, ms = ms)

    if objs[keys[0]] is not None:
        objs[keys[0]].plotRelevant(axs = axs, color = 'r', label =keys[0], lw = lw, ms = ms)

    GRAPHICStools.adjust_figure_layout(fig)

    # ********************************************************************************************************
    # Plot transition N -> N+1
    # ********************************************************************************************************

    label = "MAESTRO Transition"

    for i in range(len(objs)-1):
        obj1 = objs[keys[i]]
        obj2 = objs[keys[i+1]]

        if obj1 is None or obj2 is None:
            continue

        fig = fn.add_figure(label=f'{label} {i}->{i+1}', tab_color=2)
        axs = fig.subplot_mosaic(
            """
            ABCDH
            AEFGI
            """
        )
        axs = [ ax for ax in axs.values() ]

        obj1.plotRelevant(axs = axs, color = 'b', label =keys[i], lw = lw, ms = ms)
        obj2.plotRelevant(axs = axs, color = 'r', label =keys[i+1], lw = lw, ms = ms)

        GRAPHICStools.adjust_figure_layout(fig)

    # ********************************************************************************************************
    # Plot transition 0 -> last
    # ********************************************************************************************************

    fig = fn.add_figure(label=f'{label} {0}->{len(keys)}', tab_color=2)
    axs = fig.subplot_mosaic(
        """
        ABCDH
        AEFGI
        """
    )
    axs = [ ax for ax in axs.values() ]
    
    if ini['geqdsk'] is not None:
        plot_g_quantities(ini['geqdsk'], axs, color = 'm', lw = lw, ms = ms)
    if objs[keys[0]] is not None:
        objs[keys[0]].plotRelevant(axs = axs, color = 'b', label =keys[0], lw = lw, ms = ms)
    
    if objs[keys[-1]] is not None:
        objs[keys[-1]].plotRelevant(axs = axs, color = 'r', label =keys[-1], lw = lw, ms = ms)

    GRAPHICStools.adjust_figure_layout(fig)

    # ********************************************************************************************************
    # Plot special info
    # ********************************************************************************************************
    fig = fn.add_figure(label='MAESTRO special', tab_color=3)
    
    axs = fig.subplot_mosaic(
        """
        ABG
        ABG
        AEG
        DEH
        DFH
        DFH
        """
    )


    x, BetaN, Pfus, p_th, p_tot, Pin, Q, fG, nu_ne = [], [], [], [], [], [], [], [], []
    for p,pl in zip(ps,ps_lab):
        x.append(pl)
        BetaN.append(p.derived['BetaN'])
        Pfus.append(p.derived['Pfus'])
        p_th.append(p.derived['pthr_manual_vol'])
        p_tot.append(p.derived['ptot_manual_vol'])
        Pin.append(p.derived['qIn'])
        Q.append(p.derived['Q'])
        fG.append(p.derived['fG'])
        nu_ne.append(p.derived['ne_peaking0.2'])

    # -----------------------------------------------------------------
    ax = axs['A']
    ax.plot(x, BetaN, '-s', markersize=7, lw = 1)
    ax.set_ylabel('$\\beta_N$')
    ax.set_title('Pressure Evolution')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)

    ax.set_xticklabels([])

    ax = axs['D']
    ax.plot(x, p_th, '-s', markersize=7, lw = 1, label='Thermal')
    ax.plot(x, p_tot, '-o', markersize=7, lw = 1, label='Total')
    ax.set_ylabel('$p$ (MPa)')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)
    ax.legend()
    
    ax.tick_params(axis='x', rotation=45)

    # -----------------------------------------------------------------

    ax = axs['B']
    ax.plot(x, Q, '-s', markersize=7, lw = 1)
    ax.set_ylabel('$Q$')
    ax.set_title('Performance Evolution')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)

    ax.set_xticklabels([])


    ax = axs['E']
    ax.plot(x, Pfus, '-s', markersize=7, lw = 1)
    ax.set_ylabel('$P_{fus}$ (MW)')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)

    ax.set_xticklabels([])


    ax = axs['F']
    ax.plot(x, Pin, '-s', markersize=7, lw = 1)
    ax.set_ylabel('$P_{in}$ (MW)')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)
    
    ax.tick_params(axis='x', rotation=45)

    # -----------------------------------------------------------------
    ax = axs['G']
    ax.plot(x, fG, '-s', markersize=7, lw = 1)
    ax.set_ylabel('$f_{G}$')
    ax.set_title('Density Evolution')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)
    ax.axhline(y=1, color = 'k', lw = 2, ls = '--')

    ax.set_xticklabels([])

    ax = axs['H']
    ax.plot(x, nu_ne, '-s', markersize=7, lw = 1)
    ax.set_ylabel('$\\nu_{ne}$')
    GRAPHICStools.addDenseAxis(ax)
    ax.set_ylim(bottom = 0)
    
    ax.tick_params(axis='x', rotation=45)

    # -----------------------------------------------------------------



def plot_g_quantities(g, axs, color = 'b', lw = 1, ms = 0):

    g.plotFluxSurfaces(ax=axs[0], fluxes=np.linspace(0, 1, 21), rhoPol=False, sqrt=True, color=color,lwB=lw*3, lw = lw,label='Initial geqdsk')
    axs[3].plot(g.g['RHOVN'], g.g['PRES']*1E-6, '-o', markersize=ms, lw = lw, label='Initial geqdsk', color=color)
    axs[4].plot(g.g['RHOVN'], g.g['QPSI'], '-o', markersize=ms, lw = lw, label='Initial geqdsk', color=color)
