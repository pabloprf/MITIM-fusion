import os
import numpy as np
from collections import OrderedDict
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.transp_tools import CDFtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat

MARKERSIZE = 2
LW = 0.5

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
    objs = OrderedDict()
    for i,beat in enumerate(self.beats.values()):
        if isinstance(beat, portals_beat) and ('input.gacode' in os.listdir(beat.folder_output)):
            objs[f'PORTALS beat #{i+1} (final)'] = PROFILEStools.PROFILES_GACODE(f'{beat.folder_output}/input.gacode')

        if isinstance(beat, transp_beat):
            output_file = IOtools.findFileByExtension(beat.folder_output, '.CDF', agnostic_to_case=True, provide_full_path=True)
            if output_file is not None:
                objs[f'TRANSP beat #{i+1} (final)'] = CDFtools.transp_output(output_file)

    # Collect initialization
    g = GEQtools.MITIMgeqdsk(f'{self.beats[1].initializer.folder}/freegs.geqdsk')

    # Plot profiles
    _plot_profiles_evolution(self,objs, fn, fnlab_pre = "MAESTRO - ")

    # Plot transitions
    _plot_transitions(self, objs, fn, g= g, label = "MAESTRO Equilibria")

def _plot_profiles_evolution(self,objs, fn, fnlab_pre = ""):

    ps = []
    ps_lab = []
    for label in objs:
        if isinstance(objs[label], PROFILEStools.PROFILES_GACODE):
            ps.append(objs[label])
            ps_lab.append(label)

    # Plot profiles
    figs = PROFILEStools.add_figures(fn,fnlab_pre = fnlab_pre)
    log_file = f'{self.folder_logs}/plot_maestro.log' if (not self.terminal_outputs) else None
    with IOtools.conditional_log_to_file(log_file=log_file):
        PROFILEStools.plotAll(ps, extralabs=ps_lab, figs=figs)

    for p in ps:
        p.printInfo()

def _plot_transitions(self, objs, fn, g=None, label = ""):

    keys = list(objs.keys())

    # Plot transitions 0 -> 1
    fig = fn.add_figure(label=f'{label} 0->1', tab_color=2)
    axs = fig.subplot_mosaic(
        """
            ABCDH
            AEFGI
        """
    )
    obj1 = g
    obj1.labelMAESTRO = 'geqdsk'
    obj2 = objs[keys[0]]
    obj2.labelMAESTRO = keys[0]
    _plot_transition(self, obj1, obj2, axs)

    GRAPHICStools.adjust_figure_layout(fig)

    # Plot transitions N -> N+1
    for i in range(len(objs)-1):
        obj1 = objs[keys[i]]
        obj2 = objs[keys[i+1]]

        obj1.labelMAESTRO = keys[i]
        obj2.labelMAESTRO = keys[i+1]

        fig = fn.add_figure(label=f'{label} {i+1}->{i+2}', tab_color=2)
        axs = fig.subplot_mosaic(
            """
            ABCDH
            AEFGI
            """
        )

        _plot_transition(self, obj1, obj2, axs)

        GRAPHICStools.adjust_figure_layout(fig)

def _plot_transition(self, obj1, obj2, axs):

    # ----------------------------------
    # Equilibria
    # ----------------------------------

    ax = axs['A']
    rho = np.linspace(0, 1, 21)
    
    for obj, color, lw in zip([obj1, obj2], ['b', 'r'], [1.0,0.5]):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            obj.plotGeometry(ax=ax, surfaces_rho=rho, label=obj.labelMAESTRO, color=color, lw=lw, lw1=lw*3)
            ax.set_xlim(obj.derived['R_surface'].min()*0.9, obj.derived['R_surface'].max()*1.1)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            obj.plotGeometry(ax=ax, time=obj.t[it], rhoS=rho, label=obj.labelMAESTRO,lwB=lw*3, lw = lw,
                plotComplete=False, plotStructures=False, color=color, plotSurfs=True)
            ax.set_xlim(obj.Rmin[it].min()*0.9, obj.Rmax[it].max()*1.1)
        elif isinstance(obj, GEQtools.MITIMgeqdsk):
            obj.plotFluxSurfaces(ax=ax, fluxes=rho, rhoPol=False, sqrt=True, color=color,lwB=lw*3, lw = lw,
            label=obj.labelMAESTRO)

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Equilibria")


    # ----------------------------------
    # Kinetic Profiles
    # ----------------------------------

    # T profiles
    ax = axs['B']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.profiles['te(keV)'], '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)
            ax.plot(obj.profiles['rho(-)'], obj.profiles['ti(keV)'][:,0], '-*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', i', color=color)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.Te[it], '--o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)
            ax.plot(obj.x_lw, obj.Ti[it], '--*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', i', color=color)

    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$T$ (keV)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Temperatures")

    # ne profiles
    ax = axs['C']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.profiles['ne(10^19/m^3)']*1E-1, '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.ne[it], '--s', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)

    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Electron Density")

    # ----------------------------------
    # Pressure
    # ----------------------------------

    ax = axs['D']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.profiles['ptot(Pa)']*1E-6, '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.p_kin[it], '--s', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)
        elif isinstance(obj, GEQtools.MITIMgeqdsk):
            ax.plot(obj.g['RHOVN'], obj.g['PRES']*1E-6, '-.*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)


    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$p_{kin}$ (MPa)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Total Pressure")




    # ----------------------------------
    # Current
    # ----------------------------------

    # q-profile
    ax = axs['H']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.profiles['q(-)'], '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.xb_lw, obj.q[it], '--s', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)
        elif isinstance(obj, GEQtools.MITIMgeqdsk):
            ax.plot(obj.g['RHOVN'], obj.g['QPSI'], '-.*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)


    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$q$")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Safety Factor")


    # ----------------------------------
    # Powers
    # ----------------------------------

    # RF
    ax = axs['E']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.profiles['qrfe(MW/m^3)'], '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)
            ax.plot(obj.profiles['rho(-)'], obj.profiles['qrfi(MW/m^3)'], '-*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', i', color=color)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.Peich[it], '--o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)
            ax.plot(obj.x_lw, obj.Peich[it], '--*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', i', color=color)

    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$P_{ich}$ (MW/m$^3$)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("ICH Power Deposition")

    # Ohmic
    ax = axs['F']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.profiles['qohme(MW/m^3)'], '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)
        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.Poh[it], '--s', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO, color=color)

    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$P_{oh}$ (MW/m$^3$)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Ohmic Power Deposition")

    # ----------------------------------
    # Heat fluxes
    # ----------------------------------

    ax = axs['G']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.derived['qe_MWm2'], '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)
            ax.plot(obj.profiles['rho(-)'], obj.derived['qi_MWm2'], '-*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', i', color=color)

        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.qe_obs_GACODE[it], '--o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)
            ax.plot(obj.x_lw, obj.qi_obs_GACODE[it], '--*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', e', color=color)


    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$Q$ ($MW/m^2$)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Energy Fluxes")

    # ----------------------------------
    # Dynamic targets
    # ----------------------------------

    ax = axs['I']

    for obj, color in zip([obj1, obj2], ['b', 'r']):
        if isinstance(obj, PROFILEStools.PROFILES_GACODE):
            ax.plot(obj.profiles['rho(-)'], obj.derived['qrad'], '-o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', rad', color=color)
            ax.plot(obj.profiles['rho(-)'], obj.profiles['qei(MW/m^3)'], '-*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', exc', color=color)
            ax.plot(obj.profiles['rho(-)'], obj.profiles['qfuse(MW/m^3)']+obj.profiles['qfusi(MW/m^3)'], '-s', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', fus', color=color)

        elif isinstance(obj, CDFtools.transp_output):
            it = obj.ind_saw - 1
            ax.plot(obj.x_lw, obj.Prad[it], '--o', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', rad', color=color)
            ax.plot(obj.x_lw, obj.Pei[it], '--*', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', exc', color=color)
            ax.plot(obj.x_lw, obj.Pfuse[it]+obj.Pfusi[it], '--s', markersize=MARKERSIZE, lw = LW, label=obj.labelMAESTRO+', fus', color=color)


    ax.set_xlabel("$\\rho_N$")
    ax.set_ylabel("$Q$ ($MW/m^2$)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(0,1)
    ax.legend(prop={'size':8})
    GRAPHICStools.addDenseAxis(ax)
    ax.set_title("Dynamic Targets")
