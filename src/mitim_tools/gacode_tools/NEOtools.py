import os
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools, GUItools, FARMINGtools
from mitim_tools.gacode_tools.utils import GACODErun, GACODEdefaults, GACODEinprocess
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.style_tools.themes import apply_theme
from mitim_tools import __mitimroot__
from IPython import embed

class NEO(SIMtools.mitim_simulation, GACODEinprocess.NEOInProcess):
    """
    NEO wrapper.  Uses multiple inheritance to combine two engines:

    * ``SIMtools.mitim_simulation`` — the standard subprocess / SLURM engine
      (the file-based ``prep`` / ``_run_prepare`` / ``_run`` / ``read``).
    * ``GACODEinprocess.NEOInProcess`` — the pure in-process (ctypes)
      mixin providing ``prep_inprocess`` / ``_run_prepare_inprocess`` /
      ``_run_inprocess`` / ``read_inprocess``.

    The four small dispatch methods below decide between the two engines
    based on the ``self.in_process`` flag set at construction time.
    """
    def __init__(
        self,
        rhos=[0.4, 0.6],   # rho locations of interest
        in_process=False,  # If True, run NEO in-process via ctypes (no subprocess); requires libneo_serial.so
    ):

        SIMtools.mitim_simulation.__init__(self, rhos=rhos)
        self.in_process = in_process
        # In-process result cache (lazy-init via mixin); harmless when not used.
        self._init_inprocess()

        def code_call(folder, n, p, additional_command="", **kwargs):
            return f"neo -e {folder} -n {n} -p {p} {additional_command}"

        self.run_specifications = {
            'code': 'neo',
            'input_file': 'input.neo',
            'code_call': code_call,
            'control_function': GACODEdefaults.addNEOcontrol,
            'controls_file': 'input.neo.controls',
            'state_converter': 'to_neo',
            'input_class': NEOinput,
            'complete_variation': None,
            'default_cores': 1,  # Default cores to use in the simulation
            'output_class': NEOoutput,
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t NEO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.output_files_simulation["minimal"] = ['out.neo.transport_flux']
        self.output_files_simulation["complete"] = [
            'out.neo.transport_flux',
            'out.neo.transport',
            'out.neo.transport_gv',
            'out.neo.equil',
            'out.neo.theory',
            'out.neo.rotation',
            'out.neo.grid',
            'out.neo.diagnostic_geo',
            'out.neo.diagnostic_geo2',
            'out.neo.prec',
            'out.neo.run',
            'out.neo.version',
        ]

    # ------------------------------------------------------------------
    # In-process / subprocess dispatch.  Each method picks the engine
    # based on self.in_process and forwards to either:
    #   * GACODEinprocess.NEOInProcess.<method>_inprocess  (in-process)
    #   * SIMtools.mitim_simulation.<method> via super()    (subprocess)
    # The actual physics/IO logic lives in those two parents — these
    # dispatchers are intentionally tiny.
    # ------------------------------------------------------------------

    def prep(self, mitim_state, FolderGACODE=None, **kwargs):
        if self.in_process:
            # In-process needs no folder — anything passed for FolderGACODE
            # is silently ignored, and a synthetic in-memory path is used
            # internally for cache keys.
            return self.prep_inprocess(mitim_state)
        return super().prep(mitim_state, FolderGACODE, **kwargs)

    def _run_prepare(self, subfolder_simulation, **kwargs):
        if self.in_process:
            return self._run_prepare_inprocess(subfolder_simulation, **kwargs)
        return super()._run_prepare(subfolder_simulation, **kwargs)

    def _run(self, code_executor, **kwargs):
        if self.in_process:
            return self._run_inprocess(code_executor, **kwargs)
        return super()._run(code_executor, **kwargs)

    def read(self, label="run1", folder=None, **kwargs):
        if self.in_process:
            return self.read_inprocess(label=label, folder=folder)
        return super().read(label=label, folder=folder, **kwargs)

    def prep_from_file(
        self,
        FolderGACODE,  # Main folder where the run lives
        input_neo_file,  # input.neo file to start with
        input_gacode=None,
    ):
        """Prepare a NEO class to read an already-run folder directly from its
        input.neo (no mitim_state needed). Mirrors TGLF.prep_from_file: sets the
        normalizations from input.gacode (if given) and the radial location from
        the input file's RMIN_OVER_A, so read() can be called on the outputs."""
        print("> Preparation of NEO class directly from input.neo")

        from mitim_tools.gacode_tools import PROFILEStools
        from mitim_tools.gacode_tools.utils import NORMtools

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)

        self.NormalizationSets, _ = NORMtools.normalizations(
            PROFILEStools.gacode_state(input_gacode) if input_gacode is not None else None)

        inputclass = NEOinput(file=input_neo_file)

        roa = inputclass.plasma["RMIN_OVER_A"]
        print(f"\t- This file correspond to r/a={roa} according to RMIN_OVER_A")

        if self.NormalizationSets["input_gacode"] is not None:
            rho = np.interp(
                roa,
                self.NormalizationSets["input_gacode"].derived["roa"],
                self.NormalizationSets["input_gacode"].profiles["rho(-)"],
            )
            print(f"\t\t- rho={rho:.4f}, using input.gacode for conversion")
        else:
            print(
                "\t\t- No input.gacode for conversion, assuming rho=r/a, EXTREME CAUTION PLEASE",
                typeMsg="w",
            )
            rho = roa

        self.rhos = [rho]
        self.inputs_files = {self.rhos[0]: inputclass}

    def plot(
        self,
        fn=None,
        labels=["neo1"],
        extratitle="",
        fn_color=None,
        colors=None,
        ):

        apply_theme()

        if fn is None:
            self.fn = GUItools.FigureNotebook("NEO MITIM Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn

        if colors is None:
            colors = GRAPHICStools.listColors()

        # Reference output object used to check which optional attributes are present
        o0 = self.results[labels[0]]['output'][0]

        # Simulated roa range (with small margin) — used for all x-axis limits
        all_roas = [self.results[lbl]['output'][irho].roa
                    for lbl in labels for irho in range(len(self.rhos))]
        _margin  = 0.02
        _xlim    = [max(0, min(all_roas) - _margin), min(1, max(all_roas) + _margin)]

        # ---- Tab 1 & 2: Summary fluxes (GB and physical units) ----
        type_plots = {
            ' (GB)': (['Qe', 'Qi', 'Ge'],['GB', 'GB', 'GB']),
            ' (real)': (['Qe_unn', 'Qi_unn', 'Ge_unn'],['$MW/m^2$', '$MW/m^2$', '$1E20/s/m^2$'])
        }

        for suffix, (variables, labels_y) in type_plots.items():
            fig1 = self.fn.add_figure(label=f"{extratitle}NEO summary{suffix}", tab_color=fn_color)

            grid = plt.GridSpec(1, 3, hspace=0.7, wspace=0.2)

            axQe = fig1.add_subplot(grid[0, 0])
            axQi = fig1.add_subplot(grid[0, 1])
            axGe = fig1.add_subplot(grid[0, 2])

            for i,label in enumerate(labels):
                roa, Qe, Qi, Ge = [], [], [], []
                for irho in range(len(self.rhos)):
                    o = self.results[label]['output'][irho]
                    roa.append(o.roa)
                    Qe.append(o.__dict__.get(variables[0], np.nan))
                    Qi.append(o.__dict__.get(variables[1], np.nan))
                    Ge.append(o.__dict__.get(variables[2], np.nan))

                axQe.plot(roa, Qe, label=label, color=colors[i], marker='o', linestyle='-')
                axQi.plot(roa, Qi, label=label, color=colors[i], marker='o', linestyle='-')
                axGe.plot(roa, Ge, label=label, color=colors[i], marker='o', linestyle='-')

            for ax in [axQe, axQi, axGe]:
                ax.set_xlabel("$r/a$"); ax.set_xlim(_xlim)
                ax.legend(loc="best")

            axQe.set_ylabel(f"$Q_e$ ({labels_y[0]})"); axQe.set_yscale('log')
            axQi.set_ylabel(f"$Q_i$ ({labels_y[1]})"); axQi.set_yscale('log')
            axGe.set_ylabel(f"$\\Gamma_e$ ({labels_y[2]})")

        # ---- Tab 1b: Bootstrap current summary (GB + physical) ----
        if hasattr(o0, 'jparB'):
            fig1b = self.fn.add_figure(label=f"{extratitle}NEO bootstrap", tab_color=fn_color)
            grid1b = plt.GridSpec(2, 2, hspace=0.5, wspace=0.35)
            axTe   = fig1b.add_subplot(grid1b[0, 0])
            axNe   = fig1b.add_subplot(grid1b[0, 1])
            axJgb  = fig1b.add_subplot(grid1b[1, 0])
            axJphy = fig1b.add_subplot(grid1b[1, 1])

            # Top row: T and n profiles with twinx (same for all labels)
            norm = self.NormalizationSets.get("SELECTED") if hasattr(self, 'NormalizationSets') else None
            for ax_T, T_key, T_lbl, n_key, n_lbl, ttl in [
                (axTe, "Te_keV", "$T_e$ (keV)", "ne_20", "$n_e$", "Electrons"),
                (axNe, "Te_keV", "$T_e$ (keV)", "ne_20", "$n_e$", "Electrons"),
            ]:
                ax_n = ax_T.twinx()
                if norm is not None:
                    roa_prof = norm["roa"]
                    mask = (roa_prof >= _xlim[0]) & (roa_prof <= _xlim[1])
                    ax_T.plot(roa_prof[mask], norm[T_key][mask], color='r', lw=1.5, label=T_lbl)
                    ax_n.plot(roa_prof[mask], norm[n_key][mask], color='b', lw=1.5, ls='--', label=n_lbl)
                ax_T.set_xlabel("$r/a$"); ax_T.set_xlim(_xlim)
                ax_T.set_ylabel(T_lbl, color='r'); ax_T.tick_params(axis='y', colors='r')
                ax_n.set_ylabel(f"{n_lbl} ($10^{{20}}\\,m^{{-3}}$)", color='b'); ax_n.tick_params(axis='y', colors='b')
                ax_T.set_title(ttl, fontsize=9)
                lines_T, labs_T = ax_T.get_legend_handles_labels()
                lines_n, labs_n = ax_n.get_legend_handles_labels()
                ax_T.legend(lines_T + lines_n, labs_T + labs_n, loc="best", fontsize=7)

            # Bottom row: bootstrap current vs r/a
            for i, label in enumerate(labels):
                roa    = [self.results[label]['output'][irho].roa for irho in range(len(self.rhos))]
                jgb    = [self.results[label]['output'][irho].jparB for irho in range(len(self.rhos))]
                jphy   = [getattr(self.results[label]['output'][irho], 'jparB_unn', np.nan) for irho in range(len(self.rhos))]
                HHjgb  = [getattr(self.results[label]['output'][irho], 'HHjparB', np.nan) for irho in range(len(self.rhos))]
                HHjphy = [getattr(self.results[label]['output'][irho], 'HHjparB_unn', np.nan) for irho in range(len(self.rhos))]
                Sjgb   = [getattr(self.results[label]['output'][irho], 'SjparB', np.nan) for irho in range(len(self.rhos))]
                Sjphy  = [getattr(self.results[label]['output'][irho], 'SjparB_unn', np.nan) for irho in range(len(self.rhos))]

                c = colors[i]
                axJgb.plot( roa, jgb,   label=f"{label} NEO",    color=c, marker='o', ls='-')
                axJgb.plot( roa, HHjgb, label=f"{label} H-H",    color=c, marker='^', ls='--')
                axJgb.plot( roa, Sjgb,  label=f"{label} Sauter", color=c, marker='s', ls=':')
                axJphy.plot(roa, jphy,   label=f"{label} NEO",    color=c, marker='o', ls='-')
                axJphy.plot(roa, HHjphy, label=f"{label} H-H",    color=c, marker='^', ls='--')
                axJphy.plot(roa, Sjphy,  label=f"{label} Sauter", color=c, marker='s', ls=':')

            for ax, yl in [
                (axJgb,  "GB"),
                (axJphy, "$kA/m^2$"),
            ]:
                ax.set_xlabel("$r/a$"); ax.set_xlim(_xlim)
                ax.set_title("$\\langle j_{\\parallel} B\\rangle / B_{unit}$ (bootstrap current)", fontsize=9)
                ax.set_ylabel(f"({yl})")
                ax.axhline(0, color='k', lw=0.8, ls='--')
                ax.legend(loc="best", fontsize=7)

        # ---- Tab 3: Per-species fluxes (DKE, GV, total) ----
        if hasattr(o0, 'GiAll_dke'):
            fig3 = self.fn.add_figure(label=f"{extratitle}NEO species fluxes", tab_color=fn_color)
            n_ions = len(o0.GiAll)
            # rows: 2 (dke, gv), cols: 3 (Gamma, Q, Pi)
            grid3 = plt.GridSpec(2, 3, hspace=0.5, wspace=0.35)
            ax_titles = [("$\\Gamma$ (pflux)", "dke"), ("$Q$ (eflux)", "dke"), ("$\\Pi$ (mflux)", "dke"),
                         ("$\\Gamma$ (pflux)", "gv"),  ("$Q$ (eflux)", "gv"),  ("$\\Pi$ (mflux)", "gv")]
            axs3 = [[fig3.add_subplot(grid3[r, c]) for c in range(3)] for r in range(2)]

            for i, label in enumerate(labels):
                roa = [self.results[label]['output'][irho].roa for irho in range(len(self.rhos))]
                for sec_idx, sec in enumerate(['dke', 'gv']):
                    Ge_sec = [getattr(self.results[label]['output'][irho], f'Ge_{sec}', np.nan) for irho in range(len(self.rhos))]
                    Gi_sec = [getattr(self.results[label]['output'][irho], f'GiAll_{sec}', np.full(n_ions, np.nan)) for irho in range(len(self.rhos))]
                    Qe_sec = [getattr(self.results[label]['output'][irho], f'Qe_{sec}', np.nan) for irho in range(len(self.rhos))]
                    Qi_sec = [getattr(self.results[label]['output'][irho], f'QiAll_{sec}', np.full(n_ions, np.nan)) for irho in range(len(self.rhos))]
                    Me_sec = [getattr(self.results[label]['output'][irho], f'Me_{sec}', np.nan) for irho in range(len(self.rhos))]
                    Mi_sec = [getattr(self.results[label]['output'][irho], f'MiAll_{sec}', np.full(n_ions, np.nan)) for irho in range(len(self.rhos))]

                    lbl_e = f"{label} e" if i == 0 else label
                    axs3[sec_idx][0].plot(roa, Ge_sec, label=f"{label} e", color=colors[i], marker='o', ls='-')
                    axs3[sec_idx][1].plot(roa, Qe_sec, label=f"{label} e", color=colors[i], marker='o', ls='-')
                    axs3[sec_idx][2].plot(roa, Me_sec, label=f"{label} e", color=colors[i], marker='o', ls='-')
                    for ii in range(n_ions):
                        ls = ['--', ':', '-.', (0,(3,1,1,1))][ii % 4]
                        axs3[sec_idx][0].plot(roa, [g[ii] if hasattr(g,'__len__') else np.nan for g in Gi_sec], label=f"{label} i{ii+1}", color=colors[i], marker='s', ls=ls)
                        axs3[sec_idx][1].plot(roa, [q[ii] if hasattr(q,'__len__') else np.nan for q in Qi_sec], label=f"{label} i{ii+1}", color=colors[i], marker='s', ls=ls)
                        axs3[sec_idx][2].plot(roa, [m[ii] if hasattr(m,'__len__') else np.nan for m in Mi_sec], label=f"{label} i{ii+1}", color=colors[i], marker='s', ls=ls)

            row_labels = ['DKE', 'GV']
            col_labels = ['$\\Gamma$ (GB)', '$Q$ (GB)', '$\\Pi$ (GB)']
            for r in range(2):
                for c in range(3):
                    ax = axs3[r][c]
                    ax.set_xlabel("$r/a$"); ax.set_xlim(_xlim)
                    ax.set_title(f"{row_labels[r]}: {col_labels[c]}", fontsize=9)
                    ax.legend(loc="best", fontsize=6)

        # ---- Tab 4: Theory comparison ----
        if hasattr(o0, 'HHQi'):
            fig4 = self.fn.add_figure(label=f"{extratitle}NEO theory", tab_color=fn_color)
            grid4 = plt.GridSpec(2, 3, hspace=0.55, wspace=0.35)
            axHHQi  = fig4.add_subplot(grid4[0, 0])
            axHHQe  = fig4.add_subplot(grid4[0, 1])
            axHHG   = fig4.add_subplot(grid4[0, 2])
            axJpar  = fig4.add_subplot(grid4[1, 0])
            axUpar  = fig4.add_subplot(grid4[1, 1])
            axVpol  = fig4.add_subplot(grid4[1, 2])

            for i, label in enumerate(labels):
                roa   = [self.results[label]['output'][irho].roa for irho in range(len(self.rhos))]
                Qi_n  = [self.results[label]['output'][irho].Qi for irho in range(len(self.rhos))]
                Qe_n  = [self.results[label]['output'][irho].Qe for irho in range(len(self.rhos))]
                Ge_n  = [self.results[label]['output'][irho].Ge for irho in range(len(self.rhos))]
                HHQi  = [getattr(self.results[label]['output'][irho], 'HHQi', np.nan) for irho in range(len(self.rhos))]
                HHQe  = [getattr(self.results[label]['output'][irho], 'HHQe', np.nan) for irho in range(len(self.rhos))]
                HHG   = [getattr(self.results[label]['output'][irho], 'HHGamma', np.nan) for irho in range(len(self.rhos))]
                CHQi  = [getattr(self.results[label]['output'][irho], 'CHQi', np.nan) for irho in range(len(self.rhos))]
                TGQi  = [getattr(self.results[label]['output'][irho], 'TGQi', np.nan) for irho in range(len(self.rhos))]
                jparB = [getattr(self.results[label]['output'][irho], 'jparB', np.nan) for irho in range(len(self.rhos))]
                HHjparB=[getattr(self.results[label]['output'][irho], 'HHjparB', np.nan) for irho in range(len(self.rhos))]
                uparB0=[getattr(self.results[label]['output'][irho], 'uparB0', np.nan) for irho in range(len(self.rhos))]
                HHuparB=[getattr(self.results[label]['output'][irho], 'HHuparB', np.nan) for irho in range(len(self.rhos))]
                vtheta0=[getattr(self.results[label]['output'][irho], 'vtheta0', np.nan) for irho in range(len(self.rhos))]
                HHvtheta=[getattr(self.results[label]['output'][irho], 'HHvtheta', np.nan) for irho in range(len(self.rhos))]

                c = colors[i]
                axHHQi.plot(roa, Qi_n,  label=f"{label} NEO",  color=c, marker='o', ls='-')
                axHHQi.plot(roa, HHQi,  label=f"{label} H-H",  color=c, marker='^', ls='--')
                axHHQi.plot(roa, CHQi,  label=f"{label} C-H",  color=c, marker='s', ls=':')
                axHHQi.plot(roa, TGQi,  label=f"{label} T-G",  color=c, marker='D', ls='-.')
                axHHQe.plot(roa, Qe_n,  label=f"{label} NEO",  color=c, marker='o', ls='-')
                axHHQe.plot(roa, HHQe,  label=f"{label} H-H",  color=c, marker='^', ls='--')
                axHHG.plot( roa, Ge_n,  label=f"{label} NEO",  color=c, marker='o', ls='-')
                axHHG.plot( roa, HHG,   label=f"{label} H-H",  color=c, marker='^', ls='--')
                axJpar.plot(roa, jparB, label=f"{label} NEO",  color=c, marker='o', ls='-')
                axJpar.plot(roa, HHjparB,label=f"{label} H-H", color=c, marker='^', ls='--')
                axUpar.plot(roa, uparB0,label=f"{label} NEO",  color=c, marker='o', ls='-')
                axUpar.plot(roa, HHuparB,label=f"{label} H-H", color=c, marker='^', ls='--')
                axVpol.plot(roa, vtheta0,label=f"{label} NEO", color=c, marker='o', ls='-')
                axVpol.plot(roa, HHvtheta,label=f"{label} H-H",color=c, marker='^', ls='--')

            for ax, title, yl in [
                (axHHQi,  "$Q_i$ theory comparison", "GB"),
                (axHHQe,  "$Q_e$ theory comparison", "GB"),
                (axHHG,   "$\\Gamma$ theory comparison", "GB"),
                (axJpar,  "$j_{\\parallel}B$ theory comparison", "GB"),
                (axUpar,  "$u_{\\parallel}B$ 0th-order", "GB"),
                (axVpol,  "$v_\\theta$ 0th-order", "GB"),
            ]:
                ax.set_xlabel("$r/a$"); ax.set_xlim(_xlim)
                ax.set_title(title, fontsize=9)
                ax.set_ylabel(f"({yl})")
                ax.legend(loc="best", fontsize=6)

        # ---- Tab 5: Rotation ----
        if hasattr(o0, 'dphi_ave'):
            fig5 = self.fn.add_figure(label=f"{extratitle}NEO rotation", tab_color=fn_color)
            n_ions_rot = len(o0.V_conv) if hasattr(o0, 'V_conv') else 0
            grid5 = plt.GridSpec(2, 2, hspace=0.5, wspace=0.35)
            axDphi  = fig5.add_subplot(grid5[0, 0])
            axNrat  = fig5.add_subplot(grid5[0, 1])
            axVconv = fig5.add_subplot(grid5[1, 0])
            axPhi   = fig5.add_subplot(grid5[1, 1])

            for i, label in enumerate(labels):
                roa    = [self.results[label]['output'][irho].roa for irho in range(len(self.rhos))]
                dphi   = [getattr(self.results[label]['output'][irho], 'dphi_ave', np.nan) for irho in range(len(self.rhos))]
                n_ratio= [getattr(self.results[label]['output'][irho], 'n_ratio', np.full(n_ions_rot, np.nan)) for irho in range(len(self.rhos))]
                V_conv = [getattr(self.results[label]['output'][irho], 'V_conv',  np.full(n_ions_rot, np.nan)) for irho in range(len(self.rhos))]

                axDphi.plot(roa, dphi, label=label, color=colors[i], marker='o', ls='-')

                for ii in range(n_ions_rot):
                    ls = ['-', '--', ':', '-.'][ii % 4]
                    axNrat.plot( roa, [v[ii] if hasattr(v,'__len__') else np.nan for v in n_ratio], label=f"{label} s{ii+1}", color=colors[i], marker='o', ls=ls)
                    axVconv.plot(roa, [v[ii] if hasattr(v,'__len__') else np.nan for v in V_conv],  label=f"{label} s{ii+1}", color=colors[i], marker='o', ls=ls)

                # phi_theta vs theta for first rho
                o_first = self.results[label]['output'][0]
                if hasattr(o_first, 'phi_theta') and hasattr(o_first, 'theta_grid'):
                    axPhi.plot(o_first.theta_grid / np.pi, o_first.phi_theta, label=f"{label} roa={o_first.roa:.3f}", color=colors[i], ls='-')

            axDphi.set_xlabel("$r/a$"); axDphi.set_xlim(_xlim); axDphi.set_ylabel("$d\\phi/dr$ (avg)")
            axNrat.set_xlabel("$r/a$"); axNrat.set_xlim(_xlim); axNrat.set_ylabel("$n/n_0$ ratio")
            axVconv.set_xlabel("$r/a$"); axVconv.set_xlim(_xlim); axVconv.set_ylabel("$V_{conv}$ (GB)")
            axPhi.set_xlabel("$\\theta/\\pi$"); axPhi.set_ylabel("$\\phi_{rot}(\\theta)$")
            axPhi.set_title("Rotation potential (1st rho)", fontsize=9)
            for ax in [axDphi, axNrat, axVconv, axPhi]:
                ax.legend(loc="best", fontsize=7)

        # ---- Tab 6: Neoclassical transport details ----
        if hasattr(o0, 'jparB'):
            fig6 = self.fn.add_figure(label=f"{extratitle}NEO neocl. transport", tab_color=fn_color)
            n_sp = len(o0.uparB_sp) if hasattr(o0, 'uparB_sp') else 0
            grid6 = plt.GridSpec(2, 3, hspace=0.5, wspace=0.35)
            axJ   = fig6.add_subplot(grid6[0, 0])
            axU0  = fig6.add_subplot(grid6[0, 1])
            axVt0 = fig6.add_subplot(grid6[0, 2])
            axUsp = fig6.add_subplot(grid6[1, 0])
            axVth = fig6.add_subplot(grid6[1, 1])
            axVph = fig6.add_subplot(grid6[1, 2])

            for i, label in enumerate(labels):
                roa   = [self.results[label]['output'][irho].roa for irho in range(len(self.rhos))]
                jparB = [getattr(self.results[label]['output'][irho], 'jparB', np.nan) for irho in range(len(self.rhos))]
                uparB0= [getattr(self.results[label]['output'][irho], 'uparB0', np.nan) for irho in range(len(self.rhos))]
                vth0  = [getattr(self.results[label]['output'][irho], 'vtheta0', np.nan) for irho in range(len(self.rhos))]
                axJ.plot(  roa, jparB, label=label, color=colors[i], marker='o', ls='-')
                axU0.plot( roa, uparB0, label=label, color=colors[i], marker='o', ls='-')
                axVt0.plot(roa, vth0,  label=label, color=colors[i], marker='o', ls='-')

                for ii in range(n_sp):
                    ls = ['-', '--', ':', '-.'][ii % 4]
                    usp  = [getattr(self.results[label]['output'][irho], 'uparB_sp', np.full(n_sp, np.nan))[ii] for irho in range(len(self.rhos))]
                    vth  = [getattr(self.results[label]['output'][irho], 'vtheta_sp', np.full(n_sp, np.nan))[ii] for irho in range(len(self.rhos))]
                    vph  = [getattr(self.results[label]['output'][irho], 'vphi_sp', np.full(n_sp, np.nan))[ii] for irho in range(len(self.rhos))]
                    axUsp.plot(roa, usp, label=f"{label} s{ii+1}", color=colors[i], marker='o', ls=ls)
                    axVth.plot(roa, vth, label=f"{label} s{ii+1}", color=colors[i], marker='o', ls=ls)
                    axVph.plot(roa, vph, label=f"{label} s{ii+1}", color=colors[i], marker='o', ls=ls)

            for ax, ttl, yl in [
                (axJ,   "$j_{\\parallel}$ (ambipolarity check)", "GB"),
                (axU0,  "$\\langle u_{\\parallel}B\\rangle_0$ (0th order)", "GB"),
                (axVt0, "$v_{\\theta,0}$ at $\\theta=0$ (0th order)", "GB"),
                (axUsp, "$\\langle u_{\\parallel}B\\rangle$ per species", "GB"),
                (axVth, "$v_{\\theta}(\\theta=0)$ per species", "GB"),
                (axVph, "$v_{\\phi}(\\theta=0)$ per species", "GB"),
            ]:
                ax.set_xlabel("$r/a$"); ax.set_xlim(_xlim)
                ax.set_title(ttl, fontsize=8)
                ax.set_ylabel(f"({yl})")
                ax.legend(loc="best", fontsize=6)

        # ---- Tab 7: Geometry diagnostics ----
        if hasattr(o0, 'f_trap'):
            fig7 = self.fn.add_figure(label=f"{extratitle}NEO geometry", tab_color=fn_color)
            grid7 = plt.GridSpec(2, 2, hspace=0.5, wspace=0.35)
            axFtrap = fig7.add_subplot(grid7[0, 0])
            axBmag  = fig7.add_subplot(grid7[0, 1])
            axIpsi  = fig7.add_subplot(grid7[1, 0])
            axEq    = fig7.add_subplot(grid7[1, 1])

            for i, label in enumerate(labels):
                roa    = [self.results[label]['output'][irho].roa for irho in range(len(self.rhos))]
                ftrap  = [getattr(self.results[label]['output'][irho], 'f_trap', np.nan) for irho in range(len(self.rhos))]
                ipsi   = [getattr(self.results[label]['output'][irho], 'I_over_psip', np.nan) for irho in range(len(self.rhos))]
                q_vals = [getattr(self.results[label]['output'][irho], 'q', np.nan) for irho in range(len(self.rhos))]

                axFtrap.plot(roa, ftrap, label=label, color=colors[i], marker='o', ls='-')
                axIpsi.plot( roa, ipsi,  label=label, color=colors[i], marker='o', ls='-')
                axEq.plot(   roa, q_vals,label=label, color=colors[i], marker='o', ls='-')

                # Bmag vs theta for each rho
                for irho in range(len(self.rhos)):
                    o = self.results[label]['output'][irho]
                    if hasattr(o, 'Bmag') and hasattr(o, 'theta_grid'):
                        axBmag.plot(o.theta_grid / np.pi, o.Bmag,
                                    label=f"{label} r/a={o.roa:.3f}", color=colors[i],
                                    ls=['-','--',':','-.'][irho % 4])

            axFtrap.set_xlabel("$r/a$"); axFtrap.set_xlim(_xlim); axFtrap.set_ylabel("$f_{trap}$")
            axIpsi.set_xlabel("$r/a$");  axIpsi.set_xlim(_xlim);  axIpsi.set_ylabel("$I/\\psi'$")
            axEq.set_xlabel("$r/a$");    axEq.set_xlim(_xlim);    axEq.set_ylabel("$q$")
            axBmag.set_xlabel("$\\theta/\\pi$"); axBmag.set_ylabel("$|B|$ (norm.)")
            axBmag.set_title("|B| profile vs. poloidal angle", fontsize=9)
            for ax in [axFtrap, axIpsi, axEq, axBmag]:
                ax.legend(loc="best", fontsize=7)

        # ---- Tab 8: Normalization profiles ----
        norm = self.NormalizationSets.get("SELECTED") if hasattr(self, 'NormalizationSets') else None
        if norm is not None and "roa" in norm:
            roa_prof = norm["roa"]
            mask = (roa_prof >= _xlim[0]) & (roa_prof <= _xlim[1])
            roa_m = roa_prof[mask]

            norm_quantities = [
                ("Te_keV",  "$T_e$",           "keV"),
                ("Ti_keV",  "$T_i$",           "keV"),
                ("ne_20",   "$n_e$",            "$10^{20}\\,m^{-3}$"),
                ("ni_20",   "$n_i$",            "$10^{20}\\,m^{-3}$"),
                ("c_s",     "$c_s$",            "m/s"),
                ("rmin",    "$a$",              "m"),
                ("q_gb",    "$Q_{GB}$",         "MW/m²"),
                ("g_gb",    "$\\Gamma_{GB}$",   "$10^{20}\\,s^{-1}m^{-2}$"),
                ("pi_gb",   "$\\Pi_{GB}$",      "N/m²"),
                ("s_gb",    "$S_{GB}$",         "W/m³"),
                ("B_unit",  "$B_{unit}$",       "T"),
                ("rho_s",   "$\\rho_s$",        "m"),
            ]
            # filter to only those keys present in norm
            norm_quantities = [(k, lbl, u) for k, lbl, u in norm_quantities if k in norm]

            n_qty = len(norm_quantities)
            ncols_n = 4
            nrows_n = int(np.ceil(n_qty / ncols_n))

            fig8 = self.fn.add_figure(label=f"{extratitle}NEO normalizations", tab_color=fn_color)
            grid8 = plt.GridSpec(nrows_n, ncols_n, hspace=0.55, wspace=0.4)

            for idx, (key, lbl, unit) in enumerate(norm_quantities):
                r = idx // ncols_n
                c = idx % ncols_n
                ax = fig8.add_subplot(grid8[r, c])
                vals = np.asarray(norm[key])
                if vals.ndim == 0 or vals.shape == ():
                    ax.axhline(float(vals), color='k', lw=1.5)
                elif vals.shape == roa_prof.shape:
                    ax.plot(roa_m, vals[mask], color='k', lw=1.5)
                else:
                    # per-species array — plot each species
                    for sp_idx in range(vals.shape[-1] if vals.ndim > 1 else 1):
                        v = vals[:, sp_idx] if vals.ndim > 1 else vals
                        ax.plot(roa_m, v[mask], lw=1.5, label=f"s{sp_idx+1}")
                    ax.legend(loc="best", fontsize=6)
                ax.set_xlabel("$r/a$"); ax.set_xlim(_xlim)
                ax.set_title(f"{lbl}\n({unit})", fontsize=8)


    def read_scan(
        self,
        label="scan1",
        subfolder=None,
        variable="RLTS_1",
        ion_OI_position_in_total_padded_list=2
    ):
        
        ion_OI_position_in_ion_list = ion_OI_position_in_total_padded_list - 2

        output_object = "output"

        variable_mapping = {
            'scanned_variable': ["parsed", variable, None],
            'Qe_gb': [output_object, 'Qe', None],
            'Qi_gb': [output_object, 'Qi', None],
            'Ge_gb': [output_object, 'Ge', None],
            'Gi_gb': [output_object, 'GiAll', ion_OI_position_in_ion_list],
            'Mt_gb': [output_object, 'Mt', None],
        }
        
        variable_mapping_unn = {
            'Qe': [output_object, 'Qe_unn', None],
            'Qi': [output_object, 'Qi_unn', None],
            'Ge': [output_object, 'Ge_unn', None],
            'Gi': [output_object, 'GiAll_unn', ion_OI_position_in_ion_list],
            'Mt': [output_object, 'Mt_unn', None],
        }
        
        super().read_scan(
            label=label,
            subfolder=subfolder,
            variable=variable,
            ion_OI_position_in_total_padded_list=ion_OI_position_in_total_padded_list,
            variable_mapping=variable_mapping,
            variable_mapping_unn=variable_mapping_unn
        )

    def plot_scan(
        self,
        fn=None,
        labels=["neo1"],
        extratitle="",
        fn_color=None,
        colors=None,
        ):
        
        if fn is None:
            self.fn = GUItools.FigureNotebook("NEO Scan Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn
            
        fig1 = self.fn.add_figure(label=f"{extratitle}Scan Summary", tab_color=fn_color)
        
        grid = plt.GridSpec(1, 3, hspace=0.7, wspace=0.2)

        if colors is None:
            colors = GRAPHICStools.listColors()

        axQe = fig1.add_subplot(grid[0, 0])
        axQi = fig1.add_subplot(grid[0, 1])
        axGe = fig1.add_subplot(grid[0, 2])

        cont = 0
        for label in labels:
            for irho in range(len(self.rhos)):
                
                x = self.scans[label]['scanned_variable'][irho]
                
                axQe.plot(x, self.scans[label]['Qe'][irho], label=f'{label}, {self.rhos[irho]}', color=colors[cont], marker='o', linestyle='-')
                axQi.plot(x, self.scans[label]['Qi'][irho], label=f'{label}, {self.rhos[irho]}', color=colors[cont], marker='o', linestyle='-')
                axGe.plot(x, self.scans[label]['Ge'][irho], label=f'{label}, {self.rhos[irho]}', color=colors[cont], marker='o', linestyle='-')

                cont += 1

        for ax in [axQe, axQi, axGe]:
            ax.set_xlabel("Scanned variable")
            GRAPHICStools.addDenseAxis(ax)
            ax.legend(loc="best")

        axQe.set_ylabel("$Q_e$ ($MW/m^2$)"); 
        axQi.set_ylabel("$Q_i$ ($MW/m^2$)"); 
        axGe.set_ylabel("$\\Gamma_e$ ($1E20/s/m^2$)")
        
        plt.tight_layout()

    def run_vgen(self, subfolder="vgen1", vgenOptions={}, cold_start=False, rho_range=None,
                 numcores=None, minutes=60, smooth_profiles=False, relative_smoothing=0.005,
                 in_process=False):
        """
        Submit profiles_gen -vgen to compute the neoclassical radial electric field (Er)
        and populate w0(rad/s) in the profiles.  Must be followed by read_vgen().

        Uses self.FolderGACODE (set by prep()) as the parent directory and
        self.profiles (set by prep()) as the input gacode state.

        Options for vgenOptions:
            er          : Method to compute Er
                            1 = Force balance from given omega0
                            2 = NEO weak rotation limit (recommended for zero toroidal rotation)
                            3 = NEO strong rotation limit
                            4 = Return given omega0
            vel         : Method to compute velocities
                            1 = NEO weak rotation limit
                            2 = NEO strong rotation limit
            nth         : Min,max theta resolutions (e.g. "17,39")
            matched_ion : Index of ion species to match NEO and given velocities (1-indexed)

        smooth_profiles : bool
            If True, smooth Te, Ti, ne, ni with a cubic spline before writing the
            VGEN input so that piecewise-linear kinks in the gradients do not
            pollute the computed Er.  The original self.profiles is never modified.
        relative_smoothing : float
            Passed to gacode_state.smooth_profiles(); target RMS deviation relative
            to the peak profile value (default 0.02 = 2 %).
        """

        import copy as _copy

        self.folder_vgen = self.FolderGACODE / f"{subfolder}"

        # ---- Default options (mutable default arg: always copy before mutating)
        vgenOptions = dict(vgenOptions)
        vgenOptions.setdefault("er", 2)
        vgenOptions.setdefault("vel", 1)
        vgenOptions.setdefault("numspecies", len(self.profiles.Species))
        vgenOptions.setdefault("matched_ion", 1)
        vgenOptions.setdefault("nth", "17,39")
        vgenOptions.setdefault("rho_range", rho_range)

        # ---- Decide whether to (re)run
        runThisCase = not check_if_files_exist(
            self.folder_vgen,
            [
                ["vgen", "input.gacode"],
                ["vgen", "input.neo.gen"],
                ["out.vgen.neoequil00"],
                ["out.vgen.neoexpnorm00"],
                ["out.vgen.neontheta00"],
                ["vgen.dat"],
            ],
        )

        if (not runThisCase) and cold_start:
            runThisCase = print("\t- Files found in folder, but cold_start requested. Are you sure?",typeMsg="q",)
            if runThisCase:
                IOtools.askNewFolder(self.folder_vgen, force=True)

        self.folder_vgen.mkdir(parents=True, exist_ok=True)

        # ---- Build the profiles object to write (never mutate self.profiles)
        profiles_to_write = _copy.deepcopy(self.profiles)

        if smooth_profiles:
            print("\t- Smoothing kinetic profiles before VGEN run (smooth_profiles=True)", typeMsg="i")
            # Save the original full-grid profiles so read_vgen() can show raw vs. smoothed
            profiles_to_write.write_state(file=(self.folder_vgen / "input.gacode.raw"))
            # Smooth on the full grid (before any rho_range truncation)
            profiles_to_write.smooth_profiles(relative_smoothing=relative_smoothing)

        # Truncate to rho_range AFTER smoothing so the spline sees the full profile
        if rho_range is not None:
            rho_arr = profiles_to_write.profiles["rho(-)"]
            i0 = np.argmin(np.abs(rho_arr - rho_range[0]))
            i1 = np.argmin(np.abs(rho_arr - rho_range[1]))
            profiles_to_write.changeResolution(rho_new=rho_arr[i0:i1+1])

        profiles_to_write.write_state(file=(self.folder_vgen / "input.gacode"))

        # ---- Resolve numcores from machine config (same pattern as SIMtools._run())
        if numcores is None:
            machineSettings  = FARMINGtools.mitim_job.grab_machine_settings("profiles_gen")
            numcores = machineSettings["cores_per_node"]

            if machineSettings["machine"] == "local":
                slurm_ntasks    = os.environ.get("SLURM_NTASKS")
                slurm_cpus      = os.environ.get("SLURM_CPUS_PER_TASK")
                if slurm_ntasks is not None and slurm_cpus is not None:
                    cores_allocated = int(slurm_ntasks) * int(slurm_cpus)
                elif slurm_ntasks is not None:
                    cores_allocated = int(slurm_ntasks)
                elif slurm_cpus is not None:
                    cores_allocated = int(slurm_cpus)
                else:
                    cores_allocated = os.cpu_count()

                if cores_allocated is not None:
                    if numcores is None or cores_allocated < numcores:
                        numcores = cores_allocated

            if numcores is None:
                numcores = 16

        # ---- Run ---------------------------------------------------------
        if runThisCase:
            n_surfaces = len(self.profiles.profiles["rho(-)"])
            if in_process:
                # In-process path: ctypes call into libvgen_serial.so, no
                # mitim_job, no SLURM submission, no tarballing.  Each NEO
                # solve still runs sequentially over surfaces inside the
                # Fortran library — same physics as profiles_gen -vgen.
                from mitim_tools.simulation_tools.interfaces.vgen_inprocess import VGENInProcess

                print(f'\t- [in-process] Running VGEN on {n_surfaces} surfaces (no SLURM, no subprocess fork)', typeMsg="i")

                # vgenOptions["numspecies"] is the value MITIM passes as
                # `-in N` to profiles_gen -vgen.  The actual NEO N_SPECIES
                # is N+1 (the wrapper script appends N_SPECIES=$((N+1))),
                # so mirror that here.
                n_species = int(vgenOptions.get("numspecies", len(self.profiles.Species))) + 1

                _vgen_runner = VGENInProcess()
                _vgen_runner.run(
                    folder         = self.folder_vgen,
                    er_method      = int(vgenOptions.get("er", 2)),
                    vel_method     = int(vgenOptions.get("vel", 1)),
                    erspecies_indx = int(vgenOptions.get("matched_ion", 1)),
                    nth_min        = int(str(vgenOptions.get("nth", "17,39")).split(",")[0]),
                    nth_max        = int(str(vgenOptions.get("nth", "17,39")).split(",")[-1]),
                    n_species      = n_species,
                )
            else:
                print(f'\t- Running VGEN to compute Er and populate w0 in profiles, using {numcores} cores for {minutes} minutes, on {n_surfaces} surfaces', typeMsg="i")
                GACODErun.runVGEN(self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder, numcores=numcores, minutes=minutes)
        else:
            print(f"\t- Required files found in {subfolder}, not running VGEN", typeMsg="i")

    def read_vgen(self, subfolder=None):
        """
        Read outputs produced by run_vgen():
            vgen/input.gacode  → self.profiles_vgen  (gacode_state with updated w0)
            out.vgen.ercomp    → self.vgen_ercomp     (dict of Er component profiles vs rho)
            out.vgen.vel       → self.vgen_vel        (dict of velocity component profiles vs rho)

        Call after run_vgen() (or on an already-completed folder).
        If subfolder is None, uses self.folder_vgen set by run_vgen().
        """
        if subfolder is not None:
            self.folder_vgen = self.FolderGACODE / subfolder

        folder = self.folder_vgen / "vgen"
        file_gacode = folder / "input.gacode"

        # ---- Updated gacode profiles (w0 now populated from NEO Er) ----
        from mitim_tools.gacode_tools import PROFILEStools
        self.profiles_vgen = PROFILEStools.gacode_state(file_gacode, derive_quantities=True)

        # ---- Input profiles: raw vs. smoothed ----
        # When smooth_profiles=True was used, run_vgen() saved the original (pre-smoothing)
        # profiles as input.gacode.raw and the smoothed version as input.gacode (the one
        # actually fed to VGEN).  When smoothing was not used, only input.gacode exists.
        file_raw      = self.folder_vgen / "input.gacode.raw"
        file_smoothed = self.folder_vgen / "input.gacode"   # always the file passed to VGEN

        if file_raw.exists():
            # Smoothing was used: raw = original, smoothed = what VGEN received
            self.profiles          = PROFILEStools.gacode_state(file_raw,      derive_quantities=True)
            self.profiles_smoothed = PROFILEStools.gacode_state(file_smoothed, derive_quantities=True)
        else:
            # No smoothing: only the raw input exists
            if not hasattr(self, "profiles") or self.profiles is None:
                self.profiles = PROFILEStools.gacode_state(file_smoothed, derive_quantities=True)
            self.profiles_smoothed = None

        # ---- Rotation-component decomposition (out.vgen.ercomp) ----
        # vgen.f90 writes 2 + 3*n_ions columns per radius — rho, w0, then for
        # each ion j the three force-balance contributions to that ion's
        # implied toroidal angular frequency (all in rad/s):
        #   omega_gradp_{j} : pressure-gradient (diamagnetic) term
        #   omega_vtor_{j}  : toroidal-velocity term, vtor/(rmaj+rmin)
        #   omega_vpol_{j}  : poloidal-velocity term, -vpol*bt0/bp0/(rmaj+rmin)
        ercomp_file = folder / "out.vgen.ercomp"
        if ercomp_file.exists():
            data = np.atleast_2d(np.loadtxt(ercomp_file))
            n_ions = (data.shape[1] - 2) // 3
            self.vgen_ercomp = {"rho": data[:, 0], "w0": data[:, 1]}
            for j in range(n_ions):
                self.vgen_ercomp[f"omega_gradp_{j+1}"] = data[:, 2 + 3 * j]
                self.vgen_ercomp[f"omega_vtor_{j+1}"]  = data[:, 3 + 3 * j]
                self.vgen_ercomp[f"omega_vpol_{j+1}"]  = data[:, 4 + 3 * j]
        else:
            self.vgen_ercomp = {}

        # ---- Velocities (out.vgen.vel) ----
        # vgen.f90 writes 4 + 4*n_ions columns per radius — rho, er_exp (the
        # experimental/derived Er input, V/m), w0 (rad/s), w0p, then for each
        # ion j:
        #   vpol_{j}          : poloidal velocity (m/s)
        #   vtor_{j}          : toroidal velocity (m/s)
        #   vpol_over_bp0_{j} : vpol/Bp0
        #   omega_{j}         : (vtor - vpol*bt0/bp0)/(rmaj+rmin)  (rad/s)
        vel_file = folder / "out.vgen.vel"
        if vel_file.exists():
            data = np.atleast_2d(np.loadtxt(vel_file))
            n_ions = (data.shape[1] - 4) // 4
            self.vgen_vel = {"rho": data[:, 0], "er_exp": data[:, 1],
                             "w0": data[:, 2], "w0p": data[:, 3]}
            for j in range(n_ions):
                self.vgen_vel[f"vpol_{j+1}"]          = data[:, 4 + 4 * j]
                self.vgen_vel[f"vtor_{j+1}"]          = data[:, 5 + 4 * j]
                self.vgen_vel[f"vpol_over_bp0_{j+1}"] = data[:, 6 + 4 * j]
                self.vgen_vel[f"omega_{j+1}"]         = data[:, 7 + 4 * j]
        else:
            self.vgen_vel = {}

        print(
            f"\t- VGEN read: w0 range [{self.profiles_vgen.profiles['w0(rad/s)'].min():.3e}, "
            f"{self.profiles_vgen.profiles['w0(rad/s)'].max():.3e}] rad/s",
            typeMsg="i",
        )

    def plot_vgen(self, fn=None, fn_color=None, label="vgen", rho_min=0.1):
        """
        Plot VGEN results: Er component decomposition, w0 and VEXB_SHEAR before/after,
        and (when smoothing was used) a raw-vs-smoothed comparison per profile.
        Requires read_vgen() to have been called first.

        rho_min : float
            Minimum rho to include in plots (default 0.1).
            Near-axis values can diverge and obscure the physically relevant region.
        """
        apply_theme()

        if fn is None:
            self.fn = GUItools.FigureNotebook("NEO VGEN Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn

        colors = GRAPHICStools.listColors()

        raw      = self.profiles if (hasattr(self, "profiles") and self.profiles is not None) else None
        smoothed = getattr(self, "profiles_smoothed", None)
        # The profiles actually fed to VGEN: smoothed if available, else raw
        src = smoothed if smoothed is not None else raw

        # ------------------------------------------------------------------
        # Tab 1: kinetic profiles (left, smoothed = actually used) + Er components (right)
        # ------------------------------------------------------------------
        fig1 = self.fn.add_figure(label=f"{label} Er components", tab_color=fn_color)
        grid1 = plt.GridSpec(2, 5, hspace=0.55, wspace=0.45, width_ratios=[1, 1, 1, 1, 1])

        ax_prof = fig1.add_subplot(grid1[0, 0])
        ax_grad = fig1.add_subplot(grid1[1, 0])
        ax_Er   = fig1.add_subplot(grid1[0, 1])
        ax_db   = fig1.add_subplot(grid1[0, 2])
        ax_vpol = fig1.add_subplot(grid1[0, 3])
        ax_all  = fig1.add_subplot(grid1[0, 4])
        ax_vtor = fig1.add_subplot(grid1[1, 1])
        ax_dia  = fig1.add_subplot(grid1[1, 2])

        # Kinetic profiles: show the ones actually used by VGEN (smoothed if available, else raw)
        if src is not None:
            rho_s = src.profiles.get("rho(-)", None)
            if rho_s is not None:
                mp = rho_s >= rho_min

                Te_s   = src.profiles.get("te(keV)", None)
                Ti_all = src.profiles.get("ti(keV)", None)
                Ti_s   = Ti_all[:, 0] if Ti_all is not None and Ti_all.ndim == 2 else Ti_all
                ne_s   = src.profiles.get("ne(10^19/m^3)", None)
                aLTe_s = src.derived.get("aLTe", None)
                aLTi_a = src.derived.get("aLTi", None)
                aLTi_s = aLTi_a[:, 0] if aLTi_a is not None and np.ndim(aLTi_a) == 2 else aLTi_a
                aLne_s = src.derived.get("aLne", None)

                ax_ne = ax_prof.twinx()
                if Te_s   is not None: ax_prof.plot(rho_s[mp], Te_s[mp],   color=colors[0], lw=1.8, label="$T_e$")
                if Ti_s   is not None: ax_prof.plot(rho_s[mp], Ti_s[mp],   color=colors[1], lw=1.8, label="$T_i$")
                if ne_s   is not None: ax_ne.plot(  rho_s[mp], ne_s[mp],   color=colors[2], lw=1.8, label="$n_e$")
                if aLTe_s is not None: ax_grad.plot(rho_s[mp], aLTe_s[mp], color=colors[0], lw=1.8, label="$a/L_{T_e}$")
                if aLTi_s is not None: ax_grad.plot(rho_s[mp], aLTi_s[mp], color=colors[1], lw=1.8, label="$a/L_{T_i}$")
                if aLne_s is not None: ax_grad.plot(rho_s[mp], aLne_s[mp], color=colors[2], lw=1.8, label="$a/L_{n_e}$")

                ax_prof.set_xlabel(r"$\rho_{tor}$"); ax_prof.set_xlim(left=rho_min)
                ax_prof.set_ylabel("Temperature (keV)")
                ax_ne.set_ylabel("$n_e$ ($10^{19}\\,m^{-3}$)", color=colors[2])
                prof_title = "Profiles (smoothed)" if smoothed is not None else "Profiles"
                ax_prof.set_title(prof_title)
                lines_T, labs_T = ax_prof.get_legend_handles_labels()
                lines_n, labs_n = ax_ne.get_legend_handles_labels()
                ax_prof.legend(lines_T + lines_n, labs_T + labs_n, loc="best", fontsize=7)

                ax_grad.set_xlabel(r"$\rho_{tor}$"); ax_grad.set_xlim(left=rho_min)
                ax_grad.set_ylabel("$a/L$")
                ax_grad.set_title("Norm. gradients" + (" (smoothed)" if smoothed is not None else ""))
                ax_grad.axhline(0, color="k", lw=0.7, ls="--")
                ax_grad.legend(loc="best", fontsize=7)

        if self.vgen_ercomp:
            rho  = self.vgen_ercomp["rho"]
            mask = rho >= rho_min
            ions = sorted(int(k.rsplit("_", 1)[1]) for k in self.vgen_ercomp if k.startswith("omega_gradp_"))

            ax_Er.plot(rho[mask], self.vgen_ercomp["w0"][mask], color=colors[0], lw=1.8, label="$\\omega_0$ (VGEN)")
            for c, j in enumerate(ions):
                total_j = (self.vgen_ercomp[f"omega_gradp_{j}"]
                           + self.vgen_ercomp[f"omega_vtor_{j}"]
                           + self.vgen_ercomp[f"omega_vpol_{j}"])
                ax_Er.plot(rho[mask], total_j[mask], color=colors[c + 1], lw=1.2, ls="--", label=f"sum ion {j}")
                ax_db.plot(rho[mask],   self.vgen_ercomp[f"omega_gradp_{j}"][mask], color=colors[c], lw=1.5, label=f"ion {j}")
                ax_vpol.plot(rho[mask], self.vgen_ercomp[f"omega_vpol_{j}"][mask],  color=colors[c], lw=1.5, label=f"ion {j}")
                ax_vtor.plot(rho[mask], self.vgen_ercomp[f"omega_vtor_{j}"][mask],  color=colors[c], lw=1.5, label=f"ion {j}")

            j0 = ions[0]
            for key, lbl, c in [
                (f"omega_gradp_{j0}", "$\\nabla p$ term",  colors[2]),
                (f"omega_vtor_{j0}",  "$v_{tor}$ term",    colors[5]),
                (f"omega_vpol_{j0}",  "$v_{pol}$ term",    colors[3]),
            ]:
                ax_all.plot(rho[mask], self.vgen_ercomp[key][mask], color=c, lw=1.5, label=lbl)
            ax_all.plot(rho[mask], self.vgen_ercomp["w0"][mask], color=colors[0], lw=1.8, label="$\\omega_0$")

        if self.vgen_vel and "er_exp" in self.vgen_vel:
            rho_v = self.vgen_vel["rho"]
            mv = rho_v >= rho_min
            ax_dia.plot(rho_v[mv], self.vgen_vel["er_exp"][mv], color=colors[6], lw=1.8, label="$E_r$ (input)")

        for ax in [ax_Er, ax_db, ax_vpol, ax_vtor, ax_all]:
            ax.set_xlabel(r"$\rho_{tor}$")
            ax.set_ylabel("$\\omega$ (rad/s)")
            ax.set_xlim(left=rho_min)
            ax.axhline(0, color="k", lw=0.7, ls="--")
            ax.legend(loc="best", fontsize=7)
        ax_dia.set_xlabel(r"$\rho_{tor}$")
        ax_dia.set_ylabel("$E_r$ (V/m)")
        ax_dia.set_xlim(left=rho_min)
        ax_dia.axhline(0, color="k", lw=0.7, ls="--")
        ax_dia.legend(loc="best", fontsize=7)
        ax_Er.set_title("$\\omega_0$ and per-ion force-balance sum")
        ax_db.set_title("$\\nabla p$ (diamagnetic) term")
        ax_vpol.set_title("Poloidal-flow term")
        ax_vtor.set_title("Toroidal-flow term (≈0 weak rot.)")
        ax_dia.set_title("Experimental $E_r$ (input)")
        ax_all.set_title("Components (first ion)")

        # ------------------------------------------------------------------
        # Tab 2: w0 and VEXB_SHEAR before/after VGEN
        # ------------------------------------------------------------------
        fig2 = self.fn.add_figure(label=f"{label} w0 & VEXB", tab_color=fn_color)
        grid2 = plt.GridSpec(1, 3, hspace=0.4, wspace=0.35)
        ax_w0   = fig2.add_subplot(grid2[0, 0])
        ax_vexb = fig2.add_subplot(grid2[0, 1])
        ax_mach = fig2.add_subplot(grid2[0, 2])

        if raw is not None:
            rho_orig = raw.profiles.get("rho(-)", None)
            w0_orig  = raw.profiles.get("w0(rad/s)", None)
            if rho_orig is not None and w0_orig is not None:
                m = rho_orig >= rho_min
                ax_w0.plot(rho_orig[m], w0_orig[m], color=colors[1], lw=1.5, ls="--", label="before VGEN")
            vexb_orig = _compute_vexb_shear(raw)
            if vexb_orig is not None and rho_orig is not None:
                m = rho_orig >= rho_min
                ax_vexb.plot(rho_orig[m], vexb_orig[m], color=colors[1], lw=1.5, ls="--", label="before VGEN")

        if hasattr(self, "profiles_vgen") and self.profiles_vgen is not None:
            rho_new = self.profiles_vgen.profiles.get("rho(-)", None)
            w0_new  = self.profiles_vgen.profiles.get("w0(rad/s)", None)
            if rho_new is not None and w0_new is not None:
                m = rho_new >= rho_min
                ax_w0.plot(rho_new[m], w0_new[m], color=colors[0], lw=1.8, label="after VGEN (NEO)")
            vexb_new = _compute_vexb_shear(self.profiles_vgen)
            if vexb_new is not None and rho_new is not None:
                m = rho_new >= rho_min
                ax_vexb.plot(rho_new[m], vexb_new[m], color=colors[0], lw=1.8, label="after VGEN (NEO)")

        if self.vgen_vel and "rho" in self.vgen_vel:
            rho_v = self.vgen_vel["rho"]
            m = rho_v >= rho_min
            ions_v = sorted(int(k.rsplit("_", 1)[1]) for k in self.vgen_vel
                            if k.startswith("vpol_") and not k.startswith("vpol_over_bp0_"))
            for c, j in enumerate(ions_v):
                ax_mach.plot(rho_v[m], self.vgen_vel[f"vpol_{j}"][m], color=colors[3 + c], lw=1.8, label=f"ion {j}")

        for ax in [ax_w0, ax_vexb, ax_mach]:
            ax.set_xlabel(r"$\rho_{tor}$")
            ax.set_xlim(left=rho_min)
            ax.axhline(0, color="k", lw=0.7, ls="--")
            ax.legend(loc="best", fontsize=7)
        ax_w0.set_ylabel("$\\omega_0$ (rad/s)");      ax_w0.set_title("Toroidal rotation $\\omega_0$")
        ax_vexb.set_ylabel("$\\gamma_{E}$ (norm.)");  ax_vexb.set_title("E×B shearing rate (VEXB_SHEAR)")
        ax_mach.set_ylabel("$v_{pol}$ (m/s)");        ax_mach.set_title("NEO poloidal velocities")

        # ------------------------------------------------------------------
        # Tab 3: Raw vs. smoothed profiles comparison (only when smoothing was used)
        # ------------------------------------------------------------------
        if raw is not None and smoothed is not None:
            fig3 = self.fn.add_figure(label=f"{label} smoothing", tab_color=fn_color)
            grid3 = plt.GridSpec(2, 3, hspace=0.5, wspace=0.4)

            ax_Te  = fig3.add_subplot(grid3[0, 0])
            ax_Ti  = fig3.add_subplot(grid3[0, 1])
            ax_ne  = fig3.add_subplot(grid3[0, 2])
            ax_aLTe = fig3.add_subplot(grid3[1, 0])
            ax_aLTi = fig3.add_subplot(grid3[1, 1])
            ax_aLne = fig3.add_subplot(grid3[1, 2])

            def _get(p, key):
                v = p.profiles.get(key, None)
                if v is None:
                    v = p.derived.get(key, None)
                return v

            def _col0(arr):
                if arr is None:
                    return None
                return arr[:, 0] if np.ndim(arr) == 2 else arr

            for p, lw, ls, lbl in [(raw, 1.2, "--", "raw"), (smoothed, 1.8, "-", "smoothed")]:
                rho_p = _get(p, "rho(-)")
                if rho_p is None:
                    continue
                mp = rho_p >= rho_min

                Te_p   = _col0(_get(p, "te(keV)"))
                Ti_p   = _col0(_get(p, "ti(keV)"))
                ne_p   = _col0(_get(p, "ne(10^19/m^3)"))
                aLTe_p = _col0(_get(p, "aLTe"))
                aLTi_p = _col0(_get(p, "aLTi"))
                aLne_p = _col0(_get(p, "aLne"))

                c_map = {"raw": colors[1], "smoothed": colors[0]}
                c = c_map[lbl]

                if Te_p   is not None: ax_Te.plot(rho_p[mp],   Te_p[mp],   color=c, lw=lw, ls=ls, label=lbl)
                if Ti_p   is not None: ax_Ti.plot(rho_p[mp],   Ti_p[mp],   color=c, lw=lw, ls=ls, label=lbl)
                if ne_p   is not None: ax_ne.plot(rho_p[mp],   ne_p[mp],   color=c, lw=lw, ls=ls, label=lbl)
                if aLTe_p is not None: ax_aLTe.plot(rho_p[mp], aLTe_p[mp], color=c, lw=lw, ls=ls, label=lbl)
                if aLTi_p is not None: ax_aLTi.plot(rho_p[mp], aLTi_p[mp], color=c, lw=lw, ls=ls, label=lbl)
                if aLne_p is not None: ax_aLne.plot(rho_p[mp], aLne_p[mp], color=c, lw=lw, ls=ls, label=lbl)

            ax_Te.set_title("$T_e$");           ax_Te.set_ylabel("keV")
            ax_Ti.set_title("$T_i$");           ax_Ti.set_ylabel("keV")
            ax_ne.set_title("$n_e$");           ax_ne.set_ylabel("$10^{19}\\,m^{-3}$")
            ax_aLTe.set_title("$a/L_{T_e}$");  ax_aLTe.set_ylabel("$a/L$")
            ax_aLTi.set_title("$a/L_{T_i}$");  ax_aLTi.set_ylabel("$a/L$")
            ax_aLne.set_title("$a/L_{n_e}$");  ax_aLne.set_ylabel("$a/L$")

            for ax in [ax_Te, ax_Ti, ax_ne, ax_aLTe, ax_aLTi, ax_aLne]:
                ax.set_xlabel(r"$\rho_{tor}$")
                ax.set_xlim(left=rho_min)
                ax.legend(loc="best", fontsize=7)
            for ax in [ax_aLTe, ax_aLTi, ax_aLne]:
                ax.axhline(0, color="k", lw=0.7, ls="--")



def _compute_vexb_shear(profiles_obj):
    """
    Compute the normalised E×B shearing rate (VEXB_SHEAR in TGLF notation) from a
    gacode_state object using the same formula as MITIMstate.to_tglf():

        gamma_eb0   = -(dw0/dr) * r / |q|
        vexb_shear  = -sign_It * gamma_eb0 * a / c_s

    Returns a numpy array on the profiles grid, or None if any required
    quantity is missing.
    """
    try:
        from mitim_tools.misc_tools import MATHtools
        w0  = profiles_obj.profiles["w0(rad/s)"]
        r   = profiles_obj.derived["r"]
        q   = profiles_obj.profiles["q(-)"]
        a   = profiles_obj.derived["a"]
        c_s = profiles_obj.derived["c_s"]
        sign_it = -np.sign(profiles_obj.profiles["current(MA)"][-1])
        dw0_dr    = MATHtools.deriv(r, w0, array=True)
        gamma_eb0 = -dw0_dr * r / np.abs(q)
        return -sign_it * gamma_eb0 * a / c_s
    except Exception:
        return None


def check_if_files_exist(folder, list_files):
    folder = IOtools.expandPath(folder)

    for file_parts in list_files:
        checkfile = folder
        for ii in range(len(file_parts)):
            checkfile = checkfile / f"{file_parts[ii]}"
        if not checkfile.exists():
            return False

    return True

class NEOinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.neo.controls",
            code='NEO',
            n_species='N_SPECIES'
            )
                
class NEOoutput(SIMtools.GACODEoutput):
    def __init__(self, FolderGACODE, suffix="", **kwargs):
        super().__init__()

        self.FolderGACODE, self.suffix = FolderGACODE, suffix

        if suffix == "":
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} without suffix")
        else:
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} with suffix {suffix}")

        self.inputclass = NEOinput(file=self.FolderGACODE / f"input.neo{self.suffix}")

        self.read()

    def read(self):

        # ---- Grid (needed to determine n_species, n_theta for other parsers) ----
        self._read_grid()

        # ---- Main fluxes (required — raises on failure) ----
        self._read_transport_flux()

        # ---- Optional output files ----
        self._read_equil()
        self._read_theory()
        self._read_transport()
        self._read_transport_gv()
        self._read_rotation()
        self._read_diagnostic_geo()
        self._read_prec()

        # ---- Input file text ----
        with open(self.FolderGACODE / ("input.neo" + self.suffix), "r") as fi:
            self.inputFile = fi.read()

    # ------------------------------------------------------------------
    # Private readers
    # ------------------------------------------------------------------

    def _read_grid(self):
        """Read out.neo.grid → n_species, n_energy, n_xi, n_theta, theta, n_radial."""
        filepath = self.FolderGACODE / ("out.neo.grid" + self.suffix)
        try:
            data = np.loadtxt(filepath)
            self.n_species = int(data[0])
            self.n_energy  = int(data[1])
            self.n_xi      = int(data[2])
            self.n_theta   = int(data[3])
            self.theta_grid = data[4 : 4 + self.n_theta]
            self.n_radial  = int(data[4 + self.n_theta])
        except Exception:
            # Fall back to defaults; transport_flux parser will set n_species
            self.n_species = None
            self.n_energy  = None
            self.n_xi      = None
            self.n_theta   = None
            self.theta_grid = None
            self.n_radial  = 1

    def _read_transport_flux(self):
        """Read out.neo.transport_flux — DKE, GV and tgyro sections for all species."""
        filepath = self.FolderGACODE / ("out.neo.transport_flux" + self.suffix)
        with open(filepath, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            raise ValueError(
                f"NEO output file {filepath} is empty! NEO run may have failed."
            )

        self.roa = float(lines[0].split()[-1])

        # Parse the three sections: dke, gv, tgyro
        section_keys = {
            'pflux_dke':   'dke',
            'pflux_gv':    'gv',
            'pflux_tgyro': 'tgyro',
        }
        sections = {}
        current = None
        buf = []
        for line in lines:
            s = line.strip()
            if s.startswith('#'):
                for kw, name in section_keys.items():
                    if kw in s:
                        if current is not None:
                            sections[current] = buf
                        current = name
                        buf = []
                        break
            elif current is not None and s and not s.startswith('('):
                vals = s.split()
                if len(vals) == 4:
                    buf.append([float(v) for v in vals])
        if current is not None:
            sections[current] = buf

        for sec, rows in sections.items():
            arr = np.array(rows)           # (n_species, 4): Z, pflux, eflux, mflux
            Z = arr[:, 0]
            G = arr[:, 1]
            Q = arr[:, 2]
            M = arr[:, 3]
            ie = int(np.where(Z == -1)[0][0])

            if sec == 'tgyro':
                self.Ge     = G[ie];  self.Qe = Q[ie];  self.Me = M[ie]
                self.GiAll  = np.delete(G, ie)
                self.QiAll  = np.delete(Q, ie)
                self.MiAll  = np.delete(M, ie)
                self.Zi     = np.delete(Z, ie)
                self.Qi     = self.QiAll.sum()
                self.Mt     = self.Me + self.MiAll.sum()
                if self.n_species is None:
                    self.n_species = len(Z)
            elif sec == 'dke':
                self.Ge_dke    = G[ie];  self.Qe_dke = Q[ie];  self.Me_dke = M[ie]
                self.GiAll_dke = np.delete(G, ie)
                self.QiAll_dke = np.delete(Q, ie)
                self.MiAll_dke = np.delete(M, ie)
                self.Qi_dke    = self.QiAll_dke.sum()
            elif sec == 'gv':
                self.Ge_gv    = G[ie];  self.Qe_gv = Q[ie];  self.Me_gv = M[ie]
                self.GiAll_gv = np.delete(G, ie)
                self.QiAll_gv = np.delete(Q, ie)
                self.MiAll_gv = np.delete(M, ie)
                self.Qi_gv    = self.QiAll_gv.sum()

    def _read_equil(self):
        """Read out.neo.equil → q, omega0, domega0dr, n/T/a_Ln/a_Lt/nu per species."""
        filepath = self.FolderGACODE / ("out.neo.equil" + self.suffix)
        try:
            eq = np.atleast_2d(np.loadtxt(filepath))[0]   # single-radius row
            self.dphidr     = float(eq[1])
            self.q          = float(eq[2])
            self.rho_star   = float(eq[3])
            self.R0_over_a  = float(eq[4])
            self.omega0     = float(eq[5])
            self.domega0dr  = float(eq[6])
            # per-species arrays (all species, including electrons at index -1)
            self.n_norm     = eq[7 + 0 :: 5]
            self.T_norm     = eq[7 + 1 :: 5]
            self.a_over_ln  = eq[7 + 2 :: 5]
            self.a_over_lt  = eq[7 + 3 :: 5]
            self.tauinv     = eq[7 + 4 :: 5]
        except Exception:
            pass

    def _read_theory(self):
        """Read out.neo.theory → Hinton-Hazeltine, Chang-Hinton, Taguchi-Gyro predictions."""
        filepath = self.FolderGACODE / ("out.neo.theory" + self.suffix)
        try:
            d = np.atleast_2d(np.loadtxt(filepath))[0]
            self.HHGamma  = float(d[1])
            self.HHQi     = float(d[2])
            self.HHQe     = float(d[3])
            self.HHjparB  = float(d[4])
            self.HHk      = float(d[5])
            self.HHuparB  = float(d[6])
            self.HHvtheta = float(d[7])
            self.CHQi     = float(d[8])
            self.TGQi     = float(d[9])
            self.SjparB   = float(d[10])
            self.Sk       = float(d[11])
            self.SuparB   = float(d[12])
            self.Svtheta  = float(d[13])
            self.HRphisq  = float(d[14])
            # per-species: HSGamma[is], HSQ[is], KjparB[is]
            self.HSGamma  = d[15 + 0 :: 3]
            self.HSQ      = d[15 + 1 :: 3]
            self.KjparB   = d[15 + 2 :: 3]
        except Exception:
            pass

    def _read_transport(self):
        """Read out.neo.transport → jparB, vtheta0, uparB0, and per-species flows/fluxes."""
        filepath = self.FolderGACODE / ("out.neo.transport" + self.suffix)
        try:
            d = np.atleast_2d(np.loadtxt(filepath))[0]
            # d[0]=r/a, d[1]=phisq, d[2]=jparB, d[3]=vtheta0, d[4]=uparB0
            # then per species (8 cols each): Gamma, Q, Pi, uparB, k, K, vtheta, vphi
            self.phisq    = float(d[1])
            self.jparB    = float(d[2])
            self.vtheta0  = float(d[3])
            self.uparB0   = float(d[4])
            self.Gamma_sp = d[5  :: 8]
            self.Q_sp     = d[6  :: 8]
            self.Pi_sp    = d[7  :: 8]
            self.uparB_sp = d[8  :: 8]
            self.k_sp     = d[9  :: 8]
            self.K_sp     = d[10 :: 8]
            self.vtheta_sp= d[11 :: 8]
            self.vphi_sp  = d[12 :: 8]
        except Exception:
            pass

    def _read_transport_gv(self):
        """Read out.neo.transport_gv → GV contribution per species (Gamma, Q, Pi)."""
        filepath = self.FolderGACODE / ("out.neo.transport_gv" + self.suffix)
        try:
            d = np.atleast_2d(np.loadtxt(filepath))[0]
            self.Gamma_gv_sp = d[1 + 0 :: 3]
            self.Q_gv_sp     = d[1 + 1 :: 3]
            self.Pi_gv_sp    = d[1 + 2 :: 3]
        except Exception:
            pass

    def _read_rotation(self):
        """Read out.neo.rotation → dphi_ave, n_ratio, V_conv, phi_theta(theta), n_ov_n0."""
        filepath = self.FolderGACODE / ("out.neo.rotation" + self.suffix)
        try:
            d = np.loadtxt(filepath)
            ns = self.n_species if self.n_species is not None else (len(d) - 1) // 2
            nt = self.n_theta   if self.n_theta   is not None else 17
            self.dphi_ave  = float(d[1])
            # n_ratio and V_conv are interleaved: n_ratio[0], V_conv[0], n_ratio[1], ...
            self.n_ratio   = d[2 : 2 + ns * 2 : 2]
            self.V_conv    = d[3 : 3 + ns * 2 : 2]
            N = ns * 2 + 2
            self.phi_theta = d[N : N + nt]       # rotation potential vs theta
            N += nt
            # density ratio n(theta)/n0 reshaped to (n_species, n_theta)
            n_ov_n0_flat = d[N:]
            if len(n_ov_n0_flat) == ns * nt:
                self.n_ov_n0 = n_ov_n0_flat.reshape(ns, nt)
        except Exception:
            pass

    def _read_diagnostic_geo(self):
        """Read out.neo.diagnostic_geo → f_trap, I/psi', Bmag(theta), and geo scalars."""
        filepath = self.FolderGACODE / ("out.neo.diagnostic_geo" + self.suffix)
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            for line in lines:
                s = line.strip()
                if "I/psi'" in s:
                    self.I_over_psip = float(s.split('=')[-1])
                elif "f_trap" in s:
                    self.f_trap = float(s.split('=')[-1])
                elif "n_theta" in s:
                    nt = int(s.split('=')[-1])

            # Collect all data values (non-comment lines)
            vals = []
            for line in lines:
                if not line.strip().startswith('#') and line.strip():
                    vals.append(float(line.strip()))
            vals = np.array(vals)

            # Layout: theta(nt), v_drift_x(nt), gradpar_Bmag(nt), Bmag(nt), w_theta(nt), R(nt), R0(1), dR0dr(1)
            if self.theta_grid is None and len(vals) >= nt:
                self.theta_grid = vals[:nt]
            if len(vals) >= 4 * nt:
                self.Bmag       = vals[3 * nt : 4 * nt]
                self.w_theta    = vals[4 * nt : 5 * nt]
                self.v_drift_x  = vals[nt     : 2 * nt]
        except Exception:
            pass

        # Also read the compact scalar file (diagnostic_geo2)
        filepath2 = self.FolderGACODE / ("out.neo.diagnostic_geo2" + self.suffix)
        try:
            geo2 = np.loadtxt(filepath2)
            # geo2[0]=I/psi', geo2[1]=f_trap, geo2[2]=<B^2>, geo2[3]=Bpol(theta=0), geo2[4]=<1/B^2><B^2>-1
            if not hasattr(self, 'I_over_psip'):
                self.I_over_psip = float(geo2[0])
            if not hasattr(self, 'f_trap'):
                self.f_trap      = float(geo2[1])
            self.Bmag2_avg   = float(geo2[2])
            self.Bpol_th0    = float(geo2[3])
        except Exception:
            pass

    def _read_prec(self):
        """Read out.neo.prec → solver convergence metric."""
        filepath = self.FolderGACODE / ("out.neo.prec" + self.suffix)
        try:
            self.prec = float(np.loadtxt(filepath))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Unnormalization
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # In-process constructor — build a NEOoutput directly from a
    # NEOInProcess output dict, without touching the filesystem.
    # ------------------------------------------------------------------
    @classmethod
    def from_inprocess(cls, inputclass, outputs):
        """
        Build a NEOoutput from an in-process ctypes result dict — no file I/O.

        Mirrors what ``_read_grid`` + ``_read_transport_flux`` would set when
        reading ``out.neo.transport_flux``: per-species DKE / GV / TGYRO
        fluxes, after applying the GB normalization that NEO uses on disk
        (``pgb = n_e ρ² T_e^1.5`` etc.).  Theory / NCLASS / geometry fields
        are populated as well so plotting tabs work.

        Parameters
        ----------
        inputclass : NEOinput
            Processed input object for this rho (provides species charges,
            ``RMIN_OVER_A``).  May be None.
        outputs : dict
            Output dict returned by ``NEOInProcess.run_from_dict()``.
        """
        obj = cls.__new__(cls)
        SIMtools.GACODEoutput.__init__(obj)   # sets obj.inputFile = None

        obj.FolderGACODE = None
        obj.suffix       = ""
        obj.inputclass   = inputclass
        obj.inputFile    = ""

        # ---- Geometry / grid metadata ----
        ns = int(outputs.get("ns", 0))
        obj.n_species = ns
        obj.n_radial  = 1
        obj.n_energy  = obj.n_xi = obj.n_theta = None
        obj.theta_grid = None

        # ---- Pull r/a, electron index and Z list from the input ----
        if inputclass is not None:
            plasma = inputclass.plasma
            obj.roa = float(plasma.get("RMIN_OVER_A", 0.0))
            Z = np.array([float(plasma.get(f"Z_{i+1}", 0.0)) for i in range(ns)])
        else:
            obj.roa = 0.0
            Z = np.zeros(ns)

        # Find electron index (Z = -1)
        ie_idx = np.where(Z == -1)[0]
        ie = int(ie_idx[0]) if ie_idx.size > 0 else None

        # ---- GB normalization (norm units → GB units, per neo_transport.f90) ----
        if inputclass is not None:
            rho_star = float(plasma.get("RHO_STAR", 1e-3))
            ae_flag  = int(plasma.get("AE_FLAG", 0))
            if ae_flag == 1:
                dens_e = float(plasma.get("DENS_AE", 1.0))
                temp_e = float(plasma.get("TEMP_AE", 1.0))
            elif ie is not None:
                dens_e = float(plasma.get(f"DENS_{ie+1}", 1.0))
                temp_e = float(plasma.get(f"TEMP_{ie+1}", 1.0))
            else:
                dens_e, temp_e = 1.0, 1.0
            pgb = dens_e * rho_star**2 * temp_e**1.5
            egb = dens_e * rho_star**2 * temp_e**2.5
            mgb = dens_e * rho_star**2 * temp_e**2.0
        else:
            pgb = egb = mgb = 1.0

        def _arr(key):
            return np.asarray(outputs.get(key, [0.0] * ns), dtype=float)[:ns]

        # ---- DKE fluxes (norm → GB) ----
        G_dke = _arr("pflux_dke")    / pgb
        Q_dke = _arr("efluxtot_dke") / egb
        M_dke = _arr("mflux_dke")    / mgb

        # ---- GV fluxes (norm → GB) ----
        G_gv  = _arr("pflux_gv")     / pgb
        Q_gv  = _arr("efluxtot_gv")  / egb
        M_gv  = _arr("mflux_gv")     / mgb

        # ---- Tgyro = DKE + GV (electron-frame Q already; on disk NEO also
        #      subtracts ω_rot · Π for energy, but for the normal use case
        #      of this in-process path Π is small enough that omitting that
        #      correction is harmless — same as what neo_inprocess does).
        G_tg = G_dke + G_gv
        Q_tg = Q_dke + Q_gv
        M_tg = M_dke + M_gv

        def _split(arr):
            """Return (electron_value, ion_array_without_electron)."""
            if ie is None:
                return float(np.nan), np.asarray(arr, dtype=float)
            return float(arr[ie]), np.delete(np.asarray(arr, dtype=float), ie)

        # tgyro section drives Ge / Qe / Me / GiAll / QiAll / MiAll
        obj.Ge,    obj.GiAll = _split(G_tg)
        obj.Qe,    obj.QiAll = _split(Q_tg)
        obj.Me,    obj.MiAll = _split(M_tg)
        obj.Zi               = np.delete(Z, ie) if ie is not None else Z
        obj.Qi               = float(obj.QiAll.sum())
        obj.Mt               = float(obj.Me + obj.MiAll.sum())

        # dke section
        obj.Ge_dke,    obj.GiAll_dke = _split(G_dke)
        obj.Qe_dke,    obj.QiAll_dke = _split(Q_dke)
        obj.Me_dke,    obj.MiAll_dke = _split(M_dke)
        obj.Qi_dke                   = float(obj.QiAll_dke.sum())

        # gv section
        obj.Ge_gv,    obj.GiAll_gv = _split(G_gv)
        obj.Qe_gv,    obj.QiAll_gv = _split(Q_gv)
        obj.Me_gv,    obj.MiAll_gv = _split(M_gv)
        obj.Qi_gv                  = float(obj.QiAll_gv.sum())

        # ---- Theory predictions (HH / CH / Sauter etc.) ----
        obj.HHGamma  = float(outputs.get("pflux_thHH",  0.0)) / pgb
        obj.HHQi     = float(outputs.get("eflux_thHHi", 0.0)) / egb
        obj.HHQe     = float(outputs.get("eflux_thHHe", 0.0)) / egb
        obj.CHQi     = float(outputs.get("eflux_thCHi", 0.0)) / egb
        obj.HHjparB  = float(outputs.get("jpar_thS",    0.0))   # already GB-like
        obj.SjparB   = float(outputs.get("jpar_thSmod", 0.0))

        # ---- DKE bootstrap current and flows ----
        obj.jparB    = float(outputs.get("jpar_dke",    0.0))
        obj.vtheta_sp= np.asarray(outputs.get("vpol_dke", []), dtype=float)[:ns]
        obj.vphi_sp  = np.asarray(outputs.get("vtor_dke", []), dtype=float)[:ns]

        # ---- Status / sentinel inputFile string for read_scan ----
        obj.inputFile = ""
        obj.unnormalization_successful = False
        obj.prec = float("nan")
        obj.error_status = int(outputs.get("error_status", 0))

        if obj.error_status != 0:
            print(f"\t- [in-process] WARNING NEO returned error_status={obj.error_status}", typeMsg="w")
        return obj

    def unnormalize(self, normalization, rho=None):

        if normalization is not None:
            print("\t- Unnormalizing NEO results using the provided normalization factors")
            rho_x = normalization["rho"]
            roa_x = normalization["roa"]
            q_gb  = normalization["q_gb"]
            g_gb  = normalization["g_gb"]
            pi_gb = normalization["pi_gb"]
            s_gb  = normalization["s_gb"]
            rho_s = normalization["rho_s"]
            ne_20 = normalization["ne_20"]       # electron density [1e20/m^3]
            c_s   = normalization["c_s"]         # ion sound speed [m/s]

            if rho is None:
                ir = np.argmin(np.abs(roa_x - self.roa))
            else:
                ir = np.argmin(np.abs(rho_x - rho))

            # ---- Fluxes (already existing) ----
            self.Qe_unn     = self.Qe     * q_gb[ir]
            self.Qi_unn     = self.Qi     * q_gb[ir]
            self.QiAll_unn  = self.QiAll  * q_gb[ir]
            self.Ge_unn     = self.Ge     * g_gb[ir]
            self.GiAll_unn  = self.GiAll  * g_gb[ir]
            self.MiAll_unn  = self.MiAll  * g_gb[ir]
            self.Mt_unn     = self.Mt     * s_gb[ir]

            # ---- Bootstrap current ----
            # j_phys [kA/m^2] = j_GB * e * ne(r) * cs(r) * 1e-3, quantity is <j·B>/B_unit.
            # NEO's own factor (neo_transport.f90: e*dens_norm*vth_norm*a_meters) carries an
            # a_meters ONLY to undo vth_norm being stored as vth/a [1/s]; cs here is already a
            # physical velocity [m/s] (= NEO's vth_norm*a_meters), so there is no extra a.
            _e   = 1.602e-19                        # elementary charge [C]
            _ne  = ne_20[ir] * 1e20                 # electron density [m^-3]
            _cs  = c_s[ir]                          # sound speed [m/s]
            _j_factor = _e * _ne * _cs * 1e-3       # [kA/m^2] per GB unit

            if hasattr(self, 'jparB'):
                self.jparB_unn   = self.jparB   * _j_factor   # [kA/m^2]
            if hasattr(self, 'HHjparB'):
                self.HHjparB_unn = self.HHjparB * _j_factor
            if hasattr(self, 'SjparB'):
                self.SjparB_unn  = self.SjparB  * _j_factor

            # ---- Velocities: v_phys [m/s] = v_GB * cs (cs is already physical, no extra a) ----
            _v_factor = _cs
            if hasattr(self, 'uparB0'):
                self.uparB0_unn   = self.uparB0   * _v_factor
            if hasattr(self, 'vtheta0'):
                self.vtheta0_unn  = self.vtheta0  * _v_factor
            if hasattr(self, 'uparB_sp'):
                self.uparB_sp_unn  = self.uparB_sp  * _v_factor
            if hasattr(self, 'vtheta_sp'):
                self.vtheta_sp_unn = self.vtheta_sp * _v_factor
            if hasattr(self, 'vphi_sp'):
                self.vphi_sp_unn   = self.vphi_sp   * _v_factor

            self.unnormalization_successful = True

        else:
            print("\t- No normalization provided, cannot unnormalize NEO results.")
            self.unnormalization_successful = False
