import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools, GUItools
from mitim_tools.gacode_tools.utils import GACODErun, GACODEdefaults
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.style_tools.themes import apply_theme
from mitim_tools import __mitimroot__
from IPython import embed

class NEO(SIMtools.mitim_simulation):
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        def code_call(folder, n, p, additional_command="", **kwargs):
            return f"neo -e {folder} -n {n} -p {p} {additional_command}"

        def code_slurm_settings(name, minutes, total_cores_required, cores_per_code_call, type_of_submission, array_list=None, **kwargs_slurm):

            slurm_settings = {
                "name": name,
                "minutes": minutes,
                'job_array_limit': None,    # Limit to this number at most running jobs at the same time?
            }

            if type_of_submission == "slurm_standard":
                
                slurm_settings['ntasks'] = total_cores_required // cores_per_code_call  # How many independent NEO calls is this?
                
            elif type_of_submission == "slurm_array":

                slurm_settings['ntasks'] = 1                                            # Each job in the array is one NEO call
                
                slurm_settings['job_array'] = ",".join(array_list)

            # Each simulation call will use these resources (must match what the code_call requests)
            slurm_settings['cpuspertask'] = cores_per_code_call

            return slurm_settings

        self.run_specifications = {
            'code': 'neo',
            'input_file': 'input.neo',
            'code_call': code_call,
            'code_slurm_settings': code_slurm_settings,
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



    # def prep(self, inputgacode, folder):
    #     self.inputgacode = inputgacode
    #     self.folder = IOtools.expandPath(folder)

    #     self.folder.mkdir(parents=True, exist_ok=True)



    def run_vgen(self, subfolder="vgen1", vgenOptions={}, cold_start=False):

        self.folder_vgen = self.folder / f"{subfolder}"

        # ---- Default options

        vgenOptions.setdefault("er", 2)
        vgenOptions.setdefault("vel", 1)
        vgenOptions.setdefault("numspecies", len(self.inputgacode.Species))
        vgenOptions.setdefault("matched_ion", 1)
        vgenOptions.setdefault("nth", "17,39")

        # ---- Prepare

        runThisCase = check_if_files_exist(
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

        self.inputgacode.write_state(file=(self.folder_vgen / f"input.gacode"))

        # ---- Run

        if runThisCase:
            file_new = GACODErun.runVGEN(
                self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder
            )
        else:
            print(f"\t- Required files found in {subfolder}, not running VGEN",typeMsg="i",)
            file_new = self.folder_vgen / f"vgen" / f"input.gacode"

        # ---- Postprocess

        from mitim_tools.gacode_tools import PROFILEStools
        self.inputgacode_vgen = PROFILEStools.gacode_state(file_new, derive_quantities=True, mi_ref=self.inputgacode.mi_ref)


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
            a     = normalization["rmin"][-1]   # LCFS minor radius [m]
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
            # From NEO source: jpar_phys [kA/m^2] = jpar_GB * e * ne(r) * cs(r) * a * 1e-3
            # (quantity is <j·B>/B_unit in kA/m^2)
            _e   = 1.602e-19                        # elementary charge [C]
            _ne  = ne_20[ir] * 1e20                 # electron density [m^-3]
            _cs  = c_s[ir]                          # sound speed [m/s]
            _j_factor = _e * _ne * _cs * a * 1e-3   # [kA/m^2] per GB unit

            if hasattr(self, 'jparB'):
                self.jparB_unn   = self.jparB   * _j_factor   # [kA/m^2]
            if hasattr(self, 'HHjparB'):
                self.HHjparB_unn = self.HHjparB * _j_factor
            if hasattr(self, 'SjparB'):
                self.SjparB_unn  = self.SjparB  * _j_factor

            # ---- Velocities: v_phys [m/s] = v_GB * cs * a ----
            _v_factor = _cs * a
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
