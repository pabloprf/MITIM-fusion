import netCDF4
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools, GUItools, CONFIGread
from mitim_tools.gacode_tools.utils import GACODEdefaults, CGYROutils
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.utils import SIMplot
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from mitim_tools import __version__ as mitim_version
from IPython import embed

class GX(SIMtools.mitim_simulation, SIMplot.GKplotting):
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        def code_call(folder, n, p, additional_command="", **kwargs):
            return f"cd {folder}; gx -n {n} gxplasma.in > gxplasma.mitim.log"

        def code_slurm_settings(name, minutes, total_cores_required, cores_per_code_call, type_of_submission, raise_warning=True,array_list=None):

            slurm_settings = {
                "name": name,
                "minutes": minutes,
            }

            # Gather if this is a GPU enabled machine
            machineSettings = CONFIGread.machineSettings(code='gx')
            
            if machineSettings['gpus_per_node'] == 0:
                if raise_warning:
                    raise Exception("[MITIM] GX needs GPUs to run, but the selected machine does not have any GPU configured. Please select another machine in the config file with gpus_per_node>0.")
                else:
                    print("[MITIM] Warning: GX needs GPUs to run, but the selected machine does not have any GPU configured. Running without GPUs, but this will likely fail.", typeMsg="w")

            if type_of_submission == "slurm_standard":
                
                slurm_settings['ntasks'] = total_cores_required
                slurm_settings['gpuspertask'] = 1 # Because of MPI, each task needs a GPU, and I'm passing cores_per_code_call per task
                slurm_settings['job_array'] = None

            elif type_of_submission == "slurm_array":

                slurm_settings['ntasks'] = cores_per_code_call
                slurm_settings['gpuspertask'] = 1
                slurm_settings['job_array'] = ",".join(array_list)

            return slurm_settings

        self.run_specifications = {
            'code': 'gx',
            'input_file': 'gxplasma.in',
            'code_call': code_call,
            'code_slurm_settings': code_slurm_settings,
            'control_function': GACODEdefaults.addGXcontrol,
            'controls_file': 'input.gx.controls',
            'state_converter': 'to_gx',
            'input_class': GXinput,
            'complete_variation': None,
            'default_cores': 4,  # Default gpus to use in the simulation
            'output_class': GXoutput,
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t GX class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles_minimal = [
            'gxplasma.out.nc'
        ]

        self.ResultsFiles = self.ResultsFiles_minimal + [
            'gxplasma.eik.out',
            'gxplasma.eiknc.nc',
            'gxplasma.gx_geo.log',
            'gxplasma.big.nc',
            'gxplasma.mitim.log',
            'gxplasma.restart.nc',
            ]

    '''
    Redefined here so that I handle restart properly and
    I can choose numerical setup based on plasma
    '''
    def run(
        self,
        subfolder,
        numerics_based_on_plasma = None, # A dictionary with the parameters to match
        **kwargs_sim_run
    ):
        
        # ------------------------------------
        # Check about restarts
        # ------------------------------------
        
        # Assume every template writes a restart file named "gxplasma.restart.nc"
        # If extraOptions indicate not to write a restart, remove the file
        if not kwargs_sim_run.get('extraOptions', {}).get('save_for_restart', True):
            self.ResultsFiles.remove("gxplasma.restart.nc")
            print("\t- Not saving restart file")

        # If the name has changed, update the results files list
        if kwargs_sim_run.get('extraOptions', {}).get('restart_to_file', None) is not None:
            restart_name = kwargs_sim_run['extraOptions']['restart_to_file']
            self.ResultsFiles.remove("gxplasma.restart.nc")
            self.ResultsFiles.append(restart_name)
            print(f"\t- Saving restart file as {restart_name}")

        if ('Stellarator' in kwargs_sim_run.get('code_settings', 'Linear Tokamak')) or \
            (kwargs_sim_run.get('extraOptions', {}).get('geo_option', 'miller') == 'vmec'):
            self.ResultsFiles.remove('gxplasma.eik.out')
            self.ResultsFiles.remove('gxplasma.eiknc.nc')

        # ------------------------------------
        # Add numerical setup based on plasma
        # ------------------------------------
        if numerics_based_on_plasma is not None:
            pass
        #TODO

        # ------------------------------------
        # Run the super run
        # ------------------------------------
        
        super().run(subfolder, **kwargs_sim_run)

    def plot(
        self,
        fn=None,
        labels=["gx1"],
        extratitle="",
        fn_color=None,
        colors=None,
        ):
        
        if fn is None:
            self.fn = GUItools.FigureNotebook("GX MITIM Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn

        if colors is None:
            colors = GRAPHICStools.listColors()

        # Fluxes
        fig = self.fn.add_figure(label=f"{extratitle}Transport Fluxes", tab_color=fn_color)

        grid = plt.GridSpec(1, 3, hspace=0.7, wspace=0.2)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])

        i = 0
        for label in labels:
            for irho in range(len(self.rhos)):
                c = self.results[label]['output'][irho]
                
                typeLs = '-' if c.t.shape[0]>20 else '-s'
                
                self._plot_trace(ax1,self.results[label]['output'][irho],"Qe",c=colors[i],lw=1.0,ls='-',label_plot=f"{label}, Total")
                self._plot_trace(ax2,self.results[label]['output'][irho],"Qi",c=colors[i],lw=1.0,ls='-',label_plot=f"{label}, Total")
                self._plot_trace(ax3,self.results[label]['output'][irho],"Ge",c=colors[i],lw=1.0,ls='-',label_plot=f"{label}, Total")

                i += 1

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("Time ($L_{ref}/c_s$)")
            ax.set_xlim(left=0)
            GRAPHICStools.addDenseAxis(ax)

        ax1.set_title('Electron heat flux')
        ax1.set_ylabel("Electron heat flux ($Q_e/Q_{GB}$)")
        ax1.legend(loc='best', prop={'size': 12})

        ax2.set_title('Ion heat flux')
        ax2.set_ylabel("Ion heat flux ($Q_i/Q_{GB}$)")
        ax2.legend(loc='best', prop={'size': 12})

        ax3.set_title('Electron particle flux')
        ax3.set_ylabel("Electron particle flux ($\\Gamma_e/\\Gamma_{GB}$)")
        ax3.legend(loc='best', prop={'size': 12})
        
        plt.tight_layout()

            
        # Linear stability
        fig = self.fn.add_figure(label=f"{extratitle}Linear Stability", tab_color=fn_color)
        
        grid = plt.GridSpec(2, 2, hspace=0.7, wspace=0.2)


        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        
        i = 0
        for label in labels:
            for irho in range(len(self.rhos)):
                c = self.results[label]['output'][irho]
                
                typeLs = '-' if c.t.shape[0]>20 else '-s'
                
                for iky in range(len(c.ky)):
                    ax1.plot(c.t, c.w[:, iky], typeLs, label=f"{label} rho={self.rhos[irho]} ky={c.ky[iky]}", color=colors[i])
                    ax2.plot(c.t, c.g[:, iky], typeLs, label=f"{label} rho={self.rhos[irho]} ky={c.ky[iky]}", color=colors[i])
                    i += 1
                    
        for ax in [ax1, ax2]:
            ax.set_xlabel("Time ($L_{ref}/c_s$)")
            ax.set_xlim(left=0)
            GRAPHICStools.addDenseAxis(ax)
        ax1.set_ylabel("Real frequency")
        ax1.legend(loc='best', prop={'size': 4})
        ax2.set_ylabel("Growth rate")

        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])

        i = 0
        for label in labels:
            for irho in range(len(self.rhos)):
                c = self.results[label]['output'][irho]
                ax3.plot(c.ky, c.w[-1, :], '-s', markersize = 5, label=f"{label} rho={self.rhos[irho]}", color=colors[i])
                ax4.plot(c.ky, c.g[-1, :], '-s', markersize = 5, label=f"{label} rho={self.rhos[irho]}", color=colors[i])
                i += 1

        for ax in [ax3, ax4]:
            ax.set_xlabel("$k_\\theta\\rho_s$")
            ax.set_xlim(left=0)
            GRAPHICStools.addDenseAxis(ax)

        ax3.set_ylabel("Real frequency")
        ax3.legend(loc='best', prop={'size': 12})
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax4.set_ylabel("Growth rate")
        ax4.set_ylim(bottom=0)

        plt.tight_layout()


class GXinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.gx.controls",
            code='GX',
            n_species='nspecies'
        )

    # GX has a very particular way to write its state
    def write_state(self, file=None):
        
        if file is None:
            file = self.file


        with open(file, "w") as f:
            f.write("#-------------------------------------------------------------------------\n")
            f.write(f"# {self.code} input file modified by MITIM {mitim_version}\n")
            f.write("#-------------------------------------------------------------------------\n")

            # title: [controls], [plasma]
            blocks = {
                '': 
                    [ ['debug'], [] ],
                '[Dimensions]': 
                    [ ['ntheta', 'nperiod', 'ny', 'nx', 'nhermite', 'nlaguerre'], ['nspecies'] ],
                '[Domain]':
                    [ ['y0', 'boundary'], [] ],
                '[Physics]': 
                    [ ['nonlinear_mode', 'ei_colls'], ['beta'] ],
                '[Time]':
                    [ ['t_max', 'scheme', 'dt', 'nstep'], [] ],
                '[Initialization]':
                    [ ['ikpar_init', 'init_field', 'init_amp', 'gaussian_init'], [] ],
                '[Geometry]':
                    [ 
                       ['geo_option', 'alpha', 'npol', 'vmec_file'],
                       ['rhoc', 'Rmaj', 'R_geo', 'shift', 'qinp', 'shat', 'akappa', 'akappri', 'tri', 'tripri', 'betaprim', 'torflux']
                    ],
                '[Dissipation]':
                    [ ['closure_model', 'hypercollisions', 'nu_hyper_m', 'p_hyper_m', 'nu_hyper_l', 'p_hyper_l', 'hyper', 'D_hyper', 'p_hyper', 'D_H', 'w_osc', 'p_HB', 'HB_hyper'], [] ],
                '[Restart]':
                    [ ['save_for_restart', 'nsave','restart_to_file', 'restart', 'restart_from_file'], [] ],
                '[Diagnostics]':
                    [ ['nwrite', 'omega', 'fluxes', 'fields', 'moments'], [] ]
            }

            param_written = []
            for block_name, params in blocks.items():
                param_written = self._write_block(f, f"{block_name}", params, param_written)
                
            param_written = self._write_block_species(f, param_written)

        # Check that parameters were all considerd in the blocks
        for param in self.controls | self.plasma:
            if param not in param_written:
                print(f"Warning: {param} not written to file", typeMsg="q")

    def _write_block(self,f,name, param, param_written):

        # Local formatter: floats -> 6 significant figures in exponential (uppercase),
        # ints stay as ints, bools as 0/1, sequences space-separated with same rule.
        def _fmt_num(x):
            import numpy as _np
            if isinstance(x, (bool, _np.bool_)):
                return "true" if x else "false"
            if isinstance(x, (_np.floating, float)):
                # 6 significant figures in exponential => 5 digits after decimal
                return f"{float(x):.5E}"
            if isinstance(x, (_np.integer, int)):
                return f"{int(x)}"
            return str(x)

        def _fmt_value(val):
            import numpy as _np
            if isinstance(val, (list, tuple, _np.ndarray)):
                # Flatten numpy arrays but keep ordering; join with spaces
                if isinstance(val, _np.ndarray):
                    flat = val.flatten().tolist()
                else:
                    flat = list(val)
                return " ".join(_fmt_num(v) for v in flat)
            return _fmt_num(val)

        f.write(f'{name}\n')
        for p in param[0]:
            if p in self.controls:
                if self.controls[p] is not None:
                    f.write(f" {p.ljust(23)} = {_fmt_value(self.controls[p])}\n")
                param_written.append(p)
        for p in param[1]:
            if p in self.plasma:
                if self.plasma[p] is not None:
                    f.write(f" {p.ljust(23)} = {_fmt_value(self.plasma[p])}\n")
                param_written.append(p)
        f.write(f'\n')

        return param_written

    def _write_block_species(self, f, param_written):

        # Local formatter: floats -> 6 significant figures in exponential (uppercase),
        # ints stay as ints, bools as 0/1, sequences space-separated with same rule.
        def _fmt_num(x):
            import numpy as _np
            if isinstance(x, (bool, _np.bool_)):
                return "true" if x else "false"
            if isinstance(x, (_np.floating, float)):
                # 6 significant figures in exponential => 5 digits after decimal
                return f"{float(x):.5E}"
            if isinstance(x, (_np.integer, int)):
                return f"{int(x)}"
            return str(x)

        def _fmt_value(val):
            import numpy as _np
            if isinstance(val, (list, tuple, _np.ndarray)):
                # Flatten numpy arrays but keep ordering; join with spaces
                if isinstance(val, _np.ndarray):
                    flat = val.flatten().tolist()
                else:
                    flat = list(val)
                return " ".join(_fmt_num(v) for v in flat)
            return _fmt_num(val)

        self.num_recorded = 0
        for i in range(1000):
            if f"z_{i+1}" in self.plasma:
                self.num_recorded += 1
            else:
                break

        z, dens, temp, mass, fprim, tprim, vnewk, typeS = '[  ', '[  ', '[  ', '[  ', '[  ', '[  ', '[  ', '[  '
        for i in range(self.num_recorded):
            typeS += f'"{_fmt_value(self.plasma[f"type_{i+1}"])}",   '
            z += f'{_fmt_value(self.plasma[f"z_{i+1}"])},   '
            mass += f'{_fmt_value(self.plasma[f"mass_{i+1}"])},   '
            dens += f'{_fmt_value(self.plasma[f"dens_{i+1}"])},   '
            temp += f'{_fmt_value(self.plasma[f"temp_{i+1}"])},   '
            fprim += f'{_fmt_value(self.plasma[f"fprim_{i+1}"])},   '
            tprim += f'{_fmt_value(self.plasma[f"tprim_{i+1}"])},   '
            vnewk += f'{_fmt_value(self.plasma[f"vnewk_{i+1}"])},   '
            
            param_written.append(f"type_{i+1}")
            param_written.append(f"z_{i+1}")
            param_written.append(f"mass_{i+1}")
            param_written.append(f"dens_{i+1}")
            param_written.append(f"temp_{i+1}")
            param_written.append(f"fprim_{i+1}")
            param_written.append(f"tprim_{i+1}")
            param_written.append(f"vnewk_{i+1}")

        f.write("[species]\n")
        f.write(f" type  = {typeS[:-4]}  ]\n")
        f.write(f" z     = {z[:-4]}  ]\n")
        f.write(f" mass  = {mass[:-4]}  ]\n")
        f.write(f" dens  = {dens[:-4]}  ]\n")
        f.write(f" temp  = {temp[:-4]}  ]\n")
        f.write(f" fprim = {fprim[:-4]}  ]\n")
        f.write(f" tprim = {tprim[:-4]}  ]\n")
        f.write(f" vnewk = {vnewk[:-4]}  ]\n")
        
        f.write("\n")

        return param_written

class GXoutput(SIMtools.GACODEoutput):
    def __init__(self, FolderGACODE, suffix="", tmin = 0.0,  **kwargs):
        super().__init__()
        
        self.FolderGACODE, self.suffix = FolderGACODE, suffix
        
        self.tmin = tmin

        if suffix == "":
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} without suffix")
        else:
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} with suffix {suffix}")

        self.inputclass = GXinput(file=self.FolderGACODE / f"gxplasma.in{self.suffix}")

        self.read()

    def read(self):

        data = netCDF4.Dataset(self.FolderGACODE / f"gxplasma.out.nc{self.suffix}")
        
        self.t = data.groups['Grids'].variables['time'][:] # (time)
        
        # Growth rates
        ikx = 0
        self.ky = data.groups['Grids'].variables['ky'][1:]   # (ky)
        self.w = data.groups['Diagnostics'].variables['omega_kxkyt'][:,1:,ikx,0]    # (time, ky)
        self.g = data.groups['Diagnostics'].variables['omega_kxkyt'][:,1:,ikx,1]    # (time, ky)

        # Fluxes
        Q = data.groups['Diagnostics'].variables['HeatFlux_st']     # (time, species)
        G = data.groups['Diagnostics'].variables['ParticleFlux_st'] # (time, species)

        # Assume electrons are always last
        self.Qe = Q[:,-1]
        self.QiAll = Q[:,:-1]
        self.Qi = self.QiAll.sum(axis=1)
        self.Ge = G[:,-1]
        self.GiAll = G[:,:-1]
        self.Gi = self.GiAll.sum(axis=1)

        self._signal_analysis()

    def _signal_analysis(self):
        
        flags = [
            'Qe',
            'Qi',
            'Ge',
        ]
        
        for iflag in flags:
            self.__dict__[iflag+'_mean'], self.__dict__[iflag+'_std'] = CGYROutils.apply_ac(
                    self.t,
                    self.__dict__[iflag],
                    tmin=self.tmin,
                    label_print=iflag,
                    print_msg=True,
                    )

