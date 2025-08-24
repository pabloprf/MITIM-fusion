import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools, GUItools
from mitim_tools.gacode_tools.utils import GACODErun, GACODEdefaults
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from mitim_tools import __version__ as mitim_version
from IPython import embed

class GX(GACODErun.gacode_simulation):
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        def code_call(folder, n, p, additional_command="", **kwargs):
            return f"    gx {folder}/gxplasma.in  &\n"

        self.run_specifications = {
            'code': 'gx',
            'input_file': 'gxplasma.in',
            'code_call': code_call,
            'control_function': GACODEdefaults.addGXcontrol,
            'controls_file': 'input.gx.controls',
            'state_converter': 'to_gx',
            'input_class': GXinput,
            'complete_variation': None,
            'default_cores': 4,  # Default gpus to use in the simulation
            'output_class': GXoutput,
            'output_store': 'GXout'
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t GX class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles = self.ResultsFiles_minimal = [
            'gxplasma.eik.out',
            'gxplasma.eiknc.nc',
            'gxplasma.gx_geo.log',
            'gxplasma.restart.nc',
            'gxplasma.big.nc',
            'gxplasma.out.nc'
            ]

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
            
        fig1 = self.fn.add_figure(label=f"{extratitle}Summary", tab_color=fn_color)
        
        grid = plt.GridSpec(1, 2, hspace=0.7, wspace=0.2)

        if colors is None:
            colors = GRAPHICStools.listColors()

        ax1 = fig1.add_subplot(grid[0, 0])
        ax2 = fig1.add_subplot(grid[0, 1])
        
        i = 0
        for label in labels:
            for irho in range(len(self.rhos)):
                c = self.results[label]['GXout'][irho]
                for iky in range(len(c.ky)):
                    ax1.plot(c.t, c.w[:, iky], label=f"{label} rho={self.rhos[irho]} ky={c.ky[iky]}", color=colors[i])
                    ax2.plot(c.t, c.g[:, iky], label=f"{label} rho={self.rhos[irho]} ky={c.ky[iky]}", color=colors[i])
                    i += 1
                    
        for ax in [ax1, ax2]:
            ax.set_xlabel("Time")
            ax.set_xlim(left=0)
            GRAPHICStools.addDenseAxis(ax)
        ax1.set_ylabel("Real frequency")
        ax1.legend(loc='best', prop={'size': 4})
        
        ax2.set_ylabel("Growth rate")

class GXinput(GACODErun.GACODEinput):
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
                'Dimensions': 
                    [ ['ntheta', 'nperiod', 'ny', 'nx', 'nhermite', 'nlaguerre', 'nspecies'], [] ],
                'Domain':
                    [ ['y0', 'boundary'], [] ],
                'Physics': 
                    [ ['nonlinear_mode', 'ei_colls'], ['beta'] ],
                'Time':
                    [ ['t_max', 'scheme'], [] ],
                'Initialization':
                    [ ['ikpar_init', 'init_field', 'init_amp', 'gaussian_init'], [] ],
                'Geometry':
                    [ 
                       ['geo_option'],
                       ['rhoc', 'Rmaj', 'R_geo', 'shift', 'qinp', 'shat', 'akappa', 'akappri', 'tri', 'tripri', 'betaprim']
                    ],
                'Dissipation':
                    [ ['closure_model', 'hypercollisions', 'nu_hyper_m', 'p_hyper_m', 'nu_hyper_l', 'p_hyper_l', 'hyper', 'D_hyper', 'p_hyper'], [] ],
                'Restart':
                    [ ['restart', 'save_for_restart'], [] ],
                'Diagnostics':
                    [ ['nwrite', 'omega', 'fluxes', 'fields'], [] ]
            }

            param_written = []
            for block_name, params in blocks.items():
                param_written = self._write_block(f, f"[{block_name}]", params, param_written)
                
            param_written = self._write_block_species(f, param_written)

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
        for i in range(100):
            if f"z_{i+1}" in self.plasma:
                self.num_recorded += 1
            else:
                break

        z, dens, temp, mass, fprim, tprim, vnewk, typeS = '[ ', '[ ', '[ ', '[ ', '[ ', '[ ', '[ ', '[ '
        for i in range(self.num_recorded):
            typeS += f'"{_fmt_value(self.plasma[f"type_{i+1}"])}", '
            z += f'{_fmt_value(self.plasma[f"z_{i+1}"])}, '
            mass += f'{_fmt_value(self.plasma[f"mass_{i+1}"])}, '
            dens += f'{_fmt_value(self.plasma[f"dens_{i+1}"])}, '
            temp += f'{_fmt_value(self.plasma[f"temp_{i+1}"])}, '
            fprim += f'{_fmt_value(self.plasma[f"fprim_{i+1}"])}, '
            tprim += f'{_fmt_value(self.plasma[f"tprim_{i+1}"])}, '
            vnewk += f'{_fmt_value(self.plasma[f"vnewk_{i+1}"])}, '
            
            param_written.append(f"type_{i+1}")
            param_written.append(f"z_{i+1}")
            param_written.append(f"dens_{i+1}")
            param_written.append(f"temp_{i+1}")
            param_written.append(f"mass_{i+1}")
            param_written.append(f"fprim_{i+1}")
            param_written.append(f"tprim_{i+1}")
            param_written.append(f"vnewk_{i+1}")

        f.write("[species]\n")
        f.write(f" z     = {z[:-2]} ]\n")
        f.write(f" dens  = {dens[:-2]} ]\n")
        f.write(f" temp  = {temp[:-2]} ]\n")
        f.write(f" mass  = {mass[:-2]} ]\n")
        f.write(f" fprim = {fprim[:-2]} ]\n")
        f.write(f" tprim = {tprim[:-2]} ]\n")
        f.write(f" vnewk = {vnewk[:-2]} ]\n")
        f.write(f" type  = {typeS[:-2]} ]\n")
        f.write("\n")

        return param_written

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
            f.write(f" {p.ljust(23)} = {_fmt_value(self.controls[p])}\n")
            param_written.append(p)
        for p in param[1]:
            f.write(f" {p.ljust(23)} = {_fmt_value(self.plasma[p])}\n")
            param_written.append(p)
        f.write(f'\n')

        return param_written


class GXoutput(GACODErun.GACODEoutput):
    def __init__(self, FolderGACODE, suffix="", **kwargs):
        super().__init__()
        
        self.FolderGACODE, self.suffix = FolderGACODE, suffix

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
        self.g = data.groups['Diagnostics'].variables['omega_kxkyt'][:,1:,ikx,1]      # (time, ky)

        # get fluxes
        # Qi = data.groups['Fluxes'].variables['qflux'][:,1]
        # Qe = data.groups['Fluxes'].variables['qflux'][:,0]
        # Ge = data.groups['Fluxes'].variables['pflux'][:,0]
        

