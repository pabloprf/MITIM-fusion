import netCDF4
import numpy as np
from pathlib import Path
from mitim_tools.misc_tools import GRAPHICStools, IOtools, GUItools, CONFIGread
from mitim_tools.gacode_tools.utils import GACODEdefaults, CGYROutils
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.utils import SIMplot
from mitim_tools.simulation_tools.utils.GKaveraging import GKaverager, PRIMARY_CHANNELS, resolve_fixed_tmin
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from mitim_tools import __version__ as mitim_version
from IPython import embed

class GX(SIMtools.mitim_simulation, SIMplot.GKplotting):
    def __init__(
        self,
        rhos=[None],  # List of rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        def code_call(folder, n, p, additional_command="", **kwargs):
            return f"cd {folder}; gx -n {n} gxplasma.in > gxplasma.mitim.log"

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
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t GX class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.output_files_simulation["minimal"] = [
            'gxplasma.out.nc'
        ]

        self.output_files_simulation["complete"] = self.output_files_simulation["minimal"] + [
            'gxplasma.eik.out',
            'gxplasma.eiknc.nc',
            'gxplasma.gx_geo.log',
            'gxplasma.big.nc',
            'gxplasma.mitim.log',
            'gxplasma.restart.nc',
            ]
        

    '''
    Redefined here so that I handle restart properly and
        I can choose numerical setup based on plasma and
        I can send VMEC file if needed and
        I can send restarts too
    '''
    def run(
        self,
        subfolder,
        numerics_based_on_plasma = None,    # A dictionary with the parameters to match
        restart_files = None,               # If provided, dictionary with rhos as keys and restart file paths as values
        **kwargs_sim_run
    ):
        
        # ------------------------------------
        # If it's a case with VMEC, send the file
        # ------------------------------------
        from mitim_tools.plasmastate_tools.utils import VMECtools
        if isinstance(self.profiles, VMECtools.vmec_state):
            print('- Plasma comes from VMEC file, sending it along the GX run', typeMsg='i')
            
            # Get the VMEC file path
            vmec_file = Path(self.profiles.header[0].split('VMEC location')[-1][2:-1])
            
            # Add "vmec_file" to GX namelist
            
            if 'extraOptions' not in kwargs_sim_run:
                kwargs_sim_run['extraOptions'] = {}
            if 'vmec_file' in kwargs_sim_run['extraOptions']:
                print('\t- Overwriting vmec_file in extraOptions', typeMsg='w')

            kwargs_sim_run['extraOptions']['geo_option'] = f'"vmec"'
            kwargs_sim_run['extraOptions']['vmec_file'] = f'"{vmec_file.name}"'
            
            # Add the file to the list of additional files to send, equal for both radii
            
            if 'additional_files_to_send' not in kwargs_sim_run:
                kwargs_sim_run['additional_files_to_send'] = {}
            for rho in self.rhos:
                if rho not in kwargs_sim_run['additional_files_to_send']:
                    kwargs_sim_run['additional_files_to_send'][float(rho)] = []
                if vmec_file not in kwargs_sim_run['additional_files_to_send'][rho]:
                    kwargs_sim_run['additional_files_to_send'][float(rho)].append(vmec_file)
            
        # ------------------------------------
        # Check about restarts
        # ------------------------------------
        
        # Assume every template writes a restart file named "gxplasma.restart.nc"
        # If extraOptions indicate not to write a restart, remove the file
        if not kwargs_sim_run.get('extraOptions', {}).get('save_for_restart', True):
            self.output_files_simulation["complete"].remove("gxplasma.restart.nc")
            print("\t- Not saving restart file")

        # If the name has changed, update the results files list
        if kwargs_sim_run.get('extraOptions', {}).get('restart_to_file', None) is not None:
            restart_name = kwargs_sim_run['extraOptions']['restart_to_file']
            self.output_files_simulation["complete"].remove("gxplasma.restart.nc")
            self.output_files_simulation["complete"].append(restart_name)
            print(f"\t- Saving restart file as {restart_name}")

        if (self.profiles.type == 'vmec') or (kwargs_sim_run.get('extraOptions', {}).get('geo_option', 'miller') == 'vmec'):
            self.output_files_simulation["complete"].remove('gxplasma.eik.out')
            self.output_files_simulation["complete"].remove('gxplasma.eiknc.nc')

        # ------------------------------------
        # Add numerical setup based on plasma
        # ------------------------------------
        
        if restart_files is not None:
            if 'extraOptions' not in kwargs_sim_run:
                kwargs_sim_run['extraOptions'] = {}

            kwargs_sim_run['extraOptions']['restart'] = True
            kwargs_sim_run['extraOptions']['restart_from_file'] = []
            
            for rho in self.rhos:
                if rho not in restart_files:
                    raise Exception(f"[MITIM] You provided restart_files dictionary but {rho = } is missing.")
                
                # Add restart file specification
                kwargs_sim_run['extraOptions']['restart_from_file'].append(f'"{Path(restart_files[rho]).name}"')
                
                # Add the file to the list of additional files to sendi
                if 'additional_files_to_send' not in kwargs_sim_run:
                    kwargs_sim_run['additional_files_to_send'] = {}
                if rho not in kwargs_sim_run['additional_files_to_send']:
                    kwargs_sim_run['additional_files_to_send'][float(rho)] = []

                kwargs_sim_run['additional_files_to_send'][float(rho)].append(Path(restart_files[rho]))

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
        
        # If it has radii, we need to correct the labels
        labels = self._correct_rhos_labels(labels)
        
        if fn is None:
            self.fn = GUItools.FigureNotebook("GX MITIM Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn

        if colors is None:
            colors = GRAPHICStools.listColors()

        # Fluxes
        fig = self.fn.add_figure(label=f"{extratitle}Transport Fluxes", tab_color=fn_color)
        axsFluxes_t = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )

        # Fluxes (ky)
        fig = self.fn.add_figure(label=f"{extratitle}Transport Fluxes (ky)")
        axsFluxes_ky = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )

        # Turbulence
        # One "Averaging" figure per case: window selection diagnostics of the primary fluxes
        for i, label in enumerate(labels):
            if hasattr(self.results[label], 'averaging'):
                fig = self.fn.add_figure(label=f"Averaging, {label}")
                self.results[label].averaging.plot(fig=fig, color=GRAPHICStools.listColors()[i % len(GRAPHICStools.listColors())], label_plot=label)

        fig = self.fn.add_figure(label="Turbulence (linear)")
        axsTurbulence = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )

        colors = GRAPHICStools.listColors()

        for j in range(len(labels)):
            
            self.plot_fluxes(
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )

            self.plot_fluxes_ky(
                axs=axsFluxes_ky,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )

            self.plot_turbulence(
                axs=axsTurbulence,
                label=labels[j],
                c=colors[j],
            )
           
        # Back to the original labels before _correct_rhos_labels
        self.results = self.results_all

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
                    [ ['save_for_restart', 'nsave','restart_to_file', 'restart', 'restart_from_file', 'append_on_restart'], [] ],
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
    def __init__(self, FolderGACODE, suffix="", tmin = 0.0, tmin_is_rel=True, minimal = False, averaging=None, **kwargs):
        '''
        tmin sets the left edge of the window used for signal analysis (averaging method "fixed").
          tmin >= 0                    : absolute time (a/cs).
          tmin <  0, tmin_is_rel=True  : fraction of the total simulation time
                                         counted from the end. e.g. tmin=-0.25
                                         -> the last 25% of the run.
          tmin <  0, tmin_is_rel=False : absolute offset (a/cs) from the end.
                                         e.g. tmin=-200 -> the last 200 a/cs
                                         of the run (self.tmin = t[-1] - 200).
        averaging: dict selecting the window/uncertainty method, {'method': 'fixed' | 'quends' | 'howard_gkav', ...};
          see GKaveraging.py. Linear runs always use the last time point.
        '''
        super().__init__()

        self.FolderGACODE, self.suffix = Path(FolderGACODE), suffix

        if suffix == "":
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} without suffix")
        else:
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} with suffix {suffix}")

        self.inputclass = GXinput(file=self.FolderGACODE / f"gxplasma.in{self.suffix}")

        self.read(tmin, tmin_is_rel=tmin_is_rel)

    def read(self, tmin, tmin_is_rel=True):

        data = netCDF4.Dataset(self.FolderGACODE / f"gxplasma.out.nc{self.suffix}")

        # GX build provenance from the file's global attributes (names vary by GX version; empty if none)
        try:
            self.gx_version = "; ".join(f"{k}={data.getncattr(k)}" for k in data.ncattrs()
                                        if any(s in k.lower() for s in ('version', 'git', 'hash', 'commit')))
        except Exception:
            self.gx_version = ""

        self.t = data.groups['Grids'].variables['time'][:] # (time)
        self.theta = data.groups['Grids'].variables['theta'][:]
        
        # Growth rates
        ikx = 0
        self.ky = data.groups['Grids'].variables['ky'][:]   # (ky)
        self.f = np.transpose(data.groups['Diagnostics'].variables['omega_kxkyt'][:,:,ikx,0])    # (ky, time)
        self.g = np.transpose(data.groups['Diagnostics'].variables['omega_kxkyt'][:,:,ikx,1])    # (ky, time)

        # Fluxes
        Q = data.groups['Diagnostics'].variables['HeatFlux_st']     # (time, species)
        G = data.groups['Diagnostics'].variables['ParticleFlux_st'] # (time, species)
        
        # Fluxes per ky
        Q_ky = data.groups['Diagnostics'].variables['HeatFlux_kyst'][:,:,:]   # (time, species, ky)
        G_ky = data.groups['Diagnostics'].variables['ParticleFlux_kyst'][:,:,:]   # (time, species, ky)

        # Assume electrons are always last
        self.Qe = Q[:,-1]
        self.Qi_all = np.transpose(Q[:,:-1]) # (species-1, time)
        self.Qi = self.Qi_all.sum(axis=0) # (time)
        self.Ge = G[:,-1]
        self.Gi_all = np.transpose(G[:,:-1]) # (species-1, time)
        self.Gi = self.Gi_all.sum(axis=0) # (time)

        self.Qe_ky = np.transpose(Q_ky[:,-1,:])   # (ky, time)
        self.Qi_all_ky = np.transpose(Q_ky[:,:-1,:], (1,2,0))   # (species-1, ky, time)
        self.Qi_ky = self.Qi_all_ky.sum(axis=0)  # (ky, time)
        self.Ge_ky = np.transpose(G_ky[:,-1,:])   # (time, ky)
        self.Gi_all_ky = np.transpose(G_ky[:,:-1,:], (1,2,0))   # (species-1, ky, time)
        self.Gi_ky = self.Gi_all_ky.sum(axis=0)  # (time, ky)
        
        self.ions_flags = [i for i in range(1, self.Qi_all.shape[0])]
        self.all_names = [f'i{i}' for i in range(1, self.Qi_all.shape[0]+1)]
        
        # If linear, last tmin
        self.linear = not bool(data.groups['Inputs'].groups['Controls'].variables['nonlinear_mode'][:])

        self.tmin = resolve_fixed_tmin(self.t, tmin=tmin, tmin_is_rel=tmin_is_rel)
        if self.linear:
            self.tmin = self.t[-1]
            print(f"\t- Linear simulation, setting tmin to last time", typeMsg='i')

        self._build_averager(averaging, tmin, tmin_is_rel)
        self._signal_analysis()

    def _build_averager(self, averaging, tmin, tmin_is_rel):
        '''
        Selects the saturated window from Qi/Qe/Ge (GB) and updates self.tmin to it (see GKaveraging.py).
        Note: GX time is in its own normalization (a/v_ti), not a/cs; the howard_gkav thresholds were tuned for CGYRO time units.
        '''
        options = {'method': 'fixed', **(averaging or {})}
        method = options.pop('method')
        if self.linear and method != 'fixed':
            print(f"\t- Linear run: averaging method '{method}' ignored, using the last time point", typeMsg='i')
            method = 'fixed'
        traces = {k: self.__dict__[k] for k in PRIMARY_CHANNELS if k in self.__dict__}
        tmin, tmin_is_rel = (self.tmin, True) if self.linear else (tmin, tmin_is_rel)
        self.averaging = GKaverager(self.t, traces, method=method, tmin=tmin, tmin_is_rel=tmin_is_rel, label=IOtools.clipstr(self.FolderGACODE), **options)
        self.tmin = self.averaging.t_start

    # ---- harvest interface (SIMtools.GACODEoutput): inputs from the gxplasma.in file, fluxes are time averages by self.averaging
    def harvest_inputs(self):
        '''
        Every key of the gxplasma.in this radius ran with. The [species] arrays (`z = [ 1.0, -1.0 ]`)
        become z_1, z_2, ... (GXinput's own names) and quotes are stripped, then parsed like TGLF/NEO
        inputs; GXinput itself would keep '[' as the value of every species key.
        '''
        file = getattr(self.inputclass, 'file', None)
        if file is None or not Path(file).is_file():
            return {**self.inputclass.controls, **self.inputclass.plasma}
        lines = []
        for line in Path(file).read_text().splitlines():
            line = line.split('#')[0].strip()
            if '=' not in line or line.startswith('['):
                continue
            key, val = (s.strip() for s in line.split('=', 1))
            if val.startswith('['):
                lines += [f"{key}_{i+1} = {v.strip().strip(chr(34))}" for i, v in enumerate(val.strip('[] ').split(',')) if v.strip()]
            else:
                lines.append(f"{key} = {val.strip(chr(34))}")
        return SIMtools.buildDictFromInput("\n".join(lines))

    def harvest_outputs(self):
        out = {}
        for k in ('Qe', 'Qi', 'Ge'):
            for s in ('mean', 'std'):
                v = getattr(self, f'{k}_{s}', None)
                if v is not None:
                    out[f'{k}_{s}'] = float(v)
        for name, arr in (('mean', 'Qi_all_mean'), ('std', 'Qi_all_std')):
            vals = getattr(self, arr, None)
            if vals is not None:
                for i, v in enumerate(np.atleast_1d(vals)):
                    out[f'Qi_{i+1}_{name}'] = float(v)
        out.update(CGYROutils.harvest_averaging_fields(self, ('Qe', 'Qi', 'Ge')))
        return out

    def harvest_provenance(self):
        return {'averaging': CGYROutils.harvest_averaging_provenance(self)}

    def harvest_hash_extra(self):
        return {'t_last': float(self.t[-1]) if getattr(self, 't', None) is not None and len(self.t) else None}

    def _signal_analysis(self):

        flags = [
            'g',
            'f',
            'Qe',
            'Qi_all',
            'Qi',
            'Ge',
            'Qe_ky',
            'Qi_all_ky',
            'Qi_ky',
            'Ge_ky',
        ]
        
        for iflag in flags:
            self.__dict__[iflag+'_mean'], self.__dict__[iflag+'_std'] = self.averaging.mean_std(
                    self.__dict__[iflag],
                    label_print=iflag,
                    print_msg=True,
                    )

