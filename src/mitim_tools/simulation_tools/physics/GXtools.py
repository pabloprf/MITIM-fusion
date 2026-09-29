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

# Bash run right before `gx` in its folder, so a requeued array element resumes from its own checkpoint
_REQUEUE_RESUME_BASH = (__mitimroot__ / "templates" / "gx_requeue_resume.sh").read_text()

class GX(SIMtools.mitim_simulation, SIMplot.GKplotting):

    # Persist the slurm submission (run_type 'submit'), so a restarted PORTALS driver with check_existing_runs
    # re-attaches to the in-flight array instead of resubmitting (SIMtools.SubmissionRecord, gk_submission.py)
    _submission_metadata_filename = "gx_submission.json"

    # Restart file GX writes every nsave steps and reads back through restart_if_exists (input.gx.controls).
    # The PORTALS warm-start chain (restart_from_cases) retrieves it as <file>_<rho:.4f> and stages it back
    # under this name (cgyro_restart.RestartChain).
    _warm_start_file = "gxplasma.restart.nc"

    # Set by the PORTALS evaluator when restart_from_cases reads this run's restart files back (_restart_retrieval)
    keep_warm_start_file = False

    def __init__(
        self,
        rhos=[None],  # List of rho locations of interest
    ):

        super().__init__(rhos=rhos)

        def code_call(folder, n, p, additional_command="", **kwargs):
            # >> so the log of the first launch survives a requeue.
            # With n > 1, GX writes its netCDF files as parallel HDF5 through MPI-IO (nc_create_par, NC_COLLECTIVE). On NFS
            # (engaging's /orcd/pool and /orcd/scratch) Open MPI's file-I/O layer byte-range-locks every collective write
            # and the ranks stall there with the GPUs idle (2 GPUs: hung; 1 GPU writes serially and is unaffected).
            # Skipping those locks gives local-disk speed and bitwise-identical output (the ranks write disjoint regions).
            return (f"cd {folder}\n{_REQUEUE_RESUME_BASH}"
                    f"export OMPI_MCA_fs_ufs_lock_algorithm=1\n"
                    f"gx -n {n} gxplasma.in >> gxplasma.mitim.log 2>&1\n")   # stderr per radius (the array shares one slurm_error.dat)

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
            ]

        # The restart file joins 'complete' or 'optional' at run() time (_restart_retrieval)
        self.output_files_simulation["optional"] = []

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
        restart_files = None,               # If provided, dictionary with rhos as keys and restart file paths as values (warm start, see _warm_start_t_max)
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

        self._restart_retrieval(kwargs_sim_run.get('extraOptions', {}))
        self._big_retrieval(kwargs_sim_run.get('code_settings'), kwargs_sim_run.get('extraOptions', {}))
        self._check_nkx(kwargs_sim_run.get('code_settings'), kwargs_sim_run.get('extraOptions', {}))

        if (self.profiles.type == 'vmec') or (kwargs_sim_run.get('extraOptions', {}).get('geo_option', 'miller') == 'vmec'):
            self.output_files_simulation["complete"].remove('gxplasma.eik.out')
            self.output_files_simulation["complete"].remove('gxplasma.eiknc.nc')

        # ------------------------------------
        # Warm starts: restart_files are staged like the PORTALS chain stages its parents
        # ------------------------------------

        if restart_files is not None:
            files = dict(kwargs_sim_run.get('additional_files_to_send') or {})
            for rho in self.rhos:
                files[float(rho)] = list(files.get(float(rho), [])) + [(Path(restart_files[rho]), self._warm_start_file)]
            kwargs_sim_run['additional_files_to_send'] = files

        kwargs_sim_run['extraOptions'] = self._warm_start_t_max(
            kwargs_sim_run.get('code_settings'), kwargs_sim_run.get('extraOptions') or {}, kwargs_sim_run.get('additional_files_to_send'))

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

    def _restart_retrieval(self, extraOptions):
        '''
        Where the restart file GX writes goes among the retrieved files: nowhere with save_for_restart off;
        'optional' (retrieved whatever keep_files says and never unlinked by the keep_files 'pickle' cleanup, like
        CGYRO's bin.cgyro.restart) when a PORTALS warm-start chain reads it back (keep_warm_start_file); 'complete'
        (keep_files 'all') otherwise, so a run that is not warm-started never carries ~100s of MB per radius.
        '''
        name = str(extraOptions.get('restart_to_file', self._warm_start_file)).strip('"')
        for key in ("complete", "optional"):
            self.output_files_simulation[key] = [f for f in self.output_files_simulation[key] if f not in (self._warm_start_file, name)]

        if not extraOptions.get('save_for_restart', True):
            print("\t- Not saving restart file")
            return
        self.output_files_simulation["optional" if self.keep_warm_start_file else "complete"].append(name)
        if name != self._warm_start_file:
            print(f"\t- Saving restart file as {name} (warm starts and requeue resumes only read {self._warm_start_file})", typeMsg='w')

    def _check_nkx(self, code_settings, extraOptions):
        '''
        nkx = 1 + 2*((nx-1)//3) must have no prime factor > 127: where the shear is low GX sets jtwist = 1 and links all
        nkx modes into one extended flux tube, and cuFFT callbacks abort on a chain length with a large prime factor
        (GX exits within seconds, e.g. nx 288 -> nkx 191 at r/a 0.40 of ARC V3A). Caught here instead of on the GPU.
        '''
        controls = {**GACODEdefaults.addGXcontrol(code_settings), **extraOptions}
        if not controls.get('nonlinear_mode'):
            return
        nkx = 1 + 2 * ((int(controls["nx"]) - 1) // 3)
        n, largest, f = nkx, 1, 2
        while f * f <= n:
            while n % f == 0:
                largest, n = f, n // f
            f += 1
        largest = max(largest, n)
        if largest > 127:
            raise Exception(f"[MITIM] GX nx = {controls['nx']} gives nkx = {nkx}, whose prime factor {largest} > 127 makes GX abort "
                            "at low magnetic shear (jtwist = 1): choose an nx whose nkx = 1 + 2*((nx-1)//3) factors into small primes")

    def _big_retrieval(self, code_settings, extraOptions):
        '''
        GX writes gxplasma.big.nc only when [Diagnostics] fields or moments is on (diagnostics.cu:149), so it is an
        expected 'complete' file only then (the analogue presets turn both off; a missing expected file makes the
        retrieval fail and the run be repeated).
        '''
        controls = {**GACODEdefaults.addGXcontrol(code_settings), **extraOptions}
        complete = self.output_files_simulation["complete"]
        if controls.get('fields') or controls.get('moments'):
            if 'gxplasma.big.nc' not in complete:
                complete.append('gxplasma.big.nc')
        elif 'gxplasma.big.nc' in complete:
            complete.remove('gxplasma.big.nc')

    def _warm_start_t_max(self, code_settings, extraOptions, additional_files_to_send):
        '''
        t_max of the radii staged a restart file under the name GX reads (_warm_start_file), by the PORTALS
        restart_from_cases chain or by restart_files. GX restarts whenever that file is in its folder
        (restart_if_exists, input.gx.controls), continues the clock of the saved state (MomentsG::restart_read
        reads its `time`) and stops at t_max as an ABSOLUTE time (run_gx.cu: `while (... time < t_max)`).
        In MITIM, t_max on a warm start is the time ADDED on top of the saved state, as CGYRO's MAX_TIME, so
        those radii get t_max = t_saved + t_max (a per-rho list, which modifyInputs indexes in self.rhos order).
        Absolute rather than GX's own t_add, which is added to the time of whatever restart the run reads: an
        element requeued from its own checkpoint (gx_requeue_resume.sh) would run the full t_max again.
        '''
        parents = {}
        for rho in self.rhos:
            for entry in (additional_files_to_send or {}).get(float(rho), []):
                if isinstance(entry, tuple) and entry[1] == self._warm_start_file:
                    parents[float(rho)] = Path(entry[0])
        if not parents:
            return extraOptions

        t_add = extraOptions['t_max'] if 't_max' in extraOptions else GACODEdefaults.addGXcontrol(code_settings)['t_max']
        t_max = []
        for i, rho in enumerate(self.rhos):
            t_rho = float(t_add[i] if isinstance(t_add, (list, np.ndarray)) else t_add)
            if float(rho) in parents:
                with netCDF4.Dataset(parents[float(rho)]) as d:
                    t_saved = float(d.variables['time'][:])
                print(f"\t- rho={rho:.4f}: warm start from {IOtools.clipstr(parents[float(rho)])} saved at t={t_saved:.3f} a/cs, "
                      f"runs {t_rho:g} a/cs more (t_max={t_saved + t_rho:.3f})", typeMsg='i')
                t_rho += t_saved
            t_max.append(t_rho)
        return {**extraOptions, 't_max': t_max}

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
                    [ ['y0', 'x0', 'jtwist', 'boundary'], [] ],
                '[Physics]': 
                    [ ['nonlinear_mode', 'ei_colls', 'coll_conservation', 'fphi', 'fapar', 'fbpar', 'g_exb', 'ExBshear'], ['beta'] ],
                '[Time]':
                    [ ['t_max', 'scheme', 'dt', 'cfl', 'nstep'], [] ],
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
                    [ ['save_for_restart', 'nsave','restart_to_file', 'restart', 'restart_if_exists', 'restart_from_file', 'append_on_restart'], [] ],
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

        self.read(tmin, tmin_is_rel=tmin_is_rel, averaging=averaging)

    def read(self, tmin, tmin_is_rel=True, averaging=None):

        data = netCDF4.Dataset(self.FolderGACODE / f"gxplasma.out.nc{self.suffix}")

        # GX build provenance from the file's global attributes (names vary by GX version; empty if none)
        try:
            self.gx_version = "; ".join(f"{k}={data.getncattr(k)}" for k in data.ncattrs()
                                        if any(s in k.lower() for s in ('version', 'git', 'hash', 'commit')))
        except Exception:
            self.gx_version = ""

        # A run resumed after a requeue appends from its last checkpoint (gx_requeue_resume.sh), so the rows written
        # between that checkpoint and the preemption appear twice: keep the later ones, i.e. drop every row whose
        # time is not below all the times written after it
        t = data.groups['Grids'].variables['time'][:]
        keep = t < np.append(np.minimum.accumulate(t[::-1])[::-1][1:], np.inf)
        diag = data.groups['Diagnostics'].variables

        self.t = t[keep] # (time)
        self.theta = data.groups['Grids'].variables['theta'][:]

        # Growth rates
        ikx = 0
        self.ky = data.groups['Grids'].variables['ky'][:]   # (ky)
        self.f = np.transpose(diag['omega_kxkyt'][:][keep][:,:,ikx,0])    # (ky, time)
        self.g = np.transpose(diag['omega_kxkyt'][:][keep][:,:,ikx,1])    # (ky, time)

        # Fluxes
        Q = diag['HeatFlux_st'][:][keep]     # (time, species)
        G = diag['ParticleFlux_st'][:][keep] # (time, species)

        # Fluxes per ky
        Q_ky = diag['HeatFlux_kyst'][:][keep]   # (time, species, ky)
        G_ky = diag['ParticleFlux_kyst'][:][keep]   # (time, species, ky)

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

        # GX writes heat, particle and turbulent-heating fluxes but no momentum flux: Mt = 0 (w0 cannot be predicted with GX)
        self.Mt = np.zeros_like(self.Qe)

        # Turbulent energy exchange per species, GX "TurbulentHeating" (written with fluxes = true), in GB units
        # Q_GB/a: n_s Z_s < (h_s dchi/dt - chi dh_s/dt)/2 > (GX device_funcs.cu turbulent_heating_summand), positive
        # when species s gains energy. Same form, sign and units as CGYRO's exchange moment (cgyro_flux.f90, moment
        # 4), so the electron one is what PORTALS reads as QieGB_turb (transport_cgyro._EXCHANGE_SPEC), as for CGYRO.
        if 'TurbulentHeating_st' in diag:
            H = diag['TurbulentHeating_st'][:][keep]   # (time, species)
            self.Se = H[:,-1]
            self.Si_all = np.transpose(H[:,:-1]) # (species-1, time)
            self.Si = self.Si_all.sum(axis=0) # (time)
        
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
        Note: GX time is L_ref/sqrt(T_ref/m_ref) = a/c_s, as in CGYRO (MITIMstate.to_gx sets T_ref=Te, m_ref=m_D, L_ref=a).
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
        for k in ('Qe', 'Qi', 'Ge', 'Se'):
            for s in ('mean', 'std'):
                v = getattr(self, f'{k}_{s}', None)
                if v is not None:
                    out[f'{k}_{s}'] = float(v)
        for name, arr in (('mean', 'Qi_all_mean'), ('std', 'Qi_all_std')):
            vals = getattr(self, arr, None)
            if vals is not None:
                for i, v in enumerate(np.atleast_1d(vals)):
                    out[f'Qi_{i+1}_{name}'] = float(v)
        out.update(CGYROutils.harvest_averaging_fields(self, ('Qe', 'Qi', 'Ge', 'Se')))
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
            'Gi_all',
            'Gi',
            'Mt',
            'Qe_ky',
            'Qi_all_ky',
            'Qi_ky',
            'Ge_ky',
        ] + [k for k in ('Se', 'Si', 'Si_all') if k in self.__dict__]

        for iflag in flags:
            self.__dict__[iflag+'_mean'], self.__dict__[iflag+'_std'] = self.averaging.mean_std(
                    self.__dict__[iflag],
                    label_print=iflag,
                    print_msg=True,
                    )

