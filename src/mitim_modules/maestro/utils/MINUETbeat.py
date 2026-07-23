import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GUItools, GRAPHICStools, PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat, _format_seconds
from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from IPython import embed

'''
MINUET beat: in-process substitute for the transp_soft beat.

Runs current diffusion + sawteeth at FIXED kinetic profiles and fixed-boundary
equilibrium using the standalone MINUET package (current diffusion coupled to a
fixed-boundary Grad-Shafranov solver). The kinetics, species and source columns
of the incoming state pass through VERBATIM; only the equilibrium blocks and the
q/johm/jbs columns are evolved.

MINUET is an optional MITIM dependency (pip install "mitim-fusion[minuet]"), so
it is imported lazily and only when a minuet beat actually runs or plots.
'''

def _import_minuet():
    try:
        import minuet as minuet_pkg
    except ImportError as e:
        raise ImportError(
            '[MITIM] The minuet beat requires the standalone MINUET package. '
            'Install it with: pip install "mitim-fusion[minuet]" '
            '(or an editable install of the minuet repo)') from e
    return minuet_pkg


class minuet_beat(beat):

    def __init__(self, maestro_instance, folder_name = None):
        super().__init__(maestro_instance, beat_name = 'minuet', folder_name = folder_name)

    def prepare(
            self,
            t_end               = 20.0,         # [s] simulation length (transp_soft flattop_window analog)
            evolve_equilibrium  = True,         # True: coupled CD+GS (transp_soft-equivalent); False: pure CD on frozen geometry
            sawteeth            = True,
            sawtooth_model      = 'porcelli',   # 'porcelli' (self-consistent trigger) | 'fixed_period'
            sawtooth_period     = None,         # [s], fixed_period model only
            reconnection_model  = 'kadomtsev',  # 'kadomtsev' | 'porcelli' | 'partial'
            ensure_sawtooths    = None,         # extend t_end so at least this many crashes are expected (transp beat contract)
            resistivity_model   = 'sauter',     # 'sauter' | 'spitzer'
            bootstrap_model     = 'sauter',     # 'sauter' | None (purely ohmic)
            Ip_from_frozen      = True,         # command Ip to the frozen engineering current(MA) (CUR-ufile analog)
            gs_ns               = 128,
            gs_ntheta           = 256,
            n_cells             = 200,          # current-diffusion radial cells
            rtol                = None,         # BDF relative tolerance (None -> MINUET default)
            n_save              = 201,          # saved time frames (plot granularity of run.minuet)
            **kwargs
            ):

        self.prepare_minimal(
            t_end = t_end,
            evolve_equilibrium = evolve_equilibrium,
            sawteeth = sawteeth,
            sawtooth_model = sawtooth_model,
            sawtooth_period = sawtooth_period,
            reconnection_model = reconnection_model,
            ensure_sawtooths = ensure_sawtooths,
            resistivity_model = resistivity_model,
            bootstrap_model = bootstrap_model,
            Ip_from_frozen = Ip_from_frozen,
            gs_ns = gs_ns,
            gs_ntheta = gs_ntheta,
            n_cells = n_cells,
            rtol = rtol,
            n_save = n_save,
        )

        # Grab things from previous beats (e.g. extend t_end to ensure sawtooth crashes)
        self._inform(ensure_sawtooths = ensure_sawtooths)

    def prepare_minimal(self, **kwargs):
        '''
        Stash the namelist knobs on self. finalize()/merge_parameters() (which also run
        on the skip path) do not need them, but summary() reports the models used.
        '''
        self.minuet_config = kwargs

    # -----------------------------------------------------------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------------------------------------------------------

    def _build_models(self, mn):
        '''
        Translate the string knobs of the namelist into MINUET model objects
        (resistivity, bootstrap, sawtooth with its redistribution component).
        '''
        cfg = self.minuet_config

        resistivity = {
            'sauter':  mn.SauterResistivity,
            'spitzer': mn.SpitzerResistivity,
        }[cfg['resistivity_model']]()

        bootstrap = mn.SauterBootstrap() if cfg['bootstrap_model'] == 'sauter' else None

        if not cfg['sawteeth']:
            sawtooth = None
        else:
            redistribution = {
                'kadomtsev': mn.KadomtsevReconnection,
                'porcelli':  mn.PorcelliReconnection,
                'partial':   mn.PartialReconnection,
            }[cfg['reconnection_model']]()

            if cfg['sawtooth_model'] == 'fixed_period':
                if cfg['sawtooth_period'] is None:
                    raise ValueError('[MITIM] minuet beat: sawtooth_model "fixed_period" requires sawtooth_period [s]')
                sawtooth = mn.FixedPeriodSawtooth(period = cfg['sawtooth_period'], redistribution = redistribution)
            else:
                sawtooth = mn.PorcelliSawtooth(redistribution = redistribution)
                if not cfg['evolve_equilibrium']:
                    print('\t- CD-only mode supports only fixed-period sawteeth; the Porcelli trigger will be inert', typeMsg='w')

        return resistivity, bootstrap, sawtooth

    def _trim_folded_surfaces(self, mn, input_file):
        '''
        MINUET's gacode reader refuses states whose outermost MXH surfaces self-intersect
        (their FSA metrics carry a 1/|J2| pole -- e.g. the near-X-point separatrix of a
        FreeGS-initialized MAESTRO state). The remedy is the minuet analog of the transp
        beat's boundary_surface_psin backoff: cut the state at the last NON-FOLDED surface
        and run MINUET with that (slightly interior, rounder) boundary. The export is then
        grafted onto the trimmed grid and merge_parameters() re-grids back to the frozen
        resolution (MITIM extrapolates the thin cut band of equilibrium columns).

        Returns the file to hand to MINUET (the original when nothing folds).
        '''
        from minuet.geometry import mxh_surface_family, mxh_fold_statistic

        ig = mn.InputGacode.from_file(str(input_file))
        pr = ig.profiles
        sel = pr['rho'] > 0.0
        shp_c = np.column_stack([pr.get(f'shape_cos{n}', np.zeros_like(pr['rho']))[sel] for n in range(0, 7)])
        shp_s = np.column_stack([pr.get(f'shape_sin{n}', np.zeros_like(pr['rho']))[sel] for n in range(0, 7)])
        R, Z = mxh_surface_family(pr['rmin'][sel], pr['rmaj'][sel], pr['zmag'][sel],
                                  pr['kappa'][sel], pr['delta'][sel], pr['zeta'][sel],
                                  shp_c, shp_s, n_theta = 512)
        fold, _ = mxh_fold_statistic(R, Z, pr['rho'][sel])

        # Only OUTERMOST folded surfaces are cut (an interior fold is a deeper pathology to surface loudly)
        bad_outer = 0
        for f in fold[::-1]:
            if f <= 0:
                bad_outer += 1
            else:
                break
        if bad_outer == 0:
            return input_file

        rho_full = self.profiles_current.profiles['rho(-)']
        rho_new = rho_full[:len(rho_full) - bad_outer]
        print(f'\t- Incoming state has {bad_outer} self-intersecting outermost MXH surface(s); '
              f'trimming to rho <= {rho_new[-1]:.4f} as MINUET boundary (boundary_surface_psin-style backoff)', typeMsg='w')

        profiles_trimmed = copy.deepcopy(self.profiles_current)
        profiles_trimmed.changeResolution(rho_new = rho_new)
        trimmed_file = self.folder / 'input.gacode_trimmed'
        profiles_trimmed.write_state(file = trimmed_file)
        self._trim_rho = float(rho_new[-1])   # merge_parameters restores the frozen band beyond this
        return trimmed_file

    def run(self, **kwargs):

        mn = _import_minuet()
        cfg = self.minuet_config

        # ---------------------------------------------------------------------------------------
        # Write the incoming state to the run folder (audit trail + MINUET source), cutting
        # any self-intersecting outermost surfaces (see _trim_folded_surfaces)
        # ---------------------------------------------------------------------------------------
        input_file = self.folder / 'input.gacode'
        self.profiles_current.write_state(file = input_file)
        input_file = self._trim_folded_surfaces(mn, input_file)

        # ---------------------------------------------------------------------------------------
        # Build MINUET settings
        # ---------------------------------------------------------------------------------------
        resistivity, bootstrap, sawtooth = self._build_models(mn)

        diffusion_kwargs = dict(n_cells = cfg['n_cells'], n_save = cfg['n_save'])
        if cfg['rtol'] is not None:
            diffusion_kwargs['rtol'] = cfg['rtol']
        if cfg['Ip_from_frozen']:
            # Command Ip to the frozen engineering current (the CUR-ufile analog); MINUET
            # distributes the initial commanded-vs-state mismatch over its edge buffer
            Ip_MA = float(self.maestro_instance.profiles_with_engineering_parameters.profiles['current(MA)'][0])
            diffusion_kwargs['Ip'] = abs(Ip_MA) * 1e6  # [A]
            print(f'\t- Commanding Ip = {abs(Ip_MA):.3f} MA from frozen engineering parameters')

        settings = mn.Settings(
            t_end = cfg['t_end'],
            evolve_equilibrium = cfg['evolve_equilibrium'],
            resistivity = resistivity,
            bootstrap = bootstrap,
            sawtooth = sawtooth,
            gs_ns = cfg['gs_ns'],
            gs_ntheta = cfg['gs_ntheta'],
            diffusion = mn.DiffusionSettings(**diffusion_kwargs),
        )

        # ---------------------------------------------------------------------------------------
        # Run the discharge and persist the MINUET object (mitim_plot_minuet consumes this)
        # ---------------------------------------------------------------------------------------
        print(f'\t- Running MINUET for {cfg["t_end"]:.1f} s ({"coupled CD+GS" if cfg["evolve_equilibrium"] else "CD-only"})')
        # Pass an explicit InputGacode (not the path): minuet's path sniffer keys on the
        # first header line, which for MITIM scratch-created states ("# Created from
        # scratch...") would misroute the file to the geqdsk reader
        m = mn.minuet(mn.InputGacode.from_file(str(input_file)), settings = settings)
        m.run()
        m.save(self.folder / 'run.minuet')

        # ---------------------------------------------------------------------------------------
        # Export the evolved equilibrium ONTO the untouched kinetics/grid of the input file
        # ---------------------------------------------------------------------------------------
        m.export_input_gacode(str(self.folder / 'input.gacode_minuet'), keep_kinetics = str(input_file))

        # ---------------------------------------------------------------------------------------
        # Sidecar with the scalars that _inform_save()/summary() need after cleanup
        # ---------------------------------------------------------------------------------------
        crashes = np.asarray(m.history['crashes']) if m.history is not None else np.array([])
        minuet_results = {
            'sawtooth_times': crashes,
            't_end': float(m.result.t[-1]),
            'Ip_MA_realized': float(m.result.ip_enc[-1, -1]) * 1e-6,
            'q0_initial': float(m.result.q0[0]),
            'q0_final': float(m.result.q0[-1]),
            'evolve_equilibrium': cfg['evolve_equilibrium'],
            'models': {
                'resistivity': cfg['resistivity_model'],
                'bootstrap': cfg['bootstrap_model'],
                'sawtooth': (cfg['sawtooth_model'] if cfg['sawteeth'] else None),
                'reconnection': (cfg['reconnection_model'] if cfg['sawteeth'] else None),
            },
        }
        np.save(self.folder / 'minuet_results.npy', minuet_results)

    # -----------------------------------------------------------------------------------------------------------------------
    # Finalize and merge
    # -----------------------------------------------------------------------------------------------------------------------

    def finalize(self, force_auxiliary_heating_at_output = None, **kwargs):

        # Refresh folder_output from self.folder only if the source still exists.
        # On a re-invocation after `maestro.keep_all_files: false` wiped self.folder,
        # folder_output already has the authoritative content from the prior run.
        if (self.folder / 'input.gacode_minuet').exists():

            # Remove previous output files
            for item in self.folder_output.glob('*'):
                if item.is_file():
                    item.unlink(missing_ok=True)

            # Persist sidecar + discharge object (copy under keep_all_files: true; move otherwise),
            # so plotting and _inform_save survive the cleanup loop
            self._persist(self.folder / 'minuet_results.npy', self.folder_output / 'minuet_results.npy')
            if (self.folder / 'run.minuet').exists():
                self._persist(self.folder / 'run.minuet', self.folder_output / 'run.minuet')

            # Write profiles to output folder
            self.profiles_output = PROFILEStools.gacode_state(self.folder / 'input.gacode_minuet')
            self.profiles_output.write_state(file = self.folder_output / 'input.gacode')

        else:
            # Cleanup case: load profiles from the existing folder_output snapshot
            self.profiles_output = PROFILEStools.gacode_state(self.folder_output / 'input.gacode')

        # Gaussian-source injection is deferred to merge_parameters(): its normalization
        # needs volume integrals, which are only reliable on the full frozen-resolution
        # grid (a trimmed-boundary state can have NaN edge volume integrals)
        self._force_auxiliary_heating = force_auxiliary_heating_at_output

    def merge_parameters(self):
        '''
        MINUET evolves only the equilibrium blocks and current columns; the export with
        keep_kinetics carries the grid, kinetics, species and source columns of the incoming
        state verbatim. This merge therefore only guards against grid leaks (safety re-grid)
        and re-pins the engineering scalars (MINUET's exported current is the realized Ampere
        Ip, which can differ at the closure level from the commanded engineering value).
        No auxiliary-power rescale is needed: sources either pass through verbatim or (for
        gaussian_sources) are injected at finalize already normalized exactly to the
        engineering Pe/Pi, so the transp beat's rescale-to-frozen dance does not apply.
        '''

        # Write the pre-merge input.gacode before modifying it
        profiles_output_pre_merge = copy.deepcopy(self.profiles_output)
        profiles_output_pre_merge.write_state(file = self.folder_output / 'input.gacode_pre_merge')

        p_frozen = self.maestro_instance.profiles_with_engineering_parameters

        # MINUET's trustworthy radial span BEFORE re-gridding: the keep_kinetics export
        # CLAMPS the equilibrium columns below its innermost stored surface (constant rmin
        # over any base-grid points finer than the CD grid near the axis -> zero Jacobian
        # -> NaN volume integrals downstream). Detect the clamp on the export grid.
        rho_exp = self.profiles_output.profiles['rho(-)']
        rmin_exp = self.profiles_output.profiles['rmin(m)']
        i_clamp = 0
        for i in range(1, min(30, len(rmin_exp))):
            if rmin_exp[i] <= rmin_exp[i-1]:
                i_clamp = i
        rho_lo = rho_exp[i_clamp + 1] if i_clamp > 0 else None

        # Re-grid to the frozen resolution: a no-op when keep_kinetics preserved the grid
        # exactly; real work when the incoming state was trimmed (folded-surface backoff)
        self.profiles_output.changeResolution(rho_new = p_frozen.profiles['rho(-)'])

        # Below the near-axis clamp of the export, CONTINUE MINUET's own first healthy
        # row inward instead of restoring frozen rows: MXH shape moments are EVEN in rho
        # (zero axis slope), so a flat continuation is the correct limit, rmin goes
        # linearly through zero and polflux quadratically. A frozen-row restore here
        # created a shape DISCONTINUITY at the seam between frozen and minuet rows --
        # on ~mm-radius surfaces even a small jump flips the family Jacobian and the
        # fold guard would refuse the file on a later minuet ingest (seen once the
        # export fitter became accurate, 2026-07-22).
        rho = p_frozen.profiles['rho(-)']
        if rho_lo is not None:
            h = int(np.searchsorted(rho, rho_lo))
            if h > 0:
                print(f'\t\t\t* Continuing MINUET axis rows inward through the export clamp '
                      f'(rho < {rho[h]:.4f}: flat shapes, linear rmin, quadratic polflux)')
                prof = self.profiles_output.profiles
                for key in ('rmaj(m)', 'zmag(m)', 'kappa(-)', 'delta(-)', 'zeta(-)'):
                    prof[key][:h] = prof[key][h]
                for key in list(prof.keys()):
                    if key.startswith('shape_'):
                        prof[key][:h] = prof[key][h]
                prof['rmin(m)'][:h] = prof['rmin(m)'][h] * rho[:h] / rho[h]
                prof['polflux(Wb/radian)'][:h] = (
                    prof['polflux(Wb/radian)'][h] * (rho[:h] / rho[h]) ** 2)

        # Restore the frozen equilibrium VERBATIM beyond the trimmed boundary (the
        # equilibrium there was NOT evolved -- MINUET ran with a backed-off boundary).
        # Extrapolated MXH surfaces there would produce NaN volume integrals that
        # poison the frozen state for every later beat. The small q/polflux seam is
        # the price of the backoff; kinetics/sources/engineering are frozen-inserted
        # anyway.
        mask = np.zeros(len(rho), dtype=bool)
        if getattr(self, '_trim_rho', None) is not None:
            mask |= rho > self._trim_rho
        if mask.any():
            print(f'\t\t\t* Restoring frozen equilibrium beyond MINUET\'s trimmed boundary '
                  f'({int(mask.sum())} points: rho > {self._trim_rho:.4f})')
            for key, arr in self.profiles_output.profiles.items():
                if key in p_frozen.profiles and isinstance(arr, np.ndarray) and arr.ndim >= 1 \
                        and arr.shape == p_frozen.profiles[key].shape and len(arr) == len(mask):
                    arr[mask] = p_frozen.profiles[key][mask]
            # Rebase the restored OUTER polflux band so the column stays continuous at the
            # seam (the axis band needs no rebase: polflux(0) = 0 anchors both sides and
            # the values there are ~1e-5 Wb/rad)
            if getattr(self, '_trim_rho', None) is not None:
                outer = rho > self._trim_rho
                i_seam = int(np.argmax(outer))
                pol = self.profiles_output.profiles['polflux(Wb/radian)']
                pol[outer] += pol[i_seam - 1] - p_frozen.profiles['polflux(Wb/radian)'][i_seam - 1]

        # Re-insert frozen kinetic profiles exactly (guards interpolation leaks)
        print('\t\t\t* Bringing kinetic profiles of frozen plasma state to new plasma state')
        self.profiles_output.profiles['ne(10^19/m^3)'] = p_frozen.profiles['ne(10^19/m^3)']
        self.profiles_output.profiles['te(keV)'] = p_frozen.profiles['te(keV)']
        self.profiles_output.profiles['ti(keV)'][:,0] = p_frozen.profiles['ti(keV)'][:,0]
        self.profiles_output.makeAllThermalIonsHaveSameTemp()

        # Re-insert engineering parameters (except shape)
        print('\t\t\t* Bringing Bt and Ip of frozen plasma state to new plasma state')
        for key in ['current(MA)', 'bcentr(T)']:
            self.profiles_output.profiles[key] = p_frozen.profiles[key]

        self.profiles_output.derive_quantities()

        # Gaussian-source injection (heating.type = gaussian_sources): MINUET does not run
        # a heating model, so -- like the transp beat -- the prescribed Pe/Pi gaussians are
        # written straight into qrfe/qrfi at beat output (wired by preprocess_run_minuet;
        # reuses the transp beat's method, which only touches self.profiles_output).
        # Done here, on the full frozen-resolution grid, so the power normalization uses
        # healthy volume integrals. Idempotent (the parabola overwrites the columns), so
        # the skip path re-injecting is harmless.
        if getattr(self, '_force_auxiliary_heating', None) is not None:
            transp_beat._add_heating_profiles(self, self._force_auxiliary_heating)
            self.profiles_output.derive_quantities()

        # Write to final input.gacode
        self.profiles_output.write_state(file = self.folder_output / 'input.gacode')

    # -----------------------------------------------------------------------------------------------------------------------
    # MAESTRO interface
    # -----------------------------------------------------------------------------------------------------------------------

    def _inform_save(self, *args, **kwargs):

        summary_file = self.folder_output / 'minuet_results.npy'

        if summary_file.exists():
            minuet_results = np.load(summary_file, allow_pickle=True).item()
            sawtooth_times = np.asarray(minuet_results['sawtooth_times'])
        else:
            sawtooth_times = np.array([])

        self.maestro_instance.parameters_trans_beat['sawtooth_times'] = sawtooth_times
        # The q-profile / equilibrium changed, so previous PORTALS surrogate data is stale
        self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] = None
        # Note: impurity_order_transp is NOT touched -- MINUET never reorders species

    def _inform(self, ensure_sawtooths = None):
        '''
        Same minimum-time contract as the transp beat: if a previous beat recorded
        sawtooth_times, extend t_end so at least ensure_sawtooths crashes are expected.
        Caveat: TRANSP-produced sawtooth_times carry TRANSP's ~0.1 s pre-flattop offset
        while MINUET's clock starts at 0 -- acceptable for an over-estimate.
        '''

        t_end_minimum = 0.0
        if 'sawtooth_times' in self.maestro_instance.parameters_trans_beat and ensure_sawtooths is not None:

            t_end_minimum = self._determine_minimum_time(ensure_sawtooths = ensure_sawtooths)

            if self.minuet_config['t_end'] < t_end_minimum:
                print(f'\t- Extending MINUET simulation t_end from {self.minuet_config["t_end"]:.4f} s to {t_end_minimum:.4f} s to ensure at least {ensure_sawtooths} sawtooths (estimate)', typeMsg='i')

        self.minuet_config['t_end'] = max(self.minuet_config['t_end'], t_end_minimum)

    def _determine_minimum_time(self, ensure_sawtooths = None):
        # Same period-extrapolation logic as TRANSPbeat._determine_minimum_time

        sawtooth_times = self.maestro_instance.parameters_trans_beat['sawtooth_times']

        # No sawteeth in the previous run (e.g. q0 > 1): no period to estimate from
        if len(sawtooth_times) == 0:
            print('\t- Previous run had no sawtooth crashes; cannot estimate a minimum time from the sawtooth period', typeMsg='w')
            return 0.0

        if len(sawtooth_times) >= ensure_sawtooths:
            t_end_minimum = sawtooth_times[-1]
        else:
            howmany_missing = ensure_sawtooths - len(sawtooth_times)
            if len(sawtooth_times) >= 2:
                last_period = sawtooth_times[-1] - sawtooth_times[-2]
                t_end_minimum = sawtooth_times[-1] + howmany_missing * last_period * 1.1  # Overestimation factor of 1.1
            else:
                last_period = sawtooth_times[-1] - 0.0
                t_end_minimum = sawtooth_times[-1] + howmany_missing * last_period * 1.5  # Overestimation factor of 1.5

        return t_end_minimum

    # -----------------------------------------------------------------------------------------------------------------------
    # Outputs: grab, plot, summary
    # -----------------------------------------------------------------------------------------------------------------------

    def grab_output(self, **kwargs):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if not isitfinished:
            return None, None

        # The MINUET discharge object (lazily loaded; None if minuet not importable or file missing)
        m = None
        for candidate in [self.folder_output / 'run.minuet', self.folder / 'run.minuet']:
            if candidate.exists():
                try:
                    mn = _import_minuet()
                    m = mn.minuet.load(candidate)
                except Exception as e:
                    print(f'\t- Could not load {candidate.name}: {e}', typeMsg='w')
                break

        profiles = PROFILEStools.gacode_state(self.folder_output / 'input.gacode')

        return m, profiles

    def plot(self, fn = None, counter = 0, full_plot = True):

        if fn is None:
            fn = GUItools.FigureNotebook("MINUET")

        m, profiles = self.grab_output()

        # ---------------------------------------------------------------------------------------
        # Tab 1: input vs output plasma states
        # ---------------------------------------------------------------------------------------
        fig = fn.add_figure(label='MINUET state', tab_color=counter)
        axs = fig.subplot_mosaic(
            """
            ABCDHJ
            AEFGIK
            """
        )
        axs = [ ax for ax in axs.values() ]

        if (self.folder / 'input.gacode').exists():
            profiles_input = PROFILEStools.gacode_state(self.folder / 'input.gacode')
            profiles_input.plotRelevant(axs = axs, color = 'b', label = 'orig')

        if profiles is not None:
            profiles.plotRelevant(axs = axs, color = 'r', label = 'MINUET')

        GRAPHICStools.adjust_figure_layout(fig)

        # ---------------------------------------------------------------------------------------
        # Full MINUET notebook, appended into the MAESTRO FigureNotebook (needs the saved
        # discharge object). minuet's notebook() supports fn/label_prefix/tab_color natively;
        # the beat number keeps labels unique when several minuet beats are plotted.
        # ---------------------------------------------------------------------------------------
        if m is not None:
            beat_number = self.folder_beat.name.split('_')[-1]
            m.notebook(fn = fn, label_prefix = f'MINUET b#{beat_number}: ', tab_color = counter, show = False)

        msg = '\t\t- Plotting of MINUET beat done'

        return msg

    def summary(self, output_dir, counter = None, wall_time_s = None):
        '''
        Markdown section for the last MINUET beat: models, sawtooth statistics, q evolution.
        '''

        results_file = self.folder_output / 'minuet_results.npy'
        header_extra = f' (Beat {counter})' if counter is not None else ''
        if not results_file.exists():
            return f'## MINUET{header_extra}\n*(minuet_results.npy missing; no summary available)*\n'

        try:
            d = np.load(results_file, allow_pickle=True).item()
        except Exception as e:
            return f'## MINUET{header_extra}\n*(could not load minuet_results.npy: {e})*\n'

        sawtooth_times = np.asarray(d.get('sawtooth_times', []))
        models = d.get('models', {})

        md = [f'## MINUET{header_extra}', '']
        if wall_time_s is not None:
            md.append(f'- **Wall time:** {_format_seconds(wall_time_s)}')
        md.append(f'- **Mode:** {"coupled CD+GS" if d.get("evolve_equilibrium", True) else "CD-only (frozen geometry)"}')
        md.append(f'- **Duration:** {d.get("t_end", float("nan")):.2f} s')
        md.append(f'- **Models:** resistivity = {models.get("resistivity")}, bootstrap = {models.get("bootstrap")}, '
                  f'sawtooth = {models.get("sawtooth")} ({models.get("reconnection")})')
        md.append(f'- **q0:** {d.get("q0_initial", float("nan")):.3f} (initial) -> {d.get("q0_final", float("nan")):.3f} (final)')
        md.append(f'- **Realized Ip:** {d.get("Ip_MA_realized", float("nan")):.3f} MA')
        if len(sawtooth_times) >= 2:
            periods = np.diff(sawtooth_times)
            md.append(f'- **Sawteeth:** {len(sawtooth_times)} crashes, mean period {np.mean(periods)*1e3:.1f} ms')
        else:
            md.append(f'- **Sawteeth:** {len(sawtooth_times)} crashes')
        md.append('')

        return '\n'.join(md)


# -----------------------------------------------------------------------------------------------------------------------
# Defaults to help MAESTRO
# -----------------------------------------------------------------------------------------------------------------------

def preprocess_run_minuet(run_namelist, maestro_namelist, cpus, cold_start):
    '''
    MINUET runs no heating model, so for heating.type = gaussian_sources the prescribed
    Pe/Pi gaussians must be injected at beat output (same contract as preprocess_run_transp's
    force_auxiliary_heating_at_output, consumed by minuet_beat.finalize). For ICRH/NBI the
    incoming state's sources pass through verbatim (a real transp beat is needed to compute
    those depositions).
    '''

    if maestro_namelist["plasma"]["heating"]["type"] == 'gaussian_sources':

        print('\t- Gaussian sources specified, adding to run_namelist of MINUET beat')

        Pe = maestro_namelist["plasma"]["heating"]["parameters"]["Pe"]
        Pi = maestro_namelist["plasma"]["heating"]["parameters"]["Pi"]
        nu_source = maestro_namelist["plasma"]["heating"]["parameters"]["nu_source"]

        def P_auxiliary(rhotor):
            _, y = PLASMAtools.parabolicProfile(Tbar=1.0, nu=nu_source, rho=rhotor, Tedge=0.0)
            return y

        run_namelist['force_auxiliary_heating_at_output'] = {
            'Pe': [P_auxiliary, Pe],
            'Pi': [P_auxiliary, Pi],
            }

    return run_namelist
