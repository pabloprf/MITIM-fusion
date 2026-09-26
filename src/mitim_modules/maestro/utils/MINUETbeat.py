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
fixed-boundary Grad-Shafranov solver). The kinetics and auxiliary source columns
of the incoming state pass through VERBATIM; the equilibrium blocks and the
q/johm/jbs/qohme columns are evolved. When no transp beat ran upstream, the beat also
does the transp beat's other job: it composes the ions from plasma.species and
recomputes the fusion/radiation/exchange columns (analytic targets).

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
            gs_ns               = 256,          # (256,512) is the knee of the exported-gradient convergence: at (128,256) the
            gs_ntheta           = 512,          # traced-surface knots carry GS discretization jitter that shows up as jagged a/L* downstream
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
        and run MINUET with that (slightly interior, rounder) boundary. MINUET puts x = 1 at the
        last point of the file, so the trimmed state is RELABELLED exactly like MINUET's own
        interior boundary cut (Settings.boundary): rho -> rho/rho_cut, torfluxa -> torfluxa*rho_cut^2
        (without it, MINUET extrapolates the kinetics from rho_cut out to x = 1 -- negative n, T
        across a pedestal). merge_parameters() undoes the relabel and re-grids back to the frozen
        resolution, restoring the frozen equilibrium beyond the cut.

        Returns the file to hand to MINUET (the original when nothing folds).
        '''
        from minuet.geometry import mxh_surface_family, mxh_fold_statistic

        ig = mn.InputGacode.from_file(str(input_file))
        pr = ig.profiles
        sel = pr['rho'] > 0.0
        # Arbitrary MXH moment count: both stacks must span n = 0...n_max (sin0-2 are carried by delta/zeta)
        n_max = max([0] +
                    [int(k[len('shape_cos'):]) for k in pr if k.startswith('shape_cos') and k[len('shape_cos'):].isdigit()] +
                    [int(k[len('shape_sin'):]) for k in pr if k.startswith('shape_sin') and k[len('shape_sin'):].isdigit()])
        shp_c = np.column_stack([pr.get(f'shape_cos{n}', np.zeros_like(pr['rho']))[sel] for n in range(0, n_max + 1)])
        shp_s = np.column_stack([pr.get(f'shape_sin{n}', np.zeros_like(pr['rho']))[sel] for n in range(0, n_max + 1)])
        R, Z = mxh_surface_family(pr['rmin'][sel], pr['rmaj'][sel], pr['zmag'][sel],
                                  pr['kappa'][sel], pr['delta'][sel], pr['zeta'][sel],
                                  shp_c, shp_s, n_theta = 512)
        fold, _ = mxh_fold_statistic(R, Z, pr['rho'][sel])

        # Only OUTERMOST surfaces are cut (an interior fold is a deeper pathology to surface loudly): folded ones
        # (fold <= 0) and near-folded ones, whose fold collapses relative to the next surface inward. The latter
        # still self-intersect nowhere but sit next to the 1/|J2| pole of the FSA metrics (a surface with fold
        # 0.14 behind a 0.22 one read 9.19 vs 8.75 MA of Ampere Ip on a SPARC geqdsk state; a smooth Miller-like
        # edge decays only ~2-5% per surface, e.g. 0.25 -> 0.18 over the last 8 surfaces)
        bad_outer = 0
        for k in range(len(fold) - 1, 0, -1):
            if fold[k] <= 0 or fold[k-1] <= 0 or fold[k] < 0.8 * fold[k-1]:
                bad_outer += 1
            else:
                break
        if bad_outer == 0:
            return input_file

        rho_full = self.profiles_current.profiles['rho(-)']
        rho_new = rho_full[:len(rho_full) - bad_outer]
        print(f'\t- Incoming state has {bad_outer} self-intersecting or near-folded outermost MXH surface(s); '
              f'trimming to rho <= {rho_new[-1]:.4f} as MINUET boundary (boundary_surface_psin-style backoff)', typeMsg='w')
        if rho_new[-1] < 0.97:
            print(f'\t- The trim reaches rho = {rho_new[-1]:.4f}: the incoming edge shaping is badly represented '
                  f'(more MXH moments, or a boundary_surface_psin back-off, would help)', typeMsg='w')

        profiles_trimmed = copy.deepcopy(self.profiles_current)
        profiles_trimmed.changeResolution(rho_new = rho_new)
        self._trim_rho = float(rho_new[-1])   # merge_parameters undoes the relabel and restores the frozen band beyond this
        profiles_trimmed.profiles['rho(-)'] = profiles_trimmed.profiles['rho(-)'] / self._trim_rho
        profiles_trimmed.profiles['torfluxa(Wb/radian)'] = profiles_trimmed.profiles['torfluxa(Wb/radian)'] * self._trim_rho**2
        trimmed_file = self.folder / 'input.gacode_trimmed'
        profiles_trimmed.write_state(file = trimmed_file)
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
            # Also on a trimmed boundary: the thin cut band carries ~0.2% of Ip, whereas the current
            # MINUET would reconstruct from the edge metrics of the incoming state is off by several %
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
            # Near-uniform radial map: minuet's global default (2.0) packs cells to the edge
            # for diverted/X-point boundaries, but the beat solves a smooth fixed-boundary MXH
            # curve -- packing there starves the core grid (~2x coarser, equivalent to halving
            # gs_ns) and puts GS jitter into the exported a/L* gradients, while buying nothing
            # at the edge (kappa995/delta995 unchanged at the 1e-6 level, zero cost in time)
            gs_s_packing = 0.5,
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
        x_ohm, p_ohm, P_ohm = self._ohmic_power(m)
        print(f'\t- MINUET ohmic power (last 20% of the run): {P_ohm:.3f} MW, written as qohme at beat output')
        minuet_results = {
            'sawtooth_times': crashes,
            't_end': float(m.result.t[-1]),
            'Ip_MA_realized': float(m.result.ip_enc[-1, -1]) * 1e-6,
            'q0_initial': float(m.result.q0[0]),
            'q0_final': float(m.result.q0[-1]),
            'evolve_equilibrium': cfg['evolve_equilibrium'],
            'trim_rho': getattr(self, '_trim_rho', None),   # input.gacode_minuet is on the relabelled grid when set
            'ohmic_x': x_ohm, 'ohmic_MWm3': p_ohm, 'P_ohm_MW': P_ohm,
            'models': {
                'resistivity': cfg['resistivity_model'],
                'bootstrap': cfg['bootstrap_model'],
                'sawtooth': (cfg['sawtooth_model'] if cfg['sawteeth'] else None),
                'reconnection': (cfg['reconnection_model'] if cfg['sawteeth'] else None),
            },
        }
        np.save(self.folder / 'minuet_results.npy', minuet_results)

    @staticmethod
    def _ohmic_power(m, last_fraction = 0.2):
        '''
        Ohmic power density by the parallel-Ohm route, <E.B><J.B>/<B^2> with <E.B> = (V_loop/2pi) F <1/R^2>
        (MINUET's own verification routes; uses the SAVED V_loop, so it is independent of eta), averaged over
        the saved frames of the last `last_fraction` of the run so that a crash-instant V_loop transient does
        not set it. Returns x (MINUET CD cells), p [MW/m^3], and its volume integral [MW].
        '''
        res, g = m.result, m.geom_last
        x = res.x_c
        F, g1R2, B2, Vp = (g.interp(k, x) for k in ('F', 'g_1R2', 'B2_avg', 'Vprime'))
        k0 = int((1.0 - last_fraction) * (res.t.size - 1))
        p = np.mean([res.v_loop[k] / (2 * np.pi) * F * g1R2 * res.jb[k] / B2 for k in range(k0, res.t.size)], axis=0) * 1E-6
        P = float(np.trapezoid(p * Vp, x))
        return x, p, P

    # -----------------------------------------------------------------------------------------------------------------------
    # Finalize and merge
    # -----------------------------------------------------------------------------------------------------------------------

    def finalize(self, force_auxiliary_heating_at_output = None, **kwargs):

        # Refresh folder_output from self.folder only if the source still exists.
        # On a re-invocation after pruning (`maestro.prune_level` >= run-folder level) wiped self.folder,
        # folder_output already has the authoritative content from the prior run.
        if (self.folder / 'input.gacode_minuet').exists():

            # Remove previous output files
            for item in self.folder_output.glob('*'):
                if item.is_file():
                    item.unlink(missing_ok=True)

            # Persist sidecar + discharge object (copy when the run folder is kept; move when prune_level wipes it),
            # so plotting and _inform_save survive the cleanup loop
            self._persist(self.folder / 'minuet_results.npy', self.folder_output / 'minuet_results.npy')
            if (self.folder / 'run.minuet').exists():
                self._persist(self.folder / 'run.minuet', self.folder_output / 'run.minuet')

            # A re-invocation (new process) must know whether the export is on the trimmed, relabelled grid
            self._trim_rho = np.load(self.folder_output / 'minuet_results.npy', allow_pickle=True).item().get('trim_rho')

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

        # Undo the trimmed-state relabel (see _trim_folded_surfaces): back to the full-plasma rho
        if getattr(self, '_trim_rho', None) is not None:
            self.profiles_output.profiles['rho(-)'] = self.profiles_output.profiles['rho(-)'] * self._trim_rho
            self.profiles_output.profiles['torfluxa(Wb/radian)'] = self.profiles_output.profiles['torfluxa(Wb/radian)'] / self._trim_rho**2
            rho_lo = rho_lo * self._trim_rho if rho_lo is not None else None

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

        # Ohmic heating from the evolved current (the transp beat's POH analog); x is MINUET's label, which on a
        # trimmed boundary is rho/rho_cut
        results_file = self.folder_output / 'minuet_results.npy'
        d = np.load(results_file, allow_pickle=True).item() if results_file.exists() else {}
        if d.get('ohmic_MWm3') is not None:
            self.profiles_output.profiles['qohme(MW/m^3)'] = np.interp(rho, np.asarray(d['ohmic_x']) * (d.get('trim_rho') or 1.0), d['ohmic_MWm3'])

        # MINUET's vacuum field must be the frozen one: torfluxa (kept) was solved with it, so a mismatch
        # here would silently desync torfluxa from the re-pinned bcentr and ratchet over later beats
        RB_minuet = float(self.profiles_output.profiles['bcentr(T)'][0] * self.profiles_output.profiles['rcentr(m)'][0])
        RB_frozen = float(p_frozen.profiles['bcentr(T)'][0] * p_frozen.profiles['rcentr(m)'][0])
        if abs(RB_minuet / RB_frozen - 1) > 0.01:
            print(f'\t\t\t* MINUET vacuum R*Bt ({RB_minuet:.3f} T*m) differs from the frozen one ({RB_frozen:.3f} T*m) '
                  f'by {100*(RB_minuet/RB_frozen-1):+.1f}%: torfluxa/q are inconsistent with the frozen bcentr', typeMsg='w')

        # Re-insert engineering parameters (except shape)
        print('\t\t\t* Bringing Bt and Ip of frozen plasma state to new plasma state')
        for key in ['current(MA)', 'bcentr(T)']:
            self.profiles_output.profiles[key] = p_frozen.profiles[key]

        self.profiles_output.derive_quantities()

        # Without a transp beat upstream nobody has applied plasma.species, and the fusion/radiation/exchange
        # columns are the initializer's zeros (portals_soft, which evolves only qie, would read them as they are)
        if not self._transp_upstream():
            apply_species_composition(self.profiles_output, self.maestro_instance.maestro_namelist)
            self.profiles_output.recompute_targets(targets = ["qie", "qrad", "qfus"])
            print(f'\t\t\t* Targets recomputed on the composed plasma: Pfus = {self.profiles_output.derived["Pfus"]:.1f} MW, '
                  f'Prad = {self.profiles_output.derived["Prad"]:.1f} MW')

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

    def _transp_upstream(self):
        '''
        Did a transp beat run before this one in the chain? (then the ions and the fusion/radiation columns are
        TRANSP's, which the minuet beat passes through)
        '''
        counter = int(self.folder_beat.name.split('_')[-1])
        return any(isinstance(b, transp_beat) for k, b in self.maestro_instance.beats.items() if k < counter)

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
        if d.get('P_ohm_MW') is not None:
            md.append(f'- **Ohmic power (qohme at output):** {d["P_ohm_MW"]:.3f} MW')
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


def apply_species_composition(profiles, maestro_namelist):
    '''
    Rebuild the thermal ions of `profiles` from plasma.species (+ the ICRH minority), the job the transp beat does
    through its zlump/DTplasma namelist (TRANSPbeat.preprocess_prepare_transp). Chains without a transp beat need it
    here: the initializers only lay down a placeholder main ion plus one Z = 9 impurity at the target Zeff, which
    has no tritium (no fusion) and radiates as fluorine.

    Same recipe as the transp beat: fuel fraction fmain (split evenly among the fuel species), high-Z impurity
    at fhighZ with charge CShighZ_estimate, the minority [Z, A] at fmini when present, and ONE low-Z impurity whose
    (integer) Z makes Zeff and quasineutrality close (PLASMAtools.estimateLowZ; A = 2Z). The minority is laid down
    as a THERMAL ion (dilution only; there is no ICRF physics without TRANSP). ne, Te and Ti are untouched; every ion
    gets the main-ion Ti.
    '''
    import periodictable as pt

    species = maestro_namelist["plasma"]["species"]
    mix = species["mix"]
    heating_type = maestro_namelist["plasma"]["heating"]["type"]
    heating = maestro_namelist["plasma"]["heating"]["parameters"]
    Zmini, Amini = heating.get("minority", [1, 1])
    # Minority as in the transp beat: ICRH with power, or gaussian_sources with fmini > 0 (dilution accounting)
    fmini = heating.get("fmini", 0.0) or 0.0
    if not ((heating_type == 'ICRH' and heating.get('P_icrh', 0.0) > 0.0) or heating_type == 'gaussian_sources'):
        fmini = 0.0

    high = pt.elements.symbol(mix["highZ"])
    Zhigh = float(mix["CShighZ_estimate"])
    fhigh = float(mix["fhighZ"])
    fmain = float(mix["fmain"])

    Zlow, _ = PLASMAtools.estimateLowZ(fmain, species["Zeff"], Zmini, fmini, Zhigh, fhigh)
    flow = (1.0 - (fmain + Zmini * fmini + Zhigh * fhigh)) / Zlow    # quasineutrality closes exactly with the integer Z
    if flow < 0:
        raise ValueError(f'[MITIM] plasma.species cannot be closed: fmain={fmain}, fmini={fmini}, fhighZ={fhigh} leave no room for a low-Z impurity')
    low = pt.elements[int(Zlow)]

    fuel_mass = {'H': 1.0, 'D': 2.0, 'T': 3.0}
    ions = [(f, 1.0, fuel_mass[f], fmain / len(species["fuel"])) for f in species["fuel"]]
    if fmini > 0:
        ions.append((pt.elements[int(Zmini)].symbol if Zmini > 1 or Amini != 1 else 'H', float(Zmini), float(Amini), fmini))
    ions.append((high.symbol, Zhigh, float(high.mass), fhigh))
    ions.append((low.symbol, float(Zlow), 2.0 * Zlow, flow))

    P = profiles.profiles
    ne, ti = P["ne(10^19/m^3)"], P["ti(keV)"][:, 0]
    P["nion"] = np.array([f"{len(ions)}"])
    P["name"] = np.array([i[0] for i in ions])
    P["z"] = np.array([i[1] for i in ions])
    P["mass"] = np.array([i[2] for i in ions])
    P["type"] = np.array(["[therm]"] * len(ions))
    P["ni(10^19/m^3)"] = np.column_stack([i[3] * ne for i in ions])
    P["ti(keV)"] = np.column_stack([ti] * len(ions))
    if "vtor(m/s)" in P:
        P["vtor(m/s)"] = np.column_stack([P["vtor(m/s)"][:, 0]] * len(ions))

    profiles.readSpecies()
    profiles.derive_quantities(rederiveGeometry=False)

    zeff_vol = float(np.mean(profiles.profiles["z_eff(-)"]))
    print(f'\t\t\t* Ion composition from plasma.species: ' + ', '.join(f'{n} {f:.2e}' for n, _, _, f in ions) +
          f' (n_i/n_e); Zeff ~ {zeff_vol:.3f} (target {species["Zeff"]})')
