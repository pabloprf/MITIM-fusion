import inspect
import json
from pathlib import Path
import numpy as np
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.powertorch.physics_models.utils.cgyro_extra_points import ExtraPointHarvester
from mitim_modules.powertorch.physics_models.utils.cgyro_restart import RestartChain, RestartPlan
from mitim_modules.powertorch.physics_models.utils.gk_submission import GKSubmission
from mitim_modules.powertorch.physics_models.utils.per_iter_overrides import PerIterOverrides


# powerstate attribute <key>_turb / <key>_turb_stds <- (mean, std) attributes of one code output.
# "per species" entries are read at the turbulence-side impurity position.
_FLUX_SPEC = (
    ("QeGB", "Qe_mean",     "Qe_std",     False),
    ("QiGB", "Qi_mean",     "Qi_std",     False),
    ("GeGB", "Ge_mean",     "Ge_std",     False),
    # GZ: particle flux of the PORTALS trace impurity. MITIM writes input.cgyro species in
    # input.gacode ion order (electrons last), so the impurity position indexes Gi_all directly.
    ("GZGB", "Gi_all_mean", "Gi_all_std", True),
    # Mt: momentum flux summed over all species (same convention as TGLF's Mt = Me + sum(Mi)).
    # No sign flip: CGYRO's native sign is the physical GACODE convention (it receives
    # MACH/GAMMA_E/GAMMA_P unflipped, like NEO); TGLF's -SIGN_IT flip only undoes its
    # parity-mapped rotation inputs (tgyro_tglf_map.f90:197-199 / tgyro_flux.f90:199,208).
    ("MtGB", "Mt_mean",     "Mt_std",     False),
)

# Qie: electron turbulent energy exchange (the quantity TGLF passes as Se). Older CGYRO outputs
# carry no exchange moment (n_flux=3) and get zeros instead.
_EXCHANGE_SPEC = ("QieGB", "Se_mean", "Se_std", False)

# Passed by name at the call site, so they must never also come from the namelist `run:` block
_SINGLE_EXPLICIT_KWARGS = {"subfolder", "cold_start", "forceIfcold_start", "only_minimal_files", "job_name_suffix"}
_BATCHED_EXPLICIT_KWARGS = {"list_of_states", "base_subfolder", "cold_start", "forceIfcold_start",
                            "extra_name", "attempts_execution", "only_minimal_files", "job_name_suffix"}


def _forwardable_kwargs(target, method_name, explicit):
    '''
    The `transport.options.<code>.run` keys that `target.<method_name>` actually accepts. The MRO
    is walked so a wrapper forwarding **kwargs (CGYROtools.CGYRO.run -> mitim_simulation.run)
    contributes its own parameters and the search continues below it. Deriving the set from the
    callee is what keeps a namelist knob from being forwarded to a method that does not take it.
    '''
    accepted = set()
    for klass in target.__mro__:
        func = klass.__dict__.get(method_name)
        if func is None:
            continue
        params = inspect.signature(func).parameters
        accepted |= {n for n, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            break
    return accepted - {"self"} - explicit


# Namelist run keys the batched path forwards; kept as a module constant because
# tests/dev_tests/test_run_over_plasmas_kwargs.py checks it against the callee signature.
_RUN_OVER_PLASMAS_KEYS = _forwardable_kwargs(SIMtools.mitim_simulation, "run_over_plasmas", _BATCHED_EXPLICIT_KWARGS)

# An extra case is usable when it ran to MAX_TIME (EXIT) or was stopped past min_time (tag)
_extra_point_usable = ExtraPointHarvester.usable


def _check_exchange_moment(outputs, labels):
    '''
    Either every output carries the exchange moment (Se_mean) or none does (old CGYRO,
    n_flux=3). A mix means the output files of the odd radii are inconsistent (time
    rows vs flux records), which must not be silently passed on as Qie = 0.
    '''
    missing = [str(l) for l, o in zip(labels, outputs) if not hasattr(o, 'Se_mean')]
    if missing and len(missing) < len(outputs):
        raise RuntimeError(
            f"CGYRO exchange moment missing at {missing} but present elsewhere: "
            "out.cgyro.time and bin.cgyro.ky_flux disagree there (interrupted/continued run?)"
        )


def _averaging_records(outputs):
    '''Per-rho GKaverager.to_dict() (window, flag, provenance) of the read outputs, for the fluxes JSON.'''
    return [o.averaging.to_dict() for o in outputs] if all(hasattr(o, 'averaging') for o in outputs) else None


def _flux_arrays(outputs_per_plasma, mean_attr, std_attr, per_species, impurity_position):
    '''(N, nrho) mean and std arrays for one entry of _FLUX_SPEC.'''
    def value(o, attr):
        return getattr(o, attr)[impurity_position] if per_species else getattr(o, attr)

    return (np.array([[value(o, mean_attr) for o in outputs] for outputs in outputs_per_plasma]),
            np.array([[value(o, std_attr) for o in outputs] for outputs in outputs_per_plasma]))


class _GKRun:
    '''
    The configuration and live state of one gyrokinetic evaluation, assembled by `_gk_options`
    and threaded through `_gk_prepare` / `_gk_execute` / `_gk_collect_fluxes`. `batched` is the
    only switch between the single-plasma and the multi-plasma dispatch.
    '''

    batched = False
    list_of_states = None
    plasma_labels = None
    plasma_subfolder = None
    unpickled = False
    submission = None
    pickle_file = None


class gyrokinetic_model:

    # ------------------------------------------------------------------------------------------
    # One evaluation of a gyrokinetic backend (CGYRO, GX): namelist -> simulation object ->
    # results on local disk -> the turbulent fluxes power_transport reads.
    # ------------------------------------------------------------------------------------------

    def _evaluate_gyrokinetic_model(self, code='cgyro', gk_object=None):

        ctx = self._gk_options(code, gk_object)
        gk = self._gk_prepare(ctx, gk_object)
        outputs = self._gk_execute(ctx, gk)

        if ctx.run_type is SIMtools.RunType.PREP:
            return self._run_externally_and_wait(ctx)

        if outputs is not None:
            self._gk_collect_fluxes(ctx, outputs)
            if ctx.extra_points:
                self._extra_points(ctx).harvest()
            ctx.submission.cleanup(
                remove_scratch=(ctx.run_type is SIMtools.RunType.SUBMIT) and ctx.remove_scratch_after_fetch)

        return gk

    # ------------------------------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------------------------------

    def _gk_options(self, code, gk_class, list_of_states=None):
        '''
        Namelist -> everything the rest of the evaluation needs: the kwargs the backend's runner
        accepts, the re-attach and retry controls, and the resolved warm-start plan.
        '''
        simulation_options = self.transport_evaluator_options[code]
        run_options = simulation_options["run"]
        batched = list_of_states is not None

        ctx = _GKRun()
        ctx.code = code
        ctx.batched = batched
        ctx.list_of_states = list_of_states
        ctx.simulation_options = simulation_options
        ctx.read_options = simulation_options["read"]
        ctx.cold_start = self.cold_start
        ctx.keep_gk_files = simulation_options.get("keep_files", 'all')
        ctx.evaluation_number = getattr(self, "evaluation_number", 0)
        # Every on-disk artifact carries the instance name, so a named multi-fidelity config
        # ('cgyro1') never collides with plain 'cgyro'.
        ctx.subfolder_name = f"base_{code}"
        ctx.plasma_subfolder = f"{ctx.subfolder_name}_plasma0" if batched else None
        ctx.rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item()
                             for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        ctx.run_type = SIMtools.RunType.parse(run_options.get("run_type", "normal"))

        if batched and ctx.run_type is SIMtools.RunType.PREP:
            raise NotImplementedError(
                "run_type='prep' (interactive external CGYRO run) is not supported in "
                "batched mode. Use single-plasma evaluation or run_type='normal'/'submit'.")

        # Read through .get() so the shared namelist dict is not mutated between PORTALS iterations
        ctx.every_n_minutes = run_options.get("every_n_minutes", 10)
        ctx.remove_scratch_after_fetch = run_options.get("remove_scratch_after_fetch", False)
        ctx.check_existing_runs = run_options.get("check_existing_runs", False)
        if ctx.check_existing_runs and ctx.run_type is not SIMtools.RunType.SUBMIT:
            print(f"\t- check_existing_runs=True has no effect when run_type='{ctx.run_type.value}' "
                  "(only 'submit' supports re-attach); ignoring", typeMsg='w')
            ctx.check_existing_runs = False

        # Forwarded to the mitim_job and consumed inside connect_ssh(), so submit, check and fetch
        # share one retry policy. attempts=None retries forever, which is what a long
        # PORTALS-CGYRO run needs to ride out overnight VPN flaps.
        ctx.connection_retry_settings = {
            "wait_seconds": run_options.get("ssh_retry_wait_seconds", 5),
            "attempts":     run_options.get("ssh_retry_attempts", 3),
        }
        # Per-rho stall rescue (CGYROtools._cgyro_handle_stalled_tasks)
        ctx.auto_resubmit_settings = {
            "enabled":                    run_options.get("auto_resubmit_enabled", True),
            "stall_init_kill_seconds":    run_options.get("stall_init_kill_seconds", 1800),
            "stall_running_kill_seconds": run_options.get("stall_running_kill_seconds", 1800),
            "max_resubmits_per_rho":      run_options.get("max_resubmits_per_rho", 1),
        }

        forwardable = _forwardable_kwargs(
            gk_class,
            "run_over_plasmas" if batched else "run",
            _BATCHED_EXPLICIT_KWARGS if batched else _SINGLE_EXPLICIT_KWARGS)
        ctx.run_kwargs = {k: v for k, v in run_options.items() if k in forwardable}
        ctx.extra_points = (ctx.run_kwargs.get("load_balance") or {}).get("strategy") == "extra_points"

        self._gk_warm_start(ctx, gk_class, run_options)

        # Per-iteration overrides. extraOptions is coerced to {} because SIMtools.run does not
        # accept None; allocation may stay None so that run() still sizes it from _default_allocation.
        ctx.run_kwargs["extraOptions"] = PerIterOverrides.from_run_options(run_options, "extraOptions").merge(
            ctx.run_kwargs.get("extraOptions"), ctx.evaluation_number) or {}
        ctx.run_kwargs["allocation"] = PerIterOverrides.from_run_options(run_options, "allocation").merge(
            ctx.run_kwargs.get("allocation"), ctx.evaluation_number)

        return ctx

    def _gk_warm_start(self, ctx, gk_class, run_options):
        '''
        Resolve the restart staging into ctx.run_kwargs["additional_files_to_send"].
        `restart_from_folder` is staged on every path; the automatic chain is SKIPPED on a
        re-attach, because re-running it would re-pick parents against the current BO state and
        misrepresent what was actually staged at the original submit. That original pick travels
        in the submission metadata and is put back on disk by load_submission_state.
        '''
        ctx.chain = RestartChain(run_options, ctx.evaluation_number, self.folder, ctx.rho_locations,
                                 base_subfolder=ctx.subfolder_name, plasma_subfolder=ctx.plasma_subfolder)
        ctx.turb_target_GB = RestartChain.turbulent_target_GB(self, plasma_index=0)

        staged = ctx.chain.stage_explicit_folder(ctx.run_kwargs.get("additional_files_to_send"))
        if staged is not None:
            ctx.run_kwargs["additional_files_to_send"] = staged

        metadata_name = getattr(gk_class, "_submission_metadata_filename", None)
        metadata = (self.folder / ctx.subfolder_name / metadata_name) if metadata_name else None
        if ctx.check_existing_runs and metadata is not None and metadata.is_file():
            print(f"\t- [CGYRO restart] Re-attach detected ({metadata.name} present); "
                  "preserving original parent-pick (skipping resolver)", typeMsg='i')
            ctx.plan = RestartPlan(files_per_rho=ctx.run_kwargs.get("additional_files_to_send"))
        else:
            ctx.plan = ctx.chain.resolve(ctx.run_kwargs.get("additional_files_to_send"), ctx.turb_target_GB)

        if ctx.plan.files_per_rho is not None:
            ctx.run_kwargs["additional_files_to_send"] = ctx.plan.files_per_rho

    def _gk_prepare(self, ctx, gk_class):
        '''
        The simulation object, either restored from the pickle a previous run left or freshly
        constructed and prepped. Retry settings and the harvest recorder are attached either way.
        '''
        ctx.pickle_file = self.folder / ctx.subfolder_name / ("gk_object_batched.pkl" if ctx.batched else "gk_object.pkl")

        gk = None
        if ctx.keep_gk_files in ['pickle']:
            try:
                gk = SIMtools.restore_class_pickle(ctx.pickle_file)
                ctx.unpickled = True
                if ctx.batched:
                    ctx.plasma_labels = {p: f"{ctx.subfolder_name}_plasma{p}" for p in range(len(ctx.list_of_states))}
                print(f"\t- Pickle file with {'batched ' if ctx.batched else ''}GK object information "
                      "has been restored successfully", typeMsg='i')
            except Exception as e:
                ctx.unpickled = False
                print('\t- Pickle file could not be read, with error:', typeMsg='w')
                print(e)

        if not ctx.unpickled:
            gk = gk_class(rhos=ctx.rho_locations)
            if ctx.batched:
                # run_over_plasmas / _prepare_plasmas_state call _run_prepare directly (not run()),
                # so preprocess_options has to be on the object before they do
                gk._preprocess_options = ctx.simulation_options["run"].get("preprocess_options")
            else:
                # Side-aware: a turbulence backend consumes the turb-side post-processed profiles
                gk.prep(self._profiles_transport_for("turb"), self.folder)
                if ctx.extra_points:
                    gk.extra_point_builder = self._extra_points(ctx).builder()

        # On the object whether it was unpickled or not: connect_ssh() reads the retry config from
        # the mitim_job, which may already exist on a restored instance.
        gk.connection_retry_settings = ctx.connection_retry_settings
        gk.auto_resubmit_settings = ctx.auto_resubmit_settings
        self._harvest_attach(gk)
        # Embedded in the submission metadata by _write_submission_metadata, so that a later
        # re-attach can put restart_sources.json back on disk.
        gk._restart_sources_payload = ctx.plan.payload
        if getattr(gk, "simulation_job", None) is not None:
            gk.simulation_job.connection_retry_settings = ctx.connection_retry_settings

        return gk

    def _gk_execute(self, ctx, gk):
        '''
        Results onto local disk: re-attach to the job a previous process submitted, or submit a
        fresh one; poll and fetch when the submission is detached; read. Returns one list of
        per-rho output objects per plasma, or None when this run_type leaves nothing to read.
        '''
        ctx.submission = GKSubmission(
            gk, self.folder, ctx.subfolder_name,
            every_n_minutes=ctx.every_n_minutes,
            enabled=ctx.check_existing_runs,
            connection_retry_settings=ctx.connection_retry_settings,
            label=f"batched {ctx.code.upper()}" if ctx.batched else ctx.code.upper(),
            submit_name="run_over_plasmas()" if ctx.batched else "run()",
            reader_name="read_plasma()" if ctx.batched else "read()",
            organize_label="per-(plasma,rho) folders" if ctx.batched else "per-rho folders",
        )

        if not ctx.unpickled:
            outcome = ctx.submission.try_reattach(
                on_fresh_fallback=lambda: ctx.chain.rerun_for_fresh_submission(gk, ctx.run_kwargs, ctx.turb_target_GB),
                before_load=lambda: self._gk_restore_reader_state(ctx, gk),
            )
            if not outcome.reattached:
                self._gk_submit(ctx, gk)
                # run() -> _run_prepare recreated <subfolder_name>/ through askNewFolder, wiping the
                # resolver's restart_sources.json. Put it back so the warm-start parent map is on
                # disk for the trace plotter, including during polling.
                RestartPlan.restore_json(self.folder, ctx.subfolder_name,
                                         getattr(gk, "_restart_sources_payload", None))

        # 'send' stages the inputs without submitting and 'prep' only builds them: neither
        # leaves anything on disk to read
        if ctx.run_type in (SIMtools.RunType.SEND, SIMtools.RunType.PREP):
            return None

        if not ctx.unpickled:
            if ctx.run_type is SIMtools.RunType.SUBMIT:
                ctx.submission.poll_and_fetch(outcome.reattached)

            self._gk_read(ctx, gk)
            self._gk_save_pickle(ctx, gk, outcome.reattached)

        labels = list(ctx.plasma_labels.values()) if ctx.batched else [ctx.subfolder_name]
        return [gk.results[label]['output'] for label in labels]

    def _gk_collect_fluxes(self, ctx, outputs_per_plasma, pass_info=True):
        '''
        The turbulent fluxes power_transport expects: QeGB_turb, QiGB_turb, GeGB_turb, GZGB_turb,
        MtGB_turb, QieGB_turb and their _stds, as (nrho,) arrays for one plasma and (N, nrho) for
        a batch, plus the per-rho averaging record that goes into fluxes_turb.json.
        '''
        flat = [o for outputs in outputs_per_plasma for o in outputs]
        _check_exchange_moment(flat, [
            f"plasma {p} rho={rho:.4f}" if ctx.batched else f"{rho:.4f}"
            for p in range(len(outputs_per_plasma)) for rho in ctx.rho_locations])

        if not pass_info:
            return

        has_exchange = hasattr(flat[0], 'Se_mean')
        spec = (_FLUX_SPEC + (_EXCHANGE_SPEC,)) if has_exchange else _FLUX_SPEC
        impurity_position = self._impurity_position_transport_for("turb")

        for key, mean_attr, std_attr, per_species in spec:
            mean, std = _flux_arrays(outputs_per_plasma, mean_attr, std_attr, per_species, impurity_position)
            setattr(self, f"{key}_turb", mean if ctx.batched else mean[0])
            setattr(self, f"{key}_turb_stds", std if ctx.batched else std[0])

        if not has_exchange:
            print("\t- CGYRO output carries no turbulent-exchange moment (n_flux=3); passing QieGB_turb = 0", typeMsg='w')
            self.QieGB_turb = self.QeGB_turb * 0.0
            self.QieGB_turb_stds = self.QeGB_turb_stds * 0.0

        averaging = [_averaging_records(outputs) for outputs in outputs_per_plasma]
        self.averaging_info_turb = averaging if ctx.batched else averaging[0]

    # ------------------------------------------------------------------------------------------
    # Pieces of a stage
    # ------------------------------------------------------------------------------------------

    def _extra_points(self, ctx):
        return ExtraPointHarvester(self, ctx.code, ctx.rho_locations, ctx.run_kwargs, ctx.read_options)

    def _gk_submit(self, ctx, gk):
        job_name_suffix = f"_ev{ctx.evaluation_number}"
        if ctx.batched:
            gk.prep(ctx.list_of_states[0], self.folder, cold_start=ctx.cold_start)
            ctx.plasma_labels = gk.run_over_plasmas(
                ctx.list_of_states,
                base_subfolder=ctx.subfolder_name,
                cold_start=ctx.cold_start,
                forceIfcold_start=True,
                extra_name=self.name,
                attempts_execution=2,
                only_minimal_files=ctx.keep_gk_files in ['none', 'pickle'],
                job_name_suffix=job_name_suffix,
                **ctx.run_kwargs,
            )
        else:
            gk.run(
                ctx.subfolder_name,
                cold_start=ctx.cold_start,
                forceIfcold_start=True,
                only_minimal_files=ctx.keep_gk_files in ['none', 'pickle'],
                job_name_suffix=job_name_suffix,
                **ctx.run_kwargs,
            )

    def _gk_restore_reader_state(self, ctx, gk):
        '''
        What the reader needs but a re-attach never staged. _run_prepare normally sets
        FolderSimLast; the batched reader additionally needs every per-plasma folder rebuilt
        (profiles, inputs_files, normalizations) without contacting the cluster.
        '''
        if not ctx.batched:
            gk.FolderSimLast = self.folder / ctx.subfolder_name
            return

        print(f"\t- Rebuilding per-plasma state for {len(ctx.list_of_states)} plasma(s) "
              "(profiles, inputs, normalizations) without re-submitting...", typeMsg='i')
        print("")
        gk.prep(ctx.list_of_states[0], self.folder, cold_start=False)
        # forceIfcold_start=True so askNewFolder() does not prompt mid-re-attach; the inputs are
        # deterministic from namelist+powerstate and the slurm job already has its own copy
        _, _, ctx.plasma_labels = gk._prepare_plasmas_state(
            ctx.list_of_states,
            base_subfolder=ctx.subfolder_name,
            cold_start=False,
            forceIfcold_start=True,
            code_settings=ctx.run_kwargs.get("code_settings"),
            extraOptions=ctx.run_kwargs.get("extraOptions"),
            multipliers=ctx.run_kwargs.get("multipliers"),
            minimum_delta_abs=ctx.run_kwargs.get("minimum_delta_abs"),
            only_minimal_files=ctx.keep_gk_files in ['none', 'pickle'],
            launchSlurm=ctx.run_kwargs.get("launchSlurm", True),
            allocation=ctx.run_kwargs.get("allocation"),
            additional_files_to_send=ctx.run_kwargs.get("additional_files_to_send"),
            ApplyCorrections=ctx.run_kwargs.get("ApplyCorrections", True),
            Quasineutral=ctx.run_kwargs.get("Quasineutral", False),
            announce=False,
        )

    def _gk_read(self, ctx, gk):
        # minimal=True: a pickle carrying the full output would be extra heavy
        if ctx.batched:
            for p, label in ctx.plasma_labels.items():
                gk.read_plasma(p, label=label, minimal=True, **ctx.read_options)
        else:
            gk.read(label=ctx.subfolder_name, minimal=True, **ctx.read_options)

    def _gk_save_pickle(self, ctx, gk, reattached):
        '''
        keep_files 'pickle': the object first, the heavy per-rho files only once the pickle is on
        disk, so a failed save never leaves the evaluation with no data at all.
        '''
        if ctx.keep_gk_files not in ['pickle']:
            return
        if reattached:
            # inputs_files / normalization state were reconstructed through prep only
            print("\t- pickle requested but run was re-attached; skipping pickle save (inputs not fully available)", typeMsg='i')
            return

        ctx.pickle_file.parent.mkdir(parents=True, exist_ok=True)
        gk.save_pickle(ctx.pickle_file)
        if not ctx.pickle_file.exists():
            print(f"\t- save_pickle did not produce {ctx.pickle_file}; keeping raw CGYRO files", typeMsg='w')
            return

        labels = list(ctx.plasma_labels.values()) if ctx.batched else [ctx.subfolder_name]
        for label in labels:
            for file in gk.output_files_simulation["complete"]:
                for rho in gk.rhos:
                    (self.folder / label / f"{file}_{rho:.4f}").unlink(missing_ok=True)

    def _run_externally_and_wait(self, ctx):
        '''
        run_type 'prep': MITIM builds the inputs, the user runs the code elsewhere and drops
        fluxes_turb.json in the evaluation folder. Loop until that file is there and its
        gradients agree with the powerstate.
        '''
        # the JSON comes from the user, so do not write one from our own variables
        self._write_json_from_variables_turb = False
        self._profiles_transport_for("turb").write_state(self.folder / ctx.subfolder_name / "input.gacode")

        self.pre_checks()

        file_path = self.folder / 'fluxes_turb.json'
        attempts = 0
        all_good = self.post_checks() if file_path.exists() else False
        while (file_path.exists() is False) or (not all_good):
            if attempts > 0:
                print(f"\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                print(f" MITIM could not find the file... looping back", typeMsg='i')
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
            logic_to_wait(self.folder, self.folder / ctx.subfolder_name)
            attempts += 1

            if file_path.exists():
                all_good = self.post_checks()

    # ------------------------------------------------------------------------------------------
    # Interactive helpers of run_type 'prep'
    # ------------------------------------------------------------------------------------------

    def pre_checks(self):
        '''Gradients and the flux the turbulence has to carry (target - neoclassical), per radius.'''
        plasma = self.powerstate.plasma

        txt = "\nFluxes to be matched by turbulence ( Target - Neoclassical ):"

        for var, varn in zip(
            ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
            ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
        ):
            txt += f"\n{var}   = "
            for j in range(plasma["rho"].shape[1] - 1):
                txt += f"{plasma[varn][0,j+1]:.6f}   "

        for var, varn in zip(
            ["Qe (GB)", "Qi (GB)", "Ge (GB)", "GZ (GB)", "Mt (GB)"],
            ["QeGB", "QiGB", "GeGB", "GZGB", "MtGB"],
        ):
            # neoclassical_model None leaves no *GB_neoc: the target itself is what turbulence carries
            neoc = self.__dict__.get(f'{varn}_neoc')
            txt += f"\n{var}  = "
            for j in range(plasma["rho"].shape[1] - 1):
                txt += f"{plasma[varn][0,j+1] - (neoc[j] if neoc is not None else 0.0):.4e}   "

        print(txt)

    def post_checks(self, rtol=1e-3):
        '''Compare the additional_info of a user-supplied fluxes_turb.json against the powerstate.'''
        with open(self.folder / 'fluxes_turb.json', 'r') as f:
            json_dict = json.load(f)

        additional_info_from_json = json_dict.get('additional_info', {})

        all_good = True

        if len(additional_info_from_json) == 0:
            print(f"\t- No additional info found in fluxes_turb.json to be compared with", typeMsg='i')

        else:
            print(f"\t- Additional info found in fluxes_turb.json:", typeMsg='i')
            for k, v in additional_info_from_json.items():
                vP = self.powerstate.plasma[k].cpu().numpy()[0,1:]

                crit = not np.allclose(v, vP, rtol=rtol)

                print(f"\t   {k} from JSON      : {[round(float(i),4) for i in v]}", typeMsg='' if not crit else 'i')
                print(f"\t   {k} from POWERSTATE: {[round(float(i),4) for i in vP]}", typeMsg='' if not crit else 'i')

                if crit:
                    all_good = print(f"{k} does not match with a relative tolerance of {rtol*100.0:.3f}%, max rel difference: {np.max(np.abs(v - vP) / np.maximum(np.abs(v), np.abs(vP)))*100.0:.3f}%", typeMsg='q')

        return all_good

    # ------------------------------------------------------------------------------------------

    def _stable_correction(self, simulation_options_all):

        simulation_options = simulation_options_all[getattr(self, "_active_turb_options_key", None) or "cgyro"]

        Qi_stable_criterion = simulation_options["Qi_stable_criterion"]
        # Setting Qi_stable_criterion to null/None in the namelist disables the check entirely.
        if Qi_stable_criterion is None:
            print("\n- Qi_stable_criterion is null; skipping CGYRO stable-flux check", typeMsg='i')
            return

        print(f"\n- Checking if any radius has Qi below the stability criterion to apply a stable correction if needed...", typeMsg='i')

        Qi_stable_percent_error = simulation_options["Qi_stable_percent_error"]

        # Check if Qi in MW/m2 < Qi_stable_criterion
        QiMWm2 = self.powerstate.plasma['QiMWm2_tr_turb']
        QiMWm2_stds = self.powerstate.plasma['QiMWm2_tr_turb_stds']

        # Handle both single-plasma (1D: nrho) and batched (2D: N x nrho) arrays
        is_batched = np.ndim(QiMWm2) >= 2

        if is_batched:
            N = QiMWm2.shape[0]
            nrho = QiMWm2.shape[1]
            for b in range(N):
                QiMWm2_target_b = self.powerstate.plasma['QiMWm2'][b, 1:].cpu().numpy()
                for i in range(nrho):
                    if QiMWm2[b, i] < Qi_stable_criterion:
                        print(f"\n\t- Qi considered stable at plasma #{b}, radius #{i}: {QiMWm2[b, i]:.2e} MW/m^2 in CGYRO simulation < {Qi_stable_criterion:.2e} MW/m^2 criterion (see namelist)", typeMsg='i')
                        Qi_std = QiMWm2_target_b[i] * Qi_stable_percent_error / 100
                        print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target value as standard deviation: sigma = {Qi_std:.2e} MW/m^2 instead of {QiMWm2_stds[b, i]:.2e} MW/m^2", typeMsg='i')
                        QiMWm2_stds[b, i] = Qi_std
        else:
            QiMWm2_target = self.powerstate.plasma['QiMWm2'][0, 1:].cpu().numpy()
            for i in range(len(QiMWm2)):
                if QiMWm2[i] < Qi_stable_criterion:
                    print(f"\n\t- Qi considered stable at radius #{i}: {QiMWm2[i]:.2e} MW/m^2 in CGYRO simulation < {Qi_stable_criterion:.2e} MW/m^2 criterion (see namelist)", typeMsg='i')
                    Qi_std = QiMWm2_target[i] * Qi_stable_percent_error / 100
                    print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target value as standard deviation: sigma = {Qi_std:.2e} MW/m^2 instead of {QiMWm2_stds[i]:.2e} MW/m^2", typeMsg='i')
                    QiMWm2_stds[i] = Qi_std


class cgyro_model(gyrokinetic_model):

    def evaluate_turbulence(self):

        # Active cgyro options block (defaults to "cgyro"; differs in named multi-fidelity
        # instances like "cgyro1"). The base-TGLF diagnostic intentionally stays on
        # options["tglf"] — see namelist.portals.yaml notes on the multi-fidelity scoping.
        cgyro_key = getattr(self, "_active_turb_options_key", None) or "cgyro"
        self._run_base_tglf(cgyro_key)

        self._evaluate_gyrokinetic_model(code=cgyro_key, gk_object=CGYROtools.CGYRO)

    def evaluate_turbulence_batched(self, list_of_states, pass_info=True):
        '''
        Fan a list of profile states through run_over_plasmas so that every (plasma, rho) work
        unit is dispatched concurrently by the existing FARMINGtools pipeline. Reached from
        portals_transport_model.evaluate_turbulence_batched when powerstate.batch_size > 1.
        '''
        cgyro_key = getattr(self, "_active_turb_options_key", None) or "cgyro"
        self._run_base_tglf(cgyro_key, list_of_states=list_of_states)

        ctx = self._gk_options(cgyro_key, CGYROtools.CGYRO, list_of_states=list_of_states)
        gk = self._gk_prepare(ctx, CGYROtools.CGYRO)
        outputs = self._gk_execute(ctx, gk)

        if outputs is not None:
            self._gk_collect_fluxes(ctx, outputs, pass_info=pass_info)
            ctx.submission.cleanup(
                remove_scratch=(ctx.run_type is SIMtools.RunType.SUBMIT) and ctx.remove_scratch_after_fetch)

        return gk

    def _run_base_tglf(self, cgyro_key, list_of_states=None):
        '''Base TGLF alongside CGYRO, to keep track of discrepancies. pass_info=False so it
        computes its fluxes without overwriting the CGYRO ones.'''
        if not self.transport_evaluator_options[cgyro_key].get("run_base_tglf", True):
            return

        from mitim_modules.powertorch.physics_models.transport_tglf import tglf_model
        self.transport_evaluator_options["tglf"]["use_scan_trick_for_stds"] = None
        if list_of_states is None:
            self._evaluate_tglf(pass_info=False, options_key="tglf")
        else:
            tglf_model._evaluate_tglf_batched(self, list_of_states, pass_info=False, options_key="tglf")


def logic_to_wait(folder, subfolder):
    print(f"\n**** Simulation inputs prepared. Please, run it from the simulation setup in folder:\n", typeMsg='i')
    print(f"\t {subfolder}\n", typeMsg='i')
    print(f"**** When finished, the fluxes_turb.json file should be placed in:\n", typeMsg='i')
    print(f"\t {folder}/fluxes_turb.json\n", typeMsg='i')
    while not print(f"**** When you have done that, please say yes", typeMsg='q'):
        pass


def write_json_CGYRO(roa, fluxes_mean, fluxes_stds, additional_info = None, file = 'fluxes_turb.json'):
    '''
    *********************
    Helper to write JSON
    *********************
        roa
            Must be an array: [0.25, 0.35, ...]
        fluxes_mean
            Must be a dictionary with the fields and arrays:
                'QeMWm2': [0.1, 0.2, ...],
                'QiMWm2': ...,
                'Ge1E20m2': ...,
                'GZ1E20m2': ...,
                'MtJm2': ...,
                'QieMWm3': ..
            or, alternatively (or complementary), in GB units:
                'QeGB': [0.1, 0.2, ...],
                'QiGB': ...,
                'GeGB': ...,
                'GZGB': ...,
                'MtGB': ...,
                'QieGB': ..
        fluxes_stds
            Exact same structure as fluxes_mean
        additional_info
            A dictionary with any additional information to include in the JSON and compare to powerstate,
            for example (and recommended):
                'aLte': [0.2, 0.5, ...],
                'aLti': [0.3, 0.6, ...],
                'aLne': [0.3, 0.6, ...],
                'Qgb': [0.4, 0.7, ...],
                'rho': [0.2, 0.5, ...],
    '''

    if additional_info is None:
        additional_info = {}

    with open(file, 'w') as f:

        additional_info_extended = additional_info | {'roa': roa.tolist() if not isinstance(roa, list) else roa}

        json_dict = {
            'fluxes_mean': fluxes_mean,
            'fluxes_stds': fluxes_stds,
            'additional_info': additional_info_extended
        }

        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.generic,)):
                return obj.item()
            else:
                return obj

        json.dump(convert_numpy(json_dict), f, indent=4)
