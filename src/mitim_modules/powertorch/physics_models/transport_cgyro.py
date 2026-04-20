import json
from pathlib import Path
import numpy as np
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


def _resolve_cgyro_restart_folder(run_options, rho_locations, existing_additional_files_to_send=None):
    '''
    Translate the namelist-level `restart_from_folder` option into per-rho
    `additional_files_to_send` entries.

    `restart_from_folder` is expected to be a directory containing the
    per-radius binary restart file CGYRO produces (and MITIM retrieves):
    `bin.cgyro.restart_<rho:.4f>`. For every rho in `rho_locations`, the
    binary is staged as `bin.cgyro.restart` inside the rho subfolder via
    SIMtools' rename-on-copy mechanism.

    CGYRO auto-detects the restart at startup (cgyro_init_h.f90):
      - tag present + bin present  -> restart_flag=1 (TRUE restart;
        continues from t_current stamped in the tag; requires new
        MAX_TIME > t_current AND the full out.cgyro.* time-series bundle
        on disk to rewind via io_control=3 — cgyro_init_kernel.F90:82).
      - tag missing, bin present   -> restart_flag=2 (warm start; uses
        restart data as initial condition, t resets to 0, out.cgyro.*
        files are overwritten fresh via io_control=1).

    For PORTALS this helper deliberately stages ONLY the binary (no tag):
    each iteration changes input.cgyro parameters, so *continuing* in
    time from a prior run with different physics is not physically
    meaningful; warm-start is the correct semantics, and it also avoids
    the io_control=3 rewind crash when the full output bundle isn't
    staged alongside the restart file.

    Raises if the folder doesn't exist or if any rho is missing its
    `bin.cgyro.restart_<rho:.4f>` file.
    '''

    restart_folder = run_options.get("restart_from_folder")
    if restart_folder in (None, ""):
        return existing_additional_files_to_send

    restart_folder = Path(restart_folder).expanduser()
    if not restart_folder.is_dir():
        raise FileNotFoundError(
            f"[MITIM] CGYRO restart_from_folder does not exist or is not a directory: {restart_folder}"
        )

    print(f"\n- [CGYRO restart] Staging per-radius restart files from:\n\t{restart_folder}", typeMsg='i')

    resolved = dict(existing_additional_files_to_send) if existing_additional_files_to_send else {}
    missing_bin = []
    for rho in rho_locations:
        bin_file = restart_folder / f"bin.cgyro.restart_{rho:.4f}"
        if not bin_file.is_file():
            missing_bin.append(bin_file.name)
            continue
        print(f"\t  rho={rho:.4f}: {bin_file.name} -> bin.cgyro.restart (warm start)", typeMsg='i')
        resolved.setdefault(float(rho), []).append((bin_file, "bin.cgyro.restart"))

    if missing_bin:
        raise FileNotFoundError(
            "[MITIM] CGYRO restart_from_folder is missing per-rho binary restart files: "
            f"{missing_bin}. Expected one file per predicted radius, named "
            f"bin.cgyro.restart_<rho:.4f>, in {restart_folder}."
        )

    return resolved


def _resolve_cgyro_restart_chain(
    run_options,
    evaluation_number,
    folder,
    rho_locations,
    existing_additional_files_to_send=None,
    plasma_subfolder=None,
):
    '''
    Automatic-restart companion to `_resolve_cgyro_restart_folder`. Driven
    by the namelist-level `restart_from_cases` option:

      None / null  -> no-op (no automatic restart)
      "first"      -> every iteration N >= 1 restarts from iteration 0
      "all"        -> every iteration N >= 1 restarts from iteration N-1
                      (chained; each iter warm-starts from the previous)

    Source folder for iteration N is derived as:
      <root>/Execution/Evaluation.{source_iter}/transport_simulation_folder/base_cgyro
        (+ /base_cgyro_plasma0 in batched mode)
    or, during the simple-relax initializer:
      <root>/Initialization/initialization_simple_relax/portals_sr_ev_{source_iter}/
        transport_simulation_folder/base_cgyro

    where source_iter is 0 for "first" and (N-1) for "all".

    The binary restart file is named bin.cgyro.restart_<rho:.4f>, matching
    what CGYRO writes and MITIM retrieves after a PORTALS run. It is
    staged into the rho subfolder renamed to "bin.cgyro.restart" via the
    (src, dst) tuple mechanism in SIMtools. See
    `_resolve_cgyro_restart_folder` for the rationale on why only the
    binary is staged (warm start, restart_flag=2) and not the companion
    out.cgyro.tag file.

    Missing files are FATAL: if restart_from_cases was requested and either
    the source base_cgyro folder is absent, or the per-rho restart binary
    is absent for any rho, a FileNotFoundError is raised. A silent partial
    restart produced one-rho-cold/others-warm ensembles that were almost
    impossible to diagnose downstream (and broke the "one key per rho"
    invariant the stage-in consumer relies on). If you want to proceed
    without restart, clear restart_from_cases in the namelist.

    Backward-compat: legacy `restart_from_first: true` still works and is
    mapped to `restart_from_cases: "first"` (with a one-line notice).

    Returns the merged additional_files_to_send dict, or the existing one
    unchanged when the mode is None/empty, restart_from_folder takes
    precedence, or this is iteration 0 (no prior iteration to restart
    from — not an error for N=0).
    '''

    # Resolve the mode, with backward-compat for the retired
    # `restart_from_first: true` flag.
    mode = run_options.get("restart_from_cases")
    if mode in (None, "", "null"):
        # Check legacy flag.
        if run_options.get("restart_from_first", False):
            print(
                "\t- [CGYRO restart] `restart_from_first: true` is deprecated; "
                "mapping to `restart_from_cases: \"first\"`. Please update your namelist.",
                typeMsg='w',
            )
            mode = "first"
        else:
            return existing_additional_files_to_send

    mode_lower = str(mode).lower()
    if mode_lower not in ("first", "all"):
        print(
            f"\t- [CGYRO restart] Unknown restart_from_cases={mode!r}; expected one of "
            "null / \"first\" / \"all\". Ignoring.",
            typeMsg='w',
        )
        return existing_additional_files_to_send

    # restart_from_folder explicit beats automatic chain.
    if run_options.get("restart_from_folder") not in (None, ""):
        print(
            f"\t- [CGYRO restart] restart_from_folder is set; restart_from_cases={mode_lower!r} ignored.",
            typeMsg='w',
        )
        return existing_additional_files_to_send

    # Detect context: simple-relax initialization places per-iteration folders
    # under <root>/Initialization/initialization_simple_relax/portals_sr_ev_{N}/
    # transport_simulation_folder. The BO loop uses <root>/Execution/Evaluation.{N}/
    # transport_simulation_folder. Both share the pattern folder.parent.parent /
    # <sibling name> / folder.name; only the sibling's stem differs.
    in_simple_relax = "initialization_simple_relax" in folder.parts
    context_label = "portals_sr_ev" if in_simple_relax else "Evaluation"

    if evaluation_number == 0:
        print(
            f"\n- [CGYRO restart_from_cases={mode_lower!r}] This is {context_label}.0 — no prior iteration to restart from.\n"
            "\t  REMINDER: for subsequent iterations to resume from this one, RESTART_STEP\n"
            "\t  (and any related CGYRO restart settings) MUST be set in extraOptions so\n"
            "\t  that bin.cgyro.restart_<rho:.4f> files are written, and keep_files must\n"
            "\t  preserve them (keep_files: \"all\" is safest).",
            typeMsg='w',
        )
        return existing_additional_files_to_send

    source_iter = 0 if mode_lower == "first" else (evaluation_number - 1)
    source_sibling = (f"portals_sr_ev_{source_iter}" if in_simple_relax
                      else f"Evaluation.{source_iter}")

    source_folder = folder.parent.parent / source_sibling / folder.name / "base_cgyro"
    if plasma_subfolder:
        source_folder = source_folder / plasma_subfolder

    if not source_folder.is_dir():
        raise FileNotFoundError(
            f"[MITIM] CGYRO restart_from_cases={mode_lower!r} was requested for "
            f"{context_label}.{evaluation_number}, but the source base_cgyro "
            f"folder does not exist:\n\t{source_folder}\n"
            f"Clear restart_from_cases in the namelist to run without restart."
        )

    print(
        f"\n- [CGYRO restart_from_cases={mode_lower!r}] {context_label}.{evaluation_number} will restart from {source_sibling}:\n"
        f"\t{source_folder}",
        typeMsg='i',
    )

    resolved = dict(existing_additional_files_to_send) if existing_additional_files_to_send else {}
    missing_bin = []
    for rho in rho_locations:
        bin_file = source_folder / f"bin.cgyro.restart_{rho:.4f}"
        if not bin_file.is_file():
            missing_bin.append(bin_file.name)
            continue
        print(f"\t  rho={rho:.4f}: {bin_file.name} -> bin.cgyro.restart (warm start)", typeMsg='i')
        resolved.setdefault(float(rho), []).append((bin_file, "bin.cgyro.restart"))

    if missing_bin:
        raise FileNotFoundError(
            f"[MITIM] CGYRO restart_from_cases={mode_lower!r} was requested for "
            f"{context_label}.{evaluation_number}, but binary restart files are missing in "
            f"{source_sibling}: {missing_bin}.\n"
            "Check that RESTART_STEP was set in extraOptions on the source iteration and "
            "that keep_files did not unlink them. Clear restart_from_cases in the namelist "
            "to run without restart."
        )

    return resolved


# Deprecated alias — kept so any external imports of the old name keep working
# until the retiring release. Delegates to the new chain resolver.
_resolve_cgyro_restart_from_first = _resolve_cgyro_restart_chain


def _iteration_matches_spec_key(key, iteration):
    '''
    Test whether a PORTALS iteration index matches a *_special spec key:

      "N"    -> exact match: iteration == N
      ">N"   -> iteration >  N
      ">=N"  -> iteration >= N
      "<N"   -> iteration <  N
      "<=N"  -> iteration <= N

    Malformed keys (non-integer right-hand-side, unknown operator) return
    False and a warning is printed.
    '''
    key = str(key).strip()
    for op, cmp in (
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        (">",  lambda a, b: a >  b),
        ("<",  lambda a, b: a <  b),
    ):
        if key.startswith(op):
            try:
                return cmp(iteration, int(key[len(op):].strip()))
            except ValueError:
                print(f"\t- [CGYRO *_special] Malformed spec key {key!r}; ignoring", typeMsg='w')
                return False
    try:
        return iteration == int(key)
    except ValueError:
        print(f"\t- [CGYRO *_special] Malformed spec key {key!r}; expected integer or comparison (e.g. '5', '>5'); ignoring", typeMsg='w')
        return False


def _resolve_cgyro_per_iter_override(spec_dict, evaluation_number, existing_baseline):
    '''
    Generic helper for extraOptions_special / allocation_special: walk
    the per-iteration spec dict and merge any override entries whose key
    matches `evaluation_number` on top of `existing_baseline`.

    Match semantics:
      - Range-style keys (">N", "<N", ">=N", "<=N") are applied first.
      - Exact-integer keys ("N") are applied last so they override
        range-matched values when both match (e.g. "5" wins over ">4"
        for iteration 5).

    Returns (merged_dict, matched_keys). When no key matches or the spec
    is empty, returns (existing_baseline, []) unchanged.

    Never mutates the incoming namelist dict — PORTALS re-reads
    simulation_options on every iteration, so mutation would leak
    per-iter overrides across iterations.
    '''
    if not spec_dict:
        return existing_baseline, []

    range_keys = []
    exact_keys = []
    for k in spec_dict:
        ks = str(k).strip()
        if any(ks.startswith(op) for op in (">=", "<=", ">", "<")):
            range_keys.append(k)
        else:
            exact_keys.append(k)

    merged = dict(existing_baseline) if existing_baseline else {}
    matched = []

    # Ranges first (so exact matches applied after take precedence).
    for k in range_keys:
        if _iteration_matches_spec_key(k, evaluation_number):
            matched.append(str(k))
            for kk, vv in (spec_dict[k] or {}).items():
                merged[kk] = vv

    # Exact matches last so they win on conflicts.
    for k in exact_keys:
        if _iteration_matches_spec_key(k, evaluation_number):
            matched.append(str(k))
            for kk, vv in (spec_dict[k] or {}).items():
                merged[kk] = vv

    if not matched:
        return existing_baseline, []
    return merged, matched


def _resolve_cgyro_extra_options_special(run_options, evaluation_number, existing_extra_options):
    '''
    Per-iteration CGYRO `extraOptions` overrides driven by
    `extraOptions_special`, a dict keyed by iteration selectors. See
    `_resolve_cgyro_per_iter_override` for match semantics.

    Backward compatibility: the retired `extraOptions_first: {...}` is
    treated as an implicit `{"0": {...}}` under the new spec, with a
    one-line deprecation notice.
    '''
    spec = run_options.get("extraOptions_special")
    legacy_first = run_options.get("extraOptions_first")
    if not spec and legacy_first:
        print(
            "\t- [CGYRO extraOptions_special] `extraOptions_first` is deprecated; "
            "treating as `extraOptions_special: {\"0\": ...}`. Please update your namelist.",
            typeMsg='w',
        )
        spec = {"0": legacy_first}

    merged, matched = _resolve_cgyro_per_iter_override(
        spec or {}, evaluation_number, existing_extra_options,
    )
    if matched:
        print(
            f"\n- [CGYRO extraOptions_special] Iteration {evaluation_number}: "
            f"overrides from keys {matched} -> {sorted(set(merged) - set(existing_extra_options or {}))}"
            if existing_extra_options is not None
            else f"\n- [CGYRO extraOptions_special] Iteration {evaluation_number}: overrides from keys {matched}",
            typeMsg='i',
        )
    return merged


def _resolve_cgyro_allocation_special(run_options, evaluation_number, existing_allocation):
    '''
    Per-iteration SLURM allocation overrides driven by
    `allocation_special`, a dict keyed by iteration selectors. See
    `_resolve_cgyro_per_iter_override` for match semantics.

    Backward compatibility: the retired `allocation_first: {...}` is
    treated as an implicit `{"0": {...}}` under the new spec, with a
    one-line deprecation notice.
    '''
    spec = run_options.get("allocation_special")
    legacy_first = run_options.get("allocation_first")
    if not spec and legacy_first:
        print(
            "\t- [CGYRO allocation_special] `allocation_first` is deprecated; "
            "treating as `allocation_special: {\"0\": ...}`. Please update your namelist.",
            typeMsg='w',
        )
        spec = {"0": legacy_first}

    merged, matched = _resolve_cgyro_per_iter_override(
        spec or {}, evaluation_number, existing_allocation,
    )
    if matched:
        print(
            f"\n- [CGYRO allocation_special] Iteration {evaluation_number}: "
            f"overrides from keys {matched} -> {dict((k, merged[k]) for k in sorted(set(merged) - set(existing_allocation or {})))}"
            if existing_allocation is not None
            else f"\n- [CGYRO allocation_special] Iteration {evaluation_number}: overrides from keys {matched}",
            typeMsg='i',
        )
    return merged


# Deprecated aliases — preserve the names from the previous API so any
# external import paths keep working while callers migrate.
_resolve_cgyro_extra_options_first = _resolve_cgyro_extra_options_special
_resolve_cgyro_allocation_first = _resolve_cgyro_allocation_special


class gyrokinetic_model:

    def _evaluate_gyrokinetic_model(self, code = 'cgyro', gk_object = None):
        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options
        # ------------------------------------------------------------------------------------------------------------------------

        simulation_options = self.transport_evaluator_options[code]
        cold_start = self.cold_start

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        run_type = SIMtools._normalize_run_type(simulation_options["run"]["run_type"])
        keep_gk_files = simulation_options.get("keep_files", 'all')

        # Re-attach controls (see templates/namelist.portals.yaml). Read via .get()
        # so we do not mutate the shared namelist dict between PORTALS iterations;
        # the two keys are stripped from the kwargs actually forwarded to run().
        check_existing_runs = simulation_options["run"].get("check_existing_runs", False)
        every_n_minutes = simulation_options["run"].get("every_n_minutes", 10)
        if check_existing_runs and run_type != 'submit':
            print(f"\t- check_existing_runs=True has no effect when run_type='{run_type}' (only 'submit' supports re-attach); ignoring", typeMsg='w')
            check_existing_runs = False
        run_kwargs = {k: v for k, v in simulation_options["run"].items() if k not in ('check_existing_runs', 'every_n_minutes', 'restart_from_folder', 'restart_from_first', 'restart_from_cases', 'extraOptions_first', 'extraOptions_special', 'allocation_first', 'allocation_special', 'remove_scratch_after_fetch')}
        remove_scratch_after_fetch = simulation_options["run"].get("remove_scratch_after_fetch", False)

        # Translate namelist-level restart_from_folder into per-rho
        # additional_files_to_send tuples (renamed to out.cgyro.restart on stage-in).
        resolved_additional = _resolve_cgyro_restart_folder(
            simulation_options["run"],
            rho_locations,
            run_kwargs.get("additional_files_to_send"),
        )
        if resolved_additional is not None:
            run_kwargs["additional_files_to_send"] = resolved_additional

        # Automatic restart chain from prior iteration's base_cgyro folder,
        # driven by restart_from_cases ("first"=from iter 0, "all"=from iter N-1).
        # Skipped when restart_from_folder is set (the helper short-circuits).
        resolved_additional = _resolve_cgyro_restart_chain(
            simulation_options["run"],
            getattr(self, "evaluation_number", 0),
            self.folder,
            rho_locations,
            run_kwargs.get("additional_files_to_send"),
        )
        if resolved_additional is not None:
            run_kwargs["additional_files_to_send"] = resolved_additional

        # Per-iteration extraOptions overrides (extraOptions_special), e.g.
        # RESTART_STEP/MAX_TIME tuning for the seed iteration vs. the rest.
        # No-op for iterations that match no spec key.
        run_kwargs["extraOptions"] = _resolve_cgyro_extra_options_special(
            simulation_options["run"],
            getattr(self, "evaluation_number", 0),
            run_kwargs.get("extraOptions"),
        )

        # Per-iteration SLURM allocation overrides (allocation_special), e.g.
        # longer `minutes` on the seed iteration that has to converge the
        # full transient before writing its restart blob; shorter on later
        # iterations that only run MAX_TIME past the warm-start.
        run_kwargs["allocation"] = _resolve_cgyro_allocation_special(
            simulation_options["run"],
            getattr(self, "evaluation_number", 0),
            run_kwargs.get("allocation"),
        )

        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare object
        # ------------------------------------------------------------------------------------------------------------------------

        subfolder_name = f"base_{code}"
        metadata_path = self.folder / subfolder_name / "cgyro_submission.json"

        # <><><><><><>
        # If the way to store data is in pickle, try first to read the stored pickled in the folder (e.g. for SR stage)
        # <><><><><><>
        gk_object_unpickled = False
        if keep_gk_files in ['pickle']:
            try:
                pickle_file = self.folder / f"{subfolder_name}" / "gk_object.pkl"
                gk_object = SIMtools.restore_class_pickle(pickle_file)
                gk_object_unpickled = True
                print('\t- Pickle file with GK object information has been restored successfully', typeMsg='i')
            except Exception as e:
                gk_object_unpickled = False
                print('\t- Pickle file could not be read, with error:', typeMsg='w')
                print(e)

        # <><><><><><>
        # Standard run
        # <><><><><><>
        reattached = False
        skip_check_fetch = False
        if not gk_object_unpickled:
            gk_object = gk_object(rhos=rho_locations)

            _ = gk_object.prep(
                self.powerstate.profiles_transport,
                self.folder,
                )

            # <><><><><><>
            # Optional re-attach: if a prior process submitted this job and
            # wrote submission metadata, skip run() and go straight to
            # check/fetch/read against the existing slurm allocation.
            # <><><><><><>
            if check_existing_runs:
                if metadata_path.exists():
                    print("")
                    print(f"\t==================== [check_existing_runs] Re-attach to existing CGYRO submission ====================", typeMsg='i')
                    print("")
                    print(f"\t- Submission metadata found at:", typeMsg='i')
                    print(f"\t     {metadata_path}", typeMsg='i')
                    # Normally _run_prepare sets FolderSimLast (consumed by
                    # read()); re-attach skips that call to avoid the folder
                    # wipe / prompt, so set it manually here.
                    gk_object.FolderSimLast = self.folder / subfolder_name
                    data = gk_object.load_submission_state(metadata_path)
                    _jobinfo = data.get("job", {})
                    print("")
                    print(f"\t- Prior submission: jobid={_jobinfo.get('jobid')} on {_jobinfo.get('machineSettings', {}).get('machine')}", typeMsg='i')
                    print(f"\t     remote folder: {_jobinfo.get('folderExecution')}", typeMsg='i')
                    print(f"\t     submitted at:  {data.get('created_utc')} (schema v{data.get('schema_version')})", typeMsg='i')
                    print("")
                    print(f"\t- Skipping run()/sbatch; polling this job with every_n_minutes={every_n_minutes}", typeMsg='i')
                    print("")
                    reattached = True

                    # Liveness probe — decision tree when the job is gone:
                    #   1. local result files already complete  -> skip to read()
                    #   2. otherwise try fetch() once: the job may have
                    #      finished cleanly while we were offline and the
                    #      results are still sitting in the remote scratch
                    #      folder waiting to be pulled.
                    #   3. only fall back to a fresh submission when fetch()
                    #      also can't produce a complete local set.
                    print(f"\t- Liveness probe via squeue...", typeMsg='i')
                    gk_object.simulation_job.check(file_output=gk_object.slurm_output)
                    print("")
                    if gk_object.simulation_job.status == 2:
                        print(f"\t- Slurm reports job is NOT in the queue (state={gk_object.simulation_job.infoSLURM.get('STATE')})", typeMsg='i')
                        if gk_object._local_results_complete():
                            print(f"\t- All expected CGYRO output files are already on local disk — skipping check()/fetch() and jumping to read()", typeMsg='i')
                            skip_check_fetch = True
                        else:
                            print(f"\t- Local results incomplete; attempting fetch() from remote scratch folder in case the job finished while we were offline...", typeMsg='i')
                            try:
                                gk_object.fetch()
                            except Exception as _fe:
                                print(f"\t- fetch() raised ({_fe})", typeMsg='w')
                            if gk_object._local_results_complete():
                                print(f"\t- Remote scratch had the results — fetch complete, skipping check()/fetch() in the main loop and jumping to read()", typeMsg='i')
                                skip_check_fetch = True
                            else:
                                print(f"\t- Even after fetch() the expected CGYRO output files are incomplete — the prior submission apparently failed.", typeMsg='w')
                                print(f"\t  Removing {metadata_path.name} and falling back to a fresh submission", typeMsg='w')
                                metadata_path.unlink(missing_ok=True)
                                reattached = False
                    else:
                        print(f"\t- Slurm reports job is still live (jobid={gk_object.simulation_job.jobid}, state={gk_object.simulation_job.infoSLURM.get('STATE')}); proceeding with check()/fetch()", typeMsg='i')
                    print("")
                else:
                    print("")
                    print(f"\t==================== [check_existing_runs] No prior CGYRO submission to re-attach ====================", typeMsg='i')
                    print("")
                    print(f"\t- Looked for metadata at:", typeMsg='i')
                    print(f"\t     {metadata_path}", typeMsg='i')
                    print(f"\t- File does not exist; this is a fresh PORTALS evaluation, submitting CGYRO normally", typeMsg='i')
                    print("")

            if not reattached:
                _ = gk_object.run(
                    subfolder_name,
                    cold_start=cold_start,
                    forceIfcold_start=True,
                    only_minimal_files=keep_gk_files in ['none', 'pickle'],
                    job_name_suffix=f"_ev{getattr(self, 'evaluation_number', 0)}",
                    **run_kwargs
                    )

        if run_type in ['normal', 'submit', 'send']:

            if not gk_object_unpickled:

                if run_type in ['submit'] and not skip_check_fetch:
                    print("")
                    print(f"\t- [submit] Polling slurm every {every_n_minutes} min until the job leaves the queue (state NOT FOUND / squeue returns nothing).", typeMsg='i')
                    print(f"\t  You can ^C at any time; {metadata_path.name} is on disk so re-attach will resume from where we left off.", typeMsg='i')
                    print("")
                    gk_object.check(
                        every_n_minutes=every_n_minutes,
                        skip_first_iteration_squeue=reattached,
                        custom_checker=getattr(gk_object, "_custom_check_callback", None),
                    )

                    print("")
                    print(f"\t- [submit] Job finished on the cluster — pulling the result tarball and organizing files into per-rho folders.", typeMsg='i')
                    print("")
                    gk_object.fetch()
                elif run_type in ['submit'] and skip_check_fetch:
                    print("")
                    print(f"\t- [submit] Results were already local — reading them directly without polling or fetching.", typeMsg='i')
                    print("")

                gk_object.read(
                    label=subfolder_name,
                    minimal=True,  # In case I pickle, I don't want to be extra heavy
                    **simulation_options["read"]
                    )

                # Optional remote scratch cleanup on the submit path. run_type
                # 'normal' already cleans up via run(removeScratchFolders=True);
                # submit mode defers that because fetch is decoupled from run.
                # Wrapped in a try/except so a flaky connection doesn't abort
                # the PORTALS iteration just because the rm -rf failed.
                if run_type == 'submit' and remove_scratch_after_fetch:
                    try:
                        gk_object.simulation_job.connect()
                        gk_object.simulation_job.remove_scratch_folder()
                        gk_object.simulation_job.close()
                        print(f"\t- [submit] remove_scratch_after_fetch=true — removed remote scratch {gk_object.simulation_job.folderExecution}", typeMsg='i')
                    except Exception as _rs_e:
                        print(f"\t- remote scratch removal raised ({_rs_e}); leaving the folder in place", typeMsg='w')

                # Invariant for re-attach: "metadata present => job in flight (or
                # retrieval not yet complete)". Read succeeded, so drop the file.
                if metadata_path.exists():
                    print(f"\t- [check_existing_runs] Read finished — removing stale submission metadata at {metadata_path.name} so the next PORTALS iteration submits fresh", typeMsg='i')
                metadata_path.unlink(missing_ok=True)

                # Special case to keep only the pickle file but remove all heavy files.
                # Save pickle FIRST; only unlink the heavy files if the pickle is on disk,
                # otherwise a save failure would leave the run with no data at all.
                # Skip pickle on a re-attached run — `inputs_files` / normalization
                # state may be incomplete (reconstructed via prep only), which would
                # produce an unusable pickle.
                if keep_gk_files in ['pickle'] and reattached:
                    print("\t- pickle requested but run was re-attached; skipping pickle save (inputs not fully available)", typeMsg='i')
                elif keep_gk_files in ['pickle']:
                    pickle_file.parent.mkdir(parents=True, exist_ok=True)
                    gk_object.save_pickle(pickle_file)
                    if pickle_file.exists():
                        for file in gk_object.output_files_simulation["complete"]:
                            for rho in gk_object.rhos:
                                fileN = f"{file}_{rho:.4f}"
                                (self.folder / f"{subfolder_name}" / fileN).unlink(missing_ok=True)
                    else:
                        print(
                            f"\t- save_pickle did not produce {pickle_file}; keeping raw CGYRO files",
                            typeMsg='w',
                        )
        
            # ------------------------------------------------------------------------------------------------------------------------
            # Pass the information to what power_transport expects
            # ------------------------------------------------------------------------------------------------------------------------

            self.QeGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Qe_mean for i in range(len(rho_locations))])
            self.QeGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Qe_std for i in range(len(rho_locations))])
                    
            self.QiGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Qi_mean for i in range(len(rho_locations))])
            self.QiGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Qi_std for i in range(len(rho_locations))])
                    
            self.GeGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Ge_mean for i in range(len(rho_locations))])
            self.GeGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Ge_std for i in range(len(rho_locations))]) 
            
            self.GZGB_turb = self.QeGB_turb*0.0 #TODO     
            self.GZGB_turb_stds = self.QeGB_turb*0.0 #TODO          

            self.MtGB_turb = self.QeGB_turb*0.0 #TODO     
            self.MtGB_turb_stds = self.QeGB_turb*0.0 #TODO     

            self.QieGB_turb = self.QeGB_turb*0.0 #TODO     
            self.QieGB_turb_stds = self.QeGB_turb*0.0 #TODO     

        elif run_type == 'prep':
            
            # Prevent writing the json file from variables, as we will wait for the user to run CGYRO externally and provide the json themselves
            self._write_json_from_variables_turb = False
            
            # Wait until the user has placed the json file in the right folder
            
            self.powerstate.profiles_transport.write_state(self.folder / subfolder_name / "input.gacode")
            
            pre_checks(self)

            file_path = self.folder / 'fluxes_turb.json'

            attempts = 0
            all_good = post_checks(self) if file_path.exists() else False
            while (file_path.exists() is False) or (not all_good):
                if attempts > 0:
                    print(f"\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f" MITIM could not find the file... looping back", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                logic_to_wait(self.folder, self.folder / subfolder_name)
                attempts += 1

                if file_path.exists():
                    all_good = post_checks(self)

    def _stable_correction(self, simulation_options_all):

        simulation_options = simulation_options_all["cgyro"]

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
                        print(f"\n\t- Qi considered stable at plasma #{b}, radius #{i}: {QiMWm2[b, i]:.2e} MW/m^2 in CGYRO simulation < {Qi_stable_criterion:.2e} MW/m^2 criterion (see namelist)", typeMsg='q')
                        Qi_std = QiMWm2_target_b[i] * Qi_stable_percent_error / 100
                        print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target value as standard deviation: sigma = {Qi_std:.2e} MW/m^2 instead of {QiMWm2_stds[b, i]:.2e} MW/m^2", typeMsg='i')
                        QiMWm2_stds[b, i] = Qi_std
        else:
            QiMWm2_target = self.powerstate.plasma['QiMWm2'][0, 1:].cpu().numpy()
            for i in range(len(QiMWm2)):
                if QiMWm2[i] < Qi_stable_criterion:
                    print(f"\n\t- Qi considered stable at radius #{i}: {QiMWm2[i]:.2e} MW/m^2 in CGYRO simulation < {Qi_stable_criterion:.2e} MW/m^2 criterion (see namelist)", typeMsg='q')
                    Qi_std = QiMWm2_target[i] * Qi_stable_percent_error / 100
                    print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target value as standard deviation: sigma = {Qi_std:.2e} MW/m^2 instead of {QiMWm2_stds[i]:.2e} MW/m^2", typeMsg='i')
                    QiMWm2_stds[i] = Qi_std

class cgyro_model(gyrokinetic_model):

    def evaluate_turbulence(self):

        if self.transport_evaluator_options["cgyro"].get("run_base_tglf", True):
            # Run base TGLF, to keep track of discrepancies! ---------------------------------------------
            simulation_options_tglf = self.transport_evaluator_options["tglf"]
            simulation_options_tglf["use_scan_trick_for_stds"] = None
            self._evaluate_tglf(pass_info = False)
            # --------------------------------------------------------------------------------------------

        self._evaluate_gyrokinetic_model(code = 'cgyro', gk_object = CGYROtools.CGYRO)

    # ----------------------------------------------------------------------------------------
    # Multi-plasma CGYRO — fan a list of profile states through run_over_plasmas so that every
    # (plasma, rho) work unit is dispatched concurrently by the existing FARMINGtools pipeline.
    # Used by power_transport._evaluate_batched() when the powerstate carries batch_size > 1.
    # ----------------------------------------------------------------------------------------
    def evaluate_turbulence_batched(self, list_of_states, pass_info=True):

        # Run base TGLF for diagnostics, same as single-plasma path (pass_info=False
        # so TGLF results are computed for comparison but don't overwrite the flux arrays)
        if self.transport_evaluator_options["cgyro"].get("run_base_tglf", True):
            from mitim_modules.powertorch.physics_models.transport_tglf import tglf_model
            simulation_options_tglf = self.transport_evaluator_options["tglf"]
            simulation_options_tglf["use_scan_trick_for_stds"] = None
            tglf_model._evaluate_tglf_batched(self, list_of_states, pass_info=False)

        simulation_options = self.transport_evaluator_options["cgyro"]
        cold_start = self.cold_start

        run_type = SIMtools._normalize_run_type(simulation_options["run"].get("run_type", "normal"))
        if run_type == "prep":
            raise NotImplementedError(
                "run_type='prep' (interactive external CGYRO run) is not supported in "
                "batched mode. Use single-plasma evaluation or run_type='normal'/'submit'."
            )

        # Re-attach controls (see templates/namelist.portals.yaml). Read via .get()
        # so we do not mutate the shared namelist between PORTALS iterations.
        check_existing_runs = simulation_options["run"].get("check_existing_runs", False)
        every_n_minutes = simulation_options["run"].get("every_n_minutes", 10)
        remove_scratch_after_fetch = simulation_options["run"].get("remove_scratch_after_fetch", False)
        if check_existing_runs and run_type != 'submit':
            print(f"\t- check_existing_runs=True has no effect when run_type='{run_type}' (only 'submit' supports re-attach); ignoring", typeMsg='w')
            check_existing_runs = False

        keep_gk_files = simulation_options.get("keep_files", "all")

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        N = len(list_of_states)
        nrho = len(rho_locations)

        metadata_path = self.folder / "base_cgyro" / "cgyro_submission.json"

        # Try to restore from pickle if keep_files == 'pickle' (mirrors single-plasma path)
        cgyro_unpickled = False
        pickle_file = self.folder / "base_cgyro" / "gk_object_batched.pkl"
        if keep_gk_files in ["pickle"]:
            try:
                cgyro = SIMtools.restore_class_pickle(pickle_file)
                cgyro_unpickled = True
                plasma_labels = {p: f"base_cgyro_plasma{p}" for p in range(N)}
                print("\t- Pickle file with batched GK object information has been restored successfully", typeMsg="i")
            except Exception as e:
                cgyro_unpickled = False
                print("\t- Pickle file could not be read, with error:", typeMsg="w")
                print(e)

        reattached = False
        skip_check_fetch = False
        if not cgyro_unpickled:
            cgyro = CGYROtools.CGYRO(rhos=rho_locations)

            # run_over_plasmas / _prepare_plasmas_state call _run_prepare directly
            # (not CGYRO.run()), so preprocess_options must be set beforehand for
            # _run_prepare -> _apply_cgyro_preprocessing to pick it up.
            cgyro._preprocess_options = simulation_options["run"].get("preprocess_options")

            # Filter simulation_options["run"] to only keys that run_over_plasmas accepts;
            # CGYRO-specific keys (preprocess_options) are handled above; re-attach
            # controls (check_existing_runs, every_n_minutes) are consumed here.
            # restart_from_folder is resolved below into additional_files_to_send.
            _run_over_plasmas_keys = {
                "code_settings", "extraOptions", "multipliers", "minimum_delta_abs",
                "ApplyCorrections", "Quasineutral", "launchSlurm", "allocation",
                "run_type", "additional_files_to_send", "helper_lostconnection",
            }
            run_kwargs = {k: v for k, v in simulation_options["run"].items() if k in _run_over_plasmas_keys}

            # Translate namelist-level restart_from_folder into per-rho
            # additional_files_to_send tuples (renamed to out.cgyro.restart on stage-in).
            # Apply before both the re-attach and fresh-submit branches so either path
            # sees the resolved dict via run_kwargs["additional_files_to_send"].
            resolved_additional = _resolve_cgyro_restart_folder(
                simulation_options["run"],
                rho_locations,
                run_kwargs.get("additional_files_to_send"),
            )
            if resolved_additional is not None:
                run_kwargs["additional_files_to_send"] = resolved_additional

            # Automatic restart chain in batched mode: for every iteration N>=1,
            # pull bin.cgyro.restart from plasma 0 of the source iteration
            # (iter 0 for "first", iter N-1 for "all"). All plasmas in the
            # current batched call warm-start from the same plasma-0 reference
            # since they share the seed-iteration semantics.
            resolved_additional = _resolve_cgyro_restart_chain(
                simulation_options["run"],
                getattr(self, "evaluation_number", 0),
                self.folder,
                rho_locations,
                run_kwargs.get("additional_files_to_send"),
                plasma_subfolder="base_cgyro_plasma0",
            )
            if resolved_additional is not None:
                run_kwargs["additional_files_to_send"] = resolved_additional

            # Per-iteration extraOptions overrides (extraOptions_special). All
            # plasmas in a batched call share the iteration index and therefore
            # apply the same merged overrides.
            run_kwargs["extraOptions"] = _resolve_cgyro_extra_options_special(
                simulation_options["run"],
                getattr(self, "evaluation_number", 0),
                run_kwargs.get("extraOptions"),
            )

            # Per-iteration SLURM allocation overrides (allocation_special).
            run_kwargs["allocation"] = _resolve_cgyro_allocation_special(
                simulation_options["run"],
                getattr(self, "evaluation_number", 0),
                run_kwargs.get("allocation"),
            )

            if check_existing_runs:
                if metadata_path.exists():
                    print("")
                    print(f"\t==================== [check_existing_runs] Re-attach to existing batched CGYRO submission ====================", typeMsg='i')
                    print("")
                    print(f"\t- Submission metadata found at:", typeMsg='i')
                    print(f"\t     {metadata_path}", typeMsg='i')
                    print("")
                    print(f"\t- Rebuilding per-plasma state for {N} plasma(s) (profiles, inputs, normalizations) without re-submitting...", typeMsg='i')
                    print("")

                    # Rebuild per-plasma state (profiles, inputs_files, normalizations)
                    # that read_plasma() relies on, without contacting the cluster.
                    _ = cgyro.prep(
                        list_of_states[0],
                        self.folder,
                        cold_start=False,
                    )
                    # forceIfcold_start=True so _run_prepare's askNewFolder() does
                    # not prompt the user mid-re-attach; inputs are deterministic
                    # from namelist+powerstate so re-staging the per-plasma folder
                    # is harmless (and the slurm job already has its own copy).
                    _, _, plasma_labels = cgyro._prepare_plasmas_state(
                        list_of_states,
                        base_subfolder="base_cgyro",
                        cold_start=False,
                        forceIfcold_start=True,
                        code_settings=run_kwargs.get("code_settings"),
                        extraOptions=run_kwargs.get("extraOptions"),
                        multipliers=run_kwargs.get("multipliers"),
                        minimum_delta_abs=run_kwargs.get("minimum_delta_abs"),
                        only_minimal_files=keep_gk_files in ["none", "pickle"],
                        launchSlurm=run_kwargs.get("launchSlurm", True),
                        allocation=run_kwargs.get("allocation"),
                        additional_files_to_send=run_kwargs.get("additional_files_to_send"),
                        ApplyCorrections=run_kwargs.get("ApplyCorrections", True),
                        Quasineutral=run_kwargs.get("Quasineutral", False),
                        announce=False,
                    )
                    data = cgyro.load_submission_state(metadata_path)
                    _jobinfo = data.get("job", {})
                    print("")
                    print(f"\t- Prior submission: jobid={_jobinfo.get('jobid')} on {_jobinfo.get('machineSettings', {}).get('machine')}", typeMsg='i')
                    print(f"\t     remote folder: {_jobinfo.get('folderExecution')}", typeMsg='i')
                    print(f"\t     submitted at:  {data.get('created_utc')} (schema v{data.get('schema_version')})", typeMsg='i')
                    print("")
                    print(f"\t- Skipping run_over_plasmas()/sbatch; polling this job with every_n_minutes={every_n_minutes}", typeMsg='i')
                    print("")
                    reattached = True

                    # Liveness probe — same stale-job decision tree as
                    # single-plasma: job gone + local complete -> read;
                    # job gone + local incomplete -> try fetch() once
                    # (results may still be in remote scratch); if fetch
                    # still can't fill the local set -> resubmit.
                    print(f"\t- Liveness probe via squeue...", typeMsg='i')
                    cgyro.simulation_job.check(file_output=cgyro.slurm_output)
                    print("")
                    if cgyro.simulation_job.status == 2:
                        print(f"\t- Slurm reports job is NOT in the queue (state={cgyro.simulation_job.infoSLURM.get('STATE')})", typeMsg='i')
                        if cgyro._local_results_complete():
                            print(f"\t- All expected CGYRO output files are already on local disk — skipping check()/fetch() and jumping to read_plasma()", typeMsg='i')
                            skip_check_fetch = True
                        else:
                            print(f"\t- Local results incomplete; attempting fetch() from remote scratch folder in case the job finished while we were offline...", typeMsg='i')
                            try:
                                cgyro.fetch()
                            except Exception as _fe:
                                print(f"\t- fetch() raised ({_fe})", typeMsg='w')
                            if cgyro._local_results_complete():
                                print(f"\t- Remote scratch had the results — fetch complete, skipping check()/fetch() in the main loop and jumping to read_plasma()", typeMsg='i')
                                skip_check_fetch = True
                            else:
                                print(f"\t- Even after fetch() the expected CGYRO output files are incomplete — the prior submission apparently failed.", typeMsg='w')
                                print(f"\t  Removing {metadata_path.name} and falling back to a fresh submission", typeMsg='w')
                                metadata_path.unlink(missing_ok=True)
                                reattached = False
                    else:
                        print(f"\t- Slurm reports job is still live (jobid={cgyro.simulation_job.jobid}, state={cgyro.simulation_job.infoSLURM.get('STATE')}); proceeding with check()/fetch()", typeMsg='i')
                    print("")
                else:
                    print("")
                    print(f"\t==================== [check_existing_runs] No prior batched CGYRO submission to re-attach ====================", typeMsg='i')
                    print("")
                    print(f"\t- Looked for metadata at:", typeMsg='i')
                    print(f"\t     {metadata_path}", typeMsg='i')
                    print(f"\t- File does not exist; this is a fresh PORTALS evaluation, submitting CGYRO normally", typeMsg='i')
                    print("")

            if not reattached:
                _ = cgyro.prep(
                    list_of_states[0],
                    self.folder,
                    cold_start=cold_start,
                )

                plasma_labels = cgyro.run_over_plasmas(
                    list_of_states,
                    base_subfolder="base_cgyro",
                    cold_start=cold_start,
                    forceIfcold_start=True,
                    extra_name=self.name,
                    attempts_execution=2,
                    only_minimal_files=keep_gk_files in ["none", "pickle"],
                    job_name_suffix=f"_ev{getattr(self, 'evaluation_number', 0)}",
                    **run_kwargs,
                )

            # run_over_plasmas does not poll/fetch for run_type='submit'; do it
            # here so batched `submit` actually works (also covers the re-attach
            # path, since reattached runs need the same check/fetch cycle).
            if run_type == "submit" and not skip_check_fetch:
                print("")
                print(f"\t- [submit] Polling slurm every {every_n_minutes} min until the batched CGYRO job leaves the queue (state NOT FOUND / squeue returns nothing).", typeMsg='i')
                print(f"\t  You can ^C at any time; {metadata_path.name} is on disk so re-attach will resume from where we left off.", typeMsg='i')
                print("")
                cgyro.check(
                    every_n_minutes=every_n_minutes,
                    skip_first_iteration_squeue=reattached,
                    custom_checker=getattr(cgyro, "_custom_check_callback", None),
                )

                print("")
                print(f"\t- [submit] Job finished on the cluster — pulling the result tarball and organizing files into per-(plasma,rho) folders.", typeMsg='i')
                print("")
                cgyro.fetch()
            elif run_type == "submit" and skip_check_fetch:
                print("")
                print(f"\t- [submit] Results were already local — reading them directly without polling or fetching.", typeMsg='i')
                print("")
        Qe_batch     = np.zeros((N, nrho))
        Qe_std_batch = np.zeros((N, nrho))
        Qi_batch     = np.zeros((N, nrho))
        Qi_std_batch = np.zeros((N, nrho))
        Ge_batch     = np.zeros((N, nrho))
        Ge_std_batch = np.zeros((N, nrho))
        GZ_batch     = np.zeros((N, nrho))
        GZ_std_batch = np.zeros((N, nrho))
        Mt_batch     = np.zeros((N, nrho))
        Mt_std_batch = np.zeros((N, nrho))
        S_batch      = np.zeros((N, nrho))
        S_std_batch  = np.zeros((N, nrho))

        for p, label in plasma_labels.items():
            if not cgyro_unpickled:
                cgyro.read_plasma(
                    p,
                    label=label,
                    minimal=True,
                    **simulation_options["read"],
                )
            outputs = cgyro.results[label]["output"]

            Qe_batch[p, :]     = np.array([outputs[i].Qe_mean for i in range(nrho)])
            Qe_std_batch[p, :] = np.array([outputs[i].Qe_std for i in range(nrho)])
            Qi_batch[p, :]     = np.array([outputs[i].Qi_mean for i in range(nrho)])
            Qi_std_batch[p, :] = np.array([outputs[i].Qi_std for i in range(nrho)])
            Ge_batch[p, :]     = np.array([outputs[i].Ge_mean for i in range(nrho)])
            Ge_std_batch[p, :] = np.array([outputs[i].Ge_std for i in range(nrho)])

            # GZ, Mt, Qie not yet available from CGYRO — zero as in single-plasma path
            GZ_batch[p, :]     = 0.0  # TODO
            GZ_std_batch[p, :] = 0.0  # TODO
            Mt_batch[p, :]     = 0.0  # TODO
            Mt_std_batch[p, :] = 0.0  # TODO
            S_batch[p, :]      = 0.0  # TODO
            S_std_batch[p, :]  = 0.0  # TODO

        # Optional remote scratch cleanup on the submit path — see the
        # single-plasma branch for the rationale. Same try/except shape so
        # a connection hiccup doesn't abort the PORTALS iteration.
        if run_type == 'submit' and remove_scratch_after_fetch:
            try:
                cgyro.simulation_job.connect()
                cgyro.simulation_job.remove_scratch_folder()
                cgyro.simulation_job.close()
                print(f"\t- [submit] remove_scratch_after_fetch=true — removed remote scratch {cgyro.simulation_job.folderExecution}", typeMsg='i')
            except Exception as _rs_e:
                print(f"\t- remote scratch removal raised ({_rs_e}); leaving the folder in place", typeMsg='w')

        # Invariant for re-attach: "metadata present => job in flight (or
        # retrieval not yet complete)". Read loop succeeded, so drop the file.
        if metadata_path.exists():
            print(f"\t- [check_existing_runs] Batched read finished — removing stale submission metadata at {metadata_path.name} so the next PORTALS iteration submits fresh", typeMsg='i')
        metadata_path.unlink(missing_ok=True)

        # Save pickle first, then remove heavy files only if the pickle is on disk.
        # This mirrors the single-plasma path and prevents data loss if save_pickle raises.
        # Skip on a re-attached run — per-plasma `inputs_files` / normalization
        # state was reconstructed minimally via prep, so a pickle would be incomplete.
        if keep_gk_files in ["pickle"] and reattached:
            print("\t- pickle requested but run was re-attached; skipping pickle save (inputs not fully available)", typeMsg="i")
        elif keep_gk_files in ["pickle"] and not cgyro_unpickled:
            pickle_file.parent.mkdir(parents=True, exist_ok=True)
            cgyro.save_pickle(pickle_file)
            if pickle_file.exists():
                for p, label in plasma_labels.items():
                    for file in cgyro.output_files_simulation["complete"]:
                        for rho in cgyro.rhos:
                            fileN = f"{file}_{rho:.4f}"
                            (self.folder / label / fileN).unlink(missing_ok=True)
            else:
                print(
                    f"\t- save_pickle did not produce {pickle_file}; keeping raw CGYRO files",
                    typeMsg='w',
                )

        if pass_info:
            self.QeGB_turb      = Qe_batch
            self.QeGB_turb_stds = Qe_std_batch

            self.QiGB_turb      = Qi_batch
            self.QiGB_turb_stds = Qi_std_batch

            self.GeGB_turb      = Ge_batch
            self.GeGB_turb_stds = Ge_std_batch

            self.GZGB_turb      = GZ_batch
            self.GZGB_turb_stds = GZ_std_batch

            self.MtGB_turb      = Mt_batch
            self.MtGB_turb_stds = Mt_std_batch

            self.QieGB_turb      = S_batch
            self.QieGB_turb_stds = S_std_batch

        return cgyro


def pre_checks(self):
    
    plasma = self.powerstate.plasma

    txt = "\nFluxes to be matched by turbulence ( Target - Neoclassical ):"

    # Print gradients
    for var, varn in zip(
        ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
        ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
    ):
        txt += f"\n{var}   = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]:.6f}   "

    # Print target fluxes
    for var, varn in zip(
        ["Qe (GB)", "Qi (GB)", "Ge (GB)", "GZ (GB)", "Mt (GB)"],
        ["QeGB", "QiGB", "GeGB", "GZGB", "MtGB"],
    ):
        txt += f"\n{var}  = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]-self.__dict__[f'{varn}_neoc'][j]:.4e}   "

    print(txt)

def logic_to_wait(folder, subfolder):
    print(f"\n**** Simulation inputs prepared. Please, run it from the simulation setup in folder:\n", typeMsg='i')
    print(f"\t {subfolder}\n", typeMsg='i')
    print(f"**** When finished, the fluxes_turb.json file should be placed in:\n", typeMsg='i')
    print(f"\t {folder}/fluxes_turb.json\n", typeMsg='i')
    while not print(f"**** When you have done that, please say yes", typeMsg='q'):
        pass

def post_checks(self, rtol = 1e-3):
    
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
