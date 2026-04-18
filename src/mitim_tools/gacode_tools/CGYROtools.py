import math
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as plt
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools.utils import GACODEdefaults, CGYROutils
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.utils import SIMplot
from mitim_tools.misc_tools import GRAPHICStools, CONFIGread
from mitim_tools.gacode_tools.utils import GACODEplotting
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


def _annotate_missing(ax, reason):
    '''
    Stamp a small "data unavailable" note on an axes when the underlying
    CGYRO output files weren't written or weren't retrieved (e.g.
    MOMENT_PRINT_FLAG=0 drops kxky_n/e/v; FIELD_PRINT_FLAG=0 drops
    kxky_apar/bpar). Keeps the surrounding title/labels intact so the reader
    sees which panel was supposed to be there.
    '''
    ax.text(
        0.5, 0.5, f"Data unavailable\n({reason})",
        ha='center', va='center', transform=ax.transAxes,
        color='gray', style='italic', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.4),
    )


def _format_wall_seconds(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def cgyro_per_task_status(sim):
    '''
    Custom checker for `mitim_simulation.check(custom_checker=...)` that
    prints, for every (subfolder, rho) currently tracked by
    `sim.kwargs_organize`, a per-task status line derived from the CGYRO
    output files on the remote (`out.cgyro.info`, `out.cgyro.timing`,
    `out.cgyro.tag`) plus the global slurm STATE captured by the outer
    poller in `simulation_job.infoSLURM`.

    One SSH round-trip per check poll — the remote side loops through all
    folders and emits a
    `folder|raw_state|avg_total|steps|wall_seconds|since_update_seconds|tag_token`
    line per element, which we then reclassify and pretty-print locally.

    Per-task classification, in priority order:
        1. `out.cgyro.tag` present  -> trust its token (FINISHED/TIMEOUT/ERROR/...)
        2. `out.cgyro.timing` mtime stale (no append for max(60, min(600, 3*avg))s)
              -> STALLED (or TIMED_OUT if slurm STATE is also terminal). This is
                 the signal that catches slurm wall-clock kills, since
                 `out.cgyro.info` mtime never moves after init and the prior
                 "wall since init" alone made killed runs look identical to
                 live ones.
        3. slurm global STATE in {NOT FOUND, COMPLETED, TIMEOUT, FAILED,
              CANCELLED} while the per-task raw state is still RUNNING
              -> TIMED_OUT (catches the case where the staleness threshold
                 has not yet elapsed but the job is already gone).
        4. otherwise -> raw state (NOT_STARTED / INITIALIZED / RUNNING).
    '''

    job = getattr(sim, "simulation_job", None)
    kwargs_organize = getattr(sim, "kwargs_organize", None)
    if job is None or kwargs_organize is None or not getattr(job, "launchSlurm", False):
        return

    # Global slurm STATE from the outer poller. Coarse (one row per job, not
    # per array task), so we use it only as a tiebreaker on top of the
    # per-task filesystem signals — never as the primary classifier.
    info_slurm = getattr(job, "infoSLURM", None) or {}
    slurm_state = info_slurm.get("STATE")
    job_terminal = slurm_state in ("NOT FOUND", "COMPLETED", "TIMEOUT", "FAILED", "CANCELLED")

    # Flatten code_executor into an ordered list of "subfolder/rho_{val:.4f}"
    # folder paths matching the slurm-array layout built by SIMtools._run.
    folders = []
    for sub, rhos in kwargs_organize["code_executor"].items():
        for rho in rhos:
            folders.append(f"{sub}/rho_{float(rho):.4f}")
    if not folders:
        return

    folder_list_sh = " ".join(f'"{f}"' for f in folders)

    # Single shell script executed remotely. Avg/steps awk reads every numeric
    # line in the "Run time" block (header "... TOTAL" is skipped by the
    # numeric-field guard) and averages the last column (TOTAL). We capture
    # mtime of out.cgyro.timing (live-append signal — moves every step) in
    # addition to mtime of out.cgyro.info (init timestamp — never moves), and
    # the first non-empty token of out.cgyro.tag if it exists.
    script = (
        f"cd {job.folderExecution} && for folder in {folder_list_sh}; do\n"
        '    info="$folder/out.cgyro.info"; timing="$folder/out.cgyro.timing"; tag="$folder/out.cgyro.tag"\n'
        '    now=$(date +%s)\n'
        '    if [ -f "$info" ]; then\n'
        '        info_mtime=$(stat -c %Y "$info"); wall=$((now - info_mtime))\n'
        '        if [ -f "$timing" ]; then\n'
        "            avg=$(awk '/^Run time/{run=1; next} run && NF>=14 && ($NF+0==$NF){s+=$NF; n++} END{if(n>0) printf \"%.3f\", s/n; else printf \"NA\"}' \"$timing\")\n"
        "            steps=$(awk '/^Run time/{run=1; next} run && NF>=14 && ($NF+0==$NF){n++} END{print n+0}' \"$timing\")\n"
        '            timing_mtime=$(stat -c %Y "$timing"); since_update=$((now - timing_mtime))\n'
        '            state="RUNNING"\n'
        '        else\n'
        '            avg="NA"; steps="0"; since_update=$wall; state="INITIALIZED"\n'
        '        fi\n'
        '        tag_token="-"\n'
        '        if [ -f "$tag" ]; then\n'
        "            tk=$(awk 'NF>0 {print $1; exit}' \"$tag\")\n"
        '            [ -n "$tk" ] && tag_token="$tk"\n'
        '        fi\n'
        '        echo "$folder|$state|$avg|$steps|$wall|$since_update|$tag_token"\n'
        '    else\n'
        '        echo "$folder|NOT_STARTED|NA|0|0|0|-"\n'
        '    fi\n'
        "done"
    )

    try:
        job.connect()
        out, _err = job.execute(script, printYN=False)
        job.close()
    except Exception as e:
        print(f"\t- [per-task status] remote inspection failed ({e}); continuing", typeMsg='w')
        return

    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    out = out or ""

    print(f"\t- Per-task CGYRO status ({len(folders)} element(s)):")
    for raw in out.splitlines():
        parts = raw.strip().split("|")
        if len(parts) < 7:
            continue
        folder, state, avg, steps, wall, since_update, tag_token = parts[:7]

        try:
            wall_i = max(0, int(wall))
        except ValueError:
            wall_i = 0
        try:
            since_update_i = max(0, int(since_update))
        except ValueError:
            since_update_i = 0
        try:
            avg_f = float(avg)
        except ValueError:
            avg_f = None

        # Stale-output threshold scales with the run's own step time so a slow
        # nonlinear case (~30s/step) is not flagged after a single missed step
        # while a fast linear case (~0.5s/step) still gets a generous grace
        # window. Floor 300s (5min) so short I/O pauses, checkpoint writes,
        # and intermittent filesystem blips don't false-flag healthy runs;
        # ceiling 600s. When avg is unknown (INITIALIZED), fall back to 180s.
        if avg_f is not None and avg_f > 0:
            stale_threshold = max(300, min(600, int(3 * avg_f)))
        else:
            stale_threshold = 180

        wall_str = _format_wall_seconds(wall_i) if wall_i > 0 else "—"
        update_str = _format_wall_seconds(since_update_i)

        # Reclassify (priority: terminal tag > staleness > slurm-terminal > raw).
        # Only FINISHED/TIMEOUT/ERROR tag tokens override the state. Any other
        # non-empty token (e.g. CGYRO phase indicators like "100"/"200") is
        # informational — surfaced as a suffix on the normal per-state line so
        # the step/avg/wall detail is preserved instead of being replaced by a
        # terse raw= fallback.
        effective = state
        reason = ""
        tag_suffix = ""
        tk = tag_token.upper() if (tag_token and tag_token != "-") else ""

        if tk == "FINISHED":
            effective, reason = "FINISHED", " (out.cgyro.tag=FINISHED)"
        elif tk == "TIMEOUT":
            effective, reason = "TIMED_OUT", " (out.cgyro.tag=TIMEOUT)"
        elif tk == "ERROR":
            effective, reason = "ERROR", " (out.cgyro.tag=ERROR)"
        else:
            if tk:
                tag_suffix = f" [out.cgyro.tag={tag_token}]"
            if state == "RUNNING" and since_update_i > stale_threshold:
                if job_terminal:
                    effective = "TIMED_OUT"
                    reason = f" (no out.cgyro.timing update for {update_str}; slurm STATE={slurm_state})"
                else:
                    effective = "STALLED"
                    reason = f" (no out.cgyro.timing update for {update_str}; threshold {stale_threshold}s — slurm wall-clock kill or rank crash likely)"
            elif state == "INITIALIZED" and since_update_i > stale_threshold:
                effective = "STALLED_INIT"
                reason = f" (no out.cgyro.timing after {update_str}; threshold {stale_threshold}s)"
            elif state == "RUNNING" and job_terminal:
                effective = "TIMED_OUT"
                reason = f" (slurm STATE={slurm_state})"

        if effective == "NOT_STARTED":
            print(f"\t     {folder}: pending — no out.cgyro.info on disk yet{tag_suffix}")
        elif effective == "INITIALIZED":
            print(f"\t     {folder}: initialized — out.cgyro.info present, awaiting out.cgyro.timing (wall since init: {wall_str}){tag_suffix}")
        elif effective == "RUNNING":
            print(f"\t     {folder}: running — {steps} step(s), avg TOTAL/step = {avg}s (wall since init: {wall_str}, last update {update_str} ago){tag_suffix}")
        elif effective == "STALLED":
            print(f"\t     {folder}: stalled{reason} — {steps} step(s), avg TOTAL/step = {avg}s (wall since init: {wall_str}){tag_suffix}", typeMsg='w')
        elif effective == "STALLED_INIT":
            print(f"\t     {folder}: stalled at init{reason} — out.cgyro.timing never appeared (wall since init: {wall_str}){tag_suffix}", typeMsg='w')
        elif effective == "TIMED_OUT":
            print(f"\t     {folder}: timed out{reason} — {steps} step(s), avg TOTAL/step = {avg}s (wall since init: {wall_str}, last update {update_str} ago){tag_suffix}", typeMsg='w')
        elif effective == "FINISHED":
            print(f"\t     {folder}: finished{reason} — {steps} step(s), avg TOTAL/step = {avg}s (wall since init: {wall_str}){tag_suffix}", typeMsg='i')
        elif effective == "ERROR":
            print(f"\t     {folder}: ERROR{reason} — {steps} step(s), avg TOTAL/step = {avg}s (wall since init: {wall_str}){tag_suffix}", typeMsg='w')
        else:
            print(f"\t     {folder}: {effective.lower()}{reason} — raw='{raw}'")

class CGYRO(SIMtools.mitim_simulation, SIMplot.GKplotting):

    # Opts CGYRO into persisting slurm-submission metadata (jobid, remote
    # folder, retrieval plan) whenever run_type='submit' is used, so a later
    # PORTALS restart can re-attach to the in-flight job rather than resubmit.
    _submission_metadata_filename = "cgyro_submission.json"

    # Per-task inspection for `check(custom_checker=...)`. Picked up by
    # transport_cgyro.py via `getattr(gk_object, '_custom_check_callback', None)`
    # so the generic gyrokinetic_model evaluator stays code-agnostic (GX etc.
    # simply get None and skip).
    _custom_check_callback = staticmethod(cgyro_per_task_status)

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Transient state used by run() to feed preprocess_options into _run_prepare()
        self._preprocess_options = None

        def code_call(folder, p, n=1, additional_command="", **kwargs):
            # MPI layout is resolved centrally in SLURMtools so the invented
            # knobs (full-node MPI on GPU machines, MPS sharing) live in one
            # place instead of being duplicated here and in code_slurm_settings.
            from mitim_tools.misc_tools import SLURMtools
            resolved = SLURMtools.resolve(code='cgyro', allocation={'resources_per_call': int(n)})
            mpi = resolved.mpi

            if mpi.get("numa") is not None:
                cgyro_cmd = (f"cgyro -e {folder} -n {mpi['n']} -nomp {mpi['nomp']} "
                             f"-numa {mpi['numa']} -mpinuma {mpi['mpinuma']} "
                             f"-p {p} {additional_command}")
            else:
                cgyro_cmd = (f"cgyro -e {folder} -n {mpi['n']} -nomp {mpi['nomp']} "
                             f"-p {p} {additional_command}")

            # Post-CGYRO: drop a warm-start bin.cgyro.restart that the run did
            # not overwrite, so the retrieval tarball doesn't ferry back an
            # unchanged blob we already have locally. out.cgyro.info is
            # written by CGYRO at every startup (cgyro_write_timedata.f90),
            # so its mtime is a reliable "this run started after here"
            # baseline. If bin.cgyro.restart is strictly newer, CGYRO wrote
            # it during the run -> keep. Otherwise (older or equal, i.e.
            # staged by PORTALS before the run, not rewritten) -> delete.
            # No-op when either file is absent (e.g. CGYRO crashed at init
            # or the run didn't use a warm-start at all). Wrapped in a
            # block so the slurm_array additional_command's trailing newline
            # doesn't break chaining.
            restart_path = f"{p}/{folder}/bin.cgyro.restart"
            info_path = f"{p}/{folder}/out.cgyro.info"
            cleanup_cmd = (
                f'if [ -f "{restart_path}" ] && [ -f "{info_path}" ] && '
                f'[ ! "{restart_path}" -nt "{info_path}" ]; then '
                f'rm -f "{restart_path}"; fi'
            )

            return cgyro_cmd.rstrip("\n") + "\n" + cleanup_cmd + "\n"

        # On GPU machines, always use a job array so each radius gets its own GPU allocation.
        _cgyro_machine_settings = CONFIGread.machineSettings(code='cgyro')
        _force_submission_type = 'slurm_array' if (_cgyro_machine_settings.get('gpus_per_node') or 0) > 0 else None

        self.run_specifications = {
            'code': 'cgyro',
            'input_file': 'input.cgyro',
            'code_call': code_call,
            'control_function': GACODEdefaults.addCGYROcontrol,
            'controls_file': 'input.cgyro.controls',
            'state_converter': 'to_cgyro',
            'input_class': CGYROinput,
            'complete_variation': None,
            'default_cores': 16,  # Default cores to use in the simulation
            'output_class': CGYROutils.CGYROoutput,
            'force_submission_type': _force_submission_type,
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t CGYRO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.output_files_simulation["minimal_base"] = [
            "bin.cgyro.geo",
            "bin.cgyro.ky_cflux",
            "bin.cgyro.ky_flux",
            "input.cgyro.gen",
            "out.cgyro.egrid",
            "out.cgyro.equilibrium",
            "out.cgyro.grids",
            "out.cgyro.hosts",
            "out.cgyro.info",
            "out.cgyro.memory",
            "out.cgyro.mpi",
            "out.cgyro.prec",
            "out.cgyro.rotation",
            "out.cgyro.startups",
            "out.cgyro.time",
            "out.cgyro.timing",
            "out.cgyro.version",
        ]

        self.output_files_simulation["complete_base"] = self.output_files_simulation["minimal_base"] + [
            "mitim.out",
        ]

        # Best-effort retrievals: tarred if present, absence logged once (no
        # 60s retry, no cold-start trigger). Two groups:
        #   - restart blobs + companion .flag/.tag that CGYRO may skip writing
        #     on short runs, crashes, or COMPLETING-timeouts.
        #   - large bin.cgyro.kxky_* dumps (tens to hundreds of MB each) that
        #     diagnostics can do without and whose retrieval over a slow
        #     shared filesystem was the dominant cost of fetch().
        self.output_files_simulation["optional_base"] = [
            "bin.cgyro.restart",
            "bin.cgyro.restart.flag",
            "out.cgyro.tag",
            "bin.cgyro.kxky_apar",
            "bin.cgyro.kxky_bpar",
            "bin.cgyro.kxky_e",
            "bin.cgyro.kxky_n",
            "bin.cgyro.kxky_phi",
            "bin.cgyro.kxky_v",
        ]

        # Nonlinear sim
        for key in ['minimal', 'complete']:
            self.output_files_simulation[f"{key}_nonlinear"] = self.output_files_simulation[f"{key}_base"] + [
                "bin.cgyro.freq"
                ]

        # Linear sim
        for key in ['minimal', 'complete']:
            self.output_files_simulation[f"{key}_linear"] = self.output_files_simulation[f"{key}_base"] + [
                "out.cgyro.freq",
                "bin.cgyro.phib",
                "bin.cgyro.aparb",
                "bin.cgyro.bparb",
                ]

        # Make sure, just in case, that "complete" and "minimal" are populated from this __init__, even if it will be re-defined later
        self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_nonlinear"])
        self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_nonlinear"])
        self.output_files_simulation["optional"] = copy.deepcopy(self.output_files_simulation["optional_base"])
        

    # Thin wrapper: capture preprocess_options and delegate to the generic run()
    def run(self, *args, preprocess_options=None, **kwargs):
        self._preprocess_options = preprocess_options
        try:
            return super().run(*args, **kwargs)
        finally:
            self._preprocess_options = None

    # Redefine to raise warning and allow selection of output files
    def _run_prepare(
        self,
        subfolder_simulation,
        extraOptions=None,
        multipliers=None,
        **kwargs,
    ):

        # ---------------------------------------------
        # Check if any *_SCALE_* variable is being used
        # ---------------------------------------------
        dictionary_check = {}
        if extraOptions is not None:
            if multipliers is not None:
                dictionary_check = {**extraOptions, **multipliers}
            else:
                dictionary_check = extraOptions
        elif multipliers is not None:
                dictionary_check = multipliers

        for key in dictionary_check:
            if '_SCALE_' in key:
                print("The use of *_SCALE_* is discouraged, please use the appropriate variable instead.", typeMsg='q')

        # ---------------------------------------------

        # Check if it's linear
        if 'Nonlinear' not in kwargs.get('code_settings', ''):
            self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_linear"])
            self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_linear"])
        else:
            self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_nonlinear"])
            self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_nonlinear"])
        # Optional-retrieval set is the same for linear and nonlinear.
        self.output_files_simulation["optional"] = copy.deepcopy(self.output_files_simulation["optional_base"])

        # Pre-process BOX_SIZE / N_RADIAL from local equilibrium if requested.
        # Model yaml (input.cgyro.models.yaml) can supply per-model defaults;
        # user-supplied self._preprocess_options override on a per-key basis.
        from mitim_tools.gacode_tools.utils import GACODEdefaults
        model_preprocess = GACODEdefaults.getCGYROpreprocessDefaults(kwargs.get("code_settings"))
        user_preprocess = getattr(self, "_preprocess_options", None) or {}
        merged_preprocess = {**model_preprocess, **user_preprocess}
        if merged_preprocess:
            saved_preprocess = getattr(self, "_preprocess_options", None)
            self._preprocess_options = merged_preprocess
            try:
                extraOptions = self._apply_cgyro_preprocessing(extraOptions or {})
            finally:
                self._preprocess_options = saved_preprocess

        # Enforce TOROIDALS_PER_PROC compatibility with N_TOROIDAL and MPI rank count.
        # Resolution order matches what _run_prepare will eventually write:
        #   input.cgyro.controls  ->  input.cgyro.models.yaml[code_settings]  ->  extraOptions
        extraOptions = self._enforce_toroidals_per_proc(
            extraOptions or {},
            kwargs.get("allocation"),
            code_settings=kwargs.get("code_settings"),
        )

        # Enforce PRINT_STEP so that DELTA_T * PRINT_STEP == 1.0 (integer PRINT_STEP).
        # Same resolution order as above: controls -> yaml -> extraOptions.
        extraOptions = self._enforce_print_step(
            extraOptions or {},
            code_settings=kwargs.get("code_settings"),
        )

        # Ensure RESTART_STEP >= total number of data outputs so the restart file
        # is written at least once at the end. One data output == DELTA_T*PRINT_STEP,
        # so n_outputs = ceil(MAX_TIME / (DELTA_T*PRINT_STEP)).
        extraOptions = self._enforce_restart_step(
            extraOptions or {},
            code_settings=kwargs.get("code_settings"),
        )

        return super()._run_prepare(
            subfolder_simulation,
            extraOptions=extraOptions,
            multipliers=multipliers,
            **kwargs,
        )

    def _enforce_toroidals_per_proc(self, extraOptions, allocation, code_settings=None):
        """
        CGYRO distributes N_TOROIDAL across MPI ranks (= resources_per_call GPUs).
        Each rank holds N_TOROIDAL/resources toroidal modes, so TOROIDALS_PER_PROC
        must be a multiple of that ratio. If the user's value is incompatible (or
        missing), coerce it to the minimum valid value and warn.

        Resolution order for N_TOROIDAL / TOROIDALS_PER_PROC mirrors the one
        _run_prepare applies when materializing input.cgyro:
            inputs_files[rho].controls       (base input.cgyro.controls)
                ->  input.cgyro.models.yaml[code_settings]   (e.g. Nonlinear sets N_TOROIDAL=12)
                ->  extraOptions             (user/per-rho override)
        """
        from mitim_tools.misc_tools import SLURMtools
        from mitim_tools.gacode_tools.utils import GACODEdefaults

        allocation = allocation or {}
        default_rpc = SLURMtools.CODE_HINTS.get('cgyro', {}).get("default_resources_per_call", 1)
        resources_per_call = int(allocation.get("resources_per_call", default_rpc))
        if resources_per_call <= 0:
            return extraOptions

        # Start from the per-rho controls snapshot (input.cgyro.controls) and,
        # if a code_settings label was provided, layer the yaml overrides on top
        # so we see the same N_TOROIDAL that will be written to disk.
        controls = {}
        if self.rhos is not None and len(self.rhos) > 0:
            controls = dict(self.inputs_files[self.rhos[0]].controls)
        if code_settings is not None:
            try:
                controls = GACODEdefaults.addCGYROcontrol(code_settings)
            except Exception as e:
                print(
                    f"\t- [preprocess] Could not resolve code_settings={code_settings!r} "
                    f"against input.cgyro.models.yaml ({e}); falling back to controls file only",
                    typeMsg="w",
                )

        n_tor_src = extraOptions.get('N_TOROIDAL', controls.get('N_TOROIDAL', 1))
        n_tor_list = [int(v) for v in n_tor_src] if isinstance(n_tor_src, (list, np.ndarray)) else [int(n_tor_src)]

        if any(nt <= 0 or nt % resources_per_call != 0 for nt in n_tor_list):
            print(
                f"\t- [preprocess] N_TOROIDAL={n_tor_list} not divisible by "
                f"resources_per_call={resources_per_call}; leaving TOROIDALS_PER_PROC as-is",
                typeMsg="w",
            )
            return extraOptions

        # Respect an explicit user-supplied TOROIDALS_PER_PROC in extraOptions:
        # if set there, leave it untouched (user is overriding on purpose).
        if 'TOROIDALS_PER_PROC' in extraOptions:
            return extraOptions

        required_divisors = [nt // resources_per_call for nt in n_tor_list]
        extraOptions = copy.deepcopy(extraOptions)
        tpp_src = controls.get('TOROIDALS_PER_PROC', required_divisors[0])
        tpp_list = [int(v) for v in tpp_src] if isinstance(tpp_src, (list, np.ndarray)) else [int(tpp_src)] * len(required_divisors)

        # Broadcast a scalar constraint to match len(tpp_list); pad divisors if N_TOROIDAL was scalar.
        if len(required_divisors) == 1 and len(tpp_list) > 1:
            required_divisors = required_divisors * len(tpp_list)
        if len(tpp_list) != len(required_divisors):
            print(
                f"\t- [preprocess] TOROIDALS_PER_PROC length {len(tpp_list)} mismatches "
                f"N_TOROIDAL length {len(required_divisors)}; leaving as-is",
                typeMsg="w",
            )
            return extraOptions

        coerced = [
            v if (v > 0 and v % d == 0) else d
            for v, d in zip(tpp_list, required_divisors)
        ]

        if coerced != tpp_list:
            print(
                f"\t- [preprocess] TOROIDALS_PER_PROC adjusted to be a multiple of "
                f"N_TOROIDAL/resources={required_divisors}: {coerced}",
                typeMsg="w",
            )

        # Preserve scalar shape if the caller originally gave a scalar and all coerced match.
        if not isinstance(tpp_src, (list, np.ndarray)) and len(set(coerced)) == 1:
            extraOptions['TOROIDALS_PER_PROC'] = coerced[0]
        else:
            extraOptions['TOROIDALS_PER_PROC'] = coerced

        return extraOptions

    def _enforce_print_step(self, extraOptions, code_settings=None):
        """
        Set PRINT_STEP so that DELTA_T * PRINT_STEP == 1.0 (PRINT_STEP integer).
        DELTA_T resolves through the same layering _run_prepare applies:
            inputs_files[rho].controls  ->  input.cgyro.models.yaml[code_settings]  ->  extraOptions
        If the user explicitly sets PRINT_STEP in extraOptions, it is respected.
        """
        from mitim_tools.gacode_tools.utils import GACODEdefaults

        if 'PRINT_STEP' in extraOptions:
            return extraOptions

        controls = {}
        if self.rhos is not None and len(self.rhos) > 0:
            controls = dict(self.inputs_files[self.rhos[0]].controls)
        if code_settings is not None:
            try:
                controls = GACODEdefaults.addCGYROcontrol(code_settings)
            except Exception as e:
                print(
                    f"\t- [preprocess] Could not resolve code_settings={code_settings!r} "
                    f"against input.cgyro.models.yaml ({e}); falling back to controls file only",
                    typeMsg="w",
                )

        dt_src = extraOptions.get('DELTA_T', controls.get('DELTA_T', None))
        if dt_src is None:
            return extraOptions

        dt_list = (
            [float(v) for v in dt_src]
            if isinstance(dt_src, (list, np.ndarray))
            else [float(dt_src)]
        )
        if any(dt <= 0 for dt in dt_list):
            print(
                f"\t- [preprocess] DELTA_T={dt_list} contains non-positive values; leaving PRINT_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        # PRINT_STEP * DELTA_T should equal 1.0 -> PRINT_STEP = round(1.0 / DELTA_T).
        print_steps = [max(1, int(round(1.0 / dt))) for dt in dt_list]

        extraOptions = copy.deepcopy(extraOptions)
        if not isinstance(dt_src, (list, np.ndarray)) and len(set(print_steps)) == 1:
            extraOptions['PRINT_STEP'] = print_steps[0]
        else:
            extraOptions['PRINT_STEP'] = print_steps

        print(
            f"\t- [preprocess] PRINT_STEP set to {extraOptions['PRINT_STEP']} "
            f"(DELTA_T={dt_src} -> DELTA_T*PRINT_STEP ~= 1.0)",
            typeMsg="i",
        )
        return extraOptions

    def _enforce_restart_step(self, extraOptions, code_settings=None):
        """
        Ensure CGYRO writes a restart by the end of the run. The restart trigger
        inside cgyro_restart.F90 is `mod(i_time, restart_step*print_step) == 0`,
        with i_time running 1..n_time and n_time = nint(MAX_TIME/DELTA_T). So the
        firing condition requires RESTART_STEP*PRINT_STEP <= n_time, i.e.
        RESTART_STEP <= n_outputs = MAX_TIME / (DELTA_T*PRINT_STEP).

        To guarantee exactly one restart at end-of-run, we set RESTART_STEP equal
        to n_outputs (this divides itself and fires at i_time == n_time).
        If the user's controls-file / yaml value is already <= n_outputs and
        divides it evenly, we keep it; otherwise we coerce to n_outputs.

        Resolution order matches _run_prepare:
            inputs_files[rho].controls  ->  input.cgyro.models.yaml[code_settings]  ->  extraOptions
        If the user explicitly sets RESTART_STEP in extraOptions, it is respected.
        """
        import math
        from mitim_tools.gacode_tools.utils import GACODEdefaults

        if 'RESTART_STEP' in extraOptions:
            return extraOptions

        controls = {}
        if self.rhos is not None and len(self.rhos) > 0:
            controls = dict(self.inputs_files[self.rhos[0]].controls)
        if code_settings is not None:
            try:
                controls = GACODEdefaults.addCGYROcontrol(code_settings)
            except Exception as e:
                print(
                    f"\t- [preprocess] Could not resolve code_settings={code_settings!r} "
                    f"against input.cgyro.models.yaml ({e}); falling back to controls file only",
                    typeMsg="w",
                )

        def _as_list(key):
            src = extraOptions.get(key, controls.get(key, None))
            if src is None:
                return None, None
            if isinstance(src, (list, np.ndarray)):
                return [float(v) for v in src], src
            return [float(src)], src

        dt_list, dt_src = _as_list('DELTA_T')
        ps_list, ps_src = _as_list('PRINT_STEP')
        mt_list, mt_src = _as_list('MAX_TIME')

        if dt_list is None or ps_list is None or mt_list is None:
            return extraOptions
        if any(v <= 0 for v in dt_list + ps_list + mt_list):
            print(
                f"\t- [preprocess] Non-positive DELTA_T/PRINT_STEP/MAX_TIME; leaving RESTART_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        # Broadcast to common length.
        n = max(len(dt_list), len(ps_list), len(mt_list))
        def _bcast(lst):
            return lst * n if len(lst) == 1 else lst
        dt_list, ps_list, mt_list = _bcast(dt_list), _bcast(ps_list), _bcast(mt_list)
        if not (len(dt_list) == len(ps_list) == len(mt_list) == n):
            print(
                "\t- [preprocess] DELTA_T/PRINT_STEP/MAX_TIME length mismatch; leaving RESTART_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        n_outputs = [max(1, math.ceil(mt / (dt * ps))) for dt, ps, mt in zip(dt_list, ps_list, mt_list)]

        rs_src = extraOptions.get('RESTART_STEP', controls.get('RESTART_STEP', 0))
        rs_list = (
            [int(v) for v in rs_src]
            if isinstance(rs_src, (list, np.ndarray))
            else [int(rs_src)] * n
        )
        if len(rs_list) != n:
            rs_list = rs_list + [rs_list[-1]] * (n - len(rs_list)) if len(rs_list) < n else rs_list[:n]

        # Keep the user/controls value if it is a valid divisor of n_outputs
        # (fires at least once by end-of-run); otherwise coerce to n_outputs so
        # a single restart is written exactly at t = MAX_TIME.
        coerced = [
            rs if (0 < rs <= no and no % rs == 0) else no
            for rs, no in zip(rs_list, n_outputs)
        ]

        extraOptions = copy.deepcopy(extraOptions)
        # Preserve scalar shape if inputs were all scalar and all coerced agree.
        scalars = not any(isinstance(x, (list, np.ndarray)) for x in (dt_src, ps_src, mt_src, rs_src))
        if scalars and len(set(coerced)) == 1:
            extraOptions['RESTART_STEP'] = coerced[0]
        else:
            extraOptions['RESTART_STEP'] = coerced

        print(
            f"\t- [preprocess] RESTART_STEP set to {extraOptions['RESTART_STEP']} "
            f"(<= n_outputs = ceil(MAX_TIME/(DELTA_T*PRINT_STEP)) = {n_outputs}; "
            f"restart fires when mod(i_time, RESTART_STEP*PRINT_STEP) == 0)",
            typeMsg="i",
        )
        return extraOptions

    def _apply_cgyro_preprocessing(self, extraOptions):
        """
        Compute BOX_SIZE and N_RADIAL per rho from the caller-provided
        ky_min plus Q, S, RMIN from self.inputs_files[rho], and inject them
        (along with KY=ky_min) into a copy of extraOptions as per-rho arrays.
        """

        allowed_keys = {'ky_min', 'L_x', 'N_radial', 'min_box_size'}
        opts = dict(self._preprocess_options) if self._preprocess_options else {}
        unknown = set(opts) - allowed_keys
        if unknown:
            raise ValueError(
                f"[MITIM] Unknown preprocess_options keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed_keys)}"
            )
        ky_min_opt    = opts.get('ky_min', 0.1)
        L_x           = opts.get('L_x', 90.0)
        N_radial      = opts.get('N_radial', 256)
        min_box_size  = opts.get('min_box_size', 100)

        extraOptions = copy.deepcopy(extraOptions)

        for conflict_key in ('KY', 'BOX_SIZE', 'N_RADIAL'):
            if conflict_key in extraOptions:
                print(
                    f"\t- [preprocess] {conflict_key} was set in extraOptions; "
                    f"it will be overwritten by the preprocessing result",
                    typeMsg="w",
                )

        box_sizes = []
        n_radials = []
        ky_mins   = []

        print("\t- [preprocess] Computing BOX_SIZE and N_RADIAL per rho:")
        for i, rho in enumerate(self.rhos):
            input_rho = self.inputs_files[rho]
            q     = float(input_rho.plasma['Q'])
            shear = float(input_rho.plasma['S'])
            rmin  = float(input_rho.plasma['RMIN'])

            if isinstance(ky_min_opt, (list, np.ndarray)):
                ky_min = float(ky_min_opt[i])
            else:
                ky_min = float(ky_min_opt)

            box_size, n_radial_val = CGYROutils.compute_box_and_nradial(
                q=q,
                shear=shear,
                rmin=rmin,
                ky_min=ky_min,
                L_x=L_x,
                N_radial=N_radial,
                min_box_size=min_box_size,
            )

            print(
                f"\t\t* rho={rho:.4f}: q={q:.3f} s={shear:.3f} r/a={rmin:.3f} "
                f"KY={ky_min:.4f} -> BOX_SIZE={box_size} N_RADIAL={n_radial_val}",
                typeMsg="i",
            )

            box_sizes.append(box_size)
            n_radials.append(n_radial_val)
            ky_mins.append(ky_min)

        extraOptions['KY']       = ky_mins
        extraOptions['BOX_SIZE'] = box_sizes
        extraOptions['N_RADIAL'] = n_radials
        return extraOptions

    # Re-defined to make specific arguments explicit
    def read(
        self,
        tmin = 0.0,
        tmin_is_rel = True,
        minimal = False,
        last_tmin_for_linear = True,
        **kwargs
    ):

        super().read(
            tmin = tmin,
            tmin_is_rel = tmin_is_rel,
            minimal = minimal,
            last_tmin_for_linear = last_tmin_for_linear,
            **kwargs)

    def read_linear_scan(
        self,
        folder=None,
        preffix="scan",
        store_as_label=None,
        irho = 0,
        **kwargs
    ):
        '''
        Useful utility for when a folder contains subfolders like... scan0, scan1, scan2... with different ky
        '''
        
        if folder is None:
            folder = self.FolderGACODE
        
        main_label = kwargs.get('label', 'run1')
        del kwargs['label']
        
        # Get all folders inside "folder" that start with "preffix"
        subfolders = [subfolder for subfolder in Path(folder).glob(f"*{preffix}*") if subfolder.is_dir()]

        # ----------------------------------------------------------
        # Store in resutls
        # ----------------------------------------------------------

        # Store results in the form of {main_label}_KY_{subfolder}
        
        labels_in_results = []
        if len(subfolders) == 0:
            print(f"No subfolders found in {folder} with preffix {preffix}. Reading the folder directly.")
            labels_in_results.append(f'{main_label}_KY_scan0')
            self.read(label=labels_in_results[-1], folder=folder, **kwargs)
        else:   
            for subfolder in subfolders:
                labels_in_results.append(f'{main_label}_KY_{subfolder.name}')
                self.read(label=labels_in_results[-1], folder=subfolder, **kwargs)        

        # ----------------------------------------------------------
        # Make it a linear scan for the main label
        # ----------------------------------------------------------
        
        # Store special linear scan class as {main_label}
        
        labelsD = []
        for label in labels_in_results:
            parts = label.split('_')
            if len(parts) >= 3 and parts[-2] == "KY":
                # Extract the base name (scan1) and middle value (0.3/0.4)
                base_name = '_'.join(parts[0:-2])               
                labelsD.append(label)

        if store_as_label is not None:
            main_label = store_as_label

        self.results[main_label] = CGYROutils.CGYROlinear_scan(labelsD, self.results, irho=irho)

    # Redefined to remove potential large objects
    def save_pickle(self, file, **kwargs):

        class_to_store = super().prepare_for_save()

        # cgyrodata carries _thread.lock through its internal HDF/file handles,
        # which breaks copy.deepcopy. Temporarily detach each cgyrodata from the
        # live object, deepcopy the rest, then restore them so the in-memory
        # object is unchanged after save.
        stashed = []
        for key in class_to_store.results:
            for irho in range(len(class_to_store.results[key]['output'])):
                out = class_to_store.results[key]['output'][irho]
                if 'cgyrodata' in out.__dict__:
                    stashed.append((out, out.cgyrodata))
                    out.cgyrodata = None
        try:
            class_to_store = copy.deepcopy(class_to_store)
        finally:
            for out, data in stashed:
                out.cgyrodata = data

        super().save_pickle(file, class_to_store = class_to_store, **kwargs)
                
                
    def plot(
        self,
        labels=[""],
        fn=None,
        include_2D=True,
        common_colorbar=True):
        
        # If it has radii, we need to correct the labels
        labels = self._correct_rhos_labels(labels)
    
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook
            self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")
        else:
            self.fn = fn

        fig = self.fn.add_figure(label="Fluxes (time)")
        axsFluxes_t = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Fluxes (ky)")
        axsFluxes_ky = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Intensities (time)")
        axsIntensities = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (ky)")
        axsIntensities_ky = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (kx)")
        axsIntensities_kx = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Cross-phases (ky)")
        axsCrossPhases = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Turbulence (linear)")
        axsTurbulence = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
      
        create_ballooning = False
        for label in labels:
            if 'phi_ballooning' in self.results[label].__dict__:
                create_ballooning = True
            
        if create_ballooning:

            fig = self.fn.add_figure(label="Ballooning")
            axsBallooning = fig.subplot_mosaic(
                """
                135
                246
                """
                )
        else:
            axsBallooning = None
        
        
        if include_2D:
            axs2D = []
            for i in range(len(labels)):
                fig = self.fn.add_figure(label="Turbulence (2D), " + labels[i])
                
                mosaic = _2D_mosaic(4) # Plot 4 times by default
                
                axs2D.append(fig.subplot_mosaic(mosaic))
        
        fig = self.fn.add_figure(label="Inputs")
        axsInputs = fig.subplot_mosaic(
            """
            A
            B
            """
        )

        
        colors = GRAPHICStools.listColors()

        # Safety net: if one sub-plot method raises (e.g. a missing optional
        # CGYRO output file that the per-panel guards didn't cover), we do
        # NOT want it to abort the whole notebook build. Wrap each call in a
        # try/except that logs a warning and continues.
        def _safe_plot(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as _e:
                print(f"\t- {fn.__name__} failed ({_e}); skipping this figure and continuing", typeMsg='w')
                return None

        colorbars_all = []  # Store all colorbars for later use
        for j in range(len(labels)):

            _safe_plot(self.plot_fluxes,
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            _safe_plot(self.plot_fluxes_ky,
                axs=axsFluxes_ky,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            _safe_plot(self.plot_intensities_ky,
                axs=axsIntensities_ky,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,
            )
            _safe_plot(self.plot_intensities,
                axs=axsIntensities,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            _safe_plot(self.plot_intensities_kx,
                axs=axsIntensities_kx,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            _safe_plot(self.plot_turbulence,
                axs=axsTurbulence,
                label=labels[j],
                c=colors[j],
            )
            _safe_plot(self.plot_cross_phases,
                axs=axsCrossPhases,
                label=labels[j],
                c=colors[j],
            )
            if create_ballooning:
                _safe_plot(self.plot_ballooning,
                    axs=axsBallooning,
                    label=labels[j],
                    c=colors[j],
                )

            if include_2D:

                colorbars = _safe_plot(self.plot_2D,
                    axs=axs2D[j],
                    label=labels[j],
                )

                colorbars_all.append(colorbars)

            _safe_plot(self.plot_inputs,
                ax=axsInputs["A"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
                normalization_label= labels[0],  # Normalize to the first label
                only_plot_differences=len(labels) > 1,  # Only plot differences if there are multiple labels
            )

            _safe_plot(self.plot_inputs,
                ax=axsInputs["B"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
            )
            
        axsInputs["A"].axhline(
            1.0,
            color="k",
            ls="--",
            lw=2.0
        )
        
        GRAPHICStools.adjust_subplots(axs=axsInputs, vertical=0.4, horizontal=0.3)
        
        # Modify the colorbars to have a common range
        if include_2D and common_colorbar and len(colorbars_all) > 0:
            for var in ['phi', 'n', 'e']:
                min_val = np.inf
                max_val = -np.inf
                for ilabel in range(len(colorbars_all)):
                    cb = colorbars_all[ilabel][0][var]
                    vals = cb.mappable.get_clim()
                    min_val = min(min_val, vals[0])
                    max_val = max(max_val, vals[1])
                
                for ilabel in range(len(colorbars_all)):
                    for it in range(len(colorbars_all[ilabel])):
                        cb = colorbars_all[ilabel][it][var]
                        cb.mappable.set_clim(min_val, max_val)
                        cb.update_ticks()
                        #cb.set_label(f"{var} (common range)")

        # Back to the original labels before _correct_rhos_labels
        self.results = self.results_all

    def plot_inputs(self, ax = None, label="", c="b", ms = 10, normalization_label=None, only_plot_differences=False):
        
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(18, 9))

        rel_tol = 1e-2

        legadded = False
        for i, ikey in enumerate(self.results[label].params1D):
            
            z = self.results[label].params1D[ikey]
            
            if normalization_label is not None:
                z0 = self.results[normalization_label].params1D[ikey]
                zp = z/z0 if z0 != 0 else 0
                label_plot = f"{label} / {normalization_label}"
            else:
                label_plot = label
                zp = z

            if (not only_plot_differences) or (not np.isclose(z, z0, rtol=rel_tol)):
                ax.plot(ikey,zp,'o',markersize=ms,color=c,label=label_plot if not legadded else '')
                legadded = True

        if normalization_label is not None:
            if only_plot_differences:
                ylabel = f"Parameters (DIFFERENT by {rel_tol*100:.2f}%) relative to {normalization_label}"
            else:
                ylabel = f"Parameters relative to {normalization_label}"
        else:
            ylabel = "Parameters"

        ax.set_xlabel("Parameter")
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel(ylabel)
        GRAPHICStools.addDenseAxis(ax)
        if legadded:
            ax.legend(loc='best')

    def plot_intensities(self, axs = None, label= "cgyro1", c="b", addText=True):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
            
        ax = axs["A"]
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta \\phi/\\phi_0$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].t, self.results[label].apar_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}, $A_\\parallel$")
            ax.plot(self.results[label].t, self.results[label].bpar_rms_sumnr_sumn*100.0, '--', c=c, lw=2, label=f"{label}, $B_\\parallel$")
            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity fluctuations')
        

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["C"]
        try:
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_e/n_{e,0}/n_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Density intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["D"]
        try:
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_e/T_{e,0}/T_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Temperature intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))




        ax = axs["E"]
        try:
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Density intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["F"]
        try:
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/T_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Temperature intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].t, self.results[label].ni_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Density intensity fluctuations')


        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].t, self.results[label].Ti_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Temperature intensity fluctuations')


        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_ky(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].phi_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].phi_rms_sumnr_mean-self.results[label].phi_rms_sumnr_std, self.results[label].phi_rms_sumnr_mean+self.results[label].phi_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta\phi/\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].ky, self.results[label].apar_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].apar_rms_sumnr_mean-self.results[label].apar_rms_sumnr_std, self.results[label].apar_rms_sumnr_mean+self.results[label].apar_rms_sumnr_std, color=c, alpha=0.2)
            ax.plot(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean, '--', markersize=5, color=c, label=label+', $B_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean-self.results[label].bpar_rms_sumnr_std, self.results[label].bpar_rms_sumnr_mean+self.results[label].bpar_rms_sumnr_std, color=c, alpha=0.2)

            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta F_\parallel/F_{\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs. $k_\\theta\\rho_s$')
        
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["C"]
        try:
            ax.plot(self.results[label].ky, self.results[label].ne_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].ne_rms_sumnr_mean-self.results[label].ne_rms_sumnr_std, self.results[label].ne_rms_sumnr_mean+self.results[label].ne_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Electron temperature intensity
        ax = axs["D"]
        try:
            ax.plot(self.results[label].ky, self.results[label].Te_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].Te_rms_sumnr_mean-self.results[label].Te_rms_sumnr_std, self.results[label].Te_rms_sumnr_mean+self.results[label].Te_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["E"]
        try:
            ax.plot(self.results[label].ky, self.results[label].ni_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].ni_rms_sumnr_mean-self.results[label].ni_rms_sumnr_std, self.results[label].ni_rms_sumnr_mean+self.results[label].ni_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Ion temperature intensity
        ax = axs["F"]
        try:
            ax.plot(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean-self.results[label].Ti_rms_sumnr_std, self.results[label].Ti_rms_sumnr_mean+self.results[label].Ti_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].ni_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)


        # Ion temperature intensity
        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].Ti_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_kx(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta \\phi/\\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs kx')
        ax.legend(loc='best', prop={'size': 8},)
        ax.set_yscale('log')
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["C"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].kx, self.results[label].apar_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].bpar_rms_sumn_mean, '--', markersize=1.0, lw=1.0, color=c, label=label+', $B_\\parallel$ (mean)')

            ax.legend(loc='best', prop={'size': 8},)


        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs kx')
        ax.set_yscale('log')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["B"]
        try:
            ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
            ax.plot(self.results[label].kx, self.results[label].ne_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')
            ax.legend(loc='best', prop={'size': 8},)
            ax.set_yscale('log')
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs kx')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        # Electron temperature intensity
        ax = axs["D"]
        try:
            ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
            ax.plot(self.results[label].kx, self.results[label].Te_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')
            ax.legend(loc='best', prop={'size': 8},)
            ax.set_yscale('log')
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs kx')
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)


    def plot_cross_phases(self, axs = None, label= "cgyro1", c="b"):

        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
        m = GRAPHICStools.listmarkers()
            
        ax = axs["A"]
        try:
            ax.plot(self.results[label].ky, self.results[label].neTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].neTe_kx0_mean-self.results[label].neTe_kx0_std, self.results[label].neTe_kx0_mean+self.results[label].neTe_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n + kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_e-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_e-T_e$ cross-phase ($k_x=0$)')


        ax = axs["B"]
        try:
            ax.plot(self.results[label].ky, self.results[label].niTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].niTi_kx0_mean-self.results[label].niTi_kx0_std, self.results[label].niTi_kx0_mean+self.results[label].niTi_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n + kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_i-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_i-T_i$ cross-phase ($k_x=0$)')

        ax = axs["C"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phine_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phine_kx0_mean-self.results[label].phine_kx0_std, self.results[label].phine_kx0_mean+self.results[label].phine_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_e$ cross-phase ($k_x=0$)')

        ax = axs["D"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phini_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phini_kx0_mean-self.results[label].phini_kx0_std, self.results[label].phini_kx0_mean+self.results[label].phini_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ cross-phase ($k_x=0$)')


        ax = axs["E"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phiTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phiTe_kx0_mean-self.results[label].phiTe_kx0_std, self.results[label].phiTe_kx0_mean+self.results[label].phiTe_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_e$ cross-phase ($k_x=0$)')


        ax = axs["F"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phiTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phiTi_kx0_mean-self.results[label].phiTi_kx0_std, self.results[label].phiTi_kx0_mean+self.results[label].phiTi_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ cross-phase ($k_x=0$)')


        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].phiTi_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ (all) cross-phase ($k_x=0$)')


        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].phini_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ (all) cross-phase ($k_x=0$)')
        
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_ballooning(self, time = None, label="cgyro1", c="b", axs=None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                135
                246
                """
            )

        if time is None:
            time = np.min([self.results[label].tmin, self.results[label].tmax_fluct])
        
        it = np.argmin(np.abs(self.results[label].t - time))

        colorsC, _ = GRAPHICStools.colorTableFade(
            len(self.results[label].ky),
            startcolor=c,
            endcolor=c,
            alphalims=[1.0, 0.4],
        )

        ax = axs['1']
        for ky in range(len(self.results[label].ky)):
            for var, axsT in zip(
                ["phi_ballooning", "apar_ballooning", "bpar_ballooning"],
                [[axs['1'], axs['2']], [axs['3'], axs['4']], [axs['5'], axs['6']]],
            ):

                f = self.results[label].__dict__[var][:, it]
                y1 = np.real(f)
                y2 = np.imag(f)
                x = self.results[label].theta_ballooning / np.pi

                # Normalize
                y1_max = np.max(np.abs(y1))
                y2_max = np.max(np.abs(y2))
                y1 /= y1_max
                y2 /= y2_max

                ax = axsT[0]
                ax.plot(
                    x,
                    y1,
                    color=colorsC[ky],
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y1_max:.2e})",
                )
                ax = axsT[1]
                ax.plot(
                    x, 
                    y2, 
                    color=colorsC[ky], 
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y2_max:.2e})",
                )


        ax = axs['1']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta\\phi$)")
        ax.set_title("$\\delta\\phi$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([-2 * np.pi, 2 * np.pi])

        ax = axs['3']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta A\\parallel$)")
        ax.set_title("$\\delta A\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['5']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta B\\parallel$)")
        ax.set_title("$\\delta B\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['2']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta\\phi$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['4']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta A\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['6']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta B\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)


        for ax in [axs['1'], axs['3'], axs['5'], axs['2'], axs['4'], axs['6']]:
            ax.axvline(x=0, lw=0.5, ls="--", c="k")
            ax.axhline(y=0, lw=0.5, ls="--", c="k")
            
            
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_2D(self, label="cgyro1", axs=None, times = None):

        # plot_2D needs kxky_phi (always), kxky_n (MOMENT_PRINT_FLAG=1) and
        # kxky_e (MOMENT_PRINT_FLAG=1). If any of the underlying fluctuation
        # arrays is missing (typically because the user disabled the print
        # flags to save disk / retrieval time), skip cleanly instead of
        # aborting the whole plot chain.
        _res = self.results.get(label) if hasattr(self, 'results') else None
        if _res is None or not all(hasattr(_res, _a) for _a in ('phi', 'ne', 'Te')):
            print("\t- plot_2D skipped: needs phi/ne/Te (requires bin.cgyro.kxky_phi + kxky_n + kxky_e; enable MOMENT_PRINT_FLAG=1 / FIELD_PRINT_FLAG=1)", typeMsg='w')
            return

        if times is None:
            times = []
            
            number_times = len(axs)//3 if axs is not None else 4

            try:
                times = [self.results[label].t[-1-i*10] for i in range(number_times)]
            except IndexError:
                 times = [self.results[label].t[-1-i*1] for i in range(number_times)]

        if axs is None:

            mosaic = _2D_mosaic(len(times))

            plt.ion()
            fig = plt.figure(figsize=(18, 9))
            axs = fig.subplot_mosaic(mosaic)

        # Pre-calculate global min/max for each field type across all times
        phi_values = []
        n_values = []
        e_values = []
        
        for time in times:
            it = np.argmin(np.abs(self.results[label].t - time))
            
            # Get phi values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)
            phi_values.append(fp)
            
            # Get n values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)
            n_values.append(fp)
            
            # Get e values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)
            e_values.append(fp)
        
        # Calculate global ranges
        phi_max = np.max([np.max(np.abs(fp)) for fp in phi_values])
        phi_min, phi_max = -phi_max, +phi_max
        
        n_max = np.max([np.max(np.abs(fp)) for fp in n_values])
        n_min, n_max = -n_max, +n_max
        
        e_max = np.max([np.max(np.abs(fp)) for fp in e_values])
        e_min, e_max = -e_max, +e_max

        colorbars = []  # Store colorbar references
        # Now plot with consistent colorbar ranges
        for time_i, time in enumerate(times):
            
            print(f"\t- Plotting 2D turbulence for {label} at time {time}")
            
            it = np.argmin(np.abs(self.results[label].t - time))
            
            cfig = axs[str(time_i+1)].get_figure()
            
            # Phi plot
            ax = axs[str(time_i+1)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)

            cs1 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(phi_min,phi_max,(phi_max-phi_min)/256),cmap=plt.get_cmap('jet'))
            cphi = cfig.colorbar(cs1, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta\\phi/\\phi_0$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # N plot
            ax = axs[str(time_i+1+len(times))]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)

            cs2 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(n_min,n_max,(n_max-n_min)/256),cmap=plt.get_cmap('jet'))
            cn = cfig.colorbar(cs2, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta n_e/n_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # E plot
            ax = axs[str(time_i+1+len(times)*2)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)

            cs3 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(e_min,e_max,(e_max-e_min)/256),cmap=plt.get_cmap('jet'))
            ce = cfig.colorbar(cs3, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta E_e/E_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            # Store the colorbar objects with their associated contour plots
            colorbars.append({
                'phi': cphi,
                'n': cn,
                'e': ce
            })

        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.4, horizontal=0.3)

        return colorbars
        
    def _to_real_space(self, variable = 'kxky_phi', species = None, label="cgyro1", theta_plot = 0, it = -1):
        
        # from pygacode
        def maptoreal_fft(nr,nn,nx,ny,c):

            d = np.zeros([nx,nn],dtype=complex)
            for i in range(nr):
                p = i-nr//2
                if -p < 0:
                    k = -p+nx
                else:
                    k = -p
                d[k,0:nn] = np.conj(c[i,0:nn])
            f = np.fft.irfft2(d,s=[nx,ny],norm='forward')*0.5

            # Correct for half-sum
            f = 2*f

            return f

        # Real space
        nr = self.results[label].cgyrodata.n_radial
        nn = self.results[label].cgyrodata.n_n
        craw = self.results[label].cgyrodata.__dict__[variable]
        
        itheta = np.argmin(np.abs(self.results[label].theta_stored-theta_plot))
        if species is None:
            c = craw[:,itheta,:,it]
        else:
            c = craw[:,itheta,species,:,it]

        nx = self.results[label].cgyrodata.__dict__[variable].shape[0]
        ny = nx
        
        # Arrays
        x = np.arange(nx)*2*np.pi/nx
        y = np.arange(ny)*2*np.pi/ny
        f = maptoreal_fft(nr,nn,nx,ny,c)
        
        # Physical maxima
        ky1 = self.results[label].cgyrodata.ky[1] if len(self.results[label].cgyrodata.ky) > 1 else self.results[label].cgyrodata.ky[0]
        xmax = self.results[label].cgyrodata.length
        ymax = (2*np.pi)/np.abs(ky1)
        xp = x/(2*np.pi)*xmax
        yp = y/(2*np.pi)*ymax

        # Periodic extensions
        xp = np.append(xp,xmax)
        yp = np.append(yp,ymax)
        fp = np.zeros([nx+1,ny+1])
        fp[0:nx,0:ny] = f[:,:]
        fp[-1,:] = fp[0,:]
        fp[:,-1] = fp[:,0]
        
        return xp, yp, fp
        
    def plot_quick_linear(self, labels=["cgyro1"], fig=None):
 
        colors = GRAPHICStools.listColors()
        ls = GRAPHICStools.listLS()

        if fig is None:
            fig = plt.figure(figsize=(15,9))

        axs = fig.subplot_mosaic(
            """
            12
            34
            """
        )
            
        def _plot_linear_stability(axs, labels, label_base,col_lin ='b', start_cont=0):

            irho = self.results[label_base].irho

            for cont, label in enumerate(labels):
                c = self.results[label]['output'][irho]
                baseColor = colors[cont+start_cont+1]
                colorsC, _ = GRAPHICStools.colorTableFade(
                    len(c.ky),
                    startcolor=baseColor,
                    endcolor=baseColor,
                    alphalims=[1.0, 0.4],
                )

                ax = axs['1']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.g[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$, $r/a={c.roa:.2f}$",
                        ls = ls[irho]
                    )

                ax = axs['2']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.f[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$, $r/a={c.roa:.2f}$",
                        ls = ls[irho]
                    )

            roa = self.results[self.results[label_base].labels[0]]['output'][irho].roa

            GACODEplotting.plotTGLFspectrum(
                [axs['3'], axs['4']],
                abs(self.results[label_base].ky),
                self.results[label_base].g_mean,
                freq=self.results[label_base].f_mean,
                coeff=0.0,
                c=col_lin,
                ls="-",
                lw=1,
                label=f"r/a = {roa}",
                facecolors=colors,
                markersize=50,
                alpha=1.0,
                titles=["Growth Rate", "Real Frequency"],
                removeLow=1e-4,
                ylabel=True,
            )
            axs['3'].legend(loc='best', prop={'size': 8},)
            
            return cont

        co = -1
        for i,label0 in enumerate(labels):
            co = _plot_linear_stability(axs, self.results[label0].labels, label0, start_cont=co, col_lin=colors[i])

        ax = axs['1']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$\\gamma$ $(c_s/a)$")
        ax.set_title("Growth Rate")
        ax.set_xlim(left=0)
        ax.legend(loc='best', prop={'size': 8},)
        
        ax = axs['2']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.set_ylabel("$\\omega$ $(c_s/a)$")
        ax.set_title("Real Frequency")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_xlim(left=0)
        
        for ax in [axs['1'], axs['2'], axs['3'], axs['4']]:
            GRAPHICStools.addDenseAxis(ax)
        
        plt.tight_layout()

class CGYROinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.cgyro.controls",
            code="CGYRO",
            n_species='N_SPECIES',
        )

def _2D_mosaic(n_times):

    num_cols = n_times

    # Create the mosaic layout dynamically
    mosaic = []
    counter = 1
    for _ in range(3):
        row = []
        for _ in range(num_cols):
            row.append(str(counter))
            counter += 1
        mosaic.append(row)
        
    return mosaic