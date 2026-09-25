'''
Warm-start staging for the gyrokinetic codes (CGYRO, GX): which prior iteration each radius
restarts from, and the per-rho restart files handed to `additional_files_to_send`. The file is
the code's own (`_warm_start_file` of the simulation class: bin.cgyro.restart, gxplasma.restart.nc),
retrieved as <file>_<rho:.4f> and staged back under its plain name.

What every knob means, what the user must set for it, and the precedence rules live in
templates/namelist.portals.yaml (`transport.options.cgyro.run`, blocks `restart_from_folder`
and `restart_from_cases`; the GX block points there). That file is the single source of truth;
this module implements it.
'''

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mitim_tools.misc_tools.LOGtools import printMsg as print

# Predicted channel -> GB flux key, shared by the powerstate targets and fluxes_turb.json
_CHANNEL_TO_GB = {"te": "QeGB", "ti": "QiGB", "ne": "GeGB", "nZ": "GZGB", "w0": "MtGB"}

# Name CGYRO reads the warm-start blob from inside each rho subfolder (default of RestartChain)
_RESTART_DST = "bin.cgyro.restart"


@dataclass
class RestartPlan:
    '''
    Outcome of resolving the chain for one evaluation: the merged
    `additional_files_to_send` dict and the parent iteration each rho took its blob from.
    `files_per_rho` is None only when the caller passed None and nothing was staged.
    '''

    files_per_rho: dict = None
    sources: dict = field(default_factory=dict)
    mode: str = None
    evaluation_number: int = 0
    context_label: str = "Evaluation"
    payload: dict = None

    def write_json(self, folder, base_subfolder):
        '''
        Write the per-rho parent map to <folder>/<base_subfolder>/restart_sources.json and
        return the payload (None when nothing was written). The trace plotter
        (CGYROplot.load_restart_sources_for_iterations) reads it to offset the time axis of a
        warm-started iteration; rhos absent from `sources` are drawn as cold starts.
        '''
        if not self.sources:
            return None
        out_dir = Path(folder) / base_subfolder
        out_path = out_dir / "restart_sources.json"
        payload = {
            "mode": self.mode,
            "evaluation_number": int(self.evaluation_number),
            "context_label": self.context_label,
            "sources": {str(k): int(v) for k, v in self.sources.items()},
        }
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            print(f"\t- [restart] Could not write {out_path}: {e}", typeMsg='w')
            return None
        return payload

    @staticmethod
    def restore_json(folder, base_subfolder, payload):
        '''
        Put restart_sources.json back after run(): _run_prepare recreates <base_subfolder>/
        through askNewFolder, wiping the copy the resolver wrote. The payload is read from the
        simulation object, because the re-attach -> fresh-fallback path refreshes it there.
        '''
        if not payload:
            return
        out_path = Path(folder) / base_subfolder / "restart_sources.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            print(f"\t- [restart] Could not restore {out_path}: {e}", typeMsg='w')


class RestartChain:
    '''
    Resolves `restart_from_folder` (explicit directory) and `restart_from_cases`
    ("first" / "all" / "best") into per-rho (source, restart_file) tuples.

    CGYRO: only the binary is staged, never the companion out.cgyro.tag: that is what makes CGYRO
    treat the blob as a warm start (restart_flag=2, t reset to 0) instead of a true rewind.
    See the namelist for the full rationale and for the MAX_TIME convention that follows
    from it. GX: the staged gxplasma.restart.nc is picked up by restart_if_exists and GX.run
    turns t_max into the time added on top of it.
    '''

    MODES = ("first", "all", "best")

    def __init__(self, run_options, evaluation_number, folder, rho_locations,
                 base_subfolder="base_cgyro", plasma_subfolder=None, restart_file=_RESTART_DST, label="CGYRO"):
        self.run_options = run_options or {}
        # The code's restart file (retrieved per rho as <restart_file>_<rho:.4f>) and its log label
        self.restart_file = restart_file
        self.label = label
        # PORTALS sources the evaluation number from the Dakota-style filename, i.e. a *string*
        # in the Execution phase and an int during the SR initializer.
        try:
            self.evaluation_number = int(evaluation_number)
        except (TypeError, ValueError):
            self.evaluation_number = evaluation_number
        self.folder = Path(folder)
        self.rho_locations = list(rho_locations)
        self.base_subfolder = base_subfolder
        # In the SIMtools layout the per-plasma folder is a SIBLING of base_subfolder, so in
        # batched mode it REPLACES it as the source directory instead of nesting under it.
        self.plasma_subfolder = plasma_subfolder
        self.in_simple_relax = "initialization_simple_relax" in self.folder.parts
        self.context_label = "portals_sr_ev" if self.in_simple_relax else "Evaluation"

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    @property
    def active(self):
        '''restart_from_cases is set: this iteration's restart files are read back by later ones.'''
        return self.run_options.get("restart_from_cases") not in (None, "", "null") or bool(self.run_options.get("restart_from_first", False))

    def stage_explicit_folder(self, existing_files=None):
        '''
        `restart_from_folder`: one directory holding <restart_file>_<rho:.4f> for every
        predicted radius. Raises when the folder or any per-rho file is missing.
        '''
        restart_folder = self.run_options.get("restart_from_folder")
        if restart_folder in (None, ""):
            return existing_files

        restart_folder = Path(restart_folder).expanduser()
        if not restart_folder.is_dir():
            raise FileNotFoundError(
                f"[MITIM] {self.label} restart_from_folder does not exist or is not a directory: {restart_folder}"
            )

        print(f"\n- [{self.label} restart] Staging per-radius restart files from:\n\t{restart_folder}", typeMsg='i')

        resolved, missing = self._stage_rhos(restart_folder, existing_files)
        if missing:
            raise FileNotFoundError(
                f"[MITIM] {self.label} restart_from_folder is missing per-rho restart files: "
                f"{missing}. Expected one file per predicted radius, named "
                f"{self.restart_file}_<rho:.4f>, in {restart_folder}."
            )
        return resolved

    def resolve(self, existing_files=None, turb_target_GB=None):
        '''
        `restart_from_cases`. Returns a RestartPlan whose `files_per_rho` is the merged
        `additional_files_to_send` (the incoming one unchanged when the mode is null,
        `restart_from_folder` wins, or this is iteration 0) and whose `payload` is the
        restart_sources.json content, already on disk.
        '''
        mode = self._mode()
        if mode is None:
            return RestartPlan(files_per_rho=existing_files)

        if self.run_options.get("restart_from_folder") not in (None, ""):
            print(f"\t- [{self.label} restart] restart_from_folder is set; restart_from_cases={mode!r} ignored.", typeMsg='w')
            return RestartPlan(files_per_rho=existing_files)

        if self.evaluation_number == 0:
            print(
                f"\n- [{self.label} restart_from_cases={mode!r}] This is {self.context_label}.0 — no prior iteration to restart from.\n"
                f"\t  REMINDER: for subsequent iterations to resume from this one, every radius must\n"
                f"\t  write {self.restart_file} (CGYRO: RESTART_STEP in extraOptions; GX: save_for_restart,\n"
                f"\t  on by default) and {self.restart_file}_<rho:.4f> must stay on disk (see keep_files).",
                typeMsg='w',
            )
            return RestartPlan(files_per_rho=existing_files)

        if mode == "best":
            files, sources = self._resolve_best(existing_files, turb_target_GB)
        else:
            files, sources = self._resolve_uniform(mode, existing_files)

        plan = RestartPlan(
            files_per_rho=files, sources=sources, mode=mode,
            evaluation_number=self.evaluation_number, context_label=self.context_label,
        )
        plan.payload = plan.write_json(self.folder, self.base_subfolder)
        return plan

    def rerun_for_fresh_submission(self, gk_object, run_kwargs, turb_target_GB=None):
        '''
        Re-attach preserves the parent pick of the original submit by skipping the resolver.
        When it falls back to a fresh submission the chain has to be resolved after all,
        otherwise the job cold-starts while MAX_TIME (sized as warm-start-additional time) and
        the restored restart_sources.json still claim a warm start. Mutates
        run_kwargs["additional_files_to_send"] and refreshes gk_object._restart_sources_payload.
        '''
        plan = self.resolve(run_kwargs.get("additional_files_to_send"), turb_target_GB)
        if plan.files_per_rho is not None:
            run_kwargs["additional_files_to_send"] = plan.files_per_rho
        gk_object._restart_sources_payload = plan.payload
        return plan

    @staticmethod
    def turbulent_target_GB(power_transport, plasma_index=0):
        '''
        Per-channel target the turbulence has to carry, in GB units:

            target_GB - neoclassical_GB

        for every active channel, as a per-rho array with rho=0 stripped to match the
        fluxes_*.json indexing. Targets come from powerstate.plasma (populated by
        calculateTargets()), neoclassical from the evaluate_neoclassical() that ran just
        before in TRANSPORTtools.evaluate(). Channels whose neoclassical array is missing or
        shape-mismatched fall back to neoc=0. `plasma_index` picks the reference plasma in
        batched mode (0, matching the plasma-0 convention of the batched chain).
        '''
        out = {}
        for ch in (getattr(power_transport.powerstate, "predicted_channels", []) or []):
            gb_key = _CHANNEL_TO_GB.get(ch)
            if gb_key is None:
                continue
            target_t = power_transport.powerstate.plasma.get(gb_key)
            if target_t is None:
                continue
            # powerstate.plasma["{Q*GB}"] is [batch, nrho] with rho=0 at index 0
            if hasattr(target_t, "detach"):
                target_arr = target_t[plasma_index, 1:].detach().cpu().numpy()
            else:
                target_arr = np.asarray(target_t)[plasma_index, 1:]

            neoc_val = getattr(power_transport, f"{gb_key}_neoc", None)
            neoc_arr = np.zeros_like(target_arr)
            if neoc_val is not None:
                arr = np.asarray(neoc_val)
                if arr.ndim >= 2:
                    arr = arr[plasma_index] if arr.shape[0] > plasma_index else arr[0]
                if arr.shape == target_arr.shape:
                    neoc_arr = arr.astype(target_arr.dtype, copy=False)

            out[gb_key] = target_arr - neoc_arr
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mode(self):
        '''`restart_from_cases`, with the retired `restart_from_first: true` mapped onto "first".'''
        mode = self.run_options.get("restart_from_cases")
        if mode in (None, "", "null"):
            if not self.run_options.get("restart_from_first", False):
                return None
            print(
                f"\t- [{self.label} restart] `restart_from_first: true` is deprecated; "
                "mapping to `restart_from_cases: \"first\"`. Please update your namelist.",
                typeMsg='w',
            )
            return "first"

        mode_lower = str(mode).lower()
        if mode_lower not in self.MODES:
            print(
                f"\t- [{self.label} restart] Unknown restart_from_cases={mode!r}; expected one of "
                "null / \"first\" / \"all\" / \"best\". Ignoring.",
                typeMsg='w',
            )
            return None
        return mode_lower

    def _sibling(self, iteration):
        return f"portals_sr_ev_{iteration}" if self.in_simple_relax else f"Evaluation.{iteration}"

    def _eval_root(self, iteration):
        return self.folder.parent.parent / self._sibling(iteration) / self.folder.name

    def _source_subfolder(self):
        return self.plasma_subfolder or self.base_subfolder

    def _stage_rhos(self, source_folder, existing_files):
        '''Per-rho (src, dst) tuples for every rho whose restart file is in source_folder.'''
        resolved = dict(existing_files) if existing_files else {}
        missing = []
        for rho in self.rho_locations:
            bin_file = source_folder / f"{self.restart_file}_{rho:.4f}"
            if not bin_file.is_file():
                missing.append(bin_file.name)
                continue
            print(f"\t  rho={rho:.4f}: {bin_file.name} -> {self.restart_file} (warm start)", typeMsg='i')
            resolved.setdefault(float(rho), []).append((bin_file, self.restart_file))
        return resolved, missing

    def _resolve_uniform(self, mode, existing_files):
        '''
        "first" (source iter 0) and "all" (source iter N-1): every rho takes the same parent.
        A missing source folder or a missing per-rho restart file is FATAL — a silent partial restart
        produces one-rho-cold/others-warm ensembles that cannot be diagnosed downstream.
        '''
        source_iter = 0 if mode == "first" else (self.evaluation_number - 1)
        source_sibling = self._sibling(source_iter)
        source_subfolder = self._source_subfolder()
        source_folder = self._eval_root(source_iter) / source_subfolder

        if not source_folder.is_dir():
            raise FileNotFoundError(
                f"[MITIM] {self.label} restart_from_cases={mode!r} was requested for "
                f"{self.context_label}.{self.evaluation_number}, but the source {source_subfolder} "
                f"folder does not exist:\n\t{source_folder}\n"
                f"Clear restart_from_cases in the namelist to run without restart."
            )

        print(
            f"\n- [{self.label} restart_from_cases={mode!r}] {self.context_label}.{self.evaluation_number} "
            f"will restart from {source_sibling}:\n\t{source_folder}",
            typeMsg='i',
        )

        resolved, missing = self._stage_rhos(source_folder, existing_files)
        if missing:
            raise FileNotFoundError(
                f"[MITIM] {self.label} restart_from_cases={mode!r} was requested for "
                f"{self.context_label}.{self.evaluation_number}, but restart files are missing in "
                f"{source_sibling}: {missing}.\n"
                "Check that the source iteration wrote them (CGYRO: RESTART_STEP in extraOptions; "
                "GX: save_for_restart) and that keep_files did not unlink them. Clear restart_from_cases "
                "in the namelist to run without restart."
            )

        return resolved, {f"{rho:.4f}": source_iter for rho in self.rho_locations}

    def _candidates(self, turb_target_GB):
        '''Prior iterations whose fluxes_turb.json exists and carries every active channel.'''
        candidates = []
        for i in range(self.evaluation_number):
            eval_root = self._eval_root(i)
            json_path = eval_root / "fluxes_turb.json"
            if self.plasma_subfolder and not json_path.is_file():
                # Batched evaluations write per-plasma JSON pairs; use plasma 0, matching the
                # plasma-0 reference convention of the batched chain.
                json_path = eval_root / "plasma_0" / "fluxes_turb.json"
            if not json_path.is_file():
                continue
            try:
                with open(json_path, "r") as f:
                    flux_mean = json.load(f).get("fluxes_mean", {})
            except (OSError, ValueError) as e:
                print(f"\t- [{self.label} restart 'best'] Could not load {json_path}: {e}; skipping iter {i}.", typeMsg='w')
                continue
            missing_keys = [k for k in turb_target_GB if k not in flux_mean]
            if missing_keys:
                print(
                    f"\t- [{self.label} restart 'best'] {self.context_label}.{i} fluxes_turb.json missing channels "
                    f"{missing_keys}; skipping as candidate.",
                    typeMsg='w',
                )
                continue
            candidates.append((i, eval_root, flux_mean))
        return candidates

    def _resolve_best(self, existing_files, turb_target_GB):
        '''
        Per-rho selection: the prior iteration whose turbulent flux at that rho is closest to
        the current turbulent target, by L2 over the active channels in RAW GB units (the heat
        channels, being one to two orders of magnitude larger, therefore decide the pick).
        Ties go to the higher iteration, whose profile is closer to the current iterate.
        Missing files are non-fatal and per-rho: a rho with no usable candidate cold-starts.
        '''
        if not turb_target_GB:
            print(
                f"\t- [{self.label} restart_from_cases='best'] No current turbulent target was provided "
                "(active channels empty or target assembly failed). No restart applied.",
                typeMsg='w',
            )
            return existing_files, {}

        candidates = self._candidates(turb_target_GB)
        if not candidates:
            print(
                f"\t- [{self.label} restart_from_cases='best'] No prior iteration in "
                f"[0, {self.evaluation_number - 1}] had a usable fluxes_turb.json; cold start.",
                typeMsg='w',
            )
            return existing_files, {}

        print(
            f"\n- [{self.label} restart_from_cases='best'] {self.context_label}.{self.evaluation_number} per-rho selection "
            f"({len(candidates)} candidate iter(s); active channels: {sorted(turb_target_GB.keys())}):",
            typeMsg='i',
        )

        resolved = dict(existing_files) if existing_files else {}
        cold_started_rhos = []
        chosen_sources = {}
        for rho_idx, rho in enumerate(self.rho_locations):
            per_rho = []
            for i, eval_root, flux_mean in candidates:
                bin_file = eval_root / self._source_subfolder() / f"{self.restart_file}_{rho:.4f}"
                if not bin_file.is_file():
                    continue
                sq_sum = sum(
                    (float(flux_mean[ch_key][rho_idx]) - float(target_arr[rho_idx])) ** 2
                    for ch_key, target_arr in turb_target_GB.items()
                )
                per_rho.append((sq_sum ** 0.5, i, bin_file))

            if not per_rho:
                cold_started_rhos.append(rho)
                print(f"\t  rho={rho:.4f}: cold start (no prior iter had restart file + json for this radius)", typeMsg='w')
                continue

            # (distance asc, iter desc) so the higher iter wins on ties
            per_rho.sort(key=lambda t: (t[0], -t[1]))
            chosen_dist, chosen_iter, chosen_bin = per_rho[0]
            print(f"\t  rho={rho:.4f} -> {self._sibling(chosen_iter)} (d={chosen_dist:.3g})", typeMsg='i')
            resolved.setdefault(float(rho), []).append((chosen_bin, self.restart_file))
            chosen_sources[f"{rho:.4f}"] = chosen_iter

        if cold_started_rhos:
            print(
                f"\t- [{self.label} restart 'best'] {len(cold_started_rhos)} of {len(self.rho_locations)} "
                f"rho(s) cold-started: {[f'{r:.4f}' for r in cold_started_rhos]}. Check that "
                f"prior iterations wrote and kept {self.restart_file}_<rho:.4f>.",
                typeMsg='w',
            )

        return resolved, chosen_sources
