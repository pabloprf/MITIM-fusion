import json
from matplotlib.pylab import f
import torch
import numpy as np
from functools import partial
import copy
import shutil
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools, NEOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def write_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
    '''
    For tracking and reproducibility (e.g. external runs), we want to write a json file
    containing the simulation results. JSON should look like:
    
    {
        'fluxes_mean': 
            {
                'QeGB': ...
                'QiGB': ...
                'GeGB': ...
                'GZGB': ...
                'MtGB': ...
                'QieGB': ...
            },
        'fluxes_stds': 
            {
                'QeGB': ...
                'QiGB': ...
                'GeGB': ...
                'GZGB': ...
                'MtGB': ...
                'QieGB': ...
            },
        'additional_info': {
                'rho': rho.tolist(),
            }
    }
    '''
    
    write_json_from_variables = self._write_json_from_variables_turb if suffix == 'turb' else self._write_json_from_variables_neoc
    
    if self.folder.exists() and write_json_from_variables:
        
        with open(self.folder / file_name, 'w') as f:

            fluxes_mean = {}
            fluxes_stds = {}

            for var in ['QeGB', 'QiGB', 'GeGB', 'GZGB', 'MtGB']:
                fluxes_mean[var] = self.__dict__[f"{var}_{suffix}"].tolist()
                fluxes_stds[var] = self.__dict__[f"{var}_{suffix}_stds"].tolist()

            try:
                var = 'QieGB'
                fluxes_mean[var] = self.__dict__[f"{var}_{suffix}"].tolist()
                fluxes_stds[var] = self.__dict__[f"{var}_{suffix}_stds"].tolist()
            except KeyError:
                # NEO file may not have it
                pass

            json_dict = {
                'fluxes_mean': fluxes_mean,
                'fluxes_stds': fluxes_stds,
                'additional_info': {
                    'rho': self.powerstate.plasma["rho"][0, 1:].cpu().numpy().tolist(),
                    'roa': self.powerstate.plasma["roa"][0, 1:].cpu().numpy().tolist(),
                    'Qgb': self.powerstate.plasma["Qgb"][0, 1:].cpu().numpy().tolist(),
                    'aLte': self.powerstate.plasma["aLte"][0, 1:].cpu().numpy().tolist(),
                    'aLti': self.powerstate.plasma["aLti"][0, 1:].cpu().numpy().tolist(),
                    'aLne': self.powerstate.plasma["aLne"][0, 1:].cpu().numpy().tolist(),
                }
            }

            json.dump(json_dict, f, indent=4)

        print(f"\t* Written JSON with {suffix} information to {IOtools.clipstr(self.folder / file_name)}")
        
    else:
        
        print(f"\t* Folder {self.folder} does not exist, cannot write {file_name}", typeMsg='w')

class power_transport:

    def __init__(self, powerstate, name = "test", folder = "~/scratch/", evaluation_number = 0):

        self.name = name
        self.folder = IOtools.expandPath(folder)
        self.evaluation_number = evaluation_number
        self.powerstate = powerstate

        self.transport_evaluator_options  = self.powerstate.transport_options["options"]
        self.cold_start                   = self.powerstate.transport_options["cold_start"]

        # Model results is None by default, but can be assigned in evaluate
        self.model_results = None

        # By default, write the json files after evaluating the variables (will be changed in gyrokinetic "prep" run mode)
        self._write_json_from_variables_turb = True
        self._write_json_from_variables_neoc = True

        # Multi-fidelity scaffolding — a single-fidelity run keeps fidelity_level=0 and the
        # active-options keys unset; the portals_transport_model dispatcher sets these just
        # before calling a backend so backend modules can index transport_evaluator_options
        # by the active instance name rather than the hardcoded code name.
        self.fidelity_level = 0
        self._active_turb_options_key = None
        self._active_neo_options_key = None

        # ----------------------------------------------------------------------------------------
        # labels for plotting
        # ----------------------------------------------------------------------------------------

        self.powerstate.labelsFluxes = {
            "te": "$Q_e$ ($MW/m^2$)",
            "ti": "$Q_i$ ($MW/m^2$)",
            "ne": "$Q_{conv}$ ($MW/m^2$)",
            "nZ": "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)",
            "w0": "$M_T$ ($J/m^2$)",
        }

    def evaluate(self):

        # Copy the input.gacode files to the output folder
        self._profiles_to_store()

        '''
        ******************************************************************************************************
        Cache short-circuit.

        If the target folder already contains both fluxes_turb.json and fluxes_neoc.json, trust them
        and skip the turbulence / neoclassical backends entirely — go straight to the populate +
        postprocess tail of this method. This is what makes initialization_simple_relax's SR copy
        actually save work at BO time under *both* subprocess and in-process TGLF: the write_json
        hook (decorating evaluate_turbulence / evaluate_neoclassical) wrote the JSON pair at every
        SR step regardless of how transport was executed, and the SR copy loop already carries that
        pair into Execution/Evaluation.{i}/transport_simulation_folder. The subprocess path used to
        rely on cold_start_checker matching out.tglf.gbflux_{rho:.4f} inside the same folder, which
        the in-process path never writes — so a pure JSON-pair gate at this level subsumes the
        lower-level reuse path and fixes both.
        ******************************************************************************************************
        '''
        cache_hit = (
            (self.folder / 'fluxes_turb.json').exists()
            and (self.folder / 'fluxes_neoc.json').exists()
        )

        batched = (
            getattr(self.powerstate, "batch_size", 0) or 0
        ) > 1

        # Initialize all GB flux arrays to zeros up front. The backends overwrite
        # whichever fluxes they compute; anything left untouched stays at zero (e.g.
        # QieGB_neoc, which NEO never produces). For the batched path the 2D (N, nrho)
        # arrays written by the batched backends overwrite these 1D zeros; the in-memory
        # fallback in _populate_from_json then broadcasts them correctly.
        nrho_m1 = self.powerstate.plasma['rho'].shape[-1] - 1
        for var in ['QeGB', 'QiGB', 'GeGB', 'GZGB', 'MtGB', 'QieGB']:
            for suffix in ['turb', 'neoc']:
                for suffix0 in ['', '_stds']:
                    self.__dict__[f"{var}_{suffix}{suffix0}"] = torch.zeros(nrho_m1)

        if cache_hit:
            print(
                f"\t- Reusing cached flux JSONs from {IOtools.clipstr(self.folder)} "
                f"(SR copy or previous evaluation); skipping turbulence / neoclassical runs",
                typeMsg='i',
            )
        elif batched:
            # Multi-plasma simple-relax path: fan N plasmas through run_over_plasmas in one
            # parallel submission via _evaluate_batched. See docstring on that method.
            self._evaluate_batched()
        else:
            '''
            ******************************************************************************************************
            Evaluate neoclassical and turbulent transport (*in GB units*).
            These functions use a hook to write the .json files to communicate the results to powerstate.plasma
            ******************************************************************************************************
            '''

            neoclassical = self.evaluate_neoclassical()
            turbulence = self.evaluate_turbulence()

        '''
        ******************************************************************************************************
        From the json to powerstate.plasma and GB to real units transformation
        ******************************************************************************************************
        '''
        self._populate_from_json(file_name = 'fluxes_turb.json', suffix= 'turb')
        self._populate_from_json(file_name = 'fluxes_neoc.json', suffix= 'neoc')

        # Some codes would like a stable correction
        self._stable_correction(self.transport_evaluator_options)

        '''
        ******************************************************************************************************
        Post-process the data: add turb and neoc, tensorize and transformations
        ******************************************************************************************************
        '''
        self._postprocess()

    def _evaluate_batched(self):
        '''
        Multi-plasma (batched) evaluate path, reached from evaluate() when
        self.powerstate.batch_size > 1 — today that is the PORTALS parallel simple-relax
        initialization, which asks one flux_match call to walk N trajectories in lockstep.

        The responsibility split:

            - Write one input.gacode per batch element under
              self.folder / f"plasma_{b}" / "input.gacode" by deepcopy + _repeat_tensors(
              batch_size=1, positionToUnrepeat=b) on self.powerstate, then call
              from_powerstate(...) to materialise a gacode_state and its input.gacode file.
            - Assemble list_of_states (one per plasma) and hand it to the batched
              backend dispatchers evaluate_neoclassical_batched / evaluate_turbulence_batched,
              which route to neo_model._evaluate_neo_batched / tglf_model._evaluate_tglf_batched
              respectively. Those methods run one `tglf.run_over_plasmas` / `neo.run_over_plasmas`
              call, reap per-plasma results, and populate self.QeGB_{turb,neoc}, ... as
              (N, nrho) numpy arrays.
            - Suppress the bundled-folder write_json decorators (via
              _write_json_from_variables_{turb,neoc} = False) so we do not emit a 2D
              fluxes_*.json at self.folder — that would confuse the single-plasma
              _populate_from_json reader that fires at BO time via the cache-hit gate.
            - Write one single-plasma fluxes_turb.json + fluxes_neoc.json pair under each
              self.folder / f"plasma_{b}" / using the *existing* 1D schema. These are the
              files the SR copy loop in initialization_simple_relax moves into
              Execution/Evaluation.{i}/transport_simulation_folder, and they are what the
              evaluate() cache-hit gate picks up at BO time for zero-work reuse.

        The in-memory (N, nrho) arrays are then fed through _populate_from_json's
        file-missing fallback (which now uses [:, 1:] broadcasting) and _postprocess
        (which now detects 1D vs 2D inputs), so the caller sees (N, nrho+1) batched
        torch tensors in self.powerstate.plasma[Q*_tr_*] exactly like the single-plasma
        path does for its own (1, nrho+1) case.
        '''

        N = int(self.powerstate.batch_size)

        # (1) Resolve per-side postproc fns once. Same logic as _modify_profiles:
        # per-model override beats the namelist global; no-neo collapses to fast path.
        turb_name = self._resolve_instance_name(self.turbulence_model)
        fn_turb = self._resolve_postproc_fun(turb_name)
        if self.neoclassical_model is not None:
            neo_name = self._resolve_instance_name(self.neoclassical_model)
            fn_neo = self._resolve_postproc_fun(neo_name)
        else:
            neo_name = None
            fn_neo = fn_turb
        split_path = fn_turb is not fn_neo

        # (2) Materialise N per-plasma input.gacodes + gacode_state profile objects.
        # On the split path we also materialise per-plasma input.gacode.turb / .neo
        # files and build two parallel lists. Per-side impurity-position resolution
        # is done once on plasma 0 (all plasmas share the same species structure).
        list_of_states_turb = []
        list_of_states_neo = []
        per_plasma_folders = []
        baseline_species = None
        for b in range(N):
            sub_folder = self.folder / f"plasma_{b}"
            sub_folder.mkdir(parents=True, exist_ok=True)
            per_plasma_folders.append(sub_folder)

            sub_powerstate = copy.deepcopy(self.powerstate)
            sub_powerstate._repeat_tensors(batch_size=1, positionToUnrepeat=b)

            gacode_file = sub_folder / "input.gacode"
            sub_profile = sub_powerstate.copy_state().from_powerstate(
                write_input_gacode=gacode_file,
                postprocess_input_gacode=self.powerstate.transport_options.get("applyCorrections", True),
                rederive_profiles=True,
                insert_highres_powers=True,
            )

            if b == 0:
                baseline_species = list(PROFILEStools.gacode_state(gacode_file).Species)
            baseline_pos = self.powerstate.impurityPosition

            if not split_path:
                # Fast path — single postproc applied to gacode_file in place.
                if fn_turb is not None:
                    sub_profile, new_pos = self._apply_postproc_and_resolve_impurity(
                        gacode_file, fn_turb,
                        baseline_species=baseline_species, baseline_pos=baseline_pos,
                    )
                    if b == 0 and new_pos != baseline_pos:
                        print(f"\t- Impurity position has changed from {baseline_pos} to {new_pos}", typeMsg="i")
                        self.powerstate.impurityPosition_transport = new_pos
                list_of_states_turb.append(sub_profile)
                list_of_states_neo.append(sub_profile)  # alias — same object
            else:
                # Split path — write per-side files and apply each fn to its own copy.
                file_turb = sub_folder / "input.gacode.turb"
                file_neo = sub_folder / "input.gacode.neo"
                shutil.copy2(gacode_file, file_turb)
                shutil.copy2(gacode_file, file_neo)

                state_turb, pos_turb = self._apply_postproc_and_resolve_impurity(
                    file_turb, fn_turb,
                    baseline_species=baseline_species, baseline_pos=baseline_pos,
                    side_label="turb",
                )
                state_neo, pos_neo = self._apply_postproc_and_resolve_impurity(
                    file_neo, fn_neo,
                    baseline_species=baseline_species, baseline_pos=baseline_pos,
                    side_label="neo",
                )
                # Canonical input.gacode mirrors the turb side so downstream
                # consumers (per-plasma JSON, plotting) see one consistent file.
                shutil.copy2(file_turb, gacode_file)
                list_of_states_turb.append(state_turb)
                list_of_states_neo.append(state_neo)

                if b == 0:
                    print(f"\t- [batched] Per-model profiles postprocessing detected:", typeMsg="i")
                    print(f"\t    turb ({turb_name!r}): {fn_turb}", typeMsg="i")
                    print(f"\t    neo  ({neo_name!r}):  {fn_neo}", typeMsg="i")
                    self.powerstate.profiles_transport_turb = state_turb
                    self.powerstate.profiles_transport_neo = state_neo
                    self.powerstate.impurityPosition_transport_turb = pos_turb
                    self.powerstate.impurityPosition_transport_neo = pos_neo
                    self.powerstate.impurityPosition_transport = pos_turb
                    if pos_turb != baseline_pos:
                        print(f"\t- [turb] Impurity position has changed from {baseline_pos} to {pos_turb}", typeMsg="i")
                    if pos_neo != pos_turb:
                        print(f"\t- [neo] Impurity position differs from turb: turb={pos_turb}, neo={pos_neo}", typeMsg="i")

        # (3) Suppress the bundled-folder write_json decorators — each per-plasma JSON pair
        # is written explicitly below so the top-level self.folder does not carry a 2D
        # fluxes_*.json that would confuse single-plasma readers (including the cache-hit
        # gate at BO time). The flag must be set *before* the decorator fires, which means
        # before evaluate_turbulence_batched / evaluate_neoclassical_batched run.
        self._write_json_from_variables_turb = False
        self._write_json_from_variables_neoc = False

        # (4) Run both backends once across all N plasmas via run_over_plasmas.
        # Each side gets its own list of post-processed gacode_states; on the
        # fast path the two lists alias the same objects so behavior is identical
        # to the legacy single-list dispatch.
        self.evaluate_neoclassical_batched(list_of_states_neo)
        self.evaluate_turbulence_batched(list_of_states_turb)

        # (4) Emit one single-plasma JSON pair per batch element for downstream reuse.
        rho_arr   = self.powerstate.plasma["rho"][0, 1:].cpu().numpy().tolist()
        roa_arr   = self.powerstate.plasma["roa"][0, 1:].cpu().numpy().tolist()
        for b, sub_folder in enumerate(per_plasma_folders):
            # Shared additional_info is taken from batch 0 — rho / roa / aLx do not vary
            # across the batch replication, they are just tiled copies.
            self._write_per_plasma_json(
                sub_folder,
                batch_index=b,
                suffix='turb',
                rho_arr=rho_arr,
                roa_arr=roa_arr,
            )
            self._write_per_plasma_json(
                sub_folder,
                batch_index=b,
                suffix='neoc',
                rho_arr=rho_arr,
                roa_arr=roa_arr,
            )

    def _write_per_plasma_json(self, sub_folder, batch_index, suffix, rho_arr, roa_arr):
        '''
        Per-plasma, single-plasma-shaped writer used by _evaluate_batched to carve one
        row out of the (N, nrho) self.QeGB_{turb,neoc} / ... arrays and drop it into
        sub_folder as a 1D fluxes_{turb,neoc}.json that matches the write_json schema.
        '''
        file_name = f"fluxes_{suffix}.json"
        fluxes_mean, fluxes_stds = {}, {}

        def _slice_batch(arr, b):
            '''Extract a 1D per-plasma slice from an array that may be 2D (N, nrho) or 1D (nrho).
            1D arrays are shared across all batches (e.g. the zero-init for QieGB_neoc which NEO
            never overwrites), so they are returned as-is regardless of batch index.'''
            arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
            return arr[b, :] if arr.ndim >= 2 else arr

        for var in ['QeGB', 'QiGB', 'GeGB', 'GZGB', 'MtGB', 'QieGB']:
            key_mean = f"{var}_{suffix}"
            key_std  = f"{var}_{suffix}_stds"
            if key_mean not in self.__dict__:
                continue
            try:
                fluxes_mean[var] = _slice_batch(self.__dict__[key_mean], batch_index).tolist()
                fluxes_stds[var] = _slice_batch(self.__dict__[key_std],  batch_index).tolist()
            except (KeyError, IndexError, TypeError):
                pass

        json_dict = {
            'fluxes_mean': fluxes_mean,
            'fluxes_stds': fluxes_stds,
            'additional_info': {
                'rho': rho_arr,
                'roa': roa_arr,
                'Qgb':  self.powerstate.plasma["Qgb"][0, 1:].cpu().numpy().tolist(),
                'aLte': self.powerstate.plasma["aLte"][0, 1:].cpu().numpy().tolist(),
                'aLti': self.powerstate.plasma["aLti"][0, 1:].cpu().numpy().tolist(),
                'aLne': self.powerstate.plasma["aLne"][0, 1:].cpu().numpy().tolist(),
            },
        }

        with open(sub_folder / file_name, 'w') as f:
            json.dump(json_dict, f, indent=4)
        print(
            f"\t* Written per-plasma JSON with {suffix} information to "
            f"{IOtools.clipstr(sub_folder / file_name)}"
        )

    def _postprocess(self):
        '''
        Curate information for the powerstate (e.g. add models, add batch dimension, rho=0.0, and tensorize)
        Before calling this function, the powerstate.plasma should have the following variables:
            'QeMWm2_tr_X', 'QiMWm2_tr_X', 'Ge1E20m2_tr_X', 'GZ1E20m2_tr_X', 'MtJm2_tr_X', 'QieMWm3_tr_X'
        where X = 'turb' or 'neoc'
        and also the corresponding _stds versions
        '''

        variables = ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']

        for variable in variables:
            for suffix in ['_tr_turb', '_tr_turb_stds', '_tr_neoc', '_tr_neoc_stds']:

                arr = self.powerstate.plasma[f"{variable}{suffix}"]
                # _populate_from_json hands us either a numpy ndarray (single plasma fallback)
                # or a torch tensor (JSON-on-disk read path, or 2D (N,nrho) from the
                # multi-plasma batched in-memory fallback). Normalise to torch first, then
                # only unsqueeze(0) if the array is still 1D (single-plasma case). For a
                # pre-2D array (batched SR path) we keep the explicit N batch axis and pad
                # the rho=0 column per batch row.
                if not isinstance(arr, torch.Tensor):
                    arr = torch.as_tensor(arr)
                arr = arr.to(self.powerstate.dfT)
                if arr.dim() == 1:
                    arr = arr.unsqueeze(0)

                # Pad with zeros at rho=0.0 — shape (N, 1) matching the current batch size
                arr = torch.cat((
                    torch.zeros((arr.shape[0], 1), dtype=arr.dtype, device=arr.device),
                    arr,
                ), dim=1)

                self.powerstate.plasma[f"{variable}{suffix}"] = arr

        # -----------------------------------------------------------
        # Sum the turbulent and neoclassical contributions
        # -----------------------------------------------------------
        
        variables = ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2']
        
        for variable in variables:
            self.powerstate.plasma[f"{variable}_tr"] = self.powerstate.plasma[f"{variable}_tr_turb"] + self.powerstate.plasma[f"{variable}_tr_neoc"]

        # ---------------------------------------------------------------------------------
        # Convective fluxes (& Re-scale the GZ flux by the original impurity concentration)
        # ---------------------------------------------------------------------------------
        
        mapper_convective = {
            'Ce': 'Ge1E20m2',
            'CZ': 'GZ1E20m2',
        }
        
        for key in mapper_convective.keys():
            for tt in ['','_turb', '_turb_stds', '_neoc', '_neoc_stds']:
                
                mult = 1/self.powerstate.fImp_orig if key == 'CZ' else 1.0
                
                self.powerstate.plasma[f"{key}_tr{tt}"] = PLASMAtools.convective_flux(
                    self.powerstate.plasma["te"],
                    self.powerstate.plasma[f"{mapper_convective[key]}_tr{tt}"]
                ) * mult
           
    def produce_profiles(self):
        # Only add self._produce_profiles() if it's needed (e.g. full TGLF), otherwise this is somewhat expensive
        # (e.g. for flux matching of analytical models)
        pass

    def _produce_profiles(self,derive_quantities=True):

        self.applyCorrections = self.powerstate.transport_options["applyCorrections"]

        # Write this updated profiles class (with parameterized profiles and target powers)
        self.file_profs = self.folder / "input.gacode"

        powerstate_detached = self.powerstate.copy_state()

        self.powerstate.profiles = powerstate_detached.from_powerstate(
            write_input_gacode=self.file_profs,
            postprocess_input_gacode=self.applyCorrections,
            rederive_profiles = derive_quantities,        # Derive quantities so that it's ready for analysis and plotting later
            insert_highres_powers = derive_quantities,    # Insert powers so that Q, Pfus and all that it's consistent when read later
        )

        self.powerstate.profiles_transport = copy.deepcopy(self.powerstate.profiles)

        self._modify_profiles()

    def _modify_profiles(self):
        '''
        Modify the profiles (e.g. lumping) before running the transport model.

        Two paths:
          - **Fast path**: turbulence and neoclassical resolve to the same
            postprocessing callable (today's default for any namelist that
            hasn't added per-model overrides). Apply once and leave the
            canonical profiles_transport / impurityPosition_transport as
            the single source of truth — output is byte-identical to the
            prior single-postproc behavior.
          - **Split path**: the two sides resolve to different callables.
            Materialise input.gacode.turb and input.gacode.neo, apply each
            function to its own copy, and set per-side aliases on the
            powerstate (profiles_transport_turb/_neo,
            impurityPosition_transport_turb/_neo). The canonical
            profiles_transport / impurityPosition_transport still point at
            the turb side so downstream consumers (VGEN, finalize_evaluation,
            plotting) keep seeing one profile.

        The per-side resolution uses `_resolve_postproc_fun(instance_name)`
        which prefers `transport.options.<name>.profiles_postprocessing_fun`
        and falls back to the namelist-global `transport.profiles_postprocessing_fun`.
        '''

        # After producing the profiles, copy for future modifications
        self.file_profs_unmod = self.file_profs.parent / f"{self.file_profs.name}_unmodified"
        shutil.copy2(self.file_profs, self.file_profs_unmod)

        # Resolve active model names + their per-side postproc fns.
        # `_resolve_instance_name` requires fidelity_level to be set, which
        # STATEtools.calculateTransport guarantees before produce_profiles.
        turb_name = self._resolve_instance_name(self.turbulence_model)
        fn_turb = self._resolve_postproc_fun(turb_name)
        if self.neoclassical_model is not None:
            neo_name = self._resolve_instance_name(self.neoclassical_model)
            fn_neo = self._resolve_postproc_fun(neo_name)
        else:
            neo_name = None
            # No neo evaluator -> force fast path (the neo side is never read)
            fn_neo = fn_turb

        baseline_state = PROFILEStools.gacode_state(self.file_profs_unmod)
        baseline_species = list(baseline_state.Species)
        baseline_pos = self.powerstate.impurityPosition

        if fn_turb is fn_neo:
            # Fast path — single postproc applied once, canonical attributes
            # only. Byte-identical to the prior implementation.
            if fn_turb is not None:
                print(f"\t- Modifying input.gacode to run transport calculations based on {fn_turb}", typeMsg="i")
                self.powerstate.profiles_transport = fn_turb(self.file_profs)
            p_new = PROFILEStools.gacode_state(self.file_profs)
            impurity_of_interest = baseline_species[baseline_pos]
            try:
                impurityPosition_new = p_new.Species.index(impurity_of_interest)
            except ValueError:
                print(f"\t- Impurity {impurity_of_interest} not found in new profiles, keeping position {baseline_pos}", typeMsg="w")
                impurityPosition_new = baseline_pos
            if impurityPosition_new != baseline_pos:
                print(f"\t- Impurity position has changed from {baseline_pos} to {impurityPosition_new}", typeMsg="i")
                self.powerstate.impurityPosition_transport = impurityPosition_new
        else:
            # Split path — turbulence and neoclassical use different postprocs.
            print(f"\t- Per-model profiles postprocessing detected:", typeMsg="i")
            print(f"\t    turb ({turb_name!r}): {fn_turb}", typeMsg="i")
            print(f"\t    neo  ({neo_name!r}):  {fn_neo}", typeMsg="i")

            file_turb = self.file_profs.parent / f"{self.file_profs.name}.turb"
            file_neo = self.file_profs.parent / f"{self.file_profs.name}.neo"
            shutil.copy2(self.file_profs, file_turb)
            shutil.copy2(self.file_profs, file_neo)

            # Turb side (canonical). After applying, copy turb file back to
            # self.file_profs so downstream (VGEN, finalize_evaluation, plotting)
            # sees the canonical post-processed file at the expected path.
            state_turb, pos_turb = self._apply_postproc_and_resolve_impurity(
                file_turb, fn_turb,
                baseline_species=baseline_species, baseline_pos=baseline_pos,
                side_label="turb",
            )
            shutil.copy2(file_turb, self.file_profs)
            self.powerstate.profiles_transport = state_turb
            self.powerstate.profiles_transport_turb = state_turb
            self.powerstate.impurityPosition_transport = pos_turb
            self.powerstate.impurityPosition_transport_turb = pos_turb
            if pos_turb != baseline_pos:
                print(f"\t- [turb] Impurity position has changed from {baseline_pos} to {pos_turb}", typeMsg="i")

            # Neo side
            state_neo, pos_neo = self._apply_postproc_and_resolve_impurity(
                file_neo, fn_neo,
                baseline_species=baseline_species, baseline_pos=baseline_pos,
                side_label="neo",
            )
            self.powerstate.profiles_transport_neo = state_neo
            self.powerstate.impurityPosition_transport_neo = pos_neo
            if pos_neo != pos_turb:
                print(f"\t- [neo] Impurity position differs from turb: turb={pos_turb}, neo={pos_neo}", typeMsg="i")

        # --- Optional: compute neoclassical E×B shear from NEO VGEN (zero toroidal rotation assumed)
        vgen_exb_options = self.powerstate.transport_options.get("options", {}).get("neo", {}).get("vgen_exb_shear", None)
        if vgen_exb_options not in (None, False):
            print("\t- Computing neoclassical ExB shear via NEO VGEN (zero toroidal rotation)", typeMsg="i")
            vgenOptions = {} if vgen_exb_options is True else dict(vgen_exb_options)

            # Limit VGEN to the region PORTALS predicts, with a 0.05 margin on each side
            rho_portals = self.powerstate.plasma["rho"][0, 1:].cpu().numpy()
            rho_lo = float(np.clip(rho_portals.min() - 0.05, 0.0, 1.0))
            rho_hi = float(np.clip(rho_portals.max() + 0.05, 0.0, 1.0))
            rho_range = [rho_lo, rho_hi]
            print(f"\t\t- VGEN rho_range: [{rho_lo:.3f}, {rho_hi:.3f}] (PORTALS rho: [{rho_portals.min():.3f}, {rho_portals.max():.3f}])", typeMsg="i")

            # minutes for VGEN from the NEO allocation (same source as neo.run()), defaulting to 60.
            neo_opts     = self.transport_evaluator_options.get("neo", {})
            neo_run      = neo_opts.get("run", {})
            neo_alloc    = neo_run.get("allocation", {})
            minutes_vgen = neo_alloc.get("minutes", 60)
            # Reuse the top-level `transport.in_process` flag for VGEN as well —
            # if the user wants in-process codes they almost certainly want
            # in-process VGEN too (same ctypes path, no SLURM overhead).
            in_process_vgen = self.powerstate.transport_options.get("in_process", False)

            neo_exb = NEOtools.NEO(rhos=[])
            neo_exb.FolderGACODE = self.folder
            neo_exb.profiles = self.powerstate.profiles_transport
            # numcores=None → run_vgen() resolves from machineSettings (same logic as SIMtools._run())
            # smooth_profiles=True: smooth Te/Ti/ne/ni before VGEN so piecewise-linear
            # kinks in the gradients do not pollute the computed Er
            neo_exb.run_vgen(subfolder="vgen_neo_exb", vgenOptions=vgenOptions, cold_start=self.cold_start,
                             rho_range=rho_range, minutes=minutes_vgen, smooth_profiles=True,
                             in_process=in_process_vgen)
            neo_exb.read_vgen()
            
            # Insert w0 by interpolating
            if rho_range is not None:
                w0 = np.interp(
                    self.powerstate.profiles_transport.profiles['rho(-)'],
                    neo_exb.profiles_vgen.profiles['rho(-)'],
                    neo_exb.profiles_vgen.profiles['w0(rad/s)']
                )
            else:
                w0 = neo_exb.profiles_vgen.profiles['w0(rad/s)']
                
            self.powerstate.profiles_transport.profiles['w0(rad/s)'] = w0

            # Propagate to powerstate.profiles so the stored iteration profiles also carry the NEO w0
            self.powerstate.profiles.profiles['w0(rad/s)'] = w0
            # Refresh the on-disk file from the POST-PROCESSED state (+ VGEN w0):
            # file_profs holds the post-processed content at this point, and the
            # per-iteration snapshots (Outputs/portals_profiles, Evaluation folders)
            # must reflect what the transport codes actually ran — writing the
            # un-post-processed powerstate.profiles here would revert it.
            self.powerstate.profiles_transport.write_state(file=self.file_profs)

            # Under the split-postproc path, mirror VGEN w0 into the neo-side
            # profile so NEO evaluators see the same VGEN result as the turb
            # side (canonical). No-op on the fast path because the neo alias
            # then points at the same `profiles_transport` object.
            neo_state = getattr(self.powerstate, "profiles_transport_neo", None)
            if neo_state is not None and neo_state is not self.powerstate.profiles_transport:
                neo_state.profiles['w0(rad/s)'] = w0

    def _profiles_to_store(self):

        if "folder" in self.powerstate.transport_options:
            whereFolder = IOtools.expandPath(self.powerstate.transport_options["folder"] / "Outputs" / "portals_profiles")
            if not whereFolder.exists():
                IOtools.askNewFolder(whereFolder)

            fil = whereFolder / f"input.gacode.{self.evaluation_number}"
            shutil.copy2(self.file_profs, fil)
            shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
            print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
        else:
            print("\t- Could not move files", typeMsg="w")
                
    def _populate_from_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
        '''
        Populate the powerstate.plasma with the results from the json file
        '''
        
        mapper = {
            'QeGB': ['Qgb', 'QeMWm2'],
            'QiGB': ['Qgb', 'QiMWm2'],
            'GeGB': ['Ggb', 'Ge1E20m2'],
            'GZGB': ['Ggb', 'GZ1E20m2'],
            'MtGB': ['Pgb', 'MtJm2'],
            'QieGB': ['Sgb', 'QieMWm3']
        }
        
        '''
        **********************************************************************************************
        If no population file exists, I only convert from GB to real units and return
        **********************************************************************************************
        '''
        if not (self.folder / file_name).exists():
            print(f"\t* File {self.folder / file_name} does not exist, cannot populate powerstate.plasma", typeMsg='w')
            print(f"\t- Tranforming from GB to real units:")

            # Use [:, 1:] rather than [0, 1:] so this in-memory fallback broadcasts for both
            # single-plasma and multi-plasma inputs without collapsing the batch axis. gb is
            # first converted to numpy because self.__dict__[*_turb] arrives as a numpy array
            # from _evaluate_tglf / _evaluate_tglf_batched and `np.ndarray * torch.Tensor`
            # raises TypeError (only the torch-lhs form works).
            for var in mapper:
                gb = self.powerstate.plasma[f"{mapper[var][0]}"][:, 1:].cpu().numpy()
                self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"] = self.__dict__[f"{var}_{suffix}"] * gb
                self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}_stds"] = self.__dict__[f"{var}_{suffix}_stds"] * gb

            return
        
        '''
        **********************************************************************************************
        Populate the powerstate.plasma from the json file
        **********************************************************************************************
        '''
        print(f"\t* Populating powerstate.plasma with JSON data from {IOtools.clipstr(self.folder / file_name)}")

        with open(self.folder / file_name, 'r') as f:
            json_dict = json.load(f)
        
        # See if the file has GB or real units
        units_GB, units_real = False, False
        if 'QeGB' in json_dict['fluxes_mean']:
            units_GB = True
        if 'QeMWm2' in json_dict['fluxes_mean']:
            units_real = True

        units = 'both' if (units_GB and units_real) else 'GB' if units_GB else 'real' if units_real else 'none'

        if units == 'real':
            
            print("\t\t- File has fluxes in real units... populating powerstate directly")

            for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
                self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
                self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        elif units == 'GB' or units == 'both':

            dum = {}
            for var in mapper:
                gb = self.powerstate.plasma[f"{mapper[var][0]}"][0,1:].cpu().numpy()
                dum[f"{mapper[var][1]}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var]) * gb
                dum[f"{mapper[var][1]}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var]) * gb

            if units == 'GB':
                
                print("\t\t- File has fluxes in GB units... using GB units from powerstate to convert to real units")

                for var in mapper:
                    self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"] = dum[f"{mapper[var][1]}_tr_{suffix}"]
                    self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}_stds"] = dum[f"{mapper[var][1]}_tr_{suffix}_stds"]

            elif units == 'both':
                
                print("\t\t- File has fluxes in both GB and real units... using real units and checking consistency")

                for var in mapper:
                    if not np.allclose(self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"], dum[f"{mapper[var][1]}_tr_{suffix}"]):
                        print(f"\t\t\t- Inconsistent values found for {mapper[var][1]}_tr_{suffix}")

                for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
                    self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
                    self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        else:
            raise ValueError("[MITIM] Unknown units in JSON file")

    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self):
        '''
        This needs to populate the following np.arrays in self., with dimensions of rho:
            - QeGB_turb
            - QiGB_turb
            - GeGB_turb
            - GZGB_turb
            - MtGB_turb
            - QieGB_turb (turbulence exchange)
        and their respective standard deviations, e.g. QeGB_turb_stds
        '''

        print(">> No turbulent fluxes to evaluate", typeMsg="w")
    
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))    
    def evaluate_neoclassical(self):
        '''
        This needs to populate the following np.arrays in self.:
            - QeGB_neoc
            - QiGB_neoc
            - GeGB_neoc
            - GZGB_neoc
            - MtGB_neoc
            - QieGB_neoc (zero)
        and their respective standard deviations, e.g. QeGB_neoc_stds
        '''

        print(">> No neoclassical fluxes to evaluate", typeMsg="w")
        
        
# *******************************************************************************************
# Combinations
# *******************************************************************************************

from mitim_modules.powertorch.physics_models.transport_tglf import tglf_model
from mitim_modules.powertorch.physics_models.transport_neo import neo_model
from mitim_modules.powertorch.physics_models.transport_cgyro import cgyro_model
from mitim_modules.powertorch.physics_models.transport_gx import gx_model

class portals_transport_model(power_transport, tglf_model, neo_model, cgyro_model, gx_model):

    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

        # Defaults — single fidelity. `turbulence_model` / `neoclassical_model` may be
        # overridden via transport_options["evaluator_instance_attributes"] to either a
        # plain string (single fidelity, as today) or an int-keyed dict (multi-fidelity,
        # e.g. {0: 'tglf', 1: 'cgyro'} or {0: 'tglf1', 1: 'tglf2'}). See namelist.portals.yaml.
        self.turbulence_model = 'tglf'
        self.neoclassical_model = 'neo'

    # ----------------------------------------------------------------------------------------
    # Multi-fidelity resolution helpers
    # ----------------------------------------------------------------------------------------
    def _resolve_instance_name(self, spec):
        '''Return the active instance-name string for a model spec. Strings pass through;
        dicts (int-keyed) are indexed by self.fidelity_level.'''
        if isinstance(spec, dict):
            key = int(round(float(self.fidelity_level)))
            if key not in spec:
                raise KeyError(
                    f"fidelity_level={key} not present in model spec {spec!r}; "
                    f"valid keys are {sorted(spec.keys())}"
                )
            return spec[key]
        return spec

    def _resolve_postproc_fun(self, instance_name):
        '''Per-model `profiles_postprocessing_fun` lookup with fallback.

        Resolution order (per evaluator side):
          1. transport.options.<instance_name>.profiles_postprocessing_fun
          2. transport.profiles_postprocessing_fun (the namelist global default)
          3. None
        Per-model overrides global. Used by `_modify_profiles` and
        `_evaluate_batched` to decide whether turbulence and neoclassical
        sides need separate post-processed input.gacode copies.
        '''
        if instance_name is None:
            return self.powerstate.transport_options.get("profiles_postprocessing_fun")
        per_model = (
            self.powerstate.transport_options.get("options", {})
            .get(instance_name, {})
            .get("profiles_postprocessing_fun")
        )
        if per_model is not None:
            return per_model
        return self.powerstate.transport_options.get("profiles_postprocessing_fun")

    def _profiles_transport_for(self, side):
        '''Side-aware accessor for the post-processed gacode_state. Returns
        powerstate.profiles_transport_<side> when set (split path active);
        otherwise falls back to the canonical powerstate.profiles_transport
        — preserving byte-identical behavior on the fast path where turb
        and neo share one postproc fn. side ∈ {"turb","neo"}.
        '''
        if side == "turb":
            return getattr(self.powerstate, "profiles_transport_turb",
                           self.powerstate.profiles_transport)
        if side == "neo":
            return getattr(self.powerstate, "profiles_transport_neo",
                           self.powerstate.profiles_transport)
        raise ValueError(f"Unknown side {side!r}; expected 'turb' or 'neo'")

    def _impurity_position_transport_for(self, side):
        '''Side-aware accessor for impurityPosition_transport. Mirrors
        `_profiles_transport_for`: per-side attribute when split, canonical
        fallback otherwise. side ∈ {"turb","neo"}.
        '''
        if side == "turb":
            return getattr(self.powerstate, "impurityPosition_transport_turb",
                           self.powerstate.impurityPosition_transport)
        if side == "neo":
            return getattr(self.powerstate, "impurityPosition_transport_neo",
                           self.powerstate.impurityPosition_transport)
        raise ValueError(f"Unknown side {side!r}; expected 'turb' or 'neo'")

    def _apply_postproc_and_resolve_impurity(self, gacode_path, fn, *,
                                              baseline_species, baseline_pos,
                                              side_label=""):
        '''Apply `fn` (or no-op) to `gacode_path` (modifies in place; returns
        gacode_state), then re-read the file and resolve the new impurity
        position by name-matching baseline_species[baseline_pos] against
        the post-processed species list. Returns (new_state, new_impurity_pos).

        `side_label` (e.g. "turb", "neo") is just used in the warning
        message when the impurity name no longer exists in the post-processed
        species list — informational only.
        '''
        if fn is not None:
            new_state = fn(gacode_path)
        else:
            new_state = PROFILEStools.gacode_state(gacode_path)

        # Re-read for the species check — the postproc may have rewritten
        # the file but not refreshed the in-memory species list.
        p_new = PROFILEStools.gacode_state(gacode_path)
        impurity_of_interest = baseline_species[baseline_pos]
        try:
            new_pos = p_new.Species.index(impurity_of_interest)
        except ValueError:
            tag = f"[{side_label}] " if side_label else ""
            print(f"\t- {tag}Impurity {impurity_of_interest} not found in post-processed profiles "
                  f"({gacode_path.name}); keeping position {baseline_pos}", typeMsg="w")
            new_pos = baseline_pos
        return new_state, new_pos

    def _resolve_code(self, instance_name):
        '''Resolve the backend code (tglf / cgyro / gx / neo) from an instance name by
        looking at transport.options.<name>.code. When absent, the instance name must
        itself equal a known code.'''
        opts = self.transport_evaluator_options.get(instance_name, {})
        if not isinstance(opts, dict):
            opts = {}
        return str(opts.get('code', instance_name)).lower()

    def _announce_multifidelity(self, kind, spec, active_name, code, batched=False):
        '''Print the resolved fidelity choice — but only when the namelist actually
        requested multi-fidelity (dict spec). Single-fidelity runs stay silent so the
        log output doesn't change from today's behaviour.'''
        if not isinstance(spec, dict):
            return
        fid = int(round(float(self.fidelity_level)))
        tag = " (batched)" if batched else ""
        fidelities_sorted = sorted(spec.keys())
        menu = ", ".join(f"{k}:{spec[k]!r}" for k in fidelities_sorted)
        print(
            f">> [multi-fidelity] {kind}{tag}: fidelity_level={fid} -> instance={active_name!r} (code={code!r}) "
            f"| ladder: {{{menu}}}",
            typeMsg="i",
        )

    def produce_profiles(self):
        self._produce_profiles()

    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self):

        active_name = self._resolve_instance_name(self.turbulence_model)
        self._active_turb_options_key = active_name
        code = self._resolve_code(active_name)
        self._announce_multifidelity("turbulence", self.turbulence_model, active_name, code)

        if code == 'tglf':
            return tglf_model.evaluate_turbulence(self)
        elif code == 'cgyro':
            return cgyro_model.evaluate_turbulence(self)
        elif code == 'gx':
            return gx_model.evaluate_turbulence(self)
        else:
            raise Exception(f"Unknown turbulence code {code!r} (instance {active_name!r})")

    def evaluate_turbulence_batched(self, list_of_states):
        # Multi-plasma dispatcher — powerstate.batch_size > 1 path. No @hook_method here:
        # _evaluate_batched() writes per-plasma JSON pairs itself and disables the single-
        # folder write_json decorator via self._write_json_from_variables_turb = False.
        active_name = self._resolve_instance_name(self.turbulence_model)
        self._active_turb_options_key = active_name
        code = self._resolve_code(active_name)
        self._announce_multifidelity("turbulence", self.turbulence_model, active_name, code, batched=True)

        if code == 'tglf':
            return tglf_model.evaluate_turbulence_batched(self, list_of_states)
        elif code == 'cgyro':
            return cgyro_model.evaluate_turbulence_batched(self, list_of_states)
        else:
            raise NotImplementedError(
                f"Batched turbulence dispatch not yet implemented for code {code!r} "
                f"(instance {active_name!r}). Use single-plasma SR or extend this dispatcher."
            )

    def _stable_correction(self, simulation_options):

        active_name = self._resolve_instance_name(self.turbulence_model)
        code = self._resolve_code(active_name)
        if code == 'cgyro':
            cgyro_model._stable_correction(self, simulation_options)

    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))
    def evaluate_neoclassical(self):
        if self.neoclassical_model is None:
            print('Neoclassical model chosen', typeMsg='w')
            return

        active_name = self._resolve_instance_name(self.neoclassical_model)
        self._active_neo_options_key = active_name
        code = self._resolve_code(active_name)
        self._announce_multifidelity("neoclassical", self.neoclassical_model, active_name, code)

        if code == 'neo':
            return neo_model.evaluate_neoclassical(self)
        else:
            raise Exception(f"Unknown neoclassical code {code!r} (instance {active_name!r})")

    def evaluate_neoclassical_batched(self, list_of_states):
        # Multi-plasma dispatcher — no @hook_method decorator for the same reason as
        # evaluate_turbulence_batched above.
        if self.neoclassical_model is None:
            print('Neoclassical model chosen', typeMsg='w')
            return

        active_name = self._resolve_instance_name(self.neoclassical_model)
        self._active_neo_options_key = active_name
        code = self._resolve_code(active_name)
        self._announce_multifidelity("neoclassical", self.neoclassical_model, active_name, code, batched=True)

        if code == 'neo':
            return neo_model.evaluate_neoclassical_batched(self, list_of_states)
        else:
            raise NotImplementedError(
                f"Batched neoclassical dispatch not yet implemented for code {code!r} "
                f"(instance {active_name!r})."
            )
