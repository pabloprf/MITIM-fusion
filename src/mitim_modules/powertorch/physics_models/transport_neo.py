import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class neo_model:

    def evaluate_neoclassical_batched(self, list_of_states):
        self._evaluate_neo_batched(list_of_states)

    # ----------------------------------------------------------------------------------------
    # Multi-plasma NEO — fan a list of profile states through run_over_plasmas so that every
    # (plasma, rho) work unit is dispatched concurrently by the existing FARMINGtools pipeline.
    # Used by power_transport._evaluate_batched() when powerstate.batch_size > 1.
    # ----------------------------------------------------------------------------------------
    def _evaluate_neo_batched(self, list_of_states):

        neo_key = getattr(self, "_active_neo_options_key", None) or "neo"
        simulation_options = self.transport_evaluator_options[neo_key]
        cold_start = self.cold_start

        percent_error = simulation_options["percent_error"]
        in_process = self.powerstate.transport_options.get("in_process", False)
        # Side-aware (neoclassical): under per-model postproc, NEO may see a
        # different species list (and impurity position) than the turbulence side.
        ion_OI_position_in_ion_list = self._impurity_position_transport_for("neo")

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        neo = NEOtools.NEO(rhos=rho_locations, in_process=in_process)

        # list_of_states is the neo-side per-plasma states under split-postproc
        # (passed in from _evaluate_batched). Aliased to the canonical states
        # under the fast path.
        _ = neo.prep(
            list_of_states[0],
            self.folder,
            cold_start=cold_start,
        )

        plasma_labels = neo.run_over_plasmas(
            list_of_states,
            base_subfolder=f"base_{neo_key}",
            cold_start=cold_start,
            forceIfcold_start=True,
            **simulation_options["run"],
        )

        N = len(plasma_labels)
        nrho = len(rho_locations)
        Qe_batch = np.zeros((N, nrho))
        Qi_batch = np.zeros((N, nrho))
        Ge_batch = np.zeros((N, nrho))
        GZ_batch = np.zeros((N, nrho))
        Mt_batch = np.zeros((N, nrho))

        for p, label in plasma_labels.items():
            neo.read_plasma(p, label=label, **simulation_options["read"])
            outputs = neo.results[label]["output"]

            Qe_batch[p, :] = np.array([outputs[i].Qe for i in range(nrho)])
            Qi_batch[p, :] = np.array([outputs[i].Qi for i in range(nrho)])
            Ge_batch[p, :] = np.array([outputs[i].Ge for i in range(nrho)])
            GZ_batch[p, :] = np.array([outputs[i].GiAll[ion_OI_position_in_ion_list] for i in range(nrho)])
            Mt_batch[p, :] = np.array([outputs[i].Mt for i in range(nrho)])

        self.QeGB_neoc = Qe_batch
        self.QiGB_neoc = Qi_batch
        self.GeGB_neoc = Ge_batch
        self.GZGB_neoc = GZ_batch
        self.MtGB_neoc = Mt_batch

        self.QeGB_neoc_stds = np.abs(Qe_batch) * percent_error / 100.0
        self.QiGB_neoc_stds = np.abs(Qi_batch) * percent_error / 100.0
        self.GeGB_neoc_stds = np.abs(Ge_batch) * percent_error / 100.0
        self.GZGB_neoc_stds = np.abs(GZ_batch) * percent_error / 100.0
        self.MtGB_neoc_stds = np.abs(Mt_batch) * percent_error / 100.0

        return neo

    def evaluate_neoclassical(self):

        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options
        # ------------------------------------------------------------------------------------------------------------------------

        neo_key = getattr(self, "_active_neo_options_key", None) or "neo"
        simulation_options = self.transport_evaluator_options[neo_key]
        cold_start = self.cold_start

        percent_error = simulation_options["percent_error"]
        # If True, NEO runs in-process via ctypes (libneo_serial.so) — no
        # subprocess fork, no folder / file I/O.  See namelist.portals.yaml.
        in_process = self.powerstate.transport_options.get("in_process", False)
        # [ion1,ion2,ion3,...], so if I want ion3, I need to do ion_OI_position_in_ion_list = 2
        # Side-aware (neoclassical): see _evaluate_neo_batched() comment.
        ion_OI_position_in_ion_list = self._impurity_position_transport_for("neo")

        # ------------------------------------------------------------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------------------------------------------------------------

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]

        neo = NEOtools.NEO(rhos=rho_locations, in_process=in_process)

        _ = neo.prep(
            self._profiles_transport_for("neo"),
            self.folder,
            cold_start = cold_start,
            )
        
        neo.run(
            f"base_{neo_key}",
            cold_start=cold_start,
            forceIfcold_start=True,
            **simulation_options["run"]
        )
    
        neo.read(
            label='base',
            **simulation_options["read"])
        
        
        Qe = np.array([neo.results['base']['output'][i].Qe for i in range(len(rho_locations))])
        Qi = np.array([neo.results['base']['output'][i].Qi for i in range(len(rho_locations))])
        Ge = np.array([neo.results['base']['output'][i].Ge for i in range(len(rho_locations))])
        GZ = np.array([neo.results['base']['output'][i].GiAll[ion_OI_position_in_ion_list] for i in range(len(rho_locations))])
        Mt = np.array([neo.results['base']['output'][i].Mt for i in range(len(rho_locations))])
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Pass the information to what power_transport expects
        # ------------------------------------------------------------------------------------------------------------------------
        
        self.QeGB_neoc = Qe
        self.QiGB_neoc = Qi
        self.GeGB_neoc = Ge
        self.GZGB_neoc = GZ
        self.MtGB_neoc = Mt
        
        # Uncertainties is just a percent of the value
        self.QeGB_neoc_stds = abs(Qe) * percent_error/100.0
        self.QiGB_neoc_stds = abs(Qi) * percent_error/100.0
        self.GeGB_neoc_stds = abs(Ge) * percent_error/100.0
        self.GZGB_neoc_stds = abs(GZ) * percent_error/100.0
        self.MtGB_neoc_stds = abs(Mt) * percent_error/100.0

        return neo
