import numpy as np
from mitim_tools.misc_tools.LOGtools import printMsg as print

try:
    from tglfnn_gknn import TGLFinput
    from tglfnn_gknn import load_withnegd_pipeline, default_models_dir, run_withnegd_gknn
except ModuleNotFoundError:
    pass

def gknn_profiles_postprocessing_fun(file_profs):
    """Lump a multi-species plasma down to 2 ions for GKNN compatibility.

    The GKNN network only sees TGLF species 1 (electrons), 2 (main ion),
    and 3 (lumped impurity).  This function:
      1. Lumps all impurities into a single effective impurity (species 3).
      2. Lumps D+T (if present) into a single "DT" main ion with A=2.5 (species 2).
      3. Enforces quasineutrality on the result.

    Signature matches the profiles_postprocessing_fun interface:
      fn(file_profs: Path) -> gacode_state
    """
    from mitim_tools.gacode_tools import PROFILEStools

    p = PROFILEStools.gacode_state(file_profs)

    if len(p.ion_list_impurities) > 1:
        p.lumpImpurities()

    p.lumpDT()

    p.enforce_quasineutrality()
    p.write_state(file=file_profs)
    return p


def _tglf_dict_to_gknn_input(flat_dict):
    """Extract the 31 GKNN parameters from a flat TGLF input dict.

    The flat dict comes from gacode_state.to_tglf() and contains controls,
    plasma, and species fields all merged into one dict with keys like
    BETAE, RMIN_LOC, RLTS_1, AS_2, TAUS_3, VPAR_1, VPAR_SHEAR_1, etc.

    Species 3 parameters may be absent when the profiles have only 2 species
    (electrons + 1 ion); defaults to 0.0 in that case.
    """

    return TGLFinput(
        AS_2=flat_dict['AS_2'],
        AS_3=flat_dict.get('AS_3', 0.0),
        BETAE=flat_dict['BETAE'],
        DEBYE=flat_dict['DEBYE'],
        DELTA_LOC=flat_dict['DELTA_LOC'],
        DRMAJDX_LOC=flat_dict['DRMAJDX_LOC'],
        DZMAJDX_LOC=flat_dict['DZMAJDX_LOC'],
        KAPPA_LOC=flat_dict['KAPPA_LOC'],
        P_PRIME_LOC=flat_dict['P_PRIME_LOC'],
        Q_LOC=flat_dict['Q_LOC'],
        Q_PRIME_LOC=flat_dict['Q_PRIME_LOC'],
        RLNS_1=flat_dict['RLNS_1'],
        RLNS_2=flat_dict['RLNS_2'],
        RLNS_3=flat_dict.get('RLNS_3', 0.0),
        RLTS_1=flat_dict['RLTS_1'],
        RLTS_2=flat_dict['RLTS_2'],
        RLTS_3=flat_dict.get('RLTS_3', 0.0),
        RMAJ_LOC=flat_dict['RMAJ_LOC'],
        RMIN_LOC=flat_dict['RMIN_LOC'],
        S_DELTA_LOC=flat_dict['S_DELTA_LOC'],
        S_KAPPA_LOC=flat_dict['S_KAPPA_LOC'],
        S_ZETA_LOC=flat_dict['S_ZETA_LOC'],
        TAUS_2=flat_dict['TAUS_2'],
        TAUS_3=flat_dict.get('TAUS_3', 0.0),
        VEXB_SHEAR=flat_dict['VEXB_SHEAR'],
        VPAR_1=flat_dict['VPAR_1'],
        VPAR_SHEAR_1=flat_dict['VPAR_SHEAR_1'],
        XNUE=flat_dict['XNUE'],
        ZEFF=flat_dict['ZEFF'],
        ZETA_LOC=flat_dict['ZETA_LOC'],
        ZMAJ_LOC=flat_dict['ZMAJ_LOC'],
    )


class gknn_model:
    """TGLF-GKNN-JAX transport model mixin for PORTALS.

    Predicts turbulent fluxes (Qe, Qi, Ge, Pi->Mt) using a JAX neural network
    that approximates TGLF SAT3 with GKNN corrections. Missing channels
    (GZ impurity, Se entropy) are set to zero.
    """

    _gknn_pipeline = None

    def evaluate_turbulence(self):
        self._evaluate_gknn()

    def evaluate_turbulence_batched(self, list_of_states):
        self._evaluate_gknn_batched(list_of_states)

    @classmethod
    def _get_pipeline(cls, models_dir=None):
        if cls._gknn_pipeline is None:
            d = models_dir or default_models_dir()
            print(f"\t- Loading GKNN-JAX pipeline from {d}")
            cls._gknn_pipeline = load_withnegd_pipeline(d)
        return cls._gknn_pipeline

    def _evaluate_gknn(self, pass_info=True):
        options_key = getattr(self, "_active_turb_options_key", None) or "gknn"
        simulation_options = self.transport_evaluator_options[options_key]
        percent_error = simulation_options.get("percent_error", 5.0)
        models_dir = simulation_options.get("models_dir", None)
        apply_gknn = simulation_options.get("apply_gknn", True)

        pipeline = self._get_pipeline(models_dir)

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        profiles = self._profiles_transport_for("turb")
        tglf_dicts = profiles.to_tglf(r=rho_locations, r_is_rho=True)

        gknn_inputs = [_tglf_dict_to_gknn_input(tglf_dicts[rho]) for rho in rho_locations]

        label = "GKNN-JAX" if apply_gknn else "TGLF-NN (no GKNN corrections)"
        print(f"\t- Running {label} inference at {len(rho_locations)} radii")
        results = run_withnegd_gknn(gknn_inputs, pipeline, apply_gknn=apply_gknn)

        Qe = np.array([r.Qe for r in results])
        Qi = np.array([r.Qi for r in results])
        Ge = np.array([r.Ge for r in results])
        Mt = np.array([r.Pi for r in results])
        GZ = np.zeros_like(Qe)
        S  = np.zeros_like(Qe)

        Flux_value = np.array([Qe, Qi, Ge, GZ, Mt, S])
        Flux_std = np.abs(Flux_value) * percent_error / 100.0

        if pass_info:
            self.QeGB_turb      = Flux_value[0]
            self.QeGB_turb_stds = Flux_std[0]
            self.QiGB_turb      = Flux_value[1]
            self.QiGB_turb_stds = Flux_std[1]
            self.GeGB_turb      = Flux_value[2]
            self.GeGB_turb_stds = Flux_std[2]
            self.GZGB_turb      = Flux_value[3]
            self.GZGB_turb_stds = Flux_std[3]
            self.MtGB_turb      = Flux_value[4]
            self.MtGB_turb_stds = Flux_std[4]
            self.QieGB_turb      = Flux_value[5]
            self.QieGB_turb_stds = Flux_std[5]

    def _evaluate_gknn_batched(self, list_of_states, pass_info=True):
        options_key = getattr(self, "_active_turb_options_key", None) or "gknn"
        simulation_options = self.transport_evaluator_options[options_key]
        percent_error = simulation_options.get("percent_error", 5.0)
        models_dir = simulation_options.get("models_dir", None)
        apply_gknn = simulation_options.get("apply_gknn", True)

        pipeline = self._get_pipeline(models_dir)

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        N = len(list_of_states)
        nrho = len(rho_locations)

        all_gknn_inputs = []
        for state in list_of_states:
            tglf_dicts = state.to_tglf(r=rho_locations, r_is_rho=True)
            for rho in rho_locations:
                all_gknn_inputs.append(_tglf_dict_to_gknn_input(tglf_dicts[rho]))

        label = "GKNN-JAX" if apply_gknn else "TGLF-NN (no GKNN corrections)"
        print(f"\t- Running {label} batched inference: {N} plasmas x {nrho} radii = {N*nrho} evaluations")
        all_results = run_withnegd_gknn(all_gknn_inputs, pipeline, apply_gknn=apply_gknn)

        Qe = np.array([r.Qe for r in all_results]).reshape(N, nrho)
        Qi = np.array([r.Qi for r in all_results]).reshape(N, nrho)
        Ge = np.array([r.Ge for r in all_results]).reshape(N, nrho)
        Mt = np.array([r.Pi for r in all_results]).reshape(N, nrho)
        GZ = np.zeros((N, nrho))
        S  = np.zeros((N, nrho))

        Flux_value = np.stack([Qe, Qi, Ge, GZ, Mt, S], axis=0)
        Flux_std = np.abs(Flux_value) * percent_error / 100.0

        if pass_info:
            self.QeGB_turb      = Flux_value[0]
            self.QeGB_turb_stds = Flux_std[0]
            self.QiGB_turb      = Flux_value[1]
            self.QiGB_turb_stds = Flux_std[1]
            self.GeGB_turb      = Flux_value[2]
            self.GeGB_turb_stds = Flux_std[2]
            self.GZGB_turb      = Flux_value[3]
            self.GZGB_turb_stds = Flux_std[3]
            self.MtGB_turb      = Flux_value[4]
            self.MtGB_turb_stds = Flux_std[4]
            self.QieGB_turb      = Flux_value[5]
            self.QieGB_turb_stds = Flux_std[5]
