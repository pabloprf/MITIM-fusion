import numpy as np
from mitim_tools.misc_tools.LOGtools import printMsg as print

try:
    from mitim_tools.qualikiz_tools import QLKtools
except (ImportError, ModuleNotFoundError):
    QLKtools = None

class qualikiz_model:

    def evaluate_turbulence(self):
        self._evaluate_qualikiz()

    # ------------------------------------------------------------------
    # Single-plasma QuaLiKiz dispatch
    # ------------------------------------------------------------------

    def _evaluate_qualikiz(self, pass_info=True):

        options_key = getattr(self, "_active_turb_options_key", None) or "qlk"
        simulation_options = self.transport_evaluator_options[options_key]
        cold_start = self.cold_start

        percent_error       = simulation_options.get("percent_error", 10.0)
        allocation          = simulation_options.get("allocation", {"resources_per_call": 1, "minutes": 60})
        attempts            = simulation_options.get("attempts_execution", 1)
        use_qlk_scan_trick  = simulation_options.get("use_scan_trick_for_stds", None)
        uncertainty_statistics = simulation_options.get("uncertainty_statistics", "gaussian")

        # Side-aware: postproc may give the turbulence side a different species
        # list than the neoclassical side.
        ion_OI_position_in_ion_list = self._impurity_position_transport_for("turb")

        rho_locations = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        qlk = QLKtools.QuaLiKiz(rhos=rho_locations)

        qlk.prep(
            self._profiles_transport_for("turb"),
            self.folder,
            cold_start=cold_start,
            forceIfcold_start=True,
        )

        # Subfolder mirrors the TGLF convention so named multi-fidelity configs
        # ('qlk1'/'qlk2') don't collide on disk.
        subfolder_name = f"base_{options_key}"
        qlk.run(
            subfolder_name,
            cold_start=cold_start,
            forceIfcold_start=True,
            extra_name=self.name,
            allocation=allocation,
            attempts_execution=attempts,
            **simulation_options.get("run", {}),
        )

        qlk.read(label="base", **simulation_options.get("read", {}))

        nrho = len(rho_locations)
        Qe = np.zeros(nrho)
        Qi = np.zeros(nrho)
        Ge = np.zeros(nrho)
        GZ = np.zeros(nrho)
        Mt = np.zeros(nrho)
        S  = np.zeros(nrho)   # Qie turbulent exchange not provided by QuaLiKiz

        for i, out in enumerate(qlk.results["base"]["output"]):
            # Read physical-unit (SI) outputs; normalise below using the
            # powerstate's Qgb/Ggb/Pgb (which use B_unit), so no explicit
            # B0→B_unit correction is needed.
            Qe[i] = float(out["efe_SI"].values)
            Ge[i] = float(out["pfe_SI"].values)
            Qi[i] = float(out["efi_SI"].values.sum())
            GZ[i] = float(out["pfi_SI"].values[ion_OI_position_in_ion_list])
            try:
                Mt[i] = float(out["vfi_SI"].values.sum())
            except KeyError:
                pass

        # Normalise SI fluxes to MITIM GB units.
        # efe/efi_SI [W/m²]  → divide by Qgb [MW/m²] × 1e6
        # pfe/pfi_SI [m⁻²s⁻¹] → divide by Ggb [1e20 m⁻²s⁻¹] × 1e20
        # vfi_SI [J/m²]      → divide by Pgb [J/m²]
        Qgb = self.powerstate.plasma["Qgb"][0, 1:].cpu().numpy()
        Ggb = self.powerstate.plasma["Ggb"][0, 1:].cpu().numpy()
        Pgb = self.powerstate.plasma["Pgb"][0, 1:].cpu().numpy()
        Qe /= Qgb * 1e6
        Qi /= Qgb * 1e6
        Ge /= Ggb * 1e20
        GZ /= Ggb * 1e20
        Mt /= Pgb

        Flux_base = np.array([Qe, Qi, Ge, GZ, Mt, S])

        if use_qlk_scan_trick is None:
            Flux_value = Flux_base
            Flux_std  = np.abs(Flux_value) * percent_error / 100.0
            Flux_extras = None
        else:
            Flux_value_list, Flux_std_list, Flux_extras = _run_qlk_uncertainty_model(
                qlk,
                rho_locations,
                self.powerstate.predicted_channels,
                Qgb, Ggb, Pgb,
                Flux_base=Flux_base,
                ion_OI_position_in_ion_list=ion_OI_position_in_ion_list,
                delta=use_qlk_scan_trick,
                cold_start=cold_start,
                extra_name=self.name,
                allocation=allocation,
                attempts_execution=attempts,
                code_settings=simulation_options.get("run", {}).get("code_settings"),
                statistics=uncertainty_statistics,
            )
            Flux_value = np.array(Flux_value_list)
            Flux_std  = np.array(Flux_std_list)

        if pass_info:
            self.QeGB_turb       = Flux_value[0]
            self.QeGB_turb_stds  = Flux_std[0]
            self.QiGB_turb       = Flux_value[1]
            self.QiGB_turb_stds  = Flux_std[1]
            self.GeGB_turb       = Flux_value[2]
            self.GeGB_turb_stds  = Flux_std[2]
            self.GZGB_turb       = Flux_value[3]
            self.GZGB_turb_stds  = Flux_std[3]
            self.MtGB_turb       = Flux_value[4]
            self.MtGB_turb_stds  = Flux_std[4]
            self.QieGB_turb      = Flux_value[5]
            self.QieGB_turb_stds = Flux_std[5]

            # Asymmetric-statistics sidecar (semi-deviations symmetric to the std, and no
            # samples, when the scan trick is off) -- same contract as transport_tglf
            for i, var in enumerate(["Qe", "Qi", "Ge", "GZ", "Mt", "Qie"]):
                setattr(self, f"{var}GB_turb_stds_minus", Flux_extras["stds_minus"][i] if Flux_extras is not None else Flux_std[i])
                setattr(self, f"{var}GB_turb_stds_plus",  Flux_extras["stds_plus"][i]  if Flux_extras is not None else Flux_std[i])
                setattr(self, f"{var}GB_turb_samples",    Flux_extras["samples"][i]    if Flux_extras is not None else None)

        return qlk


# ======================================================================================
# QuaLiKiz uncertainty quantification via gradient scan
# ======================================================================================

def _build_qlk_uncertainty_scan_setup(
    predicted_channels,
    ion_OI_position_in_ion_list,
    thermal_ion_indices,
    minimum_abs_gradient,
):
    '''
    Map each predicted channel to the QuaLiKiz scan_dict key scanned for UQ, and
    build the co-variation map that ensures physically consistent perturbations.

    QuaLiKiz analog of transport_tglf._build_uncertainty_scan_setup. Key
    differences from TGLF:

    * QuaLiKiz uses absolute temperatures (Te, Ti) and major-radius-normalised
      logarithmic gradients (Ate = R/LTe, Ane = R/Lne, Ati{i}, Ani{i}), whereas
      TGLF uses RLTS_1/RLTS_2 and RLNS_1. The representative scan key and
      co-variation partners are defined accordingly.

    * The ion/electron temperature ratio scan (TAUS_2 in TGLF) is implemented by
      uniformly multiplying all thermal-ion Ti values (Ti0, Ti1, …) while keeping
      Te fixed.

    * There is no direct QuaLiKiz analog for TGLF's XNUE (collisionality) or
      BETAE (electron beta) scan keys: QuaLiKiz derives these internally from Te,
      ne and Bo, which are not independently settable in the gradient-only scan.

    Co-variation rules (same multiplier as the representative key):
        Ati0  -> Ati{i} for every other thermal ion (all thermal ions perturbed equally)
        Ane   -> Ani{i} for every thermal ion      (quasineutrality)
        Ti0   -> Ti{i}  for every other thermal ion (ratio scan: all Ti move together)

    The minimum_delta_abs floor prevents effectively zero perturbations when a
    gradient is near zero (e.g. aLn ≈ 0 at low-gradient locations).
    '''

    variables_to_scan = []

    for ch in predicted_channels:
        if ch == 'te':
            variables_to_scan.append('Ate')
        if ch == 'ti':
            variables_to_scan.append('Ati0')
        if ch == 'ne':
            variables_to_scan.append('Ane')
        if ch == 'nZ':
            variables_to_scan.append(f'Ani{ion_OI_position_in_ion_list}')
        if ch == 'w0':
            variables_to_scan.append('gammaE')

    # Ion/electron temperature ratio (TAUS_2 analog in TGLF).
    if 'te' in predicted_channels or 'ti' in predicted_channels:
        variables_to_scan.append('Ti0')

    # Co-variation: which additional scan_dict keys to perturb with the same
    # multiplier as the representative variable.
    covariation_map = {}
    for var in variables_to_scan:
        if var == 'Ati0':
            covariation_map[var] = [f'Ati{i}' for i in thermal_ion_indices[1:]]
        elif var == 'Ane':
            covariation_map[var] = [f'Ani{i}' for i in thermal_ion_indices]
        elif var == 'Ti0':
            covariation_map[var] = [f'Ti{i}' for i in thermal_ion_indices[1:]]
        else:
            covariation_map[var] = []

    # Absolute minimum perturbation floors (avoid no-op scans near zero).
    minimum_delta_abs = {}
    for var in variables_to_scan:
        if var in ('Ate', 'Ane') or var.startswith('Ati') or var.startswith('Ani'):
            minimum_delta_abs[var] = minimum_abs_gradient
        if var == 'gammaE':
            minimum_delta_abs[var] = minimum_abs_gradient

    return variables_to_scan, covariation_map, minimum_delta_abs


def _run_qlk_uncertainty_model(
    qlk,
    rho_locations,
    predicted_channels,
    Qgb, Ggb, Pgb,
    Flux_base=None,
    ion_OI_position_in_ion_list=1,
    delta=0.02,
    minimum_abs_gradient=0.005,
    cold_start=False,
    extra_name="",
    subfolder_name='turb_drives',
    allocation=None,
    attempts_execution=1,
    code_settings=None,
    statistics="gaussian",
):
    '''
    Run QuaLiKiz with small gradient perturbations (±delta) around the base
    evaluation point to estimate flux uncertainties via a gradient scan.

    QuaLiKiz analog of transport_tglf._run_tglf_uncertainty_model. The key
    structural difference is that QuaLiKiz covers ALL rho points in a single
    MPI execution (via its own dimx scan). This function takes advantage of
    that by additionally stacking all N_vars × N_deltas perturbation cases
    onto that same dimx scan, so the entire gradient scan runs as a SINGLE
    QuaLiKiz execution (via qlk.run_cases()/read_cases()) instead of
    N_vars × N_deltas separate ones — compared with TGLF's
    N_rhos × N_vars × N_deltas total executions. Each case applies a uniform
    multiplier to the perturbed variable across all rho points simultaneously.

    All cases are generated under a single subfolder:
        {qlk.FolderGACODE}/{subfolder_name}/

    Results are stored in qlk.scans["{subfolder_name}_{variable}"] as
    (nrho, n_deltas) arrays in MITIM GB units, mirroring the structure that
    transport_tglf._aggregate_scan_fluxes reads.
    '''

    print(f"\t- Running QuaLiKiz gradient scans ({delta = }) to determine relative errors")

    n_ions = min(len(qlk.profiles.Species), QLKtools.MAX_QLK_IONS)
    thermal_ion_indices = [
        i for i in range(n_ions)
        if qlk.profiles.Species[i].get("S", "therm") != "fast"
    ]

    variables_to_scan, covariation_map, _ = _build_qlk_uncertainty_scan_setup(
        predicted_channels, ion_OI_position_in_ion_list, thermal_ion_indices, minimum_abs_gradient
    )

    relative_scan = [1 - delta, 1 + delta]
    nrho = len(rho_locations)
    n_deltas = len(relative_scan)

    # Build the full list of perturbation cases up front, tracking which
    # (variable, delta) each case corresponds to so the stacked results can be
    # unstacked back into per-variable (nrho, n_deltas) arrays below.
    cases = []
    case_index_map = []  # (i_var, i_mult) per entry of `cases`, same order
    for i_var, variable in enumerate(variables_to_scan):
        for i_mult, mult in enumerate(relative_scan):
            mult_r = round(mult, 6)
            # Apply the same relative multiplier to the representative variable
            # and all co-varied partners.
            multipliers = {variable: mult_r}
            for cov_key in covariation_map.get(variable, []):
                multipliers[cov_key] = mult_r
            cases.append(multipliers)
            case_index_map.append((i_var, i_mult))

    print(
        f"\n\t- Running QuaLiKiz scans for turbulence drives: "
        f"{len(variables_to_scan)} variables × {n_deltas} deltas = "
        f"{len(cases)} perturbation cases, stacked into a single QuaLiKiz "
        f"execution covering {len(cases) * nrho} dimx points ({nrho} rho points each)\n",
        typeMsg="i",
    )

    label = f"{subfolder_name}_all"
    qlk.run_cases(
        subfolder_name,
        cases,
        cold_start=cold_start,
        forceIfcold_start=True,
        extra_name=f"{extra_name}_{subfolder_name}",
        allocation=allocation,
        attempts_execution=attempts_execution,
        code_settings=code_settings,
    )
    qlk.read_cases(label, n_cases=len(cases))

    scan_data_by_var = {
        variable: {
            "Qe_gb": np.zeros((nrho, n_deltas)),
            "Qi_gb": np.zeros((nrho, n_deltas)),
            "Ge_gb": np.zeros((nrho, n_deltas)),
            "Gi_gb": np.zeros((nrho, n_deltas)),
            "Mt_gb": np.zeros((nrho, n_deltas)),
            "S_gb":  np.zeros((nrho, n_deltas)),
        }
        for variable in variables_to_scan
    }

    for icase, (i_var, i_mult) in enumerate(case_index_map):
        variable = variables_to_scan[i_var]
        scan_data = scan_data_by_var[variable]

        for irho, out in enumerate(qlk.results[label]["output"][icase]):
            Qe_SI = float(out["efe_SI"].values)
            Ge_SI = float(out["pfe_SI"].values)
            Qi_SI = float(out["efi_SI"].values.sum())
            GZ_SI = float(out["pfi_SI"].values[ion_OI_position_in_ion_list])
            try:
                Mt_SI = float(out["vfi_SI"].values.sum())
            except KeyError:
                Mt_SI = 0.0

            scan_data["Qe_gb"][irho, i_mult] = Qe_SI / (Qgb[irho] * 1e6)
            scan_data["Qi_gb"][irho, i_mult] = Qi_SI / (Qgb[irho] * 1e6)
            scan_data["Ge_gb"][irho, i_mult] = Ge_SI / (Ggb[irho] * 1e20)
            scan_data["Gi_gb"][irho, i_mult] = GZ_SI / (Ggb[irho] * 1e20)
            scan_data["Mt_gb"][irho, i_mult] = Mt_SI / Pgb[irho]
            scan_data["S_gb"][irho, i_mult]  = 0.0   # QuaLiKiz doesn't output Qie

    for variable in variables_to_scan:
        qlk.scans[f"{subfolder_name}_{variable}"] = scan_data_by_var[variable]

    return _aggregate_qlk_scan_fluxes(
        qlk, subfolder_name, variables_to_scan, relative_scan, rho_locations, Flux_base=Flux_base
    )


def _aggregate_qlk_scan_fluxes(
    qlk,
    scan_subfolder_name,
    variables_to_scan,
    relative_scan,
    rho_locations,
    Flux_base=None,
):
    '''
    Collect per-variable scan results from qlk.scans and compute mean / std
    across all scan cases.

    QuaLiKiz analog of transport_tglf._aggregate_scan_fluxes. Reads from
    qlk.scans["{scan_subfolder_name}_{variable}"] dicts (populated by
    _run_qlk_uncertainty_model), stacks them into (nrho, n_total_cases) arrays,
    optionally prepends the base flux as an additional column, then returns
    nanmean and nanstd along the case axis.
    '''

    nrho = len(rho_locations)
    n_scan = len(variables_to_scan) * len(relative_scan)

    Qe = np.zeros((nrho, n_scan))
    Qi = np.zeros((nrho, n_scan))
    Ge = np.zeros((nrho, n_scan))
    GZ = np.zeros((nrho, n_scan))
    Mt = np.zeros((nrho, n_scan))
    S  = np.zeros((nrho, n_scan))

    cont = 0
    for vari in variables_to_scan:
        key  = f"{scan_subfolder_name}_{vari}"
        jump = qlk.scans[key]["Qe_gb"].shape[-1]
        Qe[:, cont:cont+jump] = qlk.scans[key]["Qe_gb"]
        Qi[:, cont:cont+jump] = qlk.scans[key]["Qi_gb"]
        Ge[:, cont:cont+jump] = qlk.scans[key]["Ge_gb"]
        GZ[:, cont:cont+jump] = qlk.scans[key]["Gi_gb"]
        Mt[:, cont:cont+jump] = qlk.scans[key]["Mt_gb"]
        S[:,  cont:cont+jump] = qlk.scans[key]["S_gb"]
        cont += jump

    if Flux_base is not None:
        Qe = np.append(np.atleast_2d(Flux_base[0]).T, Qe, axis=1)
        Qi = np.append(np.atleast_2d(Flux_base[1]).T, Qi, axis=1)
        Ge = np.append(np.atleast_2d(Flux_base[2]).T, Ge, axis=1)
        GZ = np.append(np.atleast_2d(Flux_base[3]).T, GZ, axis=1)
        Mt = np.append(np.atleast_2d(Flux_base[4]).T, Mt, axis=1)
        S  = np.append(np.atleast_2d(Flux_base[5]).T, S,  axis=1)

    # Same reduction contract as transport_tglf._aggregate_scan_fluxes: the per-side
    # semi-deviations and the raw samples are computed in BOTH modes (so the diagnostics and
    # the persisted sidecar are always available) and only the reported (value, std) changes.
    # calculate_median_semistds is imported rather than reimplemented so the two evaluators
    # cannot drift apart on the tie-handling convention.
    from mitim_modules.powertorch.physics_models.transport_tglf import calculate_median_semistds

    def calculate_mean_std(Q):
        return np.nanmean(Q, axis=1), np.nanstd(Q, axis=1)

    Flux_value, Flux_std, Flux_std_minus, Flux_std_plus, Flux_samples = [], [], [], [], []
    for Q in [Qe, Qi, Ge, GZ, Mt, S]:
        med, s_minus, s_plus = calculate_median_semistds(Q)
        if statistics == "asymmetric":
            point, std = med, np.sqrt(0.5 * (s_minus**2 + s_plus**2))
        else:
            point, std = calculate_mean_std(Q)
        Flux_value.append(point)
        Flux_std.append(std)
        Flux_std_minus.append(s_minus)
        Flux_std_plus.append(s_plus)
        Flux_samples.append(Q)

    Flux_extras = {
        "statistics": statistics,
        "stds_minus": Flux_std_minus,
        "stds_plus": Flux_std_plus,
        "samples": Flux_samples,
    }

    return Flux_value, Flux_std, Flux_extras
