"""
PORTALSmain_edge.py
--------------------
``portals_edge``: PORTALS-Edge workflow class.

Inherits everything from the standard ``portals`` class and overrides only ``run()``
so that each model evaluation uses a ``powerstate_edge`` instance instead of the
base ``powerstate``.

What is *not* changed
---------------------
- ``prep()`` / ``PORTALSinit.initializeProblem()``  — initialization is identical
- The BO optimization loops (acquisition function, GP surrogates, transforms)
- The relaxation initialization (``initialization_simple_relax``)
- ``scalarized_objective()``, ``analyze_results()``, ``reuseTrainingTabular()``

The only structural modification is that each call to the real transport model goes
through a ``powerstate_edge`` that inserts the boundary / neutral / impurity
calculations between ``calculateProfileFunctions()`` and ``calculateTargets()``.

Expected namelist additions  (portals_namelist.yaml)
-----------------------------------------------------
Add an ``edge_options`` block alongside the standard ``solution`` block:

    edge_options:
      domain_roa    : [0.85, 1.0]   # restrict plasma to r/a ∈ [0.85, 1.0]
      domain_rho    : null           # or give rho limits (roa takes priority)

      # Separatrix values used for LCFS BC enforcement in powerstate_edge.modify()
      lcfs_bc:
        te : 0.080   # keV
        ti : 0.080   # keV
        ne : 0.30    # 1e19 m-3

      # Swap the targets evaluator for analytical_model_edge
      use_edge_targets : true

    # Optional: edge-UQ uncertainty inflation (applied inside powerstate_edge)
    edge_uq_enable              : true
    edge_uq_calib_dir           : ./results_offline_edge_uq
    edge_uq_calib_label         : baseline
    edge_uq_scale_factor        : 1.0
    # edge_uq_channel_mapping: {bc_ne: ne, bc_te: te}  # optional
"""

import copy
import torch
import shutil
import dill as pickle_dill
from mitim_modules.portals.PORTALSmain import portals, map_powerstate_to_portals
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
import numpy as np
import pandas as pd
from collections import OrderedDict
from mitim_tools.plasmastate_tools import MITIMstate
from mitim_modules.powertorch import STATEedge
from mitim_modules.portals import PORTALStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.LOGtools import printMsg as print


def _dv_name_for_parameterizer(parameterizer_name, channel, index, defined_on=None):
    if parameterizer_name.lower() == "mtanh":
        param_names = ["log_A", "log_u1", "delta", "m"]
        if 1 <= index <= len(param_names):
            return f"{channel}_{param_names[index - 1]}"
    if parameterizer_name in {"SplineMtanh", "spline_mtanh", "mtanh_spline", "MtanhSpline"}:
        if defined_on == "dy":
            return f"d{channel}_{index}"
        return f"aL{channel}_{index}"
    return f"aL{channel}_{index}"


def _get_parameterized_state_from_powerstate(powerstate, channels):
    """Create a lightweight state view required by parameterizer.parameterize()."""

    class _StateView:
        pass

    state = _StateView()

    rho = powerstate.plasma["rho"]
    if isinstance(rho, torch.Tensor) and rho.ndim > 1:
        rho = rho[0, :]
    state.rho = rho.detach().cpu().numpy() if isinstance(rho, torch.Tensor) else np.asarray(rho)
    state.roa = powerstate.plasma['roa']

    if "a" in powerstate.plasma:
        aval = powerstate.plasma["a"]
        if isinstance(aval, torch.Tensor):
            state.a = float(aval.detach().cpu().reshape(-1)[0].item())
        else:
            state.a = float(np.asarray(aval).reshape(-1)[0])
    else:
        state.a = 1.0

    for ch in channels:
        y = powerstate.plasma[ch]
        if isinstance(y, torch.Tensor) and y.ndim > 1:
            y = y[0, :]
        setattr(
            state,
            ch,
            y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y),
        )

        ag_name = f"aL{ch}"
        if ag_name in powerstate.plasma:
            ag = powerstate.plasma[ag_name]
            if isinstance(ag, torch.Tensor) and ag.ndim > 1:
                ag = ag[0, :]
            setattr(
                state,
                ag_name,
                ag.detach().cpu().numpy() if isinstance(ag, torch.Tensor) else np.asarray(ag),
            )

    return state


def _recover_previous_dvs(foldermitim):
    from mitim_tools.opt_tools.STRATEGYtools import opt_evaluator

    opt_fun = opt_evaluator(foldermitim)
    opt_fun.read_optimization_results(analysis_level=1)
    x = opt_fun.mitim_model.BOmetrics["overall"]["xBest"].cpu().numpy()
    dvs = opt_fun.mitim_model.optimization_options["problem_options"]["dvs"]
    dvs_dict = {dvs[j]: x[j] for j in range(len(dvs))}

    print(
        f"- Grabbing best #{opt_fun.mitim_model.BOmetrics['overall']['indBest']} from previous workflow",
        typeMsg="i",
    )
    return dvs_dict


def _as_tensor_on(ref_tensor, value):
    return torch.as_tensor(float(value), dtype=ref_tensor.dtype, device=ref_tensor.device)


def _read_parameter_ranges(ranges_dict, channel, param_idx, param_name):
    if ranges_dict is None:
        return 0.0

    if channel not in ranges_dict:
        return 0.0

    vals = ranges_dict[channel]
    if isinstance(vals, dict):
        if param_name in vals:
            return float(vals[param_name])
        return float(vals.get(str(param_idx), 0.0))

    return _pick_value(vals, param_idx)


def _previous_dv_value(previous_dvs, parameterizer_name, channel, param_idx, param_name, defined_on=None):
    if not previous_dvs:
        return None

    candidates = [
        f"{channel}_{param_name}",
        _dv_name_for_parameterizer(parameterizer_name, channel, param_idx + 1, defined_on=defined_on),
    ]

    for key in candidates:
        if key in previous_dvs:
            return previous_dvs[key]

    return None


def _pick_value(seq, idx):
    if np.isscalar(seq):
        return float(seq)
    if len(seq) == 0:
        return 0.0
    return float(seq[min(idx, len(seq) - 1)])


def initializeProblem(
    portals_fun,
    folderWork,
    fileStart,
    RelVar_y_max,
    RelVar_y_min,
    limits_are_relative=True,
    yminymax_atleast=None,
    cold_start=False,
    fixed_gradients=None,
    start_from_folder=None,
    define_ranges_from_profiles=None,
    seedInitial=None,
    checkForSpecies=True,
    tensor_options = {
        "dtype": torch.double,
        "device": torch.device("cpu"),
    }
    ):
    """
    Notes:
        - Specification of points occur in rho coordinate, although internally the work is r/a
            cold_start = True if cold_start from beginning
        - define_ranges_from_profiles is intentionally disabled in edge mode
    """

    if define_ranges_from_profiles is not None:
        raise ValueError(
            "[PORTALSedge] define_ranges_from_profiles is not supported in edge mode. "
            "This avoids legacy powerstate initialization paths."
        )

    dfT = torch.randn((2, 2), **tensor_options)

    if seedInitial is not None:
        torch.manual_seed(seed=seedInitial)

    FolderInitialization = folderWork / "Initialization"

    if (cold_start) or (not folderWork.exists()):
        IOtools.askNewFolder(folderWork, force=cold_start)

    FolderInitialization.mkdir(parents=True, exist_ok=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize file input.gacode
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ---- Copy the file of interest to initialization folder
    if isinstance(fileStart, MITIMstate.mitim_state):
        fileStart.write_state(file=FolderInitialization / "input.gacode")
    else:
        shutil.copy2(fileStart, FolderInitialization / "input.gacode")

    # ---- Make another copy to preserve the original state

    shutil.copy2(FolderInitialization / "input.gacode", FolderInitialization / "input.gacode_original")

    # ---- Initialize file to modify and increase resolution

    initialization_file = FolderInitialization / "input.gacode"
    profiles = PROFILEStools.gacode_state(initialization_file)

    # About radial locations
    if portals_fun.portals_parameters["solution"]["predicted_roa"] is not None:
        roa = portals_fun.portals_parameters["solution"]["predicted_roa"]
        rho = np.interp(roa, profiles.derived["roa"], profiles.profiles["rho(-)"])
        print("\t * r/a provided, transforming to rho:")
        print(f"\t\t r/a = {roa}")
        print(f"\t\t rho = {rho}")
        portals_fun.portals_parameters["solution"]["predicted_rho"] = rho

    # Good approach to ensure this consistency
    profiles.correct(options={"recalculate_ptot": True})

    if portals_fun.portals_parameters["solution"]["trace_impurity"] is not None:
        position_of_impurity = MITIMstate.impurity_location(profiles, portals_fun.portals_parameters["solution"]["trace_impurity"])
    else:
        position_of_impurity = 1

    if portals_fun.portals_parameters["solution"]["fZ0_as_weight"] is not None and portals_fun.portals_parameters["solution"]["trace_impurity"] is not None:
        f0 = profiles.Species[position_of_impurity]["n0"] / profiles.profiles['ne(10^19/m^3)'][0]
        portals_fun.portals_parameters["solution"]["fImp_orig"] = f0/portals_fun.portals_parameters["solution"]["fZ0_as_weight"]
        print(f'\t- Ion {portals_fun.portals_parameters["solution"]["trace_impurity"]} has original central concentration of {f0:.2e}, using its inverse multiplied by {portals_fun.portals_parameters["solution"]["fZ0_as_weight"]} as scaling factor of GZ -> {portals_fun.portals_parameters["solution"]["fImp_orig"]:.2e}',typeMsg="i")
    else:
        portals_fun.portals_parameters["solution"]["fImp_orig"] = 1.0

    # Check if I will be able to calculate radiation
    speciesNotFound = []
    for i in range(len(profiles.Species)):
        data_df = pd.read_csv(__mitimroot__ / "src" / "mitim_modules" / "powertorch" / "physics_models" / "radiation_chebyshev.csv")
        if not (data_df['Ion'].str.lower()==profiles.Species[i]["N"].lower()).any():
            speciesNotFound.append(profiles.Species[i]["N"])

    # Print warning or question to be careful!
    if len(speciesNotFound) > 0:

        if "qrad" in portals_fun.portals_parameters["target"]["options"]["targets_evolve"]:
        
            answerYN = print(f"\t- Species {speciesNotFound} not found in radiation database, radiation will be zero in PORTALS... is this ok for your predictions?",typeMsg="q" if checkForSpecies else "w")
            if checkForSpecies and (not answerYN):
                raise ValueError("Species not found")

        else:

            print(f'\t- Species {speciesNotFound} not found in radiation database, but this PORTALS prediction is not calculating radiation anyway',typeMsg="w")

    # Prepare and defaults

    xCPs = torch.from_numpy(np.array(portals_fun.portals_parameters["solution"]["predicted_rho"])).to(dfT)

    """
    ***************************************************************************************************
                                powerstate object
    ***************************************************************************************************
    """

    transport_parameters = portals_fun.portals_parameters["transport"]
    edge_options = copy.deepcopy(portals_fun.portals_parameters.get("edge_options", {}) or {})

    # Keep compatibility with workflows that place this under solution.
    if "defined_on" not in edge_options:
        defined_on = portals_fun.portals_parameters["solution"].get("defined_on", None)
        if defined_on is not None:
            edge_options["defined_on"] = defined_on
    
    # Add folder and cold_start to the simulation options
    transport_options = transport_parameters | {"folder": portals_fun.folder, "cold_start": False}
    target_options = copy.deepcopy(portals_fun.portals_parameters["target"])
    target_options.setdefault("options", {})
    if "target_multipliers" in edge_options:
        target_options["options"]["target_multipliers"] = edge_options["target_multipliers"]
    if "targets_scaled" in edge_options:
        target_options["options"]["targets_scaled"] = edge_options["targets_scaled"]

    portals_fun.powerstate = STATEedge.powerstate_edge(
        profiles,
        evolution_options={
            "ProfilePredicted": portals_fun.portals_parameters["solution"]["predicted_channels"],
            "rhoPredicted": xCPs,
            "impurityPosition": position_of_impurity,
            "fImp_orig": portals_fun.portals_parameters["solution"]["fImp_orig"],
            "parameterizer": portals_fun.portals_parameters["solution"].get("parameterizer", "spline"),
            "parameterizer_options": portals_fun.portals_parameters["solution"].get("parameterizer_options", None),
            **edge_options,
        },
        transport_options=transport_options,
        target_options=target_options,
        tensor_options=tensor_options
    )

    # After resolution and corrections, store.
    profiles.write_state(file=FolderInitialization / "input.gacode_modified")

    # ***************************************************************************************************
    # ***************************************************************************************************

    previous_dvs = _recover_previous_dvs(start_from_folder) if start_from_folder is not None else {}

    # Write this updated profiles class (with parameterized profiles)
    _ = portals_fun.powerstate.from_powerstate(
        write_input_gacode=FolderInitialization / "input.gacode",
        postprocess_input_gacode=portals_fun.portals_parameters["transport"]["applyCorrections"],
    )

    # Original complete targets
    portals_fun.powerstate.calculateBoundaryConditions()
    portals_fun.powerstate.calculateProfileFunctions()
    portals_fun.powerstate.calculateTargets()

    thr = 1E-5

    parameterizer_name = portals_fun.portals_parameters["solution"].get("parameterizer", "spline")
    parameterizer_defined_on = edge_options.get("defined_on")

    state_view = _get_parameterized_state_from_powerstate(
        portals_fun.powerstate,
        portals_fun.portals_parameters["solution"]["predicted_channels"],
    )
    param_dict, _ = portals_fun.powerstate.parameterizer.parameterize(
        state_view,
        portals_fun.powerstate.bc_dict,
    )
    # Store reconstructed profiles bookkeeping from parameterizer output.
    dictCPs_base = {
        name: param_dict[name]
        for name in portals_fun.portals_parameters["solution"]["predicted_channels"]
        if name in param_dict
    }
    param_order = list(getattr(portals_fun.powerstate.parameterizer, "param_names", []))

    dictDVs = OrderedDict()
    for var in portals_fun.portals_parameters["solution"]["predicted_channels"]:
        if var not in param_dict:
            raise ValueError(f"[PORTALSedge] Parameterizer did not return parameters for channel '{var}'")

        pvals = param_dict[var]
        if isinstance(pvals, dict):
            if len(param_order) > 0 and all(k in pvals for k in param_order):
                channel_param_names = [k for k in param_order if k in pvals]
            else:
                channel_param_names = list(pvals.keys())
            channel_base = [_as_tensor_on(dfT, pvals[pn]) for pn in channel_param_names]
        else:
            arr = np.asarray(pvals).reshape(-1)
            channel_param_names = [f"p{j}" for j in range(arr.shape[0])]
            channel_base = [_as_tensor_on(dfT, arr[j]) for j in range(arr.shape[0])]

        for conti, (pname, base_val) in enumerate(zip(channel_param_names, channel_base)):
            if limits_are_relative:
                rmin = _read_parameter_ranges(RelVar_y_min, var, conti, pname)
                rmax = _read_parameter_ranges(RelVar_y_max, var, conti, pname)
                scale = max(abs(float(base_val.item())), 1e-3)
                y1 = base_val - scale * rmin
                y2 = base_val + scale * rmax
            else:
                y1 = _as_tensor_on(dfT, _read_parameter_ranges(RelVar_y_min, var, conti, pname))
                y2 = _as_tensor_on(dfT, _read_parameter_ranges(RelVar_y_max, var, conti, pname))

            if yminymax_atleast is not None:
                if yminymax_atleast[0] is not None:
                    y1 = _as_tensor_on(dfT, np.min([float(y1.item()), yminymax_atleast[0]]))
                if yminymax_atleast[1] is not None:
                    y2 = _as_tensor_on(dfT, np.max([float(y2.item()), yminymax_atleast[1]]))

            if y2 - y1 < thr:
                print(
                    f"{var}::{pname} has a range of {y2-y1:.1e} which is less than {thr:.1e}",
                    typeMsg="q",
                )

            name = f"{var}_{pname}"
            previous_val = _previous_dv_value(
                previous_dvs,
                parameterizer_name,
                var,
                conti,
                pname,
                defined_on=parameterizer_defined_on,
            )
            if (seedInitial is None) or (seedInitial == 0):
                base_gradient = (
                    _as_tensor_on(dfT, previous_val) if previous_val is not None else base_val
                )
            else:
                base_gradient = torch.rand(1)[0].to(dfT) * (y2 - y1) / 4 + (3 * y1 + y2) / 4

            if fixed_gradients is None:
                dictDVs[name] = [y1, base_gradient, y2]
            else:
                fixed = fixed_gradients.get(name, None)
                if fixed is None:
                    legacy_name = _dv_name_for_parameterizer(
                        parameterizer_name,
                        var,
                        conti + 1,
                        defined_on=parameterizer_defined_on,
                    )
                    fixed = fixed_gradients.get(legacy_name, None)
                if fixed is None:
                    raise KeyError(
                        f"[PORTALSedge] Missing fixed gradients for DV '{name}'. "
                        f"Expected keys '{name}' or '{legacy_name}'."
                    )
                dictDVs[name] = [
                    _as_tensor_on(dfT, fixed[0]),
                    base_gradient,
                    _as_tensor_on(dfT, fixed[1]),
                ]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define output dictionaries
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ofs, name_objectives = [], []
    for ikey in dictCPs_base:
        if ikey == "te":
            var = "Qe"
        elif ikey == "ti":
            var = "Qi"
        elif ikey == "ne":
            var = "Ge"
        elif ikey == "nZ":
            var = "GZ"
        elif ikey == "w0":
            var = "Mt"

        for i in range(len(portals_fun.portals_parameters["solution"]["predicted_rho"])):
            ofs.append(f"{var}_tr_turb_{i+1}")
            ofs.append(f"{var}_tr_neoc_{i+1}")

            ofs.append(f"{var}_tar_{i+1}")

            name_objectives.append(f"{var}Res_{i+1}")

    if portals_fun.portals_parameters["solution"]["turbulent_exchange_as_surrogate"]:
        for i in range(len(portals_fun.portals_parameters["solution"]["predicted_rho"])):
            ofs.append(f"Qie_tr_turb_{i+1}")

    name_transformed_ofs = []
    for of in ofs:
        if ("GZ" in of) and (portals_fun.portals_parameters["solution"]["impurity_trick"]):
            lab = f"{of} (GB MOD)"
        else:
            lab = f"{of} (GB)"
        name_transformed_ofs.append(lab)

    portals_fun.name_objectives = name_objectives
    portals_fun.name_transformed_ofs = name_transformed_ofs
    portals_fun.optimization_options["problem_options"]["ofs"] = ofs
    portals_fun.optimization_options["problem_options"]["dvs"] = [*dictDVs]
    portals_fun.optimization_options["problem_options"]["dvs_min"] = []
    for i in dictDVs:
        portals_fun.optimization_options["problem_options"]["dvs_min"].append(dictDVs[i][0].cpu().numpy())
    portals_fun.optimization_options["problem_options"]["dvs_base"] = []
    for i in dictDVs:
        portals_fun.optimization_options["problem_options"]["dvs_base"].append(dictDVs[i][1].cpu().numpy())
    portals_fun.optimization_options["problem_options"]["dvs_max"] = []
    for i in dictDVs:
        portals_fun.optimization_options["problem_options"]["dvs_max"].append(dictDVs[i][2].cpu().numpy())

    portals_fun.optimization_options["problem_options"]["dvs_min"] = np.array(portals_fun.optimization_options["problem_options"]["dvs_min"])
    portals_fun.optimization_options["problem_options"]["dvs_max"] = np.array(portals_fun.optimization_options["problem_options"]["dvs_max"])
    portals_fun.optimization_options["problem_options"]["dvs_base"] = np.array(portals_fun.optimization_options["problem_options"]["dvs_base"])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # For surrogate
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Variables = {}
    for ikey in portals_fun.portals_parameters["solution"]["portals_transformation_variables"]:
        Variables[ikey] = prepportals_transformation_variables(portals_fun, ikey)

    portals_fun.surrogate_parameters = {
        "transformationInputs": PORTALStools.input_transform_portals,
        "transformationOutputs": PORTALStools.output_transform_portals,
        "powerstate": portals_fun.powerstate,
        "impurity_trick": portals_fun.portals_parameters["solution"]["impurity_trick"],
        "surrogate_transformation_variables_alltimes": Variables,
        "surrogate_transformation_variables_lasttime": copy.deepcopy(Variables[list(Variables.keys())[-1]]),
        "parameters_combined": {},
    }

def prepportals_transformation_variables(portals_fun, ikey, doNotFitOnFixedValues=False):
    allOuts = portals_fun.optimization_options["problem_options"]["ofs"]
    portals_transformation_variables = portals_fun.portals_parameters["solution"]["portals_transformation_variables"][ikey]
    portals_transformation_variables_trace = portals_fun.portals_parameters["solution"]["portals_transformation_variables_trace"][ikey]

    Variables = {}
    for output in allOuts:
        if IOtools.isfloat(output):
            continue

        typ = '_'.join(output.split("_")[:-1])
        pos = int(output.split("_")[-1])

        if typ in [
            "Qe",
            "Qe_tr_turb",
            "Qe_tr_neoc",
            "Qi",
            "Qi_tr_turb",
            "Qi_tr_neoc",
            "Ge",
            "Ge_tr_turb",
            "Ge_tr_neoc",
            "Qie_tr_turb",
            "Mt",
            "Mt_tr_turb",
            "Mt_tr_neoc",
        ]:
            if doNotFitOnFixedValues:
                isAbsValFixed = pos == (
                    portals_fun.powerstate.plasma["rho"].shape[-1] - 1
                )
            else:
                isAbsValFixed = False

            Variations = {
                "aLte": "te" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLti": "ti" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLne": "ne" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLw0": "w0" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "te": ("te" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "ti": ("ti" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "ne": ("ne" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "w0": ("w0" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
            }

            Variables[output] = []
            for ikey in portals_transformation_variables:
                useThisOne = False
                for varis in portals_transformation_variables[ikey]:
                    if Variations[varis]:
                        useThisOne = True
                        break

                if useThisOne:
                    Variables[output].append(ikey)

        elif typ in ["GZ", "GZ_tr_turb", "GZ_tr_neoc"]:
            if doNotFitOnFixedValues:
                isAbsValFixed = pos == portals_fun.powerstate.plasma["rho"].shape[-1] - 1
            else:
                isAbsValFixed = False

            Variations = {
                "aLte": "te" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLti": "ti" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLne": "ne" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLw0": "w0" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "aLnZ": "nZ" in portals_fun.portals_parameters["solution"]["predicted_channels"],
                "te": ("te" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "ti": ("ti" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "ne": ("ne" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "w0": ("w0" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
                "nZ": ("nZ" in portals_fun.portals_parameters["solution"]["predicted_channels"])
                and (not isAbsValFixed),
            }

            Variables[output] = []
            for ikey in portals_transformation_variables_trace:
                useThisOne = False
                for varis in portals_transformation_variables_trace[ikey]:
                    if Variations[varis]:
                        useThisOne = True
                        break

                if useThisOne:
                    Variables[output].append(ikey)

        elif typ in ["Qe_tar"]:
            Variables[output] = ["QeGB"]
        elif typ in ["Qi_tar"]:
            Variables[output] = ["QiGB"]
        elif typ in ["Ge_tar"]:
            Variables[output] = ["CeGB"]
        elif typ in ["GZ_tar"]:
            Variables[output] = ["CZGB"]
        elif typ in ["Mt_tar"]:
            Variables[output] = ["MtGB"]

    return Variables


def _dictdvs_to_x_tensor(portals_obj, powerstate, dictDVs):
    """Build a single-row design vector X from the declared DV ordering."""
    dv_names = portals_obj.optimization_options["problem_options"]["dvs"]
    x_vals = [dictDVs[name]["value"] for name in dv_names]
    return torch.as_tensor(x_vals, dtype=powerstate.dfT.dtype, device=powerstate.dfT.device).unsqueeze(0)


def runModelEvaluator_edge(
    self,
    FolderEvaluation,
    dictDVs,
    name,
    cold_start=False,
    numPORTALS=0,
    dictOFs=None,
    remove_folder_upon_completion=False,
):
    """Edge-specific evaluator decoupled from legacy fixed DV naming."""
    powerstate = copy.deepcopy(self.powerstate)

    folder_model = FolderEvaluation / "transport_simulation_folder"
    folder_model.mkdir(parents=True, exist_ok=True)

    X = _dictdvs_to_x_tensor(self, powerstate, dictDVs)
    powerstate._repeat_tensors(batch_size=X.shape[0])
    powerstate.transport_options["cold_start"] = cold_start
    powerstate.calculate(X, nameRun=name, folder=folder_model, evaluation_number=numPORTALS)

    if dictOFs is not None:
        proxy_powerstate = copy.copy(powerstate)
        proxy_powerstate.plasma = powerstate._slice_plasma_to_rhoCP(pad_zero=True)
        dictOFs = map_powerstate_to_portals(proxy_powerstate, dictOFs)

    if remove_folder_upon_completion:
        IOtools.shutil_rmtree(folder_model)

    return powerstate, dictOFs


class portals_edge(portals):
    """
    PORTALS-Edge: thin wrapper around ``portals`` that swaps the model evaluator's
    powerstate for a ``powerstate_edge`` on every call to ``run()``.

    All BO / relaxation / surrogate fitting logic is inherited unchanged.
    """

    def __init__(
        self,
        folder,
        portals_namelist=None,
        tensor_options={
            "dtype": torch.double,
            "device": torch.device("cpu"),
        },
    ):
        super().__init__(
            folder,
            portals_namelist=portals_namelist,
            tensor_options=tensor_options,
        )

        # Edge-specific options from the namelist; defaults gracefully to empty dict
        # if the ``edge_options`` block is absent so that the class can run with a
        # vanilla PORTALS namelist during early development.
        self._edge_options = self.portals_parameters.get("edge_options", {})

    # ------------------------------------------------------------------
    # Override: prep()
    # ------------------------------------------------------------------


    def prep(
        self,
        mitim_state,
        cold_start=False,
        seedInitial=None,
        askQuestions=True,
    ):

        # Grab exploration ranges
        ymax = self.portals_parameters["solution"]["exploration_ranges"]["ymax"]
        ymin = self.portals_parameters["solution"]["exploration_ranges"]["ymin"]
        limits_are_relative = self.portals_parameters["solution"]["exploration_ranges"]["limits_are_relative"]
        fixed_gradients = self.portals_parameters["solution"]["exploration_ranges"]["fixed_gradients"]
        yminymax_atleast = self.portals_parameters["solution"]["exploration_ranges"]["yminymax_atleast"]
        enforce_finite_aLT = self.portals_parameters["solution"]["exploration_ranges"]["enforce_finite_aLT"]
        start_from_folder = self.portals_parameters["solution"]["exploration_ranges"]["start_from_folder"]
        reevaluate_targets = self.portals_parameters["solution"]["exploration_ranges"]["reevaluate_targets"]

        # edge_options are often edited in scripts/workflows after __init__.
        self._edge_options = self.portals_parameters.get("edge_options", {})

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make sure that options that are required by good behavior of PORTALS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(">> PORTALS flags pre-check")

        # Check that I haven't added a deprecated variable that I expect some behavior from
        IOtools.check_flags_mitim_namelist(self.portals_parameters, self.potential_flags, avoid = ["run", "read"], askQuestions=askQuestions)

        key_rhos = "predicted_roa" if self.portals_parameters["solution"]["predicted_roa"] is not None else "predicted_rho"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if IOtools.isfloat(ymax):
            ymax0 = copy.deepcopy(ymax)

            ymax = {}
            for prof in self.portals_parameters["solution"]["predicted_channels"]:
                ymax[prof] = np.array( [ymax0] * len(self.portals_parameters["solution"][key_rhos]) )
        
        if IOtools.isfloat(ymin):
            ymin0 = copy.deepcopy(ymin)

            ymin = {}
            for prof in self.portals_parameters["solution"]["predicted_channels"]:
                ymin[prof] = np.array( [ymin0] * len(self.portals_parameters["solution"][key_rhos]) )

        if enforce_finite_aLT is not None:
            for prof in ['te', 'ti']:
                if prof in ymin:
                    ymin[prof] = np.array(ymin[prof]).clip(min=None,max=enforce_finite_aLT)

        # Initialize
        print(">> PORTALS initalization module (START)", typeMsg="i")
        initializeProblem(
            self,
            self.folder,
            mitim_state,
            ymax,
            ymin,
            start_from_folder=start_from_folder,
            fixed_gradients=fixed_gradients,
            limits_are_relative=limits_are_relative,
            cold_start=cold_start,
            yminymax_atleast=yminymax_atleast,
            tensor_options = self.tensor_options,
            seedInitial=seedInitial,
            checkForSpecies=askQuestions,
        )
        print(">> PORTALS initalization module (END)", typeMsg="i")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Option to cold_start
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if start_from_folder is not None:
            self.reuseTrainingTabular(start_from_folder, self.folder, reevaluate_targets=reevaluate_targets)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ignore targets in surrogate_data.csv
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if 'extrapointsModels' not in self.optimization_options['surrogate_options'] or \
            self.optimization_options['surrogate_options']['extrapointsModels'] is None or \
            len(self.optimization_options['surrogate_options']['extrapointsModels'])==0:

            self._define_reuse_models()

        else:
            print("\t- extrapointsModels already defined, not changing")

        # Make a copy of the namelist that was imported to the folder
        shutil.copy(self.portals_namelist, self.folder / "portals.namelist_original.yaml")

        # Write the parameters (after script modification) to a yaml namelist for tracking purposes
        IOtools.write_mitim_yaml(self.portals_parameters, self.folder / "namelist.portals.yaml")

    # Override: run()
    # ------------------------------------------------------------------

    def run(self, paramsfile, resultsfile):
        """
        Override of ``portals.run()`` that substitutes a ``powerstate_edge`` for
        the standard powerstate inside the model evaluator call.

                Differences from the base class
                ---------------------------------
                - Uses the prep-upgraded ``powerstate_edge`` stored on ``self.powerstate``.
                    ``runModelEvaluator`` still deep-copies it per evaluation, preserving
                    isolation across BO calls.
                - The run name gains an ``_edge`` infix for traceability in logs.
                - All other behaviour (reading DVs, writing OFs, extra storage) is identical
                    to ``portals.run()``.

        Parameters
        ----------
        paramsfile   : Path
        resultsfile  : Path
        """
        # 1. Read what PORTALS sends (unchanged from base)
        FolderEvaluation, numPORTALS, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        _, b = IOtools.reducePathLevel(self.folder, level=1)
        name = f"portals_edge_{b}_ev{numPORTALS}"

        # Keep a consistent scratch-root hint on the template powerstate; the
        # evaluator deep-copies this object for each model call.
        self.powerstate._solver_scratch_folder = (
            FolderEvaluation / "transport_simulation_folder"
        )

        # 2. Run edge evaluator (decoupled from legacy DV naming assumptions)
        powerstate_result, dictOFs = runModelEvaluator_edge(
            self,
            FolderEvaluation,
            dictDVs,
            name,
            numPORTALS=numPORTALS,
            dictOFs=dictOFs,
            remove_folder_upon_completion=not self.portals_parameters["solution"]["keep_full_model_folder"],
        )

        # 4. Write results (unchanged)
        self.write(dictOFs, resultsfile)

        # 5. Extra storage — identical pattern to portals.run()
        if self.optimization_extra is not None:
            dictStore = IOtools.unpickle_mitim(self.optimization_extra)
            dictStore[int(numPORTALS)] = {"powerstate": powerstate_result}
            dictStore["profiles_modified"] = PROFILEStools.gacode_state(
                self.folder / "Initialization" / "input.gacode_modified"
            )
            dictStore["profiles_original"] = PROFILEStools.gacode_state(
                self.folder / "Initialization" / "input.gacode_original"
            )
            with open(self.optimization_extra, "wb") as handle:
                pickle_dill.dump(dictStore, handle, protocol=4)

    def scalarized_objective(self, Y):
        """
        Notes
        -----
                - Y is the multi-output evaluation of the model in the shape of (dim1...N, num_ofs), i.e. this function should not care
                  about number of dimensions
        """

        ofs_ordered_names = np.array(self.optimization_options["problem_options"]["ofs"])

        """
		-------------------------------------------------------------------------
		Prepare transport dictionary
		-------------------------------------------------------------------------
			Note: var_dict['Qe_tr_turb'] must have shape (dim1...N, num_radii)
		"""

        var_dict = {}
        for of in ofs_ordered_names:

            var = '_'.join(of.split("_")[:-1])
            if var not in var_dict:
                var_dict[var] = torch.Tensor().to(Y)
            var_dict[var] = torch.cat((var_dict[var], Y[..., ofs_ordered_names == of]), dim=-1)

        """
		-------------------------------------------------------------------------
		Calculate quantities
		-------------------------------------------------------------------------
			Note: of and cal must have shape (dim1...N, num_radii*num_channels)
				  res must have shape (dim1...N)
		"""

        # Build a shallow proxy whose plasma geometry tensors (rmin, volp, …) are
        # downsampled to the rhoCP grid (with pad_zero=True so that each tensor has
        # length n_rhoCP+1, matching the zero-prepended flux arrays that
        # computeTurbExchangeIndividual / from_density_to_flux expect).  This
        # avoids a grid-length mismatch between var_dict values (n_rhoCP) and the
        # fine-grid plasma arrays stored on self.powerstate.
        proxy_powerstate = copy.copy(self.powerstate)
        proxy_powerstate.plasma = self.powerstate._slice_plasma_to_rhoCP(pad_zero=True)

        of, cal, _, res = PORTALStools.calculate_residuals(proxy_powerstate, self.portals_parameters,specific_vars=var_dict)

        return of, cal, res



def calculate_residuals(powerstate, portals_parameters, specific_vars=None):
    """
    Notes
    -----
        - Works with tensors
        - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    # Case where I have already constructed the dictionary (i.e. in scalarized objective)
    if specific_vars is not None:
        var_dict = specific_vars
    # Prepare dictionary from powerstate (for use in Analysis)
    else:
        var_dict = {}

        mapper = {
            "Qe_tr_turb": "QeMWm2_tr_turb",
            "Qi_tr_turb": "QiMWm2_tr_turb",
            "Ge_tr_turb": "Ge_tr_turb",
            "GZ_tr_turb": "GZ_tr_turb",
            "Mt_tr_turb": "MtJm2_tr_turb",
            "Qe_tr_neoc": "QeMWm2_tr_neoc",
            "Qi_tr_neoc": "QiMWm2_tr_neoc",
            "Ge_tr_neoc": "Ge_tr_neoc",
            "GZ_tr_neoc": "GZ_tr_neoc",
            "Mt_tr_neoc": "MtJm2_tr_neoc",
            "Qe_tar": "QeMWm2",
            "Qi_tar": "QiMWm2",
            "Ge_tar": "Ge",
            "GZ_tar": "GZ",
            "Mt_tar": "MtJm2",
            "Qie_tr_turb": "QieMWm3_tr_turb"
        }

        for ikey in mapper:
            var_dict[ikey] = powerstate.plasma[mapper[ikey]][..., 1:]
            if mapper[ikey] + "_stds" in powerstate.plasma:
                var_dict[ikey + "_stds"] = powerstate.plasma[mapper[ikey] + "_stds"][..., 1:]
            else:
                var_dict[ikey + "_stds"] = None

    dfT = list(var_dict.values())[0]  # as a reference for sizes

    # -------------------------------------------------------------------------
    # Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    # -------------------------------------------------------------------------

    if portals_parameters["solution"]["turbulent_exchange_as_surrogate"]:
        QieMWm2_tr_turb = PORTALStools.computeTurbExchangeIndividual(var_dict["Qie_tr_turb"], powerstate)
    else:
        QieMWm2_tr_turb = torch.zeros(dfT.shape).to(dfT)

    # ------------------------------------------------------------------------
    # Go through each profile that needs to be predicted, calculate components
    # ------------------------------------------------------------------------

    of, cal, res = (
        torch.Tensor().to(dfT),
        torch.Tensor().to(dfT),
        torch.Tensor().to(dfT),
    )
    for prof in powerstate.predicted_channels:
        if prof == "te":
            var = "Qe"
        elif prof == "ti":
            var = "Qi"
        elif prof == "ne":
            var = "Ge"
        elif prof == "nZ":
            var = "GZ"
        elif prof == "w0":
            var = "Mt"

        """
		-----------------------------------------------------------------------------------
		Transport (_tr_turb+_tr_neoc)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}_tr_turb"] + var_dict[f"{var}_tr_neoc"]

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}_tar"] + QieMWm2_tr_turb
        elif var == "Qi":
            cal0 = var_dict[f"{var}_tar"] - QieMWm2_tr_turb
        else:
            cal0 = var_dict[f"{var}_tar"]

        """
		-----------------------------------------------------------------------------------
		Ad-hoc modifications for different weighting
		-----------------------------------------------------------------------------------
		"""

        if var == "Qe":
            of0, cal0 = (
                of0 * portals_parameters["solution"]["scalar_multipliers"][0],
                cal0 * portals_parameters["solution"]["scalar_multipliers"][0],
            )
        elif var == "Qi":
            of0, cal0 = (
                of0 * portals_parameters["solution"]["scalar_multipliers"][1],
                cal0 * portals_parameters["solution"]["scalar_multipliers"][1],
            )
        elif var == "Ge":
            of0, cal0 = (
                of0 * portals_parameters["solution"]["scalar_multipliers"][2],
                cal0 * portals_parameters["solution"]["scalar_multipliers"][2],
            )
        elif var == "GZ":
            of0, cal0 = (
                of0 * portals_parameters["solution"]["scalar_multipliers"][3],
                cal0 * portals_parameters["solution"]["scalar_multipliers"][3],
            )
        elif var == "MtJm2":
            of0, cal0 = (
                of0 * portals_parameters["solution"]["scalar_multipliers"][4],
                cal0 * portals_parameters["solution"]["scalar_multipliers"][4],
            )

        of, cal = torch.cat((of, of0), dim=-1), torch.cat((cal, cal0), dim=-1)

    # -----------
    # Composition
    # -----------

    # Source term is (TARGET - TRANSPORT)
    source = (cal - of)/cal

    # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L2
    res = -1 / source.shape[-1] * torch.norm(source, p=2, dim=-1)

    return of, cal, source, res