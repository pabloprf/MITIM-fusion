import shutil
import torch
import copy
import numpy as np
import pandas as pd
from collections import OrderedDict
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.plasmastate_tools import MITIMstate
from mitim_modules.powertorch import STATEtools
from mitim_modules.portals import PORTALStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed


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
        - define_ranges_from_profiles must be PROFILES class
    """

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
    
    # Add folder and cold_start to the simulation options
    transport_options = transport_parameters | {"folder": portals_fun.folder, "cold_start": False}
    target_options = portals_fun.portals_parameters["target"]

    portals_fun.powerstate = STATEtools.powerstate(
        profiles,
        evolution_options={
            "ProfilePredicted": portals_fun.portals_parameters["solution"]["predicted_channels"],
            "rhoPredicted": xCPs,
            "impurityPosition": position_of_impurity,
            "fImp_orig": portals_fun.portals_parameters["solution"]["fImp_orig"]
        },
        transport_options=transport_options,
        target_options=target_options,
        tensor_options=tensor_options
    )

    # After resolution and corrections, store.
    profiles.write_state(file=FolderInitialization / "input.gacode_modified")

    # ***************************************************************************************************
    # ***************************************************************************************************

    # Store parameterization in dictCPs_base (to define later the relative variations) and modify profiles class with parameterized profiles
    dictCPs_base = {}
    for name in portals_fun.portals_parameters["solution"]["predicted_channels"]:
        dictCPs_base[name] = portals_fun.powerstate.update_var(name, var=None)[0, :]

    # Maybe it was provided from earlier run
    if start_from_folder is not None:
        dictCPs_base = grabPrevious(start_from_folder, dictCPs_base)
        for name in portals_fun.portals_parameters["solution"]["predicted_channels"]:
            _ = portals_fun.powerstate.update_var(
                name, var=dictCPs_base[name].unsqueeze(0)
            )

    # Write this updated profiles class (with parameterized profiles)
    _ = portals_fun.powerstate.from_powerstate(
        write_input_gacode=FolderInitialization / "input.gacode",
        postprocess_input_gacode=portals_fun.portals_parameters["transport"]["applyCorrections"],
    )

    # Original complete targets
    portals_fun.powerstate.calculateProfileFunctions()
    portals_fun.powerstate.calculateTargets()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define input dictionaries (Define ranges of variation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if define_ranges_from_profiles is not None:  # If I want to define ranges from a different profile
        powerstate_extra = STATEtools.powerstate(
            define_ranges_from_profiles,
            evolution_options={
                "ProfilePredicted": portals_fun.portals_parameters["solution"]["predicted_channels"],
                "rhoPredicted": xCPs,
                "impurityPosition": position_of_impurity,
                "fImp_orig": portals_fun.portals_parameters["solution"]["fImp_orig"]
            },
            target_options=portals_fun.portals_parameters["target"],
            tensor_options = tensor_options
        )

        dictCPs_base_extra = {}
        for name in portals_fun.portals_parameters["solution"]["predicted_channels"]:
            dictCPs_base_extra[name] = powerstate_extra.update_var(name, var=None)[0, :]

        dictCPs_base = dictCPs_base_extra

    thr = 1E-5

    dictDVs = OrderedDict()
    for var in dictCPs_base:
        for conti, i in enumerate(np.arange(1, len(dictCPs_base[var]))):
            if limits_are_relative:
                y1 = dictCPs_base[var][i] - abs(dictCPs_base[var][i])*RelVar_y_min[var][conti]
                y2 = dictCPs_base[var][i] + abs(dictCPs_base[var][i])*RelVar_y_max[var][conti]
            else:
                y1 = torch.tensor(RelVar_y_min[var][conti]).to(dfT)
                y2 = torch.tensor(RelVar_y_max[var][conti]).to(dfT)

            if yminymax_atleast is not None:
                if yminymax_atleast[0] is not None:
                    y1 = torch.tensor(np.min([y1, yminymax_atleast[0]]))
                if yminymax_atleast[1] is not None:
                    y2 = torch.tensor(np.max([y2, yminymax_atleast[1]]))

            # Check that makes sense
            if y2-y1 < thr:
                print(f"{var} @ pos={i} has a range of {y2-y1:.1e} which is less than {thr:.1e}",typeMsg="q")

            if (seedInitial is None) or (seedInitial == 0):
                base_gradient = dictCPs_base[var][i]
            else:
                # Special case where I want to randomize the initial starting case with a half bounds
                base_gradient = torch.rand(1)[0] * (y2 - y1) / 4 + (3 * y1 + y2) / 4

            name = f"aL{var}_{i}"
            if fixed_gradients is None:
                dictDVs[name] = [y1, base_gradient, y2]
            else:
                dictDVs[name] = [fixed_gradients[name][0], base_gradient, fixed_gradients[name][1]]

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
    additional_params_in_surrogate = portals_fun.portals_parameters["solution"]["additional_params_in_surrogate"]

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

            for kkey in additional_params_in_surrogate:
                Variations[kkey] = True

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

            for kkey in additional_params_in_surrogate:
                Variations[kkey] = True

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


def grabPrevious(foldermitim, dictCPs_base):
    from mitim_tools.opt_tools.STRATEGYtools import opt_evaluator

    opt_fun = opt_evaluator(foldermitim)
    opt_fun.read_optimization_results(analysis_level=1)
    x = opt_fun.mitim_model.BOmetrics["overall"]["xBest"].cpu().numpy()
    dvs = opt_fun.mitim_model.optimization_options["problem_options"]["dvs"]
    dvs_dict = {}
    for j in range(len(dvs)):
        dvs_dict[dvs[j]] = x[j]

    print(
        f"- Grabbing best #{opt_fun.mitim_model.BOmetrics['overall']['indBest']} from previous workflow",
        typeMsg="i",
    )

    for ikey in dictCPs_base:
        for ir in range(len(dictCPs_base[ikey]) - 1):
            ikey_mod = f"aL{ikey}_{ir+1}"
            try:
                dictCPs_base[ikey][ir + 1] = dvs_dict[ikey_mod]
            except:
                pass

    return dictCPs_base

