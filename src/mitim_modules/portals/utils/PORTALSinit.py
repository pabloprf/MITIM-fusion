import shutil
import torch
import copy
import numpy as np
import pandas as pd
from collections import OrderedDict
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.portals import PORTALStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed


def initializeProblem(
    portals_fun,
    folderWork,
    fileStart,
    INITparameters,
    RelVar_y_max,
    RelVar_y_min,
    limitsAreRelative=True,
    hardGradientLimits=None,
    cold_start=False,
    dvs_fixed=None,
    start_from_folder=None,
    define_ranges_from_profiles=None,
    ModelOptions=None,
    seedInitial=None,
    checkForSpecies=True,
    tensor_opts = {
        "dtype": torch.double,
        "device": torch.device("cpu"),
    }
    ):
    """
    Notes:
        - Specification of points occur in rho coordinate, although internally the work is r/a
            cold_start = True if cold_start from beginning
        - I can give ModelOptions directly (e.g. if I want chis or something)
        - define_ranges_from_profiles must be PROFILES class
    """

    dfT = torch.randn((2, 2), **tensor_opts)

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

    shutil.copy2(fileStart, FolderInitialization / "input.gacode")

    # ---- Make another copy to preserve the original state

    shutil.copy2(FolderInitialization / "input.gacode", FolderInitialization / "input.gacode_original")

    # ---- Initialize file to modify and increase resolution

    initialization_file = FolderInitialization / "input.gacode"
    profiles = PROFILEStools.PROFILES_GACODE(initialization_file)

    # About radial locations
    if portals_fun.MODELparameters["RoaLocations"] is not None:
        roa = portals_fun.MODELparameters["RoaLocations"]
        rho = np.interp(roa, profiles.derived["roa"], profiles.profiles["rho(-)"])
        print("\t * r/a provided, transforming to rho:")
        print(f"\t\t r/a = {roa}")
        print(f"\t\t rho = {rho}")
        portals_fun.MODELparameters["RhoLocations"] = rho

    if (
        len(INITparameters["removeIons"]) > 0
        or INITparameters["removeFast"]
        or INITparameters["quasineutrality"]
        or INITparameters["sameDensityGradients"]
        or INITparameters["recompute_ptot"]
    ):
        profiles.correct(options=INITparameters)

    if portals_fun.PORTALSparameters["ImpurityOfInterest"] is not None:
        position_of_impurity = PROFILEStools.impurity_location(profiles, portals_fun.PORTALSparameters["ImpurityOfInterest"])
    else:
        position_of_impurity = 1

    if portals_fun.PORTALSparameters["UseOriginalImpurityConcentrationAsWeight"] is not None and portals_fun.PORTALSparameters["ImpurityOfInterest"] is not None:
        f0 = profiles.Species[position_of_impurity]["n0"] / profiles.profiles['ne(10^19/m^3)'][0]
        portals_fun.PORTALSparameters["fImp_orig"] = f0/portals_fun.PORTALSparameters["UseOriginalImpurityConcentrationAsWeight"]
        print(f'\t- Ion {portals_fun.PORTALSparameters["ImpurityOfInterest"]} has original central concentration of {f0:.2e}, using its inverse multiplied by {portals_fun.PORTALSparameters["UseOriginalImpurityConcentrationAsWeight"]} as scaling factor of GZ -> {portals_fun.PORTALSparameters["fImp_orig"]}',typeMsg="i")
    else:
        portals_fun.PORTALSparameters["fImp_orig"] = 1.0

    # Check if I will be able to calculate radiation
    speciesNotFound = []
    for i in range(len(profiles.Species)):
        data_df = pd.read_csv(__mitimroot__ / "src" / "mitim_modules" / "powertorch" / "physics" / "radiation_chebyshev.csv")
        if not (data_df['Ion'].str.lower()==profiles.Species[i]["N"].lower()).any():
            speciesNotFound.append(profiles.Species[i]["N"])

    # Print warning or question to be careful!
    if len(speciesNotFound) > 0:

        if portals_fun.MODELparameters["Physics_options"]["TypeTarget"] == 3:
        
            answerYN = print(f"\t- Species {speciesNotFound} not found in radiation database, radiation will be zero in PORTALS... is this ok for your predictions?",typeMsg="q" if checkForSpecies else "w")
            if checkForSpecies and (not answerYN):
                raise ValueError("Species not found")

        else:

            print(f'\t- Species {speciesNotFound} not found in radiation database, but this PORTALS prediction is not calculating radiation anyway',typeMsg="w")

    # Prepare and defaults

    xCPs = torch.from_numpy(np.array(portals_fun.MODELparameters["RhoLocations"])).to(dfT)

    if ModelOptions is None:
        ModelOptions = {
            "cold_start": False,
            "launchMODELviaSlurm": portals_fun.PORTALSparameters[
                "launchEvaluationsAsSlurmJobs"
            ],
            "MODELparameters": portals_fun.MODELparameters,
            "includeFastInQi": portals_fun.PORTALSparameters["includeFastInQi"],
            "TurbulentExchange": portals_fun.PORTALSparameters["surrogateForTurbExch"],
            "profiles_postprocessing_fun": portals_fun.PORTALSparameters[
                "profiles_postprocessing_fun"
            ],
            "impurityPosition": position_of_impurity,
            "useConvectiveFluxes": portals_fun.PORTALSparameters["useConvectiveFluxes"],
            "UseFineGridTargets": portals_fun.PORTALSparameters["fineTargetsResolution"],
            "OriginalFimp": portals_fun.PORTALSparameters["fImp_orig"],
            "forceZeroParticleFlux": portals_fun.PORTALSparameters[
                "forceZeroParticleFlux"
            ],
            "percentError": portals_fun.PORTALSparameters["percentError"],
            "use_tglf_scan_trick": portals_fun.PORTALSparameters["use_tglf_scan_trick"],
        }

    if "extra_params" not in ModelOptions:
        ModelOptions["extra_params"] = {
            "PORTALSparameters": portals_fun.PORTALSparameters,
            "folder": portals_fun.folder,
        }


    """
    ***************************************************************************************************
                                powerstate object
    ***************************************************************************************************
    """

    portals_fun.powerstate = STATEtools.powerstate(
        profiles,
        EvolutionOptions={
            "ProfilePredicted": portals_fun.MODELparameters["ProfilesPredicted"],
            "rhoPredicted": xCPs,
            "useConvectiveFluxes": portals_fun.PORTALSparameters["useConvectiveFluxes"],
            "impurityPosition": position_of_impurity,
            "fineTargetsResolution": portals_fun.PORTALSparameters["fineTargetsResolution"],
        },
        TransportOptions={
               "transport_evaluator": portals_fun.PORTALSparameters["transport_evaluator"],
               "ModelOptions": ModelOptions,
        },
        TargetOptions={
            "targets_evaluator": portals_fun.PORTALSparameters["targets_evaluator"],
            "ModelOptions": {
                "TypeTarget": portals_fun.MODELparameters["Physics_options"]["TypeTarget"],
                "TargetCalc": portals_fun.PORTALSparameters["TargetCalc"]},
        },
        tensor_opts = tensor_opts
    )

    # After resolution and corrections, store.
    profiles.writeCurrentStatus(file=FolderInitialization / "input.gacode_modified")

    # ***************************************************************************************************
    # ***************************************************************************************************

    # Store parameterization in dictCPs_base (to define later the relative variations) and modify profiles class with parameterized profiles
    dictCPs_base = {}
    for name in portals_fun.MODELparameters["ProfilesPredicted"]:
        dictCPs_base[name] = portals_fun.powerstate.update_var(name, var=None)[0, :]

    # Maybe it was provided from earlier run
    if start_from_folder is not None:
        dictCPs_base = grabPrevious(start_from_folder, dictCPs_base)
        for name in portals_fun.MODELparameters["ProfilesPredicted"]:
            _ = portals_fun.powerstate.update_var(
                name, var=dictCPs_base[name].unsqueeze(0)
            )

    # Write this updated profiles class (with parameterized profiles)
    _ = portals_fun.powerstate.from_powerstate(
        write_input_gacode=FolderInitialization / "input.gacode",
        postprocess_input_gacode=portals_fun.MODELparameters["applyCorrections"],
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
            EvolutionOptions={
                "ProfilePredicted": portals_fun.MODELparameters["ProfilesPredicted"],
                "rhoPredicted": xCPs,
                "useConvectiveFluxes": portals_fun.PORTALSparameters["useConvectiveFluxes"],
                "impurityPosition": position_of_impurity,
                "fineTargetsResolution": portals_fun.PORTALSparameters["fineTargetsResolution"],
            },
            TargetOptions={
                "targets_evaluator": portals_fun.PORTALSparameters["targets_evaluator"],
                "ModelOptions": {
                    "TypeTarget": portals_fun.MODELparameters["Physics_options"]["TypeTarget"],
                    "TargetCalc": portals_fun.PORTALSparameters["TargetCalc"]},
            },
            tensor_opts = tensor_opts
        )

        dictCPs_base_extra = {}
        for name in portals_fun.MODELparameters["ProfilesPredicted"]:
            dictCPs_base_extra[name] = powerstate_extra.update_var(name, var=None)[0, :]

        dictCPs_base = dictCPs_base_extra

    thr = 1E-5

    dictDVs = OrderedDict()
    for var in dictCPs_base:
        for conti, i in enumerate(np.arange(1, len(dictCPs_base[var]))):
            if limitsAreRelative:
                y1 = dictCPs_base[var][i] - abs(dictCPs_base[var][i])*RelVar_y_min[var][conti]
                y2 = dictCPs_base[var][i] + abs(dictCPs_base[var][i])*RelVar_y_max[var][conti]
            else:
                y1 = torch.tensor(RelVar_y_min[var][conti]).to(dfT)
                y2 = torch.tensor(RelVar_y_max[var][conti]).to(dfT)

            if hardGradientLimits is not None:
                if hardGradientLimits[0] is not None:
                    y1 = torch.tensor(np.min([y1, hardGradientLimits[0]]))
                if hardGradientLimits[1] is not None:
                    y2 = torch.tensor(np.max([y2, hardGradientLimits[1]]))

            # Check that makes sense
            if y2-y1 < thr:
                print(f"{var} @ pos={i} has a range of {y2-y1:.1e} which is less than {thr:.1e}",typeMsg="q")

            if (seedInitial is None) or (seedInitial == 0):
                base_gradient = dictCPs_base[var][i]
            else:
                # Special case where I want to randomize the initial starting case with a half bounds
                base_gradient = torch.rand(1)[0] * (y2 - y1) / 4 + (3 * y1 + y2) / 4

            name = f"aL{var}_{i}"
            if dvs_fixed is None:
                dictDVs[name] = [y1, base_gradient, y2]
            else:
                dictDVs[name] = [dvs_fixed[name][0], base_gradient, dvs_fixed[name][1]]

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

        for i in range(len(portals_fun.MODELparameters["RhoLocations"])):
            ofs.append(f"{var}Turb_{i+1}")
            ofs.append(f"{var}Neo_{i+1}")

            ofs.append(f"{var}Tar_{i+1}")

            name_objectives.append(f"{var}Res_{i+1}")

    if portals_fun.PORTALSparameters["surrogateForTurbExch"]:
        for i in range(len(portals_fun.MODELparameters["RhoLocations"])):
            ofs.append(f"PexchTurb_{i+1}")

    name_transformed_ofs = []
    for of in ofs:
        if ("GZ" in of) and (portals_fun.PORTALSparameters["applyImpurityGammaTrick"]):
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
    for ikey in portals_fun.PORTALSparameters["portals_transformation_variables"]:
        Variables[ikey] = prepportals_transformation_variables(portals_fun, ikey)

    portals_fun.surrogate_parameters = {
        "transformationInputs": PORTALStools.input_transform_portals,
        "transformationOutputs": PORTALStools.output_transform_portals,
        "powerstate": portals_fun.powerstate,
        "applyImpurityGammaTrick": portals_fun.PORTALSparameters["applyImpurityGammaTrick"],
        "useFluxRatios": portals_fun.PORTALSparameters["useFluxRatios"],
        "useDiffusivities": portals_fun.PORTALSparameters["useDiffusivities"],
        "surrogate_transformation_variables_alltimes": Variables,
        "surrogate_transformation_variables_lasttime": copy.deepcopy(Variables[list(Variables.keys())[-1]]),
        "parameters_combined": {},
    }

def prepportals_transformation_variables(portals_fun, ikey, doNotFitOnFixedValues=False):
    allOuts = portals_fun.optimization_options["problem_options"]["ofs"]
    portals_transformation_variables = portals_fun.PORTALSparameters["portals_transformation_variables"][ikey]
    portals_transformation_variables_trace = portals_fun.PORTALSparameters[
        "portals_transformation_variables_trace"
    ][ikey]
    additional_params_in_surrogate = portals_fun.PORTALSparameters[
        "additional_params_in_surrogate"
    ]

    Variables = {}
    for output in allOuts:
        if IOtools.isfloat(output):
            continue

        typ, num = output.split("_")
        pos = int(num)

        if typ in [
            "Qe",
            "QeTurb",
            "QeNeo",
            "Qi",
            "QiTurb",
            "QiNeo",
            "Ge",
            "GeTurb",
            "GeNeo",
            "PexchTurb",
            "Mt",
            "MtTurb",
            "MtNeo",
        ]:
            if doNotFitOnFixedValues:
                isAbsValFixed = pos == (
                    portals_fun.powerstate.plasma["rho"].shape[-1] - 1
                )
            else:
                isAbsValFixed = False

            Variations = {
                "aLte": "te" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLti": "ti" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLne": "ne" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLw0": "w0" in portals_fun.MODELparameters["ProfilesPredicted"],
                "te": ("te" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ti": ("ti" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ne": ("ne" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "w0": ("w0" in portals_fun.MODELparameters["ProfilesPredicted"])
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

        elif typ in ["GZ", "GZTurb", "GZNeo"]:
            if doNotFitOnFixedValues:
                isAbsValFixed = pos == (
                    portals_fun.powerstate.plasma["rho"].shape[-1] - 1
                )
            else:
                isAbsValFixed = False

            Variations = {
                "aLte": "te" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLti": "ti" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLne": "ne" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLw0": "w0" in portals_fun.MODELparameters["ProfilesPredicted"],
                "aLnZ": "nZ" in portals_fun.MODELparameters["ProfilesPredicted"],
                "te": ("te" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ti": ("ti" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ne": ("ne" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "w0": ("w0" in portals_fun.MODELparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "nZ": ("nZ" in portals_fun.MODELparameters["ProfilesPredicted"])
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

        elif typ in ["QeTar"]:
            Variables[output] = ["PeGB"]
        elif typ in ["QiTar"]:
            Variables[output] = ["PiGB"]
        elif typ in ["GeTar"]:
            Variables[output] = ["CeGB"]
        elif typ in ["GZTar"]:
            Variables[output] = ["CZGB"]
        elif typ in ["MtTar"]:
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

