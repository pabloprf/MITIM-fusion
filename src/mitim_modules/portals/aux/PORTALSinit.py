import os
import torch
import copy
import numpy as np
from collections import OrderedDict
from mitim_modules.powertorch.physics import TARGETStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.portals import PORTALStools
from mitim_tools.misc_tools.IOtools import printMsg as print
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
    restartYN=False,
    dvs_fixed=None,
    grabFrom=None,
    profileForBase=None,
    dfT=torch.randn((2, 2), dtype=torch.double),
    ModelOptions=None,
    seedInitial=None,
    checkForSpecies=True,
):
    """
    Specification of points occur in rho coordinate, although internally the work is r/a
    restartYN = True if restart from beginning

    I can give ModelOptions directly (e.g. if I want chis or something)

    profileForBase must be PROFILES class
    """

    if seedInitial is not None:
        torch.manual_seed(seed=seedInitial)

    FolderInitialization = folderWork + "Initialization"

    if (restartYN) or (not os.path.exists(folderWork)):
        IOtools.askNewFolder(folderWork, force=restartYN)

    if not os.path.exists(FolderInitialization):
        os.system(f"mkdir {FolderInitialization}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize file input.gacode
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ---- Copy the file of interest to initialization folder

    os.system(f"cp {fileStart} {FolderInitialization}/input.gacode")

    # ---- Make another copy to preserve the original state

    os.system(
        f"cp {FolderInitialization}/input.gacode {FolderInitialization}/input.gacode_original_originalResol_uncorrected"
    )

    # ---- Initialize file to modify and increase resolution

    initialization_file = f"{FolderInitialization}/input.gacode"
    profiles = PROFILEStools.PROFILES_GACODE(initialization_file)

    # About radial locations
    if "RoaLocations" in portals_fun.TGYROparameters:
        roa = portals_fun.TGYROparameters["RoaLocations"]
        rho = np.interp(roa, profiles.derived["roa"], profiles.profiles["rho(-)"])
        print("\t * r/a provided, transforming to rho:")
        print(f"\t\t r/a = {roa}")
        print(f"\t\t rho = {rho}")
        portals_fun.TGYROparameters["RhoLocations"] = rho

    if (
        len(INITparameters["removeIons"]) > 0
        or INITparameters["removeFast"]
        or INITparameters["quasineutrality"]
        or INITparameters["sameDensityGradients"]
        or INITparameters["recompute_ptot"]
    ):
        profiles.correct(options=INITparameters)

    # Resolution of input.gacode
    defineNewGridmitim(profiles, np.array(portals_fun.TGYROparameters["RhoLocations"]))

    # After resolution and corrections, store.
    profiles.writeCurrentStatus(file=f"{FolderInitialization}/input.gacode_original")

    if portals_fun.PORTALSparameters["UseOriginalImpurityConcentrationAsWeight"]:
        portals_fun.PORTALSparameters["fImp_orig"] = profiles.Species[
            portals_fun.PORTALSparameters["ImpurityOfInterest"] - 1
        ]["dens"]
        print(
            f"\t- Using original concentration of {portals_fun.PORTALSparameters['fImp_orig']:.2e} for ion {portals_fun.PORTALSparameters['ImpurityOfInterest']} as scaling factor of GZ",
            typeMsg="i",
        )
    else:
        portals_fun.PORTALSparameters["fImp_orig"] = 1.0

    # Check if I will be able to calculate radiation
    if checkForSpecies:
        speciesNotFound = []
        for i in range(len(profiles.Species)):
            c = TARGETStools.get_chebyshev_coeffs(profiles.Species[i]['N'])
            if c[0]<=-1E10:
                speciesNotFound.append(profiles.Species[i]['N'])
        if len(speciesNotFound)>0:
            a = print(f"\t- Species {speciesNotFound} not found, radiation will be zero in PORTALS. Make sure this is ok with your predictions", typeMsg="q")
            if not a:
                raise ValueError("Species not found")
    """
	***************************************************************************************************
											POWER STATE
	***************************************************************************************************
	"""

    xCPs = torch.from_numpy(
        np.append([0], np.array(portals_fun.TGYROparameters["RhoLocations"]))
    ).to(
        dfT
    )  # Added zero

    # Define powerstate with the de-parameterization functions
    portals_fun.powerstate = STATEtools.powerstate(
        profiles,
        xCPs,
        ProfilesPredicted=portals_fun.TGYROparameters["ProfilesPredicted"],
        TargetOptions={
            "TypeTarget": portals_fun.TGYROparameters["TGYRO_physics_options"][
                "TargetType"
            ],
            "TargetCalc": portals_fun.PORTALSparameters["TargetCalc"],
        },  #'tgyro' if 'tgyro' in portals_fun.PORTALSparameters['model_used'] else 'powerstate'},
        useConvectiveFluxes=portals_fun.PORTALSparameters["useConvectiveFluxes"],
        impurityPosition=portals_fun.PORTALSparameters["ImpurityOfInterest"],
        fineTargetsResolution=portals_fun.PORTALSparameters["fineTargetsResolution"],
    )

    # Store parameterization in dictCPs_base (to define later the relative variations) and modify profiles class with parameterized profiles
    dictCPs_base = {}
    for name in portals_fun.TGYROparameters["ProfilesPredicted"]:
        dictCPs_base[name] = portals_fun.powerstate.update_var(name, var=None)[0, :]

    # Maybe it was provided from earlier run
    if grabFrom is not None:
        dictCPs_base = grabPrevious(grabFrom, dictCPs_base)
        for name in portals_fun.TGYROparameters["ProfilesPredicted"]:
            _ = portals_fun.powerstate.update_var(
                name, var=dictCPs_base[name].unsqueeze(0)
            )

    # Write this updated profiles class (with parameterized profiles)
    _ = portals_fun.powerstate.insertProfiles(
        profiles,
        writeFile=f"{FolderInitialization}/input.gacode",
        applyCorrections=portals_fun.TGYROparameters["applyCorrections"],
    )

    # Original complete targets
    portals_fun.powerstate.calculateProfileFunctions()
    portals_fun.powerstate.calculateTargets()

    # Prepare powerstate for evaluations
    portals_fun.powerstate.TransportOptions[
        "TypeTransport"
    ] = portals_fun.PORTALSparameters["model_used"]
    if ModelOptions is not None:
        portals_fun.powerstate.TransportOptions["ModelOptions"] = ModelOptions
    else:
        portals_fun.powerstate.TransportOptions["ModelOptions"] = {
            "restart": False,
            "launchTGYROviaSlurm": portals_fun.PORTALSparameters[
                "launchEvaluationsAsSlurmJobs"
            ],
            "TGYROparameters": portals_fun.TGYROparameters,
            "TGLFparameters": portals_fun.TGLFparameters,
            "includeFastInQi": portals_fun.PORTALSparameters["includeFastInQi"],
            "TurbulentExchange": portals_fun.PORTALSparameters["surrogateForTurbExch"],
            "profiles_postprocessing_fun": portals_fun.PORTALSparameters[
                "profiles_postprocessing_fun"
            ],
            "impurityPosition": portals_fun.PORTALSparameters["ImpurityOfInterest"],
            "useConvectiveFluxes": portals_fun.PORTALSparameters["useConvectiveFluxes"],
            "UseFineGridTargets": portals_fun.PORTALSparameters[
                "fineTargetsResolution"
            ],
            "OriginalFimp": portals_fun.PORTALSparameters["fImp_orig"],
            "forceZeroParticleFlux": portals_fun.PORTALSparameters[
                "forceZeroParticleFlux"
            ],
            "percentError": portals_fun.PORTALSparameters["percentError"],
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define input dictionaries (Define ranges of variation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (
        profileForBase is not None
    ):  # If I want to define ranges from a different profile
        powerstate_extra = STATEtools.powerstate(
            profileForBase,
            xCPs,
            ProfilesPredicted=portals_fun.TGYROparameters["ProfilesPredicted"],
            TargetOptions={
                "TypeTarget": portals_fun.TGYROparameters["TGYRO_physics_options"][
                    "TargetType"
                ],
                "TargetCalc": portals_fun.PORTALSparameters["TargetCalc"],
            },
            useConvectiveFluxes=portals_fun.PORTALSparameters["useConvectiveFluxes"],
        )

        dictCPs_base_extra = {}
        for name in portals_fun.TGYROparameters["ProfilesPredicted"]:
            dictCPs_base_extra[name] = powerstate_extra.update_var(name, var=None)[0, :]

        dictCPs_base = dictCPs_base_extra

    dictDVs = OrderedDict()
    for cont, var in enumerate(dictCPs_base):
        for conti, i in enumerate(np.arange(1, len(dictCPs_base[var]))):
            if limitsAreRelative:
                y1 = dictCPs_base[var][i] * (1 - RelVar_y_min[cont][conti])
                y2 = dictCPs_base[var][i] * (1 + RelVar_y_max[cont][conti])
            else:
                # y1 = dictCPs_base[var][i] - RelVar_y_min[cont][conti]
                # y2 = dictCPs_base[var][i] + RelVar_y_max[cont][conti]
                y1 = torch.tensor(RelVar_y_min[cont][conti]).to(dfT)
                y2 = torch.tensor(RelVar_y_max[cont][conti]).to(dfT)

            if hardGradientLimits is not None:
                y1 = torch.tensor(np.min([y1, hardGradientLimits[0]]))
                y2 = torch.tensor(np.max([y2, hardGradientLimits[1]]))

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

        for i in range(len(portals_fun.TGYROparameters["RhoLocations"])):
            ofs.append(f"{var}Turb_{i+1}")
            ofs.append(f"{var}Neo_{i+1}")

            ofs.append(f"{var}Tar_{i+1}")

            name_objectives.append(f"{var}Res_{i+1}")

    if portals_fun.PORTALSparameters["surrogateForTurbExch"]:
        for i in range(len(portals_fun.TGYROparameters["RhoLocations"])):
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
    portals_fun.Optim["ofs"] = ofs
    portals_fun.Optim["dvs"] = [*dictDVs]
    portals_fun.Optim["cvs"] = []
    portals_fun.Optim["dvs_min"] = []
    for i in dictDVs:
        portals_fun.Optim["dvs_min"].append(dictDVs[i][0].cpu().numpy())
    portals_fun.Optim["BaselineDV"] = []
    for i in dictDVs:
        portals_fun.Optim["BaselineDV"].append(dictDVs[i][1].cpu().numpy())
    portals_fun.Optim["dvs_max"] = []
    for i in dictDVs:
        portals_fun.Optim["dvs_max"].append(dictDVs[i][2].cpu().numpy())

    portals_fun.Optim["dvs_min"] = np.array(portals_fun.Optim["dvs_min"])
    portals_fun.Optim["dvs_max"] = np.array(portals_fun.Optim["dvs_max"])
    portals_fun.Optim["BaselineDV"] = np.array(portals_fun.Optim["BaselineDV"])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # For surrogate
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Variables = {}
    for ikey in portals_fun.PORTALSparameters["physicsBasedParams"]:
        Variables[ikey] = prepPhysicsBasedParams(portals_fun, ikey)

    portals_fun.surrogate_parameters = {
        "transformationInputs": PORTALStools.produceNewInputs,
        "transformationOutputs": PORTALStools.transformmitim,
        "powerstate": portals_fun.powerstate,
        "applyImpurityGammaTrick": portals_fun.PORTALSparameters[
            "applyImpurityGammaTrick"
        ],
        "useFluxRatios": portals_fun.PORTALSparameters["useFluxRatios"],
        "useDiffusivities": portals_fun.PORTALSparameters["useDiffusivities"],
        "physicsInformedParams_dict": Variables,
        "physicsInformedParamsComplete": copy.deepcopy(
            Variables[list(Variables.keys())[-1]]
        ),
        "parameters_combined": {},
    }

    # Pass an text version of the initialization file, so that I can run mitim without having the initializaiton folder
    # in the same computer
    with open(initialization_file, "r") as f:
        portals_fun.file_in_lines_initial_input_gacode = f.readlines()


def defineNewGridmitim(profiles, rhoTGYRO):
    """
    Resolution of input.gacode
    **************************
    - Change resolution to a fine grid in which doing the flattening around coarse points has a small effect on the profile.
    - It is recommended that it goes through the points, with more points around the trailing edge transition.
    - Also, avoid adding too points near axis. (NOT NOW?)
    """

    # ----------------------------------------
    # Parameters
    # ----------------------------------------

    total_points = 100
    d_spacing_coarse = 1e-3  # 1/200 #1E-3
    points_updown = 2

    # ----------------------------------------------------------------------------------
    # 1. Fill up spaces in between the points until the total is total_points
    # ----------------------------------------------------------------------------------

    num_points_rest = int(np.max([3, total_points / (len(rhoTGYRO) + 1)]))

    # Correction: If I do a very fine grid, but with a first point away from 0.0, it'll	have a piecewise behavior from 0 to the first points, so ensure a few more
    num_points_0 = int(np.max([num_points_rest, 10]))
    # *******

    rho_new0 = np.append(np.append([0], rhoTGYRO), [1])
    rho_new = np.array([])
    for i in range(rho_new0.shape[0] - 1):
        num_points = num_points_rest if i > 0 else num_points_0
        rho_new = np.append(
            rho_new, np.linspace(rho_new0[i], rho_new0[i + 1], num_points)
        )

    # ----------------------------------------------------------------------------------
    # 2. Add extra resolution around the TGYRO poitns
    # ----------------------------------------------------------------------------------

    for i in range(points_updown):
        rho_new = np.append(
            np.append(rho_new, rhoTGYRO + d_spacing_coarse * (i + 1)),
            rhoTGYRO - d_spacing_coarse * (i + 1),
        )

    # ----------------------------------------------------------------------------------
    # Change resolution
    # ----------------------------------------------------------------------------------
    profiles.changeResolution(rho_new=rho_new)


def prepPhysicsBasedParams(portals_fun, ikey, doNotFitOnFixedValues=False):
    allOuts = portals_fun.Optim["ofs"]
    physicsBasedParams = portals_fun.PORTALSparameters["physicsBasedParams"][ikey]
    physicsBasedParams_trace = portals_fun.PORTALSparameters[
        "physicsBasedParams_trace"
    ][ikey]

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
                "aLte": "te" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLti": "ti" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLne": "ne" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLw0": "w0" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "te": ("te" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ti": ("ti" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ne": ("ne" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "w0": ("w0" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
            }

            Variables[output] = []
            for ikey in physicsBasedParams:
                useThisOne = False
                for varis in physicsBasedParams[ikey]:
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
                "aLte": "te" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLti": "ti" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLne": "ne" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLw0": "w0" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "aLnZ": "nZ" in portals_fun.TGYROparameters["ProfilesPredicted"],
                "te": ("te" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ti": ("ti" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "ne": ("ne" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "w0": ("w0" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
                "nZ": ("nZ" in portals_fun.TGYROparameters["ProfilesPredicted"])
                and (not isAbsValFixed),
            }

            Variables[output] = []
            for ikey in physicsBasedParams_trace:
                useThisOne = False
                for varis in physicsBasedParams_trace[ikey]:
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
    from mitim_tools.opt_tools.STRATEGYtools import FUNmain

    opt_fun = FUNmain(foldermitim)
    opt_fun.read_optimization_results(plotYN=False, analysis_level=1)
    x = opt_fun.prfs_model.BOmetrics["overall"]["xBest"].cpu().numpy()
    dvs = opt_fun.prfs_model.Optim["dvs"]
    dvs_dict = {}
    for j in range(len(dvs)):
        dvs_dict[dvs[j]] = x[j]

    print(
        f"- Grabbing best #{opt_fun.prfs_model.BOmetrics['overall']['indBest']} from previous workflow",
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
