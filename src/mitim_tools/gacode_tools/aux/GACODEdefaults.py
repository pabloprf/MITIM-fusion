import json
import numpy as np
from mitim_tools.misc_tools import IOtools

from IPython import embed

from mitim_tools.misc_tools.IOtools import printMsg as print


def addTGLFcontrol(TGLFsettings, NS=2, minimal=False):
    """
    ********************************************************************************
    Define dictionary to start with
    ********************************************************************************
    """

    # Minimum working set
    if minimal or TGLFsettings == 0:
        TGLFoptions = {
            "USE_MHD_RULE": True,
            "USE_BPER": False,
            "USE_BPAR": False,
            "SAT_RULE": 1,
            "NKY": 12,
            "KYGRID_MODEL": 1,
            "ADIABATIC_ELEC": False,
            "ALPHA_ZF": 1.0,
            "ETG_FACTOR": 1.28,
            "XNU_MODEL": 2,
        }

    # Define every flag
    else:
        TGLFoptions = IOtools.generateMITIMNamelist(
            "$MITIM_PATH/templates/input.tglf.controls", caseInsensitive=False
        )
        TGLFoptions["NMODES"] = NS + 2

    """
	********************************************************************************
	Standard sets of TGLF control parameters
	  (rest of parameters are as defaults)
	********************************************************************************
	"""

    with open(
        IOtools.expandPath("$MITIM_PATH/templates/input.tglf.models.json"), "r"
    ) as f:
        settings = json.load(f)

    if str(TGLFsettings) in settings:
        sett = settings[str(TGLFsettings)]
        label = sett["label"]
        for ikey in sett["controls"]:
            TGLFoptions[ikey] = sett["controls"][ikey]
    else:
        print(
            "\t- TGLFsettings not found in input.tglf.models.json, using defaults",
            typeMsg="w",
        )
        label = "unspecified"

    # --------------------------------
    # From dictionary to text
    # --------------------------------

    TGLFinput = [""]
    for ikey in TGLFoptions:
        TGLFinput.append(f"{ikey} = {TGLFoptions[ikey]}")
    TGLFinput.append("")
    TGLFinput.append("# -- Begin overlay")
    TGLFinput.append("")

    return TGLFinput, TGLFoptions, label


def TGLFinTRANSP(TGLFsettings, NS=3):
    _, TGLFoptions, label = addTGLFcontrol(TGLFsettings, NS=NS)

    """
	------------------------------------------------------------------------------------------------------
	TRANSP TGLF namelist has some modifications...
	------------------------------------------------------------------------------------------------------
	"""

    # **** Modifications from TGLF standalone

    change_from_to = {
        "USE_TRANSPORT_MODEL": None,
        "SIGN_IT": None,
        "SIGN_BT": None,
        "ALPHA_E": None,
        "NN_MAX_ERROR": None,
        "GEOMETRY_FLAG": "GEOM_FLAG",
        "IFLUX": "LFLUX",
        "ETG_FACTOR": "ETG_FAC",
        "DEBYE_FACTOR": "DEBYE_FAC",
        "XNU_FACTOR": "XNU_FAC",
        "LINSKER_FACTOR": "LINSKER_FAC",
        "GRADB_FACTOR": "GRADB_FAC",
    }

    for ikey in change_from_to:
        if change_from_to[ikey] is not None:
            TGLFoptions[change_from_to[ikey]] = TGLFoptions[ikey]
        del TGLFoptions[ikey]

    # **** New ones that are TRANSP-specific

    TGLFoptions["TGLFMOD"] = 1
    TGLFoptions["NSPEC"] = (
        NS  # Number of species used in tglf model (maximum 10 species allowed)
    )
    TGLFoptions["NLGRAD"] = False  # Output flux gradients
    TGLFoptions["ALPHA_N"] = 0.0  # Scaling factor for vn shear
    TGLFoptions["ALPHA_T"] = 0.0  # Scaling factor for vt shear
    # TGLFoptions['ALPHA_DIA'] 		= 0.0 	# Scaling factor for diamagnetic terms to exb shear
    TGLFoptions["CBETAE"] = 1.0  # Betae multiplier (needed for e-m calcs)
    TGLFoptions["CXNU"] = 1.0  # Collisionality multiplier
    TGLFoptions["EM_STAB"] = (
        0.0  # EM factor for the ion temperature gradient 	--- Is this the right default?
    )
    TGLFoptions["PEVOLVING"] = (
        0  # Evolving temperature and its gradients 	   	--- Is this the right default?
    )
    TGLFoptions["kinetic_fast_ion"] = (
        0  # Fast ion species model in TGLF			   	--- Is this the right default?
    )

    # **** Other modifications
    TGLFoptions["UNITS"] = f"'{TGLFoptions['UNITS']}'"

    return TGLFoptions, label


def addCGYROcontrol(Settings, rmin):

    CGYROoptions = IOtools.generateMITIMNamelist(
        "$MITIM_PATH/templates/input.cgyro.controls", caseInsensitive=False
    )

    """
	********************************************************************************
	Standard sets of TGLF control parameters
	  (rest of parameters are as defaults)
	********************************************************************************
	"""

    with open(
        IOtools.expandPath("$MITIM_PATH/templates/input.cgyro.models.json"), "r"
    ) as f:
        settings = json.load(f)

    if str(Settings) in settings:
        sett = settings[str(Settings)]
        label = sett["label"]
        for ikey in sett["controls"]:
            CGYROoptions[ikey] = sett["controls"][ikey]
    else:
        print(
            "\t- Settings not found in input.cgyro.models.json, using defaults",
            typeMsg="w",
        )
        label = "unspecified"

    CGYROoptions["RMIN"] = rmin

    # --------------------------------
    # From dictionary to text
    # --------------------------------

    CGYROinput = [""]
    for ikey in CGYROoptions:
        CGYROinput.append(f"{ikey} = {CGYROoptions[ikey]}")
    CGYROinput.append("")

    return CGYROinput, CGYROoptions, label


def addTGYROcontrol(
    num_it=0,
    howmany=8,
    fromRho=0.2,
    ToRho=0.8,
    Tepred=1,
    Tipred=1,
    nepred=1,
    Erpred=0,
    physics_options={},
    solver_options={},
    restart=False,
    special_radii=None,
):
    """
    Special radii must have "howmany" elements, indicating the non-uniform grid I want
    """
    if special_radii is None:
        special_radii = -np.ones(howmany)

    """
	------------------------------------------------------------------------------------------------------------
	Defaults
	------------------------------------------------------------------------------------------------------------
	"""

    physics_options.setdefault("TargetType", 3)
    physics_options.setdefault("TurbulentExchange", 1)
    physics_options.setdefault("neoclassical", 2)
    physics_options.setdefault("InputType", 1)
    physics_options.setdefault(
        "quasineutrality", []
    )  # By default, let's not change ions due to quasineutrality
    physics_options.setdefault("GradientsType", 0)
    physics_options.setdefault("PtotType", 0)
    physics_options.setdefault("ParticleFlux", 3)

    solver_options.setdefault("step_jac", 0.1)
    solver_options.setdefault("res_method", 2)
    solver_options.setdefault("tgyro_method", 1)
    solver_options.setdefault("UseRho", 1)

    # Following documentation
    if solver_options["tgyro_method"] == 1:
        solver_options.setdefault("relax_param", 2.0)
        solver_options.setdefault(
            "step_max", 0.1
        )  # 1.0) I think 1.0 is too high... I've seen 0.1 in some places (OMFIT?)

    # Following documentation/recommendation (note)
    elif solver_options["tgyro_method"] == 6:
        solver_options.setdefault("relax_param", 0.1)
        solver_options.setdefault("step_max", 0.05)

    # ----------------------------------------------------------------------------------------------------------

    TGYROoptions = {}

    # ----------- Overarching Options

    TGYROoptions["TGYRO_MODE"] = "1"  # 1: Transport code, 3: multi-job generator
    TGYROoptions["LOC_RESTART_FLAG"] = (
        f"{int(restart)}"  # 0: Start from beginning, 1: Continue from last iteration
    )
    TGYROoptions["TGYRO_RELAX_ITERATIONS"] = f"{num_it}"  # Number of iterations
    TGYROoptions["TGYRO_WRITE_PROFILES_FLAG"] = (
        "1"  # 1: Create new input.profiles at end, 0: Nothing, -1: At all iterations
    )

    # ----------- Optimization

    TGYROoptions["LOC_RESIDUAL_METHOD"] = (
        f"{solver_options['res_method']}"  # 2: |F|, 3: |F|^2
    )
    TGYROoptions["TGYRO_ITERATION_METHOD"] = (
        f"{solver_options['tgyro_method']}"  # 1: Standard local residual, 2 3 4 5 6
    )
    TGYROoptions["LOC_DX"] = (
        f"{solver_options['step_jac']}"  # Step length for Jacobian calculation (df: 0.1), units of a/Lx
    )
    TGYROoptions["LOC_DX_MAX"] = (
        f"{solver_options['step_max']}"  # Max length for any Newton step (df: 1.0)
    )
    TGYROoptions["LOC_RELAX"] = (
        f"{solver_options['relax_param']}"  # Parameter 𝐶𝜂 controlling shrinkage of relaxation parameter
    )

    # ----------- Prediction Options
    TGYROoptions["LOC_SCENARIO"] = (
        f"{physics_options['TargetType']}"  # 1: Static targets, 2: dynamic exchange, 3: alpha, rad, exchange change
    )
    TGYROoptions["LOC_TI_FEEDBACK_FLAG"] = f"{Tipred}"  # Evolve Ti?
    TGYROoptions["LOC_TE_FEEDBACK_FLAG"] = f"{Tepred}"  # Evolve Te?
    TGYROoptions["LOC_ER_FEEDBACK_FLAG"] = f"{Erpred}"  # Evolve Er?
    TGYROoptions["TGYRO_DEN_METHOD0"] = f"{nepred}"  # Evolve ne?
    TGYROoptions["LOC_PFLUX_METHOD"] = (
        f"{physics_options['ParticleFlux']}"  # Particle flux method. 1 = zero target flux, 2 = beam, 3 = beam+wall
    )
    TGYROoptions["TGYRO_RMIN"] = f"{fromRho}"
    TGYROoptions["TGYRO_RMAX"] = f"{ToRho}"
    TGYROoptions["TGYRO_USE_RHO"] = (
        f"{solver_options['UseRho']}"  # 1: Grid provided in input.tgyro is for rho values
    )

    # ----------- Physics
    TGYROoptions["TGYRO_ROTATION_FLAG"] = "1"  # Trigger rotation physics?
    TGYROoptions["TGYRO_NEO_METHOD"] = (
        f"{physics_options['neoclassical']}"  # 0: None, 1: H&H, 2: NEO
    )
    TGYROoptions["TGYRO_TGLF_REVISION"] = (
        "0"  # 0: Use input.tglf in folders, instead of GA defaults.
    )
    TGYROoptions["TGYRO_EXPWD_FLAG"] = (
        f"{physics_options['TurbulentExchange']}"  # Add turbulent exchange to exchange powers in targets?
    )
    # TGYROoptions['TGYRO_ZEFF_FLAG'] 			= '1'													# 1: Use Zeff from input.gacode

    # ----------- Assumptions
    for i in physics_options["quasineutrality"]:
        TGYROoptions[f"TGYRO_DEN_METHOD{i}"] = (
            "-1"  # Species used to ensure quasineutrality
        )
    # TGYROoptions['LOC_NUM_EQUIL_FLAG'] 			= f"{physics_options['usingINPUTgeo']}"		# 0: Use Miller, 1: Use numerical equilibrium (not valid for TGLF_scans)	#DEPRECATED IN LATEST VERSIONS
    TGYROoptions["LOC_LOCK_PROFILE_FLAG"] = (
        f"{physics_options['InputType']}"  # 0: Re-compute profiles from coarse gradients grid, 	 1: Use exact profiles (only valid at first iteration)
    )
    TGYROoptions["TGYRO_CONSISTENT_FLAG"] = (
        f"{physics_options['GradientsType']}"  # 0: Finite-difference gradients used from input.gacode, 1: Gradients from coarse profiles?
    )
    TGYROoptions["LOC_EVOLVE_GRAD_ONLY_FLAG"] = "0"  # 1: Do not change absolute values
    TGYROoptions["TGYRO_PTOT_FLAG"] = (
        f"{physics_options['PtotType']}"  # 0: Compute pressure from profiles, 1: correct from input.gacode PTOT profile
    )

    # ----------- Radii

    for i in range(howmany):
        if special_radii[i] > 0:
            extra_lab = f" X={special_radii[i]}"
        else:
            extra_lab = ""
        TGYROoptions[f"DIR TGLF{i+1} 1{extra_lab}"] = None

    return TGYROoptions, physics_options, solver_options


def addTGYROspecies(Species, onlyThermal=False, limitSpecies=100):
    maxAllowedIons = 7  # 5

    txt = ""
    if onlyThermal:
        txt += " (ignoring fast species)"
    if limitSpecies < maxAllowedIons:
        txt += f", limiting to {limitSpecies}"

    print(f"\t- Adding TGYRO species{txt}:")

    avoid = len(Species) - limitSpecies

    if avoid > 0:
        print(f"\t*CAREFUL, NOT considering the last {avoid} ion(s)")
    SpeciesToCompute = np.min([limitSpecies, len(Species)])

    if SpeciesToCompute > maxAllowedIons:
        print(
            "\t*CAREFUL, number of IONS in species > {0}, but requested only {0}".format(
                maxAllowedIons
            ),
            typeMsg="w",
        )
        lenSpec = maxAllowedIons
    else:
        lenSpec = SpeciesToCompute

    TGYROoptions = {}
    cont = 0
    for i in range(lenSpec):
        if (Species[i]["S"] == "fast" and onlyThermal) or Species[i]["dens"] == 0.0:
            continue

        print(f'\t\t- Specie Z={Species[i]["Z"]} added')

        TGYROoptions[f"TGYRO_CALC_FLAG{cont+1}"] = (
            "1"  # Use this ion in the flux calculation?
        )

        if Species[i]["S"] == "fast":
            TGYROoptions[f"TGYRO_THERM_FLAG{cont+1}"] = 0
        else:
            TGYROoptions[f"TGYRO_THERM_FLAG{cont+1}"] = 1

        # TGYROoptions['TGYRO_DEN_METHOD{0}'.format(cont+1)]=f"{npred[i])
        TGYROoptions[f"TGYRO_SE_SCALE{cont+1}"] = "0"

        cont += 1

    LOC_N_ION = cont

    TGYROoptions["LOC_N_ION"] = f"{LOC_N_ION}"

    TGYROinput = [""]
    for ikey in TGYROoptions:
        if TGYROoptions[ikey] is not None:
            TGYROinput.append(f"{ikey} = {TGYROoptions[ikey]}")
        else:
            TGYROinput.append(f"{ikey}")
    TGYROinput.append("")

    return TGYROinput, TGYROoptions


def convolution_CECE(d_perp_dict, dRdx=1.0):
    """
    d_perp_dict must be a dictionary with the radii and the value of d_perp for each of them, in cm. This means
    that this value must have been already converted from d_theta to d_perp prior to enter in this tool.
    Example:
            factor_theta_to_perp = np.cos( 11* (np.pi/180) )
            d_perp_dict = { 0.65: 0.757 / np.sqrt(2) / factor_theta_to_perp ,
                                            0.75: 0.667 / np.sqrt(2) / factor_theta_to_perp }

    Because the convolution is integrated using d_perp, I need to convert ky to k_perp first.
    G. Staebler (02/23/2021):
            "Note that at the outboard midplane k_perp*rhos = k_y*(1+DRMAJDX_LOC)."
    then dRdx = DRMAJDX_LOC

    Convolution function from
            exp( - (k_perp * d_perp)**2  / 2 )
    """

    fun = lambda ky, rho_s=1.0, d_perp=d_perp_dict, dRdx=dRdx, rho_eval=0.6: np.exp(
        -((ky * (1 + dRdx[rho_eval]) / rho_s * d_perp[rho_eval]) ** 2) / 2
    )

    """
	TGLF as of 03/26/2021 gives the total fluctuations, and for CECE we need to convert into perpendicular.
	Until a better method comes in hand, we just add an ad-hoc correction factor.
		P. Molina (03/26/2021) indicated that GENE suggests that perpendicular is ~30% higher than total
	"""
    factorTot_to_Perp = 1.3

    return fun, factorTot_to_Perp
