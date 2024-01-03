import numpy as np
from IPython import embed

from mitim_tools.misc_tools.IOtools import printMsg as print


def constructStandardTGLF(NS=2):
    """
    Consistent with localdump of TGLF
    """

    TGLFoptions = {
        "USE_TRANSPORT_MODEL": True,
        "GEOMETRY_FLAG": 1,  # 0: s-a, 1: Miller, 2: Fourier, 3: ELITE
        "ADIABATIC_ELEC": False,  # T: Use adiabatic electrons.
        "SIGN_IT": 1,  # Sign of 𝐼𝑇 with repsect to CCW toroidal direction from top.
        "SIGN_BT": 1,  # Sign of 𝐵𝑇 with repsect to CCW toroidal direction from top.
        "IFLUX": True,  # Compute quasilinear weights and mode amplitudes.
        "THETA_TRAPPED": 0.7,  # Parameter to adjust trapped fraction model.
        "NN_MAX_ERROR": -1.0,  # Threshold for TGLF-NN execution versus full TGLF calculation
    }

    """
	Wavenumber grid
	-----------------------
	"""
    TGLFoptions["KYGRID_MODEL"] = 1  # 1: Standard grid
    TGLFoptions["NKY"] = 19  # Number of poloidal modes in the high-k spectrum
    TGLFoptions["KY"] = 0.3  # ky for single-mode call to TGLF.

    """
	Hermite basis functions
	-----------------------
	"""
    TGLFoptions[
        "FIND_WIDTH"
    ] = True  # T: Find the width that maximizes the growth rate, F: Use WIDTH
    TGLFoptions[
        "USE_BISECTION"
    ] = True  # T: Use bisection search method to find width that maximizes growth rate
    TGLFoptions[
        "WIDTH"
    ] = 1.65  # Max. width to search for Hermite polynomial basis (1.65 default, <1.5 recommended in Staebler PoP05)
    TGLFoptions[
        "WIDTH_MIN"
    ] = 0.3  # Min. width to search for Hermite polynomial basis (0.3 default)
    TGLFoptions["NWIDTH"] = 21  # Number of search points
    TGLFoptions["NBASIS_MIN"] = 2  # Minimum number of parallel basis functions
    TGLFoptions["NBASIS_MAX"] = 4  # Maximum number of parallel basis functions
    TGLFoptions["NXGRID"] = 16  # Number of nodes in Gauss-Hermite quadrature.

    """
	Modes options
	-----------------------
	"""
    TGLFoptions[
        "IBRANCH"
    ] = (
        -1
    )  # 0  = find two most unstable modes one for each sign of frequency, electron drift direction (1), ion drift direction (2),
    # -1 = sort the unstable modes by growthrate in rank order
    TGLFoptions["NMODES"] = int(
        NS + 2
    )  # For IBRANCH=-1, number of modes to store. ASTRA uses NS+2

    """
	Saturation model
	-----------------------
	"""
    TGLFoptions["SAT_RULE"] = 0
    TGLFoptions["ETG_FACTOR"] = 1.25

    """
	Physics included
	-----------------------
	"""
    TGLFoptions[
        "USE_BPER"
    ] = False  # Include transverse magnetic fluctuations, 𝛿𝐴‖. 					-> PERPENDICULAR FLUCTUATIONS
    TGLFoptions[
        "USE_BPAR"
    ] = False  # Include compressional magnetic fluctuations, \delta B_{\lVert }}  -> COMPRESSIONAL EFFECTS
    TGLFoptions[
        "USE_MHD_RULE"
    ] = False  # Ignore pressure gradient contribution to curvature drift.
    TGLFoptions["XNU_MODEL"] = 2  # Collision model (2=new)
    TGLFoptions["VPAR_MODEL"] = 0  # 0=low-Mach-number limit (DEPRECATED?)
    TGLFoptions["VPAR_SHEAR_MODEL"] = 1

    """
	ExB shear model
	-----------------------
	"""
    TGLFoptions[
        "ALPHA_QUENCH"
    ] = 0.0  # 1.0 = use quench rule, 0.0 = use new spectral shift model (0.0 recommeded by Gary, 05/11/2020)
    TGLFoptions[
        "ALPHA_E"
    ] = 1.0  # Multiplies ExB velocity shear for spectral shift model (1.0 ecommeded by Gary, 05/11/2020)

    TGLFoptions["ALPHA_MACH"] = 0.0
    TGLFoptions["ALPHA_ZF"] = 1.0

    """
	Multipliers
	-----------------------
	"""

    TGLFoptions["DEBYE_FACTOR"] = 1.0  # Multiplies the debye length
    TGLFoptions[
        "XNU_FACTOR"
    ] = 1.0  # Multiplies the trapped/passing boundary electron-ion collision terms
    TGLFoptions["ALPHA_P"] = 1.0  # Multiplies parallel velocity shear for all species
    TGLFoptions["PARK"] = 1.0  # Multiplies the parallel gradient term.
    TGLFoptions["GHAT"] = 1.0  # Multiplies the curvature drift closure terms.
    TGLFoptions["GCHAT"] = 1.0  # Multiplies the curvature drift irreducible terms.

    # WHY?
    TGLFoptions["DAMP_PSI"] = 0.0  # Damping factor for psi
    TGLFoptions["DAMP_SIG"] = 0.0  # Damping factor for sig
    TGLFoptions["LINSKER_FACTOR"] = 0.0  # Multiplies the Linsker terms
    TGLFoptions["GRADB_FACTOR"] = 0.0  # Multiplies the gradB terms

    TGLFoptions[
        "WD_ZERO"
    ] = 0.1  # Cutoff for curvature drift eigenvalues to prevent zero.
    TGLFoptions[
        "FILTER"
    ] = 2.0  # Sets threshold for frequency/drift frequency to filter out non-driftwave instabilities.
    TGLFoptions["WDIA_TRAPPED"] = 0.0

    """
	Other Options
	-----------------------
	"""

    TGLFoptions[
        "USE_AVE_ION_GRID"
    ] = False  # T:Use weighted average charge of ions for the gyroradius reference, F: Use first ion
    TGLFoptions[
        "USE_INBOARD_DETRAPPED"
    ] = False  # Set trapped fraction to zero if eigenmode is inward ballooning.

    TGLFoptions["RLNP_CUTOFF"] = 18.0  # Limits SAT2 factor of R/Lp < RLNP_CUTOFF

    return TGLFoptions


def addTGLFcontrol(TGLFsettings=1, NS=2, minimal=False):
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
        TGLFoptions = constructStandardTGLF(NS=NS)

    """
	********************************************************************************
	Standard sets of TGLF control parameters
	  (rest of parameters are as defaults)
	********************************************************************************
	"""

    label = "unspecified"

    # SAT1 ----- Old SAT1 standard to recover previous results
    if TGLFsettings == 1:
        TGLFoptions["SAT_RULE"] = 1
        TGLFoptions["UNITS"] = "GYRO"

        label = "SAT1"

    # SAT0 ----- Old SAT0 standard to recover previous results
    if TGLFsettings == 2:
        TGLFoptions["SAT_RULE"] = 0
        TGLFoptions[
            "UNITS"
        ] = "GYRO"  # In SAT0, this is the only option, see tglf.startup.f90
        TGLFoptions["ETG_FACTOR"] = 1.25

        label = "SAT0"

    # SAT1geo ----- SAT1 standard (CGYRO)
    if TGLFsettings == 3:
        TGLFoptions["SAT_RULE"] = 1
        TGLFoptions["UNITS"] = "CGYRO"

        label = "SAT1geo"

    # SAT2 ----- SAT2 standard
    if TGLFsettings == 4:
        label = "SAT2"

        # SAT2
        TGLFoptions["SAT_RULE"] = 2
        TGLFoptions[
            "UNITS"
        ] = "CGYRO"  # In SAT2, CGYRO/GENE are the only options, see tglf.startup.f90
        TGLFoptions["XNU_MODEL"] = 3  # This is forced anyway, see tglf.startup.f90
        TGLFoptions["WDIA_TRAPPED"] = 1.0  # This is forced anyway, see tglf.startup.f90

    # SAT2em
    if TGLFsettings == 5:
        label = "SAT2em"

        # SAT2
        TGLFoptions["SAT_RULE"] = 2
        TGLFoptions[
            "UNITS"
        ] = "CGYRO"  # In SAT2, CGYRO/GENE are the only options, see tglf.startup.f90
        TGLFoptions["XNU_MODEL"] = 3  # This is forced anyway, see tglf.startup.f90
        TGLFoptions["WDIA_TRAPPED"] = 1.0  # This is forced anyway, see tglf.startup.f90

        # EM
        TGLFoptions["USE_BPER"] = True

    # ------------------------
    # PRF's Experiments
    # ------------------------

    # SAT2em with higher resolution
    if TGLFsettings == 101:
        # SAT2
        TGLFoptions["SAT_RULE"] = 2
        TGLFoptions[
            "UNITS"
        ] = "CGYRO"  # In SAT2, CGYRO/GENE are the only options, see tglf.startup.f90
        TGLFoptions["XNU_MODEL"] = 3  # This is forced anyway, see tglf.startup.f90
        TGLFoptions["WDIA_TRAPPED"] = 1.0  # This is forced anyway, see tglf.startup.f90

        # EM
        TGLFoptions["USE_BPER"] = True

        # Extra
        TGLFoptions["KYGRID_MODEL"] = 4
        TGLFoptions["NBASIS_MAX"] = 6

        label = "SAT2em basis"

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
    _, TGLFoptions, label = addTGLFcontrol(TGLFsettings=TGLFsettings, NS=NS)

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
    TGLFoptions[
        "NSPEC"
    ] = NS  # Number of species used in tglf model (maximum 10 species allowed)
    TGLFoptions["NLGRAD"] = False  # Output flux gradients
    TGLFoptions["ALPHA_N"] = 0.0  # Scaling factor for vn shear
    TGLFoptions["ALPHA_T"] = 0.0  # Scaling factor for vt shear
    # TGLFoptions['ALPHA_DIA'] 		= 0.0 	# Scaling factor for diamagnetic terms to exb shear
    TGLFoptions["CBETAE"] = 1.0  # Betae multiplier (needed for e-m calcs)
    TGLFoptions["CXNU"] = 1.0  # Collisionality multiplier
    TGLFoptions[
        "EM_STAB"
    ] = 0.0  # EM factor for the ion temperature gradient 	--- Is this the right default?
    TGLFoptions[
        "PEVOLVING"
    ] = 0  # Evolving temperature and its gradients 	   	--- Is this the right default?
    TGLFoptions[
        "kinetic_fast_ion"
    ] = 0  # Fast ion species model in TGLF			   	--- Is this the right default?

    # **** Other modifications
    TGLFoptions["UNITS"] = f"'{TGLFoptions['UNITS']}'"

    return TGLFoptions, label


def addCGYROcontrol(rmin, ky, CGYROsettings=1):
    # Defaults OMFIT
    CGYROoptions = {
        "NONLINEAR_FLAG": 0,
        "EQUILIBRIUM_MODEL": 2,
        "RMIN": rmin,
        "IPCCW": 1.0,
        "BTCCW": -1.0,
        "UDSYMMETRY_FLAG": 1,
        "PROFILE_MODEL": 2,
        "AMP": 1e-10,
        "AMP0": 0.0,
        "N_FIELD": 2,
        "BETAE_UNIT_SCALE": 1.0,
        "N_RADIAL": 6,
        "N_THETA": 24,
        "N_XI": 16,
        "N_ENERGY": 8,
        "E_MAX": 8.0,
        "N_TOROIDAL": 1,
        "KY": ky,
        "BOX_SIZE": 1,
        "UP_THETA": 1.0,
        "UP_RADIAL": 1.0,
        "DELTA_T": 0.01,
        "MAX_TIME": 29,
        "FREQ_TOL": 1e-3,
        "PRINT_STEP": 100,
        "RESTART_STEP": 10,
        "COLLISION_MODEL": 4,
        "NU_EE_SCALE": 1.0,
        "Z_EFF_METHOD": 1,
        "GAMMA_E_SCALE": 0.0,
        "GAMMA_P_SCALE": 1.0,
        "MACH_SCALE": 1.0,
        "DLNTDR_1_SCALE": 1.0,
        "DLNTDR_2_SCALE": 1.0,
        "DLNTDR_3_SCALE": 1.0,
        "DLNNDR_1_SCALE": 1.0,
        "DLNNDR_2_SCALE": 1.0,
        "DLNNDR_3_SCALE": 1.0,
        "THETA_PLOT": 1,
    }

    # Species
    CGYROoptions["N_SPECIES"] = 3
    CGYROoptions["Z_1"] = 1.0
    CGYROoptions["MASS_1"] = 1.0
    CGYROoptions["Z_2"] = 6.0
    CGYROoptions["MASS_2"] = 6.0
    CGYROoptions["Z_3"] = -1.0
    CGYROoptions["MASS_3"] = 0.0002724486

    # ----------------------------------------------------------------
    # Options
    # ----------------------------------------------------------------

    if CGYROsettings == 1:
        CGYROoptions[
            "DELTA_T"
        ] = 2e-3  # Higher ks smaller than this maybe. 2E-3 for low-k, 1E-3 med-k 5E-4 high-k
        CGYROoptions[
            "MAX_TIME"
        ] = 30000  # NTH: If it doesn't converg earlier, do not limit by time

        CGYROoptions["N_RADIAL"] = 12  # NTH
        CGYROoptions[
            "N_THETA"
        ] = 24  # NTH: Change to 48 to see if answer changes for the most unstable (>10%?)

        CGYROoptions["RESTART_STEP"] = 500  # NTH

    CGYROinput = ["# Written by MITIM (P. Rodriguez-Fernandez, 2021)"]
    for ikey in CGYROoptions:
        CGYROinput.append(f"{ikey} = {CGYROoptions[ikey]}")

    return CGYROinput, CGYROoptions


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
    TGYROoptions[
        "LOC_RESTART_FLAG"
    ] = f"{int(restart)}"  # 0: Start from beginning, 1: Continue from last iteration
    TGYROoptions["TGYRO_RELAX_ITERATIONS"] = f"{num_it}"  # Number of iterations
    TGYROoptions[
        "TGYRO_WRITE_PROFILES_FLAG"
    ] = "1"  # 1: Create new input.profiles at end, 0: Nothing, -1: At all iterations

    # ----------- Optimization

    TGYROoptions[
        "LOC_RESIDUAL_METHOD"
    ] = f"{solver_options['res_method']}"  # 2: |F|, 3: |F|^2
    TGYROoptions[
        "TGYRO_ITERATION_METHOD"
    ] = f"{solver_options['tgyro_method']}"  # 1: Standard local residual, 2 3 4 5 6
    TGYROoptions[
        "LOC_DX"
    ] = f"{solver_options['step_jac']}"  # Step length for Jacobian calculation (df: 0.1), units of a/Lx
    TGYROoptions[
        "LOC_DX_MAX"
    ] = f"{solver_options['step_max']}"  # Max length for any Newton step (df: 1.0)
    TGYROoptions[
        "LOC_RELAX"
    ] = f"{solver_options['relax_param']}"  # Parameter 𝐶𝜂 controlling shrinkage of relaxation parameter

    # ----------- Prediction Options
    TGYROoptions[
        "LOC_SCENARIO"
    ] = f"{physics_options['TargetType']}"  # 1: Static targets, 2: dynamic exchange, 3: alpha, rad, exchange change
    TGYROoptions["LOC_TI_FEEDBACK_FLAG"] = f"{Tipred}"  # Evolve Ti?
    TGYROoptions["LOC_TE_FEEDBACK_FLAG"] = f"{Tepred}"  # Evolve Te?
    TGYROoptions["LOC_ER_FEEDBACK_FLAG"] = f"{Erpred}"  # Evolve Er?
    TGYROoptions["TGYRO_DEN_METHOD0"] = f"{nepred}"  # Evolve ne?
    TGYROoptions[
        "LOC_PFLUX_METHOD"
    ] = f"{physics_options['ParticleFlux']}"  # Particle flux method. 1 = zero target flux, 2 = beam, 3 = beam+wall
    TGYROoptions["TGYRO_RMIN"] = f"{fromRho}"
    TGYROoptions["TGYRO_RMAX"] = f"{ToRho}"
    TGYROoptions[
        "TGYRO_USE_RHO"
    ] = f"{solver_options['UseRho']}"  # 1: Grid provided in input.tgyro is for rho values

    # ----------- Physics
    TGYROoptions["TGYRO_ROTATION_FLAG"] = "1"  # Trigger rotation physics?
    TGYROoptions[
        "TGYRO_NEO_METHOD"
    ] = f"{physics_options['neoclassical']}"  # 0: None, 1: H&H, 2: NEO
    TGYROoptions[
        "TGYRO_TGLF_REVISION"
    ] = "0"  # 0: Use input.tglf in folders, instead of GA defaults.
    TGYROoptions[
        "TGYRO_EXPWD_FLAG"
    ] = f"{physics_options['TurbulentExchange']}"  # Add turbulent exchange to exchange powers in targets?
    # TGYROoptions['TGYRO_ZEFF_FLAG'] 			= '1'													# 1: Use Zeff from input.gacode

    # ----------- Assumptions
    for i in physics_options["quasineutrality"]:
        TGYROoptions[
            f"TGYRO_DEN_METHOD{i}"
        ] = "-1"  # Species used to ensure quasineutrality
    # TGYROoptions['LOC_NUM_EQUIL_FLAG'] 			= f"{physics_options['usingINPUTgeo']}"		# 0: Use Miller, 1: Use numerical equilibrium (not valid for TGLF_scans)	#DEPRECATED IN LATEST VERSIONS
    TGYROoptions[
        "LOC_LOCK_PROFILE_FLAG"
    ] = f"{physics_options['InputType']}"  # 0: Re-compute profiles from coarse gradients grid, 	 1: Use exact profiles (only valid at first iteration)
    TGYROoptions[
        "TGYRO_CONSISTENT_FLAG"
    ] = f"{physics_options['GradientsType']}"  # 0: Finite-difference gradients used from input.gacode, 1: Gradients from coarse profiles?
    TGYROoptions["LOC_EVOLVE_GRAD_ONLY_FLAG"] = "0"  # 1: Do not change absolute values
    TGYROoptions[
        "TGYRO_PTOT_FLAG"
    ] = f"{physics_options['PtotType']}"  # 0: Compute pressure from profiles, 1: correct from input.gacode PTOT profile

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

        TGYROoptions[
            f"TGYRO_CALC_FLAG{cont+1}"
        ] = "1"  # Use this ion in the flux calculation?

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
