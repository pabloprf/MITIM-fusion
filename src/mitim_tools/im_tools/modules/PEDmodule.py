import numpy as np
import math, pdb, netCDF4
import matplotlib.pyplot as plt

from mitim_tools.misc_tools import IOtools, MATHtools, PLASMAtools, FARMINGtools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.im_tools import IMparam
from mitim_tools.im_tools.aux import LUTtools

from mitim_tools.misc_tools import CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print

from IPython import embed


def calculatePedestal(
    UseShapingRatio,
    PedestalType,
    CoordinateMapping,
    Rmajor,
    epsilon,
    kappa,
    Bt,
    delta,
    BetaN,
    Zeff,
    neped,
    Ip=None,
    Ratios=[None, None],
    LuT_loc=None,
    LuT_variables=None,
    LuT_fixed=None,
    LuT_vals=None,
    nn_loc=None,
    LH_valsL=None,
    nameFolder="",
    lock=None,
):
    if UseShapingRatio:
        print(
            '\t- Applying "separatrix to 99.5 flux" shaping ratios: {0:.3f} for kappa and {1:.3f} for delta'.format(
                Ratios[0], Ratios[1]
            )
        )
        kR, dR = Ratios[0], Ratios[1]
    else:
        kR, dR = 1.0, 1.0

    if PedestalType not in ["vals", "lmode"]:
        print(
            "\t- Evaluating pedestal with: Rmajor={0:.2f},epsilon={1:.3f},Ip={2:.2f},kappa={3:.3f},delta={4:.3f},BetaN={5:.2f},neped={6:.2f},Bt={7:.2f}".format(
                Rmajor, epsilon, Ip, kappa * kR, delta * dR, BetaN, neped, Bt
            )
        )
    else:
        print("\t- Pedestal (including ne_top) specified by namelist", typeMsg="i")

    # -------------------------------------------
    # Pedestal Model (must provide m^-3, eV, psi)
    # -------------------------------------------

    if PedestalType in ["vals"]:
        # Read values direclty from namelist
        Te_height = LuT_vals[0]

        width = LuT_vals[1]
        ne_height = LuT_vals[2]
        Ip = LuT_vals[3]

        Ti_height = Te_height
        try:
            if LuT_vals[4] is not None:
                Ti_height = LuT_vals[4]
        except:
            pass

        p_height = p1_height = None

    elif PedestalType in ["lut", "surrogate_model"]:
        ne_height, Te_height, width, _, p_height, p1_height = LUTtools.search_LuT_EPED(
            PedestalType,
            LuT_loc=LuT_loc,
            LuT_variables=LuT_variables,
            LuT_fixed=LuT_fixed,
            Bt=Bt,
            Rmajor=Rmajor,
            Ip=Ip,
            kappa=kappa * kR,
            delta=delta * dR,
            BetaN=BetaN,
            neped=neped,
            Zeff=Zeff,
            epsilon=epsilon,
            nameFolder=nameFolder,
            lock=lock,
        )

        Ti_height = Te_height

    elif PedestalType in ["nn"]:
        ne_height, Te_height, width = LUTtools.evaluateNN(
            nn_loc, Bt, Rmajor, epsilon, Ip, kappa * kR, delta * dR, BetaN, neped
        )

        Ti_height = Te_height

    elif PedestalType in ["lmode"]:
        ne_LCFS, Te_LCFS, _ = PLASMAtools.evaluateLCFS_Lmode(LH_valsL[0])

        ne_height, Te_height, width = ne_LCFS * 1.5, Te_LCFS * 1.5, 0.05

        print(
            ">> L-mode boundary given by T_{{keV}}={0:.2f}, n_{{20}}={1:.2f} at rho={2:.2f}".format(
                Te_height * 1e-3, ne_height * 1e-20, width
            )
        )

        Ti_height = Te_height

    else:
        raise Exception(
            "Pedestal option not available, please choose: None, nn, vals, lut, lmode, surrogate_model"
        )

    # -------------------------------------------
    # Conversions
    # -------------------------------------------

    # Convert to RHO grid
    if width < 0:
        Te_width = np.abs(width)
        print('>> Pedestal width taken as rho, by "vals" request')
    else:
        Te_width = convertwidthToRho(width, CoordinateMapping)
    Ti_width = Te_width
    ne_width = Te_width

    # Min pedestal temperature is 100 eV (not really a pedestal)
    Te_height = np.max([Te_height, 100.0])
    # Ti_height = Te_height
    ne_height = ne_height * 1e-6  # cm**-3

    # Rotation
    omegaEdge, whereOmega = PLASMAtools.getOmegaBC()

    return (
        Te_height,
        Ti_height,
        ne_height,
        Te_width,
        Ti_width,
        ne_width,
        omegaEdge,
        whereOmega,
        p_height,
        p1_height,
    )


def runPedestal(
    mitimNML, MITIMparams, TransitionTime, FinTransition, EnforceDensityBC=None
):
    print(">> Entering pedestal module...")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	Plasma Parameters to evaluate pedestal with
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    CoordinateMapping = mitimNML.StepParams["CoordinateMapping"]

    BetaN = mitimNML.StepParams["BetaN"]
    Ip, Bt, Zeff, neped = (
        MITIMparams["Ip"],
        MITIMparams["Bt"],
        MITIMparams["Zeff"],
        MITIMparams["neped"],
    )

    if mitimNML.Pedestal_Redk is not None:
        print(
            '>> Evaluating pedestal with values established specifically in namelist "Redk" variable',
            typeMsg="i",
        )
        Rmajor, epsilon, delta, kappa = mitimNML.Pedestal_Redk
    else:
        Rmajor, epsilon, delta, kappa = (
            MITIMparams["rmajor"],
            MITIMparams["epsilon"],
            MITIMparams["delta"],
            MITIMparams["kappa"],
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	Calculate pedestal features
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    (
        Te_height,
        Ti_height,
        ne_height,
        Te_width,
        Ti_width,
        ne_width,
        omegaEdge,
        whereOmega,
        p_height,
        p1_height,
    ) = calculatePedestal(
        mitimNML.UseShapingRatio,
        mitimNML.PedestalType,
        CoordinateMapping,
        Rmajor,
        epsilon,
        kappa,
        Bt,
        delta,
        BetaN,
        Zeff,
        neped,
        Ip=Ip,
        Ratios=[mitimNML.StepParams["kappaRatio"], mitimNML.StepParams["deltaRatio"]],
        LuT_loc=mitimNML.LuT_loc,
        LuT_variables=mitimNML.LuT_variables,
        LuT_fixed=mitimNML.LuT_fixed,
        LuT_vals=mitimNML.LuT_vals,
        nn_loc=mitimNML.nn_loc,
        LH_valsL=mitimNML.LH_valsL,
        nameFolder=str(mitimNML.nameRunTot),
        lock=mitimNML.lock,
    )

    Te_height, Ti_height = (
        MITIMparams["factor_ped_degr"] * Te_height,
        MITIMparams["factor_ped_degr"] * Ti_height,
    )

    try:
        print(
            ">> Pedestal top: {0:.2}MPa, {1:.2f}keV, {2:.1f}E19 m^-3, rho = {3:.3f}".format(
                p_height, Te_height * 1e-3, ne_height * 1e-14 * 10, 1 - Te_width
            ),
            typeMsg="i",
        )
    except:
        pass

    mitimNML.MITIMparams["Te_height"] = Te_height * 1e-3  # in keV
    mitimNML.MITIMparams["Te_width"] = Te_width
    mitimNML.MITIMparams["ne_height"] = ne_height * 1e-14  # in 1E20m^-3
    mitimNML.MITIMparams["ne_width"] = ne_width

    if mitimNML.LH_time is not None:
        (
            Te_Lmode,
            Ti_Lmode,
            ne_Lmode,
            width,
            _,
            _,
            _,
            _,
            _,
            p_height,
            p1_height,
        ) = calculatePedestal(
            mitimNML.UseShapingRatio,
            "lmode",
            CoordinateMapping,
            Rmajor,
            epsilon,
            kappa,
            Bt,
            delta,
            BetaN,
            Zeff,
            neped,
            Ip=Ip,
            Ratios=[
                mitimNML.StepParams["kappaRatio"],
                mitimNML.StepParams["deltaRatio"],
            ],
            LuT_loc=mitimNML.LuT_loc,
            LuT_variables=mitimNML.LuT_variables,
            LuT_fixed=mitimNML.LuT_fixed,
            LuT_vals=mitimNML.LuT_vals,
            nn_loc=mitimNML.nn_loc,
            LH_valsL=mitimNML.LH_valsL,
            nameFolder=str(mitimNML.nameRunTot),
            lock=mitimNML.lock,
        )
    else:
        Te_Lmode, Ti_Lmode, ne_Lmode, width_Lmode = None, None, None, None

    width_Hmode = Te_width

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	Implement pedestal
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Temperature and Density B.C.

    namelistPath = mitimNML.namelistPath
    insertBCs(namelistPath, [Te_width, ne_width], EnforceDensityBC=EnforceDensityBC)

    # ~~~ Option to implement pedestal in UFILES
    if mitimNML.PedestalUfiles:
        implementUFILESpedestal(
            namelistPath,
            mitimNML.FolderTRANSP,
            mitimNML.nameBaseShot,
            mitimNML.nameRunTot,
            TransitionTime,
            FinTransition,
            Te_height,
            Ti_height,
            ne_height,
            width_Hmode,
            p_height,
            p1_height,
            PedestalShape=mitimNML.PedestalShape,
            LH_time=mitimNML.LH_time,
            Te_Lmode=Te_Lmode,
            Ti_Lmode=Ti_Lmode,
            ne_Lmode=ne_Lmode,
            width_Lmode=width_Lmode,
            StepParams=mitimNML.StepParams,
        )

    # ~~~ Option to implement pedestal in namelist instead of UFILES
    else:
        print(">> Implementing pedestal in namelist")
        IOtools.changeValue(
            namelistPath, "MODEEDG", 5, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            namelistPath, "MODIEDG", 5, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            namelistPath, "MODNEDG", 5, None, "=", MaintainComments=True
        )

    # In any case, add values to namelist, so that later I can get them from TRANSP output
    IOtools.changeValue(
        namelistPath, "TEPED", Te_height, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "TEPEDW", -width_Hmode, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "TIPED", Ti_height, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "TIPEDW", -width_Hmode, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "XNEPED", ne_height, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "XNEPEDW", -width_Hmode, None, "=", MaintainComments=True
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	Plasma rotation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    try:
        IOtools.changeValue(
            namelistPath, "OMEDGE", omegaEdge, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            namelistPath, "MODOMEDG", 2, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            namelistPath, "XPHIBOUND", whereOmega, None, "=", MaintainComments=True
        )
    except:
        print("\t- Rotation not changed")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	Option for TRANSP-based pedestal prediction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if mitimNML.pedestalPred:
        try:
            if mitimNML.runPhaseNumber == 0:  # interpretive
                offset = mitimNML.PredictionOffset
                IOtools.changeValue(
                    namelistPath,
                    "time_l2h",
                    mitimNML.BaselineTime + offset,
                    None,
                    "=",
                    MaintainComments=True,
                )
            elif mitimNML.runPhaseNumber == 1:  # predictive
                offsetHF = mitimNML.PredOffset_2ndPhase
                IOtools.changeValue(
                    namelistPath,
                    "time_l2h",
                    mitimNML.BaselineTime + offsetHF,
                    None,
                    "=",
                    MaintainComments=True,
                )

        except:
            print(">> l2h not changed")

    return width_Hmode


def implementUFILESpedestal(
    namelistPath,
    FolderTRANSP,
    nameBaseShot,
    nameRunTot,
    TransitionTime,
    FinTransition,
    Te_height,
    Ti_height,
    ne_height,
    width_Hmode,
    p_height,
    p1_height,
    PedestalShape=1,
    LH_time=None,
    Te_Lmode=None,
    Ti_Lmode=None,
    ne_Lmode=None,
    width_Lmode=None,
    StepParams={},
):
    print(">> Implementing pedestal inside UFILES, not namelist")

    IOtools.changeValue(namelistPath, "MODEEDG", 3, None, "=", MaintainComments=True)
    IOtools.changeValue(namelistPath, "MODIEDG", 4, None, "=", MaintainComments=True)
    IOtools.changeValue(namelistPath, "MODNEDG", 3, None, "=", MaintainComments=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	L-mode before the H-mode (optional)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if LH_time is not None:
        deltaLH = 0.01  # Time to take from start to end of LH transition
        # TransitionTime += 0.02 # time from LH specified transition (because at the transition, Power goes up, but pedestal later)

        print(
            ">> Introducing first L-mode boundary condition starting at t={0:.3f}s with time window of {1:.3f}s".format(
                TransitionTime, deltaLH
            )
        )

        width = np.max([width_Lmode, width_Hmode])

        addPedestalToProfile(
            f"{FolderTRANSP}/PRF{nameBaseShot}.TEL",
            PedestalTop=Te_Lmode,
            PedestalBottom=Te_Lmode / 2.0,
            PedestalWidth=width,
            TransitionTime=TransitionTime,
            TimeWidthTransition=deltaLH,
            StraightLine=(PedestalShape == 0),
        )

        addPedestalToProfile(
            f"{FolderTRANSP}/PRF{nameBaseShot}.TIO",
            PedestalTop=Ti_Lmode,
            PedestalBottom=Ti_Lmode / 2.0,
            PedestalWidth=width,
            TransitionTime=TransitionTime,
            TimeWidthTransition=deltaLH,
            StraightLine=(PedestalShape == 0),
        )

        addPedestalToProfile(
            f"{FolderTRANSP}/PRF{nameBaseShot}.NEL",
            PedestalTop=ne_Lmode,
            PedestalBottom=ne_Lmode / 1.5,
            PedestalWidth=width,
            TransitionTime=TransitionTime,
            TimeWidthTransition=deltaLH,
            StraightLine=(PedestalShape == 0),
        )

        TransitionTime, Delta_time = LH_time, 0.01

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 	H-mode implementation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else:
        Delta_time = FinTransition - TransitionTime
        # Delta_time is the time for transition to new pedestal (not to be too brusque)

    print(
        ">> H-mode pedestal condition starting at t={0:.3f}s with time window of {1:.3f}s".format(
            TransitionTime, Delta_time
        )
    )

    if Te_height > 2e3:
        Te_LCFS = 300.0
    else:
        Te_LCFS = 100.0

    if PedestalShape in [0, 1]:
        addPedestalToProfile(
            f"{FolderTRANSP}/PRF{nameBaseShot}.TEL",
            PedestalTop=Te_height,
            PedestalBottom=Te_LCFS,
            PedestalWidth=width_Hmode,
            TransitionTime=TransitionTime,
            TimeWidthTransition=Delta_time,
            StraightLine=(PedestalShape == 0),
        )

        addPedestalToProfile(
            f"{FolderTRANSP}/PRF{nameBaseShot}.TIO",
            PedestalTop=Ti_height,
            PedestalBottom=Te_LCFS * 2,
            PedestalWidth=width_Hmode,
            TransitionTime=TransitionTime,
            TimeWidthTransition=Delta_time,
            StraightLine=(PedestalShape == 0),
        )

        addPedestalToProfile(
            f"{FolderTRANSP}/PRF{nameBaseShot}.NEL",
            PedestalTop=ne_height,
            PedestalBottom=ne_height * 0.3,
            PedestalWidth=width_Hmode,
            TransitionTime=TransitionTime,
            TimeWidthTransition=Delta_time,
            StraightLine=(PedestalShape == 0),
        )

    else:
        implementPedestalMTANH(
            FolderTRANSP,
            nameBaseShot,
            nameRunTot,
            ne_height,
            width_Hmode,
            p_height,
            p1_height,
            Te_height,
            TransitionTime=TransitionTime,
            TimeWidthTransition=Delta_time,
            StepParams=StepParams,
        )


def create_dummy_plasmastate(file, rho, rhob, psipol, ne, te, ti):
    """
    To "trick" the IDL routines to think this is a plasmstate file to extract the pdestal features

    """

    # Create file
    ncfile = netCDF4.Dataset(file, mode="w", format="NETCDF4_CLASSIC")

    # --------------------------------
    # Store profiles (in center grid)
    # --------------------------------

    # Dimensions
    ncfile.createDimension("xdim", rho.shape[0])
    ncfile.createDimension("sdim", 2)
    ncfile.createDimension("ndim", 1)

    value = ncfile.createVariable("s_name", "S1", ("sdim",))
    value[:] = np.array(["e", "i"], dtype="S1")

    value = ncfile.createVariable(
        "ns",
        "f4",
        (
            "sdim",
            "xdim",
        ),
    )
    value[0, :] = ne

    value = ncfile.createVariable("ns_bdy", "f4", ("sdim",))
    value[:] = np.array([ne[-1], ne[-1]])

    value = ncfile.createVariable(
        "Ts",
        "f4",
        (
            "sdim",
            "xdim",
        ),
    )
    value[0, :] = te

    value = ncfile.createVariable("Te_bdy", "f4", ("ndim",))
    value[:] = te[-1]

    value = ncfile.createVariable("Ti", "f4", ("xdim",))
    value[:] = ti

    value = ncfile.createVariable("Ti_bdy", "f4", ("ndim",))
    value[:] = ti[-1]

    # --------------------------------
    # Store rho and psi (in boundary grid with an extra zero)
    # --------------------------------

    # add one more point to boundary
    rhob = np.append([0], rhob)
    psipol = np.append([0], psipol)

    ncfile.createDimension("xdim_rhob", rhob.shape[0])

    value = ncfile.createVariable("rho", "f4", ("xdim_rhob",))
    value[:] = rhob

    value = ncfile.createVariable("psipol", "f4", ("xdim_rhob",))
    value[:] = psipol

    ncfile.close()


def implementPedestalMTANH(
    FolderTRANSP,
    nameBaseShot,
    nameRunTot,
    netop,
    width_top,
    ptop,
    p1,
    tetop,
    TransitionTime=2.0,
    TimeWidthTransition=1e-3,
    StepParams={},
):
    print("\t- Implementing pedestal through IDL mtanh routine")

    folderWork = FolderTRANSP + "PED_folder/"

    IOtools.askNewFolder(folderWork, force=True)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Produce plasmastate
    # -------------------------------------------------------------------------------------------------------------------------------------

    plasmastate = folderWork + "state.cdf"
    rhob = StepParams["CoordinateMapping"]["rho"]
    psipol = StepParams["CoordinateMapping"]["psi"]
    rho = StepParams["rho"]
    ne = StepParams["ne"]
    te = StepParams["Te"]
    ti = StepParams["Ti"]
    create_dummy_plasmastate(plasmastate, rho, rhob, psipol, ne * 1e20, te, ti)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Create pedestal shape
    # -------------------------------------------------------------------------------------------------------------------------------------

    # By default width_top will be in rho, but I need to do this in psi
    width_top_psi = 1 - np.interp(
        1 - width_top,
        StepParams["CoordinateMapping"]["rho"],
        StepParams["CoordinateMapping"]["psi"],
    )

    x, ne, Te, Ti = fit_pedestal_mtanh(
        width_top_psi,
        netop * 1e-14,
        p1 * 1e6,
        ptop * 1e6,
        plasmastate,
        folderWork=folderWork,
        nameRunid=nameRunTot,
        tetop_previous=tetop * 1e-3,
    )

    # Only core values
    ne = ne[x <= 1.0] * 1e14  # cm^-3
    Te = Te[x <= 1.0] * 1e3  # eV
    Ti = Ti[x <= 1.0] * 1e3  # eV
    x = x[x <= 1.0]

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Insert
    # -------------------------------------------------------------------------------------------------------------------------------------

    x_rho = np.interp(
        x,
        StepParams["CoordinateMapping"]["psi"],
        StepParams["CoordinateMapping"]["rho"],
    )

    uf_file = f"{FolderTRANSP}/PRF{nameBaseShot}.TEL"
    ped_ready = [x_rho, Te]
    addPedestalToProfile(
        uf_file,
        TransitionTime=TransitionTime,
        TimeWidthTransition=TimeWidthTransition,
        ped_ready=ped_ready,
        PedestalWidth=width_top,
        PedestalTop=tetop,
    )

    uf_file = f"{FolderTRANSP}/PRF{nameBaseShot}.TIO"
    ped_ready = [x_rho, Ti]
    addPedestalToProfile(
        uf_file,
        TransitionTime=TransitionTime,
        TimeWidthTransition=TimeWidthTransition,
        ped_ready=ped_ready,
        PedestalWidth=width_top,
    )

    uf_file = f"{FolderTRANSP}/PRF{nameBaseShot}.NEL"
    ped_ready = [x_rho, ne]
    addPedestalToProfile(
        uf_file,
        TransitionTime=TransitionTime,
        TimeWidthTransition=TimeWidthTransition,
        ped_ready=ped_ready,
        PedestalWidth=width_top,
        PedestalTop=netop,
    )


def fit_pedestal_mtanh(
    width_top,
    netop,
    p1,
    ptop,
    plasmastate,
    folderWork="~/scratchFolder/",
    nameRunid=1,
    tetop_previous=None,
    debug=False,
):
    """

    Inputs:
            netop in 1E20 m^-3
            width_top in psin
            pressures in Pa

            (Note that temperature is defined from these inputs, in the IDL routine)

            tetop_previous is just for testing how different the value of the IDL calculated tetop from the pressure. Discrepancies
            may be due simply to the linear interpolation

    Outputs:
            x is psin
            ne in 1E20 m^-3
            Te, Ti in keV

    """

    pedestal_job = FARMINGtools.mitim_job(folderWork)

    pedestal_job.define_machine(
        "idl",
        f"mitim_idl_{nameRunid}/",
        launchSlurm=False,
    )

    path = pedestal_job.folderExecution
    plasmastate_path = path + IOtools.reducePathLevel(plasmastate, isItFile=True)[-1]

    with open(folderWork + "/idl_in", "w") as f:
        f.write(".r /home/nthoward/SPARC_mtanh/make_pedestal_profiles_portals.pro\n")
        f.write(
            f"make_profiles,[{width_top},{netop},{p1},{ptop}],'{plasmastate_path}','{path}'\n\n"
        )
        f.write("exit")

    start = "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/labombard/idl_lib/fortran && export IDL_STARTUP=/home/nthoward/idl/idl_startup"

    command = f"cd {path} && {start} && idl < idl_in"

    inputFiles = [folderWork + "/idl_in", plasmastate]
    outputFiles = ["mtanh_fits"]

    print(f"\t\t- Proceeding to run idl pedestal fitter (psi_pol = {width_top:.3f})")

    pedestal_job.prep(
        command,
        output_files=outputFiles,
        input_files=inputFiles,
    )

    pedestal_job.run(timeoutSecs=30)

    x, ne, Te, Ti = read_mtanh(folderWork + "/mtanh_fits")

    tetop = ptop / netop / 3.2e1 * 1e-3

    print(
        "\t- Fitted pedestal, resulting in Tetop = {0:.2f}keV, netop = {1:.1f}E19 m^-3, psipol={2:.3f}".format(
            tetop, netop * 10, 1 - width_top
        )
    )

    if tetop_previous is not None:
        percent_change = np.abs(tetop_previous - tetop) / tetop * 100
        if percent_change > 0.5:
            print(
                "\t\t- Tetop differs from the previous reported value by {0:.1f}% (likely because of linearly interpolating ptop instead of netop and Tetop)".format(
                    percent_change
                ),
                typeMsg="w",
            )

    if debug:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        ax = axs[0]
        ax.plot(x, Te, "-o", markersize=1, lw=1.0, label="fit")
        ax.scatter([1 - width_top], [tetop], label="top")
        ax = axs[1]
        ax.plot(x, Ti, "-o", markersize=1, lw=1.0)
        # ax.scatter([1-width_top],[titop])
        ax = axs[2]
        ax.plot(x, ne, "-o", markersize=1, lw=1.0)
        ax.scatter([1 - width_top], [netop])

        from mitim_tools.transp_tools.tools.PLASMASTATEtools import Plasmastate

        p = Plasmastate(plasmastate)

        p.plot(axs=axs, label=".cdf")

        axs[0].legend()

        plt.show()

        pdb.set_trace()

    return x, ne, Te, Ti


def read_mtanh(file_out):
    with open(file_out, "r") as f:
        aux = f.readlines()

    v = []
    for i in aux:
        if len(i) > 0:
            a = [float(j) for j in i.split()]
        for k in a:
            v.append(k)

    nr = int(len(v) / 4)

    x = np.array(v[:nr])
    ne = np.array(v[nr : nr * 2])  # *1E-20
    Te = np.array(v[nr * 2 : nr * 3]) * 1e-3
    Ti = np.array(v[nr * 3 :]) * 1e-3

    return x, ne, Te, Ti


def implementSweepingPedestal(
    mitimNML, MITIMparams, TransitionTime, FinTransition, minioffset=1e-4
):
    print(
        ">> Implementation of baseline pedestal, start transition at {0:.3f} and fully formed pedestal at t={1:.3f}s".format(
            TransitionTime, FinTransition
        )
    )

    w = runPedestal(mitimNML, MITIMparams, TransitionTime, FinTransition)

    MITIMparams = MITIMparams
    TransitionTime, FinTransition = FinTransition + minioffset, mitimNML.EQsweepTimes[1]
    w = runPedestal(mitimNML, MITIMparams, TransitionTime, FinTransition)

    furtherwidth = mitimNML.MITIMparams["Te_width"]
    for i in range((len(mitimNML.EQsweepTimes) - 2) // 2):
        MITIMparams = IMparam.updateParametersWithGfile(
            mitimNML.prescribedEvolution[1], MITIMparams, mitimNML.nameRunTot
        )
        TransitionTime, FinTransition = (
            FinTransition + minioffset,
            mitimNML.EQsweepTimes[(i + 1) * 2],
        )
        w = runPedestal(mitimNML, MITIMparams, TransitionTime, FinTransition)
        if mitimNML.MITIMparams["Te_width"] > furtherwidth:
            furtherwidth = mitimNML.MITIMparams["Te_width"]

        MITIMparams = IMparam.updateParametersWithGfile(
            mitimNML.prescribedEvolution[2], MITIMparams, mitimNML.nameRunTot
        )
        TransitionTime, FinTransition = (
            FinTransition + minioffset,
            mitimNML.EQsweepTimes[(i + 1) * 2 + 1],
        )
        w = runPedestal(mitimNML, MITIMparams, TransitionTime, FinTransition)
        if mitimNML.MITIMparams["Te_width"] > furtherwidth:
            furtherwidth = mitimNML.MITIMparams["Te_width"]

    insertBCs(mitimNML.namelistPath, [furtherwidth, furtherwidth])


def insertBCs(namelistPath, widths_rhoN, EnforceDensityBC=None):
    orig_width = widths_rhoN[0]

    # Start prediction a bit inside because TRANSp will choose the next point
    extrafactor = 0.005
    widths_rhoN = [round(i + extrafactor, 2) for i in widths_rhoN]
    # -------
    print(
        ">> Implementing boundary condition in TRANSP namelist at rho={0} (moved from original {1} so account for TRANSP coarse grid)".format(
            widths_rhoN[0], orig_width
        )
    )

    IOtools.changeValue(
        namelistPath, "XIBOUND", 1 - widths_rhoN[0], None, "=", MaintainComments=True
    )
    if EnforceDensityBC is None:
        IOtools.changeValue(
            namelistPath,
            "XNBOUND",
            1 - widths_rhoN[1],
            None,
            "=",
            MaintainComments=True,
        )
    else:
        IOtools.changeValue(
            namelistPath, "XNBOUND", EnforceDensityBC, None, "=", MaintainComments=True
        )


def convertwidthToRho(width, CoordinateMapping):
    Te_width = (
        1
        - np.interp([1 - width], CoordinateMapping["psi"], CoordinateMapping["rho"])[0]
    )

    print(
        "\t- Pedestal width in PSI (norm pol flux): {0:.3f}, in RHO (sqrt norm tor flux): {1:.3f}".format(
            width, Te_width
        )
    )
    print(
        "\t- Pedestal position in PSI (norm pol flux): {0:.3f}, in RHO (sqrt norm tor flux): {1:.3f}".format(
            1 - width, 1 - Te_width
        )
    )

    # nzones = {'nzones':100,'nzone_FastIonDistr':10,'nzone_MCmodel':20,'nzone_FPmodel':20}

    return Te_width


def addPedestalToProfile(
    UFfile,
    TransitionTime=2.0,
    TimeWidthTransition=1e-3,
    PedestalTop=1.0e3,
    PedestalBottom=1.0e2,
    PedestalWidth=0.05,
    StraightLine=False,
    ped_ready=None,
    debug=False,
):
    """
    This function will add a pedestal to all the profiles in the the UFile after a given time.
    TimeWidthTransition

    x in ped_ready should have the same units as the original UF (rho?)

    """
    if not StraightLine:
        mergeExtraRange = 0  # 0.01 This is used b/c TRANSP grid may pick the next point to start the pedestal (UPDATE: NOT ANYMORE CHANGE BC in NML instead)
        x = np.linspace(0, 1.0, 2001)
    else:
        mergeExtraRange = 0.0
        x = np.linspace(0, 1.0, 2001)

    TransitionTime += TimeWidthTransition

    # ~~~~~~~~~~~~~~~~~~ Grab previous profile

    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(UFfile)
    x0 = np.array(UF.Variables["X"])
    t0 = np.array(
        UF.Variables["Y"]
    )  # <--- Time is always in the 'Y' variable when reading a UFfile
    z0 = np.array(UF.Variables["Z"])

    it = [i for i, j in enumerate(t0) if j < TransitionTime][-1]
    Te_orig = z0[:, it]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~ Construct pedestal
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if ped_ready is None:
        a = (PedestalBottom - PedestalTop) / PedestalWidth
        b = PedestalBottom - a

        if StraightLine:
            ydata = a * x + b
        else:
            try:
                ydata = PLASMAtools.fitTANHPedestal(
                    TtopN=PedestalTop, Tbottom=PedestalBottom, w=PedestalWidth, xgrid=x
                )
            except:
                print("tanh routine failed... doing a StraightLine")
                ydata = a * x + b

    else:
        print("\t- Pedestal structure provided externally, just inserting it")
        [x_fit, v_fit] = ped_ready
        ydata = np.interp(x, x_fit, v_fit)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~ Construct core profile
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~ Merging core/edge point
    xmerge = 1 - (PedestalWidth + mergeExtraRange)
    ix = np.argmin(np.abs(x - xmerge))
    yped = ydata[ix:]
    xped = x[ix:]

    # ~~~~ Interpolate previous Te to new x grid
    Te_interp = np.interp(x, x0, Te_orig)

    # ~~~~ Scale up/down the entire core profile according to how much the pedestal has changed
    PedestalTop_corr = np.interp(xmerge, x, ydata)  # PedestalTop
    Te_orig_new = PLASMAtools.implementProfileBoost(
        x, Te_interp, PedestalTop_corr, ix=ix
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~ Merge profiles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Te_new = np.append(Te_orig_new[:ix], yped)

    # make sure it is smaller than 100keV
    if "NEL" not in UFfile:
        for i in range(len(Te_new)):
            if Te_new[i] > 1e5:
                Te_new[i] = 0.999e5

    # make sure not negative values.
    for i in range(len(Te_new)):
        if Te_new[i] < 0:
            Te_new[i] = 0.0

    # ~~~~~~~~~~~~~~~~~~ Join to original times

    tN, cont = (
        np.sort(np.append(t0, [TransitionTime - TimeWidthTransition, TransitionTime])),
        0,
    )

    tN, zN, already = [], [], False
    for it in range(len(t0)):
        if t0[it] > TransitionTime:
            tN.append(TransitionTime - TimeWidthTransition)
            zN.append(Te_interp.tolist())
            tN.append(TransitionTime)
            zN.append(Te_new.tolist())
            tN.append(100.0)
            zN.append(Te_new.tolist())
            break
        else:
            tN.append(t0[it])
            zN.append(np.interp(x, x0, z0[:, it]).tolist())

    time, Te, rho = np.array(tN), np.array(zN), np.array(x)

    # ~~~~~~~~~~~~~~~~~~ Write UFile

    UF.Variables["X"], UF.Variables["Y"], UF.Variables["Z"] = (
        rho,
        time,
        np.transpose(Te),
    )
    UF.writeUFILE(UFfile)

    # plot
    if debug:
        fig, ax = plt.subplots()
        # ax.plot(x0,Te_orig,label='Original')
        ax.plot(x, Te_interp, label="Original")
        ax.plot(x_fit, v_fit, label="mtanh fit (orig)")
        ax.plot(x, ydata, label="mtanh fit")
        ax.plot(rho, Te[-1, :], label="New")

        UF_new = UFILEStools.UFILEtransp()
        UF_new.readUFILE(UFfile)
        UF_new.plotVar(ax=ax, val=100.0, label="New UF")

        ax.plot([1 - PedestalWidth], [PedestalTop_corr], "o", markersize=3)
        ax.legend()
        plt.show()
        import pdb

        pdb.set_trace()
