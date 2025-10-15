import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mitim_tools.transp_tools.utils import NTCCtools
from mitim_tools.misc_tools import FARMINGtools, IOtools, MATHtools, GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

def runTGYRO(
    folderWork,
    outputFiles=None,
    nameRunid="",
    nameJob="tgyro_mitim",
    nparallel=8,
    minutes=30,
    inputFiles=None,
    launchSlurm=True,
):
    """
    This is the auxiliary function that TGYROtools will call to run TGYRO. It must be standalone.
    ------------------------------------------------------------------------------------------------
    launchSlurm = True 	-> Launch as a batch job in the machine chosen
    launchSlurm = False -> Launch locally as a bash script
    """

    if outputFiles is None:
        outputFiles = []
    
    if inputFiles is None:
        inputFiles = []

    # This routine assumes that folderWork contains input.profiles and input.tgyro already

    tgyro_job = FARMINGtools.mitim_job(folderWork)

    tgyro_job.define_machine(
        "tgyro",
        f"mitim_{nameRunid}",
        launchSlurm=launchSlurm,
        slurm_settings={
            "minutes": minutes,
            "ntasks": 1,
            "name": nameJob,
            "cpuspertask": nparallel,
        },
    )

    # ------ Run TGYRO

    inputFiles.append(folderWork / "input.tglf")
    inputFiles.append(folderWork / "input.tgyro")
    inputFiles.append(folderWork / "input.gacode")

    # ---------------
    # Execution command
    # ---------------

    folderExecution = tgyro_job.machineSettings["folderWork"]

    TGYROcommand = f"tgyro -e . -n {nparallel} -p {folderExecution}"

    # Before calling tgyro, create TGLF folders and place input.tglfs there
    shellPreCommands = [
        f"for i in `seq 1 {nparallel}`;",
        "do",
        "	mkdir TGLF$i",
        "	cp input.tglf TGLF$i/input.tglf",
        "done",
    ]

    # After calling tgyro, move the out.tglf.localdump files
    shellPostCommands = [
        f"for i in `seq 1 {nparallel}`;",
        "do",
        "	cp TGLF$i/out.tglf.localdump input.tglf.new$i",
        "done",
    ]

    # ---------------------------------------------
    # Execute
    # ---------------------------------------------

    tgyro_job.prep(
        TGYROcommand,
        input_files=inputFiles,
        output_files=outputFiles,
        shellPreCommands=shellPreCommands,
        shellPostCommands=shellPostCommands,
    )

    tgyro_job.run()

def findNamelist(LocationCDF, folderWork=None, nameRunid="10000", ForceFirst=True):
    # -----------------------------------------------------------
    # Find namelist
    # -----------------------------------------------------------

    LocationCDF = IOtools.expandPath(LocationCDF)
    Folder = LocationCDF.parent
    print(f"\t- Looking for namelist in folder ...{IOtools.clipstr(Folder)}")

    LocationNML = IOtools.findFileByExtension(Folder, "TR.DAT", ForceFirst=ForceFirst)

    # -----------------------------------------------------------
    # Copy to folder or create dummy if it has not been found
    # -----------------------------------------------------------

    if LocationNML is None:
        LocationNML = folderWork / f"{nameRunid}TR.DAT"
        print("\t\t- Creating dummy namelist because it was not found in folder",typeMsg="i",)
        with open(LocationNML, "w") as f:
            f.write(f"nshot = {nameRunid}")
        dummy = True
    else:
        dummy = False

    return LocationNML, dummy

def prepareTGYRO(
    LocationCDF,
    LocationNML,
    timeRun,
    avTime=0.0,
    BtIp_dirs=[0, 0],
    fixPlasmaState=True,
    gridsTRXPL=[151, 101, 101],
    folderWork="~/scratchFolder/",
    StateGenerated=False,
    sendState=True,
    nameRunid="",
    includeGEQ=True,
):
    nameWork = "10001"
    folderWork = IOtools.expandPath(folderWork)

    if not StateGenerated:
        print("\t- Running TRXPL to extract g-file and plasmastate")
        CDFtoTRXPLoutput(
            LocationCDF,
            LocationNML,
            timeRun,
            avTime=avTime,
            BtIp_dirs=BtIp_dirs,
            nameOutputs=nameWork,
            folderWork=folderWork,
            grids=gridsTRXPL,
            sendState=sendState,
        )

    print("\t- Running PROFILES_GEN to generate input.profiles/input.gacode files")
    runPROFILES_GEN(
        folderWork,
        nameFiles=nameWork,
        UseMITIMmodification=fixPlasmaState,
        includeGEQ=includeGEQ,
    )

def CDFtoTRXPLoutput(
    LocationCDF,
    LocationNML,
    timeRun,
    avTime=0.0,
    BtIp_dirs=[0, 0],
    nameOutputs="10001",
    folderWork="~/scratchFolder/",
    grids=[151, 101, 101],
    sendState=True,
):
    nameFiles, fail_attempts = "10000", 2
    folderWork = IOtools.expandPath(folderWork)

    folderWork.mkdir(parents=True, exist_ok=True)
    if sendState:
        cdffile = folderWork / f'{nameFiles}.CDF'
        shutil.copy2(LocationCDF, cdffile)
    trfile = folderWork / f'{nameFiles}TR.DAT'
    LocationNML.replace(trfile)

    runTRXPL(
        folderWork,
        timeRun,
        BtDir=BtIp_dirs[0],
        IpDir=BtIp_dirs[1],
        avTime=avTime,
        nameFiles=nameFiles,
        sendState=sendState,
        nameOutputs=nameOutputs,
        grids=grids,
    )

    # Retry for random error
    cont = 1
    while (
        (not (folderWork / f"{nameOutputs}.cdf").exists())
        or (not (folderWork / f"{nameOutputs}.geq").exists())
    ) and cont < fail_attempts:
        print("\t\t- Re-running to see if it was a random error", typeMsg="i")
        cont += 1
        runTRXPL(
            folderWork,
            timeRun,
            BtDir=BtIp_dirs[0],
            IpDir=BtIp_dirs[1],
            avTime=avTime,
            nameFiles=nameFiles,
            sendState=sendState,
            nameOutputs=nameOutputs,
            grids=grids,
        )

def runTRXPL(
    FolderTRXPL,
    timeRun,
    BtDir="1",
    IpDir="1",
    avTime=0.0,
    nameFiles="10000",
    nameOutputs="10001",
    sendState=True,
    grids=[151, 101, 101],
):
    """
    trxpl_path:  #theta pts for 2d splines:
    trxpl_path:  #R pts for cartesian overlay grid:
    trxpl_path:  #Z pts for cartesian overlay grid:

    TRXPL asks for direction:
            trxpl_path:  enter "1" for if Btoroidal is ccw:
              "ccw" means "counter-clockwise looking down from above".
              ...enter "-1" for clockwise orientation.
              ...enter "0" to read orientation from TRANSP data archive.
            trxpl_path:  enter "1" for if Btoroidal is ccw:
    """

    commandTRXPL = f"P\n10000\nA\n{timeRun}\n{avTime}\n{grids[0]}\n{grids[1]}\n{grids[2]}\n{BtDir}\n{IpDir}\nY\nX\nH\nW\n10001\nQ\nQ\nQ"
    with open(FolderTRXPL / "trxpl.in", "w") as f:
        f.write(commandTRXPL)

    if sendState:
        inputFiles = [
            FolderTRXPL / "trxpl.in",
            FolderTRXPL / f"{nameFiles}TR.DAT",
            FolderTRXPL / f"{nameFiles}.CDF",
        ]
    else:
        inputFiles = [
            FolderTRXPL / "trxpl.in",
            FolderTRXPL / f"{nameFiles}TR.DAT",
        ]

    if grids[0] > 301:
        raise Exception("~~~~ Max grid for TRXPL is 301")

    print(f"\t\t- testProceeding to run TRXPL with: {' '.join(commandTRXPL.splitlines())}", typeMsg="i")


    trxpl_job = FARMINGtools.mitim_job(FolderTRXPL)
    trxpl_job.define_machine(
        "trxpl",
        f"mitim_trxpl_{nameOutputs}",
    )

    command = "trxpl < trxpl.in"

    trxpl_job.prep(
        command,
        input_files=inputFiles,
        output_files=[f"{nameOutputs}.cdf", f"{nameOutputs}.geq"],
    )
    trxpl_job.run()

def runPROFILES_GEN(
    FolderTGLF,
    nameFiles="10001",
    UseMITIMmodification=False,
    includeGEQ=True,
):
    runWithoutEqIfFail = False  # If profiles_gen fails, try without the "-g" option

    if UseMITIMmodification:
        print("\t\t- Running modifyPlasmaState")
        shutil.copy2(FolderTGLF / f"{nameFiles}.cdf", FolderTGLF / f"{nameFiles}.cdf_old")
        pls = NTCCtools.Plasmastate(FolderTGLF / f"{nameFiles}.cdf_old")
        pls.modify_default(FolderTGLF / f"{nameFiles}.cdf")

    inputFiles = [
        FolderTGLF / "profiles_gen.sh",
        FolderTGLF / f"{nameFiles}.cdf",
    ]

    if includeGEQ:
        inputFiles.append(FolderTGLF / f"{nameFiles}.geq")

    # **** Write command
    txt = f"profiles_gen -i {nameFiles}.cdf"
    if includeGEQ:
        txt += f" -g {nameFiles}.geq\n"
    else:
        txt += "\n"
    with open(FolderTGLF / "profiles_gen.sh", "w") as f:
        f.write(txt)
    # ******************

    print(f"\t\t- Proceeding to run PROFILES_GEN with: {txt}")

    pgen_job = FARMINGtools.mitim_job(FolderTGLF)
    pgen_job.define_machine(
        "profiles_gen",
        f"mitim_profiles_gen_{nameFiles}",
    )

    pgen_job.prep(
        "bash profiles_gen.sh",
        input_files=inputFiles,
        output_files=["input.gacode"],
    )
    pgen_job.run()

    if (
        runWithoutEqIfFail
        and (not (FolderTGLF / "input.gacode").exists())
        and (includeGEQ)
    ):
        print(
            "\t\t- PROFILE_GEN failed, running without the geqdsk file option to see if that works...",
            typeMsg="w",
        )

        # **** Write command
        txt = f"profiles_gen -i {nameFiles}.cdf\n"
        with open(FolderTGLF / "profiles_gen.sh", "w") as f:
            f.write(txt)
        # ******************

        print(f"\t\t- Proceeding to run PROFILES_GEN with: {txt}")
        pgen_job.run()

def runVGEN(
    workingFolder,
    numcores=32,
    minutes=60,
    vgenOptions={},
    name_run="vgen1",
):
    """
    Driver for the vgen (velocity-generation) capability of NEO.
    This will write a new input.gacode with NEO-computed electric field and/or velocities.

    **** Options
            -er:  Method to compute Er
                            1 = Computing omega0 (Er) from force balance
                            2 = Computing omega0 (Er) from NEO (weak rotation limit)
                            3 = ?????NEO (strong rot)?????
                            4 = Returning given omega0 (Er)
            -vel: Method to compute velocities
                            1 = Computing velocities from NEO (weak rotation limit)
                            2 = Computing velocities from NEO (strong rotation limit)
                            3 = ?????Return given?????
            -in:  Number of ion species (uses default neo template. Otherwise, input.neo must exist)
            -ix:  Which ion to match velocities? Index of ion species to match NEO and given velocities
            -nth: Minimum and maximum theta resolutions (e.g. 17,39)
    """

    workingFolder = IOtools.expandPath(workingFolder)
    vgen_job = FARMINGtools.mitim_job(workingFolder)

    vgen_job.define_machine(
        "profiles_gen",
        f"mitim_vgen_{name_run}",
        slurm_settings={
            "minutes": minutes,
            "ntasks": numcores,
            "name": f"neo_vgen_{name_run}",
        },
    )

    print(
        f"\t- Running NEO (with {vgenOptions['numspecies']} species) to populate w0(rad/s) in input.gacode file"
    )
    print(f"\t\t> Matching ion {vgenOptions['matched_ion']} Vtor")

    options = f"-er {vgenOptions['er']} -vel {vgenOptions['vel']} -in {vgenOptions['numspecies']} -ix {vgenOptions['matched_ion']} -nth {vgenOptions['nth']}"

    # ***********************************

    print(
        f"\t\t- Proceeding to generate Er from NEO run using profiles_gen -vgen ({options})"
    )

    inputgacode_file = workingFolder / f"input.gacode"

    _, nameFile = IOtools.reducePathLevel(inputgacode_file, level=1, isItFile=True)

    command = f"cd {vgen_job.machineSettings['folderWork']} && bash profiles_vgen.sh"
    with open(workingFolder / f"profiles_vgen.sh", "w") as f:
        f.write(f"profiles_gen -vgen -i {nameFile} {options} -n {numcores}")

    # ---------------
    # Execute
    # ---------------

    vgen_job.prep(
        command,
        input_files=[inputgacode_file, workingFolder / f"profiles_vgen.sh"],
        output_files=["slurm_output.dat", "slurm_error.dat"],
    )

    vgen_job.run()

    file_new = workingFolder / "vgen" / "input.gacode"

    return file_new

# ----------------------------------------------------------------------
# 						Reading/Writing routines
# ----------------------------------------------------------------------

def obtainFluctuationLevel(
    ky,
    Amplitude,
    rhos,
    a,
    convolution_fun_fluct=None,
    rho_eval=0.6,
    factorTot_to_Perp=1.0,
    printYN=True,
):
    """
    Amplitude must be AMPLITUDE (see my notes), not intensity
    factorTot_to_Perp is applied if I want to convert TOTAL to Perpendicular

    """
    ky_integr = [0, 5]
    integrand = np.array(Amplitude) ** 2
    fluctSim = (
        rhos
        / a
        * np.sqrt(
            integrateSpectrum(
                ky,
                integrand,
                ky_integr[0],
                ky_integr[1],
                convolution_fun_fluct=convolution_fun_fluct,
                rhos=rhos,
                rho_eval=rho_eval,
            )
        )
    )

    if factorTot_to_Perp != 1.0 and printYN:
        print(
            f'\t\t- Fluctuations x{factorTot_to_Perp} to account for "total-to-perp." conversion',
            typeMsg="i",
        )

    return fluctSim * 100.0 * factorTot_to_Perp

def obtainNTphase(
    ky,
    nTphase,
    rhos,
    a,
    convolution_fun_fluct=None,
    rho_eval=0.6,
    factorTot_to_Perp=1.0,
):
    ky_integr = [0, 5]
    x, y = defineNewGrid(ky, nTphase, ky_integr[0], ky_integr[1], kind="linear")
    if convolution_fun_fluct is not None:
        gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
    else:
        gaussW = np.ones(len(x))

    neTe = np.sum(y * gaussW) / np.sum(gaussW)

    return neTe

def integrateSpectrum(
    xOriginal,
    yOriginal,
    xmin,
    xmax,
    convolution_fun_fluct=None,
    rhos=None,
    debug=False,
    rho_eval=0.6,
):
    x, y = defineNewGrid(xOriginal, yOriginal, xmin, xmax, kind="linear")

    if convolution_fun_fluct is not None:
        gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
    else:
        gaussW = np.ones(len(x))

    integ = MATHtools.integrate_definite(x, y * gaussW)

    if debug:
        fig = plt.figure(figsize=(17, 8))
        grid = plt.GridSpec(2, 3, hspace=0.2, wspace=0.35)

        ax = fig.add_subplot(grid[0, 0])
        ax.set_title("Option 1")
        ax1 = ax.twinx()
        ax.plot(x, y, "-o", c="r", markersize=2, label="interpolation")
        ax.plot(xOriginal, yOriginal, "o", c="b", markersize=4, label="TGLF output")
        ax1.plot(x, gaussW, ls="--", lw=0.5, c="k")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fluctuation Intensity ($A^2$)")
        ax1.set_ylabel("Convolution (C)")
        ax1.set_ylim([0, 1])
        ax = fig.add_subplot(grid[1, 0])
        gaussWO = np.interp(xOriginal, x, gaussW)
        ax.plot(x, y * gaussW, "-o", c="r", markersize=2)
        ax.plot(xOriginal, yOriginal * gaussWO, "o", c="b", markersize=4)
        GRAPHICStools.fillGraph(
            ax, x, y * gaussW, y_down=np.zeros(len(x)), y_up=None, alpha=0.2, color="g"
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A^2\\cdot C$")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")

        ax.text(
            0.5,
            0.5,
            f"I = {integ:.2f}",
            horizontalalignment="center",
            transform=ax.transAxes,
        )

        x, y = defineNewGrid(xOriginal, yOriginal, xmin, xmax)
        if convolution_fun_fluct is not None:
            gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
        else:
            gaussW = np.ones(len(x))
        integ = MATHtools.integrate_definite(x, y * gaussW)
        ax = fig.add_subplot(grid[0, 1])
        ax.set_title("Option 2")
        ax1 = ax.twinx()
        ax.plot(x, y, "-o", c="r", markersize=2, label="interpolation")
        ax.plot(xOriginal, yOriginal, "o", c="b", markersize=4, label="TGLF output")
        ax1.plot(x, gaussW, ls="--", lw=0.5, c="k")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fluctuation Intensity ($A^2$)")
        ax1.set_ylabel("Convolution (C)")
        ax1.set_ylim([0, 1])
        ax = fig.add_subplot(grid[1, 1])
        gaussWO = np.interp(xOriginal, x, gaussW)
        ax.plot(x, y * gaussW, "-o", c="r", markersize=2)
        ax.plot(xOriginal, yOriginal * gaussWO, "o", c="b", markersize=4)
        GRAPHICStools.fillGraph(
            ax, x, y * gaussW, y_down=np.zeros(len(x)), y_up=None, alpha=0.2, color="g"
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A^2\\cdot C$")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")

        ax.text(
            0.5,
            0.5,
            f"I = {integ:.2f}",
            horizontalalignment="center",
            transform=ax.transAxes,
        )

        x, y = defineNewGrid(xOriginal, yOriginal, xmin, xmax)
        if convolution_fun_fluct is not None:
            gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
        else:
            gaussW = np.ones(len(x))
        gaussWO = np.interp(xOriginal, x, gaussW)

        ax = fig.add_subplot(grid[0, 2])
        ax.set_title("Option 3")
        ax1 = ax.twinx()
        # ax.plot(x,y,'-o',c='r',markersize=2,label='interpolation')
        ax.plot(xOriginal, yOriginal, "o", c="b", markersize=4, label="TGLF output")
        ax1.plot(x, gaussW, ls="--", lw=0.5, c="k")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fluctuation Intensity ($A^2$)")
        ax1.set_ylabel("Convolution (C)")
        ax1.set_ylim([0, 1])
        ax = fig.add_subplot(grid[1, 2])

        x, ys = defineNewGrid(xOriginal, yOriginal * gaussWO, xmin, xmax)
        integ = MATHtools.integrate_definite(x, ys)
        ax.plot(x, ys, "-o", c="r", markersize=2)
        ax.plot(xOriginal, yOriginal * gaussWO, "o", c="b", markersize=4)
        GRAPHICStools.fillGraph(
            ax, x, ys, y_down=np.zeros(len(x)), y_up=None, alpha=0.2, color="g"
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A^2\\cdot C$")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")

        ax.text(
            0.5,
            0.5,
            f"I = {integ:.2f}",
            horizontalalignment="center",
            transform=ax.transAxes,
        )

        plt.show()

    return integ

def defineNewGrid(
    xOriginal1,
    yOriginal1,
    xmin,
    xmax,
    debug=False,
    createZero=True,
    interpolateY=True,
    kind="linear",
):
    """
    if createZero, then it adds a point at x=0
    if createZero and interpolateY, the point to be added is y_new[0] = y_old[0], i.e. extrapolation of first value
    """

    if createZero:
        xOriginal1 = np.insert(xOriginal1, 0, 0, axis=0)
        if interpolateY:
            yOriginal1 = np.insert(yOriginal1, 0, yOriginal1[0], axis=0)
        else:
            yOriginal1 = np.insert(yOriginal1, 0, 0, axis=0)

        # Making sure that xOriginal is monotonically increasing
    xOriginal, yOriginal = [], []
    prev = 0.0
    for i in range(len(xOriginal1)):
        if xOriginal1[i] >= prev:
            xOriginal.append(xOriginal1[i])
            yOriginal.append(yOriginal1[i])
            prev = xOriginal1[i]
        else:
            break

    if xOriginal[0] > xmax:
        print(
            "Wavenumber spectrum is too coarse for fluctuations analysis, using the minimum value",
            typeMsg="w",
        )
        xmax = xOriginal[0] + 1e-4

    f = interp1d(xOriginal, yOriginal, kind=kind)
    x = np.linspace(xOriginal[0], max(xOriginal), int(1e4))
    y = f(x)

    imin = np.argmin(np.abs(x - xmin))  # [i for i, j in enumerate(x) if j>=xmin][0]
    imax = np.argmin(np.abs(x - xmax))  # [i for i, j in enumerate(x) if j>=xmax][0]

    if debug:
        fn = plt.figure()
        ax = fn.add_subplot(111)
        ax.plot(x, y)
        ax.scatter(x, y, label="New points")
        ax.scatter(xOriginal, yOriginal, 100, label="Original points")
        xli = np.array(x[imin:imax])
        yli2 = np.array(y[imin:imax])
        ax.fill_between(xli, yli2, facecolor="r", alpha=0.5)
        ax.set_xlim([0, 1.0])
        ax.set_xlabel("ky")
        ax.set_ylabel("T_fluct")
        ax.legend()

        plt.show()

    return x[imin:imax], y[imin:imax]

