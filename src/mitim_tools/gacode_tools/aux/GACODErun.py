import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mitim_tools.transp_tools.tools import PLASMASTATEtools
from mitim_tools.misc_tools import FARMINGtools, IOtools, MATHtools, GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

from IPython import embed

verbose_level = read_verbose_level()


def runTGYRO(
    folderWork,
    outputFiles=[],
    nameRunid="",
    nameJob="tgyro_prf",
    nparallel=8,
    minutes=30,
    inputFiles=[],
    launchSlurm=True,
):
    """
    This is the auxiliary function that TGYROtools will call to run TGYRO. It must be standalone.
    ------------------------------------------------------------------------------------------------
    launchSlurm = True 	-> Launch as a batch job in the machine chosen
    launchSlurm = False -> Launch locally as a bash script
    """

    # This routine assumes that folderWork contains input.profiles and input.tgyro already

    tgyro_job = FARMINGtools.mitim_job(folderWork)

    tgyro_job.define_machine(
        "tgyro",
        f"mitim_{nameRunid}/",
        launchSlurm=launchSlurm,
        slurm_settings={
            "minutes": minutes,
            "ntasks": 1,
            "name": nameJob,
            "cpuspertask": nparallel,
        },
    )

    # ------ Run TGYRO

    inputFiles.append(folderWork + "input.tglf")
    inputFiles.append(folderWork + "input.tgyro")
    inputFiles.append(folderWork + "input.gacode")

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

    LocationCDF = os.path.abspath(LocationCDF)
    Folder = "/".join(LocationCDF.split("/")[:-1]) + "/"
    print(
        f"\t- Looking for namelist in folder ...{Folder[np.max([-40,-len(Folder)]):]}"
    )

    NML = IOtools.findFileByExtension(Folder, "TR.DAT", ForceFirst=ForceFirst)

    # -----------------------------------------------------------
    # Copy to folder or create dummy if it has not been found
    # -----------------------------------------------------------

    LocationNML = f"{folderWork}/{nameRunid}TR.DAT"

    if NML is None:
        print(
            "\t\t- Creating dummy namelist because it was not found in folder",
            typeMsg="w",
        )
        with open(LocationNML, "w") as f:
            f.write(f"nshot = {nameRunid}")
        dummy = True
    else:
        LocationNML_orig = Folder + NML + "TR.DAT"
        print(
            f"\t\t- Namelist was found: {NML}TR.DAT, copying to ...{LocationNML[np.max([-40,-len(LocationNML)]):]}"
        )
        os.system(f"cp {LocationNML_orig} {LocationNML}")
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
        UsePRFmodification=fixPlasmaState,
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

    if not os.path.exists(folderWork):
        os.system(f"mkdir {folderWork}")
    if sendState:
        os.system(f"cp {LocationCDF} {folderWork}/{nameFiles}.CDF")
    os.system(f"mv {LocationNML} {folderWork}/{nameFiles}TR.DAT")

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
        not os.path.exists(f"{folderWork}/{nameOutputs}.cdf")
        or (not os.path.exists(f"{folderWork}/{nameOutputs}.geq"))
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


def executeCGYRO(
    FolderCGYRO,
    linesCGYRO,
    fileProfiles,
    outputFiles=["out.cgyro.run"],
    name="",
    numcores=32,
):
    if not os.path.exists(FolderCGYRO):
        os.system(f"mkdir {FolderCGYRO}")

    cgyro_job = FARMINGtools.mitim_job(FolderCGYRO)

    cgyro_job.define_machine(
        "cgyro",
        f"mitim_cgyro_{name}/",
        slurm_settings={
            "minutes": 60,
            "ntasks": numcores,
            "name": name,
        },
    )

    # ---------------
    # Prepare files
    # ---------------

    fileCGYRO = FolderCGYRO + "/input.cgyro"
    with open(fileCGYRO, "w") as f:
        f.write("\n".join(linesCGYRO))

    # ---------------
    # Execution command
    # ---------------

    folderExecution = cgyro_job.machineSettings["folderWork"]
    CGYROcommand = f"cgyro -e . -n {numcores} -p {folderExecution}"

    shellPreCommands = []

    # ---------------
    # Execute
    # ---------------

    cgyro_job.prep(
        CGYROcommand,
        input_files=[fileProfiles, fileCGYRO],
        output_files=outputFiles,
        shellPreCommands=shellPreCommands,
    )

    cgyro_job.run()


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

    commandTRXPL = "P\n10000\nA\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\nY\nX\nH\nW\n10001\nQ\nQ\nQ".format(
        timeRun, avTime, grids[0], grids[1], grids[2], BtDir, IpDir
    )
    with open(FolderTRXPL + "trxpl.in", "w") as f:
        f.write(commandTRXPL)

    if sendState:
        inputFiles = [
            FolderTRXPL + "trxpl.in",
            FolderTRXPL + f"{nameFiles}TR.DAT",
            FolderTRXPL + f"{nameFiles}.CDF",
        ]
    else:
        inputFiles = [
            FolderTRXPL + "trxpl.in",
            FolderTRXPL + f"{nameFiles}TR.DAT",
        ]

    if grids[0] > 301:
        raise Exception("~~~~ Max grid for TRXPL is 301")

    print(
        "\t\t- Proceeding to run TRXPL with: {0}".format(
            (" ".join(commandTRXPL.split("\n")))
        ),
        typeMsg="i",
    )

    trxpl_job = FARMINGtools.mitim_job(FolderTRXPL)
    trxpl_job.define_machine(
        "trxpl",
        f"mitim_trxpl_{nameOutputs}/",
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
    UsePRFmodification=False,
    includeGEQ=True,
):
    runWithoutEqIfFail = False  # If profiles_gen fails, try without the "-g" option

    if UsePRFmodification:
        print("\t\t- Running modifyPlasmaState")
        os.system("cp {0} {0}_old".format(FolderTGLF + "{0}.cdf".format(nameFiles)))
        pls = PLASMASTATEtools.Plasmastate(FolderTGLF + f"{nameFiles}.cdf_old")
        pls.modify_default(FolderTGLF + f"{nameFiles}.cdf")

    inputFiles = [
        FolderTGLF + "profiles_gen.sh",
        FolderTGLF + f"{nameFiles}.cdf",
    ]

    if includeGEQ:
        inputFiles.append(FolderTGLF + f"{nameFiles}.geq")

    # **** Write command
    txt = f"profiles_gen -i {nameFiles}.cdf"
    if includeGEQ:
        txt += f" -g {nameFiles}.geq\n"
    else:
        txt += "\n"
    with open(FolderTGLF + "profiles_gen.sh", "w") as f:
        f.write(txt)
    # ******************

    print(f"\t\t- Proceeding to run PROFILES_GEN with: {txt}")

    pgen_job = FARMINGtools.mitim_job(FolderTGLF)
    pgen_job.define_machine(
        "profiles_gen",
        f"mitim_profiles_gen_{nameFiles}/",
    )

    pgen_job.prep(
        "bash profiles_gen.sh",
        input_files=inputFiles,
        output_files=["input.gacode"],
    )
    pgen_job.run()

    if (
        runWithoutEqIfFail
        and (not os.path.exists(FolderTGLF + "/input.gacode"))
        and (includeGEQ)
    ):
        print(
            "\t\t- PROFILE_GEN failed, running without the geqdsk file option to see if that works...",
            typeMsg="w",
        )

        # **** Write command
        txt = f"profiles_gen -i {nameFiles}.cdf\n"
        with open(FolderTGLF + "profiles_gen.sh", "w") as f:
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

    vgen_job = FARMINGtools.mitim_job(workingFolder)

    vgen_job.define_machine(
        "profiles_gen",
        f"mitim_vgen_{name_run}/",
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

    inputgacode_file = f"{workingFolder}/input.gacode"

    _, nameFile = IOtools.reducePathLevel(inputgacode_file, level=1, isItFile=True)

    command = f"cd {vgen_job.machineSettings['folderWork']} && bash profiles_vgen.sh"
    with open(f"{workingFolder}/profiles_vgen.sh", "w") as f:
        f.write(f"profiles_gen -vgen -i {nameFile} {options} -n {numcores}")

    # ---------------
    # Execute
    # ---------------

    vgen_job.prep(
        command,
        input_files=[inputgacode_file, f"{workingFolder}/profiles_vgen.sh"],
        output_files=["slurm_output.dat", "slurm_error.dat"],
    )

    vgen_job.run()

    file_new = f"{workingFolder}/vgen/input.gacode"

    return file_new


def buildDictFromInput(inputFile):
    parsed = {}

    lines = inputFile.split("\n")
    for line in lines:
        if "=" in line:
            splits = [i.split()[0] for i in line.split("=")]
            if ("." in splits[1]) and (splits[1][0].split()[0] != "."):
                parsed[splits[0].split()[0]] = float(splits[1].split()[0])
            else:
                try:
                    parsed[splits[0].split()[0]] = int(splits[1].split()[0])
                except:
                    parsed[splits[0].split()[0]] = splits[1].split()[0]

    for i in parsed:
        if isinstance(parsed[i], str):
            if (
                parsed[i].lower() == "t"
                or parsed[i].lower() == "true"
                or parsed[i].lower() == ".true."
            ):
                parsed[i] = True
            elif (
                parsed[i].lower() == "f"
                or parsed[i].lower() == "false"
                or parsed[i].lower() == ".false."
            ):
                parsed[i] = False

    return parsed


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


def runCGYRO(
    rhos,
    finalFolder,
    tmpFolder,
    inputFilesCGYRO,
    inputGacode,
    numcores=32,
    extraFlag="",
    filesToRetrieve=["cgyro/out.cgyro.info"],
    name="",
):
    MaxRunsWithoutWait, timewaitSendRun = 4, 10

    for rho in rhos:
        print(f" ~~ Running CGYRO at rho={rho:.4f}")
        if len(rhos) > MaxRunsWithoutWait:
            time.sleep(timewaitSendRun)

        fileCGYRO = executeCGYRO(
            tmpFolder,
            inputFilesCGYRO[rho],
            inputGacode,
            numcores=numcores,
            outputFiles=filesToRetrieve,
            name=name,
        )

        for file in filesToRetrieve:
            fr = f"{file}_{rho:.4f}{extraFlag}"
            ff = f"{finalFolder}/{fr}"
            if os.path.exists(ff):
                os.system("rm " + ff)
            os.system(f"mv {tmpFolder}/{file} {ff}")

            if verbose_level in [4, 5]:
                print("\t- Retrieving files and changing names for storing")
                if not os.path.exists(ff):
                    print(f"\t!! file {file} ({fr}) could not be retrived")
                # else:
                # print(
                #     f"\t\t~ {file} successfully retrieved, converted into {fr}",
                #     verbose=verbose_level,
                # )

        os.system(f"mv {fileCGYRO} {finalFolder}/input.cgyro_{rho:.4f}")


def runTGLF(
    rhos,
    finalFolder,
    inputFilesTGLF,
    minutes=5,
    cores_tglf=4,
    extraFlag="",
    filesToRetrieve=["out.tglf.gbflux"],
    name="",
    launchSlurm=True,
):
    """
    launchSlurm = True -> Launch as a batch job in the machine chosen
    launchSlurm = False -> Launch locally as a bash script
    """

    tmpFolder = finalFolder + "/tmp_tglf/"
    IOtools.askNewFolder(tmpFolder)

    tglf_job = FARMINGtools.mitim_job(tmpFolder)

    tglf_job.define_machine_quick(
        "tglf",
        f"mitim_{name}/",
    )

    # ---------------------------------------------
    # Prepare files and folders
    # ---------------------------------------------

    folders, folders_red = [], []
    for i, rho in enumerate(rhos):
        print(f"\t- Preparing TGLF at rho={rho:.4f}")

        folderTGLF_this = f"{tmpFolder}/rho_{rho:.4f}"
        folders.append(folderTGLF_this)
        folders_red.append(f"rho_{rho:.4f}")

        if not os.path.exists(folderTGLF_this):
            os.system(f"mkdir {folderTGLF_this}")

        fileTGLF = f"{folderTGLF_this}/input.tglf"
        with open(fileTGLF, "w") as f:
            f.write(inputFilesTGLF[rho])

    # ---------------------------------------------
    # Prepare command
    # ---------------------------------------------

    total_tglf_cores = int(cores_tglf * len(rhos))

    if launchSlurm and ("partition" in tglf_job.machineSettings["slurm"]):
        typeRun = "job" if total_tglf_cores <= 32 else "array"
    else:
        typeRun = "bash"

    if typeRun in ["bash", "job"]:
        TGLFcommand = ""
        for i, rho in enumerate(rhos):
            TGLFcommand += f"tglf -e rho_{rho:.4f}/ -n {cores_tglf} -p {tglf_job.folderExecution}/ &\n"

        TGLFcommand += (
            "\nwait"  # This is needed so that the script doesn't end before each job
        )
        rho_array = None

        ntasks = len(rhos)
        cpuspertask = cores_tglf

    elif typeRun in ["array"]:
        print(
            f"\t- TGLF will be executed in SLURM as job array due to its size (cpus: {total_tglf_cores})",
            typeMsg="i",
        )

        rho_array = ",".join([f"{int(rho*1E4)}" for rho in rhos])
        TGLFcommand = f'tglf -e rho_0."$SLURM_ARRAY_TASK_ID"/ -n {cores_tglf} -p {tglf_job.folderExecution}/ 1> {tglf_job.folderExecution}/rho_0."$SLURM_ARRAY_TASK_ID"/slurm_output.dat 2> {tglf_job.folderExecution}/rho_0."$SLURM_ARRAY_TASK_ID"/slurm_error.dat\n'

        ntasks = 1
        cpuspertask = cores_tglf

    # ---------------------------------------------
    # Execute
    # ---------------------------------------------

    tglf_job.define_machine(
        "tglf",
        f"mitim_{name}/",
        launchSlurm=launchSlurm,
        slurm_settings={
            "minutes": minutes,
            "ntasks": ntasks,
            "name": name,
            "cpuspertask": cpuspertask,
            "job_array": rho_array,
            "nodes": 1,
        },
    )

    # I would like the mitim_job to check if the retrieved folders were complete
    check_files_in_folder = {}
    for folder in folders_red:
        check_files_in_folder[folder] = filesToRetrieve
    # ---------------------------------------------

    tglf_job.prep(
        TGLFcommand,
        input_folders=folders,
        output_folders=folders_red,
        check_files_in_folder= check_files_in_folder
    )

    tglf_job.run() #removeScratchFolders=False)

    # ---------------------------------------------
    # Organize
    # ---------------------------------------------

    print("\t- Retrieving files and changing names for storing")
    fineall = True
    for i, rho in enumerate(rhos):
        for file in filesToRetrieve:
            original_file = f"{file}_{rho:.4f}{extraFlag}"
            final_destination = f"{finalFolder}/{original_file}"

            if os.path.exists(final_destination):
                os.system(f"rm {final_destination}")

            os.system(f"mv {tmpFolder}/rho_{rho:.4f}/{file} {final_destination}")

            fineall = fineall and os.path.exists(final_destination)

            if not os.path.exists(final_destination):
                print(
                    f"\t!! file {file} ({original_file}) could not be retrived",
                    typeMsg="w",
                    verbose=verbose_level,
                )

    if fineall:
        print("\t\t- All files were successfully retrieved")
    else:
        print("\t\t- Some files were not retrieved", typeMsg="w")
