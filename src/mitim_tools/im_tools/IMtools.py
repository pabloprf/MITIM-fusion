import sys, os, time, glob, re, copy, ast, pdb, netCDF4, traceback, pickle, torch, datetime
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

from mitim_tools.misc_tools import IOtools, MATHtools, PLASMAtools
from mitim_tools.transp_tools import UFILEStools, CDFtools
from mitim_modules.powertorch.aux import PARAMtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.im_tools import IMparam
from mitim_tools.im_tools.modules import PEDmodule, TRANSPmodule, EQmodule

from mitim_tools.misc_tools.IOtools import printMsg as print


def runIMworkflow(
    IMnamelist,
    FolderEvaluation,
    numEval,
    SpecificParams={},
    modNamelist={},
):
    checkForActive = True  # True if I want to know run status (preferred choice). False if website gives time-out.
    # Problem: False assumes that the run is running and look has been generated

    # ---------- Define debugging option
    """
    # DebugMode = 0: No Debug, start from running 1st phase
    # DebugMode = 1: Run 2nd phase (assumes FolderTRANSP exists with 1stphase)
    # DebugMode = 2: Only evaluate (assumes FolderTRANSP contains final results; ideally with results of 2nd or 2nd-double phases)
    # DebugMode = 3: Run 2nd-double phase (assumes FolderTRANSP exists with 2nd phase)
    """

    SpecificParams.setdefault("IsItSingle", True)
    SpecificParams.setdefault("forceMinimumDebug", None)
    SpecificParams.setdefault("automaticProcess", False)
    SpecificParams.setdefault("lock", None)

    DebugMode = 0
    if SpecificParams["forceMinimumDebug"] is not None:
        DebugMode = np.max([DebugMode, SpecificParams["forceMinimumDebug"]])

    # if int(numEval) < 5:    DebugMode = 1

    # ---------- All information needed for a run

    mitimNML = IMparam.MITIMnamelist(
        FolderEvaluation,
        IMnamelist,
        numEval,
        DebugMode=DebugMode,
        IsItSingle=SpecificParams["IsItSingle"],
    )

    if "lock" in SpecificParams:
        mitimNML.lock = SpecificParams["lock"]
    else:
        mitimNML.lock = None

    for ikey in modNamelist:
        mitimNML.__dict__[ikey] = modNamelist[ikey]

    # ---------- Execute the evaluation

    mitimNML, statusMITIM, DebugMode = runWithCatchingError(
        mitimNML,
        DebugMode,
        checkForActive,
        automaticProcess=SpecificParams["automaticProcess"],
    )

    # ---------- If the run failed, let's repeat it

    if not statusMITIM:
        mitimNML = rerunIMworkflow(
            mitimNML,
            DebugMode,
            checkForActive,
            retries=mitimNML.retriesmitim - 1,
            automaticProcess=SpecificParams["automaticProcess"],
        )

    # ---- Sometimes during multiprocessing the logging fails and it appends a new evaluation to current one

    new_logfile = mitimNML.FolderOutputs + "MITIM.log"
    os.system(f"cp {mitimNML.logFile} {new_logfile}")

    return mitimNML


def runWithCatchingError(mitimNML, DebugMode, checkForActive, automaticProcess=False):
    try:
        statusMITIM = launchHFModel(
            mitimNML,
            DebugMode,
            checkForActive=checkForActive,
            automaticProcess=automaticProcess,
        )
    except:
        print(">> Error was found when running MITIM:", typeMsg="w")
        print(traceback.format_exc())
        print("---------------------------------------------------")
        statusMITIM = False

    # Find which one was the last completed phase
    try:
        dictStatus = interpretStatus(mitimNML.FolderOutputs + "/MITIM.log_tmp")
        DebugForNextTrial = 0
        if dictStatus["LastFinished"] == "Interpretive":
            DebugForNextTrial = 1
        elif dictStatus["LastFinished"] == "1st predictive":
            DebugForNextTrial = 3
        elif dictStatus["LastFinished"] == "2nd predictive":
            DebugForNextTrial = 2
    except:
        pass

    return mitimNML, statusMITIM, DebugForNextTrial


def rerunIMworkflow(
    mitimNML, DebugMode, checkForActive, retries=1, automaticProcess=False
):
    statusMITIM = False

    for i in range(retries):
        if not statusMITIM:
            PreviousTrial = i + 1

            print(
                ">> Trying to run again, maybe error was random, trial #{0} (last success phase:{1})".format(
                    PreviousTrial + 1, DebugMode
                )
            )

            # -----------------
            # Depending on the debugging flag, I need to work files out
            # -----------------

            # Move FolderTRANSP (present) to _trial
            if DebugMode != 2:
                print(
                    "\t-  Moving {0} to {1}_trial{2}".format(
                        mitimNML.FolderTRANSP, mitimNML.FolderTRANSP[:-1], PreviousTrial
                    )
                )
                os.system(f"rm -r {mitimNML.FolderTRANSP[:-1]}_trial{PreviousTrial}")
                os.system(
                    "mv {0} {1}_trial{2}".format(
                        mitimNML.FolderTRANSP, mitimNML.FolderTRANSP[:-1], PreviousTrial
                    )
                )
                os.system(
                    "mv {0} {1}_trial{2}".format(
                        mitimNML.FolderEQ, mitimNML.FolderEQ[:-1], PreviousTrial
                    )
                )

                # Move _interpretive or _predictive1 to the present FolderTRANSP
                if DebugMode > 0:
                    if DebugMode == 1:
                        fileActual = "interpretive"
                    if DebugMode == 3:
                        fileActual = "predictive1"
                    print(
                        "\t-  Moving {0}_{1} to {0}".format(
                            mitimNML.FolderTRANSP[:-1], fileActual
                        )
                    )
                    os.system(
                        "mv {0}_{1} {0}".format(mitimNML.FolderTRANSP[:-1], fileActual)
                    )
                    os.system(
                        "mv {0}_{1} {0}".format(mitimNML.FolderEQ[:-1], fileActual)
                    )

            # Re-running workflow
            mitimNML, statusMITIM, DebugMode = runWithCatchingError(
                mitimNML, DebugMode, checkForActive, automaticProcess=automaticProcess
            )

    if not statusMITIM:
        print(
            '>> Sorry, it failed in all requested trials, returning "inf" flags to optimizer',
            typeMsg="w",
        )

    return mitimNML


def runBlackBox(
    mitimNML,
    stepTransition,
    numStage=0,
    checkForActive=True,
    outtims=[],
    useMMX=False,
    automaticProcess=False,
):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~ Run Equilibrium ~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n==============> Calculating equilibrium <==============\n")

    if mitimNML.EquilibriumType is not None:
        EQmodule.runEquilibrium(
            mitimNML, stepTransition, automaticProcess=automaticProcess
        )
    else:
        print(">> Equilibrium will not be calculated")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~ Run Pedestal ~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n==============> Calculating pedestal <==============\n")

    if mitimNML.TimeImposedPedestal is not None:
        TransitionTime, FinTransition = (
            mitimNML.TimeImposedPedestal,
            mitimNML.TimeImposedPedestal + 0.1,
        )

    else:
        # If first interpretive phase
        if numStage == 0:
            TransitionTime = mitimNML.BaselineTime + mitimNML.startRamp
            FinTransition = mitimNML.BaselineTime + mitimNML.endRamp
        else:
            # If thrid phase
            if numStage == 2 and mitimNML.DoubleInInterpretive:
                # if interpretrive, do not update pedestal (use from previous UFILE)
                TransitionTime, FinTransition = 1e3, 1e3
            # If second phase (or sunsequent predictive ones)
            else:
                TransitionTime = (
                    mitimNML.BaselineTime + mitimNML.PredictionOffset - 0.05
                )
                FinTransition = mitimNML.BaselineTime + mitimNML.PredictionOffset

    if mitimNML.PedestalType is not None:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~ Single pedestal evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if mitimNML.advancedEQ < 2:
            print(
                "\t- Unique pedestal calculation requested, start transition at {0:.3f} and fully formed pedestal at t={1:.3f}s".format(
                    TransitionTime, FinTransition
                )
            )
            MITIMparams = copy.deepcopy(mitimNML.MITIMparams)
            width = PEDmodule.runPedestal(
                mitimNML, MITIMparams, TransitionTime, FinTransition
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~ Advanced (sweeping) pedestal evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif mitimNML.advancedEQ == 2:
            print("\t- Sweeped-pedestal calculation requested")
            MITIMparams = copy.deepcopy(mitimNML.MITIMparams)
            PEDmodule.implementSweepingPedestal(
                mitimNML, MITIMparams, TransitionTime, FinTransition
            )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ No pedestal evaluation, use whatever is in the UFILE or the profile restarted from previous evaluation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        print(
            ">> Pedestal will not be calculated, imposing B.C. at rhoN={0}".format(
                mitimNML.PedestalBC
            )
        )
        PEDmodule.insertBCs(
            mitimNML.namelistPath, [1 - mitimNML.PedestalBC, 1 - mitimNML.PedestalBC]
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~ Run Additional Commands ~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~ Profile Smoothing in UFILES

    SmoothProfiles = mitimNML.SmoothInitialProfiles[numStage]

    # I can have the option of smoothing kinetic profile gradients and enable only density evolution
    if SmoothProfiles:
        pedPos = (1 - width) - 0.05
        checkVariation = numStage > 1
        print(
            f">> Smoothing kinetic profiles (NEL, TEL, TIO) from rho = {pedPos} to 0.0"
        )
        for ity in ["NEL", "TEL", "TIO"]:
            smoothUFILE_variable(
                f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.{ity}",
                pedPos=pedPos,
                checkVariation=checkVariation,
            )

    # Build offline density UFILE if that option is enabled
    if mitimNML.OfflineDensity and mitimNML.SpecialDensityUFILE is None:
        print(">> Building estimation for density profile")

        rho, n20 = PLASMAtools.estimateDensityProfile(mitimNML)

        fileUF = f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.NEL"

        UF = UFILEStools.UFILEtransp()
        UF.readUFILE(fileUF)

        UF.Variables["X"], UF.Variables["Y"], UF.Variables["Z"] = (
            rho,
            [0.1, 100.0],
            n20 * 1e20 * 1e-6,
        )  # needs to be in cm^-3
        UF.repeatProfile()
        UF.writeUFILE(fileUF)

    # ~~~~~~~~~~~ Complete functional forms

    if mitimNML.functionalForms is not None:
        print(">> Forcing profiles to take functional forms, as given in namelist")

        functionalForms(
            mitimNML,
            "NEL",
            mitimNML.functionalForms[0],
            mitimNML.functionalForms[1] * 1e20 * 1e-6,
            mitimNML.functionalForms[2],
        )
        mitimNML.TRANSPnamelist["nriner"] = -4
        functionalForms(
            mitimNML,
            "TEL",
            mitimNML.functionalForms[3],
            mitimNML.functionalForms[4] * 1e3,
            mitimNML.functionalForms[5],
        )
        mitimNML.TRANSPnamelist["nriter"] = -4
        functionalForms(
            mitimNML,
            "TIO",
            mitimNML.functionalForms[3],
            mitimNML.functionalForms[4] * 1e3,
            mitimNML.functionalForms[5],
        )
        mitimNML.TRANSPnamelist["nriti2"] = -4

        mitimNML.predictQuantities = []

    # Add MMX
    if useMMX:
        EQmodule.addMMX(
            mitimNML.mmx_loc,
            f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.MMX",
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~ Define Convergence Criterion ~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n==============> Defining convergence metrics <==============\n")

    mitimNML.defineConvergence(numStage=numStage)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~ Run TRANSP ~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n==============> Running TRANSP <==============\n")

    if numStage == 0:
        phasetxt = "interpretive (1st)"
    elif numStage == 1:
        phasetxt = "predictive (2nd)"
    elif numStage == 2:
        phasetxt = "predictive (3rd)"
    else:
        phasetxt = "unknown"

    HasItFailed = TRANSPmodule.runTRANSP(
        mitimNML,
        stepTransition,
        checkForActive=checkForActive,
        outtims=outtims,
        phasetxt=phasetxt,
        automaticProcess=automaticProcess,
    )

    return HasItFailed


def smoothUFILE_variable(fileUF, pedPos=0.9, debug=False, checkVariation=False):
    # Read current UFILE
    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(fileUF)
    var_orig = UF.Variables["Z"][:, 0]

    x, var = UF.Variables["X"], UF.Variables["Z"][:, 0]

    # Smooth
    numCPs = 30

    offs = 0
    for i in range(1):
        xCPs = np.linspace(0.0, pedPos - 0.01 - offs, numCPs)
        aLy_coarse, deparametrizer, _, _ = PARAMtools.performCurveRegression(
            x, var, torch.from_numpy(xCPs), preSmoothing=True, PreventNegative=True
        )
        x, var = deparametrizer(aLy_coarse[:, 0], aLy_coarse[:, 1])
        var = var[0, :]
        offs += 0.002

        x, var = x.cpu().numpy(), var.cpu().numpy()

    # Checks

    onaxis_var = np.abs(var[0] - var_orig[0]) / var_orig[0]
    print(f"\t- Profile smoothed. On-axis value changed by {onaxis_var * 100.0:.2f}%")

    if (onaxis_var > 0.15 and checkVariation) or debug:
        if onaxis_var > 0.15 and checkVariation:
            print(
                "\t- Smoothed profile differs by more than 15% from original, are you sure? (close window)",
                typeMsg="q",
            )
        fig, ax = plt.subplots()
        ax.plot(x, var)
        ax.plot(x, var_orig)
        plt.show()

    if not debug:
        # Write
        UF.Variables["X"], UF.Variables["Y"], UF.Variables["Z"] = x, [0.1, 100.0], var
        UF.repeatProfile()
        UF.writeUFILE(fileUF)

    return var


def functionalForms(mitimNML, varName, peaking, average, lambda_correction):
    if lambda_correction is not None:
        y, x = MATHtools.profileMARS_PRF(peaking, average, lambda_correction)
    else:
        y, x = MATHtools.profileMARS(peaking, average)

    fileUF = f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.{varName}"
    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(fileUF)
    UF.Variables["X"], UF.Variables["Y"], UF.Variables["Z"] = x, [0.1, 100.0], y
    UF.repeatProfile()
    UF.writeUFILE(fileUF)


def launchHFModel(mitimNML, DebugMode, checkForActive=True, automaticProcess=False):
    if automaticProcess:
        IOtools.randomWait(mitimNML.dakNumUnit)

    # ------------------------------------------------------------------------------
    # If this starts from a given CDF as interpretive
    # ------------------------------------------------------------------------------
    if mitimNML.completedFirst is not None and DebugMode == 0:
        print(">> This run starts from a specific CDF file as its interpretive phase")
        DebugMode = 1
        if os.path.exists(mitimNML.FolderTRANSP):
            os.system(f"mv {mitimNML.FolderTRANSP} {mitimNML.FolderTRANSP[:-1]}_prev")
        os.system(f"mkdir {mitimNML.FolderTRANSP}")
        os.system(f"cp {mitimNML.completedFirst} {mitimNML.FolderTRANSP}")

    # ------------------------------------------------------------------------------

    onlySecond = False  # This is a parameter for the case where I want double predictive, but only the second one now

    if DebugMode == 0:
        booleanCalculate = True
    elif DebugMode == 1 or DebugMode == 3:
        booleanCalculate, False_HasItFailed, False_separatePredictive, onlySecond = (
            False,
            False,
            True,
            DebugMode == 3,
        )
    elif DebugMode == 2:
        booleanCalculate, False_HasItFailed, False_separatePredictive = (
            False,
            False,
            False,
        )

    # ------------------------------------------------------------------------------
    # Grab reactor parameters (from here, no distinction bw DVs and unchanged parameters;
    #                           dictDVs and BASELINE_X nml parameters file shouldn't be used anymore)
    # ------------------------------------------------------------------------------

    (
        mitimNML.MITIMparams,
        mitimNML.MITIMparams_baseline,
    ) = IMparam.getParametersDictionary(mitimNML, onlyReadFinal=DebugMode == 2)

    # ------------------------------------------------------------------------------
    # Handling of UFiles that are not created during MITIM run
    #   - Te, Ti, ne, V and q variables require initialization (from a previous run)
    #   - R*Bt (RBZ) and Ip (CUR) require initialization from a previous run to ramp up to new values (if that option is enabled)
    # ------------------------------------------------------------------------------

    mitimNML.UFsToChange = {
        "TEL": ["TE", "ter"],
        "TIO": ["TI", "ti2"],
        "NEL": ["NE", "ner"],
        "QPR": ["Q", "qpr"],
        "VSF": ["VSURC", "vsf"],
        "RBZ": ["BZXR", "rbz"],
        "CUR": ["PCUR", "cur"],
    }

    # ------------------------------------------------------------------------------
    # Handling of UFiles that my require initialization
    #   - MRY,LIM Ufiles if I'm not creating an equilibrium
    # ------------------------------------------------------------------------------

    mitimNML.UFsToCopy = []
    if mitimNML.EquilibriumType is None:
        mitimNML.UFsToCopy.append("MRY")

    useMMX = mitimNML.machine == "CMOD" and mitimNML.mmx_loc is not None

    # ------------------------------------------------------------------------------
    # INTERPRETIVE PHASE
    # ------------------------------------------------------------------------------

    if booleanCalculate:
        HasItFailed, separatePredictive = runInterpretiveTRANSP(
            mitimNML,
            useMMX=useMMX,
            checkForActive=checkForActive,
            automaticProcess=automaticProcess,
        )

    else:
        HasItFailed, separatePredictive = False_HasItFailed, False_separatePredictive

        if separatePredictive and not HasItFailed and not onlySecond:
            print(
                ">> Not calculating interpretive phase, using results that were in folder",
                typeMsg="f",
            )
            evaluateWithNoRun(mitimNML)

    # ------------------------------------------------------------------------------
    # PREDICTIVE PHASES (W/ VV and WO/ Ramp)
    # ------------------------------------------------------------------------------

    if separatePredictive and not HasItFailed:
        # Run 1st predictive phase

        if not onlySecond:
            print(">> Proceeding to set up 1st predictive phase")

            HasItFailed = runPredictiveTRANSP(
                mitimNML,
                numStage=1,
                finalTimeRunOffset=mitimNML.ftimeOffsets[1],
                extraLabelPreviousRun="_interpretive/",
                checkForActive=checkForActive,
                useMMX=useMMX,
                automaticProcess=automaticProcess,
            )

    # Run 2nd predictive phase

    if separatePredictive and mitimNML.RunDoublePredictive and not HasItFailed:
        if not onlySecond:
            mitimNML.Reactor = CDFtools.CDFreactor(mitimNML.LocationNewCDF)
        else:
            print(
                ">> Not calculating 1st predictive phase, using results that were in folder",
                typeMsg="f",
            )
            evaluateWithNoRun(mitimNML)

        print(">> Running second predictive phase with updated parameters")

        HasItFailed = runPredictiveTRANSP(
            mitimNML,
            numStage=2,
            finalTimeRunOffset=mitimNML.ftimeOffsets[2],
            extraLabelPreviousRun="_predictive1/",
            checkForActive=checkForActive,
            runInterpretive=mitimNML.DoubleInInterpretive,
            useMMX=useMMX,
            automaticProcess=automaticProcess,
        )

    # ------------------------------------------------------------------------------
    # Evaluate Metrics
    # ------------------------------------------------------------------------------

    if not HasItFailed:
        CDFSrun = IOtools.findFileByExtension(mitimNML.FolderTRANSP, ".CDF")

        if CDFSrun != mitimNML.nameRunTot:
            print(">> Previous TRANSP had different name, renaming")
            IOtools.renameCommand(
                CDFSrun, mitimNML.nameRunTot, folder=mitimNML.FolderTRANSP
            )

        mitimNML.Reactor = evaluateFinalReactor(
            mitimNML.FolderOutputs, mitimNML.LocationNewCDF
        )

        # Eliminate previous CDF
        os.system(f"rm -r {mitimNML.FolderTRANSP}/*CDF_prev")

        statusMITIM = True

    else:
        statusMITIM = False

    return statusMITIM


def evaluateFinalReactor(folderOutputs, LocationNewCDF, gatherAllData=True):
    # ------------------------------------------------------------------------------------
    # Allow the possibility for parameters that require running TGLF or similar
    # ------------------------------------------------------------------------------------

    EvaluateExtraAnalysis = []
    if len(EvaluateExtraAnalysis) == 0:
        EvaluateExtraAnalysis = None

    # ------------------------------------------------------------------------------------
    # Read extra features in this last CDFreactor call
    # ------------------------------------------------------------------------------------

    if gatherAllData:
        readFBM = readTGLF = readTORIC = True
    else:
        readFBM = readTGLF = readTORIC = False

    # ------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------

    cdf = CDFtools.CDFreactor(
        LocationNewCDF,
        EvaluateExtraAnalysis=EvaluateExtraAnalysis,
        readFBM=readFBM,
        readTGLF=readTGLF,
        readTORIC=readTORIC,
    )

    cdf.writeOutput(time=-0.06, avTime=0.05)

    # ------------------------------------------------------------------------------------
    # Write run info
    # ------------------------------------------------------------------------------------

    cdf.writeResults_TXT(folderOutputs + "/infoRun.dat")
    cdf.writeResults_XLSX(folderOutputs + "/infoRun.xlsx")

    return cdf


def createUFILES(
    MITIMparams,
    FolderTRANSP,
    nameBaseShot,
    impurityMode=0,
    imp_profs=[],
    fast_anom=None,
    min_dens=None,
):
    # ~~~~~~~~~~~~~~~ He4 transport coefficients

    if "xhe4" in MITIMparams:
        xHe4, DHe4, VHe4 = MITIMparams["xHE4"], MITIMparams["DHE4"], MITIMparams["VHE4"]

        print(
            ">> Creating He4 ash particle diffusivity UFile (@rho=1 -> D = {0:.1f})".format(
                DHe4[-1]
            )
        )
        UFILEStools.quickUFILE(
            xHe4,
            DHe4 * 1e4,
            f"{FolderTRANSP}/PRF{nameBaseShot}.DHE4",
            typeuf="df4",
        )

        print(
            f">> Creating He4 ash particle pinch UFile (@rho=1 -> V = {VHe4[-1]:.1f})"
        )
        UFILEStools.quickUFILE(
            xHe4,
            VHe4 * 1e2,
            f"{FolderTRANSP}/PRF{nameBaseShot}.VHE4",
            typeuf="vc4",
        )

    else:
        print(">> He4 transport coefficients could not be found, not building UFILES")

    # ~~~~~~~~~~~~~~~ ECH

    if "polech" in MITIMparams:
        xPolech, Polech = (
            np.linspace(0, len(MITIMparams["Polech"]), len(MITIMparams["Polech"])),
            MITIMparams["Polech"],
        )

        print(">> Creating THE poloidal ECH aiming angle UFile")
        UFILEStools.quickUFILE(
            xPolech,
            Polech,
            f"{FolderTRANSP}/PRF{nameBaseShot}.THE",
            typeuf="the",
        )

    else:
        print(">> ECH  poloidal aiming angle could not be found, not building UFILE")

    if "torech" in MITIMparams:
        xPolech, Polech = (
            np.linspace(0, len(MITIMparams["Torech"]), len(MITIMparams["Torech"])),
            MITIMparams["Torech"],
        )

        print(">> Creating PHI toroidal ECH aiming angle UFile")
        UFILEStools.quickUFILE(
            xPolech,
            Polech,
            f"{FolderTRANSP}/PRF{nameBaseShot}.PHI",
            typeuf="phi",
        )

    else:
        print(">> ECH  toroidal aiming angle could not be found, not building UFILE")

    # ~~~~~~~~~~~~~~~ Zeff and impurities

    if impurityMode != 2:
        print(">> Zeff as a UFILE selected")
        xZeff, Zeff = np.linspace(0, 1, 10), np.ones(10) * MITIMparams["Zeff"]

        print("\t- Creating Zeff profile UFILE")
        UFILEStools.quickUFILE(
            xZeff,
            Zeff,
            f"{FolderTRANSP}/PRF{nameBaseShot}.ZF2",
            typeuf="zf2",
        )

    if imp_profs is not None:
        for i in range(len(imp_profs)):
            if impurityMode == 2 or (impurityMode == 3 and i == 0):
                print(f"\t- Creating impurity profile UFile for impurity #{i + 1}")
                createImpurityUFILE(imp_profs[i], FolderTRANSP, nameBaseShot, num=i + 1)

    if fast_anom is not None:
        print("\t- Creating fast ion anomalous transport UFILES")
        createFastAnomalousUFILE(fast_anom, FolderTRANSP, nameBaseShot)

    if min_dens is not None:
        print("\t- Creating minority density UFILE")
        createMinorityUFILE(min_dens, FolderTRANSP, nameBaseShot)

    # ~~~~~~~~~~~~~~~ Gas flow

    if "gasflow" in MITIMparams:
        gasflow = MITIMparams["gasflow"]
    else:
        print(">> Gas flow rate could not be found, using 0")
        gasflow = 0.0

    print(">> Creating gas flow UFile")
    UFILEStools.quickUFILE(
        None,
        gasflow * 1e20,
        f"{FolderTRANSP}/PRF{nameBaseShot}.GFD",
        typeuf="gfd",
    )


def createImpurityUFILE(imp_prof, FolderTRANSP, nameBaseShot, num=1):
    pik = pickle.load(open(imp_prof, "rb"))
    x, nimp = pik["rho"], pik["nimp"] * 1e20
    UFILEStools.quickUFILE(
        x,
        nimp * 1e-6,
        f"{FolderTRANSP}/PRF{nameBaseShot}.NIMP{num}",
        typeuf="sim",
    )


def createFastAnomalousUFILE(fast_anom, FolderTRANSP, nameBaseShot):
    pik = pickle.load(open(fast_anom, "rb"))
    x, D, V = pik["rho"], pik["D"], pik["V"]
    UFILEStools.quickUFILE(
        x, D * 1e4, f"{FolderTRANSP}/PRF{nameBaseShot}.D2F", typeuf="d2f"
    )
    UFILEStools.quickUFILE(
        x, V * 1e2, f"{FolderTRANSP}/PRF{nameBaseShot}.V2F", typeuf="v2f"
    )


def createMinorityUFILE(min_dens, FolderTRANSP, nameBaseShot):
    pik = pickle.load(open(min_dens, "rb"))
    x, n = pik["rho"], pik["n"]
    UFILEStools.quickUFILE(
        x, n * 1e-6, f"{FolderTRANSP}/PRF{nameBaseShot}.NMR", typeuf="nmr"
    )


def createRotation(mitimNML):
    Mach = mitimNML.MITIMparams["Mach"]

    cdf = CDFtools.CDFreactor(mitimNML.useRotationMach)

    Vtor = (
        PLASMAtools.constructVtorFromMach(Mach, cdf.Ti[cdf.ind_saw], cdf.mi / cdf.u)
        * 100.0
    )  # in cm/s

    fileu = "VP2"
    print(
        ">> Warning: Creating Vtor profile UFile ({3}) with Mach = {0} and using vTi from ...{1}, resulting in {2:.1f}km/s on axis".format(
            Mach, mitimNML.useRotationMach[-40:], Vtor[0] * 1e-5, fileu
        ),
        typeMsg="w",
    )

    UFILEStools.quickUFILE(
        x,
        Vtor,
        f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.VP2",
        typeuf=fileu.lower(),
    )

    mitimNML.TRANSPnamelist["NLVPHI"], mitimNML.TRANSPnamelist["NGVTOR"] = "T", "100"
    (
        mitimNML.TRANSPnamelist[f"pre{fileu}"],
        mitimNML.TRANSPnamelist[f"ext{fileu}"],
        mitimNML.TRANSPnamelist[f"nri{fileu}"],
    ) = ('"PRF"', f'"{fileu}"', -5)


def evaluateWithNoRun(mitimNML):
    CDFSrun = IOtools.findFileByExtension(mitimNML.FolderTRANSP, ".CDF")

    if CDFSrun != mitimNML.nameRunTot:
        print(">> Previous TRANSP had different name, renaming")
        IOtools.renameCommand(
            CDFSrun, mitimNML.nameRunTot, folder=mitimNML.FolderTRANSP
        )

    mitimNML.Reactor = CDFtools.CDFreactor(mitimNML.LocationNewCDF)


def runInterpretiveTRANSP(
    mitimNML, useMMX=False, checkForActive=True, automaticProcess=False
):
    mitimNML.TRANSPnamelist_store = copy.deepcopy(mitimNML.TRANSPnamelist)

    separatePredictive = mitimNML.separatePredictive

    # ~~~~~~~~~~~~~
    # Prepare folders
    # ~~~~~~~~~~~~~
    if os.path.exists(mitimNML.FolderTRANSP):
        os.system("mv {0} {0}prev".format(mitimNML.FolderTRANSP[:-1]))
        os.system(f"rm -r {mitimNML.FolderTRANSP[:-1]}prev")

    if os.path.exists(mitimNML.FolderEQ):
        os.system("rm -r " + mitimNML.FolderEQ)

    os.system("mkdir " + mitimNML.FolderTRANSP)
    os.system("mkdir " + mitimNML.FolderEQ)

    # ~~~~~~~~~~~~~
    # Parameters specific for this phase
    # ~~~~~~~~~~~~~

    mitimNML.runPhaseNumber = 0

    # Timings specific for this phase

    stepTransition = True

    mitimNML.TRANSPnamelist["tinit"] = round(mitimNML.AbsoluteTINIT, 3)
    mitimNML.TRANSPnamelist["ftime"] = round(
        mitimNML.ftimeOffsets[0] + mitimNML.TRANSPnamelist["tinit"], 3
    )

    mitimNML.TRANSPnamelist["t_sawtooth_on"] = mitimNML.timeSawtooth

    # Current diffusion

    option = 1  # 1: Enable current diffusion with matching Ip from timeCurrentDiff
    # 2: Options to not matching Ip

    if option == 1:
        mitimNML.TRANSPnamelist["nqmoda(1)"] = 4  # QPR
        mitimNML.TRANSPnamelist["nqmodb(1)"] = 1  # Match Ip

        mitimNML.TRANSPnamelist["tqmoda(1)"] = mitimNML.timeCurrentDiff

        mitimNML.TRANSPnamelist["nqmoda(2)"] = 1  # Current diffusion
        mitimNML.TRANSPnamelist["nqmodb(2)"] = 1  # Match Ip

    elif option == 2:
        mitimNML.TRANSPnamelist["nqmoda(1)"] = 4  # QPR
        mitimNML.TRANSPnamelist["nqmodb(1)"] = 2  # No Match Ip

        mitimNML.TRANSPnamelist["tqmoda(1)"] = (
            mitimNML.BaselineTime + mitimNML.endRamp + 0.005
        )

        mitimNML.TRANSPnamelist["nqmoda(2)"] = 4  # QPR
        mitimNML.TRANSPnamelist["nqmodb(2)"] = 1  # Match Ip

        mitimNML.TRANSPnamelist["tqmoda(2)"] = (
            mitimNML.BaselineTime + mitimNML.endRamp + 0.01
        )

        mitimNML.TRANSPnamelist["nqmoda(3)"] = 1  # Current diffusion
        mitimNML.TRANSPnamelist["nqmodb(3)"] = 1  # Match Ip

    # Specific aspects of this phase
    mitimNML.useAxialTrick = mitimNML.useAxialTrickPhases[0]
    if mitimNML.TransportModels[0] in ["TGLF", "tglf"]:
        mitimNML.TRANSPnamelist["NPTR_PSERVE"] = 1
    else:
        mitimNML.TRANSPnamelist["NPTR_PSERVE"] = 0
    if mitimNML.PlasmaFeatures["ICH"]:
        mitimNML.TRANSPnamelist["NTORIC_PSERVE"] = 1
    else:
        mitimNML.TRANSPnamelist["NTORIC_PSERVE"] = 0

    # ------------------------------------
    #  Produce Namelist
    # ------------------------------------

    IMparam.getBaseNamelists(
        mitimNML.nameRunTot,
        mitimNML.FolderTRANSP,
        PlasmaFeatures=mitimNML.PlasmaFeatures,
        InputFolder=mitimNML.baseNMFolder,
        tok=mitimNML.machine,
        NMLnumber=mitimNML.NMLnumber,
        useMMX=useMMX,
    )

    # ------------------------------------
    # Get parameters from previous simulation, for mapping, pedestal calculations, etc
    # ------------------------------------

    getStepParams(
        mitimNML, cdffile=mitimNML.LocationOriginalCDF, gfile=mitimNML.gfile_loc
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~ Prepare UFILES
    # LIM, RFP will be created afterwards in the MITIM framework
    # TEL,TIO,CUR,NEL,QPR,RBZ,VSF are copied from previous run (and time-stepped accordingly)
    # Rest are created
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~ Create He4 transport, Zeff and ECH aimings ~~~~~~~~~~~~~~~~~~
    createUFILES(
        mitimNML.MITIMparams,
        mitimNML.FolderTRANSP,
        mitimNML.nameBaseShot,
        impurityMode=mitimNML.impurityMode,
        imp_profs=mitimNML.imp_profs,
        fast_anom=mitimNML.fast_anom,
        min_dens=mitimNML.min_dens,
    )

    # ~~~~~~~~~~~~ Create rotation profile ~~~~~~~~~~~~~~~~~~
    if mitimNML.useRotationMach is not None:
        createRotation(mitimNML)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~ Create [Te, Ti, ne, q] profiles and V, RBt and Ip ~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~ OPTION 1: If NO excel file has been provided, initialize from previous CDF (which must be provided), and ramp up to new values if selected

    mitimNML.currentCDF = mitimNML.LocationOriginalCDF

    if mitimNML.excelProfile is None:
        print(">> CDF initialization requested")

        print(
            ">> Initial Te, Ti, ne, q, V, RBt and Ip are being read from CDF file ...{0}".format(
                mitimNML.LocationOriginalCDF[-40:]
            )
        )

        timeUpdate = mitimNML.restartTime[0]
        _ = restartUFILES(
            mitimNML,
            cdffile=mitimNML.currentCDF,
            timeUpdate=timeUpdate,
            StandaloneTransportFolder=None,
            SpecialDensityUFILE=mitimNML.SpecialDensityUFILE,
            Initialq_profile=mitimNML.Initialq_gfile,
        )

    # ~~~ OPTION 2: If excel file has been provided, take [Te, Ti, ne, q], guess V and do not ramp RBt nor Ip

    else:
        print(">> NON-CDF initialization requested")

        print(
            "\t- Initial Te, Ti, ne, and q are being read from Excel file ...{0}".format(
                mitimNML.excelProfile[-40:]
            )
        )
        rho, Te, Ti, q, ne = IOtools.getProfiles_ExcelColumns(
            mitimNML.excelProfile, fromColumn=0, fromRow=4
        )

        print(
            "\t- Initial Vsurf equal to 0.0V and no stepping in RBZ, CUR (subsequently nor MRY either)"
        )
        V = 0.0
        stepTransition = False

        UFILEStools.initializeUFILES_MinimalTRANSP(
            rho,
            Te * 1e3,
            Ti * 1e3,
            ne * 1e14,
            q,
            V,
            location=mitimNML.FolderTRANSP,
            name=mitimNML.nameBaseShot,
        )

        # mitimNML.TRANSPnamelist['nriter'] = -7
        # mitimNML.TRANSPnamelist['nriti2'] = -7
        # mitimNML.TRANSPnamelist['nriner'] = -7
        # mitimNML.TRANSPnamelist['nriqpr'] = -7

    # ~~~~~~~~~~~~~
    # Special q-profile given in a pickle
    # ~~~~~~~~~~~~~

    if mitimNML.SpecialQprofile is not None:
        print("\t- Special q-profile requested")

        aux = pickle.load(open(IOtools.expandPath(mitimNML.SpecialQprofile), "rb"))
        t = aux["t"]
        x = aux["rhopol"]
        q = aux["q"]
        it = np.argmin(np.abs(t - mitimNML.SpecialQprofile_time))

        UFILEStools.quickUFILE(
            x[it],
            q[it],
            f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.QPR",
            typeuf="qpr",
        )
        mitimNML.TRANSPnamelist["nriqpr"] = -7

    # ~~~~~~~~~~~~~
    # Run Framework
    # ~~~~~~~~~~~~~

    mitimNML.oversizedMachine = stepTransition

    HasItFailed = runBlackBox(
        mitimNML,
        stepTransition,
        numStage=0,
        checkForActive=checkForActive,
        useMMX=useMMX,
        automaticProcess=automaticProcess,
    )

    # ~~~~~~~~~~~~~
    # Results
    # ~~~~~~~~~~~~~

    if not HasItFailed:
        mitimNML.Reactor = CDFtools.CDFreactor(mitimNML.LocationNewCDF)
        mitimNML.Reactor.writeResults_TXT(mitimNML.FolderTRANSP + "/infoRun.dat")

        # Eliminate previous CDF
        os.system(f"rm -r {mitimNML.FolderTRANSP}/*CDF_prev")

    return HasItFailed, separatePredictive


def runPredictiveTRANSP(
    mitimNML,
    finalTimeRunOffset=1e4,
    extraLabelPreviousRun="/",
    checkForActive=True,
    numStage=1,
    runInterpretive=False,
    useMMX=False,
    automaticProcess=False,
):
    try:
        mitimNML.TRANSPnamelist = mitimNML.TRANSPnamelist_store
    except:
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepare folders
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print(">> Organizing folders properly")

    previousFolderTRANSP = mitimNML.FolderTRANSP[:-1] + extraLabelPreviousRun
    previousFolderEQ = mitimNML.FolderEQ[:-1] + extraLabelPreviousRun

    # ~~ Remove previous folders

    if os.path.exists(previousFolderTRANSP):
        os.system("mv {0} {0}rm".format(previousFolderTRANSP[:-1]))
        os.system(f"rm -r {previousFolderTRANSP[:-1]}rm")

    if os.path.exists(previousFolderEQ):
        os.system("rm -r " + previousFolderEQ)

    # ~~ Save previous run results in new folders

    os.system("mv " + mitimNML.FolderTRANSP + " " + previousFolderTRANSP)

    if os.path.exists(mitimNML.FolderEQ):
        os.system("mv " + mitimNML.FolderEQ + " " + previousFolderEQ)
    else:
        print("\t- EQ folder from previous stage not found, not important", typeMsg="w")

    # ~~ Create new folders with the same name to work with

    os.system(f"mkdir {mitimNML.FolderTRANSP} {mitimNML.FolderEQ}")

    mitimNML.currentCDF = previousFolderTRANSP + mitimNML.nameRunTot + ".CDF"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run Standalone Transport Analysis (optional)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # runTransportAnalysis(mitimNML,mitimNML.currentCDF)
    StandaloneTransportFolder = None

    # ------------------------------------
    #  Produce Namelist
    # ------------------------------------

    IMparam.getBaseNamelists(
        mitimNML.nameRunTot,
        mitimNML.FolderTRANSP,
        PlasmaFeatures=mitimNML.PlasmaFeatures,
        InputFolder=mitimNML.baseNMFolder,
        tok=mitimNML.machine,
        NMLnumber=mitimNML.NMLnumber,
        useMMX=useMMX,
        TGLFsettings=mitimNML.TGLFsettings,
    )

    # ------------------------------------
    # Get parameters from previous simulation, for mapping, pedestal calculations, etc
    # ------------------------------------

    getStepParams(mitimNML, cdffile=mitimNML.currentCDF)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #           Prepare UFILES
    # LIM, RFP will be created afterwards in the MITIM framework
    # TEL,TIO,CUR,NEL,QPR,RBZ,VSF are copied from previous run (and time-stepped accordingly)
    # DHE4 and VH4 are created
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Create He4 transport coefficients
    createUFILES(
        mitimNML.MITIMparams,
        mitimNML.FolderTRANSP,
        mitimNML.nameBaseShot,
        impurityMode=mitimNML.impurityMode,
        imp_profs=mitimNML.imp_profs,
        fast_anom=mitimNML.fast_anom,
        min_dens=mitimNML.min_dens,
    )

    if mitimNML.useRotationMach is not None:
        createRotation(mitimNML)

    # Copy rest
    timeUpdate = mitimNML.restartTime[numStage]

    timeExtracted = restartUFILES(
        mitimNML,
        cdffile=mitimNML.currentCDF,
        timeUpdate=timeUpdate,
        StandaloneTransportFolder=StandaloneTransportFolder,
        SpecialDensityUFILE=None,
        Initialq_profile=False,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameters specific for this phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mitimNML.runPhaseNumber = 1
    mitimNML.oversizedMachine = (
        False  # If true, machine limiters and VV very big to avoid intersection
    )
    mitimNML.useAxialTrick = mitimNML.useAxialTrickPhases[1]

    # ~~~~~~~~~~ Set time basis
    mitimNML.BaselineTime = timeExtracted
    mitimNML.TRANSPnamelist["tinit"] = mitimNML.BaselineTime
    mitimNML.TRANSPnamelist["ftime"] = (
        finalTimeRunOffset + mitimNML.TRANSPnamelist["tinit"]
    )

    # ~~~~~~~~~~ Offset for ICRF
    mitimNML.timePower = mitimNML.BaselineTime + 0.005

    # ~~~~~~~~~~ Define special options for 2nd predictive in interpretive mode
    if runInterpretive:
        outtims = defineInterpretiveFeatures(mitimNML, mitimNML.currentCDF, timeUpdate)
    # ~~~~~~~~~~ Options for standard predictive mode
    else:
        outtims = (
            []
        )  # round(i/1000.0+mitimNML.BaselineTime,3) for i in mitimNML.out_times]
        mitimNML.TRANSPnamelist["t_sawtooth_on"] = round(
            mitimNML.BaselineTime + mitimNML.SawtoothOffset_2ndPhase, 3
        )
        mitimNML.TRANSPnamelist["tqmoda(1)"] = round(
            mitimNML.BaselineTime + mitimNML.CurrentDiffOffset_2ndPhase, 3
        )
        mitimNML.PredictionOffset = mitimNML.PredOffset_2ndPhase

    # Specific aspects of this phase
    if mitimNML.TransportModels[1] in ["TGLF", "tglf"]:
        mitimNML.TRANSPnamelist["NPTR_PSERVE"] = 1
    else:
        mitimNML.TRANSPnamelist["NPTR_PSERVE"] = 0
    if mitimNML.PlasmaFeatures["ICH"]:
        mitimNML.TRANSPnamelist["NTORIC_PSERVE"] = 1
    else:
        mitimNML.TRANSPnamelist["NTORIC_PSERVE"] = 0

    # ~~~~~~~~~~~~~
    # Run Framework
    # ~~~~~~~~~~~~~
    stepTransition = False
    HasItFailed = runBlackBox(
        mitimNML,
        stepTransition,
        numStage=numStage,
        checkForActive=checkForActive,
        outtims=outtims,
        useMMX=useMMX,
        automaticProcess=automaticProcess,
    )

    return HasItFailed


def defineInterpretiveFeatures(mitimNML, cdffile, timeUpdate, calculateRadiation=True):
    startTGLFbeforeEnd = 0.004
    extractOUTbeforeEnd = (
        0.005  # Remember that an average will be applied 50ms before this
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get total run length from previous run
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Minimum_Length = mitimNML.DoubleInInterpretiveMinLength  # 1.0
    Maximum_Length = 3.0

    # Total run length for an interpretive case is determined by the TORIC and NUBEAM time scales (slowing-down mostly)
    c = mitimNML.Reactor
    taue = c.taueTot[c.ind_saw] * 1e-3
    tauSD = c.tauSD_He4_avol[c.ind_saw] * 1e-3
    tau = np.max([taue, tauSD])

    totalRunLength = np.min([np.max([Minimum_Length, tau]), Maximum_Length])

    print(
        ">> Running in interpretive mode for {0:.0f}ms, running TGLF/OUTTIMES {1:.0f}ms/{2:.0f}ms before end".format(
            totalRunLength * 1000, startTGLFbeforeEnd * 1000, extractOUTbeforeEnd * 1000
        )
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run only for 400ms (e.g. seen enough for SPARC V1E). Only thing that needs to settle
    #   is the deposited power (NUBEAM slow down and TORIC) because kinetic profiles are fixed
    #   and ash is taken interpretively
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mitimNML.TRANSPnamelist["ftime"] = mitimNML.TRANSPnamelist["tinit"] + totalRunLength

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TGLF: Run it briefly to get fluctuation data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mitimNML.TRANSPnamelist["nlgrowth_tglf"] = "T"
    mitimNML.PredictionOffset = (
        mitimNML.TRANSPnamelist["ftime"] - mitimNML.BaselineTime
    ) - startTGLFbeforeEnd
    mitimNML.predictQuantities = ["te"]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # OUTTIMES (Times to write output files from FAST IONS, NUBEAM, TORIC)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    outtims = [mitimNML.TRANSPnamelist["ftime"] - extractOUTbeforeEnd]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Increase grid for pedestal info
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # mitimNML.TRANSPnamelist['nzones']    = 300    # Originally 100
    # mitimNML.TRANSPnamelist['nzone_nb']  = 50     # Originally 20
    # mitimNML.TRANSPnamelist['nzone_fp']  = 50     # Originally 20
    # mitimNML.TRANSPnamelist['nzone_fb']  = 25     # Originally 10

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Helium ash density (take NHE4 from previous run. This is because of a weird "bug" that
    #   makes He4 explosively accumulate in the core in the interpretive runs I have tried
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    try:
        ufileN = f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.NI4"
        _, _ = UFILEStools.updateUFILEfromCDF(
            "NHE4", ufileN, cdffile, timeUpdate, [0.0, 100.0], scratch="ni4"
        )
        mitimNML.TRANSPnamelist["ndefine(3)"] = 2
        mitimNML.TRANSPnamelist["preNI4"] = "'PRF'"
        mitimNML.TRANSPnamelist["extNI4"] = "'NI4'"
        mitimNML.TRANSPnamelist["nriNI4"] = -5

    except:
        print(">> Could not write NHE4 density, probably because this is a DD plasma")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Radiation Options
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Option to just use the radiation profile from the previous interpretive run
    if not calculateRadiation:
        ufileN = f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.BOL"
        _, _ = UFILEStools.updateUFILEfromCDF(
            "PRAD", ufileN, cdffile, timeUpdate, [0.0, 100.0], scratch="bol"
        )
        mitimNML.TRANSPnamelist["preBOL"] = "'PRF'"
        mitimNML.TRANSPnamelist["extBOL"] = "'BOL'"
        mitimNML.TRANSPnamelist["nriBOL"] = -5
        mitimNML.TRANSPnamelist["nprad"] = 0

    # Option to keep calculating radiation (advantage: If it is sawtoothing, weird discontinuities may have happen previously)
    else:
        mitimNML.TRANSPnamelist["nprad"] = 2

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Current diffusion: Do not evaluate current diffusion, just use q-profile from previous run
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mitimNML.TRANSPnamelist["t_sawtooth_on"] = 100.0
    mitimNML.TRANSPnamelist["tqmoda(1)"], mitimNML.TRANSPnamelist["tqmoda(2)"] = (
        100.0,
        200.0,
    )
    mitimNML.TRANSPnamelist["nqmoda(1)"] = 4  # (equivalent to nlqdata=T)
    mitimNML.TRANSPnamelist[
        "nqmodb(1)"
    ] = 2  # Do not try to match Ip, to avoid changed to q-profile (although Ip may differ)
    mitimNML.TRANSPnamelist["nmodpoh"] = 2  # Use eta for evaluation

    return outtims


def restartUFILES(
    mitimNML,
    cdffile=None,
    timeUpdate=None,
    timeWrite=[0.0, 100.0],
    StandaloneTransportFolder=None,
    SpecialDensityUFILE=None,
    Initialq_profile=False,
):
    # timeUpdate is the time in the cdffile to extract from. timeWrite is the timea to write new UFILE (repeat profile)

    print("\n==============> Generating UFILES <==============\n")

    # ------------------------------------
    #  Copy MRY UF from baseline simulation
    # ------------------------------------

    if len(mitimNML.UFsToCopy) > 0:
        for file in mitimNML.UFsToCopy:
            os.system(f"cp {mitimNML.baseFolder}/*.{file} {mitimNML.FolderTRANSP}/.")
        IOtools.renameCommand(
            mitimNML.UFnumber, mitimNML.nameBaseShot, folder=mitimNML.FolderTRANSP
        )

    # ------------------------------------
    # Update UFILES from a given point in time in the CDF
    # ------------------------------------

    if timeUpdate is not None:
        # MRY may be varying in time, let's just extract the timing
        extrUF = ["MRY"]
        for i in extrUF:
            if extrUF in mitimNML.UFsToCopy:
                print(
                    ">> Reducing timing of {0} ufile, extracted at t = {1:.3f}s".format(
                        extrUF, timeUpdate
                    )
                )
                UFILEStools.reduceTimeUFILE(
                    f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.{extrUF}",
                    timeUpdate,
                )

        for iUF in mitimNML.UFsToChange:
            ufileN = f"{mitimNML.FolderTRANSP}/PRF{mitimNML.nameBaseShot}.{iUF}"

            # ~~~~~~~~
            # Option of special density profile (from specific CDF file)
            # ~~~~~~~~

            if iUF in ["NEL"] and SpecialDensityUFILE is not None:
                if "CDF" in SpecialDensityUFILE:
                    print(
                        ">> Restarting from a non-standard density CDF, {0}".format(
                            SpecialDensityUFILE
                        )
                    )
                    cdffileToGet = SpecialDensityUFILE
                    timeUpdateToGet = -1

                    _, _ = UFILEStools.updateUFILEfromCDF(
                        mitimNML.UFsToChange[iUF][0],
                        ufileN,
                        cdffileToGet,
                        timeUpdateToGet,
                        timeWrite,
                        scratch=mitimNML.UFsToChange[iUF][1],
                    )
                else:
                    print(
                        ">> Restarting NE from a standard density UFILE, {0}".format(
                            SpecialDensityUFILE
                        )
                    )
                    os.system(f"cp {SpecialDensityUFILE} {ufileN}")

            # ~~~~~~~~
            # Option of special q profile (from gfile)
            # ~~~~~~~~

            elif (
                iUF in ["QPR"]
                and Initialq_profile
                and mitimNML.EquilibriumType == "gfile"
            ):
                print(">> Restarting Q from gfile")

                g = GEQtools.MITIMgeqdsk(mitimNML.gfile_loc)

                psi = g.Ginfo["PSI_NORM"]
                q = g.Ginfo["QPSI"]

                # Read Ufile
                UF = UFILEStools.UFILEtransp()
                UF.readUFILE(ufileN)

                # Modify variables
                UF.Variables["X"], UF.Variables["Z"], UF.Variables["Y"] = (
                    psi,
                    q,
                    timeWrite,
                )
                UF.repeatProfile()

                #  Write new UFile
                UF.writeUFILE(ufileN)

                # Make sure that we read poloidal flux
                mitimNML.TRANSPnamelist["nriqpr"] = -6

            # ~~~~~~~~
            # Option of special q profile (from ufile)
            # ~~~~~~~~

            elif iUF in ["QPR"] and mitimNML.SpecialQUFILE is not None:
                print(
                    ">> Restarting Q from a standard q-profile UFILE, {0}".format(
                        mitimNML.SpecialQUFILE
                    )
                )
                os.system(f"cp {mitimNML.SpecialQUFILE} {ufileN}")

            # ~~~~~~~~
            # Standard restarting for the rest of variables
            # ~~~~~~~~

            else:
                cdffileToGet, timeUpdateToGet = cdffile, timeUpdate
                timeExtracted, ind = UFILEStools.updateUFILEfromCDF(
                    mitimNML.UFsToChange[iUF][0],
                    ufileN,
                    cdffileToGet,
                    timeUpdateToGet,
                    timeWrite,
                    scratch=mitimNML.UFsToChange[iUF][1],
                )

                mitimNML.TRANSPnamelist["nriqpr"] = -5

    # ------------------------------------
    # Just copy UFILES from a given folder
    # ------------------------------------

    else:
        for iUF in mitimNML.UFsToChange:
            os.system(f"cp {mitimNML.baseFolder}/*.{file} {mitimNML.FolderTRANSP}/.")
        IOtools.renameCommand(
            mitimNML.UFnumber, mitimNML.nameBaseShot, folder=mitimNML.FolderTRANSP
        )

    # ------------------------------------
    # Grab geometry parameters to later build MRY for ramp
    # ------------------------------------

    print(f">> Restarting Geometry (R,eps,k,d,z,z0) from t={timeExtracted:.2f}s")

    if timeUpdate is not None:
        cdf = mitimNML.Reactor
    else:
        cdf = CDFtools.CDFreactor(cdffile)

    it = ind
    mitimNML.GeoBase_rmajor = round(cdf.Rmajor[it], 2)
    mitimNML.GeoBase_epsilon = round(cdf.epsilon[it], 3)
    mitimNML.GeoBase_kappa = round(cdf.kappa[it], 2)
    mitimNML.GeoBase_delta = round(cdf.delta[it], 2)
    mitimNML.GeoBase_zeta = round(cdf.zeta[it], 2)
    mitimNML.GeoBase_z0 = round(cdf.Ymag[it], 2)

    mitimNML.Boundary_r, mitimNML.Boundary_z = cdf.getLCFS_time(time=cdf.t[it])

    # All time extracted should be the same, no problem with the overwritting
    return timeExtracted


def getStepParams(mitimNML, cdffile=None, gfile=None):
    """
    This routine is used to pass information to next phase. It opens the previous CDF if it exists.
    If it does not exist, tries to get information from gfile
    """

    mitimNML.Reactor, StepParams = None, {}

    # If I have the previous reactor data (CDF file), I'm all set because I have all the information

    if cdffile is not None:
        mitimNML.Reactor = CDFtools.CDFreactor(cdffile)
        StepParams = mitimNML.Reactor.reactor

    # If I don't have the previous reactor data, I need to come up with a way to get the minimum set of information

    else:
        if gfile is not None:
            print(
                "\t- Warning: No previous CDF file selected. Getting rho,psi,kR and dR from gfile, but enforcing BetaN=1.0",
                typeMsg="w",
            )

            runRemote = False
            name = mitimNML.nameRunTot

            g = GEQtools.MITIMgeqdsk(gfile, runRemote=runRemote, name=name)
            rho, psi = g.defineMapping()
            StepParams["CoordinateMapping"] = {"rho": rho, "psi": psi}
            StepParams["kappaRatio"], StepParams["deltaRatio"] = g.getShapingRatios(
                runRemote=runRemote, name=name
            )

            StepParams["BetaN"] = 1.0

        else:
            print(
                "\t- No previous CDF nor geqdsk files selected. Pedestal calculation will not be able to map psi->rho nor use BetaN, shaping ratios correctly",
                typeMsg="w",
            )

            StepParams["BetaN"] = 1.0
            StepParams["CoordinateMapping"] = {
                "rho": np.arange(0, 1.05, 0.05),
                "psi": np.arange(0, 1.05, 0.05),
            }
            StepParams["kappaRatio"] = 1.0
            StepParams["deltaRatio"] = 1.0

    mitimNML.StepParams = StepParams


# ----- MONITOR
def interpretStatus(file):
    dictStatus = {}

    with open(file, "r") as f:
        lines = np.flipud(f.readlines())

    dictStatus["LastFinished"] = findLastFinished(lines)
    dictStatus["LastCheck"] = findLastCheck(lines)
    (
        dictStatus["Status"],
        dictStatus["StatusGrid"],
        dictStatus["StatusGrid_meta"],
    ) = findCurrentStatus(lines)

    if dictStatus["LastFinished"] == "2nd predictive":
        dictStatus["StatusGrid_meta"] = "Finished"

    return dictStatus


def findCurrentStatus(lines):
    current = "Do not know"
    statusGrid = "Not found"

    for i in range(10):
        if "Waiting for run" in lines[i]:
            current = "Waiting for convergence"
            break

    for i in range(len(lines)):
        if "in the TRANSP grid" in lines[i]:
            statusGrid = lines[i].split(">")[-1].split("\n")[0]
            break

    if current == "Do not know":
        meta = "!!!!!!!!!!!!!!!! Unknown !!!!!!!!!!!!!!!!"
    elif current == "Waiting for convergence":
        meta = " ~~ Running ~~"

    return current, statusGrid, meta


def findLastCheck(lines):
    LastCheck = {"time": "Never", "phase": ""}

    for i in range(len(lines)):
        if "SUMMARY FOR RUN" in lines[i]:
            LastCheck = {
                "time": lines[i].split("(")[-1].split(")")[0],
                "phase": lines[i - 2]
                .split("phase")[-1]
                .split("\n")[0]
                .split(":")[-1]
                .replace(" ", ""),
            }

            break

    if LastCheck["time"] == "Never":
        LastCheck["time_ago"] = np.inf
    else:
        LastCheck["time_ago"] = obtainTimeDifference(LastCheck["time"])

    return LastCheck


def obtainTimeDifference(time_str):
    time = datetime.datetime(
        int(time_str.split("-")[0]),
        int(time_str.split("-")[1]),
        int(time_str.split("-")[2].split(" ")[0]),
        int(time_str.split("-")[2].split(" ")[1].split(":")[0]),
        int(time_str.split("-")[2].split(" ")[1].split(":")[1]),
        int(time_str.split("-")[2].split(" ")[1].split(":")[2]),
        0,
    )

    diff = IOtools.getTimeDifference(time)

    return diff


def findLastFinished(lines):
    LastFinished = -1
    for i in range(len(lines)):
        # ------ First check for last phase that has finished
        boolCheck = False
        linesSearch = ["has finished in the grid", "has sucessfully run and converged"]
        for k in linesSearch:
            boolCheck = boolCheck or k in lines[i]

        if boolCheck:
            for j in range(50):
                if "wait for AC generation" in lines[i + j]:
                    LastFinished = 2
                elif "MITIM phase: predictive (2nd)" in lines[i + j]:
                    LastFinished = 1
                elif "MITIM phase: interpretive (1st)" in lines[i + j]:
                    LastFinished = 0

            break

    for i in range(len(lines)):
        # ------ First check for last phase that has finished
        boolCheck = False
        linesSearch = ["phase, using results that were in folder"]
        for k in linesSearch:
            boolCheck = boolCheck or k in lines[i]

        if boolCheck:
            if "interpretive phase, using results that were in folder" in lines[i]:
                LastFinished = np.max([LastFinished, 0])
            elif "predictive phase, using results that were in folder" in lines[i]:
                LastFinished = np.max([LastFinished, 1])

    if LastFinished == -1:
        LastFinished = "No phase has finished"
    elif LastFinished == 0:
        LastFinished = "Interpretive"
    elif LastFinished == 1:
        LastFinished = "1st predictive"
    elif LastFinished == 2:
        LastFinished = "2nd predictive"

    return LastFinished
