import numpy as np
import random, time, pdb, datetime, os, sys, copy
from IPython import embed
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.im_tools.aux import LUTtools
from mitim_tools.transp_tools.tools import NMLtools
from mitim_tools.misc_tools.IOtools import printMsg as print


class MITIMnamelist:
    def __init__(
        self,
        FolderEvaluation,
        FileNamelist,
        numDakota,
        activateLogger=True,
        DebugMode=0,
        IsItSingle=True,
    ):
        self.numDakota = str(numDakota)

        # ----- Folder where simulation occurs ( ~/sparcals/SimulationFiles/Execution/Evaluation.1/ )

        self.FolderDakota = FolderEvaluation

        self.FolderSimulation = self.FolderDakota
        self.FolderTRANSP = self.FolderDakota + "FolderTRANSP/"
        self.FolderEQ = self.FolderDakota + "FolderEQ/"
        self.FolderOutputs = self.FolderDakota + "Outputs/"

        self.ReactorFile = self.FolderOutputs + "ReactorParameters.dat"

        if os.path.exists(self.FolderOutputs):
            os.system("mv {0} {0}prev".format(self.FolderOutputs[:-1]))
            os.system("mkdir " + self.FolderOutputs)
            os.system(
                "mv {0}prev/optimization_log.txt {0}/optimization_log.txt".format(self.FolderOutputs[:-1])
            )
            os.system(f"rm -r {self.FolderOutputs[:-1]}prev")
        else:
            os.system("mkdir " + self.FolderOutputs)

        # ------ Prepare log file

        self.logFile = self.FolderOutputs + "optimization_log.txt_tmp"
        if activateLogger:
            sys.stdout = IOtools.Logger(
                logFile=self.logFile, DebugMode=DebugMode, writeAlsoTerminal=IsItSingle
            )

        # ---------------------------------------------------------------------------------

        currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            "=============================================================================="
        )
        print(f"  Executing IM Evaluation #{numDakota} ({currentTime})")
        print(
            "==============================================================================\n"
        )
        print(">> Parsing mitim namelist parameters")

        # ----- Folder where inputs are

        self.FileNamelist = FileNamelist

        # ----- Read user namelist

        NMLparameters = IOtools.generateMITIMNamelist(self.FileNamelist)

        # ~~~~~~~~~~~~~~~~ Optimization Settings ~~~~~~~~~~~~~~~~

        self.OptimizationParameters = IOtools.CaseInsensitiveDict()
        for i in NMLparameters:
            if "opt_" in i:
                self.OptimizationParameters[i[4:]] = NMLparameters[i]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~ Timings ~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # TRANSP start and end

        self.AbsoluteTINIT = NMLparameters[
            "tinit"
        ]  # Time to start first TRANSP (tinit in namelist)
        self.ftimeOffsets = (
            NMLparameters["ftimeOffsets"] / 1000.0
        )  # Times to end all TRANSP simulations (ftime in namelist)

        # Ramps and other offets for 1st phase

        self.BaselineTime = NMLparameters[
            "BaselineTime"
        ]  # Time to use as baseline for various offsets

        self.startRamp = NMLparameters["startRamp"] / 1000.0
        self.endRamp = NMLparameters["endRamp"] / 1000.0

        self.PredictionOffset = NMLparameters["startPred"] / 1000.0
        self.timePower = self.BaselineTime + NMLparameters["startPower"] / 1000.0
        self.timeSawtooth = self.BaselineTime + NMLparameters["startsawtooth"] / 1000.0
        self.timeCurrentDiff = (
            self.BaselineTime + NMLparameters["startCurDiff"] / 1000.0
        )
        if self.timeCurrentDiff <= self.BaselineTime:
            raise Exception("Ramp starts too soon, please specify > 5ms")

        # Offsets for 2nd phase

        self.PredOffset_2ndPhase = (
            NMLparameters["PredsecondOffset"] / 1000.0
        )  # Amount of time to allow non-predictive ICRF
        self.CurrentDiffOffset_2ndPhase = (
            NMLparameters["CurrsecondOffset"] / 1000.0
        )  # Amount of time to allow fix q-profile
        self.SawtoothOffset_2ndPhase = (
            NMLparameters["SawsecondOffset"] / 1000.0
        )  # Amount of time to allow no-sawtooth

        # ~~~~~~~~~~~~~~~~ Run Settings ~~~~~~~~~~~~~~~~

        self.version = NMLparameters["version"]

        mpis = NMLparameters["mpis"]
        self.trmpi, self.toricmpi, self.ptrmpi = (
            int(mpis[0]),
            int(mpis[1]),
            int(mpis[2]),
        )
        self.machine = NMLparameters["machine"].upper()
        if self.machine == "SPARC":
            self.tok = "D3D"  #'SPRC'
        elif self.machine == "D3D":
            self.tok = "D3D"
        elif self.machine == "AUG":
            self.tok = "AUGD"
        elif self.machine == "CMOD":
            self.tok = "CMOD"
        elif self.machine == "ARC":
            self.tok = "D3D"

        self.nameBaseShot = str(NMLparameters["shotnumber"])
        self.runidletter = NMLparameters["runidletter"]
        # if self.numDakota == 9: self.runidletter = 'T'

        if isinstance(self.runidletter, bool):
            if self.runidletter:
                self.runidletter = "T"
            else:
                self.runidletter = "F"

        self.minWaitLook = int(NMLparameters["minWaitLook"])
        self.Reinitialize_q = NMLparameters["Reinitialize_q"]

        self.SpecialQprofile = NMLparameters["SpecialQprofile"]
        self.SpecialQprofile_time = NMLparameters["SpecialQprofile_time"]

        try:
            self.baseNMFolder = NMLparameters["baseNMFolder"]
        except:
            self.baseNMFolder = None

        try:
            self.baseFolder = NMLparameters["baseUFFolder"]
            self.UFnumber = IOtools.findFileByExtension(
                self.baseFolder, ".MRY", prefix="PRF"
            )
            if self.UFnumber is None:
                self.UFnumber = IOtools.findFileByExtension(
                    self.baseFolder, ".THE", prefix="PRF"
                )
        except:
            self.baseFolder, self.UFnumber = None, None

        self.separatePredictive = NMLparameters["separatePredictive"]

        self.TransportModels = NMLparameters["TransportModels"]

        self.minConfinements = NMLparameters["minTimes"]
        self.rotationPred = NMLparameters["rotationPred"]
        self.pedestalPred = NMLparameters["pedestalPred"]
        self.useBcoil = NMLparameters["useBcoil"]
        self.enforceICRFonaxis = NMLparameters["enforceICRFonaxis"]

        self.TGLFsettings = NMLparameters["TGLFsettings"]

        # EQ ---------
        self.EquilibriumType = NMLparameters["EquilibriumType"]
        if self.EquilibriumType is not None and self.EquilibriumType.lower() == "gfile":
            self.gfile_loc = IOtools.expandPath(NMLparameters["gfile_loc"])
            try:
                self.UseGFILEgeometryParams = NMLparameters["UseGFILEgeometryParams"]
            except:
                self.UseGFILEgeometryParams = False
        else:
            self.gfile_loc = None

        try:
            self.advancedEQ = NMLparameters["advancedEQ"]
        except:
            self.advancedEQ = 0

        if self.advancedEQ == 1:
            self.prescribedEvolution = NMLparameters["prescribedEvolutionTSC"]
            self.prescribedEvolutionTimes = NMLparameters["prescribedEvolutionTimes"]
        if self.advancedEQ == 2:
            self.prescribedEvolution = NMLparameters["prescribedEvolution"]
            EQsweepStart = NMLparameters["EQsweepStart"]
            EQsweepHz = NMLparameters["EQsweepHz"]
            howmanyrepeats = 10
            self.EQsweepTimes = [0.0, EQsweepStart]
            for i in range(howmanyrepeats):
                self.EQsweepTimes.append(self.EQsweepTimes[-1] + 0.5 / EQsweepHz)
                self.EQsweepTimes.append(self.EQsweepTimes[-1] + 0.5 / EQsweepHz)

        # q profile ---------
        try:
            self.Initialq_gfile = NMLparameters["Initialq_gfile"]
        except:
            self.Initialq_gfile = False

        try:
            self.SpecialQUFILE = NMLparameters["SpecialQUFILE"]
        except:
            self.SpecialQUFILE = None

        if self.Initialq_gfile and self.SpecialQUFILE is not None:
            raise Exception(
                "Special u-file selected, but g-file q-profile option also enabled"
            )

        # Pedestal ---------
        self.Pedestal_Redk = NMLparameters["Pedestal_Redk"]
        try:
            self.UseShapingRatio = NMLparameters["UseShapingRatio"]
        except:
            self.UseShapingRatio = False
        self.enforceIp = NMLparameters["enforce_Ip"]
        self.enforceneped = NMLparameters["enforce_neped"]
        self.PedestalType = NMLparameters["PedestalType"]

        if self.PedestalType is None:
            self.PedestalBC = NMLparameters["PedestalBC"]
        else:
            self.PedestalBC = None

        if self.PedestalType == "lut" or self.PedestalType == "surrogate_model":
            self.LuT_loc, self.LuT_variables, self.LuT_fixed = (
                IOtools.expandPath(NMLparameters["LuT_loc"]),
                NMLparameters["LuT_variables"],
                NMLparameters["LuT_fixed"],
            )
        else:
            self.LuT_loc, self.LuT_variables, self.LuT_fixed = None, None, None

        if self.PedestalType == "vals":
            self.LuT_vals = NMLparameters["LuT_vals"]
        else:
            self.LuT_vals = [None, None]

        if self.PedestalType == "nn":
            self.nn_loc = IOtools.expandPath(NMLparameters["nn_loc"])
        else:
            self.nn_loc = None

        if self.enforceIp:
            self.qPRF = float(NMLparameters["qPRF"])
        if self.enforceneped:
            self.nPRF = float(NMLparameters["nPRF"])

        if (
            self.EquilibriumType is not None
            and self.EquilibriumType.lower() == "gfile"
            and self.UseGFILEgeometryParams
            and self.Pedestal_Redk is not None
        ):
            raise Exception(
                "g-file extraction and manual specification of pedestal values both enabled"
            )

        try:
            self.PedestalShape = NMLparameters["PedestalShape"]
        except:
            self.PedestalShape = 1

        try:
            self.useRotationMach = NMLparameters["useRotationMach"]
        except:
            self.useRotationMach = None

        self.includeASH = NMLparameters["includeASH"]

        # -----------------------------

        self.restartTime = NMLparameters["restartTime"]
        self.useAxialTrickPhases = NMLparameters["useAxialTrickPhases"]
        self.PedestalUfiles = NMLparameters["PedestalUfiles"]
        self.timeLagDensity = NMLparameters["timeLagDensity"] / 1000.0

        # Density -----
        try:
            self.SpecialDensityUFILE = NMLparameters["SpecialDensityUFILE"]
        except:
            self.SpecialDensityUFILE = None
        try:
            self.OfflineDensity = NMLparameters["OfflineDensity"]
        except:
            self.OfflineDensity = False

        self.SmoothInitialProfiles = NMLparameters["SmoothInitialProfiles"]

        try:
            self.mmx_loc = NMLparameters["mmx_loc"]
        except:
            self.mmx_loc = None

        # -----------------------------

        self.RestartBaselineTime_YN = NMLparameters["RestartBaselineTime_YN"]
        if not self.RestartBaselineTime_YN:
            self.restartTime[0] = None

        self.retriesmitim = NMLparameters["retriesMITIM"]

        # --------------------------------------------------------------------------------

        self.MITIMmode = NMLparameters["MITIMmode"]

        if self.MITIMmode == 1 or self.MITIMmode == 3:
            print(">> Standard MITIM mode enabled (mode 1), running full workflow")

            self.RunDoublePredictive = NMLparameters["RunDoublePredictive"]
            if self.RunDoublePredictive:
                self.DoubleInInterpretive = NMLparameters["DoubleInInterpretive"]
            numch = NMLparameters["conv_num"]

            if self.MITIMmode == 3:
                self.completedFirst = IOtools.expandPath(
                    NMLparameters["completedFirst"]
                )
            else:
                self.completedFirst = None

        elif self.MITIMmode == 2:
            print(">> Single-phase MITIM mode enabled (mode 2)")

            self.RunDoublePredictive, self.DoubleInInterpretive, self.completedFirst = (
                False,
                False,
                None,
            )
            numch = [100, 100, 100]

        self.DoubleInInterpretiveMinLength = float(
            NMLparameters["DoubleInInterpretiveMinLength"]
        )

        # --------------------------------------------------------------------------------

        # Convergence
        self.numbercheckConvergence = [int(kk) for kk in numch]
        self.toleranceConvergence = [float(kk) for kk in NMLparameters["conv_tol"]]
        self.quantitiesConvergence = NMLparameters["conv_vars"]

        self.changeHeatingHardware = NMLparameters["changeHeatingHardware"]

        # Possibility of LH transition
        try:
            self.TimeImposedPedestal = NMLparameters["TimeImposedPedestal"]
        except:
            self.TimeImposedPedestal = None

        try:
            self.LH_time, self.LH_valsL = NMLparameters["LHtransition"], [
                float(kk) for kk in NMLparameters["Lvalues"]
            ]
        except:
            self.LH_time, self.LH_valsL = None, [None, None]

        if self.LH_time is not None and not self.PedestalUfiles:
            raise Exception("L-H transition exists but pedestal changed in namelist")

        # -------- Read TRANSP direct parameters

        self.TRANSPnamelist = {}
        for i in NMLparameters:
            if "transp_" in i:
                self.TRANSPnamelist[i[7:]] = NMLparameters[i]

        if self.toricmpi > 1:
            self.TRANSPnamelist["NTORIC_PSERVE"] = 1
        else:
            self.TRANSPnamelist["NTORIC_PSERVE"] = 0

        if self.trmpi > 1:
            self.TRANSPnamelist["NBI_PSERVE"] = 1
        else:
            self.TRANSPnamelist["NBI_PSERVE"] = 0

        # ------ Read Baseline Parameters

        self.namelistBaselineFile = NMLparameters["namelistBaselineFile"]

        # Option 1: Baseline values given in this namelist
        if self.namelistBaselineFile is None:
            whereBaseline = NMLparameters
        else:
            whereBaseline = IOtools.generateMITIMNamelist(self.namelistBaselineFile)

        self.BaselineParameters = readBaselineParameters(whereBaseline)

        # Namelist-specific params
        self.TRANSPnamelist["tauph(1)"] = self.BaselineParameters["taup"]
        self.TRANSPnamelist["tauph(2)"] = self.BaselineParameters["taup"]
        self.TRANSPnamelist["tauph(3)"] = self.BaselineParameters["taup"]
        self.TRANSPnamelist["taupo"] = self.BaselineParameters["taup"]
        self.TRANSPnamelist["taumin"] = self.BaselineParameters["taup"]

        # ------

        if len(self.numDakota) > 2:
            numDakotaModified = self.numDakota[1:]
        else:
            numDakotaModified = self.numDakota

        self.nameRun = self.runidletter + str(numDakotaModified).zfill(2)  # B02
        self.nameRunTot = self.nameBaseShot + self.nameRun  # 175844B02
        self.dakNumUnit = int(self.nameRunTot[-1])  # 2

        # ------ Files & Folders

        self.LocationNewCDF = self.FolderTRANSP + self.nameRunTot + ".CDF"
        self.namelistPath = self.FolderTRANSP + self.nameRunTot + "TR.DAT"
        self.namelistPath_ptsolver = self.FolderTRANSP + "ptsolver_namelist.dat"

        # ----

        self.predictQuantities = ["te", "ti", "ne"]

        # ------ Initialization

        try:
            self.LocationOriginalCDF = IOtools.expandPath(NMLparameters["baseCDFfile"])
        except:
            self.LocationOriginalCDF = None
        try:
            self.excelProfile = IOtools.expandPath(NMLparameters["excelProfile"])
        except:
            self.excelProfile = None

        # ------ Baseline simulation names

        if self.baseNMFolder is not None:
            self.NMLnumber = IOtools.findFileByExtension(self.baseNMFolder, "TR.DAT")
        else:
            self.NMLnumber = "10000"

        # ------ Impurities
        try:
            self.impurityMode = int(NMLparameters["impurityMode"])
        except:
            self.impurityMode = 0

        self.fmain_avol, self.zimp_high, self.imp_profs = None, None, None
        if self.impurityMode == 1 or self.impurityMode == 3:
            self.fmain_avol = NMLparameters["fmain_avol"]
            self.zimp_high = int(NMLparameters["zimp_high"])
        if self.impurityMode > 1:
            self.imp_profs = []
            for j in range(len(NMLparameters["imp_profs"])):
                self.imp_profs.append(IOtools.expandPath(NMLparameters["imp_profs"][j]))

        # ------ Fast ions transport
        try:
            self.fast_anom = IOtools.expandPath(NMLparameters["fast_anom"])
        except:
            self.fast_anom = None

        # ------ Minority density
        try:
            self.min_dens = IOtools.expandPath(NMLparameters["min_dens"])
        except:
            self.min_dens = None

        try:
            self.functionalForms = NMLparameters["forceFunctionals"]
        except:
            self.functionalForms = None

    def defineConvergence(self, numStage=0):
        minConfinements = self.minConfinements[numStage]
        numbercheck = self.numbercheckConvergence[numStage]
        tolerance = self.toleranceConvergence[numStage]
        extratime = np.max(
            [
                0,
                self.PredictionOffset
                + self.BaselineTime
                - self.TRANSPnamelist["tinit"],
            ]
        )

        if minConfinements > 0:
            tau, text_extra = 1.0, ""
        else:
            minConfinements = np.abs(minConfinements)
            tau = self.StepParams["taue"] / 1000.0
            print(
                f">> Confinement time in previous simulation is {tau*1000.0:.0f}ms, adjusting minimum time of this simulation"
            )
            text_extra = f"{minConfinements} minimum confinement times and "
        mintime = extratime + tau * minConfinements

        print(
            f">> Minimum run time set to {mintime*1000:.0f}ms (provided {text_extra}{int(extratime*1000)}ms prediction offset)"
        )
        if numbercheck > 0:
            print(
                f">> Using {numbercheck} consecutive sawteeh, {tolerance*100.0}% variation of quantities for convergence"
            )
        else:
            print(f">> Must run at least {-numbercheck} sawteeh")

        # Remove Q if plasma is not DT
        if (
            "VarPRF_Q" in self.quantitiesConvergence
            and self.machine != "SPARC"
            and self.machine != "ARC"
        ):
            print(
                ">> Removing Q from convergence criterion because machine is not SPARC"
            )
            self.quantitiesConvergence = self.quantitiesConvergence.tolist()
            self.quantitiesConvergence.remove("VarPRF_Q")

        txtvar = ""
        for i in self.quantitiesConvergence:
            txtvar += f" {i},"

        print(f">> Variables for convergence:{txtvar[:-1]}")

        self.convCrit = {"timerunMIN": mintime}
        for i in self.quantitiesConvergence:
            self.convCrit[i] = {
                "numbercheck": numbercheck,
                "tolerance": tolerance,
                "sawSmooth": True,
                "radius": None,
            }


def readBaselineParameters(whereBaseline):
    BaselineParameters = IOtools.CaseInsensitiveDict()
    for i in whereBaseline:
        if "baseline_" in i:
            BaselineParameters[i[9:]] = whereBaseline[i]

    return BaselineParameters


def getParametersDictionary(mitimNML, onlyReadFinal=False):
    # Options
    useBcoil = mitimNML.useBcoil
    enforceICRFonaxis = mitimNML.enforceICRFonaxis
    PedestalType = mitimNML.PedestalType
    ReactorFile = mitimNML.ReactorFile
    nameRunTot = mitimNML.nameRunTot

    # ------------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------------

    # ~~~ Modify baseline parameters if the geometry is read from a g-file
    if (
        not onlyReadFinal
        and mitimNML.EquilibriumType is not None
        and mitimNML.EquilibriumType.lower() == "gfile"
        and mitimNML.UseGFILEgeometryParams
    ):
        dictParamsOrig = updateParametersWithGfile(
            mitimNML.gfile_loc, mitimNML.BaselineParameters, mitimNML.nameRunTot
        )

    else:
        dictParamsOrig = mitimNML.BaselineParameters

    dictParamsOrig["eta_ICH"] = PLASMAtools.getICHefficiency(dictParamsOrig)

    # Other work
    dictParams = dictParamsOrig

    a = dictParams["epsilon"] * dictParams["rmajor"]

    # Avoid problems with inconsistent namelist and DVs in terms of Bt and Bcoil
    Bt, Bcoil = PLASMAtools.BaxisBcoil(
        a,
        dictParams["b_shield"] + dictParams["wall_gap"],
        dictParams["rmajor"],
        Bt=dictParams["Bt"],
        Bcoil=None,
    )

    dictParams["Bt"] = Bt
    dictParams["Bcoil"] = Bcoil

    # ICRF frequency
    if enforceICRFonaxis:
        print(">> Fich selected to be on axis for He3 ")
        dictParams["Fich"] = [PLASMAtools.FrequencyOnAxis(dictParams["Bt"])]

    # -----------------------------------------------
    # ------------ Look-up Table Parameters
    # -----------------------------------------------

    # ~~~~~~~~ Plasma Current ~~~~~~~~

    if mitimNML.enforceIp:
        dictParams = findPlasmaCurrent(dictParams, mitimNML)

    # ~~~~~~~~ Pedestal Density ~~~~~~~~

    if mitimNML.enforceneped:
        if mitimNML.PedestalType == "vals":
            raise Exception(
                "Namelist indicates to enforce neped but entire pedestal is being specified with VALS"
            )
        else:
            dictParams = PLASMAtools.findPedestalDensity(dictParams, mitimNML)

    # -----------------------------------------------
    # ------------ Pedestal Option
    # -----------------------------------------------

    changePed = {"rmajor": 0, "epsilon": 1, "delta": 2, "kappa": 3}
    for ikey in changePed:
        if mitimNML.Pedestal_Redk is not None:
            fac = dictParams[ikey] / dictParamsOrig[ikey]
            mitimNML.Pedestal_Redk[changePed[ikey]] = (
                mitimNML.Pedestal_Redk[changePed[ikey]] * fac
            )
            print(
                '>> Pedestal_Redk enabled and "{0}" is a DV. Rescaling {0} for pedestal to {1}'.format(
                    ikey, mitimNML.Pedestal_Redk[changePed[ikey]]
                )
            )

    # -----------------------------------------------
    # -----------------------------------------------

    # ICH efficiency

    dictParams["eta_ICH"] = PLASMAtools.getICHefficiency(dictParams)

    # Write file with final parameters to be run

    outfile = open(ReactorFile, "w")
    outfile.write(f" >> >> Summary for run {nameRunTot} << <<\n")
    outfile.write("-------- Input Parameters -------------\n")
    for i in dictParams:
        textt = f"{i} = {dictParams[i]}\n"
        outfile.write(textt)
    outfile.write("----------------------------------------\n")
    outfile.close()

    # ------------------------------------------------------------------------
    # ------------ Add those values that concern the TRANSP namelist to be modified
    # -------------------------------------------------------------------------

    # ----- Confirm plasma features
    mitimNML.PlasmaFeatures = {
        "ICH": np.sum(dictParams["Pich"]) > 0.0,
        "ECH": np.sum(dictParams["Pech"]) > 0.0,
        "NBI": np.sum(dictParams["Pnbi"]) > 0.0,
        "MC": np.sum(dictParams["Pnbi"]) > 0.0
        or mitimNML.machine == "SPARC"
        or mitimNML.machine == "ARC",
        "ASH": mitimNML.includeASH,
        "Fuel": dictParams["Fuel"],
    }

    mitimNML.TRANSPnamelist = parametersConcerningNML(
        mitimNML.TRANSPnamelist,
        dictParams,
        mitimNML.PlasmaFeatures,
        impurityMode=mitimNML.impurityMode,
        fmain=mitimNML.fmain_avol,
        zimp_high=mitimNML.zimp_high,
        fast_anom=mitimNML.fast_anom,
        min_dens=mitimNML.min_dens,
    )

    return dictParams, dictParamsOrig


def parametersConcerningNML(
    TRANSPnamelist,
    dictParams,
    PlasmaFeatures,
    impurityMode=0,
    fmain=0.85,
    zimp_high=50,
    fast_anom=None,
    min_dens=None,
):
    numAntennas_ICRF, numAntennas_ECRF, numBeams_NBI = (
        len(dictParams["Fich"]),
        len(dictParams["Fech"]),
        len(dictParams["Vnbi"]),
    )

    # Impurities present in plasma
    TRANSPnamelist = addImpuritiesNML(
        TRANSPnamelist,
        dictParams,
        impurityMode=impurityMode,
        fmain=fmain,
        zimp_high=zimp_high,
    )

    # Main plasma
    if PlasmaFeatures["Fuel"] == 2:
        if "tdens" in dictParams:
            tritium = dictParams["tdens"]
            deuterium = dictParams["ddens"]
        else:
            tritium = 0.5
            deuterium = 0.5
        TRANSPnamelist["frac(1)"] = deuterium
        TRANSPnamelist["frac(2)"] = tritium

    # Minority species

    if PlasmaFeatures["ICH"]:
        TRANSPnamelist["xzmini"] = dictParams["Zmini"]
        TRANSPnamelist["amini"] = dictParams["Amini"]

        if min_dens is not None:
            TRANSPnamelist["prenmr"] = '"PRF"'
            TRANSPnamelist["extnmr"] = '"NMR"'
            TRANSPnamelist["nrinmr"] = -5
            TRANSPnamelist["frmini"] = None

            print(
                ">> Warning: Minority fraction from BASELINE ommitted because NMR file is being created",
                typeMsg="w",
            )

        else:
            TRANSPnamelist["frmini"] = dictParams["Fmini"]

    # ICRF and ECRF Frequencies, NBI Voltages
    TRANSPnamelist = writeAntennasNML(
        TRANSPnamelist,
        PlasmaFeatures,
        dictParams,
        numAntennas_ICRF=numAntennas_ICRF,
        numAntennas_ECRF=numAntennas_ECRF,
        numBeams_NBI=numBeams_NBI,
    )

    TRANSPnamelist["NLBCCW"] = dictParams["NLBCCW"]
    TRANSPnamelist["NLJCCW"] = dictParams["NLJCCW"]

    # Fast ion transport
    if fast_anom is not None:
        TRANSPnamelist["nmdifb"] = 3

        TRANSPnamelist["pred2f"] = '"PRF"'
        TRANSPnamelist["extd2f"] = '"D2F"'
        TRANSPnamelist["nrid2f"] = -5

        TRANSPnamelist["prev2f"] = '"PRF"'
        TRANSPnamelist["extv2f"] = '"V2F"'
        TRANSPnamelist["nriv2f"] = -5

    else:
        TRANSPnamelist["nmdifb"] = 0

    return TRANSPnamelist


def updateParametersWithGfile(gfile, BaselineParameters, nameRunTot):
    print(
        ">> Warning: DV and baseline geometric parameters are being overwritten by g-file boundary in ..{0}".format(
            gfile[40:]
        ),
        typeMsg="w",
    )

    g = GEQtools.MITIMgeqdsk(gfile)
    rmajor, epsilon, kappa, delta, zeta, z0 = g.paramsLCFS()

    txt = "\t- New parameters:"
    txt += f" Rmajor = {rmajor:.3f}m,"
    BaselineParameters["rmajor"] = rmajor
    txt += f" epsilon = {epsilon:3f} (a = {epsilon * rmajor:.2f}m),"
    BaselineParameters["epsilon"] = epsilon
    txt += f" kappa = {kappa:.3f},"
    BaselineParameters["kappa"] = kappa
    txt += f" delta = {delta:.3f},"
    BaselineParameters["delta"] = delta
    txt += f" zeta = {zeta:.3f},"
    BaselineParameters["zeta"] = zeta
    txt += f" z0 = {z0:.3f}m,"
    BaselineParameters["z0"] = z0

    print(txt[:-1])

    return BaselineParameters


def addImpuritiesNML(
    TRANSPnamelist, dictParams, coronal=True, impurityMode=0, fmain=0.85, zimp_high=50
):
    impurityRotation = 0  # Position of impurity that rotates

    if impurityMode == 0:
        print(">> Standard impurity specification with DENSIMS selected")
        try:
            fractionsZimp = dictParams["Fimp"]
            while fractionsZimp.max() >= 1.0:
                print(
                    "\t- At least 1 impurity fraction above one, run would crash, dividing all by 10"
                )
                fractionsZimp = fractionsZimp / 10.0

            if "fimp_edge" in dictParams:
                print("\t- Edge density for impurities specified!")
                fractionsZimp_edge = dictParams["Fimp_edge"]
                while fractionsZimp_edge.max() > 1.0:
                    print(
                        "\t- At least 1 impurity fraction above one, run would crash, dividing all by 10"
                    )
                    fractionsZimp_edge = fractionsZimp_edge / 10.0
            else:
                fractionsZimp_edge = None

        except:
            print(">> Impurity fractions not found, using all 1.0E-1")
            fractionsZimp = np.ones(100) * 1e-1
            fractionsZimp_edge = None

    elif impurityMode == 1 or impurityMode == 3:
        print(">> Obtaining low-Z impurity...")

        ZimpL_mod, Rimp = PLASMAtools.getLowZimpurity(
            Fmain=fmain,
            Zeff=dictParams["Zeff"],
            Zmini=dictParams["Zmini"],
            Fmini=dictParams["Fmini"],
            ZimpH=zimp_high,
            FimpH=dictParams["Fimp"][0],
            Zother=2,
            Fother=0.1e-2,
        )
        fractionsZimp = [Rimp * 1e-1, 1e-1]
        fractionsZimp_edge = None
        dictParams["Zimp"][1], dictParams["Aimp"][1] = ZimpL_mod, ZimpL_mod * 2

        impurityRotation = 1  # Because the low-Z is second

    else:
        fractionsZimp, fractionsZimp_edge = [None] * 10, None

    # ---------

    if isinstance(dictParams["Zimp"], int) or isinstance(dictParams["Zimp"], float):
        Zs, As, Fs = (
            np.array([dictParams["Zimp"]]),
            np.array([dictParams["Aimp"]]),
            np.array([fractionsZimp]),
        )
        if fractionsZimp_edge is not None:
            Fs_edge = np.array([fractionsZimp_edge])
        else:
            Fs_edge = None
    else:
        Zs, As, Fs = dictParams["Zimp"], dictParams["Aimp"], fractionsZimp
        if fractionsZimp_edge is not None:
            Fs_edge = fractionsZimp_edge
        else:
            Fs_edge = None

    for i in range(len(Zs)):
        TRANSPnamelist[f"xzimps({i + 1})"] = Zs[i]
        TRANSPnamelist[f"aimps({i + 1})"] = As[i]
        if coronal:
            addCor = 1
        else:
            addCor = 0
        TRANSPnamelist[f"nadvsim({i + 1})"] = addCor

        if impurityMode < 2 or (impurityMode == 3 and i > 0):
            TRANSPnamelist[f"densim({i + 1})"] = round(Fs[i], 6)
            print(
                "\t- Added impurity to namelist: Z={0}, A={1:.1f}, frel={2:.2e}".format(
                    int(Zs[i]), As[i], Fs[i]
                )
            )
            if Fs_edge is not None:
                TRANSPnamelist[f"densima({i + 1})"] = round(Fs_edge[i], 6)

        else:
            if impurityMode == 2 or (impurityMode == 3 and i == 0):
                print(
                    f"\t- Impurity Z={int(Zs[i])}, A={As[i]:.1f} specified from UFILE"
                )
                TRANSPnamelist[f"densim({i + 1})"] = None
                TRANSPnamelist[f"presim({i + 1})"] = '"PRF"'
                TRANSPnamelist[f"extsim({i + 1})"] = f'"NIMP{i + 1}"'
                TRANSPnamelist[f"nrisim({i + 1})"] = -5

            if impurityMode == 2:
                print("\t- Because all impurities from UFILES, do not use Zeff UFILE")
                (
                    TRANSPnamelist["prezf2"],
                    TRANSPnamelist["extzf2"],
                    TRANSPnamelist["nrizf2"],
                ) = (None, None, None)
                TRANSPnamelist["nlzfi2"], TRANSPnamelist["nlzsim"] = "F", "T"

    TRANSPnamelist["NVTOR_Z"], TRANSPnamelist["XVTOR_A"] = int(
        Zs[impurityRotation]
    ), int(As[impurityRotation])

    return TRANSPnamelist


def writeAntennasNML(
    TRANSPnamelist,
    PlasmaFeatures,
    dictParams,
    numAntennas_ICRF=1,
    numAntennas_ECRF=1,
    numBeams_NBI=1,
):
    if PlasmaFeatures["ICH"]:
        strAntenna = f"{dictParams['Fich'][0] * 1000000.0:.4e}"
        for i in range(numAntennas_ICRF - 1):
            strAntenna += f", {dictParams['Fich'][i + 1] * 1000000.0:.4e}"
        TRANSPnamelist["FRQicha"] = strAntenna
        TRANSPnamelist["NICHA"] = numAntennas_ICRF

    if PlasmaFeatures["ECH"]:
        strAntenna = f"{dictParams['Fech'][0] * 1000000.0:.4e}"
        for i in range(numAntennas_ECRF - 1):
            strAntenna += f", {dictParams['Fech'][i + 1] * 1000000.0:.4e}"
        TRANSPnamelist["FREQECH"] = strAntenna
        TRANSPnamelist["NANTECH"] = numAntennas_ECRF

    if PlasmaFeatures["NBI"]:
        for cont, i in enumerate(dictParams["Vnbi"]):
            TRANSPnamelist[f"EINJA({cont + 1})"] = i * 1e3
        TRANSPnamelist["NBEAM"] = numBeams_NBI

    return TRANSPnamelist


def getBaseNamelists(
    nameRunTot,
    OutputFolder,
    PlasmaFeatures={},
    InputFolder=None,
    tok="SPARC",
    appendHeating=True,
    NMLnumber="10000",
    useMMX=False,
    TGLFsettings=5,
):
    print(">> Generating baseline NML")
    nml = NMLtools.default_nml(
        NMLnumber, tok, PlasmaFeatures=PlasmaFeatures, TGLFsettings=TGLFsettings
    )

    if InputFolder is None:
        # Is the user has indicated no NML folder, create a base namelist for this tokamak
        nml.write(BaseFile=OutputFolder + f"/{NMLnumber}TR.DAT")
    else:
        print(">> Copying baseline NML")
        os.system("cp {0}/*TR.DAT" + " {1}/.".format(InputFolder, OutputFolder))
        if appendHeating:
            for j in ["ICRF", "ECRF", "LH", "NBI"]:
                os.system(
                    "cp {0}/*TR.DAT_{2}" + " {1}/.".format(InputFolder, OutputFolder, j)
                )

    # ~~~~~~~~~~~~~~~~ Renaming ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    IOtools.renameCommand(NMLnumber, nameRunTot, folder=OutputFolder)

    namelistPath = OutputFolder + nameRunTot + "TR.DAT"

    # ~~~~~~~~~~~~~~~~ Append Heating modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if appendHeating:
        nml.BaseFile = namelistPath
        nml.appendNMLs()

    # Correct namelist
    IOtools.correctNML(namelistPath)


def writeBASELINEnamelist(
    file,
    Ip,
    kappa,
    delta,
    epsilon,
    Rmajor,
    Bt,
    neped,
    Zeff,
    Zimp,
    Aimp,
    Fimp,
    Pich,
    Fich,
    Zmini,
    Amini,
    Fmini,
    Pech,
    Fech,
    Polech,
    Torech,
    Pnbi,
    Vnbi,
    AxialDeffs,
    NLBCCW,
    NLJCCW,
    He4=True,
):
    txt = [
        "",
        "# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "# Baseline Values (the ones that can be a DV)",
        "# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        f"BASELINE_Ip             = {Ip:.2f}",
        f"BASELINE_kappa          = {kappa:.2f}",
        f"BASELINE_delta          = {delta:.2f}",
        f"BASELINE_epsilon        = {epsilon:.3f}",
        f"BASELINE_rmajor         = {Rmajor:.2f}",
        f"BASELINE_Bt             = {Bt:.2f}",
        f"BASELINE_neped          = {neped:.2f}",
        "",
        "BASELINE_factor_ped_degr = 1.0",
        "",
        f"BASELINE_Zeff           = {Zeff:.2f}",
        f"BASELINE_Zimp           = {Zimp}",
        f"BASELINE_Aimp           = {Aimp}",
        f"BASELINE_Fimp           = {Fimp}",
        "",
        "BASELINE_Pich           = [{0}]         # MW".format(
            ",".join([str(round(i, 2)) for i in Pich])
        ),
        "BASELINE_Fich           = [{0}]         # MHz".format(
            ",".join([str(round(i, 1)) for i in Fich])
        ),
        f"BASELINE_Zmini          = {Zmini:.1f}",
        f"BASELINE_Amini          = {Amini:.1f}",
        f"BASELINE_Fmini          = {Fmini:.3f}",
        "",
        "BASELINE_Pech           = [{0}]       # MW".format(
            ",".join([str(round(i, 2)) for i in Pech])
        ),
        "BASELINE_Fech           = [{0}]       # MHz".format(
            ",".join([str(round(i, 1)) for i in Fech])
        ),
        "BASELINE_Polech         = [{0}]       # In degrees, poloidal".format(
            ",".join([str(round(i, 1)) for i in Polech])
        ),
        "BASELINE_Torech         = [{0}]       # In degrees, toroidal".format(
            ",".join([str(round(i, 1)) for i in Torech])
        ),
        "",
        "BASELINE_Pnbi           = [{0}]         # MW".format(
            ",".join([str(round(i, 2)) for i in Pnbi])
        ),
        "BASELINE_Vnbi           = [{0}]         # keV".format(
            ",".join([str(round(i, 1)) for i in Vnbi])
        ),
        "",
        f"BASELINE_AxialDeffs = {AxialDeffs}",
        "",
        "BASELINE_antenna_gap = 0.02",
        "BASELINE_antenna_pol = 26.5",
        "BASELINE_wall_gap    = 0.05",
        "",
        "BASELINE_zeta           = 0.0",
        "BASELINE_z0             = 0.0",
        "",
        "BASELINE_wall_gap       = 0.05",
        "BASELINE_b_shield       = 0.01",
        "BASELINE_Bcoil          = 15.0",
        "",
        f"BASELINE_NLBCCW\t\t = {NLBCCW}",
        f"BASELINE_NLJCCW\t\t = {NLJCCW}",
        "",
    ]

    if He4:
        txt.append(
            "BASELINE_xHE4		= [ 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,1.0 ] # rhoN"
        )
        txt.append(
            "BASELINE_DHE4		= [ 0.03, 0.03, 0.03, 0.03, 0.05, 0.2, 1.0, 1.3, 1.6, 1.8, 2.0, 2.1, 2.0, 1.8, 1.6, 1.3, 0.8, 0.6, 0.4, 0.3, 0.3,100.0] # m^2/s"
        )
        txt.append(
            "BASELINE_VHE4		= [ 0.0, 0.0, 0.0, -0.3, -0.5, -0.9, -1.0, -1.1, -1.2, -1.2, -1.2, -1.1, -1.1, -1.0, -0.5, -0.5, -0.5, -1.0, -5., -30.0, -150.0,0.0] # m/s"
        )

    with open(file, "w") as f:
        f.write("\n".join(txt))


def findPlasmaCurrent(dictParams, mitimNML):
    if mitimNML.qPRF is not None:
        Rmajor, kappa, Bt, epsilon, delta = (
            dictParams["rmajor"],
            dictParams["kappa"],
            dictParams["Bt"],
            dictParams["epsilon"],
            dictParams["delta"],
        )
        dictParams["Ip"] = PLASMAtools.evaluate_qstar(
            mitimNML.qPRF, Rmajor, kappa, Bt, epsilon, delta, isInputIp=False
        )
        print(
            ">> Ip has been enforced from qPRF (value: {0}), Ip = {1:.2f}".format(
                mitimNML.qPRF, dictParams["Ip"]
            )
        )
    else:
        if mitimNML.PedestalType in ["lut", "vals"]:
            Rmajor, kappa, Bt, epsilon, delta, Zeff, neped = (
                dictParams["rmajor"],
                dictParams["kappa"],
                dictParams["Bt"],
                dictParams["epsilon"],
                dictParams["delta"],
                dictParams["Zeff"],
                dictParams["neped"],
            )

            _, _, _, dictParams["Ip"], _, _ = LUTtools.search_LuT_EPED(
                mitimNML.PedestalType,
                LuT_loc=mitimNML.LuT_loc,
                LuT_variables=mitimNML.LuT_variables,
                LuT_vals=mitimNML.LuT_vals,
                LuT_fixed=mitimNML.LuT_fixed,
                Bt=Bt,
                Rmajor=Rmajor,
                kappa=kappa,
                delta=delta,
                neped=neped,
                Zeff=Zeff,
                epsilon=epsilon,
            )

        else:
            raise Exception(
                "No pedestal specified but namelist indicates to replace Ip with LuT"
            )

        print(f">> Ip has been enforced from LuT/vals, Ip = {dictParams['Ip']:.2f}")

    return dictParams
