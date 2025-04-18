import copy
import datetime
import pandas
import freegs
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import interp1d
from mitim_modules.freegsu.utils import FREEGSUplotting, FREEGSUparams
from mitim_tools.misc_tools import IOtools, PLASMAtools, FARMINGtools, MATHtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def evaluator(
    dictDVs,
    CoilCurrents={},
    CoilCurrents_lower=None,
    Constraints={},
    plot=False,
    debug=False,
    internalParallel=False,
    printYN=True,
    optionsFREEGS={
        "n": 257,
        "equalizers": {},
        "div1div": [1.0, 0.0, 0.0, 0.0],
        "div2div": [1.0, 0.0, 0.0, 0.0],
        "csForceRatio": {
            "cs1u": None,
            "cs2u": None,
            "cs3u": None,
            "cs1l": None,
            "cs2l": None,
            "cs3l": None,
        },
        "symmetricX": True,
        "plasmaParams": {"BR": 22.49, "pAxis": 2.6e6, "Ip": 8.7, "R0": 1.85},
        "outMidplane_matching": None,
        "coilsVersion": "V2",
        "onlyUpperPlate": None,
        "outphaseForce": False,
        "profiles_settings": {"type": "pIp", "args": {"alpha_m": 2.0, "alpha_n": 2.0}},
        "psiGuess": None,
        "separatrixPoint": 2.2,
        "sortbyXpoints": "flux",
    },
    CoilCurrents_orig=None,
    ProblemExtras=None,
    paramsMods=None,
    figs=None,
    onlyPrepare=False,
    ):
    """
    outMidplane_matching is dRsep represented as follows:
    [0,x] -> dRsep = -x if the 1st x-point is upper
    [x,0] -> dRsep = +x if the 1st x-point is upper

    paramsMods is to change params to compare runs

    separatrixPoint is just used as evaluation of that R value
    """

    folderData = IOtools.expandPath("$MFEIM_PATH") / "private_code_mitim" / "FREEGS_SPARC"

    # --------------------------------------------------------
    # Join dictionaries with internal and external options
    # --------------------------------------------------------

    Params = {}
    Params.update(optionsFREEGS)
    Params.update(Constraints)
    Params.update({"printYN": printYN, "debug": debug})

    Params, ProblemExtras = curateGSoptions(Params, ProblemExtras)

    """
  --------------------------------------------------------
  Merge fixed coils with design variables during optimization.
  Output must be all simplified coils
  --------------------------------------------------------
  """

    # ---- Separation upper/lower
    dictDVs_upper, dictDVs_lower = separateUpperLowerDVs(dictDVs)
    # ---- ----  ----  ----  ----  ----

    # Upper set
    coils, coils_names = FREEGSUparams.updateCoils(
        CoilCurrents,
        dictDVs_upper,
        div1div=Params["div1div"],
        div2div=Params["div2div"],
        reverse=False,
    )

    # Lower ser (if not up-down symmetric)
    if CoilCurrents_lower is not None:
        coils_lower, _ = FREEGSUparams.updateCoils(
            CoilCurrents_lower,
            dictDVs_lower,
            div1div=Params["div1div"],
            div2div=Params["div2div"],
            reverse=Params["outphaseForce"],
        )
    else:
        CoilCurrents_lower, coils_lower = CoilCurrents, coils

    # Equalizers
    coils, coils_lower = FREEGSUparams.equalizer(
        coils, coils_lower, coils_names, equalizers=Params["equalizers"]
    )
    # ----------------------------

    Params.update({"coils": coils, "coils_lower": coils_lower})

    num_cases = []
    for i in range(len(coils)):
        num_cases.append(
            1
            if (type(coils[i]) in [float, np.float64, int] or coils[i] is None)
            else len(coils[i])
        )
    num_cases = np.max(num_cases)

    # --------------------------------------------------------
    # Create baseline class (it will be updated for each inididual)
    # --------------------------------------------------------


    # From SPARC_PATH in PYTHONPATH
    try:
        from FREEGS_SPARC import GSsparc
    except ImportError:
        raise Exception(
            "[MITIM] The FREEGS_SPARC module is not available. Please ensure it is installed and accessible."
        )

    # Change params per individual?

    ParamsAll = []
    for cont in range(num_cases):
        ParamsThis = copy.deepcopy(Params)
        if paramsMods is not None:
            for i in paramsMods[cont]:
                ParamsThis[i] = paramsMods[cont][i]

        prfBase = GSsparc.GS_MITIM(
            folderData=IOtools.expandPath(folderData),
            n=ParamsThis["n"],
            printYN=printYN,
            plasmaParams=ParamsThis["plasmaParams"],
            profiles_settings=ParamsThis["profiles_settings"],
            fileWall=ParamsThis["fileWall"],
        )

        ParamsThis.update({"prfBase": prfBase})

        ParamsAll.append(ParamsThis)

    # --------------------------------------------------------
    # Run cases either in parallel or sequencial
    # --------------------------------------------------------

    if not onlyPrepare:
        if not internalParallel:
            mitims = []
            for cont in range(num_cases):
                prf1 = runSingleFreeGS(ParamsAll, cont)
                mitims.append(prf1)
        else:
            mitims = FARMINGtools.ParallelProcedure(
                runSingleFreeGS, ParamsAll, parallel=num_cases, howmany=num_cases
            )

        # --------------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------------

        metrics, opts = calculateMetrics(
            mitims, separatrixPoint=Params["separatrixPoint"]
        )

        # --------------------------------------------------------
        # Plotting options here
        # --------------------------------------------------------

        if plot and mitims is not None:
            FREEGSUplotting.plotResult(
                mitims,
                metrics,
                Constraints,
                coils_orig=CoilCurrents_orig,
                ProblemExtras=ProblemExtras,
                ParamsAll=ParamsAll,
                figs=figs,
            )

        return mitims, metrics, opts

    else:
        return ParamsAll


def separateUpperLowerDVs(dictDVs):
    dictDVs_upper, dictDVs_lower = {}, {}
    for ikey in dictDVs:
        if "_l" in ikey:
            dictDVs_lower[ikey.split("_")[0]] = dictDVs[ikey]
        else:
            dictDVs_upper[ikey] = dictDVs[ikey]

    return dictDVs_upper, dictDVs_lower


def calculateSweepPowerTraces(Matrix_file, coils, coils_terminal, t):
    df = pandas.read_excel(IOtools.expandPath(Matrix_file), sheet_name=1)
    ncoils = len(df["Unnamed: 0"].values)

    # ------ Order coils and stuff so that they correspond to the same coil I'm talking about

    coil_names, Iturns, I, M = [], [], [], np.zeros((ncoils, ncoils))
    coils_new, coils_terminal_new = OrderedDict(), OrderedDict()
    printtx = "Calculating powers from system of coils:"
    for i in range(ncoils):
        # Coil to grab stuff
        namecol = df["Unnamed: 0"].values[i]
        printtx += f" {namecol.lower()},"
        # Grab Matrix by calling that column
        M[i, :] = df[namecol].values
        # Grab coil currents for my case, but looking in the coil dictionaries for those names
        I.append(coils_terminal[namecol.lower()])
        Iturns.append(coils[namecol.lower()])
        # Store what coils I have used
        coil_names.append(namecol.lower())
        coils_new[namecol.lower()] = coils[namecol.lower()]
        coils_terminal_new[namecol.lower()] = coils_terminal[namecol.lower()]
    I = np.array(I)
    Iturns = np.array(Iturns)

    print(printtx[:-1])

    # ------ Calculate
    V, Power, _ = PLASMAtools.calculatePowerDrawn(t, I, M)
    _, _, dIdt = PLASMAtools.calculatePowerDrawn(t, Iturns, M)

    # Convert to dictionaries
    Power1, V1, dIdt1 = OrderedDict(), OrderedDict(), OrderedDict()
    for i in range(ncoils):
        Power1[coil_names[i]] = np.array(Power[i])
        V1[coil_names[i]] = np.array(V[i])
        dIdt1[coil_names[i]] = np.array(dIdt[i])

    return coils_new, coils_terminal_new, Power1, V1, dIdt1


def calculatePowerQuantities(metrics, ProblemExtras):
    # Add plasma current to coils set
    metrics["coils"]["plasma"] = metrics["Ip"]
    metrics["coils_terminal"]["plasma"] = metrics["coils"]["plasma"] * 1e6

    # Repeat for a number of times
    coils, t = makePeriodic(metrics["coils"], ProblemExtras["times"])
    coils_terminal, t = makePeriodic(metrics["coils_terminal"], ProblemExtras["times"])

    # Calculate quantities
    coils, coils_terminal, Power, V, dIdt = calculateSweepPowerTraces(
        ProblemExtras["InductanceMatrix"], coils, coils_terminal, t
    )
    # ----------------------

    # Read ramp-up
    if "RampUpFile" in ProblemExtras:
        maxVar_dIdT, maxVar_V, maxVar_P = readCoilCalcsMatrices_RampUp(
            ProblemExtras["RampUpFile"]
        )
    else:
        maxVar_dIdT = maxVar_V = maxVar_P = None

    RampUp_limits = [maxVar_dIdT, maxVar_V, maxVar_P]

    # Read supplies team requirements
    if "RequirementsFile" in ProblemExtras:
        maxVar_I, maxVar_V, minVar_I, minVar_V = readCoilCalcsMatrices_Supplies(
            ProblemExtras["RequirementsFile"]
        )
    else:
        maxVar_I = maxVar_V = minVar_I = minVar_V = None

    Supplies_limits = [maxVar_I, maxVar_V, minVar_I, minVar_V]

    return t, coils, coils_terminal, Power, V, dIdt, RampUp_limits, Supplies_limits


def readCoilCalcsMatrices_RampUp(file):
    df = pandas.read_excel(file)
    maxVar_dIdT, maxVar_V, maxVar_P = {}, {}, {}
    for i in df:
        if "Unnamed" not in i and i[-1] == "U":
            name = i[:-1].lower()
            maxVar_dIdT[name] = df[i][1]
            maxVar_P[name] = df[i][2]
            maxVar_V[name] = df[i][3]

    maxVar_dIdT["vs1"], maxVar_V["vs1"], maxVar_P["vs1"] = 0, 0, 0

    # Max dIdT in MA-t/s, Max voltage in kV, Max power in MVA = MW
    return maxVar_dIdT, maxVar_V, maxVar_P


def readCoilCalcsMatrices_Supplies_old(file):
    df = pandas.read_excel(file, sheet_name="PRD_allChanges")
    maxVar_I, maxVar_V = {}, {}
    minVar_I, minVar_V = {}, {}
    for i in df:
        if "Unnamed" in i:
            coil_n = df[i][0]
            if coil_n[-1] == "U":
                name = coil_n[:-1].lower()
                maxVar_I[name] = df[i][2]
                maxVar_V[name] = df[i][12]
                minVar_I[name] = df[i][1]
                minVar_V[name] = df[i][11]

    maxVar_I["vs1"] = maxVar_V["vs1"] = minVar_I["vs1"] = minVar_V["vs1"] = 0
    maxVar_I["dv1"] = minVar_I["dv1"] = maxVar_I["dv2"] = minVar_I["dv2"] = 30.0

    # Max dIdT in MA-t/s, Max voltage in kV, Max power in MVA = MW
    return maxVar_I, maxVar_V, minVar_I, minVar_V


def readCoilCalcsMatrices_Supplies(file):
    df = pandas.read_excel(IOtools.expandPath(file))
    maxVar_I, maxVar_V = {}, {}
    minVar_I, minVar_V = {}, {}
    for i in df:
        if "Unnamed" in i:
            coil_n = df[i][0]
            if coil_n[-1] == "U":
                name = coil_n[:-1].lower()
                minVar_I[name] = df[i][1]
                minVar_V[name] = df[i][6]
                maxVar_I[name] = df[i][2]
                maxVar_V[name] = df[i][7]

    maxVar_I["vs1"] = maxVar_V["vs1"] = minVar_I["vs1"] = minVar_V["vs1"] = 0

    # Max dIdT in MA-t/s, Max voltage in kV, Max power in MVA = MW
    return maxVar_I, maxVar_V, minVar_I, minVar_V


def makePeriodic(coils_orig, t_orig, n=10, nEach=100):
    """
    Repeats t_orig (and corresponding coils_orig) a given number n of times
    """

    t, coils = copy.deepcopy(t_orig), copy.deepcopy(coils_orig)

    tstart = t[0]
    tO = np.array(t) - tstart

    # Complete period
    t = np.linspace(tO[0], tO[-1], nEach)
    for i in coils:
        coil = np.interp(
            t, tO, coils[i]
        )  #    np.linspace(coils[i][0],coils[i][-1],1000)
        coils[i] = np.concatenate((coil, np.flipud(coil)[1:]))
    t = np.concatenate((t, t[1:] + t[-1]))

    # Repeat previous perid
    t_new, coils_new = copy.deepcopy(t), copy.deepcopy(coils)
    for j in range(n - 1):
        t_new = np.concatenate((t_new, t[1:] + t_new[-1]))
        for i in coils_new:
            coils_new[i] = np.concatenate((coils_new[i], coils[i][1:]))

    return coils_new, t_new


def calculateMetrics(mitims, separatrixPoint=None):
    # -----
    # CALCULATE METRICS
    # Note that the PFs MA-turns sum both Upper and Lower
    # -----

    coils, coils_terminal = {}, {}
    for label in mitims[0].sparc_coils.coils_final_summary:
        coils[label] = []
        coils_terminal[label] = []

    xpoints_up, xpoints_low, midplane_sepin, midplane_sepout = [], [], [], []
    xpoints_xpt = []
    strike_inR, strike_outR, strike_inZ, strike_outZ = [], [], [], []
    strike_inAngle, strike_outAngle = [], []
    Rmajors, aminors, elongs, triangs = [], [], [], []
    Ip = []
    opoints = []
    vs_mag = []
    li1, li2, li3 = [], [], []
    betaN = []
    squareness, squarenessU, squarenessL = [], [], []
    elongsA = []

    kappaU, kappaL, kappaM = [], [], []
    deltaU, deltaL, deltaM = [], [], []

    q0, q95, qstar = [], [], []

    ShafParam, ShafShift = [], []

    Zmag = []

    separatrixPoints = []

    for cont, prf in enumerate(mitims):
        for label in mitim = sparc_coils.coils_final_summary:
            coils[label].append(mitim = sparc_coils.coils_final_summary[label])
            coils_terminal[label].append(
                mitim = sparc_coils.coils_final_summary_terminal[label]
            )

        """
        What of the x-points is which? See GSsparc order x-points
        """
        upNum, lowNum = mitim = sweep.xPoint_up, mitim = sweep.xPoint_low
        if len(mitim = sweep.xPointOut) > 2:
            sndary = mitim = sweep.xPointOut[2]
        else:
            sndary = [np.inf] * 3

        opoint = mitim = sweep.oPointOut[0]

        opoints.append(opoint)
        xpoints_up.append(upNum)
        xpoints_low.append(lowNum)
        xpoints_xpt.append(sndary)
        midplane_sepin.append(mitim = sweep.sepRmin)
        midplane_sepout.append(mitim = sweep.sepRmax)

        Zmag.append(mitim = sweep.Zmag)

        if mitim = sweep.RstrikeInner is not None:
            strike_inR.append(mitim = sweep.RstrikeInner)
        else:
            strike_inR.append(np.inf)
        if mitim = sweep.RstrikeOuter is not None:
            strike_outR.append(mitim = sweep.RstrikeOuter)
        else:
            strike_outR.append(np.inf)
        if mitim = sweep.ZstrikeInner is not None:
            strike_inZ.append(mitim = sweep.ZstrikeInner)
        else:
            strike_inZ.append(np.inf)
        if mitim = sweep.ZstrikeOuter is not None:
            strike_outZ.append(mitim = sweep.ZstrikeOuter)
        else:
            strike_outZ.append(np.inf)
        if mitim = sweep.AstrikeOuter is not None:
            strike_outAngle.append(mitim = sweep.AstrikeOuter)
        else:
            strike_outAngle.append(np.inf)
        if mitim = sweep.AstrikeInner is not None:
            strike_inAngle.append(mitim = sweep.AstrikeInner)
        else:
            strike_inAngle.append(np.inf)

        Rmajors.append(mitim = sweep.Rgeo)
        aminors.append(mitim = sweep.aMinor)
        elongs.append(mitim = sweep.elongationX)
        elongsA.append(mitim = sweep.elongationA)
        triangs.append(mitim = sweep.triangularityX)

        kappaU.append(mitim = sweep.elongationUp)
        kappaL.append(mitim = sweep.elongationLow)
        kappaM.append(mitim = sweep.elongation)

        deltaU.append(mitim = sweep.triangularityUp)
        deltaL.append(mitim = sweep.triangularityLow)
        deltaM.append(mitim = sweep.triangularity)

        q0.append(np.interp(0, mitim = profiles["psinorm"], mitim = profiles["qpsi"]))
        q95.append(np.interp(0.95, mitim = profiles["psinorm"], mitim = profiles["qpsi"]))
        qstar.append(
            PLASMAtools.evaluate_qstar(
                mitim = sweep.Ip * 1e-6,
                mitim = sweep.Rgeo,
                mitim = sweep.elongation,
                mitim = sweep.BR / mitim = sweep.Rgeo,
                mitim = sweep.aMinor / mitim = sweep.Rgeo,
                mitim = sweep.triangularity,
                isInputIp=True,
            )
        )

        squareness.append(mitim = sweep.squarenessM)
        squarenessU.append(mitim = sweep.squarenessU)
        squarenessL.append(mitim = sweep.squarenessL)

        # Ip.append(mitim = IpIn*1E-6)
        Ip.append(mitim = profiles["Ip"] * 1e-6)

        # VS magnitude total (MA-t)
        VS1U = mitim = sparc_coils.coils_final_summary_terminal["vs1u"] * 1e-3  # kA
        VS1L = mitim = sparc_coils.coils_final_summary_terminal["vs1l"] * 1e-3  # kA
        VS1 = np.abs(VS1U) + np.abs(VS1L)
        vs_mag.append(VS1)

        # Inductances
        li1.append(mitim = sweep.li1)
        li2.append(mitim = sweep.li2)
        li3.append(mitim = sweep.li3)
        betaN.append(mitim = sweep.betaN)

        ShafShift.append(mitim = sweep.ShafShift)
        ShafParam.append(mitim = sweep.ShafParam)

        if separatrixPoint is not None:
            """
            Looks for separatrix value in the lower outer midplane
            """
            R, Z = mitim = sweep.sepR, mitim = sweep.sepZ
            R0, Z0 = R[R > mitim = sweep.Rmag], Z[R > mitim = sweep.Rmag]
            R1, Z1 = R0[Z0 < 0], Z0[Z0 < 0]

            if Z1[0] > Z1[-1]:
                R1, Z1 = np.flipud(R1), np.flipud(Z1)

            zval = np.interp(separatrixPoint, R1, Z1)

            separatrixPoints.append(zval)

        else:
            separatrixPoints.append(0)

    xpoints_up = np.array(xpoints_up)
    xpoints_low = np.array(xpoints_low)
    xpoints_xpt = np.array(xpoints_xpt)
    opoints = np.array(opoints)
    squareness = np.array(squareness)
    squarenessU = np.array(squarenessU)
    squarenessL = np.array(squarenessL)

    separatrixPoints = np.array(separatrixPoints)

    Zmag = np.array(Zmag)

    ShafParam = np.array(ShafParam)
    ShafShift = np.array(ShafShift)

    li1 = np.array(li1)
    li2 = np.array(li2)
    li3 = np.array(li3)
    betaN = np.array(betaN)

    Ip = np.array(Ip)

    q0 = np.array(q0)
    q95 = np.array(q95)
    qstar = np.array(qstar)

    elongsA = np.array(elongsA)
    kappaU = np.array(kappaU)
    kappaL = np.array(kappaL)
    kappaM = np.array(kappaM)

    deltaU = np.array(deltaU)
    deltaL = np.array(deltaL)
    deltaM = np.array(deltaM)

    vs_mag = np.array(vs_mag)

    Rmajors = np.array(Rmajors)
    aminors = np.array(aminors)

    strike_inR = np.array(strike_inR)
    strike_outR = np.array(strike_outR)
    strike_inZ = np.array(strike_inZ)
    strike_inZ = np.array(strike_inZ)

    strike_outAngle = np.array(strike_outAngle)
    strike_inAngle = np.array(strike_inAngle)

    for label in coils:
        coils[label] = np.array(coils[label])
        coils_terminal[label] = np.array(coils_terminal[label])

    # --------------------------------
    metrics = {
        "coils": coils,
        "coils_terminal": coils_terminal,
        "xpoints_up": xpoints_up,
        "xpoints_low": xpoints_low,
        "xpoints_xpt": xpoints_xpt,
        "opoints": opoints,
        "midplane_sepin": midplane_sepin,
        "midplane_sepout": midplane_sepout,
        "strike_inR": strike_inR,
        "strike_outR": strike_outR,
        "strike_inZ": strike_inZ,
        "strike_outZ": strike_outZ,
        "strike_outAngle": strike_outAngle,
        "strike_inAngle": strike_inAngle,
        "Rmajors": Rmajors,
        "aminors": aminors,
        "elongs": elongs,
        "kappaU": kappaU,
        "kappaL": kappaL,
        "kappaM": kappaM,
        "elongsA": elongsA,
        "triangs": triangs,
        "deltaU": deltaU,
        "deltaL": deltaL,
        "deltaM": deltaM,
        "Ip": Ip,
        "vs_mag": vs_mag,
        "li1": li1,
        "li2": li2,
        "li3": li3,
        "betaN": betaN,
        "zetaM": squareness,
        "zetaU": squarenessU,
        "zetaL": squarenessL,
        "q0": q0,
        "q95": q95,
        "qstar": qstar,
        "ShafParam": ShafParam,
        "ShafShift": ShafShift,
        "Zmag": Zmag,
        "separatrixPoints": separatrixPoints,
    }
    # --------------------------------

    # Calculate values for OFs optimization

    val = {}

    if metrics is not None:
        for i in range(len(metrics["midplane_sepout"])):
            val[f"xpR_{i + 1}"] = metrics["xpoints_up"][i][0] * 1e2  # in cm
            val[f"xpZ_{i + 1}"] = metrics["xpoints_up"][i][1] * 1e2  # in cm
            val[f"xpF_{i + 1}"] = metrics["xpoints_up"][i][2]  # *4*np.pi
            val[f"xpdRsep_{i + 1}"] = metrics["xpoints_up"][i][3]  # in cm
            val[f"mpo_{i + 1}"] = metrics["midplane_sepout"][i] * 100
            val[f"mpi_{i + 1}"] = metrics["midplane_sepin"][i] * 100
            val[f"stiR_{i + 1}"] = metrics["strike_inR"][i] * 100
            val[f"stiZ_{i + 1}"] = metrics["strike_inZ"][i] * 100
            val[f"stoR_{i + 1}"] = metrics["strike_outR"][i] * 100
            val[f"stoZ_{i + 1}"] = metrics["strike_outZ"][i] * 100
            val[f"stoA_{i + 1}"] = metrics["strike_outAngle"][i]
            val[f"xpRlow_{i + 1}"] = metrics["xpoints_low"][i][0] * 100
            val[f"xpZlow_{i + 1}"] = metrics["xpoints_low"][i][1] * 100
            val[f"xpdRseplow_{i + 1}"] = metrics["xpoints_low"][i][3]
            val[f"xpRxpt_{i + 1}"] = metrics["xpoints_xpt"][i][0] * 1e2  # in cm
            val[f"xpZxpt_{i + 1}"] = metrics["xpoints_xpt"][i][1] * 1e2  # in cm
            val[f"squareness_{i + 1}"] = metrics["zetaM"][i]
            val[f"zmag_{i + 1}"] = metrics["Zmag"][i] * 1e2  # in cm
            val[f"vs_mag_{i + 1}"] = (
                metrics["vs_mag"][i] * 1e-1
            )  # in tens of kA. So that 1kA corresponds to 1mm

            val[f"sepZ_{i + 1}"] = metrics["separatrixPoints"][i] * 1e2  # in cm

    else:
        for i in range(len(metrics["midplane_sepout"])):
            val[f"xpR_{i + 1}"] = np.inf
            val[f"xpZ_{i + 1}"] = np.inf
            val[f"xpF_{i + 1}"] = np.inf
            val[f"xpdRsep_{i + 1}"] = np.inf
            val[f"mpo_{i + 1}"] = np.inf
            val[f"mpi_{i + 1}"] = np.inf
            val[f"stiR_{i + 1}"] = np.inf
            val[f"stiZ_{i + 1}"] = np.inf
            val[f"stoR_{i + 1}"] = np.inf
            val[f"stoZ_{i + 1}"] = np.inf
            val[f"stoA_{i + 1}"] = np.inf
            val[f"xpRlow_{i + 1}"] = np.inf
            val[f"xpZlow_{i + 1}"] = np.inf
            val[f"xpRxpt_{i + 1}"] = np.inf
            val[f"xpZxpt_{i + 1}"] = np.inf
            val[f"xpdRseplow_{i + 1}"] = np.inf
            val["squareness"] = np.inf
            val[f"vs_mag_{i + 1}"] = np.inf
            val[f"sepZ_{i + 1}"] = np.inf

    return metrics, val


def lambda_from_vectors(x, y, kind="cubic", ensure01=True):
    if ensure01:
        x_old = copy.deepcopy(x)
        x = np.linspace(0, 1, 500)
        y = np.interp(x, x_old, y)

    f = interp1d(x, y, kind=kind)
    return f

def calculateDivertorPlateCoverage(metrics, limRZ, pointsDiv=[3, 17], plotYN=False):
    Zo = metrics["strike_outZ"]
    Zi = metrics["strike_inZ"]
    Ro = metrics["strike_outR"]
    Ri = metrics["strike_inR"]
    Ao = metrics["strike_outAngle"]
    Ai = metrics["strike_inAngle"]

    xo = np.transpose(np.vstack((Ro, Zo)))
    xi = np.transpose(np.vstack((Ri, Zi)))

    iS = pointsDiv[0]
    iF = pointsDiv[1]
    di = MATHtools.calculateDistanceAlong(xi, limRZ, iS=iS)
    do = MATHtools.calculateDistanceAlong(xo, limRZ, iS=iS)

    limRZ_d = MATHtools.calculateDistanceAlong(limRZ[iS:iF], limRZ, iS=iS)

    if plotYN:
        plt.ion()
        fig = plt.figure(figsize=(18, 5))
        grid = plt.GridSpec(ncols=3, nrows=1, hspace=0.6, wspace=0.4)
        ax = fig.add_subplot(grid[0])

        ax.plot(limRZ[:, 0], limRZ[:, 1], "o-", color="k", markersize=5)
        ax.plot(xo[:, 0], xo[:, 1], "s-", color="r", markersize=5)
        ax.plot(xi[:, 0], xi[:, 1], "s-", color="g", markersize=5)
        ax.plot([limRZ[iS, 0]], [limRZ[iS, 1]], "s", color="b", markersize=10)
        ax.plot([limRZ[iF, 0]], [limRZ[iF, 1]], "s", color="m", markersize=10)
        ax.set_xlim([1.2, 1.9])
        ax.set_ylim([1.0, 1.6])
        ax.set_aspect("equal")

        ax = fig.add_subplot(grid[1:])
        ax.plot(
            limRZ_d, ["divertor points"] * len(limRZ_d), "o-", color="k", markersize=5
        )
        ax.plot(do, ["outer strike point"] * len(do), "s-", color="r", markersize=5)
        for i in do:
            ax.axvline(x=i, ls="--", lw=0.5, color="r")
        ax.plot(di, ["inner strike point"] * len(di), "s-", color="g", markersize=5)
        for i in di:
            ax.axvline(x=i, ls="--", lw=0.5, color="g")

        ax.plot([limRZ_d[0]], ["divertor points"], "s", color="b", markersize=10)
        ax.plot([limRZ_d[-1]], ["divertor points"], "s", color="m", markersize=10)

        ax.set_xlabel("Distance along divertor plates (m)")

    return limRZ_d, di, do


def curateGSoptions(Params, ProblemExtras):
    if "outphaseForce" not in Params:
        Params["outphaseForce"] = False
    if "profiles_settings" not in Params:
        Params["profiles_settings"] = {
            "type": "pIp",
            "args": {"alpha_m": 2.0, "alpha_n": 2.0},
        }
    if "outMidplane_matching" not in Params:
        Params["outMidplane_matching"] = None
    if "xpoints2" not in Params:
        Params["xpoints2"] = None
    if "equalizers" not in Params:
        Params["equalizers"] = {}
    if "separatrixPoint" not in Params:
        Params["separatrixPoint"] = None
    if "fileWall" not in Params:
        Params["fileWall"] = "reinke_20211213/v2b_t4slope_v3.csv"
    if "csForceRatio" not in Params:
        Params["csForceRatio"] = {
            "cs1u": None,
            "cs2u": None,
            "cs3u": None,
            "cs1l": None,
            "cs2l": None,
            "cs3l": None,
        }

    if "div1div" not in Params:
        Params["div1div"] = [0.25] * 4
    if "div2div" not in Params:
        Params["div2div"] = [0.25] * 4

    if "div2equal1" not in Params:
        Params["div2equal1"] = 0
    if "cs3equal2" not in Params:
        Params["cs3equal2"] = 0

    if ProblemExtras is None:
        ProblemExtras = {"times": [0]}
    if "times" not in ProblemExtras:
        ProblemExtras["times"] = [0]

    if "sortbyXpoints" not in Params:
        Params["sortbyXpoints"] = "flux"

    return Params, ProblemExtras


def makePeriodic_metrics(metrics):
    torig = copy.deepcopy(metrics["params"]["times"])
    for i in metrics:
        if i not in ["mitims", "params"]:
            if i in ["xpoints_up", "xpoints_low", "opoints", "xpoints_xpt"]:
                mn = []
                for j in range(metrics[i].shape[1]):
                    mm = metrics[i][:, j]
                    mm, metrics["params"]["times"] = makePeriodic({"a": mm}, torig, n=1)
                    mn.append(mm["a"])
                metrics[i] = np.transpose(np.array(mn))
            elif type(metrics[i]) in [np.ndarray, list]:
                mm = {"a": metrics[i]}
                mm, metrics["params"]["times"] = makePeriodic(mm, torig, n=1)
                metrics[i] = mm["a"]
            else:
                metrics[i], metrics["params"]["times"] = makePeriodic(
                    metrics[i], torig, n=1
                )

    return metrics


def runSingleFreeGS(ParamsAll, cont):
    Params = ParamsAll[cont]

    # --------------------------------------------------
    # Routine inputs
    # --------------------------------------------------

    (
        cs1,
        cs2,
        cs3,
        pf1,
        pf2,
        pf3,
        pf4,
        dv1a,
        dv1b,
        dv1c,
        dv1d,
        dv2a,
        dv2b,
        dv2c,
        dv2d,
        vs1,
    ) = Params["coils"]
    (
        cs1_l,
        cs2_l,
        cs3_l,
        pf1_l,
        pf2_l,
        pf3_l,
        pf4_l,
        dv1a_l,
        dv1b_l,
        dv1c_l,
        dv1d_l,
        dv2a_l,
        dv2b_l,
        dv2c_l,
        dv2d_l,
        vs1_l,
    ) = Params["coils_lower"]
    if len(Params["xpoints"]) == 1:
        xpoint = Params["xpoints"][0]
    else:
        xpoint = Params["xpoints"][cont]
    if "xpoints2" in Params and Params["xpoints2"] is not None:
        if len(Params["xpoints2"]) == 1:
            xpoint2 = Params["xpoints2"][0]
        else:
            xpoint2 = Params["xpoints2"][cont]
    else:
        xpoint2 = None

    n = Params["n"]
    printYN = Params["printYN"]
    prfBase = Params["prfBase"]
    midplaneR = Params["midplaneR"]
    debug = Params["debug"]
    onlyUpperPlate = Params["onlyUpperPlate"]
    coilsVersion = Params["coilsVersion"]
    csForceRatio = Params["csForceRatio"]
    outMidplane_matching = Params[
        "outMidplane_matching"
    ]  # Distance from outer midplane (+) to match flux for x-points (useful for LSN)
    symmetricX = Params["symmetricX"]
    sortbyXpoints = Params["sortbyXpoints"]

    if "extraSeparatrixPoint" in Params:
        extraSeparatrixPoint = Params["extraSeparatrixPoint"]
    else:
        extraSeparatrixPoint = None

    if "psiGuess" in Params:
        psiGuess = Params["psiGuess"]
    else:
        psiGuess = None

    if printYN:
        print("\n--> Finding solution with {0}x{0} grid".format(n))

    prf1 = copy.deepcopy(prfBase)

    """
    --------------------------------------------------
    Create coils scalars and create class
      The class contains now all info about the coils and the transformations into
      individual (in/out, up/lower), which didn't have before
    --------------------------------------------------
    """

    # ---------------
    # --- Upper Coils
    # ---------------

    # Convert from array/list to a single value, because this run is individual FREEGS
    coilsSingle = FREEGSUparams.assignCoils(
        cont,
        cs1,
        cs2,
        cs3,
        pf1,
        pf2,
        pf3,
        pf4,
        dv1a,
        dv1b,
        dv1c,
        dv1d,
        dv2a,
        dv2b,
        dv2c,
        dv2d,
        vs1,
    )

    # From SPARC_PATH in PYTHONPATH
    try:
        from FREEGS_SPARC import GSsparc_coils
    except ImportError:
        raise Exception(
            "[MITIM] The FREEGS_SPARC module is not available. Please ensure it is installed and accessible."
        )

    # Assign to class
    prf1.sparc_coils = GSsparc_coils.SPARCcoils(
        coilsSingle, coilsVersion=coilsVersion, csForceRatio=csForceRatio
    )

    # ---------------
    # --- Lower Coils
    # ---------------

    # Convert from array/list to a single value, because this run is individual FREEGS
    coilsSingle_l = FREEGSUparams.assignCoils(
        cont,
        cs1_l,
        cs2_l,
        cs3_l,
        pf1_l,
        pf2_l,
        pf3_l,
        pf4_l,
        dv1a_l,
        dv1b_l,
        dv1c_l,
        dv1d_l,
        dv2a_l,
        dv2b_l,
        dv2c_l,
        dv2d_l,
        vs1_l,
    )

    # Change the lower coils
    prf1.sparc_coils.lowerCoils(coils=coilsSingle_l)

    # Re-process everything
    prf1.sparc_coils.process(coilsVersion)

    # Options for exact out of phase
    # if outPhase:
    #   cs10l,cs20l,cs30l,pf10l,pf20l,pf30l,pf40l,
    #   dv10al, dv10bl, dv10cl, dv10dl,
    #   dv20al, dv20bl, dv20cl, dv20dl = FREEGSUparams.assignCoils(len(xpoints)-cont-1,cs1,cs2,cs3,pf1,pf2,pf3,pf4,dv1a, dv1b, dv1c, dv1d,dv2a, dv2b, dv2c, dv2d)
    #   prf1.sparc_coils.lowerCoils(coils=(cs10l,cs20l,cs30l,pf10l,pf20l,pf30l,pf40l,dv10al, dv10bl, dv10cl, dv10dl,dv20al, dv20bl, dv20cl, dv20dl,0))
    #   lower = xpoints[len(xpoints)-cont-1]
    #   prf1.sparc_coils.process()

    # --------------------------------------------------
    # Add constraints
    # --------------------------------------------------

    if midplaneR is not None:
        outMidplane, inMidplane = [midplaneR[1], 0.0], [midplaneR[0], 0.0]
    else:
        outMidplane = inMidplane = None

    prf1.controlPoints(
        xpoint=xpoint,
        xpoint2=xpoint2,
        outMidplane=outMidplane,
        outMidplane_matching=outMidplane_matching,
        inMidplane=inMidplane,
        printYN=printYN,
        symmetricX=symmetricX,
        extraSeparatrixPoint=extraSeparatrixPoint,
    )

    # --------------------------------------------------
    # Actual equilibrium work
    # --------------------------------------------------

    time1 = datetime.datetime.now()

    prf1.evaluate(
        debug=debug,
        upDownSymmetric=False,
        printYN=printYN,
        psiGuess=psiGuess,
        sortbyXpoints=sortbyXpoints,
    )  # not outPhase)

    prf1.calculateProperties(onlyUpperPlate=onlyUpperPlate)
    print("\t\t~ FreeGS took: " + IOtools.getTimeDifference(time1))

    # --------------------------------------------------
    # Uptdate class with the currents finally selected (in case of FreeGS self-optimization)
    # --------------------------------------------------
    prf1.sparc_coils.grabFinalCurrents(prf1.sweep.coils)

    return prf1


# --------------------------------------------------------------------------------------------------------------
# To avoid changing freegs source code
# --------------------------------------------------------------------------------------------------------------


class Machine(freegs.machine.Machine):
    def __init__(self, coils, wall=None):
        super().__init__(coils, wall=wall)

    def controlAdjust(self, current_change):
        controlcoils = [coil for label, coil in self.coils if coil.control]
        controlcoils_labels = [label for label, coil in self.coils if coil.control]

        for name, coil, dI in zip(controlcoils_labels, controlcoils, current_change):
            print(
                "Current in {0} changed from {1:.2f}MA-t to {2:.2f}MA-t".format(
                    name, coil.current * 1e-6, coil.current * 1e-6 + dI.item() * 1e-6
                ),
            )
            coil.current += dI.item()


