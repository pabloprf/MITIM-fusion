import os, time, pdb, math, time, datetime
import numpy as np
from IPython import embed

from mitim_tools.transp_tools import UFILEStools, TRANSPtools
from mitim_tools.transp_tools.src import TRANSPglobus
from mitim_tools.transp_tools.tools import NMLtools
from mitim_tools.misc_tools import IOtools


def runTRANSP(
    mitimNML,
    stepTransition,
    checkForActive=True,
    outtims=[],
    phasetxt="",
    automaticProcess=False,
):
    # ---------------------------------------------------------------------------
    # Run Settings
    # ---------------------------------------------------------------------------

    if not mitimNML.PlasmaFeatures["ICH"]:
        mitimNML.TRANSPnamelist["NTORIC_PSERVE"], mitimNML.toricmpi = 0, 1
    if not mitimNML.PlasmaFeatures["MC"]:
        mitimNML.TRANSPnamelist["NBI_PSERVE"], mitimNML.trmpi = 0, 1

    nameRunTot = mitimNML.nameRunTot
    Reinitialize_q = mitimNML.Reinitialize_q
    tok = mitimNML.tok

    # ---------------------------------------------------------------------------
    # Change namelist
    # ---------------------------------------------------------------------------

    print(">> Changing namelist parameters...")
    NMLtools.changeNamelist(
        mitimNML.namelistPath,
        mitimNML.nameBaseShot,
        mitimNML.TRANSPnamelist,
        mitimNML.FolderTRANSP,
        outtims=outtims,
    )

    # Add updates (has to be at the end)
    mitimNML.timeStartPrediction = round(
        mitimNML.BaselineTime + mitimNML.PredictionOffset, 3
    )

    if mitimNML.timeStartPrediction < mitimNML.TRANSPnamelist["tinit"]:
        print(
            ">> Prediction update time was earlier than tinit... correcting to tinit+1ms"
        )
        mitimNML.timeStartPrediction = mitimNML.TRANSPnamelist["tinit"] + 0.001

    model = mitimNML.TransportModels[mitimNML.runPhaseNumber]

    if model is not None:
        print(f">> Appending predictive options with {model.upper()} model")
        NMLtools.appendUpdates(
            mitimNML.namelistPath,
            mitimNML.namelistPath_ptsolver,
            mitimNML.TRANSPnamelist,
            mitimNML.MITIMparams,
            mitimNML.timeStartPrediction,
            model,
            useAxialTrick=mitimNML.useAxialTrick,
            timeLagDensity=mitimNML.timeLagDensity,
            predictQuantities=mitimNML.predictQuantities,
            pedestalPred=mitimNML.pedestalPred,
            rotationPred=mitimNML.rotationPred,
        )

    else:
        print(">> Updates not appended because no transport model has been selected")

    # ---------------------------------------------------------------------------
    # Change UFILES values
    # ---------------------------------------------------------------------------

    timesStepTransition = [
        mitimNML.BaselineTime,
        mitimNML.BaselineTime + mitimNML.startRamp,
        mitimNML.BaselineTime + mitimNML.endRamp,
    ]
    # --- Selection of time to evaluate UFiles to select what is the starting point of ramp
    if mitimNML.restartTime[0] is not None:
        timeOriginal = mitimNML.restartTime[0]
    else:
        timeOriginal = 100.0
    # --------
    print(">> Changing RBZ and CUR UFILES...")
    UFILEStools.changeUFILEs(
        timesStepTransition,
        mitimNML.nameBaseShot,
        timeOriginal,
        mitimNML.FolderTRANSP,
        mitimNML.MITIMparams,
        stepTransition=stepTransition,
    )

    # ---------------------------------------------------------------------------
    # Antennas and Beams (deserve special treat)
    # ---------------------------------------------------------------------------

    print(">> Changing antennas and/or beams...")
    changeHardware(
        mitimNML.PlasmaFeatures,
        mitimNML.MITIMparams,
        mitimNML.timePower,
        mitimNML.FolderTRANSP,
        mitimNML.nameBaseShot,
        LH_time=mitimNML.LH_time,
        PowerOff=mitimNML.LH_valsL[1],
    )

    # ---------------------------------------------------------------------------
    # Execute TRANSP
    # ---------------------------------------------------------------------------

    mpisettings = {
        "trmpi": mitimNML.trmpi,
        "ptrmpi": mitimNML.ptrmpi,
        "toricmpi": mitimNML.toricmpi,
    }

    # Define TRANSP class
    transp_run = TRANSPtools.TRANSP(mitimNML.FolderTRANSP, mitimNML.tok)

    # Define run parameters
    transp_run.defineRunParameters(
        mitimNML.nameRunTot, mitimNML.nameBaseShot, mpisettings=mpisettings
    )

    # Convergence loop
    minWaitLook = mitimNML.minWaitLook
    convCriteria = mitimNML.convCrit
    FolderOutputs = mitimNML.FolderOutputs
    timeStartPrediction = mitimNML.timeStartPrediction

    HasItFailed = transp_run.automatic(
        convCriteria,
        version=mitimNML.version,
        minWait=minWaitLook,
        timeStartPrediction=timeStartPrediction,
        FolderOutputs=FolderOutputs,
        checkForActive=checkForActive,
        phasetxt=phasetxt,
        automaticProcess=automaticProcess,
        retrieveAC=True if len(outtims) > 0 else False,
    )

    return HasItFailed


def changeHardware(
    PlasmaFeatures,
    MITIMparams,
    timePower,
    FolderTRANSP,
    nameBaseShot,
    LH_time=None,
    PowerOff=0.0,
):
    for feature in ["ICH", "ECH", "NBI"]:
        if feature == "ICH":
            Power, name1, name2 = "Pich", "RFP", "rfp"
        if feature == "ECH":
            Power, name1, name2 = "Pech", "ECH", "ecp"
        if feature == "NBI":
            Power, name1, name2 = "Pnbi", "NB2", "nb2"

        Pich = MITIMparams[Power]
        if LH_time is not None:
            print(
                " --> LH transition happens within the discharge, adjusting powers..."
            )
            PichOff = np.ones(len(Pich)) * PowerOff
            timePower = LH_time
        else:
            PichOff = None
            timePower = timePower

        timesOn = np.repeat([[timePower, 100.0]], len(Pich), axis=0)
        fileUF = FolderTRANSP + f"PRF{nameBaseShot}.{name1}"
        if PlasmaFeatures[feature]:
            print(f"\t- Changing {feature} hardware...")
            UFILEStools.writeAntenna(
                fileUF, timesOn, Pich, PowersMW_Before=PichOff, fromScratch=name2
            )
