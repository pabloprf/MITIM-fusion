import os
import time
import os.path
import datetime
import re
import numpy as np
from collections import OrderedDict
from IPython import embed
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.transp_tools import CDFtools, TRANSPtools
from mitim_tools.transp_tools.src import TRANSPglobus_ntcc
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print



##################################################
# TRANSP error avoidance
##################################################
timewaitSendRequest = 10.0
attemptsSendRequest = 5
timewaitFailedLook = 60.0 * 5
attemptsSendLook = 3
timewaitRetrieveLook = (
    60.0 * 3
)  # Time to wait to check again if the "look" file is ready
attemptsRetrieveLook = 10
timewaitNotStarted = 120.0
maxSendTrialsAfterNotFound = 2
minWaitPingWebsite = 5  # Time to wait between two consecutive website pings
##################################################

"""
------------------------------------------------------------------------------------------------------
TRANSP class to run through globus, inherits main stuff from the parent class, common with singularity
------------------------------------------------------------------------------------------------------
"""


class TRANSPglobus(TRANSPtools.TRANSPgeneric):
    def __init__(self, FolderTRANSP, tokamak):
        super().__init__(FolderTRANSP, tokamak)

    def defineRunParameters(self, *args, **kwargs):
        super().defineRunParameters(*args, **kwargs)

        s = CONFIGread.load_settings()
        self.pppluser = s["globus"]["username"]
        self.email = s["globus"]["email"]

        # I want to understand where the run goes in the PPPL grid. Note that fetching server needs year, so define later
        self.folderGRID = f"transpgrid.pppl.gov/u/tr_{self.pppluser}/transp/"
        self.folderGRID_look = f"{self.folderGRID}/look/{self.tok}/"

    """
	------------------------------------------------------------------------------------------------------
		Main routines
	------------------------------------------------------------------------------------------------------
	"""

    def run(self, version="pshare", **kwargs):
        # Make sure that the MPIs are set up properly
        self.mpisettings = TRANSPtools.ensureMPIcompatibility(
            self.nml_file, self.nml_file_ptsolver, self.mpisettings
        )

        # tr_start
        self.start(version=version)

        # tr_send
        self.send()

    def check(self, permitSomeTime=False, NBImissing_isStopped=True, **kwargs):
        permitSomeTime = False  # False has been HARD CODED for tests

        # ------------------------------------------------------------------------------------------------------------------------------
        # Some dynamics to deal with the server pings not been refreshed with enough time resolution
        # ------------------------------------------------------------------------------------------------------------------------------
        if permitSomeTime:
            print(
                f"\t* Waiting {minWaitPingWebsite}min because that seems to be the time allowed to ping the server",
                typeMsg="w",
            )
            time.sleep(minWaitPingWebsite * 60)
        else:
            print(
                f"\t* Remember that the server does not refresh constant pings, it is recommended to wait ~{minWaitPingWebsite}min",
                typeMsg="w",
            )
        # ------------------------------------------------------------------------------------------------------------------------------

        info, status, infoGrid = getGridInformation(
            self.runid, project=self.tok, NBImissing_isStopped=NBImissing_isStopped
        )
        try:
            self.yearRun = info["info"]["info_launch"]["year"]
        except:
            self.yearRun = None
            print(
                "\t- Year associated with the run could not be retrieved at this moment",
                typeMsg="w",
            )

        # Build the folderGRID result once I know the year, so update here
        if self.yearRun is not None:
            self.folderGRID_result = "{0}/result/{1}.{2}/".format(
                self.folderGRID, self.tok, self.yearRun
            )

        return info, status, infoGrid

    def get(
        self,
        label="run1",
        retrieveAC=False,
        fullRequest=True,
        checkForActive=True,
        **kwargs,
    ):
        """
        This is for unfinished runs (tr_look), grab only the CDF for checks
        """

        file = f"{self.runid}.CDF"

        if fullRequest:
            # ------------------------------------------------------------------------------------------------------------------------------
            # LOOP LOOK
            # ------------------------------------------------------------------------------------------------------------------------------

            if self.FolderTRANSP[-1] != "/":
                self.FolderTRANSP += "/"

            NBImissing_isStopped = (
                False  # If TRANSP grid indicates missing_nbi that means stopped
            )

            print(
                f">> Starting process of grabbing files of {self.runid} and asessing status"
            )

            # ---------------------
            # Remove old .CDF file
            # ---------------------

            old_file = f"{self.FolderTRANSP}/{self.runid}.CDF"
            if os.path.exists(old_file):
                os.system("mv {0} {0}_prev".format(old_file))

            # ---------------------
            # Send look request (tr_look)
            # ---------------------

            TRANSPglobus_ntcc.tr_look(self.FolderTRANSP, self.runid, self.tok)

            # ---------------------
            # Retrieve new .CDF file
            # ---------------------

            statusLook, self.statusStop, f, timeReSentLook, timeWaitLook = (
                False,
                0,
                [],
                0,
                0,
            )

            file = f"{self.runid}.CDF"

            while not statusLook:
                # ---------------------
                # Let's check what's the status
                # ---------------------
                if checkForActive:
                    dictInfo, _, _ = self.check(
                        NBImissing_isStopped=NBImissing_isStopped, permitSomeTime=True
                    )
                else:
                    dictInfo = {
                        "info": {"status": "running"},
                        "look": {"status": "generated"},
                    }
                    time.sleep(60 * 5)

                # ---------------------
                # If run is on the grid, analyze status
                # ---------------------
                if dictInfo["info"]["status"] != "not found":
                    # ---------------------
                    # If run is "missing_NBI", it means stopped, but TRANSP team may restart it if they realize
                    # ---------------------
                    NBImissing_wait = 3600 * 2
                    if dictInfo["info"]["status"] == "missing_nbi":
                        print(
                            '\t- Run {0} is stopped by "time out", wait 3h for TRANSP team to restart'.format(
                                self.runid
                            ),
                            typeMsg="w",
                        )
                        try:
                            os.system(
                                "cd " + self.FolderTRANSP + " && cp *CDF_prev saved"
                            )
                            print(
                                f'\t- {self.runid} last CDF state backed up as "saved"'
                            )
                        except:
                            print(
                                "\t- {0} last CDF state could NOT be backed up".format(
                                    self.runid
                                ),
                                typeMsg="w",
                            )

                        TRANSPglobus_ntcc.tr_get(
                            file,
                            self.folderGRID_look,
                            self.runid,
                            self.FolderTRANSP,
                            self.tok,
                        )

                        print(f"\t- {self.runid} last CDF state could retrieved")
                        time.sleep(NBImissing_wait)

                    # ---------------------
                    # If run is stopped, get out of the loop to give a FAIL
                    # ---------------------

                    elif dictInfo["info"]["status"] == "stopped":
                        print(f">> {self.runid} run stopped", typeMsg="w")
                        statusLook, self.statusStop, f = True, 1, None

                        try:
                            TRANSPglobus_ntcc.tr_get(
                                file,
                                self.folderGRID_look,
                                self.runid,
                                self.FolderTRANSP,
                                self.tok,
                            )
                            print(f">> {self.runid} last CDF state could retrieved")
                        except:
                            print(
                                f">> {self.runid} last CDF state could NOT be retrieved",
                                typeMsg="w",
                            )

                        try:
                            print(
                                f">> {self.runid} run failed due to <{dictInfo['error']['status']}>. Recommendation: <{dictInfo['error']['recommendation']}>",
                                typeMsg="w",
                            )
                        except:
                            pass

                    # ---------------------
                    # If run is finished, get out of the loop to fetch
                    # ---------------------

                    elif dictInfo["info"]["status"] == "finished":
                        print(f">> {self.runid} run finished!")
                        statusLook, self.statusStop, f = True, -2, None

                    # ---------------------
                    # If run is submitted (not enough cores to get it started, just wait)
                    # ---------------------

                    elif dictInfo["info"]["status"] == "not started yet":
                        print(
                            f">> {self.runid} run in submitted state (not started yet), waiting {timewaitNotStarted/60.:.0f}min"
                        )
                        time.sleep(timewaitNotStarted)

                    # ---------------------
                    # If look request has not been request or it has (randomly?) failed, repeat tr_look, until assume stopped if too many attemps
                    # ---------------------

                    elif (
                        dictInfo["info"]["status"] != "canceled"
                        and dictInfo["look"]["status"] == "not sent"
                    ) or dictInfo["look"]["status"] == "failed":
                        if timeReSentLook < attemptsSendLook:
                            print(f">> {self.runid} Look failed, sending tr_look again")
                            TRANSPglobus_ntcc.tr_look(
                                self.FolderTRANSP, self.runid, self.tok
                            )
                            timeReSentLook = timeReSentLook + 1
                            print(
                                ">> Waiting {0}min before getting file".format(
                                    timewaitFailedLook / 60.0
                                )
                            )
                            time.sleep(timewaitFailedLook)
                        else:
                            # print(f' >> too many looks for run {self.runid}, assume stopped')
                            # statusLook, statusStop, f = True, 1, None
                            print(
                                f'>> Too many looks for run {self.runid}, assume "not found" and delete',
                                typeMsg="w",
                            )
                            statusLook, self.statusStop, f = True, -1, None
                            TRANSPglobus_ntcc.tr_cancel(
                                self.runid, self.FolderTRANSP, self.tok, howManyCancel=2
                            )

                    # ---------------------
                    # If .CDF has not been generated yet, wait a bit, until assume stopped if too many attemps
                    # ---------------------

                    elif dictInfo["look"]["status"] == "submitted":
                        if timeWaitLook < attemptsRetrieveLook:
                            print(
                                f'>> {self.runid} Look file not generated yet ("Submitted" state), waiting {timewaitRetrieveLook/60}min...'
                            )
                            time.sleep(timewaitRetrieveLook)
                            timeWaitLook = timeWaitLook + 1
                        else:
                            print(
                                f">> Waited for {self.runid} .CDF generation for too long, assume stopped",
                                typeMsg="w",
                            )
                            statusLook, self.statusStop, f = True, 10, None

                    # ---------------------
                    # If .CDF has been correctly generated, go next step to retrieve it
                    # ---------------------

                    elif dictInfo["look"]["status"] == "generated":
                        print(f"\t\t- {self.runid} Look generated")
                        statusLook = True

                else:
                    # If the run is not there, then I need do tr_send again
                    statusLook, self.statusStop, f = True, -1, None

            if f is not None:
                TRANSPglobus_ntcc.tr_get(
                    file, self.folderGRID_look, self.runid, self.FolderTRANSP, self.tok
                )

                netCDFfile = self.FolderTRANSP + f"{self.runid}.CDF"

                # Sometimes the file is corrupted, len(t) different from len(Te)
                try:
                    Reactor = CDFtools.transp_output(netCDFfile)
                except:
                    Reactor = corruptRecover(
                        self.FolderTRANSP, self.runid, self.tok, self.folderGRID_look
                    )

                Reactor.writeResults_TXT(
                    self.FolderTRANSP + "infoRun_preconvergence.dat", ensureBackUp=False
                )

                # ---- Possibility of AC files

                if retrieveAC:
                    ICRF, TORBEAM, NUBEAM = self.determineACs(Reactor)
                else:
                    ICRF = TORBEAM = NUBEAM = False

                retrieveACfiles(
                    self.runid,
                    self.FolderTRANSP,
                    self.tok,
                    self.folderGRID_look,
                    NUBEAM=NUBEAM,
                    ICRF=ICRF,
                    TORBEAM=TORBEAM,
                )

            else:
                Reactor = None

        # ------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------------------

        else:
            print(" ~ Retrieving file from server only, not submitting request again")
            self.grab(self.folderGRID_look, retrieveAC=retrieveAC)

            Reactor = TRANSPtools.storeCDF(
                self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
            )

        self.cdfs[label] = Reactor

    def fetch(self, label="run1", retrieveAC=False, **kwargs):
        """
        This is for finished runs (tr_fetch)
        """

        # Send a check so that I can grab the year, in case I didn't before
        info, status, infoGrid = self.check()

        # Retrieve files
        self.grab(self.folderGRID_result, retrieveAC=retrieveAC)

        # Read results
        self.cdfs[label] = TRANSPtools.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
        )

        return self.cdfs[label]

    def delete(self, howManyCancel=1, MinWaitDeletion=0, **kwargs):
        TRANSPglobus_ntcc.tr_cancel(
            self.runid,
            self.FolderTRANSP,
            self.tok,
            howManyCancel=howManyCancel,
            MinWaitDeletion=MinWaitDeletion,
        )

    def automatic(
        self,
        convCriteria,
        version="pshare",
        minWait=60,
        timeStartPrediction=0,
        FolderOutputs=None,
        checkForActive=True,
        phasetxt="",
        automaticProcess=False,
        retrieveAC=False,
    ):
        if FolderOutputs is None:
            FolderOutputs = self.FolderTRANSP

        print(f"** Running tr_start ({version}) for run {self.runid}...")
        self.start(version=version)

        trialSend, statusStop = 1, -1

        # If run is not found on the grid (-1: not found, 0: running, 1: stopped, -2: success)
        while statusStop == -1:
            print(f"** Running tr_send for run {self.runid}...")
            statusRun, repeatStart, trialSend = self.send(
                trialSend=trialSend, checkForActive=checkForActive
            )

            # Sometimes the file REQUEST is not created by random bugs, repeat process
            if repeatStart:
                self.start(version=version)
                statusRun, repeatStart, trialSend = self.send(
                    trialSend=trialSend, checkForActive=checkForActive
                )

            # If run is ACTIVE (tr_send criterion), check convergence. If it stopped later, retrieve FAILURE
            if statusRun:
                # ~~~~~ Check status of run before sending look (to avoid problem with OUTTIMES)
                if retrieveAC:
                    dictInfo, _, _ = self.check()
                    infoCheck = dictInfo["info"]["status"]
                    while infoCheck != "finished":
                        mins = 10
                        currentTime = datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        print(
                            ">> {0}, run not finished yet, but wait for AC generation (wait {1} min for next check)".format(
                                currentTime, mins
                            )
                        )
                        time.sleep(60.0 * mins)
                        dictInfo, _, _ = self.check()
                        infoCheck = dictInfo["info"]["status"]

                    statusStop = -2

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~ Standard convergence test
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    ConvergedRun, statusStop = self.convergence(
                        convCriteria,
                        minWait=minWait,
                        checkForActive=checkForActive,
                        timeStartPrediction=timeStartPrediction,
                        automaticProcess=automaticProcess,
                        retrieveAC=retrieveAC,
                        phasetxt=phasetxt,
                    )

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # If run is stopped, go out and retrieve a FAILURE
            else:
                statusStop = 1

        # ---------------------------------------------------------------------------
        # Post-TRANSP
        # ---------------------------------------------------------------------------

        # If run has stopped
        if statusStop == 1:
            print(f">> Run {self.runid} has STOPPED", typeMsg="w")
            HasItFailed = True
            writeErrorInfo(self.runid, self.tok, FolderOutputs + "/ErrorInfo.txt")

        # If run has finished running
        elif statusStop == -2:
            print(f">> Run {self.runid} has finished in the grid, assume converged")
            HasItFailed = False

            self.fetch(label="run1", retrieveAC=retrieveAC)

        # If run is for some reason stuck and does not admit looks, repeat process
        elif statusStop == 10:
            print(
                ">> Run {0} does not admit looks, removing and running loop again".format(
                    self.runid
                )
            )
            self.delete(howManyCancel=2, MinWaitDeletion=2)

            HasItFailed = self.automatic(
                convCriteria,
                version=version,
                minWait=minWait,
                FolderOutputs=FolderOutputs,
                timeStartPrediction=timeStartPrediction,
                checkForActive=checkForActive,
                phasetxt=phasetxt,
                automaticProcess=automaticProcess,
                retrieveAC=retrieveAC,
            )

        # If run has sucessfully run and converged
        else:
            print(f">> Run {self.runid} has sucessfully run and converged!")
            HasItFailed = False

        # Whatever the outcome, remove run from GRID. To make sure, send several cancel requests
        self.delete(howManyCancel=2, MinWaitDeletion=2)

        return HasItFailed

    """
	------------------------------------------------------------------------------------------------------
		Auxiliary routines
	------------------------------------------------------------------------------------------------------
	"""

    def start(self, version="pshare"):
        TRANSPglobus_ntcc.tr_start(
            self.shotnumber,
            self.runid,
            self.email,
            self.FolderTRANSP,
            version,
            self.mpisettings,
            self.tok,
        )

    def send(self, trialSend=1, checkForActive=True):
        # ------------------------------------------------------------------------------------------
        # LOOP SEND
        # ------------------------------------------------------------------------------------------

        safetyCont = 0

        RunWorked, repeatStart, timeAssume, timew = False, False, 0, 0
        while not RunWorked:  # While run has not worked (i.e. it is not ACTIVE)
            # -----------
            # Send file if REQUEST exists (until it does not anymore!). If not, assume is on the grid.
            # -----------

            if timew < attemptsSendRequest:
                # --------------------------------------------------------------------------------------
                # Do we take into account if the run exists?
                # --------------------------------------------------------------------------------------
                checkIfExists = False
                # Note for PRF: checkIfExists is false because I'm finding that if the run is canceled, it may be detected as active... so this doesn't really work.

                exists, existenceFlags = False, ["running"]  # ,'finished']
                if checkIfExists:
                    dictInfo, statusRun, infoGridInfo = self.check(permitSomeTime=True)
                    if (dictInfo["info"]["status"] in existenceFlags) and (
                        not dictInfo["AssumptionNeeded"]
                    ):
                        exists = True
                        print("\t- This simulation is already ACTIVE on the grid:")
                        print("\t\t", dictInfo)
                # --------------------------------------------------------------------------------------

                if not exists:
                    TRANSPglobus_ntcc.tr_send(self.FolderTRANSP, self.runid, self.tok)
                else:
                    print(
                        "\t- Because this run is already ACTIVE on the grid, I will not launch again, since that will have no effect",
                        typeMsg="w",
                    )

            else:
                if timeAssume < 1:
                    timeAssume = timeAssume + 1
                    print(
                        ">> {0} assumed to be on the grid... cannot find REQUEST file".format(
                            self.runid
                        ),
                        typeMsg="w",
                    )
                else:
                    print(
                        ">> {0} looks like it did not start correctly, repeat".format(
                            self.runid
                        ),
                        typeMsg="w",
                    )
                    repeatStart, RunWorked = True, True

            # -----------
            # Check is the simulation is running. statusRun gives 0 for active, -1 for stopped, 1 for success
            # -----------

            if checkForActive:
                dictInfo, statusRun, infoGrid = self.check(permitSomeTime=True)

                if (
                    dictInfo["AssumptionNeeded"]
                    and trialSend < maxSendTrialsAfterNotFound
                    and statusRun != 0
                ):
                    print(
                        ">> Asumption for run {0} not accepted because this is trial #{1}, and {2} are allowed".format(
                            self.runid, trialSend, maxSendTrialsAfterNotFound
                        ),
                        typeMsg="w",
                    )
                    repeatStart, RunWorked, trialSend = True, True, 1
                else:
                    if statusRun == 0:
                        currentTime = datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if dictInfo["info"]["status"] != "not started yet":
                            print(
                                ">> {0} run is ACTIVE in the TRANSP grid ({1})".format(
                                    self.runid, currentTime
                                )
                            )
                        else:
                            print(
                                ">> {0} run is SUBMITTED in the TRANSP grid ({1})".format(
                                    self.runid, currentTime
                                )
                            )
                        RunWorked = True

                    elif statusRun == -1:
                        print(
                            f">> {self.runid} run is STOPPED in the TRANSP grid",
                            typeMsg="w",
                        )
                        break

                    elif statusRun == 1:
                        print(
                            f">> {self.runid} run is FINISHED in the TRANSP grid",
                            typeMsg="w",
                        )
                        RunWorked = True

                    else:
                        print("WHAAAAAT?", typeMsg="w")
                        embed()

            else:
                RunWorked = True

            safetyCont += 1

            if safetyCont > 20:
                break

        return RunWorked, repeatStart, trialSend

    def grab(self, folderGRID, retrieveAC=True, retrieveFolder=True):
        """
        This routine grabs all results files from the Grid, either after LOOK or FINISH
        """

        if "yearRun" not in self.__dict__:
            self.check()

        """
		Retrieve CDF file
		"""
        file = f"{self.runid}.CDF"
        TRANSPglobus_ntcc.tr_get(
            file,
            folderGRID,
            self.runid,
            self.FolderTRANSP,
            self.tok,
            remove_previous_before=True,
        )

        """
		Retrieve AC files
		"""
        if retrieveAC:
            # Determine if run has AC files requested
            Reactor = CDFtools.transp_output(f"{self.FolderTRANSP}/{file}")
            TORIC, TORBEAM, NUBEAM = self.determineACs(Reactor)
            # Retrieve
            retrieveACfiles(
                self.runid,
                self.FolderTRANSP,
                self.tok,
                folderGRID,
                NUBEAM=NUBEAM,
                ICRF=TORIC,
                TORBEAM=TORBEAM,
            )

        """
		Retrieve all results in tar file too
		"""
        if retrieveFolder:
            file = f"{self.tok}.{self.yearRun}_{self.runid}.tar.gz"
            TRANSPglobus_ntcc.tr_get(
                file, folderGRID, self.runid, self.FolderTRANSP, self.tok
            )


"""
------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------
"""


def retrieveACfiles(
    nameRunTot,
    FolderSimulation,
    tok,
    serverCDF,
    nummax=1,
    NUBEAM=True,
    ICRF=True,
    TORBEAM=True,
):
    if NUBEAM:
        # ~~~~~~~~~~~~~ NUBEAM ~~~~~~~~~~~~~
        for i in range(nummax):
            name = nameRunTot + f".DATA{i + 1}"
            getSERVERfile(
                name,
                FolderSimulation,
                nameRunTot,
                tok,
                serverCDF,
                nameNewFolder="NUBEAM_folder",
            )

            name = nameRunTot + f"_birth.cdf{i + 1}"
            getSERVERfile(
                name,
                FolderSimulation,
                nameRunTot,
                tok,
                serverCDF,
                nameNewFolder="NUBEAM_folder",
            )

    if ICRF:
        # ~~~~~~~~~~~~~ ICRF ~~~~~~~~~~~~~
        for i in range(nummax):
            name = nameRunTot + f"_ICRF_TAR.GZ{i + 1}"
            getSERVERfile(
                name,
                FolderSimulation,
                nameRunTot,
                tok,
                serverCDF,
                nameNewFolder="TORIC_folder",
            )

        # ~~~~~~~~ Fokker-Planck ~~~~~~~~~~~~~
        name = nameRunTot + "FPPRF.DATA"
        getSERVERfile(
            name,
            FolderSimulation,
            nameRunTot,
            tok,
            serverCDF,
            nameNewFolder="NUBEAM_folder",
        )

        # ~~~~~~~~~~~~~ Fast Ions ~~~~~~~~~~~~~
        for i in range(nummax):
            name = nameRunTot + f"_FI_TAR.GZ{i + 1}"
            getSERVERfile(
                name,
                FolderSimulation,
                nameRunTot,
                tok,
                serverCDF,
                nameNewFolder="FI_folder",
            )

    if TORBEAM:
        # ~~~~~~~~~~~~~ TORBEAM ~~~~~~~~~~~~~
        for i in range(nummax):
            name = nameRunTot + f"_TOR_TAR.GZ{i + 1}"
            getSERVERfile(
                name,
                FolderSimulation,
                nameRunTot,
                tok,
                serverCDF,
                nameNewFolder="TORBEAM_folder",
            )


def getSERVERfile(name, FolderSimulation, nameRunTot, tok, server, nameNewFolder=None):
    print(f" >> Retrieving {name} file")
    TRANSPglobus_ntcc.tr_get(name, server, nameRunTot, FolderSimulation, tok)

    nfold = f"{FolderSimulation}/{nameNewFolder}"

    if nameNewFolder is not None:
        if not os.path.exists(nfold):
            os.system(f"mkdir {nfold}")
        os.system(f"mv {FolderSimulation}/{name} {nfold}/.")


def corruptRecover(FolderSimulation, nameRunTot, tok, serverCDF):
    tryCorrupt, minWait, Reactor = 5, 60.0, None

    netCDFfile = obtainResult(
        FolderSimulation, nameRunTot, tok, serverCDF, minWait=minWait
    )

    passY = False
    for i in range(tryCorrupt - 1):
        try:
            # -----------------
            # Successful & correct run
            Reactor = CDFtools.transp_output(netCDFfile)
            Reactor.writeResults_TXT(
                FolderSimulation + "infoRun_preconvergence.dat", ensureBackUp=False
            )
            passY = True
            # -----------------
            break
        except:
            # -----------------
            # Corrupted again
            netCDFfile = obtainResult(
                FolderSimulation, nameRunTot, tok, serverCDF, minWait=minWait
            )

    return Reactor


def obtainResult(FolderSimulation, nameRunTot, tok, serverCDF, minWait=60.0):
    print(
        " >> {0} Reactor CDF is corrupted, trying to relaunch look in {1} minutes".format(
            nameRunTot, minWait
        )
    )

    TRANSPglobus_ntcc.tr_look(FolderSimulation, nameRunTot, tok)
    print(f" >> Waiting {minWait}min before launching a get command")
    time.sleep(minWait * 60.0)

    file = f"{nameRunTot}.CDF"
    TRANSPglobus_ntcc.tr_get(file, serverCDF, nameRunTot, FolderSimulation, tok)
    netCDFfile = FolderSimulation + f"{nameRunTot}.CDF"

    return netCDFfile


# ---------------------------------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------------------------------


def getGridInformation(runid, project=None, NBImissing_isStopped=True):
    """
    ************************************************************************************************
                    Read website
    ************************************************************************************************
    """

    try:
        parser = parseTRANSPwebsite(
            runid=runid, NBImissing_isStopped=NBImissing_isStopped
        )
    except:
        print("Trying again")
        time.sleep(30)
        parser = parseTRANSPwebsite(
            runid=runid, NBImissing_isStopped=NBImissing_isStopped
        )

    """
	************************************************************************************************
			Find Run
	************************************************************************************************
	"""

    infoGrid = grabRunid(parser, runid, project=project)

    """
	************************************************************************************************
			Further interpret results
	************************************************************************************************
	"""

    dictInfo = {}

    if infoGrid is not None:
        dictStatus, dictError, AssumptionNeeded = {}, {}, False

        dictStatus["status"] = infoGrid["status_flag"]
        status = infoGrid["status"]

        if infoGrid["status_flag"] in ["stopped"]:
            try:
                errorinfo = getErrorInfo(runid, project)
                dictError = interpretError(errorinfo)
            except:
                print(
                    "\t- Error file could not be found/interpreted, take a look at whats going on",
                    typeMsg="w",
                )
                embed()

        if "double" in infoGrid["info"]:
            dictStatus["extra"] = "doubled"
        else:
            dictStatus["extra"] = "single"

        dictStatus["user"] = {"globus": infoGrid["user_tr"]}

        dictStatus["info_launch"] = {
            "owner": infoGrid["user"],
            "year": infoGrid["year"],
            "user": infoGrid["user_tr"],
        }

        # ------------------------------------
        # LOOK data
        # ------------------------------------

        dictLook = {}
        if "LOOK" in infoGrid["info"].keys():
            if "Submitted" in infoGrid["info"]["LOOK"]:
                dictLook["status"] = "submitted"
                dictLook["location"] = ""
                dictLook["time"] = ""
            elif (
                "Failed" in infoGrid["info"]["LOOK"]
                or "Ready Since" in infoGrid["info"]["LOOK"]
            ):
                dictLook["status"] = "failed"
                dictLook["location"] = ""
                dictLook["time"] = ""
            elif "/transp/look" in infoGrid["info"]["LOOK"]:
                dictLook["status"] = "generated"
                dictLook["location"] = infoGrid["info"]["LOOK"].split(" ")[0]
                dictLook["time"] = infoGrid["info"]["LOOK"].split(" ")[3]
        else:
            dictLook["status"] = "not sent"
            dictLook["location"] = ""
            dictLook["time"] = ""

        # ------------------------------------
        # Time run
        # ------------------------------------
        if dictStatus["status"] == "running":
            try:
                cpuTime = float(
                    infoGrid["info"]["Mark"].split("cpu time =")[1].split("(h")[0]
                )
            except:
                cpuTime = 0.0

            try:
                time1 = float(
                    infoGrid["info"]["Mark"]
                    .split("Restart mark set to:")[1]
                    .split("(sec)")[0]
                    .split("/")[0]
                )
                time2 = float(
                    infoGrid["info"]["Mark"]
                    .split("Restart mark set to:")[1]
                    .split("(sec)")[0]
                    .split("/")[1]
                )
            except:
                time1 = 0.0
                time2 = 0.0

            dictStatus["cputime"] = cpuTime
            dictStatus["MarkRestart"] = np.array([time1, time2])

        infoGridInfo = infoGrid["info"]

    else:
        dictError = {}

        print(
            f">> Ping to TRANSP Grid website did not include run {runid}, but lets give it some time to update (assume ACTIVE w/ look SUBMITTED for the time being)",
            typeMsg="w",
        )
        dictStatus = {"status": "running", "info_launch": None}
        dictLook = {"status": "submitted", "location": "", "time": ""}
        status = 0
        AssumptionNeeded = True

        infoGridInfo = None

    dictInfo["info"] = dictStatus
    dictInfo["look"] = dictLook
    dictInfo["error"] = dictError
    dictInfo["AssumptionNeeded"] = AssumptionNeeded

    try:
        print(
            "\t- Run {0} is in the grid: SIMULATION = {1}, LOOK = {2}".format(
                runid, dictInfo["info"]["status"], dictInfo["look"]["status"]
            )
        )
    except:
        pass

    return dictInfo, status, infoGridInfo


# ----------------------------------------------------------------------------------------------------------------------------------------
# Tools to grab full website information and downselect by criterion
# ----------------------------------------------------------------------------------------------------------------------------------------


def parseTRANSPwebsite(runid=None, NBImissing_isStopped=True):
    # Decide if getting entire website or only the runid ------------------------
    if runid is not None:
        url = f"https://w3.pppl.gov/cgi-bin/transpgrid_monitor.frames?runid={runid}&project=&owner=&submit=Set+selection+criteria"
    else:
        url = "https://w3.pppl.gov/cgi-bin/transpgrid_monitor.frames"
    # ---------------------------------------------------------------------------

    response = IOtools.receiveWebsite(url)

    # Grab every run in the website
    framer, framer0, newer = [], [], 0
    for line in response:
        if "+tr_" in line.decode():
            framer.append(framer0)
            framer0 = []
        framer0.append(line.decode())
    framer = framer[1:-1]

    if len(framer) == 0:
        framer = [framer0]

    # Parse info
    parser = []
    for frame in framer:
        if len(frame) < 4:
            continue

        status, status_flag, parsed = interpretPage(
            frame[3], NBImissing_isStopped=NBImissing_isStopped
        )

        if "ptrmpi" in frame[3]:
            ptrmpi = (
                frame[3]
                .split("ptrmpi")[1]
                .split("</td><td>")[0]
                .split(">")[-1]
                .rjust(2)
            )
        else:
            ptrmpi = "1 "
        if "trmpi" in frame[3]:
            trmpi = (
                frame[3].split("trmpi")[1].split("</td><td>")[0].split(">")[-1].rjust(2)
            )
        else:
            trmpi = "1 "
        if "toricmpi" in frame[3]:
            toricmpi = (
                frame[3]
                .split("toricmpi")[1]
                .split("</td><td>")[0]
                .split(">")[-1]
                .rjust(2)
            )
        else:
            toricmpi = "1 "

        parser0 = {
            "user_tr": frame[0].split()[0][1:],
            "user": frame[2].split(">")[-1].split()[0],
            "runid": frame[1].split(">")[4].split("<")[0],
            "year": frame[1].split(">")[8].split("<")[0].split()[0],
            "tok": frame[1].split(">")[6].split("<")[0].split()[0],
            "info": parsed,
            "status": status,
            "status_flag": status_flag,
            "mpi": np.max([int(ptrmpi), int(trmpi), int(toricmpi)]),
            "mpis": f"{np.max([int(ptrmpi),int(trmpi),int(toricmpi)])} MPI [tr{trmpi} tor{toricmpi} ptr{ptrmpi}]",
        }

        if parser0["status_flag"] not in ["not started yet"]:
            parser0["machine"] = frame[3].split(".pppl.gov")[0].split()[-1]
        else:
            parser0["machine"] = "?"

        parser.append(parser0)

    # Sometimes runs are not mapped
    framer_extra = []
    for frame in framer:
        seper = frame[-1].split('HREF="transpgrid_perjobmenu.frames?')
        if len(seper) > 2:
            framer_extra.append(seper[1])

    for frame in framer_extra:
        status, status_flag, parsed = interpretPage(
            frame, NBImissing_isStopped=NBImissing_isStopped
        )

        if "totally more than 320 cores used" in parsed:
            parser0 = {
                "user_tr": "---",
                "user": "---",
                "runid": framer_extra[0].split("+")[0],
                "year": "---",
                "tok": framer_extra[0].split("+")[1],
                "info": parsed,
                "status": -1,
                "status_flag": "undetermined",
                "mpi": "---",
                "mpis": "---",
                "machine": "---",
            }

            parser.append(parser0)

    return parser


def interpretPage(the_page, NBImissing_isStopped=True):
    """
    statusRun gives 0 for active or submitted, -1 for stopped, 1 for success
    """

    items = ["project", "owner", "details", "status", "remarks"]

    parsed = OrderedDict()
    try:
        the_page = the_page.decode()
    except:
        pass
    for line in the_page.split("\n"):
        if all([k in line for k in items]):
            line = re.sub("&nbsp", "", line)
            line = re.sub("""<(?:"[^"]*"['"]*|'[^']*'['"]*|[^'">])+>""", "\n", line)
            line = filter(None, line.split("\n"))
            try:
                extens = zip(line[::2], line[1::2])
            except:
                extens = zip(line, line)
            for entry, value in extens:
                parsed.setdefault(entry, "")
                parsed[entry] += value

    if parsed is not None:
        titlesGrid = parsed.keys()
        if "success" in titlesGrid:
            status = 1
            status_flag = "finished"
        elif "stopped" in titlesGrid or "suspended" in titlesGrid:
            status = -1
            status_flag = "stopped"
        elif "missing_nbi" in titlesGrid:
            status = -1
            if NBImissing_isStopped:
                status_flag = "stopped"
            else:
                status_flag = "missing_nbi"
        elif "submitted" in titlesGrid and "active" not in titlesGrid:
            status = 0
            status_flag = "not started yet"
        elif "pre" in parsed.keys():
            status = 0
            status_flag = "preprocessing"
        elif (
            ("active" in titlesGrid)
            or ("Mark" in titlesGrid)
            or ("double" in titlesGrid)
            or ("submitted" in titlesGrid)
        ):
            status = 0
            status_flag = "running"
        elif "canceled" in titlesGrid:
            status_flag = "canceled"
            status = -1
        else:
            status = 0  # Sim has just being submitted
            status_flag = "undetermined"

    return status, status_flag, parsed


def printUser(users, NBImissing_isStopped=True):
    os.system("clear")
    print(
        f'Checking TRANSP Grid usage for {", ".join(users)} ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})'
    )

    try:
        parser = parseTRANSPwebsite(NBImissing_isStopped=NBImissing_isStopped)
    except:
        print("Trying again")
        time.sleep(30)
        parser = parseTRANSPwebsite(NBImissing_isStopped=NBImissing_isStopped)

    totalruns, totalmpis = 0, 0
    for par in parser:
        if par["user"] in users:
            user_complete = f"{par['user']} ({par['user_tr']})"
            typeMsg = (
                "w"
                if par["status_flag"] in ["stopped", "not started yet"]
                else "a" if par["status_flag"] in ["finished"] else "i"
            )
            print(
                f"  * {user_complete.ljust(22)}: {par['runid'].ljust(9)} {par['tok'].ljust(4)} {par['year'].ljust(2)} -- STATUS: {par['status_flag'].ljust(8)} ({par['machine'].ljust(8)}, {par['mpis']})",
                typeMsg=typeMsg,
            )

            if par["status_flag"] not in ["finished", "stopped", "not started yet"]:
                totalmpis += par["mpi"]
                totalruns += 1

    maxmpis = 320 * len(users)

    print(
        f"> Users are taking {totalmpis}/{maxmpis} MPIs with {totalruns} runs active at the moment"
    )
    print(
        "-----------------------------------------------------------------------------------"
    )


def grabRunid(parser, runid, project=None):
    infoGrid = None
    for par in parser:
        if (par["runid"] == runid) and (
            par["tok"] == project if project is not None else True
        ):
            infoGrid = par
            break

    return infoGrid


# ----------------------------------------------------------------------------------------------------------------------------------------
# ERROR INTERPRETATION
# ----------------------------------------------------------------------------------------------------------------------------------------


def getErrorInfo(nameRunTot, tok):
    the_page = IOtools.page(
        f"https://w3.pppl.gov/cgi-bin/transpgrid_listfile?{nameRunTot}+{tok}+tr.tail"
    ).decode()

    if "File not available or not readable" in the_page:
        the_page = IOtools.page(
            "https://w3.pppl.gov/cgi-bin/transpgrid_listfile?{0}+{1}+trdat.log".format(
                nameRunTot, tok
            )
        )
        if "File not available or not readable" in the_page.decode():
            the_page = IOtools.page(
                "https://w3.pppl.gov/cgi-bin/transpgrid_listfile?{0}+{1}+pretr.log".format(
                    nameRunTot, tok
                )
            )

    try:
        the_page = the_page.decode()
    except:
        pass

    return the_page


def interpretError(errorinfo):
    try:
        errorinfo = errorinfo.decode()
    except:
        pass

    if "quval: failed to converge" in errorinfo:
        status = "Equilibrium could not converge."
        recommendation = (
            "Change nteq if happened at t=0. Extend equilibrium ramp if t>0."
        )
        extra = ""

    elif "time base not strict ascending" in errorinfo:
        status = "Error in the time basis of UFiles, make sure it is monotonically increasing in time."
        recommendation = "Change time basis of UFiles."
        extra = ""

    elif "unrecognized name" in errorinfo:
        status = "There is a mistake in one of the names of the namelist variables."
        recommendation = "Change namelist."
        extra = ""

    elif "pt_dvs_solver:  too many iterations" in errorinfo:
        status = "Transport solver failed to converge. Max number of iterations reached"
        recommendation = "Probably happened after a sawtooth or some special event."
        extra = ""

    elif "input vpar_shear_in is NAN" in errorinfo:
        status = "Something fishy happened"
        recommendation = "Probably happened after a sawtooth or some special event."
        extra = ""

    elif "?" in errorinfo or len(errorinfo) < 10:
        status = "Unknown"
        recommendation = ""
        extra = ""

    else:
        status = "Run needs to be restarted, probably out of memory."
        recommendation = "Wait for TRANSP team"
        extra = "missing_nbi"

    return {"status": status, "recommendation": recommendation, "extra": extra}


def writeErrorInfo(runid, tok, fileError):
    textError = getErrorInfo(runid, tok)
    with open(fileError, "w") as fi:
        fi.write(textError)
    print(f">> Error information has been written to {fileError}", typeMsg="w")
