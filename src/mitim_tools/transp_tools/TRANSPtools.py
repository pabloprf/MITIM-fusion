import time
import datetime
import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
Overall routine to handle runs depending on being globus or singularity
"""


def TRANSP(*args):
    s = CONFIGread.load_settings()

    if s["preferences"]["transp"] == "globus":
        from mitim_tools.transp_tools.src.TRANSPglobus import (
            TRANSPglobus as TRANSPclass,
        )
    else:
        from mitim_tools.transp_tools.src.TRANSPsingularity import (
            TRANSPsingularity as TRANSPclass,
        )

    return TRANSPclass(*args)


"""
						TRANSP class to handle runs

#########################################################################################################
 	Steps
#########################################################################################################

# Initialize the class
	t = TRANSP(folder,'CMOD')

# Define user and run parameters

	t.defineRunParameters('88664Z12','88664',mpisettings={'trmpi':1,'toricmpi':64,'ptrmpi':1})

# To define the UFILES and namelist there are several options

	1. Write standard namelist
		t.writeNML(12345,TRANSPnamelist_dict={},outtims=[])
		UFILES HERE?

	2. Populate folder from MDS+ UFILES and namelist
		t.populateFromMDS(88664)

	3. Populate folder from MDS+ but use my own namelist defaults and UFILES usage
		t.populateFromMDS(88664)
		t.defaultbasedMDS(12345)
		
# Run TRANSP
	t.run(version='pshare')

# Check until it finishes
	t.checkUntilFinished()

#########################################################################################################
 	Steps
#########################################################################################################

Note that this is a parent class, and the run command must be specified depending on GLOBUS or SINGULARITY

"""


class TRANSPgeneric:
    def __init__(self, FolderTRANSP, tokamak):
        self.FolderTRANSP = IOtools.expandPath(FolderTRANSP)
        self.tok = tokamak
        self.cdfs = {}

    """
	------------------------------------------------------------------------------------------------------
		Methods to be defined for each type (minimum)
	------------------------------------------------------------------------------------------------------
	"""

    def run(self, *args, **kwargs):
        pass

    def check(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        pass

    def fetch(self, *args, **kwargs):
        pass

    def delete(self, *args, **kwargs):
        pass

    def automatic(*args, **kwargs):
        pass

    """
	------------------------------------------------------------------------------------------------------
		Workflows
	------------------------------------------------------------------------------------------------------
	"""

    def defineRunParameters(
        self,
        runid,
        shotnumber,
        mpisettings={"trmpi": 1, "toricmpi": 1, "ptrmpi": 1},
        shotNumberReal=None,
        **kwargs,
    ):
        self.runid = runid  # 12345A01
        self.shotnumber = shotnumber  # 12345
        self.mpisettings = mpisettings
        self.shotnumberReal = shotNumberReal

        # Namelist location
        self.nml_file = self.FolderTRANSP / f"{self.runid}TR.DAT"
        self.nml_file_ptsolver = self.FolderTRANSP / "ptsolver_namelist.dat"
        self.nml_file_glf23 = self.FolderTRANSP / "glf23_namelist.dat"
        self.nml_file_tglf = self.FolderTRANSP / "tglf_namelist.dat"

        """
		Make sure that the namelists end with \
		 Note: this means that NML tools must NOT add that character at the end
		"""

        for file in [self.nml_file_ptsolver, self.nml_file_glf23, self.nml_file_tglf]:
            if file.exists():
                with open(file, "a") as f:
                    f.write("\n/\n")

    def convergence(
        self,
        convCriteria,
        minWait=60,
        checkForActive=True,
        timeStartPrediction=0,
        automaticProcess=False,
        retrieveAC=False,
        phasetxt="",
    ):
        maxhoursStatic, lastTime, ConvergedRun, hoursStatic, timeTot, timeDifference = (
            10.0,
            0,
            False,
            0,
            0,
            0,
        )

        while not ConvergedRun:
            timeDifference = 0
            tt = (
                datetime.datetime.now() + datetime.timedelta(minutes=minWait)
            ).strftime("%H:%M")
            print(
                f">> Waiting for run {self.runid} to converge... (check every {minWait}min, at {tt})"
            )

            try:
                del self.cdfs["r1"]
            except:
                pass

            time.sleep(60 * minWait)

            self.get(label="r1", retrieveAC=retrieveAC, checkForActive=checkForActive)

            # Try to read the hours it has been static (not progressed a single ms)
            try:
                if self.cdfs["r1"].t[-1] - self.cdfs["r1"].t[0] == timeTot:
                    hoursStatic += minWait / 60.0
                else:
                    hoursStatic = 0
                timeTot = self.cdfs["r1"].t[-1] - self.cdfs["r1"].t[0]

                if hoursStatic > maxhoursStatic:
                    print(
                        f" >> {self.runid} run has not progressed in {hoursStatic:.1f}h (> {maxhoursStatic:.1f}h), assume there is a problem with it"
                    )
                    self.statusStop = 1  # 0

            except:
                pass

            # Check if it has been cold_started by mistake, to avoid overwritting my CDF_prev file at next time step!
            try:
                if lastTime > self.cdfs["r1"].t[-1]:
                    print(
                        f" >> {self.runid} run has come back in time! Please check what happened by comparing CDF and CDF_prev",
                        typeMsg="q" if not automaticProcess else "qa",
                    )
                else:
                    if lastTime > 0:
                        timeDifference = self.cdfs["r1"].t[-1] - lastTime
                    else:
                        timeDifference = self.cdfs["r1"].t[-1] - self.cdfs["r1"].t[0]
                    lastTime = self.cdfs["r1"].t[-1]
            except:
                pass

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~ If the run is correct, evaluate if it has converged
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if self.statusStop == 0:
                ConvergedRun = self.cdfs["r1"].fullMITIMconvergence(
                    minWait,
                    timeDifference,
                    convCriteria,
                    tstart=timeStartPrediction,
                    phasetxt=phasetxt,
                )
            else:
                break

        return ConvergedRun, self.statusStop

    def plot(self, label="run1", time=None):
        if self.cdfs[label] is not None:
            self.cdfs[label].plot(time=time)

            self.fn = self.cdfs[label].fn

    def checkUntilFinished(
        self, label="run1", checkMin=5, grabIntermediateEachMin=300.0, retrieveAC=False,
    ):
        """
        Launch this after .run(), to check if it has finished

        checkMin 					- 	Check the grid after these minutes
        grabIntermediateEachMin 	- 	If I'm checking and has passed more than grabIntermediateEachMin, retrieve intermediate files
        """

        from mitim_tools.transp_tools.src.TRANSPglobus import TRANSPglobus
        should_I_wait = isinstance(self, TRANSPglobus) or self.job.launchSlurm

        try:
            fin = self.latest_info["info"]['info']['status'] == "finished"
        except AttributeError:
            fin = False

        first = True 
        status, time_passed = 0, 0.0
        while status != "finished":

            # ------------------------------------------------------------------------------------------------------------------
            # Waiting Logic
            # ------------------------------------------------------------------------------------------------------------------
            tt = (
                datetime.datetime.now() + datetime.timedelta(minutes=checkMin)
            ).strftime("%H:%M")

            if not should_I_wait:
                print(">> Simulation not run via slurm, not waiting for check since execution was halted")
            elif fin:
                print(">> Simulation finished, not waiting for check")
            else:
                if first:
                    print(
                        f">> Simulation just submitted, will check status in {checkMin}min (at {tt})"
                    )
                else:
                    print(
                        f">> Not finished yet, will check status in {checkMin}min (at {tt})"
                    )
                time.sleep(60.0 * checkMin)
                time_passed += 60.0 * checkMin

            # ------------------------------------------------------------------------------------------------------------------
            # Check status
            # ------------------------------------------------------------------------------------------------------------------

            print(">> Checking status of run:")

            info, _, _ = self.check()
            status = info["info"]["status"]

            if status == "stopped":
                self.get(fullRequest=True, label=label + "_mid", retrieveAC=retrieveAC)
                raise Exception("[MITIM] TRANSP stopped, check the logs and intermediative files, but I need to kill this run to avoid bad results")

            # ------------------------------------------------------------------------------------------------------------------
            # Do something
            # ------------------------------------------------------------------------------------------------------------------

            print(">> Grabbing intermediate files?")
            if time_passed >= 60.0 * grabIntermediateEachMin:
                print(
                    f"\t- Yes, because {time_passed / 60.0}min passed (even though run has not finished yet)"
                )
                self.get(fullRequest=True, label=label + "_mid", retrieveAC=retrieveAC)
                time_passed = 0.0
            else:
                print(f"\t- No, because not enough time has passed, {time_passed / 60.0}min < {grabIntermediateEachMin}min")
            
            first = False

        c = self.fetch(label=label, retrieveAC=retrieveAC)
        self.delete()

        return c

    def determineACs(self, Reactor):
        # ------------------------------------------------------------------------------------------------------------------
        # TORIC
        # ------------------------------------------------------------------------------------------------------------------
        nlicrf = IOtools.findValue(
            self.nml_file, "nlicrf", "=", raiseException=False, avoidIfStartsWith="!"
        )
        fi_outtim = IOtools.findValue(
            self.nml_file, "fi_outtim", "=", raiseException=False, avoidIfStartsWith="!"
        )

        if isinstance(fi_outtim, str):  fi_outtim = float(fi_outtim.split(',')[0])

        TORIC = (
            (nlicrf is not None)
            and (bool(int(nlicrf) if isinstance(nlicrf, float) else nlicrf))
            and (bool(int(nlicrf) if isinstance(nlicrf, float) else nlicrf) and fi_outtim is not None)
        )

        # Has it run enough to get files?
        if TORIC:
            if Reactor.t[-1] < fi_outtim:
                print(
                    "\t- This run has requested TORIC information (AC file) but it has not run long enough",
                    typeMsg="w",
                )
                TORIC = False

        # ------------------------------------------------------------------------------------------------------------------
        # TORBEAM
        # ------------------------------------------------------------------------------------------------------------------

        nltorbeam = IOtools.findValue(
            self.nml_file, "EC_MODEL", "=", raiseException=False, avoidIfStartsWith="!"
        )
        fe_outtim = IOtools.findValue(
            self.nml_file, "fe_outtim", "=", raiseException=False, avoidIfStartsWith="!"
        )

        if isinstance(fe_outtim, str):  fe_outtim = float(fe_outtim.split(',')[0])

        TORBEAM = (nltorbeam == "'TORBEAM'") and (
            bool(int(nltorbeam) if isinstance(nltorbeam, float) else nltorbeam) and fe_outtim is not None
        )

        # Has it run enough to get files?
        if TORBEAM:
            if Reactor.t[-1] < fe_outtim:
                print(
                    "\t- This run has requested TORBEAM information (AC file) but it has not run long enough",
                    typeMsg="w",
                )
                TORBEAM = False

        # ------------------------------------------------------------------------------------------------------------------
        # NUBEAM
        # ------------------------------------------------------------------------------------------------------------------

        nlbeam = IOtools.findValue(
            self.nml_file, "nlbeam", "=", raiseException=False, avoidIfStartsWith="!"
        )
        nalpha = IOtools.findValue(
            self.nml_file, "nalpha", "=", raiseException=False, avoidIfStartsWith="!"
        )
        outtim = IOtools.findValue(
            self.nml_file, "outtim", "=", raiseException=False, avoidIfStartsWith="!"
        )

        if isinstance(outtim, str):  outtim = float(outtim.split(',')[0])

        if (nlbeam is not None) and bool(int(nlbeam) if isinstance(nlbeam, float) else nlbeam):
            BEAMS = True
        else:
            BEAMS = False
        if (nalpha is not None) and int(nalpha) == 0:
            ALPHAS = True
        else:
            ALPHAS = False

        if (BEAMS or ALPHAS) and (outtim is not None):
            NUBEAM = True
        else:
            NUBEAM = False

        # Has it run enough to get files?
        if NUBEAM:
            if Reactor.t[-1] < outtim:
                print(
                    "\t- This run has requested NUBEAM information (AC file) but it has not run long enough",
                    typeMsg="w",
                )
                NUBEAM = False

        # ------------------------------------------------------------------------------------------------------------------

        return TORIC, TORBEAM, NUBEAM

    """
	------------------------------------------------------------------------------------------------------
		Routines to run experimental cases
	------------------------------------------------------------------------------------------------------
	"""

    def populateFromMDS(*args, **kwargs):
        return TRANSPhelpers.populateFromMDS(*args, **kwargs)

    def defaultbasedMDS(*args, **kwargs):
        return TRANSPhelpers.defaultbasedMDS(*args, **kwargs)


def storeCDF(FolderTRANSP, runid, retrieveAC=False):
    netCDFfile = FolderTRANSP / f"{runid}.CDF"

    if retrieveAC:
        readFBM = readTORIC = True
    else:
        readFBM = readTORIC = False

    # try:
    c = CDFtools.transp_output(
        netCDFfile,
        readTGLF=True,
        readStructures=True,
        readGFILE=True,
        readGEQDSK=True,
        readFBM=readFBM,
        readTORIC=readTORIC,
    )
    # except:
    #     print(
    #         "\t- CDF file could not be processed as transp_output, possibly corrupted",
    #         typeMsg="w",
    #     )
    #     c = None

    return c


def ensureMPIcompatibility(nml_file, nml_file_ptsolver, mpisettings):

    nml_file = IOtools.expandPath(nml_file)
    nml_file_ptsolver = IOtools.expandPath(nml_file_ptsolver)

    # If no TORIC, no MPI
    val = IOtools.findValue(nml_file, "nlicrf", "=", raiseException=False)
    if (val is None) or IOtools.isFalse(val):
        mpisettings["toricmpi"] = 1
        _ = IOtools.changeValue(nml_file, "ntoric_pserve", 0, [], "=", MaintainComments=True)

    # If no TGLF, no MPI
    val = IOtools.findValue(nml_file, "lpredictive_mode", "=", raiseException=False, findOnlyLast=True)
    if nml_file_ptsolver.exists():
        val1 = IOtools.findValue(nml_file_ptsolver,"pt_confinement%tglf%active","=",raiseException=False,findOnlyLast=True)
    else:
        val1 = False
    if (val is None) or (int(val) < 3) or (not bool(int(val1) if isinstance(val1, float) else val1)):
        mpisettings["ptrmpi"] = 1
        _ = IOtools.changeValue(nml_file, "nptr_pserve", 0, [], "=", MaintainComments=True)

    # If no NUBEAM, no MPI
    val = IOtools.findValue(nml_file, "nalpha", "=", raiseException=False)
    if (val is None) or (int(val) > 0):
        mpisettings["trmpi"] = 1
        _ = IOtools.changeValue(nml_file, "nbi_pserve", 0, [], "=", MaintainComments=True)

    # -------- Further checkers
    toric_mpi = IOtools.findValue(nml_file, "ntoric_pserve", "=", raiseException=False)
    nbi_mpi = IOtools.findValue(nml_file, "nbi_pserve", "=", raiseException=False)
    tglf_mpi = IOtools.findValue(nml_file, "nptr_pserve", "=", raiseException=False)

    # Check situation in which namelist says MPI but defineRunParameters says not allocation

    if (toric_mpi is not None) and bool(int(toric_mpi) if isinstance(toric_mpi, float) else toric_mpi) and (mpisettings["toricmpi"] < 2):
        print("\t- TORIC mpi specified in namelist but not in defineRunParameters(), high risk of TRANSP failure!",typeMsg="w")
    if (nbi_mpi is not None) and bool(int(nbi_mpi) if isinstance(nbi_mpi, float) else nbi_mpi) and (mpisettings["trmpi"] < 2):
        print("\t- NUBEAM mpi specified in namelist but not in defineRunParameters(), high risk of TRANSP failure!",typeMsg="w")
    if (tglf_mpi is not None) and bool(int(tglf_mpi) if isinstance(tglf_mpi, float) else tglf_mpi) and (mpisettings["ptrmpi"] < 2):
        print("\t- TGLF mpi specified in namelist but not in defineRunParameters(), high risk of TRANSP failure!",typeMsg="w")
    
    # Check situation in which defineRunParameters says allocate MPI but namelist says no MPI run
    
    if (mpisettings["toricmpi"] > 1) and ((toric_mpi is None) or not bool(int(toric_mpi) if isinstance(toric_mpi, float) else toric_mpi)):
        print("\t- TORIC mpi specified in defineRunParameters() but not in namelist, you are probably allocating resources you may not need",typeMsg="i")
    if (mpisettings["trmpi"] > 1) and ((nbi_mpi is None) or not bool(int(nbi_mpi) if isinstance(nbi_mpi, float) else nbi_mpi)):
        print("\t- NUBEAM mpi specified in defineRunParameters() but not in namelist, you are probably allocating resources you may not need",typeMsg="i")
    if (mpisettings["ptrmpi"] > 1) and ((tglf_mpi is None) or not bool(int(tglf_mpi) if isinstance(tglf_mpi, float) else tglf_mpi)):
        print("\t- TGLF mpi specified in defineRunParameters() but not in namelist, you are probably allocating resources you may not need",typeMsg="i")

    return mpisettings
