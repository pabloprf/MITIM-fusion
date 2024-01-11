import os
import datetime
import time
import numpy as np
from IPython import embed
from mitim_tools.transp_tools.src import TRANSPmain
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.transp_tools.tools import NMLtools

from mitim_tools.misc_tools.IOtools import printMsg as print


class TRANSPsingularity(TRANSPmain.TRANSPgeneric):
    def __init__(self, FolderTRANSP, tokamak):
        super().__init__(FolderTRANSP, tokamak)

        self.job_id, self.job_name = None, None

    def defineRunParameters(self, *args, **kargs):
        super().defineRunParameters(*args, **kargs)

        self.job_name = f"transp_{self.tok}_{self.runid}"

        # Store folderExecution for later use
        machineSettings = CONFIGread.machineSettings(
            code="transp", nameScratch=f"transp_{self.tok}_{self.runid}/"
        )
        self.folderExecution = machineSettings["folderWork"]

    """
	------------------------------------------------------------------------------------------------------
		Main routines
	------------------------------------------------------------------------------------------------------
	"""

    def run(self, restartFromPrevious=False, minutesAllocation=60 * 10, **kargs):
        # Make sure that the MPIs are set up properly
        self.mpisettings = TRANSPmain.ensureMPIcompatibility(
            self.nml_file, self.nml_file_ptsolver, self.mpisettings
        )

        self.job_id = runSINGULARITY(
            self.FolderTRANSP,
            self.runid,
            self.shotnumber,
            self.tok,
            self.mpisettings,
            nameJob=self.job_name,
            minutes=minutesAllocation,
            restartFromPrevious=restartFromPrevious,
        )

    def check(self, **kargs):
        infoSLURM, self.job_id, self.log_file = getRunInformation(
            self.job_id, self.FolderTRANSP, self.job_name, self.folderExecution
        )

        info, status = interpretRun(infoSLURM, self.log_file)

        return info, status, None

    def get(
        self,
        label="run1",
        retrieveAC=False,
        fullRequest=True,
        minutesAllocation=60,
        **kargs,
    ):
        runSINGULARITY_look(
            self.FolderTRANSP,
            self.runid,
            self.tok,
            self.folderExecution,
            minutes=minutesAllocation,
        )

        self.cdfs[label] = TRANSPmain.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
        )

    def fetch(self, label="run1", retrieveAC=False, minutesAllocation=60, **kargs):
        runSINGULARITY_finish(
            self.FolderTRANSP, self.runid, self.tok, minutes=minutesAllocation
        )

        # ------------------
        # Organize AC files
        # ------------------
        if retrieveAC:
            print("Checker AC, work on it")
            embed()
            ICRF, TORBEAM, NUBEAM = self.determineACs()
            organizeACfiles(
                self.runid, self.FolderTRANSP, ICRF=ICRF, TORBEAM=TORBEAM, NUBEAM=NUBEAM
            )

        self.cdfs[label] = TRANSPmain.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
        )

        return self.cdfs[label]

    def delete(self, howManyCancel=1, MinWaitDeletion=0, **kargs):
        machineSettings = CONFIGread.machineSettings(
            code="transp", nameScratch=f"transp_{self.tok}_{self.runid}/"
        )
        machineSettings["clear"] = True
        for i in range(howManyCancel):
            FARMINGtools.runCommand(
                f"scancel {self.job_id}", [], machineSettings=machineSettings
            )

        time.sleep(MinWaitDeletion * 60.0)

    def automatic(
        self,
        convCriteria,
        minWait=60,
        timeStartPrediction=0,
        FolderOutputs=None,
        phasetxt="",
        automaticProcess=False,
        retrieveAC=False,
        **kargs,
    ):
        # Launch run
        self.run(restartFromPrevious=False)

        statusStop = -1

        # If run is not found on the grid (-1: not found, 0: running, 1: stopped, -2: success)
        while statusStop == -1:
            # ~~~~~ Check status of run before sending look (to avoid problem with OUTTIMES)
            if retrieveAC:
                dictInfo, _ = self.check()
                infoCheck = dictInfo["info"]["status"]
                while infoCheck != "finished":
                    mins = 10
                    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        " >>>>>>>>>>> {0}, run not finished yet, but wait for AC generation (wait {1} min for next check)".format(
                            currentTime, mins
                        )
                    )
                    time.sleep(60.0 * mins)
                    dictInfo, _ = self.check()
                    infoCheck = dictInfo["info"]["status"]

                statusStop = -2

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~ Standard convergence test
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                ConvergedRun, statusStop = self.convergence(
                    convCriteria,
                    minWait=minWait,
                    timeStartPrediction=timeStartPrediction,
                    automaticProcess=automaticProcess,
                    retrieveAC=retrieveAC,
                    phasetxt=phasetxt,
                )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # ~~~~~ Re-launch because of timelimit
            # self.run(restartFromPrevious=True)

        # ---------------------------------------------------------------------------
        # Post-TRANSP
        # ---------------------------------------------------------------------------

        # If run has stopped
        if statusStop == 1:
            print(f" >>>>>>>>>>> Run {self.runid} has STOPPED")
            HasItFailed = True

        # If run has finished running
        elif statusStop == -2:
            print(
                " >>>>>>>>>>> Run {0} has finished in the grid, assume converged".format(
                    self.runid
                )
            )
            HasItFailed = False

            self.fetch(label="run1", retrieveAC=retrieveAC)

        # If run is for some reason stuck and does not admit looks, repeat process
        elif statusStop == 10:
            print(
                " >>>>>>>>>>> Run {0} does not admit looks, removing and running loop again".format(
                    self.runid
                )
            )
            self.delete(howManyCancel=2, MinWaitDeletion=2)

            HasItFailed = self.automatic(
                convCriteria,
                minWaitLook=minWaitLook,
                timeStartPrediction=timeStartPrediction,
                checkForActive=checkForActive,
                phasetxt=phasetxt,
                automaticProcess=automaticProcess,
                retrieveAC=retrieveAC,
            )

        # If run has sucessfully run and converged
        else:
            print(f" >>>>>>>>>>> Run {self.runid} has sucessfully run and converged!")
            HasItFailed = False

        # Whatever the outcome, remove run from GRID. To make sure, send several cancel requests
        self.delete(howManyCancel=2, MinWaitDeletion=2)

        return HasItFailed


"""
------------------------------------------------------------------------------------------------------
	Auxiliary
------------------------------------------------------------------------------------------------------
"""


def runSINGULARITY(
    folderWork,
    runid,
    shotnumber,
    tok,
    mpis,
    minutes=60,
    nameJob="transp",
    restartFromPrevious=False,
):
    machineSettings = CONFIGread.machineSettings(
        code="transp", nameScratch=f"transp_{tok}_{runid}/"
    )

    folderExecution = machineSettings["folderWork"]

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Number of cores (must be inside 1 node)
    # ---------------------------------------------------------------------------------------------------------------------------------------
    nparallel = 1
    for j in mpis:
        nparallel = int(np.max([nparallel, mpis[j]]))
        if mpis[j] == 1:
            mpis[j] = 0  # definition used for the transp-source

    NMLtools.adaptNML(folderWork, runid, shotnumber, folderExecution)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Common things
    # ---------------------------------------------------------------------------------------------------------------------------------------

    inputFolders, inputFiles, shellPreCommands = [], [], []

    if "nobackup1" in folderExecution:
        txt_bind = (
            "--bind /nobackup1 "  # As Jai suggestion to solve problem with nobackup1
        )
    else:
        txt_bind = ""

    txt = ""
    if nparallel > 1:
        txt = " MPI"

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Preparation
    # ---------------------------------------------------------------------------------------------------------------------------------------

    # ********** Standard run, from the beginning

    if not restartFromPrevious:
        # ------------------------------------------------------------
        # Copy UFILES and NML into a self-contained folder
        # ------------------------------------------------------------
        os.system(f"rm -r {folderWork}/tmp_inputs")
        IOtools.askNewFolder(folderWork + "/tmp_inputs/", force=True)

        # os.system(f'cp {folderWork}/PRF* {folderWork}/*TR.DAT {folderWork}/*_namelist.dat {folderWork}/tmp_inputs/.')
        os.system(f"cp {folderWork}/* {folderWork}/tmp_inputs/.")

        inputFolders = [folderWork + "/tmp_inputs/"]

        shellPreCommands = ["cp ./tmp_inputs/* ."]

        # ------------------------------------------------------------
        # Pre-sets
        # ------------------------------------------------------------

        # ENV
        file = folderWork + "/env_prf"
        inputFiles.append(file)
        with open(file, "w") as f:
            f.write(
                "{0}/results/\n{0}/data/\n{0}/tmp/\ny\ny\n{1}\n{2}\n{3}\n{4}\n0\n0".format(
                    folderExecution,
                    nparallel,
                    mpis["trmpi"],
                    mpis["toricmpi"],
                    mpis["ptrmpi"],
                )
            )

        # Direct env (dirty fix until Jai fixes the env app)
        file = folderWork + "/transp-bashrc"
        inputFiles.append(file)
        ENVcommand = f"""
export WORKDIR=./
export RESULTDIR={folderExecution}/results/
mkdir -p {folderExecution}/results/
export DATADIR={folderExecution}/data/
mkdir -p {folderExecution}/data/
export TMPDIR_TR={folderExecution}/tmp/
mkdir -p {folderExecution}/tmp/
export NPROCS={nparallel}
export NBI_NPROCS={mpis["trmpi"]}
export NTOR_NPROCS={mpis["toricmpi"]}
export NPTR_NPROCS={mpis["ptrmpi"]}
export NGEN_NPROCS=0
export NCQL3D_NPROCS=0
"""

        with open(file, "w") as f:
            f.write(ENVcommand)

        # PRE
        file = folderWork + "/pre_prf"
        inputFiles.append(file)
        with open(file, "w") as f:
            f.write("00\nY\nLaunched by MITIM\nx\n")

        # ---------------
        # Execution command
        # ---------------

        TRANSPcommand_prep = f"""
#singularity run --app environ $TRANSP_SINGULARITY < {folderExecution}/env_prf
singularity run {txt_bind}--app pretr $TRANSP_SINGULARITY {tok}{txt} {runid} < {folderExecution}/pre_prf
singularity run {txt_bind}--app trdat $TRANSP_SINGULARITY {tok} {runid} w q |& tee {runid}tr_dat.log
"""

        TRANSPcommand = f"""
#singularity run --app environ $TRANSP_SINGULARITY < {folderExecution}/env_prf
singularity run {txt_bind}--app pretr $TRANSP_SINGULARITY {tok}{txt} {runid} < {folderExecution}/pre_prf
singularity run {txt_bind}--app trdat $TRANSP_SINGULARITY {tok} {runid} w q |& tee {runid}tr_dat.log
singularity run {txt_bind}--app link $TRANSP_SINGULARITY {runid}
singularity run {txt_bind}--cleanenv --app transp $TRANSP_SINGULARITY {runid} |& tee {folderExecution}/{runid}tr.log
"""

    # ********** Start from previous

    else:
        print("Launch restart request")

        TRANSPcommand_prep = None

        TRANSPcommand = """
singularity run {4}--cleanenv --app transp $TRANSP_SINGULARITY {3} R |& tee {0}/{3}tr.log
""".format(
            folderExecution, tok, txt, runid, txt_bind
        )

    # ------------------
    # Execute pre-checks
    # ------------------

    if TRANSPcommand_prep is not None:
        os.system(f"rm {folderWork}/{runid}tr_dat.log")

        # Run first the prep (with tr_dat)
        os.system(f"rm {folderWork}/tmp_inputs/bash.src")
        os.system(f"rm {folderWork}/tmp_inputs/mitim.sh")

        jobid = FARMINGtools.SLURMcomplete(
            TRANSPcommand_prep,
            folderWork,
            inputFiles,
            [f"{runid}tr_dat.log"],
            5,
            nparallel,
            nameJob,
            machineSettings,
            shellPreCommands=shellPreCommands,
            inputFolders=inputFolders,
            waitYN=True,
        )

        # Interpret
        NMLtools.interpret_trdat(f"{folderWork}/{runid}tr_dat.log")

        #
        inputFiles = inputFiles[:-2]  # Because in SLURMcomplete they are added
        os.system(f"rm {folderWork}/tmp_inputs/bash.src")
        os.system(f"rm {folderWork}/tmp_inputs/mitim.sh")

    # ---------------
    # Execute Full
    # ---------------

    jobid = FARMINGtools.SLURMcomplete(
        TRANSPcommand,
        folderWork,
        inputFiles,
        [],
        minutes,
        nparallel,
        nameJob,
        machineSettings,
        shellPreCommands=shellPreCommands,
        inputFolders=inputFolders,
        waitYN=False,
    )

    os.system(f"rm -r {folderWork}/tmp_inputs")

    return jobid


def getRunInformation(jobid, folder, job_name, folderExecution):
    print('* Submitting a "check" request to the cluster', typeMsg="i")

    machineSettings = CONFIGread.machineSettings(
        code="transp", nameScratch=f"transp_{jobid}_check/"
    )

    # Grab slurm state and log file from TRANSP
    infoSLURM = FARMINGtools.getSLURMstatus(
        folder,
        machineSettings,
        jobid=jobid,
        grablog=f"{folderExecution}/slurm_output.dat",
        name=job_name,
    )

    with open(folder + "/slurm_output.dat", "r") as f:
        log_file = f.readlines()

    # If jobid was given as None, I retrieved the info from the job_name, but now provide here the actual id
    if infoSLURM is not None:
        jobid = infoSLURM["JOBID"]

    return infoSLURM, jobid, log_file


def interpretRun(infoSLURM, log_file):
    # status gives 0 for active or submitted, -1 for stopped, 1 for success

    info = {"slurm": infoSLURM, "info": {}}

    if infoSLURM is not None:
        """
        Case is running		#if infoSLURM['STATE'] in ['RUNNING']
        """
        status = 0
        info["info"]["status"] = "running"

        print(
            f"\t- Run '{info['slurm']['NAME']}' is currently in the SLURM grid, with state '{info['slurm']['STATE']}' (jobid {info['slurm']['JOBID']})",
            typeMsg="i",
        )

    else:
        """
        Case is not running (finished or failed)
        """

        if ("Error termination" in "\n".join(log_file)) or (
            "Backtrace for this error:" in "\n".join(log_file)
        ):
            status = -1
            info["info"]["status"] = "stopped"
        elif "TERMINATE THE RUN (NORMAL EXIT)" in "\n".join(log_file):
            status = 1
            info["info"]["status"] = "finished"
        else:
            print(
                "Not identified status... assume for the moment that it has finished?",
                typeMsg="w",
            )
            embed()
            pringLogTail(log_file)
            status = 1
            info["info"]["status"] = "finished"

        print(
            f"\t- Run is not currently in the SLURM grid ({info['info']['status']})",
            typeMsg="i" if status == 1 else "w",
        )
        if status == -1:
            pringLogTail(log_file)

    return info, status


def pringLogTail(log_file, howmanylines=50):
    howmanylines = np.min([len(log_file), howmanylines])

    print(f"\t* Last {howmanylines} lines of log file:")
    txt = ""
    for i in range(howmanylines):
        txt += f"\t\t{log_file[-howmanylines+i]}"
    print(txt, typeMsg="w")


def runSINGULARITY_finish(folderWork, runid, tok, minutes=60):
    nameJob = f"transp_{tok}_{runid}_finish"

    machineSettings = CONFIGread.machineSettings(
        code="transp_look", nameScratch=f"transp_{tok}_{runid}/"
    )

    # ---------------
    # Execution command
    # ---------------

    if "nobackup1" in machineSettings["folderWork"]:
        txt_bind = (
            "--bind /nobackup1 "  # As Jai suggestion to solve problem with nobackup1
        )
    else:
        txt_bind = ""

    TRANSPcommand = f"""
cd {machineSettings['folderWork']} && singularity run {txt_bind}--app trlook $TRANSP_SINGULARITY {tok} {runid}
cd {machineSettings['folderWork']} && singularity run {txt_bind}--app finishup $TRANSP_SINGULARITY {runid}
cd {machineSettings['folderWork']} && tar -czvf TRANSPresults.tar results/{tok}.00
"""

    # ---------------
    # Execute
    # ---------------

    print('* Submitting a "finish" request to the cluster', typeMsg="i")

    machineSettings["clear"] = False
    jobid = FARMINGtools.SLURMcomplete(
        TRANSPcommand,
        folderWork,
        [],
        ["TRANSPresults.tar"],
        minutes,
        1,
        nameJob,
        machineSettings,
        extranamelogs="_finish",
    )

    os.system(
        f"cd {folderWork} && tar -xzvf TRANSPresults.tar && cp -r results/{tok}.00/* ."
    )
    os.system(f"rm -r {folderWork}/TRANSPresults.tar {folderWork}/results/")


def runSINGULARITY_look(folderWork, runid, tok, folderExecution, minutes=60):
    nameJob = f"transp_{tok}_{runid}_look"

    machineSettings = CONFIGread.machineSettings(
        code="transp_look", nameScratch=f"{nameJob}/"
    )

    # ---------------
    # Execution command
    # ---------------

    if "nobackup1" in machineSettings["folderWork"]:
        txt_bind = (
            "--bind /nobackup1 "  # As Jai suggestion to solve problem with nobackup1
        )
    else:
        txt_bind = ""

    # I have to do the look in another folder (I think it fails if I simply grab, so I copy things)
    TRANSPcommand = f"""
cd {machineSettings['folderWork']} && cp {folderExecution}/*PLN {folderExecution}/transp-bashrc {machineSettings['folderWork']}/. && singularity run {txt_bind}--app plotcon $TRANSP_SINGULARITY {runid}
"""

    # ---------------
    # Execute
    # ---------------

    print('* Submitting a "look" request to the cluster', typeMsg="i")

    outputFiles = [f"{runid}.CDF"]

    jobid = FARMINGtools.SLURMcomplete(
        TRANSPcommand, folderWork, [], outputFiles, minutes, 1, nameJob, machineSettings
    )


def organizeACfiles(
    runid, FolderTRANSP, nummax=1, ICRF=False, NUBEAM=False, TORBEAM=False
):
    os.system(f"mkdir {FolderTRANSP}/NUBEAM_folder/")
    os.system(f"mkdir {FolderTRANSP}/TORIC_folder/")
    os.system(f"mkdir {FolderTRANSP}/TORBEAM_folder/")
    os.system(f"mkdir {FolderTRANSP}/FI_folder/")

    if NUBEAM:
        for i in range(nummax):
            name = runid + f".DATA{i + 1}"
            os.system("mv {0}/{1} {0}/NUBEAM_folder".format(FolderTRANSP, name))
            name = runid + f"_birth.cdf{i + 1}"
            os.system("mv {0}/{1} {0}/NUBEAM_folder".format(FolderTRANSP, name))

    if ICRF:
        for i in range(nummax):
            name = runid + f"_ICRF_TAR.GZ{i + 1}"
            os.system("mv {0}/{1} {0}/TORIC_folder".format(FolderTRANSP, name))

            name = runid + f"_FI_TAR.GZ{i + 1}"
            os.system("mv {0}/{1} {0}/FI_folder".format(FolderTRANSP, name))

        name = runid + "FPPRF.DATA"
        os.system("mv {0}/{1} {0}/NUBEAM_folder".format(FolderTRANSP, name))

    if TORBEAM:
        for i in range(nummax):
            name = runid + f"_TOR_TAR.GZ{i + 1}"
            os.system("mv {0}/{1} {0}/TORBEAM_folder".format(FolderTRANSP, name))
