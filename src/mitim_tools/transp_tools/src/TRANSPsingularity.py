import os
import shutil
import copy
import datetime
import time
import numpy as np
from mitim_tools.transp_tools import TRANSPtools, NMLtools
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

MINUTES_ALLOWED_JOB_GET = 30
class TRANSPsingularity(TRANSPtools.TRANSPgeneric):
    def __init__(self, FolderTRANSP, tokamak):
        super().__init__(FolderTRANSP, tokamak)

        self.job_id, self.job_name = None, None

    def defineRunParameters(
        self, *args, minutesAllocation=60 * 8, ensureMPIcompatibility=True, tokamak_name = None, **kwargs
    ):
        super().defineRunParameters(*args, **kwargs)

        self.job_name = f"transp_{self.tok}_{self.runid}"

        # Store folderExecution for later use
        machineSettings = CONFIGread.machineSettings(
            code="transp", nameScratch=f"transp_{self.tok}_{self.runid}", append_folder_local=self.FolderTRANSP
        )
        self.folderExecution = machineSettings["folderWork"]

        # Make sure that the MPIs are set up properly
        if ensureMPIcompatibility:
            self.mpisettings = TRANSPtools.ensureMPIcompatibility(
                self.nml_file, self.nml_file_ptsolver, self.mpisettings
            )

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Number of cores (must be inside 1 node)
        # ---------------------------------------------------------------------------------------------------------------------------------------
        self.nparallel = 1
        for j in self.mpisettings:
            self.nparallel = int(np.max([self.nparallel, self.mpisettings[j]]))
            if self.mpisettings[j] == 1:
                self.mpisettings[j] = 0  # definition used for the transp-source



        self.job = FARMINGtools.mitim_job(self.FolderTRANSP)

        self.job.define_machine(
            "transp",
            f"transp_{self.tok if tokamak_name is None else tokamak_name}_{self.runid}",
            slurm_settings={
                "minutes": minutesAllocation,
                "ntasks": self.nparallel,
                "name": self.job_name,
                "mem": 0,                       # All memory available, since TRANSP manages a lot of in-memory operations
            },
        )

    def run(self, cold_startFromPrevious=False, **kwargs):
        runSINGULARITY(
            self.job,
            self.runid,
            self.shotnumber,
            self.tok,
            self.mpisettings,
            cold_startFromPrevious=cold_startFromPrevious,
            mpi_tasks = self.nparallel,
        )

        self.jobid = self.job.jobid

    def check(self, **kwargs):

        self.job.check(file_output =f"{self.runid}tr.log")

        if not self.job.launchSlurm:
            print('\t- (Note: MITIM "checked" on a job but was not submitted via slurm)', typeMsg='w')
            infoSLURM = None
        else:
            infoSLURM = self.job.infoSLURM
            self.jobid = self.job.jobid_found

        info, status = interpretRun(infoSLURM, self.job.log_file)

        self.latest_info = {'info': info, 'status': status, 'infoGrid': None}

        return info, status, None

    def get(
        self,
        label="run1",
        retrieveAC=False,
        **kwargs,
    ):

        runSINGULARITY_look(
            self.FolderTRANSP,
            self.job.folderExecution,
            self.runid,
            self.job_name + "_look",
        )

        self.cdfs[label] = TRANSPtools.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
        )

        # dictInfo, _, _ = self.check()

        # # THIS NEEDS MORE WORK, LIKE IN GLOBUS #TODO: Fix
        # if dictInfo["info"]["status"] == "finished":
        #     self.statusStop = -2
        # else:
        #     self.statusStop = 0

    def fetch(self, label="run1", retrieveAC=False, **kwargs):
        runSINGULARITY_finish(
            self.FolderTRANSP,
            self.runid,
            self.tok,
            self.job_name,
        )

        # Get reactor to call for ACs as well
        self.cdfs[label] = TRANSPtools.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=False
        )

        # ------------------
        # Organize AC files
        # ------------------

        if retrieveAC:
            ICRF, TORBEAM, NUBEAM = self.determineACs(self.cdfs[label])
            organizeACfiles(
                self.runid, self.FolderTRANSP, ICRF=ICRF, TORBEAM=TORBEAM, NUBEAM=NUBEAM
            )

            # Re-Read again
            self.cdfs[label] = TRANSPtools.storeCDF(
                self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
            )

        return self.cdfs[label]

    def delete(self, howManyCancel=1, MinWaitDeletion=0, **kwargs):
        transp_job = FARMINGtools.mitim_job(self.FolderTRANSP)

        transp_job.define_machine(
            "transp",
            self.job_name,
            launchSlurm=False,
        )

        transp_job.prep(
            f"scancel {self.job_id}",
            label_log_files="_finish",
        )

        for i in range(howManyCancel):
            transp_job.run()

        time.sleep(MinWaitDeletion * 60.0)

    def automatic(
        self,
        convCriteria,
        minWait=60,
        timeStartPrediction=0,
        phasetxt="",
        automaticProcess=False,
        retrieveAC=False,
        **kwargs,
        ):
        # Launch run
        self.run(cold_startFromPrevious=False)

        self.statusStop = -1

        # If run is not found on the grid (-1: not found, 0: running, 1: stopped, -2: success)
        while self.statusStop == -1:
            # ~~~~~ Check status of run before sending look (to avoid problem with OUTTIMES)
            if retrieveAC:
                dictInfo, _, _ = self.check()
                infoCheck = dictInfo["info"]["status"]
                while infoCheck != "finished":
                    mins = 10
                    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f" >>>>>>>>>>> {currentTime}, run not finished yet, but wait for AC generation (wait {mins} min for next check)"
                    )
                    time.sleep(60.0 * mins)
                    dictInfo, _, _ = self.check()
                    infoCheck = dictInfo["info"]["status"]

                self.statusStop = -2

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~ Standard convergence test
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                ConvergedRun, self.statusStop = self.convergence(
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
            # self.run(cold_startFromPrevious=True)

        # ---------------------------------------------------------------------------
        # Post-TRANSP
        # ---------------------------------------------------------------------------

        # If run has stopped
        if self.statusStop == 1:
            print(f" >>>>>>>>>>> Run {self.runid} has STOPPED")
            HasItFailed = True

        # If run has finished running
        elif self.statusStop == -2:
            print(
                f" >>>>>>>>>>> Run {self.runid} has finished in the grid, assume converged"
            )
            HasItFailed = False

            self.fetch(label="run1", retrieveAC=retrieveAC)

        # # If run is for some reason stuck and does not admit looks, repeat process
        # elif self.statusStop == 10:
        #     print(
        #         f" >>>>>>>>>>> Run {self.runid} does not admit looks, removing and running loop again"
        #     )
        #     self.delete(howManyCancel=2, MinWaitDeletion=2)

        #     HasItFailed = self.automatic(
        #         convCriteria,
        #         minWaitLook=minWaitLook,
        #         timeStartPrediction=timeStartPrediction,
        #         checkForActive=checkForActive,
        #         phasetxt=phasetxt,
        #         automaticProcess=automaticProcess,
        #         retrieveAC=retrieveAC,
        #     )

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
    transp_job,
    runid,
    shotnumber,
    tok,
    mpis,
    mpi_tasks=None,
    cold_startFromPrevious=False,
):
    folderWork = transp_job.folder_local
    nparallel = transp_job.slurm_settings["ntasks"] if mpi_tasks is None else mpi_tasks

    NMLtools.adaptNML(folderWork, runid, shotnumber, transp_job.folderExecution)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Common things
    # ---------------------------------------------------------------------------------------------------------------------------------------

    inputFolders, inputFiles, shellPreCommands = [], [], []

    # Catch the situation in which I'm running TRANSP locally
    if not isinstance(transp_job.folderExecution, str):
        transp_job.folderExecution = str(transp_job.folderExecution)

    start_folder = transp_job.folderExecution.split("/")[1]  # e.g. pool001, nobackup1

    if start_folder not in ["home", "Users"]:
        txt_bind = f"--bind /{start_folder} "  # As Jai suggestion to solve problem with nobackup1
    else:
        txt_bind = ""

    txt = ""
    if nparallel > 1:
        txt = " MPI"

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Preparation
    # ---------------------------------------------------------------------------------------------------------------------------------------

    # ********** Standard run, from the beginning

    if not cold_startFromPrevious:
        # ------------------------------------------------------------
        # Copy UFILES and NML into a self-contained folder
        # ------------------------------------------------------------
        folder_inputs = folderWork / "tmp_inputs"
        if folder_inputs.exists():
            IOtools.shutil_rmtree(folder_inputs)
        
        IOtools.askNewFolder(folder_inputs, force=True)
        for item in folderWork.glob('*'):
            if item.is_file():
                shutil.copy2(item, folder_inputs)
            elif item.is_dir():
                shutil.copytree(item, folder_inputs / item.name)

        inputFolders = [folderWork / "tmp_inputs"]

        shellPreCommands = ["cp ./tmp_inputs/* ."]

        # ------------------------------------------------------------
        # Pre-sets
        # ------------------------------------------------------------

        # ENV
        file = folderWork / "env_mitim"
        inputFiles.append(file)
        with open(file, "w") as f:
            f.write(
                f"{transp_job.folderExecution}/results/\n{transp_job.folderExecution}/data/\n{transp_job.folderExecution}/tmp/\ny\ny\n{nparallel}\n{mpis['trmpi']}\n{mpis['toricmpi']}\n{mpis['ptrmpi']}\n0\n0"
            )

        # Direct env (dirty fix until Jai fixes the env app)
        file = folderWork / "transp-bashrc"
        inputFiles.append(file)
        ENVcommand = f"""
export WORKDIR=./
export RESULTDIR={transp_job.folderExecution}/results/
mkdir -p {transp_job.folderExecution}/results/
export DATADIR={transp_job.folderExecution}/data/
mkdir -p {transp_job.folderExecution}/data/
export TMPDIR_TR={transp_job.folderExecution}/tmp/
mkdir -p {transp_job.folderExecution}/tmp/
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
        file = folderWork / "pre_mitim"
        inputFiles.append(file)
        with open(file, "w") as f:
            f.write("00\nY\nLaunched by MITIM\nx\n")

        # ---------------
        # Execution command
        # ---------------

        TRANSPcommand_prep = f"""
#singularity run --app environ $TRANSP_SINGULARITY < {transp_job.folderExecution}/env_mitim
singularity run {txt_bind}--app pretr $TRANSP_SINGULARITY {tok}{txt} {runid} < {transp_job.folderExecution}/pre_mitim
singularity run {txt_bind}--app trdat $TRANSP_SINGULARITY {tok} {runid} w q |& tee {runid}tr_dat.log
"""

        TRANSPcommand = f"""
#singularity run --app environ $TRANSP_SINGULARITY < {transp_job.folderExecution}/env_mitim
singularity run {txt_bind}--app pretr $TRANSP_SINGULARITY {tok}{txt} {runid} < {transp_job.folderExecution}/pre_mitim
singularity run {txt_bind}--app trdat $TRANSP_SINGULARITY {tok} {runid} w q |& tee {runid}tr_dat.log
singularity run {txt_bind}--app link $TRANSP_SINGULARITY {runid}
singularity run {txt_bind}--cleanenv --app transp $TRANSP_SINGULARITY {runid} |& tee {transp_job.folderExecution}/{runid}tr.log
"""

    # ********** Start from previous

    else:
        print("Launch cold_start request")

        TRANSPcommand_prep = None

        TRANSPcommand = f"""
singularity run {txt_bind}--cleanenv --app transp $TRANSP_SINGULARITY {runid} R |& tee {transp_job.folderExecution}/{runid}tr.log
"""

    # ------------------
    # Execute pre-checks
    # ------------------

    if TRANSPcommand_prep is not None:
        (folderWork / f'{runid}tr_dat.log').unlink(missing_ok=True)

        # Run first the prep (with tr_dat)
        (folderWork / f'{runid}mitim_bash.src').unlink(missing_ok=True)
        (folderWork / f'{runid}mitim_shell_executor.sh').unlink(missing_ok=True)

        transp_job.prep(
            TRANSPcommand_prep,
            input_files=inputFiles,
            input_folders=inputFolders,
            output_files=[f"{runid}tr_dat.log"],
            shellPreCommands=shellPreCommands,
        )

        # tr_dat doesn't need slurm
        lS = copy.deepcopy(transp_job.launchSlurm)
        transp_job.launchSlurm = False

        transp_job.run()

        transp_job.launchSlurm = lS  # Back to original

        # Interpret
        TRANSPhelpers.interpret_trdat( folderWork / f'{runid}tr_dat.log')

        inputFiles = inputFiles[:-2]  # Because in SLURMcomplete they are added
        (folderWork / 'tmp_inputs' / 'mitim_bash.src').unlink(missing_ok=True)
        (folderWork / 'tmp_inputs' / 'mitim_shell_executor.sh').unlink(missing_ok=True)

    # ---------------
    # Execute Full
    # ---------------

    transp_job.prep(
        TRANSPcommand,
        input_files=inputFiles,
        input_folders=inputFolders,
        shellPreCommands=shellPreCommands,
    )

    transp_job.run(waitYN=False)

    IOtools.shutil_rmtree(folderWork / 'tmp_inputs')


def interpretRun(infoSLURM, log_file):
    # status gives 0 for active or submitted, -1 for stopped, 1 for success

    info = {"slurm": infoSLURM, "info": {}}

    if (infoSLURM is not None) and (infoSLURM["STATE"] != "NOT FOUND"):
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

        if "TERMINATE THE RUN (NORMAL EXIT)" in "\n".join(log_file):
            status = 1
            info["info"]["status"] = "finished"
        elif ("Error termination" in "\n".join(log_file)) or (
            "Backtrace for this error:" in "\n".join(log_file)
            ) or (
            "TRANSP ABORTR SUBROUTINE CALLED" in "\n".join(log_file)
            ) or (
            "%bad_exit:  generic f77 error exit call" in "\n".join(log_file)
            ) or (
            "Segmentation fault - invalid memory reference" in "\n".join(log_file)
            ) or (
            "*** End of error message ***" in "\n".join(log_file)
            ):
            status = -1
            info["info"]["status"] = "stopped"
        else:
            print(
                "\t- No error nor termination found, assuming it is still running",
                typeMsg="w",
            )
            pringLogTail(log_file, typeMsg="i")
            status = 0
            info["info"]["status"] = "running"

        print(
            f"\t- Run is not currently in the SLURM grid ({info['info']['status']})",
            typeMsg="i" if status == 1 else "w",
        )
        if status == -1:
            pringLogTail(log_file)

    return info, status


def pringLogTail(log_file, howmanylines=100, typeMsg="w"):
    howmanylines = np.min([len(log_file), howmanylines])

    print(f"\t* Last {howmanylines} lines of log file:")
    txt = ""
    for i in range(howmanylines):
        txt += f"\t\t{log_file[-howmanylines+i]}"
    print(txt, typeMsg=typeMsg)

def runSINGULARITY_finish(folderWork, runid, tok, job_name):
    transp_job = FARMINGtools.mitim_job(folderWork)

    transp_job.define_machine(
        "transp",
        job_name,
        launchSlurm=True,
        slurm_settings={"name": job_name+"_finish", "minutes": MINUTES_ALLOWED_JOB_GET},
    )

    # Catch the situation in which I'm running TRANSP locally
    if not isinstance(transp_job.machineSettings["folderWork"], str):
        transp_job.machineSettings["folderWork"] = str(transp_job.machineSettings["folderWork"])

    # ---------------
    # Execution command
    # ---------------

    start_folder = transp_job.machineSettings["folderWork"].split("/")[1]  # e.g. pool001, nobackup1

    if start_folder not in ["home", "Users"]:
        txt_bind = f"--bind /{start_folder} "  # As Jai suggestion to solve problem with nobackup1
    else:
        txt_bind = ""

    TRANSPcommand = f"""
cd {transp_job.machineSettings['folderWork']} && singularity run {txt_bind}--app trlook $TRANSP_SINGULARITY {tok} {runid}
cd {transp_job.machineSettings['folderWork']} && singularity run {txt_bind}--app finishup $TRANSP_SINGULARITY {runid}
"""

    # ---------------
    # Execute
    # ---------------

    print('* Submitting a "finish" request to the cluster', typeMsg="i")

    transp_job.prep(
        TRANSPcommand,
        output_folders=["results"],
        output_files=[f"{runid}tr.log"],
        label_log_files="_finish",
    )

    transp_job.run(
        removeScratchFolders=False
    )  # Because it needs to read what it was there from run()

    odir = folderWork / "results" / f"{tok}.00"
    for item in odir.glob('*'):
        if item.is_file():
            shutil.copy2(item, folderWork)
        elif item.is_dir():
            shutil.copytree(item, folderWork / item.name)

def runSINGULARITY_look(folderWork, folderTRANSP, runid, job_name, times_retry_look = 3):

    transp_job = FARMINGtools.mitim_job(folderWork)

    transp_job.define_machine(
        "transp",
        job_name,
        launchSlurm=True,
        slurm_settings={"name": job_name+"_look", "minutes": MINUTES_ALLOWED_JOB_GET},
    )

    # Catch the situation in which I'm running TRANSP locally
    if not isinstance(transp_job.machineSettings["folderWork"], str):
        transp_job.machineSettings["folderWork"] = str(transp_job.machineSettings["folderWork"])

    # ---------------
    # Execution command
    # ---------------

    start_folder = transp_job.machineSettings["folderWork"].split("/")[
        1
    ]  # e.g. pool001, nobackup1

    if start_folder not in ["home", "Users"]:
        txt_bind = f"--bind /{start_folder} "  # As Jai suggestion to solve problem with nobackup1
    else:
        txt_bind = ""

    # Avoid copying the bash and executable, and the FI cdf files that sometimes vanish. Try to minimize copying window and not crashing after errors
    extra_commands = " --delay-updates --ignore-errors --exclude='*_state.cdf' --exclude='*.tmp' --exclude='mitim*'"

    TRANSPcommand = f"""
rsync -av{extra_commands} {folderTRANSP}/* . &&  singularity run {txt_bind}--app plotcon $TRANSP_SINGULARITY {runid}
"""

    # ---------------
    # Execute
    # ---------------

    print('* Submitting a "look" request to the cluster', typeMsg="i")

    outputFiles = [f"{runid}.CDF",f"{runid}tr.log"]

    transp_job.prep(
        TRANSPcommand,
        output_files=outputFiles,
        label_log_files="_look",
    )

    # Not sure why but the look sometimes just rabndomly (?) fails, so we need to try a few times, outside of the logic of the mitim_job checker
    for i in range(times_retry_look):
        transp_job.run(check_if_files_received=False)
        if (folderWork / f"{runid}.CDF").exists():
            break
        else:
            print(f"Singularity look failed (.CDF file not found), trying again ({i+1}/3)", typeMsg="w")
    if not (folderWork / f"{runid}.CDF").exists():
        print(f"Singularity look failed (.CDF file not found) after {times_retry_look} attempts, please check what's going on", typeMsg="q")


def organizeACfiles(
    runid, FolderTRANSP, nummax=1, ICRF=False, NUBEAM=False, TORBEAM=False
):

    for ff in ["NUBEAM_folder", "TORIC_folder", "TORBEAM_folder", "FI_folder"]:
        (FolderTRANSP / ff).mkdir(parents=True, exist_ok=True)

    if NUBEAM:
        for i in range(nummax):
            if (FolderTRANSP / f'{runid}.DATA{i + 1}').exists():
                (FolderTRANSP / f'{runid}.DATA{i + 1}').replace(FolderTRANSP / 'NUBEAM_folder' / f'{runid}.DATA{i + 1}')
            if (FolderTRANSP / f'{runid}_birth.cdf{i + 1}').exists():
                (FolderTRANSP / f'{runid}_birth.cdf{i + 1}').replace(FolderTRANSP / 'NUBEAM_folder' / f'{runid}_birth.cdf{i + 1}')
    if ICRF:
        for i in range(nummax):
            if (FolderTRANSP / f'{runid}_ICRF_TAR.GZ{i + 1}').exists():
                (FolderTRANSP / f'{runid}_ICRF_TAR.GZ{i + 1}').replace(FolderTRANSP / 'TORIC_folder' / f'{runid}_ICRF_TAR.GZ{i + 1}')
            if (FolderTRANSP / f'{runid}_FI_TAR.GZ{i + 1}').exists():
                (FolderTRANSP / f'{runid}_FI_TAR.GZ{i + 1}').replace(FolderTRANSP / 'FI_folder' / f'{runid}_FI_TAR.GZ{i + 1}')
        if (FolderTRANSP / f'{runid}FPPRF.DATA').exists():
            (FolderTRANSP / f'{runid}FPPRF.DATA').replace(FolderTRANSP / 'NUBEAM_folder' / f'{runid}FPPRF.DATA')

    if TORBEAM:
        for i in range(nummax):
            if (FolderTRANSP / f'{runid}_TOR_TAR.GZ{i + 1}').exists():
                (FolderTRANSP / f'{runid}_TOR_TAR.GZ{i + 1}').replace(FolderTRANSP / 'TORBEAM_folder' / f'{runid}_TOR_TAR.GZ{i + 1}')
