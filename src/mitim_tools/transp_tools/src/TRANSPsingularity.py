import os
import copy
import datetime
import time
import numpy as np
from mitim_tools.transp_tools.src import TRANSPmain
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.transp_tools.tools import NMLtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed


class TRANSPsingularity(TRANSPmain.TRANSPgeneric):
    def __init__(self, FolderTRANSP, tokamak):
        super().__init__(FolderTRANSP, tokamak)

        self.job_id, self.job_name = None, None

    def defineRunParameters(
        self, *args, minutesAllocation=60 * 8, ensureMPIcompatibility=True, **kwargs
    ):
        super().defineRunParameters(*args, **kwargs)

        self.job_name = f"transp_{self.tok}_{self.runid}"

        # Store folderExecution for later use
        machineSettings = CONFIGread.machineSettings(
            code="transp", nameScratch=f"transp_{self.tok}_{self.runid}/"
        )
        self.folderExecution = machineSettings["folderWork"]

        # Make sure that the MPIs are set up properly
        if ensureMPIcompatibility:
            self.mpisettings = TRANSPmain.ensureMPIcompatibility(
                self.nml_file, self.nml_file_ptsolver, self.mpisettings
            )

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Number of cores (must be inside 1 node)
        # ---------------------------------------------------------------------------------------------------------------------------------------
        nparallel = 1
        for j in self.mpisettings:
            nparallel = int(np.max([nparallel, self.mpisettings[j]]))
            if self.mpisettings[j] == 1:
                self.mpisettings[j] = 0  # definition used for the transp-source

        self.job = FARMINGtools.mitim_job(self.FolderTRANSP)

        self.job.define_machine(
            "transp",
            f"transp_{self.tok}_{self.runid}/",
            slurm_settings={
                "minutes": minutesAllocation,
                "ntasks": nparallel,
                "name": self.job_name,
            },
        )

    def run(self, restartFromPrevious=False, **kwargs):
        runSINGULARITY(
            self.job,
            self.runid,
            self.shotnumber,
            self.tok,
            self.mpisettings,
            restartFromPrevious=restartFromPrevious,
        )

        self.jobid = self.job.jobid

    def check(self, **kwargs):
        self.job.check()

        info, status = interpretRun(self.job.infoSLURM, self.job.log_file)

        self.jobid = self.job.jobid_found

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

        self.cdfs[label] = TRANSPmain.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
        )

        dictInfo, _, _ = self.check()

        # THIS NEEDS MORE WORK, LIKE IN GLOBUS # TO FIX
        if dictInfo["info"]["status"] == "finished":
            self.statusStop = -2
        else:
            self.statusStop = 0

    def fetch(self, label="run1", retrieveAC=False, minutesAllocation=60, **kwargs):
        runSINGULARITY_finish(
            self.FolderTRANSP,
            self.runid,
            self.tok,
            self.job_name,
            minutes=minutesAllocation,
        )

        # Get reactor to call for ACs as well
        self.cdfs[label] = TRANSPmain.storeCDF(
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
            self.cdfs[label] = TRANSPmain.storeCDF(
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
        self.run(restartFromPrevious=False)

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
            # self.run(restartFromPrevious=True)

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

        # If run is for some reason stuck and does not admit looks, repeat process
        elif self.statusStop == 10:
            print(
                f" >>>>>>>>>>> Run {self.runid} does not admit looks, removing and running loop again"
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
    transp_job,
    runid,
    shotnumber,
    tok,
    mpis,
    restartFromPrevious=False,
):
    folderWork = transp_job.folder_local
    nparallel = transp_job.slurm_settings["ntasks"]

    NMLtools.adaptNML(folderWork, runid, shotnumber, transp_job.folderExecution)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Common things
    # ---------------------------------------------------------------------------------------------------------------------------------------

    inputFolders, inputFiles, shellPreCommands = [], [], []

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
                f"{transp_job.folderExecution}/results/\n{transp_job.folderExecution}/data/\n{transp_job.folderExecution}/tmp/\ny\ny\n{nparallel}\n{mpis['trmpi']}\n{mpis['toricmpi']}\n{mpis['ptrmpi']}\n0\n0"
            )

        # Direct env (dirty fix until Jai fixes the env app)
        file = folderWork + "/transp-bashrc"
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
        file = folderWork + "/pre_prf"
        inputFiles.append(file)
        with open(file, "w") as f:
            f.write("00\nY\nLaunched by MITIM\nx\n")

        # ---------------
        # Execution command
        # ---------------

        TRANSPcommand_prep = f"""
#singularity run --app environ $TRANSP_SINGULARITY < {transp_job.folderExecution}/env_prf
singularity run {txt_bind}--app pretr $TRANSP_SINGULARITY {tok}{txt} {runid} < {transp_job.folderExecution}/pre_prf
singularity run {txt_bind}--app trdat $TRANSP_SINGULARITY {tok} {runid} w q |& tee {runid}tr_dat.log
"""

        TRANSPcommand = f"""
#singularity run --app environ $TRANSP_SINGULARITY < {transp_job.folderExecution}/env_prf
singularity run {txt_bind}--app pretr $TRANSP_SINGULARITY {tok}{txt} {runid} < {transp_job.folderExecution}/pre_prf
singularity run {txt_bind}--app trdat $TRANSP_SINGULARITY {tok} {runid} w q |& tee {runid}tr_dat.log
singularity run {txt_bind}--app link $TRANSP_SINGULARITY {runid}
singularity run {txt_bind}--cleanenv --app transp $TRANSP_SINGULARITY {runid} |& tee {transp_job.folderExecution}/{runid}tr.log
"""

    # ********** Start from previous

    else:
        print("Launch restart request")

        TRANSPcommand_prep = None

        TRANSPcommand = f"""
singularity run {txt_bind}--cleanenv --app transp $TRANSP_SINGULARITY {runid} R |& tee {transp_job.folderExecution}/{runid}tr.log
"""

    # ------------------
    # Execute pre-checks
    # ------------------

    if TRANSPcommand_prep is not None:
        os.system(f"rm {folderWork}/{runid}tr_dat.log")

        # Run first the prep (with tr_dat)
        os.system(f"rm {folderWork}/tmp_inputs/mitim_bash.src")
        os.system(f"rm {folderWork}/tmp_inputs/mitim_shell_executor.sh")

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
        NMLtools.interpret_trdat(f"{folderWork}/{runid}tr_dat.log")

        #
        inputFiles = inputFiles[:-2]  # Because in SLURMcomplete they are added
        os.system(f"rm {folderWork}/tmp_inputs/mitim_bash.src")
        os.system(f"rm {folderWork}/tmp_inputs/mitim_shell_executor.sh")

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

    os.system(f"rm -r {folderWork}/tmp_inputs")


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
                "Not identified status... assume for the moment that it has finished",
                typeMsg="w",
            )
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


def runSINGULARITY_finish(folderWork, runid, tok, job_name, minutes=60):
    transp_job = FARMINGtools.mitim_job(folderWork)

    transp_job.define_machine(
        "transp",
        job_name,
        launchSlurm=True,
    )

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
        output_folders=["results/"],
        label_log_files="_finish",
    )

    transp_job.run(
        removeScratchFolders=False
    )  # Because it needs to read what it was there from run()

    os.system(f"cd {folderWork}&& cp -r results/{tok}.00/* .")


def runSINGULARITY_look(folderWork, folderTRANSP, runid, job_name):

    transp_job = FARMINGtools.mitim_job(folderWork)

    transp_job.define_machine(
        "transp",
        job_name,
        launchSlurm=True,
    )

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

    # Avoid copying the bash and executable!
    TRANSPcommand = f"""
    rsync -a --exclude='mitim*' {folderTRANSP}/ . &&  singularity run {txt_bind}--app plotcon $TRANSP_SINGULARITY {runid}
"""

    # ---------------
    # Execute
    # ---------------

    print('* Submitting a "look" request to the cluster', typeMsg="i")

    outputFiles = [f"{runid}.CDF"]

    transp_job.prep(
        TRANSPcommand,
        output_files=outputFiles,
    )

    transp_job.run()


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
            os.system(f"mv {FolderTRANSP}/{name} {FolderTRANSP}/NUBEAM_folder")
            name = runid + f"_birth.cdf{i + 1}"
            os.system(f"mv {FolderTRANSP}/{name} {FolderTRANSP}/NUBEAM_folder")

    if ICRF:
        for i in range(nummax):
            name = runid + f"_ICRF_TAR.GZ{i + 1}"
            os.system(f"mv {FolderTRANSP}/{name} {FolderTRANSP}/TORIC_folder")

            name = runid + f"_FI_TAR.GZ{i + 1}"
            os.system(f"mv {FolderTRANSP}/{name} {FolderTRANSP}/FI_folder")

        name = runid + "FPPRF.DATA"
        os.system(f"mv {FolderTRANSP}/{name} {FolderTRANSP}/NUBEAM_folder")

    if TORBEAM:
        for i in range(nummax):
            name = runid + f"_TOR_TAR.GZ{i + 1}"
            os.system(f"mv {FolderTRANSP}/{name} {FolderTRANSP}/TORBEAM_folder")
