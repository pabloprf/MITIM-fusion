import os
import shutil
import copy
import datetime
import time
import numpy as np
from mitim_tools.transp_tools import TRANSPtools, NMLtools
from mitim_tools.misc_tools import IOtools, FARMINGtools, CONFIGread
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.transp_tools.src.TRANSPsingularity import interpretRun, pringLogTail, organizeACfiles
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

MINUTES_ALLOWED_JOB_GET = 30

class TRANSPdocker(TRANSPtools.TRANSPgeneric):
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
        runDOCKER(
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

        runDOCKER_look(
            self.FolderTRANSP,
            self.job.folderExecution,
            self.runid,
            self.job_name + "_look",
        )

        self.cdfs[label] = TRANSPtools.storeCDF(
            self.FolderTRANSP, self.runid, retrieveAC=retrieveAC
        )

    def fetch(self, label="run1", retrieveAC=False, **kwargs):
        runDOCKER_finish(
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


def _docker_run_cmd(workdir, inner_cmd, txt_bind=""):
    """Helper to build a docker run command string.

    Parameters
    ----------
    workdir : str
        The working directory to mount and use inside the container.
    inner_cmd : str
        The command(s) to execute inside the container via ``bash -c``.
    txt_bind : str, optional
        Extra ``-v`` volume mount flags (e.g. ``-v /pool001:/pool001:Z``).

    Returns
    -------
    str
        A complete ``docker run`` command string.
    """
    return (
        f"docker run --rm {txt_bind}-v {workdir}:{workdir}:Z -w {workdir} "
        f"$TRANSP_DOCKER bash -c \"{inner_cmd}\""
    )


def runDOCKER(
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
        txt_bind = f"-v /{start_folder}:/{start_folder}:Z "
    else:
        txt_bind = ""

    txt = ""
    if nparallel > 1:
        txt = " MPI"

    # Helper to source the Docker environment inside the container
    env_source = f"source {transp_job.folderExecution}/env_docker.sh"

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

        # ENV (Docker environment script replaces both env_mitim and transp-bashrc from Singularity)
        file = folderWork / "env_docker.sh"
        inputFiles.append(file)
        ENVcommand = f"""#!/bin/bash
pushd . > /dev/null && cd /opt/transp/ && source environ && popd > /dev/null
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

        fe = transp_job.folderExecution  # shorthand

        # Build the link step: label + copy_expert_for + tr_build.py
        link_cmd = f"{env_source} && label {runid} && copy_expert_for {runid} && tr_build.py trexe {runid}tr"

        # Build the transp execution step
        if nparallel > 1:
            transp_exec = (
                f"{env_source} && "
                f"echo 'localhost slots={nparallel}' > machines && "
                f"mpirun --allow-run-as-root -n {nparallel} -machinefile machines ./{runid}TR.EXE {runid} S"
            )
        else:
            transp_exec = f"{env_source} && ./{runid}TR.EXE {runid} S"

        TRANSPcommand_prep = f"""
{_docker_run_cmd(fe, f"{env_source} && pretr {tok}{txt} {runid} < {fe}/pre_mitim", txt_bind)}
{_docker_run_cmd(fe, f"{env_source} && trdat {tok} {runid} w q", txt_bind)} |& tee {runid}tr_dat.log
"""

        TRANSPcommand = f"""
{_docker_run_cmd(fe, f"{env_source} && pretr {tok}{txt} {runid} < {fe}/pre_mitim", txt_bind)}
{_docker_run_cmd(fe, f"{env_source} && trdat {tok} {runid} w q", txt_bind)} >> {runid}tr_dat.log 2>&1
{_docker_run_cmd(fe, link_cmd, txt_bind)}
{_docker_run_cmd(fe, transp_exec, txt_bind)} >> {fe}/{runid}tr.log 2>&1
"""

    # ********** Start from previous

    else:
        print("Launch cold_start request")

        TRANSPcommand_prep = None

        fe = transp_job.folderExecution

        if nparallel > 1:
            transp_exec = (
                f"{env_source} && "
                f"echo 'localhost slots={nparallel}' > machines && "
                f"mpirun --allow-run-as-root -n {nparallel} -machinefile machines ./{runid}TR.EXE {runid} R"
            )
        else:
            transp_exec = f"{env_source} && ./{runid}TR.EXE {runid} R"

        TRANSPcommand = f"""
{_docker_run_cmd(fe, transp_exec, txt_bind)} >> {fe}/{runid}tr.log 2>&1
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

        inputFiles = inputFiles[:-1]  # Remove pre_mitim; env_docker.sh is still needed
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

    if 'exclusive' not in transp_job.machineSettings["slurm"] or not transp_job.machineSettings["slurm"]["exclusive"]:
        print("\t- TRANSP typically requires exclusive node allocation, but that has not been requested, prone to failure", typeMsg="i")

    transp_job.run(waitYN=False)

    IOtools.shutil_rmtree(folderWork / 'tmp_inputs')


def runDOCKER_finish(folderWork, runid, tok, job_name):

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
        txt_bind = f"-v /{start_folder}:/{start_folder}:Z "
    else:
        txt_bind = ""

    fe = transp_job.machineSettings['folderWork']

    # Source only the TRANSP environment (env_docker.sh should still exist in the working directory from runDOCKER)
    env_inline = "pushd . > /dev/null && cd /opt/transp/ && source environ && popd > /dev/null"

    TRANSPcommand = f"""
cd {fe} && {_docker_run_cmd(fe, f"{env_inline} && trlook {tok} {runid}", txt_bind)}
cd {fe} && {_docker_run_cmd(fe, f"{env_inline} && finishup {runid}", txt_bind)}
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
            shutil.copytree(item, folderWork / item.name, dirs_exist_ok=True)

def runDOCKER_look(folderWork, folderTRANSP, runid, job_name, times_retry_look = 3):

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
        txt_bind = f"-v /{start_folder}:/{start_folder}:Z "
    else:
        txt_bind = ""

    fe = transp_job.machineSettings['folderWork']

    # Source only the TRANSP environment for plotcon
    env_inline = "pushd . > /dev/null && cd /opt/transp/ && source environ && popd > /dev/null"

    # Avoid copying the bash and executable, and the FI cdf files that sometimes vanish. Try to minimize copying window and not crashing after errors
    extra_commands = " --delay-updates --ignore-errors --exclude='*_state.cdf' --exclude='*.tmp' --exclude='mitim*'"

    TRANSPcommand = f"""
rsync -av{extra_commands} {folderTRANSP}/* . && {_docker_run_cmd(fe, f"{env_inline} && plotcon {runid}", txt_bind)}
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

    # Not sure why but the look sometimes just randomly (?) fails, so we need to try a few times, outside of the logic of the mitim_job checker
    for i in range(times_retry_look):
        transp_job.run(check_if_files_received=False)
        if (folderWork / f"{runid}.CDF").exists():
            break
        else:
            print(f"Docker look failed (.CDF file not found), trying again ({i+1}/3)", typeMsg="w")
    if not (folderWork / f"{runid}.CDF").exists():
        print(f"Docker look failed (.CDF file not found) after {times_retry_look} attempts, please check what's going on", typeMsg="q")
