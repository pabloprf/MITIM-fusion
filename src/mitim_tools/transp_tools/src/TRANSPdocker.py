import sys
import copy
import shutil
import subprocess
from mitim_tools.misc_tools import IOtools, FARMINGtools, CONFIGread
from mitim_tools.transp_tools import NMLtools
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.transp_tools.src.TRANSPsingularity import TRANSPsingularity, MINUTES_ALLOWED_JOB_GET
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
Runner for the TRANSP docker-lineage containers (v25.1.0+), which have NO singularity apps
and ship no environment setup. The workflow replicates what the old singularity-app container
(transp_23.1.sif) did internally, mapped onto the new versioned tree /opt/transp/vX.Y.Z:
environment exports + rc sourcing, then pretr / trdat / label / copy_expert_for / tr_build.py,
run the linked executable, and trlook to produce the final CDF.

The same OCI image runs under different engines (auto-detected where the command executes):
    - apptainer/singularity (clusters, e.g. engaging; convert the image once with
      `apptainer build transp_v25.1.0.sif oci-archive://transp_v25.1.0_1.tar`)
    - docker (workstations; the image as provided, e.g. transp_v25.1.0:latest)

Selected per machine in config_user.json (soft deprecation: key absent = old singularity
apps behavior, untouched):
    "transp": {"container_style": "docker", "image": "/path/to/transp_v25.1.0.sif"}
If "image" is not given, $TRANSP_SINGULARITY (typically exported in "modules") is used.

TRANSPdocker subclasses TRANSPsingularity, replacing only the command construction
(_backend_run/_backend_look/_backend_finish); run monitoring, MAESTRO integration and the
fail-fast container-launch catcher are inherited.
"""

# ------------------------------------------------------------------------------------------------------
# In-container script fragments (validated on engaging with transp_v25.1.0, 2026-06)
# ------------------------------------------------------------------------------------------------------

def _container_setup(nparallel, mpisettings):
    """
    Environment that the old singularity-app container exported, remapped onto the versioned
    tree (auto-detected, so a future v25.2 image works unchanged). Note TRANSP uses TMPDIR_TR,
    not TMPDIR. IMAS_PREFIX comes from the IMAS module in PPPL production and is undefined in
    the image, but transp.rc needs it (IMAS_ON=1 in transp_local.rc) to build the link line.
    """
    return f"""TR=$(ls -d /opt/transp/v* 2>/dev/null | tail -1)
echo "Using TRANSP tree: $TR"

export TRANSPROOT=$TR
export TRANSPHOME=$TR
export CODESYSDIR=$TR
export SC=$TR/csh
export SQ=$TR/qcsh
export SB=$TR/bin
export LOCAL=$TR
export BUILD=$TR
export XE=$TR/exe
export CONFIGDIR=$TR/config
export LOGDIR=$TR/log
export ADASDIR=/opt/adas
export PREACTDIR=/opt/preact
export NO_XTRANSPIN=TRUE
export PATH=$TR/exe:$TR/bin:$TR/qcsh:$TR/csh:$TR/tools:$PATH
for d in /opt/*/lib /opt/*/*/lib $TR/lib; do [ -d "$d" ] && LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"; done
export LD_LIBRARY_PATH

export IMAS_PREFIX=$(ls -d /opt/imas/* 2>/dev/null | tail -1)
export IMAS_HOME=$IMAS_PREFIX

# Build-time environment (compilers/linker flags used to link the run executable)
source $TR/transp_local.rc
source $TR/transp.rc

# Run-specific environment (self-contained under the run folder; note TRANSP uses TMPDIR_TR)
export WORKDIR=./
export RESULTDIR=$PWD/results
export DATADIR=$PWD/data
export TMPDIR_TR=$PWD/tmp
mkdir -p results data tmp

export NPROCS={nparallel}
export NBI_NPROCS={mpisettings["trmpi"]}
export NTOR_NPROCS={mpisettings["toricmpi"]}
export NPTR_NPROCS={mpisettings["ptrmpi"]}
export NGEN_NPROCS=0
export NCQL3D_NPROCS=0

# TRANSP needs an unlimited stack (large Fortran automatic arrays); the container shell
# default (typically 8 MB) produces silent, deterministic SIGSEGVs mid-run
ulimit -s unlimited 2>/dev/null || true

# pipefail so that a failing step aborts the script even when piped into tee
set -eo pipefail
"""


def _link_and_build(runid, mpisettings):
    """
    What the old singularity "link" app did, but with tr_build.py instead of uplink, as
    TRANSPhub instructs for versions >= 23.2. <runid>_pserv.tmp is written by tr_start in
    the PPPL production system (not shipped in the containers): it records the PSERVE on/off
    FLAGS (matching the namelist NBI_PSERVE etc., NOT the process counts, which go in the
    *_NPROCS env vars), and datchk_mpi inside TR.EXE cross-checks it against the namelist at
    startup. The leading blank on each line and the record order are REQUIRED (datchk_mpi
    reads the file skipping the first column, in this exact sequence). The tr_build target is
    the bare runid: the makefile rule is `$(NAME): $(NAME)ex.o` with output $(NAME)TR.EXE.
    """
    return f"""if [ ! -e {runid}TF.PLN ]; then csh $SC/label {runid}; fi
if [ ! -e {runid}ex.for ] && [ ! -e {runid}ex.f90 ]; then csh $SC/copy_expert_for {runid}; fi

cat > {runid}_pserv.tmp << EOF
 nbi_pserve = {1 if mpisettings["trmpi"] > 0 else 0}
 ntoric_pserve = {1 if mpisettings["toricmpi"] > 0 else 0}
 nptr_pserve = {1 if mpisettings["ptrmpi"] > 0 else 0}
 ndep_pserve = 0
 ncql3d_pserve = 0
 ngenray_pserve = 0
EOF

tr_build.py trexe {runid}
"""


def _pretr_trdat(tok, runid, nparallel, tee=True):
    txt_mpi = " MPI" if nparallel > 1 else ""
    trdat_log = f"2>&1 | tee {runid}tr_dat.log" if tee else f">> {runid}tr_dat.log 2>&1"
    return f"""pretr {tok}{txt_mpi} {runid} < pre_mitim
trdat {tok} {runid} w q {trdat_log}
"""


def _exe_command(runid, nparallel, restart=False):
    start_or_restart = "R" if restart else "S"
    if nparallel > 1:
        # MPI runs are single-node only (TRANSPhub known limitation).
        # --allow-run-as-root is required under docker (container user is root) and harmless under apptainer.
        # CMA single-copy shared memory must be disabled: hosts with Yama ptrace_scope=1 deny the
        # process_vm_readv between sibling ranks inside the container (dmesg "ptrace attach ...
        # attempted"), silently deadlocking all ranks at 100% CPU on the first inter-rank exchange
        # (the NUBEAM server handoff at the first sources timestep). Both spellings so it covers
        # OpenMPI 4 (vader btl) and 5 (smsc framework); unknown MCA env vars are ignored.
        return f"""export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_smsc=^cma

echo "localhost slots={nparallel}" > machines
mpirun --allow-run-as-root -n {nparallel} -machinefile machines ./{runid}TR.EXE {runid} {start_or_restart}
"""
    else:
        return f"./{runid}TR.EXE {runid} {start_or_restart}\n"


# ------------------------------------------------------------------------------------------------------
# Container invocation (engine detected where the command actually runs, which may be remote)
# ------------------------------------------------------------------------------------------------------

def _config_image():
    """Image from the machine config ("transp": {"image": ...}); None falls back to $TRANSP_SINGULARITY"""
    s = CONFIGread.load_settings()
    machine = s["preferences"]["transp"]
    return s.get(machine, {}).get("transp", {}).get("image", None)


def _wrap_in_container(script_text, script_name, folderExecution, image, redirect=""):
    """
    Build the shell command that materializes the in-container script and runs it through
    whatever engine exists on the executing machine (apptainer/singularity on clusters,
    docker on workstations). The quoted heredoc delimiter keeps $vars unexpanded outside.
    """
    # Bind the top-level folder, as in TRANSPsingularity (e.g. /pool001, /nobackup1, /orcd)
    start_folder = str(folderExecution).split("/")[1]
    txt_bind = f"--bind /{start_folder} " if start_folder not in ["home", "Users"] else ""

    image_expr = str(image) if image is not None else "$TRANSP_SINGULARITY"

    return f"""cat > {script_name} << 'MITIM_SCRIPT_EOF'
{script_text}MITIM_SCRIPT_EOF
IMAGE_TRANSP={image_expr}
if command -v apptainer > /dev/null 2>&1; then
    CONTAINER_EXEC="apptainer exec {txt_bind}--cleanenv $IMAGE_TRANSP"
elif command -v singularity > /dev/null 2>&1; then
    CONTAINER_EXEC="singularity exec {txt_bind}--cleanenv $IMAGE_TRANSP"
elif command -v docker > /dev/null 2>&1; then
    CONTAINER_EXEC="docker run --rm -v $PWD:$PWD -w $PWD $IMAGE_TRANSP"
else
    echo "[MITIM] No container engine found (apptainer/singularity/docker)" >&2
    exit 1
fi
$CONTAINER_EXEC bash {script_name}{redirect}
"""


# ------------------------------------------------------------------------------------------------------
# mitim_job-integrated backends (mirror runSINGULARITY / _look / _finish)
# ------------------------------------------------------------------------------------------------------

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

    inputFolders, inputFiles, shellPreCommands = [], [], []

    # Catch the situation in which I'm running TRANSP locally
    if not isinstance(transp_job.folderExecution, str):
        transp_job.folderExecution = str(transp_job.folderExecution)

    image = _config_image()

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

        # pretr answers (shot year code, confirm, comment, exit) - same as the singularity workflow
        file = folderWork / "pre_mitim"
        inputFiles.append(file)
        with open(file, "w") as f:
            f.write("00\nY\nLaunched by MITIM\nx\n")

        # ---------------
        # Execution commands
        # ---------------

        setup = _container_setup(nparallel, mpis)

        script_prep = setup + _pretr_trdat(tok, runid, nparallel, tee=True)

        script_main = (
            setup
            + _pretr_trdat(tok, runid, nparallel, tee=False)
            + _link_and_build(runid, mpis)
            + _exe_command(runid, nparallel)
        )

        TRANSPcommand_prep = _wrap_in_container(
            script_prep, "mitim_docker_prep.sh", transp_job.folderExecution, image
        )
        TRANSPcommand = _wrap_in_container(
            script_main, "mitim_docker_main.sh", transp_job.folderExecution, image,
            redirect=f" >> {transp_job.folderExecution}/{runid}tr.log 2>&1",
        )

    # ********** Start from previous

    else:
        print("Launch cold_start request")

        TRANSPcommand_prep = None

        script_main = _container_setup(nparallel, mpis) + _exe_command(runid, nparallel, restart=True)
        TRANSPcommand = _wrap_in_container(
            script_main, "mitim_docker_main.sh", transp_job.folderExecution, image,
            redirect=f" >> {transp_job.folderExecution}/{runid}tr.log 2>&1",
        )

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

        # Interpret (includes the container-launch fail-fast catch)
        TRANSPhelpers.interpret_trdat( folderWork / f'{runid}tr_dat.log')

        inputFiles = inputFiles[:-1]  # Because in SLURMcomplete they are added
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

    # Avoid copying the bash and executable, and the FI cdf files that sometimes vanish. Try to minimize copying window and not crashing after errors
    extra_commands = " --delay-updates --ignore-errors --exclude='*_state.cdf' --exclude='*.tmp' --exclude='mitim*'"

    script_look = (
        _container_setup(1, {"trmpi": 0, "toricmpi": 0, "ptrmpi": 0})
        + f"plotcon {runid}\n"
    )
    wrapped = _wrap_in_container(
        script_look, "look_docker.sh", transp_job.machineSettings["folderWork"], _config_image()
    )

    TRANSPcommand = f"""
rsync -av{extra_commands} {folderTRANSP}/* .
{wrapped}
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

    script_finish = (
        _container_setup(1, {"trmpi": 0, "toricmpi": 0, "ptrmpi": 0})
        + f"csh $SC/trlook {tok} {runid}\n"
        + f"csh $SC/finishup {runid}\n"
    )
    wrapped = _wrap_in_container(
        script_finish, "finish_docker.sh", transp_job.machineSettings["folderWork"], _config_image()
    )

    TRANSPcommand = f"""
cd {transp_job.machineSettings['folderWork']}
{wrapped}
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


# ------------------------------------------------------------------------------------------------------
# TRANSP class for docker-style containers: everything inherited except the command construction
# ------------------------------------------------------------------------------------------------------

class TRANSPdocker(TRANSPsingularity):

    _backend_run = staticmethod(runDOCKER)
    _backend_look = staticmethod(runDOCKER_look)
    _backend_finish = staticmethod(runDOCKER_finish)


# ------------------------------------------------------------------------------------------------------
# Standalone runner (manual testing, e.g. tests/dev_tests/transp_docker_quickrun.py)
# ------------------------------------------------------------------------------------------------------

def detect_engine():
    for engine in ["docker", "apptainer", "singularity"]:
        if shutil.which(engine) is not None:
            return engine
    raise RuntimeError("[MITIM] No container engine found in PATH (tried docker, apptainer, singularity)")


def write_commands_script(folder, runid, tok, mpisettings, nparallel):
    """
    Write the full standalone bash script executed INSIDE the container, with cwd = the run
    folder (pretr through trlook in one go). Returns the script path. Also writes the pretr
    answers file.
    """

    file_pre = folder / "pre_mitim"
    with open(file_pre, "w") as f:
        f.write("00\nY\nLaunched by MITIM\nx\n")

    script = (
        "#!/usr/bin/env bash\n"
        "# TRANSP run script generated by MITIM (TRANSPdocker.py), executed inside the container\n"
        + _container_setup(nparallel, mpisettings)
        + _pretr_trdat(tok, runid, nparallel, tee=True)
        + _link_and_build(runid, mpisettings)
        + _exe_command(runid, nparallel)
        + f"csh $SC/trlook {tok} {runid}\n"
    )

    file_script = folder / "mitim_transp_docker.sh"
    with open(file_script, "w") as f:
        f.write(script)

    return file_script


def run_transp_docker(
    folder,
    runid,
    tok,
    image=None,
    mpisettings={"trmpi": 1, "toricmpi": 1, "ptrmpi": 1},
    engine=None,
):
    """
    Run a complete TRANSP case from `folder` (must contain <runid>TR.DAT and UFILEs),
    blocking until it finishes. `image` is the docker tag (docker engine) or the path
    to the converted .sif (apptainer/singularity engine).
    """

    folder = IOtools.expandPath(folder)

    if engine is None:
        engine = detect_engine()

    if image is None:
        raise RuntimeError("[MITIM] No container image provided (docker tag or .sif path)")

    nparallel = max([1] + [int(v) for v in mpisettings.values()])

    # Same convention as TRANSPsingularity.defineRunParameters: 1 means "not used", exported as 0
    mpisettings = {k: (0 if int(v) == 1 else int(v)) for k, v in mpisettings.items()}

    # Point the namelist input paths (inputdir, solver templates) to the run folder, as
    # runSINGULARITY does via adaptNML at submission time — inputs copied from a previous
    # run carry stale absolute paths. nshot is read from the namelist itself (unchanged).
    nshot = IOtools.findValue(folder / f"{runid}TR.DAT", "nshot", "=")
    NMLtools.adaptNML(folder, runid, int(float(nshot)), str(folder))

    file_script = write_commands_script(folder, runid, tok, mpisettings, nparallel)
    log_file = folder / f"{runid}tr.log"

    if engine == "docker":
        container_command = f"docker run --rm -v {folder}:{folder} -w {folder} {image} bash {file_script.name}"
    else:
        # Bind the top-level folder, as in TRANSPsingularity (e.g. /pool001, /nobackup1, /orcd)
        start_folder = str(folder).split("/")[1]
        txt_bind = f"--bind /{start_folder} " if start_folder not in ["home", "Users"] else ""
        container_command = f"cd {folder} && {engine} exec {txt_bind}--cleanenv {image} bash {file_script.name}"

    print(f"\t- Running TRANSP ({runid}, {tok}) via {engine}, NPROCS = {nparallel}", typeMsg="i")
    print(f"\t- Log: {log_file}")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            container_command, shell=True, cwd=folder,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in process.stdout:
            sys.stdout.write(line)  # live progress on screen (print is shadowed by printMsg, which takes no `end`)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        process.wait()

    # --------------------------------------------------------------------------------------
    # Interpret outcome
    # --------------------------------------------------------------------------------------

    log_txt = log_file.read_text()

    if any(err in log_txt for err in TRANSPhelpers.CONTAINER_LAUNCH_ERRORS):
        raise RuntimeError(
            "[MITIM] The container failed to launch on this node (user namespace creation denied,"
            " check user.max_user_namespaces). TRANSP did not run"
        )

    cdf_file = folder / f"{runid}.CDF"
    if not cdf_file.exists():
        raise RuntimeError(
            f"[MITIM] TRANSP run did not produce {cdf_file.name} (exit code {process.returncode}), check {log_file.name}"
        )

    print(f"\t- TRANSP run finished, CDF produced: {cdf_file}", typeMsg="i")

    return cdf_file
