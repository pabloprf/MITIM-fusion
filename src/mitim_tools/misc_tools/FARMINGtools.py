"""
Set of tools to farm out simulations to run in either remote clusters or locally, serially or parallel
"""

from tqdm import tqdm
import shlex
import shutil
import string
import time
import sys
import subprocess
import socket
import signal
import datetime
import copy
import tarfile
import paramiko
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from mitim_tools.misc_tools import IOtools, CONFIGread
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from IPython import embed

"""
New handling of jobs in remote or local clusters. Example use:

    folderWork = path_to_local_folder

    # Define job
    job = FARMINGtools.mitim_job(folderWork)

    # Define machine
    job.define_machine(
            code_name,  # must be defined in config_user.json
            job_name,   # name you want to give to the job
            slurm_settings={
                'minutes':minutes,
                'ntasks':ntasks,
                'name':job_name,
            },
        )

    # Prepare job (remember that you can get job.folderExecution, which is where the job will be executed remotely)
    job.prep(
            Command_to_execute_as_string, # e.g. f'cd {job.folderExecution} && python3 job.py' 
            output_files=outputFiles,
            input_files=inputFiles,
            input_folders=inputFolders,
            output_folders=outputFolders,
        )

    # Run job
    job.run()

"""

@dataclass
class RetryPolicy:
    """
    Wait/attempt policy applied to every remote operation that can fail transiently
    (connect, sftp transfer, idempotent remote exec). attempts=None retries forever.

    Configured per job through mitim_job.connection_retry_settings, which PORTALS-CGYRO
    fills from portals.namelist:
        transport.options.cgyro.run.ssh_retry_wait_seconds
        transport.options.cgyro.run.ssh_retry_attempts   (int, or null/None for infinite)
    """

    wait_seconds: float = 5.0
    attempts: int | None = 3

    # Transient handshake/network errors: paramiko's own transient class, socket timeouts
    # (Errno 60), EOF/reset from a transport dropped mid-operation, and gaierror (DNS
    # resolution failing while the VPN is down). Anything outside this tuple is a real
    # failure and is re-raised on its first occurrence.
    TRANSIENT = (
        paramiko.ssh_exception.SSHException,
        TimeoutError,
        socket.timeout,
        EOFError,
        ConnectionError,
        socket.gaierror,
    )

    @classmethod
    def from_settings(cls, settings):
        settings = settings or {}
        attempts = settings.get("attempts", 3)
        if attempts is not None and (not isinstance(attempts, int) or attempts < 1):
            raise ValueError(
                f"connection_retry_settings['attempts'] must be a positive int "
                f"or None (infinite); got {attempts!r}"
            )
        return cls(wait_seconds=float(settings.get("wait_seconds", 5)), attempts=attempts)

    def run(self, what, fn, on_retry=None):
        """
        Call fn() until it succeeds or the attempts are exhausted, running on_retry
        (typically the job's connect) between attempts.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except self.TRANSIENT as e:
                if self.attempts is not None and attempt >= self.attempts:
                    raise
                cap_str = "infinite" if self.attempts is None else f"{self.attempts}"
                print(
                    f"\t<> {what} attempt {attempt}/{cap_str} failed "
                    f"({type(e).__name__}: {e}). "
                    f"{'Reconnecting & retrying' if on_retry is not None else 'Retrying'} "
                    f"in {self.wait_seconds:g}s...",
                    typeMsg="w",
                )
                time.sleep(self.wait_seconds)
                if on_retry is not None:
                    try:
                        on_retry()
                    except Exception as e_reconnect:
                        print(
                            f"\t<> reconnect during {what} failed ({e_reconnect}); "
                            "will retry on next iteration",
                            typeMsg="w",
                        )


@dataclass
class RetrievalSpec:
    """
    Everything retrieve() needs to know about what comes back from the remote:

        files           mandatory files, relative to folderExecution
        folders         mandatory folders
        selective       {folder: [file, ...]} subsets to tar instead of the whole folder
        fallbacks       {primary: fallback} renames resolved remotely before tarring
        check_in_folder {folder: [file, ...]} checked after extraction
        optional        files that are tarred if present but never flagged as missing

    Polls that must not disturb the remote (check(), submit-mode run()) retrieve with a
    narrowed spec from only() instead of blanking and restoring the job's attributes.
    """

    files: list = field(default_factory=list)
    folders: list = field(default_factory=list)
    selective: dict = field(default_factory=dict)
    fallbacks: dict = field(default_factory=dict)
    check_in_folder: dict = field(default_factory=dict)
    optional: list = field(default_factory=list)

    def only(self, *files, optional=()):
        return RetrievalSpec(files=list(files), optional=list(optional))


class SlurmState(str, Enum):
    """
    The slurm job states MITIM reasons about, as reported by squeue/sacct.
    ABSENT is MITIM's own token for "the job is no longer in the queue".
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    CONFIGURING = "CONFIGURING"
    REQUEUED = "REQUEUED"
    SUSPENDED = "SUSPENDED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    REVOKED = "REVOKED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    ABSENT = "NOT FOUND"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_token(cls, token):
        if token is None:
            return cls.UNKNOWN
        token = str(token).strip().upper()
        if not token:
            return cls.UNKNOWN
        try:
            return cls(token)
        except ValueError:
            pass
        # sacct decorates states ("CANCELLED+", "CANCELLED by 12345"); squeue does not
        try:
            return cls(token.split()[0].rstrip("+"))
        except ValueError:
            return cls.UNKNOWN

    @property
    def in_queue(self):
        # Still the queue's business: keep polling, never read as finished
        return self in (
            SlurmState.PENDING,
            SlurmState.RUNNING,
            SlurmState.COMPLETING,
            SlurmState.CONFIGURING,
            SlurmState.REQUEUED,
            SlurmState.SUSPENDED,
        )

    @property
    def terminal(self):
        # The task has stopped progressing and will not resume by itself. COMPLETING is
        # here as well as in in_queue: the queue still lists it, but the work is over.
        return self in (
            SlurmState.COMPLETED,
            SlurmState.COMPLETING,
            SlurmState.CANCELLED,
            SlurmState.FAILED,
            SlurmState.TIMEOUT,
            SlurmState.OUT_OF_MEMORY,
            SlurmState.BOOT_FAIL,
            SlurmState.NODE_FAIL,
            SlurmState.PREEMPTED,
            SlurmState.REVOKED,
            SlurmState.DEADLINE,
        )


@dataclass
class SqueueRecord:
    """
    One row of `squeue -o "%.15i %.50P %.18j %.10u %.10T %.10M %.10l %.5D %R"`. Array
    submissions print one row per element, so parse() returns all of them.
    """

    jobid: str = None
    name: str = None
    state: SlurmState = SlurmState.UNKNOWN
    nodelist: str = None
    fields: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text):
        lines = [line for line in (text or "").splitlines() if line.strip()]
        if len(lines) < 2:
            return []

        header = lines[0].split()
        # The last column header is "NODELIST(REASON)": %R carries the node once running
        # and the pending reason before that
        key_node = next((key for key in header if key.startswith("NODELIST")), None)

        records = []
        for line in lines[1:]:
            tokens = line.split()
            fields = {key: (tokens[i] if i < len(tokens) else None) for i, key in enumerate(header)}
            records.append(
                cls(
                    jobid=fields.get("JOBID"),
                    name=fields.get("NAME"),
                    state=SlurmState.from_token(fields.get("STATE")),
                    nodelist=fields.get(key_node) if key_node is not None else None,
                    fields=fields,
                )
            )
        return records

    @property
    def node(self):
        # "(null)", "(Priority)", "(Resources)", "n/a": a reason, not a node
        if self.nodelist is None or self.nodelist == "n/a" or self.nodelist.startswith("("):
            return None
        return self.nodelist


class mitim_job:
    def __init__(
            self,
            folder_local,
            log_simulation_file = None # If not None, log information of how the simulation went to this file
            ):
        
        if not isinstance(folder_local, (str, Path)):
            raise TypeError('MITIM job folder must be a valid string or pathlib.Path object to a local directory')
        self.folder_local = IOtools.expandPath(folder_local)
        self.jobid = None
        self.log_simulation_file = log_simulation_file

        # What retrieve() brings back. Populated in prep() for the submit path and in
        # load_submission_state() for the re-attach path. Initialised here so it can be
        # read unconditionally on job instances that bypass prep() entirely (e.g.
        # mitim_job.check(), which retrieves squeue output with a narrowed spec).
        self.spec = RetrievalSpec()

        # The remote session. None until connect() builds them (and for local runs).
        self.jump_client, self.ssh, self.sftp = None, None, None

        # Optional {"wait_seconds": float, "attempts": int|None} overriding the default
        # 5 s / 3 attempts of the RetryPolicy built by self.retry; None keeps the defaults.
        self.connection_retry_settings = None
        # Sub-folders (relative to folderExecution) that remove_scratch_folder() must
        # keep across the going-in wipe: interrupted runs being rescued in place.
        self.preserve_subfolders = []
        # Optional SCHEDULERtools.InAllocationScheduler: when set (local machine, bash
        # mode) full_process() runs the per-call bodies through it instead of executing
        # mitim_bash.src, and its accepted extra folders are added to the retrieval.
        self.scheduler = None
        self.scheduler_result = None

    # The five historical attribute names are views on the spec: external code
    # (SIMtools.load_submission_state, fetch_cgyro_intermediate) assigns them directly,
    # and full_process edits the containers in place, so the getters hand out the live
    # objects rather than copies.

    @property
    def output_files(self):
        return self.spec.files

    @output_files.setter
    def output_files(self, value):
        self.spec.files = value

    @property
    def output_folders(self):
        return self.spec.folders

    @output_folders.setter
    def output_folders(self, value):
        self.spec.folders = value

    @property
    def output_folders_selective(self):
        return self.spec.selective

    @output_folders_selective.setter
    def output_folders_selective(self, value):
        self.spec.selective = value

    @property
    def output_file_fallbacks(self):
        return self.spec.fallbacks

    @output_file_fallbacks.setter
    def output_file_fallbacks(self, value):
        self.spec.fallbacks = value

    @property
    def check_files_in_folder(self):
        return self.spec.check_in_folder

    @check_files_in_folder.setter
    def check_files_in_folder(self, value):
        self.spec.check_in_folder = value

    def __setstate__(self, state):
        # Pickles written before the spec existed carry the five names in the instance
        # dict, where the properties above would shadow them
        spec = state.pop("spec", None)
        if spec is None:
            spec = RetrievalSpec(
                files=state.pop("output_files", None) or [],
                folders=state.pop("output_folders", None) or [],
                selective=state.pop("output_folders_selective", None) or {},
                fallbacks=state.pop("output_file_fallbacks", None) or {},
                check_in_folder=state.pop("check_files_in_folder", None) or {},
            )
        self.__dict__.update(state)
        self.spec = spec

    def define_machine(
        self,
        code,
        nameScratch,
        launchSlurm=True,
        slurm_settings=None,
    ):
        # Separated in case I need to quickly grab the machine settings
        self.define_machine_quick(code, nameScratch, slurm_settings=slurm_settings)

        self.launchSlurm = launchSlurm

        if self.launchSlurm and (len(self.machineSettings["slurm"]) == 0):
            self.launchSlurm = False
            print("\t- slurm requested but no slurm setup to this machine in config... not doing slurm",typeMsg="i",)

        # Print Slurm info — one header line + one compact key=value line.
        if self.launchSlurm:
            host = f'{self.machineSettings["user"]}@{self.machineSettings["machine"]}'
            partition = (self.machineSettings.get("slurm") or {}).get("partition", "?")
            job = self.slurm_settings.get("job-name", "mitim_job")
            print(f"\t- SLURM: {job} @ {host}:{partition}")

            parts = []
            _skip = {"job-name"}  # already in the header
            for k, v in self.slurm_settings.items():
                if k in _skip or v is None or v is False:
                    continue
                parts.append(f"{k}={v}")
            for k in ("qos", "account", "constraint", "exclusive", "exclude"):
                v = (self.machineSettings.get("slurm") or {}).get(k)
                if v is None or v is False:
                    continue
                parts.append(f"{k}={'yes' if v is True else v}")
            if parts:
                print("\t\t" + "  ".join(parts))
        else:
            print(f"\t- Bash (no SLURM) on {self.machineSettings['machine']}")

    def define_machine_quick(self, code, nameScratch, slurm_settings=None):

        self.slurm_settings = slurm_settings if slurm_settings is not None else {}

        # Back-compat: migrate the legacy 'name' key to the native sbatch
        # 'job-name' key. mitim_job-direct callers (e.g. TRANSPsingularity, vgen)
        # still pass 'name'; without this they'd submit as the "mitim_job"
        # default while the squeue/scancel by-name fallbacks search the legacy name.
        if "job-name" not in self.slurm_settings and "name" in self.slurm_settings:
            self.slurm_settings["job-name"] = self.slurm_settings["name"]

        # In case there's no job name, ensure one (native sbatch key)
        self.slurm_settings.setdefault("job-name", "mitim_job")

        self.machineSettings = CONFIGread.machineSettings(
            code=code,
            nameScratch=nameScratch,
            append_folder_local=self.folder_local,
        )
        # Left as string due to potentially referencing a remote file system
        self.folderExecution = self.machineSettings["folderWork"]
        # In-place local execution: no scratch staging, runs directly in folder_local
        self.run_in_place = bool(self.machineSettings.get("run_in_place", False))
        if self.run_in_place:
            print("\t- In-place local execution: folderExecution == folder_local (no scratch staging)")

    @staticmethod
    def grab_machine_settings(code):
        return CONFIGread.machineSettings(code=code)

    def prep(
        self,
        command,
        input_files=None,
        input_folders=None,
        output_files=None,
        output_folders=None,
        check_files_in_folder={},
        output_folders_selective={},  # New parameter for selective folder content
        output_file_fallbacks=None,  # {primary_basename: fallback_basename} for retrieve() remote prune
        shellPreCommands=None,
        shellPostCommands=None,
        label_log_files="",
    ):
        """
        command:
            Option 1: string with commands to execute separated by &&
            Option 2: list of strings with commands to execute

        check_files_in_folder is a dictionary with the folder name as key and a list of files to check as value, optionally.
            Otherwise, it will just check if the folder was received, but not the files inside it.

        output_folders_selective is a dictionary with folder name as key and list of specific files
        to include as value, e.g. {'plots': ['figure1.png']}. Explicit names only: the entries are
        quoted into the remote tar command, so shell globs do not expand (and tar does not expand
        them either on create).

        output_file_fallbacks maps primary basename -> fallback basename. Before
        the tarball is built, retrieve() runs one remote bash snippet per
        folder in output_folders_selective that contains the primary: if the
        primary is absent but the fallback is present, the fallback is renamed
        to the primary; if both are present, the fallback is removed. This
        lets us pull exactly one file per pair (cheap transfer) while still
        picking up the fallback when the primary write didn't land (e.g. a
        CGYRO restart that only left bin.cgyro.restart.old behind after a
        timeout mid-rename).
        """

        # Pass to class
        self.command = command

        if not isinstance(self.command, list):
            self.command = [self.command]

        self.input_files = input_files if isinstance(input_files, list) else []
        self.input_folders = input_folders if isinstance(input_folders, list) else []

        self.shellPreCommands = shellPreCommands if isinstance(shellPreCommands, list) else []
        self.shellPostCommands = shellPostCommands if isinstance(shellPostCommands, list) else []
        self.label_log_files = label_log_files

        self.spec = RetrievalSpec(
            files=output_files if isinstance(output_files, list) else [],
            folders=output_folders if isinstance(output_folders, list) else [],
            selective=output_folders_selective if isinstance(output_folders_selective, dict) else {},
            fallbacks=output_file_fallbacks if isinstance(output_file_fallbacks, dict) else {},
            check_in_folder=check_files_in_folder,
        )

        # run() snapshots the input lists on its first call; a re-prep supersedes that
        # snapshot, which would otherwise re-send the previous inputs
        self.__dict__.pop("_input_lists_snapshot", None)

    def run(
            self,
            waitYN=True,
            timeoutSecs=1e6,
            removeScratchFolders=True,
            removeScratchFolders_goingIn=None,
            check_if_files_received=True,
            attempts_execution=1,
            execute_case_flag=True,
            helper_lostconnection=False,
            ):
        
        '''
        execute_case_flag is a master flag to execute or not the commands. If False, the commands will not be executed.
        
        if helper_lostconnection is True, it means that the connection to the remote machine was lost, but the files are there,
            so I just want to retrieve them. In that case, I do not remove the scratch folder going in, and I do not execute the commands.
        '''

        # Make run() idempotent w.r.t. the input file/folder lists. Below, fileSBATCH/
        # fileSHELL are appended and self.input_files/self.input_folders are rewritten to
        # paths relative to folder_local, in place. If run() is re-invoked on the same job
        # (e.g. SIMtools._run's "repeat once" retry after a transient error), repeating
        # that on the now-relative paths makes relative_to() raise ValueError. Snapshot the
        # caller's originals on the first call and restore them on every subsequent one.
        if not hasattr(self, "_input_lists_snapshot"):
            self._input_lists_snapshot = (list(self.input_files), list(self.input_folders))
        else:
            self.input_files = list(self._input_lists_snapshot[0])
            self.input_folders = list(self._input_lists_snapshot[1])

        removeScratchFolders_goingOut = removeScratchFolders
        if removeScratchFolders_goingIn is None:
            removeScratchFolders_goingIn = removeScratchFolders

        if not waitYN:
            removeScratchFolders_goingOut = False

        # Always start by going to the folder (inside sbatch file). Quoted: with
        # scratch null (in-place execution) this is the user's own working folder,
        # which may contain spaces.
        command_str_mod = [f"cd {shlex.quote(str(self.folderExecution))}"]

        for command in self.command:
            command_str_mod += [command]
            
        # ****** Prepare SLURM job *****************************
        comm, fileSBATCH, fileSHELL = create_slurm_execution_files(
            command_str_mod,
            self.folderExecution,
            modules_remote=self.machineSettings["modules"],
            folder_local=self.folder_local,
            shellPreCommands=self.shellPreCommands,
            shellPostCommands=self.shellPostCommands,
            label_log_files=self.label_log_files,
            wait_until_sbatch=waitYN,
            slurm_allocation=self.machineSettings["slurm"],
            launchSlurm=self.launchSlurm,
            slurm_settings=self.slurm_settings,
        )
        # ******************************************************

        if fileSBATCH not in self.input_files:
            self.input_files.append(fileSBATCH)
        if fileSHELL not in self.input_files:
            self.input_files.append(fileSHELL)

        self.output_files = curateOutFiles(self.output_files)

        # Relative paths
        self.input_files = [IOtools.expandPath(path).relative_to(self.folder_local) for path in self.input_files]
        self.input_folders = [IOtools.expandPath(path).relative_to(self.folder_local) for path in self.input_folders]

        # Submit-mode minimal retrieve: when waitYN=False the only thing we
        # need locally after `./mitim_shell_executor.sh > mitim.out` is
        # mitim.out itself (the sbatch banner that carries the jobid for the
        # later check()/fetch() loop). The job's own spec is the post-run
        # output spec, which at this point on the remote contains *only* what
        # we just uploaded (input.cgyro per rho, multi-GB restart binaries,
        # etc.) — pulling it back would round-trip those bytes for nothing.
        # The job keeps its full spec, so the eventual fetch() (SIMtools.fetch
        # -> simulation_job.retrieve()) sees the original CGYRO output list.
        spec = self.spec.only("mitim.out") if not waitYN else None

        self.full_process(
            comm,
            removeScratchFolders_goingIn=removeScratchFolders_goingIn and (not helper_lostconnection),
            removeScratchFolders_goingOut=removeScratchFolders_goingOut,
            timeoutSecs=timeoutSecs,
            check_if_files_received=waitYN and check_if_files_received,
            check_files_in_folder=self.check_files_in_folder,
            attempts_execution=attempts_execution,
            execute_flag=execute_case_flag and (not helper_lostconnection),
            spec=spec,
        )

        # Get jobid
        if self.launchSlurm:
            try:
                with open(self.folder_local / "mitim.out", "r") as f:
                    aux = f.readlines()
                for line in aux:
                    if "Submitted batch job " in line:
                        self.jobid = line.split()[-1]
            except FileNotFoundError:
                self.jobid = None
        else:
            self.jobid = None

    def resubmit_single_task(self, code_call_str, label, exclude_node=None):
        '''
        Submit a fresh, single-task (non-array) sbatch that runs `code_call_str`
        verbatim, reusing this job's machineSettings and live SSH session.
        Returns the new jobid (string) on success, or None on failure.

        Used by the CGYRO stall-rescue path: when one rho in a job array hangs,
        we scancel just that array index, clean its remote subfolder, and call
        this primitive to relaunch the same work as a standalone single-task job
        while sibling array tasks keep running. Output `slurm_output{label}.dat`
        and `slurm_error{label}.dat` are written at the top of folderExecution
        (sbatch banner). The body itself is responsible for any `cd <subfolder>`
        and per-task stdout/stderr redirection (caller composes these into
        `code_call_str` so the primitive stays code-agnostic).

        Connection management: requires self.ssh to be live (caller responsibility);
        does not connect/close on its own so multiple resubmits inside one
        polling cycle can share a single SSH session.

        `exclude_node` (optional): if provided, added as the SBATCH --exclude
        list on the resubmit so a known-bad node doesn't get re-tried. Falls
        back gracefully on None / empty string.
        '''
        if not self.launchSlurm:
            raise RuntimeError("resubmit_single_task requires launchSlurm=True")
        if self.ssh is None and self.machineSettings["machine"] != "local":
            raise RuntimeError("resubmit_single_task requires a live self.ssh; call self.connect() first")

        # Drop --array on a deep copy so the parent slurm_settings stays intact
        # (other resubmits in the same poll-cycle reuse it). Also rename the
        # job so squeue distinguishes the rescue from the parent array.
        slurm_settings = copy.deepcopy(self.slurm_settings or {})
        slurm_settings["array"] = None
        slurm_settings["array_limit"] = None
        parent_name = slurm_settings.get("job-name", "mitim_job")
        slurm_settings["job-name"] = f"{parent_name}{label}"

        # Machine-config slurm allocation copy. Append the bad node to whatever
        # exclude list was already in machineSettings (preserves operator-set
        # exclusions); never overwrite an existing entry.
        slurm_allocation = copy.deepcopy((self.machineSettings or {}).get("slurm", {}))
        if exclude_node:
            existing = slurm_allocation.get("exclude")
            if existing:
                slurm_allocation["exclude"] = f"{existing},{exclude_node}"
            else:
                slurm_allocation["exclude"] = exclude_node

        # The SBATCH body: caller-supplied bash, prefixed with the standard
        # mitim banner echoes from create_slurm_execution_files.
        body = [code_call_str.rstrip("\n")]

        _, fileSBATCH, _ = create_slurm_execution_files(
            body,
            self.folderExecution,
            modules_remote=self.machineSettings.get("modules"),
            folder_local=self.folder_local,
            shellPreCommands=None,
            shellPostCommands=None,
            label_log_files=label,
            wait_until_sbatch=False,
            slurm_allocation=slurm_allocation,
            launchSlurm=True,
            slurm_settings=slurm_settings,
            if_array_relabel=False,
        )

        # Ship just the sbatch file (the shell-executor wrapper isn't used —
        # we sbatch --parsable the file directly to capture the new jobid).
        sbatch_basename = Path(fileSBATCH).name
        remote_sbatch_path = f"{self.folderExecution}/{sbatch_basename}"
        try:
            if self.ssh is None:
                shutil.copy2(str(fileSBATCH), remote_sbatch_path)   # local machine: plain copy
            else:
                self._sftp_transfer_with_retry('put', str(fileSBATCH), remote_sbatch_path)
        except Exception as e:
            print(f"\t- resubmit_single_task: failed to upload {sbatch_basename} ({type(e).__name__}: {e})", typeMsg='w')
            return None

        # `--parsable` makes sbatch print just <jobid>[;<cluster>] on stdout —
        # easy to parse, no log-file scraping like the run() path needs.
        submit_cmd = f"cd {shlex.quote(str(self.folderExecution))} && chmod +x {sbatch_basename} && sbatch --parsable {sbatch_basename}"
        out, err = self.execute(submit_cmd, printYN=True)
        if isinstance(out, bytes):
            out = out.decode(errors='replace')
        if isinstance(err, bytes):
            err = err.decode(errors='replace')
        out = (out or "").strip()
        err = (err or "").strip()

        # `sbatch --parsable` returns "<jobid>" or "<jobid>;<cluster>". Pick the
        # first all-digits token (resilient to either format and to any banner
        # noise that sneaks onto stdout).
        new_jobid = None
        for token in out.replace(';', ' ').split():
            if token.isdigit():
                new_jobid = token
                break

        if new_jobid is None:
            print(
                f"\t- resubmit_single_task: sbatch --parsable did not return a numeric jobid "
                f"(stdout='{out}', stderr='{err}')",
                typeMsg='w',
            )
            return None

        print(f"\t- resubmit_single_task: launched jobid={new_jobid} (label='{label}', exclude={exclude_node})", typeMsg='i')
        return new_jobid

    # --------------------------------------------------------------------
    # SSH executions
    # --------------------------------------------------------------------

    def full_process(
        self,
        comm,
        timeoutSecs=1e6,
        removeScratchFolders_goingIn=True,
        removeScratchFolders_goingOut=True,
        check_if_files_received=True,
        check_files_in_folder={},
        attempts_execution = 1,
        execute_flag=True,
        spec=None,
    ):
        """
        My philosophy is to always wait for the execution of all commands. If I need
        to not wait, that's handled by a slurm submission without --wait, but I still
        want to finish the sbatch launch process.
        
        Notes:
         - If execute_flag is False, the commands will not be executed. This is useful,
            together with removeScratchFolders_goingIn=False if the results exist in the remote
            but the connection failed with your local machine. You can then just retrieve the results.
        """
        wait_for_all_commands = True
        spec = self.spec if spec is None else spec

        time_init = datetime.datetime.now()
        print(f"\n\t-------------- Running process ({time_init.strftime('%Y-%m-%d %H:%M:%S')}{f', will timeout execution in {timeoutSecs}s' if timeoutSecs < 1e6 else ''}) --------------")

        with self.session(log_file=self.folder_local / "paramiko.log"):
            # ~~~~~~ Prepare scratch folder
            if not self.run_in_place:
                if removeScratchFolders_goingIn:
                    self.remove_scratch_folder()
                self.create_scratch_folder()

                # ~~~~~~ Send
                self.send()
            else:
                print("\t* In-place local execution: skipping scratch setup and file staging")

            # ~~~~~~ Execute
            execution_counter = 0
            received, output, error = False, None, None

            while execution_counter < attempts_execution:

                if execute_flag and self.scheduler is not None and self.ssh is None:
                    output, error = b"", b""
                    prelude = "\n".join([self.machineSettings.get("modules") or ""] + list(self.shellPreCommands or []))
                    print(f"\t* Executing (local) through the in-allocation scheduler ({len(self.scheduler.bodies)} calls, {self.scheduler.concurrency} at a time)", typeMsg="i")
                    self.scheduler_result = self.scheduler.run(Path(self.folderExecution), prelude=prelude)
                    # accepted extras come back best-effort: tarred with the same file patterns as
                    # the main folders, never part of the mandatory check
                    patterns = next(iter(spec.selective.values()), None) if spec.selective else None
                    for rel in self.scheduler_result["accepted"]:
                        if rel not in spec.folders:
                            spec.folders.append(rel)
                        if patterns is not None:
                            spec.selective[rel] = list(patterns)
                elif execute_flag:
                    output, error = self.execute(
                        comm,
                        wait_for_all_commands=wait_for_all_commands,
                        printYN=True,
                        timeoutSecs=timeoutSecs if timeoutSecs < 1e6 else None,
                        log_file=self.log_simulation_file
                    )
                else:
                    output, error = b"", b""
                    print("\t* Not executing commands, just retrieving files (execute_flag=False)", typeMsg="q")

                # ~~~~~~ Retrieve
                received = self.retrieve(
                    check_if_files_received=check_if_files_received,
                    check_files_in_folder=check_files_in_folder,
                    spec=spec,
                )

                execution_counter += 1

                if received:
                    break
                else:
                    if execution_counter < attempts_execution:
                        print(f"\t* Unexpectedly, the run did not come back with the right outputs... repeating execution ({execution_counter}/{attempts_execution})")

            # ~~~~~~ Remove scratch folder
            if received:

                if wait_for_all_commands and removeScratchFolders_goingOut and not self.run_in_place:
                    self.remove_scratch_folder()
                
            else:

                # If not received, write output and error to files (they are None when
                # execute_remote swallowed a timeout, or when nothing was executed at all)
                if output is not None:
                    self._write_debugging_files(output, error)

                cont = print(f"\t* Not all expected files received, not removing scratch folder (mitim_farming.out and mitim_farming.err written in '{self.folder_local / 'mitim_farming.err'}')",typeMsg="q")
                if not cont:
                    print("[MITIM] Stopped with embed(), you can look at output and error",typeMsg="w",)
                    embed()

        print(f"\t-------------- Finished process (took {IOtools.getTimeDifference(time_init)}) --------------\n")

    def _write_debugging_files(self, output, error, extra_name=""):
            with open(self.folder_local / f"mitim_farming{extra_name}.out", "w") as f:
                f.write(output.decode("utf-8"))
            with open(self.folder_local / f"mitim_farming{extra_name}.err", "w") as f:
                f.write(error.decode("utf-8"))

    @property
    def retry(self):
        # Built on access, not in __init__, because callers (SIMtools, transport_cgyro)
        # assign connection_retry_settings after the job object exists
        return RetryPolicy.from_settings(self.connection_retry_settings)

    @contextmanager
    def session(self, **kwargs):
        """
        The ssh/jump/sftp lifecycle: connect on entry, close on exit no matter how the
        body ends, so an aborted execute/retrieve cannot leave the transport open.
        """
        try:
            self.connect(**kwargs)
        except Exception:
            self._close_clients()
            raise
        try:
            yield self
        finally:
            self.close()

    def _close_clients(self):
        # Closes and forgets whatever is live. Safe on a job that never connected and
        # on one whose connect half-succeeded (ssh up, open_sftp raised)
        for attribute in ("sftp", "ssh", "jump_client"):
            client = getattr(self, attribute, None)
            if client is not None:
                try:
                    client.close()
                except Exception as e_close:
                    print(f"\t<> Could not close {attribute} cleanly ({type(e_close).__name__}: {e_close})", typeMsg="w")
            setattr(self, attribute, None)

    def connect(self, *args, **kwargs):
        if self.machineSettings["machine"] != "local":
            return self.connect_ssh(*args, **kwargs)
        else:
            self._close_clients()

    def connect_ssh(self, log_file=None):
        self.jump_host = self.machineSettings["tunnel"]
        self.jump_user = self.machineSettings["user"]

        self.target_host = self.machineSettings["machine"]
        self.target_user = self.machineSettings["user"]

        print("\t* Connecting to remote server:")
        print(f'\t\t{self.target_user}@{self.target_host}{f", via tunnel {self.jump_user}@" +self.jump_host  if self.jump_host is not None else ""}{":" + str(self.machineSettings["port"]) if self.machineSettings["port"] is not None else ""}{" with key " + self.machineSettings["identity"] if self.machineSettings["identity"] is not None else ""}')

        if log_file is not None:
            paramiko.util.log_to_file(log_file)

        self.retry.run("Paramiko connect", self._connect_ssh_item)

    def _connect_ssh_item(self):

        # Whatever is live must go before new clients are built, otherwise every
        # reconnect (VPN flap mid-transfer) orphans a transport and an sftp channel
        self._close_clients()

        try:
            self.define_jump()
            self.define_server()
        except paramiko.ssh_exception.AuthenticationException:
            # If it fails, try to disable rsa-sha2-512 and rsa-sha2-256 (e.g. for iris.gat.com)
            self._close_clients()
            self.define_jump()
            self.define_server(
                disabled_algorithms={"pubkeys": ["rsa-sha2-512", "rsa-sha2-256"]}
            )

    def define_server(self, disabled_algorithms=None):
        # Create a new SSH client for the target machine
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the host
        try:
            self.ssh.connect(
                self.target_host,
                username=self.target_user,
                disabled_algorithms=disabled_algorithms,
                key_filename=str(self.key_filename) if self.key_filename is not None else None,
                port=self.port,
                sock=self.sock,
                allow_agent=True,
            )
        except paramiko.ssh_exception.NoValidConnectionsError:
            print("\t> Paramiko's connection failed! trying again in 5 seconds to avoid random drops", typeMsg="w")
            time.sleep(5)
            self.ssh.connect(
                self.target_host,
                username=self.target_user,
                disabled_algorithms=disabled_algorithms,
                key_filename=str(self.key_filename) if self.key_filename is not None else None,
                port=self.port,
                sock=self.sock,
                allow_agent=True,
            )

        try:
            self.sftp = self.ssh.open_sftp()
        except paramiko.sftp.SFTPError:
            raise Exception("[MITIM] SFTPError: Your bashrc on the server likely contains print statements")

    def define_jump(self):
        if self.jump_host is not None:
            # Create an SSH client instance for the jump host
            self.jump_client = paramiko.SSHClient()
            self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            key_jump = self.machineSettings["identity"]

            if key_jump is not None:
                key_jump = IOtools.expandPath(key_jump)
                if not key_jump.exists():
                    if print(
                        'Key file "'
                        + f'{key_jump}'
                        + '" does not exist, continue without key',
                        typeMsg="q",
                    ):
                        key_jump = None

            # Connect to the jump host
            self.jump_client.connect(
                self.jump_host,
                username=self.jump_user,
                port=(
                    self.machineSettings["port"]
                    if self.machineSettings["port"] is not None
                    else 22
                ),
                key_filename=key_jump,
                allow_agent=True,
            )

            # Use the existing transport of self.jump_client for tunneling
            transport = self.jump_client.get_transport()

            # Create a channel to the target through the tunnel
            create_port_in_tunnel = 22
            channel = transport.open_channel(
                "direct-tcpip",
                (self.target_host, create_port_in_tunnel),
                (self.target_host, 0),
            )

            # to pass to self.ssh
            self.port = create_port_in_tunnel
            self.sock = channel
            self.key_filename = None

        else:
            self.jump_client = None
            self.port = (
                self.machineSettings["port"]
                if self.machineSettings["port"] is not None
                else 22
            )
            self.sock = None

            self.key_filename = self.machineSettings["identity"]

            if self.key_filename is not None:
                self.key_filename = IOtools.expandPath(self.key_filename)
                if not self.key_filename.exists():
                    if print(
                        'Key file "'
                        + f'{self.key_filename}'
                        + '" does not exist, continue without key',
                        typeMsg="q",
                    ):
                        self.key_filename = None

    def create_scratch_folder(self):
        if getattr(self, "run_in_place", False):
            return None, None

        print(f'\t* Creating{" remote" if self.ssh is not None else ""} folder:')
        print(f"\t\t{self.folderExecution}")

        command = f"mkdir -p {self.folderExecution}"

        output, error = self.execute(command)

        return output, error

    def _sftp_transfer_with_retry(self, sftp_method_name, *args, **kwargs):
        '''
        Retry a paramiko SFTP transfer ('get' or 'put'), reconnecting between attempts.
        The remote tarballs (mitim_send.tar.gz / mitim_receive.tar.gz) survive a
        reconnect, so callers do not need to re-tar.

        The method is looked up on self.sftp inside the lambda, on every attempt: a
        reconnect replaces self.sftp with a fresh SFTPClient instance.
        '''
        return self.retry.run(
            f"Paramiko sftp.{sftp_method_name}",
            lambda: getattr(self.sftp, sftp_method_name)(*args, **kwargs),
            on_retry=self.connect,
        )

    def send(self):
        if getattr(self, "run_in_place", False):
            return

        print(f'\t* Sending files{" to remote server" if self.ssh is not None else ""}:')

        # Create a tarball of the local directory
        print("\t\t- Tarballing (local side)")
        with tarfile.open(
            self.folder_local / "mitim_send.tar.gz", "w:gz"
        ) as tar:
            for file in self.input_files + self.input_folders:
                tar.add(self.folder_local / file, arcname=file)

        # Send it
        print("\t\t- Sending (local -> remote)")
        if self.ssh is not None:
            with TqdmUpTo(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc="mitim_send.tar.gz",
                bar_format=" " * 20
                + "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
            ) as t:
                # Wrap the put in the shared transient-failure retry — large
                # tarballs (multi-GB CGYRO inputs) are particularly exposed to
                # mid-transfer SSH/VPN flaps, surfacing as paramiko EOFError /
                # SSHException / socket.timeout from sftp.put.
                self._sftp_transfer_with_retry(
                    'put',
                    self.folder_local / "mitim_send.tar.gz",
                    f"{self.folderExecution}/mitim_send.tar.gz",
                    callback=lambda sent, total_size: t.update_to(sent, total_size),
                )
        else:
            shutil.copy2(
                self.folder_local / "mitim_send.tar.gz",
                f"{self.folderExecution}/mitim_send.tar.gz"
            )

        # Extract it
        print("\t\t- Extracting tarball (remote side)")
        self.execute(
            "tar -xzf "
            + f'{self.folderExecution}/mitim_send.tar.gz'
            + " -C "
            + f'{self.folderExecution}'
        )

        # Remove tarballs
        print("\t\t- Removing tarball (local side)")
        (self.folder_local / "mitim_send.tar.gz").unlink(missing_ok=True)
        print("\t\t- Removing tarball (remote side)")
        self.execute(f"rm {self.folderExecution}/mitim_send.tar.gz")

    def execute(self, command_str, log_file=None, **kwargs):

        # self.ssh is None before connect(); on a remote machine that would otherwise
        # send the command to the local shell
        if self.ssh is None and self.machineSettings["machine"] != "local":
            raise RuntimeError("execute() on a remote machine requires a live self.ssh; call self.connect() first")

        if self.ssh is not None:
            output, error = self.execute_remote(command_str, **kwargs)
        else:
            output, error = self.execute_local(command_str, **kwargs)

        # Write information file about where and how the run took place
        if log_file is not None:
            self.write_information_file(command_str, output, error, file=log_file)

        return output, error

    def write_information_file(self, command, output, error, file = 'mitim_simulation.log'):
        """
        Write a log file with information about where the simulation happened (local/remote),
        user, host, ssh settings if remote, and head/tail of output and error.
        """
        import getpass
        import platform
        from datetime import datetime

        # Prepare context info
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_remote = self.ssh is not None
        lines = []
        lines.append("==================== MITIM Simulation Execution Log ====================\n")
        lines.append(f"Date (finished): {now}")
        if is_remote:
            exec_type = "Remote"
        elif getattr(self, "run_in_place", False):
            exec_type = "Local (in-place)"
        else:
            exec_type = "Local"
        lines.append(f"Execution Type: {exec_type}\n")
        lines.append("--- Execution Details ---")
        if is_remote:
            lines.append(f"SSH User: {getattr(self, 'target_user', 'N/A')}")
            lines.append(f"SSH Host: {getattr(self, 'target_host', 'N/A')}")
            lines.append(f"Remote Folder: {self.folderExecution}")
        else:
            lines.append(f"User: {getpass.getuser()}")
            lines.append(f"Host: {platform.node()}")
        lines.append(f"Folder: {self.folderExecution}")
        lines.append("")

        def get_head_tail(data, n=20):
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = str(data)
            lines_ = text.splitlines()
            head = lines_[:n]
            tail = lines_[-n:] if len(lines_) > n else []
            return head, tail

        out_head, out_tail = get_head_tail(output)
        err_head, err_tail = get_head_tail(error)

        lines.append("--- Output (Head) ---")
        lines.extend(out_head if out_head else ["<no output>"])
        lines.append("")
        lines.append("--- Output (Tail) ---")
        lines.extend(out_tail if out_tail else ["<no output>"])
        lines.append("")
        lines.append("--- Error (Head) ---")
        lines.extend(err_head if err_head else ["<no error>"])
        lines.append("")
        lines.append("--- Error (Tail) ---")
        lines.extend(err_tail if err_tail else ["<no error>"])
        lines.append("")
        lines.append(f"--- Command ---")
        lines.append(f"{command}")
        lines.append(f"\n--- Input Files ---")
        lines.extend([str(file) for file in self.input_files])
        lines.append(f"\n--- Input Folders ---")
        lines.extend([str(folder) for folder in self.input_folders])
        lines.append(f"\n--- Output Files ---")
        lines.extend([str(file) for file in self.output_files])
        lines.append(f"\n--- Output Folders ---")
        lines.extend([str(folder) for folder in self.output_folders])
        lines.append("\n=======================================================================\n")

        # Write to file (file can be Path or str)
        file_path = file if isinstance(file, (str, Path)) else str(file)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def execute_remote(
        self,
        command_str,
        printYN=False,
        timeoutSecs=None,
        wait_for_all_commands=True,
        retry_on_transient=False,
        **kwargs,
    ):
        if printYN:
            print("\t* Executing (remote):", typeMsg="i")
            print(f"\t\t{command_str}")

        # Default path preserves historical behavior exactly (socket.timeout is
        # swallowed with a notice, returning None outputs).
        if not retry_on_transient:
            try:
                return self._execute_remote_once(command_str, timeoutSecs, wait_for_all_commands)
            except socket.timeout:
                print("\t> Command timed out!", typeMsg="w")
                return None, None

        # Opt-in retry for IDEMPOTENT remote commands only (squeue poll, tar/rm during
        # retrieve). NEVER set retry_on_transient for a job submission -- a re-run
        # would double-launch it.
        return self.retry.run(
            "Remote exec",
            lambda: self._execute_remote_once(command_str, timeoutSecs, wait_for_all_commands),
            on_retry=self.connect,
        )

    def _execute_remote_once(self, command_str, timeoutSecs, wait_for_all_commands):
        # One exec_command attempt; raises on transient SSH/socket errors so the
        # caller's retry loop (when enabled) can reconnect. self.ssh is looked up
        # fresh each call so a reconnect's new client is picked up.
        output = None
        error = None
        stdin, stdout, stderr = self.ssh.exec_command(command_str, timeout=timeoutSecs)
        if wait_for_all_commands:
            stdin.close()
            output = stdout.read()
            error = stderr.read()
        return output, error

    def execute_local(self, command_str, printYN=False, timeoutSecs=None, **kwargs):
        if printYN:
            print("\t* Executing (local):", typeMsg="i")
            print(f"\t\t{command_str}")

        output, error = run_subprocess(
            [command_str], timeoutSecs=timeoutSecs, localRun=True
        )

        return output, error

    def retrieve(self, check_if_files_received=True, check_files_in_folder={}, optional_files=None, best_effort=False, spec=None):
        '''
        spec: what to bring back (default: the job's own RetrievalSpec, built in prep()).
        Status polls and submit-mode runs pass a narrowed spec so that nothing else on
        the remote is tarred, pruned or renamed.

        optional_files: files that we still try to pull from the remote (added
        to the tarball and unlinked locally before retrieval like the mandatory
        ones) but which are NOT flagged as "not received" when absent — used by
        `check()` for the slurm-job log, which does not exist yet while the job
        is PENDING and shouldn't cause a 60s retry on every status poll.
        '''
        spec = self.spec if spec is None else spec
        optional_files = list(optional_files) if optional_files else list(spec.optional)

        if getattr(self, "run_in_place", False):
            print("\t* In-place local execution: outputs already in folder_local (skipping retrieval)")
            if check_if_files_received:
                received = self.check_all_received(check_files_in_folder=check_files_in_folder, spec=spec)
                if received:
                    print("\t\t- All correct", typeMsg="i")
                return received
            return True

        print(f'\t* Retrieving files{" from remote server" if self.ssh is not None else ""}:')

        time_wait = 60
        received = False
        for attempt in (1, 2):

            if attempt == 2:
                print(f"\t* Not all received, trying retrieval once again after waiting {time_wait} seconds", typeMsg="i")
                time.sleep(time_wait)

            if not self._pull_outputs(spec, optional_files, best_effort=best_effort):
                return False

            if not check_if_files_received:
                return True

            received = self.check_all_received(check_files_in_folder=check_files_in_folder, spec=spec)
            if received:
                print("\t\t- All correct", typeMsg="i")
                break
            if best_effort:
                # Status poll: don't block on a 60s in-retrieve retry; the caller re-polls.
                print("\t* Not all expected files received (best-effort); caller will re-poll", typeMsg="i")
                break

        return received

    def _pull_outputs(self, spec, optional_files, best_effort=False):
        '''
        One tar + download + extract pass of `spec`. Returns False only when a
        best-effort retrieval gave up; otherwise it either succeeds or raises.
        '''

        # Defensively (re)create folder_local before any local FS or SFTP
        # operation: paramiko's sftp.get() opens the local destination via
        # `open(localpath, "wb")` and raises FileNotFoundError if the parent
        # directory is missing — this can hit a long-running PORTALS poll
        # whenever folder_local got cleaned up between submit and a later
        # check()/fetch() (mirrors load_submission_state's own re-attach
        # safety mkdir in SIMtools.py).
        self.folder_local.mkdir(parents=True, exist_ok=True)

        # Create a tarball of the output files & folders on the remote machine
        print("\t\t- Removing local output files & folders that potentially exist from previous runs")
        for file in list(spec.files) + optional_files:
            (self.folder_local / file).unlink(missing_ok=True)
        for folder in spec.folders:
            if (self.folder_local / folder).exists():
                IOtools.shutil_rmtree(self.folder_local / folder)

        # Create a tarball of the output files & folders on the remote machine
        print("\t\t- Tarballing (remote side)")

        self._resolve_fallbacks_on_remote(spec)

        # Build tar command with selective folder content
        tar_items = []

        # Add all output files (mandatory + best-effort)
        tar_items.extend(spec.files)
        tar_items.extend(optional_files)

        # Add folders - either full folders or selective content
        for folder in spec.folders:
            if folder in spec.selective:
                # Add specific files from this folder
                for pattern in spec.selective[folder]:
                    tar_items.append(f"{folder}/{pattern}")
            else:
                # Add entire folder
                tar_items.append(folder)

        # Quoted like folderExecution: tar does not expand wildcards on create and the
        # remote login shell's cwd is not folderExecution, so an item is always a literal
        # path here (a name with a space would otherwise split into two items)
        tar_items = " ".join(shlex.quote(str(item)) for item in tar_items)

        # Tar + download + extract, wrapped in a typeMsg='q' retry prompt:
        # the most common failure mode here (observed in production) is the
        # remote scratch pool filling up — tar produces no output and the
        # next sftp.get() crashes with a confusing FileNotFoundError 5
        # frames deep in paramiko. Surfacing a best-guess message and
        # blocking on 'y' lets the user free disk space (or fix whatever
        # transient remote issue) and resume the same retrieval without
        # losing the BO process. Answering 'n' re-raises the original
        # exception so the caller sees the real failure.
        while True:
            try:
                self.execute(
                    "tar -czf "
                    + f'{self.folderExecution}/mitim_receive.tar.gz'
                    + " -C "
                    + f'{self.folderExecution}'
                    + " "
                    + tar_items,
                    retry_on_transient=True,   # idempotent gather: safe to re-run across SSH flaps
                )

                # Download the tarball
                print("\t\t- Downloading (remote -> local)")
                if self.ssh is not None:
                    # Wrap the get in the shared transient-failure retry — same
                    # policy connect_ssh uses, namelist-tunable via PORTALS-CGYRO.
                    # The remote tarball at folderExecution/mitim_receive.tar.gz
                    # was created above and persists across reconnects.
                    with TqdmUpTo(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        desc="mitim_receive.tar.gz",
                        bar_format=" " * 20
                        + "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
                    ) as t:
                        self._sftp_transfer_with_retry(
                            'get',
                            f"{self.folderExecution}/mitim_receive.tar.gz",
                            self.folder_local / "mitim_receive.tar.gz",
                            callback=lambda sent, total_size: t.update_to(sent, total_size),
                        )
                else:
                    shutil.copy2(
                        f"{self.folderExecution}/mitim_receive.tar.gz",
                        self.folder_local / "mitim_receive.tar.gz"
                    )

                # Extract the tarball locally
                print("\t\t- Extracting tarball (local side)")
                with tarfile.open(self.folder_local / "mitim_receive.tar.gz", "r:gz") as tar:
                    tar.extractall(path=self.folder_local)
                break
            except (tarfile.ReadError, EOFError, OSError) as _retrieve_exc:
                if best_effort:
                    # Best-effort (status-poll) retrieval: a failure here is expected and
                    # transient -- the job is early-PENDING so its remote folder/log are not
                    # ready yet, or a brief SSH flap. Do NOT raise the interactive disk-full
                    # prompt: under an unattended run (stdout redirected to a log) that prompt
                    # hard-crashes the whole run. Abandon this attempt; the caller re-polls.
                    print(f"\t* Best-effort retrieval failed ({type(_retrieve_exc).__name__}: {_retrieve_exc}); "
                          f"outputs treated as not-yet-available (caller will re-poll)", typeMsg='w')
                    (self.folder_local / "mitim_receive.tar.gz").unlink(missing_ok=True)
                    return False
                if isinstance(_retrieve_exc, (TimeoutError, ConnectionError, socket.gaierror)):
                    # Network error that already survived the retry policy: it is not a full
                    # disk, and the prompt below would misdiagnose it
                    raise
                msg = (
                    f"Remote tar/download/extract failed "
                    f"({type(_retrieve_exc).__name__}: {_retrieve_exc}). "
                    f"Most likely cause: remote scratch is out of disk space. "
                    f"Remote folder: {self.folderExecution}. "
                    f"Free space on the cluster (delete old runs / clear scratch), "
                    f"then answer 'y' to retry; 'n' aborts and propagates the error."
                )
                if not print(msg, typeMsg='q'):
                    raise
                # Drop any partial local tarball before retrying so the next
                # attempt starts clean.
                (self.folder_local / "mitim_receive.tar.gz").unlink(missing_ok=True)

        # Remove tarballs
        print("\t\t- Removing tarball (local side)")
        (self.folder_local / "mitim_receive.tar.gz").unlink(missing_ok=True)
        print("\t\t- Removing tarball (remote side)")
        self.execute(f"rm {self.folderExecution}/mitim_receive.tar.gz")

        return True

    def _resolve_fallbacks_on_remote(self, spec):
        '''
        Remote-side primary/fallback resolution BEFORE the tar, so we only
        ever tar & transfer one file per pair (the fallback is typically
        same-order-of-magnitude size as the primary — no point paying
        double when we only want one). Per (folder, primary, fallback):
          primary present    -> remove fallback (dedup + free remote disk)
          fallback only      -> rename fallback to primary
          neither present    -> no-op
        Idempotent and safe if the folders don't exist yet.
        '''
        if not spec.fallbacks:
            return

        fallback_lines = []
        for folder, patterns in spec.selective.items():
            pattern_set = set(patterns)
            for primary, fallback in spec.fallbacks.items():
                if primary not in pattern_set:
                    continue
                p = f"{self.folderExecution}/{folder}/{primary}"
                f = f"{self.folderExecution}/{folder}/{fallback}"
                fallback_lines.append(
                    f'if [ -f "{p}" ]; then rm -f "{f}"; '
                    f'elif [ -f "{f}" ]; then mv "{f}" "{p}"; fi'
                )
        if fallback_lines:
            print(f"\t\t- Resolving {len(fallback_lines)} primary/fallback pair(s) on remote")
            self.execute(" ; ".join(fallback_lines))

    def remove_scratch_folder(self):
        # Safety guard: never rm -rf the user's working directory in in-place mode
        if getattr(self, "run_in_place", False):
            print("\t* Skipping scratch-folder removal (in-place local execution)")
            return None, None

        preserve = [str(p) for p in (self.preserve_subfolders or [])]
        if preserve:
            # Move the rescued sub-folders aside, wipe, and move them back: the
            # rest of the scratch tree is rebuilt from a fresh stage-in.
            print(f'\t* Removing{" remote" if self.ssh is not None else ""} folder, preserving {len(preserve)} rescued sub-folder(s)')
            fe = shlex.quote(str(self.folderExecution))
            aside = shlex.quote(str(self.folderExecution) + '.rescue')
            cmd = f"rm -rf {aside} && mkdir -p {aside}"
            for rel in preserve:
                r = shlex.quote(rel)
                cmd += f" && mkdir -p $(dirname {aside}/{r}) && mv {fe}/{r} {aside}/{r}"
            cmd += f" && rm -rf {fe} && mkdir -p {fe}"
            for rel in preserve:
                r = shlex.quote(rel)
                cmd += f" && mkdir -p $(dirname {fe}/{r}) && mv {aside}/{r} {fe}/{r}"
            cmd += f" && rm -rf {aside}"
            output, error = self.execute(cmd)
            # One-shot: the going-out wipe after a successful run must remove everything.
            # Cleared only once the command ran, so a failure keeps the list for the retry.
            self.preserve_subfolders = []
            return output, error

        print(f'\t* Removing{" remote" if self.ssh is not None else ""} folder')

        output, error = self.execute(f"rm -rf {self.folderExecution}")

        return output, error

    def probe_interrupted_runs(self, rel_folders, required_files, checksum_file, progress_file=None, progress_line=None, checksum_ignore_prefix=None, report_files=None):
        '''
        Look inside the (possibly remote) scratch folder for sub-folders left by an
        interrupted execution. For each entry of `rel_folders` (relative to
        folderExecution) returns {rel: (md5_of_checksum_file, progress_token, report)}
        when every file in `required_files` exists there, and nothing otherwise.
        `progress_token` is the first column of line `progress_line` (1-based; None =
        last line) of `progress_file` (e.g. the time the code will resume from), or
        None. `report` is a 'name=bytes ...' string with the sizes of `report_files`
        (forensics for the log; missing files are skipped). Lines of `checksum_file`
        starting with `checksum_ignore_prefix` (a prefix or a list of them, e.g.
        ['MAX_TIME', 'RESTART_STEP']) are excluded from the md5, so values the caller
        rewrites on rescue do not defeat the identity check. One shell round-trip in total.
        '''
        if getattr(self, 'run_in_place', False) or not rel_folders:
            return {}
        fe = str(self.folderExecution)
        checks = ' && '.join(f'[ -f "$d/{f}" ]' for f in list(required_files) + [checksum_file])
        pick = f'sed -n "{int(progress_line)}p"' if progress_line else 'tail -n 1'
        prog = f'$({pick} "$d/{progress_file}" 2>/dev/null | awk \'{{print $1}}\')' if progress_file else 'none'
        report = ' '.join(f'$([ -f "$d/{f}" ] && echo "{f}=$(wc -c < "$d/{f}" | tr -d " ")")' for f in (report_files or []))
        prefixes = [checksum_ignore_prefix] if isinstance(checksum_ignore_prefix, str) else list(checksum_ignore_prefix or [])
        filt = ''.join(f" | grep -v '^{p}'" for p in prefixes)
        lines = []
        for rel in rel_folders:
            d = f'{fe}/{rel}'
            lines.append(
                f'd={shlex.quote(d)}; if {checks}; then '
                f'h=$( cat "$d/{checksum_file}"{filt} | (md5sum 2>/dev/null || md5 -q) | cut -d" " -f1 ); '
                f'echo "MITIM_RESCUE {rel} $h {prog} | {report}"; fi'
            )
        with self.session(log_file=self.folder_local / 'paramiko.log'):
            output, _ = self.execute('; '.join(lines))
        found = {}
        for line in (output or b'').decode('utf-8', errors='ignore').splitlines():
            head, _, report_str = line.partition('|')
            parts = head.split()
            if len(parts) >= 3 and parts[0] == 'MITIM_RESCUE':
                found[parts[1]] = (parts[2], parts[3] if len(parts) > 3 else None, report_str.strip())
        return found

    def close(self, *args, **kwargs):
        if self.machineSettings["machine"] != "local":
            return self.close_ssh(*args, **kwargs)

    def close_ssh(self):
        print("\t* Closing connection")

        self._close_clients()

    # --------------------------------------------------------------------

    def _squeue_job_name(self):
        # Job name as submitted: native 'job-name' (set/migrated at define time),
        # with the legacy 'name' key as fallback for objects built without
        # going through define_machine_quick.
        return self.slurm_settings.get("job-name", self.slurm_settings.get("name", "mitim_job"))

    def check(self, file_output = "slurm_output.dat"):
        """
        Check job status slurm

            - If the job was launched with run(waitYN=False), then the script will not
                wait for the full slurm execution, but will launch it and retrieve the jobid,
                which the check command uses to check the status of the job.
            - If the class was initiated but not run, it will not have the jobid, so it will
                try to find it from the job_name, which must match the submitted one.
        """

        if self.jobid is not None:
            txt_look = f"-j {self.jobid}"
        else:
            txt_look = f"-n {self._squeue_job_name()}"

        command = f'cd {shlex.quote(str(self.folderExecution))} && squeue {txt_look} -o "%.15i %.50P %.18j %.10u %.10T %.10M %.10l %.5D %R" > squeue_output.dat'

        # Only squeue_output.dat is mandatory — it is what interpret_status() parses. The
        # slurm job log (`file_output`) is optional: it does not exist on the remote while
        # the job is still PENDING, and its absence simply means `interpret_status` sets
        # `self.log_file = None`. Everything else the job normally retrieves stays out of
        # this spec, so a mere status poll never tars a folder nor resolves the
        # primary/fallback pairs on the remote (rm/mv of bin.cgyro.restart.old).
        spec = self.spec.only("squeue_output.dat", optional=[file_output])

        # A status poll must never crash the run on a remote hiccup. The squeue
        # exec_command is retried across transient SSH/VPN flaps (retry_on_transient),
        # and the retrieval is best-effort; if a residual transient error still
        # surfaces (e.g. finite ssh_retry_attempts exhausted), degrade to "not received"
        # -> interpret_status treats it as pending/keep-polling.
        output = error = None
        with self.session():
            try:
                output, error = self.execute(command, printYN=True, retry_on_transient=True)
                received = self.retrieve(spec=spec, best_effort=True)
            except RetryPolicy.TRANSIENT as _poll_exc:
                print(f"\t* Status poll could not reach the remote ({type(_poll_exc).__name__}: {_poll_exc}); "
                      f"assuming job still pending (will re-poll)", typeMsg="w")
                received = False
            if not received and output is not None:
                self._write_debugging_files(output, error, extra_name = '_check')

        self.interpret_status(file_output = file_output)

    def interpret_status(self, file_output = "slurm_output.dat"):
        """
        Status of job:
            0: Submitted/pending
            1: Running
            2: Not found / finished
        """

        # -----------------------------------------------
        # Guard: squeue output could not be retrieved this poll (best-effort check
        # against an early-PENDING job whose remote folder is not ready yet, or a
        # transient SSH flap). Degrade to "pending / keep polling" -- NEVER "finished":
        # a missing poll must not be read as job-done, or the caller would proceed to
        # the next beat/step with no output. The polling loop retries next cycle.
        # -----------------------------------------------
        if not (self.folder_local / "squeue_output.dat").exists():
            print("\t* squeue output not retrieved this poll; assuming job still pending (will re-poll)", typeMsg="w")
            self.records = []
            self.infoSLURM = {"STATE": "UNKNOWN", "NAME": self._squeue_job_name(), "JOBID": None}
            self.jobid_found = None
            self.status = 0
            self.log_file = None
            return

        # -----------------------------------------------
        # Read output of squeue command -> self.records, self.infoSLURM
        # -----------------------------------------------

        with open(self.folder_local / "squeue_output.dat", "r") as f:
            self.records = SqueueRecord.parse(f.read())

        if not self.records:
            state = SlurmState.ABSENT
            self.infoSLURM = {"STATE": state.value}
            self.jobid_found = None
        else:
            # An array submission prints one row per element; infoSLURM describes the
            # first of them (node_of() answers the per-element questions)
            state = self.records[0].state
            self.infoSLURM = dict(self.records[0].fields)
            self.jobid_found = self.infoSLURM.get("JOBID")

        # -----------------------------------------------
        # Interpret status
        # -----------------------------------------------

        if state is SlurmState.PENDING:
            self.status = 0
        elif state in (SlurmState.RUNNING, SlurmState.COMPLETING):
            self.status = 1
        elif state is SlurmState.ABSENT:
            self.status = 2
        else:
            # Any other state the job can sit in (REQUEUED, SUSPENDED, CONFIGURING, ...)
            # means it is still in the queue: keep polling, never read it as finished.
            print(f"\t* SLURM state '{self.infoSLURM['STATE']}' not explicitly handled; assuming job still in the queue (will re-poll)", typeMsg="w")
            self.status = 0

        # ------------------------------------------------------------
        # If it was available, read the status of the ACTUAL slurm job
        # ------------------------------------------------------------

        if (self.folder_local / file_output).exists():
            with open(self.folder_local / file_output, "r") as f:
                self.log_file = f.readlines()
        else:
            self.log_file = None

        # ------------------------------------------------------------
        # Print info to screen
        # ------------------------------------------------------------

        txt = "\t* Job was checked"
        if (self.jobid is None) and (self.jobid_found is not None):
            txt += f' (jobid {self.jobid_found}, found from name "{self._squeue_job_name()}")'
        elif self.jobid is not None:
            txt += f" (jobid {self.jobid})"
        txt += f', is {self.infoSLURM["STATE"]} (job.infoSLURM)'
        if self.log_file is not None:
            txt += f". Log file (job.log_file) was retrieved, and has {len(self.log_file)} lines"
        print(txt)

    def node_of(self, array_index):
        '''
        Node running array element `array_index`, from the squeue rows of the last
        interpret_status(). Returns None when it cannot be resolved: no rows, the element
        is still inside a compressed "12345_[8-12]" pending row, or %R holds a reason
        instead of a node.
        '''
        if array_index is None:
            return None

        suffix = f"_{array_index}"
        for record in getattr(self, "records", []) or []:
            jobid = record.jobid or ""
            if jobid.endswith(suffix) and "[" not in jobid:
                return record.node
        return None

    def check_all_received(self, check_files_in_folder={}, spec=None):
        spec = self.spec if spec is None else spec
        print("\t* Checking if all files & folders that are expected were received")
        received = True

        # Check if all files were received
        for file in spec.files:
            if not (self.folder_local / file).exists():
                print(f"\t\t- File '{file}' not received", typeMsg="w")
                received = False

        for folder in spec.folders:
            # Check if all folders were received
            if not (self.folder_local / folder).exists():
                print(f"\t\t- Folder '{folder}/' not received", typeMsg="w")
                received = False
            # Check if all files in folder were received (optional information provided at job execution)
            else:
                if folder in check_files_in_folder:
                    for file in check_files_in_folder[folder]:
                        if not (self.folder_local / folder / file).exists():
                            print(f"\t\t- File '{file}' not received in folder '{folder}/'",typeMsg="w")
                            received = False

        return received

class TqdmUpTo(tqdm):
    def __init__(self, *args, **kwargs):
        self.enabled = read_verbose_level() in [4, 5]
        if not self.enabled:
            # Create a 'dummy' progress bar (does nothing)
            kwargs['disable'] = True
        super().__init__(*args, **kwargs)
        self.initialized = False

    def update_to(self, sent, total_size):
        if self.enabled:
            if not self.initialized:
                self.total = total_size
                self.initialized = True
            self.update(sent - self.n)  # will also set self.n = sent

""" 
	Timeout function
	- I just need to add a function:
		with timeout(secs):
			do things
	- I don't recommend that "do things" includes a context manager like with Popen or it won't work... not sure why
"""


def raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def timeout(time, proc=None):
    time = int(time)
    if time < 1e6:
        print(
            f'\t\t* Note: this process will be killed if time exceeds {time}sec of execution ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})',
            typeMsg="i",
        )

    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        print(
            f'\t\t\t* Killing process! ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})',
            typeMsg="w",
        )
        if proc is not None:
            proc.kill()
            outs, errs = proc.communicate()
    finally:
        # Unregister the signal so it won't be triggered if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def run_subprocess(commandExecute, timeoutSecs=None, localRun=False):
    """
    Note (PRF):
        Note that before I had a context such as "with Popen() as p:" but that failed to catch time outs!
        So, even though I don't know why... I'm doing this directly, with opening and closing it
        For local runs, I had originally:
                error=None; result=None;
                os.system(Command)
        Now, it uses subprocess with shell. This is because I couldn't load "source" because is a shell command, with simple os.system()
        New solution is not the safest but it works.
    """

    if localRun:
        shell = True
        executable = "/bin/bash"
    else:
        shell = False
        executable = None

    p = subprocess.Popen(
        commandExecute,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable=executable,
    )

    result, error = None, None
    if timeoutSecs is not None:
        with timeout(timeoutSecs, proc=p):
            result, error = p.communicate()
            p.stdout.close()
            p.stderr.close()
    else:
        result, error = p.communicate()
        p.stdout.close()
        p.stderr.close()

    return result, error


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
FUNCTIONS THAT HANDLE PARELELIZATION OF ANY FUNCTION
Usage:
	- Function must be able to take 
		Params (a list/dict containing the fixed parameters for all evaluations) and cont.
		Once inside the function, I should be able to do whatever with Params and cont, but
		cont is provided by the workflow automatically from 0 to (n-1), with n number of parallels
		Example:
			def FunctionToParallelize(Params,cont):
				return Params['FixedValue']*Params['VariableArray'][cont]

			Params = {'FixedValue': 10, 'VariableArray': [2,4,6,8,9]}
			y = ParallelProcedure(FunctionToParallelize,Params,parallel=5,howmany=5)

			This would generate y = [20,40,60,80,90] in parallel
"""


def init(l_lock):
    global lock
    lock = l_lock


class MITIM_ParallelClass_reduced(object):
    def __init__(self, Function, Params):
        self.Params = Params
        self.Function = Function

    def __call__(self, cont):
        self.Params["lock"] = lock
        return self.Function(self.Params, cont)


def ParallelProcedure(
    Function, Params, parallel=8, howmany=8, array=True, on_dill=True
):
    if on_dill:
        import multiprocessing_on_dill as multiprocessing
    else:
        import multiprocessing

    """
	This way of pooling passes a lock when initializing every child class. It handles
	a global lock, and then every child can call lock.acquire() and lock.release()
	so that for instance not two at the same time open and write the same file.
	"""

    l0 = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init, initargs=(l0,), processes=parallel)

    if array:
        print(
            f'\n~~~~~~~~~~~~~~~~~~ Launching batch of {howmany} evaluations ({parallel} in parallel), {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ~~~~~~~~~~~~~~~~~~'
        )
    res = pool.map(MITIM_ParallelClass_reduced(Function, Params), np.arange(howmany))
    if array:
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        )

    pool.close()

    if array:
        return np.array(res)
    else:
        return res


def SerialProcedure(Function, Params, howmany):
    y, yE = [], []
    for cont in range(howmany):
        y1, yE1 = Function(Params, cont)
        y.append(y1)
        yE.append(yE1)

    y = np.array(y)
    yE = np.array(yE)

    return y, yE


LOCKFILE_TEMPLATE = string.Template('''LOCK_FILE="$lock_path"
LOCK_TIMEOUT=$timeout_seconds  # $hours hours in seconds

# Check if lock file already exists
if [ -f "$$LOCK_FILE" ]; then
    # Get current time and file modification time
    CURRENT_TIME=$$(date +%s)
    FILE_TIME=$$(stat -c%Y "$$LOCK_FILE" 2>/dev/null || stat -f%m "$$LOCK_FILE" 2>/dev/null)
    AGE=$$((CURRENT_TIME - FILE_TIME))
${sp}
    # If lock is older than $hours hours, delete it and continue
    if [ "$$AGE" -gt "$$LOCK_TIMEOUT" ]; then
        echo "Lock file is stale ($$(($$AGE / 3600)) hours old), removing and proceeding..."
        rm -f "$$LOCK_FILE"
    else
        # Lock is recent, another job may be running
        echo "ERROR: Lock file exists ($$(($$AGE / 60)) minutes old). Another job may be running."
        exit 1
    fi
fi

# Cleanup function runs on exit (success, failure, or timeout)
cleanup() {
    rm -f "$$LOCK_FILE"
    echo "Lock file cleaned up at $$(date)"
}

trap cleanup EXIT

# Create lock file
touch "$$LOCK_FILE"
echo "Lock file created at $$(date)"
''')


class SbatchScript:
    '''
    Builder of the sbatch script MITIM submits: the #SBATCH directives from the per-job
    settings and the machine allocation, the banner/export preamble, and the optional
    lock-file guard. Both dicts are copied, and the 'minutes'/'name' legacy keys are
    migrated here, so nothing written back reaches the caller (slurm_allocation is
    machineSettings["slurm"], aliased by reference to the process-global config cache).
    '''

    def __init__(self, settings=None, allocation=None):
        self.settings = dict(settings or {})
        self.allocation = dict(allocation or {})

        # Back-compat: migrate the legacy 'minutes' key to the native 'time' key. mitim_job-direct
        # callers (e.g. TRANSPsingularity) still pass 'minutes' instead of going through
        # SLURMtools.resolve(); without this they'd silently fall back to the "10:00" default below.
        if "time" not in self.settings and "minutes" in self.settings:
            _m = int(self.settings["minutes"])
            self.settings["time"] = f"{_m//60:02d}:{_m%60:02d}:00" if _m >= 60 else f"{_m:02d}:00"

        # ---- Native sbatch keys (the only schema we support) -----------------
        self.name            = self.settings.setdefault("job-name", "mitim_job")
        self.time            = self.settings.setdefault("time", "10:00")
        memory_req_by_job    = self.settings.setdefault("mem", None)

        self.nodes           = self.settings.setdefault("nodes", None)
        self.ntasks          = self.settings.setdefault("ntasks", None)
        self.cpuspertask     = self.settings.setdefault("cpus-per-task", None)
        self.ntaskspernode   = self.settings.setdefault("ntasks-per-node", None)
        self.gpuspertask     = self.settings.setdefault("gpus-per-task", None)
        self.gpuspernode     = self.settings.setdefault("gpus-per-node", None)

        self.array           = self.settings.setdefault("array", None)
        self.array_limit     = self.settings.setdefault("array_limit", None)
        job_exclusive        = self.settings.setdefault("exclusive", False)

        # Requeue-ability: True (default) emits --requeue, False emits --no-requeue,
        # None leaves the cluster default. Explicit --requeue makes behavior uniform
        # across clusters (slurm.conf JobRequeue varies): on preemption or node
        # failure the job goes back in the queue under the same id instead of dying,
        # and MITIM workflows resume from their on-disk checkpoints when re-executed.
        self.requeue         = self.settings.setdefault("requeue", True)

        # ---- Machine specifications as given by the config, not by the job ---
        self.partition       = self.allocation.setdefault("partition", None)
        self.qos             = self.allocation.setdefault("qos", None)
        self.email           = self.allocation.setdefault("email", None)
        self.exclude         = self.allocation.setdefault("exclude", None)
        self.account         = self.allocation.setdefault("account", None)
        self.constraint      = self.allocation.setdefault("constraint", None)
        memory_req_by_config = self.allocation.setdefault("mem", None)
        request_exclusive_node = self.allocation.setdefault("exclusive", False)

        if memory_req_by_job == 0 :
            print("\t\t- Entire node memory requested by job, overwriting memory requested by config file", typeMsg="i")
            self.memory = memory_req_by_job
        elif memory_req_by_job is not None:
            print(f"\t\t- Memory requested by job ({memory_req_by_job}), overwriting memory requested by config file", typeMsg="i")
            self.memory = memory_req_by_job
        else:
            if memory_req_by_config is not None:
                print(f"\t\t- Memory requested by config file ({memory_req_by_config})", typeMsg="i")
            self.memory = memory_req_by_config

        # --exclusive can co-exist with arrays (one whole node per array element)
        # and with packed jobs (whole nodes via per-job slurm_settings). Honor
        # both the machine config (`allocation`) and the per-job override
        # (`settings.exclusive`). A string value (e.g. "user" or "mcs") emits
        # --exclusive=<value>, which keeps OTHER users off the node while letting this
        # user's own array tasks pack onto it; a bare True stays plain --exclusive
        # (one whole node per job).
        self.exclusive = request_exclusive_node or job_exclusive

    def directives(self, folderExecution, label_log_files="", if_array_relabel=False, append_mode=False):
        lines = ["#!/usr/bin/env bash"]

        lines.append(f"#SBATCH --job-name {self.name}")
        if (not if_array_relabel) or (self.array is None):
            lines.append(f"#SBATCH --output {folderExecution}/slurm_output{label_log_files}.dat")
            lines.append(f"#SBATCH --error {folderExecution}/slurm_error{label_log_files}.dat")
        else:
            lines.append(f"#SBATCH --output {folderExecution}/slurm_output{label_log_files}_%A_%a.dat")
            lines.append(f"#SBATCH --error {folderExecution}/slurm_error{label_log_files}_%A_%a.dat")
        lines.append(f"#SBATCH --time {self.time}")
        if self.email is not None:
            lines.append("#SBATCH --mail-user=" + self.email)
        if self.partition is not None:
            lines.append(f"#SBATCH --partition {self.partition}")
        if self.account is not None:
            lines.append(f"#SBATCH --account {self.account}")
        if self.qos is not None:
            lines.append(f"#SBATCH --qos {self.qos}")
        if self.constraint is not None:
            lines.append(f"#SBATCH --constraint {self.constraint}")
        if self.memory is not None:
            lines.append(f"#SBATCH --mem {self.memory}")
        if self.array is not None:
            lines.append(f"#SBATCH --array={self.array}{f'%{self.array_limit} ' if self.array_limit is not None else ''}")
        if self.exclusive:
            lines.append(f"#SBATCH --exclusive={self.exclusive}" if isinstance(self.exclusive, str) else "#SBATCH --exclusive")
        if self.requeue is True:
            lines.append("#SBATCH --requeue")
        elif self.requeue is False:
            lines.append("#SBATCH --no-requeue")
        if self.nodes is not None:
            lines.append(f"#SBATCH --nodes {self.nodes}")
        if self.ntasks is not None:
            lines.append(f"#SBATCH --ntasks {self.ntasks}")
        if self.ntaskspernode is not None:
            lines.append(f"#SBATCH --ntasks-per-node {self.ntaskspernode}")
        if self.cpuspertask is not None:
            lines.append(f"#SBATCH --cpus-per-task {self.cpuspertask}")
        if self.gpuspertask is not None:
            lines.append(f"#SBATCH --gpus-per-task {self.gpuspertask}")
        if self.gpuspernode is not None:
            lines.append(f"#SBATCH --gpus-per-node={self.gpuspernode}")
        if self.exclude is not None:
            lines.append(f"#SBATCH --exclude={self.exclude}")
        if append_mode:
            lines.append("#SBATCH --open-mode=append")

        lines.append("#SBATCH --profile=all")

        return lines

    def preamble(self):
        lines = ["", "export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"]
        if self.gpuspernode is not None:
            lines.append('export SLURM_CPU_BIND="cores"')
        lines.append('echo "MITIM: Submitting SLURM job $SLURM_JOBID in $HOSTNAME (host: $SLURM_SUBMIT_HOST)"')
        lines.append('echo "MITIM: Nodes have $SLURM_CPUS_ON_NODE cores and $SLURM_JOB_NUM_NODES node(s) were allocated for this job"')
        lines.append('echo "MITIM: Each of the $SLURM_NTASKS tasks allocated will run with $SLURM_CPUS_PER_TASK cores, allocating $SRUN_CPUS_PER_TASK CPUs per srun"')
        lines.append('echo "***********************************************************************************************"')
        lines.append('echo ""')
        lines.append("")
        return lines

    def lockfile_block(self, folderExecution, hours):
        # Refuses to start a second job on the same folder unless the lock has gone stale
        block = LOCKFILE_TEMPLATE.substitute(
            lock_path=f"{folderExecution}/job.lock",
            timeout_seconds=int(hours * 3600),
            hours=hours,
            sp="    ",   # the blank line inside the `if` keeps its indentation
        )
        return block.split("\n")


def create_slurm_execution_files(
    command,
    folderExecution,
    modules_remote=None,
    folder_local=None,
    shellPreCommands=None,
    shellPostCommands=None,
    label_log_files="",
    wait_until_sbatch=True,
    slurm_allocation=None,
    launchSlurm=True,
    slurm_settings = None,
    if_array_relabel = False,
    lock_file=None,
    lock_file_timeout_hours=12,
    append_mode = False
):

    fileSBATCH = folder_local / f"mitim_bash{label_log_files}.src"
    fileSHELL = folder_local / f"mitim_shell_executor{label_log_files}.sh"
    fileSBATCH_remote = f"{folderExecution}/mitim_bash{label_log_files}.src"

    script = SbatchScript(settings=slurm_settings, allocation=slurm_allocation)

    """
	********************************************************************************************
	Write mitim_bash.src file to execute
	********************************************************************************************
	"""

    command = [command] if isinstance(command, str) else command
    shellPreCommands = [] if shellPreCommands is None else shellPreCommands
    shellPostCommands = [] if shellPostCommands is None else shellPostCommands

    commandSBATCH = script.directives(
        folderExecution,
        label_log_files=label_log_files,
        if_array_relabel=if_array_relabel,
        append_mode=append_mode,
    )
    commandSBATCH.extend(script.preamble())

    if lock_file:
        commandSBATCH.extend(script.lockfile_block(folderExecution, lock_file_timeout_hours))

    # If modules, add them, but also make sure I expand the potential aliases that they may have!
    full_command = ["shopt -s expand_aliases",modules_remote] if (modules_remote is not None) else []

    full_command.extend(command)
    for c in full_command:
        commandSBATCH.append(c)

    commandSBATCH.append("")

    wait_txt = " --wait" if wait_until_sbatch else ""
    if launchSlurm:
        comm, launch = commandSBATCH, "sbatch" + wait_txt + " "
    else:
        comm, launch = ["#!/usr/bin/env bash"] + full_command, ""

    fileSBATCH.unlink(missing_ok=True)
    with open(fileSBATCH, "w", newline="") as f:
        f.write("\n".join(comm))

    """
	********************************************************************************************
	Write mitim_shell_executor.sh file that handles the execution of the mitim_bash.src with pre and post commands
	********************************************************************************************
	"""

    commandSHELL = ["#!/usr/bin/env bash"]

    commandSHELL.append("")
    if modules_remote is not None:
        commandSHELL.append(modules_remote)

    commandSHELL.extend(copy.deepcopy(shellPreCommands))

    commandSHELL.append(f"{launch} {fileSBATCH_remote}")
    commandSHELL.append("")
    for i in range(len(shellPostCommands)):
        commandSHELL.append(shellPostCommands[i])

    fileSHELL.unlink(missing_ok=True)
    with open(fileSHELL, "w", newline="") as f:
        f.write("\n".join(commandSHELL))

    """
	********************************************************************************************
	Command to send through scp
	********************************************************************************************
	"""

    comm = f"cd {shlex.quote(str(folderExecution))} && chmod +x {fileSBATCH_remote} && chmod +x mitim_shell_executor{label_log_files}.sh && ./mitim_shell_executor{label_log_files}.sh > mitim.out"

    return comm, fileSBATCH.resolve(), fileSHELL.resolve()


def curateOutFiles(outputFiles):
    # Avoid repetitions, otherwise, e.g., they will fail to rename

    if "mitim.out" not in outputFiles:
        outputFiles.append("mitim.out")

    outputFiles_new = []
    for file in outputFiles:
        if file not in outputFiles_new:
            outputFiles_new.append(file)

    return outputFiles_new


def printEfficiencySLURM(out_file):
    """
    It reads jobid from mitim.out or slurm_output.dat
    """

    with open(out_file, "r") as f:
        aux = f.readlines()

    jobid = None
    for line in aux:
        if ("Submitted batch job" in line) or ("Submitting SLURM job" in line):
            # The two messages put the id in different columns ("Submitted batch job <id>"
            # vs "MITIM: Submitting SLURM job <id> in <host> ..."): take the first number
            jobid = next((int(token) for token in line.split() if token.isdigit()), None)
            if jobid is not None:
                break

    if jobid is not None:
        print(f"Evaluating efficienty of job {jobid}:")
        try:
            print("\n****** SEFF:")
            subprocess.run(["seff", str(jobid)])
            print("\n****** SACCT:")
            subprocess.run(["sacct", "-j", str(jobid)])
        except FileNotFoundError as e:
            print(f"\t- SLURM utility not available ({e})", typeMsg='w')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for quick remote executions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def perform_quick_remote_execution(
    folder_local,
    machine,
    command,
    input_files=None,
    input_folders=None,
    output_files=None,
    output_folders=None,
    job_name = "test",
    check_if_files_received=True,
    ):

    if input_files is None:
        input_files = []
    if input_folders is None:
        input_folders = []
    if output_files is None:
        output_files = []
    if output_folders is None:
        output_folders = []

    job = mitim_job(folder_local)

    # Define machine
    job.slurm_settings, job.launchSlurm = {}, False
    job.machineSettings = CONFIGread.machineSettings(code=None,nameScratch=job_name,forceMachine=machine,append_folder_local=folder_local)
    job.folderExecution = job.machineSettings["folderWork"]
    job.run_in_place = bool(job.machineSettings.get("run_in_place", False))

    # Submit
    job.prep(
        command,
        input_files=input_files,
        input_folders=input_folders,
        output_files=output_files,
        output_folders=output_folders)
    job.run(check_if_files_received=check_if_files_received)


def retrieve_files_from_remote(
    folder_local,
    machine,
    files_remote = [],
    folders_remote = [],
    only_folder_structure_with_files = None, # If not None, only the folder structure is retrieved, with files in the list
    purge_tmp_files = False,
    ensure_files = True
    ):
    '''
    Quick routine for file retrieval from remote machine (assumes remote machine is linux)

    e.g.:
            mitim_plot_portals run2 --remote engaging:path_to_folder_remote_where_run2_is/

    '''

    # Ensure Paths
    folder_local = Path(folder_local)

    job_name = 'file_retrieval'

    # ------------------------------------------------
    # Prep files and folders to be transfered
    # ------------------------------------------------

    machineSettings = CONFIGread.machineSettings(code=None,nameScratch=job_name,forceMachine=machine,append_folder_local=folder_local)

    command, output_files, output_folders = '', [], []
    for file in files_remote:
        file0 = file.split('/')[-1]
        command += f'cp {file} {machineSettings["folderWork"]}/{file0}\n'
        output_files.append(file0)
    for folder in folders_remote:
        folder0 = f'{IOtools.expandPath(folder)}'.split('/')[-1]
        
        folder_source = folder
        folder_destination = f'{machineSettings["folderWork"]}/{folder0}'
        if only_folder_structure_with_files is None:
            # Normal full copy
            command += f'cp -r {folder_source} {folder_destination}\n'
        else:
            retrieve_files = ''
            for file in only_folder_structure_with_files:
                retrieve_files += f'-f"+ {file}" '
            # Only copy the folder structure with a few files
            command += f'rsync -av -f"+ */" {retrieve_files}-f"- *" {folder_source}/ {folder_destination}/\n'
            
        output_folders.append(folder0)

    # ------------------------------------------------
    # Run
    # ------------------------------------------------

    perform_quick_remote_execution(
        folder_local,
        machine,
        command,
        output_files = output_files,
        output_folders = output_folders,
        job_name = job_name,
        check_if_files_received = ensure_files,
    )

    if purge_tmp_files:
        # Remote files created in this process
        for file in ['mitim_bash.src', 'mitim_shell_executor.sh', 'paramiko.log', 'mitim.out']:
            (folder_local / file).unlink(missing_ok=True)
    
    # Return local addresses
    folders = [folder_local / IOtools.reducePathLevel(folder)[-1] for folder in folders_remote]
    files = [folder_local / IOtools.reducePathLevel(file)[-1] for file in files_remote]

    return files, folders


if __name__ == "__main__":
    printEfficiencySLURM(sys.argv[1])
