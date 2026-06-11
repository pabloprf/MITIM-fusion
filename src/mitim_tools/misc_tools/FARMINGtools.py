"""
Set of tools to farm out simulations to run in either remote clusters or locally, serially or parallel
"""

from math import log
from tqdm import tqdm
import os
import shlex
import shutil
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

        # Populated in prep() for the submit path and in load_submission_state()
        # for the re-attach path. Initialised here so retrieve() can read it
        # unconditionally even on job instances that bypass prep() entirely
        # (e.g. mitim_job.check() builds a temporary retrieve for squeue output).
        self.output_file_fallbacks = {}

        # Optional retry config consumed by connect_ssh(). Callers that want
        # to override the default 5 s wait / 3-attempt cap (e.g. PORTALS-CGYRO,
        # which reads it from portals.namelist) can set this to a dict of
        # {"wait_seconds": float, "attempts": int|None} where attempts=None
        # means retry forever. When self.connection_retry_settings is None,
        # connect_ssh() falls back to the historical defaults.
        self.connection_retry_settings = None

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

        output_folders_selective is a dictionary with folder name as key and list of specific files/patterns to include as value.
            e.g., {'results': ['*.dat', '*.log'], 'plots': ['figure1.png']}

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
        self.output_files = output_files if isinstance(output_files, list) else []
        self.output_folders = output_folders if isinstance(output_folders, list) else []
        self.check_files_in_folder = check_files_in_folder

        self.shellPreCommands = shellPreCommands if isinstance(shellPreCommands, list) else []
        self.shellPostCommands = shellPostCommands if isinstance(shellPostCommands, list) else []
        self.label_log_files = label_log_files

        self.output_folders_selective = output_folders_selective if isinstance(output_folders_selective, dict) else {}
        self.output_file_fallbacks = output_file_fallbacks if isinstance(output_file_fallbacks, dict) else {}

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
        # later check()/fetch() loop). The rest of self.output_files /
        # self.output_folders is the post-run output spec, which at this
        # point on the remote contains *only* what we just uploaded
        # (input.cgyro per rho, multi-GB restart binaries, etc.) — pulling
        # it back would round-trip those bytes for nothing. The full spec
        # is restored before this method returns so the eventual fetch()
        # (SIMtools.fetch -> simulation_job.retrieve()) sees the original
        # CGYRO output list. Mirrors the same backup/restore dance used in
        # `mitim_job.check()` to scope retrieve() to squeue_output.dat.
        submit_minimal_retrieve = not waitYN
        if submit_minimal_retrieve:
            _saved_output_files = list(self.output_files)
            _saved_output_folders = list(self.output_folders)
            _saved_output_folders_selective = dict(self.output_folders_selective)
            self.output_files = ["mitim.out"]
            self.output_folders = []
            self.output_folders_selective = {}

        try:
            self.full_process(
                comm,
                removeScratchFolders_goingIn=removeScratchFolders_goingIn and (not helper_lostconnection),
                removeScratchFolders_goingOut=removeScratchFolders_goingOut,
                timeoutSecs=timeoutSecs,
                check_if_files_received=waitYN and check_if_files_received,
                check_files_in_folder=self.check_files_in_folder,
                attempts_execution=attempts_execution,
                execute_flag=execute_case_flag and (not helper_lostconnection)
            )
        finally:
            if submit_minimal_retrieve:
                self.output_files = _saved_output_files
                self.output_folders = _saved_output_folders
                self.output_folders_selective = _saved_output_folders_selective

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
        if self.ssh is None:
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

        time_init = datetime.datetime.now()
        print(f"\n\t-------------- Running process ({time_init.strftime('%Y-%m-%d %H:%M:%S')}{f', will timeout execution in {timeoutSecs}s' if timeoutSecs < 1e6 else ''}) --------------")

        # ~~~~~~ Connect
        self.connect(log_file=self.folder_local / "paramiko.log")

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

        while execution_counter < attempts_execution:
            
            if execute_flag:
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

            # If not received, write output and error to files
            self._write_debugging_files(output, error)

            cont = print(f"\t* Not all expected files received, not removing scratch folder (mitim_farming.out and mitim_farming.err written in '{self.folder_local / 'mitim_farming.err'}')",typeMsg="q")
            if not cont:
                print("[MITIM] Stopped with embed(), you can look at output and error",typeMsg="w",)
                embed()

        # ~~~~~~ Close
        self.close()

        print(f"\t-------------- Finished process (took {IOtools.getTimeDifference(time_init)}) --------------\n")

    def _write_debugging_files(self, output, error, extra_name=""):
            with open(self.folder_local / f"mitim_farming{extra_name}.out", "w") as f:
                f.write(output.decode("utf-8"))
            with open(self.folder_local / f"mitim_farming{extra_name}.err", "w") as f:
                f.write(error.decode("utf-8"))

    def connect(self, *args, **kwargs):
        if self.machineSettings["machine"] != "local":
            return self.connect_ssh(*args, **kwargs)
        else:
            self.jump_client, self.ssh, self.sftp = None, None, None

    def connect_ssh(self, log_file=None):
        self.jump_host = self.machineSettings["tunnel"]
        self.jump_user = self.machineSettings["user"]

        self.target_host = self.machineSettings["machine"]
        self.target_user = self.machineSettings["user"]

        print("\t* Connecting to remote server:")
        print(f'\t\t{self.target_user}@{self.target_host}{f", via tunnel {self.jump_user}@" +self.jump_host  if self.jump_host is not None else ""}{":" + str(self.machineSettings["port"]) if self.machineSettings["port"] is not None else ""}{" with key " + self.machineSettings["identity"] if self.machineSettings["identity"] is not None else ""}')

        if log_file is not None:
            paramiko.util.log_to_file(log_file)

        # Resolve retry config. Defaults preserve historical behavior (5 s wait,
        # 3-attempt cap). PORTALS-CGYRO overrides these via portals.namelist:
        #   transport.options.cgyro.run.ssh_retry_wait_seconds
        #   transport.options.cgyro.run.ssh_retry_attempts   (int, or null/None for infinite)
        # which transport_cgyro.py forwards onto self.connection_retry_settings.
        retry_cfg = getattr(self, "connection_retry_settings", None) or {}
        retry_wait = float(retry_cfg.get("wait_seconds", 5))
        retry_attempts = retry_cfg.get("attempts", 3)
        # None (e.g. YAML `null`) means retry forever; any positive int caps the
        # attempts. Anything else is a config error and surfaces here.
        if retry_attempts is not None and (not isinstance(retry_attempts, int) or retry_attempts < 1):
            raise ValueError(
                f"connection_retry_settings['attempts'] must be a positive int "
                f"or None (infinite); got {retry_attempts!r}"
            )
        max_retries = retry_attempts  # None == unbounded

        # Transient handshake/network errors. TimeoutError is the case Pablo
        # reported (Errno 60 from paramiko's underlying socket); SSHException
        # covers paramiko's own transient class; socket.timeout / EOFError /
        # ConnectionError / socket.gaierror cover the rest of the typical
        # VPN/firewall flap modes (gaierror = DNS resolution failure when
        # the VPN drops mid-poll). Anything outside this tuple is treated
        # as a real failure and re-raised on the first occurrence.
        transient_exc = (
            paramiko.ssh_exception.SSHException,
            TimeoutError,
            socket.timeout,
            EOFError,
            ConnectionError,
            socket.gaierror,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                self._connect_ssh_item()
                break
            except transient_exc as e:
                if max_retries is not None and attempt >= max_retries:
                    raise
                cap_str = "infinite" if max_retries is None else f"{max_retries}"
                print(
                    f"\t<> Paramiko connect attempt {attempt}/{cap_str} failed "
                    f"({type(e).__name__}: {e}). Retrying in {retry_wait:g}s...",
                    typeMsg="w",
                )
                time.sleep(retry_wait)

    def _connect_ssh_item(self):

        try:
            self.define_jump()
            self.define_server()
        except paramiko.ssh_exception.AuthenticationException:
            # If it fails, try to disable rsa-sha2-512 and rsa-sha2-256 (e.g. for iris.gat.com)
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
        Retry a paramiko SFTP transfer ('get' or 'put') using the same
        policy `connect_ssh` uses, sourced from self.connection_retry_settings
        (5 s wait / 3 attempts by default; PORTALS-CGYRO namelist-tunable;
        attempts=None means infinite). On a transient transport-closed /
        socket-timeout / EOF / connection-reset failure mid-transfer,
        rebuild self.ssh + self.sftp via self.connect() and re-issue the
        op. The remote tarballs (mitim_send.tar.gz / mitim_receive.tar.gz)
        persist across reconnects, so callers do not need to re-tar.
        Persistent connection failure is caught by connect_ssh's own retry
        surface; a single reconnect failure here is logged and the next
        loop iteration retries the transfer.

        sftp_method_name is looked up on self.sftp on every attempt — the
        bound method must NOT be captured before the loop, because reconnect
        replaces self.sftp with a fresh SFTPClient instance.
        '''
        retry_cfg = getattr(self, "connection_retry_settings", None) or {}
        retry_wait = float(retry_cfg.get("wait_seconds", 5))
        retry_attempts = retry_cfg.get("attempts", 3)
        if retry_attempts is not None and (not isinstance(retry_attempts, int) or retry_attempts < 1):
            raise ValueError(
                f"connection_retry_settings['attempts'] must be a positive int "
                f"or None (infinite); got {retry_attempts!r}"
            )
        transient_exc = (
            paramiko.ssh_exception.SSHException,
            TimeoutError,
            socket.timeout,
            EOFError,
            ConnectionError,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                getattr(self.sftp, sftp_method_name)(*args, **kwargs)
                return
            except transient_exc as e:
                if retry_attempts is not None and attempt >= retry_attempts:
                    raise
                cap_str = "infinite" if retry_attempts is None else f"{retry_attempts}"
                print(
                    f"\t<> Paramiko sftp.{sftp_method_name} attempt {attempt}/{cap_str} failed "
                    f"({type(e).__name__}: {e}). Reconnecting & retrying in {retry_wait:g}s...",
                    typeMsg="w",
                )
                time.sleep(retry_wait)
                try:
                    self.connect()
                except Exception as _conn_e:
                    print(
                        f"\t<> reconnect during sftp.{sftp_method_name} failed ({_conn_e}); "
                        "will retry transfer on next loop iteration",
                        typeMsg="w",
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

        # Opt-in retry for IDEMPOTENT remote commands (squeue poll, tar/rm during
        # retrieve) across transient SSH/VPN flaps, mirroring connect_ssh /
        # _sftp_transfer_with_retry (same connection_retry_settings; ssh_retry_attempts
        # null == retry forever). Previously a drop between connect and the squeue/tar
        # exec_command raised an uncaught SSHException even with retry-forever requested.
        # NEVER set retry_on_transient for a job submission -- re-running would double-launch.
        retry_cfg = getattr(self, "connection_retry_settings", None) or {}
        retry_wait = float(retry_cfg.get("wait_seconds", 5))
        retry_attempts = retry_cfg.get("attempts", 3)
        if retry_attempts is not None and (not isinstance(retry_attempts, int) or retry_attempts < 1):
            raise ValueError(
                f"connection_retry_settings['attempts'] must be a positive int "
                f"or None (infinite); got {retry_attempts!r}"
            )
        transient_exc = (
            paramiko.ssh_exception.SSHException,
            TimeoutError,
            socket.timeout,
            EOFError,
            ConnectionError,
            socket.gaierror,
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._execute_remote_once(command_str, timeoutSecs, wait_for_all_commands)
            except transient_exc as e:
                if retry_attempts is not None and attempt >= retry_attempts:
                    raise
                cap_str = "infinite" if retry_attempts is None else f"{retry_attempts}"
                print(
                    f"\t<> Remote exec attempt {attempt}/{cap_str} failed "
                    f"({type(e).__name__}: {e}). Reconnecting & retrying in {retry_wait:g}s...",
                    typeMsg="w",
                )
                time.sleep(retry_wait)
                try:
                    self.connect()
                except Exception as _conn_e:
                    print(
                        f"\t<> reconnect during remote exec failed ({_conn_e}); "
                        "will retry on next iteration",
                        typeMsg="w",
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

    def retrieve(self, check_if_files_received=True, check_files_in_folder={}, optional_files=None, best_effort=False):
        '''
        optional_files: files that we still try to pull from the remote (added
        to the tarball and unlinked locally before retrieval like the mandatory
        ones) but which are NOT flagged as "not received" when absent — used by
        `check()` for the slurm-job log, which does not exist yet while the job
        is PENDING and shouldn't cause a 60s retry on every status poll.
        '''
        optional_files = list(optional_files) if optional_files else []

        if getattr(self, "run_in_place", False):
            print("\t* In-place local execution: outputs already in folder_local (skipping retrieval)")
            if check_if_files_received:
                received = self.check_all_received(check_files_in_folder=check_files_in_folder)
                if received:
                    print("\t\t- All correct", typeMsg="i")
                return received
            return True

        print(f'\t* Retrieving files{" from remote server" if self.ssh is not None else ""}:')

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
        for file in list(self.output_files) + optional_files:
            (self.folder_local / file).unlink(missing_ok=True)
        for folder in self.output_folders:
            if (self.folder_local / folder).exists():
                IOtools.shutil_rmtree(self.folder_local / folder)

        # Create a tarball of the output files & folders on the remote machine
        print("\t\t- Tarballing (remote side)")

        # Remote-side primary/fallback resolution BEFORE the tar, so we only
        # ever tar & transfer one file per pair (the fallback is typically
        # same-order-of-magnitude size as the primary — no point paying
        # double when we only want one). Per (folder, primary, fallback):
        #   primary present    -> remove fallback (dedup + free remote disk)
        #   fallback only      -> rename fallback to primary
        #   neither present    -> no-op
        # Idempotent and safe if the folders don't exist yet.
        if self.output_file_fallbacks:
            fallback_lines = []
            for folder, patterns in self.output_folders_selective.items():
                pattern_set = set(patterns)
                for primary, fallback in self.output_file_fallbacks.items():
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

        # Build tar command with selective folder content
        tar_items = []

        # Add all output files (mandatory + best-effort)
        tar_items.extend(self.output_files)
        tar_items.extend(optional_files)

        # Add folders - either full folders or selective content
        for folder in self.output_folders:
            if folder in self.output_folders_selective:
                # Add specific files from this folder
                for pattern in self.output_folders_selective[folder]:
                    tar_items.append(f"{folder}/{pattern}")
            else:
                # Add entire folder
                tar_items.append(folder)

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
                    + " ".join(tar_items),
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
            except (FileNotFoundError, IOError, tarfile.ReadError, EOFError) as _retrieve_exc:
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

        # Check if all files were received
        time_wait = 60
        if check_if_files_received:
            received = self.check_all_received(check_files_in_folder=check_files_in_folder)
            if received:
                print("\t\t- All correct", typeMsg="i")
            elif best_effort:
                # Status poll: don't block on a 60s in-retrieve retry; the caller re-polls.
                print("\t* Not all expected files received (best-effort); caller will re-poll", typeMsg="i")
            else:
                print(f"\t* Not all received, trying retrieval once again after waiting {time_wait} seconds", typeMsg="i")
                time.sleep(time_wait)
                _ = self.retrieve(check_if_files_received=False, optional_files=optional_files)
                received = self.check_all_received(check_files_in_folder=check_files_in_folder)
        else:
            received = True

        return received

    def remove_scratch_folder(self):
        # Safety guard: never rm -rf the user's working directory in in-place mode
        if getattr(self, "run_in_place", False):
            print("\t* Skipping scratch-folder removal (in-place local execution)")
            return None, None

        print(f'\t* Removing{" remote" if self.ssh is not None else ""} folder')

        output, error = self.execute(f"rm -rf {self.folderExecution}")

        return output, error

    def close(self, *args, **kwargs):
        if self.machineSettings["machine"] != "local":
            return self.close_ssh(*args, **kwargs)

    def close_ssh(self):
        print("\t* Closing connection")

        self.sftp.close()
        self.ssh.close()

        if self.jump_client is not None:
            self.jump_client.close()

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

        if "output_files" in self.__dict__:
            output_files_backup = copy.deepcopy(self.output_files)
            output_folders_backup = copy.deepcopy(self.output_folders)
            wasThere = True
        else:
            wasThere = False

        # Only squeue_output.dat is mandatory — it is what interpret_status()
        # parses. The slurm job log (`file_output`) is best-effort: it does not
        # exist on the remote while the job is still PENDING, and its absence
        # simply means `interpret_status` sets `self.log_file = None`. Marking
        # it optional avoids a spurious "File not received" warning plus a 60s
        # retry on every status poll while the job is queued.
        self.output_files = ["squeue_output.dat"]
        self.output_folders = []

        # A status poll must never crash the run on a remote hiccup. The squeue
        # exec_command is retried across transient SSH/VPN flaps (retry_on_transient),
        # and the retrieval is best-effort; if a residual transient error still
        # surfaces (e.g. finite ssh_retry_attempts exhausted), degrade to "not received"
        # -> interpret_status treats it as pending/keep-polling. connect() keeps its own
        # retry, so a clean connect here means close() below is safe.
        output = error = None
        self.connect()
        try:
            output, error = self.execute(command, printYN=True, retry_on_transient=True)
            received = self.retrieve(optional_files=[file_output], best_effort=True)
        except (paramiko.ssh_exception.SSHException, TimeoutError, socket.timeout,
                EOFError, ConnectionError, socket.gaierror) as _poll_exc:
            print(f"\t* Status poll could not reach the remote ({type(_poll_exc).__name__}: {_poll_exc}); "
                  f"assuming job still pending (will re-poll)", typeMsg="w")
            received = False
        if not received and output is not None:
            self._write_debugging_files(output, error, extra_name = '_check')
        self.close()
        self.interpret_status(file_output = file_output)

        # Back to original
        if wasThere:
            self.output_folders = output_folders_backup
            self.output_files = output_files_backup

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
            self.infoSLURM = {"STATE": "UNKNOWN", "NAME": self._squeue_job_name(), "JOBID": None}
            self.jobid_found = None
            self.status = 0
            self.log_file = None
            return

        # -----------------------------------------------
        # Read output of squeue command -> self.infoSLURM
        # -----------------------------------------------

        with open(self.folder_local / "squeue_output.dat", "r") as f:
            output_squeue = f.read()
        output_squeue = str(output_squeue)[3:].split("\n")

        if (len(output_squeue[0].split()) == 0) or (len(output_squeue[1]) == 0):
            self.infoSLURM = {"STATE": "NOT FOUND"}
            self.jobid_found = None
        else:
            self.infoSLURM = {}
            for i in range(len(output_squeue[0].split())):
                if i < len(output_squeue[1].split()):
                    self.infoSLURM[output_squeue[0].split()[i]] = output_squeue[1].split()[i]
                else:
                    self.infoSLURM[output_squeue[0].split()[i]] = None

            self.jobid_found = self.infoSLURM["JOBID"]

        # -----------------------------------------------
        # Interpret status
        # -----------------------------------------------

        if self.infoSLURM["STATE"] == "PENDING":
            self.status = 0
        elif (self.infoSLURM["STATE"] == "RUNNING") or (self.infoSLURM["STATE"] == "COMPLETING"):
            self.status = 1
        elif self.infoSLURM["STATE"] == "NOT FOUND":
            self.status = 2
        else:
            print("Unknown SLURM status, please check")
            embed()

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

    def check_all_received(self, check_files_in_folder={}):
        print("\t* Checking if all files & folders that are expected were received")
        received = True

        # Check if all files were received
        for file in self.output_files:
            if not (self.folder_local / file).exists():
                print(f"\t\t- File '{file}' not received", typeMsg="w")
                received = False

        for folder in self.output_folders:
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


def create_slurm_execution_files(
    command,
    folderExecution,
    modules_remote=None,
    folder_local=None,
    shellPreCommands=None,
    shellPostCommands=None,
    label_log_files="",
    wait_until_sbatch=True,
    slurm_allocation={},
    launchSlurm=True,
    slurm_settings = None,
    if_array_relabel = False
):
    
    fileSBATCH = folder_local / f"mitim_bash{label_log_files}.src"
    fileSHELL = folder_local / f"mitim_shell_executor{label_log_files}.sh"
    fileSBATCH_remote = f"{folderExecution}/mitim_bash{label_log_files}.src"
    
    # ---------------------------------------------------
    # slurm_settings indicate the job resource allocation   
    #  ---------------------------------------------------

    if slurm_settings is None:
        slurm_settings = {}

    # ---- Native sbatch keys (the only schema we support) -----------------
    # Back-compat: migrate the legacy 'minutes' key to the native 'time' key. mitim_job-direct
    # callers (e.g. TRANSPsingularity) still pass 'minutes' instead of going through
    # SLURMtools.resolve(); without this they'd silently fall back to the "10:00" default below.
    if "time" not in slurm_settings and "minutes" in slurm_settings:
        _m = int(slurm_settings["minutes"])
        slurm_settings["time"] = f"{_m//60:02d}:{_m%60:02d}:00" if _m >= 60 else f"{_m:02d}:00"

    nameJob         = slurm_settings.setdefault("job-name", "mitim_job")
    time_com        = slurm_settings.setdefault("time", "10:00")
    memory_req_by_job = slurm_settings.setdefault("mem", None)

    nodes           = slurm_settings.setdefault("nodes", None)
    ntasks          = slurm_settings.setdefault("ntasks", None)
    cpuspertask     = slurm_settings.setdefault("cpus-per-task", None)
    ntaskspernode   = slurm_settings.setdefault("ntasks-per-node", None)
    gpuspertask     = slurm_settings.setdefault("gpus-per-task", None)
    gpuspernode     = slurm_settings.setdefault("gpus-per-node", None)

    job_array       = slurm_settings.setdefault("array", None)
    job_array_limit = slurm_settings.setdefault("array_limit", None)
    job_exclusive   = slurm_settings.setdefault("exclusive", False)

    # Requeue-ability: True (default) emits --requeue, False emits --no-requeue,
    # None leaves the cluster default. Explicit --requeue makes behavior uniform
    # across clusters (slurm.conf JobRequeue varies): on preemption or node
    # failure the job goes back in the queue under the same id instead of dying,
    # and MITIM workflows resume from their on-disk checkpoints when re-executed.
    job_requeue     = slurm_settings.setdefault("requeue", True)

    # ---------------------------------------------------
    # slurm_allocation indicate the machine specifications as given by the config instead of individual job
    # ---------------------------------------------------
    
    partition = slurm_allocation.setdefault("partition", None)
    qos = slurm_allocation.setdefault("qos", None)
    email = slurm_allocation.setdefault("email", None)
    exclude = slurm_allocation.setdefault("exclude", None)
    account = slurm_allocation.setdefault("account", None)
    constraint = slurm_allocation.setdefault("constraint", None)
    memory_req_by_config = slurm_allocation.setdefault("mem", None)
    request_exclusive_node = slurm_allocation.setdefault("exclusive", False)
    
    
    if memory_req_by_job == 0 :
        print("\t\t- Entire node memory requested by job, overwriting memory requested by config file", typeMsg="i")
        memory_req = memory_req_by_job
    elif memory_req_by_job is not None:
        print(f"\t\t- Memory requested by job ({memory_req_by_job}), overwriting memory requested by config file", typeMsg="i")
        memory_req = memory_req_by_job
    else:
        if memory_req_by_config is not None:
            print(f"\t\t- Memory requested by config file ({memory_req_by_config})", typeMsg="i")
        memory_req =  memory_req_by_config
    
    # `time_com` is already a formatted sbatch --time string (set above
    # from the native 'time' key or migrated from legacy 'minutes').

    """
	********************************************************************************************
	Write mitim_bash.src file to execute
	********************************************************************************************
	"""

    command = [command] if isinstance(command, str) else command
    shellPreCommands = [] if shellPreCommands is None else shellPreCommands
    shellPostCommands = [] if shellPostCommands is None else shellPostCommands

    # ~~~~ Construct SLURM header ~~~~~~~~~~~~~~~
    commandSBATCH = []

    commandSBATCH.append("#!/usr/bin/env bash")
    commandSBATCH.append(f"#SBATCH --job-name {nameJob}")
    if (not if_array_relabel) or (job_array is None):
        commandSBATCH.append(f"#SBATCH --output {folderExecution}/slurm_output{label_log_files}.dat")
        commandSBATCH.append(f"#SBATCH --error {folderExecution}/slurm_error{label_log_files}.dat")
    else:
        commandSBATCH.append(f"#SBATCH --output {folderExecution}/slurm_output{label_log_files}_%A_%a.dat")
        commandSBATCH.append(f"#SBATCH --error {folderExecution}/slurm_error{label_log_files}_%A_%a.dat")
    commandSBATCH.append(f"#SBATCH --time {time_com}")
    if email is not None:
        commandSBATCH.append("#SBATCH --mail-user=" + email)
    if partition is not None:
        commandSBATCH.append(f"#SBATCH --partition {partition}")
    if account is not None:
        commandSBATCH.append(f"#SBATCH --account {account}")
    if qos is not None:
        commandSBATCH.append(f"#SBATCH --qos {qos}")
    if constraint is not None:
        commandSBATCH.append(f"#SBATCH --constraint {constraint}")
    if memory_req is not None:
        commandSBATCH.append(f"#SBATCH --mem {memory_req}")
    if job_array is not None:
        commandSBATCH.append(f"#SBATCH --array={job_array}{f'%{job_array_limit} ' if job_array_limit is not None else ''}")
    # --exclusive can co-exist with arrays (one whole node per array element)
    # and with packed jobs (whole nodes via per-job slurm_settings). Honor
    # both the machine config (`slurm_allocation`) and the per-job override
    # (`slurm_settings.exclusive`).
    if request_exclusive_node or job_exclusive:
        commandSBATCH.append("#SBATCH --exclusive")
    if job_requeue is True:
        commandSBATCH.append("#SBATCH --requeue")
    elif job_requeue is False:
        commandSBATCH.append("#SBATCH --no-requeue")
    if nodes is not None:
        commandSBATCH.append(f"#SBATCH --nodes {nodes}")
    if ntasks is not None:
        commandSBATCH.append(f"#SBATCH --ntasks {ntasks}")
    if ntaskspernode is not None:
        commandSBATCH.append(f"#SBATCH --ntasks-per-node {ntaskspernode}")
    if cpuspertask is not None:
        commandSBATCH.append(f"#SBATCH --cpus-per-task {cpuspertask}")
    if gpuspertask is not None:
        commandSBATCH.append(f"#SBATCH --gpus-per-task {gpuspertask}")
    if gpuspernode is not None:
        commandSBATCH.append(f"#SBATCH --gpus-per-node={gpuspernode}")
    if exclude is not None:
        commandSBATCH.append(f"#SBATCH --exclude={exclude}")

    commandSBATCH.append("#SBATCH --profile=all")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    # ~~~~ Commands ~~~~~~~~~~~~~~~
    commandSBATCH.append("")
    commandSBATCH.append("export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK")
    if gpuspernode is not None:
        commandSBATCH.append('export SLURM_CPU_BIND="cores"')
    commandSBATCH.append('echo "MITIM: Submitting SLURM job $SLURM_JOBID in $HOSTNAME (host: $SLURM_SUBMIT_HOST)"')
    commandSBATCH.append('echo "MITIM: Nodes have $SLURM_CPUS_ON_NODE cores and $SLURM_JOB_NUM_NODES node(s) were allocated for this job"')
    commandSBATCH.append('echo "MITIM: Each of the $SLURM_NTASKS tasks allocated will run with $SLURM_CPUS_PER_TASK cores, allocating $SRUN_CPUS_PER_TASK CPUs per srun"')
    commandSBATCH.append('echo "***********************************************************************************************"')
    commandSBATCH.append('echo ""')
    commandSBATCH.append("")

    # If modules, add them, but also make sure I expand the potential aliases that they may have!
    full_command = ["shopt -s expand_aliases",modules_remote] if (modules_remote is not None) else []
    
    full_command.extend(command)
    for c in full_command:
        commandSBATCH.append(c)

    commandSBATCH.append("")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

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
            jobid = int(line.split()[3])
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
