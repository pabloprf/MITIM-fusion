"""
Set of tools to farm out simulations to run in either remote clusters or locally, serially or parallel
"""

from tqdm import tqdm
import os
import time
import sys
import subprocess
import socket
import signal
import datetime
import torch
import copy
import paramiko
import tarfile
import numpy as np
from contextlib import contextmanager
from mitim_tools.misc_tools import IOtools, CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from IPython import embed

verbose_level = read_verbose_level()

if verbose_level in [4, 5]:
    quiet_tag = ""
else:
    quiet_tag = "-q "

UseCUDAifAvailable = True

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
    def __init__(self, folder_local):
        self.folder_local = folder_local
        self.jobid = None

    def define_machine(
        self,
        code,
        nameScratch,
        launchSlurm=True,
        slurm_settings={},
    ):
        # Separated in case I need to quickly grab the machine settings
        self.define_machine_quick(code, nameScratch, slurm_settings=slurm_settings)

        self.launchSlurm = launchSlurm

        if self.launchSlurm and (len(self.machineSettings["slurm"]) == 0):
            self.launchSlurm = False
            print(
                "\t- slurm requested but no slurm setup to this machine in config... not doing slurm",
                typeMsg="w",
            )

        # Print Slurm info
        if self.launchSlurm:
            print("\t- Slurm Settings:")
            print("\t\t- Job settings:")
            for key in self.slurm_settings:
                if self.slurm_settings[key] is not None:
                    print(f"\t\t\t- {key}: {self.slurm_settings[key]}")
            print("\t\t- Partition settings:")
            print(f'\t\t\t- machine: {self.machineSettings["machine"]}')
            print(f'\t\t\t- username: {self.machineSettings["user"]}')
            for key in self.machineSettings["slurm"]:
                print(f'\t\t\t- {key}: {self.machineSettings["slurm"][key]}')

    def define_machine_quick(self, code, nameScratch, slurm_settings={}):
        self.slurm_settings = slurm_settings

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Defaults for slurm
        self.slurm_settings.setdefault("name", "mitim_job")
        self.slurm_settings.setdefault("minutes", 10)
        self.slurm_settings.setdefault("cpuspertask", 1)
        self.slurm_settings.setdefault("ntasks", 1)
        self.slurm_settings.setdefault("nodes", None)
        self.slurm_settings.setdefault("job_array", None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.machineSettings = CONFIGread.machineSettings(
            code=code,
            nameScratch=nameScratch,
        )
        self.folderExecution = self.machineSettings["folderWork"]

    def prep(
        self,
        command,
        input_files=[],
        input_folders=[],
        output_files=[],
        output_folders=[],
        shellPreCommands=[],
        shellPostCommands=[],
        label_log_files="",
    ):
        """
        command:
            Option 1: string with commands to execute separated by &&
            Option 2: list of strings with commands to execute
        """

        # Pass to class
        self.command = command
        self.input_files = input_files
        self.input_folders = input_folders
        self.output_files = output_files
        self.output_folders = output_folders

        self.shellPreCommands = shellPreCommands
        self.shellPostCommands = shellPostCommands
        self.label_log_files = label_log_files

    def run(self, waitYN=True, timeoutSecs=1e6, removeScratchFolders=True):
        if not waitYN:
            removeScratchFolders = False

        # ****** Prepare SLURM job *****************************
        comm, fileSBTACH, fileSHELL = create_slurm_execution_files(
            self.command,
            self.machineSettings["folderWork"],
            self.machineSettings["modules"],
            job_array=self.slurm_settings["job_array"],
            folder_local=self.folder_local,
            shellPreCommands=self.shellPreCommands,
            shellPostCommands=self.shellPostCommands,
            nameJob=self.slurm_settings["name"],
            minutes=self.slurm_settings["minutes"],
            nodes=self.slurm_settings["nodes"],
            ntasks=self.slurm_settings["ntasks"],
            cpuspertask=self.slurm_settings["cpuspertask"],
            slurm=self.machineSettings["slurm"],
            launchSlurm=self.launchSlurm,
            label_log_files=self.label_log_files,
            wait_until_sbatch=waitYN,
        )
        # ******************************************************

        if fileSBTACH not in self.input_files:
            self.input_files.append(fileSBTACH)
        if fileSHELL not in self.input_files:
            self.input_files.append(fileSHELL)

        self.output_files = curateOutFiles(self.output_files)

        # Relative paths TO FIX
        self.input_files = [
            os.path.relpath(path, self.folder_local) for path in self.input_files
        ]
        self.input_folders = [
            os.path.relpath(path, self.folder_local) for path in self.input_folders
        ]

        # Process
        self.full_process(
            comm, removeScratchFolders=removeScratchFolders, timeoutSecs=timeoutSecs
        )

        # Get jobid
        if self.launchSlurm:
            with open(self.folder_local + "/mitim.out", "r") as f:
                aux = f.readlines()
            self.jobid = aux[0].split()[-1]
        else:
            self.jobid = None

    # --------------------------------------------------------------------
    # SSH executions
    # --------------------------------------------------------------------

    def full_process(self, comm, timeoutSecs=1e6, removeScratchFolders=True):
        """
        My philosophy is to always wait for the execution of all commands. If I need
        to not wait, that's handled by a slurm submission without --wait, but I still
        want to execute that.
        """
        wait_for_all_commands = True

        if timeoutSecs < 1e6:
            timeOut_txt = f", will timeout execution in {timeoutSecs}s"
        else:
            timeOut_txt = ""

        time_init = datetime.datetime.now()

        print(
            f"\n\t-------------- Running process ({time_init.strftime('%Y-%m-%d %H:%M:%S')}{timeOut_txt}) --------------"
        )

        self.connect(log_file=f"{self.folder_local}/paramiko.log")
        if removeScratchFolders:
            self.remove_scratch_folder()
        self.create_scratch_folder()
        self.send()

        output, error = self.execute(
            comm,
            make_relative=True,
            wait_for_all_commands=wait_for_all_commands,
            printYN=True,
            timeoutSecs=timeoutSecs if timeoutSecs < 1e6 else None,
        )

        for _ in range(2):
            self.retrieve()
            received = self.check_all_received()
            if received:
                print("\t\t- All correct")
                break
            print("\t\t- Not all received, trying again", typeMsg="w")
            time.sleep(10)

        if received:
            if wait_for_all_commands and removeScratchFolders:
                self.remove_scratch_folder()
        else:
            cont = print(
                "\t* Not all expected files received, not removing scratch folder",
                typeMsg="q",
            )
            if not cont:
                print(
                    "[mitim] Stopped with embed(), you can look at output and error",
                    typeMsg="w",
                )
                embed()

        self.close()

        print(
            f"\t-------------- Finished process (took {IOtools.getTimeDifference(time_init)}) --------------\n"
        )

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
        print(
            f'\t\t{self.target_user}@{self.target_host}{f", via tunnel {self.jump_user}@" +self.jump_host  if self.jump_host is not None else ""}{":" + str(self.machineSettings["port"]) if self.machineSettings["port"] is not None else ""}{" with identity: " + self.machineSettings["identity"] if self.machineSettings["identity"] is not None else ""}'
        )

        if log_file is not None:
            paramiko.util.log_to_file(log_file)

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
        self.ssh.connect(
            self.target_host,
            username=self.target_user,
            disabled_algorithms=disabled_algorithms,
            key_filename=self.key_filename,
            port=self.port,
            sock=self.sock,
            allow_agent=True,
        )

        try:
            self.sftp = self.ssh.open_sftp()
        except paramiko.sftp.SFTPError:
            raise Exception(
                "[mitim] SFTPError: Your bashrc on the server likely contains print statements"
            )

    def define_jump(self):
        if self.jump_host is not None:
            # Create an SSH client instance for the jump host
            self.jump_client = paramiko.SSHClient()
            self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            key_jump = IOtools.expandPath(self.machineSettings["identity"])

            if key_jump is not None:
                if not os.path.exists(key_jump):
                    if print('Key file "' + key_jump + '" does not exist, continue without key',typeMsg='q'):
                        key_jump = None
                        
            # Connect to the jump host
            self.jump_client.connect(
                self.jump_host,
                username=self.jump_user,
                port=self.machineSettings["port"]
                if self.machineSettings["port"] is not None
                else 22,
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

            self.key_filename = IOtools.expandPath(self.machineSettings["identity"])

            if self.key_filename is not None:
                if not os.path.exists(self.key_filename):
                    if print('Key file "' + self.key_filename + '" does not exist, continue without key',typeMsg='q'):
                        self.key_filename = None
                        

    def create_scratch_folder(self):
        print(f'\t* Creating{" remote" if self.ssh is not None else ""} folder:')
        print(f'\t\t{self.machineSettings["folderWork"]}')

        command = "mkdir -p " + self.machineSettings["folderWork"]

        output, error = self.execute(command)

        return output, error

    def send(self):
        print(
            f'\t* Sending files{" to remove server" if self.ssh is not None else ""}:'
        )

        # Create a tarball of the local directory
        print("\t\t- Tarballing")
        with tarfile.open(
            os.path.join(self.folder_local, "mitim_send.tar.gz"), "w:gz"
        ) as tar:
            for file in self.input_files + self.input_folders:
                tar.add(os.path.join(self.folder_local, file), arcname=file)

        # Send it
        print("\t\t- Sending")
        if self.ssh is not None:
            with TqdmUpTo(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc="mitim_send.tar.gz",
                bar_format=" " * 20
                + "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
            ) as t:
                self.sftp.put(
                    os.path.join(self.folder_local, "mitim_send.tar.gz"),
                    os.path.join(
                        self.machineSettings["folderWork"], "mitim_send.tar.gz"
                    ),
                    callback=lambda sent, total_size: t.update_to(sent, total_size),
                )
        else:
            os.system(
                "cp "
                + os.path.join(self.folder_local, "mitim_send.tar.gz")
                + " "
                + os.path.join(self.machineSettings["folderWork"], "mitim_send.tar.gz")
            )

        # Extract it
        print("\t\t- Extracting tarball")
        self.execute(
            "tar -xzf "
            + os.path.join(self.machineSettings["folderWork"], "mitim_send.tar.gz")
            + " -C "
            + self.machineSettings["folderWork"]
        )

        # Remove tarballs
        print("\t\t- Removing tarballs")
        os.remove(os.path.join(self.folder_local, "mitim_send.tar.gz"))
        self.execute(
            "rm "
            + os.path.join(self.machineSettings["folderWork"], "mitim_send.tar.gz")
        )

    def execute(self, command_str, make_relative=False, **kwargs):
        # Always start by going to the folder
        if make_relative:
            command_str_mod = f"cd {self.folderExecution} && {command_str}"
        else:
            command_str_mod = command_str

        if self.ssh is not None:
            return self.execute_remote(command_str_mod, **kwargs)
        else:
            return self.execute_local(command_str_mod, **kwargs)

    def execute_remote(
        self,
        command_str,
        printYN=False,
        timeoutSecs=None,
        wait_for_all_commands=True,
        **kwargs,
    ):
        if printYN:
            print("\t* Executing (remote):", typeMsg="i")
            print(f"\t\t{command_str}")

        output = None
        error = None

        try:
            stdin, stdout, stderr = self.ssh.exec_command(
                command_str, timeout=timeoutSecs
            )
            # Wait for the command to complete and read the output
            if wait_for_all_commands:
                stdin.close()
                output = stdout.read()
                error = stderr.read()
        except socket.timeout:
            print("Command timed out")

        return output, error

    def execute_local(self, command_str, printYN=False, timeoutSecs=None, **kwargs):
        if printYN:
            print("\t* Executing (local):")
            print(f"\t\t{command_str}")

        output, error = run_subprocess(
            [command_str], timeoutSecs=timeoutSecs, localRun=True
        )

        return output, error

    def retrieve(self):
        print(
            f'\t* Retrieving files{" from remote server" if self.ssh is not None else ""}:'
        )

        # Create a tarball of the output files and folders on the remote machine
        print("\t\t- Tarballing")
        self.execute(
            "tar -czf "
            + os.path.join(self.machineSettings["folderWork"], "mitim_receive.tar.gz")
            + " -C "
            + self.machineSettings["folderWork"]
            + " "
            + " ".join(self.output_files + self.output_folders)
        )

        # Download the tarball
        print("\t\t- Downloading")
        if self.ssh is not None:
            with TqdmUpTo(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc="mitim_receive.tar.gz",
                bar_format=" " * 20
                + "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
            ) as t:
                self.sftp.get(
                    os.path.join(
                        self.machineSettings["folderWork"], "mitim_receive.tar.gz"
                    ),
                    os.path.join(self.folder_local, "mitim_receive.tar.gz"),
                    callback=lambda sent, total_size: t.update_to(sent, total_size),
                )
        else:
            os.system(
                "cp "
                + os.path.join(
                    self.machineSettings["folderWork"], "mitim_receive.tar.gz"
                )
                + " "
                + os.path.join(self.folder_local, "mitim_receive.tar.gz")
            )

        # Extract the tarball locally
        print("\t\t- Extracting tarball")
        with tarfile.open(
            os.path.join(self.folder_local, "mitim_receive.tar.gz"), "r:gz"
        ) as tar:
            tar.extractall(path=self.folder_local)

        # Remove tarballs
        print("\t\t- Removing tarballs")
        os.remove(os.path.join(self.folder_local, "mitim_receive.tar.gz"))
        self.execute(
            "rm "
            + os.path.join(self.machineSettings["folderWork"], "mitim_receive.tar.gz")
        )

    def remove_scratch_folder(self):
        print(f'\t* Removing{" remote" if self.ssh is not None else ""} folder')

        output, error = self.execute("rm -rf " + self.machineSettings["folderWork"])

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

    def check(self):
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
            txt_look = f"-n {self.slurm_settings['name']}"

        command = (
            f'squeue {txt_look} -o "%.15i %.24P %.18j %.10u %.10T %.10M %.10l %.5D %R" '
        )

        self.connect()
        self.output_files.append("slurm_output.dat")
        output, error = self.execute(command, printYN=True)
        self.retrieve()  # Retrieve self.output_files too
        self.close()

        output = str(output)[3:].split("\\n")
        if len(output[1].split()) == 1:
            self.infoSLURM = None
        else:
            self.infoSLURM = {}
            for i in range(len(output[0].split())):
                self.infoSLURM[output[0].split()[i]] = output[1].split()[i]

        with open(self.folder_local + "/slurm_output.dat", "r") as f:
            self.log_file = f.readlines()

        # If jobid was given as None, I retrieved the info from the job_name, but now provide here the actual id
        if self.infoSLURM is not None:
            self.jobid_found = self.infoSLURM["JOBID"]
        else:
            self.jobid_found = None

        # Print info
        txt = "\t- Job was checked"
        if (self.jobid is None) and (self.jobid_found is not None):
            txt += f' (jobid {self.jobid_found}, found from name "{self.slurm_settings["name"]}")'
        elif self.jobid is not None:
            txt += f" (jobid {self.jobid})"
        if self.infoSLURM is not None:
            txt += f', is {self.infoSLURM["STATE"]} (job.infoSLURM)'
        else:
            txt += ", is not on the grid (job.infoSLURM is None)"
        if self.log_file is not None:
            txt += f". Log file (job.log_file) was retrieved, and has {len(self.log_file)} lines"
        print(txt)

    def check_all_received(self):
        print("\t* Checking if all expected files and folders were received")
        received = True
        for file in self.output_files:
            if not os.path.exists(os.path.join(self.folder_local, file)):
                print(f"\t\t- File {file} not received", typeMsg="w")
                received = False
        for folder in self.output_folders:
            if not os.path.exists(os.path.join(self.folder_local, folder)):
                print(f"\t\t- Folder {folder} not received", typeMsg="w")
                received = False

        return received


class TqdmUpTo(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = False

    def update_to(self, sent, total_size):
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


class PRF_ParallelClass_reduced(object):
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

    if UseCUDAifAvailable and torch.cuda.is_available():
        multiprocessing.set_start_method("spawn")

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
    res = pool.map(PRF_ParallelClass_reduced(Function, Params), np.arange(howmany))
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
    folder_remote,
    modules_remote,
    slurm={},
    folder_local=None,
    shellPreCommands=[],
    shellPostCommands=[],
    launchSlurm=True,
    nameJob="test",
    minutes=5,
    ntasks=1,
    cpuspertask=4,
    job_array=None,
    nodes=None,
    label_log_files="",
    wait_until_sbatch=True,
):
    if folder_local is None:
        folder_local = folder_remote
    if isinstance(command, str):
        command = [command]

    folderExecution = IOtools.expandPath(folder_remote)
    fileSBTACH = f"{folder_local}/mitim_bash.src"
    fileSHELL = f"{folder_local}/mitim_shell_executor.sh"
    fileSBTACH_remote = f"{folder_remote}/mitim_bash.src"

    minutes = int(minutes)

    partition = slurm.setdefault("partition", None)
    email = slurm.setdefault("email", None)
    exclude = slurm.setdefault("exclude", None)
    account = slurm.setdefault("account", None)
    constraint = slurm.setdefault("constraint", None)
    memory_req = slurm.setdefault("mem", None)

    """
	********************************************************************************************
	Write mitim_bash.src file to execute
	********************************************************************************************
		- Contains sourcing of mitim.bashrc, so that it's done at node level
	"""

    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes - hours * 60
        time_com = f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:00"
    else:
        time_com = f"{str(minutes).zfill(2)}:00"

    commandSBATCH = []

    # ******* Basics
    commandSBATCH.append("#!/bin/bash -l")
    commandSBATCH.append(f"#SBATCH --job-name {nameJob}")
    commandSBATCH.append(
        f"#SBATCH --output {folderExecution}/slurm_output{label_log_files}.dat"
    )
    commandSBATCH.append(
        f"#SBATCH --error {folderExecution}/slurm_error{label_log_files}.dat"
    )
    if email is not None:
        commandSBATCH.append("#SBATCH --mail-user=" + email)

    # ******* Partition / Billing
    commandSBATCH.append(f"#SBATCH --partition {partition}")

    if account is not None:
        commandSBATCH.append(f"#SBATCH --account {account}")
    if constraint is not None:
        commandSBATCH.append(f"#SBATCH --constraint {constraint}")

    if memory_req is not None:
        commandSBATCH.append(f"#SBATCH --mem {memory_req}")

    commandSBATCH.append(f"#SBATCH --time {time_com}")

    if job_array is None:
        commandSBATCH.append("#SBATCH --exclusive")
    else:
        commandSBATCH.append(f"#SBATCH --array={job_array}")

    # ******* CPU setup
    if nodes is not None:
        commandSBATCH.append(f"#SBATCH --nodes {nodes}")
    commandSBATCH.append(f"#SBATCH --ntasks {ntasks}")
    commandSBATCH.append(f"#SBATCH --cpus-per-task {cpuspertask}")

    if exclude is not None:
        commandSBATCH.append(f"#SBATCH --exclude={exclude}")

    commandSBATCH.append("export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK")

    # ******* Commands
    commandSBATCH.append("")
    commandSBATCH.append(
        'echo "MITIM: Submitting SLURM job $SLURM_JOBID in $HOSTNAME (host: $SLURM_SUBMIT_HOST)"'
    )
    commandSBATCH.append(
        'echo "MITIM: Nodes have $SLURM_CPUS_ON_NODE cores and $SLURM_JOB_NUM_NODES nodes were allocated for this job"'
    )
    commandSBATCH.append(
        'echo "MITIM: Each of the $SLURM_NTASKS tasks allocated will run with $SLURM_CPUS_PER_TASK cores, allocating $SRUN_CPUS_PER_TASK CPUs per srun"'
    )
    commandSBATCH.append("")

    full_command = [modules_remote] if (modules_remote is not None) else []
    full_command.extend(command)
    for c in full_command:
        commandSBATCH.append(c)

    commandSBATCH.append("")

    wait_txt = " --wait" if wait_until_sbatch else ""
    if launchSlurm:
        comm, launch = commandSBATCH, "sbatch" + wait_txt
    else:
        comm, launch = full_command, "bash"

    if os.path.exists(fileSBTACH):
        os.system(f"rm {fileSBTACH}")
    with open(fileSBTACH, "w") as f:
        f.write("\n".join(comm))

    """
	********************************************************************************************
	Write mitim_shell_executor.sh file that handles the execution of the mitim_bash.src with pre and post commands
	********************************************************************************************
		- Contains sourcing of mitim.bashrc, so that it's done at machine level
	"""

    commandSHELL = copy.deepcopy(shellPreCommands)
    commandSHELL.append("")
    if modules_remote is not None:
        commandSHELL.append(modules_remote)
    commandSHELL.append(f"{launch} {fileSBTACH_remote}")
    commandSHELL.append("")
    for i in range(len(shellPostCommands)):
        commandSHELL.append(shellPostCommands[i])
    # Evaluate Job performance
    # commandSHELL.append(
    #     "python3 $MITIM_PATH/src/mitim_tools/misc_tools/FARMINGtools.py mitim.out"
    # )

    if os.path.exists(fileSHELL):
        os.system(f"rm {fileSHELL}")
    with open(fileSHELL, "w") as f:
        f.write("\n".join(commandSHELL))

    """
	********************************************************************************************
	Command to send through scp
	********************************************************************************************
	"""

    comm = "bash mitim_shell_executor.sh > mitim.out"

    return comm, fileSBTACH, fileSHELL


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
        print("\n****** SEFF:")
        os.system(f"seff {jobid}")
        print("\n****** SACCT:")
        os.system(f"sacct -j {jobid}")


if __name__ == "__main__":
    printEfficiencySLURM(sys.argv[1])
