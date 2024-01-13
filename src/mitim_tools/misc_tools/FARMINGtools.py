"""

    Set of tools to farm out simulations to run in either remote clusters or locally, serially or parallel

"""

import os
import sys
import subprocess
import socket
import signal
import pickle
import datetime
import torch
import copy
import numpy as np
from IPython import embed
from contextlib import contextmanager
from mitim_tools.misc_tools import IOtools,CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

if verbose_level in [4, 5]:
    quiet_tag = ""
else:
    quiet_tag = "-q "

UseCUDAifAvailable = True

class mitim_job:
    def __init__(self,folder_local):

        self.folder_local = folder_local

    def define_machine(
            self,
            code,
            nameScratch,
            launchSlurm=True,
            slurm_settings={},
            ):

        self.launchSlurm = launchSlurm
        self.slurm_settings = slurm_settings

        self.machineSettings = CONFIGread.machineSettings(
            code=code,
            nameScratch=nameScratch,
            )

        if self.launchSlurm and (len(self.machineSettings['slurm']) == 0):
            self.launchSlurm = False
            print(
                "\t- slurm requested but no slurm setup to this machine in config... not doing slurm",
                typeMsg="w",
            )

        # Defaults for slurm (give them even in LaunchSlurm=False)
        self.slurm_settings.setdefault('name','mitim_job')
        self.slurm_settings.setdefault('job_array',None)
        self.slurm_settings.setdefault('nodes',None)
        self.slurm_settings.setdefault('cpuspertask',1)
        self.slurm_settings.setdefault('email',"noemail")
        self.slurm_settings.setdefault('minutes',10)
        self.slurm_settings.setdefault('ntasks',1)
        self.slurm_settings.setdefault('nodes',1)

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
            extranamelogs="",
            default_exclusive=False
            ):

        # Pass to class
        self.command = command
        self.input_files = input_files
        self.input_folders = input_folders
        self.output_files = output_files
        self.output_folders = output_folders
        
        self.shellPreCommands = shellPreCommands
        self.shellPostCommands = shellPostCommands
        self.extranamelogs = extranamelogs
        self.default_exclusive = default_exclusive

    def run(self,waitYN=True):

        if not waitYN:
            # If I'm not waiting, make sure i don't clear the folder
            self.machineSettings["clear"] = False

        # ****** Prepare SLURM job *****************************
        comm, fileSBTACH, fileSHELL = SLURM(
            self.command,
            self.machineSettings["folderWork"],
            self.machineSettings['modules'],
            job_array=self.slurm_settings['job_array'],
            folder_local=self.folder_local,
            shellPreCommands=self.shellPreCommands,
            shellPostCommands=self.shellPostCommands,
            nameJob=self.slurm_settings['name'],
            minutes=self.slurm_settings['minutes'],
            nodes=self.slurm_settings['nodes'],
            ntasks=self.slurm_settings['ntasks'],
            cpuspertask=self.slurm_settings['cpuspertask'],
            slurm=self.machineSettings["slurm"],
            launchSlurm=self.launchSlurm,
            email=self.slurm_settings['email'],
            extranamelogs=self.extranamelogs,
            default_exclusive=self.default_exclusive,
            waitYN=waitYN,
        )
        # ******************************************************

        if fileSBTACH not in self.input_files:
            self.input_files.append(fileSBTACH)
        if fileSHELL not in self.input_files:
            self.input_files.append(fileSHELL)

        self.output_files = curateOutFiles(self.output_files)

        runCommand(
            comm,
            self.input_files,
            inputFolders=self.input_folders,
            outputFiles=self.output_files,
            outputFolders=self.output_folders,
            whereOutput=self.folder_local,
            machineSettings=self.machineSettings
            )

        with open(self.folder_local + "/sbatch.out", "r") as f:
            aux = f.readlines()

        self.jobid = aux[0].split()[-1]


    def check(self):
        '''
        Check job status slurm
        '''

        # Grab slurm state and log file from TRANSP
        self.infoSLURM = getSLURMstatus(
            self.folder_local,
            self.machineSettings,
            jobid=self.jobid,
            grablog=f"{self.folderExecution}/slurm_output.dat",
            name=self.slurm_settings['name'],
        )

        with open(self.folder_local + "/slurm_output.dat", "r") as f:
            self.log_file = f.readlines()

        # If jobid was given as None, I retrieved the info from the job_name, but now provide here the actual id
        if self.infoSLURM is not None:
            self.jobid_found = self.infoSLURM["JOBID"]
        else:
            self.jobid_found = None

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


# -------------------------------------


def runFunction_Complete(
    FunctionToRun,
    Params_in,
    WhereIsFunction="",
    machineSettings={"machine": "local", "folderWork": "~/", "clear": False},
    scratchFolder="~/scratch/",
):
    """
    Runs function externally.
    Important note... this will use the function as is at the time of execution
    """

    txt = f"""
import sys,pickle
{WhereIsFunction}

paramF 		= sys.argv[1]
Params_in 	= pickle.load(open(paramF,'rb'))
print(' ~~ Parameters successfully read')

Params_for_function = Params_in[:-2]
folder 				= Params_in[-2]

Params_out = {FunctionToRun}(*Params_for_function)

with open(folder+'/Params_out.pkl','wb') as handle:	pickle.dump(Params_out,handle,protocol=2)
"""

    IOtools.askNewFolder(scratchFolder, force=True)
    fileExe = scratchFolder + "/exe.py"
    with open(fileExe, "w") as f:
        f.write(txt)

    ScriptToRun = f'python3 {machineSettings["folderWork"]}/exe.py'

    Params_out = runFunction(
        ScriptToRun,
        Params_in,
        InputFiles=[fileExe],
        machineSettings=machineSettings,
        scratchFolder=scratchFolder,
    )

    return Params_out


def runFunction(
    ScriptToRun,
    Params_in,
    InputFiles=[],
    machineSettings={"machine": "local", "folderWork": "~/", "clear": False},
    scratchFolder="~/scratch/",
):
    """
    Params_in is tuple
    """

    scratchFolder = IOtools.expandPath(scratchFolder)

    if not os.path.exists(scratchFolder):
        IOtools.askNewFolder(scratchFolder, force=True)

    # First, put parameters into a pickle
    Params_in += (machineSettings["folderWork"],)
    pickF = "parameters_in.pkl"
    with open(scratchFolder + "/" + pickF, "wb") as handle:
        pickle.dump(Params_in, handle, protocol=1)

    # Then, run function that takes the location of Params.pkl and creates Params_out.pkl
    commandMain = f'{ScriptToRun} {machineSettings["folderWork"]}/{pickF}'
    InputFiles.append(scratchFolder + "/" + pickF)
    runCommand(
        commandMain,
        InputFiles,
        outputFiles=["Params_out.pkl"],
        whereOutput=scratchFolder,
        machineSettings=machineSettings,
    )

    # Read output
    Params_out = pickle.load(open(scratchFolder + "/Params_out.pkl", "rb"))

    return Params_out


def run_subprocess(commandExecute, shell=False, timeoutSecs=None, localRun=False):
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
        print("\t\t\t++ Running local process")
        shell = True

    executable = (
        "/bin/bash" if shell else None
    )  # If I'm simply running on the shell, then it's better to indicate bash

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

    return error, result


def runCommand_remote(
    command,
    machine="local",
    user=None,
    tunnel=None,
    userTunnel=None,
    port=None,
    identity=None,
    shell=False,
    timeoutSecs=None,
):
    # Building command for port and identity options --------------

    sshCommand = ["ssh"]

    if port is not None:
        portCommand = f"-p {port}"
        sshCommand.append(portCommand)
    else:
        portCommand = ""
    if identity is not None:
        sshCommand.append("-i")
        sshCommand.append(identity)

    if user is None:
        userCommand = ""
    else:
        userCommand = user + "@"

    if userTunnel is None:
        userTunnelCommand = ""
    else:
        userTunnelCommand = userTunnel + "@"

    # --------------------------------------------------------------

    if machine == "local" or machine == socket.gethostname():
        error, result = run_subprocess(command, localRun=True)
    else:
        if tunnel is None or len(tunnel) == 0:
            commandExecute = [f"{userCommand}{machine}", command]
        else:
            commandExecute = [
                f"{userTunnelCommand}{tunnel}",
                f'ssh {user}@{machine} "{command}"',
            ]

        sshCommand.extend(commandExecute)
        error, result = run_subprocess(sshCommand, shell=shell, timeoutSecs=timeoutSecs)

    return error, result


def sendCommand_remote(
    file,
    folderWork,
    machine="local",
    user=None,
    tunnel=None,
    userTunnel=None,
    folderWorkTunnel=None,
    port=None,
    identity=None,
    isItFolders=False,
):
    # Building command for port and identity options --------------

    sshCommand = ["ssh"]

    if port is not None:
        portCommand = f"-p {port}"
        sshCommand.append(portCommand)
    else:
        portCommand = ""
    if identity is not None:
        identityCommand = f"-i {identity}"
        sshCommand.append(identityCommand)
    else:
        identityCommand = ""

    if user is None:
        userCommand = ""
    else:
        userCommand = user + "@"

    if userTunnel is None:
        userTunnelCommand = ""
    else:
        userTunnelCommand = userTunnel + "@"

    if folderWorkTunnel is None:
        folderWorkTunnel = folderWork

    if machine == "local" and not isItFolders:
        inFiles = file[1:-1].split(",")
    else:
        inFiles = file.split(" ")

    # --------------------------------------------------------------

    if machine == "local" or machine == socket.gethostname():
        if not isItFolders:
            for file0 in inFiles:
                if len(file0) > 0:
                    os.system(f"cp {file0} {folderWork}/.")
        else:
            for file0 in inFiles:
                os.system(f"cp -r {file0} {folderWork}/.")
    elif tunnel is None:
        if not isItFolders:
            commai = f"scp {quiet_tag}{portCommand.upper()} {identityCommand} {file} {userCommand}{machine}:{folderWork}/."
        else:
            addSolutionForPathCanonicalization = ' -O' # This was needed for iris for a particular user
            commai = f"scp {quiet_tag}{portCommand.upper()} {identityCommand} -r{addSolutionForPathCanonicalization} {file} {userCommand}{machine}:{folderWork}/."
        # run_subprocess(commai,localRun=True)
        os.system(commai)
    else:
        # Send files to tunnel
        sendCommand_remote(
            file,
            folderWorkTunnel,
            machine=tunnel,
            user=userTunnel,
            tunnel=None,
            port=port,
            isItFolders=isItFolders,
        )

        # Send files from tunnel to remote (this ssh connections is better to do file by file)

        for file0 in inFiles:
            if not isItFolders:
                if len(file0.split("/")) > 1:
                    file1 = IOtools.reducePathLevel(file0, level=1, isItFile=True)[1]
                else:
                    file1 = file0
                commandExecute = [
                    f"{userTunnelCommand}{tunnel}",
                    f"scp {quiet_tag}{folderWorkTunnel}/{file1} {userCommand}{machine}:{folderWork}/.",
                ]
            else:
                file1 = IOtools.reducePathLevel(file0, level=1, isItFile=False)[1]
                commandExecute = [
                    f"{userTunnelCommand}{tunnel}",
                    f"scp -r {quiet_tag}{folderWorkTunnel}/{file1} {userCommand}{machine}:{folderWork}/.",
                ]
            sshCommand.extend(commandExecute)

            with subprocess.Popen(
                sshCommand, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as p:
                result, error = p.communicate()

def receiveCommand_remote(
    file,
    folderWork,
    f2,
    machine="local",
    user=None,
    tunnel=None,
    userTunnel=None,
    port=None,
    identity=None,
    whereFiles=None,
    folderWorkTunnel=None,
    isItFolders=False,
    scpCommand="scp -TO",
):
    if folderWorkTunnel is None:
        folderWorkTunnel = folderWork

    if whereFiles is None:
        folderWork_extra = folderWork
    else:
        folderWork_extra = whereFiles

    # Building command for port and identity options --------------

    sshCommand = ["ssh"]

    if port is not None:
        portCommand = f"-p {port}"
        sshCommand.append(portCommand)
    else:
        portCommand = ""
    if identity is not None:
        identityCommand = f"-i {identity}"
        sshCommand.append(identityCommand)
    else:
        identityCommand = ""

    # ----
    if not isItFolders:
        outputFiles = file[1:-1].split(",")
    else:
        outputFiles = [file]
    for i in range(len(outputFiles)):
        outputFiles[i] = folderWork + outputFiles[i]

    if len(outputFiles) < 2:
        file_mod = outputFiles[0]
    else:
        # file_mod = '{'+','.join(outputFiles)+'}'
        file_mod = '"' + " ".join(outputFiles) + '"'

    # --------------------------------------------------------------

    if user is None:
        userCommand = ""
    else:
        userCommand = user + "@"

    if userTunnel is None:
        userTunnelCommand = ""
    else:
        userTunnelCommand = userTunnel + "@"

    if machine == "local":
        if verbose_level in [4, 5]:
            print("\t\t\t++ Copying local file")
        for file0 in outputFiles:
            if isItFolders:
                os.system(f"cp -r {file0} {f2}/.")
            else:
                os.system(f"cp {file0} {f2}/.")
    elif tunnel is None:
        if verbose_level in [4, 5]:
            print(f"\t\t\t- Secure ssh copy from {machine}:{folderWork} to local")
        port_iden = f"{portCommand.upper()} {identityCommand}"
        if isItFolders:
            commandExecute = f"{scpCommand} -r {quiet_tag}{port_iden} {userCommand}{machine}:{file_mod} {f2}/."  # https://unix.stackexchange.com/questions/708517/scp-multiple-files-with-single-command
        else:
            commandExecute = f"{scpCommand} {quiet_tag}{port_iden} {userCommand}{machine}:{file_mod} {f2}/."
        os.system(commandExecute)
    else:
        # Execute command to retrieve files from remote to tunnel (this ssh connections is better to do file by file)
        for file_mod_extra in outputFiles:
            if verbose_level in [4, 5]:
                print(
                    f"\t\t\t- In {tunnel}: Secure ssh copy from {machine}:{folderWork_extra} to {tunnel}:{folderWorkTunnel}"
                )
            if isItFolders:
                comm = f"scp -r {quiet_tag}{userCommand}{machine}:{file_mod_extra} {folderWorkTunnel}/."
            else:
                comm = f"scp {quiet_tag}{userCommand}{machine}:{file_mod_extra} {folderWorkTunnel}/."

            commandExecute = [f"{userTunnelCommand}{tunnel}", comm]
            sshCommand.extend(commandExecute)

            with subprocess.Popen(
                sshCommand, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as p:
                result, error = p.communicate()

        # Execute command to retrieve files from tunnel to local
        receiveCommand_remote(
            file,
            folderWorkTunnel,
            f2,
            machine=tunnel,
            user=userTunnel,
            tunnel=None,
            port=port,
            isItFolders=isItFolders,
            scpCommand=scpCommand,
        )


def runCommand(
    commandMain,
    inputFiles,
    outputFiles=[],
    inputFolders=[],
    outputFolders=[],
    whereOutput="~/.",
    machineSettings={"machine": "local", "folderWork": "~/", "clear": False},
    timeoutSecs=1e6,
    whereFiles=None,
):
    timeOut_txt = (
        f", will timeout execution in {timeoutSecs}s" if timeoutSecs < 1e6 else ""
    )

    machine = machineSettings["machine"]
    folderWork = machineSettings["folderWork"]

    clearYN = machineSettings.setdefault("clear", False)
    user = machineSettings.setdefault("user", None)
    tunnel = machineSettings.setdefault("tunnel", None)
    port = machineSettings.setdefault("port", None)
    identity = machineSettings.setdefault("identity", None)
    userTunnel = machineSettings.setdefault("userTunnel", user)
    folderWorkTunnel = machineSettings.setdefault("folderWorkTunnel", folderWork)

    isTunnelSameMachine = machineSettings.setdefault("isTunnelSameMachine", 0)

    # --- If I am at the tunnel, do not tunnel!
    if (tunnel is not None) and (socket.gethostname() in tunnel):
        tunnel = None
    # ----

    if (tunnel is not None) and isTunnelSameMachine:
        print(
            "\t- Tunnel and machine share the same file system, do not tunnel for file handling",
            typeMsg="i",
        )
        tunnelF = None
        machineF = tunnel
    else:
        tunnelF = tunnel
        machineF = machine

    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(
        f"\n\t------------------------ Running process ({currentTime}{timeOut_txt}) ------------------------"
    )
    if verbose_level in [4, 5]:
        print(f"\t * process: {commandMain}")
        machine_info = (
            f"\t * in machine {machine}"
            if machine != "local"
            else "\t * locally in this machine"
        )
        if tunnel is not None:
            tunnel_info = f" through {tunnel}"
            if identity is not None:
                tunnel_info += f" with identity {identity}"
            if port is not None:
                tunnel_info += f" with port {port}"
            machine_info += tunnel_info

        print(machine_info)

        print("\t * following these steps:")

    # ------------ Create temporal folders if they do not exist

    contStep = 1

    print(f"\t\t{contStep}. Creating temporal folders")

    if tunnelF is not None:
        # -------- Sometimes cybele doesn't have scratch... not sure why
        folderScratch = IOtools.reducePathLevel(
            folderWorkTunnel, level=1, isItFile=False
        )[0]
        command = f"mkdir {folderScratch}"
        error, result = runCommand_remote(
            command,
            machine=tunnel,
            user=userTunnel,
            userTunnel=None,
            tunnel=None,
            port=port,
            identity=identity,
        )
        # ---------------------------

        command = f"mkdir {folderWorkTunnel}"
        error, result = runCommand_remote(
            command,
            machine=tunnel,
            user=userTunnel,
            userTunnel=None,
            tunnel=None,
            port=port,
            identity=identity,
        )

    command = f"mkdir {folderWork}"
    error, result = runCommand_remote(
        command,
        machine=machineF,
        user=user,
        tunnel=tunnelF,
        userTunnel=userTunnel,
        port=port,
        identity=identity,
    )

    # ------------ Send files to machine

    if len(inputFiles) > 0:
        contStep += 1
        print(f"\t\t{contStep}. Sending required files")
        if machine == "local":
            combined_files = "{" + ",".join(inputFiles) + ",}"
        else:
            combined_files = " ".join(inputFiles)
        sendCommand_remote(
            combined_files,
            folderWork,
            folderWorkTunnel=folderWorkTunnel,
            machine=machineF,
            user=user,
            tunnel=tunnelF,
            userTunnel=userTunnel,
            port=port,
            identity=identity,
            isItFolders=False,
        )

    # ------------ Send folders to machine

    if len(inputFolders) > 0:
        contStep += 1
        print(f"\t\t{contStep}. Sending required folders")
        combined_folders = " ".join(inputFolders)
        sendCommand_remote(
            combined_folders,
            folderWork,
            folderWorkTunnel=folderWorkTunnel,
            machine=machineF,
            user=user,
            tunnel=tunnelF,
            userTunnel=userTunnel,
            port=port,
            identity=identity,
            isItFolders=True,
        )

    # ------------ Run command

    error, result = "", ""
    contStep += 1
    print(f"\t\t{contStep}. Running command")
    if verbose_level in [4, 5]:
        print(f"\t\t\t{commandMain}")
        
    error, result = runCommand_remote(
        commandMain,
        machine=machine,
        user=user,
        userTunnel=userTunnel,
        tunnel=tunnel,
        port=port,
        identity=identity,
        timeoutSecs=timeoutSecs,
    )

    # ---------------------------------------------------------------
    # ------------ Bring files from machine
    # ---------------------------------------------------------------

    scpCommand = "scp -TO"
    allFilesCorrect = bringFiles(
        outputFiles,
        outputFolders,
        folderWork,
        whereOutput,
        folderWorkTunnel,
        machineF,
        user,
        tunnelF,
        userTunnel,
        port,
        identity,
        whereFiles,
        scpCommand,
        contStep=contStep,
    )

    if not allFilesCorrect:
        print(
            "* Exploring SCP option -T instead of -TO because retrieval failed",
            typeMsg="w",
        )
        scpCommand = "scp -T"
        allFilesCorrect = bringFiles(
            outputFiles,
            outputFolders,
            folderWork,
            whereOutput,
            folderWorkTunnel,
            machineF,
            user,
            tunnelF,
            userTunnel,
            port,
            identity,
            whereFiles,
            scpCommand,
            contStep=contStep,
        )

    if not allFilesCorrect:
        print(
            "\t- Not all output files received from remote computer, printing process error:",
            typeMsg="w",
        )
        try:
            print("".join(error))
        except:
            print(error)
        print("\t- Printing process result:")
        try:
            print("".join(result))
        except:
            print(result)
        print(
            "\t- For easy debugging, I am not clearing the remote folder", typeMsg="w"
        )
        clearYN = False

    # Clear stuff
    if clearYN:
        command = f"rm -r {folderWork}"

        error, result = runCommand_remote(
            command,
            machine=machine,
            user=user,
            userTunnel=userTunnel,
            tunnel=tunnel,
            port=port,
            identity=identity,
        )
        if tunnelF is not None:
            command = f"rm -r {folderWorkTunnel}"
            error, result = runCommand_remote(
                command,
                machine=tunnel,
                user=user,
                tunnel=None,
                port=port,
                identity=identity,
            )

    print(
        f"\t------------------------ Finished process ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ------------------------"
    )
    print("\n")


def bringFiles(
    outputFiles,
    outputFolders,
    folderWork,
    whereOutput,
    folderWorkTunnel,
    machine,
    user,
    tunnel,
    userTunnel,
    port,
    identity,
    whereFiles,
    scpCommand,
    contStep=0,
):
    
    if len(outputFiles) > 0:
        contStep += 1
        print(f"\t\t{contStep}. Receiving files")
        combined_files = "{" + ",".join(outputFiles) + "}"
        receiveCommand_remote(
            combined_files,
            folderWork,
            whereOutput,
            isItFolders=False,
            folderWorkTunnel=folderWorkTunnel,
            machine=machine,
            user=user,
            tunnel=tunnel,
            userTunnel=userTunnel,
            port=port,
            identity=identity,
            whereFiles=whereFiles,
            scpCommand=scpCommand,
        )

    if len(outputFolders) > 0:
        contStep += 1
        print(f"\t\t{contStep}. Receiving folders")
        for outputFolder in outputFolders:
            # First remove current folders
            upo = 0
            tester = f"{whereOutput}/{outputFolder}_old{upo}"
            while os.path.exists(f"{whereOutput}/{outputFolder}"):
                if os.path.exists(tester):
                    upo += 1
                    tester = f"{whereOutput}/{outputFolder}_old{upo}"
                else:
                    os.system(f"mv {whereOutput}/{outputFolder} {tester}")

            receiveCommand_remote(
                outputFolder,
                folderWork,
                whereOutput,
                isItFolders=True,
                folderWorkTunnel=folderWorkTunnel,
                machine=machine,
                user=user,
                tunnel=tunnel,
                userTunnel=userTunnel,
                port=port,
                identity=identity,
                whereFiles=whereFiles,
                scpCommand=scpCommand,
            )

    allFilesCorrect = True
    for file in outputFiles:
        fileF = os.path.exists(f"{whereOutput}/{file}")
        allFilesCorrect = allFilesCorrect and fileF
        if not fileF:
            print(f"\t\t\t- File {file} not found", typeMsg="w")
    for folder in outputFolders:
        folderF = os.path.exists(f"{whereOutput}/{folder}")
        allFilesCorrect = allFilesCorrect and folderF
        if not folderF:
            print(f"\t\t\t- Folder {folder} not found", typeMsg="w")

    return allFilesCorrect


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


def SLURM(
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
    email="noemail",
    waitYN=True,
    extranamelogs="",
    default_exclusive=False,
):
    if folder_local is None:
        folder_local = folder_remote
    if isinstance(command, str):
        command = [command]

    folderExecution = IOtools.expandPath(folder_remote)
    fileSBTACH = f"{folder_local}/bash.src"
    fileSHELL = f"{folder_local}/mitim.sh"
    fileSBTACH_remote = f"{folder_remote}/bash.src"

    minutes = int(minutes)

    partition = slurm.setdefault("partition",None)
    exclude = slurm.setdefault("exclude",None)
    account = slurm.setdefault("account",None)
    constraint = slurm.setdefault("constraint",None)


    """
	********************************************************************************************
	Write bash.src file to execute
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
        f"#SBATCH --output {folderExecution}/slurm_output{extranamelogs}.dat"
    )
    commandSBATCH.append(
        f"#SBATCH --error {folderExecution}/slurm_error{extranamelogs}.dat"
    )
    if email != "noemail":
        commandSBATCH.append("#SBATCH --mail-user=" + email)

    # ******* Partition / Billing
    commandSBATCH.append(f"#SBATCH --partition {partition}")

    if account is not None:
        commandSBATCH.append(f"#SBATCH --account {account}")
    if constraint is not None:
        commandSBATCH.append(f"#SBATCH --constraint {constraint}")

    commandSBATCH.append(f"#SBATCH --time {time_com}")

    if job_array is None:
        if default_exclusive:
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

    wait_txt = " --wait" if waitYN else ""
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
	Write mitim.sh file that handles the execution of the bash.src with pre and post commands
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
    commandSHELL.append(
        "python3 $MITIM_PATH/src/mitim_tools/misc_tools/FARMINGtools.py sbatch.out"
    )

    if os.path.exists(fileSHELL):
        os.system(f"rm {fileSHELL}")
    with open(fileSHELL, "w") as f:
        f.write("\n".join(commandSHELL))

    """
	********************************************************************************************
	Command to send through scp
	********************************************************************************************
	"""

    comm = f"cd {folderExecution} && bash mitim.sh > sbatch.out"

    return comm, fileSBTACH, fileSHELL

def curateOutFiles(outputFiles):
    # Avoid repetitions, otherwise, e.g., they will fail to rename

    if "sbatch.out" not in outputFiles:
        outputFiles.append("sbatch.out")

    outputFiles_new = []
    for file in outputFiles:
        if file not in outputFiles_new:
            outputFiles_new.append(file)

    return outputFiles_new


def getSLURMstatus(FolderSLURM, machineSettings, jobid=None, name=None, grablog=None):
    """
    Search by jobid or by name
    """

    folderWork = machineSettings["folderWork"]

    if jobid is not None:
        txt_look = f"-j {jobid}"
    else:
        txt_look = f"-n {name}"

    command = f'cd {folderWork} && squeue {txt_look} -o "%.15i %.24P %.18j %.10u %.10T %.10M %.10l %.5D %R" > squeue.out'

    outputFiles = ["squeue.out"]

    if grablog is not None:
        command += f" && cp {grablog} {folderWork}/."
        outputFiles.append(IOtools.reducePathLevel(grablog)[-1])

    runCommand(
        command,
        [],
        outputFiles=outputFiles,
        machineSettings=machineSettings,
        whereOutput=FolderSLURM,
    )

    with open(f"{FolderSLURM}/squeue.out", "r") as f:
        aux = f.readlines()

    if len(aux) > 1:
        info = {}
        for i in range(len(aux[0].split())):
            info[aux[0].split()[i]] = aux[1].split()[i]
    else:
        info = None

    return info


def printEfficiencySLURM(out_file):
    """
    It reads jobid from sbatch.out or slurm_output.dat
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
