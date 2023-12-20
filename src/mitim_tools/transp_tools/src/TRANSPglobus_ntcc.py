import sys, os, time
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import FARMINGtools, IOtools
from mitim_tools.gacode_tools.aux import GACODEdefaults
from mitim_tools.transp_tools.src import TRANSPglobus, TRANSPmain
from mitim_tools.transp_tools.tools import NMLtools
from mitim_tools.misc_tools import CONFIGread

from mitim_tools.misc_tools.IOtools import printMsg as print

retrieveTerminalOutput = True


def tr_start(
    shotnumber,
    runid,
    email,
    FolderTRANSP,
    version,
    mpisettings,
    tok,
    typeStart=1,
    runTRDAT=True,
):
    if FolderTRANSP[-1] != "/":
        FolderTRANSP += "/"

    subfolder = IOtools.reducePathLevel(FolderTRANSP, level=1, isItFile=False)[
        1
    ]  #'/FolderTRANSP/'

    # --------------------------------
    # Where to run
    # --------------------------------

    machineSettings = CONFIGread.machineSettings(
        code="ntcc", nameScratch=f"mitim_tmp_{runid}/"
    )
    scratchArea = f"{machineSettings['folderWork']}/{subfolder}/"

    # -------------------------------------
    # Prepare tr_start IDL script responses
    # -------------------------------------

    trmpi, toricmpi, ptrmpi, txt_version = "", "", "", ""
    if mpisettings["trmpi"] > 1:
        trmpi = f"{mpisettings['trmpi']}\n"
    if mpisettings["toricmpi"] > 1:
        toricmpi = f"{mpisettings['toricmpi']}\n"
    if mpisettings["ptrmpi"] > 1:
        ptrmpi = f"{mpisettings['ptrmpi']}\n"
    if version == "tshare":
        txt_version = " tshare"

    with open(FolderTRANSP + "responsestrstart.txt", "w") as f:
        if typeStart == 1:
            f.write(f"{trmpi}{toricmpi}{ptrmpi}{tok}\n60\nY\nY")
        if typeStart == 2:
            f.write(f"{tok}\nY\nY\n{trmpi}{toricmpi}{ptrmpi}")
    with open(FolderTRANSP + f"{runid}CC.TMP", "w") as f:
        f.write("TRANSP run performed via MITIM")

    # --------------------------------
    # Prepare associated files
    # --------------------------------

    NMLtools.adaptNML(FolderTRANSP, runid, shotnumber, scratchArea)

    # Run trdat to make sure things are fine
    if runTRDAT:
        tr_dat(runid, tok, FolderTRANSP)

    # --------------------------------
    # Command
    # --------------------------------

    inputFiles = []
    inputFolders = [FolderTRANSP]
    outputFiles = [
        f"{runid}_{tok}_tmp.tar.gz",
        f"{runid}_{tok}.REQUEST",
    ]
    if retrieveTerminalOutput:
        outputFiles.append("outputtrstart.txt")
        extraout = " >> outputtrstart.txt"
        os.system(f"rm {FolderTRANSP}/outputtrstart.txt")
    else:
        extraout = ""

    notriage_option = " notriage"

    extracoms = f"export TR_EMAIL={email} && export EDITOR=cat"

    # extracoms = "export RESULTDIR={0}\nexport DATADIR={1}\nexport TMPDIR={2}\nexport DATADIR=''".format(email)
    Command = (
        extracoms
        + " && cd {0} && tr_start {1}{2}{3} < responsestrstart.txt{4}".format(
            scratchArea, runid, notriage_option, txt_version, extraout
        )
    )

    for file in outputFiles:
        Command += f" && cp {file} ../."

    # --------------------------------
    # Run
    # --------------------------------

    timeoutSecs = 120  # THis is needed, to avoid crazy high file sizes!!

    FARMINGtools.runCommand(
        Command,
        inputFiles,
        outputFiles=outputFiles,
        inputFolders=inputFolders,
        machineSettings=machineSettings,
        whereOutput=FolderTRANSP,
        timeoutSecs=timeoutSecs,
    )

    # --------------------------------
    # Checker
    # --------------------------------
    if not os.path.exists(f"{FolderTRANSP}/{runid}_{tok}.REQUEST"):
        print(
            f"\n\nFile {FolderTRANSP}/{runid}_{tok}.REQUEST was not generated, likely due to a tr_start failure (namelist and UFILES are not correct)",
            typeMsg="w",
        )
        print("Disregard this message and continue? (c)", typeMsg="q")


def tr_dat(runid, tok, FolderTRANSP):
    print(
        "\t- It has been requested that I run TRDAT to make sure things are right",
        typeMsg="i",
    )

    machineSettings = CONFIGread.machineSettings(
        code="ntcc", nameScratch=f"mitim_tmp_{runid}/"
    )

    subfolder = IOtools.reducePathLevel(FolderTRANSP, level=1, isItFile=False)[
        1
    ]  #'/FolderTRANSP/'
    scratchArea = f"{machineSettings['folderWork']}/{subfolder}/"

    # --------------------------------
    # Command
    # --------------------------------

    inputFiles = []
    inputFolders = [FolderTRANSP]
    outputFiles = [f"{subfolder}/outputtrdat.txt"]
    os.system(f"rm {FolderTRANSP}/outputtrdat.txt")

    Command = f"cd {scratchArea} && trdat {tok} {runid} Q >> outputtrdat.txt"

    # --------------------------------
    # Run
    # --------------------------------

    timeoutSecs = 120  # THis is needed, to avoid crazy high file sizes!!

    FARMINGtools.runCommand(
        Command,
        inputFiles,
        outputFiles=outputFiles,
        inputFolders=inputFolders,
        machineSettings=machineSettings,
        whereOutput=FolderTRANSP,
        timeoutSecs=timeoutSecs,
    )

    # --------------------------------
    # Interpret
    # --------------------------------

    NMLtools.interpret_trdat(f"{FolderTRANSP}/outputtrdat.txt")


def tr_send(FolderTRANSP, runid, tok, waitseconds=1 * 60):
    """
    Kill look command after waitseconds
    """

    if FolderTRANSP[-1] != "/":
        FolderTRANSP += "/"

    # --------------------------------
    # Where to run
    # --------------------------------

    machineSettings = CONFIGread.machineSettings(
        code="ntcc", nameScratch=f"mitim_tmp_{runid}/"
    )
    scratchArea = machineSettings["folderWork"]

    # --------------------------------
    # Command
    # --------------------------------

    Command = "cd {0} && tr_send_pppl.pl {1} {2} NOMDSPLUS >> outputtrsend.txt".format(
        scratchArea, tok, runid
    )

    outputFiles = []
    if retrieveTerminalOutput:
        outputFiles.append("outputtrsend.txt")
        os.system(f"rm {FolderTRANSP}/outputtrsend.txt")

    inputFiles = [
        FolderTRANSP + f"{runid}_{tok}_tmp.tar.gz",
        FolderTRANSP + f"{runid}_{tok}.REQUEST",
    ]

    # --------------------------------
    # Run
    # --------------------------------

    FARMINGtools.runCommand(
        Command,
        inputFiles,
        outputFiles=outputFiles,
        machineSettings=machineSettings,
        whereOutput=FolderTRANSP,
        timeoutSecs=waitseconds,
    )


def tr_look(FolderTRANSP, runid, tok, waitseconds=60):
    """
    Kill look command after waitseconds. Since I'm not waiting for generation here, this can be short.
    """

    if FolderTRANSP[-1] != "/":
        FolderTRANSP += "/"

    # --------------------------------
    # Where to run
    # --------------------------------

    machineSettings = CONFIGread.machineSettings(
        code="ntcc", nameScratch=f"mitim_tmp_{runid}/"
    )
    scratchArea = machineSettings["folderWork"]

    """
	--------------------------------
	Command
		e.g. trlook 12345B10 AUGD nomdsplus
	--------------------------------
	"""

    Command = f"cd {scratchArea} && tr_look {runid} {tok} nomdsplus >> outputtrlook.txt"

    inputFiles = []
    outputFiles = []
    if retrieveTerminalOutput:
        outputFiles.append("outputtrlook.txt")
        os.system(f"rm {FolderTRANSP}/outputtrlook.txt")

    # --------------------------------
    # Run
    # --------------------------------

    FARMINGtools.runCommand(
        Command,
        inputFiles,
        outputFiles=outputFiles,
        machineSettings=machineSettings,
        whereOutput=FolderTRANSP,
        timeoutSecs=waitseconds,
    )


def tr_get(file, server, runid, FolderTRANSP, tok, remove_previous_before=False):
    """
    I unified the LOOK and the FINISH getter routines, only differentiated by the server
    """

    file_in_server = f"{server}/{file}"

    if FolderTRANSP[-1] != "/":
        FolderTRANSP += "/"

    # --------------------------------
    # Where to run
    # --------------------------------

    machineSettings = CONFIGread.machineSettings(
        code="ntcc", nameScratch=f"mitim_tmp_{runid}/"
    )
    scratchArea = machineSettings["folderWork"]

    # --------------------------------
    # Command
    # --------------------------------

    Command = "cd {0} && globus-url-copy -nodcau gsiftp://{1} file://{0}/{2} >> outputtrfetch.txt".format(
        scratchArea, file_in_server, file
    )

    inputFiles = []
    outputFiles = [file]
    if retrieveTerminalOutput:
        outputFiles.append("outputtrfetch.txt")

    # --------------------------------
    # Run
    # --------------------------------

    if remove_previous_before:
        error, result = FARMINGtools.run_subprocess(
            f"cd {FolderTRANSP} && rm {file}", localRun=True
        )

    FARMINGtools.runCommand(
        Command,
        inputFiles,
        outputFiles=outputFiles,
        machineSettings=machineSettings,
        whereOutput=FolderTRANSP,
    )


def tr_cancel(runid, FolderTRANSP, tok, howManyCancel=1, MinWaitDeletion=2):
    if FolderTRANSP[-1] != "/":
        FolderTRANSP += "/"

    # --------------------------------
    # Where to run
    # --------------------------------

    machineSettings = CONFIGread.machineSettings(
        code="ntcc", nameScratch=f"mitim_tmp_{runid}/"
    )
    scratchArea = machineSettings["folderWork"]

    # --------------------------------
    # Command
    # --------------------------------

    Command = "cd {0} && globus-job-submit transpgrid.pppl.gov /u/pshare/globus/transp_cleanup {1} {2} >> outputtrcancel.txt".format(
        scratchArea, runid, tok
    )

    inputFiles = []
    outputFiles = []
    if retrieveTerminalOutput:
        outputFiles.append("outputtrcancel.txt")

    # --------------------------------
    # Run
    # --------------------------------

    print(f">> Deleting {runid} from the grid")
    for k in range(howManyCancel):
        FARMINGtools.runCommand(
            Command,
            inputFiles,
            outputFiles=outputFiles,
            timeoutSecs=5,
            machineSettings=machineSettings,
            whereOutput=FolderTRANSP,
        )

    # Leave some more time for deletion
    print(
        ">> Cancel request has been submitted, but waiting now {0}min to allow for deletion to happen properly".format(
            MinWaitDeletion
        )
    )
    time.sleep(MinWaitDeletion * 60.0)
