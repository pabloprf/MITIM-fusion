import re
import os
import shutil
import sys
import time
import datetime
import socket
import random
import zipfile
import cProfile
import termios
import tty
import h5py
import warnings
import subprocess
import json
import functools
import contextlib
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas
except ImportError:
    pass

try:
    from IPython import embed
except ImportError:
    pass

import urllib.request as urlREQ  # urllibR
import urllib.error as urlERR  # urllibE


class speeder(object):
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        
        self.profiler = cProfile.Profile()
        self.timeBeginning = datetime.datetime.now()

        print(">>> Profiling started")
        self.profiler.enable()

        return self

    def __exit__(self, *args):

        self.profiler.disable()
        print(">>> Profiling ended")

        self._get_time()

    def _get_time(self):

        self.timeDiff = getTimeDifference(self.timeBeginning, niceText=False)
        self.profiler.dump_stats(self.file)

        print(
            f'Script took {createTimeTXT(self.timeDiff)}, profiler stats dumped to {self.file} (open with "python3 -m snakeviz {self.file}")'
        )

class timer(object):

    def __init__(self, name="\t* Script"):
        self.name = name

    def __enter__(self):
        self.timeBeginning = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        self._get_time()

    def _get_time(self):

        self.timeDiff = getTimeDifference(self.timeBeginning, niceText=False)

        print(f'{self.name} took {createTimeTXT(self.timeDiff)}')

# Decorator to time functions

def mitim_timer(name="\t* Script"):
    def decorator_timer(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with timer(name):
                return func(*args, **kwargs)
        return wrapper_timer
    return decorator_timer

def clipstr(txt, chars=40):
    return f"{'...' if len(txt) > chars else ''}{txt[-chars:]}" if txt is not None else None

def receiveWebsite(url, data=None):
    NumTriesAfterTimeOut = 60
    secWaitTimeOut = 10

    # ------------------------------------------------------------------------------------------
    # Loop to retry in case of `URLError: <urlopen error [Errno 110] Connection timed out>`
    # ------------------------------------------------------------------------------------------
    response, status = None, 0
    for k in range(NumTriesAfterTimeOut):
        # Standard response (works on mfews)
        try:
            # ip = random.randint(1,999)
            # proxy = urlREQ.ProxyHandler({"http": f"{ip}.69.140.{ip}:53281"})
            # opener = urlREQ.build_opener(proxy)
            # urlREQ.install_opener(opener)

            req = urlREQ.Request(url, data)
            # response = urlREQ.urlopen(req) #mfews
            import ssl

            response = urlREQ.urlopen(
                req, context=ssl._create_unverified_context()
            )  # iris

            break

        except (urlERR.URLError, urlERR.HTTPError) as _excp:
            print(
                " -------> Website did not respond, relaunching new info request in {0}s".format(
                    secWaitTimeOut
                )
            )  # repr(_excp)
            time.sleep(secWaitTimeOut)

    return response

def page(url):
    req = urlREQ.Request(url)
    try:
        response = urlREQ.urlopen(req)
    except:
        import ssl

        context = ssl._create_unverified_context()
        response = urlREQ.urlopen(req, context=context)
    the_page = response.read()

    return the_page

def printMsg(*args, typeMsg="", verbose=None):
    """
    Print messages with different colors (blue-red is better for colorblind)
    """

    # Info (about a choice or something found): Blue
    if typeMsg == "i":
        extra = "\u001b[34m"
    # Warning (about something to be careful about, even if chosen): Red
    elif typeMsg == "w":
        extra = "\u001b[31;1m"
    # Decision to go faster (e.g. because results are found are a step is skipped): Cyan
    elif typeMsg == "f":
        extra = "\u001b[36m"
    # Question or something that is stopped
    elif (typeMsg == "q") or (typeMsg == "qa"):
        extra = "\u001b[44;1m\u001b[37m"
    # Note: Nothing
    else:
        extra = "\u001b[0m"

    total = (extra,) + args + ("\u001b[0m",)

    # Take into account the verbose level is it was provided
    if (verbose is None) or (verbose in [4, 5]):
        print(*total)

    if typeMsg == "q":
        return query_yes_no("\t\t>> Do you want to continue?", extra=extra)

class HiddenPrints:
    """
    Usage:
            with IOtools.HiddenPrints():
                    print("This will not be printed")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_git_info(repo_path):
    # Get branch
    result = subprocess.run(['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        branch = result.stdout.strip()
    else:
        branch = None

    # Get hash
    result = subprocess.run(['git', '-C', repo_path, 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        commit_hash = result.stdout.strip()
    else:
        commit_hash = None

    return branch, commit_hash

def createCDF_simple(file, zvals, names):
    """
    This creates a simple netCDF of 1D variables of the same size

    zvals must be (variable,rho)
    """

    import netCDF4  # Importing here because some machines require netCDF4 libraries

    ncfile = netCDF4.Dataset(file, mode="w", format="NETCDF4_CLASSIC")

    x = ncfile.createDimension("xdim", zvals.shape[1])

    for i, name in enumerate(names):
        value = ncfile.createVariable(name, "f4", ("xdim",))
        value[:] = zvals[i, :]

    ncfile.close()

def printPoints(x, numtabs=1):
    """
    x is (batch, dim)
    """

    tabs = ""
    for i in range(numtabs):
        tabs += "\t"

    for i in range(x.shape[0]):
        txt = f"x{i + 1} = "
        for j in range(x.shape[1]):
            txt += f"{x[i, j]:.3f}, "
        txt = txt[:-2]
        print(tabs + txt)


def isfloat(x):
    return isinstance(x, (np.floating, float))


def isint(x):
    return isinstance(x, (np.integer, int))


def isnum(x):
    return isinstance(x, (np.floating, float)) or isinstance(x, (np.integer, int))


def islistarray(x):
    return type(x) in [list, np.ndarray]


def isAnyNan(x):
    try:
        aux = len(x)
    except:
        aux = None

    if aux is None:
        isnan = np.isnan(x)
    else:
        isnan = False
        for j in range(aux):
            isnan = isnan or np.isnan(x[j]).any()

    return isnan


def randomWait(dakNumUnit, multiplierMin=1):
    # Maximum is (multiplier*9) minutes. So, for 10minutes, multiplierMin~1

    maxSeconds = int(dakNumUnit * multiplierMin * 60.0)
    waitSec = random.randint(0, maxSeconds)
    print(f" >>>>>>>>>>> Waiting {waitSec}s (random, up to {maxSeconds}s)")
    time.sleep(waitSec)


def safeBackUp(FolderToZip, NameZippedFile="Contents", locationZipped="~/scratch/"):
    f1 = moveRecursive(
        check=1,
        commonpreffix=expandPath(locationZipped) + NameZippedFile + "_",
        commonsuffix=".zip",
        eliminateAfter=5,
    )
    zipFolder(FolderToZip, ZippedFile=f1)
    print(f" --> Most current {expandPath(FolderToZip)} folder zipped to {f1}")


def zipFolder(FolderToZip, ZippedFile="Contents.zip"):
    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    with zipfile.ZipFile(ZippedFile, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(FolderToZip, zipf)


def moveRecursive(check=1, commonpreffix="~/Contents_", commonsuffix=".zip", eliminateAfter=5):
    file_current = f"{commonpreffix}{check}{commonsuffix}"

    if os.path.exists(file_current):
        if check >= eliminateAfter:
            os.system("rm " + file_current)
        else:
            file_next = f"{commonpreffix}{check + 1}{commonsuffix}"
            if os.path.exists(file_next):
                moveRecursive(
                    check=check + 1,
                    commonpreffix=commonpreffix,
                    commonsuffix=commonsuffix,
                    eliminateAfter=eliminateAfter,
                )
            os.system(f"mv {file_current} {file_next}")

    return file_current


def getsize(obj):
    print(f"\t{getsizeObject(obj) * 1e-06:.1f} MB")


def getsizeObject(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([getsizeObject(v, seen) for v in obj.values()])
        size += sum([getsizeObject(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += getsizeObject(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([getsizeObject(i, seen) for i in obj])
    return size

def read_mitim_nml(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    optimization_options = data["optimization"]
    optimization_options["StrategyOptions"] =  data["StrategyOptions"]
    optimization_options["surrogateOptions"] = data["surrogateOptions"]

    return optimization_options

def getpythonversion():
    return [
        int(i.split("\n")[0].split("+")[0]) for i in sys.version.split()[0].split(".")
    ]

def zipFiles(files, outputFolder, name="info"):
    os.system(f"mkdir {outputFolder}/{name}")
    for i in files:
        os.system(f"cp {i} {outputFolder}/{name}/")

    shutil.make_archive(f"{outputFolder}/{name}", "zip", outputFolder)

    os.system(f"rm -r {outputFolder}/{name}")


def unzipFiles(file, destinyFolder, clear=True):
    shutil.unpack_archive(file, destinyFolder + "/")

    if clear:
        os.system("rm " + file)


def getProfiles_ExcelColumns(file, fromColumn=0, fromRow=4, rhoNorm=None, sheet_name=0):
    df = pandas.read_excel(file, sheet_name=sheet_name)

    rho = getVar_ExcelColumn(df, df.keys()[fromColumn + 0], fromRow=fromRow)
    Te = getVar_ExcelColumn(df, df.keys()[fromColumn + 1], fromRow=fromRow) * 1e-3
    Ti = getVar_ExcelColumn(df, df.keys()[fromColumn + 2], fromRow=fromRow) * 1e-3
    q = getVar_ExcelColumn(df, df.keys()[fromColumn + 3], fromRow=fromRow)
    ne = getVar_ExcelColumn(df, df.keys()[fromColumn + 4], fromRow=fromRow) * 1e-20

    if rhoNorm is not None:
        Te = np.interp(rhoNorm, rho, Te)
        Ti = np.interp(rhoNorm, rho, Ti)
        q = np.interp(rhoNorm, rho, q)
        ne = np.interp(rhoNorm, rho, ne)

    return rho, Te, Ti, q, ne


def getVar_ExcelColumn(pandafile, columnName, fromRow=4):
    var0 = np.array(pandafile[columnName].values)[fromRow:]
    var = []
    for i in var0:
        if not np.isnan(float(i)):
            var.append(float(i))

    return np.array(var)


def writeProfiles_ExcelColumns(file, rho, Te, q, ne, Ti=None, fromColumn=0, fromRow=4):
    os.system(f"rm {file}")

    if Ti is None:
        Ti = Te

    dictExcel = OrderedDict()
    dictExcel["rho"] = rho
    dictExcel["Te"] = Te
    dictExcel["Ti"] = Ti
    dictExcel["q"] = q
    dictExcel["ne"] = ne

    writeExcel_fromDict(dictExcel, file, fromColumn=fromColumn, fromRow=fromRow)


def writeExcel_fromDict(dictExcel, file, fromColumn=0, fromRow=4):
    df = pandas.DataFrame(dictExcel)
    writer = pandas.ExcelWriter(file, engine="xlsxwriter")
    df.to_excel(
        writer,
        sheet_name="Sheet1",
        header=True,
        index=False,
        startrow=fromRow - 1,
        startcol=fromColumn,
    )
    writer.save()


def createExcelRow(dataSet_dict, row_name="row 1"):
    columns, data = [], []
    for i in dataSet_dict:
        columns.append(i)
        data.append([dataSet_dict[i]])
    data = np.transpose(data)

    df = pandas.DataFrame(data, index=[row_name], columns=columns)

    return df


def addRowToExcel(file, dataSet_dict, row_name="row 1", repeatIfIndexExist=True):
    df = createExcelRow(dataSet_dict, row_name=row_name)

    if os.path.exists(file):
        df_orig = pandas.read_excel(file, index_col=0)
        df_new = df_orig
        if not repeatIfIndexExist and df.index[0] in df_new.index:
            df_new = df_new.drop(df.index[0])
            print(f" ~~~ Row with index {df.index[0]} removed")
        df_new = df_new.append(df)
        print(f" ~~~ Row with index {df.index[0]} added")
    else:
        df_new = df

    with pandas.ExcelWriter(file, mode="w") as writer:
        df_new.to_excel(writer, sheet_name="Sheet1")


def correctNML(BaseFile):
    """
    Note: Sometimes I have found that python changes the way line breaks occur in a file,
    leading to tr_start not being able to read correctly "inputdir". If this happens,
    simply apply the command-line "tr -d '\r' < file_in > file_out". Example:
    """

    os.system("tr -d '\r' < {0} > {0}_new && mv {0}_new {0}".format(BaseFile))


def getTimeDifference(previousTime, newTime=None, niceText=True, factor=1):
    if newTime is None:
        newTime = datetime.datetime.now()

    difference = newTime - previousTime

    duration_in_s = difference.total_seconds() / factor

    if niceText:
        return createTimeTXT(duration_in_s)
    else:
        return duration_in_s


def getnowString(time=True):
    date = datetime.datetime.now()
    txt = f"{date.month}{date.day}{date.year}"
    if time:
        txt += f"_{date.hour}{date.minute}"
    return txt


def getTimeFromString(time_str):
    return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


def getStringFromTime(object_time=None):
    if object_time is None:
        object_time = datetime.datetime.now()

    return object_time.strftime("%Y-%m-%d %H:%M:%S")


def loopFileBackUp(file):
    if os.path.exists(file):
        copyToFile = f"{file}_0"
        if os.path.exists(copyToFile):
            loopFileBackUp(copyToFile)
        os.system(f"mv {file} {copyToFile}")


def createTimeTXT(duration_in_s, until=3):
    days = divmod(duration_in_s, 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds

    try:
        milisec = int(
            (
                duration_in_s
                - (days[0] * 24 * 3600 + hours[0] * 3600 + minutes[0] * 60 + seconds[0])
            )
            * 1000
        )
        milisec_txt = f" ({str(milisec).zfill(2)}ms)"
    except:
        milisec_txt = ""

    if days[0] > 0:
        txt = f"{days[0]}d "
    else:
        txt = ""

    if until > 0:
        if hours[0] > 0:
            txt += f"{str(hours[0]).zfill(2)}h "
        if until > 1:
            if minutes[0] > 0:
                txt += f"{str(minutes[0]).zfill(2)}min "
            if until > 2:
                if seconds[0] > 0:
                    txt += f"{str(seconds[0]).zfill(2)}s "

    if len(txt) == 0:
        if until == 0:
            txt = "<1d "
        elif until == 1:
            txt = "<1h "
        elif until == 2:
            txt = "<1min "
        else:
            txt = f"<1s{milisec_txt} "

    return txt[:-1]


def renameCommand(ini, fin, folder="~/"):
    folder = expandPath(folder)

    if ini is not None:
        if "mfe" in socket.gethostname():
            os.system(f'cd {folder} && rename "s/{ini}/{fin}/" *')
        else:
            for filename in os.listdir(folder):
                if ini in filename:
                    newname = filename.split(ini)[0] + fin + filename.split(ini)[-1]
                    os.system("mv {0}/{1} {0}/{2}".format(folder, filename, newname))


def readExecutionParams(folderExecution, nums=[0, 9]):
    x = []
    for i in np.arange(nums[0], nums[1] + 1, 1):
        params = generateDictionaries(
            folderExecution + "Execution/Evaluation.{0}/params.in.{0}".format(i)
        )

        dictDVs = params["dictDVs"]

        xp = []
        for ikey in dictDVs:
            xp.append(float(dictDVs[ikey]["value"]))

        x.append(np.array(xp))

    return np.array(x)


def askNewFolder(folderWork, force=False, move=None):
    if os.path.exists(folderWork):
        if force:
            os.system(f"rm -r {folderWork}")
        else:
            if move is not None:
                os.system(f"mv {folderWork[:-1]}/ {folderWork[:-1]}_{move}")
            else:
                printMsg(
                    f"You are about to erase the content of {folderWork}", typeMsg="q"
                )
                os.system(f"rm -r {folderWork}")

    os.system(f"mkdir {folderWork}")

    if os.path.exists(folderWork):
        print(f" \t\t~ Folder ...{folderWork[np.max([-40,-len(folderWork)]):]} created")
    else:
        fo = reducePathLevel(folderWork, level=1, isItFile=False)[0]
        askNewFolder(fo, force=False, move=None)
        askNewFolder(folderWork, force=False, move=None)


def removeRepeatedPoints_2D(rs, zs, FirstEqualToLast=True):
    r, z = [], []
    for i in range(len(rs)):
        pointToAdd = round(rs[i], 4), round(zs[i], 4)
        pointAlreadyThere = False
        if pointToAdd[0] in r:
            possible_z = []
            for i in range(len(r)):
                if r[i] == pointToAdd[0]:
                    possible_z.append(z[i])
            if pointToAdd[1] in possible_z:
                pointAlreadyThere = True

        if not pointAlreadyThere:
            r.append(pointToAdd[0])
            z.append(pointToAdd[1])

    if FirstEqualToLast:
        r.append(r[0])
        z.append(z[0])

    return r, z


def getLocInfo(locFile, removeSpaces=True):
    locFile = expandPath(locFile)

    folderWork = "/".join(locFile.split("/")[:-1]) + "/"
    nameRunid = locFile.split("/")[-1].split(".")[0]

    # if removeSpaces:
    #     embed()
    #     folderWork.replace(' ','\ ')
    return folderWork, nameRunid


def findFileByExtension(
    folder, extension, prefix=" ", fixSpaces=False, ForceFirst=False, agnostic_to_case=False, provide_full_path=False
    ):
    """
    Retrieves the file without folder and extension
    """

    folder = expandPath(folder, fixSpaces=fixSpaces)

    if os.path.exists(folder):
        allfiles = findExistingFiles(folder, extension, agnostic_to_case = agnostic_to_case)

        if len(allfiles) > 1:
            # print(allfiles)
            if not ForceFirst:
                raise Exception("More than one file with same extension in the folder!")
            else:
                allfiles = [allfiles[0]]

        if len(allfiles) == 1:
            fileReturn = allfiles[0].split(extension)[0].split(prefix)[-1]
        else:
            print(
                f"\t\t~ File with extension {extension} not found in {clipstr(folder)}, returning None"
            )
            fileReturn = None
    else:
        from mitim_tools.misc_tools.CONFIGread import read_verbose_level

        
        printMsg(
            f"\t\t\t~ Folder ...{folder[np.max([-40,-len(folder)]):]} does not exist, returning None",
            verbose=read_verbose_level(),
        )
        fileReturn = None

    if provide_full_path and fileReturn is not None:
        fileReturn = folder + fileReturn + extension

    return fileReturn

def findExistingFiles(folder, extension, agnostic_to_case=False):
    allfiles = []
    for file in os.listdir(folder):

        if not agnostic_to_case:
            if file.endswith(extension):
                allfiles.append(file)
        else:
            if file.lower().endswith(extension.lower()):
                allfiles.append(file)

    return allfiles

def writeOFs(resultsFile, dictOFs, dictErrors=None):
    """
    By PRF's convention, error here refers to the standard deviation
    """

    for iOF in dictOFs:
        if "error" not in dictOFs[iOF]:
            dictOFs[iOF]["error"] = 0.0

    typeF = "w"

    for cont, iOF in enumerate(dictOFs):
        writeresults(
            resultsFile, dictOFs[iOF]["value"], dictOFs[iOF]["error"], iOF, typeF=typeF
        )
        typeF = "a"


def generateDictionaries(InputsFile):
    dictDVs, dictOFs = OrderedDict(), OrderedDict()

    with open(InputsFile, "r") as f:
        numvar = int(f.readline().split()[0])

        for i in range(numvar):
            stt = f.readline().split()
            dic = {}
            dic["value"] = float(stt[0])
            dictDVs[stt[1]] = dic

        numobj = int(f.readline().split()[0])

        for i in range(numobj):
            stt = f.readline().split(":")
            dic = {}
            dic["value"] = "nan"
            dictOFs[stt[-1].split("\n")[0]] = dic

    return {"dictDVs": dictDVs, "dictOFs": dictOFs}


# From https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, extra=""):
    valid = {"y": True, "n": False, "e": None}
    prompt = " [y/n/e] (yes, no, exit)"

    while True:
        total = (extra,) + (question,) + (prompt,) + ("\u001b[0m",)
        print(*total)
        with promting_context():
            choice = sys.stdin.read(1)
        if choice.lower() in valid:
            print(f"\t\t>> Answer: {choice.lower()}")
            if valid[choice.lower()] is not None:
                print(
                    f'\t\t>> Proceeding sending "{valid[choice.lower()]}" flag to main program'
                )
                return valid[choice.lower()]
            else:
                raise Exception("[mitim] Exit request")
        else:
            print("Please respond with 'y' (yes) or 'n' (no)\n")


class promting_context(object):
    def __init__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)

    def __enter__(self):
        tty.setraw(sys.stdin.fileno())

    def __exit__(self, *args):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def findValue(
    FilePath,
    ParamToFind,
    SplittingChar,
    isitArray=False,
    raiseException=True,
    findOnlyLast=False,
    avoidIfStartsWith=None,
):
    """
    FilePath:       File to read
    ParamtoFind:    Parameter that you're looking for
    SplittingChar:  Character that exists between name and value (e.g. "=")

    upper, lower agnostic
    """

    f = open(FilePath, "r")

    for line in f:
        # If line contain that variable name, let's grab it!
        if ParamToFind.upper() in line.upper():
            if avoidIfStartsWith is not None:
                if line.strip()[0] == avoidIfStartsWith:
                    continue

            try:
                # Assume first that it's just a float
                val = float(line.split(SplittingChar)[1].split()[0])
            except:
                if not isitArray:
                    # If not float, and no array, maybe it's just a string
                    try:
                        val = str(line.split(SplittingChar)[1].split()[0])
                    # If not float, no array, and no string, keep looking, something is wrong
                    except:
                        continue
                else:
                    # If it is array
                    val = str(line.split(SplittingChar)[1])

            # Have I really found it? (checking that it's not commented)
            if line.split(SplittingChar)[0].upper().split()[0] == ParamToFind.upper():
                val_final = val
                if not findOnlyLast:
                    break

    f.close()

    try:
        return val
    except:
        if raiseException:
            raise Exception(f"{ParamToFind} Value not found in namelist {FilePath}")
        else:
            # printMsg('{} Value not found in namelist {}, returning None'.format(ParamToFind,FilePath),verbose=read_verbose_level()) #,typeMsg='w')
            return None


def cleanArray(liststr):
    return np.array(
        [
            float(i.strip(",").strip("\t").strip("\n"))
            for i in liststr.split("!")[0].split()
        ]
    )


def changeValue(
    FilePath,
    ParamToChange,
    Value,
    InfoCommand,
    SplittingChar,
    TryAgain=True,
    CommentChar="!",
    MaintainComments=False,
    NoneMeansRemove=True,
):
    """
    Inputs:
            - SplittingChar:  Character that exists between name and value (e.g. "=")
    Notes:
            - There has to be spaces in the namelist 'e.g. nshot = 10000, not nshot=1000'
            - If Value is None, remove from namelist
    """

    passlast = False
    if InfoCommand is not None and len(InfoCommand) == 0:
        InfoCommand, passlast = [""], True

    if CommentChar is not None:
        AddTextToChangedParam = " " + CommentChar + " Changed by MITIM"
        separator_space = "  "
    else:
        AddTextToChangedParam = ""
        separator_space = ""

    f1, f2 = open(FilePath, "r"), open(FilePath + "_new", "w")

    FoundAtLeastOnce = False
    for line in f1:
        lineSep = line.upper().split()
        if len(lineSep) > 0:
            # Allowing for possibility that the '=' is not separated by spaces
            if SplittingChar in lineSep[0]:
                lineCheck = lineSep[0].upper().split(SplittingChar)[0]
            else:
                lineCheck = lineSep[0].upper()

            varFound = ParamToChange.upper() == lineCheck

        else:
            varFound = False

        # ~~~~~~ Modification if it has been found
        if varFound:
            # Cases that the TRANSP namelist may be picky about (in terms of spaces and comments)
            if ParamToChange.lower() == "nshot" or "_pserve" in ParamToChange.lower():
                if Value is None and NoneMeansRemove:
                    line = ""
                else:
                    line = f"{line.split(SplittingChar)[0]}{SplittingChar}{Value}\n"

            # General cases
            else:
                # Do I keep original comments?
                possibleComment = ""
                if CommentChar is not None and MaintainComments:
                    try:
                        if line.split(SplittingChar)[1].split()[1] == CommentChar:
                            try:
                                possibleComment = " ".join(
                                    line.split(SplittingChar)[1].split()[1:]
                                ).split(AddTextToChangedParam)[0]
                            except:
                                possibleComment = " ".join(
                                    line.split(SplittingChar)[1].split()[1:]
                                )
                    except:
                        pass
                    AddTextToChangedParam = ""

                if Value is None and NoneMeansRemove:
                    line = ""
                else:
                    line = "{0}{5}{1}{5}{2}{5}{3}{5}{4}\n".format(
                        line.split(SplittingChar)[0],
                        SplittingChar,
                        Value,
                        possibleComment,
                        AddTextToChangedParam,
                        separator_space,
                    )

            FoundAtLeastOnce = True

        f2.write(line)

    f1.close()
    f2.close()

    os.rename(FilePath + "_new", FilePath)

    # If not found at least once, then write it, but make sure it is after the updates flag
    if not FoundAtLeastOnce and Value is not None:
        if CommentChar is not None:
            extt = f"{CommentChar} Added by MITIM"
        else:
            extt = ""
        lines_add = "{0}{4}{1}{4}{2}{4}{3}\n".format(
            ParamToChange, SplittingChar, Value, extt, separator_space
        )

        with open(FilePath, "r") as f:
            lines = f.readlines()
        done, lines_n = False, ""
        for i in lines:
            if "~update" in i and not done:
                lines_n += lines_add
                done = True
            lines_n += i
        if not done:
            lines_n += lines_add
        with open(FilePath, "w") as f:
            f.write(lines_n)
    # ------------------

    if InfoCommand is None:
        try:
            try:
                print(
                    f'\t- Namelist parameter "{ParamToChange:s}" changed to {Value:.4f}'
                )
            except:
                print(f'\t- Namelist parameter "{ParamToChange:s}" changed to {Value}')
        except:
            if TryAgain:
                changeValue(
                    FilePath,
                    ParamToChange.upper(),
                    Value,
                    None,
                    SplittingChar,
                    TryAgain=False,
                    CommentChar=CommentChar,
                )
            else:
                pass

    else:
        try:
            try:
                InfoCommand.append(
                    f'\t- Namelist parameter "{ParamToChange:s}" changed to {Value:.4f}'
                )
            except:
                InfoCommand.append(
                    f'\t- Namelist parameter "{ParamToChange:s}" changed to {Value}'
                )
        except:
            if TryAgain:
                InfoCommand = changeValue(
                    FilePath,
                    ParamToChange.upper(),
                    Value,
                    InfoCommand,
                    SplittingChar,
                    TryAgain=False,
                    CommentChar=CommentChar,
                )
            else:
                pass
        if passlast:
            InfoCommand = InfoCommand[1:]

        return InfoCommand


def writeQuickParams(folder, num=1):
    txt = [
        "                                          1 variables",
        "                      1.000000000000000e+00 factor_ped_degr",
        "                                          1 functions",
        "                                          1 ASV_1:Q_plasma",
        "                                          0 derivative_variables",
        "                                          0 analysis_components",
        "                                          1 eval_id",
    ]

    with open(folder + f"/params.in.{num}", "w") as f:
        f.write("\n".join(txt))


def readValueinFile(file, variable, positionReturn=0):
    f = open(file, "r")

    for line in f:
        if line.split()[1] == variable:
            varAux = line.split()[positionReturn]
            try:
                var = float(varAux)
            except:
                var = varAux

        # For the case of array (e.g. fast species)
        elif line.split()[-1] == variable:
            var = line.split()[positionReturn]
            for i in range(len(line.split()) - positionReturn - 2):
                var = var + line.split()[positionReturn + i + 1]

    f.close()

    return var


def writeresults(resespec, final_results, final_errors, vartag, typeF="a"):
    with open(resespec, typeF) as outfile:
        if isnum(final_results):
            outfile.write(
                f"{vartag:15s}: {final_results:1.15e},   {final_errors:1.15e}\n"
            )
        else:
            outfile.write(f"{vartag:15s}: {final_results},   {final_errors}\n")


def readresults(fileresults):
    with open(fileresults, "r") as outfile:
        aux = outfile.readlines()

    y, yE = [], []
    for i in aux:
        label = i.split(":")[0]
        y0 = float(i.split(":")[1].split(",")[0])
        yE0 = float(i.split(":")[1].split(",")[1])

        y.append(round(y0, 16))
        yE.append(round(yE0, 16))

    y = np.array(y)
    yE = np.array(yE)

    return y, yE


def writeparams(x, fileparams, inputs, outputs, numEval):
    with open(fileparams, "w") as outfile:
        outfile.write(
            f"                                          {len(inputs)} variables\n"
        )
        for cont, i in enumerate(inputs):
            outfile.write(f"                      {x[cont]:1.15e} {i}\n")
        outfile.write(
            f"                                          {len(outputs)} functions\n"
        )
        for cont, i in enumerate(outputs):
            outfile.write(
                f"                                          1 ASV_{cont + 1}:{i}\n"
            )
        outfile.write(
            "                                          0 derivative_variables\n"
        )
        outfile.write(
            "                                          0 analysis_components\n"
        )
        outfile.write(f"                                       {numEval:4d} eval_id\n")


class CaseInsensitiveDict(OrderedDict):
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(key.lower())


def getLinesNamelist(filename, commentCommand, separator, boolLower=None):
    with open(filename, "r") as f:
        allLines = f.readlines()
    allLines_clean = []
    for i in range(len(allLines)):
        cl = (
            allLines[i]
            .split(commentCommand)[0]
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
        )
        if len(cl) > 1 and cl[0] != commentCommand:
            par = cl.split(separator)[0].lower()
            if par[0] != "&":
                if boolLower is not None:
                    if boolLower(par):
                        allLines_clean.append(cl.lower())
                    else:
                        allLines_clean.append(cl)
                else:
                    allLines_clean.append(cl.lower())

    return allLines_clean


def getDictionaryNamelist(allLines_clean, separator, caseInsensitive=True):
    if caseInsensitive:
        dictParams = CaseInsensitiveDict()
    else:
        dictParams = OrderedDict()

    for i in allLines_clean:
        splitted = i.split(separator)
        if len(splitted) < 2:
            continue

        tag = splitted[0]
        val = splitted[1]

        isNum, isFloat = isNumber(val)

        if len(val) == 0:
            val = ""
        elif isFalse(val):
            val = False
        elif isTrue(val):
            val = True
        elif val == "none" or val == "None":
            val = None
        elif val[0] == "[" or val[0] == "(":
            valis = val[1:-1].split(",")
            if len(valis[0] + "a") > 1:
                val = []
                for j in valis:
                    try:
                        val.append(float(j))
                    except:
                        if j == "none" or j == "None":
                            val.append(None)
                        elif isTrue(j):
                            val.append(True)
                        elif isFalse(j):
                            val.append(False)
                        else:
                            val.append(j)
            else:
                val = []
            val = np.array(val)
        elif isNum:
            if isFloat:
                val = float(val)
            else:
                val = int(val)

        dictParams[tag] = val

    return dictParams


def isTrue(val):
    return (
        val.lower() == "T"
        or val.lower() == "t"
        or val.lower() == "true"
        or val.lower() == ".T."
        or val.lower() == ".t."
        or val.lower() == ".True."
        or val.lower() == ".true."
    )


def isFalse(val):
    return (
        val.lower() == "F"
        or val.lower() == "f"
        or val.lower() == "false"
        or val.lower() == ".F."
        or val.lower() == ".f."
        or val.lower() == ".False."
        or val.lower() == ".false."
    )


def boolMITIMnoLower(par):
    return (
        par
        not in [
            "opt_dvs",
            "opt_ofs",
            "namelistbaselinefile",
            "userotationmach",
            "opt_cvs",
            "basegeom",
            "opt_fileinitialization",
            "baseuffolder",
            "basecdffile",
            "basenmfolder",
            "runidletter",
            "shotnumber",
            "scriptsfolder",
            "excelprofile",
            "completedfirst",
            "prescribedevolution",
            "prescribedevolutiontsc",
            "imp_profs",
            "fast_anom",
            "min_dens",
            "lut_loc",
            "gfile_loc",
            "specialqprofile",
            "mmx_loc",
            "nn_loc",
            "opt_normalizationfolder",
            "specialdensityufile",
            "specialqufile",
            "conv_vars",
        ]
        and ".name" not in par
        and ".email" not in par
        and "cmd" not in par
    )


def false(par):
    return False


def generateMITIMNamelist(
    orig, commentCommand="#", separator="=", WriteNew=None, caseInsensitive=True
):
    orig = expandPath(orig)

    # Read values from namelist

    if caseInsensitive:
        boolLower = boolMITIMnoLower
    else:
        boolLower = false

    allLines_clean = getLinesNamelist(
        orig, commentCommand, separator, boolLower=boolLower
    )
    dictParams = getDictionaryNamelist(
        allLines_clean, separator, caseInsensitive=caseInsensitive
    )

    if WriteNew is not None:
        with open(WriteNew, "w") as f:
            for i in dictParams:
                f.write(i + "=" + str(dictParams[i]) + "\n")

    return dictParams


def obtainGeneralParams(inputFile, resultsFile):
    FolderEvaluation = "/".join(os.path.realpath(inputFile).split("/")[:-1]) + "/"

    if FolderEvaluation[0] != "/":
        FolderEvaluation = "/home/" + FolderEvaluation

    # In case
    inputFile, resultsFile = inputFile.split("/")[-1], resultsFile.split("/")[-1]

    numDakota = inputFile.split(".")[2]

    inputFilePath, outputFilePath = (
        FolderEvaluation + inputFile,
        FolderEvaluation + resultsFile,
    )

    return FolderEvaluation, numDakota, inputFilePath, outputFilePath


def isNumber(val):
    try:
        aux = float(val)
        if "." in val or "e" in val or "E" in val:
            return True, True
        else:
            return True, False
    except:
        return False, False


def ArrayToString(ll):
    nn = []
    for i in ll:
        nn.append(f"{i:.2f}")

    return ",".join(nn)


def expandPath(txt, fixSpaces=False, ensurePathValid=False):
    while txt[-2:] == "//":
        txt = txt[:-1]

    if txt[0] == ".":
        if (len(txt) == 1) or (len(txt) == 2 and txt[-1] == "/"):
            txt = os.path.realpath(txt) + "/"
        else:
            txt = os.path.realpath(txt)

    if ensurePathValid and (txt[0] not in ["~", "/"]):
        txt = os.path.realpath("./" + txt + "/")

    pathn = os.path.expanduser(os.path.expandvars(txt))

    if fixSpaces:
        pathn = pathn.replace(" ", "\ ")
        pathn = pathn.replace("(", "\(")
        pathn = pathn.replace(")", "\)")

    return pathn


def cleanPath(txt, isItFile=False):
    txt = expandPath(txt, ensurePathValid=True).split("/")
    aux = []
    for i in txt:
        if len(i) > 0:
            aux.append(i)

    txt = "/".join(aux) + "/"

    if txt[0] != "/":
        txt = "/" + txt

    if not isItFile and txt[-1] != "/":
        txt = txt + "/"
    if isItFile and txt[-1] == "/":
        txt = txt[:-1]

    return txt


def reducePathLevel(txt, level=1, isItFile=False):
    txt = cleanPath(txt, isItFile=isItFile)
    if isItFile:
        level += -1

    sep1 = "/".join(txt.split("/")[: -(level + 1)]) + "/"
    sep2 = txt.split("/".join(txt.split("/")[: -(level + 1)]))[-1][1:]

    if not isItFile and sep2[-1] == "/":
        sep2 = sep2[:-1]

    return sep1, sep2


def read_pfile(filepath="./JWH_pedestal_profiles.p", plot=False):
    """
    Method to parse p-files for pedestal modeling.
    sciortino, 2020
    """
    with open(filepath, "r") as f:
        contents = f.readlines()

    # find end of header
    for ii, line in enumerate(contents):
        if line.startswith("***"):
            break
    # '****BEGIN P FILE****\n'
    ii += 1

    num = int(contents[ii].split()[0])
    psin = np.zeros(num)
    ne = np.zeros(num)
    dnedpsi = np.zeros(num)
    Te = np.zeros(num)
    dTedpsi = np.zeros(num)
    ni = np.zeros(num)
    dnidpsi = np.zeros(num)
    Ti = np.zeros(num)
    dTidpsi = np.zeros(num)

    # skip new header
    ii += 1
    for jj in np.arange(num):
        elems = contents[ii].split()
        psin[jj] = float(elems[0])
        ne[jj] = float(elems[1])
        dnedpsi[jj] = float(elems[2])
        ii += 1

    # skip new header
    ii += 1
    for kk in np.arange(num):
        elems = contents[ii].split()
        psin[kk] = float(elems[0])
        Te[kk] = float(elems[1])
        dTedpsi[kk] = float(elems[2])
        ii += 1

    # skip new header
    ii += 1
    for mm in np.arange(num):
        elems = contents[ii].split()
        psin[mm] = float(elems[0])
        ni[mm] = float(elems[1])
        dnidpsi[mm] = float(elems[2])
        ii += 1

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        ax = ax.flatten()
        ax[0].plot(np.sqrt(psin), ne, label="p-file")
        ax[1].plot(np.sqrt(psin), Te)
        ax[2].plot(np.sqrt(psin), ni)
        ax[3].plot(np.sqrt(psin), Ti)
        ax[0].set_ylabel(r"$n_e$ [$10^{20}$ $m^{-3}$]")
        ax[1].set_ylabel(r"$T_e$ [$keV$]")
        ax[2].set_ylabel(r"$n_i$ [$10^{20}$ $m^{-3}$]")
        ax[3].set_ylabel(r"$T_i$ [$keV$]")
        ax[2].set_xlabel(r"$\rho_\psi$")
        ax[3].set_xlabel(r"$\rho_\psi$")
        ax[0].legend().set_draggable(True)
    else:
        fig = None
        ax = None

    return psin, ne, dnedpsi, Te, dTedpsi, ni, dnidpsi, Ti, dTidpsi, fig, ax

'''
Log file utilities
--------------------------------
chatGPT 4o as of 08/18/2024
'''

@contextlib.contextmanager
def conditional_log_to_file(log_file=None, msg=None):
    if log_file is not None:
        with log_to_file(log_file, msg) as logger:
            yield logger
    else:
        yield None  # Simply pass through without logging

def strip_ansi_codes(text):
    if not isinstance(text, (str, bytes)):
        text = str(text)  # Convert non-string types to string
    # Strip ANSI escape codes
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)

class log_to_file:
    def __init__(self, log_file, msg=None):
        if msg is not None:
            print(msg)
        self.log_file = log_file
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        self.log = open(self.log_file, 'a')
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)

        # Redirect stdout and stderr to the log file.
        os.dup2(self.log.fileno(), self.stdout_fd)
        os.dup2(self.log.fileno(), self.stderr_fd)

        # Redirect Python's sys.stdout and sys.stderr to the log file.
        sys.stdout = self
        sys.stderr = self

        # Redirect warnings to the log file
        logging_handler = lambda message, category, filename, lineno, file=None, line=None: \
            self.log.write(f"{category.__name__}: {strip_ansi_codes(message)}\n")
        warnings.showwarning = logging_handler

    def write(self, message):
        # Remove ANSI codes from the message before writing to the log
        clean_message = strip_ansi_codes(message)
        self.log.write(clean_message)
        self.log.flush()  # Ensure each write is immediately flushed

    def flush(self):
        # Ensure sys.stdout and sys.stderr are flushed
        self.log.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        # Flush and restore Python's sys.stdout and sys.stderr
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

        # Restore the original file descriptors for stdout and stderr
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)

        # Close the duplicate file descriptors
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)

        # Close the log file
        self.log.close()

        # Restore the original warnings behavior
        warnings.showwarning = warnings._showwarning_orig

"""
This HDF5 tool was originally designed by A.J. Creely, but modifications 
by P. Rodriguez-Fernandez have been made.

Example use:
    plt.ion(); fig, ax = plt.subplots()
    ax.plot([1.0,2.0],[0.0,1.0],'-o')

    axesToHDF5({'a':ax},filename='dataset1',check=True)
"""

class hdf5figurefile(object):
    def __init__(self, filename):
        self.fig = h5py.File(filename + ".hdf5", "w")

    def makeHDF5group(
        self,
        plotGroup,
        numTraces=1,
        xlabel=None,
        ylabel=None,
        title=None,
        xdata=None,
        ydata=None,
        zdata=None,
        ldata=None,
        udata=None,
        vdata=None,
        color=None,
        linestyle=None,
        marker=None,
        displayname=None,
        linewidth=None,
    ):
        # def makeHDF5file(self,*args):

        self.plotGroup = self.fig.create_group(plotGroup)

        self.numTraces = numTraces

        self.plotGroup.attrs["XLabel1"] = xlabel
        self.plotGroup.attrs["YLabel1"] = ylabel
        self.plotGroup.attrs["Title1"] = title

        self.xdata = xdata
        self.ydata = ydata
        self.zdata = zdata
        self.ldata = ldata
        self.udata = udata
        self.vdata = vdata
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.displayname = displayname
        self.linewidth = linewidth

        for i in range(0, self.numTraces):
            self.makeDataSet(i)

        return

    def makeDataSet(self, i):
        dataTemp = self.plotGroup.create_group("Data" + str(i))

        # This might be a problem... Uneven array size.
        # print(self.xdata)
        # print(self.xdata[0][:])##WORKING HERE##

        xTemp = dataTemp.create_dataset("XData", (self.xdata[i].size,), dtype="f")
        yTemp = dataTemp.create_dataset("YData", (self.ydata[i].size,), dtype="f")

        xTemp[:] = self.xdata[i][:]
        yTemp[:] = self.ydata[i][:]

        if self.zdata is not None:
            zTemp = dataTemp.create_dataset("ZData", (self.zdata[i].size,), dtype="f")
            zTemp[:] = self.zdata[i][:]

        if self.ldata is not None:
            lTemp = dataTemp.create_dataset("LData", (self.ldata[i].size,), dtype="f")
            lTemp[:] = self.ldata[i][:]

        if self.udata is not None:
            uTemp = dataTemp.create_dataset("UData", (self.udata[i].size,), dtype="f")
            uTemp[:] = self.udata[i][:]

        if self.vdata is not None:
            vTemp = dataTemp.create_dataset("VData", (self.vdata[i].size,), dtype="f")
            vTemp[:] = self.vdata[i][:]

        if self.color is not None:
            dataTemp.attrs["Color [r g b]"] = self.color[i]
        if self.linestyle is not None:
            dataTemp.attrs["LineStyle"] = self.linestyle[i]
        if self.marker is not None:
            dataTemp.attrs["Marker"] = self.marker[i]
        if self.displayname is not None:
            dataTemp.attrs["DisplayName"] = self.displayname[i]
        if self.linewidth is not None:
            dataTemp.attrs["LineWidth"] = self.linewidth[i]

    def subplotToHDF5(self, ax, name="a"):
        xlabel = [ax.get_xlabel()]
        ylabel = [ax.get_ylabel()]

        title = ax.get_title()
        numTraces = len(ax.get_lines())
        color, linestyle, linewidth, marker, xdata, ydata, displayname = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(numTraces):
            color.append(ax.get_lines()[i].get_color())
            linestyle.append(ax.get_lines()[i].get_linestyle())
            linewidth.append(ax.get_lines()[i].get_linewidth())
            marker.append(ax.get_lines()[i].get_marker())

            x = np.array(ax.get_lines()[i].get_data()[0])
            y = np.array(ax.get_lines()[i].get_data()[1])

            if len(x.shape) > 1:
                x = x[:, 0]

            xdata.append(x)
            ydata.append(y)

            displayname.append(ax.get_lines()[i].get_label())

        zdata, ldata, udata, vdata = None, None, None, None
        displayname = None

        self.makeHDF5group(
            name,
            numTraces,
            xlabel,
            ylabel,
            title,
            xdata,
            ydata,
            zdata,
            ldata,
            udata,
            vdata,
            color,
            linestyle,
            marker,
            displayname,
            linewidth,
        )


def axesToHDF5(axesarray_dict, filename="dataset1", check=True):
    h5file = hdf5figurefile(filename)

    for ikey in axesarray_dict:
        name = ikey
        ax = axesarray_dict[name]
        h5file.subplotToHDF5(ax, name=name)

    print(" --> Written " + filename)

    if check:
        # Check
        f = h5py.File(filename + ".hdf5", "r")
        for ikey in f.keys():
            print(np.array(f["a"]["Data0"]["XData"]))
