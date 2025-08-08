import os
import re
import shutil
import psutil
import copy
from typing import Callable
import dill as pickle_dill
import pandas as pd
from mitim_tools.misc_tools import GRAPHICStools
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import datetime
import socket
import random
import zipfile
import cProfile
import h5py
import subprocess
import json
import functools
import hashlib
from collections import OrderedDict
from pathlib import Path
import platform
import torch
try:
    from IPython import embed
except ImportError:
    pass

import urllib.request as urlREQ  # urllibR
import urllib.error as urlERR  # urllibE

from mitim_tools.misc_tools.LOGtools import printMsg as print

class speeder(object):
    def __init__(self, file='./profiler.prof'):
        self.file = Path(file).expanduser()

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

        print(f'Script took {createTimeTXT(self.timeDiff)}, profiler stats dumped to {self.file} (open with "python3 -m snakeviz {self.file}")')

class timer:
    '''
    Context manager to time a script or function execution.
    '''
    # ────────────────────────────────────────────────────────────────────
    def __init__(self,
                 name: str = "Script",                  # Name of the script for printing, visualization
                 print_at_entering: str | None = None,  # Prefix printed right before the timer starts
                 log_file: Path | None = None):         # File to log the timing information in JSON format
        self.name       = name
        self.print_at_entering = print_at_entering
        self.log_file   = log_file

    # ────────────────────────────────────────────────────────────────────
    def __enter__(self):
        # high-resolution timer + wall-clock stamp
        
        self.t0_wall    = time.perf_counter()
        self.t0         = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.print_at_entering:
            print(f'{self.print_at_entering}{self.t0}')
        return self

    # ────────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc, tb):
        self._finish() 
        return False            # propagate any exception

    # ────────────────────────────────────────────────────────────────────
    def _finish(self):
        
        dt = time.perf_counter() - self.t0_wall
        t1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f'\t\t* {self.name} took {createTimeTXT(dt)}')

        if self.log_file:
            record = {
                "script"      : self.name,
                "t_start"     : self.t0,
                "ts_end"      : t1,
                "duration_s"  : dt,
            }
            with Path(self.log_file).open("a", buffering=1) as f:
                f.write(json.dumps(record) + "\n")

# Decorator to time functions

def mitim_timer(
        name: str | None = None,
        print_at_entering: str | None = None,
        log_file: str | Path | Callable[[object], str | Path] | None = None
    ):
    """
    Decorator that times a function / method and optionally appends one JSON
    line to *log_file* after the call finishes.

    Parameters
    ----------
    name : str | None
        Human-readable beat name.  If None, defaults to the wrapped function's __name__.
    print_at_entering : str
        Prefix printed right before the timer starts
    log_file : str | Path | callable(self) -> str | Path | None
        • str / Path  → literal path written every time the beat finishes  
        • callable    → called **at call time** with the bound instance
                        (`self`) and must return the path to use  
        • None        → no file is written, only console timing is printed

    Notes
    -----
    *When* the wrapper runs it has access to the bound instance (`self`), so
    callable argument values let you access self variables.
    """

    def decorator_timer(func):
        script_name = name or func.__name__

        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            # -------------------- resolve name --------------------------
            if callable(script_name):
                # assume first positional arg is `self` for bound methods
                instance = args[0] if args else None
                chosen_script_name = script_name(instance)
            else:
                chosen_script_name = script_name
            # ---------------------------------------------------------------
            # -------------------- resolve log_file --------------------------
            if callable(log_file):
                # assume first positional arg is `self` for bound methods
                instance = args[0] if args else None
                chosen_log_file = log_file(instance)
            else:
                chosen_log_file = log_file
            # ---------------------------------------------------------------

            # Your original context-manager timer class:
            with timer(chosen_script_name,
                       print_at_entering=print_at_entering,
                       log_file=chosen_log_file):
                return func(*args, **kwargs)

        return wrapper_timer

    return decorator_timer

# ---------------------------------------------------------------------------
def plot_timings(jsonl_path, axs = None, unit: str = "min", color = "b", label= '', log=False):
    """
    Plot cumulative durations from a .jsonl timing ledger written by @mitim_timer,
    with vertical lines when the beat number changes.

    Parameters
    ----------
    jsonl_path : str | Path
        File with one JSON record per line.
    unit : {"s", "min", "h"}
        Unit for the y-axis.
    """
    multiplier = {"s": 1, "min": 1 / 60, "h": 1 / 3600}[unit]

    scripts, script_time, cumulative, beat_nums, script_restarts = [], [], [], [], []
    running = 0.0
    beat_pat = re.compile(r"Beat\s*#\s*(\d+)")

    # ── read the file ───────────────────────────────────────────────────────
    with Path(jsonl_path).expanduser().open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            
            if rec["script"] not in scripts:
            
                scripts.append(rec["script"])
                script_time.append(rec["duration_s"] * multiplier)
                running += rec["duration_s"]* multiplier
                cumulative.append(running)

                m = beat_pat.search(rec["script"])
                beat_nums.append(int(m.group(1)) if m else None)
                
                script_restarts.append(0.0)
                
            else:
                # If the script is already in the list, it means it was restarted
                idx = scripts.index(rec["script"])
                script_restarts[idx] += rec["duration_s"] * multiplier
                
                cumulative[-1] += script_restarts[idx] 
                running += script_restarts[idx] 
                

    if not scripts:
        raise ValueError(f"No records found in {jsonl_path}")

    beat_nums = [0] + beat_nums  # Start with zero beat
    scripts = ['ini'] + scripts  # Add initial beat
    script_time = [0.0] + script_time  # Start with zero time
    cumulative = [0.0] + cumulative  # Start with zero time
    script_restarts = [0.0] + script_restarts  # Start with zero restarts

    # ── plot ────────────────────────────────────────────────────────────────
    x = list(range(len(scripts)))
    
    if axs is None:
        plt.ion()
        fig = plt.figure()
        axs = fig.subplot_mosaic("""
                                A
                                B
                                """)
    
    try:
        axs = [ax for ax in axs.values()]
    except:
        pass

    ax = axs[0]
    ax.plot(x, cumulative, "-s", markersize=8, color=color, label=label)
    
    # Add restarts as vertical lines
    for i in range(len(script_restarts)):
        if script_restarts[i] > 0:
            ax.plot(
                [x[i],x[i]],
                [cumulative[i],cumulative[i]-script_restarts[i]],
                "-.o", markersize=5, color=color)
    
    
    for i in range(1, len(beat_nums)):
        if beat_nums[i] != beat_nums[i - 1]:
            ax.axvline(i - 0.5, color='k',linestyle="-.")

    #ax.set_xlim(left=0)
    ax.set_ylabel(f"Cumulative time ({unit})"); #ax.set_ylim(bottom=0)
    ax.set_xticks(x, scripts, rotation=10, ha="right", fontsize=8)
    GRAPHICStools.addDenseAxis(ax)
    ax.legend(loc='upper left', fontsize=8)


    ax = axs[1]
    for i in range(len(scripts)-1):
        ax.plot([x[i], x[i+1]], [0, script_time[i+1]], "-s", markersize=8, color=color)

    # Add restarts as vertical lines
    for i in range(len(script_restarts)-1):
        if script_restarts[i] > 0:
            ax.plot(
                [x[i+1],x[i+1]],
                [script_time[i+1],script_time[i+1]+script_restarts[i+1]],
                "-.o", markersize=5, color=color)

    for i in range(1, len(beat_nums)):
        if beat_nums[i] != beat_nums[i - 1]:
            ax.axvline(i - 0.5, color='k',linestyle="-.")

    #ax.set_xlim(left=0)
    ax.set_ylabel(f"Time ({unit})"); #ax.set_ylim(bottom=0)
    ax.set_xticks(x, scripts, rotation=10, ha="right", fontsize=8)
    GRAPHICStools.addDenseAxis(ax)
    if log:
        ax.set_yscale('log')
    
    return x, scripts


# ------------------------------------

# Decorator to hook methods before and after execution
def hook_method(before=None, after=None):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if before:
                before(self)
            result = func(self, *args, **kwargs)
            if after:
                after(self)
            return result
        return wrapper
    return decorator

def clipstr(txt, chars=40):
    if not isinstance(txt, str):
        txt = f"{txt}"
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

def get_git_info(repo_path):
    
    repopath = f"{repo_path.expanduser().resolve()}" if isinstance(repo_path, Path) else expandPath(repo_path)

    # Get branch
    result = subprocess.run(['git', '-C', repopath, 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        branch = result.stdout.strip()
    else:
        branch = None

    # Get hash
    result = subprocess.run(['git', '-C', repopath, 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

    fpath = Path(file).expanduser()
    ncfile = netCDF4.Dataset(fpath, mode="w", format="NETCDF4_CLASSIC")

    x = ncfile.createDimension("xdim", zvals.shape[1])

    for i, name in enumerate(names):
        value = ncfile.createVariable(name, "f4", ("xdim",))
        value[:] = zvals[i, :]

    ncfile.close()

def printPoints(x, numtabs=2):
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
    ziptargetpath = Path(FolderToZip).expanduser()
    zipdir = Path(locationZipped).expanduser()
    f1 = moveRecursive(
        check=1,
        commonprefix=NameZippedFile + "_",
        commonsuffix=".zip",
        eliminateAfter=5,
        rootFolder=zipdir
    )
    zipFolder(ziptargetpath, ZippedFile=f1)
    print(f" --> Most current {ziptargetpath.resolve()} folder zipped to {f1}")


def zipFolder(FolderToZip, ZippedFile="Contents.zip"):
    zpath = Path(FolderToZip).expanduser()
    zipitems = zpath.glob('**/*')
    with zipfile.ZipFile(ZippedFile, "w", zipfile.ZIP_DEFLATED) as zipf:
        for itempath in zipitems:
            if itempath.is_file():
                zipf.write(itempath)


def moveRecursive(check=1, commonprefix="Contents_", commonsuffix=".zip", eliminateAfter=5, rootFolder="~"):
    '''
    Shifts all existing file numbers up by one, keeping only a limited number due to memory requirements
    '''

    root_current = Path(rootFolder).expanduser()
    file_current = root_current / f"{commonprefix}{check}{commonsuffix}"

    if file_current.exists():
        if check >= eliminateAfter:
            file_current.unlink(missing_ok=True)
        else:
            file_next = root_current / f"{commonprefix}{check + 1}{commonsuffix}"
            if file_next.exists():
                moveRecursive(
                    check=check + 1,
                    commonprefix=commonprefix,
                    commonsuffix=commonsuffix,
                    eliminateAfter=eliminateAfter,
                    rootFolder=root_current,
                )
            file_current.replace(file_next)

    return file_current #f"{file_current}"

def calculate_sizes_obj_recursive(obj, N=5, parent_name="", recursion = 5):
    '''
    Calculate the size of the top N attributes of an object and recursively calculate the sizes of the attributes of the top item. 
    '''

    from pympler import asizeof

    sizes = {}
    prefix = f"{parent_name}." if parent_name else ""

    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, np.ndarray, tuple)):
        items = enumerate(obj)
    elif isinstance(obj, str):
        return
    else:
        try:
            items = vars(obj).items()
        except:
            print('Type not recognized, probably out of depth:')
            print(obj)
            return

    # Collect the size of each item in the object
    for attr_name, attr_value in items:
        sizes[attr_name] = (asizeof.asizeof(attr_value) * 1E-6, type(attr_value).__name__)

    # Sort the items by size (from high to low)
    sorted_sizes = dict(sorted(sizes.items(), key=lambda item: item[1][0], reverse=True))

    # Determine the maximum length of the attribute names for alignment
    max_attr_name_length = max(len(str(attr_name)) for attr_name in sorted_sizes)

    # Print the sizes of the top N items
    print(f'\nSize of {N} largest attributes of {type(obj).__name__}:')
    for attr_name, (size, attr_type) in list(sorted_sizes.items())[:N]:
        full_attr_name = f"{prefix}{attr_name}"
        print(f'\t{full_attr_name.ljust(max_attr_name_length + len(prefix))}: {size:>10.6f} MB ({attr_type})')

    # Sum the sizes of the remaining attributes
    remaining_size = sum(size for size, _ in list(sorted_sizes.values())[N:])
    
    # Print the total size of the remaining attributes if any
    if remaining_size > 0:
        print(f'\t{prefix}Remaining attributes combined size: {remaining_size:.6f} MB')

    # Recursively calculate the sizes of the attributes of the top item
    if recursion > 0:

        if isinstance(obj, dict):
            parent_name = list(sorted_sizes.keys())[0]
            child_obj = obj[list(sorted_sizes.keys())[0]]
        elif isinstance(obj, (list, np.ndarray, tuple)):
            parent_name = f"{prefix}{list(sorted_sizes.keys())[0]}"
            child_obj = obj[list(sorted_sizes.keys())[0]]
        else:
            parent_name = list(sorted_sizes.items())[0][0]
            child_obj = getattr(obj,parent_name)
        calculate_sizes_obj_recursive(child_obj, N, recursion = recursion - 1, parent_name=parent_name)

def calculate_size_pickle(file):
    '''
    Calculate the size of the object stored in a pickle file.
    '''

    import pickle
    
    ifile = Path(file).expanduser()
    with open(ifile, 'rb') as f:
        obj = pickle.load(f)
    calculate_sizes_obj_recursive(obj, recursion = 20)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MITIM optimization namelist
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_mitim_nml(json_file):
    jpath = Path(json_file).expanduser()
    with open(jpath, 'r') as file:
        optimization_options = json.load(file)

    return optimization_options

def curate_mitim_nml(optimization_options, stopping_criteria_default = None):

    # Optimization criterion
    if optimization_options['convergence_options']['stopping_criteria'] is None:
        optimization_options['convergence_options']['stopping_criteria'] = stopping_criteria_default

    # Add optimization print
    if optimization_options is not None:
        unprint_fun = copy.deepcopy(optimization_options['convergence_options']['stopping_criteria'])
        def opt_crit(*args,**kwargs):
            print('\n')
            print('--------------------------------------------------')
            print('Convergence criteria')
            print('--------------------------------------------------')
            v = unprint_fun(*args, **kwargs)
            print('--------------------------------------------------\n')
            return v
        optimization_options['convergence_options']['stopping_criteria'] = opt_crit

    # Check if the optimization options are in the namelist
    from mitim_tools import __mitimroot__
    Optim_potential = read_mitim_nml(__mitimroot__ / "templates" / "main.namelist.json")
    for ikey in optimization_options:
        if ikey not in Optim_potential:
            print(f"\t- Option {ikey} is an unexpected variable, prone to errors", typeMsg="q")

    return optimization_options

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getpythonversion():
    return [ int(i.split("\n")[0].split("+")[0]) for i in sys.version.split()[0].split(".") ]

def zipFiles(files, outputFolder, name="info"):
    odir = Path(outputFolder).expanduser()
    opath = odir / name
    if not opath.is_dir():
        opath.mkdir(parents=True)
    for i in files:
        shutil.copy2(expandPath(i), opath)
    shutil.make_archive(f"{opath}", "zip", odir)  # Apparently better to keep string as first argument
    shutil_rmtree(opath)


def unzipFiles(file, destinyFolder, clear=True):
    zpath = Path(file).expanduser()
    odir = Path(destinyFolder).expanduser()
    shutil.unpack_archive(zpath, odir)
    if clear:
        zpath.unlink(missing_ok=True)


def getProfiles_ExcelColumns(file, fromColumn=0, fromRow=4, rhoNorm=None, sheet_name=0):

    ifile = Path(file).expanduser()
    df = pd.read_excel(ifile, sheet_name=sheet_name)

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


def getVar_ExcelColumn(df, columnName, fromRow=4):
    var0 = np.array(df[columnName].values)[fromRow:]
    var = []
    for i in var0:
        if not np.isnan(float(i)):
            var.append(float(i))
    return np.array(var)


def writeProfiles_ExcelColumns(file, rho, Te, q, ne, Ti=None, fromColumn=0, fromRow=4):

    ofile = Path(file).expanduser()
    ofile.unlink(missing_ok=True)

    if Ti is None:
        Ti = Te

    dictExcel = OrderedDict()
    dictExcel["rho"] = rho
    dictExcel["Te"] = Te
    dictExcel["Ti"] = Ti
    dictExcel["q"] = q
    dictExcel["ne"] = ne

    writeExcel_fromDict(dictExcel, ofile, fromColumn=fromColumn, fromRow=fromRow)


def writeExcel_fromDict(dictExcel, file, fromColumn=0, fromRow=4):
    ofile = Path(file).expanduser()
    df = pd.DataFrame(dictExcel)
    writer = pd.ExcelWriter(ofile, engine="xlsxwriter")
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

    df = pd.DataFrame(data, index=[row_name], columns=columns)

    return df


def addRowToExcel(file, dataSet_dict, row_name="row 1", repeatIfIndexExist=True):

    fpath = Path(file).expanduser()
    df = createExcelRow(dataSet_dict, row_name=row_name)

    if fpath.is_file():
        df_orig = pd.read_excel(fpath, index_col=0)
        df_new = df_orig
        if not repeatIfIndexExist and df.index[0] in df_new.index:
            df_new = df_new.drop(df.index[0])
            print(f" ~~~ Row with index {df.index[0]} removed")
        df_new = df_new.append(df)
        print(f" ~~~ Row with index {df.index[0]} added")
    else:
        df_new = df

    with pd.ExcelWriter(fpath, mode="w") as writer:
        df_new.to_excel(writer, sheet_name="Sheet1")


def correctNML(BaseFile):
    """
    Note: Sometimes I have found that python changes the way line breaks occur in a file,
    leading to tr_start not being able to read correctly "inputdir". If this happens,
    simply apply the command-line "tr -d '\r' < file_in > file_out". Example:
    """

    fpath = Path(BaseFile).expanduser()
    if fpath.is_file():
        fpath_new = fpath.with_name(f'{fpath.name}_new')
        with open(fpath, 'r') as ff:
            all_lines = ff.read()
        with open(fpath_new, 'w') as wf:
            wf.write(all_lines.translate(str.maketrans('', '', '\r')))
        if fpath_new.exists():
            fpath_new.replace(fpath)
        #os.system(f"tr -d '\r' < {fpath} > {fpath}_new && mv {fpath}_new {fpath}")


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
    fpath = Path(file).expanduser()
    if fpath.is_file():
        copyToPath = fpath.parent / (fpath.name + "_0")
        if copyToPath.exists():
            loopFileBackUp(copyToPath)
        fpath.replace(copyToPath)


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
    ipath = Path(folder).expanduser()
    if ini is not None:
        if "mfe" in socket.gethostname():
            os.chdir(ipath)
            os.system(f'rename "s/{ini}/{fin}/" *')
        else:
            for filepath in ipath.glob(f"*{ini}*"):
                newname = filepath.name
                newname = newname.sub(f"{ini}", f"{fin}")
                opath = filepath.parent / newname
                filepath.replace(opath)


def readExecutionParams(folderExecution, nums=[0, 9]):
    fpath = Path(folderExecution).expanduser()
    x = []
    for i in np.arange(nums[0], nums[1] + 1, 1):
        params = generateDictionaries(
            fpath / f"Execution" / f"Evaluation.{i}" / f"params.in.{i}"
        )

        dictDVs = params["dictDVs"]

        xp = []
        for ikey in dictDVs:
            xp.append(float(dictDVs[ikey]["value"]))

        x.append(np.array(xp))

    return np.array(x)


def askNewFolder(folderWork, force=False, move=None):
    workpath = Path(folderWork).expanduser()
    if workpath.exists():
        if force:
            shutil_rmtree(workpath)
        else:
            if move is not None:
                workpath.replace(workpath.parent / f"{workpath.name}_{move}")
            else:
                print(
                    f"You are about to erase the content of {workpath.resolve()}", typeMsg="q"
                )
                shutil_rmtree(workpath)
    if not workpath.exists():
        workpath.mkdir(parents=True)
    if workpath.is_dir():
        print(f" \t\t~ Folder ...{clipstr(workpath)} created")
    else:  # What is this?
        fo = reducePathLevel(workpath, level=1, isItFile=False)[0]
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


def getLocInfo(locFile, with_extension=False):
    # First return value is a pathlib.Path object, second return value is a string
    ipath = Path(locFile).expanduser()
    return ipath.parent, ipath.name if with_extension else ipath.stem #f"{ipath.parent}", f"{ipath.stem}"


def findFileByExtension(
    folder, extension, prefix=" ", fixSpaces=False, ForceFirst=False, agnostic_to_case=False, do_not_consider_files=None
    ):
    """
    Retrieves the file without folder and extension
    """

    fpath = Path(folder).expanduser()

    retpath = None
    if fpath.exists():
        allfiles = findExistingFiles(fpath, extension, agnostic_to_case = agnostic_to_case)

        # Filter out files that contain any of the strings in do_not_consider_files
        if do_not_consider_files is not None:
            filtered_files = []
            for file_path in allfiles:
                file_name = file_path.name
                should_exclude = any(exclude_str in file_name for exclude_str in do_not_consider_files)
                if not should_exclude:
                    filtered_files.append(file_path)
            allfiles = filtered_files

        if len(allfiles) > 1:
            # print(allfiles)
            if not ForceFirst:
                raise Exception("More than one file with same extension in the folder!")
            else:
                allfiles = [allfiles[0]]

        if len(allfiles) == 1:
            retpath = allfiles[0]
        else:
            print(
                f"\t\t~ File with extension {extension} not found in {fpath}, returning None"
            )
    else:
        fstr = clipstr(f"{fpath}")
        print(
            f"\t\t\t~ Folder ...{fstr} does not exist, returning None",
        )

    #TODO: We really should not change return type
    #retval = None
    #if retpath is not None:
    #    if not provide_full_path:
    #        retval = f"{retpath.name}".replace(extension, "")
    #    else:
    #        retval = f"{retpath}"

    return retpath


def findExistingFiles(folder, extension, agnostic_to_case=False):
    fpath = Path(folder).expanduser()
    allfiles = []
    for filepath in fpath.glob("*"):
        if filepath.is_file():
            if not agnostic_to_case:
                if f"{filepath.resolve()}".endswith(extension):
                    allfiles.append(filepath.resolve())
            else:
                if f"{filepath.resolve()}".lower().endswith(extension.lower()):
                    allfiles.append(filepath.resolve())
    return allfiles


def writeOFs(resultsFile, dictOFs, dictErrors=None):
    """
    By PRF's convention, error here refers to the standard deviation
    """

    rpath = Path(resultsFile).expanduser()
    for iOF in dictOFs:
        if "error" not in dictOFs[iOF]:
            dictOFs[iOF]["error"] = 0.0

    typeF = "w"

    for cont, iOF in enumerate(dictOFs):
        writeresults(
            rpath, dictOFs[iOF]["value"], dictOFs[iOF]["error"], iOF, typeF=typeF
        )
        typeF = "a"


def generateDictionaries(InputsFile):

    dictDVs = {}
    dictOFs = {}

    ipath = Path(InputsFile).expanduser()
    with open(ipath, "r") as f:
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

    fpath = Path(FilePath).expanduser()

    if not fpath.is_file():
        if raiseException:
            raise Exception(f"File {fpath} not found")
        else:
            return None

    with open(fpath, "r") as f:

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

    try:
        return val
    except:
        if raiseException:
            raise Exception(f"{ParamToFind} Value not found in namelist {fpath}")
        else:
            # print('{} Value not found in namelist {}, returning None'.format(ParamToFind,FilePath)) #,typeMsg='w')
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

    fpath1 = Path(FilePath).expanduser()
    fpath2 = Path(f"{fpath1}_new").expanduser()
    #f1, f2 = open(fpath1, "r"), open(fpath2, "w")

    FoundAtLeastOnce = False
    with open(fpath1, "r") as f1:
        with open(fpath2, "w") as f2:
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

    fpath2.replace(fpath1)

    # If not found at least once, then write it, but make sure it is after the updates flag
    if not FoundAtLeastOnce and Value is not None:
        if CommentChar is not None:
            extt = f"{CommentChar} Added by MITIM"
        else:
            extt = ""
        lines_add = "{0}{4}{1}{4}{2}{4}{3}\n".format(
            ParamToChange, SplittingChar, Value, extt, separator_space
        )

        with open(fpath1, "r") as f:
            lines = f.readlines()
        done, lines_n = False, ""
        for i in lines:
            if "~update" in i and not done:
                lines_n += lines_add
                done = True
            lines_n += i
        if not done:
            lines_n += lines_add
        with open(fpath1, "w") as f:
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
                    fpath1,
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
                    fpath1,
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
    fdir = Path(folder).expanduser()
    fpath = fdir / f"params.in.{num}"
    txt = [
        "                                          1 variables",
        "                      1.000000000000000e+00 factor_ped_degr",
        "                                          1 functions",
        "                                          1 ASV_1:Q_plasma",
        "                                          0 derivative_variables",
        "                                          0 analysis_components",
        "                                          1 eval_id",
    ]
    with open(fpath, "w") as f:
        f.write("\n".join(txt))


def readValueinFile(filename, variable, positionReturn=0):
    fpath = Path(filename).expanduser()
    with open(fpath, "r") as f:

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

    return var


def writeresults(resespec, final_results, final_errors, vartag, typeF="a"):
    opath = Path(resespec).expanduser()
    with open(opath, typeF) as outfile:
        if isnum(final_results):
            outfile.write(
                f"{vartag:15s}: {final_results:1.15e},   {final_errors:1.15e}\n"
            )
        else:
            outfile.write(f"{vartag:15s}: {final_results},   {final_errors}\n")


def readresults(fileresults):
    ipath = Path(fileresults).expanduser()
    with open(ipath, "r") as outfile:
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
    ofile = Path(fileparams).expanduser()
    with open(ofile, "w") as outfile:
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
    fpath = Path(filename).expanduser()
    with open(fpath, "r") as f:
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
    origpath = Path(orig).expanduser()

    # Read values from namelist

    if caseInsensitive:
        boolLower = boolMITIMnoLower
    else:
        boolLower = false

    allLines_clean = getLinesNamelist(
        origpath, commentCommand, separator, boolLower=boolLower
    )
    dictParams = getDictionaryNamelist(
        allLines_clean, separator, caseInsensitive=caseInsensitive
    )

    if WriteNew is not None:
        opath = Path(WriteNew).expanduser()
        with open(opath, "w") as f:
            for i in dictParams:
                f.write(i + "=" + str(dictParams[i]) + "\n")

    return dictParams


def obtainGeneralParams(inputFile, resultsFile):
    ipath = Path(inputFile).expanduser()
    rpath = Path(resultsFile).expanduser()
    FolderEvaluation = ipath.parent if not ipath.is_dir() else ipath

    # In case
    iname = ipath.name
    rname = rpath.name

    numDakota = iname.split(".")[2]

    #return f"{FolderEvaluation}", numDakota, f"{ipath}", f"{rpath}"
    return FolderEvaluation, numDakota, ipath, rpath


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


def expandPath(path, fixSpaces=False, ensurePathValid=False):
    npath = Path(os.path.expandvars(path)).expanduser()
    if ensurePathValid:
        assert npath.exists()
    return npath.resolve() if npath.exists() else npath # To cover cases in which the path is an environment variable that does not exist as file/dir


def reducePathLevel(path, level=1, isItFile=False):
    npath = Path(path).expanduser()
    npath_before = npath
    if len(npath.parents) > level:
        npath_before = npath.parents[level - 1]
    #path_before = f"{npath_before}"
    #if level > 0:
    #    path_before += "/"
    #path_after = f"{npath}"
    #if path_before in path_after:
    #    path_after = path_after.replace(path_before, "")
    #return path_before, path_after
    return npath_before, npath.relative_to(npath_before)


def read_pfile(filepath="./JWH_pedestal_profiles.p", plot=False):
    """
    Method to parse p-files for pedestal modeling.
    sciortino, 2020
    """
    fpath = Path(filepath).expanduser()
    with open(fpath, "r") as f:
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
        fname = str(filename)
        if not fname.endswith(".hdf5"):
            fname += ".hdf5"
        self.fpath = Path(fname).expanduser()
        self.fig = h5py.File(self.fpath, "w")

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
    fname = filename
    if not fname.endswith(".hdf5"):
        fname += ".hdf5"
    fpath = Path(fname).expanduser()
    h5file = hdf5figurefile(fpath)

    for ikey in axesarray_dict:
        name = ikey
        ax = axesarray_dict[name]
        h5file.subplotToHDF5(ax, name=name)

    print(f" --> Written {h5file.fpath}")

    if check:
        # Check
        f = h5py.File(fpath, "r")
        for ikey in f.keys():
            print(np.array(f["a"]["Data0"]["XData"]))


# chatGPT 4o (08/31/2024)
def string_to_sequential_number(input_string, num_digits=5): #TODO: Create a better convertor from path to number to avoid clashes in scratch
    # Separate the last character and base part
    base_part = input_string[:-1]
    last_char = input_string[-1]
    
    # If the last character is a digit, use it as the sequence number
    sequence_digit = int(last_char) if last_char.isdigit() else 0

    # Combine the base part and the sequence digit
    combined_string = f"{base_part}{sequence_digit}"
    
    # Create a hash of the combined string using SHA-256
    hash_object = hashlib.sha256(combined_string.encode())
    
    # Convert the hash to an integer
    hash_int = int(hash_object.hexdigest(), 16)
    
    # Mod the hash to get a number with the desired number of digits
    mod_value = 10**num_digits
    final_number = hash_int % mod_value
    
    # Format the number to ensure it has exactly `num_digits` digits
    return f'{final_number:0{num_digits}d}'

def path_overlapping(nameScratch, append_folder_local, hash_length=20):
    '''
    (chatGPT 4o help)
    This function is used to avoid overlapping of paths in scratch.
    It generates a unique folder name by appending a hashed representation
    of the input folder path to a base name.
    '''

    # Convert the append_folder_local path to a string and encode it in UTF-8,
    # then generate a SHA-256 hash. This ensures a unique, deterministic hash
    # value for the folder path.
    hash_object = hashlib.sha256(str(append_folder_local).encode('utf-8'))

    # Convert the hash object into a hexadecimal string and truncate it to
    # the first 20 characters. This creates a compact, unique identifier for
    # the folder path while reducing the risk of collision.
    unique_hash = hash_object.hexdigest()[:hash_length]
    
    # Combine the base name (nameScratch) with the unique hash to create the
    # final folder name. This ensures the folder is identifiable and unique
    # across different runs or processes.
    nameScratch_full = f"{nameScratch}_{unique_hash}"

    return nameScratch_full


def print_machine_info(output_file=None):

    info_lines = []

    # System Information
    info_lines.append("=== System Information ===")
    info_lines.append(f"System: {platform.system()}")
    info_lines.append(f"Node Name: {platform.node()}")
    info_lines.append(f"Release: {platform.release()}")
    info_lines.append(f"Version: {platform.version()}")
    info_lines.append(f"Machine: {platform.machine()}")
    info_lines.append(f"Processor: {platform.processor()}")

    # CPU Information
    info_lines.append("\n=== CPU Information ===")
    logical_cpus = os.cpu_count()
    info_lines.append(f"Logical CPUs (os.cpu_count()): {logical_cpus}")

    # Attempt to get CPU frequency (limited without external packages)
    try:
        if platform.system() == "Windows":
            import subprocess
            cmd = 'wmic cpu get MaxClockSpeed'
            max_freq = subprocess.check_output(cmd, shell=True).decode().split('\n')[1].strip()
            info_lines.append(f"Max Frequency: {max_freq} MHz")
        elif platform.system() == "Linux":
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read()
            import re
            matches = re.findall(r"cpu MHz\s+:\s+([\d.]+)", cpuinfo)
            if matches:
                current_freq = matches[0]
                info_lines.append(f"Current Frequency: {current_freq} MHz")
        else:
            info_lines.append("CPU Frequency information not available.")
    except Exception as e:
        info_lines.append("Error retrieving CPU Frequency information.")

    # PyTorch CPU Information
    info_lines.append("\n=== PyTorch Information ===")
    num_threads = torch.get_num_threads()
    num_interop_threads = torch.get_num_interop_threads()
    openmp_enabled = getattr(torch.backends, 'openmp', None)
    mkl_enabled = getattr(torch.backends, 'mkl', None)

    info_lines.append(f"PyTorch Intraop Threads: {num_threads}")
    info_lines.append(f"PyTorch Interop Threads: {num_interop_threads}")
    info_lines.append(f"OpenMP Enabled in PyTorch: {openmp_enabled.is_available() if openmp_enabled else 'N/A'}")
    info_lines.append(f"MKL Enabled in PyTorch: {mkl_enabled.is_available() if mkl_enabled else 'N/A'}")

    # Output to screen or file
    output = '\n'.join(info_lines)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
    else:
        print(output)


def monitor_resources(pid, log_file="resource_log.txt", interval=1):

    # Include machine info
    print_machine_info(output_file=log_file)

    process = psutil.Process(pid)
    start_time = time.time()  # Record the start time of logging
    
    with open(log_file, "a") as log:
        # Write header with proper column alignment
        log.write(f"Monitoring resources for PID: {pid}\n")
        log.write(
            f"{'Timestamp':<20} {'Elapsed Time (s)':<18} {'Memory (GB)':<12} "
            f"{'CPU (%)':<8} {'Threads':<10} {'Open Files':<12} {'IO Read (MB)':<15} {'IO Write (MB)':<15}\n"
        )
        log.write("=" * 100 + "\n")  # Add a separator line for clarity
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

                # Gather metrics
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=interval)
                num_threads = process.num_threads()
                open_files = len(process.open_files())
                
                # Safely handle io_counters
                try:
                    io_counters = process.io_counters()
                    read_bytes = io_counters.read_bytes / 1e6  # Convert to MB
                    write_bytes = io_counters.write_bytes / 1e6  # Convert to MB
                except AttributeError:
                    read_bytes = write_bytes = 0.0  # Fallback values
                
                # Format and write log entry
                log_entry = (
                    f"{timestamp:<20} {elapsed_time:<18.2f} {memory_info.rss / 1e9:<12.2f} "
                    f"{cpu_percent:<8.2f} {num_threads:<10} {open_files:<12} {read_bytes:<15.2f} {write_bytes:<15.2f}\n"
                )
                log.write(log_entry)
                log.flush()  # Ensure logs are updated in real-time
        except (psutil.NoSuchProcess, KeyboardInterrupt):
            log.write("Monitoring stopped.\n")
            print("Monitoring stopped.")

def plot_metrics(log_file="resource_log.txt", output_image="resource_metrics.png"):
    
    column_names = [
        "Timestamp", "ElapsedTime", "MemoryGB", "CPUPercent", 
        "Threads", "OpenFiles", "IOReadMB", "IOWriteMB"
    ]

    data = pd.read_csv(
        log_file,
        sep=r"\s\s+",
        skiprows=2,  # Skip the first two header lines
        names=column_names,
        parse_dates=["Timestamp"],  # Automatically parse Timestamp
        on_bad_lines="skip",  # Skip malformed lines
    )

    # Clean the data
    # Replace non-numeric or NaN values with 0 or a suitable default
    for col in column_names[1:]:  # Skip Timestamp
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    # Create a figure with subplots
    plt.ion()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

    # Define metrics and labels
    metrics = [
        ("MemoryGB", "Memory Usage (GB)"),
        ("CPUPercent", "CPU Usage (%)"),
        ("Threads", "Threads"),
        ("OpenFiles", "Open Files"),
        ("IOReadMB", "IO Read (MB)"),
        ("IOWriteMB", "IO Write (MB)"),
    ]

    # Plot each metric
    for ax, (metric, label) in zip(axes.flat, metrics):
        ax.plot(data["ElapsedTime"], data[metric], marker="o", linestyle="-",markersize=0.5)
        ax.set_ylabel(label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Elapsed Time (s)")
        GRAPHICStools.addDenseAxis(ax)

    plt.tight_layout()

def shutil_rmtree(item):
    '''
    Removal of folders may fail because of a "Directory not empty" error, 
    even if the files were properly removed. This is because of potential syncronization
    or back-up processes that may be running in the background in some file systems.
    Temporary solution for now is to use the shutil.rmtree function with a try-except block,
    one that waits a second and one that just renames the folder to a temporary name.
    '''

    try:
        shutil.rmtree(item)
    except OSError:
        time.sleep(1)
        try:
            shutil.rmtree(item)
        except OSError:
            new_item = item.with_name(item.name + "_cannotrm")
            shutil.move(item, new_item)
            print(f"> Folder {clipstr(item)} could not be removed. Renamed to {clipstr(new_item)}",typeMsg='w')

def recursive_backup(file, extension='bak'):
    
    index = 0
    file_new = file.with_suffix(f".{extension}.{index}")
    
    while file_new.exists():
        index += 1
        file_new = file.with_suffix(f".{extension}.{index}")

    shutil.copy2(file, file_new)
    print(f"> File {clipstr(file)} backed up to {clipstr(file_new)}", typeMsg='i')


def unpickle_mitim(file):

    with open(str(file), "rb") as handle:
        try:
            state = pickle_dill.load(handle)
        except:
            print("\t- Pickled file could not be opened, going with custom unpickler...",typeMsg='w')
            handle.seek(0)
            state = CPU_Unpickler(handle).load()

    return state

"""
To load pickled GPU-cuda classes on a CPU machine
From:
	https://github.com/pytorch/pytorch/issues/16797
	https://stackoverflow.com/questions/35879096/pickle-unpicklingerror-could-not-find-mark
"""
class CPU_Unpickler(pickle_dill.Unpickler):
    def find_class(self, module, name):
        import io

        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
        else:
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                print(f"\t\tModule not found: {module} {name}; returning dummy", typeMsg="i")
                return super().find_class("torch._utils", name)

