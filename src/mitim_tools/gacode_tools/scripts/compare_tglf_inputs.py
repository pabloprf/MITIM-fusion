import sys, copy
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools

"""
This is used to commpare namelists values
e.g.
		compareNML.py namelist1TR.DAT namelist2TR.DAT

"""


def compareNML(file1, file2):
    d1 = IOtools.generateMITIMNamelist(file1, commentCommand="!", separator="=")
    d2 = IOtools.generateMITIMNamelist(file2, commentCommand="!", separator="=")

    d1 = separateArrays(d1)
    d2 = separateArrays(d2)

    diff = compareDictionaries(d1, d2)

    diffo = cleanDifferences(diff)

    k = sorted([i for i in diffo.keys()])
    diff = IOtools.CaseInsensitiveDict()
    for i in k:
        diff[i] = diffo[i]

    return diff


def separateArrays(d):
    # Separate values in commas
    dnew = IOtools.CaseInsensitiveDict()
    for ikey in d:
        if type(d[ikey]) == str and "," in d[ikey]:
            arr = d[ikey].split(",")
            for cont, i in enumerate(arr):
                try:
                    conv = float(i)
                except:
                    conv = i
                dnew[ikey + f"({cont + 1})"] = conv
        else:
            dnew[ikey] = d[ikey]

    # Quotes
    for ikey in dnew:
        if type(dnew[ikey]) == str and "'" in dnew[ikey]:
            dnew[ikey] = dnew[ikey].replace("'", '"')

    return dnew


def cleanDifferences(d, tol_rel=0.0001):
    d_new = {}
    for key in d:
        if key not in ["inputdir"]:
            if (
                d[key][0] is None
                or d[key][1] is None
                or isinstance(d[key][0], str)
                or isinstance(d[key][1], str)
                or isinstance(d[key][0], bool)
                or isinstance(d[key][1], bool)
                or d[key][0] == 0
                or np.abs((d[key][0] - d[key][1]) / d[key][0]) > tol_rel
            ):
                d_new[key] = d[key]

    return d_new


def compareDictionaries(d1, d2):
    different = {}

    for key in d1:
        # Exists in d1 but not in d2
        if key not in d2:
            different[key] = [d1[key], None]
        # Values are different
        else:
            if d1[key] != d2[key]:
                different[key] = [d1[key], d2[key]]

    for key in d2:
        # Exists in d2 but not in d1
        if key not in d1:
            different[key] = [None, d2[key]]

    return different


def printTable(diff):
    try:    
        print(f"{'':>15}{file1.split('/')[-1]:>25}{file2.split('/')[-1]:>25}")
    except:
        pass
    for key in diff:
        if diff[key][0] is not None:
            if diff[key][1] is not None:
                print(f"{key:>15}{str(diff[key][0]):>25}{str(diff[key][1]):>25}")
            else:
                print(f"{key:>15}{str(diff[key][0]):>25}{'':>25}")
        else:
            print(f"{key:>15}{'':>25}{str(diff[key][1]):>25}")
        print("-----------------------------------------------------------------------")


if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    diff = compareNML(file1, file2)

    printTable(diff)
    print(f"Differences: {len(diff)}")
