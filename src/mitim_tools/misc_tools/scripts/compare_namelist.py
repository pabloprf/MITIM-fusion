import argparse
import numpy as np
from mitim_tools.misc_tools import IOtools
from IPython import embed

from mitim_tools.misc_tools.LOGtools import printMsg as print

"""
This is used to commpare namelists values
e.g.
		compareNML.py namelist1TR.DAT namelist2TR.DAT =

"""


def compareNML(file1, file2, commentCommand="!", separator="=", precision_of=None, close_enough=1e-7):
    d1 = IOtools.generateMITIMNamelist(
        file1, commentCommand=commentCommand, separator=separator
    )
    d2 = IOtools.generateMITIMNamelist(
        file2, commentCommand=commentCommand, separator=separator
    )

    d1 = separateArrays(d1)
    d2 = separateArrays(d2)

    diff = compareDictionaries(d1, d2, precision_of=precision_of, close_enough=close_enough)

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
        if isinstance(d[ikey], str) and "," in d[ikey]:
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
        if isinstance(dnew[ikey], str) and "'" in dnew[ikey]:
            dnew[ikey] = dnew[ikey].replace("'", '"')

    return dnew


def cleanDifferences(d, tol_rel=1e-7):
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

def compare_number(a,b,precision_of=None, close_enough=1e-7):

    if precision_of is None:
        a_rounded = a
        b_rounded = b

    elif precision_of == 1:
        # Round to the same number of decimal places
        a_str = str(a)
        if '.' in a_str:
            decimal_places = len(a_str.split('.')[1])
        else:
            decimal_places = 0

        if isinstance(b, str):
            b_rounded = b
        else:  
            b_rounded = round(b, decimal_places)
        a_rounded = a

    elif precision_of == 2:

        # Round to the same number of significant figures
        b_str = str(b)
        if '.' in b_str:
            decimal_places = len(b_str.split('.')[1])
        else:
            decimal_places = 0

        b_rounded = b
        a_rounded = round(a, decimal_places)

    # Compare the two numbers
    if isinstance(a_rounded, str) or isinstance(b_rounded, str):
        # If either is a string, we cannot compare numerically
        are_equal = a_rounded == b_rounded
    else:
        are_equal = np.isclose(a_rounded, b_rounded, rtol=close_enough)

    return are_equal

def compareDictionaries(d1, d2, precision_of=None, close_enough=1e-7):
    different = {}

    for key in d1:
        # Exists in d1 but not in d2
        if key not in d2:
            different[key] = [d1[key], None]
        # Values are different
        else:
            if not compare_number(d1[key],d2[key],precision_of=precision_of, close_enough=close_enough):
                different[key] = [d1[key], d2[key]]

    for key in d2:
        # Exists in d2 but not in d1
        if key not in d1:
            different[key] = [None, d2[key]]

    return different


def printTable(diff, printing_percent = 1e-5, warning_percent=1e-1):

    for key in diff:
        if diff[key][0] is not None:
            if diff[key][1] is not None:
                if diff[key][0] != 0.0:
                    try:
                        perc = 100 * np.abs(
                            (diff[key][0] - diff[key][1]) / diff[key][0]
                        )
                    except:
                        perc = np.nan
                else:
                    perc = np.nan
                print(
                    f"{key:>15}{str(diff[key][0]):>25}{str(diff[key][1]):>25}  (~{perc:.0e}%)",
                    typeMsg="i" if perc > warning_percent else "",
                )
            else:
                print(f"{key:>15}{str(diff[key][0]):>25}{'':>25}  (100%)", typeMsg="i")
        else:
            print(f"{key:>15}{'':>25}{str(diff[key][1]):>25}  (100%)", typeMsg="i")
        print(
            "--------------------------------------------------------------------------------"
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=str, help="First namelist file to compare")
    parser.add_argument("file2", type=str, help="Second namelist file to compare")
    parser.add_argument("--separator", type=str, required=False, default="=",
                        help="Separator used in the namelist files, default is '='")
    parser.add_argument("--precision", type=int, required=False, default=None,
                        help="Precision for comparing numbers: 1 for decimal places, 2 for significant figures, None for exact comparison")
    parser.add_argument("--close_enough", type=float, required=False, default=1e-7,
                        help="Tolerance for comparing numbers, default is 1e-7")
    args = parser.parse_args()

    # Get arguments
    file1 = args.file1
    file2 = args.file2
    separator = args.separator
    precision = args.precision
    close_enough = args.close_enough

    diff = compareNML(file1, file2, separator=separator, precision_of=precision, close_enough=close_enough)

    printTable(diff)
    print(f"Differences: {len(diff)}")

if __name__ == "__main__":
    main()
