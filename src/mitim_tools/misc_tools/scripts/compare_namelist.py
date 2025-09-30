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


def printTable(diff, warning_percent=1e-1):

    # Compute percent differences first so we can sort by them
    percs = {}
    for key in diff:
        v0, v1 = diff[key]
        if v0 is None or v1 is None:
            # Treat missing values as 100% difference for sorting
            percs[key] = 100.0
        else:
            if v0 != 0.0:
                try:
                    percs[key] = 100 * np.abs((v0 - v1) / v0)
                except Exception:
                    percs[key] = np.nan
            else:
                percs[key] = np.nan

    # Sort keys by descending percent; NaNs go last
    def sort_key(k):
        p = percs[k]
        return (1, 0) if (p is None or (isinstance(p, float) and np.isnan(p))) else (0, -p)

    for key in sorted(diff.keys(), key=sort_key):
        v0, v1 = diff[key]
        if v0 is not None:
            if v1 is not None:
                perc = percs[key]
                if perc<1e-2:
                    perc_str = f"{perc:.2e}"
                elif perc<1.0:
                    perc_str = f"{perc:.3f}"
                else:
                    perc_str = f"{perc:.1f}"
                print(f"{key:>15}{str(v0):>25}{str(v1):>25}  ({perc_str} %)",typeMsg="i" if perc > warning_percent else "",)
            else:
                print(f"{key:>15}{str(v0):>25}{'':>25}  (100%)", typeMsg="i")
        else:
            print(f"{key:>15}{'':>25}{str(v1):>25}  (100%)", typeMsg="i")
        print("--------------------------------------------------------------------------------")


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
