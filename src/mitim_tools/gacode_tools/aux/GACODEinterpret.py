import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print


def Waveform_read(file, fileOut):
    """
    Provides:
            theta, results[field_name][mode,theta]

            where: field_name = RE(phi)    IM(phi)    RE(Bper)    IM(Bper)    RE(Bpar)    IM(Bpar)
    """

    with open(file, "r") as f:
        aux = f.readlines()

    results = []
    for i in range(len(aux) - 3):
        line = aux[i + 3].split()
        for i in line:
            results.append(float(i))

    line1 = aux[0].split()
    nmodes = int(line1[0])
    nfields = int(line1[1])

    results = np.array(results).reshape(
        (int(len(results) / (nmodes * nfields * 2 + 1)), nmodes * nfields * 2 + 1)
    )

    fields = aux[1].split()[1:]
    theta = results[:, 0]

    """
	Until here, results is [:,colum], where colums go through fields first, and then modes
	"""

    # Now convert to [field,mode,theta]

    results2 = np.zeros((nfields * 2, nmodes, len(theta)))
    for imode in range(nmodes):
        for ifield in range(nfields * 2):  # x2 for RE and IM
            results2[ifield, imode, :] = results[:, 1 + imode + ifield]

    # Now understand what fields are those
    # possible: RE(phi)    IM(phi)    RE(Bper)    IM(Bper)    RE(Bpar)    IM(Bpar)
    resultsField = {}
    for ifield, field in enumerate(fields):
        resultsField[field] = results2[ifield, :, :]

    # If BPER or BPAR does not exist, create with zeros
    if "RE(Bper)" not in resultsField:
        resultsField["RE(Bper)"] = np.zeros((nmodes, len(theta)))
    if "RE(Bpar)" not in resultsField:
        resultsField["RE(Bpar)"] = np.zeros((nmodes, len(theta)))
    if "IM(Bper)" not in resultsField:
        resultsField["IM(Bper)"] = np.zeros((nmodes, len(theta)))
    if "IM(Bpar)" not in resultsField:
        resultsField["IM(Bpar)"] = np.zeros((nmodes, len(theta)))

    resultsField["theta"] = theta

    with open(fileOut, "r") as f:
        aux = f.readlines()

    for ir in range(len(aux)):
        if "ky:" in aux[ir]:
            break

    ky = float(aux[ir].split()[-1])
    g, f = [], []
    for i in range(len(aux[ir + 2 :])):
        a = aux[ir + 2 + i].split()
        f.append(float(a[1]))
        g.append(float(a[2]))

    resultsField["ky"] = np.array([ky] * len(g))
    resultsField["gamma"] = np.array(g)
    resultsField["freq"] = np.array(f)

    return resultsField


def string_is_float(element):
    # Ignore also integers
    if "." not in element:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def TGLFreader(file, blocks=3, columns=5, numky=None):
    """
    Only one of them can be None
    """

    with open(file, "r") as f:
        aux = f.readlines()

    # Read full file in a single array
    aux_array = np.array([])
    for line_cont in range(len(aux)):
        if string_is_float(aux[line_cont].split()[0]):
            aux_array = np.append(aux_array, [float(i) for i in aux[line_cont].split()])

    # Reshape
    if numky is None:
        numky = int(len(aux_array) / (blocks * columns))
    elif columns is None:
        columns = int(len(aux_array) / (blocks * numky))
    elif blocks is None:
        blocks = int(len(aux_array) / (columns * numky))
    data = np.reshape(aux_array, (blocks, numky, columns))

    return data


# --------- TGYRO


def readGeneral(file, numcols=3, maskfun=None):
    with open(file, "r") as f:
        aux = f.readlines()

    num = len(aux)

    vecT = {}
    for k in range(numcols):
        vecT[k] = []

    i = 0
    while i < num:
        if "r/a" in aux[i]:
            if i > 0:
                for k in range(numcols):
                    vecT[k].append(vec[k])
            vec = {}
            for k in range(numcols):
                vec[k] = []
            i += 1
        else:
            v = [float(j) for j in aux[i].split()]
            for k in range(numcols):
                vec[k].append(v[k])
        i += 1

    for k in range(numcols):
        vecT[k].append(vec[k])

    """
	vecT is a dictionary, with keys each of the columms (proxy for variables)
	vecT[0] is a 2D array, with (iteration,radial)
	"""

    vecTarr = []
    for k in range(numcols):
        vecTarr.append(np.array(vecT[k]))
    vecTarr = np.array(vecTarr)

    if maskfun is not None:
        vecTarr_new = []
        for j in range(vecTarr.shape[0]):
            vecTarr_new.append(maskfun(vecTarr[j, :, :]))
        vecTarr = np.array(vecTarr_new)

    return vecTarr
