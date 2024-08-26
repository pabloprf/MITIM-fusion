import torch
import numpy as np
from IPython import embed

from mitim_tools.misc_tools import IOtools
from mitim_modules.freegsu import FREEGSUmain


def createProblemParameters(
    InitialCurrents,
    maxVariation1,
    div2equal1=0,
    cs3equal2=0,
    xp=[[157.5, 112.0], [151.0, 110.0]],
    maxVariation2=None,
    fileRequirements=None,
    coilsVersion="V2new",
):


    try:
        from FREEGS_SPARC import GSsparc_coils
    except ImportError:
        raise Exception(
            "[mitim] The FREEGS_SPARC module is not available. Please ensure it is installed and accessible."
        )


    # I have to please change div2equal1 and cs3equal remove from here
    # embed()

    # Specifics for this case

    ofs = ["xpR_1", "xpZ_1", "mpo_1", "mpi_1"]
    calofs = [xp[0][0], xp[0][1], 242.0, 128]

    # dvs  	= ['pf4','dv1','dv2','pf3','pf2','pf1','cs3','cs2']
    # if 'cs1' in maxVariation1: dvs.append('cs1')

    dvs = [i for i in maxVariation1]

    if div2equal1 != 0:
        dvs.remove("dv2")
    if cs3equal2 != 0:
        dvs.remove("cs3")

    wofs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Read limits from file and make sure I don't surpass it
    if fileRequirements is not None:
        maxVar_I, _, minVar_I, _ = FREEGSUmain.readCoilCalcsMatrices_Supplies(
            fileRequirements
        )

        # Convert to MA-t, which are my DVs
        sparc_coils = GSsparc_coils.SPARCcoils(None, coilsVersion=coilsVersion)
        for i in maxVar_I:
            maxVar_I[i] = maxVar_I[i] * sparc_coils.turns_real[i + "u"] * 1e-3
            minVar_I[i] = minVar_I[i] * sparc_coils.turns_real[i + "u"] * 1e-3

    dvs_base, dvs_min, dvs_max = [], [], []
    for cont, i in enumerate(dvs):
        base = InitialCurrents[i]

        if not IOtools.isnum(base):
            base = base[0]

        dvs_min.append(base - maxVariation1["".join(i)])
        dvs_max.append(base + maxVariation1["".join(i)])

        # Requirements
        if fileRequirements is not None:
            dvs_min[cont] = np.max([dvs_min[cont], minVar_I[i]])
            dvs_max[cont] = np.min([dvs_max[cont], maxVar_I[i]])

        # Make sure base point is in between limits (could not be due to requirements)
        if (base <= dvs_min[cont]) or (base >= dvs_max[cont]):
            print(i, " baseline is out of bounds")
            base = np.mean([dvs_min[cont], dvs_max[cont]])

        dvs_base.append(base)

    # ------------------------------------------------------------
    # In the case of second one
    # ------------------------------------------------------------
    if maxVariation2 is not None:
        ofs.extend(["xpR_2", "xpZ_2", "mpo_2", "mpi_2"])
        calofs.extend([xp[1][0], xp[1][1], 242.0, 128.0])
        wofs.extend([1.0, 1.0, 1.0, 1.0])

        dvs_2 = []
        for i in dvs:
            dvs_2.append(i + "_delta")

        dvs_base2, dvs_min2, dvs_max2 = [], [], []
        for cont, i in enumerate(dvs):
            base = 0.0

            dvs_min2.append(base - maxVariation2["".join(i)])
            dvs_base2.append(base)
            dvs_max2.append(base + maxVariation2["".join(i)])

            embed()
            # missing requirements part

        dvs.extend(dvs_2)
        dvs_base.extend(dvs_base2)
        dvs_min.extend(dvs_min2)
        dvs_max.extend(dvs_max2)

        transformation = produceNewInputs
    else:
        transformation = None

    return transformation, dvs, ofs, calofs, wofs, dvs_base, dvs_min, dvs_max


def updateCoils(
    fixedstepSettings,
    dictDVs,
    div1div=[1, 0, 0, 0],
    div2div=[1, 0, 0, 0],
    reverse=False,
):
    """
    Goal of this routine:
            Merging of fixedstepSettings and dictDVs, separate div coils in contributions.
            Outputs can be scalars or list for a scan. There can be a mix among coils
    """

    div1div, div2div = np.array(div1div), np.array(div2div)

    # --------------------------------------------------

    CoilsDict = {}

    cs1 = fixedstepSettings["cs1"]
    cs2 = fixedstepSettings["cs2"]
    cs3 = fixedstepSettings["cs3"]
    pf1 = fixedstepSettings["pf1"]
    pf2 = fixedstepSettings["pf2"]
    pf3 = fixedstepSettings["pf3"]
    pf4 = fixedstepSettings["pf4"]
    dv1 = fixedstepSettings["dv1"]
    dv2 = fixedstepSettings["dv2"]
    if "vs1" in fixedstepSettings:
        vs1 = fixedstepSettings["vs1"]
    else:
        vs1 = 0.0

    # ------------------------------------------------------------

    if "cs1" in dictDVs:
        cs1 = dictDVs["cs1"]["value"]
    if "cs2" in dictDVs:
        cs2 = dictDVs["cs2"]["value"]
    if "cs3" in dictDVs:
        cs3 = dictDVs["cs3"]["value"]
    if "pf1" in dictDVs:
        pf1 = dictDVs["pf1"]["value"]
    if "pf2" in dictDVs:
        pf2 = dictDVs["pf2"]["value"]
    if "pf3" in dictDVs:
        pf3 = dictDVs["pf3"]["value"]
    if "pf4" in dictDVs:
        pf4 = dictDVs["pf4"]["value"]
    if "dv1" in dictDVs:
        dv1 = dictDVs["dv1"]["value"]
    if "dv2" in dictDVs:
        dv2 = dictDVs["dv2"]["value"]
    if "vs1" in dictDVs:
        dv2 = dictDVs["vs1"]["value"]

    # ------------------------------------------------------------
    # In the case of second one
    # ------------------------------------------------------------

    if "cs1_delta" in dictDVs:
        cs1 = [cs1, cs1 + dictDVs["cs1_delta"]["value"]]
    if "cs2_delta" in dictDVs:
        cs2 = [cs2, cs2 + dictDVs["cs2_delta"]["value"]]
    if "cs3_delta" in dictDVs:
        cs3 = [cs3, cs3 + dictDVs["cs3_delta"]["value"]]
    if "pf1_delta" in dictDVs:
        pf1 = [pf1, pf1 + dictDVs["pf1_delta"]["value"]]
    if "pf2_delta" in dictDVs:
        pf2 = [pf2, pf2 + dictDVs["pf2_delta"]["value"]]
    if "pf3_delta" in dictDVs:
        pf3 = [pf3, pf3 + dictDVs["pf3_delta"]["value"]]
    if "pf4_delta" in dictDVs:
        pf4 = [pf4, pf4 + dictDVs["pf4_delta"]["value"]]
    if "dv1_delta" in dictDVs:
        dv1 = [dv1, dv1 + dictDVs["dv1_delta"]["value"]]
    if "dv2_delta" in dictDVs:
        dv2 = [dv2, dv2 + dictDVs["dv2_delta"]["value"]]
    if "vs1_delta" in dictDVs:
        vs1 = [vs1, vs1 + dictDVs["vs1_delta"]["value"]]

    # ------------------------------------------------------------
    # DIV A,B,C,D issue
    # ------------------------------------------------------------

    if IOtools.isnum(dv1):
        [div1a, div1b, div1c, div1d] = div1div * dv1
    else:
        div1a = [i * div1div[0] for i in dv1]
        div1b = [i * div1div[1] for i in dv1]
        div1c = [i * div1div[2] for i in dv1]
        div1d = [i * div1div[3] for i in dv1]

    if IOtools.isnum(dv2):
        [div2a, div2b, div2c, div2d] = div2div * dv2
    else:
        div2a = [i * div2div[0] for i in dv2]
        div2b = [i * div2div[1] for i in dv2]
        div2c = [i * div2div[2] for i in dv2]
        div2d = [i * div2div[3] for i in dv2]

        # div1a = np.array(div1a)
        # div1b = np.array(div1b)
        # div1c = np.array(div1c)
        # div1d = np.array(div1d)
        # div2a = np.array(div2a)
        # div2b = np.array(div2b)
        # div2c = np.array(div2c)
        # div2d = np.array(div2d)

    coils_names = (
        "cs1",
        "cs2",
        "cs3",
        "pf1",
        "pf2",
        "pf3",
        "pf4",
        "div1a",
        "div1b",
        "div1c",
        "div1d",
        "div2a",
        "div2b",
        "div2c",
        "div2d",
        "vs1",
    )
    coils = (
        cs1,
        cs2,
        cs3,
        pf1,
        pf2,
        pf3,
        pf4,
        div1a,
        div1b,
        div1c,
        div1d,
        div2a,
        div2b,
        div2c,
        div2d,
        vs1,
    )

    if reverse:
        newcoils = []
        for i in coils:
            if not IOtools.isnum(i) and i is not None:
                newcoils.append(np.flipud(i))
            else:
                newcoils.append(i)
        coils = tuple(newcoils)

    return coils, coils_names


def findposcoil(coil, coils, coils_names):
    for i in range(len(coils_names)):
        if coil == coils_names[i]:
            return i


def equalizer(coils, coils_lower, coils_names, equalizers={}):
    coils = list(coils)
    coils_lower = list(coils_lower)

    for i in equalizers:
        keys1 = i.split("_")
        # Find position of the one to be modified
        upper1 = keys1[1] == "u"
        if upper1:
            pos1 = findposcoil(keys1[0], coils, coils_names)
        else:
            pos1 = findposcoil(keys1[0], coils_lower, coils_names)

        keys2 = equalizers[i].split("_")
        # Find value of the one to modify with
        upper2 = keys2[1] == "u"
        if upper2:
            pos2 = findposcoil(keys2[0], coils, coils_names)
            val = coils[pos2]
        else:
            pos2 = findposcoil(keys2[0], coils_lower, coils_names)
            val = coils_lower[pos2]

        # Do I request a negative:
        if len(keys2) > 2 and keys2[2] == "neg":
            val = [-p for p in val]

        # ** Modify **
        if upper1:
            coils[pos1] = val
        else:
            coils_lower[pos1] = val

    return tuple(coils), tuple(coils_lower)


def assignCoils(
    cont,
    cs1,
    cs2,
    cs3,
    pf1,
    pf2,
    pf3,
    pf4,
    dv1a,
    dv1b,
    dv1c,
    dv1d,
    dv2a,
    dv2b,
    dv2c,
    dv2d,
    vs1,
):
    """
    Goal of this routine:
            Because I may be doing scans in some of the coils, extract here
            those in positions "cont".
            Some coils may be scalars because they don't vary
            during the scans, so just take those
    """

    fin = []
    for coil in (
        cs1,
        cs2,
        cs3,
        pf1,
        pf2,
        pf3,
        pf4,
        dv1a,
        dv1b,
        dv1c,
        dv1d,
        dv2a,
        dv2b,
        dv2c,
        dv2d,
        vs1,
    ):
        fin.append(extractCont(coil, cont))

    return tuple(fin)


def extractCont(x, cont):
    if IOtools.islistarray(x):
        v = x[cont]
    else:
        v = x

    return v


def produceNewInputs(X, output, bounds, ParamProfile):
    """
    X will be a tensor (with or without gradients) batch*dim,   unnormalized
    """

    stringVariation = "_delta"

    Xnew = torch.Tensor().to(X)

    for cont, i in enumerate(bounds):
        if output[-1] == "1":
            if stringVariation not in i:
                vv = X[:, cont]
                Xnew = torch.cat((Xnew, vv.unsqueeze(1)), axis=1)

        if output[-1] == "2":
            if stringVariation in i:
                for cont_abs, i2 in enumerate(bounds):
                    if i2 == i[: -len(stringVariation)]:
                        break
                vv = X[:, cont] + X[:, cont_abs]
                Xnew = torch.cat((Xnew, vv.unsqueeze(1)), axis=1)

    return Xnew
