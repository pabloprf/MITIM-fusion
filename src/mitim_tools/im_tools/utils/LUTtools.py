import copy, pickle
import numpy as np

from mitim_tools.misc_tools import MATHtools, IOtools, FARMINGtools
from mitim_tools.misc_tools import CONFIGread

from IPython import embed

from mitim_tools.misc_tools.IOtools import printMsg as print


def understandLuT(allLines):
    linesHeader = 0
    for i in range(1000):
        try:
            aux = float(allLines[i].split()[0])
            break
        except:
            linesHeader += 1

    linesInBetween = 0
    for i in range(1000):
        try:
            aux = float(allLines[linesHeader + i + 1].split()[0])
            break
        except:
            linesInBetween += 1

    linesvalues = (len(allLines) - linesHeader) // (1 + linesInBetween)

    print(
        "\t- LuT has {0} lines for header and {1} actual EPED entries (separated by {2} blank spaces)".format(
            linesHeader, linesvalues, linesInBetween
        )
    )

    return linesvalues, linesHeader, linesInBetween


def readLuT(lut_file, typeLUT="NTH2"):  #'JWH','NTH'
    # Read values from LuT EPED
    with open(lut_file, "r") as f:
        allLines = f.readlines()

    # Understand content
    linesvalues, linesHeader, linesInBetween = understandLuT(allLines)

    # Construct vectors
    vec = []
    for i in range(linesvalues):
        vec.append(
            [float(j) for j in allLines[i * (1 + linesInBetween) + linesHeader].split()]
        )
    vec = np.array(vec).transpose()

    # Check for issues (negative number)
    map1 = vec[-1, :] > 0
    vec = vec[:, map1]

    # ----

    if typeLUT == "NTH":
        # Inputs
        Bt, Ip, R, kappa, delta, BetaN, neped, Zeff, epsilon, Aimp, Zimp, m, z = vec[
            :13
        ]
        # Outputs
        width_top, presstop, netop, Tetop = vec[13:]
    if typeLUT == "NTH2":
        # Inputs
        Bt, Ip, R, kappa, delta, BetaN, neped, Zeff, epsilon, Aimp, Zimp, m, z = vec[
            :13
        ]
        # Outputs
        width_top, pressped, presstop, netop, Tetop = vec[13:]
    elif typeLUT == "JWH":
        # Inputs
        neped, delta, kappa, R, a, Ip, Bt, BetaN, Zeff, m, z = vec[3:]
        epsilon = a / R
        Aimp, Zimp = 0, 0
        # Outputs
        presstop, width_top, Tetop = vec[:3]
        Tetop = Tetop * 1e3
        netop = presstop * 1e6 / (2 * Tetop * 1.6e-19)

    return (
        Bt,
        Ip,
        R,
        kappa,
        delta,
        BetaN,
        neped,
        Zeff,
        epsilon,
        Aimp,
        Zimp,
        m,
        z,
        width_top,
        netop,
        Tetop,
        presstop,
        pressped,
    )


def prepare_LuT_EPED(
    LuT_loc,
    LuT_variables,
    Bt,
    Rmajor,
    Ip,
    kappa,
    delta,
    BetaN,
    neped,
    epsilon,
    Aimp=22.0,
    Zimp=11.0,
    m=2.5,
    z=1.0,
    Zeff=1.5,
    enforceClosestShaping=True,
    LuT_fixed=np.zeros(100),
):
    # ~~~~ Read LuT

    (
        v_Bt,
        v_Ip,
        v_Rmajor,
        v_kappa,
        v_delta,
        v_BetaN,
        v_neped,
        v_Zeff,
        v_epsilon,
        v_Aimp,
        v_Zimp,
        v_m,
        v_z,
        v_width_top,
        v_netop,
        v_Tetop,
        v_presstop,
        v_pressped,
    ) = readLuT(LuT_loc)

    # Check interpolation
    if BetaN > np.max(v_BetaN):
        print(
            "\t- BetaN outside of LUT bounds (BetaN={0:.3f}), taking pedestal value at maximum BetaN={1:.3f} in LuT".format(
                BetaN, np.max(v_BetaN)
            ),
            typeMsg="w",
        )
        BetaN = np.max(v_BetaN)
    if BetaN < np.min(v_BetaN):
        print(
            "\t- BetaN outside of LUT bounds (BetaN={0:.3f}), taking pedestal value at minimum BetaN={1:.3f} in LuT".format(
                BetaN, np.min(v_BetaN)
            ),
            typeMsg="w",
        )
        BetaN = np.min(v_BetaN)

    if enforceClosestShaping:
        delta_orig = copy.deepcopy(delta)
        kappa_orig = copy.deepcopy(kappa)
        delta = np.unique(v_delta)[np.argmin(np.abs(np.unique(v_delta) - delta))]
        kappa = np.unique(v_kappa)[np.argmin(np.abs(np.unique(v_kappa) - kappa))]

        if round(delta, 3) - round(delta_orig, 3) > 1e-3:
            print(
                "\t- Warning: Original delta = {0:.3f} converted to delta = {2:.3f} for LuT search".format(
                    delta_orig, kappa_orig, delta, kappa
                ),
                typeMsg="w",
            )
        if round(kappa, 3) - round(kappa_orig, 3) > 1e-3:
            print(
                "\t- Warning: Original kappa = {1:.3f} converted to kappa = {3:.3f} for LuT search".format(
                    delta_orig, kappa_orig, delta, kappa
                ),
                typeMsg="w",
            )

    # ~~~~ Construct vectors for search

    vectorsLUT = []
    valuesEval = []
    for i in LuT_variables:
        if i == "bt":
            vectorsLUT.append(v_Bt)
            valuesEval.append(Bt)
        if i == "ip":
            vectorsLUT.append(v_Ip)
            valuesEval.append(Ip)
        if i == "rmajor":
            vectorsLUT.append(v_Rmajor)
            valuesEval.append(Rmajor)
        if i == "kappa":
            vectorsLUT.append(v_kappa)
            valuesEval.append(kappa)
        if i == "delta":
            vectorsLUT.append(v_delta)
            valuesEval.append(delta)
        if i == "betan":
            vectorsLUT.append(v_BetaN)
            valuesEval.append(BetaN)
        if i == "neped":
            vectorsLUT.append(v_neped)
            valuesEval.append(neped)
        if i == "zeff":
            vectorsLUT.append(v_Zeff)
            valuesEval.append(Zeff)
        if i == "epsilon":
            vectorsLUT.append(v_epsilon)
            valuesEval.append(epsilon)

    vectorsLUT = tuple(vectorsLUT)
    valuesEval = tuple(valuesEval)

    vectorsLUT_out = (v_width_top, v_netop, v_Tetop, v_presstop, v_pressped)

    # ~~~~ Fixed LuT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (LuT_fixed == 1).any():
        print("\t- Option to include fixed, non-interpolable values in LuT search")
        vectorsLUT, valuesEval_new, vectorsLUT_out = correctByNonInterpolationValues(
            vectorsLUT, vectorsLUT_out, valuesEval, LuT_fixed, LuT_variables
        )
    else:
        valuesEval_new = valuesEval

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return vectorsLUT, valuesEval_new, vectorsLUT_out


def correctByNonInterpolationValues(
    vectorsLUT, vectorsLUT_out, valuesEval, LuT_fixed, LuT_variables
):
    valuesEval_new = []
    mask = np.ones(len(vectorsLUT[0]), dtype=bool)
    iEvals = []
    for i in range(len(LuT_variables)):
        if LuT_fixed[i] == 1:
            mask = mask & (vectorsLUT[i] == valuesEval[i])
        else:
            iEvals.append(i)
            valuesEval_new.append(valuesEval[i])

    vectorsLUT_new = []
    for i in range(len(vectorsLUT)):
        if i in iEvals:
            vectorsLUT_new.append(vectorsLUT[i][mask])

    vectorsLUT_out_new = []
    for i in range(len(vectorsLUT_out)):
        vectorsLUT_out_new.append(vectorsLUT_out[i][mask])

    return tuple(vectorsLUT_new), tuple(valuesEval_new), tuple(vectorsLUT_out_new)


def search_LuT_EPED(
    PedestalType,
    LuT_loc=None,
    LuT_variables=None,
    LuT_fixed=None,
    enforceClosestShaping=True,
    Bt=12.0,
    Rmajor=1.65,
    Ip=None,
    kappa=1.8,
    delta=0.36,
    BetaN=1.35,
    neped=33.0,
    Zeff=1.5,
    epsilon=0.303,
    Aimp=22.0,
    Zimp=11.0,
    m=2.5,
    z=1.0,
    nameFolder="",
    lock=None,
):
    # ~~~~ Read LuT

    vectorsLUT, valuesEval, vectorsLUT_out = prepare_LuT_EPED(
        LuT_loc,
        LuT_variables,
        Bt,
        Rmajor,
        Ip,
        kappa,
        delta,
        BetaN,
        neped,
        epsilon,
        LuT_fixed=LuT_fixed,
        Aimp=Aimp,
        Zimp=Zimp,
        m=m,
        z=z,
        Zeff=Zeff,
        enforceClosestShaping=enforceClosestShaping,
    )

    if PedestalType == "lut":
        width = MATHtools.HighDimSearch(vectorsLUT, vectorsLUT_out[0], valuesEval)
        ne_height = MATHtools.HighDimSearch(vectorsLUT, vectorsLUT_out[1], valuesEval)
        Te_height = MATHtools.HighDimSearch(vectorsLUT, vectorsLUT_out[2], valuesEval)

        p_height = MATHtools.HighDimSearch(vectorsLUT, vectorsLUT_out[3], valuesEval)
        p1_height = MATHtools.HighDimSearch(vectorsLUT, vectorsLUT_out[4], valuesEval)

        if Ip is None:
            Ip = MATHtools.HighDimSearch(vectorsLUT, v_Ip, valuesEval)

    # if PedestalType == 'surrogate_model':

    #     runRemote = True

    #     if not runRemote:

    #         ne_height,Te_height,width = getsurrogate_model(vectorsLUT,vectorsLUT_out,valuesEval)

    #     else:
    #         '''
    #         This is recommended when running inside multiprocessing. For some unknown reason,
    #         gpytorch MLL evaluation gets stuck otherwise... so this runs the command in another machine
    #         '''

    #         print('- Option to run getsurrogate_model remotely has been requested')

    #         folderLocal     = IOtools.expandPath('~/scratch/findPRFLocal_{0}/'.format(nameFolder))
    #         folderRemote0   = 'findPRFRemote_{0}/'.format(nameFolder)
    #         folderRemote    = IOtools.expandPath('~/scratch/{0}'.format(folderRemote0))
    #         Params_in       = (vectorsLUT,vectorsLUT_out,valuesEval,folderRemote)
    #         machineSettings = CONFIGread.machineSettings(code='eq',nameScratch=folderRemote0)

    #         ne_height,Te_height,width = FARMINGtools.runFunction_Complete('getsurrogate_model',Params_in,WhereIsFunction='from im_tools.utils.LUTtools import getsurrogate_model',
    #                 machineSettings=machineSettings,scratchFolder=folderLocal)

    #     if Ip is None:
    #         ypred,GP,_,_    = BOtools.fitQuickModel(train_X,train_Y,(v_Ip))
    #         Ip          = ypred[0]

    # eV, m^-3, psi, MA
    return ne_height, Te_height, width, Ip, p_height, p1_height


def getsurrogate_model(vectorsLUT, vectorsLUT_out, valuesEval):
    # Construct training data
    train_X = []
    train_Y = []
    for i in range(len(vectorsLUT[0])):
        xp = []
        for j in range(len(vectorsLUT)):
            xp.append(vectorsLUT[j][i])
        train_X.append(xp)

        yp = []
        for j in range(len(vectorsLUT_out)):
            yp.append(vectorsLUT_out[j][i])
        train_Y.append(yp)

    train_X, train_Y = np.array(train_X), np.array(train_Y)

    # ---------------------------------------------------------------------------------------------------

    import torch
    from opt_tools.models import surrogates

    # One by one because not sure why, it doesn't work several outputs at a time. Have to fix

    i = 0
    GP, predictor, axs = surrogates.simpleModel(train_X, train_Y[:, i], plotYN=False)
    a, _ = predictor(torch.from_numpy(valuesEval).unsqueeze(0))
    width = a[0, 0]

    i = 1
    GP, predictor, axs = surrogates.simpleModel(train_X, train_Y[:, i], plotYN=False)
    a, _ = predictor(torch.from_numpy(valuesEval).unsqueeze(0))
    ne_height = a[0, 0]

    i = 2
    GP, predictor, axs = surrogates.simpleModel(train_X, train_Y[:, i], plotYN=False)
    a, _ = predictor(torch.from_numpy(valuesEval).unsqueeze(0))
    Te_height = a[0, 0]

    return ne_height, Te_height, width


def evaluateNN(
    nn_weights_loc, bt, rmajor, epsilon, ip, kappa, delta, betan, nped, zeff=2.0
):
    x = [[bt, rmajor, epsilon, ip, kappa, delta, betan, nped, zeff]]
    out = np.zeros(6)

    # x is array of eped inputs
    # x=[bt(tesla),rmajor(m),epsilon,ip(MA),kappa,delta,betan,nped(1e19),zeff]

    pres = pickle.load(open(nn_weights_loc + "/eped_network_p.sav", "rb"))
    out[0] = pres.predict(x)

    wid = pickle.load(open(nn_weights_loc + "/eped_network_wid.sav", "rb"))
    out[1] = wid.predict(x)

    ptop = pickle.load(open(nn_weights_loc + "/eped_network_ptop.sav", "rb"))
    out[2] = ptop.predict(x)

    wtop = pickle.load(open(nn_weights_loc + "/eped_network_wtop.sav", "rb"))
    out[3] = wtop.predict(x)

    dens = pickle.load(open(nn_weights_loc + "/eped_network_dens.sav", "rb"))
    out[4] = dens.predict(x) * x[0][7]

    temp = pickle.load(open(nn_weights_loc + "/eped_network_temp.sav", "rb"))
    out[5] = temp.predict(x)

    # out is an array of the outputs from the nn
    # out=[pressure(MPa),width(psinorm),pressuretop(MPa),widthtop(psinorm),densitytop(1e19),temperature(eV)]

    ne_height = out[4] * 1e19
    Te_height = out[5]
    width = out[3]

    # m^-3,eV,psi
    return ne_height, Te_height, width
