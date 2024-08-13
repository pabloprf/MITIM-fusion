import os
import math
import numpy as np

from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.transp_tools.tools import TRANSPhelpers


def defineTRANSPnmlStructures():
    limiters = [
        [104.0, 0.0, 90.0],
        [114.0, 70.0, 105.0],
        [135.0, 110.0, 130.0],
        [143.0, 114.0, 146.0],
        [163.0, 114.0, 20.0],
        [196.0, 78.0, 60.0],
        [215.0, 39.0, 70.0],
        [221.0, 0.0, 90.0],
        [213.0, -28.0, 105.0],
        [187.0, -68.0, 130.0],
        [145.0, -106.0, 0.0],
        [106.0, -30.0, 82.0],
        [123.0, -82.0, 61.0],
    ]
    VVmoms = None

    return limiters, VVmoms


def getHeatingParams(folderTRgui, shotnumber, trange=[0, 100]):
    uf = UFILEStools.UFILEtransp()
    uf.readUFILE(f"{folderTRgui}/THE{shotnumber}.ECH")
    cPol, tPol, zPol = uf.Variables["X"], uf.Variables["Y"], uf.Variables["Z"]

    uf = UFILEStools.UFILEtransp()
    uf.readUFILE(f"{folderTRgui}/PHI{shotnumber}.ECH")
    cTor, tTor, zTor = uf.Variables["X"], uf.Variables["Y"], uf.Variables["Z"]

    Polech, Torech = [], []
    for i in range(len(cPol)):
        it1, it2 = np.argmin(np.abs(tPol - trange[0])), np.argmin(
            np.abs(tPol - trange[1])
        )
        Polech.append(np.mean(zPol[i, it1:it2]))
        it1, it2 = np.argmin(np.abs(tTor - trange[0])), np.argmin(
            np.abs(tTor - trange[1])
        )
        Torech.append(np.mean(zTor[i, it1:it2]))

    Vnbi = []
    for i in range(10):
        try:
            Vnbi.append(
                float(
                    IOtools.findValue(
                        f"{folderTRgui}/{shotnumber}TR.DAT",
                        f"EINJA({i + 1})",
                        "=",
                        isitArray=False,
                    )
                )
            )
        except:
            break
    Vnbi = np.array(Vnbi) * 1e-3

    Fech = [
        float(i) * 1e-6
        for i in IOtools.findValue(
            f"{folderTRgui}/{shotnumber}TR.DAT",
            "FREQECH",
            "=",
            isitArray=True,
        ).split(",")
    ]
    Fech = np.array(Fech)

    return Polech, Torech, Vnbi, Fech


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ ASTRA to TRANSP routines
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def createFolderFromASTRA(fileN, folderUFs):
    with open(fileN, "r") as f:
        allLines = f.readlines()

    # Scalars
    convertScalarsToUFILE(folderUFs, allLines)

    # Profiles
    convertVariableToUFILE(folderUFs, "TEX", allLines, "ter", "TEL", scale=1e3)
    convertVariableToUFILE(folderUFs, "NEX", allLines, "ner", "NEL", scale=1e19 * 1e-6)
    convertVariableToUFILE(folderUFs, "TIX", allLines, "ti2", "TIO", scale=1e3)
    convertVariableToUFILE(folderUFs, "VTORX", allLines, "vp2", "VP2", scale=1e2)
    convertVariableToUFILE(folderUFs, "MUX", allLines, "qpr", "QPR", div=True)

    # Heating
    convertHeatingsToUFILE(folderUFs, allLines)

    # Boundary
    convertBoundaryToUFILE(allLines, folderUFs)


def findVariable(varName, allLines, numperline=6, thereIsRho=True):
    for i in range(len(allLines)):
        if varName in allLines[i]:
            try:
                num = int(allLines[i].split("POINTS")[1].split("GRIDTYPE")[0])
            except:
                num = int(allLines[i].split("POINTS")[1].split("NAMEXP")[0])

            lines = int(math.ceil(num / float(numperline)))

            if thereIsRho:
                rhoN = ""
                for j in range(lines):
                    rhoN += allLines[i + 2 + j]
                rho = np.array([float(p) for p in rhoN.replace("\n", "").split()])
            else:
                rho = None
                j = -1

            TeN = ""
            for k in range(lines):
                TeN += allLines[i + 2 + j + k + 1]
            Te = np.array([float(p) for p in TeN.replace("\n", "").split()])

            break

    return rho, Te


def findScalar(varNames, allLines):
    vals = []
    for varName in varNames:
        for i in range(len(allLines)):
            if varName in allLines[i]:
                try:
                    val = float(allLines[i].split()[2])
                except:
                    val = float(allLines[i].split()[1])
                break
        vals.append(val)
    return np.array(vals)


def convertScalarsToUFILE(folder, allLines, shotnum="54321"):
    Ip = findScalar(["IPL"], allLines)[0]
    UF = UFILEStools.UFILEtransp(scratch="cur")
    UF.Variables["Z"] = Ip * 1e6
    UF.repeatProfile()
    UF.writeUFILE(folder + f"OMF{shotnum}.CUR")

    Rmajor = findScalar(["RTOR"], allLines)[0]
    Bt = findScalar(["BTOR"], allLines)[0]
    UF = UFILEStools.UFILEtransp(scratch="rbz")
    UF.Variables["Z"] = Rmajor * 100.0 * Bt
    UF.repeatProfile()
    UF.writeUFILE(folder + f"OMF{shotnum}.RBZ")

    Vsurf = 0.0
    UF = UFILEStools.UFILEtransp(scratch="vsf")
    UF.Variables["Z"] = Vsurf
    UF.repeatProfile()
    UF.writeUFILE(folder + f"OMF{shotnum}.VSF")


def convertHeatingsToUFILE(folder, allLines, shotnum="54321"):
    # ECH
    Pech = findScalar(
        ["ZRD11", "ZRD12", "ZRD13", "ZRD14", "ZRD15", "ZRD16", "ZRD17", "ZRD18"],
        allLines,
    )
    timesOn = np.repeat([[0.0, 10.0]], 10, axis=0)
    UFILEStools.writeAntenna(
        folder + f"OMF{shotnum}.ECH",
        timesOn,
        Pech,
        fromScratch="ecp",
        flipAxis=True,
    )

    # NBI

    timesOn = np.repeat([[0.0, 10.0]], 10, axis=0)
    UFILEStools.writeAntenna(
        folder + f"OMF{shotnum}.NB2",
        timesOn,
        Pnbi,
        fromScratch="nb2",
        flipAxis=True,
    )


def convertVariableToUFILE(
    folder, varName, allLines, UFtype, UFname, scale=1.0, div=False, shotnum="54321"
):
    rho, var = findVariable(varName, allLines)

    UF = UFILEStools.UFILEtransp(scratch=UFtype)
    UF.Variables["X"] = rho
    if div:
        var = 1.0 / var
    UF.Variables["Z"] = var * scale
    UF.repeatProfile()
    # pdb.set_trace()
    UF.writeUFILE(folder + f"OMF{shotnum}.{UFname}")


def convertBoundaryToUFILE(allLines, folder):
    _, RZ = findVariable("BND", allLines, numperline=1, thereIsRho=False)

    R = []
    Z = []
    for i in range(len(RZ) / 2 - 1):
        R.append(RZ[i * 2])
        Z.append(RZ[i * 2 + 1])

    R = np.array(R)
    Z = np.array(Z)

    name = folder + "/BOUNDARY_123456_01000.DAT"

    # Write the file for scruncher
    with open(name, "w") as f:
        f.write("Boundary description for timeslice 123456 1000 msec\n")
        f.write("Shot date: 29AUG-2018\n")
        f.write("UNITS ARE METERS\n")
        f.write(f"Number of points: {len(R)}\n")

        f.write(
            "Begin R-array ==================================================================\n"
        )
        writeLines(f, R, perline=10)

        f.write(
            "Begin z-array ==================================================================\n"
        )
        writeLines(f, Z, perline=10)

    os.system(f"cp {name} {folder}/BOUNDARY_123456_10000.DAT")

    # MRY
    TRANSPhelpers.generateMRY(folder, ["01000", "10000"], folder, "54321")


def writeLines(f, z, perline=10):
    numRows = int(math.ceil(len(z) / float(perline)))
    for i in range(numRows):
        stri = ""
        for j in range(perline):
            try:
                stri += f"{z[i * perline + j]:7.3f} "
            except:
                break
        f.write(stri + "\n")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ AUG parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def defineFirstWall():
    r = [
        1.16611e00,
        1.18605e00,
        1.20266e00,
        1.22924e00,
        1.23920e00,
        1.26578e00,
        1.28571e00,
        1.32226e00,
        1.33223e00,
        1.38206e00,
        1.42525e00,
        1.46844e00,
        1.50498e00,
        1.66113e00,
        1.77741e00,
        1.77409e00,
        1.86379e00,
        2.06977e00,
        2.09967e00,
        2.10299e00,
        2.11960e00,
        2.17608e00,
        2.19601e00,
        2.19934e00,
        2.18605e00,
        2.16611e00,
        2.14286e00,
        2.10631e00,
        2.06645e00,
        2.01661e00,
        1.99003e00,
        1.99003e00,
        1.75083e00,
        1.75083e00,
        1.72425e00,
        1.69435e00,
        1.63787e00,
        1.58140e00,
        1.56478e00,
        1.55482e00,
        1.46179e00,
        1.32226e00,
        1.27907e00,
        1.24585e00,
        1.23588e00,
        1.28571e00,
        1.28904e00,
        1.27575e00,
        1.23588e00,
        1.19934e00,
        1.13953e00,
        1.11296e00,
        1.08970e00,
        1.06312e00,
        1.04983e00,
        1.04319e00,
        1.04651e00,
        1.06312e00,
        1.08306e00,
        1.10963e00,
        1.14950e00,
        1.16279e00,
        1.16611e00,
    ]

    z = [
        8.00000e-01,
        8.33333e-01,
        8.83333e-01,
        9.36667e-01,
        9.66667e-01,
        1.00000e00,
        1.02667e00,
        1.07667e00,
        1.09333e00,
        1.12000e00,
        1.14333e00,
        1.17000e00,
        1.18667e00,
        1.13000e00,
        1.08667e00,
        1.06667e00,
        9.40000e-01,
        5.96667e-01,
        5.46667e-01,
        5.13333e-01,
        4.73333e-01,
        3.23333e-01,
        2.13333e-01,
        2.66667e-02,
        -7.33333e-02,
        -1.60000e-01,
        -2.36667e-01,
        -3.16667e-01,
        -3.96667e-01,
        -4.76667e-01,
        -5.16667e-01,
        -5.53333e-01,
        -8.26667e-01,
        -8.40000e-01,
        -8.43333e-01,
        -8.70000e-01,
        -9.86667e-01,
        -1.20667e00,
        -1.20667e00,
        -1.17333e00,
        -1.06333e00,
        -1.06000e00,
        -1.09667e00,
        -1.14333e00,
        -1.12667e00,
        -9.80000e-01,
        -9.33333e-01,
        -8.76667e-01,
        -8.03333e-01,
        -7.46667e-01,
        -6.56667e-01,
        -5.80000e-01,
        -4.80000e-01,
        -3.16667e-01,
        -1.80000e-01,
        -7.33333e-02,
        9.66667e-02,
        3.06667e-01,
        4.43333e-01,
        5.73333e-01,
        7.16667e-01,
        7.70000e-01,
        8.00000e-01,
    ]

    return r, z


def ECRFgyrotrons():
    """
    For the gyros, the frequencies FREQECH specified here are unimportant. It will be changed by the
    mitim.baseline specification. However, the rest of parameters (geometry, positions) are important
    and need to be accurate here.
    Poloidal and toroidal angles are given in UFILES THE and PHI respectively.
    """

    lines = [
        "",
        "! ~~~~~~~~~~ Gyrotrons set-up",
        "",
        "NLAUNCHECH 	= 1",
        "NANTECH        = 8",
        "",
        "XECECH         = 238., 238., 231., 231., 236.1, 236.1, 236.1, 236.1",
        "ZECECH         = 0., 0., 0., 0., 32., 32., -32., -32.",
        "FREQECH        =  1.398e+11, 1.398e+11, 1.398e+11, 1.398e+11, 1.400e+11, 1.400e+11, 1.400e+11, 1.400e+11",
        "RFMODECH       = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
        "ECB_WIDTH_HOR  =    3.0,    3.0,    3.0,    3.0,    2.5,    2.5,    2.5,    2.5",
        "ECB_WIDTH_VERT =    3.0,    3.0,    3.0,    3.0,    2.5,    2.5,    2.5,    2.5",
        "ECB_CURV_HOR   =  129.4,  129.4,  129.4,  129.4,   85.0,   85.0,   85.0,   85.0",
        "ECB_CURV_VERT  =  129.4,  129.4,  129.4,  129.4,   85.0,   85.0,   85.0,   85.0",
        "",
        "!==================================================================",
        "",
    ]

    return "\n".join(lines)


def NBIbeams():
    """
    For the beams, the voltages EINJA specified here are unimportant. It will be changed by the
    mitim.baseline specification. However, the rest of parameters (geometry, positions) are important
    and need to be accurate here
    """

    lines = """

nbeam = 8

! ~~~~~~~~~~ Beamlines set-up
!==================================================================

! ~~~~~~~~~~ Beam #1

NLCO(1) =  T 
XBZETA(1) = 318.6794 
RTCENA(1) = 53.5608 
XLBTNA(1) = 934.1600 
XYBSCA(1) = 60.0000 

NBSHAPA(1) = 1 
BMWIDRA(1) = 11.0000 
BMWIDZA(1) = 25.0000 
DIVRA(1) = 0.0099 
DIVZA(1) = 0.0099 
FOCLRA(1) = 650.0000 
FOCLZA(1) = 850.0000 

NBAPSHA(1) = 1 
XLBAPA(1) = 590.3171 
XYBAPA(1) = 9.7166 
RAPEDGA(1) = 30.0000 
XZPEDGA(1) = 41.4200 
XRAPOFFA(1) = 10.8710 
XZAPOFFA(1) = -9.7460 

NBAPSH2(1) = 1 
XLBAPA2(1) = 665.3952 
RAPEDG2(1) = 30.0000 
XZPEDG2(1) = 46.1440 
XRAPOFF2(1) = -16.3080 
XZAPOFF2(1) = -0.8710 

ABEAMA(1) = 2.000000
XZBEAMA(1)= 1.000000
EINJA(1)  =  58512.863
FFULLA(1) =      0.469
FHALFA(1) =      0.364

! ~~~~~~~~~~ Beam #2

NLCO(2) =  T 
XBZETA(2) = 312.8954 
RTCENA(2) = 93.1688 
XLBTNA(2) = 923.5200 
XYBSCA(2) = 60.0000 

NBSHAPA(2) = 1 
BMWIDRA(2) = 11.0000 
BMWIDZA(2) = 25.0000 
DIVRA(2) = 0.0099 
DIVZA(2) = 0.0099 
FOCLRA(2) = 650.0000 
FOCLZA(2) = 850.0000 

NBAPSHA(2) = 1 
XLBAPA(2) = 587.3534 
XYBAPA(2) = 9.9690 
RAPEDGA(2) = 30.0000 
XZPEDGA(2) = 41.1810 
XRAPOFFA(2) = 19.9540 
XZAPOFFA(2) = -9.9850 

NBAPSH2(2) = 1 
XLBAPA2(2) = 667.2220 
RAPEDG2(2) = 30.0000 
XZPEDG2(2) = 48.3420 
XRAPOFF2(2) = -18.4300 
XZAPOFF2(2) = -0.2610 

ABEAMA(2) = 2.000000
XZBEAMA(2)= 1.000000
EINJA(2)  =  60000.000
FFULLA(2) =      0.469
FHALFA(2) =      0.364

! ~~~~~~~~~~ Beam #3

NLCO(3) =  T 
XBZETA(3) = 312.8894 
RTCENA(3) = 94.0390 
XLBTNA(3) = 923.4400 
XYBSCA(3) = -60.0000 

NBSHAPA(3) = 1 
BMWIDRA(3) = 11.0000 
BMWIDZA(3) = 25.0000 
DIVRA(3) = 0.0099 
DIVZA(3) = 0.0099 
FOCLRA(3) = 650.0000 
FOCLZA(3) = 850.0000 

NBAPSHA(3) = 1 
XLBAPA(3) = 587.3427 
XYBAPA(3) = -9.9700 
RAPEDGA(3) = 30.0000 
XZPEDGA(3) = 40.4920 
XRAPOFFA(3) = 19.4380 
XZAPOFFA(3) = 9.2160 

NBAPSH2(3) = 1 
XLBAPA2(3) = 667.9514 
RAPEDG2(3) = 30.0000 
XZPEDG2(3) = 48.2250 
XRAPOFF2(3) = -18.9290 
XZAPOFF2(3) = 0.3740 

ABEAMA(3) = 2.000000
XZBEAMA(3)= 1.000000
EINJA(3)  =  59174.336
FFULLA(3) =      0.469
FHALFA(3) =      0.364

! ~~~~~~~~~~ Beam #4

NLCO(4) =  T 
XBZETA(4) = 318.6794 
RTCENA(4) = 53.5608 
XLBTNA(4) = 934.1600 
XYBSCA(4) = -60.0000 

NBSHAPA(4) = 1 
BMWIDRA(4) = 11.0000 
BMWIDZA(4) = 25.0000 
DIVRA(4) = 0.0099 
DIVZA(4) = 0.0099 
FOCLRA(4) = 650.0000 
FOCLZA(4) = 850.0000 

NBAPSHA(4) = 1 
XLBAPA(4) = 590.3171 
XYBAPA(4) = -9.7166 
RAPEDGA(4) = 30.0000 
XZPEDGA(4) = 40.4600 
XRAPOFFA(4) = 10.8710 
XZAPOFFA(4) = 8.8590 

NBAPSH2(4) = 1 
XLBAPA2(4) = 666.0908 
RAPEDG2(4) = 30.0000 
XZPEDG2(4) = 46.3310 
XRAPOFF2(4) = -16.1060 
XZAPOFF2(4) = 0.6840 

ABEAMA(4) = 2.000000
XZBEAMA(4)= 1.000000
EINJA(4)  =  58904.281
FFULLA(4) =      0.469
FHALFA(4) =      0.364

! ~~~~~~~~~~ Beam #5

NLCO(5) =  T 
XBZETA(5) = 131.6736 
RTCENA(5) = 85.5282 
XLBTNA(5) = 973.8400 
XYBSCA(5) = 60.0927 

NBSHAPA(5) = 1 
BMWIDRA(5) = 11.0000 
BMWIDZA(5) = 25.0000 
DIVRA(5) = 0.0099 
DIVZA(5) = 0.0099 
FOCLRA(5) = 723.0000 
FOCLZA(5) = 1194.0000 

NBAPSHA(5) = 2 
XLBAPA(5) = 492.8980 
XYBAPA(5) = 17.5933 
RAPEDGA(5) = 39.7500 
XZPEDGA(5) = -1.0000
XRAPOFFA(5) = -12.0870 
XZAPOFFA(5) = -17.8450 

NBAPSH2(5) = 1 
XLBAPA2(5) = 735.9146 
RAPEDG2(5) = 25.0000 
XZPEDG2(5) = 39.6960 
XRAPOFF2(5) = 2.4580 
XZAPOFF2(5) = 2.2690 

ABEAMA(5) = 2.000000
XZBEAMA(5)= 1.000000
EINJA(5)  =  93000.000
FFULLA(5) =      0.434
FHALFA(5) =      0.390

! ~~~~~~~~~~ Beam #6

NLCO(6) =  T 
XBZETA(6) = 126.1698 
RTCENA(6) = 125.9036 
XLBTNA(6) =    961.516
XYBSCA(6) = 69.4550 

NBSHAPA(6) = 1 
BMWIDRA(6) = 11.0000 
BMWIDZA(6) = 25.0000 
DIVRA(6) = 0.0099 
DIVZA(6) = 0.0099 
FOCLRA(6) = 723.0000 
FOCLZA(6) = 1194.0000 

NBAPSHA(6) = 1 
XLBAPA(6) =    656.114
XYBAPA(6) =     -5.807
RAPEDGA(6) = 25.0000 
XZPEDGA(6) = 41.0160 
XRAPOFFA(6) = 14.0400 
XZAPOFFA(6) = 0.4080 

NBAPSH2(6) = 1 
XLBAPA2(6) = 741.8481 
RAPEDG2(6) = 25.0000 
XZPEDG2(6) = 39.5900 
XRAPOFF2(6) = -6.7300 
XZAPOFF2(6) = 9.5430 

ABEAMA(6) = 2.000000
XZBEAMA(6)= 1.000000
EINJA(6)  =  93000.000
FFULLA(6) =      0.434
FHALFA(6) =      0.390

! ~~~~~~~~~~ Beam #7

NLCO(7) =  T 
XBZETA(7) = 126.1624 
RTCENA(7) = 127.4349 
XLBTNA(7) =    961.515
XYBSCA(7) = -69.6733 

NBSHAPA(7) = 1 
BMWIDRA(7) = 11.0000 
BMWIDZA(7) = 25.0000 
DIVRA(7) = 0.0099 
DIVZA(7) = 0.0099 
FOCLRA(7) = 723.0000 
FOCLZA(7) = 1194.0000 

NBAPSHA(7) = 1 
XLBAPA(7) =    656.114
XYBAPA(7) =      5.805
RAPEDGA(7) = 25.0000 
XZPEDGA(7) = 41.0910 
XRAPOFFA(7) = 13.0360 
XZAPOFFA(7) = -2.4190 

NBAPSH2(7) = 1 
XLBAPA2(7) = 741.6901 
RAPEDG2(7) = 25.0000 
XZPEDG2(7) = 39.5540 
XRAPOFF2(7) = -7.8660 
XZAPOFF2(7) = -11.6470 

ABEAMA(7) = 2.000000
XZBEAMA(7)= 1.000000
EINJA(7)  =  93000.000
FFULLA(7) =      0.434
FHALFA(7) =      0.390

! ~~~~~~~~~~ Beam #8

NLCO(8) =  T
XBZETA(8) = 131.6737 
RTCENA(8) = 85.5283 
XLBTNA(8) = 973.8400 
XYBSCA(8) = -60.1545 

NBSHAPA(8) = 1 
BMWIDRA(8) = 11.0000 
BMWIDZA(8) = 25.0000 
DIVRA(8) = 0.0099 
DIVZA(8) = 0.0099 
FOCLRA(8) = 723.0000 
FOCLZA(8) = 1194.0000 

NBAPSHA(8) = 2 
XLBAPA(8) = 492.8803 
XYBAPA(8) = -17.3138 
RAPEDGA(8) = 39.7500 
XZPEDGA(8) = -1.0000
XRAPOFFA(8) = -12.0860 
XZAPOFFA(8) = 17.5670 

NBAPSH2(8) = 1 
XLBAPA2(8) = 735.8206 
RAPEDG2(8) = 25.0000 
XZPEDG2(8) = 39.6580 
XRAPOFF2(8) = 2.4540 
XZAPOFF2(8) = -3.0340 

ABEAMA(8) = 2.000000
XZBEAMA(8)= 1.000000
EINJA(8)  =  93375.594
FFULLA(8) =      0.434
FHALFA(8) =      0.390

!==================================================================

	"""

    return lines
