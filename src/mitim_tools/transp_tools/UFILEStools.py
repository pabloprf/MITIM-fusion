import re
import copy
import numpy as np
import matplotlib.pyplot as plt

try:
    from IPython import embed
except:
    pass

from mitim_tools.misc_tools import IOtools, GRAPHICStools


class UFILEtransp:
    def __init__(self, scratch=None, rhos=np.linspace(0, 1, 100), labelX=None):
        if scratch is not None:
            self.initializeBaseUFILE(scratch, rhos=rhos, labelX=labelX)

        self.VarLabelX = "Time"

    def readContHeader(self, lines):
        contL = 3
        while "INDEPENDENT VARIABLE" not in lines[contL]:
            contL += 1

        posEnd = 0
        while (
            "END-OF-DATA" not in lines[posEnd] and "index_filename" not in lines[posEnd]
        ):
            posEnd += 1

        return contL, posEnd

    def readUFILE(self, filename):
        """
        Notes:
            This will read the coordinates and will shift them if needed so that Variables['Y'] is always time, leaving
            rho or channel as 'X'. If a file is read, and then written (see functions in class below), then the coordinates
            order may change. The file written could be different regardless of data being changed.
        """

        with open(filename, "r") as f:
            allLines = f.readlines()

        # --- Read Header Lines

        contH, posEnd = self.readContHeader(allLines)

        self.STR_header = allLines[:contH]

        self.STR_labelX = allLines[contH]
        self.labelX = self.STR_labelX.strip().split(":")[-1].split("-")[0]
        self.STR_footer = allLines[posEnd:]  # -1]

        if "INDEPENDENT VARIABLE" in allLines[contH + 1]:
            self.dim = 2
            self.STR_labelY = allLines[contH + 1]
            self.labelY = self.STR_labelY.strip().split(":")[-1].split("-")[0]
            cont = contH + 1

            if "INDEPENDENT VARIABLE" in allLines[contH + 2]:
                self.dim = 3
                self.STR_labelQ = allLines[contH + 2]
                self.labelQ = self.STR_labelQ.strip().split(":")[-1].split("-")[0]
                cont = contH + 2

        else:
            self.dim = 1
            cont = contH

        self.STR_labelZ = allLines[cont + 1]
        self.STR_proc = allLines[cont + 2]

        self.STR_numX = allLines[cont + 3]
        self.numX = int(self.STR_numX.split()[0])

        if self.dim > 1:
            self.STR_numY = allLines[cont + 4]
            self.numY = int(self.STR_numY.split()[0])
            cont += 1
            self.numZ = self.numX * self.numY

            if self.dim == 3:
                self.STR_numQ = allLines[cont + 4]
                self.numQ = int(self.STR_numQ.split()[0])
                cont += 1
                self.numZ = self.numX * self.numY * self.numQ
        else:
            self.numZ = self.numX

        # --- Read Variables
        allnums = []
        for i in allLines[cont + 4 : posEnd]:
            valy = self.extractNumbersFromLine(i)
            allnums.extend(valy)

        self.Variables = {}
        self.Variables["X"] = np.array(allnums[: self.numX])
        contv = self.numX
        if self.dim > 1:
            self.Variables["Y"] = np.array(allnums[contv : contv + self.numY])
            contv = contv + self.numY
            if self.dim == 3:
                self.Variables["Q"] = np.array(allnums[contv : contv + self.numQ])
                contv = contv + self.numQ
        Zall = allnums[contv : contv + self.numZ]

        if self.dim == 1:
            self.Variables["Z"] = np.array(Zall)
        elif self.dim == 2:
            self.Variables["Z"] = (
                np.array(Zall).reshape((self.numY, self.numX)).transpose()
            )
        else:
            self.Variables["Z"] = (
                np.array(Zall).reshape((self.numQ, self.numY, self.numX)).transpose()
            )

        # ----- Shift so that time is always second

        self.shiftVariables()

    def extractNumbersFromLine(self, line):
        nums = line.split("\n")[0].split()
        for i in range(len(nums)):
            if nums[i] == "nan":
                print(" !!!!!! nan found in UFILE")
                nums[i] = "0.0e+00"

        vecy = "".join(nums)

        pose = [m.start() for m in re.finditer("e", vecy)]
        posE = [m.start() for m in re.finditer("E", vecy)]
        pose.extend(posE)

        valy = []
        cont = 0
        for i in pose:
            valy.append(float(vecy[cont : i + 4]))
            cont = i + 4

        return valy

    def shiftVariables(self):
        # Make sure that time is always second

        self.VarLabelX = self.STR_labelX.split()[0]

        if (
            self.dim == 2
            and ("rho" not in self.VarLabelX)
            and ("r/a" not in self.VarLabelX)
            and ("X" not in self.VarLabelX)
            and ("phi" not in self.VarLabelX)
            and ("Channel" not in self.VarLabelX)
        ):
            print("Shifting variables in UFILES read process")

            # Change labels
            aux = self.STR_labelX
            self.STR_labelX = self.STR_labelY
            self.STR_labelY = aux

            aux = self.STR_numX
            self.STR_numX = self.STR_numY
            self.STR_numY = aux
            self.numY = int(self.STR_numY.split()[0])
            self.numX = int(self.STR_numX.split()[0])
            self.labelY = self.STR_labelY.strip().split(":")[-1].split("-")[0]
            self.labelX = self.STR_labelX.strip().split(":")[-1].split("-")[0]

            # Change Variables
            aux = self.Variables["X"]
            self.Variables["X"] = self.Variables["Y"]
            self.Variables["Y"] = aux
            self.Variables["Z"] = self.Variables["Z"].transpose()

            self.VarLabelX = "Time"

    def repeatProfile(self):
        if self.dim == 1:
            varPermut = "X"
        elif self.dim == 2:
            varPermut = "Y"

        aux = []
        for i in range(len(self.Variables[varPermut])):
            aux.append(self.Variables["Z"])
        self.Variables["Z"] = np.array(aux)

        if self.dim == 2:
            self.Variables["Z"] = np.transpose(self.Variables["Z"])

    def writeUFILE(self, filename, orderZvariable="C"):
        """
        For this to work correctly, the Z array must have (time,rho), which should have happened automatically
        when reading (through the shiftingVariables method). I think...
        """

        # Check whether dimensions have changed
        self.numX = len(self.Variables["X"])
        if self.dim >= 2:
            self.numY = len(self.Variables["Y"])
        if self.dim == 3:
            self.numQ = len(self.Variables["Q"])

        with open(filename, "w") as f:
            # Header
            f.write("".join(self.STR_header))

            # Variable labels
            f.write(self.STR_labelX)
            if self.dim > 1:
                f.write(self.STR_labelY)
            if self.dim == 3:
                f.write(self.STR_labelQ)
            f.write(self.STR_labelZ)
            f.write(self.STR_proc)

            # Labels with number of points
            f.write(
                "{0}                    ;-# OF {1} PTS-\n".format(
                    str(self.numX).rjust(11), self.labelX
                )
            )
            if self.dim > 1:
                f.write(
                    "{0}                    ;-# OF {1} PTS-\n".format(
                        str(self.numY).rjust(11), self.labelY
                    )
                )
            if self.dim == 3:
                f.write(
                    "{0}                    ;-# OF {1} PTS-\n".format(
                        str(self.numQ).rjust(11), self.labelQ
                    )
                )

            # ------------------
            # Variables
            # ------------------

            # ~~~~~~ Write X variable
            self.writeVar(f, self.Variables["X"])
            if self.dim > 1:
                # ~~~~~~ Write Y variable
                self.writeVar(f, self.Variables["Y"])
                # ~~~~~~ Prepare Z variable
                if self.dim == 3:
                    self.writeVar(f, self.Variables["Q"])
                    Zall = np.transpose(self.Variables["Z"]).reshape(
                        self.numX * self.numY * self.numQ
                    )
                else:
                    Zall = (
                        self.Variables["Z"]
                        .transpose()
                        .reshape(self.numX * self.numY, order=orderZvariable)
                    )
                timepoints = len(self.Variables["Y"])
            else:
                Zall = self.Variables["Z"]
                timepoints = len(self.Variables["X"])
            # ~~~~~~ Write Z variable
            self.writeVar(f, Zall)

            # Footer
            f.write("".join(self.STR_footer))

        print(
            "\t\t- UFILE written with {1} time points: ...{0}".format(
                filename[np.max([-40, -len(filename)]) :], timepoints
            )
        )

    def writeVar(self, f, var, ncols=6):
        n, cont = len(var), 0

        while cont < n:
            vec = var[cont : cont + ncols]

            vecS = []
            for j in vec:
                if np.abs(j) < 1.0e-30 or IOtools.isAnyNan(j):
                    j = 0
                vecS.append(f"{j:.6e}".rjust(13))
            lineToWrite = "".join(vecS) + "\n"

            cont += ncols

            f.write(lineToWrite)

    def plot(self,         
        ax=None,
        axis="X",
        axisSlice="Y",
        lw=1.0,
        ms=1.0,
        alpha=1.0,
        label=""
        ):

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots()

        colors = GRAPHICStools.listColors()

        for i in range(len(self.Variables[axisSlice])):
            self.plotVar(ax=ax, axis=axis, axisSlice=axisSlice, val=self.Variables[axisSlice][i], lw=lw, ms=ms, alpha=alpha,
                label = label + f" {self.Variables[axisSlice][i]:.2f}", color = colors[i])

        ax.set_xlabel(axis)
        ax.legend()
        GRAPHICStools.addDenseAxis(ax)


    def plotVar(
        self,
        ax=None,
        axis="X",
        axisSlice="Y",
        val=0.0,
        lw=1.0,
        ms=1.0,
        alpha=1.0,
        label="",
        color = 'b'
    ):
        import matplotlib.pyplot as plt

        x = np.array(self.Variables[axis])
        z = np.array(self.Variables["Z"])

        if self.dim == 2:
            ind = np.argmin(np.abs(np.array(self.Variables[axisSlice]) - val))
            if axisSlice == "X":
                z = z[ind, :]
            else:
                z = z[:, ind]

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots()
        ax.plot(x, z, "-o", markersize=ms, lw=lw, alpha=alpha, label=label, color=color)

    def addEQheader(self, BtSign=-1, IpSign=-1):
        Btstr = f"{BtSign:.4e}".rjust(11)
        Ipstr = f"{IpSign:.4e}".rjust(11)

        self.STR_header = "\n".join(
            [
                " 123456None {0} 0 6              ;-SHOT #- F(X) DATA WRITEUF OMFIT".format(
                    self.dim
                ),
                "                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM",
                "   2                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-",
                f" {Btstr}                   ;-SCALAR, LABEL FOLLOWS:",
                " BTOR_CCW: Dir. of B",
                f" {Ipstr}                   ;-SCALAR, LABEL FOLLOWS:",
                " ITOR_CCW: Dir. of I",
                "",
            ]
        )

    def addImpurityheader(self, Z, A):
        self.STR_header = "\n".join(
            [
                " 123456None {0} 0 6              ;-SHOT #- F(X) DATA WRITEUF OMFIT".format(
                    self.dim
                ),
                "                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM",
                "   2                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-",
                f" {A}                   ;-SCALAR, LABEL FOLLOWS:",
                " A",
                f" {Z}                   ;-SCALAR, LABEL FOLLOWS:",
                " Z",
                "",
            ]
        )

    def initializeBaseUFILE(
        self,
        varLabel,
        rhos=np.linspace(0, 1, 100),
        times=[0.0, 100.0],
        moms=[7, 4],
        thetas=np.linspace(0, 2 * np.pi, 100),
        labelX=None,
    ):
        if labelX is None:
            labelX = " rho_tor                       "

        if varLabel in ["cur", "rbz", "vsf", "zef", "ntx", "gas", "gfd", "saw"]:
            self.dim = 1
            self.STR_labelX = (
                " Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X0-\n"
            )
            self.labelX = "X0"
            self.Variables = {"X": times, "Z": np.zeros(len(times))}
        elif varLabel in [
            "df4",
            "ner",
            "nzr",
            "qpr",
            "ter",
            "ti2",
            "vc4",
            "bol",
            "vp2",
            "omg",
            "ni4",
            "sim",
            "zf2",
            "d2f",
            "v2f",
            "nmr",
            "lhe",
            "lhj",
        ]:
            self.dim = 2
            self.STR_labelX = f"{labelX};-INDEPENDENT VARIABLE LABEL: X0-\n"
            self.STR_labelY = (
                " Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X1-\n"
            )
            self.labelX = "X0"
            self.labelY = "X1"
            self.Variables = {
                "X": rhos,
                "Y": times,
                "Z": np.zeros([len(rhos), len(times)]),
            }
        elif varLabel in ["mry"]:
            self.dim = 3
            self.STR_labelX = (
                " TIME                SECONDS   ;-INDEPENDENT VARIABLE LABEL: X-\n"
            )
            self.STR_labelY = (
                " MOMENT INDEX                  ;-INDEPENDENT VARIABLE LABEL: Y-\n"
            )
            self.STR_labelQ = (
                " TERM  INDEX                   ;-INDEPENDENT VARIABLE LABEL: Z-\n"
            )
            self.labelX = "X"
            self.labelY = "Y"
            self.labelQ = "Z"
            self.Variables = {
                "X": times,
                "Y": np.linspace(0, 6, moms[0]),
                "Q": np.linspace(0, 6, moms[1]),
                "Z": np.zeros([len(times), moms[0], moms[1]]),
            }
        elif varLabel in ["rfs", "zfs"]:
            self.dim = 3
            self.STR_labelX = (
                " Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X-\n"
            )
            self.STR_labelY = (
                " poloidal angle      rad       ;-INDEPENDENT VARIABLE LABEL: Y-\n"
            )
            self.STR_labelQ = f"{labelX};-INDEPENDENT VARIABLE LABEL: Z-\n"
            self.labelX = "X"
            self.labelY = "Y"
            self.labelQ = "Z"
            self.Variables = {
                "X": times,
                "Y": thetas,
                "Q": rhos,
                "Z": np.zeros([len(times), len(thetas), len(rhos)]),
            }
        elif varLabel in ["nb2", "ecp", "rfp", "the", "phi"]:
            self.dim = 2
            self.STR_labelX = (
                " Channel number                ;-INDEPENDENT VARIABLE LABEL: X0-\n"
            )
            self.STR_labelY = (
                " Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X1-\n"
            )
            self.labelX = "X0"
            self.labelY = "X1"
            self.Variables = {
                "X": rhos,
                "Y": times,
                "Z": np.zeros([len(rhos), len(times)]),
            }
        elif varLabel in ["lim"]:
            self.dim = 1
            self.STR_labelX = (
                " R of limiter contourm         ;-INDEPENDENT VARIABLE LABEL: X0-\n"
            )
            self.labelX = "X0"
            self.Variables = {"X": times, "Z": np.zeros(len(times))}

        self.STR_header = "\n".join(
            [
                " 900052D3D  {0} 0 6              ;-SHOT #- F(X) DATA WRITEUF OMFIT".format(
                    self.dim
                ),
                "                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM",
                "   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-",
                "",
            ]
        )
        self.STR_proc = (
            " 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM 3:AVG+SM\n"
        )
        self.STR_footer = " ;----END-OF-DATA-----------------COMMENTS:-----------;"

        if varLabel == "cur":
            STR_labelZ = " Plasma Current      Amps      ;"
        elif varLabel == "df4":
            STR_labelZ = " He4++ Diffusivity   cm**2/sec ;"
        elif varLabel == "ner":
            STR_labelZ = " Electron Density    cm**-3    ;"
        elif varLabel == "nzr":
            STR_labelZ = " Impurity Density    cm**-3    ;"
        elif varLabel == "qpr":
            STR_labelZ = " EFIT q profile                ;"
        elif varLabel == "rbz":
            STR_labelZ = " Rp*Bt               T.cm      ;"
        elif varLabel == "ter":
            STR_labelZ = " Electron Temp       eV        ;"
        elif varLabel == "ti2":
            STR_labelZ = " Ion Temp            eV        ;"
        elif varLabel == "vc4":
            STR_labelZ = " He4++ v(convective) cm/sec    ;"
        elif varLabel == "vsf":
            STR_labelZ = " Vloop               V         ;"
        elif varLabel == "mry":
            STR_labelZ = " Rm Ym MOMS          CM        ;"
        elif varLabel == "bol":
            STR_labelZ = " BOLOMETER/POWER RADIWATTS/CM3 ;"
        elif varLabel == "nb2":
            STR_labelZ = " NBI power           W         ;"
        elif varLabel == "ecp":
            STR_labelZ = " ECRH power          W         ;"
        elif varLabel == "vp2":
            STR_labelZ = " Plasma Rotation     cm/sec    ;"
        elif varLabel == "omg":
            STR_labelZ = " Plasma Rotation     rad/sec   ;"
        elif varLabel == "zef":
            STR_labelZ = " Zeff                          ;"
        elif varLabel == "ntx":
            STR_labelZ = " Neutron rate                  ;"
        elif varLabel == "rfp":
            STR_labelZ = " ICRF INPUT POWER    WATTS     ;"
        elif varLabel == "ni4":
            STR_labelZ = " Helium-4 Density    cm**-3    ;"
        elif varLabel == "sim":
            STR_labelZ = " Impurity Density    cm**-3    ;"
        elif varLabel == "lim":
            STR_labelZ = " Z of limiter contourm         ;"
        elif varLabel == "zf2":
            STR_labelZ = " Zeff profile data             ;"
        elif varLabel == "the":
            STR_labelZ = " ECRH pol. angle     deg       ;"
        elif varLabel == "phi":
            STR_labelZ = " ECRH tor. angle     deg       ;"
        elif varLabel == "d2f":
            STR_labelZ = " Fast Diffusivity    cm**2/sec ;"
        elif varLabel == "v2f":
            STR_labelZ = " Fast v(convective)     cm/sec ;"
        elif varLabel == "nmr":
            STR_labelZ = " Minority Density    cm**-3    ;"
        elif varLabel == "gas":
            STR_labelZ = " Gas flow rate       /sec      ;"
        elif varLabel == "gfd":
            STR_labelZ = " D gas flow rate       /sec    ;"
        elif varLabel == "lhe":
            STR_labelZ = " Power density       WATTS/CM3 ;"
        elif varLabel == "lhj":
            STR_labelZ = " Current density         A/CM2 ;"
        elif varLabel == "rfs":
            STR_labelZ = " R(theta,x) surfaces m         ;"
        elif varLabel == "zfs":
            STR_labelZ = " Z(theta,x) surfaces m         ;"
        elif varLabel == "saw":
            STR_labelZ = " Sawtooth Times      s      ;"

        self.STR_labelZ = STR_labelZ + "-DEPENDENT VARIABLE LABEL-\n"


def writeAntenna(fileUF, timesOn, PowersMW, PowersMW_Before=None, fromScratch=None):
    timeStep = 1e-3

    numAntennas = len(PowersMW)

    if fromScratch is None:
        UF = UFILEtransp()
        UF.readUFILE(fileUF)
    else:
        UF = UFILEtransp(scratch=fromScratch)

    fullTimeOff = np.max(np.array(timesOn[:, 1]))

    timeArr = np.arange(0, fullTimeOff + 0.1, timeStep)

    print(f"\t- Found {numAntennas} hardware item(s)")

    UF.Variables["Y"] = timeArr  # ----> 'Y' is always time when reading a 2D UFfile
    UF.Variables["X"] = np.linspace(1, numAntennas, numAntennas)

    arlist = []
    for i in range(numAntennas):
        tOn = timesOn[i][0]
        tOff = timesOn[i][1]
        if PowersMW_Before is not None:
            powerOff = PowersMW_Before[i]
        else:
            powerOff = 0.0
        powerON = PowersMW[i]
        _, powerarray = buildContinuousVector(
            [
                [tOn, powerOff * 1e6],
                [tOn, powerON * 1e6],
                [tOff, powerON * 1e6],
                [tOff, powerOff * 1e6],
            ],
            t=timeArr,
        )
        # _,z = buildContinuousVector([ [0.0,0.0],[10.0,0.0],[10.01,11.1* 1E6],[20.,11.1* 1E6] ],t=timeArr)
        arlist.append(powerarray)
    UF.Variables["Z"] = np.transpose(np.array(arlist))

    UF.writeUFILE(fileUF)


def buildContinuousVector(points, t=np.linspace(0, 20, 1000)):
    points = np.array(points)

    # Beginning
    z = np.ones(len(t)) * points[0, 1]

    # Middle
    for i in range(len(points) - 1):
        # Straight vertical line
        if points[i, 0] == points[i + 1, 0]:
            zSus = points[i, 1]

        # Ramp
        else:
            it1 = np.argmin(np.abs(t - points[i, 0]))
            it2 = np.argmin(np.abs(t - points[i + 1, 0]))

            num1 = len(t[:it1])
            num2 = len(t[it1:it2])
            num3 = len(t[it2:])

            zSus1 = np.ones(num1)
            zSus2 = np.linspace(points[i, 1], points[i + 1, 1], num2)
            zSus3 = np.ones(num3)

            zSus = np.append(np.append(zSus1, zSus2), zSus3)

        z = np.where((t >= points[i, 0]) & (t < points[i + 1, 0]), zSus, z)

    # Last
    z = np.where((t >= points[-1, 0]), points[-1, 1], z)

    return t, z


def updateUFILEfromCDF(varCDF, ufile, cdffile, timeExtract, timeWrite, scratch=None):
    """Options for timeExtract:
    -1 -> use last sawtooth
    -2 -> use last time
    -3 -> use middle between last and second to last sawtooth
    #  -> Any other positive number indicates the actual time to extract from

    varCDF can be string (take from cdf) or directly a variable in time or time/rho
    """

    # --------------------------------------
    # ------- Preparation
    # --------------------------------------

    cdffile = IOtools.expandPath(cdffile)
    import netCDF4

    f = netCDF4.Dataset(cdffile).variables

    # Variable

    if type(varCDF) == str:
        Z = f[varCDF][:]
    else:
        Z = copy.deepcopy(varCDF)
        varCDF = "*"

    # Find time index to extract from UFILE

    if timeExtract == -1:
        extratim, tlastsaws = 0.01, np.unique(f["TLASTSAW"][:])
        try:
            timer = tlastsaws[-1] - extratim
            ind = np.argmin(np.abs(f["TIME"][:] - (timer)))
            print(
                ">> cold_starting {0} from top of last sawtooth ({1}ms before), t={2:.3f}s".format(
                    varCDF, extratim * 1000.0, timer
                )
            )
        except:
            print(">> Run did not contained 1 sawtooth, taking last time")
            ind = -1

    elif timeExtract == -2:
        print(f">> cold_starting {varCDF} from last time, t={f['TIME'][:][-1]:.3f}s")
        ind = -1

    elif timeExtract == -3:
        lastindex, tlastsaws = -1, np.unique(f["TLASTSAW"][:])
        try:
            timer = (tlastsaws[lastindex] + tlastsaws[lastindex - 1]) / 2.0
            ind = np.argmin(np.abs(f["TIME"][:] - timer))
            print(
                f">> cold_starting {varCDF} from middle of last sawteeth, t={timer:.3f}s"
            )

        except:
            print(
                ">> Run did not contained 2 sawteeth, taking last time, t={0:.3f}s".format(
                    f["TIME"][:][-1]
                )
            )
            ind = -1

    elif timeExtract < -5:
        tlastsaws = np.unique(f["TLASTSAW"][:])
        try:
            timer = tlastsaws[-1] + timeExtract * 1e-3
            ind = np.argmin(np.abs(f["TIME"][:] - timer))
            print(
                f">> cold_starting {-timeExtract}ms before last sawteeth, t={timer:.3f}s"
            )

        except:
            print(
                ">> Run did not contain 1 sawteeth, taking last time, t={0:.3f}s".format(
                    f["TIME"][:][-1]
                )
            )
            ind = -1

    else:
        print(f">> cold_starting {varCDF} from t={timeExtract:.3f}s")
        ind = np.argmin(np.abs(f["TIME"][:] - timeExtract))

    # Selection of rho_tor coordinate
    if varCDF == "Q":
        xvar = "XB"
    else:
        xvar = "X"

    # Dimensions of UFILE
    try:
        aux, dim = len(Z[ind]), 2
    except:
        dim = 1

    # --------------------------------------
    # ----------  Extraction
    # --------------------------------------
    if dim == 2:
        x, y, zall = np.array(f[xvar][:][ind]), np.array(timeWrite), []
        for i in y:
            zall.append(Z[ind])
        z = np.array(zall)
    else:
        x, zall = np.array(timeWrite), []
        for i in x:
            zall.append(Z[ind])
        z = np.array(zall)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       Writing to UFILE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Read Ufile
    if scratch is None:
        UF = UFILEtransp()
        UF.readUFILE(ufile)
    else:
        UF = UFILEtransp(scratch=scratch)

    # Modify variables
    UF.Variables["X"], UF.Variables["Z"] = x, np.transpose(z)
    if dim == 2:
        UF.Variables["Y"] = (
            y  # ---> 'Y' is always the time coordinate when reading a 2D UFfile
        )

    #  Write new UFile
    UF.writeUFILE(ufile)

    return f["TIME"][:][ind], ind


def updateTypicalFiles(folder_new, cdf_file, timeExtract, shot="12345"):
    _, _ = updateUFILEfromCDF(
        "Q",
        folder_new / f"PRF{shot}.QPR",
        cdf_file,
        timeExtract,
        [0.0, 100.0],
    )
    _, _ = updateUFILEfromCDF(
        "TE",
        folder_new / f"PRF{shot}.TEL",
        cdf_file,
        timeExtract,
        [0.0, 100.0],
    )
    _, _ = updateUFILEfromCDF(
        "TI",
        folder_new / f"PRF{shot}.TIO",
        cdf_file,
        timeExtract,
        [0.0, 100.0],
    )
    _, _ = updateUFILEfromCDF(
        "NE",
        folder_new / f"PRF{shot}.NEL",
        cdf_file,
        timeExtract,
        [0.0, 100.0],
    )


def changeUFILEs(
    timesStepTransition,
    nameBaseShot,
    timeOriginal,
    FolderTRANSP,
    dictParams,
    stepTransition=False,
):
    rmajor, Bt, Ip = dictParams["rmajor"], dictParams["Bt"], dictParams["Ip"]

    valToModify = {"RBZ": Bt * rmajor * 100.0, "CUR": Ip * 1e6}

    for iDV in valToModify:
        # Create UFILE with the new values but being ramped from previous ones

        if stepTransition:
            UF = UFILEtransp()
            UF.readUFILE(FolderTRANSP + f"PRF{nameBaseShot}.{iDV}")
            t_orig = np.array(UF.Variables["X"])
            val_orig = UF.Variables["Z"][np.argmin(np.abs(t_orig - timeOriginal))]

            Zvals = np.array([val_orig, val_orig, valToModify[iDV]])
            tvals = np.array(timesStepTransition)
            printt = " (ramped-up)"

        # Create UFILE with the new values (not accounting for what was contained beforehand)

        else:
            Zvals = np.array([valToModify[iDV], valToModify[iDV]])
            tvals = np.array([0.0, 100.0])
            printt = " (uniform in time)"

        UF = UFILEtransp(scratch=iDV.lower())
        UF.Variables["Z"], UF.Variables["X"] = Zvals, tvals
        UF.writeUFILE(FolderTRANSP + f"PRF{nameBaseShot}.{iDV}")

        print(f"\t- Changed {iDV} in TRANSP U-Files to {valToModify[iDV]}{printt}")


def generateInitialRampValues(dictParams, FolderTRANSP, nameBaseShot, dictParamsOrig):
    # Parameters in U-Files

    UF = UFILEtransp()
    UF.readUFILE(FolderTRANSP + f"PRF{nameBaseShot}.CUR")
    Ip_orig = UF.Variables["Z"][0]

    UF = UFILEtransp()
    UF.readUFILE(FolderTRANSP + f"PRF{nameBaseShot}.RBZ")
    RBZ_orig = UF.Variables["Z"][0]

    R_orig, epsilon_orig, a_orig = (
        dictParamsOrig["rmajor"],
        dictParamsOrig["epsilon"],
        copy.deepcopy(epsilon_orig * R_orig),
    )

    # Parameters in new baseline (which was created to avoid intersections)

    rmajor_baseline, epsilon_baseline, Bt = (
        dictParams["rmajor_startwith"],
        dictParams["epsilon_startwith"],
        dictParams["Bt"],
    )

    R_new = copy.deepcopy(rmajor_baseline)
    RBZ_new = copy.deepcopy(Bt * R_new * 100.0)
    a_new = copy.deepcopy(epsilon_baseline * R_new)
    Ip_new = copy.deepcopy(Ip_orig)

    return Ip_new, RBZ_new


def offsettimeUF(FolderTRANSP, nameBaseShot, nameufile, offsettime):
    # Read Ufile
    UF = UFILEtransp()
    UF.readUFILE(FolderTRANSP + f"PRF{nameBaseShot}.{nameufile}")

    # Modify variables
    if UF.dim == 1:
        UF.Variables["X"] = np.array(UF.Variables["X"]) + offsettime
    else:
        UF.Variables["Y"] = np.array(UF.Variables["Y"]) + offsettime

    #  Write new UFile
    UF.writeUFILE(FolderTRANSP + f"PRF{nameBaseShot}.{nameufile}")


def initializeUFILES_MinimalTRANSP(rho, Te, Ti, ne, q, V, location=".", name="12345"):
    location = IOtools.expandPath(location)

    ufiles = {
        "TEL": ["ter", Te],
        "TIO": ["ti2", Ti],
        "NEL": ["ner", ne],
        "QPR": ["qpr", q],
        "VSF": ["vsf", V],
    }

    for ufn in ufiles:
        quickUFILE(
            rho,
            ufiles[ufn][1],
            location / f"PRF{name}.{ufn}",
            typeuf=ufiles[ufn][0],
        )


def quickUFILE(x, z, file, time=[0.0, 1000.0], typeuf="ter"):
    uf = UFILEtransp(scratch=typeuf)
    if x is not None:
        uf.Variables["X"], uf.Variables["Y"], uf.Variables["Z"] = x, time, z
    else:
        uf.Variables["X"], uf.Variables["Z"] = time, z
    uf.repeatProfile()
    uf.writeUFILE(file)


def reduceTimeUFILE(file, extractTime, newTimes=None):
    if newTimes is None:
        newTimes = [extractTime]

    uf = UFILEtransp()
    uf.readUFILE(file)

    if uf.dim < 3:
        time, z = uf.Variables["Y"], uf.Variables["Z"]

        znew = []
        for i in range(z.shape[0]):
            znew.append(np.interp(extractTime, time, z[i]))
        znew = np.array(znew)

        uf.Variables["Y"] = newTimes
        uf.Variables["Z"] = znew

        if len(newTimes) > 1:
            uf.repeatProfile()

    else:
        # It looks like time is in X when I tried the MMX case
        time, z = uf.Variables["X"], uf.Variables["Z"]

        znew = []
        for i in range(z.shape[1]):
            z1 = []
            for j in range(z.shape[2]):
                z1.append(np.interp(extractTime, time, z[:, i, j]))
            znew.append(z1)
        znew = np.array(znew)

        uf.Variables["X"] = newTimes

        znew = np.expand_dims(znew, 0)

        if len(newTimes) > 1:
            for i in range(len(newTimes) - 1):
                znew = np.append(znew, znew, axis=0)

        uf.Variables["Z"] = znew

    # uf.repeatProfile()
    uf.writeUFILE(file)


def createImpurityUFILE(rho, nZ, file="PRF12345.NW", Z=74, A=183, t=None):
    # nZ in 1E20

    uf = UFILEtransp(scratch="nzr")
    uf.Variables["X"], uf.Variables["Z"] = rho, nZ
    if t is not None:
        uf.Variables["Y"] = t
    else:
        uf.repeatProfile()

    uf.addImpurityheader(Z, A)

    uf.writeUFILE(file)


def writeRFSZFS(theta, rho, R, Z, prefix="PRF12345", debug=False):
    timeAxis, timeVar = 0, "X"
    rhoVar = "Q"
    thetaVar = "Y"

    uf = UFILEtransp(scratch="rfs")
    uf.Variables[rhoVar] = rho
    uf.Variables[thetaVar] = theta
    Z0 = np.expand_dims(R, timeAxis)
    Z1 = np.concatenate((Z0, Z0), axis=0)
    uf.Variables["Z"] = Z1
    uf.writeUFILE(prefix + ".RFS")

    uf = UFILEtransp(scratch="zfs")
    uf.Variables[rhoVar] = rho
    uf.Variables[thetaVar] = theta
    Z0 = np.expand_dims(Z, timeAxis)
    Z1 = np.concatenate((Z0, Z0), axis=timeAxis)
    uf.Variables["Z"] = Z1
    uf.writeUFILE(prefix + ".ZFS")

    if debug:
        uf = UFILEtransp()
        uf.readUFILE(prefix + ".RFS")
        rho0, theta0, time0, R0 = (
            uf.Variables[rhoVar],
            uf.Variables[thetaVar],
            uf.Variables[timeVar],
            uf.Variables["Z"],
        )
        R0 = R0[0]

        uf = UFILEtransp()
        uf.readUFILE(prefix + ".ZFS")
        rho1, theta1, time1, Z1 = (
            uf.Variables[rhoVar],
            uf.Variables[thetaVar],
            uf.Variables[timeVar],
            uf.Variables["Z"],
        )
        Z1 = Z1[0]

        fig, ax = plt.subplots(ncols=2)
        cols = GRAPHICStools.listColors()
        for i in range(len(rho0)):
            ax[0].plot(R0[:, i], Z1[:, i], "o-", c=cols[i])
            ax[1].plot(theta0, R0[:, i], "-*", c=cols[i])
            ax[1].plot(theta1, Z1[:, i], "-o", c=cols[i])

        plt.show()
        embed()
