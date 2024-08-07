import sys
import os
import socket
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import FARMINGtools, MATHtools, IOtools, GRAPHICStools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.im_tools.modules import EQmodule
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed

"""
------------------------------------------------------------------------------------------------
Packages that may fail
------------------------------------------------------------------------------------------------
"""
try:
    import MDSplus
except ModuleNotFoundError:
    print("Could not load MDSplus", typeMsg="w")

sys.path.insert(0, "/home/sciortino/usr/python3modules/eqtools3")
import eqtools

# ------------------------------------------------------------------------------------------------


def getMDS_timevar(shot, tree, var):
    print(f"\t>> Extracting {var} from {tree} tree for shot {shot}")

    tree = MDSplus.Tree(tree, shot)
    Z = tree.getNode(var).record.data()
    t = tree.getNode(var).getData().dim_of(0).data()

    return Z, t


def getMDS_2Dvar(shot, tree, var):
    print(f"\t>> Extracting {var} (2D) from {tree} tree for shot {shot}")

    tree = MDSplus.Tree(tree, shot)
    Z = tree.getNode(var).record.data()
    t = tree.getNode(var).getData().dim_of(0).data()
    x = tree.getNode(var).getData().dim_of(1).data()

    return Z, t, x


class experiment:
    def __init__(self, shot):
        self.shot = shot

    def get1Dtraces(self):
        print(" >> Gathering CMOD experimental data (1D)")

        self.Wexp, self.Wexp_t = getMDS_timevar(
            self.shot, "analysis", "\efit_aeqdsk:wplasm"
        )
        self.q95, self.q95_t = getMDS_timevar(
            self.shot, "analysis", "\efit_aeqdsk:qpsib"
        )
        self.neL, self.neL_t = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT"
        )
        self.Bp, self.Bp_t = getMDS_timevar(
            self.shot, "analysis", "\\analysis::top:efit.results.a_eqdsk:betap"
        )
        self.Li, self.Li_t = getMDS_timevar(
            self.shot, "analysis", "\\analysis::top:efit.results.a_eqdsk:ali"
        )
        self.Li2Bp = self.Li / 2.0 + self.Bp
        Psurf, self.Vsurf_t = getMDS_timevar(
            self.shot, "analysis", "\efit_aeqdsk:sibdry"
        )
        self.Vsurf = MATHtools.deriv(self.Vsurf_t, Psurf * 2 * np.pi)

        try:
            self.neut, neut_t = getMDS_timevar(
                self.shot,
                "particles",
                "\particles::top.neutrons.global.results:neut_rate",
            )
        except:
            print("Could not grab neutrons... returning zeros")
            self.neut_t = self.Wexp_t
            self.neut = np.zeros(len(self.Wexp_t))

    def get2Dprofiles(self):
        self.getECE()
        self.getTS()

    def write_gfile(self, time, name="~/gfile.geq"):
        e = eqtools.CModEFITTree(self.shot, tree="analysis")
        eqtools.filewriter.gfile(e, time, name=name)

    # -----------------------------------------------------------------------------------------------------------------------
    # OPERATIONS
    # -----------------------------------------------------------------------------------------------------------------------

    def getECE(self):
        R, Te, t = [], [], []
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            Te1, Te_t1 = getMDS_timevar(
                self.shot, "electrons", f"\\ELECTRONS::gpc_te{i}"
            )
            R1, R_t1 = getMDS_timevar(self.shot, "electrons", f"\\ELECTRONS::gpc_r{i}")
            R1_mod = np.interp(Te_t1, R_t1, R1)
            R.append(R1_mod)
            Te.append(Te1)
            t.append(Te_t1)

        self.R_ECE, self.Te_ECE, self.TeError_ECE = (
            np.array(R),
            np.array(Te),
            np.array(Te) * 0.1,
        )
        self.t_ECE = np.array(t)

    def getTS(self):
        # ------------------------------------------------------------------------------------------------------------------------------------------
        # Grab data
        # ------------------------------------------------------------------------------------------------------------------------------------------

        # Edge
        neE, neE_t = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE"
        )
        RE, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID"
        )  # Time is the same
        neErrorE, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE:ERROR"
        )  # Time is the same

        TeE, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE"
        )  # Time is the same
        TeErrorE, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE:ERROR"
        )  # Time is the same

        # Core
        neC, neC_t = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ"
        )
        RC, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T"
        )  # Time is the same
        neErrorC, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_ERR"
        )  # Time is the same

        TeC, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_RZ"
        )  # Time is the same
        TeErrorC, _ = getMDS_timevar(
            self.shot, "electrons", "\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_ERR"
        )  # Time is the same

        # ------------------------------------------------------------------------------------------------------------------------------------------
        # Units converstion
        # ------------------------------------------------------------------------------------------------------------------------------------------

        neE = neE * 1e-20
        neErrorE = neErrorE * 1e-20
        TeE = TeE * 1e-3
        TeErrorE = TeErrorE * 1e-3
        neC = neC * 1e-20
        neErrorC = neErrorC * 1e-20

        # ------------------------------------------------------------------------------------------------------------------------------------------
        # Merging
        # ------------------------------------------------------------------------------------------------------------------------------------------

        # Bring core to edge time
        for i in range(neC.shape[0]):
            neC[i, :] = np.interp(neE_t, neC_t, neC[i, :])
            TeC[i, :] = np.interp(neE_t, neC_t, TeC[i, :])
            RC[i, :] = np.interp(neE_t, neC_t, RC[i, :])
            neErrorC[i, :] = np.interp(neE_t, neC_t, neErrorC[i, :])
            TeErrorC[i, :] = np.interp(neE_t, neC_t, TeErrorC[i, :])

        self.ne_TS = np.append(neC, neE, axis=0)
        self.Te_TS = np.append(TeC, TeE, axis=0)
        self.neError_TS = np.append(neErrorC, neErrorE, axis=0)
        self.TeError_TS = np.append(TeErrorC, TeErrorE, axis=0)
        self.R_TS = np.append(RC, RE, axis=0)
        self.t_TS = neE_t

        # # Order positions
        # self.ne_TS 			= np.array([a for _,a in sorted(zip(R,ne))])
        # self.Te_TS 			= np.array([a for _,a in sorted(zip(R,Te))])
        # self.neError_TS 	= np.array([a for _,a in sorted(zip(R,neError))])
        # self.TeError_TS 	= np.array([a for _,a in sorted(zip(R,TeError))])
        # self.R_TS 			= np.array([a for _,a in sorted(zip(R,R))])

    def sliceTS(self, time, avt=0):
        from mitim_tools.transp_tools.CDFtools import timeAverage

        self.time = time
        self.avt = avt

        it1 = np.argmin(np.abs(self.t_TS - (self.time - self.avt)))
        it2 = np.argmin(np.abs(self.t_TS - (self.time + self.avt))) + 1

        self.ne_TS_sliced = np.zeros(self.ne_TS.shape[0])
        self.Te_TS_sliced = np.zeros(self.Te_TS.shape[0])
        self.neError_TS_sliced = np.zeros(self.ne_TS.shape[0])
        self.TeError_TS_sliced = np.zeros(self.Te_TS.shape[0])
        self.R_TS_sliced = np.zeros(self.R_TS.shape[0])
        for c in range(self.Te_TS_sliced.shape[0]):
            self.ne_TS_sliced[c] = timeAverage(
                self.t_TS[it1:it2], self.ne_TS[c, it1:it2]
            )
            self.Te_TS_sliced[c] = timeAverage(
                self.t_TS[it1:it2], self.Te_TS[c, it1:it2]
            )
            self.neError_TS_sliced[c] = timeAverage(
                self.t_TS[it1:it2], self.neError_TS[c, it1:it2]
            )
            self.TeError_TS_sliced[c] = timeAverage(
                self.t_TS[it1:it2], self.TeError_TS[c, it1:it2]
            )
            self.R_TS_sliced[c] = timeAverage(self.t_TS[it1:it2], self.R_TS[c, it1:it2])

        (
            self.rhopol_TS_sliced,
            self.rhotor_TS_sliced,
            self.roa_TS_sliced,
        ) = self.changegrid(self.R_TS_sliced, self.time)

    def sliceECE(self, time, avt=0):
        from mitim_tools.transp_tools.CDFtools import timeAverage

        self.time = time
        self.avt = avt

        self.Te_ECE_sliced = np.zeros(self.Te_ECE.shape[0])
        self.TeError_ECE_sliced = np.zeros(self.Te_ECE.shape[0])
        self.R_ECE_sliced = np.zeros(self.R_ECE.shape[0])
        for c in range(self.Te_ECE_sliced.shape[0]):
            it1 = np.argmin(np.abs(self.t_ECE[c] - (self.time - self.avt)))
            it2 = np.argmin(np.abs(self.t_ECE[c] - (self.time + self.avt))) + 1
            self.Te_ECE_sliced[c] = timeAverage(
                self.t_ECE[c, it1:it2], self.Te_ECE[c, it1:it2]
            )
            self.TeError_ECE_sliced[c] = timeAverage(
                self.t_ECE[c, it1:it2], self.TeError_ECE[c, it1:it2]
            )
            self.R_ECE_sliced[c] = timeAverage(
                self.t_ECE[c, it1:it2], self.R_ECE[c, it1:it2]
            )

        (
            self.rhopol_ECE_sliced,
            self.rhotor_ECE_sliced,
            self.roa_ECE_sliced,
        ) = self.changegrid(self.R_ECE_sliced, self.time)

    def slice2Dprofiles(self, time, avt=0):
        self.sliceTS(time, avt=avt)
        self.sliceECE(time, avt=avt)

    def changegrid(self, Rmid, time):
        e = eqtools.CModEFITTree(self.shot, tree="analysis")

        rhopol = np.sqrt(e.rmid2psinorm(Rmid, time))
        rhotor = np.sqrt(e.rmid2phinorm(Rmid, time))
        roa = e.rmid2roa(Rmid, time)

        return rhopol, rhotor, roa

    # -----------------------------------------------------------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------------------------------------------------------

    def plotProfiles(self):
        plt.ion()
        fig, axs = plt.subplots(nrows=2, figsize=(5, 8))

        ax = axs[0]
        try:
            ax.errorbar(
                self.rhotor_TS_sliced,
                self.Te_TS_sliced,
                yerr=self.TeError_TS_sliced,
                c="r",
                label="TS",
                markersize=5,
                capsize=3.0,
                fmt="s",
                elinewidth=0.5,
                capthick=0.5,
            )
        except:
            pass
        try:
            ax.errorbar(
                self.rhotor_ECE_sliced,
                self.Te_ECE_sliced,
                yerr=self.TeError_ECE_sliced,
                c="b",
                label="ECE",
                markersize=5,
                capsize=3.0,
                fmt="s",
                elinewidth=0.5,
                capthick=0.5,
            )
        except:
            pass

        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$T_e$ (keV)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)
        ax.legend()

        ax = axs[1]
        try:
            ax.errorbar(
                self.rhotor_TS_sliced,
                self.ne_TS_sliced,
                yerr=self.neError_TS_sliced,
                c="r",
                label="TS",
                markersize=5,
                capsize=3.0,
                fmt="s",
                elinewidth=0.5,
                capthick=0.5,
            )
        except:
            pass
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        axs[0].set_title(f"Kinetic profiles @ t = {self.time:.2f} (+-{self.avt}) s")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------


def defineTRANSPnmlStructures():
    limiters = [
        [103.50, 0.00, 90.00],
        [165.00, 142.63, 0.00],
        [
            236.49,
            0.00,
            90.00,
        ],
        [165.00, -142.63, 0.00],
    ]

    VVmoms = [[64.5, 0.0], [35.0, 57.3], [3.25, -3.25], [0, 0], [0, 0]]

    return limiters, VVmoms


def defineFirstWall():
    z = [
        -0.215700001,
        0.0,
        0.432799995,
        0.432799995,
        0.402999997,
        0.385500014,
        0.388599992,
        0.337799996,
        0.337799996,
        0.266136408,
        0.266136408,
        0.264612317,
        0.263548434,
        0.262493551,
        0.230599269,
        0.226122722,
        0.195354164,
        0.1918699,
        0.169212058,
        0.166141152,
        0.141770005,
        0.138491482,
        0.112696007,
        0.109252214,
        0.0823316127,
        0.0786074847,
        0.0578148589,
        0.0507732444,
        0.0466043055,
        0.0187179465,
        0.00251530879,
        0.0,
        -0.00264220894,
        -0.0187179465,
        -0.0466043167,
        -0.0507732816,
        -0.0578148738,
        -0.0786080882,
        -0.0823322237,
        -0.109252825,
        -0.11269661,
        -0.138492092,
        -0.141770616,
        -0.166141763,
        -0.169212669,
        -0.191870511,
        -0.195342407,
        -0.226111323,
        -0.23059988,
        -0.262494147,
        -0.263549298,
        -0.264612317,
        -0.266136408,
        -0.266136408,
        -0.337799996,
        -0.337799996,
        -0.373199999,
        -0.363200009,
        -0.35710001,
        -0.372799993,
        -0.362699986,
        -0.361200005,
        -0.359699994,
        -0.358500004,
        -0.3574,
        -0.356599987,
        -0.356000006,
        -0.355699986,
        -0.355699986,
        -0.356000006,
        -0.367794007,
        -0.367793888,
        -0.370719105,
        -0.37505731,
        -0.375057012,
        -0.417571992,
        -0.417572111,
        -0.423004001,
        -0.429450691,
        -0.436699599,
        -0.444511712,
        -0.452629209,
        -0.460784405,
        -0.460783988,
        -0.573902845,
        -0.573899984,
        -0.595899999,
        -0.595899999,
        -0.579400003,
        -0.579400003,
        -0.512099981,
        -0.500100017,
        -0.474675,
        -0.29629001,
        -0.251060009,
        -0.215700001,
    ]
    r = [
        0.440200001,
        0.440200001,
        0.440200001,
        0.699899971,
        0.699899971,
        0.764900029,
        0.765699983,
        0.955299973,
        1.06159997,
        1.06159997,
        0.820685983,
        0.819162071,
        0.81916213,
        0.818088651,
        0.817809761,
        0.818863809,
        0.834543049,
        0.836682141,
        0.853755593,
        0.855821788,
        0.870345116,
        0.87206775,
        0.883877695,
        0.885235786,
        0.894189119,
        0.895626426,
        0.904622555,
        0.906196117,
        0.906889856,
        0.909967721,
        0.909967721,
        0.909967721,
        0.909967721,
        0.909967721,
        0.906889856,
        0.906196117,
        0.904622555,
        0.895626128,
        0.894188821,
        0.885235488,
        0.883877456,
        0.872067511,
        0.870344877,
        0.85582149,
        0.853755355,
        0.836681902,
        0.834549129,
        0.818871439,
        0.817809522,
        0.818088353,
        0.81916213,
        0.819162071,
        0.820685983,
        1.06159997,
        1.06159997,
        0.955299973,
        0.823499978,
        0.820800006,
        0.819199979,
        0.760500014,
        0.757799983,
        0.757200003,
        0.756399989,
        0.755299985,
        0.754000008,
        0.752900004,
        0.750999987,
        0.74940002,
        0.747699976,
        0.746100008,
        0.702022016,
        0.702022374,
        0.694146514,
        0.686951578,
        0.686951995,
        0.629375994,
        0.629375815,
        0.623269379,
        0.618246078,
        0.614471614,
        0.612070382,
        0.611121893,
        0.611657083,
        0.611657023,
        0.629500926,
        0.702899992,
        0.702899992,
        0.602199972,
        0.572899997,
        0.569899976,
        0.569899976,
        0.560699999,
        0.468349993,
        0.468349993,
        0.460399985,
        0.440200001,
    ]

    return r, z


def ICRFantennas(MHz):
    nichas = 2  # 1

    lines = [
        "! ----- Antenna Parameters",
        f"nicha       = {nichas}         \t ! Number of ICRH antennae",
        "frqicha     = 80.0e6,78.0e6 ! Frequency of antenna (Hz)".format(MHz),
        "rfartr      = 2.0           ! Distance (cm) from antenna for Faraday shield",
        "ngeoant     = 1         	 ! Geometry representation of antenna (1=traditional)",
        "rmjicha     = 60.8,60.8     ! Major radius of antenna (cm)",
        "rmnicha     = 32.5,32.5     ! Minor radius of antenna (cm)",
        "thicha      = 73.3,73.3     ! Theta extent of antenna (degrees)",
        "sepicha     = 25.6,25.6     ! Toroidal seperation strap to strap (cm)",
        "widicha     = 10.2,10.2     ! Full toroidal width of each antenna element",
        "phicha(1,1) = 0,180   		 ! Phasing of antenna elements (deg)",
        "phicha(1,2) = 0,180",
        "",
        "!==================================================================",
    ]

    return "\n".join(lines)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def grabImpurities(nml, nml_dict={}):
    nml_dict["xzmini"] = IOtools.findValue(nml, "xzmini", "=")
    nml_dict["amini"] = IOtools.findValue(nml, "amini", "=")
    nml_dict["frmini"] = IOtools.findValue(nml, "frmini", "=")

    Zs = IOtools.findValue(nml, "xzimps", "=", isitArray=True)
    Zs = [float(i) for i in Zs.split("!")[0].split(",")]
    As = IOtools.findValue(nml, "aimps", "=", isitArray=True)
    As = [float(i) for i in As.split("!")[0].split(",")]
    Ns = IOtools.findValue(nml, "densim", "=", isitArray=True)
    Ns = [float(i) for i in Ns.split("!")[0].split(",")]

    for i in range(len(Zs)):
        nml_dict[f"xzimps({i + 1})"] = Zs[i]
        nml_dict[f"aimps({i + 1})"] = As[i]
        nml_dict[f"densim({i + 1})"] = Ns[i]

        # Othewise it doesn't match dilution
        nml_dict[f"nadvsim({i + 1})"] = 0

    nml_dict["nvtor_z"] = int(IOtools.findValue(nml, "NVTOR_Z", "="))
    nml_dict["xvtor_a"] = IOtools.findValue(nml, "XVTOR_A", "=")

    return nml_dict


def updateTRANSPfromNML(nml_old, nml_new, folderWork, PRFmodified=False):
    shotnum = int(IOtools.findValue(nml_old, "nshot", "="))

    # ---------------------------------------
    # Main simulation type
    # ---------------------------------------

    # ---- Interpretive

    nml_dict = {"lpredictive_mode": 0}

    # ---- Standard C-Mod cases use current diffusion

    nml_dict["nqmoda(1)"] = 1

    # ---------------------------------------
    # Experiental data or assuptions
    # ---------------------------------------

    # ---- Use the same times

    nml_dict["tinit"] = IOtools.findValue(nml_old, "tinit", "=")
    nml_dict["ftime"] = IOtools.findValue(nml_old, "ftime", "=")

    # ---- Use same impurities

    nml_dict = grabImpurities(nml_old, nml_dict=nml_dict)

    # ---- Radiation is specified

    nml_dict["nprad"] = 0
    nml_dict["prfac"] = 0.2
    nml_dict["extbol"], nml_dict["prebol"] = "'BOL'", "'PRF'"

    # ---- Rotation is specified

    nml_dict["extvp2"], nml_dict["prevp2"] = "'VP2'", "'PRF'"

    # ---- Use the same coordinates in UFILES and names

    for i in ["ter", "ti2", "ner", "bol", "vp2"]:
        nml_dict["nri" + i] = -4

    # ----- Change names to those that I understand

    for i, j in zip(["NER", "TER", "TI2"], ["NEL", "TEL", "TIO"]):
        os.system("mv {0}/PRF{1}.{2} {0}/PRF{1}.{3}".format(folderWork, shotnum, i, j))

    # ---- Add C-Mod limiter

    rlim, zlim = defineFirstWall()
    EQmodule.addLimiters_UF(f"{folderWork}/PRF{shotnum}.LIM", rlim, zlim)

    # ---- No gas flow (my way is to give this file)

    gasflow = 0.0
    UFILEStools.quickUFILE(
        None, gasflow, f"{folderWork}/PRF{shotnum}.GFD", typeuf="gfd"
    )

    # ---- Zeff specified as a uniform profile (my way)

    xZeff, Zeff = np.linspace(0, 1, 10), np.ones(10) * IOtools.findValue(
        nml_old, "xzeffi", "="
    )
    UFILEStools.quickUFILE(xZeff, Zeff, f"{folderWork}/PRF{shotnum}.ZF2", typeuf="zf2")

    # This file is useless
    os.system(f"rm {folderWork}/PRF{shotnum}.ZEF")

    # ---- Ti validity

    nml_dict["tixlim"] = IOtools.findValue(nml_old, "tixlim", "=")

    # ---- Sawtooth trigger from UFILE

    nml_dict["nlsaw_trigger"] = False
    nml_dict["model_sawtrigger"] = 0
    nml_dict["sawtooth_period"] = 0
    nml_dict["c_sawtooth(2)"] = 0
    nml_dict["extsaw"], nml_dict["presaw"] = "'SAW'", "'PRF'"

    # Let's not include neutrons
    os.system(f"rm {folderWork}/PRF{shotnum}.NTX")

    # ---------------------------------------
    # Simulation settings
    # ---------------------------------------

    # ---- Initialized by parametrized loop voltage (no QPR)

    nml_dict["nefld"] = 3
    nml_dict["qefld"], nml_dict["rqefld"], nml_dict["xpefld"] = 0.0, 0.0, 2.0
    nml_dict["extqpr"] = nml_dict["preqpr"] = nml_dict["nriqpr"] = None

    if not PRFmodified:
        """
        These are settings that are not strickly experimental, but choices made by PRETRANSP.
        Here, I can choose them (to reproduce the PRETRANSP exactly, with PRFmodified=False)
        or use what I think it's best (PRFmodified=True)
        """

        #  ---- PRETRANSP used the default TIEDGE... which affects strongly neutrals and CX

        nml_dict["tiedge"] = 10.0

        # ---- PRETRANSP used NCLASS Resistivity and clamped it at q=1

        nml_dict["nlres_sau"], nml_dict["nletaw"], nml_dict["nlrsq1"] = (
            False,
            True,
            True,
        )

        # ---- PRETRANSP used sawtooth mixing from Kadomtsev and not applied to minority ions

        nml_dict["nmix_kdsaw"] = 1
        nml_dict["nlsawic"] = False

    # --------------------------------------------------------
    # If MRY wasn't populated for this run, use MMX
    # --------------------------------------------------------
    try:
        mry = IOtools.findValue(nml_old, "premry", "=")
    except:
        mry = None

    if mry is None:
        nml_dict["premry"] = nml_dict["extmry"] = None
        nml_dict["premmx"], nml_dict["extmmx"] = '"PRF"', '"MMX"'
    # --------------------------------------------------------

    return nml_dict


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def getTRANSP_MDS(
    runid, runid_new, folderWork="~/scratch/test/", toric_mpi=1, shotnumber=None
):
    folderWork = IOtools.expandPath(folderWork)
    if not os.path.exists(folderWork):
        IOtools.askNewFolder(folderWork)

    tree = MDSplus.Tree("transp", int(runid))

    # Namelist
    nml = tree.getNode("NAME_LIST").record.data()

    nml_file = f"{folderWork}/{runid_new}TR.DAT"
    with open(nml_file, "w") as f:
        for i in range(len(nml)):
            f.write(nml[i].decode("UTF-8") + "\n")

    IOtools.changeValue(nml_file, "NSHOT", runid, [], "=")
    IOtools.changeValue(nml_file, "KMDSPLUS", None, [], "=")

    if toric_mpi > 1:
        IOtools.changeValue(nml_file, "ntoric_pserve", 1, [], "=")

    # UFILES
    names = [
        "bol",
        "ner",
        "ter",
        "ti2",
        "vp2",
        "cur",
        "saw",
        "ntx",
        "rbz",
        "vsf",
        "zef",
        "mry",
        "rfp",
        "vp2",
    ]
    nomry = False
    for name in names:
        print(f"Reading {name}")
        nameMDS = name.upper()
        labelX = None
        if name in ["ter", "ti2", "ner", "bol", "vp2"]:
            labelX = " r/a                           "
        try:
            uf = nodeToUF(
                runid, name, nameMDS, folderWork, inputs=".INPUTS:", labelX=labelX
            )
            IOtools.changeValue(nml_file, "PRE" + nameMDS, "'PRF'", [], "=")
        except:
            print("\t~~ Could not retrieve")
            if name == "mry":
                nomry = True

    # If MRY wasn't populated, run my own equilribium scruncher
    if nomry and shotnumber is not None:
        print("** Because MRY was not produced, run scrunch2 to produce MMX")
        ff = f"{folderWork}/scrunch/"
        os.system(f"mkdir {ff}")
        getMMX(shotnumber, runid, ff)
        IOtools.changeValue(nml_file, "premry", None, [], "=")
        IOtools.changeValue(nml_file, "extmry", None, [], "=")
        os.system(f"cp {ff}/PRF{runid}.MMX {folderWork}/.")
        IOtools.changeValue(nml_file, "premmx", '"PRF"', [], "=")
        IOtools.changeValue(nml_file, "extmmx", '"MMX"', [], "=")


def nodeToUF(runid, name, nameMDS, folderWork, inputs=".INPUTS:", labelX=None):
    uf = UFILEStools.UFILEtransp(scratch=name, labelX=labelX)

    tree = MDSplus.Tree("transp", int(runid))

    uf.Variables["Z"] = tree.getNode(inputs + nameMDS).record.data()

    if uf.dim == 1:
        uf.Variables["X"] = tree.getNode(inputs + nameMDS).record.dim_of(0).data()
    elif uf.dim == 2:
        uf.Variables["X"] = tree.getNode(inputs + nameMDS).record.dim_of(1).data()
        uf.Variables["Y"] = tree.getNode(inputs + nameMDS).record.dim_of(0).data()

        uf.Variables["Z"] = np.transpose(uf.Variables["Z"])

    elif uf.dim == 3:
        uf.Variables["X"] = tree.getNode(inputs + nameMDS).record.dim_of(0).data()
        uf.Variables["Y"] = tree.getNode(inputs + nameMDS).record.dim_of(1).data()
        uf.Variables["Q"] = tree.getNode(inputs + nameMDS).record.dim_of(2).data()

        uf.Variables["Z"] = np.transpose(uf.Variables["Z"])

    filename = f"{folderWork}/PRF{runid}.{nameMDS}"

    uf.writeUFILE(filename)

    return uf


def compareMDSandCDF(runidMDS, CDFclass):
    c = CDFclass

    # Compare to original
    tree = MDSplus.Tree("transp", int(runidMDS))

    # In time
    varCompare = [
        "TE",
        "TI",
        "NE",
        "PEICH",
        "PIICH",
        "PEICH",
        "PIICH",
        "Q",
        "Q",
        "UTOTL",
        "OMEGA",
        "PRAD",
        "UTHRM",
        "UMINPA",
        "UMINPA",
        "DN0WD",
    ]
    xpos = [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0.5, 0, 0.25, 0.9]

    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    axs = GRAPHICStools.producePlotsGrid(
        len(varCompare), fig=fig, hspace=0.6, wspace=0.6, sharex=False, sharey=False
    )
    for cont, var in enumerate(varCompare):
        ax = axs[cont]

        t = tree.getNode(".TRANSP_OUT:TE0").record.dim_of(0).data()
        z = tree.getNode(".TRANSP_OUT:" + var).record.data()
        x = tree.getNode(".TRANSP_OUT:X").record.data()
        ix = np.argmin(np.abs(x - xpos[cont]))
        ax.plot(t, z[:, ix], c="b")

        t = c.f["TIME"][:]
        z = c.f[var][:]
        x = c.f["X"][:]
        ix = np.argmin(np.abs(x - xpos[cont]))
        ax.plot(t, z[:, ix], c="r")

        ax.set_title(var + "," + str(xpos[cont]))

    # In  rho
    varCompare = [
        "OMEGA",
        "TE",
        "TI",
        "NE",
        "PEICH",
        "PIICH",
        "PCNDE",
        "PCOND",
        "PRAD",
        "CUR",
        "Q",
        "ZEFFI",
        "ND",
        "UMINPP",
        "UMINPA",
        "DN0WD",
    ]
    ts = [-1] * len(varCompare)  # [0]*len(varCompare)

    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    axs = GRAPHICStools.producePlotsGrid(
        len(varCompare), fig=fig, hspace=0.6, wspace=0.6, sharex=False, sharey=False
    )
    for cont, var in enumerate(varCompare):
        ax = axs[cont]

        if ts[cont] == -1:
            ts0 = c.f["TIME"][:][-1]
        else:
            ts0 = ts[cont]

        z = tree.getNode(".TRANSP_OUT:" + var).record.data()
        t = tree.getNode(".TRANSP_OUT:TE0").record.dim_of(0).data()
        x = tree.getNode(".TRANSP_OUT:X").record.data()
        it = np.argmin(np.abs(t - ts0))
        ax.plot(x[it], z[it], c="b")

        t = c.f["TIME"][:]
        z = c.f[var][:]
        x = c.f["X"][:]
        it = np.argmin(np.abs(t - ts0))
        ax.plot(x[it], z[it], "--", c="r")

        ax.set_title(var + "," + str(ts0) + "s")


def getZeff_neo(
    shotNumber,
    folder=IOtools.expandPath("~/"),
    routine=__mitimroot__ + "/scripts/zeff_neo",
):
    with open(folder + "/idl_in", "w") as f:
        f.write(f".r {routine}\n\n")
        f.write(f"openr,1,'{folder}/shot.dat'\n")
        f.write("shot = strarr(1)\n")
        f.write("readf,1,shot\n")
        f.write("close,1\n")
        f.write(f"zeff_neo,{shotNumber},z,t\n")
        f.write(f"openw,1,'{folder}/t.dat'\n")
        f.write(f"openw,2,'{folder}/z.dat'\n")
        f.write("printf,1,t\n")
        f.write("printf,2,z\n")
        f.write("close,1\n")
        f.write("close,2\n")

    with open(folder + "/shot.dat", "w") as f:
        f.write(str(shotNumber))

    Command = f"cd {folder} && idl < idl_in"
    # FIX
    error, result = FARMINGtools.runCommand_remote(
        Command, machine=socket.gethostname()
    )

    with open(folder + "/t.dat", "r") as f:
        aux = f.readlines()
    t = []
    for i in aux:
        t.extend([float(j) for j in i.split()])
    t = np.array(t)

    with open(folder + "/z.dat", "r") as f:
        aux = f.readlines()
    zeff = []
    for i in aux:
        zeff.extend([float(j) for j in i.split()])
    zeff = np.array(zeff)

    os.system("rm " + folder + "/t.dat " + folder + "/z.dat")

    return zeff, t


def getMMX(shotNumber, runid, folderWork):
    folderWork = IOtools.expandPath(folderWork)
    folderScratch = folderWork + "/scr_mmx/"

    if not os.path.exists(folderScratch):
        IOtools.askNewFolder(folderScratch)

    with open(folderScratch + "/scrunch.in", "w") as f:
        f.write(f"CMOD\n{shotNumber}\nA\nA\nQ\nY\nP\nPRF\nW\nQ")

    Command = f"cd {folderScratch} && scrunch2 < scrunch.in"
    # FIX
    error, result = FARMINGtools.runCommand_remote(
        Command, machine=socket.gethostname()
    )
    if result is not None:
        GSerror = []
        for i in result:
            if "estimated relative GS error in data" in i:
                GSerror.append(float(i.split()[-1]))
        GSerrormax = np.max(GSerror)
        print(f" >> Maximum relative GS error in data: {GSerrormax}")

    for ufile in ["PLF", "PF0", "TRF", "PRS", "QPR", "LIM", "GRB", "MMX"]:
        os.system(
            "mv {0}/PRF{1}.{2} {3}/PRF{4}.{2}".format(
                folderScratch, str(shotNumber)[-6:], ufile, folderWork, runid
            )
        )

    os.system(f"rm -r {folderScratch}")
