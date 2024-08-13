import numpy as np
from IPython import embed

from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools import UFILEStools, TRANSPtools
from mitim_tools.transp_tools.utils import NMLtools, TRANSPhelpers


def runTRANSPfromGACODE(folderTRANSP, machine="SPARC"):
    t = TRANSPtools.TRANSP(IOtools.expandPath(folderTRANSP), machine)
    t.defineRunParameters(
        "12345A01", "12345", mpisettings={"trmpi": 1, "toricmpi": 1, "ptrmpi": 1}
    )

    t.run()
    c = t.checkUntilFinished(label="run1", checkMin=30, retrieveAC=True)

    return c


def prepareTRANSPfromGACODE(self, folderTRANSP, machine="SPARC"):
    """
    Prepares inputs to TRANSP out of input.gacode
    """

    IOtools.askNewFolder(folderTRANSP)

    # ------------------------------------------------
    # ---- Quantities needed
    # ------------------------------------------------

    Ip = self.profiles["current(MA)"][0]
    rho = self.profiles["rho(-)"]
    q = self.profiles["q(-)"]
    Bt = self.profiles["bcentr(T)"][0]  # Is this correct?
    R = self.profiles["rcentr(m)"][0]
    Zeff = self.derived["Zeff_vol"]

    Te = self.profiles["te(keV)"]
    Ti = self.profiles["ti(keV)"][:, 0]
    ne = self.profiles["ne(10^19/m^3)"]

    Prf = self.derived["qRF_MWmiller"][-1]

    Rs = self.derived["R_surface"][-1]
    Zs = self.derived["Z_surface"][-1]

    # ------------------------------------------------
    # ---- Build UFILES
    # ------------------------------------------------

    uf = UFILEStools.UFILEtransp(scratch="cur")
    uf.Variables["Z"] = np.array([Ip * 1e6, Ip * 1e6])
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.CUR")

    uf = UFILEStools.UFILEtransp(scratch="rbz")
    uf.Variables["Z"] = np.array([Bt * R * 1e2, Bt * R * 1e2])
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.RBZ")

    uf = UFILEStools.UFILEtransp(scratch="rfp")
    uf.Variables["X"] = np.array([1])
    uf.Variables["Z"] = np.array([Prf * 1e6])
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.RFP")

    uf = UFILEStools.UFILEtransp(scratch="zf2")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = np.array([Zeff] * len(rho))
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.ZF2")

    uf = UFILEStools.UFILEtransp(scratch="qpr")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = q
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.QPR")

    uf = UFILEStools.UFILEtransp(scratch="ter")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = Te
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.TEL")

    uf = UFILEStools.UFILEtransp(scratch="ti2")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = Ti
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.TIO")

    uf = UFILEStools.UFILEtransp(scratch="ner")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = ne
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.NEL")

    # Zero UFILES
    uf = UFILEStools.UFILEtransp(scratch="vsf")
    uf.Variables["Z"] = np.array([0, 0])
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.VSF")

    uf = UFILEStools.UFILEtransp(scratch="gfd")
    uf.Variables["Z"] = np.array([0, 0])
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.GFD")

    uf = UFILEStools.UFILEtransp(scratch="df4")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = Te * 0.0
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.DHE4")

    uf = UFILEStools.UFILEtransp(scratch="vc4")
    uf.Variables["X"] = rho
    uf.Variables["Z"] = Te * 0.0
    uf.repeatProfile()
    uf.writeUFILE(f"{folderTRANSP}/PRF12345.VHE4")

    # ------------------------------------------------
    # ---- Build Boundary
    # ------------------------------------------------

    TRANSPhelpers.writeBoundary(f"{folderTRANSP}/BOUNDARY_123456_00001.DAT", Rs, Zs)
    TRANSPhelpers.writeBoundary(f"{folderTRANSP}/BOUNDARY_123456_99999.DAT", Rs, Zs)

    TRANSPhelpers.generateMRY(
        folderTRANSP,
        ["00001", "99999"],
        folderTRANSP,
        "12345",
        momentsScruncher=12,
        BtSign=-1,
        IpSign=-1,
        name="",
    )

    # ------------------------------------------------
    # ---- Machine
    # ------------------------------------------------

    if machine == "AUG":
        from mitim_tools.experiment_tools.AUGtools import defineFirstWall
    elif machine == "CMOD":
        from mitim_tools.experiment_tools.CMODtools import defineFirstWall
    elif machine == "D3D":
        from mitim_tools.experiment_tools.DIIIDtools import defineFirstWall
    elif machine == "SPARC":
        from mitim_tools.experiment_tools.SPARCtools import defineFirstWall

    rlim, zlim = defineFirstWall()

    TRANSPhelpers.addLimiters_UF(f"{folderTRANSP}/PRF12345.LIM", rlim, zlim)

    # ------------------------------------------------
    # ---- Namelist
    # ------------------------------------------------

    nml = NMLtools.default_nml("12345", machine)  # ,pservers=[nbi,ntoric,nptr])
    nml.write(BaseFile=f"{folderTRANSP}/12345A01TR.DAT")
    nml.appendNMLs()
