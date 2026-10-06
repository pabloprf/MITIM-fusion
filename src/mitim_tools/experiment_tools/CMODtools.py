"""C-Mod TRANSP helpers: namelist structures, first wall, ICRF antennas, namelist
translation, and the TRANSP-tree reader (namelist + input UFILEs of a C-Mod run).
Experimental data (EFIT, Thomson, HIREXSR, ...) lives in `experiment_tools.cmod.retrieval`.
"""

import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.misc_tools.LOGtools import printMsg as print


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


def ICRFantennas(MHz=[80.0, 78.0]):

    lines = [
        "! ----- Antenna Parameters",
        f"nicha       = 2         \t ! Number of ICRH antennae",
        f"frqicha     = {MHz[0]}e6,{MHz[1]}e6 ! Frequency of antenna (Hz)",
        "!prficha    = 0.0,0.0       ! Power of antenna (W)",
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


def updateTRANSPfromNML(nml_old, nml_new, folderWork, MITIMmodified=False):
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
    nml_dict["extbol"], nml_dict["prebol"] = "'BOL'", "'MIT'"

    # ---- Rotation is specified

    nml_dict["extvp2"], nml_dict["prevp2"] = "'VP2'", "'MIT'"

    # ---- Use the same coordinates in UFILES and names

    for i in ["ter", "ti2", "ner", "bol", "vp2"]:
        nml_dict["nri" + i] = -4

    # ----- Change names to those that I understand

    for i, j in zip(["NER", "TER", "TI2"], ["NEL", "TEL", "TIO"]):
        (folderWork / f'MIT{shotnum}.{i}').replace(folderWork / f'MIT{shotnum}.{j}')

    # ---- Add C-Mod limiter

    rlim, zlim = defineFirstWall()
    TRANSPhelpers.addLimiters_UF(folderWork / f"MIT{shotnum}.LIM", rlim, zlim)

    # ---- No gas flow (my way is to give this file)

    gasflow = 0.0
    UFILEStools.quickUFILE(
        None, gasflow, folderWork / f"MIT{shotnum}.GFD", typeuf="gfd"
    )

    # ---- Zeff specified as a uniform profile (my way)

    xZeff, Zeff = np.linspace(0, 1, 10), np.ones(10) * IOtools.findValue(
        nml_old, "xzeffi", "="
    )
    UFILEStools.quickUFILE(xZeff, Zeff, folderWork / f"MIT{shotnum}.ZF2", typeuf="zf2")

    # This file is useless
    (folderWork / f"MIT{shotnum}.ZEF").unlink(missing_ok=True)

    # ---- Ti validity

    nml_dict["tixlim"] = IOtools.findValue(nml_old, "tixlim", "=")

    # ---- Sawtooth trigger from UFILE

    nml_dict["nlsaw_trigger"] = False
    nml_dict["model_sawtrigger"] = 0
    nml_dict["sawtooth_period"] = 0
    nml_dict["c_sawtooth(2)"] = 0
    nml_dict["extsaw"], nml_dict["presaw"] = "'SAW'", "'MIT'"

    # Let's not include neutrons
    (folderWork / f"MIT{shotnum}.NTX").unlink(missing_ok=True)

    # ---------------------------------------
    # Simulation settings
    # ---------------------------------------

    # ---- Initialized by parametrized loop voltage (no QPR)

    nml_dict["nefld"] = 3
    nml_dict["qefld"], nml_dict["rqefld"], nml_dict["xpefld"] = 0.0, 0.0, 2.0
    nml_dict["extqpr"] = nml_dict["preqpr"] = nml_dict["nriqpr"] = None

    if not MITIMmodified:
        """
        These are settings that are not strickly experimental, but choices made by PRETRANSP.
        Here, I can choose them (to reproduce the PRETRANSP exactly, with MITIMmodified=False)
        or use what I think it's best (MITIMmodified=True)
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
        nml_dict["premmx"], nml_dict["extmmx"] = '"MIT"', '"MMX"'
    # --------------------------------------------------------

    return nml_dict


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


_UF_INPUTS = ["bol", "ner", "ter", "ti2", "vp2", "cur", "saw", "ntx", "rbz", "vsf", "zef", "mry", "rfp"]


def getTRANSP_MDS(runid, runid_new, folderWork="~/scratch/test/", toric_mpi=1,
                  connection=None, tunnel_host=None):
    """Write the namelist and input UFILEs of C-Mod TRANSP run `runid` (the "transp" tree on
    alcdata) into `folderWork` as run `runid_new`. Reads through a CMODConnection (mdsthin):
    pass `connection` to reuse one, or `tunnel_host` off-site (see experiment_tools.MDStools)."""
    from mitim_tools.experiment_tools.cmod.retrieval import CMODConnection

    folderWork = IOtools.expandPath(folderWork)
    if not folderWork.exists():
        IOtools.askNewFolder(folderWork)

    own = connection is None
    connection = connection or CMODConnection(tunnel_host=tunnel_host)
    try:
        conn = connection.conn
        conn.openTree("transp", int(runid))

        # Namelist
        nml = np.atleast_1d(conn.get(r"\TRANSP::TOP:NAME_LIST").data())
        nml_file = folderWork / f"{runid_new}TR.DAT"
        with open(nml_file, "w") as f:
            for line in nml:
                f.write((line.decode("UTF-8") if isinstance(line, bytes) else str(line)) + "\n")

        IOtools.changeValue(nml_file, "NSHOT", runid, [], "=")
        IOtools.changeValue(nml_file, "KMDSPLUS", None, [], "=")
        if toric_mpi > 1:
            IOtools.changeValue(nml_file, "ntoric_pserve", 1, [], "=")

        # UFILES
        for name in _UF_INPUTS:
            print(f"Reading {name}")
            labelX = " r/a                           " if name in ["ter", "ti2", "ner", "bol", "vp2"] else None
            try:
                nodeToUF(conn, runid, name, name.upper(), folderWork, labelX=labelX)
                IOtools.changeValue(nml_file, "PRE" + name.upper(), "'MIT'", [], "=")
            except Exception:
                print("\t~~ Could not retrieve")
                if name == "mry":
                    print("\t~~ No MRY stored for this run: provide the equilibrium UFILE (MRY/MMX) yourself",
                          typeMsg="w")
    finally:
        if own:
            connection.close()


def nodeToUF(conn, runid, name, nameMDS, folderWork, inputs=r"\TRANSP::TOP.INPUTS:", labelX=None):
    """One TRANSP-tree input node -> UFILE MIT<runid>.<nameMDS> (conn: mdsthin connection with
    the transp tree open). mdsthin returns arrays with reversed (C-order) dims, hence the transposes."""
    uf = UFILEStools.UFILEtransp(scratch=name, labelX=labelX)
    node = inputs + nameMDS
    val = lambda e: np.asarray(conn.get(e).data())

    uf.Variables["Z"] = val(node)
    if uf.dim == 1:
        uf.Variables["X"] = val(f"dim_of({node},0)")
    elif uf.dim == 2:
        uf.Variables["X"] = val(f"dim_of({node},1)")
        uf.Variables["Y"] = val(f"dim_of({node},0)")
        uf.Variables["Z"] = np.transpose(uf.Variables["Z"])
    elif uf.dim == 3:
        uf.Variables["X"] = val(f"dim_of({node},0)")
        uf.Variables["Y"] = val(f"dim_of({node},1)")
        uf.Variables["Q"] = val(f"dim_of({node},2)")
        uf.Variables["Z"] = np.transpose(uf.Variables["Z"])

    filename = folderWork / f"MIT{runid}.{nameMDS}"
    uf.writeUFILE(filename)
    return uf
