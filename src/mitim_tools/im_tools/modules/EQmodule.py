import os, sys, math, subprocess, copy, pdb, pickle, glob, socket
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate
from collections import OrderedDict

from mitim_tools.misc_tools import IOtools, FARMINGtools, MATHtools, GRAPHICStools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import CONFIGread

from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

from IPython import embed


def runEquilibrium(mitimNML, addPredictionOffset, automaticProcess=False):
    saveFigure = not automaticProcess

    initime = mitimNML.BaselineTime
    startRamp = mitimNML.startRamp
    endRamp = mitimNML.endRamp
    time1 = str(int(initime * 1000.0)).zfill(5)
    time2 = str(int(initime * 1000.0 + startRamp * 1000.0)).zfill(5)
    time3 = str(int(initime * 1000.0 + endRamp * 1000.0)).zfill(5)
    time4 = str(int(initime * 1000.0 + endRamp * 1000.0 + 9e3)).zfill(5)

    # Actual values

    rmajor = mitimNML.MITIMparams["rmajor"]
    epsilon = mitimNML.MITIMparams["epsilon"]
    kappa = mitimNML.MITIMparams["kappa"]
    delta = mitimNML.MITIMparams["delta"]
    zeta = mitimNML.MITIMparams["zeta"]
    z0 = mitimNML.MITIMparams["z0"]

    if mitimNML.MITIMparams["NLBCCW"]:
        BtSign = 1
    else:
        BtSign = -1
    if mitimNML.MITIMparams["NLJCCW"]:
        IpSign = 1
    else:
        IpSign = -1

    # ------------------------------------------
    # Generate Geometry (BOUNDARY file)
    # ------------------------------------------

    # Standard way
    if mitimNML.advancedEQ == 0:
        times, ax, fig = writeBoundaryFiles(
            mitimNML,
            rmajor,
            epsilon,
            kappa,
            delta,
            zeta,
            z0,
            addPredictionOffset,
            time1,
            time2,
            time3,
            time4,
            saveFigure=saveFigure,
        )

    # Using g-file from TSC (which contains many times)
    elif mitimNML.advancedEQ == 1:
        print(">> Extracting boundaries from TSC g-file...")
        times, ax, fig = extractBoundaries_TSC(
            mitimNML.prescribedEvolution,
            mitimNML.prescribedEvolutionTimes,
            mitimNML.FolderEQ,
            saveFigure=saveFigure,
        )

    # Using g-file from sweeps
    elif mitimNML.advancedEQ == 2:
        print(">> Preparing divertor sweep...")
        times, ax, fig = extractBoundaries_Sweep(
            mitimNML.prescribedEvolution,
            mitimNML.EQsweepTimes,
            mitimNML.FolderEQ,
            saveFigure=saveFigure,
        )

    # ------------------------------------------
    # Run Scruncher to generate MRY UFILE
    # ------------------------------------------
    if mitimNML.EquilibriumType.lower() == "miller" or mitimNML.machine == "AUG":
        momentsScruncher = 6
    elif mitimNML.EquilibriumType.lower() == "gfile":
        momentsScruncher = 12

    # For C-Mod I have found problems running 12 moments
    if rmajor < 1.0:
        momentsScruncher = 6

    if len(glob.glob(f"{mitimNML.FolderTRANSP}/*.MRY")) > 0:
        os.system(f"rm {mitimNML.FolderTRANSP}/*.MRY")
    generateMRY(
        mitimNML.FolderEQ,
        times,
        mitimNML.FolderTRANSP,
        mitimNML.nameBaseShot,
        momentsScruncher=momentsScruncher,
        BtSign=BtSign,
        IpSign=IpSign,
        name=mitimNML.nameRunTot,
    )

    # ------------------------------------------
    # Change External Structures
    # ------------------------------------------

    print(">> Changing machine structure...")

    changeExternalStructures(
        mitimNML.machine,
        mitimNML.MITIMparams,
        mitimNML.namelistPath,
        mitimNML.FolderTRANSP + f"PRF{mitimNML.nameBaseShot}.LIM",
        ax=ax,
        oversizedMachine=mitimNML.oversizedMachine,
        changeAntennas_ICH=mitimNML.changeHeatingHardware
        and mitimNML.PlasmaFeatures["ICH"],
        changeAntennas_ECH=mitimNML.changeHeatingHardware
        and mitimNML.PlasmaFeatures["ECH"],
        changeBeams_NBI=mitimNML.changeHeatingHardware
        and mitimNML.PlasmaFeatures["NBI"],
    )

    # ~~~ Write boundary to file
    if saveFigure:
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.set_xlim(left=0.0)
        ax.legend(loc="best")
        fig.savefig(f"{mitimNML.FolderEQ}/figMITIM.svg")
        plt.close("all")


def writeBoundaryFiles(
    mitimNML,
    rmajor,
    epsilon,
    kappa,
    delta,
    zeta,
    z0,
    addPredictionOffset,
    time1,
    time2,
    time3,
    time4,
    saveFigure=True,
):
    print(">> Creating boundary flux surface (LCFS)...")

    if not addPredictionOffset:
        time3 = time1

    # ~~~~~~~~~~~~~~~~
    # Primary boundary
    # ~~~~~~~~~~~~~~~~

    nameFile = f"{mitimNML.FolderEQ}/BOUNDARY_123456_{time3}.DAT"
    nameFile_extra = f"{mitimNML.FolderEQ}/BOUNDARY_123456_{time4}.DAT"

    # Miller geometry
    if mitimNML.EquilibriumType.lower() == "miller":
        print(">> Running equilibrium parametrization (Miller)")
        rs, zs = createMillerBoundaryFile(
            [rmajor, epsilon, kappa, delta, zeta, z0], nameFile, plotYN=False
        )
        _, _ = createMillerBoundaryFile(
            [rmajor, epsilon, kappa, delta, zeta, z0], nameFile_extra, plotYN=False
        )

    # Realistic geometry from g-file
    elif mitimNML.EquilibriumType.lower() == "gfile":
        print(">> Reading boundary flux surface from g-file...")
        g, rs, zs = createGfileBoundaryFile(
            mitimNML.gfile_loc, nameFile, name=mitimNML.nameRunTot
        )
        _, _, _ = createGfileBoundaryFile(
            mitimNML.gfile_loc,
            nameFile_extra,
            gfileAlreadyExists=g,
            name=mitimNML.nameRunTot,
        )

        # Copy gfile that geometry was created from, for consistency
        folderCopy = f"{mitimNML.FolderTRANSP}/EQ_folder"
        if not os.path.exists(folderCopy):
            os.system(f"mkdir {folderCopy}")
        os.system(f"cp {mitimNML.gfile_loc} {folderCopy}/gfile1.geq")

    # ~~~ Plot boundary
    if saveFigure:
        fig, ax = plt.subplots()
        ax.plot(rs, zs, "-o", markersize=0.5, lw=0.5, c="m", label="LCFS")
    else:
        ax, fig = None, None

    # ~~~~~~~~~~~~~~~~
    # Secondary boundaries for ramp-up (always miller even if final g-file based geometry)
    # ~~~~~~~~~~~~~~~~

    if addPredictionOffset:
        print(
            ">> Creating ramp-up (from t={0}ms to t={1}ms) boundaries with equilibrium from previous CDF".format(
                time2, time3
            )
        )

        Rb, Yb = MATHtools.downsampleCurve(
            mitimNML.Boundary_r, mitimNML.Boundary_z, nsamp=100
        )

        writeBoundary(f"{mitimNML.FolderEQ}/BOUNDARY_123456_{time1}.DAT", Rb, Yb)
        writeBoundary(f"{mitimNML.FolderEQ}/BOUNDARY_123456_{time2}.DAT", Rb, Yb)

        add, times = 1, [time1, time2, time3, time4]

    else:
        print(">> Not implementing eq-ramp")
        add, times = 0, [time3, time4]

    return times, ax, fig


def generateMRY(
    FolderEquilibrium,
    times,
    FolderMRY,
    nameBaseShot,
    momentsScruncher=12,
    BtSign=-1,
    IpSign=-1,
    name="",
):
    filesInput = [FolderEquilibrium + "/scrunch_in", FolderEquilibrium + "/ga.d"]

    if momentsScruncher > 12:
        print(
            "\t- SCRUNCHER may fail because the maximum number of moments is 12",
            typeMsg="w",
        )

    with open(FolderEquilibrium + "ga.d", "w") as f:
        for i in times:
            nam = f"BOUNDARY_123456_{i}.DAT"
            f.write(nam + "\n")
            filesInput.append(FolderEquilibrium + "/" + nam)

    # Generate answers to scruncher in file scrunch_in
    with open(FolderEquilibrium + "scrunch_in", "w") as f:
        f.write(f"g\nga.d\n{momentsScruncher}\na\nY\nN\nN\nY\nX")

    # Run scruncher
    print(
        f">> Running scruncher with {momentsScruncher} moments to create MRY file from boundary files..."
    )

    scruncher_job = FARMINGtools.mitim_job(FolderEquilibrium)

    scruncher_job.define_machine(
        "scruncher",
        f"tmp_scruncher_{name}/",
        launchSlurm=False,
    )

    scruncher_job.prep(
        "scruncher < scrunch_in",
        output_files=["M123456.MRY"],
        input_files=filesInput,
    )

    scruncher_job.run()

    fileUF = f"{FolderMRY}/PRF{nameBaseShot}.MRY"
    os.system(f"cp {FolderEquilibrium}/M123456.MRY {fileUF}")

    # Check if MRY file has the number of times expected
    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(fileUF)
    if len(UF.Variables["X"]) != len(times):
        raise Exception(
            " There was a problem in scruncher, at least one boundary time not used"
        )


def changeExternalStructures(
    machine,
    MITIMparams,
    namelistPath,
    UFilePath,
    ax=None,
    oversizedMachine=False,
    changeAntennas_ICH=True,
    changeAntennas_ECH=True,
    changeBeams_NBI=True,
):
    # Params
    rmajor = MITIMparams["rmajor"]
    epsilon = MITIMparams["epsilon"]
    kappa = MITIMparams["kappa"]
    delta = MITIMparams["delta"]
    zeta = MITIMparams["zeta"]
    z0 = MITIMparams["z0"]

    gapAntenna = MITIMparams["antenna_gap"]
    polext = MITIMparams["antenna_pol"]

    wall_gap = MITIMparams["wall_gap"]

    # ------------------------------------------
    # Change Vacuum Vessel and Limiters
    # ------------------------------------------

    if not oversizedMachine:
        addVacuumVessel(
            machine,
            namelistPath,
            UFilePath,
            rmajor,
            epsilon * rmajor,
            kappa,
            delta,
            zeta,
            z0,
            wall_gap,
            ax=ax,
            useDefaultLimiters=True,
        )
    # Big VV to avoid intersection
    else:
        print(">> Generating big VV to avoid intersection...")
        addVacuumVessel(
            machine,
            namelistPath,
            UFilePath,
            rmajor,
            rmajor - 0.1,
            1.0,
            0,
            0,
            0,
            0.0,
            ax=ax,
            useDefaultLimiters=False,
        )

    # ------------------------------------------
    # Change Antenna ICRF
    # ------------------------------------------
    if changeAntennas_ICH:
        addICRFAntenna(
            namelistPath,
            rmajor,
            rmajor * epsilon,
            antgap=gapAntenna,
            polext=polext,
            ax=ax,
        )

    # ------------------------------------------
    # Change Antenna ECRF
    # ------------------------------------------

    if changeAntennas_ECH:
        pass

    # ------------------------------------------
    # Change Beams NBI
    # ------------------------------------------

    if changeBeams_NBI:
        pass

    # ------------------------------------------
    # Change Interferometer chord
    # ------------------------------------------

    # addInterferometer(rmajor,rmajor*epsilon*kappa, ax = ax)


def addVacuumVessel(
    machine,
    namelistPath,
    UFilePath,
    rmajor,
    a,
    kappa,
    delta,
    zeta,
    z0,
    wall_gap,
    ax=None,
    LimitersInNML=False,
    useDefaultLimiters=False,
):
    # ------------------------------------------
    # Find R,Z points of physical vacuum vessel w/ limiters
    # ------------------------------------------

    # Miller parametrized vessel

    epsilon_new = (a + wall_gap) / rmajor
    rvv, zvv = makeMillerGeom(rmajor, epsilon_new, kappa, delta, zeta, z0)

    # ------------------------------------------
    # Vacuum vessel moments
    # ------------------------------------------
    nfour = 5
    VVRmom, VVZmom, rvv_fit, zvv_fit = decomposeMoments(
        rvv * 100.0, zvv * 100.0, nfour=nfour
    )

    for i in range(nfour):
        IOtools.changeValue(
            namelistPath,
            "VVRmom(" + str(i + 1) + ")",
            round(VVRmom[i], 3),
            None,
            "=",
            MaintainComments=True,
        )
        IOtools.changeValue(
            namelistPath,
            "VVZmom(" + str(i + 1) + ")",
            round(VVZmom[i], 3),
            None,
            "=",
            MaintainComments=True,
        )

    if ax is not None:
        ax.plot(rvv_fit / 100.0, zvv_fit / 100.0, c="b", lw=1, label="VV")

    # ------------------------------------------
    # Limiters (i.e. physical vacuum vessel that the plasma sees (mostly for orbit losses of beams))
    # ------------------------------------------

    machinesWithFirstWall = ["AUG", "CMOD", "D3D", "SPARC"]

    if useDefaultLimiters and machine in machinesWithFirstWall:
        print(f"\t- Using limiters from {machine} defaults")

        if machine == "AUG":
            from mitim_tools.experiment_tools.AUGtools import defineFirstWall
        elif machine == "CMOD":
            from mitim_tools.experiment_tools.CMODtools import defineFirstWall
        elif machine == "D3D":
            from mitim_tools.experiment_tools.DIIIDtools import defineFirstWall
        elif machine == "SPARC":
            from mitim_tools.experiment_tools.SPARCtools import defineFirstWall

        rlim, zlim = defineFirstWall()

    if not useDefaultLimiters or machine not in machinesWithFirstWall:
        print(f"\t- Building limiter for {machine}, coincidental with VV")
        rlim, zlim = rvv_fit / 100.0, zvv_fit / 100.0

    if LimitersInNML:
        addLimiters_NML(namelistPath, rlim, zlim, [rmajor, z0], ax=ax)
    else:
        addLimiters_UF(UFilePath, rlim, zlim, ax=ax)


def addLimiters_NML(namelistPath, rs, zs, centerP, ax=None):
    numLim = 10

    x, y = MATHtools.downsampleCurve(rs, zs, nsamp=numLim + 1)
    x = x[:-1]
    y = y[:-1]

    t = []
    for i in range(numLim):
        t.append(
            -np.arctan((y[i] - centerP[1]) / (x[i] - centerP[0])) * 180 / np.pi + 90.0
        )

    alnlmr_str = IOtools.ArrayToString(x)
    alnlmy_str = IOtools.ArrayToString(y)
    alnlmt_str = IOtools.ArrayToString(t)
    IOtools.changeValue(
        namelistPath, "nlinlm", numLim, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "alnlmr", alnlmr_str, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "alnlmy", alnlmy_str, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "alnlmt", alnlmt_str, None, "=", MaintainComments=True
    )

    if ax is not None:
        ax.plot(
            x / 100.0, y / 100.0, 100, "-o", markersize=0.5, lw=0.5, c="k", label="lims"
        )


def addLimiters_UF(UFilePath, rs, zs, ax=None, numLim=100):
    # ----- ----- ----- ----- -----
    # Ensure that there are no repeats points
    rs, zs = IOtools.removeRepeatedPoints_2D(rs, zs)
    # ----- ----- ----- ----- -----

    x, y = rs, zs  # MATHtools.downsampleCurve(rs,zs,nsamp=numLim)

    # Write Ufile
    UF = UFILEStools.UFILEtransp(scratch="lim")
    UF.Variables["X"] = x
    UF.Variables["Z"] = y
    UF.writeUFILE(UFilePath)

    if ax is not None:
        ax.plot(x, y, "-o", markersize=0.5, lw=0.5, c="k", label="lims")

    print(
        f">> Limiters UFile created in ...{UFilePath[np.max([-40, -len(UFilePath)]):]}"
    )


def addICRFAntenna(namelistPath, rmajor, a, antgap=0.01, polext=73.3, ax=None):
    antrmaj = rmajor
    antrmin = (-antrmaj + rmajor) + a + antgap

    IOtools.changeValue(
        namelistPath,
        "rmjicha",
        round(antrmaj * 100.0, 3),
        None,
        "=",
        MaintainComments=True,
    )
    IOtools.changeValue(
        namelistPath,
        "rmnicha",
        round(antrmin * 100.0, 3),
        None,
        "=",
        MaintainComments=True,
    )
    IOtools.changeValue(
        namelistPath, "thicha", polext, None, "=", MaintainComments=True
    )

    # Reconstruct Antenna for plotting purposes
    R, Z = reconstructAntenna(antrmaj, antrmin, polext)

    if ax is not None:
        ax.plot(R, Z, c="g", lw=1, label="icrf")


def reconstructAntenna(antrmaj, antrmin, polext):
    theta = np.linspace(-polext / 2.0, polext / 2.0, 100)
    R = []
    Z = []
    for i in theta:
        R.append(antrmaj + antrmin * np.cos(i * np.pi / 180.0))
        Z.append(antrmin * np.sin(i * np.pi / 180.0))

    return np.array(R), np.array(Z)


def addInterferometer(namelistPath, rmajor, b, factorUp=1.5, ax=None):
    R = rmajor
    Y = b * factorUp

    IOtools.changeValue(namelistPath, "NLDA", 1, None, "=", MaintainComments=True)
    IOtools.changeValue(
        namelistPath, "RLDA(1)", R * 100.0, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "YLDA(1)", Y * 100.0, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "THLDA(1)", -90.0, None, "=", MaintainComments=True
    )
    IOtools.changeValue(namelistPath, "PHLDA(1)", 0.0, None, "=", MaintainComments=True)


def createMillerBoundaryFile(x, nameFile, plotYN=False):
    print(
        ">> Generating scruncher input, named ..."
        + nameFile[np.max([-40, -len(nameFile)]) :]
    )

    # Input values are provide via command line call (see above)
    rmajor = float(x[0])
    epsilon = float(x[1])
    kappa = float(x[2])
    delta = float(x[3])
    zeta = float(x[4])
    z0 = float(x[5])

    rs, zs = makeMillerGeom(rmajor, epsilon, kappa, delta, zeta, z0)

    if plotYN:
        plt.figure()
        plt.scatter(rs, zs)
        plt.scatter([rmajor], [0])
        plt.axis("equal")
        pdb.set_trace()

    writeBoundary(nameFile, rs, zs)

    return rs, zs


def createGfileBoundaryFile(
    gfile_loc,
    nameFile,
    plotYN=False,
    gfileAlreadyExists=None,
    runRemote=False,
    name="",
    nameRFSZFS=None,
):
    if gfileAlreadyExists is None:
        g = GEQtools.MITIMgeqdsk(gfile_loc, fullLCFS=True)
    else:
        g = gfileAlreadyExists

    # writeBoundary(nameFile,g.Rb,g.Yb)
    writeBoundary(nameFile, g.Rb_prf, g.Yb_prf)
    readGFILEmain(g)

    if nameRFSZFS is not None:
        print("\t- Manual construction of RFS and ZFS equilibrium UFILES requested")
        theta, rho, R, Z = extractRFSZFS(g, runRemote=runRemote)
        UFILEStools.writeRFSZFS(theta, rho, R, Z, prefix=nameRFSZFS)

    if plotYN:
        fig, ax = plt.subplots()
        g.plotLCFS(ax=ax)
        ax.plot(g.Rb, g.Yb, "--*", c="r")
        plt.show()

    return g, g.Rb, g.Yb


def extractRFSZFS(g, rhos=np.linspace(0, 1, 11), runRemote=False, plotYN=True):
    rmajor, epsilon, kappa, delta, zeta, z0 = g.paramsLCFS()

    theta_common = np.linspace(0, 2 * np.pi, 1000)

    R, Z = [], []
    for rho in rhos:
        if rho == 1.0:
            # rho = 0.99999
            # plotDebug 	= True
            # resol 		= 5E4

            fileCreely = "~/scratch/separatrixRZv2"
            with open(fileCreely, "r") as f:
                aux = f.readlines()
            r, z = [], []
            for line in aux:
                r0, z0 = [float(i) for i in line.split()]
                r.append(r0)
                z.append(z0)
            r, z = np.array(r), np.array(z)

        else:
            plotDebug = False
            resol = 5e3
            r, z = g.findSurface(
                rho, runRemote=runRemote, resol=resol, downsampleNum=None
            )

        theta = []
        for i in range(len(r)):
            ghZ = z[i] - 0
            ghR = r[i] - rmajor
            a = np.arctan2(ghZ, ghR)
            theta_single = a * (a >= 0) + (a + 2 * np.pi) * (
                a < 0
            )  # let's make it between 0 and 2*pi
            theta.append(theta_single)
        theta = np.array(theta)

        # Order arrays according to theta (unique)
        r = MATHtools.orderArray(r, base=theta)
        z = MATHtools.orderArray(z, base=theta)
        theta = MATHtools.orderArray(theta, base=theta)

        # Complement

        r = np.interp(theta_common, theta, r, period=2 * np.pi)
        z = np.interp(theta_common, theta, z, period=2 * np.pi)

        R.append(r)
        Z.append(z)

    R = np.transpose(np.array(R))
    Z = np.transpose(np.array(Z))

    if plotYN:
        fig, ax = plt.subplots(ncols=2)
        cols = GRAPHICStools.listColors()
        for i in range(len(rhos)):
            ax[0].plot(R[:, i], Z[:, i], "o-", c=cols[i])
            ax[1].plot(theta_common, R[:, i], "-*", c=cols[i])
            ax[1].plot(theta_common, Z[:, i], "-o", c=cols[i])
        g.plotFluxSurfaces(fluxes=[1.0], ax=ax[0], color="g", alpha=1.0, label="")
        g.plotLCFS(ax=ax[0])

        plt.show()
        embed()

    """	
	Until here, R and Z have shape (itheta,irho), with rhos and theta_common as axes
	"""

    return theta_common, np.array(rhos), R, Z


def readGFILEmain(g):
    rmajor, epsilon, kappa, delta, zeta, z0 = g.paramsLCFS()

    print(">> Boundary based on G-File has been created. Parameters:")
    print(
        "\tRmajor = {0:.2f}m, epsilon = {1:.3f} (a = {2:.2f}m), kappa = {3:.2f}, delta = {4:.2f}, zeta = {5:.2f},z0 = {6:.2f}m".format(
            rmajor, epsilon, epsilon * rmajor, kappa, delta, zeta, z0
        )
    )


def writeBoundary(nameFile, rs_orig, zs_orig):
    numpoints = len(rs_orig)

    closerInteg = int(numpoints / 10)

    if closerInteg * 10 < numpoints:
        extraneeded = True
        rs = np.reshape(rs_orig[: int(closerInteg * 10)], (int(closerInteg), 10))
        zs = np.reshape(zs_orig[: int(closerInteg * 10)], (int(closerInteg), 10))
    else:
        extraneeded = False
        rs = np.reshape(rs_orig, (int(closerInteg), 10))
        zs = np.reshape(zs_orig, (int(closerInteg), 10))

    # Write the file for scruncher
    f = open(nameFile, "w")
    f.write("Boundary description for timeslice 123456 1000 msec\n")
    f.write("Shot date: 29AUG-2018\n")
    f.write("UNITS ARE METERS\n")
    f.write(f"Number of points: {numpoints}\n")
    f.write(
        "Begin R-array ==================================================================\n"
    )
    f.close()

    f = open(nameFile, "a")
    np.savetxt(f, rs, fmt="%7.3f")
    if extraneeded:
        rs_extra = np.array([rs_orig[int(closerInteg * 10) :]])
        np.savetxt(f, rs_extra, fmt="%7.3f")
    f.write(
        "Begin z-array ==================================================================\n"
    )
    np.savetxt(f, zs, fmt="%7.3f")
    if extraneeded:
        zs_extra = np.array([zs_orig[int(closerInteg * 10) :]])
        np.savetxt(f, zs_extra, fmt="%7.3f")
    f.close()


def makeMillerGeom(rmajor, epsilon, kappa, delta, zeta, z0):
    # This script generates a flux surface based on Miller geometry.
    #
    # The inputs are a series of arguments which represent shaping and size values for the surface. The output is two arrays of values corresponding to the R and Z coordinates of the flux surface.
    #
    # =============================================================================
    # The inputs in order are
    # python make_fs.py rmajor epsilon kappa delta zeta z0
    # =============================================================================

    a = rmajor * epsilon

    if kappa < 1.1 and delta < 0.1:
        numtheta = 80
    else:
        numtheta = 210

    theta = np.zeros(numtheta)

    for i in range(0, numtheta):
        theta[i] = (i / float(numtheta - 1)) * 2 * math.pi

    # Calculate the R and Z values

    rs = np.zeros(numtheta)
    zs = np.zeros(numtheta)
    for i in range(0, numtheta):
        rs[i] = round(
            rmajor + a * math.cos(theta[i] + math.asin(delta) * math.sin(theta[i])), 3
        )
        zs[i] = round(
            z0 + a * kappa * math.sin(theta[i] + zeta * math.sin(2 * theta[i])), 3
        )

    return rs, zs


def decomposeMoments(R, Z, nfour=5):
    nth = len(R)

    # Initial Guess for R and Z array
    r = [180, 70, 3.0]  # [rmajor,anew,3.0,0,0]
    z = [0.0, 140, -3.0]  # [0.0,anew*kappa,-3.0,0.0,0.0]

    for i in range(nfour - 3):
        r.append(0)
        z.append(0)

    x = np.concatenate([r, z])

    theta = np.zeros(nth)
    for i in range(0, nth):
        theta[i] = (i / float(nth - 1)) * 2 * math.pi

    def vv_func(x):
        cosvals = np.zeros((nfour, nth))
        sinvals = np.zeros((nfour, nth))
        r_eval = np.zeros(nth)
        z_eval = np.zeros(nth)

        # Evaluate the r value thetas
        for i in range(0, nfour):
            thetas = i * theta
            for j in range(0, nth):
                cosvals[i, j] = x[i] * math.cos(thetas[j])

        # Evaluate the z value thetas
        for i in range(nfour, 2 * nfour):
            thetas = (i - nfour) * theta
            for j in range(0, nth):
                sinvals[i - nfour, j] = x[i] * math.sin(thetas[j])

            for i in range(0, nth):
                r_eval[i] = np.sum(cosvals[:, i])
                z_eval[i] = np.sum(sinvals[:, i])

        rsum = np.sum(abs(r_eval - R))
        zsum = np.sum(abs(z_eval - Z))

        metric = rsum + zsum

        return metric

    # ----------------------------------------------------------------------------------
    # Perform optimization
    # ----------------------------------------------------------------------------------

    print(f">> Performing minimization to fit VV moments ({nfour})...")

    res = minimize(
        vv_func,
        x,
        method="nelder-mead",
        options={"xtol": 1e-4, "disp": verbose_level in [4, 5]},
    )

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    x = res.x

    rmom = x[0:nfour]
    zmom = x[nfour : 2 * nfour]

    # Convert back to see error
    cosvals = np.zeros((nfour, nth))
    sinvals = np.zeros((nfour, nth))
    r_eval = np.zeros(nth)
    z_eval = np.zeros(nth)

    # Evaluate the r value thetas
    for i in range(0, nfour):
        thetas = i * theta
        for j in range(0, nth):
            cosvals[i, j] = x[i] * math.cos(thetas[j])

        # Evaluate the z value thetas
        for i in range(nfour, 2 * nfour):
            thetas = (i - nfour) * theta
            for j in range(0, nth):
                sinvals[i - nfour, j] = x[i] * math.sin(thetas[j])

            for i in range(0, nth):
                r_eval[i] = np.sum(cosvals[:, i])
                z_eval[i] = np.sum(sinvals[:, i])

            rsum = np.sum(abs(r_eval - R))
            zsum = np.sum(abs(z_eval - Z))

    return rmom, zmom, r_eval, z_eval


def reconstructVV(VVr, VVz, nth=1000):
    nfour = len(VVr)

    x = np.append(VVr, VVz)

    theta = np.zeros(nth)
    for i in range(0, nth):
        theta[i] = (i / float(nth - 1)) * 2 * math.pi

    cosvals = np.zeros((nfour, nth))
    sinvals = np.zeros((nfour, nth))
    r_eval = np.zeros(nth)
    z_eval = np.zeros(nth)

    for i in range(0, nfour):
        thetas = i * theta
        # print(x[i])
        for j in range(0, nth):
            cosvals[i, j] = x[i] * math.cos(thetas[j])

        # Evaluate the z value thetas
        for i in range(nfour, 2 * nfour):
            thetas = (i - nfour) * theta
            for j in range(0, nth):
                sinvals[i - nfour, j] = x[i] * math.sin(thetas[j])

            for i in range(0, nth):
                r_eval[i] = np.sum(cosvals[:, i])
                z_eval[i] = np.sum(sinvals[:, i])

    return r_eval, z_eval


def findParametrization(R, Y):
    rmajor = (np.max(R) + np.min(R)) / 2.0
    a = (np.max(R) - np.min(R)) / 2.0
    b = (np.max(Y) - np.min(Y)) / 2.0
    Ymajor = Y[np.argmax(R)]

    Rtop = R[np.argmax(Y)]
    Rbot = R[np.argmin(Y)]
    ctop = rmajor - Rtop
    cbot = rmajor - Rbot

    epsilon = a / rmajor
    kappa = b / a
    delta = (ctop / a + cbot / a) / 2.0
    zeta = 0.0
    z0 = Ymajor

    roundint = 2
    return (
        round(rmajor, roundint),
        round(epsilon, 3),
        round(kappa, roundint),
        round(delta, roundint),
        round(zeta, roundint),
        round(z0, roundint),
    )


def extractBoundaries_TSC(
    gfileName,
    times,
    folderOutput,
    saveFigure=True,
    runRemote=False,
    onlyExtractGfiles=False,
):
    """
    TSC produces non-standard g-files that contain many times.
    """

    # ~~~ Time strings

    timesW = []
    for i in times:
        timesW.append(str(int(i * 1000.0)).zfill(5))
    times = timesW

    # ~~~ Extract all possible individual g-files

    folderOutput = IOtools.cleanPath(folderOutput)

    cont, rsT, zsT = 0, OrderedDict(), OrderedDict()
    with open(gfileName, "r") as openfileobject:
        for line in openfileobject:
            if "c..." in line:
                if cont > 0 and not onlyExtractGfiles:
                    # Individual g-file has been written
                    f.close()

                    nameBound = f"{folderOutput}/BOUNDARY_123456_{timesec}.DAT"
                    _, rs, zs = createGfileBoundaryFile(
                        namefile, nameBound, runRemote=runRemote
                    )

                    rsT[timesec] = rs
                    zsT[str(timesec)] = zs

                cont += 1
                timesec = times[cont - 1]
                namefile = folderOutput + f"geqdsk_{timesec}s.geq"
                f = open(namefile, "w")

            # else:
            f.write(line)

    if not onlyExtractGfiles:
        # Closing last one
        f.close()
        nameBound = f"{folderOutput}/BOUNDARY_123456_{timesec}.DAT"
        _, rs, zs = createGfileBoundaryFile(namefile, nameBound, runRemote=runRemote)
        rsT[str(timesec)] = rs
        zsT[str(timesec)] = zs

    # ~~~ Plot boundary
    if saveFigure and not onlyExtractGfiles:
        fig, ax = plt.subplots()
        ax.plot(rs, zs, lw=4, c="m", label="LCFS")
        for ikey in rsT:
            ax.plot(rsT[ikey], zsT[ikey], lw=1, label=ikey)
    else:
        ax = None
        fig = None

    return timesW, ax, fig


def extractBoundaries_Sweep(gfileNames, times, folderOutput, saveFigure=True):
    # ~~~ Convert times to strings
    timesW = []
    for i in times:
        timesW.append(str(int(i * 1000.0)).zfill(5))
    times = timesW

    # ~~~ Baseline boundary (twice)

    g, rs, zs = createGfileBoundaryFile(
        gfileNames[0], f"{folderOutput}/BOUNDARY_123456_{times[0]}.DAT"
    )
    _, _, _ = createGfileBoundaryFile(
        gfileNames[0],
        f"{folderOutput}/BOUNDARY_123456_{times[1]}.DAT",
        createGfileBoundaryFile=g,
    )

    for i in range((len(times) - 2) // 2):
        # Bounday sweep extreme 1
        _, rs1, zs1 = createGfileBoundaryFile(
            gfileNames[1],
            f"{folderOutput}/BOUNDARY_123456_{times[(i + 1) * 2]}.DAT",
        )

        # Bounday sweep extreme 2
        _, rs2, zs2 = createGfileBoundaryFile(
            gfileNames[2],
            f"{folderOutput}/BOUNDARY_123456_{times[(i + 1) * 2 + 1]}.DAT",
        )

    # ~~~ Plot boundary
    if saveFigure:
        fig, ax = plt.subplots()
        ax.plot(rs, zs, lw=4, c="m", label="LCFS")
        ax.plot(rs1, zs1, lw=3, c="b", label="e1")
        ax.plot(rs2, zs2, lw=3, c="g", label="e2")
    else:
        ax = None
        fig = None

    return times, ax, fig


def addMMX(mmx_loc, file, time=1.0):
    print(f">> Using MMX file from {mmx_loc[40:]}, extracted at t={time:.3}s")
    os.system(f"cp {mmx_loc} {file}")
    UFILEStools.reduceTimeUFILE(file, time, newTimes=[0.0, 100.0])


# --------------------------- GFILE handling


def storeBoundary(g, fileStore):
    with open(fileStore, "wb") as handle:
        pickle.dump(g, handle, protocol=2)


def geqdsk_TSC(gfileName, times, scratchFolder="~/scratch/"):
    scratchFolder = IOtools.expandPath(scratchFolder) + "tmp_gfileTSC/"
    os.system("mkdir " + scratchFolder)

    timesW, _, _ = extractBoundaries_TSC(
        gfileName, times, scratchFolder, onlyExtractGfiles=True
    )

    gs = {}
    for it in range(len(times)):
        gs[times[it]] = GEQtools.MITIMgeqdsk(
            scratchFolder + f"geqdsk_{timesW[it]}s.geq"
        )

    os.system("rm -r " + scratchFolder)

    return gs


def createSetqProfilesFromTSC(
    gfileName, times, outputpickle, scratchFolder="~/scratch/", runRemote=False
):
    gs = geqdsk_TSC(gfileName, times, scratchFolder=scratchFolder, runRemote=runRemote)

    rhopol, t, q = [], [], []
    for i in gs:
        t.append(i)
        rhopol.append(gs[i].Ginfo["RHOp"])
        q.append(gs[i].Ginfo["QPSI"])

    dictQ = {"t": np.array(t), "rhopol": np.array(rhopol), "q": np.array(q)}
    with open(outputpickle, "wb") as handle:
        pickle.dump(dictQ, handle, protocol=2)


def readBoundary(RFS, ZFS, time=0.0, rho_index=-1):
    # Check if MRY file has the number of times expected
    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(RFS)
    t = UF.Variables["X"]
    theta = UF.Variables["Y"]
    Rs = UF.Variables["Z"]
    UF.readUFILE(ZFS)
    Zs = UF.Variables["Z"]

    it = np.argmin(np.abs(t - time))
    R, Z = Rs[it, :, rho_index], Zs[it, :, rho_index]

    return R, Z
