import os
import copy
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools, TGYROtools
from mitim_tools.misc_tools.IOtools import printMsg as print

"""
__________________
To run standalone:
	run ~/MITIM/mitim_opt/mitim/aux/PORTALScgyro.py ./run5/ ~/PRF/mitim_cgyro/sparc_results.txt 0,1,2,3,4
or
	run ~/MITIM/mitim_opt/mitim/aux/PORTALScgyro.py ./run5/ ~/PRF/mitim_cgyro/sparc_results.txt 0[Evaluation.X] 0[position_in_txt]
__________________
The CGYRO file must contain GB units, and the gb unit is MW/m^2, 1E19m^2/s
The CGYRO file must use particle flux. Convective transformation occurs later
"""

trick_hardcoded_f = None

# train_sep is the number of initial runs in it#0 results file. Now, it's usually 1
# start_num is the number of the first iteration, usually 0
# trick_harcoded_f is the name of the file until the iteration number. E.g. '/Users/pablorf/PROJECTS/project_2022_mitimalcc/iter/cgyro3/run3_recover/Outputs/cgyro_results/iter_rmp_75_'

# e.g.:
# includeMtAndGz_hardcoded, train_sep,start_num,last_one,trick_hardcoded_f = True, 1, 0,100, '/Users/pablorf/PROJECTS/project_2024_PORTALSdatabase/d3d_iss5chan_arx2023/mitim_run1/Outputs/cgyro_results/d3d_5chan_it_'


def evaluateCGYRO(
    PORTALSparameters, folder, numPORTALS, FolderEvaluation, unmodified_profiles, rad=4
):
    print(
        "\n ** CGYRO evaluation of fluxes has been requested before passing information to the STRATEGY module **",
        typeMsg="i",
    )

    minErrorPercent = PORTALSparameters["percentError_stable"]
    Qi_criterion_stable = PORTALSparameters["Qi_criterion_stable"]
    percentNeo = PORTALSparameters["percentError"][1]
    useConvectiveFluxes = PORTALSparameters["useConvectiveFluxes"]
    impurityPosition = PORTALSparameters["ImpurityOfInterest"]
    OriginalFimp = PORTALSparameters["fImp_orig"]

    print(
        "\t- Suggested command to send files (not .new) for CGYRO evaluation:",
        typeMsg="i",
    )
    lapfile = f"{IOtools.expandPath(folder,ensurePathValid=True)}/Outputs/ProfilesEvaluated/input.gacode.{numPORTALS}"
    mfefol = f"{IOtools.expandPath(folder,ensurePathValid=True).replace('/Users/pablorf/PRF/','/home/pablorf/PROJECTS/')}/ProfilesEvaluated/"
    print(f"\t\tscp -P 9224 {lapfile} mferws01.psfc.mit.edu:{mfefol}/.")

    cgyroing_file = (
        lambda file_cgyro, numPORTALS_this=0, includeMtAndGz=False: cgyroing(
            FolderEvaluation,
            unmodified_profiles,
            numPORTALS,
            minErrorPercent,
            Qi_criterion_stable,
            useConvectiveFluxes,
            percentNeo,
            OriginalFimp=OriginalFimp,
            evaluationsInFile=f"{numPORTALS_this}",
            rad=rad,
            impurityPosition=impurityPosition,
            file=file_cgyro,
            includeMtAndGz=includeMtAndGz,
        )
    )
    print(
        f"\t- Suggested function call for mitim evaluation {numPORTALS} (lambda for cgyroing):",
        typeMsg="i",
    )
    print(
        f"\tcgyroing_file('{IOtools.expandPath(folder,ensurePathValid=True)}/Outputs/cgyro_results/cgyro_it_{numPORTALS}.txt')"
    )

    print('\t- Then insert "exit" and RETURN', typeMsg="i")
    if (trick_hardcoded_f is None) or (int(numPORTALS) > last_one):
        embed()
    else:
        # ------------------------------------------------------------------
        # Hard-coded stuff for quick modifications
        # ------------------------------------------------------------------
        if int(numPORTALS) < train_sep:
            cgyroing_file(
                f"{trick_hardcoded_f}{start_num}.txt",
                numPORTALS_this=numPORTALS,
                includeMtAndGz=includeMtAndGz_hardcoded,
            )
        else:
            cgyroing_file(
                f"{trick_hardcoded_f}{int(numPORTALS)-train_sep+1+start_num}.txt",
                numPORTALS_this=0,
                includeMtAndGz=includeMtAndGz_hardcoded,
            )


def cgyroing(
    FolderEvaluation,
    unmodified_profiles,
    evaluations,
    minErrorPercent,
    Qi_criterion_stable,
    useConvectiveFluxes,
    percentNeo,
    OriginalFimp=1.0,
    rad=4,
    file=None,
    evaluationsInFile=0,
    impurityPosition=3,
    includeMtAndGz=False,
):
    """
    Variables need to have dimensions of (evaluation,rho)
    """

    evaluations = np.array([int(i) for i in evaluations.split(",")])
    evaluationsInFile = np.array([int(i) for i in evaluationsInFile.split(",")])

    (
        aLTe,
        aLTi,
        aLne,
        Q_gb,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        QeE,
        QiE,
        GeE,
        GZE,
        MtE,
        PexchE,
        _,
        _,
    ) = readCGYROresults(file, rad=rad, includeMtAndGz=includeMtAndGz)

    cont = 0
    for i in evaluations:
        k = evaluationsInFile[cont]
        cont += 1

        print(
            f"\t- Modifying {IOtools.clipstr(FolderEvaluation)} with position {k} in CGYRO results file {IOtools.clipstr(file)}"
        )

        # Get TGYRO
        tgyro = TGYROtools.TGYROoutput(
            FolderEvaluation,
            profiles=PROFILEStools.PROFILES_GACODE(unmodified_profiles),
        )

        # Quick checker of correct file
        wasThisTheCorrectRun(aLTe, aLTi, aLne, Q_gb, tgyro)

        modifyResults(
            Qe[k, :],
            Qi[k, :],
            Ge[k, :],
            GZ[k, :],
            Mt[k, :],
            Pexch[k, :],
            QeE[k, :],
            QiE[k, :],
            GeE[k, :],
            GZE[k, :],
            MtE[k, :],
            PexchE[k, :],
            tgyro,
            FolderEvaluation,
            minErrorPercent=minErrorPercent,
            useConvectiveFluxes=useConvectiveFluxes,
            Qi_criterion_stable=Qi_criterion_stable,
            percentNeo=percentNeo,
            impurityPosition=impurityPosition,
            OriginalFimp=OriginalFimp,
        )


def wasThisTheCorrectRun(aLTe, aLTi, aLne, Q_gb, tgyro, ErrorRaised=0.005):
    print("\t- Checking that this was the correct run...")

    tgyro_new = copy.deepcopy(tgyro)
    tgyro_new.aLti = tgyro_new.aLti[:, 0, :]

    variables = [
        [aLTe, tgyro_new.aLte, "aLTe"],
        [aLTi, tgyro_new.aLti, "aLTi"],
        [aLne, tgyro_new.aLne, "aLne"],
        [Q_gb, tgyro_new.Q_GB, "Qgb"],
    ]

    for var in variables:
        [c, t, n] = var

        for pos in range(c.shape[0]):
            for i in range(c.shape[1]):
                error = np.max(abs((t[pos, i + 1] - c[pos, i]) / t[pos, i + 1]))
                print(
                    f"\t\t* Error in {n}[{i}] was {error*100.0:.2f}% (TGYRO {t[pos,i+1]:.3f} vs. CGYRO {c[pos,i]:.3f})",
                    typeMsg="w" if error > ErrorRaised else "",
                )


def readlineNTH(line, full_file=False, unnormalize=True):
    s = line.split()

    i = 2
    roa = float(s[i])
    i += 3
    aLne = float(s[i])
    i += 3
    aLTi = float(s[i])
    i += 3
    aLTe = float(s[i])
    i += 3

    Qi = float(s[i])
    i += 3
    Qi_std = float(s[i])
    i += 3
    Qe = float(s[i])
    i += 3
    Qe_std = float(s[i])
    i += 3
    Ge = float(s[i])
    i += 3
    Ge_std = float(s[i])
    i += 3

    if full_file:
        GZ = float(s[i])
        i += 3
        GZ_std = float(s[i])
        i += 3

        Mt = float(s[i])
        i += 3
        Mt_std = float(s[i])
        i += 3

        Pexch = float(s[i])
        i += 3
        Pexch_std = float(s[i])
        i += 3

    Q_gb = float(s[i])
    i += 3
    G_gb = float(s[i]) * 1e-1
    i += 3  # From 1E19 to 1E20

    if full_file:
        Mt_gb = float(s[i])
        i += 3
        Pexch_gb = float(s[i])
        i += 3

    tstart = float(s[i])
    i += 3
    tend = float(s[i])
    i += 3

    if unnormalize:
        QiReal = Qi * Q_gb
        QiReal_std = Qi_std * Q_gb
        QeReal = Qe * Q_gb
        QeReal_std = Qe_std * Q_gb
        GeReal = Ge * G_gb
        GeReal_std = Ge_std * G_gb
    else:
        QiReal = Qi
        QiReal_std = Qi_std
        QeReal = Qe
        QeReal_std = Qe_std
        GeReal = Ge
        GeReal_std = Ge_std

    if full_file:
        if unnormalize:
            GZReal = GZ * G_gb
            GZReal_std = GZ_std * G_gb

            MtReal = Mt * Mt_gb
            MtReal_std = Mt_std * Mt_gb

            PexchReal = Pexch * Pexch_gb
            PexchReal_std = Pexch_std * Pexch_gb
        else:
            GZReal = GZ
            GZReal_std = GZ_std

            MtReal = Mt
            MtReal_std = Mt_std

            PexchReal = Pexch
            PexchReal_std = Pexch_std

        return (
            aLTe,
            aLTi,
            aLne,
            Q_gb,
            QeReal,
            QiReal,
            GeReal,
            GZReal,
            MtReal,
            PexchReal,
            QeReal_std,
            QiReal_std,
            GeReal_std,
            GZReal_std,
            MtReal_std,
            PexchReal_std,
            tstart,
            tend,
        )
    else:
        return (
            aLTe,
            aLTi,
            aLne,
            Q_gb,
            QeReal,
            QiReal,
            GeReal,
            0.0,
            0.0,
            0.0,
            QeReal_std,
            QiReal_std,
            GeReal_std,
            0.0,
            0.0,
            0.0,
            tstart,
            tend,
        )


def readCGYROresults(file, rad=4, includeMtAndGz=False, unnormalize=True):
    """
    Arrays are in (batch,radii)
    MW/m^2 and 1E20
    """

    with open(file, "r") as f:
        lines = f.readlines()

    num = int(len(lines) / rad)

    aLTe = np.zeros((num, rad))
    aLTi = np.zeros((num, rad))
    aLne = np.zeros((num, rad))
    Q_gb = np.zeros((num, rad))

    Qe = np.zeros((num, rad))
    Qe_std = np.zeros((num, rad))
    Qi = np.zeros((num, rad))
    Qi_std = np.zeros((num, rad))
    Ge = np.zeros((num, rad))
    Ge_std = np.zeros((num, rad))

    GZ = np.zeros((num, rad))
    GZ_std = np.zeros((num, rad))

    Mt = np.zeros((num, rad))
    Mt_std = np.zeros((num, rad))

    Pexch = np.zeros((num, rad))
    Pexch_std = np.zeros((num, rad))

    tstart = np.zeros((num, rad))
    tend = np.zeros((num, rad))

    p, r = 0, 0
    for i in range(len(lines)):
        (
            aLTe[p, r],
            aLTi[p, r],
            aLne[p, r],
            Q_gb[p, r],
            Qe[p, r],
            Qi[p, r],
            Ge[p, r],
            GZ[p, r],
            Mt[p, r],
            Pexch[p, r],
            Qe_std[p, r],
            Qi_std[p, r],
            Ge_std[p, r],
            GZ_std[p, r],
            Mt_std[p, r],
            Pexch_std[p, r],
            tstart[p, r],
            tend[p, r],
        ) = readlineNTH(lines[i], full_file=includeMtAndGz, unnormalize=unnormalize)
        p += 1
        if p >= num:
            r += 1
            p = 0

    return (
        aLTe,
        aLTi,
        aLne,
        Q_gb,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        Qe_std,
        Qi_std,
        Ge_std,
        GZ_std,
        Mt_std,
        Pexch_std,
        tstart,
        tend,
    )


def defineReferenceFluxes(
    tgyro, factor_tauptauE=5, useConvectiveFluxes=False, impurityPosition=3
):
    Qe_target = abs(tgyro.Qe_tar[0, 1:])
    Qi_target = abs(tgyro.Qi_tar[0, 1:])
    Mt_target = abs(tgyro.Mt_tar[0, 1:])

    # For particle fluxes, since the targets are often zero... it's more complicated
    QeMW_target = abs(tgyro.Qe_tarMW[0, 1:])
    QiMW_target = abs(tgyro.Qi_tarMW[0, 1:])
    We, Wi, Ne, NZ = tgyro.profiles.deriveContentByVolumes(
        rhos=tgyro.rho[0, 1:], impurityPosition=impurityPosition
    )

    tau_special = (
        (We + Wi) / (QeMW_target + QiMW_target) * factor_tauptauE
    )  # tau_p in seconds
    Ge_target_special = (Ne / tau_special) / tgyro.dvoldr[0, 1:]  # (1E20/seconds/m^2)

    if useConvectiveFluxes:
        Ge_target_special = PLASMAtools.convective_flux(
            tgyro.Te[0, 1:], Ge_target_special
        )  # (1E20/seconds/m^2)

    GZ_target_special = Ge_target_special * NZ / Ne

    return Qe_target, Qi_target, Ge_target_special, GZ_target_special, Mt_target


def modifyResults(
    Qe,
    Qi,
    Ge,
    GZ,
    Mt,
    Pexch,
    QeE,
    QiE,
    GeE,
    GZE,
    MtE,
    PexchE,
    tgyro,
    folder_tgyro,
    minErrorPercent=5.0,
    percentNeo=2.0,
    useConvectiveFluxes=False,
    Qi_criterion_stable=0.0025,
    impurityPosition=3,
    OriginalFimp=1.0,
):
    """
    All in real units, with dimensions of (rho) from axis to edge
    """

    # If a plasma is very close to stable... do something about error
    if minErrorPercent is not None:
        (
            Qe_target,
            Qi_target,
            Ge_target_special,
            GZ_target_special,
            Mt_target,
        ) = defineReferenceFluxes(
            tgyro,
            useConvectiveFluxes=useConvectiveFluxes,
            impurityPosition=impurityPosition,
        )

        Qe_min = Qe_target * (minErrorPercent / 100.0)
        Qi_min = Qi_target * (minErrorPercent / 100.0)
        Ge_min = Ge_target_special * (minErrorPercent / 100.0)
        GZ_min = GZ_target_special * (minErrorPercent / 100.0)
        Mt_min = Mt_target * (minErrorPercent / 100.0)

        for i in range(Qe.shape[0]):
            if Qi[i] < Qi_criterion_stable:
                print(
                    f"\t- Based on 'Qi_criterion_stable', plasma considered stable (Qi = {Qi[i]:.2e} < {Qi_criterion_stable:.2e} MW/m2) at position #{i}, using minimum errors of {minErrorPercent}% of targets",
                    typeMsg="w",
                )
                QeE[i] = Qe_min[i]
                print(f"\t\t* QeE = {QeE[i]}")
                QiE[i] = Qi_min[i]
                print(f"\t\t* QiE = {QiE[i]}")
                GeE[i] = Ge_min[i]
                print(f"\t\t* GeE = {GeE[i]}")
                GZE[i] = GZ_min[i]
                print(f"\t\t* GZE = {GZE[i]}")
                MtE[i] = Mt_min[i]
                print(f"\t\t* MtE = {MtE[i]}")

    # Heat fluxes
    QeTot = Qe + tgyro.Qe_sim_neo[0, 1:]
    QiTot = Qi + tgyro.QiIons_sim_neo_thr[0, 1:]

    # Particle fluxes
    PeTot = Ge + tgyro.Ge_sim_neo[0, 1:]
    PZTot = GZ + tgyro.Gi_sim_neo[impurityPosition - 1, 0, 1:]

    # Momentum fluxes
    MtTot = Mt + tgyro.Mt_sim_neo[0, 1:]

    # ************************************************************************************
    # **** Modify complete folder (Division of ion fluxes will be wrong, since I put everything in first ion)
    # ************************************************************************************

    # 1. Modify out.tgyro.evo files (which contain turb+neo summed together)

    print(f"\t- Modifying TGYRO out.tgyro.evo files in {IOtools.clipstr(folder_tgyro)}")
    modifyEVO(
        tgyro,
        folder_tgyro,
        QeTot,
        QiTot,
        PeTot,
        PZTot,
        MtTot,
        impurityPosition=impurityPosition,
    )

    # 2. Modify out.tgyro.flux files (which contain turb and neo separated)

    print(f"\t- Modifying TGYRO out.tgyro.flux files in {folder_tgyro}")
    modifyFLUX(
        tgyro,
        folder_tgyro,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        impurityPosition=impurityPosition,
    )

    # 3. Modify files for errors

    print(f"\t- Modifying TGYRO out.tgyro.flux_stds in {folder_tgyro}")
    modifyFLUX(
        tgyro,
        folder_tgyro,
        QeE,
        QiE,
        GeE,
        GZE,
        MtE,
        PexchE,
        impurityPosition=impurityPosition,
        special_label="_stds",
    )


def modifyEVO(
    tgyro,
    folder,
    QeT,
    QiT,
    GeT,
    GZT,
    MtT,
    impurityPosition=3,
    positionMod=1,
    special_label=None,
):
    QeTGB = QeT / tgyro.Q_GB[-1, 1:]
    QiTGB = QiT / tgyro.Q_GB[-1, 1:]
    GeTGB = GeT / tgyro.Gamma_GB[-1, 1:]
    GZTGB = GZT / tgyro.Gamma_GB[-1, 1:]
    MtTGB = MtT / tgyro.Pi_GB[-1, 1:]

    modTGYROfile(
        f"{folder}/out.tgyro.evo_te", QeTGB, pos=positionMod, fileN_suffix=special_label
    )
    modTGYROfile(
        f"{folder}/out.tgyro.evo_ti", QiTGB, pos=positionMod, fileN_suffix=special_label
    )
    modTGYROfile(
        f"{folder}/out.tgyro.evo_ne", GeTGB, pos=positionMod, fileN_suffix=special_label
    )
    modTGYROfile(
        f"{folder}/out.tgyro.evo_er", MtTGB, pos=positionMod, fileN_suffix=special_label
    )

    for i in range(tgyro.Qi_sim_turb.shape[0]):
        if i + 1 == impurityPosition:
            var = GZTGB
        else:
            var = GZTGB * 0.0
        modTGYROfile(
            f"{folder}/out.tgyro.evo_n{i+1}",
            var,
            pos=positionMod,
            fileN_suffix=special_label,
        )


def modifyFLUX(
    tgyro,
    folder,
    Qe,
    Qi,
    Ge,
    GZ,
    Mt,
    S,
    QeNeo=None,
    QiNeo=None,
    GeNeo=None,
    GZNeo=None,
    MtNeo=None,
    impurityPosition=3,
    special_label=None,
):
    QeGB = Qe / tgyro.Q_GB[-1, 1:]
    QiGB = Qi / tgyro.Q_GB[-1, 1:]
    GeGB = Ge / tgyro.Gamma_GB[-1, 1:]
    GZGB = GZ / tgyro.Gamma_GB[-1, 1:]
    MtGB = Mt / tgyro.Pi_GB[-1, 1:]
    SGB = S / tgyro.S_GB[-1, 1:]

    # ******************************************************************************************
    # Electrons
    # ******************************************************************************************

    # Particle flux: Update

    modTGYROfile(f"{folder}/out.tgyro.flux_e", GeGB, pos=2, fileN_suffix=special_label)
    if GeNeo is not None:
        GeGB_neo = GeNeo / tgyro.Gamma_GB[-1, 1:]
        modTGYROfile(
            f"{folder}/out.tgyro.flux_e", GeGB_neo, pos=1, fileN_suffix=special_label
        )

    # Energy flux: Update

    modTGYROfile(f"{folder}/out.tgyro.flux_e", QeGB, pos=4, fileN_suffix=special_label)
    if QeNeo is not None:
        QeGB_neo = QeNeo / tgyro.Q_GB[-1, 1:]
        modTGYROfile(
            f"{folder}/out.tgyro.flux_e", QeGB_neo, pos=3, fileN_suffix=special_label
        )

    # Rotation: Remove (it will be sum to the first ion)

    modTGYROfile(
        f"{folder}/out.tgyro.flux_e", GeGB * 0.0, pos=6, fileN_suffix=special_label
    )
    modTGYROfile(
        f"{folder}/out.tgyro.flux_e", GeGB * 0.0, pos=5, fileN_suffix=special_label
    )

    # Energy exchange

    modTGYROfile(f"{folder}/out.tgyro.flux_e", SGB, pos=7, fileN_suffix=special_label)

    # SMW  = S  # S is MW/m^3
    # modTGYROfile(f'{folder}/out.tgyro.power_e',SMW,pos=8,fileN_suffix=special_label)
    # print('\t\t- Modified turbulent energy exchange in out.tgyro.power_e')

    # ******************************************************************************************
    # Ions
    # ******************************************************************************************

    # Energy flux: Update

    modTGYROfile(f"{folder}/out.tgyro.flux_i1", QiGB, pos=4, fileN_suffix=special_label)

    if QiNeo is not None:
        QiGB_neo = QiNeo / tgyro.Q_GB[-1, 1:]
        modTGYROfile(
            f"{folder}/out.tgyro.flux_i1", QiGB_neo, pos=3, fileN_suffix=special_label
        )

    # Particle flux: Make ion particle fluxes zero, because I don't want to mistake TGLF with CGYRO when looking at tgyro results

    for i in range(tgyro.Qi_sim_turb.shape[0]):
        if tgyro.profiles.Species[i]["S"] == "therm":
            var = QiGB * 0.0
            modTGYROfile(
                f"{folder}/out.tgyro.flux_i{i+1}",
                var,
                pos=2,
                fileN_suffix=special_label,
            )  # Gi_turb
            modTGYROfile(
                f"{folder}/out.tgyro.evo_n{i+1}", var, pos=1, fileN_suffix=special_label
            )  # Gi (Gi_sim)

            if i + 1 != impurityPosition:
                modTGYROfile(
                    f"{folder}/out.tgyro.flux_i{i+1}",
                    var,
                    pos=1,
                    fileN_suffix=special_label,
                )  # Gi_neo

    # Rotation: Update

    modTGYROfile(f"{folder}/out.tgyro.flux_i1", MtGB, pos=6, fileN_suffix=special_label)

    if MtNeo is not None:
        MtGB_neo = MtNeo / tgyro.Pi_GB[-1, 1:]
        modTGYROfile(
            f"{folder}/out.tgyro.flux_i1", MtGB_neo, pos=5, fileN_suffix=special_label
        )

    # Energy exchange: Remove (it will be the electrons one)

    modTGYROfile(
        f"{folder}/out.tgyro.flux_i1", SGB * 0.0, pos=7, fileN_suffix=special_label
    )

    # ******************************************************************************************
    # Impurities
    # ******************************************************************************************

    # Remove everything from all the rest of non-first ions (except the particles for the impurity chosen)

    for i in range(tgyro.Qi_sim_turb.shape[0] - 1):
        if tgyro.profiles.Species[i + 1]["S"] == "therm":
            var = QiGB * 0.0
            for pos in [3, 4, 5, 6, 7]:
                modTGYROfile(
                    f"{folder}/out.tgyro.flux_i{i+2}",
                    var,
                    pos=pos,
                    fileN_suffix=special_label,
                )
            for pos in [1, 2]:
                if i + 2 != impurityPosition:
                    modTGYROfile(
                        f"{folder}/out.tgyro.flux_i{i+2}",
                        var,
                        pos=pos,
                        fileN_suffix=special_label,
                    )

    modTGYROfile(
        f"{folder}/out.tgyro.flux_i{impurityPosition}",
        GZGB,
        pos=2,
        fileN_suffix=special_label,
    )
    if GZNeo is not None:
        GZGB_neo = GZNeo / tgyro.Gamma_GB[-1, 1:]
        modTGYROfile(
            f"{folder}/out.tgyro.flux_i{impurityPosition}",
            GZGB_neo,
            pos=1,
            fileN_suffix=special_label,
        )


def modTGYROfile(file, var, pos=0, fileN_suffix=None):
    fileN = file if fileN_suffix is None else f"{file}{fileN_suffix}"

    if not os.path.exists(fileN):
        os.system(f"cp {file} {fileN}")

    with open(fileN, "r") as f:
        lines = f.readlines()

    with open(fileN, "w") as f:
        f.write(lines[0])
        f.write(lines[1])
        f.write(lines[2])
        for i in range(var.shape[0]):
            new_s = [float(k) for k in lines[3 + i].split()]
            new_s[pos] = var[i]

            line_new = " "
            for k in range(len(new_s)):
                line_new += f'{"" if k==0 else "   "}{new_s[k]:.6e}'
            f.write(line_new + "\n")
