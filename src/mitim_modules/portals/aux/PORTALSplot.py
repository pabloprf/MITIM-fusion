import copy, torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools, TGYROtools
from mitim_tools.gacode_tools.aux import PORTALSinteraction
from mitim_tools.misc_tools.IOtools import printMsg as print

factor_dw0dr = 1e-5
label_dw0dr = "$-d\\omega_0/dr$ (krad/s/cm)"


class PORTALSresults:
    def __init__(
        self,
        folderWork,
        prfs_model,
        ResultsOptimization,
        MITIMextra_dict=None,
        indecesPlot=[-1, 0, None],
        calculateRicci={"d0": 2.0, "l": 1.0},
    ):
        self.prfs_model = prfs_model
        self.ResultsOptimization = ResultsOptimization

        includeFast = self.prfs_model.mainFunction.PORTALSparameters["includeFastInQi"]
        impurityPosition = self.prfs_model.mainFunction.PORTALSparameters[
            "ImpurityOfInterest"
        ]
        self.useConvectiveFluxes = self.prfs_model.mainFunction.PORTALSparameters[
            "useConvectiveFluxes"
        ]

        self.numChannels = len(
            self.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]
        )
        self.numRadius = len(
            self.prfs_model.mainFunction.TGYROparameters["RhoLocations"]
        )
        self.numBest = (
            indecesPlot[0]
            if (indecesPlot[0] > 0)
            else (self.prfs_model.train_Y.shape[0] + indecesPlot[0])
        )
        self.numOrig = indecesPlot[1]
        self.numExtra = (
            indecesPlot[2]
            if (indecesPlot[2] is None or indecesPlot[2] > 0)
            else (self.prfs_model.train_Y.shape[0] + indecesPlot[2])
        )
        self.sepers = [self.numOrig, self.numBest]

        if (self.numExtra is not None) and (self.numExtra == self.numBest):
            self.numExtra = None

        self.posZ = prfs_model.mainFunction.PORTALSparameters["ImpurityOfInterest"] - 1

        # Profiles and tgyro results
        print("\t- Reading profiles and tgyros for each evaluation")
        self.profiles, self.tgyros = [], []
        for i in range(self.prfs_model.train_Y.shape[0]):
            # print(f'\t- Reading TGYRO results and PROFILES for evaluation {i}/{self.prfs_model.train_Y.shape[0]-1}')
            if MITIMextra_dict is not None:
                # print('\t\t* Reading from MITIMextra_dict',typeMsg='i')
                self.tgyros.append(MITIMextra_dict[i]["tgyro"].results["use"])
                self.profiles.append(
                    MITIMextra_dict[i]["tgyro"].results["use"].profiles_final
                )
            else:
                # print('\t\t* Reading from scratch from folders (will surely take longer)',typeMsg='i')
                folderEvaluation = folderWork + f"/Execution/Evaluation.{i}/"
                self.profiles.append(
                    PROFILEStools.PROFILES_GACODE(
                        folderEvaluation + "/model_complete/input.gacode.new"
                    )
                )
                self.tgyros.append(
                    TGYROtools.TGYROoutput(
                        folderEvaluation + "model_complete/", profiles=self.profiles[i]
                    )
                )

        if len(self.profiles) <= self.numBest:
            print(
                "\t- PORTALS was read after new residual was computed but before pickle was written!",
                typeMsg="w",
            )
            self.numBest -= 1
            self.numExtra = None

        # Create some metrics

        print("\t- Process results")

        self.evaluations, self.resM = [], []
        self.FusionGain, self.tauE, self.FusionPower = [], [], []
        self.resTe, self.resTi, self.resne, self.resnZ, self.resw0 = [], [], [], [], []
        if calculateRicci is not None:
            self.QR_Ricci, self.chiR_Ricci, self.points_Ricci = [], [], []
        else:
            self.QR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None
        for i, (p, t) in enumerate(zip(self.profiles, self.tgyros)):
            self.evaluations.append(i)
            self.FusionGain.append(p.derived["Q"])
            self.FusionPower.append(p.derived["Pfus"])
            self.tauE.append(p.derived["tauE"])

            # ------------------------------------------------
            # Residual definitions
            # ------------------------------------------------

            powerstate = self.prfs_model.mainFunction.powerstate

            try:
                OriginalFimp = powerstate.TransportOptions["ModelOptions"][
                    "OriginalFimp"
                ]
            except:
                OriginalFimp = 1.0

            portals_variables = t.TGYROmodeledVariables(
                useConvectiveFluxes=self.useConvectiveFluxes,
                includeFast=includeFast,
                impurityPosition=impurityPosition,
                ProfilesPredicted=self.prfs_model.mainFunction.TGYROparameters[
                    "ProfilesPredicted"
                ],
                UseFineGridTargets=self.prfs_model.mainFunction.PORTALSparameters[
                    "fineTargetsResolution"
                ],
                OriginalFimp=OriginalFimp,
                forceZeroParticleFlux=self.prfs_model.mainFunction.PORTALSparameters[
                    "forceZeroParticleFlux"
                ],
            )

            if (
                len(powerstate.plasma["volp"].shape) > 1
                and powerstate.plasma["volp"].shape[1] > 1
            ):
                powerstate.unrepeat(do_fine=False)
                powerstate.repeat(do_fine=False)

            _, _, source, res = PORTALSinteraction.calculatePseudos(
                portals_variables["var_dict"],
                self.prfs_model.mainFunction.PORTALSparameters,
                self.prfs_model.mainFunction.TGYROparameters,
                powerstate,
            )

            # Make sense of tensor "source" which are defining the entire predictive set in
            Qe_resR = np.zeros(
                len(self.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            Qi_resR = np.zeros(
                len(self.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            Ge_resR = np.zeros(
                len(self.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            GZ_resR = np.zeros(
                len(self.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            Mt_resR = np.zeros(
                len(self.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
            )
            cont = 0
            for prof in self.prfs_model.mainFunction.TGYROparameters[
                "ProfilesPredicted"
            ]:
                for ix in range(
                    len(self.prfs_model.mainFunction.TGYROparameters["RhoLocations"])
                ):
                    if prof == "te":
                        Qe_resR[ix] = source[0, cont].abs()
                    if prof == "ti":
                        Qi_resR[ix] = source[0, cont].abs()
                    if prof == "ne":
                        Ge_resR[ix] = source[0, cont].abs()
                    if prof == "nZ":
                        GZ_resR[ix] = source[0, cont].abs()
                    if prof == "w0":
                        Mt_resR[ix] = source[0, cont].abs()

                    cont += 1

            res = -res.item()

            self.resTe.append(Qe_resR)
            self.resTi.append(Qi_resR)
            self.resne.append(Ge_resR)
            self.resnZ.append(GZ_resR)
            self.resw0.append(Mt_resR)
            self.resM.append(res)

            # Ricci Metrics
            if calculateRicci is not None:
                try:
                    (
                        y1,
                        y2,
                        y1_std,
                        y2_std,
                    ) = PORTALSinteraction.calculatePseudos_distributions(
                        portals_variables["var_dict"],
                        self.prfs_model.mainFunction.PORTALSparameters,
                        self.prfs_model.mainFunction.TGYROparameters,
                        powerstate,
                    )

                    QR, chiR = PLASMAtools.RicciMetric(
                        y1,
                        y2,
                        y1_std,
                        y2_std,
                        d0=calculateRicci["d0"],
                        l=calculateRicci["l"],
                    )
                    self.QR_Ricci.append(QR[0])
                    self.chiR_Ricci.append(chiR[0])
                    self.points_Ricci.append(
                        [
                            y1.cpu().numpy()[0, :],
                            y2.cpu().numpy()[0, :],
                            y1_std.cpu().numpy()[0, :],
                            y2_std.cpu().numpy()[0, :],
                        ]
                    )
                except:
                    print("\t- Could not calculate Ricci metric", typeMsg="w")
                    calculateRicci = None
                    self.QR_Ricci, self.chiR_Ricci, self.points_Ricci = None, None, None

        self.labelsFluxes = portals_variables["labels"]

        self.FusionGain = np.array(self.FusionGain)
        self.FusionPower = np.array(self.FusionPower)
        self.tauE = np.array(self.tauE)
        self.resM = np.array(self.resM)
        self.evaluations = np.array(self.evaluations)
        self.resTe, self.resTi, self.resne, self.resnZ, self.resw0 = (
            np.array(self.resTe),
            np.array(self.resTi),
            np.array(self.resne),
            np.array(self.resnZ),
            np.array(self.resw0),
        )

        if calculateRicci is not None:
            self.chiR_Ricci = np.array(self.chiR_Ricci)
            self.QR_Ricci = np.array(self.QR_Ricci)
            self.points_Ricci = np.array(self.points_Ricci)

        # Normalized L1 norms
        self.resTeM = np.abs(self.resTe).mean(axis=1)
        self.resTiM = np.abs(self.resTi).mean(axis=1)
        self.resneM = np.abs(self.resne).mean(axis=1)
        self.resnZM = np.abs(self.resnZ).mean(axis=1)
        self.resw0M = np.abs(self.resw0).mean(axis=1)

        self.resCheck = (
            self.resTeM + self.resTiM + self.resneM + self.resnZM + self.resw0M
        ) / len(self.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"])

        # ---------------------------------------------------------------------------------------------------------------------
        # Jacobian
        # ---------------------------------------------------------------------------------------------------------------------

        DeltaQ1 = []
        for i in self.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
            if i == "te":
                DeltaQ1.append(-self.resTe)
            if i == "ti":
                DeltaQ1.append(-self.resTi)
            if i == "ne":
                DeltaQ1.append(-self.resne)
        DeltaQ1 = np.array(DeltaQ1)
        self.DeltaQ = DeltaQ1[0, :, :]
        for i in range(DeltaQ1.shape[0] - 1):
            self.DeltaQ = np.append(self.DeltaQ, DeltaQ1[i + 1, :, :], axis=1)

        self.aLTn_perc = aLTi_perc = None
        # try:	self.aLTn_perc  = aLTi_perc  = calcLinearizedModel(self.prfs_model,self.DeltaQ,numChannels=self.numChannels,numRadius=self.numRadius,sepers=self.sepers)
        # except:	print('\t- Jacobian calculation failed',typeMsg='w')

        self.DVdistMetric_x = ResultsOptimization.DVdistMetric_x
        self.DVdistMetric_y = ResultsOptimization.DVdistMetric_y


def calcLinearizedModel(
    prfs_model, DeltaQ, posBase=-1, numChannels=3, numRadius=4, sepers=[]
):
    """
    posBase = 1 is aLTi, 0 is aLTe, if the order is [a/LTe,aLTi]
    -1 is diagonal
    -2 is

    NOTE for PRF: THIS ONLY WORKS FOR TURBULENCE, nOT NEO!
    """

    trainx = prfs_model.steps[-1].GP["combined_model"].train_X.cpu().numpy()

    istep, aLTn_est, aLTn_base = 0, [], []
    for i in range(trainx.shape[0]):
        if i >= prfs_model.Optim["initialPoints"]:
            istep += 1

        # Jacobian
        J = (
            prfs_model.steps[istep]
            .GP["combined_model"]
            .localBehavior(trainx[i, :], plotYN=False)
        )

        J = 1e-3 * J[: trainx.shape[1], : trainx.shape[1]]  # Only turbulence

        print(f"\t- Reading Jacobian for step {istep}")

        Q = DeltaQ[i, :]

        if posBase < 0:
            # All channels together ------------------------------------------------
            mult = torch.Tensor()
            for i in range(12):
                if posBase == -1:
                    a = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Diagonal
                elif posBase == -2:
                    a = torch.Tensor(
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                    )  # Block diagonal
                a = torch.roll(a, i)
                mult = torch.cat((mult, a.unsqueeze(0)), dim=0)

            J_reduced = J * mult
            aLTn = (J_reduced.inverse().cpu().numpy()).dot(Q)
            aLTn_base0 = trainx[i, :]
            # ------------------------------------------------------------------------

        else:
            # Channel per channel, only ion temperature gradient ------------------------
            J_mod = []
            aLTn_base0 = []
            cont = 0
            for c in range(numChannels):
                for r in range(numRadius):
                    J_mod.append(J[cont, posBase * numRadius + r].cpu().numpy())
                    aLTn_base0.append(trainx[i, posBase * numRadius + r])
                    cont += 1
            J_mod = np.array(J_mod)
            aLTn_base0 = np.array(aLTn_base0)
            aLTn = Q / J_mod
            # ------------------------------------------------------------------------

        aLTn_base.append(aLTn_base0)
        aLTn_est.append(aLTn)

    aLTn_est = np.array(aLTn_est)
    aLTn_base = np.array(aLTn_base)

    aLTn_perc = [
        np.abs(i / j) * 100.0 if i is not None else None
        for i, j in zip(aLTn_est, aLTn_base)
    ]

    return aLTn_perc


def plotConvergencePORTALS(
    portals,
    folderWork=None,
    fig=None,
    stds=2,
    plotAllFluxes=False,
    indexToMaximize=None,
    plotFlows=True,
    fontsize_leg=5,
    includeRicci=True,
):
    """
    Either folderwork or portals need to be provided
    """

    labelsFluxes = portals.labelsFluxes  # For channel residuals

    useConvectiveFluxes = False  # portals.useConvectiveFluxes
    labelsFluxesF = {
        "te": "$Q_e$ ($MW/m^2$)",
        "ti": "$Q_i$ ($MW/m^2$)",
        "ne": "$\\Gamma_e$ ($10^{20}/s/m^2$)",
        "nZ": "$\\Gamma_Z$ ($10^{20}/s/m^2$)",
        "w0": "$M_T$ ($J/m^2$)",
    }

    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(15, 8))

    numprofs = len(portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"])

    wspace = 0.3 if numprofs <= 4 else 0.5

    grid = plt.GridSpec(nrows=8, ncols=numprofs + 1, hspace=0.3, wspace=0.35)

    # Te
    axTe = fig.add_subplot(grid[:4, 0])
    axTe.set_title("Electron Temperature")
    axTe_g = fig.add_subplot(grid[4:6, 0])
    axTe_f = fig.add_subplot(grid[6:, 0])

    axTi = fig.add_subplot(grid[:4, 1])
    axTi.set_title("Ion Temperature")
    axTi_g = fig.add_subplot(grid[4:6, 1])
    axTi_f = fig.add_subplot(grid[6:, 1])

    cont = 0
    if "ne" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        axne = fig.add_subplot(grid[:4, 2 + cont])
        axne.set_title("Electron Density")
        axne_g = fig.add_subplot(grid[4:6, 2 + cont])
        axne_f = fig.add_subplot(grid[6:, 2 + cont])
        cont += 1
    else:
        axne = axne_g = axne_f = None

    if "nZ" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        impos = portals.prfs_model.mainFunction.PORTALSparameters["ImpurityOfInterest"]
        labIon = f"Ion {impos} ({portals.profiles[0].Species[impos-1]['N']}{int(portals.profiles[0].Species[impos-1]['Z'])},{int(portals.profiles[0].Species[impos-1]['A'])})"
        axnZ = fig.add_subplot(grid[:4, 2 + cont])
        axnZ.set_title(f"{labIon} Density")
        axnZ_g = fig.add_subplot(grid[4:6, 2 + cont])
        axnZ_f = fig.add_subplot(grid[6:, 2 + cont])
        cont += 1
    else:
        axnZ = axnZ_g = axnZ_f = None

    if "w0" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        axw0 = fig.add_subplot(grid[:4, 2 + cont])
        axw0.set_title("Rotation")
        axw0_g = fig.add_subplot(grid[4:6, 2 + cont])
        axw0_f = fig.add_subplot(grid[6:, 2 + cont])
    else:
        axw0 = axw0_g = axw0_f = None

    axQ = fig.add_subplot(grid[:2, numprofs])
    axA = fig.add_subplot(grid[2:4, numprofs])
    axC = fig.add_subplot(grid[4:6, numprofs])
    axR = fig.add_subplot(grid[6:8, numprofs])

    if indexToMaximize is None:
        indexToMaximize = portals.numBest
    if indexToMaximize < 0:
        indexToMaximize = portals.prfs_model.train_Y.shape[0] + indexToMaximize

    # ---------------------------------------------------------------------------------------------------------
    # Plot all profiles
    # ---------------------------------------------------------------------------------------------------------

    lwt = 0.1
    lw = 0.2
    alph = 0.6
    for i, p in enumerate(portals.profiles):
        if p is not None:
            if i < 5:
                col = "k"
            else:
                col = "b"

            if i == 0:
                lab = "Training"
            elif i == 5:
                lab = "Optimization"
            else:
                lab = ""

            ix = np.argmin(
                np.abs(
                    p.profiles["rho(-)"] - portals.tgyros[portals.numOrig].rho[0][-1]
                )
            )
            axTe.plot(
                p.profiles["rho(-)"],
                p.profiles["te(keV)"],
                lw=lw,
                color=col,
                label=lab,
                alpha=alph,
            )
            axTe_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLTe"][:ix],
                lw=lw,
                color=col,
                alpha=alph,
            )
            axTi.plot(
                p.profiles["rho(-)"],
                p.profiles["ti(keV)"][:, 0],
                lw=lw,
                color=col,
                label=lab,
                alpha=alph,
            )
            axTi_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLTi"][:ix, 0],
                lw=lw,
                color=col,
                alpha=alph,
            )
            if axne is not None:
                axne.plot(
                    p.profiles["rho(-)"],
                    p.profiles["ne(10^19/m^3)"] * 1e-1,
                    lw=lw,
                    color=col,
                    label=lab,
                    alpha=alph,
                )
                axne_g.plot(
                    p.profiles["rho(-)"][:ix],
                    p.derived["aLne"][:ix],
                    lw=lw,
                    color=col,
                    alpha=alph,
                )

            if axnZ is not None:
                axnZ.plot(
                    p.profiles["rho(-)"],
                    p.profiles["ni(10^19/m^3)"][:, portals.posZ] * 1e-1,
                    lw=lw,
                    color=col,
                    label=lab,
                    alpha=alph,
                )
                axnZ_g.plot(
                    p.profiles["rho(-)"][:ix],
                    p.derived["aLni"][:ix, portals.posZ],
                    lw=lw,
                    color=col,
                    alpha=alph,
                )

            if axw0 is not None:
                axw0.plot(
                    p.profiles["rho(-)"],
                    p.profiles["w0(rad/s)"] * 1e-3,
                    lw=lw,
                    color=col,
                    label=lab,
                    alpha=alph,
                )
                axw0_g.plot(
                    p.profiles["rho(-)"][:ix],
                    p.derived["dw0dr"][:ix] * factor_dw0dr,
                    lw=lw,
                    color=col,
                    alpha=alph,
                )

        t = portals.tgyros[i]
        if (t is not None) and plotAllFluxes:
            axTe_f.plot(
                t.rho[0],
                t.Qe_sim_turb[0] + t.Qe_sim_neo[0],
                "-",
                c=col,
                lw=lwt,
                alpha=alph,
            )
            axTe_f.plot(t.rho[0], t.Qe_tar[0], "--", c=col, lw=lwt, alpha=alph)
            axTi_f.plot(
                t.rho[0],
                t.QiIons_sim_turb_thr[0] + t.QiIons_sim_neo_thr[0],
                "-",
                c=col,
                lw=lwt,
                alpha=alph,
            )
            axTi_f.plot(t.rho[0], t.Qi_tar[0], "--", c=col, lw=lwt, alpha=alph)

            if useConvectiveFluxes:
                Ge, Ge_tar = t.Ce_sim_turb + t.Ce_sim_neo, t.Ce_tar
            else:
                Ge, Ge_tar = (t.Ge_sim_turb + t.Ge_sim_neo), t.Ge_tar

            if axne_f is not None:
                axne_f.plot(t.rho[0], Ge[0], "-", c=col, lw=lwt, alpha=alph)
                axne_f.plot(t.rho[0], Ge_tar[0], "--", c=col, lw=lwt, alpha=alph)

            if axnZ_f is not None:
                if useConvectiveFluxes:
                    GZ, GZ_tar = (
                        t.Ci_sim_turb[portals.posZ, :, :]
                        + t.Ci_sim_turb[portals.posZ, :, :],
                        t.Ci_tar[portals.posZ, :, :],
                    )
                else:
                    GZ, GZ_tar = (
                        t.Gi_sim_turb[portals.posZ, :, :]
                        + t.Gi_sim_neo[portals.posZ, :, :]
                    ), t.Gi_tar[portals.posZ, :, :]

                axnZ_f.plot(t.rho[0], GZ[0], "-", c=col, lw=lwt, alpha=alph)
                axnZ_f.plot(t.rho[0], GZ_tar[0], "--", c=col, lw=lwt, alpha=alph)

            if axw0_f is not None:
                axw0_f.plot(
                    t.rho[0],
                    t.Mt_sim_turb[0] + t.Mt_sim_neo[0],
                    "-",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )
                axw0_f.plot(t.rho[0], t.Mt_tar[0], "--", c=col, lw=lwt, alpha=alph)

    # ---------------------------------------------------------------------------------------------------------

    msFlux = 3

    for cont, (indexUse, col, lab) in enumerate(
        zip(
            [portals.numOrig, portals.numBest, portals.numExtra],
            ["r", "g", "m"],
            [
                f"Initial (#{portals.numOrig})",
                f"Best (#{portals.numBest})",
                f"Last (#{portals.numExtra})",
            ],
        )
    ):
        if (indexUse is None) or (indexUse >= len(portals.profiles)):
            continue

        p = portals.profiles[indexUse]
        t = portals.tgyros[indexUse]

        ix = np.argmin(np.abs(p.profiles["rho(-)"] - t.rho[0][-1]))
        axTe.plot(
            p.profiles["rho(-)"], p.profiles["te(keV)"], lw=2, color=col, label=lab
        )
        axTe_g.plot(
            p.profiles["rho(-)"][:ix],
            p.derived["aLTe"][:ix],
            "-",
            markersize=msFlux,
            lw=2,
            color=col,
        )
        axTi.plot(
            p.profiles["rho(-)"],
            p.profiles["ti(keV)"][:, 0],
            lw=2,
            color=col,
            label=lab,
        )
        axTi_g.plot(
            p.profiles["rho(-)"][:ix],
            p.derived["aLTi"][:ix, 0],
            "-",
            markersize=msFlux,
            lw=2,
            color=col,
        )
        if axne is not None:
            axne.plot(
                p.profiles["rho(-)"],
                p.profiles["ne(10^19/m^3)"] * 1e-1,
                lw=2,
                color=col,
                label=lab,
            )
            axne_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLne"][:ix],
                "-",
                markersize=msFlux,
                lw=2,
                color=col,
            )

        if axnZ is not None:
            axnZ.plot(
                p.profiles["rho(-)"],
                p.profiles["ni(10^19/m^3)"][:, portals.posZ] * 1e-1,
                lw=2,
                color=col,
                label=lab,
            )
            axnZ_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLni"][:ix, portals.posZ],
                markersize=msFlux,
                lw=2,
                color=col,
            )

        if axw0 is not None:
            axw0.plot(
                p.profiles["rho(-)"],
                p.profiles["w0(rad/s)"] * 1e-3,
                lw=2,
                color=col,
                label=lab,
            )
            axw0_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["dw0dr"][:ix] * factor_dw0dr,
                "-",
                markersize=msFlux,
                lw=2,
                color=col,
            )

        (
            QeBest_min,
            QeBest_max,
            QiBest_min,
            QiBest_max,
            GeBest_min,
            GeBest_max,
            GZBest_min,
            GZBest_max,
            MtBest_min,
            MtBest_max,
        ) = plotFluxComparison(
            p,
            t,
            axTe_f,
            axTi_f,
            axne_f,
            axnZ_f,
            axw0_f,
            posZ=portals.posZ,
            fontsize_leg=fontsize_leg,
            stds=stds,
            col=col,
            lab=lab,
            msFlux=msFlux,
            useConvectiveFluxes=useConvectiveFluxes,
            maxStore=indexToMaximize == indexUse,
            decor=portals.numBest == indexUse,
            plotFlows=plotFlows and (portals.numBest == indexUse),
        )

    ax = axTe
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$T_e$ (keV)")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])
    ax.legend(prop={"size": fontsize_leg * 1.5})

    ax = axTe_g
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$a/L_{Te}$")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])

    ax = axTi
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$T_i$ (keV)")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])

    ax = axTi_g
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$a/L_{Ti}$")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])

    if axne is not None:
        ax = axne
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

        ax = axne_g
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$a/L_{ne}$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

    if axnZ is not None:
        ax = axnZ
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

        GRAPHICStools.addScientificY(ax)

    if axnZ_g is not None:
        ax = axnZ_g
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$a/L_{nZ}$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

    if axw0 is not None:
        ax = axw0
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$w_0$ (krad/s)")
        ax.set_xlim([0, 1])
        ax.set_xticklabels([])

    if axw0_g is not None:
        ax = axw0_g
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel(label_dw0dr)
        ax.set_xlim([0, 1])
        ax.set_xticklabels([])

    ax = axC
    if "te" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        v = portals.resTeM
        ax.plot(
            portals.evaluations,
            v,
            "-o",
            lw=0.5,
            c="b",
            markersize=2,
            label=labelsFluxes["te"],
        )
    if "ti" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        v = portals.resTiM
        ax.plot(
            portals.evaluations,
            v,
            "-s",
            lw=0.5,
            c="m",
            markersize=2,
            label=labelsFluxes["ti"],
        )
    if "ne" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        v = portals.resneM
        ax.plot(
            portals.evaluations,
            v,
            "-*",
            lw=0.5,
            c="k",
            markersize=2,
            label=labelsFluxes["ne"],
        )
    if "nZ" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        v = portals.resnZM
        ax.plot(
            portals.evaluations,
            v,
            "-v",
            lw=0.5,
            c="c",
            markersize=2,
            label=labelsFluxes["nZ"],
        )
    if "w0" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        v = portals.resw0M
        ax.plot(
            portals.evaluations,
            v,
            "-v",
            lw=0.5,
            c="darkred",
            markersize=2,
            label=labelsFluxes["w0"],
        )

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            [portals.numOrig, portals.numBest, portals.numExtra],
            ["r", "g", "m"],
            ["Initial", "Best", "Last"],
            ["o", "s", "*"],
        )
    ):
        if (indexUse is None) or (indexUse >= len(portals.profiles)):
            continue
        if "te" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
            v = portals.resTeM
            ax.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "ti" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
            v = portals.resTiM
            ax.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "ne" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
            v = portals.resneM
            ax.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "nZ" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
            v = portals.resnZM
            ax.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "w0" in portals.prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
            v = portals.resw0M
            ax.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )

    # Plot las point as check
    ax.plot(
        [portals.evaluations[-1]], [portals.resCheck[-1]], "-o", markersize=2, color="k"
    )

    separator = portals.prfs_model.Optim["initialPoints"] + 0.5 - 1

    if portals.evaluations[-1] < separator:
        separator = None

    GRAPHICStools.addDenseAxis(ax, n=5)

    ax.set_ylabel("Channel residual")
    ax.set_xlim(left=-0.2)
    # ax.set_ylim(bottom=0)
    try:
        ax.set_yscale("log")
    except:
        pass
    GRAPHICStools.addLegendApart(
        ax,
        ratio=0.9,
        withleg=True,
        size=fontsize_leg * 1.5,
        title="Channels $\\widehat{L_1}$",
    )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')
    ax.set_xticklabels([])

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=8,
            fromtop=0.1,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    ax = axR

    for resChosen, label, c in zip(
        [portals.resM, portals.resCheck],
        ["$\\widehat{L_2}$", "$\\widehat{L_1}$"],
        ["olive", "rebeccapurple"],
    ):
        ax.plot(
            portals.evaluations, resChosen, "-o", lw=1.0, c=c, markersize=2, label=label
        )
        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                [portals.numOrig, portals.numBest, portals.numExtra],
                ["r", "g", "m"],
                ["Initial", "Best", "Last"],
                ["o", "s", "*"],
            )
        ):
            if (indexUse is None) or (indexUse >= len(portals.profiles)):
                continue
            ax.plot(
                [portals.evaluations[indexUse]],
                [resChosen[indexUse]],
                "o",
                color=col,
                markersize=4,
            )

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=12,
            fromtop=0.75,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    GRAPHICStools.addDenseAxis(ax, n=5)
    ax.set_xlabel("Iterations (calls/radius)")
    ax.set_ylabel("Residual Definitions")
    ax.set_xlim(left=0)
    try:
        ax.set_yscale("log")
    except:
        pass
    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=True, size=fontsize_leg * 2.0
    )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')

    ax = axA

    ax.plot(
        portals.DVdistMetric_x,
        portals.DVdistMetric_y,
        "-o",
        c="olive",
        lw=1.0,
        markersize=2,
        label=r"$||\Delta x||_\infty$",
    )  #'$\\Delta$ $a/L_{X}$ (%)')

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            [portals.numOrig, portals.numBest, portals.numExtra],
            ["r", "g", "m"],
            ["Initial", "Best", "Last"],
            ["o", "s", "*"],
        )
    ):
        if (indexUse is None) or (indexUse >= len(portals.profiles)):
            continue
        v = portals.chiR_Ricci
        try:
            axt.plot(
                [portals.evaluations[indexUse]],
                [portals.DVdistMetric_y[indexUse]],
                "o",
                color=col,
                markersize=4,
            )
        except:
            pass

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=12,
            fromtop=0.75,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    ax.set_ylabel("$\\Delta$ $a/L_{X}$ (%)")
    ax.set_xlim(left=0)
    try:
        ax.set_yscale("log")
    except:
        pass
    ax.set_xticklabels([])

    if includeRicci and portals.chiR_Ricci is not None:
        axt = axA.twinx()
        (l2,) = axt.plot(
            portals.DVdistMetric_x,
            portals.DVdistMetric_y,
            "-o",
            c="olive",
            lw=1.0,
            markersize=2,
            label="$\\Delta$ $a/L_{X}$",
        )
        axt.plot(
            portals.evaluations,
            portals.chiR_Ricci,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$\\chi_R$",
        )
        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                [portals.numOrig, portals.numBest, portals.numExtra],
                ["r", "g", "m"],
                ["Initial", "Best", "Last"],
                ["o", "s", "*"],
            )
        ):
            if (indexUse is None) or (indexUse >= len(portals.profiles)):
                continue
            v = portals.chiR_Ricci
            axt.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                "o",
                color=col,
                markersize=4,
            )
        axt.set_ylabel("Ricci Metric, $\\chi_R$")
        axt.set_ylim([0, 1])
        axt.legend(loc="best", prop={"size": fontsize_leg * 1.5})
        l2.set_visible(False)
    elif portals.aLTn_perc is not None:
        ax = axA  # .twinx()

        x = portals.evaluations

        if len(x) > len(portals.aLTn_perc):
            x = x[:-1]

        x0, aLTn_perc0 = [], []
        for i in range(len(portals.aLTn_perc)):
            if portals.aLTn_perc[i] is not None:
                x0.append(x[i])
                aLTn_perc0.append(portals.aLTn_perc[i].mean())
        ax.plot(
            x0,
            aLTn_perc0,
            "-o",
            c="rebeccapurple",
            lw=1.0,
            markersize=2,
            label="$\\Delta$ $a/L_{X}^*$ (%)",
        )

        v = portals.aLTn_perc[portals.numOrig].mean()
        ax.plot([portals.evaluations[portals.numOrig]], v, "o", color="r", markersize=4)
        try:
            v = portals.aLTn_perc[portals.numBest].mean()
            ax.plot(
                [portals.evaluations[portals.numBest]],
                [v],
                "o",
                color="g",
                markersize=4,
            )
        except:
            pass

        ax.set_ylabel("$\\Delta$ $a/L_{X}^*$ (%)")
        try:
            ax.set_yscale("log")
        except:
            pass

        (l2,) = axA.plot(
            x0,
            aLTn_perc0,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$\\Delta$ $a/L_{X}^*$ (%)",
        )
        axA.legend(loc="upper center", prop={"size": 7})
        l2.set_visible(False)

    else:
        GRAPHICStools.addDenseAxis(ax, n=5)

    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=False, size=fontsize_leg
    )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')

    ax = axQ

    isThereFusion = np.nanmax(portals.FusionGain) > 0

    if isThereFusion:
        v = portals.FusionGain
        axt6 = ax.twinx()  # None
    else:
        v = portals.tauE
        axt6 = None
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")

    ax.plot(portals.evaluations, v, "-o", lw=1.0, c="olive", markersize=2, label="$Q$")
    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            [portals.numOrig, portals.numBest, portals.numExtra],
            ["r", "g", "m"],
            ["Initial", "Best", "Last"],
            ["o", "s", "*"],
        )
    ):
        if (indexUse is None) or (indexUse >= len(portals.profiles)):
            continue
        ax.plot(
            [portals.evaluations[indexUse]], [v[indexUse]], "o", color=col, markersize=4
        )

    vmin, vmax = np.max([0, np.nanmin(v)]), np.nanmax(v)
    ext = 0.8
    ax.set_ylim([vmin * (1 - ext), vmax * (1 + ext)])
    ax.set_ylim([0, vmax * (1 + ext)])

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=8,
            fromtop=0.1,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    if axt6 is None:
        GRAPHICStools.addDenseAxis(ax, n=5, grid=axt6 is None)

    if isThereFusion:
        ax.set_ylabel("$Q$")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, size=fontsize_leg
        )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')
    else:
        ax.set_ylabel("$\\tau_E$ (s)")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=False, size=fontsize_leg
        )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')
    ax.set_xlim(left=0)
    ax.set_xticklabels([])

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="surrogate",
            orientation="vertical",
            color="b",
            lw=0.25,
            ls="--",
            alpha=1.0,
            fontsize=7,
            fromtop=0.72,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="left",
            separation=0.2,
        )
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="training",
            orientation="vertical",
            color="k",
            lw=0.01,
            ls="--",
            alpha=1.0,
            fontsize=7,
            fromtop=0.72,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    if (axt6 is not None) and (isThereFusion):
        v = portals.FusionPower
        axt6.plot(
            portals.evaluations,
            v,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$P_{fus}$",
        )
        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                [portals.numOrig, portals.numBest, portals.numExtra],
                ["r", "g", "m"],
                ["Initial", "Best", "Last"],
                ["o", "s", "*"],
            )
        ):
            if (indexUse is None) or (indexUse >= len(portals.profiles)):
                continue
            axt6.plot(
                [portals.evaluations[indexUse]],
                [v[indexUse]],
                "s",
                color=col,
                markersize=4,
            )

        axt6.set_ylabel("$P_{fus}$ (MW)")
        axt6.set_ylim(bottom=0)

        (l2,) = ax.plot(
            portals.evaluations,
            v,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$P_{fus}$",
        )
        ax.legend(loc="lower left", prop={"size": fontsize_leg})
        l2.set_visible(False)

    for ax in [axQ, axA, axR, axC]:
        ax.set_xlim([0, len(portals.FusionGain) + 2])

    # for ax in [axA,axR,axC]:
    # 	ax.yaxis.tick_right()
    # 	ax.yaxis.set_label_position("right")

    # print(
    #     "\t* Reminder: With the exception of the Residual plot, the rest are calculated with the original profiles, not necesarily modified by targets",
    #     typeMsg="i",
    # )


def varToReal(y, prfs_model):
    """
    NEO
    """

    of, cal, res = prfs_model.mainFunction.scalarized_objective(
        torch.Tensor(y).to(prfs_model.mainFunction.dfT).unsqueeze(0)
    )

    cont = 0
    Qe, Qi, Ge, GZ, Mt = [], [], [], [], []
    Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = [], [], [], [], []
    for prof in prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        for rad in prfs_model.mainFunction.TGYROparameters["RhoLocations"]:
            if prof == "te":
                Qe.append(of[0, cont])
                Qe_tar.append(cal[0, cont])
            if prof == "ti":
                Qi.append(of[0, cont])
                Qi_tar.append(cal[0, cont])
            if prof == "ne":
                Ge.append(of[0, cont])
                Ge_tar.append(cal[0, cont])
            if prof == "nZ":
                GZ.append(of[0, cont])
                GZ_tar.append(cal[0, cont])
            if prof == "w0":
                Mt.append(of[0, cont])
                Mt_tar.append(cal[0, cont])

            cont += 1

    Qe, Qi, Ge, GZ, Mt = (
        np.array(Qe),
        np.array(Qi),
        np.array(Ge),
        np.array(GZ),
        np.array(Mt),
    )
    Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = (
        np.array(Qe_tar),
        np.array(Qi_tar),
        np.array(Ge_tar),
        np.array(GZ_tar),
        np.array(Mt_tar),
    )

    return Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar


def plotVars(
    prfs_model,
    y,
    axs,
    axsR,
    contP=0,
    lines=["-s", "--o"],
    yerr=None,
    plotPoints=None,
    plotResidual=True,
    lab="",
    color=None,
    plotErr=[False] * 10,
    colors=GRAPHICStools.listColors(),
):
    [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f] = axs
    [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r] = axsR

    ms, cp, lwc = 4, 2, 0.5

    if plotPoints is None:
        plotPoints = range(y.shape[0])

    cont = -1
    for i in plotPoints:
        cont += 1

        lw = 1.5 if i == 0 else 1.0

        contP += 1

        x_var = (
            prfs_model.mainFunction.surrogate_parameters["powerstate"]
            .plasma["roa"][0, 1:]
            .cpu()
            .numpy()
        )  # prfs_model.mainFunction.TGYROparameters['RhoLocations']

        try:
            Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = varToReal(
                y[i, :].detach().cpu().numpy(), prfs_model
            )
        except:
            continue

        if yerr is not None:
            (
                QeEl,
                QiEl,
                GeEl,
                GZEl,
                MtEl,
                Qe_tarEl,
                Qi_tarEl,
                Ge_tarEl,
                GZ_tarEl,
                Mt_tarEl,
            ) = varToReal(yerr[0][i, :].detach().cpu().numpy(), prfs_model)
            (
                QeEu,
                QiEu,
                GeEu,
                GZEu,
                MtEu,
                Qe_tarEu,
                Qi_tarEu,
                Ge_tarEu,
                GZ_tarEu,
                Mt_tarEu,
            ) = varToReal(yerr[1][i, :].detach().cpu().numpy(), prfs_model)

        ax = axTe_f

        if lines[0] is not None:
            ax.plot(
                x_var,
                Qe,
                lines[0],
                c=colors[contP] if color is None else color,
                label="$Q$" + lab if i == 0 else "",
                lw=lw,
                markersize=ms,
            )
        if lines[1] is not None:
            ax.plot(
                x_var,
                Qe_tar,
                lines[1],
                c=colors[contP] if color is None else color,
                lw=lw,
                markersize=ms,
                label="$Q^T$" + lab if i == 0 else "",
            )
        if yerr is not None:
            ax.errorbar(
                x_var,
                Qe,
                c=colors[contP] if color is None else color,
                yerr=[QeEl, QeEu],
                capsize=cp,
                capthick=lwc,
                fmt="none",
                lw=lw,
                markersize=ms,
                label="$Q$" + lab if i == 0 else "",
            )

        ax = axTi_f
        if lines[0] is not None:
            ax.plot(
                x_var,
                Qi,
                lines[0],
                c=colors[contP] if color is None else color,
                label=f"#{i}",
                lw=lw,
                markersize=ms,
            )
        if lines[1] is not None:
            ax.plot(
                x_var,
                Qi_tar,
                lines[1],
                c=colors[contP] if color is None else color,
                lw=lw,
                markersize=ms,
            )
        if yerr is not None:
            ax.errorbar(
                x_var,
                Qi,
                c=colors[contP] if color is None else color,
                yerr=[QiEl, QiEu],
                capsize=cp,
                capthick=lwc,
                fmt="none",
                lw=lw,
                markersize=ms,
            )

        if axne_f is not None:
            ax = axne_f
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    Ge,
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
            if lines[1] is not None:
                ax.plot(
                    x_var,
                    Ge_tar,
                    lines[1],
                    c=colors[contP] if color is None else color,
                    lw=lw,
                    markersize=ms,
                )
            if yerr is not None:
                ax.errorbar(
                    x_var,
                    Ge,
                    c=colors[contP] if color is None else color,
                    yerr=[GeEl, GeEu],
                    capsize=cp,
                    capthick=lwc,
                    fmt="none",
                    lw=lw,
                    markersize=ms,
                )

        if axnZ_f is not None:
            ax = axnZ_f
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    GZ,
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
            if lines[1] is not None:
                ax.plot(
                    x_var,
                    GZ_tar,
                    lines[1],
                    c=colors[contP] if color is None else color,
                    lw=lw,
                    markersize=ms,
                )
            if yerr is not None:
                ax.errorbar(
                    x_var,
                    GZ,
                    c=colors[contP] if color is None else color,
                    yerr=[GZEl, GZEu],
                    capsize=cp,
                    capthick=lwc,
                    fmt="none",
                    lw=lw,
                    markersize=ms,
                )

        if axw0_f is not None:
            ax = axw0_f
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    Mt,
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
            if lines[1] is not None:
                ax.plot(
                    x_var,
                    Mt_tar,
                    lines[1],
                    c=colors[contP] if color is None else color,
                    lw=lw,
                    markersize=ms,
                )
            if yerr is not None:
                ax.errorbar(
                    x_var,
                    Mt,
                    c=colors[contP] if color is None else color,
                    yerr=[MtEl, MtEu],
                    capsize=cp,
                    capthick=lwc,
                    fmt="none",
                    lw=lw,
                    markersize=ms,
                )

        if plotResidual:
            ax = axTe_r
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    (Qe - Qe_tar),
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label="$Q-Q^T$" + lab if i == 0 else "",
                    lw=lw,
                    markersize=ms,
                )
                if plotErr[cont]:
                    ax.errorbar(
                        x_var,
                        (Qe - Qe_tar),
                        c=colors[contP] if color is None else color,
                        yerr=[QeEl, QeEu],
                        capsize=cp,
                        capthick=lwc,
                        fmt="none",
                        lw=0.5,
                        markersize=0,
                    )

            ax = axTi_r
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    (Qi - Qi_tar),
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
                if plotErr[cont]:
                    ax.errorbar(
                        x_var,
                        (Qi - Qi_tar),
                        c=colors[contP] if color is None else color,
                        yerr=[QiEl, QiEu],
                        capsize=cp,
                        capthick=lwc,
                        fmt="none",
                        lw=0.5,
                        markersize=0,
                    )

            if axne_r is not None:
                ax = axne_r
                if lines[0] is not None:
                    ax.plot(
                        x_var,
                        (Ge - Ge_tar),
                        lines[0],
                        c=colors[contP] if color is None else color,
                        label=f"#{i}",
                        lw=lw,
                        markersize=ms,
                    )
                    if plotErr[cont]:
                        ax.errorbar(
                            x_var,
                            (Ge - Ge_tar),
                            c=colors[contP] if color is None else color,
                            yerr=[GeEl, GeEu],
                            capsize=cp,
                            capthick=lwc,
                            fmt="none",
                            lw=0.5,
                            markersize=0,
                        )

            if axnZ_r is not None:
                ax = axnZ_r
                if lines[0] is not None:
                    ax.plot(
                        x_var,
                        (GZ - GZ_tar),
                        lines[0],
                        c=colors[contP] if color is None else color,
                        label=f"#{i}",
                        lw=lw,
                        markersize=ms,
                    )
                    if plotErr[cont]:
                        ax.errorbar(
                            x_var,
                            (GZ - GZ_tar),
                            c=colors[contP] if color is None else color,
                            yerr=[GZEl, GZEu],
                            capsize=cp,
                            capthick=lwc,
                            fmt="none",
                            lw=0.5,
                            markersize=0,
                        )
            if axw0_r is not None:
                ax = axw0_r
                if lines[0] is not None:
                    ax.plot(
                        x_var,
                        (Mt - Mt_tar),
                        lines[0],
                        c=colors[contP] if color is None else color,
                        label=f"#{i}",
                        lw=lw,
                        markersize=ms,
                    )
                    if plotErr[cont]:
                        ax.errorbar(
                            x_var,
                            (Mt - Mt_tar),
                            c=colors[contP] if color is None else color,
                            yerr=[MtEl, MtEu],
                            capsize=cp,
                            capthick=lwc,
                            fmt="none",
                            lw=0.5,
                            markersize=0,
                        )

    return contP


def plotExpected(
    prfs_model,
    folder,
    fn,
    labelsFluxes={},
    step=-1,
    plotPoints=[0],
    labelAssigned=["0"],
    plotNext=True,
    MITIMextra_dict=None,
    stds=2,
):
    model = prfs_model.steps[step].GP["combined_model"]

    x_train_num = prfs_model.steps[step].train_X.shape[0]

    posZ = prfs_model.mainFunction.PORTALSparameters["ImpurityOfInterest"] - 1

    # ---- Training
    x_train = torch.from_numpy(prfs_model.steps[step].train_X).to(model.train_X)
    y_trainreal = torch.from_numpy(prfs_model.steps[step].train_Y).to(model.train_X)
    yL_trainreal = torch.from_numpy(prfs_model.steps[step].train_Ystd).to(model.train_X)
    yU_trainreal = torch.from_numpy(prfs_model.steps[step].train_Ystd).to(model.train_X)

    y_train, yU_train, yL_train, _ = model.predict(x_train)

    # ---- Next
    x_next = y_next = yU_next = yL_next = None
    if plotNext:
        try:
            x_next = prfs_model.steps[step].x_next
            y_next, yU_next, yL_next, _ = model.predict(x_next)
        except:
            pass

    # ---- Get profiles
    profiles = []
    if MITIMextra_dict is not None:
        print(f"\t- Reading TGYRO and PROFILES from MITIMextra_dict")
        for i in plotPoints:
            profiles.append(MITIMextra_dict[i]["tgyro"].results["use"].profiles)
    else:
        for i in plotPoints:
            file = f"{folder}/Execution/Evaluation.{i}/model_complete/input.gacode"
            p = PROFILEStools.PROFILES_GACODE(file, calculateDerived=False)
            profiles.append(p)

    profiles_next = None
    if x_next is not None:
        try:
            file = f"{folder}/Execution/Evaluation.{x_train_num}/model_complete/input.gacode"
            profiles_next = PROFILEStools.PROFILES_GACODE(file, calculateDerived=False)

            try:
                file = f"{folder}/Execution/Evaluation.{x_train_num}/model_complete/input.gacode.new"
                profiles_next_new = PROFILEStools.PROFILES_GACODE(
                    file, calculateDerived=True
                )
                profiles_next_new.printInfo(label="NEXT")
            except:
                profiles_next_new = profiles_next
                profiles_next_new.deriveQuantities()
        except:
            pass

    # ---- Plot
    fig = fn.add_figure(label="PORTALS Expected")

    numprofs = len(prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"])

    if numprofs <= 4:
        wspace = 0.3
    else:
        wspace = 0.5

    grid = plt.GridSpec(nrows=4, ncols=numprofs, hspace=0.2, wspace=wspace)

    axTe = fig.add_subplot(grid[0, 0])
    axTe.set_title("Electron Temperature")
    axTe_g = fig.add_subplot(grid[1, 0], sharex=axTe)
    axTe_f = fig.add_subplot(grid[2, 0], sharex=axTe)
    axTe_r = fig.add_subplot(grid[3, 0], sharex=axTe)

    axTi = fig.add_subplot(grid[0, 1], sharex=axTe)
    axTi.set_title("Ion Temperature")
    axTi_g = fig.add_subplot(grid[1, 1], sharex=axTe)
    axTi_f = fig.add_subplot(grid[2, 1], sharex=axTe)
    axTi_r = fig.add_subplot(grid[3, 1], sharex=axTe)

    cont = 0
    if "ne" in prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        axne = fig.add_subplot(grid[0, 2 + cont], sharex=axTe)
        axne.set_title("Electron Density")
        axne_g = fig.add_subplot(grid[1, 2 + cont], sharex=axTe)
        axne_f = fig.add_subplot(grid[2, 2 + cont], sharex=axTe)
        axne_r = fig.add_subplot(grid[3, 2 + cont], sharex=axTe)
        cont += 1
    else:
        axne = axne_g = axne_f = axne_r = None
    if "nZ" in prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        impos = prfs_model.mainFunction.PORTALSparameters["ImpurityOfInterest"]
        labIon = f"Ion {impos} ({profiles[0].Species[impos-1]['N']}{int(profiles[0].Species[impos-1]['Z'])},{int(profiles[0].Species[impos-1]['A'])})"
        axnZ = fig.add_subplot(grid[0, 2 + cont], sharex=axTe)
        axnZ.set_title(f"{labIon} Density")
        axnZ_g = fig.add_subplot(grid[1, 2 + cont], sharex=axTe)
        axnZ_f = fig.add_subplot(grid[2, 2 + cont], sharex=axTe)
        axnZ_r = fig.add_subplot(grid[3, 2 + cont], sharex=axTe)
        cont += 1
    else:
        axnZ = axnZ_g = axnZ_f = axnZ_r = None

    if "w0" in prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        axw0 = fig.add_subplot(grid[0, 2 + cont], sharex=axTe)
        axw0.set_title("Rotation")
        axw0_g = fig.add_subplot(grid[1, 2 + cont], sharex=axTe)
        axw0_f = fig.add_subplot(grid[2, 2 + cont], sharex=axTe)
        axw0_r = fig.add_subplot(grid[3, 2 + cont], sharex=axTe)
        cont += 1
    else:
        axw0 = axw0_g = axw0_f = axw0_r = None

    colorsA = GRAPHICStools.listColors()
    colors = []
    coli = -1
    for label in labelAssigned:
        if "best" in label:
            colors.append("g")
        elif "last" in label:
            colors.append("m")
        elif "base" in label:
            colors.append("r")
        else:
            coli += 1
            while colorsA[coli] in ["g", "m", "r"]:
                coli += 1
            colors.append(colorsA[coli])

    rho = profiles[0].profiles["rho(-)"]
    roa = profiles[0].derived["roa"]
    rhoVals = prfs_model.mainFunction.TGYROparameters["RhoLocations"]
    roaVals = np.interp(rhoVals, rho, roa)
    lastX = roaVals[-1]

    # ---- Plot profiles
    cont = -1
    for i in range(len(profiles)):
        cont += 1

        p = profiles[i]

        ix = np.argmin(np.abs(p.derived["roa"] - lastX)) + 1

        lw = 1.0 if cont > 0 else 1.5

        ax = axTe
        ax.plot(
            p.derived["roa"],
            p.profiles["te(keV)"],
            "-",
            c=colors[cont],
            label=labelAssigned[cont],
            lw=lw,
        )
        ax = axTi
        ax.plot(
            p.derived["roa"], p.profiles["ti(keV)"][:, 0], "-", c=colors[cont], lw=lw
        )
        if axne is not None:
            ax = axne
            ax.plot(
                p.derived["roa"],
                p.profiles["ne(10^19/m^3)"] * 1e-1,
                "-",
                c=colors[cont],
                lw=lw,
            )
        if axnZ is not None:
            ax = axnZ
            ax.plot(
                p.derived["roa"],
                p.profiles["ni(10^19/m^3)"][:, posZ] * 1e-1,
                "-",
                c=colors[cont],
                lw=lw,
            )
        if axw0 is not None:
            ax = axw0
            ax.plot(
                p.derived["roa"],
                p.profiles["w0(rad/s)"] * 1e-3,
                "-",
                c=colors[cont],
                lw=lw,
            )

        ax = axTe_g
        ax.plot(
            p.derived["roa"][:ix],
            p.derived["aLTe"][:ix],
            "-o",
            c=colors[cont],
            markersize=0,
            lw=lw,
        )
        ax = axTi_g
        ax.plot(
            p.derived["roa"][:ix],
            p.derived["aLTi"][:ix, 0],
            "-o",
            c=colors[cont],
            markersize=0,
            lw=lw,
        )
        if axne_g is not None:
            ax = axne_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["aLne"][:ix],
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )

        if axnZ_g is not None:
            ax = axnZ_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["aLni"][:ix, posZ],
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )
        if axw0_g is not None:
            ax = axw0_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["dw0dr"][:ix] * factor_dw0dr,
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )

    cont += 1

    # ---- Plot profiles next

    if profiles_next is not None:
        p = profiles_next
        roa = profiles_next_new.derived["roa"]
        dw0dr = profiles_next_new.derived["dw0dr"]

        ix = np.argmin(np.abs(roa - lastX)) + 1

        lw = 1.5

        ax = axTe
        ax.plot(
            roa,
            p.profiles["te(keV)"],
            "-",
            c="k",
            label=f"#{x_train_num} (next)",
            lw=lw,
        )
        ax = axTi
        ax.plot(roa, p.profiles["ti(keV)"][:, 0], "-", c="k", lw=lw)
        if axne is not None:
            ax = axne
            ax.plot(roa, p.profiles["ne(10^19/m^3)"] * 1e-1, "-", c="k", lw=lw)

        if axnZ is not None:
            ax = axnZ
            ax.plot(roa, p.profiles["ni(10^19/m^3)"][:, posZ] * 1e-1, "-", c="k", lw=lw)
        if axw0 is not None:
            ax = axw0
            ax.plot(roa, p.profiles["w0(rad/s)"] * 1e-3, "-", c="k", lw=lw)

        ax = axTe_g
        ax.plot(roa[:ix], p.derived["aLTe"][:ix], "o-", c="k", markersize=0, lw=lw)
        ax = axTi_g
        ax.plot(roa[:ix], p.derived["aLTi"][:ix, 0], "o-", c="k", markersize=0, lw=lw)

        if axne_g is not None:
            ax = axne_g
            ax.plot(roa[:ix], p.derived["aLne"][:ix], "o-", c="k", markersize=0, lw=lw)

        if axnZ_g is not None:
            ax = axnZ_g
            ax.plot(
                roa[:ix], p.derived["aLni"][:ix, posZ], "-o", c="k", markersize=0, lw=lw
            )
        if axw0_g is not None:
            ax = axw0_g
            ax.plot(
                roa[:ix], dw0dr[:ix] * factor_dw0dr, "-o", c="k", markersize=0, lw=lw
            )

        axTe_g_twin = axTe_g.twinx()
        axTi_g_twin = axTi_g.twinx()

        ranges = [-30, 30]

        rho = profiles_next_new.profiles["rho(-)"]
        rhoVals = prfs_model.mainFunction.TGYROparameters["RhoLocations"]
        roaVals = np.interp(rhoVals, rho, roa)

        p0 = profiles[0]
        zVals = []
        z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
        for roai in roaVals:
            zVals.append(np.interp(roai, roa, z))
        axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

        if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
            p0 = profiles[1]
            zVals = []
            z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4)

        axTe_g_twin.set_ylim(ranges)
        axTe_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        p0 = profiles[0]
        zVals = []
        z = (
            (p.derived["aLTi"][:, 0] - p0.derived["aLTi"][:, 0])
            / p0.derived["aLTi"][:, 0]
        ) * 100.0
        for roai in roaVals:
            zVals.append(np.interp(roai, roa, z))
        axTi_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

        if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
            p0 = profiles[1]
            zVals = []
            z = (
                (p.derived["aLTi"][:, 0] - p0.derived["aLTi"][:, 0])
                / p0.derived["aLTi"][:, 0]
            ) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTi_g_twin.plot(roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4)

        axTi_g_twin.set_ylim(ranges)
        axTi_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        for ax in [axTe_g_twin, axTi_g_twin]:
            ax.axhline(y=0, ls="-.", lw=0.2, c="k")

        if axne_g is not None:
            axne_g_twin = axne_g.twinx()

            p0 = profiles[0]
            zVals = []
            z = ((p.derived["aLne"] - p0.derived["aLne"]) / p0.derived["aLne"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axne_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = profiles[1]
                zVals = []
                z = (
                    (p.derived["aLne"] - p0.derived["aLne"]) / p0.derived["aLne"]
                ) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axne_g_twin.plot(
                    roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4
                )

            axne_g_twin.set_ylim(ranges)
            axne_g_twin.set_ylabel("(%) from last or best", fontsize=8)

            axne_g_twin.axhline(y=0, ls="-.", lw=0.2, c="k")

        if axnZ_g is not None:
            axnZ_g_twin = axnZ_g.twinx()

            p0 = profiles[0]
            zVals = []
            z = (
                (p.derived["aLni"][:, posZ] - p0.derived["aLni"][:, posZ])
                / p0.derived["aLni"][:, posZ]
            ) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axnZ_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = profiles[1]
                zVals = []
                z = (
                    (p.derived["aLni"][:, posZ] - p0.derived["aLni"][:, posZ])
                    / p0.derived["aLni"][:, posZ]
                ) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axnZ_g_twin.plot(
                    roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4
                )

            axnZ_g_twin.set_ylim(ranges)
            axnZ_g_twin.set_ylabel("(%) from last or best", fontsize=8)
        else:
            axnZ_g_twin = None

        if axw0_g is not None:
            axw0_g_twin = axw0_g.twinx()

            p0 = profiles[0]
            zVals = []
            z = ((dw0dr - p0.derived["dw0dr"]) / p0.derived["dw0dr"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axw0_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = profiles[1]
                zVals = []
                z = ((dw0dr - p0.derived["dw0dr"]) / p0.derived["dw0dr"]) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axw0_g_twin.plot(
                    roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4
                )

            axw0_g_twin.set_ylim(ranges)
            axw0_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        else:
            axw0_g_twin = None

        for ax in [axnZ_g_twin, axw0_g_twin]:
            if ax is not None:
                ax.axhline(y=0, ls="-.", lw=0.2, c="k")

    else:
        axTe_g_twin = axTi_g_twin = axne_g_twin = axnZ_g_twin = axw0_g_twin = None

    # ---- Plot fluxes
    cont = plotVars(
        prfs_model,
        y_trainreal,
        [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
        [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r],
        contP=-1,
        lines=["-s", "--o"],
        plotPoints=plotPoints,
        yerr=[yL_trainreal * stds, yU_trainreal * stds],
        lab="",
        plotErr=np.append([True], [False] * len(y_trainreal)),
        colors=colors,
    )
    _ = plotVars(
        prfs_model,
        y_train,
        [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
        [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r],
        contP=-1,
        lines=["-.*", None],
        plotPoints=plotPoints,
        plotResidual=False,
        lab=" (surr)",
        colors=colors,
    )  # ,yerr=[yL_train,yU_train])

    if y_next is not None:
        cont = plotVars(
            prfs_model,
            y_next,
            [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
            [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r],
            contP=cont,
            lines=["-s", "--o"],
            yerr=[y_next - yL_next * stds / 2.0, yU_next - y_next * stds / 2.0],
            plotPoints=None,
            color="k",
            plotErr=[True],
            colors=colors,
        )

    # ---------------
    n = 10  # 5
    ax = axTe
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylabel("Te (keV)")
    ax.set_ylim(bottom=0)
    GRAPHICStools.addDenseAxis(ax, n=n)
    # ax.	set_xticklabels([])
    ax = axTi
    ax.set_xlim([0, 1])
    ax.set_ylabel("Ti (keV)")
    ax.set_ylim(bottom=0)
    GRAPHICStools.addDenseAxis(ax, n=n)
    # ax.set_xticklabels([])
    if axne is not None:
        ax = axne
        ax.set_xlim([0, 1])
        ax.set_ylabel("ne ($10^{20}m^{-3}$)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax, n=n)
    # ax.set_xticklabels([])

    if axnZ is not None:
        ax = axnZ
        ax.set_xlim([0, 1])
        ax.set_ylabel("nZ ($10^{20}m^{-3}$)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axw0 is not None:
        ax = axw0
        ax.set_xlim([0, 1])
        ax.set_ylabel("$w_0$ (krad/s)")
        GRAPHICStools.addDenseAxis(ax, n=n)

    roacoarse = (
        prfs_model.mainFunction.surrogate_parameters["powerstate"]
        .plasma["roa"][0, 1:]
        .cpu()
        .numpy()
    )

    ax = axTe_g
    ax.set_xlim([0, 1])
    ax.set_ylabel("$a/L_{Te}$")
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0,5]);
    # ax.set_xticklabels([])
    if axTe_g_twin is not None:
        axTe_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
        if len(roacoarse) < 6:
            axTe_g_twin.set_xticks([round(i, 2) for i in roacoarse])
        GRAPHICStools.addDenseAxis(axTe_g_twin, n=n)
    else:
        GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTi_g
    ax.set_xlim([0, 1])
    ax.set_ylabel("$a/L_{Ti}$")
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0,5]);
    # ax.set_xticklabels([])
    if axTi_g_twin is not None:
        axTi_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
        if len(roacoarse) < 6:
            axTi_g_twin.set_xticks([round(i, 2) for i in roacoarse])
        GRAPHICStools.addDenseAxis(axTi_g_twin, n=n)
    else:
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axne_g is not None:
        ax = axne_g
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{ne}$")
        ax.set_ylim(bottom=0)
        # ax.set_ylim([0,5]);
        # ax.set_xticklabels([])
        if axne_g_twin is not None:
            axne_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
            if len(roacoarse) < 6:
                axne_g_twin.set_xticks([round(i, 2) for i in roacoarse])
            GRAPHICStools.addDenseAxis(axne_g_twin, n=n)
        else:
            GRAPHICStools.addDenseAxis(ax, n=n)

    if axnZ_g is not None:
        ax = axnZ_g
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{nZ}$")
        ax.set_ylim(bottom=0)
        # ax.set_ylim([0,5]);
        if axnZ_g_twin is not None:
            axnZ_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
            if len(roacoarse) < 6:
                axnZ_g_twin.set_xticks([round(i, 2) for i in roacoarse])
            GRAPHICStools.addDenseAxis(axnZ_g_twin, n=n)
        else:
            GRAPHICStools.addDenseAxis(ax, n=n)

    if axw0_g is not None:
        ax = axw0_g
        ax.set_xlim([0, 1])
        ax.set_ylabel(label_dw0dr)
        # ax.set_ylim(bottom=0); #ax.set_ylim([0,5]);
        if axw0_g_twin is not None:
            axw0_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
            if len(roacoarse) < 6:
                axw0_g_twin.set_xticks([round(i, 2) for i in roacoarse])
            GRAPHICStools.addDenseAxis(axw0_g_twin, n=n)
        else:
            GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTe_f
    ax.set_xlim([0, 1])
    ax.set_ylabel(labelsFluxes["te"])
    ax.set_ylim(bottom=0)
    # ax.legend(loc='best',prop={'size':6})
    # ax.set_xticklabels([])
    GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTi_f
    ax.set_xlim([0, 1])
    ax.set_ylabel(labelsFluxes["ti"])
    ax.set_ylim(bottom=0)
    # ax.set_xticklabels([])
    GRAPHICStools.addDenseAxis(ax, n=n)

    if axne_f is not None:
        ax = axne_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(labelsFluxes["ne"])
        # GRAPHICStools.addDenseAxis(ax,n=n)
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axnZ_f is not None:
        ax = axnZ_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(labelsFluxes["nZ"])
        # GRAPHICStools.addDenseAxis(ax,n=n)
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axw0_f is not None:
        ax = axw0_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(labelsFluxes["w0"])
        # GRAPHICStools.addDenseAxis(ax,n=n)
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTe_r
    ax.set_xlim([0, 1])
    ax.set_xlabel("$r/a$")
    ax.set_ylabel("Residual " + labelsFluxes["te"])
    GRAPHICStools.addDenseAxis(ax, n=n)
    ax.axhline(y=0, lw=0.5, ls="--", c="k")

    ax = axTi_r
    ax.set_xlim([0, 1])
    ax.set_xlabel("$r/a$")
    ax.set_ylabel("Residual " + labelsFluxes["ti"])
    GRAPHICStools.addDenseAxis(ax, n=n)
    ax.axhline(y=0, lw=0.5, ls="--", c="k")

    if axne_r is not None:
        ax = axne_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + labelsFluxes["ne"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")  #

    if axnZ_r is not None:
        ax = axnZ_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + labelsFluxes["nZ"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

    if axw0_r is not None:
        ax = axw0_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + labelsFluxes["w0"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

    try:
        Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = varToReal(
            y_trainreal[prfs_model.BOmetrics["overall"]["indBest"], :]
            .detach()
            .cpu()
            .numpy(),
            prfs_model,
        )
        rangePlotResidual = np.max([np.max(Qe_tar), np.max(Qi_tar), np.max(Ge_tar)])
        for ax in [axTe_r, axTi_r, axne_r]:
            ax.set_ylim(
                [-rangePlotResidual * 0.5, rangePlotResidual * 0.5]
            )  # 50% of max targets
    except:
        pass


def plotFluxComparison(
    p,
    t,
    axTe_f,
    axTi_f,
    axne_f,
    axnZ_f,
    axw0_f,
    posZ=3,
    labZ="Z",
    includeFirst=True,
    alpha=1.0,
    stds=2,
    col="b",
    lab="",
    msFlux=1,
    useConvectiveFluxes=False,
    maxStore=False,
    plotFlows=True,
    plotTargets=True,
    decor=True,
    fontsize_leg=12,
    useRoa=False,
    locLeg="upper left",
):
    labelsFluxesF = {
        "te": "$Q_e$ ($MW/m^2$)",
        "ti": "$Q_i$ ($MW/m^2$)",
        "ne": "$\\Gamma_e$ ($10^{20}/s/m^2$)",
        "nZ": f"$\\Gamma_{labZ}$ ($10^{{20}}/s/m^2$)",
        "w0": "$M_T$ ($J/m^2$)",
    }

    r = t.rho if not useRoa else t.roa

    ixF = 0 if includeFirst else 1

    (
        QeBest_min,
        QeBest_max,
        QiBest_min,
        QiBest_max,
        GeBest_min,
        GeBest_max,
        GZBest_min,
        GZBest_max,
        MtBest_min,
        MtBest_max,
    ) = [None] * 10

    axTe_f.plot(
        r[0][ixF:],
        t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:],
        "-s",
        c=col,
        lw=2,
        markersize=msFlux,
        label="Transport",
        alpha=alpha,
    )
    if plotTargets:
        axTe_f.plot(
            r[0][ixF:],
            t.Qe_tar[0][ixF:],
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

    try:
        sigma = t.Qe_sim_turb_stds[0][ixF:] + t.Qe_sim_neo_stds[0][ixF:]
    except:
        print("Could not find errors to plot!", typeMsg="w")
        sigma = t.Qe_sim_turb[0][ixF:] * 0.0

    m, M = (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) - stds * sigma, (
        t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]
    ) + stds * sigma
    axTe_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

    if maxStore:
        QeBest_max = np.max([M.max(), t.Qe_tar[0][ixF:].max()])
        QeBest_min = np.min([m.min(), t.Qe_tar[0][ixF:].min()])

    axTi_f.plot(
        r[0][ixF:],
        t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:],
        "-s",
        markersize=msFlux,
        c=col,
        lw=2,
        label="Transport",
        alpha=alpha,
    )
    if plotTargets:
        axTi_f.plot(
            r[0][ixF:],
            t.Qi_tar[0][ixF:],
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

    try:
        sigma = t.QiIons_sim_turb_thr_stds[0][ixF:] + t.QiIons_sim_neo_thr_stds[0][ixF:]
    except:
        sigma = t.Qe_sim_turb[0][ixF:] * 0.0

    m, M = (
        t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:]
    ) - stds * sigma, (
        t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:]
    ) + stds * sigma
    axTi_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

    if maxStore:
        QiBest_max = np.max([M.max(), t.Qi_tar[0][ixF:].max()])
        QiBest_min = np.min([m.min(), t.Qi_tar[0][ixF:].min()])

    if useConvectiveFluxes:
        Ge, Ge_tar = t.Ce_sim_turb + t.Ce_sim_neo, t.Ce_tar
        try:
            sigma = t.Ce_sim_turb_stds[0][ixF:] + t.Ce_sim_neo_stds[0][ixF:]
        except:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0
    else:
        Ge, Ge_tar = (t.Ge_sim_turb + t.Ge_sim_neo), t.Ge_tar
        try:
            sigma = t.Ge_sim_turb_stds[0][ixF:] + t.Ge_sim_neo_stds[0][ixF:]
        except:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

    if axne_f is not None:
        axne_f.plot(
            r[0][ixF:],
            Ge[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )
        if plotTargets:
            axne_f.plot(
                r[0][ixF:],
                Ge_tar[0][ixF:],
                "--",
                c=col,
                lw=2,
                label="Target",
                alpha=alpha,
            )

        m, M = Ge[0][ixF:] - stds * sigma, Ge[0][ixF:] + stds * sigma
        axne_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

    if maxStore:
        GeBest_max = np.max([M.max(), Ge_tar[0][ixF:].max()])
        GeBest_min = np.min([m.min(), Ge_tar[0][ixF:].min()])

    if axnZ_f is not None:
        if useConvectiveFluxes:
            GZ, GZ_tar = (
                t.Ci_sim_turb[posZ, :, :] + t.Ci_sim_neo[posZ, :, :],
                t.Ge_tar * 0.0,
            )
            try:
                sigma = (
                    t.Ci_sim_turb_stds[posZ, 0][ixF:] + t.Ci_sim_neo_stds[posZ, 0][ixF:]
                )
            except:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0
        else:
            GZ, GZ_tar = (
                t.Gi_sim_turb[posZ, :, :] + t.Gi_sim_neo[posZ, :, :]
            ), t.Ge_tar * 0.0
            try:
                sigma = (
                    t.Gi_sim_turb_stds[posZ, 0][ixF:] + t.Gi_sim_neo_stds[posZ, 0][ixF:]
                )
            except:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0
        axnZ_f.plot(
            r[0][ixF:],
            GZ[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )
        if plotTargets:
            axnZ_f.plot(
                r[0][ixF:],
                GZ_tar[0][ixF:],
                "--",
                c=col,
                lw=2,
                label="Target",
                alpha=alpha,
            )

        m, M = (
            GZ[0][ixF:] - stds * sigma,
            GZ[0][ixF:] + stds * sigma,
        )
        axnZ_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

        if maxStore:
            GZBest_max = np.max([M.max(), GZ_tar[0][ixF:].max()])
            GZBest_min = np.min([m.min(), GZ_tar[0][ixF:].min()])

    if axw0_f is not None:
        axw0_f.plot(
            r[0][ixF:],
            t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )
        if plotTargets:
            axw0_f.plot(
                r[0][ixF:],
                t.Mt_tar[0][ixF:],
                "--*",
                c=col,
                lw=2,
                markersize=0,
                label="Target",
                alpha=alpha,
            )

        try:
            sigma = t.Mt_sim_turb_stds[0][ixF:] + t.Mt_sim_neo_stds[0][ixF:]
        except:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

        m, M = (t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:]) - stds * sigma, (
            t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:]
        ) + stds * sigma
        axw0_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

        if maxStore:
            MtBest_max = np.max([M.max(), t.Mt_tar[0][ixF:].max()])
            MtBest_min = np.min([m.min(), t.Mt_tar[0][ixF:].min()])

    if plotFlows:
        tBest = t.profiles_final
        for ax, var, mult in zip(
            [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
            ["qe_MWm2", "qi_MWm2", "ge_10E20m2", None, "mt_Jm2"],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ):
            if ax is not None:
                if var is None:
                    y = tBest.profiles["rho(-)"] * 0.0
                else:
                    y = tBest.derived[var] * mult
                if plotTargets:
                    ax.plot(
                        tBest.profiles["rho(-)"],
                        y,
                        "-.",
                        lw=0.5,
                        c=col,
                        label="Flow",
                        alpha=alpha,
                    )
                else:
                    ax.plot(
                        tBest.profiles["rho(-)"],
                        y,
                        "--",
                        lw=2,
                        c=col,
                        label="Target",
                        alpha=alpha,
                    )

    # -- for legend
    (l1,) = axTe_f.plot(
        r[0][ixF:],
        t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:],
        "-",
        c="k",
        lw=2,
        markersize=0,
        label="Transport",
    )
    (l2,) = axTe_f.plot(
        r[0][ixF:], t.Qe_tar[0][ixF:], "--*", c="k", lw=2, markersize=0, label="Target"
    )
    l3 = axTe_f.fill_between(
        t.roa[0][ixF:],
        (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) - stds,
        (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) + stds,
        facecolor="k",
        alpha=0.3,
    )
    if plotTargets:
        setl = [l1, l3, l2]
        setlab = ["Transport", f"$\\pm{stds}\\sigma$", "Target"]
    else:
        setl = [l1, l3]
        setlab = ["Transport", f"$\\pm{stds}\\sigma$"]
    if plotFlows:
        (l4,) = axTe_f.plot(
            tBest.profiles["rho(-)"],
            tBest.derived["qe_MWm2"],
            "-.",
            c="k",
            lw=1,
            markersize=0,
            label="Transport",
        )
        setl.append(l4)
        setlab.append("Target HR")
    else:
        l4 = l3

    axTe_f.legend(setl, setlab, loc=locLeg, prop={"size": fontsize_leg})
    l1.set_visible(False)
    l2.set_visible(False)
    l3.set_visible(False)
    l4.set_visible(False)
    # ---------------

    if decor:
        ax = axTe_f
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
        ax.set_ylabel(labelsFluxesF["te"])
        ax.set_xlim([0, 1])

        ax = axTi_f
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
        ax.set_ylabel(labelsFluxesF["ti"])
        ax.set_xlim([0, 1])

        if axne_f is not None:
            ax = axne_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["ne"])
            ax.set_xlim([0, 1])

        if axnZ_f is not None:
            ax = axnZ_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["nZ"])
            ax.set_xlim([0, 1])

            GRAPHICStools.addScientificY(ax)

        if axw0_f is not None:
            ax = axw0_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["w0"])
            ax.set_xlim([0, 1])

        if maxStore:
            Qmax = QeBest_max
            Qmax += np.abs(Qmax) * 0.5
            Qmin = QeBest_min
            Qmin -= np.abs(Qmin) * 0.5
            axTe_f.set_ylim([0, Qmax])

            Qmax = QiBest_max
            Qmax += np.abs(Qmax) * 0.5
            Qmin = QiBest_min
            Qmin -= np.abs(Qmin) * 0.5
            axTi_f.set_ylim([0, Qmax])

            if axne_f is not None:
                Qmax = GeBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = GeBest_min
                Qmin -= np.abs(Qmin) * 0.5
                Q = np.max([np.abs(Qmin), np.abs(Qmax)])
                axne_f.set_ylim([-Q, Q])

            if axnZ_f is not None:
                Qmax = GZBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = GZBest_min
                Qmin -= np.abs(Qmin) * 0.5
                Q = np.max([np.abs(Qmin), np.abs(Qmax)])
                axnZ_f.set_ylim([-Q, Q])

            if axw0_f is not None:
                Qmax = MtBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = MtBest_min
                Qmin -= np.abs(Qmin) * 0.5
                Q = np.max([np.abs(Qmin), np.abs(Qmax)])
                axw0_f.set_ylim([-Q, Q])

    return (
        QeBest_min,
        QeBest_max,
        QiBest_min,
        QiBest_max,
        GeBest_min,
        GeBest_max,
        GZBest_min,
        GZBest_max,
        MtBest_min,
        MtBest_max,
    )
