import torch
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import PLASMAtools
from mitim_modules.powertorch import STATEtools
from mitim_modules.portals import PORTALStools
from mitim_tools.misc_tools.IOtools import printMsg as print


def parabolizePlasma(self):
    _, T = PLASMAtools.parabolicProfile(
        Tbar=self.derived["Te_vol"],
        nu=self.derived["Te_peaking"],
        rho=self.profiles["rho(-)"],
        Tedge=self.profiles["te(keV)"][-1],
    )
    _, Ti = PLASMAtools.parabolicProfile(
        Tbar=self.derived["Ti_vol"],
        nu=self.derived["Ti_peaking"],
        rho=self.profiles["rho(-)"],
        Tedge=self.profiles["ti(keV)"][-1, 0],
    )
    _, n = PLASMAtools.parabolicProfile(
        Tbar=self.derived["ne_vol20"] * 1e1,
        nu=self.derived["ne_peaking"],
        rho=self.profiles["rho(-)"],
        Tedge=self.profiles["ne(10^19/m^3)"][-1],
    )

    self.profiles["te(keV)"] = T

    self.profiles["ti(keV)"][:, 0] = Ti
    self.makeAllThermalIonsHaveSameTemp(refIon=0)

    factor_n = n / self.profiles["ne(10^19/m^3)"]
    self.profiles["ne(10^19/m^3)"] = n
    self.scaleAllThermalDensities(scaleFactor=factor_n)

    self.deriveQuantities()


def changeRFpower(self, PrfMW=25.0):
    """
    keeps same partition
    """
    print(
        f"- Changing the RF power from {self.derived['qRF_MWmiller'][-1]:.1f} MW to {PrfMW:.1f} MW",
        typeMsg="i",
    )
    for i in ["qrfe(MW/m^3)", "qrfi(MW/m^3)"]:
        self.profiles[i] = self.profiles[i] * PrfMW / self.derived["qRF_MWmiller"][-1]


def imposeBCtemps(self, TkeV=0.5, rho=0.9, typeEdge="linear", Tesep=0.1, Tisep=0.2):
    ix = np.argmin(np.abs(rho - self.profiles["rho(-)"]))

    self.profiles["te(keV)"] = (
        self.profiles["te(keV)"] * TkeV / self.profiles["te(keV)"][ix]
    )

    print(
        f"- Producing {typeEdge} boundary condition @ rho = {rho}, T = {TkeV} keV",
        typeMsg="i",
    )

    for sp in range(len(self.Species)):
        if self.Species[sp]["S"] == "therm":
            self.profiles["ti(keV)"][:, sp] = (
                self.profiles["ti(keV)"][:, sp]
                * TkeV
                / self.profiles["ti(keV)"][ix, sp]
            )

    if typeEdge == "linear":
        self.profiles["te(keV)"][ix:] = np.linspace(
            TkeV, Tesep, len(self.profiles["rho(-)"][ix:])
        )

        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "therm":
                self.profiles["ti(keV)"][ix:, sp] = np.linspace(
                    TkeV, Tisep, len(self.profiles["rho(-)"][ix:])
                )

    elif typeEdge == "same":
        pass
    else:
        raise Exception("no edge")


def imposeBCdens(self, n20=2.0, rho=0.9, typeEdge="linear", nedge20=0.5):
    ix = np.argmin(np.abs(rho - self.profiles["rho(-)"]))

    print(
        f"- Changing the initial average density from {self.derived['ne_vol20']:.1f} 1E20/m3 to {n20:.1f} 1E20/m3",
        typeMsg="i",
    )

    factor = n20 / self.derived["ne_vol20"]

    for i in ["ne(10^19/m^3)", "ni(10^19/m^3)"]:
        self.profiles[i] = self.profiles[i] * factor

    if typeEdge == "linear":
        factor_x = (
            np.linspace(
                self.profiles["ne(10^19/m^3)"][ix],
                nedge20 * 1e1,
                len(self.profiles["rho(-)"][ix:]),
            )
            / self.profiles["ne(10^19/m^3)"][ix:]
        )

        self.profiles["ne(10^19/m^3)"][ix:] = (
            self.profiles["ne(10^19/m^3)"][ix:] * factor_x
        )
        for i in range(self.profiles["ni(10^19/m^3)"].shape[1]):
            self.profiles["ni(10^19/m^3)"][ix:, i] = (
                self.profiles["ni(10^19/m^3)"][ix:, i] * factor_x
            )
    elif typeEdge == "same":
        pass
    else:
        raise Exception("no edge")


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# This is where the definitions for the summation variables happen for mitim and PORTALSplot
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def TGYROmodeledVariables(
    self,
    useConvectiveFluxes=False,
    forceZeroParticleFlux=False,
    includeFast=False,
    impurityPosition=1,
    UseFineGridTargets=False,
    OriginalFimp=1.0,
    dfT=torch.Tensor(),
):
    '''
    impurityPosition will be substracted one
    '''
    if "tgyro_stds" not in self.__dict__:
        self.tgyro_stds = False

    if UseFineGridTargets:
        self.useFineGridTargets(impurityPosition=impurityPosition)

    portals_variables = {}

    # **********************************
    # *********** Electron Energy Fluxes
    # **********************************

    portals_variables["Qe_turb"] = self.Qe_sim_turb[:, :]
    portals_variables["Qe_neo"] = self.Qe_sim_neo[:, :]
    portals_variables["Qe"] = self.Qe_tar[:, :]

    portals_variables["Qe_turb_stds"] = (
        self.Qe_sim_turb_stds if self.tgyro_stds else None
    )
    portals_variables["Qe_neo_stds"] = self.Qe_sim_neo_stds if self.tgyro_stds else None
    portals_variables["Qe_stds"] = self.Qe_tar_stds if self.tgyro_stds else None

    # **********************************
    # *********** Ion Energy Fluxes
    # **********************************

    if includeFast:
        portals_variables["Qi_turb"] = self.QiIons_sim_turb[:, :]
        portals_variables["Qi_neo"] = self.QiIons_sim_neo[:, :]

        portals_variables["Qi_turb_stds"] = (
            self.QiIons_sim_turb_stds if self.tgyro_stds else None
        )
        portals_variables["Qi_neo_stds"] = (
            self.QiIons_sim_neo_stds if self.tgyro_stds else None
        )

    else:
        portals_variables["Qi_turb"] = self.QiIons_sim_turb_thr[:, :]
        portals_variables["Qi_neo"] = self.QiIons_sim_neo_thr[:, :]

        portals_variables["Qi_turb_stds"] = (
            self.QiIons_sim_turb_thr_stds if self.tgyro_stds else None
        )
        portals_variables["Qi_neo_stds"] = (
            self.QiIons_sim_neo_thr_stds if self.tgyro_stds else None
        )

    portals_variables["Qi"] = self.Qi_tar[:, :]
    portals_variables["Qi_stds"] = self.Qi_tar_stds[:, :] if self.tgyro_stds else None

    # **********************************
    # *********** Momentum Fluxes
    # **********************************

    portals_variables["Mt_turb"] = self.Mt_sim_turb[
        :, :
    ]  # So far, let's include fast in momentum
    portals_variables["Mt_neo"] = self.Mt_sim_neo[:, :]
    portals_variables["Mt"] = self.Mt_tar[:, :]

    portals_variables["Mt_turb_stds"] = (
        self.Mt_sim_turb_stds if self.tgyro_stds else None
    )
    portals_variables["Mt_neo_stds"] = self.Mt_sim_neo_stds if self.tgyro_stds else None
    portals_variables["Mt_stds"] = self.Mt_tar_stds[:, :] if self.tgyro_stds else None

    # **********************************
    # *********** Particle Fluxes
    # **********************************

    if not useConvectiveFluxes:
        portals_variables["Ge_turb"] = self.Ge_sim_turb[:, :]
        portals_variables["Ge_neo"] = self.Ge_sim_neo[:, :]
        portals_variables["Ge"] = self.Ge_tar[:, :]

        portals_variables["Ge_turb_stds"] = (
            self.Ge_sim_turb_stds if self.tgyro_stds else None
        )
        portals_variables["Ge_neo_stds"] = (
            self.Ge_sim_neo_stds if self.tgyro_stds else None
        )
        portals_variables["Ge_stds"] = (
            self.Ge_tar_stds[:, :] if self.tgyro_stds else None
        )

    else:
        portals_variables["Ge_turb"] = self.Ce_sim_turb[:, :]
        portals_variables["Ge_neo"] = self.Ce_sim_neo[:, :]
        portals_variables["Ge"] = self.Ce_tar[:, :]

        portals_variables["Ge_turb_stds"] = (
            self.Ce_sim_turb_stds if self.tgyro_stds else None
        )
        portals_variables["Ge_neo_stds"] = (
            self.Ce_sim_neo_stds if self.tgyro_stds else None
        )
        portals_variables["Ge_stds"] = (
            self.Ce_tar_stds[:, :] if self.tgyro_stds else None
        )

    # **********************************
    # *********** Impurity Fluxes
    # **********************************

    if not useConvectiveFluxes:
        portals_variables["GZ_turb"] = (
            self.Gi_sim_turb[impurityPosition - 1, :, :] / OriginalFimp
        )
        portals_variables["GZ_neo"] = (
            self.Gi_sim_neo[impurityPosition - 1, :, :] / OriginalFimp
        )
        portals_variables["GZ"] = self.Gi_tar[impurityPosition - 1, :, :] / OriginalFimp

        portals_variables["GZ_turb_stds"] = (
            self.Gi_sim_turb_stds[impurityPosition - 1, :, :] / OriginalFimp
            if self.tgyro_stds
            else None
        )
        portals_variables["GZ_neo_stds"] = (
            self.Gi_sim_neo_stds[impurityPosition - 1, :, :] / OriginalFimp
            if self.tgyro_stds
            else None
        )
        portals_variables["GZ_stds"] = (
            self.Gi_tar_stds[impurityPosition - 1, :, :] / OriginalFimp
            if self.tgyro_stds
            else None
        )

    else:
        portals_variables["GZ_neo"] = (
            self.Ci_sim_neo[impurityPosition - 1, :, :] / OriginalFimp
        )
        portals_variables["GZ_turb"] = (
            self.Ci_sim_turb[impurityPosition - 1, :, :] / OriginalFimp
        )
        portals_variables["GZ"] = self.Ci_tar[impurityPosition - 1, :, :] / OriginalFimp

        portals_variables["GZ_turb_stds"] = (
            self.Ci_sim_turb_stds[impurityPosition - 1, :, :] / OriginalFimp
            if self.tgyro_stds
            else None
        )
        portals_variables["GZ_neo_stds"] = (
            self.Ci_sim_neo_stds[impurityPosition - 1, :, :] / OriginalFimp
            if self.tgyro_stds
            else None
        )
        portals_variables["GZ_stds"] = (
            self.Ci_tar_stds[impurityPosition - 1, :, :] / OriginalFimp
            if self.tgyro_stds
            else None
        )

    # **********************************
    # *********** Energy Exchange
    # **********************************

    portals_variables["PexchTurb"] = self.EXe_sim_turb[:, :]  # MW/m^3
    portals_variables["PexchTurb_stds"] = (
        self.EXe_sim_turb_stds[:, :] if self.tgyro_stds else None
    )

    # portals_variables['PexchTurbFlux'] 	= self.Qe_tarMW_expwd/self.dvoldr 		# MW/m^2

    if forceZeroParticleFlux:
        portals_variables["Ge"] = self.Ge_tar[:, :] * 0.0

    # ----------------------------------------------------------------------------------------
    # Prepare dictionary that is equal to what portals pseudo does in PORTALSmain (calculatePseudos)
    # ----------------------------------------------------------------------------------------

    portals_variables["var_dict"] = {}

    mapper = {
        "QeTurb": "Qe_turb",
        "QiTurb": "Qi_turb",
        "GeTurb": "Ge_turb",
        "GZTurb": "GZ_turb",
        "MtTurb": "Mt_turb",
        "QeNeo": "Qe_neo",
        "QiNeo": "Qi_neo",
        "GeNeo": "Ge_neo",
        "GZNeo": "GZ_neo",
        "MtNeo": "Mt_neo",
        "QeTar": "Qe",
        "QiTar": "Qi",
        "GeTar": "Ge",
        "GZTar": "GZ",
        "MtTar": "Mt",
        "PexchTurb": "PexchTurb",
    }

    for ikey in mapper:
        portals_variables["var_dict"][ikey] = torch.Tensor(
            portals_variables[mapper[ikey]]
        ).to(dfT)[:, 1:]
        if self.tgyro_stds:
            portals_variables["var_dict"][ikey + "_stds"] = torch.Tensor(
                portals_variables[mapper[ikey] + "_stds"]
            ).to(dfT)[:, 1:]
        else:
            portals_variables["var_dict"][ikey + "_stds"] = None

    # ----------------------------------------------------------------------------------------
    # labels for plotting
    # ----------------------------------------------------------------------------------------

    portals_variables["labels"] = {
        "te": "$Q_e$ ($MW/m^2$)",
        "ti": "$Q_i$ ($MW/m^2$)",
        "ne": "$Q_{conv}$ ($MW/m^2$)"
        if useConvectiveFluxes
        else "$\\Gamma_e$ ($10^{20}/s/m^2$)",
        "nZ": "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)"
        if useConvectiveFluxes
        else "$\\Gamma_Z$ $\\cdot f_{Z,0}$ ($10^{20}/s/m^2$)",
        "w0": "$M_T$ ($J/m^2$)",
    }

    return portals_variables


def calculatePseudos(var_dict, PORTALSparameters, TGYROparameters, powerstate):
    """
    Notes
    -----
            - Works with tensors
            - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    dfT = var_dict["QeTurb"]  # as a reference for sizes

    # -------------------------------------------------------------------------
    # Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    # -------------------------------------------------------------------------

    if PORTALSparameters["surrogateForTurbExch"]:
        PexchTurb_integrated = PORTALStools.computeTurbExchangeIndividual(
            var_dict["PexchTurb"], powerstate
        )
    else:
        PexchTurb_integrated = torch.zeros(dfT.shape).to(dfT)

    # ------------------------------------------------------------------------
    # Go through each profile that needs to be predicted, calculate components
    # ------------------------------------------------------------------------

    of, cal, res = (
        torch.Tensor().to(dfT),
        torch.Tensor().to(dfT),
        torch.Tensor().to(dfT),
    )
    for prof in TGYROparameters["ProfilesPredicted"]:
        if prof == "te":
            var = "Qe"
        elif prof == "ti":
            var = "Qi"
        elif prof == "ne":
            var = "Ge"
        elif prof == "nZ":
            var = "GZ"
        elif prof == "w0":
            var = "Mt"

        """
		-----------------------------------------------------------------------------------
		Transport (Turb+Neo)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}Turb"] + var_dict[f"{var}Neo"]

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}Tar"] + PexchTurb_integrated
        elif var == "Qi":
            cal0 = var_dict[f"{var}Tar"] - PexchTurb_integrated
        else:
            cal0 = var_dict[f"{var}Tar"]

        """
		-----------------------------------------------------------------------------------
		Ad-hoc modifications for different weighting
		-----------------------------------------------------------------------------------
		"""

        if var == "Qe":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][0],
                cal0 * PORTALSparameters["Pseudo_multipliers"][0],
            )
        elif var == "Qi":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][1],
                cal0 * PORTALSparameters["Pseudo_multipliers"][1],
            )
        elif var == "Ge":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][2],
                cal0 * PORTALSparameters["Pseudo_multipliers"][2],
            )
        elif var == "GZ":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][3],
                cal0 * PORTALSparameters["Pseudo_multipliers"][3],
            )
        elif var == "Mt":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][4],
                cal0 * PORTALSparameters["Pseudo_multipliers"][4],
            )

        of, cal = torch.cat((of, of0), dim=-1), torch.cat((cal, cal0), dim=-1)

    # -----------
    # Composition
    # -----------

    # Source term is (TARGET - TRANSPORT)
    source = cal - of

    # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L2
    res = -1 / source.shape[-1] * torch.norm(source, p=2, dim=-1)

    return of, cal, source, res


def calculatePseudos_distributions(
    var_dict, PORTALSparameters, TGYROparameters, powerstate
):
    """
    Notes
    -----
            - Works with tensors
            - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    dfT = var_dict["QeTurb"]  # as a reference for sizes

    # -------------------------------------------------------------------------
    # Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    # -------------------------------------------------------------------------

    if PORTALSparameters["surrogateForTurbExch"]:
        PexchTurb_integrated = PORTALStools.computeTurbExchangeIndividual(
            var_dict["PexchTurb"], powerstate
        )
        PexchTurb_integrated_stds = PORTALStools.computeTurbExchangeIndividual(
            var_dict["PexchTurb_stds"], powerstate
        )
    else:
        PexchTurb_integrated = torch.zeros(dfT.shape).to(dfT)
        PexchTurb_integrated_stds = torch.zeros(dfT.shape).to(dfT)

    # ------------------------------------------------------------------------
    # Go through each profile that needs to be predicted, calculate components
    # ------------------------------------------------------------------------

    of, cal = torch.Tensor().to(dfT), torch.Tensor().to(dfT)
    ofE, calE = torch.Tensor().to(dfT), torch.Tensor().to(dfT)
    for prof in TGYROparameters["ProfilesPredicted"]:
        if prof == "te":
            var = "Qe"
        elif prof == "ti":
            var = "Qi"
        elif prof == "ne":
            var = "Ge"
        elif prof == "nZ":
            var = "GZ"
        elif prof == "w0":
            var = "Mt"

        """
		-----------------------------------------------------------------------------------
		Transport (Turb+Neo)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}Turb"] + var_dict[f"{var}Neo"]
        of0E = (
            var_dict[f"{var}Turb_stds"] ** 2 + var_dict[f"{var}Neo_stds"] ** 2
        ) ** 0.5

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}Tar"] + PexchTurb_integrated
            cal0E = (
                var_dict[f"{var}Tar_stds"] ** 2 + PexchTurb_integrated_stds**2
            ) ** 0.5
        elif var == "Qi":
            cal0 = var_dict[f"{var}Tar"] - PexchTurb_integrated
            cal0E = (
                var_dict[f"{var}Tar_stds"] ** 2 + PexchTurb_integrated_stds**2
            ) ** 0.5
        else:
            cal0 = var_dict[f"{var}Tar"]
            cal0E = var_dict[f"{var}Tar_stds"]

        of, cal = torch.cat((of, of0), dim=-1), torch.cat((cal, cal0), dim=-1)
        ofE, calE = torch.cat((ofE, of0E), dim=-1), torch.cat((calE, cal0E), dim=-1)

    return of, cal, ofE, calE
