import torch
import numpy as np
from mitim_tools.misc_tools import PLASMAtools
from mitim_modules.portals import PORTALStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

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
    print(f"- Changing the RF power from {self.derived['qRF_MWmiller'][-1]:.1f} MW to {PrfMW:.1f} MW",typeMsg="i",)
    
    if self.derived["qRF_MWmiller"][-1] == 0.0:
        raise Exception("No RF power in the input.gacode, cannot modify the RF power")

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

def TGYROmodeledVariables(TGYROresults,
    powerstate,
    useConvectiveFluxes=False,
    forceZeroParticleFlux=False,
    includeFast=False,
    impurityPosition=1,
    UseFineGridTargets=False,
    OriginalFimp=1.0,
    provideTurbulentExchange=False,
    provideTargets=False
    ):
    """
    This function is used to extract the TGYRO results and store them in the powerstate object, from numpy arrays to torch tensors.
    """

    if "tgyro_stds" not in TGYROresults.__dict__:
        TGYROresults.tgyro_stds = False

    if UseFineGridTargets:
        TGYROresults.useFineGridTargets(impurityPosition=impurityPosition)

    nr = powerstate.plasma['rho'].shape[-1]
    if powerstate.plasma['rho'].shape[-1] != TGYROresults.rho.shape[-1]:
        print('\t- TGYRO was run with an extra point in the grid, treating it carefully now')

    # **********************************
    # *********** Electron Energy Fluxes
    # **********************************

    powerstate.plasma["QeMWm2_tr_turb"] = torch.Tensor(TGYROresults.Qe_sim_turb[:, :nr]).to(powerstate.dfT)
    powerstate.plasma["QeMWm2_tr_neo"] = torch.Tensor(TGYROresults.Qe_sim_neo[:, :nr]).to(powerstate.dfT)

    powerstate.plasma["QeMWm2_tr_turb_stds"] = torch.Tensor(TGYROresults.Qe_sim_turb_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    powerstate.plasma["QeMWm2_tr_neo_stds"] = torch.Tensor(TGYROresults.Qe_sim_neo_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    
    if provideTargets:
        powerstate.plasma["QeMWm2"] = torch.Tensor(TGYROresults.Qe_tar[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["QeMWm2_stds"] = torch.Tensor(TGYROresults.Qe_tar_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    # **********************************
    # *********** Ion Energy Fluxes
    # **********************************

    if includeFast:

        powerstate.plasma["QiMWm2_tr_turb"] = torch.Tensor(TGYROresults.QiIons_sim_turb[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["QiMWm2_tr_neo"] = torch.Tensor(TGYROresults.QiIons_sim_neo[:, :nr]).to(powerstate.dfT)
        
        powerstate.plasma["QiMWm2_tr_turb_stds"] = torch.Tensor(TGYROresults.QiIons_sim_turb_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
        powerstate.plasma["QiMWm2_tr_neo_stds"] = torch.Tensor(TGYROresults.QiIons_sim_neo_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    else:

        powerstate.plasma["QiMWm2_tr_turb"] = torch.Tensor(TGYROresults.QiIons_sim_turb_thr[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["QiMWm2_tr_neo"] = torch.Tensor(TGYROresults.QiIons_sim_neo_thr[:, :nr]).to(powerstate.dfT)

        powerstate.plasma["QiMWm2_tr_turb_stds"] = torch.Tensor(TGYROresults.QiIons_sim_turb_thr_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
        powerstate.plasma["QiMWm2_tr_neo_stds"] = torch.Tensor(TGYROresults.QiIons_sim_neo_thr_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    if provideTargets:
        powerstate.plasma["QiMWm2"] = torch.Tensor(TGYROresults.Qi_tar[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["QiMWm2_stds"] = torch.Tensor(TGYROresults.Qi_tar_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    # **********************************
    # *********** Momentum Fluxes
    # **********************************

    powerstate.plasma["Mt_tr_turb"] = torch.Tensor(TGYROresults.Mt_sim_turb[:, :nr]).to(powerstate.dfT) # So far, let's include fast in momentum
    powerstate.plasma["Mt_tr_neo"] = torch.Tensor(TGYROresults.Mt_sim_neo[:, :nr]).to(powerstate.dfT)

    powerstate.plasma["Mt_tr_turb_stds"] = torch.Tensor(TGYROresults.Mt_sim_turb_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    powerstate.plasma["Mt_tr_neo_stds"] = torch.Tensor(TGYROresults.Mt_sim_neo_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    
    if provideTargets:
        powerstate.plasma["Mt"] = torch.Tensor(TGYROresults.Mt_tar[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["Mt_stds"] = torch.Tensor(TGYROresults.Mt_tar_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    # **********************************
    # *********** Particle Fluxes
    # **********************************

    # Store raw fluxes for better plotting later
    powerstate.plasma["Ce_raw_tr_turb"] = torch.Tensor(TGYROresults.Ge_sim_turb[:, :nr]).to(powerstate.dfT)
    powerstate.plasma["Ce_raw_tr_neo"] = torch.Tensor(TGYROresults.Ge_sim_neo[:, :nr]).to(powerstate.dfT)

    powerstate.plasma["Ce_raw_tr_turb_stds"] = torch.Tensor(TGYROresults.Ge_sim_turb_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    powerstate.plasma["Ce_raw_tr_neo_stds"] = torch.Tensor(TGYROresults.Ge_sim_neo_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    
    if provideTargets:
        powerstate.plasma["Ce_raw"] = torch.Tensor(TGYROresults.Ge_tar[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["Ce_raw_stds"] = torch.Tensor(TGYROresults.Ge_tar_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    if not useConvectiveFluxes:

        powerstate.plasma["Ce_tr_turb"] = powerstate.plasma["Ce_raw_tr_turb"]
        powerstate.plasma["Ce_tr_neo"] = powerstate.plasma["Ce_raw_tr_neo"]

        powerstate.plasma["Ce_tr_turb_stds"] = powerstate.plasma["Ce_raw_tr_turb_stds"]
        powerstate.plasma["Ce_tr_neo_stds"] = powerstate.plasma["Ce_raw_tr_neo_stds"]
        
        if provideTargets:
            powerstate.plasma["Ce"] = powerstate.plasma["Ce_raw"]
            powerstate.plasma["Ce_stds"] = powerstate.plasma["Ce_raw_stds"]    

    else:

        powerstate.plasma["Ce_tr_turb"] = torch.Tensor(TGYROresults.Ce_sim_turb[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["Ce_tr_neo"] = torch.Tensor(TGYROresults.Ce_sim_neo[:, :nr]).to(powerstate.dfT)

        powerstate.plasma["Ce_tr_turb_stds"] = torch.Tensor(TGYROresults.Ce_sim_turb_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
        powerstate.plasma["Ce_tr_neo_stds"] = torch.Tensor(TGYROresults.Ce_sim_neo_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
        
        if provideTargets:
            powerstate.plasma["Ce"] = torch.Tensor(TGYROresults.Ce_tar[:, :nr]).to(powerstate.dfT)
            powerstate.plasma["Ce_stds"] = torch.Tensor(TGYROresults.Ce_tar_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None

    # **********************************
    # *********** Impurity Fluxes
    # **********************************

    # Store raw fluxes for better plotting later
    powerstate.plasma["CZ_raw_tr_turb"] = torch.Tensor(TGYROresults.Gi_sim_turb[impurityPosition, :, :nr]).to(powerstate.dfT) 
    powerstate.plasma["CZ_raw_tr_neo"] = torch.Tensor(TGYROresults.Gi_sim_neo[impurityPosition, :, :nr]).to(powerstate.dfT) 
    
    powerstate.plasma["CZ_raw_tr_turb_stds"] = torch.Tensor(TGYROresults.Gi_sim_turb_stds[impurityPosition, :, :nr]).to(powerstate.dfT)  if TGYROresults.tgyro_stds else None
    powerstate.plasma["CZ_raw_tr_neo_stds"] = torch.Tensor(TGYROresults.Gi_sim_neo_stds[impurityPosition, :, :nr]).to(powerstate.dfT)  if TGYROresults.tgyro_stds else None

    if provideTargets:
        powerstate.plasma["CZ_raw"] = torch.Tensor(TGYROresults.Gi_tar[impurityPosition, :, :nr]).to(powerstate.dfT) 
        powerstate.plasma["CZ_raw_stds"] = torch.Tensor(TGYROresults.Gi_tar_stds[impurityPosition, :, :nr]).to(powerstate.dfT)  if TGYROresults.tgyro_stds else None

    if not useConvectiveFluxes:

        powerstate.plasma["CZ_tr_turb"] = powerstate.plasma["CZ_raw_tr_turb"] / OriginalFimp
        powerstate.plasma["CZ_tr_neo"] = powerstate.plasma["CZ_raw_tr_neo"] / OriginalFimp
        
        powerstate.plasma["CZ_tr_turb_stds"] = powerstate.plasma["CZ_raw_tr_turb_stds"] / OriginalFimp if TGYROresults.tgyro_stds else None
        powerstate.plasma["CZ_tr_neo_stds"] = powerstate.plasma["CZ_raw_tr_neo_stds"] / OriginalFimp if TGYROresults.tgyro_stds else None
        
        if provideTargets:
            powerstate.plasma["CZ"] = powerstate.plasma["CZ_raw"] / OriginalFimp
            powerstate.plasma["CZ_stds"] = powerstate.plasma["CZ_raw_stds"] / OriginalFimp if TGYROresults.tgyro_stds else None

    else:

        powerstate.plasma["CZ_tr_turb"] = torch.Tensor(TGYROresults.Ci_sim_turb[impurityPosition, :, :nr]).to(powerstate.dfT) / OriginalFimp
        powerstate.plasma["CZ_tr_neo"] = torch.Tensor(TGYROresults.Ci_sim_neo[impurityPosition, :, :nr]).to(powerstate.dfT) / OriginalFimp

        powerstate.plasma["CZ_tr_turb_stds"] = torch.Tensor(TGYROresults.Ci_sim_turb_stds[impurityPosition, :, :nr]).to(powerstate.dfT) / OriginalFimp if TGYROresults.tgyro_stds else None
        powerstate.plasma["CZ_tr_neo_stds"] = torch.Tensor(TGYROresults.Ci_sim_neo_stds[impurityPosition, :, :nr]).to(powerstate.dfT) / OriginalFimp if TGYROresults.tgyro_stds else None
        
        if provideTargets:
            powerstate.plasma["CZ"] = torch.Tensor(TGYROresults.Ci_tar[impurityPosition, :, :nr]).to(powerstate.dfT) / OriginalFimp
            powerstate.plasma["CZ_stds"] = torch.Tensor(TGYROresults.Ci_tar_stds[impurityPosition, :, :nr]).to(powerstate.dfT) / OriginalFimp if TGYROresults.tgyro_stds else None

    # **********************************
    # *********** Energy Exchange
    # **********************************

    if provideTurbulentExchange:
        powerstate.plasma["PexchTurb"] = torch.Tensor(TGYROresults.EXe_sim_turb[:, :nr]).to(powerstate.dfT)
        powerstate.plasma["PexchTurb_stds"] = torch.Tensor(TGYROresults.EXe_sim_turb_stds[:, :nr]).to(powerstate.dfT) if TGYROresults.tgyro_stds else None
    else:
        powerstate.plasma["PexchTurb"] = powerstate.plasma["QeMWm2_tr_turb"] * 0.0
        powerstate.plasma["PexchTurb_stds"] = powerstate.plasma["QeMWm2_tr_turb"] * 0.0

    # **********************************
    # *********** Traget extra
    # **********************************

    if forceZeroParticleFlux and provideTargets:
        powerstate.plasma["Ce"] = powerstate.plasma["Ce"] * 0.0
        powerstate.plasma["Ce_stds"] = powerstate.plasma["Ce_stds"] * 0.0

    # ------------------------------------------------------------------------------------------------------------------------
    # Sum here turbulence and neoclassical, after modifications
    # ------------------------------------------------------------------------------------------------------------------------

    quantities = ['QeMWm2', 'QiMWm2', 'Ce', 'CZ', 'Mt', 'Ce_raw', 'CZ_raw']
    for ikey in quantities:
        powerstate.plasma[ikey+"_tr"] = powerstate.plasma[ikey+"_tr_turb"] + powerstate.plasma[ikey+"_tr_neo"]
    
    return powerstate


def calculate_residuals(powerstate, PORTALSparameters, specific_vars=None):
    """
    Notes
    -----
        - Works with tensors
        - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    # Case where I have already constructed the dictionary (i.e. in scalarized objective)
    if specific_vars is not None:
        var_dict = specific_vars
    # Prepare dictionary from powerstate (for use in Analysis)
    else:
        var_dict = {}

        mapper = {
            "Qe_tr_turb": "QeMWm2_tr_turb",
            "Qi_tr_turb": "QiMWm2_tr_turb",
            "Ge_tr_turb": "Ce_tr_turb",
            "GZ_tr_turb": "CZ_tr_turb",
            "Mt_tr_turb": "Mt_tr_turb",
            "Qe_tr_neo": "QeMWm2_tr_neo",
            "Qi_tr_neo": "QiMWm2_tr_neo",
            "Ge_tr_neo": "Ce_tr_neo",
            "GZ_tr_neo": "CZ_tr_neo",
            "Mt_tr_neo": "Mt_tr_neo",
            "Qe_tar": "QeMWm2",
            "Qi_tar": "QiMWm2",
            "Ge_tar": "Ce",
            "GZ_tar": "CZ",
            "Mt_tar": "Mt",
            "PexchTurb": "PexchTurb"
        }

        for ikey in mapper:
            var_dict[ikey] = powerstate.plasma[mapper[ikey]][..., 1:]
            if mapper[ikey] + "_stds" in powerstate.plasma:
                var_dict[ikey + "_stds"] = powerstate.plasma[mapper[ikey] + "_stds"][..., 1:]
            else:
                var_dict[ikey + "_stds"] = None

    dfT = list(var_dict.values())[0]  # as a reference for sizes

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
    for prof in powerstate.ProfilesPredicted:
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
		Transport (_tr_turb+_tr_neo)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}_tr_turb"] + var_dict[f"{var}_tr_neo"]

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}_tar"] + PexchTurb_integrated
        elif var == "Qi":
            cal0 = var_dict[f"{var}_tar"] - PexchTurb_integrated
        else:
            cal0 = var_dict[f"{var}_tar"]

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


def calculate_residuals_distributions(powerstate, PORTALSparameters):
    """
    - Works with tensors
    - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    # Prepare dictionary from powerstate (for use in Analysis)
    
    mapper = {
        "Qe_tr_turb": "QeMWm2_tr_turb",
        "Qi_tr_turb": "QiMWm2_tr_turb",
        "Ge_tr_turb": "Ce_tr_turb",
        "GZ_tr_turb": "CZ_tr_turb",
        "Mt_tr_turb": "Mt_tr_turb",
        "Qe_tr_neo": "QeMWm2_tr_neo",
        "Qi_tr_neo": "QiMWm2_tr_neo",
        "Ge_tr_neo": "Ce_tr_neo",
        "GZ_tr_neo": "CZ_tr_neo",
        "Mt_tr_neo": "Mt_tr_neo",
        "Qe_tar": "QeMWm2",
        "Qi_tar": "QiMWm2",
        "Ge_tar": "Ce",
        "GZ_tar": "CZ",
        "Mt_tar": "Mt",
        "PexchTurb": "PexchTurb"
    }

    var_dict = {}
    for ikey in mapper:
        var_dict[ikey] = powerstate.plasma[mapper[ikey]][:, 1:]
        if mapper[ikey] + "_stds" in powerstate.plasma:
            var_dict[ikey + "_stds"] = powerstate.plasma[mapper[ikey] + "_stds"][:, 1:]
        else:
            var_dict[ikey + "_stds"] = None

    dfT = var_dict["Qe_tr_turb"]  # as a reference for sizes

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
    for prof in powerstate.ProfilesPredicted:
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
		Transport (_tr_turb+_tr_neo)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}_tr_turb"] + var_dict[f"{var}_tr_neo"]
        of0E = (
            var_dict[f"{var}_tr_turb_stds"] ** 2 + var_dict[f"{var}_tr_neo_stds"] ** 2
        ) ** 0.5

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}_tar"] + PexchTurb_integrated
            cal0E = (
                var_dict[f"{var}_tar_stds"] ** 2 + PexchTurb_integrated_stds**2
            ) ** 0.5
        elif var == "Qi":
            cal0 = var_dict[f"{var}_tar"] - PexchTurb_integrated
            cal0E = (
                var_dict[f"{var}_tar_stds"] ** 2 + PexchTurb_integrated_stds**2
            ) ** 0.5
        else:
            cal0 = var_dict[f"{var}_tar"]
            cal0E = var_dict[f"{var}_tar_stds"]

        of, cal = torch.cat((of, of0), dim=-1), torch.cat((cal, cal0), dim=-1)
        ofE, calE = torch.cat((ofE, of0E), dim=-1), torch.cat((calE, cal0E), dim=-1)

    return of, cal, ofE, calE
