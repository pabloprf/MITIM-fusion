import copy
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools, MATHtools, PLASMAtools, IOtools
from mitim_modules.powertorch.physics import GEOMETRYtools, CALCtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.gacode_tools import NEOtools
from mitim_tools.gacode_tools.aux import TRANSPinteraction, PROFILEStoMODELS
from mitim_tools.transp_tools import CDFtools
from mitim_tools.im_tools.modules import PEDmodule
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

from mitim_tools.misc_tools.IOtools import printMsg as print

verbose_level = read_verbose_level()

try:
    from mitim_tools.gacode_tools.aux import PORTALSinteraction
except ImportError:
    print(
        "- I could not import PORTALSinteraction, likely a consequence of botorch incompatbility",
        typeMsg="w",
    )

# -------------------------------------------------------------------------------------
# 		input.gacode
# -------------------------------------------------------------------------------------


class PROFILES_GACODE:
    def __init__(self, file, calculateDerived=True, mi_ref=None):
        """
        Depending on resolution, derived can be expensive, so I mmay not do it every time
        """

        self.titles_singleNum = ["nexp", "nion", "shot", "name", "type", "time"]
        self.titles_singleArr = [
            "masse",
            "mass",
            "ze",
            "z",
            "torfluxa(Wb/radian)",
            "rcentr(m)",
            "bcentr(T)",
            "current(MA)",
        ]
        self.titles_single = self.titles_singleNum + self.titles_singleArr

        self.file = file
        with open(self.file, "r") as f:
            self.lines = f.readlines()

        # Read file and store raw data
        self.readHeader()
        self.readProfiles()
        self.readSpecies()

        """
		Perform MITIM derivations (can be expensive, only if requested)
			Note: One can force what mi_ref to use (in a.m.u.). This is because, by default, MITIM
				  will use the mass of the first thermal ion to produce quantities such as Q_GB, rho_s, etc.
				  However, in some ocasions (like when running TGLF), the normalization that must be used
				  for those quantities is a fixed one (e.g. Deuterium)
		"""
        self.mi_ref = mi_ref
        if mi_ref is not None:
            print(
                "\t* Reference mass ({0:.2f}) to use was forced by class initialization".format(
                    mi_ref
                ),
                typeMsg="w",
            )

        # Useful to have gradients in the basic ----------------------------------------------------------
        if "derived" not in self.__dict__:
            self.derived = {}
        self.derived["aLTe"] = aLT(self.profiles["rmin(m)"], self.profiles["te(keV)"])
        self.derived["aLne"] = aLT(
            self.profiles["rmin(m)"], self.profiles["ne(10^19/m^3)"]
        )

        self.derived["aLTi"] = self.profiles["ti(keV)"] * 0.0
        self.derived["aLni"] = []
        for i in range(self.profiles["ti(keV)"].shape[1]):
            self.derived["aLTi"][:, i] = aLT(
                self.profiles["rmin(m)"], self.profiles["ti(keV)"][:, i]
            )
            self.derived["aLni"].append(
                aLT(self.profiles["rmin(m)"], self.profiles["ni(10^19/m^3)"][:, i])
            )
        self.derived["aLni"] = np.transpose(np.array(self.derived["aLni"]))
        # ------------------------------------------------------------------------------------------------

        if calculateDerived:
            self.deriveQuantities()

    # ***** Operations

    def runTRANSPfromGACODE(self, folderTRANSP, machine="SPARC"):
        # Prepare inputs
        TRANSPinteraction.prepareTRANSPfromGACODE(self, folderTRANSP, machine=machine)

        # Run TRANSP
        self.cdf_transp = TRANSPinteraction.runTRANSPfromGACODE(
            folderTRANSP, machine=machine
        )

    def calculate_Er(
        self,
        folder,
        rhos=None,
        vgenOptions={},
        name="vgen1",
        includeAll=False,
        write_new_file=None,
        restart=False,
    ):
        profiles = copy.deepcopy(self)

        # Resolution?
        resol_changed = False
        if rhos is not None:
            profiles.changeResolution(rho_new=rhos)
            resol_changed = True

        self.neo = NEOtools.NEO()
        self.neo.prep(profiles, folder)
        self.neo.run_vgen(subfolder=name, vgenOptions=vgenOptions, restart=restart)

        profiles_new = copy.deepcopy(self.neo.inputgacode_vgen)
        if resol_changed:
            profiles_new.changeResolution(rho_new=self.profiles["rho(-)"])

        # Get the information from the NEO run

        variables = ["w0(rad/s)"]
        if includeAll:
            variables += [
                "vpol(m/s)",
                "vtor(m/s)",
                "jbs(MA/m^2)",
                "jbstor(MA/m^2)",
                "johm(MA/m^2)",
            ]

        for ikey in variables:
            if ikey in profiles_new.profiles:
                print(
                    f'\t- Inserting {ikey} from NEO run{" (went back to original resolution by interpolation)" if resol_changed else ""}'
                )
                self.profiles[ikey] = profiles_new.profiles[ikey]

        self.deriveQuantities()

        if write_new_file is not None:
            self.writeCurrentStatus(file=write_new_file)

    # *****************

    def readHeader(self):
        for i in range(len(self.lines)):
            if "# nexp" in self.lines[i]:
                istartProfs = i
        self.header = self.lines[:istartProfs]

    def readProfiles(self):
        singleLine, title, var = None, None, None  # for ruff complaints

        # ---
        found = False
        self.profiles = OrderedDict()
        for i in range(len(self.lines)):
            if self.lines[i][0] == "#" and self.lines[i + 1][0] != "#":
                # previous
                if found and not singleLine:
                    self.profiles[title] = np.array(var)
                    if self.profiles[title].shape[1] == 1:
                        self.profiles[title] = self.profiles[title][:, 0]

                linebr = self.lines[i].split("#")[1].split("\n")[0].split()
                title_Orig = linebr[0]
                if len(linebr) > 1:
                    unit = self.lines[i].split("#")[1].split("\n")[0].split()[2]
                    title = title_Orig + f"({unit})"
                else:
                    title = title_Orig
                found, var = True, []

                if title in self.titles_single:
                    singleLine = True
                else:
                    singleLine = False
            elif found:
                var0 = self.lines[i].split()
                if singleLine:
                    if title in self.titles_singleArr:
                        self.profiles[title] = np.array([float(i) for i in var0])
                    else:
                        self.profiles[title] = np.array(var0)
                else:
                    # varT = [float(j) for j in var0[1:]]
                    """
                    Sometimes there's a bug in TGYRO, where the powers may be too low (E-191) that cannot be properly written
                    """
                    varT = [
                        float(j) if (j[-4].upper() == "E" or "." in j) else 0.0
                        for j in var0[1:]
                    ]

                    var.append(varT)

        # last
        if not singleLine:
            while len(var[-1]) < 1:
                var = var[:-1]  # Sometimes there's an extra space, remove
            self.profiles[title] = np.array(var)
            if self.profiles[title].shape[1] == 1:
                self.profiles[title] = self.profiles[title][:, 0]

        if "qpar_beam(MW/m^3)" in self.profiles:
            self.varqpar, self.varqpar2 = "qpar_beam(MW/m^3)", "qpar_wall(MW/m^3)"
        else:
            self.varqpar, self.varqpar2 = "qpar_beam(1/m^3/s)", "qpar_wall(1/m^3/s)"

        if "qmom(Nm)" in self.profiles:
            self.varqmom = (
                "qmom(Nm)"  # Old, wrong one. But Candy fixed it as of 02/24/2023
            )
        else:
            self.varqmom = "qmom(N/m^2)"  # CORRECT ONE

        # -------------------------------------------------------------------------------------------------------------------
        # Insert zeros in those cases whose column are not there
        # -------------------------------------------------------------------------------------------------------------------

        some_times_are_not_here = [
            "qei(MW/m^3)",
            "qohme(MW/m^3)",
            "johm(MA/m^2)",
            "jbs(MA/m^2)",
            "jbstor(MA/m^2)",
            "w0(rad/s)",
            "ptot(Pa)",  # e.g. if I haven't written that info from ASTRA
            "zeta(-)",  # e.g. if TGYRO is run with zeta=0, it won't write this column in .new
            "zmag(m)",
            self.varqpar,
            self.varqpar2,
            "shape_cos0(-)",
            self.varqmom,
        ]

        num_moments = 6  # This is the max number of moments I'll be considering. If I don't have that many (usually there are 5 or 3), it'll be populated with zeros
        for i in range(num_moments):
            some_times_are_not_here.append(f"shape_cos{i + 1}(-)")
            if i > 1:
                some_times_are_not_here.append(f"shape_sin{i + 1}(-)")

        for ikey in some_times_are_not_here:
            if ikey not in self.profiles.keys():
                self.profiles[ikey] = copy.deepcopy(self.profiles["rmin(m)"]) * 0.0

        self.produce_shape_lists()

    def produce_shape_lists(self):
        self.shape_cos = [
            self.profiles["shape_cos0(-)"],  # tilt
            self.profiles["shape_cos1(-)"],
            self.profiles["shape_cos2(-)"],
            self.profiles["shape_cos3(-)"],
            self.profiles["shape_cos4(-)"],
            self.profiles["shape_cos5(-)"],
            self.profiles["shape_cos6(-)"],
        ]
        self.shape_sin = [
            None,
            None,  # s1 is triangularity
            None,  # s2 is minus squareness
            self.profiles["shape_sin3(-)"],
            self.profiles["shape_sin4(-)"],
            self.profiles["shape_sin5(-)"],
            self.profiles["shape_sin6(-)"],
        ]

    def readSpecies(self, maxSpecies=100):
        maxSpecies = int(self.profiles["nion"][0])

        Species = []
        for j in range(maxSpecies):
            # To determine later if this specie has zero density
            niT = self.profiles["ni(10^19/m^3)"][:, j].max()

            sp = {
                "N": self.profiles["name"][j],
                "Z": float(self.profiles["z"][j]),
                "A": float(self.profiles["mass"][j]),
                "S": self.profiles["type"][j].split("[")[-1].split("]")[0],
                "dens": niT,
            }

            Species.append(sp)

        self.Species = Species

        self.mi_first = self.Species[0]["A"]

        self.DTplasma()
        self.sumFast()

    def sumFast(self):
        self.nFast = self.profiles["ne(10^19/m^3)"] * 0.0
        self.nZFast = self.profiles["ne(10^19/m^3)"] * 0.0
        self.nThermal = self.profiles["ne(10^19/m^3)"] * 0.0
        self.nZThermal = self.profiles["ne(10^19/m^3)"] * 0.0
        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "fast":
                self.nFast += self.profiles["ni(10^19/m^3)"][:, sp]
                self.nZFast += (
                    self.profiles["ni(10^19/m^3)"][:, sp] * self.profiles["z"][sp]
                )
            else:
                self.nThermal += self.profiles["ni(10^19/m^3)"][:, sp]
                self.nZThermal += (
                    self.profiles["ni(10^19/m^3)"][:, sp] * self.profiles["z"][sp]
                )

    def deriveQuantities(self, mi_ref=None, n_theta_geo=1001, rederiveGeometry=True):
        """
        deriving geometry is expensive, so if I'm just updating profiles it may not be needed
        """

        self.varqmom = "qmom(N/m^2)"
        if self.varqmom not in self.profiles:
            self.profiles[self.varqmom] = self.profiles["rho(-)"] * 0.0

        if "derived" not in self.__dict__:
            self.derived = {}

        # ---------------------------------------------------------------------------------------------------------------------
        # --------- MAIN (useful for STATEtools)
        # ---------------------------------------------------------------------------------------------------------------------

        self.derived["a"] = self.profiles["rmin(m)"][-1]
        # self.derived['epsX'] = self.profiles['rmaj(m)'] / self.profiles['rmin(m)']
        # self.derived['eps'] = self.derived['epsX'][-1]
        self.derived["eps"] = (
            self.profiles["rmin(m)"][-1] / self.profiles["rmaj(m)"][-1]
        )

        self.derived["roa"] = self.profiles["rmin(m)"] / self.derived["a"]

        self.derived["torflux"] = (
            float(self.profiles["torfluxa(Wb/radian)"][0])
            * 2
            * np.pi
            * self.profiles["rho(-)"] ** 2
        )  # Wb
        self.derived["B_unit"] = PLASMAtools.Bunit(
            self.derived["torflux"], self.profiles["rmin(m)"]
        )

        self.derived["psi_pol_n"] = (
            self.profiles["polflux(Wb/radian)"] - self.profiles["polflux(Wb/radian)"][0]
        ) / (
            self.profiles["polflux(Wb/radian)"][-1]
            - self.profiles["polflux(Wb/radian)"][0]
        )
        self.derived["rho_pol"] = self.derived["psi_pol_n"] ** 0.5

        self.derived["q95"] = np.interp(
            0.95, self.derived["psi_pol_n"], self.profiles["q(-)"]
        )

        # --------- Geometry (only if it doesn't exist or if I ask to recalculate)

        if rederiveGeometry or ("volp_miller" not in self.derived):

            self.produce_shape_lists()

            (
                self.derived["volp_miller"],
                self.derived["surf_miller"],
                self.derived["gradr_miller"],
                self.derived["geo_bt"],
            ) = GEOMETRYtools.calculateGeometricFactors(
                self,
                n_theta=n_theta_geo,
            )

            try:
                (
                    self.derived["R_surface"],
                    self.derived["Z_surface"],
                ) = GEQtools.create_geo_MXH3(
                    self.profiles["rmaj(m)"],
                    self.profiles["rmin(m)"],
                    self.profiles["zmag(m)"],
                    self.profiles["kappa(-)"],
                    self.profiles["delta(-)"],
                    self.profiles["zeta(-)"],
                    self.shape_cos,
                    self.shape_sin,
                    debugPlot=False
                )
            except:
                self.derived["R_surface"] = self.derived["Z_surface"] = None
                print(
                    "\t- Cannot calculate flux surface geometry out of the MXH3 moments",
                    typeMsg="w",
                )
            self.derived["R_LF"] = self.derived["R_surface"].max(
                axis=1
            )  # self.profiles['rmaj(m)'][0]+self.profiles['rmin(m)']

            # For Synchrotron
            self.derived["B_ref"] = np.abs(
                self.derived["B_unit"] * self.derived["geo_bt"]
            )

        # Forcing mass from this specific deriveQuantities call
        if mi_ref is not None:
            self.derived["mi_ref"] = mi_ref
        # Using mass that was used at class initialization if it was provided
        elif self.mi_ref is not None:
            self.derived["mi_ref"] = self.mi_ref
        # If nothing else provided, using the mass of the first ion
        else:
            self.derived["mi_ref"] = self.mi_first

        # ---------------------------------------------------------------------------------------------------------------------
        # --------- Important for scaling laws
        # ---------------------------------------------------------------------------------------------------------------------

        self.derived["kappa95"] = np.interp(
            0.95, self.derived["psi_pol_n"], self.profiles["kappa(-)"]
        )

        # I need to to kappa_a with the cross section...
        self.derived["kappa_a"] = self.derived[
            "kappa95"
        ]  # self.derived['surfXS_miller'][-1] / (np.pi*self.derived['a']**2)
        # print('\t- NOTE: Using kappa95 ({0:.2f}) as areal elongation for scalings. I need to work on the implementation of the XS surface area'.format(self.derived['kappa95']),typeMsg='w')

        self.derived["Rgeo"] = float(self.profiles["rcentr(m)"][-1])
        self.derived["B0"] = np.abs(float(self.profiles["bcentr(T)"][-1]))

        # ---------------------------------------------------------------------------------------------------------------------

        """
		surf_miller is truly surface area, but because of the GACODE definitions of flux, 
		Surf 		= V' <|grad r|>	 
		Surf_GACODE = V'
		"""

        self.derived["surfGACODE_miller"] = (
            self.derived["surf_miller"] / self.derived["gradr_miller"]
        )

        self.derived["surfGACODE_miller"][
            np.isnan(self.derived["surfGACODE_miller"])
        ] = 0

        self.derived["c_s"] = PLASMAtools.c_s(
            self.profiles["te(keV)"], self.derived["mi_ref"]
        )
        self.derived["rho_s"] = PLASMAtools.rho_s(
            self.profiles["te(keV)"], self.derived["mi_ref"], self.derived["B_unit"]
        )

        self.derived["q_gb"], self.derived["g_gb"], _, _, _ = PLASMAtools.gyrobohmUnits(
            self.profiles["te(keV)"],
            self.profiles["ne(10^19/m^3)"] * 1e-1,
            self.derived["mi_ref"],
            np.abs(self.derived["B_unit"]),
            self.profiles["rmin(m)"][-1],
        )

        """
		In prgen_map_plasmastate:
			qspow_e = expro_qohme+expro_qbeame+expro_qrfe+expro_qfuse-expro_qei &
				-expro_qsync-expro_qbrem-expro_qline
			qspow_i = expro_qbeami+expro_qrfi+expro_qfusi+expro_qei
		"""

        qe_terms = {
            "qohme(MW/m^3)": 1,
            "qbeame(MW/m^3)": 1,
            "qrfe(MW/m^3)": 1,
            "qfuse(MW/m^3)": 1,
            "qei(MW/m^3)": -1,
            "qsync(MW/m^3)": -1,
            "qbrem(MW/m^3)": -1,
            "qline(MW/m^3)": -1,
            "qione(MW/m^3)": 1,
        }

        self.derived["qe"] = np.zeros(len(self.profiles["rho(-)"]))
        for i in qe_terms:
            if i in self.profiles:
                self.derived["qe"] += qe_terms[i] * self.profiles[i]

        qrad = {
            "qsync(MW/m^3)": 1,
            "qbrem(MW/m^3)": 1,
            "qline(MW/m^3)": 1,
        }

        self.derived["qrad"] = np.zeros(len(self.profiles["rho(-)"]))
        for i in qrad:
            if i in self.profiles:
                self.derived["qrad"] += qrad[i] * self.profiles[i]

        qi_terms = {
            "qbeami(MW/m^3)": 1,
            "qrfi(MW/m^3)": 1,
            "qfusi(MW/m^3)": 1,
            "qei(MW/m^3)": 1,
            "qioni(MW/m^3)": 1,
        }

        self.derived["qi"] = np.zeros(len(self.profiles["rho(-)"]))
        for i in qi_terms:
            if i in self.profiles:
                self.derived["qi"] += qi_terms[i] * self.profiles[i]

        # Depends on GACODE version
        ge_terms = {self.varqpar: 1, self.varqpar2: 1}

        self.derived["ge"] = np.zeros(len(self.profiles["rho(-)"]))
        for i in ge_terms:
            if i in self.profiles:
                self.derived["ge"] += ge_terms[i] * self.profiles[i]

        """
		Careful, that's in MW/m^3. I need to find the volumes. Using here the Miller
		calculation. Should be consistent with TGYRO

		profiles_gen puts any missing power into the CX: qioni, qione
		"""

        r = self.profiles["rmin(m)"]
        volp = self.derived["volp_miller"]

        self.derived["qe_MWmiller"] = CALCtools.integrateFS(self.derived["qe"], r, volp)
        self.derived["qi_MWmiller"] = CALCtools.integrateFS(self.derived["qi"], r, volp)
        self.derived["ge_10E20miller"] = CALCtools.integrateFS(
            self.derived["ge"] * 1e-20, r, volp
        )  # Because the units were #/sec/m^3

        self.derived["geIn"] = self.derived["ge_10E20miller"][-1]  # 1E20 particles/sec

        self.derived["qe_MWm2"] = self.derived["qe_MWmiller"] / (volp)
        self.derived["qi_MWm2"] = self.derived["qi_MWmiller"] / (volp)
        self.derived["ge_10E20m2"] = self.derived["ge_10E20miller"] / (volp)

        self.derived["QiQe"] = self.derived["qi_MWm2"] / np.where(self.derived["qe_MWm2"] == 0, 1e-10, self.derived["qe_MWm2"]) # to avoid division by zero

        # "Convective" flux
        self.derived["ce_MWmiller"] = PLASMAtools.convective_flux(
            self.profiles["te(keV)"], self.derived["ge_10E20miller"]
        )
        self.derived["ce_MWm2"] = PLASMAtools.convective_flux(
            self.profiles["te(keV)"], self.derived["ge_10E20m2"]
        )

        # qmom
        self.derived["mt_Jmiller"] = CALCtools.integrateFS(
            self.profiles[self.varqmom], r, volp
        )
        self.derived["mt_Jm2"] = self.derived["mt_Jmiller"] / (volp)

        # Extras for plotting in TGYRO for comparison
        P = np.zeros(len(self.profiles["rmin(m)"]))
        if "qsync(MW/m^3)" in self.profiles:
            P += self.profiles["qsync(MW/m^3)"]
        if "qbrem(MW/m^3)" in self.profiles:
            P += self.profiles["qbrem(MW/m^3)"]
        if "qline(MW/m^3)" in self.profiles:
            P += self.profiles["qline(MW/m^3)"]
        self.derived["qe_rad_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = self.profiles["qei(MW/m^3)"]
        self.derived["qe_exc_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        """
		---------------------------------------------------------------------------------------------------------------------
		Note that the real auxiliary power is RF+BEAMS+OHMIC, 
		The QIONE is added by TGYRO, but sometimes it includes radiation and direct RF to electrons
		---------------------------------------------------------------------------------------------------------------------
		"""

        # ** Electrons

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qrfe(MW/m^3)", "qohme(MW/m^3)", "qbeame(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qe_auxONLY"] = copy.deepcopy(P)
        self.derived["qe_auxONLY_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        for i in ["qione(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qe_aux"] = copy.deepcopy(P)
        self.derived["qe_aux_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        # ** Ions

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qrfi(MW/m^3)", "qbeami(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qi_auxONLY"] = copy.deepcopy(P)
        self.derived["qi_auxONLY_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        for i in ["qioni(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qi_aux"] = copy.deepcopy(P)
        self.derived["qi_aux_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        # ** General

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qohme(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qOhm_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qrfe(MW/m^3)", "qrfi(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qRF_MWmiller"] = CALCtools.integrateFS(P, r, volp)
        if "qrfe(MW/m^3)" in self.profiles:
            self.derived["qRFe_MWmiller"] = CALCtools.integrateFS(
                self.profiles["qrfe(MW/m^3)"], r, volp
            )
        if "qrfi(MW/m^3)" in self.profiles:
            self.derived["qRFi_MWmiller"] = CALCtools.integrateFS(
                self.profiles["qrfi(MW/m^3)"], r, volp
            )

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qbeame(MW/m^3)", "qbeami(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qBEAM_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = self.derived["qrad"]
        self.derived["qrad_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qfuse(MW/m^3)", "qfusi(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qFus_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qioni(MW/m^3)", "qione(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qz_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        self.derived["q_MWmiller"] = (
            self.derived["qe_MWmiller"] + self.derived["qi_MWmiller"]
        )

        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------

        P = np.zeros(len(self.profiles["rho(-)"]))
        if "qfuse(MW/m^3)" in self.profiles:
            P = self.profiles["qfuse(MW/m^3)"]
        self.derived["qe_fus_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        if "qfusi(MW/m^3)" in self.profiles:
            P = self.profiles["qfusi(MW/m^3)"]
        self.derived["qi_fus_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        if "qfusi(MW/m^3)" in self.profiles:
            self.derived["q_fus"] = (
                self.profiles["qfuse(MW/m^3)"] + self.profiles["qfusi(MW/m^3)"]
            ) * 5
            P = self.derived["q_fus"]
        self.derived["q_fus"] = P
        self.derived["q_fus_MWmiller"] = CALCtools.integrateFS(P, r, volp)

        """
		Derivatives
		"""
        self.derived["aLTe"] = aLT(self.profiles["rmin(m)"], self.profiles["te(keV)"])
        self.derived["aLTi"] = self.profiles["ti(keV)"] * 0.0
        for i in range(self.profiles["ti(keV)"].shape[1]):
            self.derived["aLTi"][:, i] = aLT(
                self.profiles["rmin(m)"], self.profiles["ti(keV)"][:, i]
            )
        self.derived["aLne"] = aLT(
            self.profiles["rmin(m)"], self.profiles["ne(10^19/m^3)"]
        )
        self.derived["aLni"] = []
        for i in range(self.profiles["ni(10^19/m^3)"].shape[1]):
            self.derived["aLni"].append(
                aLT(self.profiles["rmin(m)"], self.profiles["ni(10^19/m^3)"][:, i])
            )
        self.derived["aLni"] = np.transpose(np.array(self.derived["aLni"]))

        if "w0(rad/s)" not in self.profiles:
            self.profiles["w0(rad/s)"] = self.profiles["rho(-)"] * 0.0
        self.derived["aLw0"] = aLT(self.profiles["rmin(m)"], self.profiles["w0(rad/s)"])
        self.derived["dw0dr"] = -grad(
            self.profiles["rmin(m)"], self.profiles["w0(rad/s)"]
        )

        self.derived["dqdr"] = grad(self.profiles["rmin(m)"], self.profiles["q(-)"])

        """
		Other, performance
		"""
        qFus = self.derived["qe_fus_MWmiller"] + self.derived["qi_fus_MWmiller"]
        self.derived["Pfus"] = qFus[-1] * 5

        # Note that in cases with NPRAD=0 in TRANPS, this includes radiation! no way to deal wit this...
        qIn = self.derived["qe_aux_MWmiller"] + self.derived["qi_aux_MWmiller"]
        self.derived["qIn"] = qIn[-1]
        self.derived["Q"] = self.derived["Pfus"] / self.derived["qIn"]
        self.derived["qHeat"] = qIn[-1] + qFus[-1]

        self.derived["qTr"] = (
            self.derived["qe_aux_MWmiller"]
            + self.derived["qi_aux_MWmiller"]
            + (self.derived["qe_fus_MWmiller"] + self.derived["qi_fus_MWmiller"])
            - self.derived["qrad_MWmiller"]
        )

        self.derived["Prad"] = self.derived["qrad_MWmiller"][-1]
        self.derived["Psol"] = self.derived["qHeat"] - self.derived["Prad"]

        self.derived["ni_thr"] = []
        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "therm":
                self.derived["ni_thr"].append(self.profiles["ni(10^19/m^3)"][:, sp])
        self.derived["ni_thr"] = np.transpose(self.derived["ni_thr"])
        self.derived["ni_thrAll"] = self.derived["ni_thr"].sum(axis=1)

        (
            self.derived["ptot_manual"],
            self.derived["pe"],
            self.derived["pi"],
        ) = PLASMAtools.calculatePressure(
            np.expand_dims(self.profiles["te(keV)"], 0),
            np.expand_dims(np.transpose(self.profiles["ti(keV)"]), 0),
            np.expand_dims(self.profiles["ne(10^19/m^3)"] * 0.1, 0),
            np.expand_dims(np.transpose(self.profiles["ni(10^19/m^3)"] * 0.1), 0),
        )
        self.derived["ptot_manual"], self.derived["pe"], self.derived["pi"] = (
            self.derived["ptot_manual"][0],
            self.derived["pe"][0],
            self.derived["pi"][0],
        )

        (
            self.derived["pthr_manual"],
            _,
            self.derived["pi_thr"],
        ) = PLASMAtools.calculatePressure(
            np.expand_dims(self.profiles["te(keV)"], 0),
            np.expand_dims(np.transpose(self.profiles["ti(keV)"]), 0),
            np.expand_dims(self.profiles["ne(10^19/m^3)"] * 0.1, 0),
            np.expand_dims(np.transpose(self.derived["ni_thr"] * 0.1), 0),
        )
        self.derived["pthr_manual"], self.derived["pi_thr"] = (
            self.derived["pthr_manual"][0],
            self.derived["pi_thr"][0],
        )

        # -------
        # Content
        # -------

        (
            self.derived["We"],
            self.derived["Wi_thr"],
            self.derived["Ne"],
            self.derived["Ni_thr"],
        ) = PLASMAtools.calculateContent(
            np.expand_dims(r, 0),
            np.expand_dims(self.profiles["te(keV)"], 0),
            np.expand_dims(np.transpose(self.profiles["ti(keV)"]), 0),
            np.expand_dims(self.profiles["ne(10^19/m^3)"] * 0.1, 0),
            np.expand_dims(np.transpose(self.derived["ni_thr"] * 0.1), 0),
            np.expand_dims(volp, 0),
        )

        (
            self.derived["We"],
            self.derived["Wi_thr"],
            self.derived["Ne"],
            self.derived["Ni_thr"],
        ) = (
            self.derived["We"][0],
            self.derived["Wi_thr"][0],
            self.derived["Ne"][0],
            self.derived["Ni_thr"][0],
        )

        self.derived["Nthr"] = self.derived["Ne"] + self.derived["Ni_thr"]
        self.derived["Wthr"] = self.derived["We"] + self.derived["Wi_thr"]  # Thermal

        self.derived["tauE"] = self.derived["Wthr"] / self.derived["qHeat"]  # Seconds

        self.derived["tauP"] = self.derived["Ne"] / self.derived["geIn"]  # Seconds

        self.derived["tauPotauE"] = self.derived["tauP"] / self.derived["tauE"]

        # Dilutions
        self.derived["fi"] = self.profiles["ni(10^19/m^3)"] / np.atleast_2d(
            self.profiles["ne(10^19/m^3)"]
        ).transpose().repeat(self.profiles["ni(10^19/m^3)"].shape[1], axis=1)

        # Vol-avg density
        self.derived["volume"] = CALCtools.integrateFS(np.ones(r.shape[0]), r, volp)[
            -1
        ]  # m^3
        self.derived["ne_vol20"] = (
            CALCtools.integrateFS(self.profiles["ne(10^19/m^3)"] * 0.1, r, volp)[-1]
            / self.derived["volume"]
        )  # 1E20/m^3

        self.derived["ni_vol20"] = np.zeros(self.profiles["ni(10^19/m^3)"].shape[1])
        self.derived["fi_vol"] = np.zeros(self.profiles["ni(10^19/m^3)"].shape[1])
        for i in range(self.profiles["ni(10^19/m^3)"].shape[1]):
            self.derived["ni_vol20"][i] = (
                CALCtools.integrateFS(
                    self.profiles["ni(10^19/m^3)"][:, i] * 0.1, r, volp
                )[-1]
                / self.derived["volume"]
            )  # 1E20/m^3
            self.derived["fi_vol"][i] = (
                self.derived["ni_vol20"][i] / self.derived["ne_vol20"]
            )

        self.derived["fi_onlyions_vol"] = self.derived["ni_vol20"] / np.sum(
            self.derived["ni_vol20"]
        )

        self.derived["ne_peaking"] = (
            self.profiles["ne(10^19/m^3)"][0] * 0.1 / self.derived["ne_vol20"]
        )

        xcoord = self.derived[
            "rho_pol"
        ]  # to find the peaking at rho_pol (with square root) as in Angioni PRL 2003
        self.derived["ne_peaking0.2"] = (
            self.profiles["ne(10^19/m^3)"][np.argmin(np.abs(xcoord - 0.2))]
            * 0.1
            / self.derived["ne_vol20"]
        )

        self.derived["Te_vol"] = (
            CALCtools.integrateFS(self.profiles["te(keV)"], r, volp)[-1]
            / self.derived["volume"]
        )  # keV
        self.derived["Te_peaking"] = (
            self.profiles["te(keV)"][0] / self.derived["Te_vol"]
        )
        self.derived["Ti_vol"] = (
            CALCtools.integrateFS(self.profiles["ti(keV)"][:, 0], r, volp)[-1]
            / self.derived["volume"]
        )  # keV
        self.derived["Ti_peaking"] = (
            self.profiles["ti(keV)"][0, 0] / self.derived["Ti_vol"]
        )

        self.derived["ptot_manual_vol"] = (
            CALCtools.integrateFS(self.derived["ptot_manual"], r, volp)[-1]
            / self.derived["volume"]
        )  # MPa
        self.derived["pthr_manual_vol"] = (
            CALCtools.integrateFS(self.derived["pthr_manual"], r, volp)[-1]
            / self.derived["volume"]
        )  # MPa

        # Quasineutrality
        self.derived["QN_Error"] = np.abs(
            1 - np.sum(self.derived["fi_vol"] * self.profiles["z"])
        )
        self.derived["Zeff"] = (
            np.sum(self.profiles["ni(10^19/m^3)"] * self.profiles["z"] ** 2, axis=1)
            / self.profiles["ne(10^19/m^3)"]
        )
        self.derived["Zeff_vol"] = (
            CALCtools.integrateFS(self.derived["Zeff"], r, volp)[-1]
            / self.derived["volume"]
        )

        self.derived["nu_eff"] = PLASMAtools.coll_Angioni07(
            self.derived["ne_vol20"] * 1e1,
            self.derived["Te_vol"],
            self.derived["Rgeo"],
            Zeff=self.derived["Zeff_vol"],
        )

        self.derived["nu_eff2"] = PLASMAtools.coll_Angioni07(
            self.derived["ne_vol20"] * 1e1,
            self.derived["Te_vol"],
            self.derived["Rgeo"],
            Zeff=2.0,
        )

        # Avg mass
        self.calculateMass()

        params_set_scaling = (
            np.abs(float(self.profiles["current(MA)"][-1])),
            self.derived["Rgeo"],
            self.derived["kappa_a"],
            self.derived["ne_vol20"],
            self.derived["a"] / self.derived["Rgeo"],
            self.derived["B0"],
            self.derived["mbg_main"],
            self.derived["qHeat"],
        )

        self.derived["tau98y2"], self.derived["H98"] = PLASMAtools.tau98y2(
            *params_set_scaling, tauE=self.derived["tauE"]
        )
        self.derived["tau89p"], self.derived["H89"] = PLASMAtools.tau89p(
            *params_set_scaling, tauE=self.derived["tauE"]
        )
        self.derived["tau97L"], self.derived["H97L"] = PLASMAtools.tau97L(
            *params_set_scaling, tauE=self.derived["tauE"]
        )

        """
		Mach number
		"""

        Vtor_LF_Mach1 = PLASMAtools.constructVtorFromMach(
            1.0, self.profiles["ti(keV)"][:, 0], self.derived["mbg"]
        )  # m/s
        w0_Mach1 = Vtor_LF_Mach1 / (self.derived["R_LF"])  # rad/s
        self.derived["MachNum"] = self.profiles["w0(rad/s)"] / w0_Mach1
        self.derived["MachNum_vol"] = (
            CALCtools.integrateFS(self.derived["MachNum"], r, volp)[-1]
            / self.derived["volume"]
        )

        """
		This beta is very approx, since I should be using the average of B**2? is B_ref?
		"""
        # Beta = CALCtools.integrateFS( self.derived['ptot_manual']*1E6 / (self.derived['B_ref']**2/(2*4*np.pi*1E-7 )),r,volp)[-1] / self.derived['volume']
        # Beta = self.derived['ptot_manual_vol']*1E6 / (self.derived['B0']**2/(2*4*np.pi*1E-7 ))
        # Beta = self.derived['ptot_manual_vol']*1E6 / ( CALCtools.integrateFS(  self.derived['B_ref']**2 ,r,volp)[-1] / self.derived['volume'] /(2*4*np.pi*1E-7) )

        Beta = (
            self.derived["pthr_manual_vol"]
            * 1e6
            / (self.derived["B0"] ** 2 / (2 * 4 * np.pi * 1e-7))
        )
        self.derived["BetaN"] = (
            Beta
            / (
                np.abs(float(self.profiles["current(MA)"][-1]))
                / (self.derived["a"] * self.derived["B0"])
            )
            * 100.0
        )
        # ---

        nG = PLASMAtools.Greenwald_density(
            np.abs(float(self.profiles["current(MA)"][-1])),
            float(self.profiles["rmin(m)"][-1]),
        )
        self.derived["fG"] = self.derived["ne_vol20"] / nG

        self.derived["tite"] = self.profiles["ti(keV)"][:, 0] / self.profiles["te(keV)"]
        self.derived["tite_vol"] = self.derived["Ti_vol"] / self.derived["Te_vol"]

        self.derived["LH_nmin"] = PLASMAtools.LHthreshold_nmin(
            np.abs(float(self.profiles["current(MA)"][-1])),
            self.derived["B0"],
            self.derived["a"],
            self.derived["Rgeo"],
        )

        self.derived["LH_Martin2"] = (
            PLASMAtools.LHthreshold_Martin2(
                self.derived["ne_vol20"],
                self.derived["B0"],
                self.derived["a"],
                self.derived["Rgeo"],
                nmin=self.derived["LH_nmin"],
            )
            * (2 / self.derived["mbg_main"]) ** 1.11
        )

        self.derived["LHratio"] = self.derived["Psol"] / self.derived["LH_Martin2"]

        self.readSpecies()

    def calculateMass(self):
        self.derived["mbg"] = 0.0
        self.derived["fmain"] = 0.0
        for i in range(self.derived["ni_vol20"].shape[0]):
            self.derived["mbg"] += (
                float(self.profiles["mass"][i]) * self.derived["fi_onlyions_vol"][i]
            )

        if self.DTplasmaBool:
            self.derived["mbg_main"] = (
                self.profiles["mass"][self.Dion]
                * self.derived["fi_onlyions_vol"][self.Dion]
                + self.profiles["mass"][self.Tion]
                * self.derived["fi_onlyions_vol"][self.Tion]
            ) / (
                self.derived["fi_onlyions_vol"][self.Dion]
                + self.derived["fi_onlyions_vol"][self.Tion]
            )
            self.derived["fmain"] = (
                self.derived["fi_vol"][self.Dion] + self.derived["fi_vol"][self.Tion]
            )
        else:
            self.derived["mbg_main"] = self.profiles["mass"][self.Mion]
            self.derived["fmain"] = self.derived["fi_vol"][self.Mion]

    def deriveContentByVolumes(self, rhos=[0.5], impurityPosition=3):
        """
        Calculates total particles and energy for ions and electrons, at a given volume
        It fails near axis because of the polynomial integral, requiring a number of poitns
        """

        min_number_points = 3

        We_x = np.zeros(self.profiles["te(keV)"].shape[0])
        Wi_x = np.zeros(self.profiles["te(keV)"].shape[0])
        Ne_x = np.zeros(self.profiles["te(keV)"].shape[0])
        Ni_x = np.zeros(self.profiles["te(keV)"].shape[0])
        for j in range(self.profiles["te(keV)"].shape[0] - min_number_points):
            i = j + min_number_points
            We_x[i], Wi_x[i], Ne_x[i], _ = PLASMAtools.calculateContent(
                np.expand_dims(self.profiles["rmin(m)"][:i], 0),
                np.expand_dims(self.profiles["te(keV)"][:i], 0),
                np.expand_dims(np.transpose(self.profiles["ti(keV)"][:i]), 0),
                np.expand_dims(self.profiles["ne(10^19/m^3)"][:i] * 0.1, 0),
                np.expand_dims(
                    np.transpose(self.profiles["ni(10^19/m^3)"][:i] * 0.1), 0
                ),
                np.expand_dims(self.derived["volp_miller"][:i], 0),
            )

            _, _, Ni_x[i], _ = PLASMAtools.calculateContent(
                np.expand_dims(self.profiles["rmin(m)"][:i], 0),
                np.expand_dims(self.profiles["te(keV)"][:i], 0),
                np.expand_dims(np.transpose(self.profiles["ti(keV)"][:i]), 0),
                np.expand_dims(
                    self.profiles["ni(10^19/m^3)"][:i, impurityPosition - 1] * 0.1, 0
                ),
                np.expand_dims(
                    np.transpose(self.profiles["ni(10^19/m^3)"][:i] * 0.1), 0
                ),
                np.expand_dims(self.derived["volp_miller"][:i], 0),
            )

        We, Wi, Ne, Ni = (
            np.zeros(len(rhos)),
            np.zeros(len(rhos)),
            np.zeros(len(rhos)),
            np.zeros(len(rhos)),
        )
        for i in range(len(rhos)):
            We[i] = np.interp(rhos[i], self.profiles["rho(-)"], We_x)
            Wi[i] = np.interp(rhos[i], self.profiles["rho(-)"], Wi_x)
            Ne[i] = np.interp(rhos[i], self.profiles["rho(-)"], Ne_x)
            Ni[i] = np.interp(rhos[i], self.profiles["rho(-)"], Ni_x)

        return We, Wi, Ne, Ni

    def printInfo(self, label="", reDeriveIfNotFound=True):
        try:
            ImpurityText = ""
            for i in range(len(self.Species)):
                ImpurityText += f"{self.Species[i]['N']}({self.Species[i]['Z']:.0f},{self.Species[i]['A']:.0f}) = {self.derived['fi_vol'][i]:.1e}, "
            ImpurityText = ImpurityText[:-2]

            print(f"\n***********************{label}****************")
            print("Performance:")
            print(
                "\tQ     =  {0:.2f}   (Pfus = {1:.1f}MW, Pin = {2:.1f}MW)".format(
                    self.derived["Q"], self.derived["Pfus"], self.derived["qIn"]
                )
            )
            print(
                "\tH98y2 =  {0:.2f}   (tauE  = {1:.3f} s)".format(
                    self.derived["H98"], self.derived["tauE"]
                )
            )
            print(
                "\tH89p  =  {0:.2f}   (H97L  = {1:.2f})".format(
                    self.derived["H89"], self.derived["H97L"]
                )
            )
            print(
                "\tnu_ne =  {0:.2f}   (nu_eff = {1:.2f})".format(
                    self.derived["ne_peaking"], self.derived["nu_eff"]
                )
            )
            print(
                "\tnu_ne0.2 =  {0:.2f}   (nu_eff w/Zeff2 = {1:.2f})".format(
                    self.derived["ne_peaking0.2"], self.derived["nu_eff2"]
                )
            )
            print(f"\tnu_Ti =  {self.derived['Ti_peaking']:.2f}")
            print(
                "\tBetaN =  {0:.3f} (approx, based on B0 and p_thr)".format(
                    self.derived["BetaN"]
                )
            )
            print(
                "\tPrad  =  {0:.1f}MW ({1:.1f}% of total)".format(
                    self.derived["Prad"],
                    self.derived["Prad"] / self.derived["qHeat"] * 100.0,
                )
            )
            print(
                "\tPsol  =  {0:.1f}MW (fLH = {1:.2f})".format(
                    self.derived["Psol"], self.derived["LHratio"]
                )
            )
            print(
                "Operational point ( [<ne>,<Te>] = [{0:.2f},{1:.2f}] ) and species:".format(
                    self.derived["ne_vol20"], self.derived["Te_vol"]
                )
            )
            print(
                "\t<Ti>  = {0:.2f} keV   (<Ti>/<Te> = {1:.2f}, Ti0/Te0 = {2:.2f})".format(
                    self.derived["Ti_vol"],
                    self.derived["tite_vol"],
                    self.derived["tite"][0],
                )
            )
            print(
                "\tfG    = {0:.2f}   (<ne> = {1:.2f} * 10^20 m^-3)".format(
                    self.derived["fG"], self.derived["ne_vol20"]
                )
            )
            print(
                f"\tZeff  = {self.derived['Zeff_vol']:.2f}   (M_main = {self.derived['mbg_main']:.2f}, f_main = {self.derived['fmain']:.2f}) [QN err = {self.derived['QN_Error']:.1e}]"
            )
            print(f"\tMach  = {self.derived['MachNum_vol']:.2f} (vol avg)")
            print("Content:")
            print(
                "\tWe = {0:.2f} MJ,   Wi_thr = {1:.2f} MJ    (W_thr = {2:.2f} MJ)".format(
                    self.derived["We"], self.derived["Wi_thr"], self.derived["Wthr"]
                )
            )
            print(
                "\tNe = {0:.1f}*10^20, Ni_thr = {1:.1f}*10^20 (N_thr = {2:.1f}*10^20)".format(
                    self.derived["Ne"], self.derived["Ni_thr"], self.derived["Nthr"]
                )
            )
            print(
                f"\ttauE  = { self.derived['tauE']:.3f} s,  tauP = {self.derived['tauP']:.3f} s (tauP/tauE = {self.derived['tauPotauE']:.2f})"
            )
            print("Species concentration:")
            print(f"\t{ImpurityText}")
            print("******************************************************")
        except KeyError:
            print(
                "\t- When printing info, not all keys found, probably because this input.gacode class came from an old MITIM version",
                typeMsg="w",
            )
            if reDeriveIfNotFound:
                self.deriveQuantities()
                self.printInfo(label=label, reDeriveIfNotFound=False)

    def export_to_table(self, table=None, name=None):

        if table is None:
            table = DataTable()

        data = [name]
        for var in table.variables:
            if table.variables[var][1] is not None:
                if table.variables[var][1].split("_")[0] == "rho":
                    ix = np.argmin(
                        np.abs(
                            self.profiles["rho(-)"]
                            - float(table.variables[var][1].split("_")[1])
                        )
                    )
                elif table.variables[var][1].split("_")[0] == "psi":
                    ix = np.argmin(
                        np.abs(
                            self.derived["psi_pol_n"]
                            - float(table.variables[var][1].split("_")[1])
                        )
                    )
                elif table.variables[var][1].split("_")[0] == "pos":
                    ix = int(table.variables[var][1].split("_")[1])
                vari = self.__dict__[table.variables[var][2]][table.variables[var][0]][
                    ix
                ]
            else:
                vari = self.__dict__[table.variables[var][2]][table.variables[var][0]]

            data.append(f"{vari*table.variables[var][4]:{table.variables[var][3]}}")

        table.data.append(data)
        print(f"\t* Exported {name} to table")

        return table

    def makeAllThermalIonsHaveSameTemp(self, refIon=0):
        SpecRef = self.Species[refIon]["N"]
        tiRef = self.profiles["ti(keV)"][:, refIon]

        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "therm" and sp != refIon:
                print(
                    f"\t\t\t- Temperature forcing {self.Species[sp]['N']} --> {SpecRef}"
                )
                self.profiles["ti(keV)"][:, sp] = tiRef

    def scaleAllThermalDensities(self, scaleFactor=1.0):
        scaleFactor_ions = scaleFactor

        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "therm":
                print(
                    f"\t\t\t- Scaling density of {self.Species[sp]['N']} by an average factor of {np.mean(scaleFactor_ions):.3f}"
                )
                ni_orig = self.profiles["ni(10^19/m^3)"][:, sp]
                self.profiles["ni(10^19/m^3)"][:, sp] = scaleFactor_ions * ni_orig

    def writeCurrentStatus(self, file=None, limitedNames=False):
        print("\t- Writting input.gacode file")

        if file is None:
            file = self.file

        with open(file, "w") as f:
            for line in self.header:
                f.write(line)

            for i in self.profiles:
                if "(" not in i:
                    f.write(f"# {i}\n")
                else:
                    f.write(f"# {i.split('(')[0]} | {i.split('(')[-1].split(')')[0]}\n")

                if i in self.titles_single:
                    if i == "name" and limitedNames:
                        newlist = [self.profiles[i][0]]
                        for k in self.profiles[i][1:]:
                            if k not in [
                                "D",
                                "H",
                                "T",
                                "He4",
                                "he4",
                                "C",
                                "O",
                                "Ar",
                                "W",
                            ]:
                                newlist.append("C")
                            else:
                                newlist.append(k)
                        print(
                            f"\n\n!! Correcting ion names from {self.profiles[i]} to {newlist} to avoid TGYRO radiation error (to solve in future?)\n\n",
                            typeMsg="w",
                        )
                        listWrite = newlist
                    else:
                        listWrite = self.profiles[i]

                    if IOtools.isfloat(listWrite[0]):
                        listWrite = [f"{i:.7e}".rjust(14) for i in listWrite]
                        f.write(f"{''.join(listWrite)}\n")
                    else:
                        f.write(f"{' '.join(listWrite)}\n")

                else:
                    if len(self.profiles[i].shape) == 1:
                        for j, val in enumerate(self.profiles[i]):
                            pos = f"{j + 1}".rjust(3)
                            valt = f"{val:.7e}".rjust(15)
                            f.write(f"{pos}{valt}\n")
                    else:
                        for j, val in enumerate(self.profiles[i]):
                            pos = f"{j + 1}".rjust(3)
                            txt = "".join([f"{k:.7e}".rjust(15) for k in val])
                            f.write(f"{pos}{txt}\n")

        print(
            f"\t\t~ File {IOtools.clipstr(file)} written",
            verbose=verbose_level,
        )

        # Update file
        self.file = file

    def writeMiminalKinetic(self, file):
        setProfs = [
            "rho(-)",
            "polflux(Wb/radian)",
            "q(-)",
            "te(keV)",
            "ti(keV)",
            "ne(10^19/m^3)",
        ]

        with open(file, "w") as f:
            for i in setProfs:
                if "(" not in i:
                    f.write(f"# {i}\n")
                else:
                    f.write(f"# {i.split('(')[0]} | {i.split('(')[-1].split(')')[0]}\n")

                if len(self.profiles[i].shape) > 1:
                    p = self.profiles[i][:, 0]
                else:
                    p = self.profiles[i]

                for j, val in enumerate(p):
                    pos = f"{j + 1}".rjust(3)
                    valt = f"{val:.7e}".rjust(15)
                    f.write(f"{pos}{valt}\n")

    def changeResolution(
        self, n=100, rho_new=None, interpolation_function=MATHtools.extrapolateCubicSpline
    ):
        rho = copy.deepcopy(self.profiles["rho(-)"])

        if rho_new is None:
            n = int(n)
            rho_new = np.linspace(rho[0], rho[-1], n)
        else:
            rho_new = np.unique(np.sort(rho_new))
            n = len(rho_new)

        self.profiles["nexp"] = [str(n)]

        pro = self.profiles
        for i in pro:
            if i not in self.titles_single:
                if len(pro[i].shape) == 1:
                    pro[i] = interpolation_function(rho_new, rho, pro[i])
                else:
                    prof = []
                    for j in range(pro[i].shape[1]):
                        pp = interpolation_function(rho_new, rho, pro[i][:, j])
                        prof.append(pp)
                    prof = np.array(prof)

                    pro[i] = np.transpose(prof)

        self.produce_shape_lists()

        self.deriveQuantities(mi_ref=self.derived["mi_ref"])

        print(
            f"\t\t- Resolution of profiles changed to {n} points with function {interpolation_function}"
        )

    def DTplasma(self):
        self.Dion, self.Tion = None, None
        try:
            self.Dion = np.where(self.profiles["name"] == "D")[0][0]
        except:
            pass
        try:
            self.Tion = np.where(self.profiles["name"] == "T")[0][0]
        except:
            pass

        if self.Dion is not None and self.Tion is not None:
            self.DTplasmaBool = True
        else:
            self.DTplasmaBool = False
            if self.Dion is not None:
                self.Mion = self.Dion  # Main
            elif self.Tion is not None:
                self.Mion = self.Tion  # Main
            else:
                self.Mion = (
                    0  # If no D or T, assume that the main ion is the first and only
                )

    def remove(self, ions_list):
        # First order them
        ions_list.sort()
        print(
            "\t\t- Removing ions in positions (of ions order, no zero): ",
            ions_list,
            typeMsg="i",
        )

        ions_list = [i - 1 for i in ions_list]

        fail = False

        var_changes = ["name", "type", "mass", "z"]
        for i in var_changes:
            try:
                self.profiles[i] = np.delete(self.profiles[i], ions_list)
            except:
                print(
                    f"\t\t\t* Ions {[k+1 for k in ions_list]} could not be removed",
                    typeMsg="w",
                )
                fail = True
                break

        if not fail:
            var_changes = ["ni(10^19/m^3)", "ti(keV)"]
            for i in var_changes:
                self.profiles[i] = np.delete(self.profiles[i], ions_list, axis=1)

        if not fail:
            self.profiles["nion"] = np.array(
                [str(int(self.profiles["nion"]) - len(ions_list))]
            )

        self.readSpecies()
        self.deriveQuantities()

        print("\t\t\t- Set of ions in updated profiles: ", self.profiles["name"])

    def lumpSpecies(
        self, ions_list=[2, 3], allthermal=False, forcename=None, force_integer=False,
    ):
        """
        if (D,Z1,Z2), lumping Z1 and Z2 requires ions_list = [2,3]

        if force_integer, the Zeff won't be kept exactly
        """

        # All thermal except first
        if allthermal:
            ions_list = []
            for i in range(len(self.Species) - 1):
                if self.Species[i + 1]["S"] == "therm":
                    ions_list.append(i + 2)
            lab = "therm"
        else:
            lab = "therm"

        print(
            "\t\t- Lumping ions in positions (of ions order, no zero): ",
            ions_list,
            typeMsg="i",
        )

        if forcename is None:
            forcename = "LUMPED"

        # Contributions to dilution and to Zeff
        fZ1 = np.zeros(self.derived["fi"].shape[0])
        fZ2 = np.zeros(self.derived["fi"].shape[0])
        for i in ions_list:
            fZ1 += self.Species[i - 1]["Z"] * self.derived["fi"][:, i - 1]
            fZ2 += self.Species[i - 1]["Z"] ** 2 * self.derived["fi"][:, i - 1]

        Zr = fZ2 / fZ1
        Zr_vol = (
            CALCtools.integrateFS(
                Zr, self.profiles["rmin(m)"], self.derived["volp_miller"]
            )[-1]
            / self.derived["volume"]
        )

        print(
            f'\t\t\t* Original plasma had Zeff_vol={self.derived["Zeff_vol"]:.2f}, QN error={self.derived["QN_Error"]:.4f}'
        )

        # New specie parameters
        if force_integer:
            Z = round(Zr_vol)
            print(
                f"\t\t\t* Lumped Z forced to be an integer ({Zr_vol}->{Z}), so plasma may not be quasineutral or fulfill original Zeff",
                typeMsg="w",
            )
        else:
            Z = Zr_vol

        A = Z * 2
        nZ = fZ1 / Z * self.profiles["ne(10^19/m^3)"]

        print(f"\t\t\t* New lumped impurity has Z={Z:.2f}, A={A:.2f} (calculated as 2*Z)")

        # Insert cases
        self.profiles["nion"] = np.array([f"{int(self.profiles['nion'][0])+1}"])
        self.profiles["name"] = np.append(self.profiles["name"], forcename)
        self.profiles["mass"] = np.append(self.profiles["mass"], A)
        self.profiles["z"] = np.append(self.profiles["z"], Z)
        self.profiles["type"] = np.append(self.profiles["type"], f"[{lab}]")
        self.profiles["ni(10^19/m^3)"] = np.append(
            self.profiles["ni(10^19/m^3)"], np.transpose(np.atleast_2d(nZ)), axis=1
        )
        self.profiles["ti(keV)"] = np.append(
            self.profiles["ti(keV)"],
            np.transpose(np.atleast_2d(self.profiles["ti(keV)"][:, 0])),
            axis=1,
        )

        self.readSpecies()
        self.deriveQuantities()

        # Remove species
        self.remove(ions_list)

        # Contributions to dilution and to Zeff
        print(
            f'\t\t\t* New plasma has Zeff_vol={self.derived["Zeff_vol"]:.2f}, QN error={self.derived["QN_Error"]:.4f}'
        )

    def changeZeff(self, Zeff, ion_pos=2, quasineutral_ions=None, enforceSameGradients=False):
        """
        if (D,Z1,Z2), pos 1 -> change Z1
        """

        if quasineutral_ions is None:
            if self.DTplasmaBool:
                quasineutral_ions = [self.Dion, self.Tion]
            else:
                quasineutral_ions = [self.Mion]

        print(
            f'\t\t- Changing Zeff (from {self.derived["Zeff_vol"]:.3f} to {Zeff=:.3f}) by changing content of ion in position {ion_pos} {self.Species[ion_pos]["N"],self.Species[ion_pos]["Z"]}, quasineutralized by ions {quasineutral_ions}',
            typeMsg="i",
        )

        # Plasma needs to be in quasineutrality to start with
        self.enforceQuasineutrality()

        # ------------------------------------------------------
        # Contributions to equations
        # ------------------------------------------------------
        Zq = np.zeros(self.derived["fi"].shape[0])
        Zq2 = np.zeros(self.derived["fi"].shape[0])
        fZj = np.zeros(self.derived["fi"].shape[0])
        fZj2 = np.zeros(self.derived["fi"].shape[0])
        for i in range(len(self.Species)):
            if i in quasineutral_ions:
                Zq += self.Species[i]["Z"] 
                Zq2 += self.Species[i]["Z"] ** 2 
            elif i != ion_pos:
                fZj += self.Species[i]["Z"] * self.derived["fi"][:, i]
                fZj2 += self.Species[i]["Z"] ** 2 * self.derived["fi"][:, i]
            else:
                Zk = self.Species[i]["Z"]

        # ------------------------------------------------------
        # Find free parameters (fk and fq)
        # ------------------------------------------------------

        fk = ( Zeff - (1-fZj)*Zq2/Zq - fZj2 ) / ( Zk**2 - Zk*Zq2/Zq)
        fq = ( 1 - fZj - fk*Zk ) / Zq

        if (fq<0).any():
            raise ValueError(f"Zeff cannot be reduced by changing ion {ion_pos}")

        # ------------------------------------------------------
        # Insert
        # ------------------------------------------------------

        fi_orig = self.derived["fi"][:, ion_pos]

        self.profiles["ni(10^19/m^3)"][:, ion_pos] = fk * self.profiles["ne(10^19/m^3)"]
        for i in quasineutral_ions:
            self.profiles["ni(10^19/m^3)"][:, i] = fq * self.profiles["ne(10^19/m^3)"]

        self.readSpecies()

        self.deriveQuantities()

        if enforceSameGradients:
            self.scaleAllThermalDensities()
            self.deriveQuantities()

        print(
            f'\t\t\t* Dilution changed from {fi_orig.mean():.2e} (vol avg) to { self.derived["fi"][:, ion_pos].mean():.2e} to achieve Zeff={self.derived["Zeff_vol"]:.3f} (fDT={self.derived["fmain"]:.3f}) [quasineutrality error = {self.derived["QN_Error"]:.1e}]',
        )


    def moveSpecie(self, pos=2, pos_new=1):
        """
        if (D,Z1,Z2), pos 1 pos_new 2-> (Z1,D,Z2)
        """

        position_to_moveFROM_in_profiles = pos - 1
        position_to_moveTO_in_profiles = pos_new - 1

        print(
            f'\t\t- Moving ion in position (of ions order, no zero) {pos} ({self.profiles["name"][position_to_moveFROM_in_profiles]}) to {pos_new}',
            typeMsg="i",
        )

        self.profiles["nion"] = np.array([f"{int(self.profiles['nion'][0])+1}"])

        for ikey in ["name", "mass", "z", "type", "ni(10^19/m^3)", "ti(keV)"]:
            if len(self.profiles[ikey].shape) > 1:
                axis = 1
                newly = self.profiles[ikey][:, position_to_moveFROM_in_profiles]
            else:
                axis = 0
                newly = self.profiles[ikey][position_to_moveFROM_in_profiles]
            self.profiles[ikey] = np.insert(
                self.profiles[ikey], position_to_moveTO_in_profiles, newly, axis=axis
            )

        self.readSpecies()
        self.deriveQuantities()

        if position_to_moveTO_in_profiles > position_to_moveFROM_in_profiles:
            self.remove([position_to_moveFROM_in_profiles + 1])
        else:
            self.remove([position_to_moveFROM_in_profiles + 2])

    def addSpecie(self, Z=5.0, mass=10.0, fi_vol=0.1, forcename=None):
        print(
            f"\t\t- Creating new specie with Z={Z}, mass={mass}, fi_vol={fi_vol}",
            typeMsg="i",
        )

        if forcename is None:
            forcename = "LUMPED"

        lab = "therm"
        nZ = fi_vol * self.profiles["ne(10^19/m^3)"]

        self.profiles["nion"] = np.array([f"{int(self.profiles['nion'][0])+1}"])
        self.profiles["name"] = np.append(self.profiles["name"], forcename)
        self.profiles["mass"] = np.append(self.profiles["mass"], mass)
        self.profiles["z"] = np.append(self.profiles["z"], Z)
        self.profiles["type"] = np.append(self.profiles["type"], f"[{lab}]")
        self.profiles["ni(10^19/m^3)"] = np.append(
            self.profiles["ni(10^19/m^3)"], np.transpose(np.atleast_2d(nZ)), axis=1
        )
        self.profiles["ti(keV)"] = np.append(
            self.profiles["ti(keV)"],
            np.transpose(np.atleast_2d(self.profiles["ti(keV)"][:, 0])),
            axis=1,
        )
        if "vtor(m/s)" in self.profiles:
            self.profiles["vtor(m/s)"] = np.append(
                self.profiles["vtor(m/s)"],
                np.transpose(np.atleast_2d(self.profiles["vtor(m/s)"][:, 0])),
                axis=1,
            )

        self.readSpecies()
        self.deriveQuantities()

    def correct(self, options={}, write=False, new_file=None):
        """
        if name= T D LUMPED, and I want to eliminate D, removeIons = [2]
        """

        recompute_ptot = options.get("recompute_ptot", True)  # Only done by default
        removeIons = options.get("removeIons", [])
        removeFast = options.get("removeFast", False)
        quasineutrality = options.get("quasineutrality", False)
        sameDensityGradients = options.get("sameDensityGradients", False)
        groupQIONE = options.get("groupQIONE", False)
        ensurePostiveGamma = options.get("ensurePostiveGamma", False)
        ensureMachNumber = options.get("ensureMachNumber", None)
        FastIsThermal = options.get("FastIsThermal", False)

        print("\t- Custom correction of input.gacode file has been requested")

        # ----------------------------------------------------------------------
        # Correct
        # ----------------------------------------------------------------------

        # Remove desired ions
        if len(removeIons) > 0:
            self.remove(removeIons)

        # Remove fast
        if removeFast:
            ions_fast = []
            for sp in range(len(self.Species)):
                if self.Species[sp]["S"] != "therm":
                    ions_fast.append(sp + 1)
            if len(ions_fast) > 0:
                print(
                    f"\t\t- Detected fast ions in positions {ions_fast}, removing them..."
                )
                self.remove(ions_fast)
        # Fast as thermal
        elif FastIsThermal:
            modified_num = 0
            for i in range(len(self.Species)):
                if self.Species[i]["S"] != "therm":
                    print(
                        f'\t\t- Specie {i} ({self.profiles["name"][i]}) was fast, but now it is considered thermal'
                    )
                    self.Species[i]["S"] = "therm"
                    self.profiles["type"][i] = "[therm]"
                    self.profiles["ti(keV)"][:, i] = self.profiles["ti(keV)"][:, 0]
                    modified_num += 1
            if modified_num > 0:
                print(
                    "\t- Making fast species as if they were thermal (to keep dilution effect and Qi-sum of fluxes)",
                    typeMsg="w",
                )

        # Correct LUMPED
        for i in range(len(self.profiles["name"])):
            if self.profiles["name"][i] in ["LUMPED", "None"]:
                name = ionName(
                    int(self.profiles["z"][i]), int(self.profiles["mass"][i])
                )
                if name is not None:
                    print(
                        f'\t\t- Ion in position #{i+1} was named LUMPED with Z={self.profiles["z"][i]}, now it is renamed to {name}',
                        typeMsg="i",
                    )
                    self.profiles["name"][i] = name
                else:
                    print(
                        f'\t\t- Ion in position #{i+1} was named LUMPED with Z={self.profiles["z"][i]}, but I could not find what element it is, so doing nothing',
                        typeMsg="w",
                    )

        # Correct qione
        if groupQIONE and (np.abs(self.profiles["qione(MW/m^3)"].sum()) > 1e-14):
            print('\t\t- Inserting "qione" into "qrfe"', typeMsg="i")
            self.profiles["qrfe(MW/m^3)"] += self.profiles["qione(MW/m^3)"]
            self.profiles["qione(MW/m^3)"] = self.profiles["qione(MW/m^3)"] * 0.0

        # Make all thermal ions have the same gradient as the electron density, by keeping volume average constant
        if sameDensityGradients:
            print(
                "\t\t- Making all thermal ions have the same a/Ln as electrons (making them an exact flat fraction)",
                typeMsg="i",
            )
            for sp in range(len(self.Species)):
                if self.Species[sp]["S"] == "therm":
                    self.profiles["ni(10^19/m^3)"][:, sp] = (
                        self.derived["fi_vol"][sp] * self.profiles["ne(10^19/m^3)"]
                    )

        # Enforce quasineutrality
        if quasineutrality:
            self.enforceQuasineutrality()

        print(f"\t\t\t* Quasineutrality error = {self.derived['QN_Error']:.1e}")

        # Recompute ptot
        if recompute_ptot:
            self.deriveQuantities()
            self.selfconsistentPTOT()

        # If I don't trust the negative particle flux in the core that comes from TRANSP...
        if ensurePostiveGamma:
            print("\t\t- Making particle flux always positive", typeMsg="i")
            self.profiles[self.varqpar] = self.profiles[self.varqpar].clip(0)
            self.profiles[self.varqpar2] = self.profiles[self.varqpar2].clip(0)

        # Mach
        if ensureMachNumber is not None:
            self.introduceRotationProfile(Mach_LF=ensureMachNumber)

        # ----------------------------------------------------------------------
        # Re-derive
        # ----------------------------------------------------------------------

        self.deriveQuantities()

        # ----------------------------------------------------------------------
        # Write
        # ----------------------------------------------------------------------
        if write:
            self.writeCurrentStatus(file=new_file)
            self.printInfo()

    def selfconsistentPTOT(self):
        print(
            f"\t\t* Recomputing ptot and inserting it as ptot(Pa), changed from p0 = {self.profiles['ptot(Pa)'][0] * 1e-3:.1f} to {self.derived['ptot_manual'][0]*1e+3:.1f} kPa",
            typeMsg="i",
        )
        self.profiles["ptot(Pa)"] = self.derived["ptot_manual"] * 1e6

    def enforceQuasineutrality(self):
        print(
            f"\t\t- Enforcing quasineutrality (error = {self.derived['QN_Error']:.1e})",
            typeMsg="i",
        )

        # What's the lack of quasineutrality?
        ni = self.profiles["ne(10^19/m^3)"] * 0.0
        for sp in range(len(self.Species)):
            ni += self.profiles["ni(10^19/m^3)"][:, sp] * self.profiles["z"][sp]
        ne_missing = self.profiles["ne(10^19/m^3)"] - ni

        # What ion to modify?
        if self.DTplasmaBool:
            print("\t\t\t* Enforcing quasineutrality by modifying D and T equally")
            prev_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Dion])
            self.profiles["ni(10^19/m^3)"][:, self.Dion] += ne_missing / 2
            self.profiles["ni(10^19/m^3)"][:, self.Tion] += ne_missing / 2
            new_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Dion])
        else:
            print(
                f"\t\t\t* Enforcing quasineutrality by modifying main ion (position #{self.Mion})"
            )
            prev_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Mion])
            self.profiles["ni(10^19/m^3)"][:, self.Mion] += ne_missing
            new_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Mion])

        print(
            f"\t\t\t\t- Changed on-axis density from n0 = {prev_on_axis:.2f} to {new_on_axis:.2f} ({100*(new_on_axis-prev_on_axis)/prev_on_axis:.1f}%)"
        )

        self.deriveQuantities()

    def calculateModeledQuantities(self):
        n20 = self.derived["ne_vol20"]
        p_atm = self.derived["ptot_manual_vol"] * 1e6 / 101325.0
        Psol_MW = self.derived["Psol"]
        R = self.profiles["rcentr(m)"][0]
        Bt = self.profiles["bcentr(T)"][0]
        q95 = self.derived["q95"]
        Bp = (
            self.derived["eps"] * Bt / q95
        )  # ----------------------------------- VERY ROUGH APPROXIMATION!!!!

        ne_LCFS, Te_LCFS, Lambda_q = PLASMAtools.evaluateLCFS_Lmode(
            n20, pressure_atm=p_atm, Psol_MW=Psol_MW, R=R, Bp=Bp, Bt=Bt, q95=q95
        )

        print(
            f"- ne_sep ~ {ne_LCFS*1E-20:.1f}    1E20/m3 ({(ne_LCFS*1E-20)/n20 *100:.1f}% from vol.avg.)"
        )
        print(
            f"- Te_sep ~ {Te_LCFS:.1f}  eV (using Psol = {Psol_MW:.1f}MW, p = {p_atm:.1f} atm --> lambda_q = {Lambda_q:.2f} mm)"
        )
        print(f"- Ti_sep ~ {2*Te_LCFS:.1f}  eV")

    def introduceRotationProfile(self, Mach_LF=1.0, new_file=None):
        print(f"\t- Enforcing Mach Number in LF of {Mach_LF}")
        self.deriveQuantities()
        Vtor_LF = PLASMAtools.constructVtorFromMach(
            Mach_LF, self.profiles["ti(keV)"][:, 0], self.derived["mbg"]
        )  # m/s

        self.profiles["w0(rad/s)"] = Vtor_LF / (self.derived["R_LF"])  # rad/s

        self.deriveQuantities()

        if new_file is not None:
            self.writeCurrentStatus(file=new_file)

    def introducePedestalProfile(
        self, rho_loc_top=0.95, folderWork="~/scratch/", p1_over_ptot=0.79, debug=False
    ):
        folderWork = IOtools.expandPath(folderWork)

        plasmastate = folderWork + "state.cdf"
        rhob = self.profiles["rho(-)"]
        psipol = self.derived["psi_pol_n"]
        rho = self.profiles["rho(-)"]
        ne = self.profiles["ne(10^19/m^3)"] * 1e19
        Te = self.profiles["te(keV)"] * 1e3
        Ti = self.profiles["ti(keV)"][:, 0] * 1e3
        PEDmodule.create_dummy_plasmastate(
            plasmastate, rho, rhob, psipol, ne, Te * 1e-3, Ti * 1e-3
        )

        ix = np.argmin(np.abs(self.profiles["rho(-)"] - rho_loc_top))

        width_top_psi = 1 - psipol[ix]
        netop = ne[ix]
        tetop = Te[ix]
        ptop = tetop * (netop * 1e-20) * 3.2e1 * 1e-6
        p1 = ptop * p1_over_ptot

        x, neP, TeP, TiP = PEDmodule.fit_pedestal_mtanh(
            width_top_psi,
            netop * 1e-20,
            p1 * 1e6,
            ptop * 1e6,
            plasmastate,
            folderWork=folderWork,
        )

        neP = neP[x <= 1.0]
        TeP = TeP[x <= 1.0]
        TiP = TiP[x <= 1.0]
        x = x[x <= 1.0]

        neP = np.interp(psipol, x, neP)
        TeP = np.interp(psipol, x, TeP)
        TiP = np.interp(psipol, x, TiP)

        ne_new = np.append(ne[:ix] * 1e-20, neP[ix:])
        Te_new = np.append(Te[:ix] * 1e-3, TeP[ix:])
        Ti_new = np.append(Ti[:ix] * 1e-3, TiP[ix:])

        if debug:
            fig, axs = plt.subplots(nrows=3)
            axs[0].plot(rho, Te * 1e-3, "r")
            axs[0].plot(rho, TeP, "g")
            axs[0].plot(rho, Te_new, "m")
            axs[1].plot(rho, Ti * 1e-3, "r")
            axs[1].plot(rho, TiP, "g")
            axs[1].plot(rho, Ti_new, "m")
            axs[2].plot(rho, ne * 1e-20, "r")
            axs[2].plot(rho, neP, "g")
            axs[2].plot(rho, ne_new, "m")

            plt.show()
            embed()

        # Introduce
        self.profiles["te(keV)"] = Te_new

        self.profiles["ti(keV)"][:, 0] = Ti_new
        self.makeAllThermalIonsHaveSameTemp()

        scaleFactor = ne_new / (ne * 1e-20)
        self.profiles["ne(10^19/m^3)"] = ne_new * 1e1
        self.scaleAllThermalDensities(scaleFactor=scaleFactor)

    def plot(
        self,
        axs1=None,
        axs2=None,
        axs3=None,
        axs4=None,
        axsFlows=None,
        axs6=None,
        axsImps=None,
        color="b",
        legYN=True,
        extralab="",
        fn=None,
        fnlab="",
        lsFlows="-",
        legFlows=True,
        showtexts=True,
        lastRhoGradients=0.89,
    ):
        if axs1 is None:
            if fn is None:
                from mitim_tools.misc_tools.GUItools import FigureNotebook

                self.fn = FigureNotebook("PROFILES Notebook", geometry="1600x1000")

            fig, fig2, fig3, fig4, fig5, fig6, fig7 = add_figures(self.fn, fnlab=fnlab)

            grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
            axs1 = [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[2, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 1]),
                fig.add_subplot(grid[2, 1]),
                fig.add_subplot(grid[0, 2]),
                fig.add_subplot(grid[1, 2]),
                fig.add_subplot(grid[2, 2]),
            ]

            
            grid = plt.GridSpec(3, 2, hspace=0.3, wspace=0.3)
            axs2 = [
                fig2.add_subplot(grid[0, 0]),
                fig2.add_subplot(grid[0, 1]),
                fig2.add_subplot(grid[1, 0]),
                fig2.add_subplot(grid[1, 1]),
                fig2.add_subplot(grid[2, 0]),
                fig2.add_subplot(grid[2, 1]),
            ]

            
            grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.5)
            ax00c = fig3.add_subplot(grid[0, 0])
            axs3 = [
                ax00c,
                fig3.add_subplot(grid[1, 0], sharex=ax00c),
                fig3.add_subplot(grid[2, 0], sharex=ax00c),
                fig3.add_subplot(grid[0, 1], sharex=ax00c),
                fig3.add_subplot(grid[1, 1], sharex=ax00c),
                fig3.add_subplot(grid[2, 1], sharex=ax00c),
                fig3.add_subplot(grid[0, 2], sharex=ax00c),
                fig3.add_subplot(grid[1, 2], sharex=ax00c),
                fig3.add_subplot(grid[2, 2], sharex=ax00c),
                fig3.add_subplot(grid[0, 3], sharex=ax00c),
                fig3.add_subplot(grid[1, 3], sharex=ax00c),
                fig3.add_subplot(grid[2, 3], sharex=ax00c),
            ]

            
            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
            axs4 = [
                fig4.add_subplot(grid[0, 0]),
                fig4.add_subplot(grid[1, 0]),
                fig4.add_subplot(grid[0, 1]),
                fig4.add_subplot(grid[1, 1]),
                fig4.add_subplot(grid[0, 2]),
                fig4.add_subplot(grid[1, 2]),
            ]

            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

            axsFlows = [
                fig5.add_subplot(grid[0, 0]),
                fig5.add_subplot(grid[1, 0]),
                fig5.add_subplot(grid[0, 1]),
                fig5.add_subplot(grid[0, 2]),
                fig5.add_subplot(grid[1, 1]),
                fig5.add_subplot(grid[1, 2]),
            ]

            
            grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
            axs6 = [
                fig6.add_subplot(grid[0, 0]),
                fig6.add_subplot(grid[:, 1]),
                fig6.add_subplot(grid[0, 2]),
                fig6.add_subplot(grid[1, 0]),
                fig6.add_subplot(grid[1, 2]),
                fig6.add_subplot(grid[0, 3]),
                fig6.add_subplot(grid[1, 3]),
            ]

            
            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
            axsImps = [
                fig7.add_subplot(grid[0, 0]),
                fig7.add_subplot(grid[0, 1]),
                fig7.add_subplot(grid[0, 2]),
                fig7.add_subplot(grid[1, 0]),
                fig7.add_subplot(grid[1, 1]),
                fig7.add_subplot(grid[1, 2]),
            ]

        [ax00, ax10, ax20, ax01, ax11, ax21, ax02, ax12, ax22] = axs1
        [ax00b, ax01b, ax10b, ax11b, ax20b, ax21b] = axs2
        [
            ax00c,
            ax10c,
            ax20c,
            ax01c,
            ax11c,
            ax21c,
            ax02c,
            ax12c,
            ax22c,
            ax03c,
            ax13c,
            ax23c,
        ] = axs3

        lw = 1
        fs = 6
        rho = self.profiles["rho(-)"]

        lines = ["-", "--", "-.", ":", "-", "--", "-."]

        self.plot_temps(ax=ax00, leg=legYN, col=color, lw=lw, fs=fs, extralab=extralab)
        self.plot_dens(ax=ax01, leg=legYN, col=color, lw=lw, fs=fs, extralab=extralab)
        # self.plot_powers(axs=[ax00b,ax01b,ax10b],leg=True,col='b',lw=2,fs=fs)

        ax = ax10
        cont = 0
        for i in range(len(self.Species)):
            if self.Species[i]["S"] == "therm":
                var = self.profiles["ti(keV)"][:, i]
                ax.plot(
                    rho,
                    var,
                    lw=lw,
                    ls=lines[cont],
                    c=color,
                    label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
                )
                cont += 1
        varL = "Thermal $T_i$ (keV)"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0);
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax20
        cont = 0
        for i in range(len(self.Species)):
            if self.Species[i]["S"] == "fast":
                var = self.profiles["ti(keV)"][:, i]
                ax.plot(
                    rho,
                    var,
                    lw=lw,
                    ls=lines[cont],
                    c=color,
                    label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
                )
                cont += 1
        varL = "Fast $T_i$ (keV)"
        ax.plot(
            rho,
            self.profiles["ti(keV)"][:, 0],
            lw=0.5,
            ls="-",
            alpha=0.5,
            c=color,
            label=extralab + "$T_{i,1}$",
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0);
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax11
        cont = 0
        for i in range(len(self.Species)):
            if self.Species[i]["S"] == "therm":
                var = self.profiles["ni(10^19/m^3)"][:, i] * 1e-1
                ax.plot(
                    rho,
                    var,
                    lw=lw,
                    ls=lines[cont],
                    c=color,
                    label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
                )
                cont += 1
        varL = "Thermal $n_i$ ($10^{20}/m^3$)"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0);
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax21
        cont = 0
        for i in range(len(self.Species)):
            if self.Species[i]["S"] == "fast":
                var = self.profiles["ni(10^19/m^3)"][:, i] * 1e-1 * 1e5
                ax.plot(
                    rho,
                    var,
                    lw=lw,
                    ls=lines[cont],
                    c=color,
                    label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
                )
                cont += 1
        varL = "Fast $n_i$ ($10^{15}/m^3$)"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0);
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax02
        var = self.profiles["w0(rad/s)"]
        ax.plot(rho, var, lw=lw, ls="-", c=color)
        varL = "$\\omega_{0}$ (rad/s)"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax12
        var = self.profiles["ptot(Pa)"] * 1e-6
        ax.plot(rho, var, lw=lw, ls="-", c=color, label=extralab + "ptot")
        if "ptot_manual" in self.derived:
            ax.plot(
                rho,
                self.derived["ptot_manual"],
                lw=lw,
                ls="--",
                c=color,
                label=extralab + "check",
            )
            # ax.plot(rho,np.abs(var-self.derived['ptot_manual']),lw=lw,ls='-.',c=color,label=extralab+'diff')
        varL = "$p$ (MPa)"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        # ax.set_ylim(bottom=0)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax00b
        varL = "$MW/m^3$"
        cont = 0
        var = -self.profiles["qei(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "i->e", c=color)
        cont += 1
        if "qrfe(MW/m^3)" in self.profiles:
            var = self.profiles["qrfe(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "rf", c=color)
            cont += 1
        if "qfuse(MW/m^3)" in self.profiles:
            var = self.profiles["qfuse(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "fus", c=color)
            cont += 1
        if "qbeame(MW/m^3)" in self.profiles:
            var = self.profiles["qbeame(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "beam", c=color)
            cont += 1
        if "qione(MW/m^3)" in self.profiles:
            var = self.profiles["qione(MW/m^3)"]
            ax.plot(
                rho, var, lw=lw / 2, ls=lines[cont], label=extralab + "extra", c=color
            )
            cont += 1
        if "qohme(MW/m^3)" in self.profiles:
            var = self.profiles["qohme(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "ohmic", c=color)
            cont += 1

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)
        ax.set_title("Electron Power Density")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax01b
        if "varqmom" not in self.__dict__:
            self.varqmom = "qmom(N/m^2)"
            self.profiles[self.varqmom] = self.profiles["rho(-)"] * 0.0

        ax.plot(rho, self.profiles[self.varqmom], lw=lw, ls="-", c=color)
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$N/m^2$, $J/m^3$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)
        ax.set_title("Momentum Source Density")

        ax = ax21b
        ax.plot(
            rho, self.derived["qe_MWm2"], lw=lw, ls="-", label=extralab + "qe", c=color
        )
        ax.plot(
            rho, self.derived["qi_MWm2"], lw=lw, ls="--", label=extralab + "qi", c=color
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("Heat Flux ($MW/m^2$)")
        if legYN:
            ax.legend(loc="lower left", fontsize=fs)
        ax.set_title("Flux per unit area (gacode: P/V')")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax21b.twinx()
        ax.plot(
            rho,
            self.derived["ge_10E20m2"],
            lw=lw,
            ls="-.",
            label=extralab + "$\\Gamma_e$",
            c=color,
        )
        ax.set_ylabel("Particle Flux ($10^{20}/m^2/s$)")
        if legYN:
            ax.legend(loc="lower right", fontsize=fs)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax20b
        varL = "$Q_{rad}$ ($MW/m^3$)"
        if "qbrem(MW/m^3)" in self.profiles:
            var = self.profiles["qbrem(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls="-", label=extralab + "brem", c=color)
        if "qline(MW/m^3)" in self.profiles:
            var = self.profiles["qline(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls="--", label=extralab + "line", c=color)
        if "qsync(MW/m^3)" in self.profiles:
            var = self.profiles["qsync(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=":", label=extralab + "sync", c=color)

        var = self.derived["qrad"]
        ax.plot(rho, var, lw=lw * 1.5, ls="-", label=extralab + "Total", c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0);
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)
        ax.set_title("Radiation Contributions")
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax10b
        varL = "$MW/m^3$"
        cont = 0
        var = self.profiles["qei(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "e->i", c=color)
        cont += 1
        if "qrfi(MW/m^3)" in self.profiles:
            var = self.profiles["qrfi(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "rf", c=color)
            cont += 1
        if "qfusi(MW/m^3)" in self.profiles:
            var = self.profiles["qfusi(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "fus", c=color)
            cont += 1
        if "qbeami(MW/m^3)" in self.profiles:
            var = self.profiles["qbeami(MW/m^3)"]
            ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "beam", c=color)
            cont += 1
        if "qioni(MW/m^3)" in self.profiles:
            var = self.profiles["qioni(MW/m^3)"]
            ax.plot(
                rho, var, lw=lw / 2, ls=lines[cont], label=extralab + "extra", c=color
            )
            cont += 1

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)
        ax.set_title("Ion Power Density")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        """
		Note that in prgen_map_plasmastate, that variable:
		expro_qpar_beam(i) = plst_sn_trans(i-1)/dvol

		Note that in prgen_read_plasmastate, that variable:
		  ! Particle source
  			err = nf90_inq_varid(ncid,trim('sn_trans'),varid)
  			err = nf90_get_var(ncid,varid,plst_sn_trans(1:nx-1))
  			plst_sn_trans(nx) = 0.0

		Note that in the plasmastate file, the variable "sn_trans":

		    long_name:      particle transport (loss)
		    units:          #/sec
		    component:      PLASMA
		    section:        STATE_PROFILES
		    specification:  R|units=#/sec|step*dV sn_trans(~nrho,0:nspec_th)

		So, this means that expro_qpar_beam is in units of #/sec/m^3, meaning that
		it is a particle flux DENSITY. It therefore requires volume integral and
		divide by surface to produce a flux.

		The units of this qpar_beam column is NOT MW/m^3. In the gacode source codes
		they also say that those units are wrong.

		"""

        ax = ax11b
        cont = 0
        var = self.profiles[self.varqpar] * 1e-20
        ax.plot(rho, var, lw=lw, ls=lines[0], c=color, label=extralab + "beam")
        var = self.profiles[self.varqpar2] * 1e-20
        ax.plot(rho, var, lw=lw, ls=lines[1], c=color, label=extralab + "wall")

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0);
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$10^{20}m^{-3}s^{-1}$")
        ax.set_title("Particle Source Density")
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax00c
        varL = "cos Shape Params"
        yl = 0
        cont = 0

        for i, s in enumerate(self.shape_cos):
            if s is not None:
                valmax = np.abs(s).max()
                if valmax > 1e-10:
                    lab = f"c{i}"
                    ax.plot(rho, s, lw=lw, ls=lines[cont], label=lab, c=color)
                    cont += 1

                yl = np.max([yl, valmax])

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)

        

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        if legYN:
            ax.legend(loc="best", fontsize=fs)

        ax = ax01c
        varL = "sin Shape Params"
        cont = 0
        for i, s in enumerate(self.shape_sin):
            if s is not None:
                valmax = np.abs(s).max()
                if valmax > 1e-10:
                    lab = f"s{i}"
                    ax.plot(rho, s, lw=lw, ls=lines[cont], label=lab, c=color)
                    cont += 1

                yl = np.max([yl, valmax])

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax02c
        var = self.profiles["q(-)"]
        ax.plot(rho, var, lw=lw, ls="-", c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0)
        ax.set_ylabel("q")

        ax.axhline(y=1, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0.0)

        

        ax = ax12c
        var = self.profiles["polflux(Wb/radian)"]
        ax.plot(rho, var, lw=lw, ls="-", c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("Poloidal $\\psi$ ($Wb/rad$)")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax10c

        var = self.profiles["rho(-)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])

        ax.set_xlabel("$\\rho$")
        # ax.set_ylim(bottom=0)
        ax.set_ylabel("$\\rho$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax11c

        var = self.profiles["rmin(m)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$r_{min}$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax20c

        var = self.profiles["rmaj(m)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$R_{maj}$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax21c

        var = self.profiles["zmag(m)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        yl = np.max([0.1, np.max(np.abs(var))])
        ax.set_ylim([-yl, yl])
        ax.set_ylabel("$Z_{maj}$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax22c

        var = self.profiles["kappa(-)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$\\kappa$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=1)

        ax = ax03c

        var = self.profiles["delta(-)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$\\delta$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax13c

        var = self.profiles["zeta(-)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("zeta")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax23c

        var = self.profiles["johm(MA/m^2)"]
        ax.plot(rho, var, "-", lw=lw, c=color, label=extralab + "$J_{OH}$")
        var = self.profiles["jbs(MA/m^2)"]
        ax.plot(rho, var, "--", lw=lw, c=color, label=extralab + "$J_{BS,par}$")
        var = self.profiles["jbstor(MA/m^2)"]
        ax.plot(rho, var, "-.", lw=lw, c=color, label=extralab + "$J_{BS,tor}$")

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("J ($MA/m^2$)")
        if legYN:
            ax.legend(loc="best", prop={"size": 7})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        # Derived
        self.plotGradients(
            axs4, color=color, lw=lw, lastRho=lastRhoGradients, label=extralab
        )

        # Others
        ax = axs6[0]
        ax.plot(self.profiles["rho(-)"], self.derived["dw0dr"] * 1e-5, c=color, lw=lw)
        ax.set_ylabel("$-d\\omega_0/dr$ (krad/s/cm)")
        ax.set_xlabel("$\\rho$")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)
        ax.axhline(y=0, lw=1.0, c="k", ls="--")

        ax = axs6[2]
        ax.plot(self.profiles["rho(-)"], self.derived["q_fus"], c=color, lw=lw)
        ax.set_ylabel("$q_{fus}$ ($MW/m^3$)")
        ax.set_xlabel("$\\rho$")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axs6[3]
        ax.plot(self.profiles["rho(-)"], self.derived["q_fus_MWmiller"], c=color, lw=lw)
        ax.set_ylabel("$P_{fus}$ ($MW$)")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axs6[4]
        ax.plot(self.profiles["rho(-)"], self.derived["tite"], c=color, lw=lw)
        ax.set_ylabel("$T_i/T_e$")
        ax.set_xlabel("$\\rho$")
        ax.set_xlim([0, 1])
        ax.axhline(y=1, ls="--", lw=1.0, c="k")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = axs6[5]
        if "MachNum" in self.derived:
            ax.plot(self.profiles["rho(-)"], self.derived["MachNum"], c=color, lw=lw)
        ax.set_ylabel("Mach Number")
        ax.set_xlabel("$\\rho$")
        ax.set_xlim([0, 1])
        ax.axhline(y=0, ls="--", c="k", lw=0.5)
        ax.axhline(y=1, ls="--", c="k", lw=0.5)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = axs6[6]
        safe_division = np.divide(
            self.derived["qi_MWm2"],
            self.derived["qe_MWm2"],
            where=self.derived["qe_MWm2"] != 0,
            out=np.full_like(self.derived["qi_MWm2"], np.nan),
        )
        ax.plot(
            self.profiles["rho(-)"],
            safe_division,
            c=color,
            lw=lw,
            label=extralab + "Q_i/Q_e",
        )
        safe_division = np.divide(
            self.derived["qi_aux_MWmiller"],
            self.derived["qe_MWm2"],
            where=self.derived["qe_aux_MWmiller"] != 0,
            out=np.full_like(self.derived["qi_aux_MWmiller"], np.nan),
        )
        ax.plot(
            self.profiles["rho(-)"],
            safe_division,
            c=color,
            lw=lw,
            ls="--",
            label=extralab + "P_i/P_e",
        )
        ax.set_ylabel("Power ratios")
        ax.set_xlabel("$\\rho$")
        ax.set_xlim([0, 1])
        ax.axhline(y=1.0, ls="--", c="k", lw=1.0)
        GRAPHICStools.addDenseAxis(ax)
        # GRAPHICStools.autoscale_y(ax,bottomy=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=fs)

        # Final
        if axsFlows is not None:
            self.plotBalance(
                axs=axsFlows, ls=lsFlows, leg=legFlows, showtexts=showtexts
            )

        # Geometry
        ax = axs6[1]
        self.plotGeometry(ax=ax, color=color)

        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        GRAPHICStools.addDenseAxis(ax)

        # Impurities
        ax = axsImps[0]
        for i in range(len(self.Species)):
            var = (
                self.profiles["ni(10^19/m^3)"][:, i]
                / self.profiles["ni(10^19/m^3)"][0, i]
            )
            ax.plot(
                rho,
                var,
                lw=lw,
                ls=lines[i],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
        varL = "$n_i/n_{i,0}$"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axsImps[1]
        for i in range(len(self.Species)):
            var = self.derived["fi"][:, i]
            ax.plot(
                rho,
                var,
                lw=lw,
                ls=lines[i],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
        varL = "$f_i$"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        ax.set_ylim([0, 1])
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axsImps[2]

        lastRho = 0.9

        ix = np.argmin(np.abs(self.profiles["rho(-)"] - lastRho)) + 1
        ax.plot(
            rho[:ix], self.derived["aLne"][:ix], lw=lw * 3, ls="-", c=color, label="e"
        )
        for i in range(len(self.Species)):
            var = self.derived["aLni"][:, i]
            ax.plot(
                rho[:ix],
                var[:ix],
                lw=lw,
                ls=lines[i],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
        varL = "$a/L_{ni}$"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        if legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axsImps[5]

        ax = axsImps[3]
        ax.plot(self.profiles["rho(-)"], self.derived["Zeff"], c=color, lw=lw)
        ax.set_ylabel("$Z_{eff}$")
        ax.set_xlabel("$\\rho$")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axsImps[4]
        cont = 0
        if "vtor(m/s)" in self.profiles:
            for i in range(len(self.Species)):
                try:  # REMOVE FOR FUTURE
                    var = self.profiles["vtor(m/s)"][:, i] * 1e-3
                    ax.plot(
                        rho,
                        var,
                        lw=lw,
                        ls=lines[cont],
                        c=color,
                        label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
                    )
                    cont += 1
                except:
                    break
        varL = "$V_{tor}$ (km/s)"
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        if "vtor(m/s)" in self.profiles and legYN:
            ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

    def plotGradients(
        self,
        axs4,
        color="b",
        lw=1.0,
        label="",
        ls="-o",
        lastRho=0.89,
        ms=2,
        alpha=1.0,
        useRoa=False,
        RhoLocationsPlot=[],
        plotImpurity=None,
        plotRotation=False,
        autoscale=True,
    ):
        if axs4 is None:
            plt.ion()
            fig, axs = plt.subplots(
                ncols=3 + int(plotImpurity is not None) + int(plotRotation),
                nrows=2,
                figsize=(12, 5),
            )

            axs4 = []
            for i in range(axs.shape[-1]):
                axs4.append(axs[0, i])
                axs4.append(axs[1, i])

        ix = np.argmin(np.abs(self.profiles["rho(-)"] - lastRho)) + 1

        xcoord = self.profiles["rho(-)"] if (not useRoa) else self.derived["roa"]
        labelx = "$\\rho$" if (not useRoa) else "$r/a$"

        ax = axs4[0]
        ax.plot(
            xcoord,
            self.profiles["te(keV)"],
            ls,
            c=color,
            lw=lw,
            label=label,
            markersize=ms,
            alpha=alpha,
        )
        ax = axs4[2]
        ax.plot(
            xcoord,
            self.profiles["ti(keV)"][:, 0],
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )
        ax = axs4[4]
        ax.plot(
            xcoord,
            self.profiles["ne(10^19/m^3)"] * 1e-1,
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )

        if "derived" in self.__dict__:
            ax = axs4[1]
            ax.plot(
                xcoord[:ix],
                self.derived["aLTe"][:ix],
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )
            ax = axs4[3]
            ax.plot(
                xcoord[:ix],
                self.derived["aLTi"][:ix, 0],
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )
            ax = axs4[5]
            ax.plot(
                xcoord[:ix],
                self.derived["aLne"][:ix],
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )

        for ax in axs4:
            ax.set_xlim([0, 1])

        ax = axs4[0]
        ax.set_ylabel("$T_e$ (keV)")
        ax.set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        ax.legend(loc="best", fontsize=7)
        ax = axs4[2]
        ax.set_ylabel("$T_i$ (keV)")
        ax.set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        ax = axs4[4]
        ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        ax.set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axs4[1]
        ax.set_ylabel("$a/L_{Te}$")
        ax.set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        ax = axs4[3]
        ax.set_ylabel("$a/L_{Ti}$")
        ax.set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        ax = axs4[5]
        ax.set_ylabel("$a/L_{ne}$")
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        ax.set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)

        cont = 0
        if plotImpurity is not None:
            axs4[6 + cont].plot(
                xcoord,
                self.profiles["ni(10^19/m^3)"][:, plotImpurity] * 1e-1,
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )
            axs4[6 + cont].set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
            axs4[6].set_xlabel(labelx)
            if autoscale:
                GRAPHICStools.autoscale_y(ax, bottomy=0)
            if "derived" in self.__dict__:
                axs4[7 + cont].plot(
                    xcoord[:ix],
                    self.derived["aLni"][:ix, plotImpurity],
                    ls,
                    c=color,
                    lw=lw,
                    markersize=ms,
                    alpha=alpha,
                )
            axs4[7 + cont].set_ylabel("$a/L_{nZ}$")
            axs4[7 + cont].axhline(y=0, ls="--", lw=0.5, c="k")
            axs4[7 + cont].set_xlabel(labelx)
            if autoscale:
                GRAPHICStools.autoscale_y(ax, bottomy=0)
            cont += 2

        if plotRotation:
            axs4[6 + cont].plot(
                xcoord,
                self.profiles["w0(rad/s)"] * 1e-3,
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )
            axs4[6 + cont].set_ylabel("$w_0$ (krad/s)")
            axs4[6 + cont].set_xlabel(labelx)
            if "derived" in self.__dict__:
                axs4[7 + cont].plot(
                    xcoord[:ix],
                    self.derived["dw0dr"][:ix] * 1e-5,
                    ls,
                    c=color,
                    lw=lw,
                    markersize=ms,
                    alpha=alpha,
                )
            axs4[7 + cont].set_ylabel("-$d\\omega_0/dr$ (krad/s/cm)")
            axs4[7 + cont].axhline(y=0, ls="--", lw=0.5, c="k")
            axs4[7 + cont].set_xlabel(labelx)
            if autoscale:
                GRAPHICStools.autoscale_y(ax, bottomy=0)
            cont += 2

        for x0 in RhoLocationsPlot:
            ix = np.argmin(np.abs(self.profiles["rho(-)"] - x0))
            for ax in axs4:
                ax.axvline(x=xcoord[ix], ls="--", lw=0.5, c=color)

        for i in range(len(axs4)):
            ax = axs4[i]
            GRAPHICStools.addDenseAxis(ax)

    def plotBalance(self, axs=None, limits=None, ls="-", leg=True, showtexts=True):
        if axs is None:
            fig1 = plt.figure()
            grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

            axs = [
                fig1.add_subplot(grid[0, 0]),
                fig1.add_subplot(grid[1, 0]),
                fig1.add_subplot(grid[0, 1]),
                fig1.add_subplot(grid[0, 2]),
                fig1.add_subplot(grid[1, 1]),
                fig1.add_subplot(grid[1, 2]),
            ]

        # Profiles

        ax = axs[0]
        axT = axs[1]
        roa = self.profiles["rmin(m)"] / self.profiles["rmin(m)"][-1]
        Te = self.profiles["te(keV)"]
        ne = self.profiles["ne(10^19/m^3)"] * 1e-1
        ni = self.profiles["ni(10^19/m^3)"] * 1e-1
        niT = np.sum(ni, axis=1)
        Ti = self.profiles["ti(keV)"][:, 0]
        ax.plot(roa, Te, lw=2, c="r", label="$T_e$" if leg else "", ls=ls)
        ax.plot(roa, Ti, lw=2, c="b", label="$T_i$" if leg else "", ls=ls)
        axT.plot(roa, ne, lw=2, c="m", label="$n_e$" if leg else "", ls=ls)
        axT.plot(roa, niT, lw=2, c="c", label="$\\sum n_i$" if leg else "", ls=ls)
        if limits is not None:
            [roa_first, roa_last] = limits
            ax.plot(roa_last, np.interp(roa_last, roa, Te), "s", c="r", markersize=3)
            ax.plot(roa_first, np.interp(roa_first, roa, Te), "s", c="r", markersize=3)
            ax.plot(roa_last, np.interp(roa_last, roa, Ti), "s", c="b", markersize=3)
            ax.plot(roa_first, np.interp(roa_first, roa, Ti), "s", c="b", markersize=3)
            axT.plot(roa_last, np.interp(roa_last, roa, ne), "s", c="m", markersize=3)
            axT.plot(roa_first, np.interp(roa_first, roa, ne), "s", c="m", markersize=3)

        ax.set_xlabel("r/a")
        ax.set_xlim([0, 1])
        axT.set_xlabel("r/a")
        axT.set_xlim([0, 1])
        ax.set_ylabel("$T$ (keV)")
        ax.set_ylim(bottom=0)
        axT.set_ylabel("$n$ ($10^{20}m^{-3}$)")
        axT.set_ylim(bottom=0)
        # axT.set_ylim([0,np.max(ne)*1.5])
        ax.legend()
        axT.legend()
        ax.set_title("Final Temperature profiles")
        axT.set_title("Final Density profiles")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        GRAPHICStools.addDenseAxis(axT)
        GRAPHICStools.autoscale_y(axT, bottomy=0)

        if showtexts:
            if self.derived["Q"] > 0.005:
                ax.text(
                    0.05,
                    0.05,
                    f"Pfus = {self.derived['Pfus']:.1f}MW, Q = {self.derived['Q']:.2f}",
                    color="k",
                    fontsize=10,
                    fontweight="normal",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    rotation=0,
                    transform=ax.transAxes,
                )

            axT.text(
                0.05,
                0.4,
                "ne_20 = {0:.1f} (fG = {1:.2f}), Zeff = {2:.1f}".format(
                    self.derived["ne_vol20"],
                    self.derived["fG"],
                    self.derived["Zeff_vol"],
                ),
                color="k",
                fontsize=10,
                fontweight="normal",
                horizontalalignment="left",
                verticalalignment="bottom",
                rotation=0,
                transform=axT.transAxes,
            )

        # F
        ax = axs[2]
        P = (
            self.derived["qe_fus_MWmiller"]
            + self.derived["qe_aux_MWmiller"]
            + -self.derived["qe_rad_MWmiller"]
            + -self.derived["qe_exc_MWmiller"]
        )

        ax.plot(
            roa,
            -self.derived["qe_MWmiller"],
            c="g",
            lw=2,
            label="$P_{e}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qe_fus_MWmiller"],
            c="r",
            lw=2,
            label="$P_{fus,e}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qe_aux_MWmiller"],
            c="b",
            lw=2,
            label="$P_{aux,e}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            -self.derived["qe_exc_MWmiller"],
            c="m",
            lw=2,
            label="$P_{exc,e}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            -self.derived["qe_rad_MWmiller"],
            c="c",
            lw=2,
            label="$P_{rad,e}$" if leg else "",
            ls=ls,
        )
        ax.plot(roa, -P, lw=1, c="y", label="sum" if leg else "", ls=ls)

        # Pe = self.profiles['te(keV)']*1E3*e_J*self.profiles['ne(10^19/m^3)']*1E-1*1E20 *1E-6
        # ax.plot(roa,Pe,ls='-',lw=3,alpha=0.1,c='k',label='$W_e$ (MJ/m^3)')

        ax.plot(
            roa,
            -self.derived["ce_MWmiller"],
            c="k",
            lw=1,
            label="($P_{conv,e}$)" if leg else "",
        )

        ax.set_xlabel("r/a")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P$ (MW)")
        # ax.set_ylim(bottom=0)
        ax.set_title("Electron Thermal Flows")
        ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
        )

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axs[3]
        P = (
            self.derived["qi_fus_MWmiller"]
            + self.derived["qi_aux_MWmiller"]
            + self.derived["qe_exc_MWmiller"]
        )

        ax.plot(
            roa,
            -self.derived["qi_MWmiller"],
            c="g",
            lw=2,
            label="$P_{i}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qi_fus_MWmiller"],
            c="r",
            lw=2,
            label="$P_{fus,i}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qi_aux_MWmiller"],
            c="b",
            lw=2,
            label="$P_{aux,i}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qe_exc_MWmiller"],
            c="m",
            lw=2,
            label="$P_{exc,i}$" if leg else "",
            ls=ls,
        )
        ax.plot(roa, -P, lw=1, c="y", label="sum" if leg else "", ls=ls)

        # Pi = self.profiles['ti(keV)'][:,0]*1E3*e_J*self.profiles['ni(10^19/m^3)'][:,0]*1E-1*1E20 *1E-6
        # ax.plot(roa,Pi,ls='-',lw=3,alpha=0.1,c='k',label='$W_$ (MJ/m^3)')

        ax.set_xlabel("r/a")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P$ (MW)")
        # ax.set_ylim(bottom=0)
        ax.set_title("Ion Thermal Flows")
        ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
        )

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        # F
        ax = axs[4]

        ax.plot(
            roa,
            self.derived["ge_10E20miller"],
            c="g",
            lw=2,
            label="$\\Gamma_{e}$" if leg else "",
            ls=ls,
        )
        # ax.plot(roa,self.profiles['ne(10^19/m^3)']*1E-1,lw=3,alpha=0.1,c='k',label='$n_e$ ($10^{20}/m^3$)' if leg else '',ls=ls)

        ax.set_xlabel("r/a")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\Gamma$ ($10^{20}/s$)")
        ax.set_title("Particle Flows")
        ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
        )

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        # TOTAL
        ax = axs[5]
        P = (
            self.derived["qOhm_MWmiller"]
            + self.derived["qRF_MWmiller"]
            + self.derived["qFus_MWmiller"]
            + -self.derived["qe_rad_MWmiller"]
            + self.derived["qz_MWmiller"]
            + self.derived["qBEAM_MWmiller"]
        )

        ax.plot(
            roa,
            -self.derived["q_MWmiller"],
            c="g",
            lw=2,
            label="$P$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qOhm_MWmiller"],
            c="k",
            lw=2,
            label="$P_{Oh}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qRF_MWmiller"],
            c="b",
            lw=2,
            label="$P_{RF}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qBEAM_MWmiller"],
            c="pink",
            lw=2,
            label="$P_{NBI}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qFus_MWmiller"],
            c="r",
            lw=2,
            label="$P_{fus}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            -self.derived["qe_rad_MWmiller"],
            c="c",
            lw=2,
            label="$P_{rad}$" if leg else "",
            ls=ls,
        )
        ax.plot(
            roa,
            self.derived["qz_MWmiller"],
            c="orange",
            lw=1,
            label="$P_{ionz.}$" if leg else "",
            ls=ls,
        )

        # P = Pe+Pi
        # ax.plot(roa,P,ls='-',lw=3,alpha=0.1,c='k',label='$W$ (MJ)')

        ax.plot(roa, -P, lw=1, c="y", label="sum" if leg else "", ls=ls)
        ax.set_xlabel("r/a")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$P$ (MW)")
        # ax.set_ylim(bottom=0)
        ax.set_title("Total Thermal Flows")

        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
        )

        ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
        # GRAPHICStools.drawLineWithTxt(ax,0.0,label='',orientation='vertical',color='k',lw=1,ls='--',alpha=1.0,fontsize=10,fromtop=0.85,fontweight='normal',
        # 				verticalalignment='bottom',horizontalalignment='left',separation=0)

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

    def plot_temps(self, ax=None, leg=False, col="b", lw=2, extralab="", fs=10):
        if ax is None:
            fig, ax = plt.subplots()

        rho = self.profiles["rho(-)"]

        var = self.profiles["te(keV)"]
        varL = "$T_e$ , $T_i$ (keV)"
        if leg:
            lab = extralab + "e"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="-", label=lab, c=col)
        var = self.profiles["ti(keV)"][:, 0]
        if leg:
            lab = extralab + "i"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="--", label=lab, c=col)
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        GRAPHICStools.autoscale_y(ax, bottomy=0)
        ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)

    def plot_dens(self, ax=None, leg=False, col="b", lw=2, extralab="", fs=10):
        if ax is None:
            fig, ax = plt.subplots()

        rho = self.profiles["rho(-)"]

        var = self.profiles["ne(10^19/m^3)"] * 1e-1
        varL = "$n_e$ ($10^{20}/m^3$)"
        if leg:
            lab = extralab + "e"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="-", label=lab, c=col)
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        GRAPHICStools.autoscale_y(ax, bottomy=0)
        ax.legend(loc="best", fontsize=fs)

        GRAPHICStools.addDenseAxis(ax)

    def plot_powers(self, axs=None, leg=False, col="b", lw=2, extralab="", fs=10):
        if axs is None:
            fig, axs = plt.subplots(ncols=3)

        rho = self.profiles["rho(-)"]

        ax = axs[0]
        var = self.profiles["pow_e(MW)"]
        varL = "$Q_e$, $Q_i$ (MW)"
        if leg:
            lab = extralab + "e"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="-", label=lab, c=col)
        var = self.profiles["pow_i(MW)"]
        if leg:
            lab = "i"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="--", label=lab, c=col)
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        ax.legend(loc="best", fontsize=fs)
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = axs[1]
        var = self.profiles["pow_e(MW)"]
        varL = "$Q_e$ (MW)"
        if leg:
            lab = extralab + "total"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="-", label=lab, c=col)
        var1 = self.profiles["pow_e_aux(MW)"]
        if leg:
            lab = extralab + "aux"
        else:
            lab = ""
        ax.plot(rho, var1, lw=lw, ls="--", label=lab, c=col)
        var2 = self.profiles["pow_e_fus(MW)"]
        if leg:
            lab = extralab + "fus"
        else:
            lab = ""
        ax.plot(rho, var2, lw=lw, ls="-.", label=lab, c=col)
        var3 = (
            self.profiles["pow_e_sync(MW)"]
            + self.profiles["pow_e_brem(MW)"]
            + self.profiles["pow_e_line(MW)"]
        )
        if leg:
            lab = extralab + "rad"
        else:
            lab = ""
        ax.plot(rho, var3, lw=lw, ls=":", label=lab, c=col)
        var4 = self.profiles["pow_ei(MW)"]
        if leg:
            lab = extralab + "e->i"
        else:
            lab = ""
        ax.plot(rho, -var4, lw=lw / 2, ls="-", label=lab, c=col)
        # ax.plot(rho,var1+var2-var3-var4,lw=lw,ls='--',c='y',label='check')
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        ax.legend(loc="best", fontsize=fs)
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = axs[2]
        var = self.profiles["pow_i(MW)"]
        varL = "$Q_i$ (MW)"
        if leg:
            lab = extralab + "total"
        else:
            lab = ""
        ax.plot(rho, var, lw=lw, ls="-", label=lab, c=col)
        var1 = self.profiles["pow_i_aux(MW)"]
        if leg:
            lab = extralab + "aux"
        else:
            lab = ""
        ax.plot(rho, var1, lw=lw, ls="--", label=lab, c=col)
        var2 = self.profiles["pow_i_fus(MW)"]
        if leg:
            lab = extralab + "fus"
        else:
            lab = ""
        ax.plot(rho, var2, lw=lw, ls="-.", label=lab, c=col)
        var4 = self.profiles["pow_ei(MW)"]
        if leg:
            lab = extralab + "e->i"
        else:
            lab = ""
        ax.plot(rho, var4, lw=lw / 2, ls="-", label=lab, c=col)
        # ax.plot(rho,var1+var2+var4,lw=lw,ls='--',c='y',label='check')
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(varL)
        ax.legend(loc="best", fontsize=fs)
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

    def plotGeometry(self, ax=None, surfaces_rho=np.linspace(0, 1, 11), color="b"):
        if ("R_surface" in self.derived) and (self.derived["R_surface"] is not None):
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots()
                provided = False
            else:
                provided = True

            for rho in surfaces_rho:
                ir = np.argmin(np.abs(self.profiles["rho(-)"] - rho))

                ax.plot(
                    self.derived["R_surface"][ir, :],
                    self.derived["Z_surface"][ir, :],
                    "-",
                    lw=1.0,
                    c=color,
                )

            ax.axhline(y=0, ls="--", lw=0.2, c="k")
            ax.plot(
                [self.profiles["rmaj(m)"][0]],
                [self.profiles["zmag(m)"][0]],
                "o",
                markersize=2,
                c=color,
            )

            if not provided:
                ax.set_xlabel("R (m)")
                ax.set_ylabel("Z (m)")
                ax.set_title("Surfaces @ rho=" + str(surfaces_rho), fontsize=8)
            ax.set_aspect("equal")
        else:
            print("\t- Cannot plot flux surface geometry", typeMsg="w")

    def csv(self, file="input.gacode.xlsx"):
        dictExcel = IOtools.OrderedDict()

        for ikey in self.profiles:
            print(ikey)
            if len(self.profiles[ikey].shape) == 1:
                dictExcel[ikey] = self.profiles[ikey]
            else:
                dictExcel[ikey] = self.profiles[ikey][:, 0]

        IOtools.writeExcel_fromDict(dictExcel, file, fromRow=1)

    def parabolizePlasma(self):
        PORTALSinteraction.parabolizePlasma(self)

    def changeRFpower(self, PrfMW=25.0):
        PORTALSinteraction.changeRFpower(self, PrfMW=PrfMW)

    def imposeBCtemps(self, TkeV=0.5, rho=0.9, typeEdge="linear", Tesep=0.1, Tisep=0.2):
        PORTALSinteraction.imposeBCtemps(
            self, TkeV=TkeV, rho=rho, typeEdge=typeEdge, Tesep=Tesep, Tisep=Tisep
        )

    def imposeBCdens(self, n20=2.0, rho=0.9, typeEdge="linear", nedge20=0.5):
        PORTALSinteraction.imposeBCdens(
            self, n20=n20, rho=rho, typeEdge=typeEdge, nedge20=nedge20
        )

    def addSawtoothEffectOnOhmic(self, PohTot, mixRadius=None, plotYN=False):
        """
        This will implement a flat profile inside the mixRadius to reduce the ohmic power by certain amount
        """

        if mixRadius is None:
            mixRadius = self.profiles["rho(-)"][np.where(self.profiles["q(-)"] > 1)][0]

        print(f"\t- Original Ohmic power: {self.derived['qOhm_MWmiller'][-1]:.2f}MW")
        Ohmic_old = copy.deepcopy(self.profiles["qohme(MW/m^3)"])

        dvol = self.derived["volp_miller"] * np.append(
            [0], np.diff(self.profiles["rmin(m)"])
        )

        print(
            f"\t- Will implement sawtooth ohmic power correction inside rho={mixRadius}"
        )
        Psaw = CDFtools.profilePower(
            self.profiles["rho(-)"],
            dvol,
            PohTot - self.derived["qOhm_MWmiller"][-1],
            mixRadius,
        )
        self.profiles["qohme(MW/m^3)"] += Psaw
        self.deriveQuantities()

        print(f"\t- New Ohmic power: {self.derived['qOhm_MWmiller'][-1]:.2f}MW")
        Ohmic_new = copy.deepcopy(self.profiles["qohme(MW/m^3)"])

        if plotYN:
            fig, ax = plt.subplots()
            ax.plot(self.profiles["rho(-)"], Ohmic_old, "r", lw=2)
            ax.plot(self.profiles["rho(-)"], Ohmic_new, "g", lw=2)
            plt.show()

    def to_TGLF(self, rhos=[0.5], TGLFsettings=5):
        PROFILEStoMODELS.profiles_to_tglf(self, rhos=rhos, TGLFsettings=TGLFsettings)

    def plotPeaking(
        self, ax, c="b", marker="*", label="", debugPlot=False, printVals=False
    ):
        nu_effCGYRO = self.derived["nu_eff"] * 2 / self.derived["Zeff_vol"]
        ne_peaking = self.derived["ne_peaking0.2"]
        ax.scatter([nu_effCGYRO], [ne_peaking], s=400, c=c, marker=marker, label=label)

        if printVals:
            print(f"\t- nu_eff = {nu_effCGYRO}, ne_peaking = {ne_peaking}")

        # Extra
        r = self.profiles["rmin(m)"]
        volp = self.derived["volp_miller"]
        ix = np.argmin(np.abs(self.profiles["rho(-)"] - 0.9))

        if debugPlot:
            fig, axq = plt.subplots()

            ne = self.profiles["ne(10^19/m^3)"]
            axq.plot(self.profiles["rho(-)"], ne, color="m")
            ne_vol = (
                CALCtools.integrateFS(ne * 0.1, r, volp)[-1] / self.derived["volume"]
            )
            axq.axhline(y=ne_vol * 10, color="m")

        ne = copy.deepcopy(self.profiles["ne(10^19/m^3)"])
        ne[ix:] = (0,) * len(ne[ix:])
        ne_vol = CALCtools.integrateFS(ne * 0.1, r, volp)[-1] / self.derived["volume"]
        ne_peaking0 = (
            ne[np.argmin(np.abs(self.derived["rho_pol"] - 0.2))] * 0.1 / ne_vol
        )

        if debugPlot:
            axq.plot(self.profiles["rho(-)"], ne, color="r")
            axq.axhline(y=ne_vol * 10, color="r")

        ne = copy.deepcopy(self.profiles["ne(10^19/m^3)"])
        ne[ix:] = (ne[ix],) * len(ne[ix:])
        ne_vol = CALCtools.integrateFS(ne * 0.1, r, volp)[-1] / self.derived["volume"]
        ne_peaking1 = (
            ne[np.argmin(np.abs(self.derived["rho_pol"] - 0.2))] * 0.1 / ne_vol
        )

        ne_peaking0 = ne_peaking

        ax.errorbar(
            [nu_effCGYRO],
            [ne_peaking],
            yerr=[[ne_peaking - ne_peaking1], [ne_peaking0 - ne_peaking]],
            marker=marker,
            c=c,
            markersize=16,
            capsize=2.0,
            fmt="s",
            elinewidth=1.0,
            capthick=1.0,
        )

        if debugPlot:
            axq.plot(self.profiles["rho(-)"], ne, color="b")
            axq.axhline(y=ne_vol * 10, color="b")
            plt.show()

        # print(f'{ne_peaking0}-{ne_peaking}-{ne_peaking1}')

        return nu_effCGYRO, ne_peaking


class DataTable:
    def __init__(self, variables=None):

        if variables is not None:
            self.variables = variables
        else:

            # Default for confinement mode access studies (JWH 03/2024)
            self.variables = {
                "Rgeo": ["rcentr(m)", "pos_0", "profiles", ".2f", 1, "m"],
                "ageo": ["a", None, "derived", ".2f", 1, "m"],
                "kappa @psi=0.95": ["kappa(-)", "psi_0.95", "profiles", ".2f", 1, None],
                "delta @psi=0.95": ["delta(-)", "psi_0.95", "profiles", ".2f", 1, None],
                "Bt": ["bcentr(T)", "pos_0", "profiles", ".1f", 1, "T"],
                "Ip": ["current(MA)", "pos_0", "profiles", ".1f", 1, "MA"],
                "Pin": ["qIn", None, "derived", ".1f", 1, "MW"],
                "Te @rho=0.9": ["te(keV)", "rho_0.90", "profiles", ".2f", 1, "keV"],
                "Ti/Te @rho=0.9": ["tite", "rho_0.90", "derived", ".2f", 1, None],
                "ne @rho=0.9": [
                    "ne(10^19/m^3)",
                    "rho_0.90",
                    "profiles",
                    ".2f",
                    0.1,
                    "E20m-3",
                ],
                "ptot @rho=0.9": [
                    "ptot_manual",
                    "rho_0.90",
                    "derived",
                    ".1f",
                    1e3,
                    "kPa",
                ],
                "Zeff": ["Zeff_vol", None, "derived", ".1f", 1, None],
                "fDT": ["fmain", None, "derived", ".2f", 1, None],
                "H89p": ["H89", None, "derived", ".2f", 1, None],
                "ne (vol avg)": ["ne_vol20", None, "derived", ".2f", 1, "E20m-3"],
                "fG": ["fG", None, "derived", ".2f", 1, None],
                "Pfus": ["Pfus", None, "derived", ".1f", 1, "MW"],
                "Prad": ["Prad", None, "derived", ".1f", 1, "MW"],
                "Q": ["Q", None, "derived", ".2f", 1, None],
                "Pnet @rho=0.9": ["qTr", "rho_0.90", "derived", ".1f", 1, "MW"],
                "Qi/Qe @rho=0.9": ["QiQe", "rho_0.90", "derived", ".2f", 1, None],
            }

        self.data = []

    def export_to_csv(self, filename, title=None):

        title_data = [""]
        for key in self.variables:
            if self.variables[key][5] is None:
                title_data.append(f"{key}")
            else:
                title_data.append(f"{key} ({self.variables[key][5]})")

        # Open a file with the given filename in write mode
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write the title row first if it is provided
            if title:
                writer.writerow([title] + [""] * (len(self.data[0]) - 1))

            writer.writerow(title_data)

            # Write each row in self.data to the CSV file
            for row in self.data:
                writer.writerow(row)


def compareProfiles(profiles_list, fig=None, labs_list=[""] * 10, lws=[3] * 10):
    if fig is None:
        fig = plt.figure()

    grid = plt.GridSpec(2, 3, hspace=0.6, wspace=0.2)
    ax00 = fig.add_subplot(grid[0, 0])
    ax10 = fig.add_subplot(grid[1, 0])
    ax01 = fig.add_subplot(grid[0, 1])
    ax11 = fig.add_subplot(grid[1, 1])
    ax02 = fig.add_subplot(grid[0, 2])
    # ax12 = fig.add_subplot(grid[1, 2])

    cols = GRAPHICStools.listColors()

    for cont, profile in enumerate(profiles_list):
        if cont == 0:
            leg = True
        else:
            leg = False

        profile.plot_temps(
            ax=ax00,
            leg=True,
            col=cols[cont],
            lw=lws[cont],
            extralab=f" {labs_list[cont]}",
        )
        profile.plot_dens(ax=ax10, leg=leg, col=cols[cont], lw=lws[cont])
        profile.plot_powers(
            axs=[ax01, ax11, ax02], leg=leg, col=cols[cont], lw=lws[cont]
        )


def plotAll(profiles_list, figs=None, extralabs=None, lastRhoGradients=0.89):
    if figs is not None:
        figProf_1, figProf_2, figProf_3, figProf_4, figFlows, figProf_6, fig7 = figs
        fn = None
    else:
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        fn = FigureNotebook("Profiles", geometry="1800x900")
        figProf_1, figProf_2, figProf_3, figProf_4, figFlows, figProf_6, fig7 = add_figures(fn)

    grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    axsProf_1 = [
        figProf_1.add_subplot(grid[0, 0]),
        figProf_1.add_subplot(grid[1, 0]),
        figProf_1.add_subplot(grid[2, 0]),
        figProf_1.add_subplot(grid[0, 1]),
        figProf_1.add_subplot(grid[1, 1]),
        figProf_1.add_subplot(grid[2, 1]),
        figProf_1.add_subplot(grid[0, 2]),
        figProf_1.add_subplot(grid[1, 2]),
        figProf_1.add_subplot(grid[2, 2]),
    ]

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsProf_2 = [
        figProf_2.add_subplot(grid[0, 0]),
        figProf_2.add_subplot(grid[0, 1]),
        figProf_2.add_subplot(grid[1, 0]),
        figProf_2.add_subplot(grid[1, 1]),
        figProf_2.add_subplot(grid[0, 2]),
        figProf_2.add_subplot(grid[1, 2]),
    ]
    grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.3)
    ax00c = figProf_3.add_subplot(grid[0, 0])
    axsProf_3 = [
        ax00c,
        figProf_3.add_subplot(grid[1, 0], sharex=ax00c),
        figProf_3.add_subplot(grid[2, 0]),
        figProf_3.add_subplot(grid[0, 1]),
        figProf_3.add_subplot(grid[1, 1]),
        figProf_3.add_subplot(grid[2, 1]),
        figProf_3.add_subplot(grid[0, 2]),
        figProf_3.add_subplot(grid[1, 2]),
        figProf_3.add_subplot(grid[2, 2]),
        figProf_3.add_subplot(grid[0, 3]),
        figProf_3.add_subplot(grid[1, 3]),
        figProf_3.add_subplot(grid[2, 3]),
    ]

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsProf_4 = [
        figProf_4.add_subplot(grid[0, 0]),
        figProf_4.add_subplot(grid[1, 0]),
        figProf_4.add_subplot(grid[0, 1]),
        figProf_4.add_subplot(grid[1, 1]),
        figProf_4.add_subplot(grid[0, 2]),
        figProf_4.add_subplot(grid[1, 2]),
    ]

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsFlows = [
        figFlows.add_subplot(grid[0, 0]),
        figFlows.add_subplot(grid[1, 0]),
        figFlows.add_subplot(grid[0, 1]),
        figFlows.add_subplot(grid[0, 2]),
        figFlows.add_subplot(grid[1, 1]),
        figFlows.add_subplot(grid[1, 2]),
    ]

    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    axsProf_6 = [
        figProf_6.add_subplot(grid[0, 0]),
        figProf_6.add_subplot(grid[:, 1]),
        figProf_6.add_subplot(grid[0, 2]),
        figProf_6.add_subplot(grid[1, 0]),
        figProf_6.add_subplot(grid[1, 2]),
        figProf_6.add_subplot(grid[0, 3]),
        figProf_6.add_subplot(grid[1, 3]),
    ]
    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsImps = [
        fig7.add_subplot(grid[0, 0]),
        fig7.add_subplot(grid[0, 1]),
        fig7.add_subplot(grid[0, 2]),
        fig7.add_subplot(grid[1, 0]),
        fig7.add_subplot(grid[1, 1]),
        fig7.add_subplot(grid[1, 2]),
    ]

    ls = GRAPHICStools.listLS()
    colors = GRAPHICStools.listColors()
    for i, profiles in enumerate(profiles_list):
        if extralabs is None:
            extralab = f"#{i}, "
        else:
            extralab = f"{extralabs[i]}, "
        profiles.plot(
            axs1=axsProf_1,
            axs2=axsProf_2,
            axs3=axsProf_3,
            axs4=axsProf_4,
            axsFlows=axsFlows,
            axs6=axsProf_6,
            axsImps=axsImps,
            color=colors[i],
            legYN=True,
            extralab=extralab,
            lsFlows=ls[i],
            legFlows=i == 0,
            showtexts=False,
            lastRhoGradients=lastRhoGradients,
        )

    return fn


def readTGYRO_profile_extra(file, varLabel="B_unit (T)"):
    with open(file) as f:
        aux = f.readlines()

    lenn = int(aux[36].split()[-1])

    i = 38
    allVec = []
    while i < len(aux):
        vec = np.array([float(j) for j in aux[i : i + lenn]])
        i += lenn
        allVec.append(vec)
    allVec = np.array(allVec)

    dictL = OrderedDict()
    for line in aux[2:35]:
        lab = line.split("(:)")[-1].split("\n")[0]
        try:
            dictL[lab] = int(line.split()[1])
        except:
            dictL[lab] = [int(j) for j in line.split()[1].split("-")]

    for i in dictL:
        if i.strip(" ") == varLabel:
            val = allVec[dictL[i] - 1]
            break

    return val


def aLT(r, p):
    return (
        r[-1]
        * CALCtools.produceGradient(
            torch.from_numpy(r).to(torch.double), torch.from_numpy(p).to(torch.double)
        )
        .cpu()
        .numpy()
    )


def grad(r, p):
    return MATHtools.deriv(torch.from_numpy(r), torch.from_numpy(p), array=False)


def ionName(Z, A):
    # Based on Z
    if Z == 2:
        return "He"
    elif Z == 9:
        return "F"
    elif Z == 6:
        return "C"
    elif Z == 11:
        return "Na"
    elif Z == 30:
        return "Zn"
    elif Z == 31:
        return "Ga"

    # # Based on Mass (this is the correct way, since the radiation needs to be calculated with the full element)
    # if A in [3,4]: 		return 'He'
    # elif A == 18: 	return 'F'
    # elif A == 12:	return 'C'
    # elif A == 22:	return 'Na'
    # elif A == 60:	return 'Zn'
    # elif A == 69: 	return 'Ga'


def gradientsMerger(p0, p_true, roa=0.46, blending=0.1):
    p = copy.deepcopy(p0)

    aLTe_true = np.interp(
        p.derived["roa"], p_true.derived["roa"], p_true.derived["aLTe"]
    )
    aLTi_true = np.interp(
        p.derived["roa"], p_true.derived["roa"], p_true.derived["aLTi"][:, 0]
    )
    aLne_true = np.interp(
        p.derived["roa"], p_true.derived["roa"], p_true.derived["aLne"]
    )

    ix1 = np.argmin(np.abs(p.derived["roa"] - roa + blending))
    ix2 = np.argmin(np.abs(p.derived["roa"] - roa))

    aLT0 = aLTe_true[: ix1 + 1]
    aLT2 = p.derived["aLTe"][ix2:]
    aLT1 = np.interp(
        p.derived["roa"][ix1 : ix2 + 1],
        [p.derived["roa"][ix1], p.derived["roa"][ix2]],
        [aLT0[-1], aLT2[0]],
    )[1:-1]

    aLTe = np.append(np.append(aLT0, aLT1), aLT2)
    Te = (
        CALCtools.integrateGradient(
            torch.from_numpy(p.derived["roa"]).unsqueeze(0),
            torch.Tensor(aLTe).unsqueeze(0),
            p.profiles["te(keV)"][-1],
        )
        .cpu()
        .numpy()[0]
    )

    aLT0 = aLTi_true[: ix1 + 1]
    aLT2 = p.derived["aLTi"][ix2:, 0]
    aLT1 = np.interp(
        p.derived["roa"][ix1 : ix2 + 1],
        [p.derived["roa"][ix1], p.derived["roa"][ix2]],
        [aLT0[-1], aLT2[0]],
    )[1:-1]

    aLTi = np.append(np.append(aLT0, aLT1), aLT2)
    Ti = (
        CALCtools.integrateGradient(
            torch.from_numpy(p.derived["roa"]).unsqueeze(0),
            torch.Tensor(aLTi).unsqueeze(0),
            p.profiles["ti(keV)"][-1, 0],
        )
        .cpu()
        .numpy()[0]
    )

    aLT0 = aLne_true[: ix1 + 1]
    aLT2 = p.derived["aLne"][ix2:]
    aLT1 = np.interp(
        p.derived["roa"][ix1 : ix2 + 1],
        [p.derived["roa"][ix1], p.derived["roa"][ix2]],
        [aLT0[-1], aLT2[0]],
    )[1:-1]

    aLne = np.append(np.append(aLT0, aLT1), aLT2)
    ne = (
        CALCtools.integrateGradient(
            torch.from_numpy(p.derived["roa"]).unsqueeze(0),
            torch.Tensor(aLne).unsqueeze(0),
            p.profiles["ne(10^19/m^3)"][-1],
        )
        .cpu()
        .numpy()[0]
    )

    p.profiles["te(keV)"] = Te
    p.profiles["ti(keV)"][:, 0] = Ti
    p.profiles["ne(10^19/m^3)"] = ne

    p.deriveQuantities()

    return p

def add_figures(fn, fnlab='', fnlab_pre='', tab_color=None):

    figProf_1 = fn.add_figure(label= fnlab_pre + "Profiles" + fnlab, tab_color=tab_color)
    figProf_2 = fn.add_figure(label= fnlab_pre + "Powers" + fnlab, tab_color=tab_color)
    figProf_3 = fn.add_figure(label= fnlab_pre + "Geometry" + fnlab, tab_color=tab_color)
    figProf_4 = fn.add_figure(label= fnlab_pre + "Gradients" + fnlab, tab_color=tab_color)
    figFlows = fn.add_figure(label= fnlab_pre + "Flows" + fnlab, tab_color=tab_color)
    figProf_6 = fn.add_figure(label= fnlab_pre + "Other" + fnlab, tab_color=tab_color)
    fig7 = fn.add_figure(label= fnlab_pre + "Impurities" + fnlab, tab_color=tab_color)
    figs = [figProf_1, figProf_2, figProf_3, figProf_4, figFlows, figProf_6, fig7]

    return figs

