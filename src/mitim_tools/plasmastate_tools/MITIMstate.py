import copy
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, MATHtools, PLASMAtools, IOtools
from mitim_modules.powertorch.utils import CALCtools
from mitim_tools.gacode_tools.utils import GACODEdefaults
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __version__
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

def ensure_variables_existence(self):
    # ---------------------------------------------------------------------------
    # Determine minimal set of variables that should be present in the profiles
    # ---------------------------------------------------------------------------
    
    # Kinetics
    required_profiles = {
        "te(keV)": 1,
        "ti(keV)": 2,
        "ne(10^19/m^3)": 1,
        "ni(10^19/m^3)": 2,
        "w0(rad/s)": 1,
        "ptot(Pa)": 1,
        "z_eff(-)": 1,
    }
    
    # Electromagnetics
    required_profiles.update({
        "q(-)": 1,
        "polflux(Wb/radian)": 1,
        "johm(MA/m^2)": 1,
        "jbs(MA/m^2)": 1,
        "jbstor(MA/m^2)": 1,
    })
    
    # Geometry
    required_profiles.update({
        "rho(-)": 1,
        "rmin(m)": 1,
        "rmaj(m)": 1,
        "zmag(m)": 1,
        "kappa(-)": 1,
        "delta(-)": 1,
        "zeta(-)": 1,
    })

    # Sources and Sinks
    required_profiles.update({
        "qohme(MW/m^3)": 1,
        "qei(MW/m^3)": 1,
        "qbeame(MW/m^3)": 1,
        "qbeami(MW/m^3)": 1,
        "qrfe(MW/m^3)": 1,
        "qrfi(MW/m^3)": 1,
        "qfuse(MW/m^3)": 1,
        "qfusi(MW/m^3)": 1,
        "qsync(MW/m^3)": 1,
        "qbrem(MW/m^3)": 1,
        "qline(MW/m^3)": 1,
        "qpar_beam(1/m^3/s)": 1,
        "qpar_wall(1/m^3/s)": 1,
        "qmom(N/m^2)": 1,
    })
    
    # ---------------------------------------------------------------------------
    # Insert zeros in those cases whose column are not there
    # ---------------------------------------------------------------------------

    # Choose a template for dimensionality
    template_key_1d = "rho(-)"

    # Ensure required keys exist
    for key, dim in required_profiles.items():
        if key not in self.profiles:
            self.profiles[key] = copy.deepcopy(self.profiles[template_key_1d]) * 0.0 if dim == 1 else np.atleast_2d(copy.deepcopy(self.profiles[template_key_1d]) * 0.0).T


'''
The mitim_state class is the base class for manipulating plasma states in MITIM.
Any class that inherits from this class should implement the methods:

    - derive_quantities: to derive quantities from the plasma state (must at least define "r" and call the derive_quantities_base method).

    - derive_geometry: to derive the geometry of the plasma state.
    
    - write_state: to write the plasma state to a file.
    
    - plot_geometry: to plot the geometry of the plasma state.

'''

class mitim_state:
    '''
    Class to manipulate the plasma state in MITIM.
    '''

    def __init__(self, type_file = 'input.gacode'):

        self.type = type_file

    @classmethod
    def scratch(cls, profiles, label_header='', **kwargs_process):
        # Method to write a scratch file
        
        instance = cls(None)

        # Header
        instance.header = f'''
#  Created from scratch with MITIM version {__version__}
#  {label_header}                                                       
#
'''
        # Add data to profiles
        instance.profiles = profiles

        instance.derive_quantities(**kwargs_process)

        return instance

    @IOtools.hook_method(before=ensure_variables_existence)
    def derive_quantities_base(self, mi_ref=None, derive_quantities=True, rederiveGeometry=True):

        # Make sure the profiles have the required dimensions
        if len(self.profiles["ni(10^19/m^3)"].shape) == 1:
            self.profiles["ni(10^19/m^3)"] = self.profiles["ni(10^19/m^3)"].reshape(-1, 1)
            self.profiles["ti(keV)"] = self.profiles["ti(keV)"].reshape(-1, 1)

        # -------------------------------------
        self.readSpecies()
        self.mi_first = self.Species[0]["A"]
        self.DTplasma()
        self.sumFast()
        # -------------------------------------

        if "derived" not in self.__dict__:
            self.derived = {}

        if mi_ref is not None:
            self.derived["mi_ref"] = mi_ref
            print(f"\t* Reference mass ({self.derived['mi_ref']}) to use was forced by class initialization",typeMsg="w")
        else:
            self.derived["mi_ref"] = md_u #2.0 #md_u #self.mi_first
            print(f"\t* Reference mass ({self.derived['mi_ref']}) from Deuterium, as convention in gacode",typeMsg="i")

        # Useful to have gradients in the basic ----------------------------------------------------------
        self.derived["aLTe"] = aLT(self.derived["r"], self.profiles["te(keV)"])
        self.derived["aLne"] = aLT(self.derived["r"], self.profiles["ne(10^19/m^3)"])

        self.derived["aLTi"] = self.profiles["ti(keV)"] * 0.0
        self.derived["aLni"] = []
        for i in range(self.profiles["ti(keV)"].shape[1]):
            self.derived["aLTi"][:, i] = aLT(self.derived["r"], self.profiles["ti(keV)"][:, i])
            self.derived["aLni"].append(aLT(self.derived["r"], self.profiles["ni(10^19/m^3)"][:, i]))
        self.derived["aLni"] = np.transpose(np.array(self.derived["aLni"]))
        # ------------------------------------------------------------------------------------------------

        if derive_quantities:
            
            # Avoid division by zero warning by using np.errstate
            with np.errstate(divide='ignore', invalid='ignore'):
                self.derive_quantities_full(rederiveGeometry=rederiveGeometry)

    def write_state(self, file=None):
        print("\t- Writting input.gacode file")

        if file is None:
            file = self.files[0]

        with open(file, "w") as f:
            for line in self.header:
                f.write(line)

            for i in self.profiles:
                if "(" not in i:
                    f.write(f"# {i}\n")
                else:
                    f.write(f"# {i.split('(')[0]} | {i.split('(')[-1].split(')')[0]}\n")

                if i in self.titles_single:
                    listWrite = self.profiles[i]

                    if IOtools.isnum(listWrite[0]):
                        listWrite = [f"{i:.7e}".rjust(14) for i in listWrite]
                        f.write(f"{''.join(listWrite)}\n")
                    else:
                        f.write(f"{' '.join(listWrite)}\n")

                else:
                    if len(self.profiles[i].shape) == 1:
                        for j, val in enumerate(self.profiles[i]):
                            pos = f"{j + 1}".rjust(3)
                            valt = f"{round(val,99):.7e}".rjust(15)
                            f.write(f"{pos}{valt}\n")
                    else:
                        for j, val in enumerate(self.profiles[i]):
                            pos = f"{j + 1}".rjust(3)
                            txt = "".join([f"{k:.7e}".rjust(15) for k in val])
                            f.write(f"{pos}{txt}\n")

        print(f"\t\t~ File {IOtools.clipstr(file)} written")

        # Update file
        self.files[0] = file

    # ************************************************************************************************************************************************
    # Derivation methods that children classes should implement
    # ************************************************************************************************************************************************

    def derive_quantities(self, *args, **kwargs):
        raise Exception('[MITIM] derive_quantities method is not implemented in the base class (to define "r"). Please use a derived class that implements it.')

    def derive_geometry(self, *args, **kwargs):
        raise Exception('[MITIM] This method is not implemented in the base class. Please use a derived class that implements it.')

    def plot_geometry(self, *args, **kwargs):
        print('[MITIM] Method plot_geometry() is not implemented in the base class. Please use a derived class that implements it.')
        pass

    # ************************************************************************************************************************************************
    # Derivation methods
    # ************************************************************************************************************************************************

    def calculate_Er(
        self,
        folder,
        rhos=None,
        vgenOptions={},
        name="vgen1",
        includeAll=False,
        write_new_file=None,
        cold_start=False,
        ):
        profiles = copy.deepcopy(self)

        # Resolution?
        resol_changed = False
        if rhos is not None:
            profiles.changeResolution(rho_new=rhos)
            resol_changed = True

        from mitim_tools.gacode_tools import NEOtools
        self.neo = NEOtools.NEO()
        self.neo.prep(profiles, folder)
        self.neo.run_vgen(subfolder=name, vgenOptions=vgenOptions, cold_start=cold_start)

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
                print(f'\t- Inserting {ikey} from NEO run{" (went back to original resolution by interpolation)" if resol_changed else ""}')
                self.profiles[ikey] = profiles_new.profiles[ikey]

        self.derive_quantities()

        if write_new_file is not None:
            self.write_state(file=write_new_file)

    def readSpecies(self, maxSpecies=100, correct_zeff = True):
        maxSpecies = int(self.profiles["nion"][0])

        Species = []
        for j in range(maxSpecies):
            # To determine later if this specie has zero density
            niT = self.profiles["ni(10^19/m^3)"][0, j]

            sp = {
                "N": self.profiles["name"][j],
                "Z": float(self.profiles["z"][j]),
                "A": float(self.profiles["mass"][j]),
                "S": self.profiles["type"][j].split("[")[-1].split("]")[0],
                "n0": niT,
            }

            Species.append(sp)

        self.Species = Species
        
        # Correct Zeff if needed
        if correct_zeff:
            self.correct_zeff_array()
            
    def correct_zeff_array(self):
        
        self.profiles["z_eff(-)"] = np.sum(self.profiles["ni(10^19/m^3)"] * self.profiles["z"] ** 2, axis=1) / self.profiles["ne(10^19/m^3)"]
            
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
                self.nZThermal += self.profiles["ni(10^19/m^3)"][:, sp] * self.profiles["z"][sp]

    def derive_quantities_full(self, mi_ref=None, rederiveGeometry=True):
        """
        deriving geometry is expensive, so if I'm just updating profiles it may not be needed
        """
        
        if "derived" not in self.__dict__:
            self.derived = {}

        # ---------------------------------------------------------------------------------------------------------------------
        # --------- MAIN (useful for STATEtools)
        # ---------------------------------------------------------------------------------------------------------------------

        self.derived["a"] = self.derived["r"][-1]
        # self.derived['epsX'] = self.profiles['rmaj(m)'] / self.profiles['rmin(m)']
        # self.derived['eps'] = self.derived['epsX'][-1]
        self.derived["eps"] = self.derived["r"][-1] / self.profiles["rmaj(m)"][-1]

        self.derived["roa"] = self.derived["r"] / self.derived["a"]
        self.derived["Rmajoa"] = self.profiles["rmaj(m)"] / self.derived["a"]
        self.derived["Zmagoa"] = self.profiles["zmag(m)"] / self.derived["a"]

        self.derived["torflux"] = float(self.profiles["torfluxa(Wb/radian)"][0])* 2* np.pi* self.profiles["rho(-)"] ** 2 # Wb
        self.derived["B_unit"] = PLASMAtools.Bunit(self.derived["torflux"], self.derived["r"])

        self.derived["psi_pol_n"] = (
            self.profiles["polflux(Wb/radian)"] - self.profiles["polflux(Wb/radian)"][0]
        ) / (
            self.profiles["polflux(Wb/radian)"][-1]
            - self.profiles["polflux(Wb/radian)"][0]
        )
        self.derived["rho_pol"] = self.derived["psi_pol_n"] ** 0.5

        self.derived["q95"] = np.interp(0.95, self.derived["psi_pol_n"], self.profiles["q(-)"])

        self.derived["q0"] = self.profiles["q(-)"][0]

        if self.profiles["q(-)"].min() > 1.0: 
            self.derived["rho_saw"] = np.nan
        else:
            self.derived["rho_saw"] = np.interp(1.0, self.profiles["q(-)"], self.profiles["rho(-)"])

        # --------- Geometry (only if it doesn't exist or if I ask to recalculate)

        if rederiveGeometry or ("volp_geo" not in self.derived):
            self.derive_geometry()

        # --------------------------------------------------------------------------
        # Reference mass
        # --------------------------------------------------------------------------

        # Forcing mass from this specific derive_quantities call
        if mi_ref is not None:
            self.derived["mi_ref"] = mi_ref
            print(f'\t- Using mi_ref={self.derived["mi_ref"]} provided in this particular derive_quantities method, subtituting initialization one',typeMsg='i')

        # ---------------------------------------------------------------------------------------------------------------------
        # --------- Important for scaling laws
        # ---------------------------------------------------------------------------------------------------------------------

        self.derived["Rgeo"] = float(self.profiles["rcentr(m)"][-1])
        self.derived["B0"] = np.abs(float(self.profiles["bcentr(T)"][-1]))

        # ---------------------------------------------------------------------------------------------------------------------


        self.derived["c_s"] = PLASMAtools.c_s(self.profiles["te(keV)"], self.derived["mi_ref"])
        self.derived["rho_s"] = PLASMAtools.rho_s(self.profiles["te(keV)"], self.derived["mi_ref"], self.derived["B_unit"])
        self.derived["rho_sa"] = self.derived["rho_s"] / self.derived["a"]

        self.derived["q_gb"], self.derived["g_gb"], self.derived["pi_gb"], self.derived["s_gb"], _ = PLASMAtools.gyrobohmUnits(
            self.profiles["te(keV)"],
            self.profiles["ne(10^19/m^3)"] * 1e-1,
            self.derived["mi_ref"],
            np.abs(self.derived["B_unit"]),
            self.derived["r"][-1],
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
        ge_terms = {"qpar_beam(1/m^3/s)": 1, "qpar_wall(1/m^3/s)": 1}

        self.derived["ge"] = np.zeros(len(self.profiles["rho(-)"]))
        for i in ge_terms:
            if i in self.profiles:
                self.derived["ge"] += ge_terms[i] * self.profiles[i]

        """
		Careful, that's in MW/m^3. I need to find the volumes. Using here the Miller
		calculation. Should be consistent with TGYRO

		profiles_gen puts any missing power into the CX: qioni, qione
		"""

        r = self.derived["r"]
        volp = self.derived["volp_geo"]

        self.derived["qe_MW"] = CALCtools.volume_integration(self.derived["qe"], r, volp)
        self.derived["qi_MW"] = CALCtools.volume_integration(self.derived["qi"], r, volp)
        self.derived["ge_10E20"] = CALCtools.volume_integration(self.derived["ge"] * 1e-20, r, volp)  # Because the units were #/sec/m^3

        self.derived["geIn"] = self.derived["ge_10E20"][-1]  # 1E20 particles/sec

        self.derived["qe_MWm2"] = self.derived["qe_MW"] / (volp)
        self.derived["qi_MWm2"] = self.derived["qi_MW"] / (volp)
        self.derived["ge_10E20m2"] = self.derived["ge_10E20"] / (volp)

        self.derived["QiQe"] = self.derived["qi_MWm2"] / np.where(self.derived["qe_MWm2"] == 0, 1e-10, self.derived["qe_MWm2"]) # to avoid division by zero

        # "Convective" flux
        self.derived["ce_MW"] = PLASMAtools.convective_flux(self.profiles["te(keV)"], self.derived["ge_10E20"])
        self.derived["ce_MWm2"] = PLASMAtools.convective_flux(self.profiles["te(keV)"], self.derived["ge_10E20m2"])

        # qmom
        self.derived["mt_Jmiller"] = CALCtools.volume_integration(self.profiles["qmom(N/m^2)"], r, volp)
        self.derived["mt_Jm2"] = self.derived["mt_Jmiller"] / (volp)

        # Extras for plotting in TGYRO for comparison
        P = np.zeros(len(self.derived["r"]))
        if "qsync(MW/m^3)" in self.profiles:
            P += self.profiles["qsync(MW/m^3)"]
        if "qbrem(MW/m^3)" in self.profiles:
            P += self.profiles["qbrem(MW/m^3)"]
        if "qline(MW/m^3)" in self.profiles:
            P += self.profiles["qline(MW/m^3)"]
        self.derived["qe_rad_MW"] = CALCtools.volume_integration(P, r, volp)

        P = self.profiles["qei(MW/m^3)"]
        self.derived["qe_exc_MW"] = CALCtools.volume_integration(P, r, volp)

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
        self.derived["qe_auxONLY_MW"] = CALCtools.volume_integration(P, r, volp)

        for i in ["qione(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qe_aux"] = copy.deepcopy(P)
        self.derived["qe_aux_MW"] = CALCtools.volume_integration(P, r, volp)

        # ** Ions

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qrfi(MW/m^3)", "qbeami(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qi_auxONLY"] = copy.deepcopy(P)
        self.derived["qi_auxONLY_MW"] = CALCtools.volume_integration(P, r, volp)

        for i in ["qioni(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]

        self.derived["qi_aux"] = copy.deepcopy(P)
        self.derived["qi_aux_MW"] = CALCtools.volume_integration(P, r, volp)

        # ** General

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qohme(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qOhm_MW"] = CALCtools.volume_integration(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qrfe(MW/m^3)", "qrfi(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qRF_MW"] = CALCtools.volume_integration(P, r, volp)
        if "qrfe(MW/m^3)" in self.profiles:
            self.derived["qRFe_MW"] = CALCtools.volume_integration(
                self.profiles["qrfe(MW/m^3)"], r, volp
            )
        if "qrfi(MW/m^3)" in self.profiles:
            self.derived["qRFi_MW"] = CALCtools.volume_integration(
                self.profiles["qrfi(MW/m^3)"], r, volp
            )

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qbeame(MW/m^3)", "qbeami(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qBEAM_MW"] = CALCtools.volume_integration(P, r, volp)

        self.derived["qrad_MW"] = CALCtools.volume_integration(self.derived["qrad"], r, volp)
        if "qsync(MW/m^3)" in self.profiles:
            self.derived["qrad_sync_MW"] = CALCtools.volume_integration(self.profiles["qsync(MW/m^3)"], r, volp)
        else:
            self.derived["qrad_sync_MW"] = self.derived["qrad_MW"]*0.0
        if "qbrem(MW/m^3)" in self.profiles:
            self.derived["qrad_brem_MW"] = CALCtools.volume_integration(self.profiles["qbrem(MW/m^3)"], r, volp)
        else:
            self.derived["qrad_brem_MW"] = self.derived["qrad_MW"]*0.0
        if "qline(MW/m^3)" in self.profiles:    
            self.derived["qrad_line_MW"] = CALCtools.volume_integration(self.profiles["qline(MW/m^3)"], r, volp)
        else:
            self.derived["qrad_line_MW"] = self.derived["qrad_MW"]*0.0

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qfuse(MW/m^3)", "qfusi(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qFus_MW"] = CALCtools.volume_integration(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        for i in ["qioni(MW/m^3)", "qione(MW/m^3)"]:
            if i in self.profiles:
                P += self.profiles[i]
        self.derived["qz_MW"] = CALCtools.volume_integration(P, r, volp)

        self.derived["q_MW"] = (
            self.derived["qe_MW"] + self.derived["qi_MW"]
        )

        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------

        P = np.zeros(len(self.profiles["rho(-)"]))
        if "qfuse(MW/m^3)" in self.profiles:
            P = self.profiles["qfuse(MW/m^3)"]
        self.derived["qe_fus_MW"] = CALCtools.volume_integration(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        if "qfusi(MW/m^3)" in self.profiles:
            P = self.profiles["qfusi(MW/m^3)"]
        self.derived["qi_fus_MW"] = CALCtools.volume_integration(P, r, volp)

        P = np.zeros(len(self.profiles["rho(-)"]))
        if "qfusi(MW/m^3)" in self.profiles:
            self.derived["q_fus"] = (
                self.profiles["qfuse(MW/m^3)"] + self.profiles["qfusi(MW/m^3)"]
            ) * 5
            P = self.derived["q_fus"]
        self.derived["q_fus"] = P
        self.derived["q_fus_MW"] = CALCtools.volume_integration(P, r, volp)

        """
		Derivatives
		"""
        self.derived["aLTe"] = aLT(self.derived["r"], self.profiles["te(keV)"])
        self.derived["aLTi"] = self.profiles["ti(keV)"] * 0.0
        for i in range(self.profiles["ti(keV)"].shape[1]):
            self.derived["aLTi"][:, i] = aLT(
                self.derived["r"], self.profiles["ti(keV)"][:, i]
            )
        self.derived["aLne"] = aLT(
            self.derived["r"], self.profiles["ne(10^19/m^3)"]
        )
        self.derived["aLni"] = []
        for i in range(self.profiles["ni(10^19/m^3)"].shape[1]):
            self.derived["aLni"].append(
                aLT(self.derived["r"], self.profiles["ni(10^19/m^3)"][:, i])
            )
        self.derived["aLni"] = np.transpose(np.array(self.derived["aLni"]))

        if "w0(rad/s)" not in self.profiles:
            self.profiles["w0(rad/s)"] = self.profiles["rho(-)"] * 0.0
        self.derived["aLw0"] = aLT(self.derived["r"], self.profiles["w0(rad/s)"])
        self.derived["dw0dr"] = -grad(
            self.derived["r"], self.profiles["w0(rad/s)"]
        )

        self.derived["dqdr"] = grad(self.derived["r"], self.profiles["q(-)"])

        """
		Other, performance
		"""
        qFus = self.derived["qe_fus_MW"] + self.derived["qi_fus_MW"]
        self.derived["Pfus"] = qFus[-1] * 5

        # Note that in cases with NPRAD=0 in TRANPS, this includes radiation! no way to deal wit this...
        qIn = self.derived["qe_aux_MW"] + self.derived["qi_aux_MW"]
        self.derived["qIn"] = qIn[-1]
        self.derived["Q"] = self.derived["Pfus"] / self.derived["qIn"]
        self.derived["qHeat"] = qIn[-1] + qFus[-1]

        self.derived["qTr"] = (
            self.derived["qe_aux_MW"]
            + self.derived["qi_aux_MW"]
            + (self.derived["qe_fus_MW"] + self.derived["qi_fus_MW"])
            - self.derived["qrad_MW"]
        )

        self.derived["Prad"] = self.derived["qrad_MW"][-1]
        self.derived["Prad_sync"] = self.derived["qrad_sync_MW"][-1]
        self.derived["Prad_brem"] = self.derived["qrad_brem_MW"][-1]
        self.derived["Prad_line"] = self.derived["qrad_line_MW"][-1]
        self.derived["Psol"] = self.derived["qHeat"] - self.derived["Prad"]

        self.derived["Ti_thr"] = []
        self.derived["ni_thr"] = []
        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "therm":
                self.derived["ni_thr"].append(self.profiles["ni(10^19/m^3)"][:, sp])
                self.derived["Ti_thr"].append(self.profiles["ti(keV)"][:, sp])
                
        self.derived["ni_thr"] = np.transpose(self.derived["ni_thr"])
        self.derived["Ti_thr"] = np.transpose(np.array(self.derived["Ti_thr"]))
        
        if len(self.derived["ni_thr"].shape) == 1:
            self.derived["ni_thr"] = self.derived["ni_thr"].reshape(-1, 1)
            self.derived["Ti_thr"] = self.derived["Ti_thr"].reshape(-1, 1)
        
        self.derived["ni_thrAll"] = self.derived["ni_thr"].sum(axis=1)

        self.derived["ni_All"] = self.profiles["ni(10^19/m^3)"].sum(axis=1)


        (
            self.derived["ptot_manual"],
            self.derived["pe"],
            self.derived["pi"],
            self.derived["pi_all"],
        ) = PLASMAtools.calculatePressure(
            np.expand_dims(self.profiles["te(keV)"], 0),
            np.expand_dims(np.transpose(self.profiles["ti(keV)"]), 0),
            np.expand_dims(self.profiles["ne(10^19/m^3)"] * 0.1, 0),
            np.expand_dims(np.transpose(self.profiles["ni(10^19/m^3)"] * 0.1), 0),
        )
        self.derived["ptot_manual"], self.derived["pe"], self.derived["pi"], self.derived["pi_all"] = (
            self.derived["ptot_manual"][0,...],
            self.derived["pe"][0,...],
            self.derived["pi"][0,...],
            self.derived["pi_all"][0,...],
        )
        self.derived['pi_all'] = np.transpose(self.derived['pi_all'])  # to have the same shape as ni_thr


        (
            self.derived["pthr_manual"],
            _,
            self.derived["pi_thr"],
            _,
        ) = PLASMAtools.calculatePressure(
            np.expand_dims(self.profiles["te(keV)"], 0),
            np.expand_dims(np.transpose(self.derived["Ti_thr"]), 0),
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
            np.expand_dims(np.transpose(self.derived["Ti_thr"]), 0),
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

        self.derived["tauP"] = np.where(self.derived["geIn"] != 0, self.derived["Ne"] / self.derived["geIn"], np.inf)   # Seconds
        

        self.derived["tauPotauE"] = self.derived["tauP"] / self.derived["tauE"]

        # Dilutions
        self.derived["fi"] = self.profiles["ni(10^19/m^3)"] / np.atleast_2d(
            self.profiles["ne(10^19/m^3)"]
        ).transpose().repeat(self.profiles["ni(10^19/m^3)"].shape[1], axis=1)

        # Vol-avg density
        self.derived["volume"] = CALCtools.volume_integration(np.ones(r.shape[0]), r, volp)[
            -1
        ]  # m^3
        self.derived["ne_vol20"] = (
            CALCtools.volume_integration(self.profiles["ne(10^19/m^3)"] * 0.1, r, volp)[-1]
            / self.derived["volume"]
        )  # 1E20/m^3

        self.derived["ni_vol20"] = np.zeros(self.profiles["ni(10^19/m^3)"].shape[1])
        self.derived["fi_vol"] = np.zeros(self.profiles["ni(10^19/m^3)"].shape[1])
        for i in range(self.profiles["ni(10^19/m^3)"].shape[1]):
            self.derived["ni_vol20"][i] = (
                CALCtools.volume_integration(
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
            CALCtools.volume_integration(self.profiles["te(keV)"], r, volp)[-1]
            / self.derived["volume"]
        )  # keV
        self.derived["Te_peaking"] = (
            self.profiles["te(keV)"][0] / self.derived["Te_vol"]
        )
        self.derived["Ti_vol"] = (
            CALCtools.volume_integration(self.profiles["ti(keV)"][:, 0], r, volp)[-1]
            / self.derived["volume"]
        )  # keV
        self.derived["Ti_peaking"] = (
            self.profiles["ti(keV)"][0, 0] / self.derived["Ti_vol"]
        )

        self.derived["ptot_manual_vol"] = (
            CALCtools.volume_integration(self.derived["ptot_manual"], r, volp)[-1]
            / self.derived["volume"]
        )  # MPa
        self.derived["pthr_manual_vol"] = (
            CALCtools.volume_integration(self.derived["pthr_manual"], r, volp)[-1]
            / self.derived["volume"]
        )  # MPa

        self.derived['pfast_manual'] = self.derived['ptot_manual'] - self.derived['pthr_manual']
        self.derived["pfast_manual_vol"] = (
            CALCtools.volume_integration(self.derived["pfast_manual"], r, volp)[-1]
            / self.derived["volume"]
        )  # MPa

        self.derived['pfast_fraction'] = self.derived['pfast_manual_vol'] / self.derived['ptot_manual_vol']

        #approximate pedestal top density
        self.derived['ptop(Pa)'] = np.interp(0.90, self.profiles['rho(-)'], self.profiles['ptot(Pa)'])

        # Quasineutrality
        self.derived["QN_Error"] = np.abs(
            1 - np.sum(self.derived["fi_vol"] * self.profiles["z"])
        )
        self.derived["Zeff"] = (
            np.sum(self.profiles["ni(10^19/m^3)"] * self.profiles["z"] ** 2, axis=1)
            / self.profiles["ne(10^19/m^3)"]
        )
        self.derived["Zeff_vol"] = (
            CALCtools.volume_integration(self.derived["Zeff"], r, volp)[-1]
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
            CALCtools.volume_integration(self.derived["MachNum"], r, volp)[-1]
            / self.derived["volume"]
        )

        # Retain the old beta definition for comparison with 0D modeling
        Beta_old = (self.derived["ptot_manual_vol"]* 1e6 / (self.derived["B0"] ** 2 / (2 * 4 * np.pi * 1e-7)))
        self.derived["BetaN_engineering"] = (Beta_old / 
                                        (np.abs(float(self.profiles["current(MA)"][-1])) / 
                                         (self.derived["a"] * self.derived["B0"])
                                         )* 100.0
                                         ) # expressed in percent

        ''' 
        ---------------------------------------------------------------------------------------------------
        Using B_unit, derive <B_p^2> and <Bt^2> for betap and betat calculations.
        Equivalent to GACODE expro_bp2, expro_bt2
        ---------------------------------------------------------------------------------------------------
        '''

        self.derived["bp2_exp"] = self.derived["bp2_geo"] * self.derived["B_unit"] ** 2
        self.derived["bt2_exp"] = self.derived["bt2_geo"] * self.derived["B_unit"] ** 2

        # Calculate the volume averages of bt2 and bp2

        P = self.derived["bp2_exp"]
        self.derived["bp2_vol_avg"] = CALCtools.volume_integration(P, r, volp)[-1] / self.derived["volume"]
        P = self.derived["bt2_exp"]
        self.derived["bt2_vol_avg"] = CALCtools.volume_integration(P, r, volp)[-1] / self.derived["volume"]

        # calculate beta_poloidal and beta_toroidal using volume averaged values
        # mu0 = 4pi x 10^-7, also need to convert MPa to Pa

        self.derived["Beta_p"] = (2 * 4 * np.pi * 1e-7)*self.derived["ptot_manual_vol"]* 1e6/self.derived["bp2_vol_avg"]
        self.derived["Beta_t"] = (2 * 4 * np.pi * 1e-7)*self.derived["ptot_manual_vol"]* 1e6/self.derived["bt2_vol_avg"]

        self.derived["Beta"] = 1/(1/self.derived["Beta_p"]+1/self.derived["Beta_t"])

        TroyonFactor = np.abs(float(self.profiles["current(MA)"][-1])) / (self.derived["a"] * self.derived["B0"])

        self.derived["BetaN"] = self.derived["Beta"] / TroyonFactor * 100.0

        # ---

        nG = PLASMAtools.Greenwald_density(
            np.abs(float(self.profiles["current(MA)"][-1])),
            float(self.derived["r"][-1]),
        )
        self.derived["fG"] = self.derived["ne_vol20"] / nG
        self.derived["fG_x"] = self.profiles["ne(10^19/m^3)"]* 0.1 / nG

        self.derived["tite_all"] = self.profiles["ti(keV)"] / self.profiles["te(keV)"][:, np.newaxis]
        self.derived["tite"] = self.derived["tite_all"][:, 0]
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

        # -------------------------------------------------------
        # q-star
        # -------------------------------------------------------

        self.derived["qstar"] = PLASMAtools.evaluate_qstar(
            self.profiles['current(MA)'][0],
            self.profiles['rcentr(m)'],
            self.derived['kappa95'],
            self.profiles['bcentr(T)'],
            self.derived['eps'],
            self.derived['delta95'],
            ITERcorrection=False,
            includeShaping=True,
        )[0]
        self.derived["qstar_ITER"] = PLASMAtools.evaluate_qstar(
            self.profiles['current(MA)'][0],
            self.profiles['rcentr(m)'],
            self.derived['kappa95'],
            self.profiles['bcentr(T)'],
            self.derived['eps'],
            self.derived['delta95'],
            ITERcorrection=True,
            includeShaping=True,
        )[0]

        # -------------------------------------------------------
        # Separatrix estimations
        # -------------------------------------------------------

        # ~~~~ Estimate lambda_q
        pressure_atm = self.derived["ptot_manual_vol"] * 1e6 / 101325.0
        Lambda_q = PLASMAtools.calculateHeatFluxWidth_Brunner(pressure_atm)

        # ~~~~ Estimate upstream temperature
        Bt = self.profiles["bcentr(T)"][0]
        Bp = self.derived["eps"] * Bt / self.derived["q95"] #TODO: VERY ROUGH APPROXIMATION!!!!

        self.derived['Te_lcfs_estimate'] = PLASMAtools.calculateUpstreamTemperature(
                Lambda_q, 
                self.derived["q95"], 
                self.derived["ne_vol20"], 
                self.derived["Psol"], 
                self.profiles["rcentr(m)"][0], 
                Bp, 
                Bt
                )[0]
                
        # ~~~~ Estimate upstream density
        self.derived['ne_lcfs_estimate'] = self.derived["ne_vol20"] * 0.6

        # -------------------------------------------------------
        # Transport parameters
        # -------------------------------------------------------

        self.derived['betae'] = PLASMAtools.betae(
            self.profiles['te(keV)'],
            self.profiles['ne(10^19/m^3)']*0.1,
            self.derived["B_unit"]
            )

        self.derived['xnue'] = PLASMAtools.xnue(
            torch.from_numpy(self.profiles['te(keV)']).to(torch.double),
            torch.from_numpy(self.profiles['ne(10^19/m^3)']*0.1).to(torch.double),
            self.derived["a"],
            mref_u=self.derived["mi_ref"]
            ).cpu().numpy()

        self.derived['debye'] = PLASMAtools.debye(
            self.profiles['te(keV)'],
            self.profiles['ne(10^19/m^3)']*0.1,
            self.derived["mi_ref"],
            self.derived["B_unit"]
            )
        self.derived['s_hat'] =  self.derived["r"]*self._deriv_gacode( np.log(abs(self.profiles["q(-)"])) )
        self.derived['s_q'] = (self.profiles["q(-)"] / self.derived['roa'])**2 * self.derived['s_hat']
        self.derived['s_q'][0] = 0.0 # infinite in first location

    # Derivate function
    def _deriv_gacode(self,y):
        return grad(self.derived["r"],y).cpu().numpy()

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
                np.expand_dims(self.derived["r"][:i], 0),
                np.expand_dims(self.profiles["te(keV)"][:i], 0),
                np.expand_dims(np.transpose(self.profiles["ti(keV)"][:i]), 0),
                np.expand_dims(self.profiles["ne(10^19/m^3)"][:i] * 0.1, 0),
                np.expand_dims(
                    np.transpose(self.profiles["ni(10^19/m^3)"][:i] * 0.1), 0
                ),
                np.expand_dims(self.derived["volp_geo"][:i], 0),
            )

            _, _, Ni_x[i], _ = PLASMAtools.calculateContent(
                np.expand_dims(self.derived["r"][:i], 0),
                np.expand_dims(self.profiles["te(keV)"][:i], 0),
                np.expand_dims(np.transpose(self.profiles["ti(keV)"][:i]), 0),
                np.expand_dims(
                    self.profiles["ni(10^19/m^3)"][:i, impurityPosition] * 0.1, 0
                ),
                np.expand_dims(
                    np.transpose(self.profiles["ni(10^19/m^3)"][:i] * 0.1), 0
                ),
                np.expand_dims(self.derived["volp_geo"][:i], 0),
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

        Prad_ratio = self.derived['Prad'] / self.derived['qHeat'] 
        Prad_ratio_brem = self.derived['Prad_brem']/self.derived['Prad'] 
        Prad_ratio_line = self.derived['Prad_line']/self.derived['Prad'] 
        Prad_ratio_sync = self.derived['Prad_sync']/self.derived['Prad'] 

        try:
            ImpurityText = ""
            for i in range(len(self.Species)):
                ImpurityText += f"{self.Species[i]['N']}({self.Species[i]['Z']:.0f},{self.Species[i]['A']:.0f}) = {self.derived['fi_vol'][i]:.1e}, "
            ImpurityText = ImpurityText[:-2]

            print(f"\n***********************{label}****************")
            print("Engineering Parameters:")
            print(f"\tBt = {self.profiles['bcentr(T)'][0]:.2f}T, Ip = {self.profiles['current(MA)'][0]:.2f}MA (q95 = {self.derived['q95']:.2f}, q* = {self.derived['qstar']:.2f}, q*ITER = {self.derived['qstar_ITER']:.2f}), Pin = {self.derived['qIn']:.2f}MW")
            print(f"\tR  = {self.profiles['rcentr(m)'][0]:.2f}m, a  = {self.derived['a']:.2f}m (eps = {self.derived['eps']:.3f})")
            print(f"\tkappa_sep = {self.profiles['kappa(-)'][-1]:.2f}, kappa_995 = {self.derived['kappa995']:.2f}, kappa_95 = {self.derived['kappa95']:.2f}, kappa_a = {self.derived['kappa_a']:.2f}")
            print(f"\tdelta_sep  = {self.profiles['delta(-)'][-1]:.2f}, delta_995  = {self.derived['delta995']:.2f}, delta_95  = {self.derived['delta95']:.2f}")
            print("Performance:")
            print("\tQ     =  {0:.2f}   (Pfus = {1:.1f}MW, Pin = {2:.1f}MW)".format(self.derived["Q"], self.derived["Pfus"], self.derived["qIn"]))
            print("\tH98y2 =  {0:.2f}   (tauE  = {1:.3f} s)".format(self.derived["H98"], self.derived["tauE"]))
            print("\tH89p  =  {0:.2f}   (H97L  = {1:.2f})".format(self.derived["H89"], self.derived["H97L"]))
            print("\tnu_ne =  {0:.2f}   (nu_eff = {1:.2f})".format(self.derived["ne_peaking"], self.derived["nu_eff"]))
            print("\tnu_ne0.2 =  {0:.2f}   (nu_eff w/Zeff2 = {1:.2f})".format(self.derived["ne_peaking0.2"], self.derived["nu_eff2"]))
            print(f"\tnu_Ti =  {self.derived['Ti_peaking']:.2f}")
            print(f"\tp_vol =  {self.derived['ptot_manual_vol']:.2f} MPa ({self.derived['pfast_fraction']*100.0:.1f}% fast)")
            print(f"\tBetaN =  {self.derived['BetaN']:.3f} (BetaN w/B0 = {self.derived['BetaN_engineering']:.3f})")
            print(f"\tPrad  =  {self.derived['Prad']:.1f}MW ({Prad_ratio*100.0:.1f}% of total) ({Prad_ratio_brem*100.0:.1f}% brem, {Prad_ratio_line*100.0:.1f}% line, {Prad_ratio_sync*100.0:.1f}% sync)")
            print("\tPsol  =  {0:.1f}MW (fLH = {1:.2f})".format(self.derived["Psol"], self.derived["LHratio"]))
            print("Operational point ( [<ne>,<Te>] = [{0:.2f},{1:.2f}] ) and species:".format(self.derived["ne_vol20"], self.derived["Te_vol"]))
            print("\t<Ti>  = {0:.2f} keV   (<Ti>/<Te> = {1:.2f}, Ti0/Te0 = {2:.2f})".format(self.derived["Ti_vol"],self.derived["tite_vol"],self.derived["tite"][0],))
            print("\tfG    = {0:.2f}   (<ne> = {1:.2f} * 10^20 m^-3)".format(self.derived["fG"], self.derived["ne_vol20"]))
            print(f"\tZeff  = {self.derived['Zeff_vol']:.2f}   (M_main = {self.derived['mbg_main']:.2f}, f_main = {self.derived['fmain']:.2f}) [QN err = {self.derived['QN_Error']:.1e}]")
            print(f"\tMach  = {self.derived['MachNum_vol']:.2f} (vol avg)")
            print("Content:")
            print("\tWe = {0:.2f} MJ,   Wi_thr = {1:.2f} MJ    (W_thr = {2:.2f} MJ)".format(self.derived["We"], self.derived["Wi_thr"], self.derived["Wthr"]))
            print("\tNe = {0:.1f}*10^20, Ni_thr = {1:.1f}*10^20 (N_thr = {2:.1f}*10^20)".format(self.derived["Ne"], self.derived["Ni_thr"], self.derived["Nthr"]))
            print(f"\ttauE  = { self.derived['tauE']:.3f} s,  tauP = {self.derived['tauP']:.3f} s (tauP/tauE = {self.derived['tauPotauE']:.2f})")
            print("Species concentration:")
            print(f"\t{ImpurityText}")
            print("******************************************************")
        except KeyError:
            print("\t- When printing info, not all keys found, probably because this input.gacode class came from an old MITIM version",typeMsg="w",)
            if reDeriveIfNotFound:
                self.derive_quantities()
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
                print(f"\t\t\t- Scaling density of {self.Species[sp]['N']} by an average factor of {np.mean(scaleFactor_ions):.3f}")
                ni_orig = self.profiles["ni(10^19/m^3)"][:, sp]
                self.profiles["ni(10^19/m^3)"][:, sp] = scaleFactor_ions * ni_orig

    def toNumpyArrays(self):
        self.profiles.update({key: tensor.cpu().detach().cpu().numpy() for key, tensor in self.profiles.items() if isinstance(tensor, torch.Tensor)})
        self.derived.update({key: tensor.cpu().detach().cpu().numpy() for key, tensor in self.derived.items() if isinstance(tensor, torch.Tensor)})

    def changeResolution(self, n=100, rho_new=None, interpolation_function=MATHtools.extrapolateCubicSpline):
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

        self.derive_quantities()

        print(f"\t\t- Resolution of profiles changed to {n} points with function {interpolation_function}")

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
                self.Mion = 0  # If no D or T, assume that the main ion is the first and only

        self.ion_list_main = [self.Dion+1, self.Tion+1] if self.DTplasmaBool else [self.Mion+1]
        self.ion_list_impurities = [i+1 for i in range(len(self.Species)) if i+1 not in self.ion_list_main]

    def remove(self, ions_list):
        # First order them
        ions_list.sort()
        print("\t\t- Removing ions in positions (of ions order, no zero): ",ions_list,typeMsg="i",)

        ions_list = [(i - 1 if i >-1 else i) for i in ions_list]

        fail = False

        var_changes = ["name", "type", "mass", "z"]
        for i in var_changes:
            try:
                self.profiles[i] = np.delete(self.profiles[i], ions_list)
            except:
                print(f"\t\t\t* Ions {[k+1 for k in ions_list]} could not be removed",typeMsg="w")
                fail = True
                break

        if not fail:
            var_changes = ["ni(10^19/m^3)", "ti(keV)", "vpol(m/s)", "vtor(m/s)"]
            for i in var_changes:
                if i in self.profiles:
                    self.profiles[i] = np.delete(self.profiles[i], ions_list, axis=1)

        if not fail:
            # Ensure we extract the scalar value from the array
            self.profiles["nion"] = np.array([str(int(self.profiles["nion"][0]) - len(ions_list))])

        self.readSpecies()
        self.derive_quantities(rederiveGeometry=False)

        print("\t\t\t- Set of ions in updated profiles: ", self.profiles["name"])

    def lumpSpecies(
        self, ions_list=[2, 3], allthermal=False, forcename=None, force_integer=False, force_mass=None
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

        print("\t\t- Lumping ions in positions (of ions order, no zero): ",ions_list,typeMsg="i",)

        if forcename is None:
            forcename = "LUMPED"

        # Contributions to dilution and to Zeff
        fZ1 = np.zeros(self.derived["fi"].shape[0])
        fZ2 = np.zeros(self.derived["fi"].shape[0])
        for i in ions_list:
            fZ1 += self.Species[i - 1]["Z"] * self.derived["fi"][:, i - 1]
            fZ2 += self.Species[i - 1]["Z"] ** 2 * self.derived["fi"][:, i - 1]

        Zr = fZ2 / fZ1
        Zr_vol = CALCtools.volume_integration(Zr, self.derived["r"], self.derived["volp_geo"])[-1] / self.derived["volume"]

        print(f'\t\t\t* Original plasma had Zeff_vol={self.derived["Zeff_vol"]:.2f}, QN error={self.derived["QN_Error"]:.4f}')

        # New specie parameters
        if force_integer:
            Z = round(Zr_vol)
            print(f"\t\t\t* Lumped Z forced to be an integer ({Zr_vol}->{Z}), so plasma may not be quasineutral or fulfill original Zeff",typeMsg="w",)
        else:
            Z = Zr_vol

        A = Z * 2 if force_mass is None else force_mass
        nZ = fZ1 / Z * self.profiles["ne(10^19/m^3)"]
        mass_density = A * self.derived["fi"]

        # Compute the mass weighted average velocity profiles
        if "vpol(m/s)" in self.profiles:
            vpol = np.sum((mass_density * self.profiles["vpol(m/s)"])[:,np.array(ions_list)-1],axis=1) / np.sum(mass_density[:,np.array(ions_list)-1],axis=1)
        if "vtor(m/s)" in self.profiles:
            vtor = np.sum((mass_density * self.profiles["vtor(m/s)"])[:,np.array(ions_list)-1],axis=1) / np.sum(mass_density[:,np.array(ions_list)-1],axis=1)

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
        if "vpol(m/s)" in self.profiles:
            self.profiles["vpol(m/s)"] = np.append(
                self.profiles["vpol(m/s)"], np.transpose(np.atleast_2d(vpol)), axis=1
            )
        if "vtor(m/s)" in self.profiles:
            self.profiles["vtor(m/s)"] = np.append(
                self.profiles["vtor(m/s)"], np.transpose(np.atleast_2d(vtor)), axis=1
            )

        self.readSpecies()
        self.derive_quantities(rederiveGeometry=False)

        # Remove species
        self.remove(ions_list)

        # Contributions to dilution and to Zeff
        print(f'\t\t\t* New plasma has Zeff_vol={self.derived["Zeff_vol"]:.2f}, QN error={self.derived["QN_Error"]:.4f}')

    def lumpImpurities(self):

        self.lumpSpecies(ions_list=self.ion_list_impurities)

    def lumpIons(self):

        self.lumpSpecies(ions_list=self.ion_list_main+self.ion_list_impurities)

    def lumpDT(self):

        if self.DTplasmaBool:
            self.lumpSpecies(ions_list=self.ion_list_main, forcename="DT", force_mass=2.5)
        else:
            print('\t\t- No DT plasma, so no lumping of main ions')

        self.moveSpecie(pos=len(self.Species), pos_new=1)

    def changeZeff(
        self,
        Zeff,
        ion_pos = 2,                  # Position of ion to change (if (D,Z1,Z2), pos 1 -> change Z1)
        keep_fmain = False,           # If True, it will keep fmain and change Z of ion in position ion_pos. If False, it will change the content of ion in position ion_pos and the content of quasineutral ions to achieve Zeff
        fmain_force = None,           # If keep_fmain is True, it will force fmain to this value. If None, it will use the current fmain
        enforceSameGradients = False  # If True, it will scale all thermal densities to have the same gradients after changing Zeff
        ):

        if not keep_fmain and fmain_force is not None:
            raise ValueError("[MITIM] fmain_force can only be used if keep_fmain is True")

        if fmain_force is not None:
            fmain_factor = fmain_force / self.derived["fmain"]
        else:
            fmain_factor = 1.0

        if self.DTplasmaBool:
            quasineutral_ions = [self.Dion, self.Tion]
        else:
            quasineutral_ions = [self.Mion]

        if not keep_fmain:
            print(f'\t\t- Changing Zeff (from {self.derived["Zeff_vol"]:.3f} to {Zeff=:.3f}) by changing content of ion in position {ion_pos} {self.Species[ion_pos]["N"],self.Species[ion_pos]["Z"]}, quasineutralized by ions {quasineutral_ions}',typeMsg="i")
        else:
            print(f'\t\t- Changing Zeff (from {self.derived["Zeff_vol"]:.3f} to {Zeff=:.3f}) by changing content and Z of ion in position {ion_pos} {self.Species[ion_pos]["N"],self.Species[ion_pos]["Z"]}, quasineutralized by ions {quasineutral_ions} and keeping fmain={self.derived["fmain"]*fmain_factor:.3f}',typeMsg="i")

        # Plasma needs to be in quasineutrality to start with
        self.enforceQuasineutrality()

        # ------------------------------------------------------
        # Contributions to equations
        # ------------------------------------------------------
        Zq = np.zeros(self.derived["fi"].shape[0])
        Zq2 = np.zeros(self.derived["fi"].shape[0])
        fZq = np.zeros(self.derived["fi"].shape[0])
        fZq2 = np.zeros(self.derived["fi"].shape[0])
        fZj = np.zeros(self.derived["fi"].shape[0])
        fZj2 = np.zeros(self.derived["fi"].shape[0])
        for i in range(len(self.Species)):
            
            # Ions for quasineutrality (main ones)
            if i in quasineutral_ions:
                Zq += self.Species[i]["Z"] 
                Zq2 += self.Species[i]["Z"] ** 2 
                
                fZq += self.Species[i]["Z"] * self.derived["fi"][:, i]          * fmain_factor
                fZq2 += self.Species[i]["Z"] ** 2 * self.derived["fi"][:, i]    * fmain_factor
            # Non-quasineutral and not the ion to change
            elif i != ion_pos:
                fZj += self.Species[i]["Z"] * self.derived["fi"][:, i]
                fZj2 += self.Species[i]["Z"] ** 2 * self.derived["fi"][:, i]
            # Ion to change
            else:
                Zk = self.Species[i]["Z"]

        fi_orig = self.derived["fi"][:, ion_pos]
        Zi_orig = self.Species[ion_pos]["Z"]
        Ai_orig = self.Species[ion_pos]["A"]

        if not keep_fmain:
            # ------------------------------------------------------
            # Find free parameters (fk and fq)
            # ------------------------------------------------------

            fk = ( Zeff - (1-fZj)*Zq2/Zq - fZj2 ) / ( Zk**2 - Zk*Zq2/Zq)
            fq = ( 1 - fZj - fk*Zk ) / Zq

            if (fq<0).any():
                raise ValueError(f"Zeff cannot be reduced by changing ion #{ion_pos} because it would require negative densities for quasineutral ions")

            # ------------------------------------------------------
            # Insert
            # ------------------------------------------------------

            self.profiles["ni(10^19/m^3)"][:, ion_pos] = fk * self.profiles["ne(10^19/m^3)"]
            for i in quasineutral_ions:
                self.profiles["ni(10^19/m^3)"][:, i] = fq * self.profiles["ne(10^19/m^3)"]
        else:
            # ------------------------------------------------------
            # Find free parameters (fk and Zk)
            # ------------------------------------------------------

            Zk = (Zeff - fZq2 - fZj2) / (1 - fZq - fZj)
            
            # I need a single value
            Zk_ave = CALCtools.volume_integration(Zk, self.profiles["rmin(m)"], self.derived["volp_geo"])[-1] / self.derived["volume"]

            fk = (1 - fZq - fZj) / Zk_ave

            # ------------------------------------------------------
            # Insert
            # ------------------------------------------------------

            self.profiles['z'][ion_pos] = Zk_ave
            self.profiles['mass'][ion_pos] = Zk_ave * 2
            self.profiles["ni(10^19/m^3)"][:, ion_pos] = fk * self.profiles["ne(10^19/m^3)"]
            
            if fmain_force is not None:
                for i in quasineutral_ions:
                    self.profiles["ni(10^19/m^3)"][:, i] *= fmain_factor

        self.readSpecies()

        self.derive_quantities(rederiveGeometry=False)

        if enforceSameGradients:
            self.scaleAllThermalDensities()
            self.derive_quantities(rederiveGeometry=False)

        print(f'\t\t\t* Dilution changed from {fi_orig.mean():.2e} (vol avg) of ion [{Zi_orig:.2f},{Ai_orig:.2f}] to { self.derived["fi"][:, ion_pos].mean():.2e} of ion [{self.profiles["z"][ion_pos]:.2f}, {self.profiles["mass"][ion_pos]:.2f}] to achieve Zeff={self.derived["Zeff_vol"]:.3f} (fDT={self.derived["fmain"]:.3f}) [quasineutrality error = {self.derived["QN_Error"]:.1e}]')

    def moveSpecie(self, pos=2, pos_new=1):
        """
        if (D,Z1,Z2), pos 1 pos_new 2-> (Z1,D,Z2)
        """
        
        if pos_new > pos:
            pos, pos_new = pos_new, pos

        position_to_moveFROM_in_profiles = pos - 1
        position_to_moveTO_in_profiles = pos_new - 1

        print(f'\t\t- Moving ion in position (of ions order, no zero) {pos} ({self.profiles["name"][position_to_moveFROM_in_profiles]}) to {pos_new}',typeMsg="i",)

        self.profiles["nion"] = np.array([f"{int(self.profiles['nion'][0])+1}"])

        for ikey in ["name", "mass", "z", "type", "ni(10^19/m^3)", "ti(keV)", "vpol(m/s)", "vtor(m/s)"]:
            if ikey in self.profiles:
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
        self.derive_quantities(rederiveGeometry=False)

        if position_to_moveTO_in_profiles > position_to_moveFROM_in_profiles:
            self.remove([position_to_moveFROM_in_profiles + 1])
        else:
            self.remove([position_to_moveFROM_in_profiles + 2])

    def addSpecie(self, Z=5.0, mass=10.0, fi_vol=0.1, forcename=None):
        print(f"\t\t- Creating new specie with Z={Z}, mass={mass}, fi_vol={fi_vol}",typeMsg="i",)

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
        self.derive_quantities(rederiveGeometry=False)

    def correct(self, options={}, write=False, new_file=None):
        """
        if name= T D LUMPED, and I want to eliminate D, removeIons = [2]
        """

        recalculate_ptot = options.get("recalculate_ptot", True)  # Only done by default
        removeIons = options.get("removeIons", [])
        remove_fast = options.get("remove_fast", False)
        quasineutrality = options.get("quasineutrality", False)
        enforce_same_aLn = options.get("enforce_same_aLn", False)
        groupQIONE = options.get("groupQIONE", False)
        ensure_positive_Gamma = options.get("ensure_positive_Gamma", False)
        force_mach = options.get("force_mach", None)
        thermalize_fast = options.get("thermalize_fast", False)

        print("\t- Custom correction of input.gacode file has been requested")

        # ----------------------------------------------------------------------
        # Correct
        # ----------------------------------------------------------------------

        # Remove desired ions
        if len(removeIons) > 0:
            self.remove(removeIons)

        # Remove fast
        if remove_fast:
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
        elif thermalize_fast:
            self.make_fast_ions_thermal()

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
        if enforce_same_aLn:
            self.enforce_same_density_gradients()

        # Enforce quasineutrality
        if quasineutrality:
            self.enforceQuasineutrality()

        print(f"\t\t\t* Quasineutrality error = {self.derived['QN_Error']:.1e}")

        # Recompute ptot
        if recalculate_ptot:
            self.derive_quantities(rederiveGeometry=False)
            self.selfconsistentPTOT()

        # If I don't trust the negative particle flux in the core that comes from TRANSP...
        if ensure_positive_Gamma:
            print("\t\t- Making particle flux always positive", typeMsg="i")
            self.profiles["qpar_beam(1/m^3/s)"] = self.profiles["qpar_beam(1/m^3/s)"].clip(0)
            self.profiles["qpar_wall(1/m^3/s)"] = self.profiles["qpar_wall(1/m^3/s)"].clip(0)

        # Mach
        if force_mach is not None:
            self.introduceRotationProfile(Mach_LF=force_mach)

        # ----------------------------------------------------------------------
        # Re-derive
        # ----------------------------------------------------------------------

        self.derive_quantities(rederiveGeometry=False) 

        # ----------------------------------------------------------------------
        # Write
        # ----------------------------------------------------------------------
        if write:
            self.write_state(file=new_file)
            self.printInfo()
        
    def enforce_same_density_gradients(self, onlyThermal=False):
        txt = ""
        for sp in range(len(self.Species)):
            if (not onlyThermal) or (self.Species[sp]["S"] == "therm"):
                self.profiles["ni(10^19/m^3)"][:, sp] = self.derived["fi_vol"][sp] * self.profiles["ne(10^19/m^3)"]
                txt += f"{self.Species[sp]['N']} "
        print(f"\t\t- Making all {'thermal ' if onlyThermal else ''}ions ({txt}) have the same a/Ln as electrons (making them an exact flat fraction)",typeMsg="i",)
        self.derive_quantities(rederiveGeometry=False)

    def make_fast_ions_thermal(self):
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
            print("\t- Making fast species as if they were thermal (to keep dilution effect and Qi-sum of fluxes)",typeMsg="w")
    
    def selfconsistentPTOT(self):
        print(f"\t\t* Recomputing ptot and inserting it as ptot(Pa), changed from p0 = {self.profiles['ptot(Pa)'][0] * 1e-3:.1f} to {self.derived['ptot_manual'][0]*1e+3:.1f} kPa",typeMsg="i")
        self.profiles["ptot(Pa)"] = self.derived["ptot_manual"] * 1e6

    def enforceQuasineutrality(self, using_ion = None):
        print(f"\t\t- Enforcing quasineutrality (error = {self.derived['QN_Error']:.1e})",typeMsg="i",)

        # What's the lack of quasineutrality?
        ni = self.profiles["ne(10^19/m^3)"] * 0.0
        for sp in range(len(self.Species)):
            ni += self.profiles["ni(10^19/m^3)"][:, sp] * self.profiles["z"][sp]
        ne_missing = self.profiles["ne(10^19/m^3)"] - ni

        # What ion to modify?
        if using_ion is None:
            if self.DTplasmaBool:
                print("\t\t\t* Enforcing quasineutrality by modifying D and T equally")
                prev_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Dion])
                self.profiles["ni(10^19/m^3)"][:, self.Dion] += ne_missing / 2
                self.profiles["ni(10^19/m^3)"][:, self.Tion] += ne_missing / 2
                new_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Dion])
            else:
                print(f"\t\t\t* Enforcing quasineutrality by modifying main ion (position #{self.Mion})")
                prev_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Mion])
                self.profiles["ni(10^19/m^3)"][:, self.Mion] += ne_missing
                new_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, self.Mion])
        else:
            print(f"\t\t\t* Enforcing quasineutrality by modifying ion (position #{using_ion})")
            prev_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, using_ion])
            self.profiles["ni(10^19/m^3)"][:, using_ion] += ne_missing
            new_on_axis = copy.deepcopy(self.profiles["ni(10^19/m^3)"][0, using_ion])


        print(f"\t\t\t\t- Changed on-axis density from n0 = {prev_on_axis:.2f} to {new_on_axis:.2f} ({100*(new_on_axis-prev_on_axis)/prev_on_axis:.1f}%)")

        self.derive_quantities(rederiveGeometry=False)

    def introduceRotationProfile(self, Mach_LF=1.0, new_file=None):
        print(f"\t- Enforcing Mach Number in LF of {Mach_LF}")
        self.derive_quantities()
        Vtor_LF = PLASMAtools.constructVtorFromMach(
            Mach_LF, self.profiles["ti(keV)"][:, 0], self.derived["mbg"]
        )  # m/s

        self.profiles["w0(rad/s)"] = Vtor_LF / (self.derived["R_LF"])  # rad/s

        self.derive_quantities()

        if new_file is not None:
            self.write_state(file=new_file)

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

        self.derive_quantities()

    def changeRFpower(self, PrfMW=25.0):
        """
        keeps same partition
        """
        print(f"- Changing the RF power from {self.derived['qRF_MW'][-1]:.1f} MW to {PrfMW:.1f} MW",typeMsg="i",)
        
        if self.derived["qRF_MW"][-1] == 0.0:
            raise Exception("No RF power in the input.gacode, cannot modify the RF power")

        for i in ["qrfe(MW/m^3)", "qrfi(MW/m^3)"]:
            self.profiles[i] = self.profiles[i] * PrfMW / self.derived["qRF_MW"][-1]

        self.derive_quantities()

    def imposeBCtemps(self, TkeV=0.5, rho=0.9, typeEdge="linear", Tesep=0.1, Tisep=0.2):

        ix = np.argmin(np.abs(rho - self.profiles["rho(-)"]))

        self.profiles["te(keV)"] = self.profiles["te(keV)"] * TkeV / self.profiles["te(keV)"][ix]

        print(f"- Producing {typeEdge} boundary condition @ rho = {rho}, T = {TkeV} keV",typeMsg="i",)

        for sp in range(len(self.Species)):
            if self.Species[sp]["S"] == "therm":
                self.profiles["ti(keV)"][:, sp] = self.profiles["ti(keV)"][:, sp] * TkeV / self.profiles["ti(keV)"][ix, sp]

        if typeEdge == "linear":
            self.profiles["te(keV)"][ix:] = np.linspace(TkeV, Tesep, len(self.profiles["rho(-)"][ix:]))

            for sp in range(len(self.Species)):
                if self.Species[sp]["S"] == "therm":
                    self.profiles["ti(keV)"][ix:, sp] = np.linspace(TkeV, Tisep, len(self.profiles["rho(-)"][ix:]))

        elif typeEdge == "same":
            pass
        else:
            raise Exception("no edge")

    def imposeBCdens(self, n20=2.0, rho=0.9, typeEdge="linear", nedge20=0.5, isn20_edge=True):
        ix = np.argmin(np.abs(rho - self.profiles["rho(-)"]))

        # Determine the factor to scale the density (either average or at rho)
        if not isn20_edge:
            print(f"- Changing the initial average density from {self.derived['ne_vol20']:.1f} 1E20/m3 to {n20:.1f} 1E20/m3",typeMsg="i")
            factor = n20 / self.derived["ne_vol20"]
        else:
            print(f"- Changing the density at rho={rho} from {self.profiles['ne(10^19/m^3)'][ix]*1E-1:.1f} 1E20/m3 to {n20:.1f} 1E20/m3",typeMsg="i")
            factor = n20 / (self.profiles["ne(10^19/m^3)"][ix]*1E-1)
        # ------------------------------------------------------------------

        # Scale the density profiles
        for i in ["ne(10^19/m^3)", "ni(10^19/m^3)"]:
            self.profiles[i] = self.profiles[i] * factor

        # Apply the edge condition
        if typeEdge == "linear":
            factor_x = np.linspace(self.profiles["ne(10^19/m^3)"][ix],nedge20 * 1e1,len(self.profiles["rho(-)"][ix:]),)/ self.profiles["ne(10^19/m^3)"][ix:]

            self.profiles["ne(10^19/m^3)"][ix:] = self.profiles["ne(10^19/m^3)"][ix:] * factor_x

            for i in range(self.profiles["ni(10^19/m^3)"].shape[1]):
                self.profiles["ni(10^19/m^3)"][ix:, i] = self.profiles["ni(10^19/m^3)"][ix:, i] * factor_x

        elif typeEdge == "same":
            pass
        else:
            raise Exception("no edge")
        
    def addSawtoothEffectOnOhmic(self, PohTot, mixRadius=None, plotYN=False):
        """
        This will implement a flat profile inside the mixRadius to reduce the ohmic power by certain amount
        """

        if mixRadius is None:
            mixRadius = self.profiles["rho(-)"][np.where(self.profiles["q(-)"] > 1)][0]

        print(f"\t- Original Ohmic power: {self.derived['qOhm_MW'][-1]:.2f}MW")
        Ohmic_old = copy.deepcopy(self.profiles["qohme(MW/m^3)"])

        dvol = self.derived["volp_geo"] * np.append(
            [0], np.diff(self.derived["r"])
        )

        print(
            f"\t- Will implement sawtooth ohmic power correction inside rho={mixRadius}"
        )
        from mitim_tools.transp_tools import CDFtools
        Psaw = CDFtools.profilePower(
            self.profiles["rho(-)"],
            dvol,
            PohTot - self.derived["qOhm_MW"][-1],
            mixRadius,
        )
        self.profiles["qohme(MW/m^3)"] += Psaw
        self.derive_quantities()

        print(f"\t- New Ohmic power: {self.derived['qOhm_MW'][-1]:.2f}MW")
        Ohmic_new = copy.deepcopy(self.profiles["qohme(MW/m^3)"])

        if plotYN:
            fig, ax = plt.subplots()
            ax.plot(self.profiles["rho(-)"], Ohmic_old, "r", lw=2)
            ax.plot(self.profiles["rho(-)"], Ohmic_new, "g", lw=2)
            plt.show()

    # ************************************************************************************************************************************************
    # Plotting methods for the state class, which is used to plot the profiles, powers, geometry, gradients, flows, and other quantities.
    # ************************************************************************************************************************************************

    def plot(
        self,
        fn=None,fnlab="",
        axs1=None, axs2=None, axs3=None, axs4=None, axsFlows=None, axs6=None, axsImps=None,
        color="b",legYN=True,extralab="",lsFlows="-",legFlows=True,showtexts=True,lastRhoGradients=0.89,
        ):
        if axs1 is None:
            if fn is None:
                from mitim_tools.misc_tools.GUItools import FigureNotebook

                self.fn = FigureNotebook("PROFILES Notebook", geometry="1600x1000")

            figs = state_plotting.add_figures(self.fn, fnlab=fnlab)
            axs1, axs2, axs3, axs4, axsFlows, axs6, axsImps = state_plotting.add_axes(figs)

        lw, fs = 1, 6

        state_plotting.plot_profiles(self,axs1, color=color, legYN=legYN, extralab=extralab, lw=lw, fs=fs)
        state_plotting.plot_powers(self,axs2, color=color, legYN=legYN, extralab=extralab, lw=lw, fs=fs)
        self.plot_geometry(axs3, color=color, legYN=legYN, extralab=extralab, lw=lw, fs=fs)
        state_plotting.plot_gradients(self,axs4, color=color, lw=lw, lastRho=lastRhoGradients, label=extralab)
        if axsFlows is not None:
            state_plotting.plot_flows(self, axsFlows, ls=lsFlows, leg=legFlows, showtexts=showtexts)
        state_plotting.plot_other(self,axs6, color=color, lw=lw, extralab=extralab, fs=fs)
        state_plotting.plot_ions(self,axsImps, color=color, legYN=legYN, extralab=extralab, lw=lw, fs=fs)
        
    # To allow this to be called from the object
    def plot_gradients(self, *args, **kwargs):
        return state_plotting.plot_gradients(self, *args, **kwargs)
        
    def plot_geometry(self, *args, **kwargs):
        pass

    def plot_flows(self, *args, **kwargs):
        return state_plotting.plot_flows(self, *args, **kwargs)

    def plotPeaking(
        self, ax, c="b", marker="*", label="", debugPlot=False, printVals=False
        ):
        nu_effCGYRO = self.derived["nu_eff"] * 2 / self.derived["Zeff_vol"]
        ne_peaking = self.derived["ne_peaking0.2"]
        ax.scatter([nu_effCGYRO], [ne_peaking], s=400, c=c, marker=marker, label=label)

        if printVals:
            print(f"\t- nu_eff = {nu_effCGYRO}, ne_peaking = {ne_peaking}")

        # Extra
        r = self.derived["r"]
        volp = self.derived["volp_geo"]
        ix = np.argmin(np.abs(self.profiles["rho(-)"] - 0.9))

        if debugPlot:
            fig, axq = plt.subplots()

            ne = self.profiles["ne(10^19/m^3)"]
            axq.plot(self.profiles["rho(-)"], ne, color="m")
            ne_vol = (
                CALCtools.volume_integration(ne * 0.1, r, volp)[-1] / self.derived["volume"]
            )
            axq.axhline(y=ne_vol * 10, color="m")

        ne = copy.deepcopy(self.profiles["ne(10^19/m^3)"])
        ne[ix:] = (0,) * len(ne[ix:])
        ne_vol = CALCtools.volume_integration(ne * 0.1, r, volp)[-1] / self.derived["volume"]
        ne_peaking0 = (
            ne[np.argmin(np.abs(self.derived["rho_pol"] - 0.2))] * 0.1 / ne_vol
        )

        if debugPlot:
            axq.plot(self.profiles["rho(-)"], ne, color="r")
            axq.axhline(y=ne_vol * 10, color="r")

        ne = copy.deepcopy(self.profiles["ne(10^19/m^3)"])
        ne[ix:] = (ne[ix],) * len(ne[ix:])
        ne_vol = CALCtools.volume_integration(ne * 0.1, r, volp)[-1] / self.derived["volume"]
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

    def plotRelevant(self, axs = None, color = 'b', label ='', lw = 1, ms = 1):

        if axs is None:
            fig = plt.figure()
            axs = fig.subplot_mosaic(
                """
                    ABCDH
                    AEFGI
                """
            )
            axs = [axs['A'], axs['B'], axs['C'], axs['D'], axs['E'], axs['F'], axs['G'], axs['H'], axs['I']]

        # ----------------------------------
        # Equilibria
        # ----------------------------------

        ax = axs[0]
        rho = np.linspace(0, 1, 21)
        
        self.plot_state_flux_surfaces(ax=ax, surfaces_rho=rho, label=label, color=color, lw=lw, lw1=lw*3)

        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Equilibria")

        # ----------------------------------
        # Kinetic Profiles
        # ----------------------------------

        # T profiles
        ax = axs[1]

        ax.plot(self.profiles['rho(-)'], self.profiles['te(keV)'], '-o', markersize=ms, lw = lw, label=label+', e', color=color)
        ax.plot(self.profiles['rho(-)'], self.profiles['ti(keV)'][:,0], '--*', markersize=ms, lw = lw, label=label+', i', color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$T$ (keV)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Temperatures")

        # ne profiles
        ax = axs[2]

        ax.plot(self.profiles['rho(-)'], self.profiles['ne(10^19/m^3)']*1E-1, '-o', markersize=ms, lw = lw, label=label, color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Electron Density")

        # ----------------------------------
        # Pressure
        # ----------------------------------

        ax = axs[3]

        ax.plot(self.profiles['rho(-)'], self.derived['ptot_manual'], '-o', markersize=ms, lw = lw, label=label, color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$p_{kin}$ (MPa)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Total Pressure")

        # ----------------------------------
        # Current
        # ----------------------------------

        # q-profile
        ax = axs[4]

        ax.plot(self.profiles['rho(-)'], self.profiles['q(-)'], '-o', markersize=ms, lw = lw, label=label, color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$q$")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Safety Factor")

        # ----------------------------------
        # Powers
        # ----------------------------------

        # RF
        ax = axs[5]

        ax.plot(self.profiles['rho(-)'], self.profiles['qrfe(MW/m^3)'], '-o', markersize=ms, lw = lw, label=label+', e', color=color)
        ax.plot(self.profiles['rho(-)'], self.profiles['qrfi(MW/m^3)'], '--*', markersize=ms, lw = lw, label=label+', i', color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$P_{ich}$ (MW/m$^3$)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("ICH Power Deposition")

        # Ohmic
        ax = axs[6]

        ax.plot(self.profiles['rho(-)'], self.profiles['qohme(MW/m^3)'], '-o', markersize=ms, lw = lw, label=label, color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$P_{oh}$ (MW/m$^3$)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Ohmic Power Deposition")

        # ----------------------------------
        # Heat fluxes
        # ----------------------------------

        ax = axs[7]

        ax.plot(self.profiles['rho(-)'], self.derived['qe_MWm2'], '-o', markersize=ms, lw = lw, label=label+', e', color=color)
        ax.plot(self.profiles['rho(-)'], self.derived['qi_MWm2'], '--*', markersize=ms, lw = lw, label=label+', i', color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$Q$ ($MW/m^2$)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Energy Fluxes")

        # ----------------------------------
        # Dynamic targets
        # ----------------------------------

        ax = axs[8]

        ax.plot(self.profiles['rho(-)'], self.derived['qrad'], '-o', markersize=ms, lw = lw, label=label+', rad', color=color)
        ax.plot(self.profiles['rho(-)'], self.profiles['qei(MW/m^3)'], '--*', markersize=ms, lw = lw, label=label+', exc', color=color)
        if 'qfuse(MW/m^3)' in self.profiles:
            ax.plot(self.profiles['rho(-)'], self.profiles['qfuse(MW/m^3)']+self.profiles['qfusi(MW/m^3)'], '-.s', markersize=ms, lw = lw, label=label+', fus', color=color)

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$Q$ ($MW/m^2$)")
        #ax.set_ylim(bottom = 0)
        ax.set_xlim(0,1)
        ax.legend(prop={'size':8})
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title("Dynamic Targets")

    def csv(self, file="input.gacode.xlsx"):
        dictExcel = IOtools.OrderedDict()

        for ikey in self.profiles:
            print(ikey)
            if len(self.profiles[ikey].shape) == 1:
                dictExcel[ikey] = self.profiles[ikey]
            else:
                dictExcel[ikey] = self.profiles[ikey][:, 0]

        IOtools.writeExcel_fromDict(dictExcel, file, fromRow=1)

    # ************************************************************************************************************************************************
    # Code conversions
    # ************************************************************************************************************************************************

    def _print_gb_normalizations(self,L_label,Z_label,A_label,n_label,T_label, B_label, L, Z, A):
        print(f'\t- GB normalizations, such that Q_gb = n_ref * T_ref^5/2 * m_ref^0.5 / (Z_ref * L_ref * B_ref)^2')
        print(f'\t\t* L_ref = {L_label} = {L:.3f}')
        print(f'\t\t* Z_ref = {Z_label} = {Z:.3f}')
        print(f'\t\t* A_ref = {A_label} = {A:.3f}')
        print(f'\t\t* B_ref = {B_label}')
        print(f'\t\t* n_ref = {n_label}')
        print(f'\t\t* T_ref = {T_label}')
        print(f'')

    def _calculate_pressure_gradient_from_aLx(self, pe, pi, aLTe, aLTi, aLne, aLni, a):
        '''
        pe and pi in MPa. pi two dimensional
        '''

        adpedr = - pe * 1E6 * (aLTe + aLne) 
        adpjdr = - pi * 1E6 * (aLTi + aLni)

        dpdr  = ( adpedr + adpjdr.sum(axis=-1)) / a 
        
        return dpdr

    def to_tglf(self, r=[0.5], code_settings='SAT0', r_is_rho = True):

        # <> Function to interpolate a curve <> 
        from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

        # Determine if the input radius is rho toroidal or r/a
        if r_is_rho:
            r_interpolation = self.profiles['rho(-)']
        else:
            r_interpolation = self.derived['roa']

        # Determine the number of species to use in TGLF
        max_species_tglf = 6  # TGLF only accepts up to 6 species  
        if len(self.Species) > max_species_tglf-1:
            print(f"\t- Warning: TGLF only accepts {max_species_tglf} species, but there are {len(self.Species)} ions pecies in the GACODE input. The first {max_species_tglf-1} will be used.", typeMsg="w")
            tglf_ions_num = max_species_tglf - 1
        else:
            tglf_ions_num = len(self.Species)

        # Determine the mass reference for TGLF (use 2.0 for D-mass normalization; derivatives use mD_u elsewhere)
        mass_ref = 2.0

        self._print_gb_normalizations('a', 'Z_D', 'A_D', 'n_e', 'T_e', 'B_unit', self.derived["a"], 1.0, mass_ref)

        # -----------------------------------------------------------------------
        # Derived profiles
        # -----------------------------------------------------------------------
        
        sign_it = -np.sign(self.profiles["current(MA)"][-1])
        sign_bt = -np.sign(self.profiles["bcentr(T)"][-1])

        s_kappa  = self.derived["r"] / self.profiles["kappa(-)"] * self._deriv_gacode(self.profiles["kappa(-)"])
        s_delta  = self.derived["r"]                             * self._deriv_gacode(self.profiles["delta(-)"])
        s_zeta   = self.derived["r"]                             * self._deriv_gacode(self.profiles["zeta(-)"])
        
        '''
        Total pressure
        --------------------------------------------------------
            Recompute pprime with those species that belong to this run             #TODO not exact?
        '''
        
        dpdr = self._calculate_pressure_gradient_from_aLx(
            self.derived['pe'], self.derived['pi_all'][:,:tglf_ions_num],
            self.derived['aLTe'], self.derived['aLTi'][:,:tglf_ions_num],
            self.derived['aLne'], self.derived['aLni'][:,:tglf_ions_num],
            self.derived['a']
        )
        
        pprime = 1E-7 * abs(self.profiles["q(-)"])*self.derived['a']**2/self.derived["r"]/self.derived["B_unit"]**2*dpdr
        pprime[0] = 0 # infinite in first location

        '''
        Rotations
        --------------------------------------------------------
            From TGYRO/TGLF definitions
                  w0p = expro_w0p(:)/100.0
                  f_rot(:) = w0p(:)/w0_norm
                  gamma_p0  = -r_maj(i_r)*f_rot(i_r)*w0_norm
                  gamma_eb0 = gamma_p0*r(i_r)/(q_abs*r_maj(i_r)) 
        '''

        w0p         = self._deriv_gacode(self.profiles["w0(rad/s)"])
        gamma_p0    = -self.profiles["rmaj(m)"]*w0p
        gamma_eb0   = -self._deriv_gacode(self.profiles["w0(rad/s)"]) * self.derived["r"]/ np.abs(self.profiles["q(-)"])

        vexb_shear  = -sign_it * gamma_eb0 * self.derived["a"]/self.derived['c_s']
        vpar_shear  = -sign_it * gamma_p0  * self.derived["a"]/self.derived['c_s']
        vpar        = -sign_it * self.profiles["rmaj(m)"]*self.profiles["w0(rad/s)"]/self.derived['c_s']

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Prepare the inputs for TGLF
        # ---------------------------------------------------------------------------------------------------------------------------------------

        input_parameters = {}
        for rho in r:

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Define interpolator at this rho
            # ---------------------------------------------------------------------------------------------------------------------------------------

            def interpolator(y):
                return interpolation_function(rho, r_interpolation,y).item()
            
            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Controls come from options
            # ---------------------------------------------------------------------------------------------------------------------------------------
            
            controls = GACODEdefaults.addTGLFcontrol(code_settings)

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Species come from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            species = {
                1: {
                    'ZS': -1.0,
                    'MASS': PLASMAtools.me_u / mass_ref,
                    'RLNS': interpolator(self.derived['aLne']),
                    'RLTS': interpolator(self.derived['aLTe']),
                    'TAUS': 1.0,
                    'AS': 1.0,
                    'VPAR': interpolator(vpar),
                    'VPAR_SHEAR': interpolator(vpar_shear),
                    'VNS_SHEAR': 0.0,
                    'VTS_SHEAR': 0.0},
            }

            for i in range(min(len(self.Species), max_species_tglf-1)):
                species[i+2] = {
                    'ZS': self.Species[i]['Z'],
                    'MASS': self.Species[i]['A']/mass_ref,
                    'RLNS': interpolator(self.derived['aLni'][:,i]),
                    'RLTS': interpolator(self.derived["aLTi"][:,i]),
                    'TAUS': interpolator(self.derived["tite_all"][:,i]),
                    'AS': interpolator(self.derived['fi'][:,i]),
                    'VPAR': interpolator(vpar),
                    'VPAR_SHEAR': interpolator(vpar_shear),
                    'VNS_SHEAR': 0.0,
                    'VTS_SHEAR': 0.0
                    }

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Plasma comes from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            plasma = {
                'NS': len(species),
                'SIGN_BT': sign_bt,
                'SIGN_IT': sign_it,
                'VEXB': 0.0,
                'VEXB_SHEAR': interpolator(vexb_shear),
                'XNUE': interpolator(self.derived['xnue']),
                'ZEFF': interpolator(self.derived['Zeff']),
                'DEBYE': interpolator(self.derived['debye']),
                'BETAE': interpolator(self.derived['betae']),
                }


            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Geometry comes from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            parameters = {
                'RMIN_LOC':     self.derived['roa'],
                'RMAJ_LOC':     self.derived['Rmajoa'],
                'ZMAJ_LOC':     self.derived["Zmagoa"],
                'DRMINDX_LOC':  np.ones(self.profiles["rho(-)"].shape), # Force 1.0 because of numerical issues in TGLF
                'DRMAJDX_LOC':  self._deriv_gacode(self.profiles["rmaj(m)"]),
                'DZMAJDX_LOC':  self._deriv_gacode(self.profiles["zmag(m)"]),
                'Q_LOC':        np.abs(self.profiles["q(-)"]),
                'KAPPA_LOC':    self.profiles["kappa(-)"],
                'S_KAPPA_LOC':  s_kappa,
                'DELTA_LOC':    self.profiles["delta(-)"],
                'S_DELTA_LOC':  s_delta,
                'ZETA_LOC':     self.profiles["zeta(-)"],
                'S_ZETA_LOC':   s_zeta,
                'Q_PRIME_LOC':  self.derived['s_q'],
                'P_PRIME_LOC':  pprime,
            }
            
            # Add MXH and derivatives
            for ikey in self.profiles:
                if 'shape_cos' in ikey or 'shape_sin' in ikey:
                    
                    # TGLF only accepts 6, as of July 2025
                    if int(ikey[-4]) > 6:
                        continue
                    
                    key_mod = ikey.upper().split('(')[0]
                    
                    parameters[key_mod] = self.profiles[ikey]
                    parameters[f"{key_mod.split('_')[0]}_S_{key_mod.split('_')[-1]}"] = self.derived["r"] * self._deriv_gacode(self.profiles[ikey])

            for k in parameters:
                par = torch.nan_to_num(torch.from_numpy(parameters[k]) if type(parameters[k]) is np.ndarray else parameters[k], nan=0.0, posinf=1E10, neginf=-1E10)
                plasma[k] = interpolator(par)

            plasma['BETA_LOC'] = 0.0
            plasma['KX0_LOC'] = 0.0

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Merging
            # ---------------------------------------------------------------------------------------------------------------------------------------

            input_dict = controls | plasma

            for i in range(len(species)):
                for k in species[i+1]:
                    input_dict[f'{k}_{i+1}'] = species[i+1][k]

            input_parameters[rho] = input_dict
            
        return input_parameters

    def to_neo(self, r=[0.5], r_is_rho = True, code_settings='Sonic'):

        # <> Function to interpolate a curve <> 
        from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

        # Determine if the input radius is rho toroidal or r/a
        if r_is_rho:
            r_interpolation = self.profiles['rho(-)']
        else:
            r_interpolation = self.derived['roa']

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Prepare the inputs
        # ---------------------------------------------------------------------------------------------------------------------------------------

        # Determine the mass reference
        mass_ref = 2.0
        
        sign_it = int(-np.sign(self.profiles["current(MA)"][-1]))
        sign_bt = int(-np.sign(self.profiles["bcentr(T)"][-1]))
        
        s_kappa  = self.derived["r"] / self.profiles["kappa(-)"] * self._deriv_gacode(self.profiles["kappa(-)"])
        s_delta  = self.derived["r"]                             * self._deriv_gacode(self.profiles["delta(-)"])
        s_zeta   = self.derived["r"]                             * self._deriv_gacode(self.profiles["zeta(-)"])

        # Rotations
        rmaj = self.derived['Rmajoa']
        cs = self.derived['c_s']
        a = self.derived['a']
        mach = self.profiles["w0(rad/s)"] * (self.derived['Rmajoa']*a)
        gamma_p = self._deriv_gacode(self.profiles["w0(rad/s)"]) * (self.derived['Rmajoa']*a)

        # NEO definition: 'OMEGA_ROT=',mach_loc/rmaj_loc/cs_loc
        omega_rot = mach / rmaj / cs        # Equivalent to: self.profiles["w0(rad/s)"] / self.derived['c_s'] * a
        
        # NEO definition: 'OMEGA_ROT_DERIV=',-gamma_p_loc*a/cs_loc/rmaj_loc
        omega_rot_deriv = gamma_p * a / cs / rmaj # Equivalent to: self._deriv_gacode(self.profiles["w0(rad/s)"])/ self.derived['c_s'] * self.derived['a']**2

        self._print_gb_normalizations('a', 'Z_D', 'A_D', 'n_e', 'T_e', 'B_unit', self.derived["a"], 1.0, mass_ref)

        input_parameters = {}
        for rho in r:

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Define interpolator at this rho
            # ---------------------------------------------------------------------------------------------------------------------------------------

            def interpolator(y):
                return interpolation_function(rho, r_interpolation,y).item()

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Controls come from options
            # ---------------------------------------------------------------------------------------------------------------------------------------
            
            controls = GACODEdefaults.addNEOcontrol(code_settings)

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Species come from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            species = {}
            for i in range(len(self.Species)):
                species[i+1] = {
                    'Z': self.Species[i]['Z'],
                    'MASS': self.Species[i]['A']/mass_ref,
                    'DLNNDR': interpolator(self.derived['aLni'][:,i]),
                    'DLNTDR': interpolator(self.derived["aLTi"][:,i]),
                    'TEMP': interpolator(self.derived["tite_all"][:,i]),
                    'DENS': interpolator(self.derived['fi'][:,i]),
                    }

            ie = i+2
            species[ie] = {
                    'Z': -1.0,
                    'MASS': 0.000272445,
                    'DLNNDR': interpolator(self.derived['aLne']),
                    'DLNTDR': interpolator(self.derived['aLTe']),
                    'TEMP': 1.0,
                    'DENS': 1.0,
                }

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Plasma comes from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            #TODO  Does this work with no deuterium first ion?
            factor_nu = species[1]['Z']**4 * species[1]['DENS'] * (species[ie]['MASS']/species[1]['MASS'])**0.5 * species[1]['TEMP']**(-1.5)
            
            plasma = {
                'N_SPECIES': len(species),
                'IPCCW': sign_bt,
                'BTCCW': sign_it,
                'OMEGA_ROT': interpolator(omega_rot),
                'OMEGA_ROT_DERIV': interpolator(omega_rot_deriv),
                'NU_1': interpolator(self.derived['xnue'])* factor_nu,
                'RHO_STAR': interpolator(self.derived["rho_sa"]),
                }


            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Geometry comes from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            parameters = {
                'RMIN_OVER_A':  self.derived['roa'],
                'RMAJ_OVER_A':  self.derived['Rmajoa'],
                'SHIFT':         self._deriv_gacode(self.profiles["rmaj(m)"]),
                'ZMAG_OVER_A':  self.derived["Zmagoa"],
                'S_ZMAG':       self._deriv_gacode(self.profiles["zmag(m)"]),
                'Q':            np.abs(self.profiles["q(-)"]),
                'SHEAR':        self.derived["s_hat"],
                'KAPPA':        self.profiles["kappa(-)"],
                'S_KAPPA':      s_kappa,
                'DELTA':        self.profiles["delta(-)"],
                'S_DELTA':      s_delta,
                'ZETA':         self.profiles["zeta(-)"],
                'S_ZETA':       s_zeta,
            }
            
            # Add MXH and derivatives
            for ikey in self.profiles:
                if 'shape_cos' in ikey or 'shape_sin' in ikey:
                    
                    # TGLF only accepts 6, as of July 2025
                    if int(ikey[-4]) > 6:
                        continue
                    
                    key_mod = ikey.upper().split('(')[0]
                    
                    parameters[key_mod] = self.profiles[ikey]
                    parameters[f"{key_mod.split('_')[0]}_S_{key_mod.split('_')[-1]}"] = self.derived["r"] * self._deriv_gacode(self.profiles[ikey])

            for k in parameters:
                par = torch.nan_to_num(torch.from_numpy(parameters[k]) if type(parameters[k]) is np.ndarray else parameters[k], nan=0.0, posinf=1E10, neginf=-1E10)
                plasma[k] = interpolator(par)

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Merging
            # ---------------------------------------------------------------------------------------------------------------------------------------

            input_dict = controls | plasma

            for i in range(len(species)):
                for k in species[i+1]:
                    input_dict[f'{k}_{i+1}'] = species[i+1][k]

            input_parameters[rho] = input_dict

        return input_parameters

    def to_cgyro(self, r=[0.5], r_is_rho = True, code_settings = 'Linear'):

        # <> Function to interpolate a curve <> 
        from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

        # Determine if the input radius is rho toroidal or r/a
        if r_is_rho:
            r_interpolation = self.profiles['rho(-)']
        else:
            r_interpolation = self.derived['roa']
            
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Prepare the inputs
        # ---------------------------------------------------------------------------------------------------------------------------------------

        # Determine the mass reference
        mass_ref = 2.0
        
        sign_it = int(-np.sign(self.profiles["current(MA)"][-1]))
        sign_bt = int(-np.sign(self.profiles["bcentr(T)"][-1]))
        
        s_kappa  = self.derived["r"] / self.profiles["kappa(-)"] * self._deriv_gacode(self.profiles["kappa(-)"])
        s_delta  = self.derived["r"]                             * self._deriv_gacode(self.profiles["delta(-)"])
        s_zeta   = self.derived["r"]                             * self._deriv_gacode(self.profiles["zeta(-)"])

        # Rotations
        cs = self.derived['c_s']
        a = self.derived['a']
        mach = self.profiles["w0(rad/s)"] * (self.derived['Rmajoa']*a)
        gamma_p = -self._deriv_gacode(self.profiles["w0(rad/s)"]) * (self.derived['Rmajoa']*a)
        gamma_e = -self._deriv_gacode(self.profiles["w0(rad/s)"]) * (self.profiles['rmin(m)'] / self.profiles['q(-)'])

        # CGYRO definition: 'MACH=',mach_loc/cs_loc
        mach = mach / cs

        # CGYRO definition: 'GAMMA_P=',gamma_p_loc*a/cs_loc
        gamma_p = gamma_p * a / cs

        # CGYRO definition: 'GAMMA_E=',gamma_e_loc*a/cs_loc
        gamma_e = gamma_e * a / cs
            
        # Because in MITIMstate I keep Bunit always positive, but CGYRO routines may need it negative? #TODO
        sign_Bunit = np.sign(self.profiles['torfluxa(Wb/radian)'][0])
            
        self._print_gb_normalizations('a', 'Z_D', 'A_D', 'n_e', 'T_e', 'B_unit', self.derived["a"], 1.0, mass_ref)
            
        input_parameters = {}
        for rho in r:

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Define interpolator at this rho
            # ---------------------------------------------------------------------------------------------------------------------------------------

            def interpolator(y):
                return interpolation_function(rho, r_interpolation,y).item()

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Controls come from options
            # ---------------------------------------------------------------------------------------------------------------------------------------
            
            controls = GACODEdefaults.addCGYROcontrol(code_settings)
            controls['PROFILE_MODEL'] = 1

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Species come from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            species = {}
            for i in range(len(self.Species)):
                species[i+1] = {
                    'Z': self.Species[i]['Z'],
                    'MASS': self.Species[i]['A']/mass_ref,
                    'DLNNDR': interpolator(self.derived['aLni'][:,i]),
                    'DLNTDR': interpolator(self.derived["aLTi"][:,i]),
                    'TEMP': interpolator(self.derived["tite_all"][:,i]),
                    'DENS': interpolator(self.derived['fi'][:,i]),
                    }

            ie = i+2
            species[ie] = {
                    'Z': -1.0,
                    'MASS': 0.000272445,
                    'DLNNDR': interpolator(self.derived['aLne']),
                    'DLNTDR': interpolator(self.derived['aLTe']),
                    'TEMP': 1.0,
                    'DENS': 1.0,
                }

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Plasma comes from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            plasma = {
                'N_SPECIES': len(species),
                'IPCCW': sign_bt,
                'BTCCW': sign_it,
                'MACH': interpolator(mach),
                'GAMMA_E': interpolator(gamma_e),
                'GAMMA_P': interpolator(gamma_p),
                'NU_EE': interpolator(self.derived['xnue']),
                'BETAE_UNIT': interpolator(self.derived['betae']),
                'LAMBDA_STAR': interpolator(self.derived['debye']) * sign_Bunit,
                }


            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Geometry comes from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            parameters = {
                'RMIN':     self.derived['roa'],
                'RMAJ':     self.derived['Rmajoa'],
                'SHIFT':    self._deriv_gacode(self.profiles["rmaj(m)"]),
                'ZMAG':     self.derived["Zmagoa"],
                'DZMAG':    self._deriv_gacode(self.profiles["zmag(m)"]),
                'Q':        np.abs(self.profiles["q(-)"]),
                'S':        self.derived["s_hat"],
                'KAPPA':    self.profiles["kappa(-)"],
                'S_KAPPA':  s_kappa,
                'DELTA':    self.profiles["delta(-)"],
                'S_DELTA':  s_delta,
                'ZETA':     self.profiles["zeta(-)"],
                'S_ZETA':   s_zeta,
            }
            
            # Add MXH and derivatives
            for ikey in self.profiles:
                if 'shape_cos' in ikey or 'shape_sin' in ikey:
                    
                    # TGLF only accepts 6, as of July 2025
                    if int(ikey[-4]) > 6:
                        continue
                    
                    key_mod = ikey.upper().split('(')[0]
                    
                    parameters[key_mod] = self.profiles[ikey]
                    parameters[f"{key_mod.split('_')[0]}_S_{key_mod.split('_')[-1]}"] = self.derived["r"] * self._deriv_gacode(self.profiles[ikey])

            for k in parameters:
                par = torch.nan_to_num(torch.from_numpy(parameters[k]) if type(parameters[k]) is np.ndarray else parameters[k], nan=0.0, posinf=1E10, neginf=-1E10)
                plasma[k] = interpolator(par)

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Merging
            # ---------------------------------------------------------------------------------------------------------------------------------------

            input_dict = controls | plasma

            for i in range(len(species)):
                for k in species[i+1]:
                    input_dict[f'{k}_{i+1}'] = species[i+1][k]

            input_parameters[rho] = input_dict

        return input_parameters

    def to_gx(self, r=[0.5], r_is_rho = True, code_settings = 'Linear'):

        # <> Function to interpolate a curve <> 
        from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

        # Determine if the input radius is rho toroidal or r/a
        if r_is_rho:
            r_interpolation = self.profiles['rho(-)']
        else:
            r_interpolation = self.derived['roa']
            
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Prepare the inputs
        # ---------------------------------------------------------------------------------------------------------------------------------------
          
        # Determine the mass reference
        mass_ref = 2.0

        dpdr = self._calculate_pressure_gradient_from_aLx(
            self.derived['pe'], self.derived['pi_all'][:,:],
            self.derived['aLTe'], self.derived['aLTi'][:,:],
            self.derived['aLne'], self.derived['aLni'][:,:],
            self.derived['a']
        )
        betaprim = -(8*np.pi*1E-7) * self.derived['a'] / self.derived['B_unit']**2 * dpdr
        
        #TODO #to check
        s_kappa  = self.derived["r"] / self.profiles["kappa(-)"] * self._deriv_gacode(self.profiles["kappa(-)"])
        s_delta  = self.derived["r"]                             * self._deriv_gacode(self.profiles["delta(-)"])

        self._print_gb_normalizations('a', 'Z_D', 'A_D', 'n_e', 'T_e', 'B_unit', self.derived["a"], 1.0, mass_ref)
            
        input_parameters = {}
        for rho in r:

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Define interpolator at this rho
            # ---------------------------------------------------------------------------------------------------------------------------------------

            def interpolator(y):
                return interpolation_function(rho, r_interpolation,y).item()

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Controls come from options
            # ---------------------------------------------------------------------------------------------------------------------------------------
            
            controls = GACODEdefaults.addGXcontrol(code_settings)

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Species come from profiles
            # ---------------------------------------------------------------------------------------------------------------------------------------

            species = {}

            # Ions
            for i in range(len(self.Species)):

                nu_ii = self.derived['xnue'] * \
                    (self.Species[i]['Z']/self.profiles['ze'][0])**4 * \
                        (self.profiles['ni(10^19/m^3)'][:,0]/self.profiles['ne(10^19/m^3)']) * \
                            (self.profiles['mass'][i]/self.profiles['masse'][0])**-0.5 * \
                                (self.profiles['ti(keV)'][:,0]/self.profiles['te(keV)'])**-1.5

                species[i+1] = {
                    'z': self.Species[i]['Z'],
                    'mass': self.Species[i]['A']/mass_ref,
                    'temp': interpolator(self.derived["tite_all"][:,i]),
                    'dens': interpolator(self.derived['fi'][:,i]),
                    'fprim': interpolator(self.derived['aLni'][:,i]),
                    'tprim': interpolator(self.derived["aLTi"][:,i]),
                    'vnewk': interpolator(nu_ii),
                    'type': 'ion',
                    }
                
            # Electrons
            ie = i+2
            species[ie] = {
                    'z': -1.0,
                    'mass': 0.000272445,
                    'temp': 1.0,
                    'dens': 1.0,
                    'fprim': interpolator(self.derived['aLne']),
                    'tprim': interpolator(self.derived['aLTe']),
                    'vnewk': interpolator(self.derived['xnue']),
                    'type': 'electron'
                }

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Plasma and geometry
            # ---------------------------------------------------------------------------------------------------------------------------------------

            plasma = {
                'nspecies': len(species)
            } 

            parameters = {
                'beta':     self.derived['betae'],
                'rhoc':     self.derived['roa'],
                'Rmaj':     self.derived['Rmajoa'],
                'R_geo':    self.derived['Rmajoa'] / abs(self.derived['B_unit'] / self.derived['B0']),
                'shift':    self._deriv_gacode(self.profiles["rmaj(m)"]),
                'qinp':     np.abs(self.profiles["q(-)"]),
                'shat':     self.derived["s_hat"],
                'akappa':    self.profiles["kappa(-)"],
                'akappri':  s_kappa,
                'tri':    self.profiles["delta(-)"],
                'tripri':   s_delta,
                'betaprim':    betaprim,
            }
            
            for k in parameters:
                par = torch.nan_to_num(torch.from_numpy(parameters[k]) if type(parameters[k]) is np.ndarray else parameters[k], nan=0.0, posinf=1E10, neginf=-1E10)
                plasma[k] = interpolator(par)

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Merging
            # ---------------------------------------------------------------------------------------------------------------------------------------

            input_dict = controls | plasma

            for i in range(len(species)):
                for k in species[i+1]:
                    input_dict[f'{k}_{i+1}'] = species[i+1][k]

            input_parameters[rho] = input_dict

        return input_parameters
    

    def to_transp(self, folder = '~/scratch/', shot = '12345', runid = 'P01', times = [0.0,1.0], Vsurf = 0.0):

        print("\t- Converting to TRANSP")
        folder = IOtools.expandPath(folder)
        folder.mkdir(parents=True, exist_ok=True)

        from mitim_tools.transp_tools.utils import TRANSPhelpers
        transp = TRANSPhelpers.transp_run(folder, shot, runid)
        for time in times:
            transp.populate_time.from_profiles(time,self, Vsurf = Vsurf)

        transp.write_ufiles()

        return transp

    def to_eped(self, ped_rho = 0.95):

        neped_19 = np.interp(ped_rho, self.profiles['rho(-)'], self.profiles['ne(10^19/m^3)'])

        eped_evaluation = {
            'Ip': np.abs(self.profiles['current(MA)'][0]),
            'Bt': np.abs(self.profiles['bcentr(T)'][0]),
            'R': np.abs(self.profiles['rcentr(m)'][0]),
            'a': np.abs(self.derived['a']),
            'kappa995': np.abs(self.derived['kappa995']),
            'delta995': np.abs(self.derived['delta995']),
            'neped': np.abs(neped_19),
            'betan': np.abs(self.derived['BetaN_engineering']),
            'zeff': np.abs(self.derived['Zeff_vol']),
            'tesep': np.abs(self.profiles['te(keV)'][-1])*1E3,
            'nesep_ratio': np.abs(self.profiles['ne(10^19/m^3)'][-1] / neped_19),
        }

        return eped_evaluation


class DataTable:
    def __init__(self, variables=None):

        if variables is not None:
            self.variables = variables
        else:

            # Default for confinement mode access studies (JWH 03/2024)
            self.variables = {
                "Rgeo": ["rcentr(m)", "pos_0", "profiles", ".2f", 1, "m"],
                "ageo": ["a", None, "derived", ".2f", 1, "m"],
                "volume": ["volume", None, "derived", ".2f", 1, "m"],
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
                "H98y2": ["H98", None, "derived", ".2f", 1, None],
                "ne (vol avg)": ["ne_vol20", None, "derived", ".2f", 1, "E20m-3"],
                "Ptop": ["ptop", None, "derived", ".1f", 1, "Pa"],
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

def aLT(r, p):
    return (
        r[-1]
        * CALCtools.derivation_into_Lx(
            torch.from_numpy(r).to(torch.double), torch.from_numpy(p).to(torch.double)
        )
        .cpu()
        .cpu().numpy()
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
        CALCtools.integration_Lx(
            torch.from_numpy(p.derived["roa"]).unsqueeze(0),
            torch.Tensor(aLTe).unsqueeze(0),
            p.profiles["te(keV)"][-1],
        )
        .cpu()
        .cpu().numpy()[0]
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
        CALCtools.integration_Lx(
            torch.from_numpy(p.derived["roa"]).unsqueeze(0),
            torch.Tensor(aLTi).unsqueeze(0),
            p.profiles["ti(keV)"][-1, 0],
        )
        .cpu()
        .cpu().numpy()[0]
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
        CALCtools.integration_Lx(
            torch.from_numpy(p.derived["roa"]).unsqueeze(0),
            torch.Tensor(aLne).unsqueeze(0),
            p.profiles["ne(10^19/m^3)"][-1],
        )
        .cpu()
        .cpu().numpy()[0]
    )

    p.profiles["te(keV)"] = Te
    p.profiles["ti(keV)"][:, 0] = Ti
    p.profiles["ne(10^19/m^3)"] = ne

    p.derive_quantities()

    return p

def impurity_location(profiles, impurity_of_interest):

    position_of_impurity = None
    for i in range(len(profiles.Species)):
        if profiles.Species[i]["N"] == impurity_of_interest:
            if position_of_impurity is not None:
                raise ValueError(f"[MITIM] Species {impurity_of_interest} found at positions {position_of_impurity} and {i}")
            position_of_impurity = i
    if position_of_impurity is None:
        raise ValueError(f"[MITIM] Species {impurity_of_interest} not found in profiles")

    return position_of_impurity
