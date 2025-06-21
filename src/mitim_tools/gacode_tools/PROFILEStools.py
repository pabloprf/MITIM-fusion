import copy
import numpy as np
from collections import OrderedDict
from mitim_tools.plasmastate_tools.MITIMstate import mitim_state
from mitim_tools.gs_tools import GEQtools
from mitim_tools.gacode_tools.utils import GEOMETRYtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class gacode_state(mitim_state):
    '''
    Class to read and manipulate GACODE profiles files (input.gacode).
    It inherits from the main MITIMstate class, which provides basic
    functionality for plasma state management.

    The class reads the GACODE profiles file, extracts relevant data,
    and writes them in the way that MITIMstate class expects.
    '''

    # ------------------------------------------------------------------
    # Reading and interpreting
    # ------------------------------------------------------------------

    def __init__(self, file, calculateDerived=True, mi_ref=None):

        super().__init__(type_file='input.gacode')

        self.file = file

        self._read_inputgacocde()

        if self.file is not None:
            # Derive (Depending on resolution, derived can be expensive, so I mmay not do it every time)
            self.derive_quantities(mi_ref=mi_ref, calculateDerived=calculateDerived)

    def _read_inputgacocde(self):

        self.titles_singleNum = ["nexp", "nion", "shot", "name", "type", "time"]
        self.titles_singleArr = ["masse","mass","ze","z","torfluxa(Wb/radian)","rcentr(m)","bcentr(T)","current(MA)"]
        self.titles_single = self.titles_singleNum + self.titles_singleArr
        
        if self.file is not None:
            with open(self.file, "r") as f:
                self.lines = f.readlines()

            # Read file and store raw data
            self._read_header()
            self._read_profiles()
            self._ensure_existence()

    def _read_header(self):
        for i in range(len(self.lines)):
            if "# nexp" in self.lines[i]:
                istartProfs = i
        self.header = self.lines[:istartProfs]

    def _read_profiles(self):
        singleLine, title, var = None, None, None 

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
                    varT = [float(j) if (j[-4].upper() == "E" or "." in j) else 0.0for j in var0[1:]]

                    var.append(varT)

        # last
        if not singleLine:
            while len(var[-1]) < 1:
                var = var[:-1]  # Sometimes there's an extra space, remove
            self.profiles[title] = np.array(var)
            if self.profiles[title].shape[1] == 1:
                self.profiles[title] = self.profiles[title][:, 0]

        # Accept omega0
        if ("w0(rad/s)" not in self.profiles) and ("omega0(rad/s)" in self.profiles):
            self.profiles["w0(rad/s)"] = self.profiles["omega0(rad/s)"]
            del self.profiles["omega0(rad/s)"]

    def _ensure_existence(self):
        # Calculate necessary quantities

        if "qpar_beam(MW/m^3)" in self.profiles:
            self.varqpar, self.varqpar2 = "qpar_beam(MW/m^3)", "qpar_wall(MW/m^3)"
        else:
            self.varqpar, self.varqpar2 = "qpar_beam(1/m^3/s)", "qpar_wall(1/m^3/s)"

        if "qmom(Nm)" in self.profiles:
            self.varqmom = "qmom(Nm)"  # Old, wrong one. But Candy fixed it as of 02/24/2023
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
            "qsync(MW/m^3)",
            "qbrem(MW/m^3)",
            "qline(MW/m^3)",
            self.varqpar,
            self.varqpar2,
            "shape_cos0(-)",
            self.varqmom,
        ]

        num_moments = 7  # This is the max number of moments I'll be considering. If I don't have that many (usually there are 5 or 3), it'll be populated with zeros
        for i in range(num_moments):
            some_times_are_not_here.append(f"shape_cos{i + 1}(-)")
            if i > 1:
                some_times_are_not_here.append(f"shape_sin{i + 1}(-)")

        for ikey in some_times_are_not_here:
            if ikey not in self.profiles.keys():
                self.profiles[ikey] = copy.deepcopy(self.profiles["rmin(m)"]) * 0.0

    # ------------------------------------------------------------------
    # Derivation (different from MITIMstate)
    # ------------------------------------------------------------------
   
    def derive_quantities(self, **kwargs):
 
        self._produce_shape_lists()

        super().derive_quantities(**kwargs)

    def _produce_shape_lists(self):
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
            None,  # s1 is arcsin(triangularity)
            None,  # s2 is minus squareness
            self.profiles["shape_sin3(-)"],
            self.profiles["shape_sin4(-)"],
            self.profiles["shape_sin5(-)"],
            self.profiles["shape_sin6(-)"],
        ]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
   
    def derive_geometry(self, n_theta_geo=1001):

        self._produce_shape_lists()

        (
            self.derived["volp_geo"],
            self.derived["surf_geo"],
            self.derived["gradr_geo"],
            self.derived["bp2_geo"],
            self.derived["bt2_geo"],
            self.derived["bt_geo"],
        ) = GEOMETRYtools.calculateGeometricFactors(self,n_theta=n_theta_geo)

        # Calculate flux surfaces
        cn = np.array(self.shape_cos).T
        sn = copy.deepcopy(self.shape_sin)
        sn[0] = self.profiles["rmaj(m)"]*0.0
        sn[1] = np.arcsin(self.profiles["delta(-)"])
        sn[2] = -self.profiles["zeta(-)"]
        sn = np.array(sn).T
        flux_surfaces = GEQtools.mitim_flux_surfaces()
        flux_surfaces.reconstruct_from_mxh_moments(
            self.profiles["rmaj(m)"],
            self.profiles["rmin(m)"],
            self.profiles["kappa(-)"],
            self.profiles["zmag(m)"],
            cn,
            sn)
        self.derived["R_surface"],self.derived["Z_surface"] = flux_surfaces.R, flux_surfaces.Z
        # -----------------------------------------------

        #cross-sectional area of each flux surface
        self.derived["surfXS"] = GEOMETRYtools.xsec_area_RZ(self.derived["R_surface"],self.derived["Z_surface"])

        self.derived["R_LF"] = self.derived["R_surface"].max(axis=1)  # self.profiles['rmaj(m)'][0]+self.profiles['rmin(m)']

        # For Synchrotron
        self.derived["B_ref"] = np.abs(self.derived["B_unit"] * self.derived["bt_geo"])
