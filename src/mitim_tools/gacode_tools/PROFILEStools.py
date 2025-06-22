import copy
import numpy as np
from collections import OrderedDict
from mitim_tools.plasmastate_tools import MITIMstate
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import MATHtools, IOtools, GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class gacode_state(MITIMstate.mitim_state):
    '''
    Class to read and manipulate GACODE profiles files (input.gacode).
    It inherits from the main MITIMstate class, which provides basic
    functionality for plasma state management.

    The class reads the GACODE profiles file, extracts relevant data,
    and writes them in the way that MITIMstate class expects.
    '''

    # ------------------------------------------------------------------
    # Reading and interpreting input.gacode files
    # ------------------------------------------------------------------

    def __init__(self, file, derive_quantities=True, mi_ref=None):

        # Initialize the base class and tell it the type of file
        super().__init__(type_file='input.gacode')

        # Read the input file and store the raw data
        self.file = file
        self._read_inputgacocde()

        # Derive quantities if requested
        if self.file is not None:
            # Derive (Depending on resolution, derived can be expensive, so I mmay not do it every time)
            self.derive_quantities(mi_ref=mi_ref, derive_quantities=derive_quantities)

    @IOtools.hook_method(after=MITIMstate.ensure_variables_existence)
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

            # Ensure correctness (wrong names in older input.gacode files)
            if "qmom(Nm)" in self.profiles:
                self.profiles["qmom(N/m^2)"] = self.profiles.pop("qmom(Nm)")
            if "qpar_beam(MW/m^3)" in self.profiles:
                self.profiles["qpar_beam(1/m^3/s)"] = self.profiles.pop("qpar_beam(MW/m^3)")
            if "qpar_wall(MW/m^3)" in self.profiles:
                self.profiles["qpar_wall(1/m^3/s)"] = self.profiles.pop("qpar_wall(MW/m^3)")
            
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

            # Ensure that we also have the shape coefficients
            num_moments = 7  # This is the max number of moments I'll be considering. If I don't have that many (usually there are 5 or 3), it'll be populated with zeros
            if "shape_cos0(-)" not in self.profiles:
                self.profiles["shape_cos0(-)"] = np.ones(self.profiles["rmaj(m)"].shape)
            for i in range(num_moments):
                if f"shape_cos{i + 1}(-)" not in self.profiles:
                    self.profiles[f"shape_cos{i + 1}(-)"] = np.zeros(self.profiles["rmaj(m)"].shape)
                if f"shape_sin{i + 1}(-)" not in self.profiles and i > 1:
                    self.profiles[f"shape_sin{i + 1}(-)"] = np.zeros(self.profiles["rmaj(m)"].shape)

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
        ) = calculateGeometricFactors(self,n_theta=n_theta_geo)

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
        self.derived["R_surface"],self.derived["Z_surface"] = np.array([flux_surfaces.R]), np.array([flux_surfaces.Z])
        
        # R and Z have [toroidal, radius, point], to allow for non-axisymmetric cases
        # -----------------------------------------------

        #cross-sectional area of each flux surface
        self.derived["surfXS"] = xsec_area_RZ(self.derived["R_surface"][0,...],self.derived["Z_surface"][0,...])

        self.derived["R_LF"] = self.derived["R_surface"][0,...].max(axis=-1)  # self.profiles['rmaj(m)'][0]+self.profiles['rmin(m)']

        # For Synchrotron
        self.derived["B_ref"] = np.abs(self.derived["B_unit"] * self.derived["bt_geo"])

        """
		surf_geo is truly surface area, but because of the GACODE definitions of flux, 
		Surf 		= V' <|grad r|>	 
		Surf_GACODE = V'
		"""
        self.derived["surfGACODE_geo"] = (self.derived["surf_geo"] / self.derived["gradr_geo"])
        self.derived["surfGACODE_geo"][np.isnan(self.derived["surfGACODE_geo"])] = 0

    def plot_geometry(self, axs3, color="b", legYN=True, extralab="", lw=1, fs=6):

        [ax00c,ax10c,ax20c,ax01c,ax11c,ax21c,ax02c,ax12c,ax22c,ax13c] = axs3

        rho = self.profiles["rho(-)"]
        lines = GRAPHICStools.listLS()

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
        var = self.profiles["polflux(Wb/radian)"]
        ax.plot(rho, var, lw=lw, ls="-", c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("Poloidal $\\psi$ ($Wb/rad$)")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax = ax10c
        var = self.profiles["delta(-)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$\\delta$")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax, bottomy=0)


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

        ax = ax12c

        var = self.profiles["zeta(-)"]
        ax.plot(rho, var, "-", lw=lw, c=color)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("zeta")

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.autoscale_y(ax)

        ax = ax13c
        self.plot_state_flux_surfaces(ax=ax, color=color)

        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        GRAPHICStools.addDenseAxis(ax)




def calculateGeometricFactors(profiles, n_theta=1001):

    # ----------------------------------------
    # Raw parameters from the file
    # 	in expro_util.f90, it performs those divisions to pass to geo library
    # ----------------------------------------

    r = profiles.profiles["rmin(m)"] / profiles.profiles["rmin(m)"][-1]
    R = profiles.profiles["rmaj(m)"] / profiles.profiles["rmin(m)"][-1]
    kappa = profiles.profiles["kappa(-)"]
    delta = profiles.profiles["delta(-)"]
    zeta = profiles.profiles["zeta(-)"]
    zmag = profiles.profiles["zmag(m)"] / profiles.profiles["rmin(m)"][-1]
    q = profiles.profiles["q(-)"]

    shape_coeffs = profiles.shape_cos + profiles.shape_sin

    # ----------------------------------------
    # Derivatives as defined in expro_util.f90
    # ----------------------------------------

    s_delta = r * MATHtools.deriv(r, delta)
    s_kappa = r / kappa * MATHtools.deriv(r, kappa)
    s_zeta = r * MATHtools.deriv(r, zeta)
    dzmag = MATHtools.deriv(r, zmag)
    dRmag = MATHtools.deriv(r, R)

    s_shape_coeffs = []
    for i in range(len(shape_coeffs)):
        if shape_coeffs[i] is not None:
            s_shape_coeffs.append(r * MATHtools.deriv(r, shape_coeffs[i]))
        else:
            s_shape_coeffs.append(None)

    # ----------------------------------------
    # Calculate the differencial volume at each radii
    # 	from f2py/geo/geo.f90 in gacode source we have geo_volume_prime.
    # ----------------------------------------

    # Prepare cos_sins
    cos_sin = []
    cos_sin_s = []
    for j in range(len(R)):
        cos_sin0 = []
        cos_sin_s0 = []
        for k in range(len(shape_coeffs)):
            if shape_coeffs[k] is not None:
                cos_sin0.append(shape_coeffs[k][j])
                cos_sin_s0.append(s_shape_coeffs[k][j])
            else:
                cos_sin0.append(None)
                cos_sin_s0.append(None)
        cos_sin.append(cos_sin0)
        cos_sin_s.append(cos_sin_s0)

    (
        geo_volume_prime,
        geo_surf,
        geo_fluxsurfave_grad_r,
        geo_fluxsurfave_bp2,
        geo_fluxsurfave_bt2,
        bt_geo0,
    ) = volp_surf_geo_vectorized(
        R,
        r,
        delta,
        kappa,
        cos_sin,
        cos_sin_s,
        zeta,
        zmag,
        s_delta,
        s_kappa,
        s_zeta,
        dzmag,
        dRmag,
        q,
        n_theta=n_theta,
    )

    """
	from expro_util.f90 we have:
		expro_volp(i) = geo_volume_prime*r_min**2, where r_min = expro_rmin(expro_n_exp)
		expro_surf(i) = geo_surf*r_min**2
	"""

    volp = geo_volume_prime * profiles.profiles["rmin(m)"][-1] ** 2
    surf = geo_surf * profiles.profiles["rmin(m)"][-1] ** 2

    return volp, surf, geo_fluxsurfave_grad_r, geo_fluxsurfave_bp2, geo_fluxsurfave_bt2, bt_geo0

def volp_surf_geo_vectorized(
    geo_rmaj_in,
    geo_rmin_in,
    geo_delta_in,
    geo_kappa_in,
    cos_sin,
    cos_sin_s,
    geo_zeta_in,
    geo_zmag_in,
    geo_s_delta_in,
    geo_s_kappa_in,
    geo_s_zeta_in,
    geo_dzmag_in,
    geo_drmaj_in,
    geo_q_in,
    n_theta=1001):
    """
    Completety from f2py/geo/geo.f90
    """

    geo_rmin_in = geo_rmin_in.clip(
        1e-10
    )  # To avoid problems at 0 (Implemented by PRF, not sure how TGYRO deals with this)

    geo_q_in = geo_q_in.clip(1e-2) # To avoid problems at 0 with some geqdsk files that are corrupted...


    [
        geo_shape_cos0_in,
        geo_shape_cos1_in,
        geo_shape_cos2_in,
        geo_shape_cos3_in,
        geo_shape_cos4_in,
        geo_shape_cos5_in,
        geo_shape_cos6_in,
        _,
        _,
        _,
        geo_shape_sin3_in,
        geo_shape_sin4_in,
        geo_shape_sin5_in,
        geo_shape_sin6_in,
    ] = np.array(cos_sin).astype(float).T

    [
        geo_shape_s_cos0_in,
        geo_shape_s_cos1_in,
        geo_shape_s_cos2_in,
        geo_shape_s_cos3_in,
        geo_shape_s_cos4_in,
        geo_shape_s_cos5_in,
        geo_shape_s_cos6_in,
        _,
        _,
        _,
        geo_shape_s_sin3_in,
        geo_shape_s_sin4_in,
        geo_shape_s_sin5_in,
        geo_shape_s_sin6_in,
    ] = np.array(cos_sin_s).astype(float).T

    geo_signb_in = 1.0

    geov_theta = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_bigr = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_bigr_r = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_bigr_t = np.zeros((n_theta,geo_rmin_in.shape[0]))
    bigz = np.zeros((n_theta,geo_rmin_in.shape[0]))
    bigz_r = np.zeros((n_theta,geo_rmin_in.shape[0]))
    bigz_t = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_jac_r = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_grad_r = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_l_t = np.zeros((n_theta,geo_rmin_in.shape[0]))
    r_c = np.zeros((n_theta,geo_rmin_in.shape[0]))
    bigz_l = np.zeros((n_theta,geo_rmin_in.shape[0]))
    bigr_l = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_l_r = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_nsin = np.zeros((n_theta,geo_rmin_in.shape[0]))

    pi_2 = 8.0 * np.arctan(1.0)
    d_theta = pi_2 / (n_theta - 1)

    for i in range(n_theta):
        #!-----------------------------------------
        #! Generalized Miller-type parameterization
        #!-----------------------------------------

        theta = -0.5 * pi_2 + (i - 1) * d_theta

        geov_theta[i] = theta

        x = np.arcsin(geo_delta_in)

        #! A
        #! dA/dtheta
        #! d^2A/dtheta^2
        a = (
            theta
            + geo_shape_cos0_in
            + geo_shape_cos1_in * np.cos(theta)
            + geo_shape_cos2_in * np.cos(2 * theta)
            + geo_shape_cos3_in * np.cos(3 * theta)
            + geo_shape_cos4_in * np.cos(4 * theta)
            + geo_shape_cos5_in * np.cos(5 * theta)
            + geo_shape_cos6_in * np.cos(6 * theta)
            + geo_shape_sin3_in * np.sin(3 * theta)
            + x * np.sin(theta)
            - geo_zeta_in * np.sin(2 * theta)
            + geo_shape_sin3_in * np.sin(3 * theta)
            + geo_shape_sin4_in * np.sin(4 * theta)
            + geo_shape_sin5_in * np.sin(5 * theta)
            + geo_shape_sin6_in * np.sin(6 * theta)
        )
        a_t = (
            1.0
            - geo_shape_cos1_in * np.sin(theta)
            - 2 * geo_shape_cos2_in * np.sin(2 * theta)
            - 3 * geo_shape_cos3_in * np.sin(3 * theta)
            - 4 * geo_shape_cos4_in * np.sin(4 * theta)
            - 5 * geo_shape_cos5_in * np.sin(5 * theta)
            - 6 * geo_shape_cos6_in * np.sin(6 * theta)
            + x * np.cos(theta)
            - 2 * geo_zeta_in * np.cos(2 * theta)
            + 3 * geo_shape_sin3_in * np.cos(3 * theta)
            + 4 * geo_shape_sin4_in * np.cos(4 * theta)
            + 5 * geo_shape_sin5_in * np.cos(5 * theta)
            + 6 * geo_shape_sin6_in * np.cos(6 * theta)
        )
        a_tt = (
            -geo_shape_cos1_in * np.cos(theta)
            - 4 * geo_shape_cos2_in * np.cos(2 * theta)
            - 9 * geo_shape_cos3_in * np.cos(3 * theta)
            - 16 * geo_shape_cos4_in * np.cos(4 * theta)
            - 25 * geo_shape_cos5_in * np.cos(5 * theta)
            - 36 * geo_shape_cos6_in * np.cos(6 * theta)
            - x * np.sin(theta)
            + 4 * geo_zeta_in * np.sin(2 * theta)
            - 9 * geo_shape_sin3_in * np.sin(3 * theta)
            - 16 * geo_shape_sin4_in * np.sin(4 * theta)
            - 25 * geo_shape_sin5_in * np.sin(5 * theta)
            - 36 * geo_shape_sin6_in * np.sin(6 * theta)
        )

        #! R(theta)
        #! dR/dr
        #! dR/dtheta
        #! d^2R/dtheta^2
        geov_bigr[i] = geo_rmaj_in + geo_rmin_in * np.cos(a)
        geov_bigr_r[i] = (
            geo_drmaj_in
            + np.cos(a)
            - np.sin(a)
            * (
                geo_shape_s_cos0_in
                + geo_shape_s_cos1_in * np.cos(theta)
                + geo_shape_s_cos2_in * np.cos(2 * theta)
                + geo_shape_s_cos3_in * np.cos(3 * theta)
                + geo_shape_s_cos4_in * np.cos(4 * theta)
                + geo_shape_s_cos5_in * np.cos(5 * theta)
                + geo_shape_s_cos6_in * np.cos(6 * theta)
                + geo_s_delta_in / np.cos(x) * np.sin(theta)
                - geo_s_zeta_in * np.sin(2 * theta)
                + geo_shape_s_sin3_in * np.sin(3 * theta)
                + geo_shape_s_sin4_in * np.sin(4 * theta)
                + geo_shape_s_sin5_in * np.sin(5 * theta)
                + geo_shape_s_sin6_in * np.sin(6 * theta)
            )
        )
        geov_bigr_t[i] = -geo_rmin_in * a_t * np.sin(a)
        bigr_tt = -geo_rmin_in * a_t**2 * np.cos(a) - geo_rmin_in * a_tt * np.sin(a)

        #!-----------------------------------------------------------

        #! A
        #! dA/dtheta
        #! d^2A/dtheta^2
        a = theta
        a_t = 1.0
        a_tt = 0.0

        #! Z(theta)
        #! dZ/dr
        #! dZ/dtheta
        #! d^2Z/dtheta^2
        bigz[i] = geo_zmag_in + geo_kappa_in * geo_rmin_in * np.sin(a)
        bigz_r[i] = geo_dzmag_in + geo_kappa_in * (1.0 + geo_s_kappa_in) * np.sin(a)
        bigz_t[i] = geo_kappa_in * geo_rmin_in * np.cos(a) * a_t
        bigz_tt = (
            -geo_kappa_in * geo_rmin_in * np.sin(a) * a_t**2
            + geo_kappa_in * geo_rmin_in * np.cos(a) * a_tt
        )

        g_tt = geov_bigr_t[i] ** 2 + bigz_t[i] ** 2

        geov_jac_r[i] = geov_bigr[i] * (
            geov_bigr_r[i] * bigz_t[i] - geov_bigr_t[i] * bigz_r[i]
        )

        geov_grad_r[i] = geov_bigr[i] * np.sqrt(g_tt) / geov_jac_r[i]

        geov_l_t[i] = np.sqrt(g_tt)

        r_c[i] = geov_l_t[i] ** 3 / (geov_bigr_t[i] * bigz_tt - bigz_t[i] * bigr_tt)

        bigz_l[i] = bigz_t[i] / geov_l_t[i]

        bigr_l[i] = geov_bigr_t[i] / geov_l_t[i]

        geov_l_r[i] = bigz_l[i] * bigz_r[i] + bigr_l[i] * geov_bigr_r[i]

        geov_nsin[i] = (
            geov_bigr_r[i] * geov_bigr_t[i] + bigz_r[i] * bigz_t[i]
        ) / geov_l_t[i]

    c = 0.0
    for i in range(n_theta):
        c = c + geov_l_t[i] / (geov_bigr[i] * geov_grad_r[i])

    f = geo_rmin_in / (c * d_theta / pi_2)

    c = 0.0
    for i in range(n_theta - 1):
        c = c + geov_l_t[i] * geov_bigr[i] / geov_grad_r[i]

    geo_volume_prime = pi_2 * c * d_theta

    # Line 716 in geo.f90
    geo_surf = 0.0
    for i in range(n_theta - 1):
        geo_surf = geo_surf + geov_l_t[i] * geov_bigr[i]
    geo_surf = pi_2 * geo_surf * d_theta

    # -----
    c = 0.0
    for i in range(n_theta - 1):
        c = c + geov_l_t[i] / (geov_bigr[i] * geov_grad_r[i])
    f = geo_rmin_in / (c * d_theta / pi_2)

    geov_b = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_g_theta = np.zeros((n_theta,geo_rmin_in.shape[0]))
    geov_bt = np.zeros((n_theta,geo_rmin_in.shape[0]))
    for i in range(n_theta):
        geov_bt[i] = f / geov_bigr[i]
        geov_bp = (geo_rmin_in / geo_q_in) * geov_grad_r[i] / geov_bigr[i]

        geov_b[i] = geo_signb_in * (geov_bt[i] ** 2 + geov_bp**2) ** 0.5
        geov_g_theta[i] = (
            geov_bigr[i]
            * geov_b[i]
            * geov_l_t[i]
            / (geo_rmin_in * geo_rmaj_in * geov_grad_r[i])
        )

    theta_0 = 0
    dx = geov_theta[1,0] - geov_theta[0,0]
    x0 = theta_0 - geov_theta[0,0]
    i1 = int(x0 / dx) + 1
    i2 = i1 + 1
    x1 = (i1 - 1) * dx
    z = (x0 - x1) / dx
    if i2 == n_theta:
        i2 -= 1
    bt_geo0 = geov_bt[i1] + (geov_bt[i2] - geov_bt[i1]) * z

    denom = 0
    for i in range(n_theta - 1):
        denom = denom + geov_g_theta[i] / geov_b[i]

    geo_fluxsurfave_grad_r = 0
    for i in range(n_theta - 1):
        geo_fluxsurfave_grad_r = (
            geo_fluxsurfave_grad_r
            + geov_grad_r[i] * geov_g_theta[i] / geov_b[i] / denom
        )

    geo_fluxsurfave__bp2 = 0
    for i in range(n_theta - 1):
        geo_fluxsurfave__bp2 = (
            geo_fluxsurfave__bp2
            + geov_bt[i] ** 2 * geov_g_theta[i] / geov_b[i] / denom
        )

    geo_fluxsurfave_bt2 = 0
    for i in range(n_theta - 1):
        geo_fluxsurfave_bt2 = (
            geo_fluxsurfave_bt2
            + geov_bp ** 2 * geov_g_theta[i] / geov_b[i] / denom
        )

    return geo_volume_prime, geo_surf, geo_fluxsurfave_grad_r, geo_fluxsurfave__bp2, geo_fluxsurfave_bt2, bt_geo0

def xsec_area_RZ(R,Z):
    # calculates the cross-sectional area of the plasma for each flux surface
    xsec_area = []
    for i in range(R.shape[0]):
        R0 = np.max(R[i,:]) - np.min(R[i,:])
        Z0 = np.max(Z[i,:]) - np.min(Z[i,:])
        xsec_area.append(np.trapz(R[i], Z[i]))

    xsec_area = np.array(xsec_area)

    return xsec_area

