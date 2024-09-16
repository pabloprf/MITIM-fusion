import os
import io
import tempfile
import copy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools.utils import GEQplotting
from shapely.geometry import LineString
from scipy.integrate import quad
import freegs
from freegs import geqdsk
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
Note that this module relies on OMFIT classes (https://omfit.io/classes.html) procedures to intrepret the content of g-eqdsk files.
Modifications are made in MITINM for visualizations and a few extra derivations.
"""

def fix_file(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    # -----------------------------------------------------------------------
    # Remove coils (chatGPT 4o as of 08/24/24)
    # -----------------------------------------------------------------------
    # Use StringIO to simulate the file writing
    noCoils_file = io.StringIO()
    for cont, line in enumerate(lines):
        if cont > 0 and line[:2] == "  ":
            break
        noCoils_file.write(line)

    # Reset cursor to the start of StringIO
    noCoils_file.seek(0)

    # Write the StringIO content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(noCoils_file.getvalue().encode('utf-8'))
        noCoils_file = tmp_file.name
    # -----------------------------------------------------------------------

    with open(filename, 'r') as file1, open(noCoils_file, 'r') as file2:
        file1_content = file1.read()
        file2_content = file2.read()
        
    if file1_content != file2_content:
        print(f"\t- geqdsk file {IOtools.clipstr(filename)} had coils, I have removed them")

    filename = noCoils_file

    return filename
class MITIMgeqdsk:
    def __init__(self, filename):

        # Fix file by removing coils if it has them
        filename = fix_file(filename)

        # Read GEQDSK file using OMFIT
        import omfit_classes.omfit_eqdsk
        self.g = omfit_classes.omfit_eqdsk.OMFITgeqdsk(filename, forceFindSeparatrix=True)

        # Extra derivations in MITIM
        self.derive()

        # Remove temporary file
        os.remove(filename)

    @classmethod
    def timeslices(cls, filename, **kwargs):
        print("\n...Opening GEQ file with several time slices")

        with open(filename, "rb") as f:
            lines_full = f.readlines()

        resolutions = [int(a) for a in lines_full[0].split()[-3:]]

        lines_files = []
        lines_files0 = []
        for i in range(len(lines_full)):
            line = lines_full[i]
            resols = []
            try:
                resols = [int(a) for a in line.split()[-3:]]
            except:
                pass
            if (resols == resolutions) and (i != 0):
                lines_files.append(lines_files0)
                lines_files0 = []
            lines_files0.append(line)
        lines_files.append(lines_files0)

        # Write files
        gs = []
        for i in range(len(lines_files)):
            with open(f"{filename}_time{i}.geqdsk", "wb") as f:
                f.writelines(lines_files[i])

            gs.append(
                cls(
                    f"{filename}_time{i}.geqdsk",**kwargs,
                )
            )
            os.system(f"rm {filename}_time{i}.geqdsk")

        return gs

    def derive(self, debug=False):

        self.Jt = self.g.surfAvg("Jt") * 1e-6
        self.Jt_fb = self.g.surfAvg("Jt_fb") * 1e-6

        self.Jerror = np.abs(self.Jt - self.Jt_fb)

        self.Ip = self.g["CURRENT"]

        # Parameterizations of LCFS
        self.kappa = self.g["fluxSurfaces"]["geo"]["kap"][-1]
        self.kappaU = self.g["fluxSurfaces"]["geo"]["kapu"][-1]
        self.kappaL = self.g["fluxSurfaces"]["geo"]["kapl"][-1]

        self.delta = self.g["fluxSurfaces"]["geo"]["delta"][-1]
        self.deltaU = self.g["fluxSurfaces"]["geo"]["dell"][-1]
        self.deltaL = self.g["fluxSurfaces"]["geo"]["dell"][-1]

        self.zeta = self.g["fluxSurfaces"]["geo"]["zeta"][-1]

        self.a = self.g["fluxSurfaces"]["geo"]["a"][-1]
        self.Rmag = self.g["fluxSurfaces"]["geo"]["R"][0]
        self.Zmag = self.g["fluxSurfaces"]["geo"]["Z"][0]
        self.Rmajor = np.mean(
            [
                self.g["fluxSurfaces"]["geo"]["Rmin_centroid"][-1],
                self.g["fluxSurfaces"]["geo"]["Rmax_centroid"][-1],
            ]
        )

        self.Zmajor = self.Zmag

        self.eps = self.a / self.Rmajor

        # Core values

        self.kappa_a = self.g["fluxSurfaces"]["geo"]["cxArea"][-1] / (np.pi * self.a**2)

        self.kappa995 = np.interp(
            0.995,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["kap"],
        )
        self.kappa95 = np.interp(
            0.95,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["kap"],
        )
        self.delta995 = np.interp(
            0.995,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["delta"],
        )
        self.delta95 = np.interp(
            0.95,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["delta"],
        )

        """
        --------------------------------------------------------------------------------------------------------------------------------------
        Boundary
        --------------------------------------------------------------------------------------------------------------------------------------
            Note that the RBBS and ZBBS values in the gfile are often too scattered and do not reproduce the boundary near x-points.
            The shaping parameters calculated using fluxSurfaces are correct though.
        """

        self.Rb_gfile, self.Yb_gfile = self.g["RBBBS"], self.g["ZBBBS"]
        self.Rb, self.Yb = self.g["fluxSurfaces"].sep.transpose()
        
        if len(self.Rb) == 0:
            print("\t- MITIM > No separatrix found in the OMFIT fluxSurfaces, increasing resolution and going all in!",typeMsg='i')

            flx = copy.deepcopy(self.g['fluxSurfaces'])
            flx._changeResolution(6)
            flx.findSurfaces([0.0,0.5,1.0])
            fs = flx['flux'][list(flx['flux'].keys())[-1]]
            self.Rb, self.Yb = fs['R'], fs['Z']

        if debug:
            fig, ax = plt.subplots()

            # OMFIT
            ax.plot(self.Rb, self.Yb, "-s", c="r", label="OMFIT")

            # GFILE
            ax.plot(self.Rb_gfile, self.Yb_gfile, "-s", c="y", label="GFILE")

            # Extras
            self.plotFluxSurfaces(
                ax=ax, fluxes=[0.99999, 1.0], rhoPol=True, sqrt=False, color="m"
            )
            self.plotXpointEnvelope(
                ax=ax, color="c", alpha=1.0, rhoPol=True, sqrt=False
            )
            self.plotEnclosingBox(ax=ax)

            ax.legend()
            ax.set_aspect("equal")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")

            plt.show()

    def plotEnclosingBox(self, ax=None, c= "k"):
        if ax is None:
            fig, ax = plt.subplots()

        Rmajor, a, Zmajor, kappaU, kappaL, deltaU, deltaL = (
            self.Rmajor,
            self.a,
            self.Zmajor,
            self.kappaU,
            self.kappaL,
            self.deltaU,
            self.deltaL,
        )

        ax.axhline(y=Zmajor, ls="--", c=c, lw=0.5)
        ax.axvline(x=Rmajor, ls="--", c=c, lw=0.5)
        ax.axvline(x=Rmajor - a, ls="--", c=c, lw=0.5)
        ax.axvline(x=Rmajor + a, ls="--", c=c, lw=0.5)
        Rtop = Zmajor + a * kappaU
        ax.axhline(y=Rtop, ls="--", c=c, lw=0.5)
        Rbot = Zmajor - a * kappaL
        ax.axhline(y=Rbot, ls="--", c=c, lw=0.5)
        ax.axvline(x=Rmajor - a * deltaU, ls="--", c=c, lw=0.5)
        ax.axvline(x=Rmajor - a * deltaL, ls="--", c=c, lw=0.5)

        ax.plot([self.Rmajor, self.Rmag], [self.Zmajor, self.Zmag], "o", markersize=5)

    def write(self, filename=None):
        """
        If filename is None, use the original one
        """

        if filename is not None:
            self.g.filename = filename

        self.g.save()

    # -----------------------------------------------------------------------------
    # Parameterizations
    # -----------------------------------------------------------------------------

    def get_MXH_coeff_new(self, n_coeff=7, plotYN=False): 

        psis = self.g["AuxQuantities"]["PSI_NORM"]
        flux_surfaces = self.g['fluxSurfaces']['flux']
        
        # Cannot parallelize because different number of points?
        kappa, rmin, rmaj, zmag, sn, cn = [],[],[],[],[],[]
        
        for flux in range(len(flux_surfaces)):
            if flux == len(flux_surfaces)-1:
                Rf, Zf = self.Rb, self.Yb
            else:
                Rf, Zf = flux_surfaces[flux]['R'],flux_surfaces[flux]['Z']

            # Perform the MXH decompositionusing the MITIM surface class
            surfaces = mitim_flux_surfaces()
            surfaces.reconstruct_from_RZ(Rf,Zf)
            surfaces._to_mxh(n_coeff=n_coeff)

            kappa.append(surfaces.kappa[0])
            rmin.append(surfaces.a[0])
            rmaj.append(surfaces.R0[0])
            zmag.append(surfaces.Z0[0])

            sn.append(surfaces.sn[0,:])
            cn.append(surfaces.cn[0,:])

        kappa = np.array(kappa)
        rmin = np.array(rmin)
        rmaj = np.array(rmaj)
        zmag = np.array(zmag)
        sn = np.array(sn)
        cn = np.array(cn)

        if plotYN:
            fig, ax = plt.subplots()
            ax.plot(self.Rb, self.Yb, 'o-', c = 'b')

            surfaces = mitim_flux_surfaces()
            surfaces.reconstruct_from_RZ(self.Rb, self.Yb)
            surfaces._to_mxh(n_coeff=n_coeff)
            surfaces._from_mxh()

            ax.plot(surfaces.R[0], surfaces.Z[0], 'o-', c = 'r')

            plt.show()

        '''
        Reminder that 
            sn = [0.0, np.arcsin(delta), -zeta, ...] 
        '''

        return psis, rmaj, rmin, zmag, kappa, cn, sn
        
    # -----------------------------------------------------------------------------
    # For MAESTRO and TRANSP converstions
    # -----------------------------------------------------------------------------
    def to_profiles(self, ne0_20 = 1.0, Zeff = 1.5, PichT = 1.0,  Z = 9, coeffs_MXH = 7, plotYN = False):

        # -------------------------------------------------------------------------------------------------------
        # Quantities from the equilibrium
        # -------------------------------------------------------------------------------------------------------

        rhotor = self.g['RHOVN']
        psi = self.g['AuxQuantities']['PSI']                           # Wb/rad
        torfluxa =  self.g['AuxQuantities']['PHI'][-1] / (2*np.pi)     # Wb/rad
        q = self.g['QPSI']
        pressure = self.g['PRES']       # Pa
        Ip = self.g['CURRENT']*1E-6     # MA

        RZ = np.array([self.Rb,self.Yb]).T
        R0 = (RZ.max(axis=0)[0] + RZ.min(axis=0)[0])/2
        
        B0 = self.g['RCENTR']*self.g['BCENTR'] / R0

        _, rmaj, rmin, zmag, kappa, cn, sn = self.get_MXH_coeff_new(n_coeff=coeffs_MXH)

        delta = np.sin(sn[:,1])
        zeta = -sn[:,2]

        # -------------------------------------------------------------------------------------------------------
        # Pass to profiles
        # -------------------------------------------------------------------------------------------------------

        profiles = {}

        profiles['nexp'] = np.array([f'{rhotor.shape[0]}'])
        profiles['nion'] = np.array(['2'])
        profiles['shot'] = np.array(['12345'])

        # Just one specie
        profiles['name'] = np.array(['D','F'])
        profiles['type'] = np.array(['therm','therm'])
        profiles['masse'] = np.array([5.4488748e-04])
        profiles['mass'] = np.array([2.0, Z*2])
        profiles['ze'] = np.array([-1.0])
        profiles['z'] = np.array([1.0, Z])


        profiles['torfluxa(Wb/radian)'] = np.array([torfluxa])
        profiles['rcentr(m)'] = np.array([R0])
        profiles['bcentr(T)'] = np.array([B0])
        profiles['current(MA)'] = np.array([Ip])

        profiles['rho(-)'] = rhotor
        profiles['polflux(Wb/radian)'] = psi
        profiles['q(-)'] = q

        # -------------------------------------------------------------------------------------------------------
        # Flux surfaces
        # -------------------------------------------------------------------------------------------------------

        profiles['kappa(-)'] = kappa
        profiles['delta(-)'] = delta
        profiles['zeta(-)'] = zeta
        profiles['rmin(m)'] = rmin
        profiles['rmaj(m)'] = rmaj
        profiles['zmag(m)'] = zmag

        sn, cn = np.array(sn), np.array(cn)
        for i in range(coeffs_MXH):
            profiles[f'shape_cos{i}(-)'] = cn[:,i]
        for i in range(coeffs_MXH-3):
            profiles[f'shape_sin{i+3}(-)'] = sn[:,i+3]

        '''
        -------------------------------------------------------------------------------------------------------
        Kinetic profiles
        -------------------------------------------------------------------------------------------------------
        Pressure division into temperature and density
            p_Pa = p_e + p_i = Te_eV * e_J * ne_20 * 1e20  + Ti_eV * e_J * ni_20 * 1e20
            if T=Te=Ti and ne=ni
            p_Pa = 2 * T_eV * e_J * ne_20 * 1e20
            T_eV = p_Pa / (2 * e_J * ne_20 * 1e20)
        '''

        C = 1 / (2 * 1.60217662e-19 * 1e20)
        _, ne_20 = PLASMAtools.parabolicProfile(Tbar=ne0_20/1.25,nu=1.25,rho=rhotor,Tedge=ne0_20/5)
        T_keV = C * (pressure / ne_20) * 1E-3

        fZ = (Zeff-1) / (Z**2-Z)  # One-impurity model to give desired Zeff

        profiles['te(keV)'] = T_keV
        profiles['ti(keV)'] = np.array([T_keV]*2).T
        profiles['ne(10^19/m^3)'] = ne_20*10.0
        profiles['ni(10^19/m^3)'] = np.array([profiles['ne(10^19/m^3)']*(1-Z*fZ),profiles['ne(10^19/m^3)']*fZ]).T

        # -------------------------------------------------------------------------------------------------------
        # Power: insert parabolic and use PROFILES volume integration to find desired power
        # -------------------------------------------------------------------------------------------------------

        _, profiles["qrfe(MW/m^3)"] = PLASMAtools.parabolicProfile(Tbar=1.0,nu=5.0,rho=rhotor,Tedge=0.0)

        p = PROFILEStools.PROFILES_GACODE.scratch(profiles)

        p.profiles["qrfe(MW/m^3)"] = p.profiles["qrfe(MW/m^3)"] *  PichT/p.derived['qRF_MWmiller'][-1] /2
        p.profiles["qrfi(MW/m^3)"] = p.profiles["qrfe(MW/m^3)"]

        # -------------------------------------------------------------------------------------------------------
        # Ready to go
        # -------------------------------------------------------------------------------------------------------

        p.deriveQuantities()

        # -------------------------------------------------------------------------------------------------------
        # Plotting
        # -------------------------------------------------------------------------------------------------------

        if plotYN:

            fig, ax = plt.subplots()
            ff = np.linspace(0, 1, 11)
            self.plotFluxSurfaces(ax=ax, fluxes=ff, rhoPol=False, sqrt=True, color="r", plot1=False)
            p.plotGeometry(ax=ax, surfaces_rho=ff, color="b")
            plt.show()

        return p

    def to_transp(self, folder = '~/scratch/', shot = '12345', runid = 'P01', ne0_20 = 1E19, Vsurf = 0.0, Zeff = 1.5, PichT_MW = 11.0, times = [0.0,1.0]):

        print("\t- Converting to TRANSP")
        if not os.path.exists(folder):
            os.makedirs(folder)

        p = self.to_profiles(ne0_20 = ne0_20, Zeff = Zeff, PichT = PichT_MW)
        p.writeCurrentStatus(f'{folder}/input.gacode')

        transp = p.to_transp(folder = folder, shot = shot, runid = runid, times = times, Vsurf = Vsurf)

        return transp

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------------------------------------------------------------------

    def plot(self, fn=None, extraLabel=""):
        GEQplotting.plot(self, fn=fn, extraLabel=extraLabel)

    def plotFS(self, axs=None, color="b", label=""):
        GEQplotting.plotFS(self, axs=axs, color=color, label=label)

    def plotPlasma(self, axs=None, legendYN=False, color="r", label=""):
        GEQplotting.plotPlasma(self, axs=axs, legendYN=legendYN, color=color, label=label)

    def plotGeometry(self, axs=None, color="r"):
        GEQplotting.plotGeometry(self, axs=axs, color=color)

    def plotFluxSurfaces(self, ax=None, fluxes=[1.0], color="b", alpha=1.0, rhoPol=True, sqrt=False, lw=1, lwB=2, plot1=True, label = ''):
        return GEQplotting.plotFluxSurfaces(self, ax=ax, fluxes=fluxes, color=color, alpha=alpha, rhoPol=rhoPol, sqrt=sqrt, lw=lw, lwB=lwB, plot1=plot1, label = label)

    def plotXpointEnvelope(self, ax=None, color="b", alpha=1.0, rhoPol=True, sqrt=False):
        GEQplotting.plotXpointEnvelope(self, ax=ax, color=color, alpha=alpha, rhoPol=rhoPol, sqrt=sqrt)

# -----------------------------------------------------------------------------
# Tools to handle flux surface definitions
# -----------------------------------------------------------------------------
class mitim_flux_surfaces:

    def reconstruct_from_mxh_moments(self,R0, a, kappa, Z0, cn, sn, thetas = None):
        '''
        sn = [0.0, np.arcsin(delta), -zeta, ...]
        cn = [...]

        You can provide a multi-dim array of (radii, )
        '''

        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.Z0 = Z0
        self.cn = cn
        self.sn = sn

        self.delta = np.sin(self.sn[...,1])
        self.zeta = -self.sn[...,2]

        self._from_mxh(thetas = thetas)
        
    def reconstruct_from_miller(self,R0, a, kappa, Z0, delta, zeta, thetas = None):
        '''
        sn = [0.0, np.arcsin(delta), -zeta, ...]
        cn = [...]

        You can provide a multi-dim array of (radii, ) or not
        '''

        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.Z0 = Z0
        self.delta = delta
        self.zeta = zeta

        self.cn = np.array([0.0,0.0,0.0])
        self.sn = np.array([0.0,np.arcsin(self.delta),-self.zeta])

        self._from_mxh(thetas = thetas)

    def _from_mxh(self, thetas = None):

        self.R, self.Z = from_mxh_to_RZ(self.R0, self.a, self.kappa, self.Z0, self.cn, self.sn, thetas = thetas)

    def reconstruct_from_RZ(self, R, Z):

        self.R = R
        self.Z = Z

        if len(self.R.shape) == 1:
            self.R = self.R[np.newaxis,:]
            self.Z = self.Z[np.newaxis,:]

        self._to_miller()

    def _to_mxh(self, n_coeff=6):

        self.cn = np.zeros((self.R.shape[0],n_coeff))
        self.sn = np.zeros((self.R.shape[0],n_coeff))
        self.gn = np.zeros((self.R.shape[0],4))
        for i in range(self.R.shape[0]):
            self.cn[i,:], self.sn[i,:], self.gn[i,:] = from_RZ_to_mxh(self.R[i,:], self.Z[i,:], n_coeff=n_coeff)

        [self.R0, self.a, self.Z0, self.kappa] = self.gn.T

    def _to_miller(self):

        Rmin = np.min(self.R, axis=-1)
        Rmax = np.max(self.R, axis=-1)
        Zmax = np.max(self.Z, axis=-1)
        Zmin = np.min(self.Z, axis=-1)

        self.R0 = 0.5* (Rmax + Rmin)
        self.Z0 = 0.5* (Zmax + Zmin)

        self.a = (Rmax - Rmin) / 2

        # Elongations

        self.kappa_u = (Zmax - self.Z0) / self.a
        self.kappa_l = (self.Z0 - Zmin) / self.a
        self.kappa = (self.kappa_u + self.kappa_l) / 2

        # Triangularities

        RatZmax = self.R[np.arange(self.R.shape[0]),np.argmax(self.Z,axis=-1)]
        self.delta_u = (self.R0-RatZmax) / self.a

        RatZmin = self.R[np.arange(self.R.shape[0]),np.argmin(self.Z,axis=-1)]
        self.delta_l = (self.R0-RatZmin) / self.a

        self.delta = (self.delta_u + self.delta_l) / 2

        # Squareness (not parallel for the time being)
        self.zeta = np.zeros(self.R0.shape)
        for i in range(self.R0.shape[0]):
            Ri, Zi, zeta_uo = find_squareness_points(self.R[i,:], self.Z[i,:])
            self.zeta[i] = zeta_uo

    def plot(self, ax = None, color = 'r', label = None, plot_extremes=False):

        if ax is None:
            fig, ax = plt.subplots()

        for i in range(self.R.shape[0]):
            ax.plot(self.R[i,:], self.Z[i,:], color = color, label = label)

            if plot_extremes:
                ax.plot([self.R[i,self.Z[i,:].argmax()]], [self.Z[i,:].max()], 'o', color=color, markersize=5)
                ax.plot([self.R[i,self.Z[i,:].argmin()]], [self.Z[i,:].min()], 'o', color=color, markersize=5)

        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        if label is not None: 
            ax.legend()
        GRAPHICStools.addDenseAxis(ax)

def find_squareness_points(R, Z, debug = False):

    # Reference point (A)
    A_r = R_of_maxZ = R[Z.argmax()] 
    A_z = Z_of_maxR = Z[R.argmax()]

    # Upper Outer Squareness point (D)
    C_r = R_of_maxR = R.max()
    C_z = Z_of_maxZ = Z.max()
    
    # Find intersection with separatrix (C)
    Ri_uo, Zi_uo = find_intersection_squareness(R, Z, R_of_maxZ, Z_of_maxR, R_of_maxR, Z_of_maxZ)
    C_r, C_z = Ri_uo, Zi_uo

    # Find point B
    B_r = Rs_uo = R_of_maxZ + (R_of_maxR-R_of_maxZ)/2
    B_z = Zs_uo = Z_of_maxR + (Z_of_maxZ-Z_of_maxR)/2

    # Squareness is defined as the distance BC divided by the distance AB
    zeta_uo = np.sqrt((C_r-B_r)**2 + (C_z-B_z)**2) / np.sqrt((A_r-B_r)**2 + (A_z-B_z)**2)
    #zeta_uo = np.sqrt((Ri_uo-Rs_uo)**2 + (Zi_uo-Zs_uo)**2) / np.sqrt((R_of_maxZ-R_of_maxR)**2 + (Z_of_maxZ-Z_of_maxR)**2)

    if debug:
        plt.ion()
        fig, ax = plt.subplots()
        ax.plot(R, Z, 'o-', markersize=3, color='k')
        ax.plot([Ri_uo], [Zi_uo], 'o', color='k', label = 'C', markersize=6)
        
        ax.plot([R_of_maxZ], [Z_of_maxR], 'o', color='b', label = 'A')
        ax.axvline(x=R_of_maxZ, ls='--', color='b')
        ax.axhline(y=Z_of_maxR, ls='--', color='b')

        ax.plot([R_of_maxR], [Z_of_maxZ], 'o', color='r', label = 'D')
        ax.axhline(y=Z_of_maxZ, ls='--', color='r')
        ax.axvline(x=R_of_maxR, ls='--', color='r')

        ax.plot([Rs_uo], [Zs_uo], 'o', color='g', label = 'B')

        # Connect A and D
        ax.plot([R_of_maxZ, R_of_maxR], [Z_of_maxR, Z_of_maxZ], 'm--')

        # Connect maxZ with maxR
        ax.plot([R_of_maxZ, R_of_maxR], [Z_of_maxZ, Z_of_maxR], 'm--')

        ax.set_aspect('equal')
        ax.legend()
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')

    return Ri_uo, Zi_uo, zeta_uo

def find_intersection_squareness(R, Z, Ax, Az, Dx, Dz):

    R_line = np.linspace(Ax, Dx, 100)
    Z_line = np.linspace(Az, Dz, 100)
    line1 = LineString(zip(R_line, Z_line))
    line2 = LineString(zip(R, Z))
    intersection = line1.intersection(line2)

    return intersection.x, intersection.y

# -----------------------------------------------------------------------------
# Utilities for parameterizations
# -----------------------------------------------------------------------------

def from_RZ_to_mxh(R, Z, n_coeff=3):
    """
    Calculates MXH Coefficients for a flux surface
    """
    Z = np.roll(Z, -np.argmax(R))
    R = np.roll(R, -np.argmax(R))
    if Z[1] < Z[0]: # reverses array so that theta increases
        Z = np.flip(Z)
        R = np.flip(R)

    # compute bounding box for each flux surface
    r = 0.5*(np.max(R)-np.min(R))
    kappa = 0.5*(np.max(Z) - np.min(Z))/r
    R0 = 0.5*(np.max(R)+np.min(R))
    Z0 = 0.5*(np.max(Z)+np.min(Z))
    bbox = [R0, r, Z0, kappa]

    # solve for polar angles
    # need to use np.clip to avoid floating-point precision errors
    theta_r = np.arccos(np.clip(((R - R0) / r), -1, 1))
    theta = np.arcsin(np.clip(((Z - Z0) / r / kappa),-1,1))

    # Find the continuation of theta and theta_r to [0,2pi]
    theta_r_cont = np.copy(theta_r)
    theta_cont = np.copy(theta)

    max_theta = np.argmax(theta)
    min_theta = np.argmin(theta)
    max_theta_r = np.argmax(theta_r)
    min_theta_r = np.argmin(theta_r)

    theta_cont[:max_theta] = theta_cont[:max_theta]
    theta_cont[max_theta:max_theta_r] = np.pi-theta[max_theta:max_theta_r]
    theta_cont[max_theta_r:min_theta] = np.pi-theta[max_theta_r:min_theta]
    theta_cont[min_theta:] = 2*np.pi+theta[min_theta:]

    theta_r_cont[:max_theta] = theta_r_cont[:max_theta]
    theta_r_cont[max_theta:max_theta_r] = theta_r[max_theta:max_theta_r]
    theta_r_cont[max_theta_r:min_theta] = 2*np.pi - theta_r[max_theta_r:min_theta]
    theta_r_cont[min_theta:] = 2*np.pi - theta_r[min_theta:]

    theta_r_cont = theta_r_cont - theta_cont ; theta_r_cont[-1] = theta_r_cont[0]
    
    # Fourier decompose to find coefficients
    c, s = np.zeros(n_coeff), np.zeros(n_coeff)
    def f_theta_r(theta):
        return np.interp(theta, theta_cont, theta_r_cont)
    
    for i in np.arange(n_coeff):
        def integrand_sin(theta):
            return np.sin(i*theta)*(f_theta_r(theta))
        def integrand_cos(theta):
            return np.cos(i*theta)*(f_theta_r(theta))

        s[i] = quad(integrand_sin,0,2*np.pi)[0]/np.pi
        c[i] = quad(integrand_cos,0,2*np.pi)[0]/np.pi
        
    return c, s, bbox

def from_mxh_to_RZ(R0, a, kappa, Z0, cn, sn, thetas = None):

    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 100)

    # Prepare data to always have the first dimension a batch (e.g. a radius) for parallel computation
    if IOtools.isfloat(R0):
        R0 = [R0]
        a = [a]
        kappa = [kappa]
        Z0 = [Z0]
        cn = np.array(cn)[np.newaxis,:]
        sn = np.array(sn)[np.newaxis,:]

    R0 = np.array(R0)
    a = np.array(a)
    kappa = np.array(kappa)
    Z0 = np.array(Z0)

    R = np.zeros((R0.shape[0],len(thetas)))
    Z = np.zeros((R0.shape[0],len(thetas)))
    n = np.arange(1, sn.shape[1])
    for i,theta in enumerate(thetas):
        theta_R = theta + cn[:,0] + np.sum( cn[:,1:]*np.cos(n*theta) + sn[:,1:]*np.sin(n*theta), axis=-1 )
        R[:,i] = R0 + a*np.cos(theta_R)
        Z[:,i] = Z0 + kappa*a*np.sin(theta)

    return R, Z

# --------------------------------------------------------------------------------------------------------------
# Fixed boundary stuff
# --------------------------------------------------------------------------------------------------------------

class freegs_millerized:

    def __init__(self, R, a, kappa_sep, delta_sep, zeta_sep, z0):

        print("> Fixed-boundary equilibrium with FREEGS")

        print("\t- Initializing miller geometry")
        print(f"\t\t* R={R} m, a={a} m, kappa_sep={kappa_sep}, delta_sep={delta_sep}, zeta_sep={zeta_sep}, z0={z0} m")

        self.R0 = R
        self.a = a
        self.kappa_sep = kappa_sep
        self.delta_sep = delta_sep
        self.zeta_sep = zeta_sep
        self.Z0 = z0

        thetas = np.linspace(0, 2*np.pi, 1000, endpoint=False)

        self.mitim_separatrix = mitim_flux_surfaces()
        self.mitim_separatrix.reconstruct_from_miller(self.R0, self.a, self.kappa_sep, self.Z0, self.delta_sep, self.zeta_sep, thetas = thetas)
        self.R_sep, self.Z_sep = self.mitim_separatrix.R[0,:], self.mitim_separatrix.Z[0,:]

    def prep(self,  p0_MPa, Ip_MA, B_T,
            beta_pol = None, n_coils = 10, resol_eq = 2**8+1,
            parameters_profiles = {'alpha_m':2.0, 'alpha_n':2.0, 'Raxis':1.0},
            constraint_miller_squareness_point = False):

        print("\t- Initializing plasma parameters")
        if beta_pol is not None:
            print(f"\t\t* beta_pol={beta_pol:.5f}, Ip={Ip_MA:.5f} MA, B={B_T:.5f} T")
        else:
            print(f"\t\t* p0={p0_MPa:.5f} MPa, Ip={Ip_MA:.5f} MA, B={B_T:.5f} T")

        self.p0_MPa = p0_MPa
        self.Ip_MA = Ip_MA
        self.B_T = B_T
        self.beta_pol = beta_pol
        self.parameters_profiles = parameters_profiles

        print(f"\t- Preparing equilibrium with FREEGS, with a resolution of {resol_eq}x{resol_eq}")
        self._define_coils(n_coils)
        self._define_eq(resol = resol_eq)
        self._define_gs()

        # Define xpoints
        print("\t\t* Defining upper and lower x-points")
        self.xpoints = [
            (self.R0 - self.a*self.delta_sep, self.Z0+self.a*self.kappa_sep),
            (self.R0 - self.a*self.delta_sep, self.Z0-self.a*self.kappa_sep),
        ]

        # Define isoflux
        print("\t\t* Defining midplane separatrix (isoflux)")
        self.isoflux = [
            (self.xpoints[0][0], self.xpoints[0][1], self.R0 + self.a, self.Z0),                 # Upper x-point with outer midplane
            (self.xpoints[0][0], self.xpoints[0][1], self.R0 - self.a, self.Z0),                 # Upper x-point with inner midplane
            (self.xpoints[0][0], self.xpoints[0][1], self.xpoints[1][0], self.xpoints[1][1]),   # Between x-points
        ]

        print("\t\t* Defining squareness isoflux point")

        # Find squareness point
        if constraint_miller_squareness_point:
            Rsq, Zsq, _ = find_squareness_points(self.R_sep, self.Z_sep)

            self.isoflux.append(
                (self.xpoints[0][0], self.xpoints[0][1], Rsq, Zsq)         # Upper x-point with squareness point
            )
            self.isoflux.append(
                (self.xpoints[0][0], self.xpoints[0][1], Rsq, -Zsq)        # Upper x-point with squareness point
            )

        # Combine
        self.constrain = freegs.control.constrain(
            isoflux=self.isoflux,
            xpoints=self.xpoints,
            )

    def _define_coils(self, n, rel_distance_coils = 0.5, updown_coils = True):

        print(f"\t- Defining {n} coils{' (up-down symmetric)' if updown_coils else ''} at a distance of {rel_distance_coils}*a from the separatrix")

        self.distance_coils = self.a*rel_distance_coils
        self.updown_coils = updown_coils

        if self.updown_coils:
            thetas = np.linspace(0, np.pi, n)
        else:
            thetas = np.linspace(0, 2*np.pi, n, endpoint=False)

        self.mitim_coils_surface = mitim_flux_surfaces()
        self.mitim_coils_surface.reconstruct_from_miller(self.R0, (self.a+self.distance_coils), self.kappa_sep, self.Z0, self.delta_sep, self.zeta_sep, thetas = thetas)
        self.Rcoils, self.Zcoils = self.mitim_coils_surface.R[0,:], self.mitim_coils_surface.Z[0,:]

        self.coils = []
        for num, (R, Z) in enumerate(zip(self.Rcoils, self.Zcoils)):

            if self.updown_coils and Z > 0:
                coilU = freegs.machine.Coil(
                    R,
                    Z
                    )
                coilL = freegs.machine.Coil(
                    R,
                    -Z
                    )
                coil = freegs.machine.Circuit( [ ('U', coilU, 1.0 ), ('L', coilL, 1.0 ) ] )
            else:

                coil = freegs.machine.Coil(
                    R,
                    Z
                    )

            self.coils.append(
                (f"coil_{num}", coil)
                )

    def _define_eq(self, resol=2**9+1):

        print("\t- Defining equilibrium")
        self.tokamak = freegs.machine.Machine(self.coils)

        a = self.a + self.distance_coils

        Rmin = (self.R0-a) - a*0.25
        Rmax = (self.R0+a) + a*0.25

        b = a*self.kappa_sep
        Zmin = (self.Z0 - b) - b*0.25
        Zmax = (self.Z0 + b) + b*0.25

        self.eq = freegs.Equilibrium(tokamak=self.tokamak,
                                Rmin=Rmin, Rmax=Rmax,
                                Zmin=Zmin, Zmax=Zmax,
                                nx=resol, ny=resol,
                                boundary=freegs.boundary.freeBoundaryHagenow)

    def _define_gs(self):

        if self.beta_pol is None:
            print("\t- Defining Grad-Shafranov equilibrium: p0, Ip and vaccum R*Bt")

            self.profiles = freegs.jtor.ConstrainPaxisIp(self.eq,
                self.p0_MPa*1E6, self.Ip_MA*1E6, self.R0*self.B_T,
                alpha_m=self.parameters_profiles['alpha_m'], alpha_n=self.parameters_profiles['alpha_n'], Raxis=self.parameters_profiles['Raxis'])

        else:
            print("\t- Defining Grad-Shafranov equilibrium: beta_pol, Ip and vaccum R*Bt")

            self.profiles = freegs.jtor.ConstrainBetapIp(self.eq,
                self.beta_pol, self.Ip_MA*1E6, self.R0*self.B_T,
                alpha_m=self.parameters_profiles['alpha_m'], alpha_n=self.parameters_profiles['alpha_n'], Raxis=self.parameters_profiles['Raxis'])

    def solve(self, show = False, rtol=1e-6):

        print("\t- Solving equilibrium with FREEGS")
        with IOtools.timer():
            self.x,self.y = freegs.solve(self.eq,         # The equilibrium to adjust
                self.profiles,                 # The toroidal current profile function
                constrain=self.constrain,      # Constraint function to set coil currents
                show=show,
                rtol=rtol,             # Default is 1e-3
                atol=1e-10,
                maxits=100,            # Default is 50
                convergenceInfo=True)  
            print("\t\t * Done!")

        self.check()

    def check(self, warning_error = 0.01, plotYN = False):

        print("\t- Checking separatrix quality (Miller vs FREEGS)")
        RZ = self.eq.separatrix()

        self.mitim_separatrix_eq = mitim_flux_surfaces()
        self.mitim_separatrix_eq.reconstruct_from_RZ(RZ[:,0], RZ[:,1])

        # --------------------------------------------------------------
        # Check errors
        # --------------------------------------------------------------

        max_error = 0.0
        for key in ['R0', 'a', 'kappa_sep', 'delta_sep', 'zeta_sep']:
            miller_value = getattr(self, key)
            sep_value = getattr(self.mitim_separatrix_eq, key.replace('_sep', ''))[0]
            error = abs( (miller_value-sep_value)/miller_value )
            print(f"\t\t* {key}: {miller_value:.3f} vs {sep_value:.3f} ({100*error:.2f}%)")

            max_error = np.max([max_error, error])

        if max_error > warning_error:
            print(f"\t\t- WARNING: maximum error is {100*max_error:.2f}%", typeMsg='w')
        else:
            print(f"\t\t- Maximum error is {100*max_error:.2f}%", typeMsg='i')

        # --------------------------------------------------------------
        # Plotting
        # --------------------------------------------------------------

        if plotYN:

            fig = plt.figure(figsize=(12,8))
            axs = fig.subplot_mosaic(
                """
                AB
                AB
                CB
                """
            )

            # Plot direct FreeGS output

            ax = axs['A']
            self.eq.plot(axis=ax,show=False)
            self.constrain.plot(axis=ax, show=False)

            for coil in self.coils:
                if isinstance(coil[1],freegs.machine.Circuit):
                    ax.plot([coil[1]['U'].R], [coil[1]['U'].Z], 's', c='k', markersize=2)
                    ax.plot([coil[1]['L'].R], [coil[1]['L'].Z], 's', c='k', markersize=2)
                else:
                    ax.plot([coil[1].R], [coil[1].Z], 's', c='k', markersize=2)

            GRAPHICStools.addLegendApart(ax,ratio=0.9,size=10)

            ax = axs['C']
            ax.plot(self.x,'-o', markersize=3, color='b', label = '$\\psi$ max change')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('$\\psi$ max change')
            ax.set_yscale('log')
            ax.legend(loc='lower left',prop={'size': 10})

            ax = axs['C'].twinx()
            ax.plot(self.y,'-o', markersize=3, color='r', label = '$\\psi$ max relative change')
            ax.set_ylabel('$\\psi$ max relative change')
            ax.set_yscale('log')
            ax.legend(loc='upper right',prop={'size': 10})

            # Plot comparison of equilibria

            ax = axs['B']

            self.mitim_separatrix.plot(ax=ax, color = 'b', label = 'Miller (original)', plot_extremes=True)
            self.mitim_separatrix_eq.plot(ax=ax, color = 'r', label = 'Separatrix (freegs)', plot_extremes=True)

            ax.legend(prop={'size': 10})

            plt.show()

    def derive(self, psi_surfaces = np.linspace(0,1.0,10), psi_profiles = np.linspace(0,1.0,100)):

        # Grab surfaces
        Rs, Zs = [], []
        for psi_norm in psi_surfaces:
            R, Z = self.find_surface(psi_norm)
            Rs.append(R)
            Zs.append(Z)
        Rs = np.array(Rs)
        Zs = np.array(Zs)
            
        # Calculate surface stuff in parallel
        self.surfaces = mitim_flux_surfaces()
        self.surfaces.reconstruct_from_RZ(Rs, Zs)
        self.surfaces.psi = psi_surfaces

        # Grab profiles
        self.profile_psi_norm = psi_profiles
        self.profile_pressure = self.eq.pressure(psinorm =psi_profiles)*1E-6
        self.profile_q = self.eq.q(psinorm = psi_profiles)
        self.profile_RB = self.eq.fpol(psinorm = psi_profiles)

        # Grab quantities
        self.profile_q95 = self.eq.q(psinorm = 0.95)
        self.profile_q0 = self.eq.q(psinorm = 0.0)
        self.profile_betaN = self.eq.betaN()
        self.profile_Li2 = self.eq.internalInductance2()
        self.profile_pave = self.eq.pressure_ave()
        self.profile_beta_pol =  self.eq.poloidalBeta()
        self.profile_Ashaf = self.eq.shafranovShift

    def find_surface(self, psi_norm = 0.5, thetas = None):

        if psi_norm == 0.0:
            psi_norm = 1E-6

        if psi_norm == 1.0:
            RZ = self.eq.separatrix(npoints= 1000 if thetas is None else len(thetas))
            R, Z = RZ[:,0], RZ[:,1]
        else:
            if thetas is None:
                thetas = np.linspace(0, 2*np.pi, 1000, endpoint=False)

            from freegs.critical import find_psisurface, find_critical
            from scipy import interpolate

            psi = self.eq.psi()
            opoint, xpoint = find_critical(self.eq.R, self.eq.Z, psi)
            psinorm = (psi - opoint[0][2]) / (self.eq.psi_bndry - opoint[0][2])
            psifunc = interpolate.RectBivariateSpline(self.eq.R[:, 0], self.eq.Z[0, :], psinorm)
            r0, z0 = opoint[0][0:2]

            R = np.zeros(len(thetas))
            Z = np.zeros(len(thetas))
            for i,theta in enumerate(thetas):
                R[i],Z[i] = find_psisurface(
                    self.eq,
                    psifunc,
                    r0,
                    z0,
                    r0 + 10.0 * np.sin(theta),
                    z0 + 10.0 * np.cos(theta),
                    psival=psi_norm,
                    n=1000,
                )
        return R,Z

    # --------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------

    def plot(self, axs = None, color = 'b', label = ''):

        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(16,7))
            axs = fig.subplot_mosaic(
                """
                A12
                A34
                """)
            axs = [axs['A'], axs['1'], axs['2'], axs['3'], axs['4']]

        self.plot_flux_surfaces(ax = axs[0], color = color)
        self.plot_profiles(axs = axs[1:], color = color, label = label)

    def plot_flux_surfaces(self, ax = None, color = 'b'):

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(figsize=(12,8))

        for i in range(self.surfaces.R.shape[0]):
            ax.plot(self.surfaces.R[i],self.surfaces.Z[i], '-', label = f'$\\psi$ = {self.surfaces.psi[i]:.2f}', color = color, markersize=3)

        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        GRAPHICStools.addDenseAxis(ax)

    def plot_profiles(self, axs = None, color = 'b', label = ''):

        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
            axs = axs.flatten()

        ax = axs[0]
        ax.plot(self.profile_psi_norm,self.profile_pressure,'-',color=color, label = label)
        ax.set_xlabel('$\\psi$')
        ax.set_xlim([0,1])
        ax.set_ylabel('Pressure (MPa)')
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1]
        ax.plot(self.profile_psi_norm,self.profile_q,'-',color=color)
        ax.axhline(y=1, color='k', ls='--', lw=0.5)
        ax.set_xlabel('$\\psi$')
        ax.set_ylabel('q')
        ax.set_xlim([0,1])
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[2]
        ax.plot(self.profile_psi_norm,self.profile_RB,'-',color=color)
        ax.axhline(y=self.R0*self.B_T, color=color, ls='--', lw=0.5)
        ax.set_xlabel('$\\psi$')
        ax.set_ylabel('$R\\cdot B_t$ ($T\\cdot m$)')
        ax.set_xlim([0,1])
        GRAPHICStools.addDenseAxis(ax)

    def plot_flux_surfaces_characteristics(self, axs = None, color = 'b', label = ''):

        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
            axs = axs.flatten()

        ax = axs[0]
        ax.plot(self.surfaces.psi, self.surfaces.kappa, '-o', color=color, label = label, markersize=3)
        ax.set_xlabel('$\\psi$')
        ax.set_ylabel('$\\kappa$')
        ax.set_xlim([0,1])
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1]
        ax.plot(self.surfaces.psi, self.surfaces.delta, '-o', color=color, label = label, markersize=3)
        ax.set_xlabel('$\\psi$')
        ax.set_ylabel('$\\delta$')
        ax.set_xlim([0,1])
        GRAPHICStools.addDenseAxis(ax)

    # --------------------------------------------------------------
    # Writing
    # --------------------------------------------------------------

    def write(self, filename = "mitim_freegs.geqdsk"):

        print(f"\t- Writing equilibrium to {IOtools.clipstr(filename)}")

        with open(filename, "w") as f:
            geqdsk.write(self.eq, f)

    def to_profiles(self, scratch_folder = '~/scratch/'):

        # Produce geqdsk object
        scratch_folder = IOtools.expandPath(scratch_folder)
        file_scratch = f'{scratch_folder}/mitim_freegs.geqdsk'
        self.write(file_scratch)
        g = MITIMgeqdsk(file_scratch)

        os.remove(file_scratch)

        # From geqdsk to profiles
        return g.to_profiles()

    def to_transp(self, folder = '~/scratch/', shot = '12345', runid = 'P01', ne0_20 = 1E19, Vsurf = 0.0, Zeff = 1.5, PichT_MW = 11.0, times = [0.0,1.0]):

        # Produce geqdsk object
        scratch_folder = IOtools.expandPath(folder)
        file_scratch = f'{scratch_folder}/mitim_freegs.geqdsk'
        self.write(file_scratch)
        g = MITIMgeqdsk(file_scratch)

        return g.to_transp(folder=folder, shot=shot, runid=runid, ne0_20=ne0_20, Vsurf=Vsurf, Zeff=Zeff, PichT_MW=PichT_MW, times=times)
