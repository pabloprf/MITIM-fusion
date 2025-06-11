import netCDF4
import math
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import MATHtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


class transp_output:
    """
    Some variables are model-specific, because the formulas are not automatically saved in every astra run.
    In particular be careful when dealing with variables defined through dummy variables, whose names are not
    exactly the same of the dummy variables themself (e.g. self.chi_e_TGLF_smoothed = self.f['CAR22'][:])
    """

    def __init__(self, netCDFfile, boundary=0.85):
        self.LocationCDF = netCDFfile
        self.nc_file = netCDF4.Dataset(self.LocationCDF, 'r')
        self.f = self.nc_file.variables
        #self.f = netCDF4.Dataset(self.LocationCDF).variables
        self.boundary = boundary

    def close_file(self):
        if hasattr(self, 'nc_file'):
            self.nc_file.close()
            del self.nc_file

    def getProfiles(self):

        ### Constants

        self.GP = self.f["GP"][:]       # pi
        self.GP2 = self.f["GP2"][:]     # 2*pi

        ### Few control parameters

        try:
            self.t = self.f[
                "TIME"
            ].data  # New ASTRA update needs this patch, for old version still need [:]
        except:
            self.t = self.f["TIME"][:]
        self.t = np.array([self.t]) if np.isscalar(self.t) else np.array(self.t)
        self.tau = self.f["TAU"][:]     # simulation time step
        self.na1 = self.f["NA1"][:]     # transport grid size

        ### Geometry
        
        try:
            self.R = self.f["r2d"][:]       # R coordinates
            self.Z = self.f["z2d"][:]       # Z coordinates
        except:
            self.R = self.f["R"][:]
            self.Z = self.f["Z"][:]
        self.ROC = self.f["ROC"][:]         # effective minor radius
        self.rho = self.f["RHO"][:]         # main magnetic surface label
        self.xrho = self.f["XRHO"][:]       # sqrt of normalized toroidal magnetic flux
        self.XRHO = self.xrho
        self.HRO = self.f["HRO"][:]         # radial grid step in rho
        self.rmin = self.f["AMETR"][:]      # minor radius
        self.elong = self.f["ELONG"][:]     # separatrix elongation
        self.elon = self.f["ELON"][:]       # elongation profile
        self.trian = self.f["TRIAN"][:]     # separatrix triangularity
        self.tria = self.f["TRIA"][:]       # triangularity
        self.UPDWN = self.f["UPDWN"][:]     # vertical shift separatrix      
        self.shift = self.f["SHIFT"][:]     # Shafranov shift separatrix
        self.shif = self.f["SHIF"][:]       # Shafranov shift profile
        self.RTOR = self.f["RTOR"][:]       # Major radius
        self.ABC = self.f["ABC"][:]         # minor radius at separatrix on OMP
        self.AB = self.f["AB"][:]           # maximum value allowed for ABC
        self.vol = self.f["VOLUM"][:]       # plasma volume profile
        self.VOLUME = self.f["VOLUME"][:]   # plasma volume inside separatrix
        self.AREAT = self.f['AREAT'][:]     # cross section area
        self.SLAT = self.f['SLAT'][:]       # lateral plasma surface
        self.G11 = self.f["G11"][:]         # geometrical factor 1
        self.G22 = self.f["G22"][:]         # geometrical factor 2
        self.G33 = self.f["G33"][:]         # geometrical factor 3

        self.timesteps = len(self.t)
        self.radialsize = int(self.na1[-1])
        self.area = np.zeros([self.timesteps,self.radialsize])
        for ii in range(0,int(self.na1[-1])):
            if ii>0:
                self.area[:,ii] = self.AREAT[:,ii]-self.AREAT[:,ii-1]
            else:
                self.area[:,ii] = self.AREAT[:,ii]    # cross section differential area

        ### Essential parameters

        self.BTOR = self.f["BTOR"][:]   # toroidal magnetic field
        self.IPL = self.f["IPL"][:]     # plasma current
        self.TEX = self.f["TEX"][:]     # experimental electron temperature
        self.TIX = self.f["TIX"][:]     # experimental ion temperature
        self.NEX = self.f["NEX"][:]     # experimental electron density
        self.NIX = self.f["NIX"][:]     # experimental ion density
        self.Te = self.f["TE"][:]       # electron temperature
        self.Ti = self.f["TI"][:]       # ion temperature
        self.ne = self.f["NE"][:]       # electron density
        self.NE = self.ne
        self.ni = self.f["NI"][:]       # ion density
        self.NI = self.ni
        self.NMAIN = self.f["NMAIN"][:]     # main ion density
        self.NDEUT = self.f["NDEUT"][:]     # D density
        self.NTRIT = self.f["NTRIT"][:]     # T density
        self.NIZ1 = self.f["NIZ1"][:]       # 1st impurity density
        self.NIZ2 = self.f["NIZ2"][:]       # 2nd impurity density
        self.NIZ3 = self.f["NIZ3"][:]       # 3rd impurity density
        self.NALF = self.f["NALF"][:]       # alpha density
        self.ZMJ = self.f["ZMJ"][:]         # main ion charge
        self.ZMAIN = self.f["ZMAIN"][:]     # main ion charge
        self.ZIM1 = self.f["ZIM1"][:]       # 1st impurity charge
        self.ZIM2 = self.f["ZIM2"][:]       # 2nd impurity charge
        self.ZIM3 = self.f["ZIM3"][:]       # 3rd impurity charge
        self.AMJ = self.f["AMJ"][:]         # main ion mass
        self.AMAIN = self.f["AMAIN"][:]     # main ion mass
        self.AIM1 = self.f["AIM1"][:]       # 1st impurity mass
        self.AIM2 = self.f["AIM2"][:]       # 2nd impurity mass
        self.AIM3 = self.f["AIM3"][:]       # 3rd impurity mass
        self.FP = self.f["FP"][:]           # poloidal magnetic flux
        self.TF = self.rho[:,-1] * self.rho[:,-1] * self.BTOR[-1] / 2       # ~average toroidal flux?? (Wb/rad)
        self.ER = self.f["ER"][:]           # radial electric field
        self.VPOL = self.f["VPOL"][:]       # poloidal plasma velocity
        self.VTOR = self.f["VTOR"][:]       # toroidal plasma velocity
        self.F1 = self.f["F1"][:]           # density of 1st additional transported species
        self.F2 = self.f["F2"][:]           # density of 2nd additional transported species
        self.F3 = self.f["F3"][:]           # density of 3rd additional transported species
        self.VR = self.f["VR"][:]           # volume derivative
        self.Cu = self.f["CU"][:]           # current density
        self.Cubs = self.f["CUBS"][:]       # bootstrap current density
        self.CuTor = self.f["CUTOR"][:]     # toroidal current density
        self.CD = self.f["CD"][:]           # driven current density
        self.Mu = self.f["MU"][:]           # rotational transform
        self.q  = 1/self.Mu                 # safety factor
        self.q_onaxis = 1/self.Mu[:,0]      # q on axis
        self.MV = self.f["MV"][:]           # vacuum rotational transform
        self.FV = self.f["FV"][:]           # poloidal flux for vacuum magnetic field
        self.VP = self.f["VP"][:]           # pinch velocity
        self.Qi = self.f["QI"][:]           # total transported ion heat flux
        self.Qe = self.f["QE"][:]           # total transported electron heat flux
        self.Qn = self.f["QN"][:]           # total transported particles flux
        self.ZEF = self.f["ZEF"][:]         # Zeff
        self.FTO = self.f["FTO"][:]         # toroidal magnetic flux at the edge
        self.DN = self.f["DN"][:]           # main ion particle diffusivity
        # self.HN    = self.f['HN'][:]
        # self.XN    = self.f['XN'][:]
        self.CN = self.f["CN"][:]           # particle convection
        # self.DE    = self.f['DE'][:]
        self.HE = self.f["HE"][:]           # electron heat diffusivity due to Te gradient
        # self.XE    = self.f['XE'][:]
        self.CE = self.f["CE"][:]           # electron heat convection
        # self.DI    = self.f['DI'][:]
        # self.HI    = self.f['HI'][:]
        self.XI = self.f["XI"][:]           # main ion heat diffusivity due to Ti gradient
        self.CI = self.f["CI"][:]           # ion heat convection
        self.DC = self.f["DC"][:]           # current diffusivity due to n gradient
        self.HC = self.f["HC"][:]           # current diffusivity due to Te gradient
        self.XC = self.f["XC"][:]           # current diffusivity due to Ti gradient
        self.CC = self.f["CC"][:]           # conductivity
        self.UPAR = self.f['UPAR'][:]       # toroidal velocity
        self.XUPAR = self.f['XUPAR'][:]     # Momentum diffusivity
        self.CNPAR = self.f['CNPAR'][:]     # Momentum convective velocity
        self.RUPFR = self.f['RUPFR'][:]     # Toroidal turbulence-driven instrinsique torque
        self.TTRQ = self.f['TTRQ'][:]       # Applied external torque (e.g. NBI)
        self.SN = self.f["SN"][:]           # particle source
        self.SNTOT = self.f["SNTOT"][:]
        self.SNEBM = self.f['SNEBM'][:]     # particle source due to NBI
        # self.SNN   = self.f['SNN'][:]
        # self.SNNEU  = self.f['SNNEU'][:]
        self.PBPER = self.f['PBPER'][:]     # pressure of fast ions in the perpendicular direction (wrt Bt)
        self.PBLON = self.f['PBLON'][:]     # pressure of fast ions in the longitudinal direction (wrt Bt)
        self.ULON = self.f["ULON"][:]       # longitudinal loop voltage
        self.UPL = self.f["UPL"][:]         # toroidal loop voltage
        self.IPOL = self.f["IPOL"][:]       # normalized poloidal current

        ### Power sources and sinks

        self.PE = self.f["PE"][:]           # local electron power density
        self.PI = self.f["PI"][:]           # local ion power density
        self.PEBM = self.f["PEBM"][:]       # NBI power to electrons
        self.PIBM = self.f["PIBM"][:]       # NBI power to ions
        self.PEECR = self.f["PEECR"][:]     # ECH heating to electrons
        self.PRAD = self.f["PRAD"][:]       # Radiated Power
        self.PEICR = self.f["PEICR"][:]     # ICH heating to electrons
        self.PIICR = self.f["PIICR"][:]     # ICH heating to ions
        self.POH = self.CC*(self.ULON/(self.GP2[-1]*self.RTOR[-1]*self.IPOL))**2/self.G33
        #### --------------- Calculation of fusion partition between main ions and electrons ------------------------- ###
        YVALP = 1.2960e+07
        ne = np.maximum(self.ne, 1e-30)
        te = np.maximum(self.Te, 1e-30)
        YLLAME = 23.9 + np.log(1e3 * te / np.sqrt(1e19 * ne))
        yy6 = np.sqrt(1e3 * te / 1e19 / ne) * (4.0 * self.AMAIN * YVALP) / (4.0 + self.AMAIN)
        YLLAMI = 14.2 + np.log(np.maximum(yy6, 0.1))
        yy6 = np.sqrt(1e3 * te / 1e19 / ne) * 2.0 * YVALP
        YLLAMA = 14.2 + np.log(np.maximum(yy6, 0.01))
        yy6 = (YLLAMI * self.NI / (self.AMAIN * ne) + YLLAMA * self.NALF / ne) * 7.3e-4 / YLLAME
        yy6 = np.maximum(yy6, 1e-4)
        yvc = yy6**0.33 * np.sqrt(2.0 * te * 1.7564e14)
        yeps = YVALP / (yvc + 1e-4)
        yy6 = np.arctan(0.577 * (2.0 * yeps - 1.0))
        yy7 = np.log((1.0 + yeps)**2 / (1.0 - yeps + yeps**2))
        self.PAION1 = 2.0 / yeps**2 * (0.577 * yy6 - 0.167 * yy7 + 0.3)     # alpha power fraction to main ions
        self.SVDT = self.Ti**(-1/3)
        self.SVDT = 8.972*np.exp(-19.9826*self.SVDT)*self.SVDT*self.SVDT*((self.Ti+1.0134)/(1.+6.386E-3*(self.Ti+1.0134)**2)+1.877*np.exp(-0.16176*self.Ti*np.sqrt(self.Ti)))       # nuclear fusion cross section
        self.PDT = 5.632*self.NDEUT*self.NTRIT*self.SVDT                # total alpha power
        self.PEDT = (1-self.PAION1)*self.PDT                            # alpha power to electrons
        self.PIDT = self.PAION1*self.PDT                                # alpha power to main ions
        self.COULG = 15.9-0.5*np.log(self.ne)+np.log(self.Te)           # Coloumb logarithm
        self.PEICL = 0.00246*self.COULG*self.ne*self.ni*self.ZMAIN**2
        self.PEICL = self.PEICL*(self.Te-self.Ti)/(self.AMAIN*self.Te*np.sqrt(self.Te))     # Collisional exchange power

        ### Dummy arrays (used for user-specified quantities)

        self.CAR1 = self.f["CAR1"][:]
        self.CAR2 = self.f["CAR2"][:]
        self.CAR3 = self.f["CAR3"][:]
        self.CAR4 = self.f["CAR4"][:]
        self.CAR5 = self.f["CAR5"][:]
        self.CAR6 = self.f["CAR6"][:]
        self.CAR7 = self.f["CAR7"][:]
        self.CAR8 = self.f["CAR8"][:]
        self.CAR9 = self.f["CAR9"][:]
        self.CAR10 = self.f["CAR10"][:]
        self.CAR11 = self.f["CAR11"][:]
        self.CAR12 = self.f["CAR12"][:]
        self.CAR13 = self.f["CAR13"][:]
        self.CAR14 = self.f["CAR14"][:]
        self.CAR15 = self.f["CAR15"][:]
        self.CAR16 = self.f["CAR16"][:]
        self.CAR17 = self.f["CAR17"][:]
        self.CAR18 = self.f["CAR18"][:]
        self.CAR19 = self.f["CAR19"][:]
        self.CAR20 = self.f["CAR20"][:]
        self.CAR21 = self.f["CAR21"][:]
        self.CAR22 = self.f["CAR22"][:]
        self.CAR23 = self.f["CAR23"][:]
        self.CAR24 = self.f["CAR24"][:]
        self.CAR25 = self.f["CAR25"][:]
        self.CAR26 = self.f["CAR26"][:]
        self.CAR27 = self.f["CAR27"][:]
        self.CAR28 = self.f["CAR28"][:]
        self.CAR29 = self.f["CAR29"][:]
        self.CAR30 = self.f["CAR30"][:]
        self.CAR31 = self.f["CAR31"][:]
        self.CAR32 = self.f["CAR32"][:]
        self.CAR33 = self.f["CAR33"][:]
        self.CAR34 = self.f["CAR34"][:]
        self.CAR35 = self.f["CAR35"][:]
        self.CAR36 = self.f["CAR36"][:]
        self.CAR37 = self.f["CAR37"][:]
        self.CAR38 = self.f["CAR38"][:]
        self.CAR39 = self.f["CAR39"][:]
        self.CAR40 = self.f["CAR40"][:]
        self.CAR41 = self.f["CAR41"][:]
        self.CAR42 = self.f["CAR42"][:]
        self.CAR43 = self.f["CAR43"][:]
        self.CAR44 = self.f["CAR44"][:]
        self.CAR45 = self.f["CAR45"][:]
        self.CAR46 = self.f["CAR46"][:]
        self.CAR47 = self.f["CAR47"][:]
        self.CAR48 = self.f["CAR48"][:]
        self.CAR49 = self.f["CAR49"][:]
        self.CAR50 = self.f["CAR50"][:]
        self.CAR51 = self.f["CAR51"][:]
        self.CAR52 = self.f["CAR52"][:]
        self.CAR53 = self.f["CAR53"][:]
        self.CAR54 = self.f["CAR54"][:]
        self.CAR55 = self.f["CAR55"][:]
        self.CAR56 = self.f["CAR56"][:]
        self.CAR57 = self.f["CAR57"][:]
        self.CAR58 = self.f["CAR58"][:]
        self.CAR59 = self.f["CAR59"][:]
        self.CAR60 = self.f["CAR60"][:]
        self.CAR61 = self.f["CAR61"][:]
        self.CAR62 = self.f["CAR62"][:]
        self.CAR63 = self.f["CAR63"][:]
        self.CAR64 = self.f["CAR64"][:]
        self.CAR1X = self.f["CAR1X"][:]
        self.CAR2X = self.f["CAR2X"][:]
        self.CAR3X = self.f["CAR3X"][:]
        self.CAR4X = self.f["CAR4X"][:]
        self.CAR5X = self.f["CAR5X"][:]
        self.CAR6X = self.f["CAR6X"][:]
        self.CAR7X = self.f["CAR7X"][:]
        self.CAR8X = self.f["CAR8X"][:]
        self.CAR9X = self.f["CAR9X"][:]
        self.CAR10X = self.f["CAR10X"][:]
        self.CAR11X = self.f["CAR11X"][:]
        self.CAR12X = self.f["CAR12X"][:]
        self.CAR13X = self.f["CAR13X"][:]
        self.CAR14X = self.f["CAR14X"][:]
        self.CAR15X = self.f["CAR15X"][:]
        self.CAR16X = self.f["CAR16X"][:]
        self.CAR17X = self.f["CAR17X"][:]
        self.CAR18X = self.f["CAR18X"][:]
        self.CAR19X = self.f["CAR19X"][:]
        self.CAR20X = self.f["CAR20X"][:]
        self.CAR21X = self.f["CAR21X"][:]
        self.CAR22X = self.f["CAR22X"][:]
        self.CAR23X = self.f["CAR23X"][:]
        self.CAR24X = self.f["CAR24X"][:]
        self.CAR25X = self.f["CAR25X"][:]
        self.CAR26X = self.f["CAR26X"][:]
        self.CAR27X = self.f["CAR27X"][:]
        self.CAR28X = self.f["CAR28X"][:]
        self.CAR29X = self.f["CAR29X"][:]
        self.CAR30X = self.f["CAR30X"][:]
        self.CAR31X = self.f["CAR31X"][:]
        self.CAR32X = self.f["CAR32X"][:]
        self.CAR33X = self.f["CAR33X"][:]
        self.CAR34X = self.f["CAR34X"][:]
        self.CAR35X = self.f["CAR35X"][:]
        self.CAR36X = self.f["CAR36X"][:]
        self.CAR37X = self.f["CAR37X"][:]
        self.CAR38X = self.f["CAR38X"][:]
        self.CAR39X = self.f["CAR39X"][:]
        self.CAR40X = self.f["CAR40X"][:]
        self.CAR41X = self.f["CAR41X"][:]
        self.CAR42X = self.f["CAR42X"][:]
        self.CAR43X = self.f["CAR43X"][:]
        self.CAR44X = self.f["CAR44X"][:]
        self.CAR45X = self.f["CAR45X"][:]
        self.CAR46X = self.f["CAR46X"][:]
        self.CAR47X = self.f["CAR47X"][:]
        self.CAR48X = self.f["CAR48X"][:]
        self.CAR49X = self.f["CAR49X"][:]
        self.CAR50X = self.f["CAR50X"][:]
        self.CAR51X = self.f["CAR51X"][:]
        self.CAR52X = self.f["CAR52X"][:]
        self.CAR53X = self.f["CAR53X"][:]
        self.CAR54X = self.f["CAR54X"][:]
        self.CAR55X = self.f["CAR55X"][:]
        self.CAR56X = self.f["CAR56X"][:]
        self.CAR57X = self.f["CAR57X"][:]
        self.CAR58X = self.f["CAR58X"][:]
        self.CAR59X = self.f["CAR59X"][:]
        self.CAR60X = self.f["CAR60X"][:]
        self.CAR61X = self.f["CAR61X"][:]
        self.CAR62X = self.f["CAR62X"][:]
        self.CAR63X = self.f["CAR63X"][:]
        self.CAR64X = self.f["CAR64X"][:]

        ### Dummy scalars (used for user-specified quantities)

        self.CRAD1   = self.f['CRAD1'][:]
        self.CRAD2   = self.f['CRAD2'][:]
        self.CRAD3   = self.f['CRAD3'][:]
        self.CRAD4   = self.f['CRAD4'][:]
        self.CIMP1   = self.f['CIMP1'][:]
        self.CIMP2   = self.f['CIMP2'][:]
        self.CIMP3   = self.f['CIMP3'][:]
        self.CIMP4   = self.f['CIMP4'][:]
        self.ZRD1 = self.f["ZRD1"][:]
        self.ZRD2 = self.f["ZRD2"][:]
        self.ZRD3 = self.f["ZRD3"][:]
        self.ZRD4 = self.f["ZRD4"][:]
        self.ZRD5 = self.f["ZRD5"][:]
        self.ZRD6 = self.f["ZRD6"][:]
        self.ZRD7 = self.f["ZRD7"][:]
        self.ZRD8 = self.f["ZRD8"][:]
        self.ZRD9 = self.f["ZRD9"][:]
        self.ZRD10 = self.f["ZRD1"][:]
        self.ZRD11 = self.f["ZRD11"][:]
        self.ZRD12 = self.f["ZRD12"][:]
        self.ZRD13 = self.f["ZRD13"][:]
        self.ZRD14 = self.f["ZRD14"][:]
        self.ZRD15 = self.f["ZRD15"][:]
        self.ZRD16 = self.f["ZRD16"][:]
        self.ZRD17 = self.f["ZRD17"][:]
        self.ZRD18 = self.f["ZRD18"][:]
        self.ZRD19 = self.f["ZRD19"][:]
        self.ZRD20 = self.f["ZRD20"][:]
        self.ZRD21 = self.f["ZRD21"][:]
        self.ZRD22 = self.f["ZRD22"][:]
        self.ZRD23 = self.f["ZRD23"][:]
        self.ZRD24 = self.f["ZRD24"][:]
        self.ZRD25 = self.f["ZRD25"][:]
        self.ZRD26 = self.f["ZRD26"][:]
        self.ZRD27 = self.f["ZRD27"][:]
        self.ZRD28 = self.f["ZRD28"][:]
        self.ZRD29 = self.f["ZRD29"][:]
        self.ZRD30 = self.f["ZRD30"][:]
        self.ZRD31 = self.f["ZRD31"][:]
        self.ZRD32 = self.f["ZRD32"][:]
        self.ZRD33 = self.f["ZRD33"][:]
        self.ZRD34 = self.f["ZRD34"][:]
        self.ZRD35 = self.f["ZRD35"][:]
        self.ZRD36 = self.f["ZRD36"][:]
        self.ZRD37 = self.f["ZRD37"][:]
        self.ZRD38 = self.f["ZRD38"][:]
        self.ZRD39 = self.f["ZRD39"][:]
        self.ZRD40 = self.f["ZRD40"][:]
        self.ZRD41 = self.f["ZRD41"][:]
        self.ZRD42 = self.f["ZRD42"][:]
        self.ZRD43 = self.f["ZRD43"][:]
        self.ZRD44 = self.f["ZRD44"][:]
        self.ZRD45 = self.f["ZRD45"][:]
        self.ZRD46 = self.f["ZRD46"][:]
        self.ZRD47 = self.f["ZRD47"][:]
        self.ZRD48 = self.f["ZRD48"][:]
        self.ZRD49 = self.f["ZRD49"][:]
        self.ZRD50 = self.f["ZRD50"][:]
        self.ZRD51 = self.f["ZRD51"][:]
        self.ZRD52 = self.f["ZRD52"][:]
        self.ZRD53 = self.f["ZRD53"][:]
        self.ZRD54 = self.f["ZRD54"][:]
        self.ZRD55 = self.f["ZRD55"][:]
        self.ZRD56 = self.f["ZRD56"][:]
        self.ZRD57 = self.f["ZRD57"][:]
        self.ZRD58 = self.f["ZRD58"][:]
        self.ZRD59 = self.f["ZRD59"][:]
        self.ZRD60 = self.f["ZRD60"][:]
        self.ZRD1X = self.f["ZRD1X"][:]
        self.ZRD2X = self.f["ZRD2X"][:]
        self.ZRD3X = self.f["ZRD3X"][:]
        self.ZRD4X = self.f["ZRD4X"][:]
        self.ZRD5X = self.f["ZRD5X"][:]
        self.ZRD6X = self.f["ZRD6X"][:]
        self.ZRD7X = self.f["ZRD7X"][:]
        self.ZRD8X = self.f["ZRD8X"][:]
        self.ZRD9X = self.f["ZRD9X"][:]
        self.ZRD10X = self.f["ZRD10X"][:]
        self.ZRD11X = self.f["ZRD11X"][:]
        self.ZRD12X = self.f["ZRD12X"][:]
        self.ZRD13X = self.f["ZRD13X"][:]
        self.ZRD14X = self.f["ZRD14X"][:]
        self.ZRD15X = self.f["ZRD15X"][:]
        self.ZRD16X = self.f["ZRD16X"][:]
        self.ZRD17X = self.f["ZRD17X"][:]
        self.ZRD18X = self.f["ZRD18X"][:]
        self.ZRD19X = self.f["ZRD19X"][:]
        self.ZRD20X = self.f["ZRD20X"][:]
        self.ZRD21X = self.f["ZRD21X"][:]
        self.ZRD22X = self.f["ZRD22X"][:]
        self.ZRD23X = self.f["ZRD23X"][:]
        self.ZRD24X = self.f["ZRD24X"][:]
        self.ZRD25X = self.f["ZRD25X"][:]
        self.ZRD26X = self.f["ZRD26X"][:]
        self.ZRD27X = self.f["ZRD27X"][:]
        self.ZRD28X = self.f["ZRD28X"][:]
        self.ZRD29X = self.f["ZRD29X"][:]
        self.ZRD30X = self.f["ZRD30X"][:]
        self.ZRD31X = self.f["ZRD31X"][:]
        self.ZRD32X = self.f["ZRD32X"][:]
        self.ZRD33X = self.f["ZRD33X"][:]
        self.ZRD34X = self.f["ZRD34X"][:]
        self.ZRD35X = self.f["ZRD35X"][:]
        self.ZRD36X = self.f["ZRD36X"][:]
        self.ZRD37X = self.f["ZRD37X"][:]
        self.ZRD38X = self.f["ZRD38X"][:]
        self.ZRD39X = self.f["ZRD39X"][:]
        self.ZRD40X = self.f["ZRD40X"][:]
        self.ZRD41X = self.f["ZRD41X"][:]
        self.ZRD42X = self.f["ZRD42X"][:]
        self.ZRD43X = self.f["ZRD43X"][:]
        self.ZRD44X = self.f["ZRD44X"][:]
        self.ZRD45X = self.f["ZRD45X"][:]
        self.ZRD46X = self.f["ZRD46X"][:]
        self.ZRD47X = self.f["ZRD47X"][:]
        self.ZRD48X = self.f["ZRD48X"][:]
        self.ZRD49X = self.f["ZRD49X"][:]
        self.ZRD50X = self.f["ZRD50X"][:]
        self.ZRD51X = self.f["ZRD51X"][:]
        self.ZRD52X = self.f["ZRD52X"][:]
        self.ZRD53X = self.f["ZRD53X"][:]
        self.ZRD54X = self.f["ZRD54X"][:]
        self.ZRD55X = self.f["ZRD55X"][:]
        self.ZRD56X = self.f["ZRD56X"][:]
        self.ZRD57X = self.f["ZRD57X"][:]
        self.ZRD58X = self.f["ZRD58X"][:]
        self.ZRD59X = self.f["ZRD59X"][:]
        self.ZRD60X = self.f["ZRD60X"][:]
        self.CF1 = self.f["CF1"][:]
        self.CF2 = self.f["CF2"][:]
        self.CF3 = self.f["CF3"][:]
        self.CF4 = self.f["CF4"][:]
        self.CF5 = self.f["CF5"][:]
        self.CF6 = self.f["CF6"][:]
        self.CF7 = self.f["CF7"][:]
        self.CF8 = self.f["CF8"][:]
        self.CF9 = self.f["CF9"][:]
        self.CF10 = self.f["CF10"][:]
        self.CF11 = self.f["CF11"][:]
        self.CF12 = self.f["CF12"][:]
        self.CF13 = self.f["CF13"][:]
        self.CF14 = self.f["CF14"][:]
        self.CF15 = self.f["CF15"][:]
        self.CF16 = self.f["CF16"][:]
        self.CV1 = self.f["CV1"][:]
        self.CV2 = self.f["CV2"][:]
        self.CV3 = self.f["CV3"][:]
        self.CV4 = self.f["CV4"][:]
        self.CV5 = self.f["CV5"][:]
        self.CV6 = self.f["CV6"][:]
        self.CV7 = self.f["CV7"][:]
        self.CV8 = self.f["CV8"][:]
        self.CV9 = self.f["CV9"][:]
        self.CV10 = self.f["CV10"][:]
        self.CV11 = self.f["CV11"][:]
        self.CV12 = self.f["CV12"][:]
        self.CV13 = self.f["CV13"][:]
        self.CV14 = self.f["CV14"][:]
        self.CV15 = self.f["CV15"][:]
        self.CV16 = self.f["CV16"][:]

        ### Initialize derived and integral quantities
        
        self.QIDT   = np.zeros([self.timesteps,self.radialsize])
        self.QEDT   = np.zeros([self.timesteps,self.radialsize])
        self.QDT   = np.zeros([self.timesteps,self.radialsize])
        self.QEICRH = np.zeros([self.timesteps,self.radialsize])
        self.QIICRH = np.zeros([self.timesteps,self.radialsize])
        self.QICRH = np.zeros([self.timesteps,self.radialsize])
        self.QNBI = np.zeros([self.timesteps,self.radialsize])
        self.QECRH = np.zeros([self.timesteps,self.radialsize])
        self.QEICL = np.zeros([self.timesteps,self.radialsize])
        self.Cu_tot = np.zeros([self.timesteps,self.radialsize])
        self.CuTor_tot = np.zeros([self.timesteps,self.radialsize])
        self.Cubs_tot = np.zeros([self.timesteps,self.radialsize])
        self.QE = np.zeros([self.timesteps,self.radialsize])
        self.QI = np.zeros([self.timesteps,self.radialsize])
        self.QRAD = np.zeros([self.timesteps,self.radialsize])
        self.QOH = np.zeros([self.timesteps,self.radialsize])
        self.Wtot = np.zeros([self.timesteps,self.radialsize])
        self.ne_avg = np.zeros([self.timesteps])
        self.NIZ1_avg = np.zeros([self.timesteps])
        self.NIZ2_avg = np.zeros([self.timesteps])
        self.NIZ3_avg = np.zeros([self.timesteps])
        self.NI_avg = np.zeros([self.timesteps])
        self.ne_lineavg = np.zeros([self.timesteps])
        self.Te_avg = np.zeros([self.timesteps])
        self.Ti_avg = np.zeros([self.timesteps])
        self.tau98 = np.zeros([self.timesteps])
        self.tau89 = np.zeros([self.timesteps])
        self.tau98_lineavg = np.zeros([self.timesteps])
        self.beta = np.zeros(self.timesteps)
        self.betaN = np.zeros(self.timesteps)
        self.FP_norm = np.zeros([self.timesteps,self.radialsize])
        self.q95position = [0]*self.timesteps
        self.q95 = np.zeros(self.timesteps)
        self.delta95 = np.zeros(self.timesteps)
        self.kappa95 = np.zeros(self.timesteps)
        self.n_Angioni = np.zeros(self.timesteps)
        self.SNEBM_tot = np.zeros(self.timesteps)
        self.shear = np.zeros([self.timesteps,self.radialsize])
        self.PBRAD = np.zeros([self.timesteps,self.radialsize])
        self.PSYNC = np.zeros([self.timesteps,self.radialsize])
        self.QBRAD = np.zeros([self.timesteps,self.radialsize])
        self.QSYNC = np.zeros([self.timesteps,self.radialsize])
        self.PRWOL_PUET_dens = np.zeros([self.timesteps,self.radialsize])
        self.rlte  = np.zeros([self.timesteps,self.radialsize])
        self.rlti  = np.zeros([self.timesteps,self.radialsize])
        self.rlne  = np.zeros([self.timesteps,self.radialsize])

        ### Integrated quantities

        for kk in range(0,self.timesteps):
             # volumetric density variables
             self.QIDT[kk,:] = np.cumsum(self.PIDT[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QEICL[kk,:] = np.cumsum(self.PEICL[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QEDT[kk,:] = np.cumsum(self.PEDT[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QDT[kk,:] = np.cumsum((self.PEDT[kk,:]+self.PIDT[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QNBI[kk,:] = np.cumsum((self.PEBM[kk,:]+self.PIBM[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QECRH[kk,:] = np.cumsum(self.PEECR[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QIICRH[kk,:] = np.cumsum((self.PIICR[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QEICRH[kk,:] = np.cumsum((self.PEICR[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QICRH[kk,:] = np.cumsum((self.PIICR[kk,:]+self.PEICR[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QE[kk,:] = np.cumsum(self.PE[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QI[kk,:] = np.cumsum(self.PI[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QRAD[kk,:] = np.cumsum(self.PRAD[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QOH[kk,:] = np.cumsum(self.POH[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.SNEBM_tot[kk] = np.cumsum(self.SNEBM[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]
             # areal density variables
             self.Cu_tot[kk,:] = np.cumsum(self.Cu[kk,:]*self.area[kk,:])
             self.CuTor_tot[kk,:] = np.cumsum(self.CuTor[kk,:]*self.area[kk,:])
             self.Cubs_tot[kk,:] = np.cumsum(self.Cubs[kk,:]*self.area[kk,:])
        self.QETOT  = self.QE
        self.QITOT  = self.QI

        ### Derived quantities

        for kk in range(0,self.timesteps):
             # average values
             self.ne_avg[kk] = np.cumsum(self.ne[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]          # volume average electron density
             self.NIZ1_avg[kk] = np.cumsum(self.NIZ1[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]      # volume average 1st impurity density
             self.NIZ2_avg[kk] = np.cumsum(self.NIZ2[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]      # volume average 2nd impurity density
             self.NI_avg[kk] = np.cumsum(self.NI[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]          # volume average ion density
             self.ne_lineavg[kk] = np.cumsum(self.ne[kk,:])[-1]/len(self.ne[kk,:])                              # line average electron density
             self.Te_avg[kk] = np.cumsum(self.Te[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]          # volume average Te
             self.Ti_avg[kk] = np.cumsum(self.Ti[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]          # volume average Ti
             # derived quantities
             self.FP_norm[kk,:] = (self.FP[kk,:]-self.FP[kk,0])/(self.FP[kk,-1]-self.FP[kk,0])              # normalized poloidal flux
             self.q95position[kk] = np.abs(self.FP_norm[kk] - 0.95).argmin()                                # coordinate at 95% of poloidal normalized flux
             self.q95[kk] = 1/self.Mu[kk,self.q95position[kk]]                                              # q at 95% of poloidal normalized flux
             self.delta95[kk] = self.tria[kk,self.q95position[kk]]                                          # triangularity at 95% of poloidal normalized flux
             self.kappa95[kk] = self.elon[kk,self.q95position[kk]]                                          # elongation at 95% of poloidal normalized flux
             self.beta[kk] = 0.00402*np.cumsum((self.ne[kk,:]*self.Te[kk,:]+self.ni[kk,:]*self.Ti[kk,:]+0.5*(self.PBPER[kk,:]+self.PBLON[kk,:]))*self.VR[kk,:])[-1]/np.cumsum(self.VR[kk,:])[-1]/(self.BTOR[kk]**2)                             # plasma beta
             self.betaN[kk] = 0.402*np.cumsum((self.ne[kk,:]*self.Te[kk,:]+self.ni[kk,:]*self.Ti[kk,:]+0.5*(self.PBPER[kk,:]+self.PBLON[kk,:]))*self.VR[kk,:])[-1]/np.cumsum(self.VR[kk,:])[-1]*self.ABC[kk]/(self.BTOR[kk]*self.IPL[kk])       # normalized plasma beta
             self.Wtot[kk,:] = 0.024*np.cumsum((self.ne[kk,:]*self.Te[kk,:]+self.ni[kk,:]*self.Ti[kk,:])*self.HRO[kk]*self.VR[kk,:])          # total plasma energy
             self.tau89[kk] = 0.048*(self.AMAIN[kk,1])**0.5*(self.IPL[kk])**0.85*(self.RTOR[kk])**1.2*(self.ABC[kk])**0.3*(self.AREAT[kk,-1]/(3.1415*self.rmin[kk,-1]**2))**0.5*max(1.e-12,self.ne_lineavg[kk])**0.1*(self.BTOR[kk])**0.2*max(1.e-12,self.QDT[kk,-1]+self.QICRH[kk,-1]+self.QECRH[kk,-1]+self.QNBI[kk,-1]+self.QOH[kk,-1])**(-0.5)          # tau89
             self.tau98[kk] = 0.0562*(self.IPL[kk])**0.93*(self.BTOR[kk])**0.15*max(1.e-12,self.ne_avg[kk])**0.41*max(1.e-12,self.QE[kk,-1]+self.QI[kk,-1]+self.QRAD[kk,-1])**(-0.69)*(self.RTOR[kk])**1.97*(self.AREAT[kk,-1]/(3.1415*self.rmin[kk,-1]**2))**0.78*(self.rmin[kk,-1]/self.RTOR[kk])**0.58*(self.AMAIN[kk,1])**0.19                          # tau98
             self.tau98_lineavg[kk] = 0.0562*(self.IPL[kk])**0.93*(self.BTOR[kk])**0.15*max(1.e-12,self.ne_lineavg[kk])**0.41*max(1.e-12,self.QE[kk,-1]+self.QI[kk,-1]+self.QRAD[kk,-1])**(-0.69)*(self.RTOR[kk])**1.97*(self.AREAT[kk,-1]/(3.1415*self.rmin[kk,-1]**2))**0.78*(self.rmin[kk,-1]/self.RTOR[kk])**0.58*(self.AMAIN[kk,1])**0.19              # tau98 computed with line avg density
             self.n_Angioni[kk] = 1.347-0.117*math.log(max(1.e-12,0.2*self.ne_avg[kk]*self.RTOR[kk]*self.Te_avg[kk]**(-2)))+1.331*self.SNEBM_tot[kk]-4.03*self.beta[kk]     # Angioni density peaking scaling
             self.shear[kk,:] = -self.rmin[kk,:]/self.Mu[kk,:]*np.gradient(self.Mu[kk,:]/self.rmin[kk,:])       # magnetic shear
             self.PBRAD[kk,:] = 5.06E-5*self.ZEF[kk,:]*self.ne[kk,:]**2*self.Te[kk,:]**0.5                      # Bremmstrahlung radiation
             self.PSYNC[kk,:] = 1.32E-7*(self.Te_avg[kk]*self.BTOR[kk])**2.5*np.sqrt(self.ne_avg[kk]/self.AB[kk]*(1.+18.*self.AB[kk]/(self.RTOR[kk]*np.sqrt(self.Te_avg[kk]))))         # Synchrotron radiation
             # normalized gradients
             for jj in range(0,self.radialsize-1):
                 self.rlte[kk,jj]=-self.RTOR[-1]/(0.5*(self.Te[kk,jj]+self.Te[kk,jj+1])*(self.rmin[kk,jj+1]-self.rmin[kk,jj])/(self.Te[kk,jj+1]-self.Te[kk,jj]))
                 self.rlti[kk,jj]=-self.RTOR[-1]/(0.5*(self.Ti[kk,jj]+self.Ti[kk,jj+1])*(self.rmin[kk,jj+1]-self.rmin[kk,jj])/(self.Ti[kk,jj+1]-self.Ti[kk,jj]))
                 self.rlne[kk,jj]=-self.RTOR[-1]/(0.5*(self.ne[kk,jj]+self.ne[kk,jj+1])*(self.rmin[kk,jj+1]-self.rmin[kk,jj])/(self.ne[kk,jj+1]-self.ne[kk,jj]))
             self.rlte[kk,jj+1]=self.rlte[kk,jj]        # normalized logaritmic Te gradient
             self.rlti[kk,jj+1]=self.rlti[kk,jj]        # normalized logaritmic Ti gradient
             self.rlne[kk,jj+1]=self.rlne[kk,jj]        # normalized logaritmic ne gradient
             #### --------------- Calculation of W radiation by Puetterich formula ------------------------- ###
             for jj in range(0,self.radialsize):
                 T = self.Te[kk,jj]*1000.
                 Z = np.log10(self.Te[kk,jj])
                 if T <= 25.25:
                     self.PRWOL_PUET_dens[kk,jj] = 20.*self.ne[kk,jj]
                 elif T > 25.25 and T <= 300.:
                     self.PRWOL_PUET_dens[kk,jj] = (-(150.984*Z**4 + 566.56*Z**3 + 729.562*Z**2 + 377.649*Z + 47.922))*self.ne[kk,jj]
                 elif T > 300. and T <= 3350.:
                     self.PRWOL_PUET_dens[kk,jj] = (-119.946*Z**3 - 82.821*Z**2 + 32.707*Z + 42.603)*self.ne[kk,jj]
                 elif T > 3350.:
                     self.PRWOL_PUET_dens[kk,jj] = (4.7 + 14.484*np.exp(-3.4196*(Z - 0.602)**2))*self.ne[kk,jj]
             self.QBRAD[kk,:] = np.cumsum(self.PBRAD[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QSYNC[kk,:] = np.cumsum(self.PSYNC[kk,:]*self.HRO[kk]*self.VR[kk,:])
        
        self.CuOhm = self.CC*self.ULON/(self.RTOR[-1]*2*np.pi)      # Ohmic current density
        self.n_Gr = self.IPL/(np.pi*self.ABC**2)                # Greenwald density
        self.f_Gr = self.ne_avg/10/self.n_Gr                    # Greenwald fraction
        self.tauE = self.Wtot/(self.QRAD+self.QE+self.QI)       # energy confinement time
        self.H98 = self.tauE[:,-1]/self.tau98                   # H98
        self.H89 = self.tauE[:,-1]/self.tau89                   # H89
        self.H98_lineavg = self.tauE[:,-1]/self.tau98_lineavg   # H98 with line average density
        self.fMAIN = self.NMAIN/self.ne                         # main ion concentration
        self.f1 = self.NIZ1/self.ne                             # 1st impurity concentration
        self.f2 = self.NIZ2/self.ne                             # 2nd impurity concentration
        self.f3 = self.NIZ3/self.ne                             # 3rd impurity concentration
        self.ptot  = (self.ne*self.Te+self.ni*self.Ti+0.5*(self.PBPER+self.PBLON))*1.6e-3  # total pressure, in MPa
        self.quasi = (self.f['NE'][:]-self.f['NMAIN'][:]*self.f['ZMAIN'][:]-self.f['NIZ1'][:]*self.f['ZIM1'][:]-self.f['NIZ2'][:]*self.f['ZIM2'][:]-self.f['NIZ3'][:]*self.f['ZIM3'][:])/self.f['NE'][:]        # check if QN is fulfilled (low value expected)
        #  some global and performance parameters
        self.Pfus = self.QDT/0.2     # fusion power (in the D+T fusion reactions 20% goes to He and 80% to neutrons)
        self.Q = self.Pfus[:,-1]/(self.QICRH[:,-1]+self.QOH[:,-1])      # fusion gain
        self.q_Uckan = 5*self.ABC**2*self.BTOR/(self.RTOR*self.IPL)*(1+self.kappa95**2*(1+2*self.delta95**2-1.2*self.delta95**3))/2
        self.ne_PLHmin = 0.07*(self.IPL)**0.34*(self.BTOR)**0.62*(self.RTOR)**(-0.95)*(self.RTOR/self.ABC)**0.4      # LH transition minimum density
        self.ne_PLHmin_perc = self.ne_avg/10/self.ne_PLHmin        # percentage of LH transition minimum density (>1 to use Martin scaling)
        # Martin scaling
        self.PLH = 0.0488*(self.ne_avg/10.)**0.717*(self.BTOR)**0.803*(self.SLAT[:,-1])**0.941*(2/self.AMAIN[:,-1])                             # LH power threshold
        self.PLH_lower = 0.0488*math.exp(-0.057)*(self.ne_avg/10.)**0.682*(self.BTOR)**0.771*(self.SLAT[:,-1])**0.922*(2/self.AMAIN[:,-1])      # lower error bar of LH power threshold
        self.PLH_upper = 0.0488*math.exp(0.057)*(self.ne_avg/10.)**0.752*(self.BTOR)**0.835*(self.SLAT[:,-1])**0.96*(2/self.AMAIN[:,-1])        # upper error bar of LH power threshold
        self.PLH_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH                     # Psep/PLH
        self.PLH_lower_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH_lower                                                             # lower error bar of LH power threshold
        self.PLH_upper_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH_upper                                                             # upper error bar of LH power threshold
        self.fLH_Martin = self.PLH_perc
        # "Metal wall" scaling
        self.PLH_metal = 0.044*(self.ne_avg/10.)**1.06*(self.BTOR)**0.54*(self.SLAT[:,-1])*(2/self.AMAIN[:,-1])**(0.965)               # LH power threshold
        self.PLH_metal_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH_metal         # Psep/PLH
        self.fLH_metal = self.PLH_metal_perc
        # Schmidtmayr scaling
        self.PLH_schmidtmayr = 0.0325*(self.ne_avg/10.)**1.05*(self.BTOR)**0.68*(self.SLAT[:,-1])**0.93*(2/self.AMAIN[:,-1])            # LH power threshold
        self.PLH_schmidt_perc = (self.QI[:,-1])/self.PLH_schmidtmayr               # Psep/PLH
        self.fLH_Schmidt = self.PLH_schmidt_perc
        self.a = self.rmin[:, -1]                   # LCFS minor radius at OMP
        rtor_matrix = np.zeros(self.rho.shape)
        for i in range(rtor_matrix.shape[1]):
            rtor_matrix[:, i] = self.RTOR[:]
        self.rmaj_LFx = (
            rtor_matrix + self.shif + self.rmin
        )                                           # major radius on the low field side

    def calcProfiles(self):
        self.getProfiles()

        # calculate rho_tor
        rho_tor = []
        for t in range(self.rho.shape[0]):
            rho_tor.append(self.rho[t, :] / self.ROC[t])
        self.rho_tor = np.array(rho_tor)

        # calculate safety factor profile
        self.q = 1 / self.Mu

    def time_index(self, time):
        return np.argmin(np.abs(self.t - time))

    def build_notebook(self, time_aims):
        self.getProfiles()
        # time_index  = time_index(time)
        name = "ASTRA CDF Viewer"
        self.fn = FigureNotebook(name)

        self.plot_temp(time_aims=time_aims)
        self.plot_gradients(time_aims=time_aims)
        self.plot_density(time_aims=time_aims)
        self.plot_powers_t(time_aims=time_aims)
        self.plot_powers_r(time_aims=time_aims)
        self.plot_chi_e(time_aims=time_aims)
        self.plot_chi_i(time_aims=time_aims)
        self.plot_heat_fluxes(time_aims=time_aims)
        # self.plot_flux_matching(time_aims = time_aims)
        self.plot_pulse()

        self.fn.show()
        # sys.exit(self.fn.app.exec_())
        # self.fn.deleteGui()

    def get_rho_tor_indices(self, rho_tor_aims):
        """
        Output: Array w/ shape (number of times, number of rho values of interest)
        """
        self.calcProfiles()

        self.rho_tor_aims = rho_tor_aims
        self.i_rho_tor_aims = []

        for t in range(self.timesteps):
            rho_t = np.array(self.rho[t, :])
            ROC_t = self.ROC[t]
            rho_tor_t = rho_t / ROC_t
            i_rho_tor = []
            for rho_tor in rho_tor_aims:
                i_rho_tor.append(np.argmin(np.abs(rho_tor_t - rho_tor)))
            self.i_rho_tor_aims.append(i_rho_tor)

        self.i_rho_tor_aims = np.array(self.i_rho_tor_aims)

    def get_time_indices(self, time_aims):
        self.i_time_aims = []
        for time_aim in time_aims:
            self.i_time_aims.append(np.argmin(np.abs(self.t - time_aim)))

        self.i_time_aims = np.array(self.i_time_aims)

    def get_transport(self):
        self.aLTe = gradNorm(self, self.Te)
        self.aLTi = gradNorm(self, self.Ti)
        self.aLne = gradNorm(self, self.ne)

    def make_temporal_plots(self, axis, param, rho_tor_aims, linestyle="solid"):
        """
        Inputs: self, matplotlib axis, parameter array for data (e.g. Te (time x radial)), radii of interest [list]
        """
        self.get_rho_tor_indices(rho_tor_aims)

        size = 12
        plt.rc("axes", labelsize=size)
        plt.rc("xtick", labelsize=size)
        plt.rc("ytick", labelsize=size)

        for i in range(self.i_rho_tor_aims.shape[1]):
            param_list = []

            for t in range(self.timesteps):
                i_rho_tor = self.i_rho_tor_aims[t, i]
                param_list.append(param[t, i_rho_tor])

            axis.plot(
                self.t,
                param[:, i_rho_tor],
                label=self.rho_tor_aims[i],
                linestyle=linestyle,
            )

        axis.set_xlabel("Time")

    def make_radial_plots(self, axis, param, time_aims, linestyle="solid"):
        """
        Inputs: self, matplotlib axis, parameter array for data (e.g. Te (time x radial)), times of interest [list]
        """
        self.get_time_indices(time_aims)

        size = 12
        plt.rc("axes", labelsize=size)
        plt.rc("xtick", labelsize=size)
        plt.rc("ytick", labelsize=size)

        param_list = []

        for i in range(self.i_time_aims.shape[0]):
            i_time = self.i_time_aims[i]
            axis.plot(
                self.rho_tor[i_time, :],
                param[i_time, :],
                label=time_aims[i],
                linestyle=linestyle,
            )

        axis.set_xlabel(r"$\rho_{tor}$")

    def plot_temp(self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        fig = self.fn.add_figure(label="Temp Profiles")

        ## Make temporal figures ##
        self.axTet = fig.add_subplot(2, 2, 1)
        self.make_temporal_plots(self.axTet, self.Te, rho_tor_aims)
        self.axTet.set_xlabel("Time (s)")
        self.axTet.set_ylabel("$T_e$ (keV)")

        self.axTit = fig.add_subplot(2, 2, 3)
        self.make_temporal_plots(self.axTit, self.Ti, rho_tor_aims)
        self.axTit.set_xlabel("Time (s)")
        self.axTit.set_ylabel("$T_i$ (keV)")

        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        ## Make radial figures ##

        self.axTer = fig.add_subplot(2, 2, 2)
        self.make_radial_plots(self.axTer, self.Te, time_aims)
        self.axTer.set_ylabel("$T_e$ (keV)")

        self.axTir = fig.add_subplot(2, 2, 4)
        self.make_radial_plots(self.axTir, self.Ti, time_aims)
        self.axTir.set_ylabel("$T_i$ (keV)")

        plt.legend(title="Times") #, bbox_to_anchor=(1, 1))

        fig.tight_layout()
    
    def plot_gradients(
        self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ):
        fig = self.fn.add_figure(label="Temp Gradient Profiles")
        self.get_transport()

        ## Make temporal figures ##
        self.axaLTet = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axaLTet, self.aLTe, rho_tor_aims)
        self.axaLTet.set_ylabel("$a\\nabla T_e/T_e$")
        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        self.axaLTit = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(self.axaLTit, self.aLTi, rho_tor_aims)
        self.axaLTit.set_ylabel("$a\\nabla T_i/T_i$")
        #plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        self.axaLnet = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(self.axaLnet, self.aLne, rho_tor_aims)
        self.axaLnet.set_ylabel("$a\\nabla n_e/n_e$")
        #plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        ##Make radial figures ##
        self.axaLTer = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axaLTer, self.aLTe, time_aims)
        self.axaLTer.set_ylabel("$a\\nabla T_e/T_e$")
        plt.legend(title="Times") #, bbox_to_anchor=(1, 1))

        self.axaLTir = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(self.axaLTir, self.aLTi, time_aims)
        self.axaLTir.set_ylabel("$a\\nabla T_i/T_i$")
        #plt.legend(title="Times") #, bbox_to_anchor=(1, 1))

        self.axaLner = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axaLner, self.aLne, time_aims)
        self.axaLner.set_ylabel("$a\\nabla n_e/n_e$")
        #plt.legend(title="Times") #, bbox_to_anchor=(1, 1))

        #fig.tight_layout()

    def plot_density(self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        fig = self.fn.add_figure(label="Density Profiles")

        # Make temporal figures
        self.axnet = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axnet, self.ne, rho_tor_aims)
        self.axnet.set_ylabel("Density [$10^{19}/m^3$]")
        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        self.axCut = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(self.axCut, self.Cu, rho_tor_aims)
        self.axCut.set_ylabel("J [$MA/m^2$]")

        self.axqt = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(self.axqt, self.q, rho_tor_aims)
        self.axqt.set_ylabel("q")

        # Make radial figures

        self.axner = fig.add_subplot(2, 3, 4)
        self.axner.set_ylabel("Density ($10^{19}/m^3$)")
        self.make_radial_plots(self.axner, self.ne, time_aims)

        self.axCur = fig.add_subplot(2, 3, 5)
        self.axCur.set_ylabel("J [$MA/m^2$]")
        self.make_radial_plots(self.axCur, self.Cu, time_aims)
        plt.legend(title="Times") #, bbox_to_anchor=(1, 1))

        self.axqr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axqr, self.q, time_aims)
        self.axqr.set_ylabel("q")

        #fig.tight_layout()

    def plot_powers_t(
        self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ):
        # PE = PEBM + POH + PEECR - PRAD - PEICL
        # PI = PIBM + PEICL
        fig = self.fn.add_figure(label="Power(t)")

        # Make temporal figures
        self.axPEt = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axPEt, self.PE, rho_tor_aims)
        self.axPEt.set_ylabel("$P_E$ ($MW/m^3$)")
        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        self.axPIt = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(self.axPIt, self.PI, rho_tor_aims)
        self.axPIt.set_ylabel("$P_I$ ($MW/m^3$)")

        self.axPBMt = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(self.axPBMt, self.PEBM + self.PIBM, rho_tor_aims)
        self.axPBMt.set_ylabel("Total NBI ($MW/m^3$)")

        self.axPECRt = fig.add_subplot(2, 3, 4)
        self.make_temporal_plots(self.axPECRt, self.PEECR, rho_tor_aims)
        self.axPECRt.set_ylabel("Total ECH ($MW/m^3$)")

        self.axPRADt = fig.add_subplot(2, 3, 5)
        self.make_temporal_plots(self.axPRADt, self.PRAD, rho_tor_aims)
        self.axPRADt.set_ylabel("$P_{RAD}$ ($MW/m^3$)")

        self.axPFUSt = fig.add_subplot(2, 3, 6)
        self.make_temporal_plots(self.axPFUSt, (self.PEDT+self.PIDT)*5, rho_tor_aims)
        self.axPFUSt.set_ylabel("$P_{FUS}$ ($MW/m^3$)")

        #fig.tight_layout()

    def plot_powers_r(
        self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ):
        # PE = PEBM + POH + PEECR - PRAD - PEICL
        fig = self.fn.add_figure(label="Power(r)")

        # Make temporal figures
        self.axPEr = fig.add_subplot(2, 3, 1)
        self.make_radial_plots(self.axPEr, self.PE, time_aims)
        self.axPEr.set_ylabel("$P_E$ ($MW/m^3$)")
        plt.legend(title="Times [s]") #, bbox_to_anchor=(1, 1))

        self.axPIr = fig.add_subplot(2, 3, 2)
        self.make_radial_plots(self.axPIr, self.PI, time_aims)
        self.axPIr.set_ylabel("$P_I$ ($MW/m^3$)")

        self.axPBMr = fig.add_subplot(2, 3, 3)
        self.make_radial_plots(self.axPBMr, self.PEBM + self.PIBM, time_aims)
        self.axPBMr.set_ylabel("Total NBI ($MW/m^3$)")

        self.axPECRr = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axPECRr, self.PEECR, time_aims)
        self.axPECRr.set_ylabel("Total ECH ($MW/m^3$)")

        self.axPRADr = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(self.axPRADr, self.PRAD, time_aims)
        self.axPRADr.set_ylabel("$P_{rad}$ ($MW/m^3$)")

        self.axPFUSr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axPFUSr, (self.PEDT+self.PIDT)*5, time_aims)
        self.axPFUSr.set_ylabel("$P_{fus}$ ($MW/m^3$)")

        #fig.tight_layout()

    def plot_chi_e(self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        fig = self.fn.add_figure(label="Chi_e")

        # Make temporal figures
        self.axchi_et = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axchi_et, self.chi_e_TGLF, rho_tor_aims)
        self.axchi_et.set_ylabel("TGLF ($m^2/s$)")

        self.axchi_e_smoothedt = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(
            self.axchi_e_smoothedt, self.chi_e_TGLF_smoothed, rho_tor_aims
        )
        self.axchi_e_smoothedt.set_ylabel("Smoothed ($m^2/s$)")

        self.axHEt = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(self.axHEt, self.HE, rho_tor_aims)
        self.axHEt.set_ylabel("ASTRA ($m^2/s$)")
        plt.legend(title="Times [s]") #, bbox_to_anchor=(1, 1))

        # Make radial figures
        self.axchi_er = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axchi_er, self.chi_e_TGLF, time_aims)
        self.axchi_er.set_ylabel("TGLF ($m^2/s$)")
        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        self.axchi_e_smoothedr = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(
            self.axchi_e_smoothedr, self.chi_e_TGLF_smoothed, time_aims
        )
        self.axchi_e_smoothedr.set_ylabel("Smoothed ($m^2/s$)")

        self.axHEr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axHEr, self.HE, time_aims)
        self.axHEr.set_ylabel("ASTRA ($m^2/s$)")

        #fig.tight_layout()

    def plot_chi_i(self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        fig = self.fn.add_figure(label="Chi_i")

        # Make temporal figures
        self.axchi_it = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axchi_it, self.chi_i_TGLF, rho_tor_aims)
        self.axchi_it.set_ylabel("TGLF ($m^2/s$)")

        self.axchi_i_smoothedt = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(
            self.axchi_i_smoothedt, self.chi_i_TGLF_smoothed, rho_tor_aims
        )
        self.axchi_i_smoothedt.set_ylabel("Smoothed ($m^2/s$)")

        self.axXIt = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(self.axXIt, self.XI, rho_tor_aims)
        self.axXIt.set_ylabel("ASTRA ($m^2/s$)")
        plt.legend(title="Times [s]") #, bbox_to_anchor=(1, 1))

        # Make radial figures
        self.axchi_ir = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axchi_ir, self.chi_i_TGLF, time_aims)
        self.axchi_ir.set_ylabel("TGLF ($m^2/s$)")
        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        self.axchi_i_smoothedr = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(
            self.axchi_i_smoothedr, self.chi_i_TGLF_smoothed, time_aims
        )
        self.axchi_i_smoothedr.set_ylabel("Smoothed ($m^2/s$)")

        self.axXIr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axXIr, self.XI, time_aims)
        self.axXIr.set_ylabel("ASTRA ($m^2/s$)")

        #fig.tight_layout()

    def plot_heat_fluxes(
        self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ):
        fig = self.fn.add_figure(label="Heat Fluxes")

        # Make temporal figures
        self.axQet = fig.add_subplot(2, 2, 1)
        self.make_temporal_plots(self.axQet, self.Qe, rho_tor_aims)
        self.axQet.set_ylabel("Qe (MW)")

        self.axQit = fig.add_subplot(2, 2, 2)
        self.make_temporal_plots(self.axQit, self.Qi, rho_tor_aims)
        self.axQit.set_ylabel("Qi (MW)")

        plt.legend(title=r"$\rho_{tor}$",loc='upper left') #, bbox_to_anchor=(1, 1))

        # Make radial figures

        self.axQer = fig.add_subplot(2, 2, 3)
        self.axQer.set_ylabel("Qe (MW)")
        self.make_radial_plots(self.axQer, self.Qe, time_aims)

        self.axQir = fig.add_subplot(2, 2, 4)
        self.axQir.set_ylabel("Qi (MW)")
        self.make_radial_plots(self.axQir, self.Qi, time_aims)

        plt.legend(title="Times") #, bbox_to_anchor=(1, 1))

        #fig.tight_layout()

    def plot_flux_matching(
        self, time_aims, rho_tor_aims=[0.3, 0.5, 0.7], qe_lim=[0, 5], qi_lim=[0, 5], qn_lim=[0, 5]
    ):
        last_time = [time_aims[-1]]

        fig = self.fn.add_figure(label="Flux Matching")

        self.get_rho_tor_indices(rho_tor_aims)
        self.get_time_indices(time_aims)

        ## Make temporal figures ##

        self.axQet = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axQet, self.Qe, rho_tor_aims, linestyle="dashed")
        # self.make_temporal_plots(self.axQet, self.QETOT, rho_tor_aims)
        self.axQet.set_ylabel("Qe")
        # self.axQet.set_ylim(qe_lim)
        #plt.legend(["Qe", "Qetot"], title=r"$\rho_{tor}$ = " + str(rho_tor_aims[0])) #, bbox_to_anchor=(1, 1))
        plt.legend(title=r"$\rho_{tor}$",loc='upper left')

        self.axQit = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(self.axQit, self.Qi, rho_tor_aims, linestyle="dashed")
        # self.make_temporal_plots(self.axQit, self.QITOT, rho_tor_aims)
        self.axQit.set_ylabel("Qi")
        # self.axQit.set_ylim(qi_lim)
        #plt.legend(["Qi", "Qitot"], title=r"$\rho_{tor}$ = " + str(rho_tor_aims[0])) #, bbox_to_anchor=(1, 1))
        plt.legend(title=r"$\rho_{tor}$",loc='upper left')

        self.axQnt = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(
            self.axQnt, self.Qn / self.G11, rho_tor_aims, linestyle="dashed"
        )
        # self.make_temporal_plots(self.axQnt, self.QNTOT / self.G11, rho_tor_aims)
        self.axQnt.set_ylabel("Qn/volume")
        # self.axQnt.set_ylim(qn_lim)
        #plt.legend(
        #    ["Qn/volume", "Qntot/volume"],
        #    title=r"$\rho_{tor}$ = " + str(rho_tor_aims[0]), bbox_to_anchor=(1, 1),
        #)
        plt.legend(title=r"$\rho_{tor}$",loc='upper left')
        self.axQnt.axhline(y=0.,linestyle='-.',c='k')

        ## Make radial figures ##

        self.axQer = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axQer, self.Qe, last_time, linestyle="dashed")
        # self.make_radial_plots(self.axQer, self.QETOT, last_time)
        self.axQer.set_xlim([0, self.boundary])
        self.axQer.set_ylim(qe_lim)
        self.axQer.set_ylabel("Qe")
        plt.legend(["Qe", "Qetot"], title="Time = " + str(last_time[0])) #, bbox_to_anchor=(1, 1))

        self.axQir = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(self.axQir, self.Qi, last_time, linestyle="dashed")
        # self.make_radial_plots(self.axQir, self.QITOT, last_time)
        self.axQir.set_xlim([0, self.boundary])
        self.axQir.set_ylim(qi_lim)
        self.axQir.set_ylabel("Qi")
        plt.legend(["Qi", "Qitot"], title="Time = " + str(last_time[0])) #, bbox_to_anchor=(1, 1))

        self.axQnr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axQnr, self.Qn, last_time, linestyle="dashed")
        # self.make_radial_plots(self.axQnr, self.QNTOT, last_time)
        self.axQnr.set_xlim([0, self.boundary])
        self.axQnr.set_ylim(qn_lim)
        self.axQnr.set_ylabel("Qn")
        plt.legend(["Qn/volume", "Qntot/volume"], title="Time = " + str(last_time[0])) #, bbox_to_anchor=(1, 1))

        #fig.tight_layout()

    def plot_pulse(
        self,
        time_aims=[10.20, 10.201, 10.2015, 10.202, 10.210, 10.212],
        rho_tor_aims=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
    ):
        fig = self.fn.add_figure(label="Pulse")

        ## plot performance

        '''
        self.axQ = fig.add_subplot(2, 3, 1)
        self.axQ.plot(self.t, self.Pfus[:,-1],label="$P_{fus}$ (MW)")
        self.axQ.plot(self.t, self.Q,label="Q")
        self.axQ.set_yscale('log')
        self.axQ.set_ylabel("performance parameters")
        self.axQ.set_xlabel("time (s)")
        plt.legend()
        '''

        ## plot EPED stuff

        self.axEPED = fig.add_subplot(2, 3, 1)
        self.axEPED.plot(self.t, self.ZRD50/10,label="$n_{e,top}$ ($10^{20}m^{-3}$)")
        self.axEPED.plot(self.t, self.ZRD49/1.e3,label="$p_{top}$ (MPa)")
        self.axEPED.set_ylabel("EPED values")
        self.axEPED.set_xlabel("time (s)")
        plt.legend()

        ## plot confinement

        self.axtau = fig.add_subplot(2, 3, 3)
        self.axtau.plot(self.t, self.tauE[:,-1],label="$\\tau_{e}$ (s)")
        self.axtau.plot(self.t, self.H98,label="H98")
        self.axtau.set_ylabel("performance parameters")
        self.axtau.set_xlabel("time (s)")
        plt.legend()

        ## plot shaping and q values

        self.axq = fig.add_subplot(2, 3, 2)
        self.axq.plot(self.t, self.q95/self.q95[0], label='$q_{95}$ normalized')
        self.axq.plot(self.t, self.q_onaxis/self.q_onaxis[0], label='q0 normalized')
        self.axq.plot(self.t, self.kappa95, label='$k_{95}$')
        self.axq.plot(self.t, self.delta95, label='$\\delta_{95}$')
        self.axq.plot(self.t, self.trian, label='$\\delta_{sep}$')
        self.axq.plot(self.t, self.elong, label='$k_{sep}$')
        self.axq.set_ylabel("shaping and safety factor")
        self.axq.set_xlabel("time (s)")
        plt.legend()

       ## plot beta and averaged kinetic profiles

        self.axglob = fig.add_subplot(2, 3, 4)
        self.axglob.plot(self.t, self.betaN/self.betaN[0],label="$\\beta_N$ normalized")
        self.axglob.plot(self.t, self.ne_avg/self.ne_avg[0],label="$n_{e,avg}$ normalized")
        self.axglob.plot(self.t, self.Te_avg/self.Te_avg[0],label="$T_{e,avg}$ normalized")
        self.axglob.plot(self.t, self.Ti_avg/self.Ti_avg[0],label="$T_{i,avg}$ normalized")
        self.axglob.set_ylabel("global parameters")
        self.axglob.set_xlabel("time (s)")
        #self.axglob.set_yscale("log")
        plt.legend()

        ## plot Hmode parameters
        
        self.axPLH = fig.add_subplot(2, 3, 5)
        self.axPLH.plot(self.t, self.PLH_perc,label="Martin")
        self.axPLH.plot(self.t, self.PLH_schmidt_perc,label="Schmidtmayr")
        self.axPLH.set_ylabel("$P_{sep}/P_{LH}$")
        self.axPLH.set_xlabel("time (s)")
        plt.legend()

        ## plot total powers
        
        self.axP = fig.add_subplot(2, 3, 6)
        self.axP.plot(self.t, self.QDT[:,-1]*5,label="fusion")
        self.axP.plot(self.t, self.QICRH[:,-1],label="ICRH")
        self.axP.plot(self.t, self.QECRH[:,-1],label="ECRH")
        self.axP.plot(self.t, self.QNBI[:,-1],label="NBI")
        self.axP.plot(self.t, self.QRAD[:,-1],label="radiation")
        self.axP.plot(self.t, self.QOH[:,-1],label="ohmic")
        self.axP.plot(self.t, self.QETOT[:,-1],label="electron total")
        self.axP.plot(self.t, self.QITOT[:,-1],label="ion total")
        self.axP.set_ylabel("P (MW)")
        self.axP.set_xlabel("time (s)")
        plt.legend()

        #fig.tight_layout()

    def plot_2_pulses(
        self,second_pulse,
        time_aims=[10.20, 10.201, 10.2015, 10.202, 10.210, 10.212],
        rho_tor_aims=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
    ):
        self.getProfiles()
        second_pulse.getProfiles()
        # time_index  = time_index(time)
        name = "ASTRA CDF Viewer"
        self.fn = FigureNotebook(name,vertical=False)
        fig = self.fn.add_figure(label="Solid = first pulse, dashed = second pulse")

        ## plot performance

        '''
        self.axQ = fig.add_subplot(2, 3, 1)
        self.axQ.plot(self.t, self.Pfus[:,-1],label="$P_{fus}$ (MW)")
        self.axQ.plot(self.t, self.Q,label="Q")
        self.axQ.set_yscale('log')
        self.axQ.set_ylabel("performance parameters")
        self.axQ.set_xlabel("time (s)")
        plt.legend()
        '''

        ## plot EPED stuff

        self.axEPED = fig.add_subplot(2, 3, 1)
        self.axEPED.plot(self.t, self.ZRD50/10,label="$n_{e,top}$ ($10^{20}m^{-3}$)",c='b')
        self.axEPED.plot(self.t, self.ZRD49/1.e3,label="$p_{top}$ (MPa)",c='r')
        self.axEPED.plot(second_pulse.t, second_pulse.ZRD50/10,c='b',linestyle='--')
        self.axEPED.plot(second_pulse.t, second_pulse.ZRD49/1.e3,c='r',linestyle='--')
        self.axEPED.set_ylabel("EPED values")
        self.axEPED.set_xlabel("time (s)")
        plt.legend()

        ## plot confinement

        self.axtau = fig.add_subplot(2, 3, 3)
        self.axtau.plot(self.t, self.tauE[:,-1],label="$\\tau_{e}$ (s)",c='b')
        self.axtau.plot(self.t, self.H98,label="H98",c='r')
        self.axtau.plot(second_pulse.t, second_pulse.tauE[:,-1],linestyle='--')
        self.axtau.plot(second_pulse.t, second_pulse.H98,c='r',linestyle='--')
        self.axtau.set_ylabel("performance parameters")
        self.axtau.set_xlabel("time (s)")
        self.axtau.axhline(y=1.0, ls='-.',c='k')
        self.axtau.set_ylim(bottom=0)
        plt.legend()

        ## plot shaping and q values

        self.axq = fig.add_subplot(2, 3, 2)
        self.axq.plot(self.t, self.kappa95, label='$k_{95}$',c='b')
        self.axq.plot(self.t, self.delta95, label='$\\delta_{95}$',c='r')
        self.axq.plot(self.t, self.trian, label='$\\delta_{sep}$',c='g')
        self.axq.plot(self.t, self.elong, label='$k_{sep}$',c='k')
        self.axq.plot(self.t, self.q95, label='$q_{95}$',c='y')
        self.axq.plot(self.t, self.q_onaxis, label='q0',c='orange')
        self.axq.plot(second_pulse.t, second_pulse.kappa95,c='b',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.delta95,c='r',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.trian,c='g',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.elong,c='k',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.q95,c='y',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.q_onaxis,c='orange',linestyle='--')
        self.axq.set_ylabel("shaping and safety factor")
        self.axq.set_xlabel("time (s)")
        plt.legend()

       ## plot beta and averaged kinetic profiles

        self.axglob = fig.add_subplot(2, 3, 4)
        self.axglob.plot(self.t, self.betaN,label="$\\beta_N$",c='b')
        self.axglob.plot(self.t, self.ne_avg,label="$n_{e,avg}$",c='k')
        self.axglob.plot(self.t, self.Te_avg,label="$T_{e,avg}$",c='y')
        self.axglob.plot(self.t, self.Ti_avg,label="$T_{i,avg}$",c='orange')
        self.axglob.plot(self.t, self.ne[:,int(0.2*self.na1[-1])]/self.ne_avg,label="$\\nu_{n_e}$",c='purple')
        self.axglob.plot(second_pulse.t, second_pulse.betaN,c='b',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.ne_avg,c='k',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.Te_avg,c='y',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.Ti_avg,c='orange',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.ne[:,int(0.2*second_pulse.na1[-1])]/second_pulse.ne_avg,c='purple',linestyle='--')
        self.axglob.set_ylabel("global parameters")
        self.axglob.set_xlabel("time (s)")
        #self.axglob.set_yscale('log')
        plt.legend()

        ## plot Hmode parameters
        
        self.axPLH = fig.add_subplot(2, 3, 5)
        self.axPLH.plot(self.t, self.PLH_perc,label="Martin",c='b')
        self.axPLH.plot(self.t, self.PLH_schmidt_perc,label="Schmidtmayr",c='r')
        self.axPLH.plot(second_pulse.t, second_pulse.PLH_perc,c='b',linestyle='--')
        self.axPLH.plot(second_pulse.t, second_pulse.PLH_schmidt_perc,c='r',linestyle='--')
        self.axPLH.set_ylabel("$P_{sep}/P_{LH}$")
        self.axPLH.set_xlabel("time (s)")
        self.axPLH.axhline(y=1.0, ls='-.',c='k')
        self.axPLH.set_ylim(bottom=0)
        plt.legend()

        ## plot total powers
        
        self.axP = fig.add_subplot(2, 3, 6)
        self.axP.plot(self.t, self.QDT[:,-1]*5,label="fusion",c='b')
        self.axP.plot(self.t, self.QICRH[:,-1],label="ICRH",c='r')
        self.axP.plot(self.t, self.QECRH[:,-1],label="ECRH",c='g')
        self.axP.plot(self.t, self.QNBI[:,-1],label="NBI",c='k')
        self.axP.plot(self.t, self.QRAD[:,-1],label="radiation",c='y')
        self.axP.plot(self.t, self.QOH[:,-1],label="ohmic",c='orange')
        self.axP.plot(self.t, self.QETOT[:,-1],label="electron total",c='purple')
        self.axP.plot(self.t, self.QITOT[:,-1],label="ion total",c='cyan')
        self.axP.plot(second_pulse.t, second_pulse.QDT[:,-1]*5,c='b',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QICRH[:,-1],c='r',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QECRH[:,-1],c='g',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QNBI[:,-1],c='k',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QRAD[:,-1],c='y',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QOH[:,-1],c='orange',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QETOT[:,-1],c='purple',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QITOT[:,-1],c='cyan',linestyle='--')
        self.axP.set_ylabel("P (MW)")
        self.axP.set_xlabel("time (s)")
        self.axP.axhline(y=500, ls='-.',c='k')
        self.axP.axhline(y=1000, ls='-.',c='k')
        #self.axP.set_yscale('log')
        plt.legend()

        GRAPHICStools.addDenseAxis(self.axP)
        GRAPHICStools.addDenseAxis(self.axtau)
        GRAPHICStools.addDenseAxis(self.axglob)
        GRAPHICStools.addDenseAxis(self.axPLH)
        GRAPHICStools.addDenseAxis(self.axq)
        GRAPHICStools.addDenseAxis(self.axEPED)

        GRAPHICStools.addLegendApart(self.axP)
        GRAPHICStools.addLegendApart(self.axtau)
        GRAPHICStools.addLegendApart(self.axglob)
        GRAPHICStools.addLegendApart(self.axPLH)
        GRAPHICStools.addLegendApart(self.axq)
        GRAPHICStools.addLegendApart(self.axEPED)

    def plot_3_pulses(
        self,second_pulse,third_pulse,
        time_aims=[10.20, 10.201, 10.2015, 10.202, 10.210, 10.212],
        rho_tor_aims=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
    ):
        self.getProfiles()
        second_pulse.getProfiles()
        third_pulse.getProfiles()
        # time_index  = time_index(time)
        name = "ASTRA CDF Viewer"
        self.fn = FigureNotebook(name,vertical=False)
        fig = self.fn.add_figure(label="Solid = first pulse, dashed = second pulse, dots= third pulse")

        ## plot performance

        '''
        self.axQ = fig.add_subplot(2, 3, 1)
        self.axQ.plot(self.t, self.Pfus[:,-1],label="$P_{fus}$ (MW)")
        self.axQ.plot(self.t, self.Q,label="Q")
        self.axQ.set_yscale('log')
        self.axQ.set_ylabel("performance parameters")
        self.axQ.set_xlabel("time (s)")
        plt.legend()
        '''

        ## plot EPED stuff

        self.axEPED = fig.add_subplot(2, 3, 1)
        self.axEPED.plot(self.t, self.ZRD50/10,label="$n_{e,top}$ ($10^{20}m^{-3}$)",c='b')
        self.axEPED.plot(self.t, self.ZRD49/1.e3,label="$p_{top}$ (MPa)",c='r')
        self.axEPED.plot(second_pulse.t, second_pulse.ZRD50/10,c='b',linestyle='--')
        self.axEPED.plot(second_pulse.t, second_pulse.ZRD49/1.e3,c='r',linestyle='--')
        self.axEPED.plot(third_pulse.t, third_pulse.ZRD50/10,c='b',linestyle='-.')
        self.axEPED.plot(third_pulse.t, third_pulse.ZRD49/1.e3,c='r',linestyle='-.')
        self.axEPED.set_ylabel("EPED values")
        self.axEPED.set_xlabel("time (s)")
        plt.legend()

        ## plot confinement

        self.axtau = fig.add_subplot(2, 3, 3)
        self.axtau.plot(self.t, self.tauE[:,-1],label="$\\tau_{e}$ (s)",c='b')
        self.axtau.plot(self.t, self.H98,label="H98",c='r')
        self.axtau.plot(second_pulse.t, second_pulse.tauE[:,-1],linestyle='--')
        self.axtau.plot(second_pulse.t, second_pulse.H98,c='r',linestyle='--')
        self.axtau.plot(third_pulse.t, third_pulse.tauE[:,-1],linestyle='-.')
        self.axtau.plot(third_pulse.t, third_pulse.H98,c='r',linestyle='-.')
        self.axtau.set_ylabel("performance parameters")
        self.axtau.set_xlabel("time (s)")
        self.axtau.axhline(y=1.0, ls='-.',c='k')
        self.axtau.set_ylim(bottom=0)
        plt.legend()

        ## plot shaping and q values

        self.axq = fig.add_subplot(2, 3, 2)
        self.axq.plot(self.t, self.kappa95, label='$k_{95}$',c='b')
        self.axq.plot(self.t, self.delta95, label='$\\delta_{95}$',c='r')
        self.axq.plot(self.t, self.trian, label='$\\delta_{sep}$',c='g')
        self.axq.plot(self.t, self.elong, label='$k_{sep}$',c='k')
        self.axq.plot(self.t, self.q95, label='$q_{95}$',c='y')
        self.axq.plot(self.t, self.q_onaxis, label='q0',c='orange')
        self.axq.plot(second_pulse.t, second_pulse.kappa95,c='b',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.delta95,c='r',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.trian,c='g',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.elong,c='k',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.q95,c='y',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.q_onaxis,c='orange',linestyle='--')
        self.axq.plot(third_pulse.t, third_pulse.kappa95,c='b',linestyle='-.')
        self.axq.plot(third_pulse.t, third_pulse.delta95,c='r',linestyle='-.')
        self.axq.plot(third_pulse.t, third_pulse.trian,c='g',linestyle='-.')
        self.axq.plot(third_pulse.t, third_pulse.elong,c='k',linestyle='-.')
        self.axq.plot(third_pulse.t, third_pulse.q95,c='y',linestyle='-.')
        self.axq.plot(third_pulse.t, third_pulse.q_onaxis,c='orange',linestyle='-.')
        self.axq.set_ylabel("shaping and safety factor")
        self.axq.set_xlabel("time (s)")
        plt.legend()

       ## plot beta and averaged kinetic profiles

        self.axglob = fig.add_subplot(2, 3, 4)
        self.axglob.plot(self.t, self.betaN,label="$\\beta_N$",c='b')
        self.axglob.plot(self.t, self.ne_avg,label="$n_{e,avg}$",c='k')
        self.axglob.plot(self.t, self.Te_avg,label="$T_{e,avg}$",c='y')
        self.axglob.plot(self.t, self.Ti_avg,label="$T_{i,avg}$",c='orange')
        self.axglob.plot(self.t, self.ne[:,int(0.2*self.na1[-1])]/self.ne_avg,label="$\\nu_{n_e}$",c='purple')
        self.axglob.plot(second_pulse.t, second_pulse.betaN,c='b',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.ne_avg,c='k',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.Te_avg,c='y',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.Ti_avg,c='orange',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.ne[:,int(0.2*second_pulse.na1[-1])]/second_pulse.ne_avg,c='purple',linestyle='--')
        self.axglob.plot(third_pulse.t, third_pulse.betaN,c='b',linestyle='-.')
        self.axglob.plot(third_pulse.t, third_pulse.ne_avg,c='k',linestyle='-.')
        self.axglob.plot(third_pulse.t, third_pulse.Te_avg,c='y',linestyle='-.')
        self.axglob.plot(third_pulse.t, third_pulse.Ti_avg,c='orange',linestyle='-.')
        self.axglob.plot(third_pulse.t, third_pulse.ne[:,int(0.2*third_pulse.na1[-1])]/third_pulse.ne_avg,c='purple',linestyle='-.')
        self.axglob.set_ylabel("global parameters")
        self.axglob.set_xlabel("time (s)")
        #self.axglob.set_yscale('log')
        plt.legend()

        ## plot Hmode parameters
        
        self.axPLH = fig.add_subplot(2, 3, 5)
        self.axPLH.plot(self.t, self.PLH_perc,label="Martin",c='b')
        self.axPLH.plot(self.t, self.PLH_schmidt_perc,label="Schmidtmayr",c='r')
        self.axPLH.plot(second_pulse.t, second_pulse.PLH_perc,c='b',linestyle='--')
        self.axPLH.plot(second_pulse.t, second_pulse.PLH_schmidt_perc,c='r',linestyle='--')
        self.axPLH.plot(third_pulse.t, third_pulse.PLH_perc,c='b',linestyle='-.')
        self.axPLH.plot(third_pulse.t, third_pulse.PLH_schmidt_perc,c='r',linestyle='-.')
        self.axPLH.set_ylabel("$P_{sep}/P_{LH}$")
        self.axPLH.set_xlabel("time (s)")
        self.axPLH.axhline(y=1.0, ls='-.',c='k')
        self.axPLH.set_ylim(bottom=0)
        plt.legend()

        ## plot total powers
        
        self.axP = fig.add_subplot(2, 3, 6)
        self.axP.plot(self.t, self.QDT[:,-1]*5,label="fusion",c='b')
        self.axP.plot(self.t, self.QICRH[:,-1],label="ICRH",c='r')
        self.axP.plot(self.t, self.QECRH[:,-1],label="ECRH",c='g')
        self.axP.plot(self.t, self.QNBI[:,-1],label="NBI",c='k')
        self.axP.plot(self.t, self.QRAD[:,-1],label="radiation",c='y')
        self.axP.plot(self.t, self.QOH[:,-1],label="ohmic",c='orange')
        self.axP.plot(self.t, self.QETOT[:,-1],label="electron total",c='purple')
        self.axP.plot(self.t, self.QITOT[:,-1],label="ion total",c='cyan')
        self.axP.plot(second_pulse.t, second_pulse.QDT[:,-1]*5,c='b',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QICRH[:,-1],c='r',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QECRH[:,-1],c='g',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QNBI[:,-1],c='k',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QRAD[:,-1],c='y',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QOH[:,-1],c='orange',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QETOT[:,-1],c='purple',linestyle='--')
        self.axP.plot(second_pulse.t, second_pulse.QITOT[:,-1],c='cyan',linestyle='--')
        self.axP.plot(third_pulse.t, third_pulse.QDT[:,-1]*5,c='b',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QICRH[:,-1],c='r',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QECRH[:,-1],c='g',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QNBI[:,-1],c='k',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QRAD[:,-1],c='y',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QOH[:,-1],c='orange',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QETOT[:,-1],c='purple',linestyle='-.')
        self.axP.plot(third_pulse.t, third_pulse.QITOT[:,-1],c='cyan',linestyle='-.')
        self.axP.set_ylabel("P (MW)")
        self.axP.set_xlabel("time (s)")
        self.axP.axhline(y=500, ls='-.',c='k')
        self.axP.axhline(y=1000, ls='-.',c='k')
        #self.axP.set_yscale('log')
        plt.legend()

        GRAPHICStools.addDenseAxis(self.axP)
        GRAPHICStools.addDenseAxis(self.axtau)
        GRAPHICStools.addDenseAxis(self.axglob)
        GRAPHICStools.addDenseAxis(self.axPLH)
        GRAPHICStools.addDenseAxis(self.axq)
        GRAPHICStools.addDenseAxis(self.axEPED)

        GRAPHICStools.addLegendApart(self.axP)
        GRAPHICStools.addLegendApart(self.axtau)
        GRAPHICStools.addLegendApart(self.axglob)
        GRAPHICStools.addLegendApart(self.axPLH)
        GRAPHICStools.addLegendApart(self.axq)
        GRAPHICStools.addLegendApart(self.axEPED)

### Operations: Not part of the CDF class ###
def gradNorm(CDFc, varData, specialDerivative=None):
    """
    This gives:
        a/var * dvar/dr
    """

    grad = derivativeVar(CDFc, varData, specialDerivative=specialDerivative)

    Ld_inv = -1.0 * grad / varData

    aLd = []
    for it in range(len(CDFc.t)):
        aLd.append(CDFc.a[it] * Ld_inv[it, :])

    return np.array(aLd)


def derivativeVar(CDFc, varData, specialDerivative=None, onlyOneTime=False):
    if specialDerivative is None:
        specialDerivative = CDFc.rmin

    if not onlyOneTime:
        grad = []
        for it in range(len(CDFc.t)):
            grad.append(MATHtools.deriv(specialDerivative[it, :], varData[it, :]))
        grad = np.array(grad)
    else:
        grad = MATHtools.deriv(specialDerivative, varData)

    return grad
