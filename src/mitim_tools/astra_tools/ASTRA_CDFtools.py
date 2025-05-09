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
        try:
            self.R = self.f["r2d"][:]
            self.Z = self.f["z2d"][:]
        except:
            self.R = self.f["r"][:]
            self.Z = self.f["z"][:]
        self.rho = self.f["RHO"][:]
        self.xrho = self.f["XRHO"][:]
        self.na1 = self.f["NA1"][:]
        self.BTOR = self.f["BTOR"][:]
        self.IPL = self.f["IPL"][:]
        self.Te = self.f["TE"][:]
        self.TEX = self.f["TEX"][:]
        self.TIX = self.f["TIX"][:]
        self.NEX = self.f["NEX"][:]
        self.Ti = self.f["TI"][:]
        self.ne = self.f["NE"][:]
        self.ni = self.f["NI"][:]
        self.FP = self.f["FP"][:]
        self.TF = self.rho[:,-1] * self.rho[:,-1] * self.BTOR[-1] / 2 # Wb/rad
        self.ER = self.f["ER"][:]
        self.VPOL = self.f["VPOL"][:]
        self.VTOR = self.f["VTOR"][:]
        self.F1 = self.f["F1"][:]
        self.F2 = self.f["F2"][:]
        self.F3 = self.f["F3"][:]
        self.VR = self.f["VR"][:]
        self.Cu = self.f["CU"][:]
        self.Cubs = self.f["CUBS"][:]
        #self.CuOhm = self.f["CUOHM"][:]
        self.CuTor = self.f["CUTOR"][:]
        self.CD = self.f["CD"][:]
        self.Mu = self.f["MU"][:]
        self.q_onaxis = 1/self.Mu[:,0]
        self.MV = self.f["MV"][:]
        self.FV = self.f["FV"][:]
        self.VP = self.f["VP"][:]
        self.Qi = self.f["QI"][:]
        self.Qe = self.f["QE"][:]
        self.Qn = self.f["QN"][:]
        self.PEECR = self.f["PEECR"][:]
        self.G11 = self.f["G11"][:]
        self.NALF = self.f["NALF"][:]

        # dummy variables

        self.CAR1 = self.f["CAR1"][:]
        self.CAR2 = self.f["CAR2"][:]
        self.CAR2X = self.f["CAR2X"][:]
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
        self.CAR14X = self.f["CAR14X"][:]
        self.CAR15 = self.f["CAR15"][:]
        self.CAR15X = self.f["CAR15X"][:]
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
        self.CAR40X = self.f["CAR40X"][:]
        self.CAR41X = self.f["CAR41X"][:]
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
        self.CRAD1   = self.f['CRAD1'][:]
        self.CRAD2   = self.f['CRAD2'][:]
        self.CRAD3   = self.f['CRAD3'][:]
        self.CRAD4   = self.f['CRAD4'][:]
        self.CIMP1   = self.f['CIMP1'][:]
        self.CIMP2   = self.f['CIMP2'][:]
        self.CIMP3   = self.f['CIMP3'][:]
        self.ZRD1 = self.f["ZRD1"][:]
        self.ZRD1X = self.f["ZRD1X"][:]
        self.ZRD2 = self.f["ZRD2"][:]
        self.ZRD2X = self.f["ZRD2X"][:]
        self.ZRD3 = self.f["ZRD3"][:]
        self.ZRD3X = self.f["ZRD3X"][:]
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
        self.ZRD50X = self.f["ZRD50X"][:]
        self.ZRD51X = self.f["ZRD51X"][:]
        self.ZRD52X = self.f["ZRD52X"][:]
        self.ZRD53X = self.f["ZRD53X"][:]
        self.ZRD54X = self.f["ZRD54X"][:]
        self.ZRD55X = self.f["ZRD55X"][:]

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

        self.AMJ = self.f["AMJ"][:]
        self.AMAIN = self.f['AMAIN'][:]
        self.AIM1 = self.f['AIM1'][:]
        self.AIM2 = self.f['AIM2'][:]
        self.AIM3 = self.f['AIM3'][:]
        self.ZIM1 = self.f['ZIM1'][:]
        self.ZIM2 = self.f['ZIM2'][:]
        self.ZIM3 = self.f['ZIM3'][:]
        self.ZMJ = self.f["ZMJ"][:]
        self.ZEF = self.f["ZEF"][:]
        self.ROC = self.f["ROC"][:]
        self.tau = self.f["TAU"][:]
        self.vol = self.f["VOLUM"][:]
        self.VOLUME = self.f["VOLUME"][:]
        try:
            self.t = self.f[
                "TIME"
            ].data  # New ASTRA update needs this patch, for old version still need [:]
        except:
            self.t = self.f["TIME"][:]
        self.rmin = self.f["AMETR"][:]
        self.elong = self.f["ELONG"][:]
        self.elon = self.f["ELON"][:]
        self.trian = self.f["TRIAN"][:]
        self.tria = self.f["TRIA"][:]
        self.UPDWN = self.f["UPDWN"][:]
        self.shif = self.f["SHIF"][:]
        self.shift = self.f["SHIFT"][:]
        self.RTOR = self.f["RTOR"][:]
        self.AB = self.f["AB"][:]
        self.ABC = self.f["ABC"][:]
        self.PE = self.f["PE"][:]
        self.PI = self.f["PI"][:]
        self.PEBM = self.f["PEBM"][:]  # NBI power to electrons
        self.PIBM = self.f["PIBM"][:]  # NBI power to ions
        # self.POH    = self.f['POH'][:] # Ohmic Power
        self.PEECR = self.f["PEECR"][:]  # ECH heating to electrons
        self.PRAD = self.f["PRAD"][:]  # Radiated Power
        self.PEICR = self.f["PEICR"][:]
        self.PIICR = self.f["PIICR"][:]
        # self.PEICL  = self.f['PEICL'][:] # Exchange power, given to ions --> not saved rn
        self.chi_e_TGLF = self.f["CAR18"][:]  # TGLF effective electron diffusivity
        self.chi_i_TGLF = self.f["CAR17"][:]  # TGLF effective ion plot_diffusivity
        self.chi_e_TGLF_smoothed = self.f["CAR22"][:]
        self.chi_i_TGLF_smoothed = self.f["CAR21"][:]
        self.pinch_TGLF_smoothed = self.f["CAR24"][:]
        self.FTO = self.f["FTO"][:]
        self.DN = self.f["DN"][:]
        # self.HN    = self.f['HN'][:]
        # self.XN    = self.f['XN'][:]
        # self.DE    = self.f['DE'][:]
        self.HE = self.f["HE"][:]
        # self.XE    = self.f['XE'][:]
        # self.DI    = self.f['DI'][:]
        # self.HI    = self.f['HI'][:]
        self.XI = self.f["XI"][:]
        self.CN = self.f["CN"][:]
        self.CE = self.f["CE"][:]
        self.CI = self.f["CI"][:]
        self.DC = self.f["DC"][:]
        self.HC = self.f["HC"][:]
        self.XC = self.f["XC"][:]
        self.SN = self.f["SN"][:]
        # self.SNN   = self.f['SNN'][:]
        # self.SNNEU  = self.f['SNNEU'][:]
        self.XRHO = self.f["XRHO"][:]
        self.HRO = self.f["HRO"][:]
        self.PBPER = self.f['PBPER'][:]
        self.PBLON = self.f['PBLON'][:]
        self.SNEBM = self.f['SNEBM'][:]

        self.CC = self.f["CC"][:]
        self.ULON = self.f["ULON"][:]
        self.UPL = self.f["UPL"][:]
        self.GP2 = self.f["GP2"][:]
        self.IPOL = self.f["IPOL"][:]
        self.G22 = self.f["G22"][:]
        self.G33 = self.f["G33"][:]
        # self.POH   = self.CC/(self.ULON/(self.GP2[-1]*self.RTOR[-1]*self.IPOL))**2/self.G33*1e-6
        self.PEDT = self.f["CAR3"][:]
        self.PIDT = self.f["CAR4"][:]
        self.PEICL = self.f["CAR5"][:]
        self.POH = self.f["CAR6"][:]
        self.beta = np.zeros(len(self.PEDT[:,-1]))
        self.betaN = np.zeros(len(self.PEDT[:,-1]))
        for kk in range(0,len(self.PEDT[:,-1])):
            self.beta[kk] = 0.00402*np.cumsum((self.ne[kk,:]*self.Te[kk,:]+self.ni[kk,:]*self.Ti[kk,:]+0.5*(self.PBPER[kk,:]+self.PBLON[kk,:]))*self.VR[kk,:])[-1]/np.cumsum(self.VR[kk,:])[-1]/(self.BTOR[kk]**2)
            self.betaN[kk] = 0.402*np.cumsum((self.ne[kk,:]*self.Te[kk,:]+self.ni[kk,:]*self.Ti[kk,:]+0.5*(self.PBPER[kk,:]+self.PBLON[kk,:]))*self.VR[kk,:])[-1]/np.cumsum(self.VR[kk,:])[-1]*self.ABC[kk]/(self.BTOR[kk]*self.IPL[kk])
        
        self.QIDT   = np.zeros([len(self.PEDT[:,-1]),len(self.PEDT[-1,:])])
        self.QEDT   = np.zeros([len(self.PEDT[:,-1]),len(self.PEDT[-1,:])])
        self.QDT   = np.zeros([len(self.PEDT[:,-1]),len(self.PEDT[-1,:])])
        self.QICRH = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.QNBI = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.QECRH = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.Cu_tot = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.Cubs_tot = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.QE = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.QI = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.QRAD = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.QOH = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.Wtot = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.ne_avg = np.zeros([len(self.PEICR[:,-1])])
        self.Te_avg = np.zeros([len(self.PEICR[:,-1])])
        self.Ti_avg = np.zeros([len(self.PEICR[:,-1])])
        self.n_Gr = self.IPL/(np.pi*self.ABC**2)
        self.tau98 = np.zeros([len(self.PEICR[:,-1])])
        self.AREAT = self.f['AREAT'][:]
        self.SLAT = self.f['SLAT'][:]
        self.area = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.FP_norm = np.zeros([len(self.PEICR[:,-1]),len(self.PEICR[-1,:])])
        self.q95position = [0]*len(self.PEICR[:,-1])
        self.q95 = np.zeros(len(self.PEICR[:,-1]))
        self.delta95 = np.zeros(len(self.PEICR[:,-1]))
        self.kappa95 = np.zeros(len(self.PEICR[:,-1]))
        self.n_Angioni = np.zeros(len(self.PEICR[:,-1]))
        self.SNEBM_tot = np.zeros(len(self.PEICR[:,-1]))
        for ii in range(0,int(self.na1[-1])):
             if ii>0:
                  self.area[:,ii] = self.AREAT[:,ii]-self.AREAT[:,ii-1]
             else:
                  self.area[:,ii] = self.AREAT[:,ii]
        
        for kk in range(0,len(self.PEDT[:,-1])):
             self.FP_norm[kk,:] = (self.FP[kk,:]-self.FP[kk,0])/(self.FP[kk,-1]-self.FP[kk,0])
             self.QIDT[kk,:] = np.cumsum(self.PIDT[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QEDT[kk,:] = np.cumsum(self.PEDT[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QDT[kk,:] = np.cumsum((self.PEDT[kk,:]+self.PIDT[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QNBI[kk,:] = np.cumsum((self.PEBM[kk,:]+self.PIBM[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.QECRH[kk,:] = np.cumsum(self.PEECR[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QICRH[kk,:] = np.cumsum((self.PIICR[kk,:]+self.PEICR[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.Cu_tot[kk,:] = np.cumsum(self.Cu[kk,:]*self.area[kk,:])
             self.Cubs_tot[kk,:] = np.cumsum(self.Cubs[kk,:]*self.area[kk,:])
             self.QE[kk,:] = np.cumsum(self.PE[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QI[kk,:] = np.cumsum(self.PI[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QRAD[kk,:] = np.cumsum(self.PRAD[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.QOH[kk,:] = np.cumsum(self.POH[kk,:]*self.HRO[kk]*self.VR[kk,:])
             self.Wtot[kk,:] = np.cumsum((self.ne[kk,:]*self.Te[kk,:]+self.ni[kk,:]*self.Ti[kk,:])*self.HRO[kk]*self.VR[kk,:])
             self.ne_avg[kk] = np.cumsum(self.ne[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]
             self.Te_avg[kk] = np.cumsum(self.Te[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]
             self.Ti_avg[kk] = np.cumsum(self.Ti[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]
             self.SNEBM_tot[kk] = np.cumsum(self.SNEBM[kk,:]*self.HRO[kk]*self.VR[kk,:])[-1]/self.vol[kk,-1]
             self.tau98[kk] = 0.0562*(self.IPL[kk])**0.93*(self.BTOR[kk])**0.15*(self.ne_avg[kk])**0.41*(self.QE[kk,-1]+self.QI[kk,-1]+self.QRAD[kk,-1])**(-0.69)*(self.RTOR[kk])**1.97*(self.AREAT[kk,-1]/(3.1415*self.rmin[kk,-1]**2))**0.78*(self.rmin[kk,-1]/self.RTOR[kk])**0.58*(self.AMAIN[kk,1])**0.19
             self.q95position[kk] = np.abs(self.FP_norm[kk] - 0.95).argmin()
             self.q95[kk] = 1/self.Mu[kk,self.q95position[kk]]
             self.delta95[kk] = self.tria[kk,self.q95position[kk]]
             self.kappa95[kk] = self.elon[kk,self.q95position[kk]]
             self.n_Angioni[kk] = 1.347-0.117*math.log(0.2*self.ne_avg[kk]*self.RTOR[kk]*self.Te_avg[kk]**(-2))+1.331*self.SNEBM_tot[kk]-4.03*self.beta[kk]

        self.f_Gr = self.ne_avg/10/self.n_Gr
        # self.QNTOT  = self.f['CAR8'][:]
        self.QETOT  = self.QE
        self.QITOT  = self.QI
        self.SNTOT = self.f["SNTOT"][:]
        self.Wtot = 0.0024*self.Wtot   #check formula in ASTRA
        self.tauE = self.Wtot/(self.QRAD+self.QE+self.QI)
        self.H98 = self.tauE[:,-1]/self.tau98
        self.NDEUT = self.f["NDEUT"][:]
        self.NTRIT = self.f["NTRIT"][:]
        self.NIZ1 = self.f["NIZ1"][:]
        self.NIZ2 = self.f["NIZ2"][:]
        self.NIZ3 = self.f["NIZ3"][:]
        self.CAR1 = self.f["CAR1"][:]
        self.NMAIN = self.f["NMAIN"][:]
        self.ZIM1 = self.f["ZIM1"][:]
        self.ZIM2 = self.f["ZIM2"][:]
        self.ZIM3 = self.f["ZIM3"][:]
        self.fMAIN = self.NMAIN/self.ne
        self.f1 = self.NIZ1/self.ne
        self.f2 = self.NIZ2/self.ne
        self.f3 = self.NIZ3/self.ne
        self.CAR7 = self.f["CAR7"][:]
        self.ZMAIN = self.f["ZMAIN"][:]
        self.ptot  = (self.ne*self.Te+self.ni*self.Ti+0.5*(self.PBPER+self.PBLON))*1.6e-3  #in MPa
        self.rlte  = np.zeros([len(self.PEDT[:,-1]),len(self.PEDT[-1,:])])
        self.rlti  = np.zeros([len(self.PEDT[:,-1]),len(self.PEDT[-1,:])])
        self.rlne  = np.zeros([len(self.PEDT[:,-1]),len(self.PEDT[-1,:])])
        for kk in range(0,len(self.Te[:,-1])):
             for jj in range(0,len(self.Te[-1,:])-1):
                  self.rlte[kk,jj]=-self.RTOR[-1]/(0.5*(self.Te[kk,jj]+self.Te[kk,jj+1])*(self.rmin[kk,jj+1]-self.rmin[kk,jj])/(self.Te[kk,jj+1]-self.Te[kk,jj]))
                  self.rlti[kk,jj]=-self.RTOR[-1]/(0.5*(self.Ti[kk,jj]+self.Ti[kk,jj+1])*(self.rmin[kk,jj+1]-self.rmin[kk,jj])/(self.Ti[kk,jj+1]-self.Ti[kk,jj]))
                  self.rlne[kk,jj]=-self.RTOR[-1]/(0.5*(self.ne[kk,jj]+self.ne[kk,jj+1])*(self.rmin[kk,jj+1]-self.rmin[kk,jj])/(self.ne[kk,jj+1]-self.ne[kk,jj]))
             self.rlte[kk,jj+1]=self.rlte[kk,jj]
             self.rlti[kk,jj+1]=self.rlti[kk,jj]
             self.rlne[kk,jj+1]=self.rlne[kk,jj]
 
        ##  check on quasi-neutrality
        self.quasi = (self.f['NE'][:]-self.f['NMAIN'][:]*self.f['ZMAIN'][:]-self.f['NIZ1'][:]*self.f['ZIM1'][:]-self.f['NIZ2'][:]*self.f['ZIM2'][:]-self.f['NIZ3'][:]*self.f['ZIM3'][:])/self.f['NE'][:]

        ##  some global and performance parameters
        self.Q = (self.QDT[:,-1]/(self.QICRH[:,-1]+self.QOH[:,-1]))/0.2    ## in teh D+T fusion reactions 20% goes to He and 80% to neutrons
        self.Pfus = self.QDT/0.2
        self.PLH = 0.0488*(self.ne_avg/10.)**0.717*(self.BTOR)**0.803*(self.SLAT[:,-1])**0.941*(2/self.AMAIN[:,-1])
        self.PLH_lower = 0.0488*math.exp(-0.057)*(self.ne_avg/10.)**0.682*(self.BTOR)**0.771*(self.SLAT[:,-1])**0.922*(2/self.AMAIN[:,-1])
        self.PLH_upper = 0.0488*math.exp(0.057)*(self.ne_avg/10.)**0.752*(self.BTOR)**0.835*(self.SLAT[:,-1])**0.96*(2/self.AMAIN[:,-1])
        self.PLH_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH
        self.PLH_lower_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH_lower
        self.PLH_upper_perc = (self.QE[:,-1]+self.QI[:,-1])/self.PLH_upper
        self.PLH_schmidtmayr = 0.0325*(self.ne_avg/10.)**1.05*(self.BTOR)**0.68*(self.SLAT[:,-1])**0.93*(2/self.AMAIN[:,-1])
        self.PLH_schmidt_perc = (self.QI[:,-1])/self.PLH_schmidtmayr
        self.q_Uckan = 5*self.ABC**2*self.BTOR/(self.RTOR*self.IPL)*(1+self.kappa95**2*(1+2*self.delta95**2-1.2*self.delta95**3))/2

        rtor_matrix = np.zeros(self.rho.shape)
        for i in range(rtor_matrix.shape[1]):
            rtor_matrix[:, i] = self.RTOR[:]

        self.a = self.rmin[:, -1]
        self.rmaj_LFx = (
            rtor_matrix + self.shif + self.rmin
        )  # major radius on the low field side

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
        self.fn.deleteGui()

    def get_rho_tor_indices(self, rho_tor_aims):
        """
        Output: Array w/ shape (number of times, number of rho values of interest)
        """
        self.calcProfiles()

        self.rho_tor_aims = rho_tor_aims
        self.i_rho_tor_aims = []

        for t in range(len(self.t)):
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

            for t in range(len(self.t)):
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

        plt.legend(title=r"$\rho_{tor}$")

        ## Make radial figures ##

        self.axTer = fig.add_subplot(2, 2, 2)
        self.make_radial_plots(self.axTer, self.Te, time_aims)
        self.axTer.set_ylabel("$T_e$ (keV)")

        self.axTir = fig.add_subplot(2, 2, 4)
        self.make_radial_plots(self.axTir, self.Ti, time_aims)
        self.axTir.set_ylabel("$T_i$ (keV)")

        plt.legend(title="Times")

    def plot_gradients(
        self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ):
        fig = self.fn.add_figure(label="Temp Gradient Profiles")
        self.get_transport()

        ## Make temporal figures ##
        self.axaLTet = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axaLTet, self.aLTe, rho_tor_aims)
        self.axaLTet.set_ylabel("$a\\nabla T_e/T_e$")
        plt.legend(title=r"$\rho_{tor}$")

        self.axaLTit = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(self.axaLTit, self.aLTi, rho_tor_aims)
        self.axaLTit.set_ylabel("$a\\nabla T_i/T_i$")
        plt.legend(title=r"$\rho_{tor}$")

        self.axaLnet = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(self.axaLnet, self.aLne, rho_tor_aims)
        self.axaLnet.set_ylabel("$a\\nabla n_e/n_e$")
        plt.legend(title=r"$\rho_{tor}$")

        ##Make radial figures ##
        self.axaLTer = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axaLTer, self.aLTe, time_aims)
        self.axaLTer.set_ylabel("$a\\nabla T_e/T_e$")
        plt.legend(title="Times")

        self.axaLTir = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(self.axaLTir, self.aLTi, time_aims)
        self.axaLTir.set_ylabel("$a\\nabla T_i/T_i$")
        plt.legend(title="Times")

        self.axaLner = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axaLner, self.aLne, time_aims)
        self.axaLner.set_ylabel("$a\\nabla n_e/n_e$")
        plt.legend(title="Times")

    def plot_density(self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        fig = self.fn.add_figure(label="Density Profiles")

        # Make temporal figures
        self.axnet = fig.add_subplot(2, 3, 1)
        self.make_temporal_plots(self.axnet, self.ne, rho_tor_aims)
        self.axnet.set_ylabel("Density [$10^{19}/m^3$]")
        #plt.legend(title=r"$\rho_{tor}$")

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
        plt.legend(title="Times")

        self.axqr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axqr, self.q, time_aims)
        self.axqr.set_ylabel("q")

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
        plt.legend(title=r"$\rho_{tor}$")

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

    def plot_powers_r(
        self, time_aims, rho_tor_aims=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ):
        # PE = PEBM + POH + PEECR - PRAD - PEICL
        fig = self.fn.add_figure(label="Power(r)")

        # Make temporal figures
        self.axPEr = fig.add_subplot(2, 3, 1)
        self.make_radial_plots(self.axPEr, self.PE, time_aims)
        self.axPEr.set_ylabel("$P_E$ ($MW/m^3$)")
        plt.legend(title="Times [s]")

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
        plt.legend(title="Times [s]")

        # Make radial figures
        self.axchi_er = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axchi_er, self.chi_e_TGLF, time_aims)
        self.axchi_er.set_ylabel("TGLF ($m^2/s$)")
        plt.legend(title=r"$\rho_{tor}$")

        self.axchi_e_smoothedr = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(
            self.axchi_e_smoothedr, self.chi_e_TGLF_smoothed, time_aims
        )
        self.axchi_e_smoothedr.set_ylabel("Smoothed ($m^2/s$)")

        self.axHEr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axHEr, self.HE, time_aims)
        self.axHEr.set_ylabel("ASTRA ($m^2/s$)")

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
        plt.legend(title="Times [s]")

        # Make radial figures
        self.axchi_ir = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axchi_ir, self.chi_i_TGLF, time_aims)
        self.axchi_ir.set_ylabel("TGLF ($m^2/s$)")
        plt.legend(title=r"$\rho_{tor}$")

        self.axchi_i_smoothedr = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(
            self.axchi_i_smoothedr, self.chi_i_TGLF_smoothed, time_aims
        )
        self.axchi_i_smoothedr.set_ylabel("Smoothed ($m^2/s$)")

        self.axXIr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axXIr, self.XI, time_aims)
        self.axXIr.set_ylabel("ASTRA ($m^2/s$)")

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

        plt.legend(title=r"$\rho_{tor}$")

        # Make radial figures

        self.axQer = fig.add_subplot(2, 2, 3)
        self.axQer.set_ylabel("Qe (MW)")
        self.make_radial_plots(self.axQer, self.Qe, time_aims)

        self.axQir = fig.add_subplot(2, 2, 4)
        self.axQir.set_ylabel("Qi (MW)")
        self.make_radial_plots(self.axQir, self.Qi, time_aims)

        plt.legend(title="Times")

    def plot_flux_matching(
        self, time_aims, rho_tor_aims=[0.5], qe_lim=[0, 5], qi_lim=[0, 5], qn_lim=[0, 5]
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
        plt.legend(["Qe", "Qetot"], title=r"$\rho_{tor}$ = " + str(rho_tor_aims[0]))

        self.axQit = fig.add_subplot(2, 3, 2)
        self.make_temporal_plots(self.axQit, self.Qi, rho_tor_aims, linestyle="dashed")
        # self.make_temporal_plots(self.axQit, self.QITOT, rho_tor_aims)
        self.axQit.set_ylabel("Qi")
        # self.axQit.set_ylim(qi_lim)
        plt.legend(["Qi", "Qitot"], title=r"$\rho_{tor}$ = " + str(rho_tor_aims[0]))

        self.axQnt = fig.add_subplot(2, 3, 3)
        self.make_temporal_plots(
            self.axQnt, self.Qn / self.G11, rho_tor_aims, linestyle="dashed"
        )
        # self.make_temporal_plots(self.axQnt, self.QNTOT / self.G11, rho_tor_aims)
        self.axQnt.set_ylabel("Qn")
        # self.axQnt.set_ylim(qn_lim)
        plt.legend(
            ["Qn/volume", "Qntot/volume"],
            title=r"$\rho_{tor}$ = " + str(rho_tor_aims[0]),
        )

        ## Make radial figures ##

        self.axQer = fig.add_subplot(2, 3, 4)
        self.make_radial_plots(self.axQer, self.Qe, last_time, linestyle="dashed")
        # self.make_radial_plots(self.axQer, self.QETOT, last_time)
        self.axQer.set_xlim([0, self.boundary])
        self.axQer.set_ylim(qe_lim)
        self.axQer.set_ylabel("Qe")
        plt.legend(["Qe", "Qetot"], title="Time = " + str(last_time[0]))

        self.axQir = fig.add_subplot(2, 3, 5)
        self.make_radial_plots(self.axQir, self.Qi, last_time, linestyle="dashed")
        # self.make_radial_plots(self.axQir, self.QITOT, last_time)
        self.axQir.set_xlim([0, self.boundary])
        self.axQir.set_ylim(qi_lim)
        self.axQir.set_ylabel("Qi")
        plt.legend(["Qi", "Qitot"], title="Time = " + str(last_time[0]))

        self.axQnr = fig.add_subplot(2, 3, 6)
        self.make_radial_plots(self.axQnr, self.Qn, last_time, linestyle="dashed")
        # self.make_radial_plots(self.axQnr, self.QNTOT, last_time)
        self.axQnr.set_xlim([0, self.boundary])
        self.axQnr.set_ylim(qn_lim)
        self.axQnr.set_ylabel("Qn")
        plt.legend(["Qn/volume", "Qntot/volume"], title="Time = " + str(last_time[0]))

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

        ## plot shaping values

        self.axq = fig.add_subplot(2, 3, 2)
        self.axq.plot(self.t, self.kappa95, label='$k_{95}$')
        self.axq.plot(self.t, self.delta95, label='$\\delta_{95}$')
        self.axq.plot(self.t, self.trian, label='$\\delta_{sep}$')
        self.axq.plot(self.t, self.elong, label='$k_{sep}$')
        self.axq.set_ylabel("shaping")
        self.axq.set_xlabel("time (s)")
        plt.legend()

       ## plot beta and averaged kinetic profiles

        self.axglob = fig.add_subplot(2, 3, 4)
        self.axglob.plot(self.t, self.betaN,label="$\\beta_N$")
        self.axglob.plot(self.t, self.q95, label='$q_{95}$')
        self.axglob.plot(self.t, self.q_onaxis, label='q0')
        self.axglob.plot(self.t, self.ne_avg/self.ne_avg[0],label="$n_{e,avg} normalized$")
        self.axglob.plot(self.t, self.Te_avg/self.Te_avg[0],label="$T_{e,avg} normalized$")
        self.axglob.plot(self.t, self.Ti_avg/self.Ti_avg[0],label="$T_{i,avg} normalized$")
        self.axglob.set_ylabel("global parameters")
        self.axglob.set_xlabel("time (s)")
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
        fig = self.fn.add_figure(label="Pulses: dashed = V2B baseline, solid = higher Zeff and k V2B")

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

        ## plot shaping values

        self.axq = fig.add_subplot(2, 3, 2)
        self.axq.plot(self.t, self.kappa95, label='$k_{95}$',c='b')
        self.axq.plot(self.t, self.delta95, label='$\\delta_{95}$',c='r')
        self.axq.plot(self.t, self.trian, label='$\\delta_{sep}$',c='g')
        self.axq.plot(self.t, self.elong, label='$k_{sep}$',c='k')
        self.axq.plot(second_pulse.t, second_pulse.kappa95,c='b',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.delta95,c='r',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.trian,c='g',linestyle='--')
        self.axq.plot(second_pulse.t, second_pulse.elong,c='k',linestyle='--')
        self.axq.set_ylabel("shaping")
        self.axq.set_xlabel("time (s)")
        plt.legend()

       ## plot beta and averaged kinetic profiles

        self.axglob = fig.add_subplot(2, 3, 4)
        self.axglob.plot(self.t, self.betaN,label="$\\beta_N$",c='b')
        self.axglob.plot(self.t, self.q95, label='$q_{95}$',c='r')
        self.axglob.plot(self.t, self.q_onaxis, label='q0',c='g')
        self.axglob.plot(self.t, self.ne_avg,label="$n_{e,avg}$",c='k')
        self.axglob.plot(self.t, self.Te_avg,label="$T_{e,avg}$",c='y')
        self.axglob.plot(self.t, self.Ti_avg,label="$T_{i,avg}$",c='orange')
        self.axglob.plot(self.t, self.ne[:,int(0.2*self.na1[-1])]/self.ne_avg,label="$\\nu_{n_e}$",c='purple')
        self.axglob.plot(second_pulse.t, second_pulse.betaN,c='b',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.q95,c='r',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.q_onaxis,c='g',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.ne_avg,c='k',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.Te_avg,c='y',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.Ti_avg,c='orange',linestyle='--')
        self.axglob.plot(second_pulse.t, second_pulse.ne[:,int(0.2*second_pulse.na1[-1])]/second_pulse.ne_avg,c='purple',linestyle='--')
        self.axglob.set_ylabel("global parameters")
        self.axglob.set_xlabel("time (s)")
        self.axglob.set_yscale('log')
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