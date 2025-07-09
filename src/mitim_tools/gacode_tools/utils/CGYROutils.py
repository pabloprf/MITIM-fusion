import os
import numpy as np
import statsmodels.api as sm
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from pygacode.cgyro.data_plot import cgyrodata_plot
from pygacode import gacodefuncs
from IPython import embed

class CGYROlinear_scan:
    def __init__(self, labels, cgyro_data):   

        self.labels = labels

        # Store the data in a structured way        
        self.aLTi = []
        self.ky = []
        self.g_mean = []
        self.f_mean = []

        for label in labels:
            self.ky.append(cgyro_data[label].ky[0])
            self.aLTi.append(cgyro_data[label].aLTi)
            self.g_mean.append(cgyro_data[label].g_mean[0])
            self.f_mean.append(cgyro_data[label].f_mean[0])

        self.ky = np.array(self.ky)
        self.aLTi = np.array(self.aLTi)
        self.g_mean = np.array(self.g_mean)
        self.f_mean = np.array(self.f_mean)

class CGYROout:
    def __init__(self, folder, tmin=0.0):

        original_dir = os.getcwd()

        self.folder = folder
        self.tmin = tmin

        try:
            print(f"\t- Reading CGYRO data from {self.folder.resolve()}")
            self.cgyrodata = cgyrodata_plot(f"{self.folder.resolve()}{os.sep}")
        except FileNotFoundError:
            raise Exception(f"[MITIM] Could not find CGYRO data in {self.folder.resolve()}. Please check the folder path or run CGYRO first.")
        except Exception as e:
            print(f"\t- Error reading CGYRO data: {e}")
            if print('- Could not read data, do you want me to try do "cgyro -t" in the folder?',typeMsg='q'):
                os.chdir(self.folder)
                os.system("cgyro -t")
            self.cgyrodata = cgyrodata_plot(f"{self.folder.resolve()}{os.sep}")

        os.chdir(original_dir)
            
        # Process the data
        self._process()

    def _process(self):
        
        # --------------------------------------------------------------
        # Read inputs
        # --------------------------------------------------------------
        
        self.params1D = {}
        for var in self.cgyrodata.__dict__:
            par = self.cgyrodata.__dict__[var]
            if isinstance(par, bool) or IOtools.isnum(par):
                self.params1D[var] = par
            elif isinstance(par, (list, np.ndarray)) and par.ndim==1 and len(par) <= 5:
                for i, p in enumerate(par):
                    self.params1D[f"{var}_{i}"] = p
    
        # --------------------------------------------------------------
        # Postprocess with MITIM-curated structures and variables
        # --------------------------------------------------------------

        if 'phib' in self.cgyrodata.__dict__:
            print('\t- Forcing tmin to the last time point because this is a linear run', typeMsg='i')
            self.tmin = self.cgyrodata.t[-1]

        self.cgyrodata.getflux(cflux='auto')
        self.cgyrodata.getnorm("elec")

        # Understand positions
        self.electron_flag = np.where(self.cgyrodata.z == -1)[0][0]
        self.all_flags = np.arange(0, len(self.cgyrodata.z), 1)
        self.ions_flags = self.all_flags[self.all_flags != self.electron_flag]

        self.all_names = [f"{gacodefuncs.specmap(self.cgyrodata.mass[i],self.cgyrodata.z[i])}({self.cgyrodata.z[i]},{self.cgyrodata.mass[i]:.1f})" for i in self.all_flags]

        self.fields = np.arange(self.cgyrodata.n_field)

        # ************************
        # Inputs
        # ************************

        self.aLTi = self.cgyrodata.dlntdr[0]
        self.aLTe = self.cgyrodata.dlntdr[self.electron_flag]
        self.aLne = self.cgyrodata.dlnndr[self.electron_flag]
        self.Qgb = self.cgyrodata.q_gb_norm
        self.Ggb = self.cgyrodata.gamma_gb_norm

        # ************************
        # Turbulence
        # ************************
        self.ky = self.cgyrodata.kynorm
        self.f = self.cgyrodata.fnorm[0,:,:]                # (ky, time)
        self.g = self.cgyrodata.fnorm[1,:,:]                # (ky, time)

        if 'phib' in self.cgyrodata.__dict__:
            self.phi = self.cgyrodata.phib                  # (ball, time)
            self.apar = self.cgyrodata.aparb                # (ball, time)
            self.bpar = self.cgyrodata.bparb                # (ball, time)
            self.theta_ballooning = self.cgyrodata.thetab   # (ball, time)

        # ************************
        # Fluxes
        # ************************
        
        self.t = self.cgyrodata.tnorm

        flux = np.sum(self.cgyrodata.ky_flux, axis=3)       # (species, moments, fields, time)

        # Electron energy flux
        
        i_species, i_moment = -1, 1
        i_fields = 0
        self.Qe_ES = flux[i_species, i_moment, i_fields, :] / self.cgyrodata.qc
        i_fields = 1
        self.Qe_EM_apar = flux[i_species, i_moment, i_fields, :] / self.cgyrodata.qc
        i_fields = 2
        self.Qe_EM_aper = flux[i_species, i_moment, i_fields, :] / self.cgyrodata.qc

        self.Qe_EM = self.Qe_EM_apar + self.Qe_EM_aper
        self.Qe = self.Qe_ES + self.Qe_EM

        # Electron particle flux
        
        i_species, i_moment = -1, 0
        i_fields = 0
        self.Ge_ES = flux[i_species, i_moment, i_fields, :]
        i_fields = 1
        self.Ge_EM_apar = flux[i_species, i_moment, i_fields, :]
        i_fields = 2
        self.Ge_EM_aper = flux[i_species, i_moment, i_fields, :]
        
        self.Ge_EM = self.Ge_EM_apar + self.Ge_EM_aper
        self.Ge = self.Ge_ES + self.Ge_EM
        
        # Ions energy flux
        
        i_species, i_moment = self.ions_flags, 1
        i_fields = 0
        self.Qi_all_ES = flux[i_species, i_moment, i_fields, :] / self.cgyrodata.qc
        i_fields = 1
        self.Qi_all_EM_apar = flux[i_species, i_moment, i_fields, :] / self.cgyrodata.qc
        i_fields = 2
        self.Qi_all_EM_aper = flux[i_species, i_moment, i_fields, :] / self.cgyrodata.qc
        
        self.Qi_all_EM = self.Qi_all_EM_apar + self.Qi_all_EM_aper
        self.Qi_all = self.Qi_all_ES + self.Qi_all_EM
        
        self.Qi = self.Qi_all.sum(axis=0)
        self.Qi_EM = self.Qi_all_EM.sum(axis=0)
        self.Qi_ES = self.Qi_all_ES.sum(axis=0)
        
        # ************************
        # Saturated
        # ************************
        
        flags = {
        'Qe': ['Qgb', 'MWm2'], 
        'Qi': ['Qgb', 'MWm2'], 
        'Ge': ['Ggb', '?'], 
        'Qe_ES': ['Qgb', 'MWm2'], 
        'Qi_ES': ['Qgb', 'MWm2'], 
        'Ge_ES': ['Qgb', 'MWm2'], 
        'Qe_EM': ['Qgb', 'MWm2'], 
        'Qi_EM': ['Qgb', 'MWm2'], 
        'Ge_EM': ['Ggb', '?'],
        'g': [None, None],
        'f': [None, None],
        }
        
        for iflag in flags:
            Qm, Qstd = apply_ac(
                    self.t,
                    self.__dict__[iflag],
                    tmin=self.tmin,
                    label_print=iflag
                    )
                
            self.__dict__[iflag+'_mean'] = Qm
            self.__dict__[iflag+'_std'] = Qstd
                
            # Real units
            if flags[iflag][0] is not None:
                self.__dict__[iflag+flags[iflag][1]+'_mean'] = self.__dict__[iflag+'_mean'] * self.__dict__[flags[iflag][0]]
                self.__dict__[iflag+flags[iflag][1]+'_std'] = self.__dict__[iflag+'_std'] * self.__dict__[flags[iflag][0]]


def apply_ac(t, S, tmin = 0, label_print = ''):
    
    # Correct the standard deviation
    def grab_ncorrelation(S, tmin):
        # Calculate the autocorrelation function
        i_acf = sm.tsa.acf(S)
        
        # Calculate how many time slices make the autocorrelation function is 0.36
        icor = np.abs(i_acf-0.36).argmin()
        
        # Define number of samples
        n_corr = ( len(t) - it0 ) / ( 3.0 * icor ) #Define "sample" as 3 x autocor time
        
        return n_corr, icor
    
    it0 = np.argmin(np.abs(t - tmin))
    
    # Calculate the mean and std of the signal after tmin
    S_mean = np.mean(S[...,it0:],axis=-1)
    S_std = np.std(S[...,it0:],axis=-1) # To follow NTH convention

    if S.ndim == 1:
        n_corr, icor = grab_ncorrelation(S, tmin)
        
        S_std = S_std / np.sqrt(n_corr)
        
        print(f"\t- {(label_print + ': a') if len(label_print)>0 else 'A'}utocorr time: {icor:.1f} -> {n_corr:.1f} samples -> {S_mean:.2e} +-{S_std:.2e}")
        
    elif S.ndim == 2:
        n_corr = np.zeros(S.shape[0])
        icor = np.zeros(S.shape[0])
        for i in range(S.shape[0]):
            n_corr[i], icor[i] = grab_ncorrelation(S[i], tmin)
        S_std = S_std / np.sqrt(n_corr)

        for i in range(S.shape[0]):
            print(f"\t- {(label_print + f'_{i}: a') if len(label_print)>0 else 'A'}utocorr: {icor[i]:.1f} -> {n_corr[i]:.1f} samples -> {S_mean[i]:.2e} +-{S_std[i]:.2e}")

    return S_mean, S_std
