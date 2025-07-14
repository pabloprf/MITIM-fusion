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
        self.cgyrodata.getgeo()
        self.cgyrodata.getxflux()
        self.cgyrodata.getbigfield()

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
        self.kx = self.cgyrodata.kxnorm
        self.f = self.cgyrodata.fnorm[0,:,:]                # (ky, time)
        self.g = self.cgyrodata.fnorm[1,:,:]                # (ky, time)

        # Ballooning Modes (complex eigenfunctions)
        if 'phib' in self.cgyrodata.__dict__:
            self.phi_ballooning = self.cgyrodata.phib       # (ball, time)
            self.apar_ballooning = self.cgyrodata.aparb     # (ball, time)
            self.bpar_ballooning = self.cgyrodata.bparb     # (ball, time)
            self.theta_ballooning = self.cgyrodata.thetab   # (ball, time)

        # Fluctuations (complex numbers)
        
        gbnorm = False
        
        theta = -1

        moment, species, field = 'phi', None, 0
        self.phi, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)        # [COMPLEX] (nradial,ntoroidal,time)
        if 'kxky_apar' in self.cgyrodata.__dict__:
            field = 1
            self.apar, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)   # [COMPLEX] (nradial,ntoroidal,time)
            field = 2
            self.bpar, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)   # [COMPLEX] (nradial,ntoroidal,time)
        
        moment, species, field = 'n', self.electron_flag, 0
        self.ne, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)         # [COMPLEX] (nradial,ntoroidal,time)

        species = self.ions_flags
        self.ni_all, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)     # [COMPLEX] (nradial,nions,ntoroidal,time)
        self.ni = self.ni_all.sum(axis=1)                                                         # [COMPLEX] (nradial,ntoroidal,time)

        moment, species, field = 'e', self.electron_flag, 0
        Ee, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)              # [COMPLEX] (nradial,ntoroidal,time)
        
        species = self.ions_flags
        Ei_all, _ = self.cgyrodata.kxky_select(theta,field,moment,species,gbnorm=gbnorm)          # [COMPLEX] (nradial,nions,ntoroidal,time)
        Ei = Ei_all.sum(axis=1)

        # Transform to temperature
        self.Te         = 2/3 * Ee - self.ne
        self.Ti_all     = 2/3 * Ei_all - self.ni_all
        self.Ti         = 2/3 * Ei - self.ni

        # Sum over radial modes and divide between n=0 and n>0 modes, RMS
        variables = ['phi', 'ne', 'ni_all', 'Te', 'Ti_all']
        for var in variables:
            self.__dict__[var+'_rms_sumnr_n0'] = (abs(self.__dict__[var][:,0,:])**2).sum(axis=0)**0.5       # (time)
            self.__dict__[var+'_rms_sumnr_sumnumn1'] = (abs(self.__dict__[var][:,1:,:])**2).sum(axis=(0,1))**0.5  # (time)
            self.__dict__[var+'_rms_sumnr_sumn'] = (abs(self.__dict__[var][:,:,:])**2).sum(axis=(0,1))**0.5    # (time)
            self.__dict__[var+'_rms_sumnr'] = (abs(self.__dict__[var][:,:,:])**2).sum(axis=(0))**0.5        # (ntoroidal, time)
            self.__dict__[var+'_rms_n0'] = (abs(self.__dict__[var][:,0,:])**2)**0.5                         # (nradial,time) 
            self.__dict__[var+'_rms_sumn1'] = (abs(self.__dict__[var][:,1:,:])**2).sum(axis=(1))**0.5          # (nradial,time)
            self.__dict__[var+'_rms_sumn'] = (abs(self.__dict__[var][:,:,:])**2).sum(axis=(1))**0.5          # (nradial,time)


        # Cross-phases
        self.nT = _cross_phase(self.ne, self.Te) * 180/ np.pi  # (nradial, ntoroidal, time)
        self.nT_kx0 = self.nT[np.argmin(np.abs(self.kx)),:,:]  # (ntoroidal, time)

        self.phiT = _cross_phase(self.phi, self.Te) * 180/ np.pi  # (nradial, ntoroidal, time)
        self.phiT_kx0 = self.phiT[np.argmin(np.abs(self.kx)),:,:]
        
        self.phin = _cross_phase(self.phi, self.ne) * 180/ np.pi  # (nradial, ntoroidal, time)
        self.phin_kx0 = self.phin[np.argmin(np.abs(self.kx)),:,:]

        # ************************
        # Fluxes
        # ************************
        
        self.t = self.cgyrodata.tnorm
        
        ky_flux = self.cgyrodata.ky_flux # (species, moments, fields, ntoroidal, time)

        # Electron energy flux
        
        i_species, i_moment = -1, 1
        i_fields = 0
        self.Qe_ES_ky = ky_flux[i_species, i_moment, i_fields, :, :]
        i_fields = 1
        self.Qe_EM_apar_ky = ky_flux[i_species, i_moment, i_fields, :, :]
        i_fields = 2
        self.Qe_EM_aper_ky = ky_flux[i_species, i_moment, i_fields, :, :]

        self.Qe_EM_ky = self.Qe_EM_apar_ky + self.Qe_EM_aper_ky
        self.Qe_ky = self.Qe_ES_ky + self.Qe_EM_ky

        # Electron particle flux
        
        i_species, i_moment = -1, 0
        i_fields = 0
        self.Ge_ES_ky = ky_flux[i_species, i_moment, i_fields, :]
        i_fields = 1
        self.Ge_EM_apar_ky = ky_flux[i_species, i_moment, i_fields, :]
        i_fields = 2
        self.Ge_EM_aper_ky = ky_flux[i_species, i_moment, i_fields, :]
        
        self.Ge_EM_ky = self.Ge_EM_apar_ky + self.Ge_EM_aper_ky
        self.Ge_ky = self.Ge_ES_ky + self.Ge_EM_ky
        
        # Ions energy flux
        
        i_species, i_moment = self.ions_flags, 1
        i_fields = 0
        self.Qi_all_ES_ky = ky_flux[i_species, i_moment, i_fields, :]
        i_fields = 1
        self.Qi_all_EM_apar_ky = ky_flux[i_species, i_moment, i_fields, :]
        i_fields = 2
        self.Qi_all_EM_aper_ky = ky_flux[i_species, i_moment, i_fields, :]
        
        self.Qi_all_EM_ky = self.Qi_all_EM_apar_ky + self.Qi_all_EM_aper_ky
        self.Qi_all_ky = self.Qi_all_ES_ky + self.Qi_all_EM_ky
        
        self.Qi_ky = self.Qi_all_ky.sum(axis=0)
        self.Qi_EM_ky = self.Qi_all_EM_ky.sum(axis=0)
        self.Qi_EM_apar_ky = self.Qi_all_EM_apar_ky.sum(axis=0)
        self.Qi_EM_aper_ky = self.Qi_all_EM_aper_ky.sum(axis=0)
        self.Qi_ES_ky = self.Qi_all_ES_ky.sum(axis=0)
        
        # ************************
        # Sum total 
        # ************************
        variables = ['Qe','Ge','Qi','Qi_all']
        for var in variables:
            for i in ['', '_ES', '_EM_apar', '_EM_aper', '_EM']:
                self.__dict__[var+i] = self.__dict__[var+i+'_ky'].sum(axis=-2)  # (time)
        
        # ************************
        # Saturated
        # ************************
        
        flags = {
        'Qe': ['Qgb', 'MWm2'], 
        'Qe_ky': ['Qgb', 'MWm2'], 
        'Qi': ['Qgb', 'MWm2'], 
        'Qi_ky': ['Qgb', 'MWm2'],
        'Ge': ['Ggb', '?'], 
        'Ge_ky': ['Ggb', '?'], 
        'Qe_ES': ['Qgb', 'MWm2'], 
        'Qi_ES': ['Qgb', 'MWm2'], 
        'Ge_ES': ['Qgb', 'MWm2'], 
        'Qe_EM': ['Qgb', 'MWm2'], 
        'Qi_EM': ['Qgb', 'MWm2'], 
        'Ge_EM': ['Ggb', '?'],
        'g': [None, None],
        'f': [None, None],
        'phi_rms_sumnr': [None, None],
        'ne_rms_sumnr': [None, None],
        'Te_rms_sumnr': [None, None],
        'phi_rms_n0': [None, None],
        'phi_rms_sumn1': [None, None],
        'phi_rms_sumn': [None, None],
        'ne_rms_n0': [None, None],
        'ne_rms_sumn1': [None, None],
        'ne_rms_sumn': [None, None],
        'Te_rms_n0': [None, None],
        'Te_rms_sumn1': [None, None],
        'Te_rms_sumn': [None, None],
        'nT_kx0': [None, None],
        'phiT_kx0': [None, None],
        'phin_kx0': [None, None],
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
    def grab_ncorrelation(S, it0):
        # Calculate the autocorrelation function
        i_acf = sm.tsa.acf(S)
        
        # Calculate how many time slices make the autocorrelation function is 0.36
        icor = np.abs(i_acf-0.36).argmin()
        
        # Define number of samples
        n_corr = ( len(t) - it0 ) / ( 3.0 * icor ) #Define "sample" as 3 x autocor time
        
        return n_corr, icor
    
    it0 = np.argmin(np.abs(t - tmin))
    
    # Calculate the mean and std of the signal after tmin (last dimension is time)
    S_mean = np.mean(S[..., it0:], axis=-1)
    S_std = np.std(S[..., it0:], axis=-1) # To follow NTH convention

    if S.ndim == 1:
        # 1D case: single time series
        n_corr, icor = grab_ncorrelation(S[it0:], it0)
        S_std = S_std / np.sqrt(n_corr)
        
        print(f"\t- {(label_print + ': a') if len(label_print)>0 else 'A'}utocorr time: {icor:.1f} -> {n_corr:.1f} samples -> {S_mean:.2e} +-{S_std:.2e}")
        
    else:
        # Multi-dimensional case: flatten all dimensions except the last one
        shape_orig = S.shape[:-1]  # Original shape without time dimension
        S_reshaped = S.reshape(-1, S.shape[-1])  # Flatten to (n_series, n_time)
        
        n_series = S_reshaped.shape[0]
        n_corr = np.zeros(n_series)
        icor = np.zeros(n_series)
        
        # Calculate correlation for each flattened time series
        for i in range(n_series):
            n_corr[i], icor[i] = grab_ncorrelation(S_reshaped[i, it0:], it0)
        
        # Reshape correlation arrays back to original shape (without time dimension)
        n_corr = n_corr.reshape(shape_orig)
        icor = icor.reshape(shape_orig)
        
        # Apply correlation correction to standard deviation
        S_std = S_std / np.sqrt(n_corr)

        # Print results - handle different dimensionalities
        if S.ndim == 2:
            # 2D case: print each series
            for i in range(S.shape[0]):
                print(f"\t- {(label_print + f'_{i}: a') if len(label_print)>0 else 'A'}utocorr: {icor[i]:.1f} -> {n_corr[i]:.1f} samples -> {S_mean[i]:.2e} +-{S_std[i]:.2e}")
        else:
            # Higher dimensional case: print summary statistics
            print(f"\t- {(label_print + ': a') if len(label_print)>0 else 'A'}utocorr time: {icor.mean():.1f}±{icor.std():.1f} -> {n_corr.mean():.1f}±{n_corr.std():.1f} samples -> shape {S_mean.shape}")

    return S_mean, S_std


def _cross_phase(f1, f2):
    """
    Calculate the cross-phase between two complex signals.
    
    Parameters:
    f1, f2 : np.ndarray
        Complex signals (e.g., fluctuations).
        
    Returns:
    np.ndarray
        Cross-phase in radians.
    """
    return np.angle(f1 * np.conj(f2))

