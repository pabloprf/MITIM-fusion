import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
    def __init__(self, folder, tmin=0.0, minimal=False):

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
            self.linear = True
        else:
            self.linear = False

        self.cgyrodata.getflux(cflux='auto')
        self.cgyrodata.getnorm("elec")
        self.cgyrodata.getgeo()
        self.cgyrodata.getxflux()

        # Understand positions
        self.electron_flag = np.where(self.cgyrodata.z == -1)[0][0]
        self.all_flags = np.arange(0, len(self.cgyrodata.z), 1)
        self.ions_flags = self.all_flags[self.all_flags != self.electron_flag]

        self.all_names = [f"{gacodefuncs.specmap(self.cgyrodata.mass[i],self.cgyrodata.z[i])}({self.cgyrodata.z[i]},{self.cgyrodata.mass[i]:.1f})" for i in self.all_flags]

        self.fields = np.arange(self.cgyrodata.n_field)

        self.aLTi = self.cgyrodata.dlntdr[0]
        self.aLTe = self.cgyrodata.dlntdr[self.electron_flag]
        self.aLne = self.cgyrodata.dlnndr[self.electron_flag]
    

        # ************************
        # Normalization
        # ************************
        
        self.t = self.cgyrodata.tnorm
        self.ky = self.cgyrodata.kynorm
        self.kx = self.cgyrodata.kxnorm
        self.theta = self.cgyrodata.theta
        
        if self.cgyrodata.theta_plot == 1:
            self.theta_stored = np.array([0.0])
        else:
            self.theta_stored = np.array([-1+2.0*i/self.cgyrodata.theta_plot for i in range(self.cgyrodata.theta_plot)])
        
        self.Qgb = self.cgyrodata.q_gb_norm
        self.Ggb = self.cgyrodata.gamma_gb_norm
        
        self.artificial_rhos_factor = self.cgyrodata.rho_star_norm / self.cgyrodata.rhonorm

        self._process_linear()
        
        if not minimal:
            self.cgyrodata.getbigfield()

            if 'kxky_phi' in self.cgyrodata.__dict__:
                self._process_fluctuations()
            else:
                print('\t- No fluctuations found in CGYRO data, skipping fluctuation processing and will not be able to plot default Notebook', typeMsg='w')
        else:
            print('\t- Minimal mode, skipping fluctuations processing', typeMsg='i')
            
        self._process_fluxes()        
        self._saturate_signals()

    def _process_linear(self):
        
        self.f = self.cgyrodata.fnorm[0,:,:]                # (ky, time)
        self.g = self.cgyrodata.fnorm[1,:,:]                # (ky, time)

        # Ballooning Modes (complex eigenfunctions)
        if 'phib' in self.cgyrodata.__dict__:
            self.phi_ballooning = self.cgyrodata.phib       # (ball, time)
            self.apar_ballooning = self.cgyrodata.aparb     # (ball, time)
            self.bpar_ballooning = self.cgyrodata.bparb     # (ball, time)
            self.theta_ballooning = self.cgyrodata.thetab   # (ball, time)


    def _process_fluctuations(self):
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
        variables = ['phi', 'apar', 'bpar', 'ne', 'ni_all', 'ni', 'Te', 'Ti', 'Ti_all']
        for var in variables:
            if var in self.__dict__:
                
                # Make sure I go to the real units for all of them *******************
                self.__dict__[var] = self.__dict__[var] * self.artificial_rhos_factor
                # ********************************************************************

                # Sum over radial modes
                self.__dict__[var+'_rms_sumnr'] = (abs(self.__dict__[var][:,:,:])**2).sum(axis=(0))**0.5                # (ntoroidal, time)
                 
                # Sum over radial modes AND separate n=0 and n>0 (sum) modes
                self.__dict__[var+'_rms_sumnr_n0'] = (abs(self.__dict__[var][:,0,:])**2).sum(axis=0)**0.5               # (time)
                self.__dict__[var+'_rms_sumnr_sumn1'] = (abs(self.__dict__[var][:,1:,:])**2).sum(axis=(0,1))**0.5    # (time)
                
                # Sum over radial modes and toroidal modes
                self.__dict__[var+'_rms_sumnr_sumn'] = (abs(self.__dict__[var][:,:,:])**2).sum(axis=(0,1))**0.5         # (time)
               
                # Separate n=0, n>0 (sum) modes, and all n (sum) modes
                self.__dict__[var+'_rms_n0'] = (abs(self.__dict__[var][:,0,:])**2)**0.5                         # (nradial,time) 
                self.__dict__[var+'_rms_sumn1'] = (abs(self.__dict__[var][:,1:,:])**2).sum(axis=(1))**0.5       # (nradial,time)
                self.__dict__[var+'_rms_sumn'] = (abs(self.__dict__[var][:,:,:])**2).sum(axis=(1))**0.5         # (nradial,time)

        # Cross-phases
        self.nT = _cross_phase(self.ne, self.Te) * 180/ np.pi  # (nradial, ntoroidal, time)
        self.nT_kx0 = self.nT[np.argmin(np.abs(self.kx)),:,:]  # (ntoroidal, time)

        self.phiT = _cross_phase(self.phi, self.Te) * 180/ np.pi  # (nradial, ntoroidal, time)
        self.phiT_kx0 = self.phiT[np.argmin(np.abs(self.kx)),:,:]
        
        self.phin = _cross_phase(self.phi, self.ne) * 180/ np.pi  # (nradial, ntoroidal, time)
        self.phin_kx0 = self.phin[np.argmin(np.abs(self.kx)),:,:]

    def _process_fluxes(self):
        
        # ************************
        # Fluxes
        # ************************
        
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
        
        # Convert to MW/m^2     
        self.QeMWm2 = self.Qe * self.Qgb
        self.QiMWm2 = self.Qi * self.Qgb
        self.Qi_allMWm2 = self.Qi_all * self.Qgb
        
    def _saturate_signals(self):
        
        # ************************
        # Saturated
        # ************************
        
        flags = [
            'Qe',
            'QeMWm2',
            'Qe_ky',
            'Qi',
            'QiMWm2',
            'Qi_ky',
            'Ge',
            'Ge_ky',
            'Qe_ES',
            'Qi_ES',
            'Ge_ES',
            'Qe_EM',
            'Qi_EM',
            'Ge_EM',
            'g',
            'f',
            'phi_rms_sumnr',
            'apar_rms_sumnr',
            'bpar_rms_sumnr',
            'ne_rms_sumnr',
            'Te_rms_sumnr',
            'phi_rms_n0',
            'phi_rms_sumn1',
            'phi_rms_sumn',
            'apar_rms_n0',
            'apar_rms_sumn1',
            'apar_rms_sumn',
            'bpar_rms_n0',
            'bpar_rms_sumn1',
            'bpar_rms_sumn',
            'ne_rms_n0',
            'ne_rms_sumn1',
            'ne_rms_sumn',
            'Te_rms_n0',
            'Te_rms_sumn1',
            'Te_rms_sumn',
            'nT_kx0',
            'phiT_kx0',
            'phin_kx0',
        ]
        
        for iflag in flags:
            if iflag in self.__dict__:
                Qm, Qstd = apply_ac(
                        self.t,
                        self.__dict__[iflag],
                        tmin=self.tmin,
                        label_print=iflag,
                        print_msg=iflag in ['Qi', 'Qe', 'Ge'],
                        #debug=iflag == 'Qi'
                        )
                # Qm, Qstd = apply_ac_fixed_signal(
                #         self.t,
                #         self.__dict__[iflag],
                #         self.phi_rms_sumnr_sumn,
                #         tmin=self.tmin,
                #         label_print=iflag,
                #         print_msg=iflag == 'Qi'
                #         )  
                    
                self.__dict__[iflag+'_mean'] = Qm
                self.__dict__[iflag+'_std'] = Qstd
      
def _grab_ncorrelation(t, S, it0, debug=False):
    # Calculate the autocorrelation function
    i_acf = sm.tsa.acf(S, nlags=len(S))

    if i_acf.min() > 1/np.e:
        print("Autocorrelation function does not reach 1/e, will use full length of time series for n_corr.", typeMsg='w')

    # Calculate how many time slices make the autocorrelation function is 1/e (conventional decorrelation level)
    icor = np.abs(i_acf-1/np.e).argmin()
    
    # Define number of samples
    n_corr = ( len(t) - it0 ) / ( 3.0 * icor ) #Define "sample" as 3 x autocor time
    
    if debug:
        fig, ax = plt.subplots()
        ax.plot(i_acf, '-o', label='ACF')
        ax.axhline(1/np.e, color='r', linestyle='--', label='1/e')
        ax.set_xlabel('Lags'); ax.set_xlim([0, icor+20])
        ax.set_ylabel('ACF')
        ax.legend()
        plt.show()
        embed()
    
    return n_corr, icor
      
def apply_ac_fixed_signal(t, S, Scorr, tmin = 0, label_print = '', print_msg = False, debug=False):

    it0 = np.argmin(np.abs(t - tmin))
    
    # Calculate the mean and std of the signal after tmin (last dimension is time)
    S_mean = np.mean(S[..., it0:], axis=-1)
    S_std = np.std(S[..., it0:], axis=-1)

    # 1D case: single time series
    n_corr, icor = _grab_ncorrelation(t, Scorr[it0:], it0, debug=debug)
    S_std = S_std / np.sqrt(n_corr)
    
    if print_msg:
        print(f"\t- {(label_print + ': a') if len(label_print)>0 else 'A'}utocorr time: {icor:.1f} -> {n_corr:.1f} samples")# -> {S_mean:.2e} +-{S_std:.2e}")
    
    return S_mean, S_std
                

def apply_ac(t, S, tmin = 0, label_print = '', print_msg = False, debug=False):
    
    it0 = np.argmin(np.abs(t - tmin))
    
    # Calculate the mean and std of the signal after tmin (last dimension is time)
    S_mean = np.mean(S[..., it0:], axis=-1)
    S_std = np.std(S[..., it0:], axis=-1) # To follow NTH convention

    if S.ndim == 1:
        # 1D case: single time series
        n_corr, icor = _grab_ncorrelation(t, S[it0:], it0, debug=debug)
        S_std = S_std / np.sqrt(n_corr)
        
        if print_msg:
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
            n_corr[i], icor[i] = _grab_ncorrelation(t, S_reshaped[i, it0:], it0, debug=debug)
        
        # Reshape correlation arrays back to original shape (without time dimension)
        n_corr = n_corr.reshape(shape_orig)
        icor = icor.reshape(shape_orig)
        
        # Apply correlation correction to standard deviation
        S_std = S_std / np.sqrt(n_corr)

        # Print results - handle different dimensionalities
        if print_msg:
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

