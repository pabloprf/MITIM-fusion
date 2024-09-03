import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

class eped_beat(beat):

    def __init__(self, maestro_instance, folder_name = None):
        super().__init__(maestro_instance, beat_name = 'eped', folder_name = folder_name)

    def prepare(self, nn_location = None, norm_location = None, neped_20 = None, BetaN = None, Tesep_keV = None, nesep_20 = None, **kwargs):
        ''' 
        EPED beat may receive the following parameters: neped_20, BetaN, Tesep_keV, nesep_20.
        If they are not provided, they will be taken from the profiles_current.
        '''

        self.nn = NNtools.eped_nn(type='tf')
        nn_location = IOtools.expandPath(nn_location)
        norm_location = IOtools.expandPath(norm_location)

        self.nn.load(nn_location, norm=norm_location)

        # Parameters to run EPED with instead of those from the profiles
        self.neped_20 = neped_20
        self.BetaN = BetaN
        self.Tesep_keV = Tesep_keV
        self.nesep_20 = nesep_20 

        self._inform()

    def run(self, **kwargs):

        os.system(f'cp {self.initialize.folder}/input.gacode {self.folder}/input.gacode')

        # -------------------------------------------------------
        # Run the NN
        # -------------------------------------------------------

        eped_results = self._run(loopBetaN = 1)

        # -------------------------------------------------------
        # Save stuff
        # -------------------------------------------------------

        np.save(f'{self.folder_output}/eped_results.npy', eped_results)

        self.rhotop = eped_results['rhotop']

    def _run(self, loopBetaN = 1):

        # -------------------------------------------------------
        # Grab inputs from profiles_current
        # -------------------------------------------------------

        Ip = self.profiles_current.profiles['current(MA)'][0]
        Bt = self.profiles_current.profiles['bcentr(T)'][0]
        R = self.profiles_current.profiles['rcentr(m)'][0]
        a = self.profiles_current.derived['a']
        zeff = self.profiles_current.derived['Zeff_vol'] #TODO: Use pedestal Zeff

        '''
        -----------------------------------------------------------
        Grab inputs from profiles_current if not available
        -----------------------------------------------------------
            - kappa and delta can be provided via inform() from a previous geqdsk! which is a better near separatrix descriptor
            - beta_N and ne_top can be provided as input to prepare(), recommended in first EPED beat
            - tesep and nesep can be provided as input to prepare(), recommended in first EPED beat to define the profiles "forever"
        '''

        # Check if neped_20 is already defined by the prepare() method (e.g. in first beat) or via inform() (e.g. from a previous EPED beat)
        if self.neped_20 is None:
            # If not, using simply the density at rho = 0.95
            self.neped_20 = np.interp(0.95,self.profiles_current.profiles['rho(-)'],self.profiles_current.profiles['ne(10^19/m^3)'])*1E-1

        neped_20 = self.neped_20

        kappa995 = self.profiles_current.derived['kappa995']
        delta995 = self.profiles_current.derived['delta995']
        BetaN = self.profiles_current.derived['BetaN']
        Tesep_keV = self.profiles_current.profiles['te(keV)'][-1]
        nesep_20 = self.profiles_current.profiles['ne(10^19/m^3)'][-1]*0.1
        
        if 'kappa995' in self.__dict__ and self.kappa995 is not None:     kappa995 = self.kappa995
        if 'delta995' in self.__dict__ and self.delta995 is not None:     delta995 = self.delta995
        if "BetaN" in self.__dict__ and self.BetaN is not None:           BetaN = self.BetaN
        if "Tesep_keV" in self.__dict__ and self.Tesep_keV is not None:   Tesep_keV = self.Tesep_keV
        if "nesep_20" in self.__dict__ and self.nesep_20 is not None:     nesep_20 = self.nesep_20

        nesep_ratio = nesep_20 / neped_20

        print('\n\t- Running EPED with:')
        print(f'\t\t- Ip: {Ip:.2f} MA')
        print(f'\t\t- Bt: {Bt:.2f} T')
        print(f'\t\t- R: {R:.2f} m')
        print(f'\t\t- a: {a:.2f} m')
        print(f'\t\t- kappa995: {kappa995:.3f}')
        print(f'\t\t- delta995: {delta995:.3f}')
        print(f'\t\t- neped: {neped_20*10:.2f} 10^19 m^-3')
        print(f'\t\t- zeff: {zeff:.2f}')
        print(f'\t\t- tesep: {Tesep_keV*1E3:.1f} eV')
        print(f'\t\t- nesep_ratio: {nesep_ratio:.2f}')

        # -------------------------------------------------------
        # Run NN
        # -------------------------------------------------------

        BetaNs, ptop_kPas, wtop_psipols  = [], [], []
        for i in range(loopBetaN):
            print(f'\t\t- BetaN: {BetaN:.2f}')

            ptop_kPa, wtop_psipol = self.nn(Ip, Bt, R, a, kappa995, delta995, neped_20*10, BetaN, zeff, tesep=Tesep_keV* 1E3,nesep_ratio=nesep_ratio)

            BetaNs.append(BetaN)
            ptop_kPas.append(ptop_kPa)
            wtop_psipols.append(wtop_psipol)

            # -------------------------------------------------------
            # Produce relevant quantities
            # -------------------------------------------------------

            # psi_pol to rhoN
            rhotop = np.interp(1-wtop_psipol,self.profiles_current.derived['psi_pol_n'],self.profiles_current.profiles['rho(-)'])
            rhoped = np.interp(1-2*wtop_psipol/3,self.profiles_current.derived['psi_pol_n'],self.profiles_current.profiles['rho(-)'])

            # Find ne at the top
            # basically, we are finding Ytop such that the functional form goes through the Yped and Ysep
            # this technically doesn't need to be done after the first time EPED is run, but I'm doing it now for completeness
            pedestal_profile = lambda x, Y: FunctionalForms.pedestal_tanh(Y, 
                                                            nesep_20, 
                                                            1-rhotop, 
                                                            x=x
                                                            )[1]

            n0, _ = curve_fit(pedestal_profile, [rhoped], [neped_20])
            netop_20 = n0[0]

            # Find factor to account that it's not a pure plasma
            n = self.profiles_current.derived['ni_thrAll']/self.profiles_current.profiles['ne(10^19/m^3)']
            factor = 1 + np.interp(rhotop, self.profiles_current.profiles['rho(-)'], n )

            # Temperature from pressure, assuming Te=Ti
            Ttop_keV = (ptop_kPa*1E3) / (1.602176634E-19 * factor * netop_20 * 1e20) * 1E-3 #TODO: Relax this assumption and allow TiTe_ratio as input

            # -------------------------------------------------------
            # Put into profiles #TODO: This should be looped with the NN evaluation to find the self-consisent betaN with the current profiles
            # -------------------------------------------------------

            self.profiles_output = copy.deepcopy(self.profiles_current)

            x = self.profiles_current.profiles['rho(-)']
            xp = rhotop
            xp_old = self.rhotop if 'rhotop' in self.__dict__ else 0.9

            self.profiles_output.profiles['te(keV)'] = scale_profile_by_stretching(x,self.profiles_output.profiles['te(keV)'],xp,Ttop_keV,xp_old)

            self.profiles_output.profiles['ti(keV)'][:,0] = scale_profile_by_stretching(x,self.profiles_output.profiles['ti(keV)'][:,0],xp,Ttop_keV,xp_old)
            self.profiles_output.makeAllThermalIonsHaveSameTemp()

            self.profiles_output.profiles['ne(10^19/m^3)'] = scale_profile_by_stretching(x,self.profiles_output.profiles['ne(10^19/m^3)'],xp,netop_20*1E1,xp_old)
            self.profiles_output.enforceQuasineutrality()

            # ---------------------------------
            # Re-derive
            # ---------------------------------

            self.profiles_output.deriveQuantities()

            BetaN = self.profiles_output.derived['BetaN']

        if loopBetaN > 1:
            print('\t- Looping over BetaN:')
            print(f'\t\t* BetaN: {BetaNs}')
            print(f'\t\t* ptop_kPa: {ptop_kPas}')
            print(f'\t\t* wtop_psipol: {wtop_psipols}')

        # ---------------------------------
        # Store
        # ---------------------------------

        eped_results = {
            'ptop_kPa': ptop_kPa,
            'wtop_psipol': wtop_psipol,
            'Ttop_keV': Ttop_keV,
            'netop_20': netop_20,
            'neped_20': neped_20,
            'nesep_20': nesep_20,
            'rhotop': rhotop,
            'Tesep_keV': Tesep_keV,
        }

        for key in eped_results:
            print(f'\t\t- {key}: {eped_results[key]}')

        return eped_results

    def finalize(self, **kwargs):
        
        self.profiles_output.writeCurrentStatus(file=f"{self.folder_output}/input.gacode")

    def merge_parameters(self):
        # EPED beat does not modify the profiles grid or anything, so I can keep it fine
        pass
    
    def grab_output(self):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:

            loaded_results =  np.load(f'{self.folder_output}/eped_results.npy', allow_pickle=True).item()

            profiles = PROFILEStools.PROFILES_GACODE(f'{self.folder_output}/input.gacode') if isitfinished else None
            
        else:

            loaded_results = None
            profiles = None

        return loaded_results, profiles

    def plot(self,  fn = None, counter = 0, full_plot = True):

        fig = fn.add_figure(label='EPED', tab_color=5)
        axs = fig.subplot_mosaic(
            """
            ABCDH
            AEFGI
            """
        )
        axs = [ ax for ax in axs.values() ]

        loaded_results, profiles = self.grab_output()

        profiles_current = PROFILEStools.PROFILES_GACODE(f'{self.folder}/input.gacode')

        profiles_current.plotRelevant(axs = axs, color = 'b', label = 'orig')
        
        if loaded_results is not None:
            profiles.plotRelevant(axs = axs, color = 'r', label = 'EPED')

            axs[1].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[1].axhline(loaded_results['Ttop_keV'], color='k', ls='--',lw=2)

            axs[2].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[2].axhline(loaded_results['netop_20'], color='k', ls='--',lw=2)

            axs[3].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[3].axhline(loaded_results['ptop_kPa']*1E-3, color='k', ls='--',lw=2)

        GRAPHICStools.adjust_figure_layout(fig)

        msg = '\t\t- Plotting of EPED beat done'

        return msg

    def finalize_maestro(self):

        self.maestro_instance.final_p = self.profiles_output
        
        final_file = f'{self.maestro_instance.folder_output}/input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

    # --------------------------------------------------------------------------------------------
    # Additional EPED utilities
    # --------------------------------------------------------------------------------------------
    def _inform(self):

        # From a previous EPED beat
        if 'neped_20' in self.maestro_instance.parameters_trans_beat:
            self.neped_20 = self.maestro_instance.parameters_trans_beat['neped_20']
            print(f"\t\t- Using previous neped_20: {self.neped_20}")

        # From a geqdsk initialization
        if 'kappa995' in self.maestro_instance.parameters_trans_beat:
            self.kappa995 = self.maestro_instance.parameters_trans_beat['kappa995']
            print(f"\t\t- Using previous kappa995: {self.kappa995}")
        
         # From a geqdsk initialization
        if 'delta995' in self.maestro_instance.parameters_trans_beat:
            self.delta995 = self.maestro_instance.parameters_trans_beat['delta995']
            print(f"\t\t- Using previous delta995: {self.delta995}")

    def _inform_save(self, eped_output = None):

        if eped_output is None:
            eped_output, _ = self.grab_output()

        self.maestro_instance.parameters_trans_beat['neped_20'] = eped_output['neped_20']

        print('\t\t- neped_20 saved for future beats')



def scale_profile_by_stretching(x,y,xp,yp,xp_old, plotYN=False):
    '''
    This code keeps the separatrix fixed, moves the top of the pedestal, fits pedestal and stretches the core
    xp: top of the pedestal
    '''

    # Fit new pedestal
    _, yped = FunctionalForms.pedestal_tanh(yp, y[-1], 1-xp, x=x)

    # Find old core
    ibc = np.argmin(np.abs(x-xp_old))
    xcore_old = x[:ibc+1]
    ycore_old = y[:ibc+1]

    # Find extension of new core
    ibc = np.argmin(np.abs(x-xp))
    xcore = x[:ibc+1]

    # Scale core
    ycore_new = ycore_old * yped[ibc] / ycore_old[-1]

    # Stretch old core into the new extension
    x_core_old_mod = xcore_old * xcore[-1] / xcore_old[-1]
    ycore_new = np.interp(xcore,x_core_old_mod,ycore_new)

    # Merge
    ynew = copy.deepcopy(y)
    ynew[:ibc+1] = ycore_new
    ynew[ibc+1:] = yped[ibc+1:]

    if plotYN:
        fig, ax = plt.subplots()
        ax.plot(x,y,'-o',color='b', label='old')
        ax.axvline(x=xp_old,color='b',ls='--')
        ax.plot(x,ynew,'-o',color='r',label='new')
        ax.axvline(x=xp,color='r',ls='--')
        ax.axhline(y=yp,color='r',ls='--')
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_xlim([0,1]); ax.set_ylim(bottom=0)
        ax.legend()
    
        plt.show()


    return ynew
