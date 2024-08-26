import os
import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

class eped_beat(beat):

    def __init__(self, maestro_instance):
        super().__init__(maestro_instance, beat_name = 'eped')

    def prepare(self, nn_location, norm_location, netop_20 = None):

        self.nn = NNtools.eped_nn(type='tf')
        nn_location = IOtools.expandPath(nn_location)
        norm_location = IOtools.expandPath(norm_location)

        self.nn.load(nn_location, norm=norm_location)

        self.netop = netop_20

        self._inform()

    def run(self, **kwargs):

        os.system(f'cp {self.initialize.folder}/input.gacode {self.folder}/input.gacode')

        # -------------------------------------------------------
        # Grab inputs from profiles_current and run the NN
        # -------------------------------------------------------

        if self.netop is None:
            # Trying to get from the previous run
            if 'rhotop' in self.__dict__:
                print(f"\t\t- Using previous rhotop: {self.rhotop}")
            else:
                self.rhotop = 0.9
            self.netop = np.interp(self.rhotop,self.profiles_current.profiles['rho(-)'],self.profiles_current.profiles['ne(10^19/m^3)']) * 1E-1

        neped = self.netop/1.08

        ptop_kPa, wtop_psipol = self.nn(
            self.profiles_current.profiles['current(MA)'][0],
            self.profiles_current.profiles['bcentr(T)'][0],
            self.profiles_current.profiles['rcentr(m)'][0],
            self.profiles_current.derived['a'],
            self.profiles_current.derived['kappa995'],
            self.profiles_current.derived['delta995'],
            neped,
            self.profiles_current.derived['BetaN'],
            self.profiles_current.derived['Zeff_vol'],
            tesep=self.profiles_current.profiles['te(keV)'][-1],
            nesep_ratio=self.profiles_current.profiles['ne(10^19/m^3)'][-1]*1E-1 / neped
        )

        # -------------------------------------------------------
        # Put into profiles
        # -------------------------------------------------------

        self.profiles_output, eped_results = add_eped_pressure(copy.deepcopy(self.profiles_current), ptop_kPa, wtop_psipol, self.netop)
    
        self.rhotop = eped_results['rhotop']

        np.save(f'{self.folder_output}/eped_results.npy', eped_results)

    def finalize(self):
        
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
            axs[1].axhline(loaded_results['Ttop'], color='k', ls='--',lw=2)

            axs[2].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[2].axhline(loaded_results['netop'], color='k', ls='--',lw=2)

            axs[3].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[3].axhline(loaded_results['ptop']*1E-3, color='k', ls='--',lw=2)

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

        if 'rhotop' in self.maestro_instance.parameters_trans_beat:
            self.rhotop = self.maestro_instance.parameters_trans_beat['rhotop']
            print(f"\t\t- Using previous rhotop: {self.rhotop}")

    def _inform_save(self):

        eped_output, _ = self.grab_output()

        self.maestro_instance.parameters_trans_beat['netop'] = eped_output['netop']
        self.maestro_instance.parameters_trans_beat['rhotop'] = eped_output['rhotop']

        print('\t\t- rhotop and netop saved for future beats')

# ---------------------------------------------------------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------------------------------------------------------

def add_eped_pressure(profiles, ptop, wtop, netop):
    '''
    Inputs:
        ptop in kPa
        wtop in psi_pol
        netop in 10^20/m^3
    Notes:
        - This function modifies the profiles with the right EPED pressure
    '''

    # ---------------------------------
    # Convert
    # ---------------------------------
    
    # psi_pol to rhoN
    rhotop = np.interp(1-wtop,profiles.derived['psi_pol_n'],profiles.profiles['rho(-)'])

    # Find factor to account that it's not a pure plasma
    n = profiles.derived['ni_thrAll']/profiles.profiles['ne(10^19/m^3)']
    factor = 1 + np.interp(rhotop, profiles.profiles['rho(-)'], n )

    # Temperature from pressure, assuming Te=Ti
    Ttop = (ptop*1E3) / (1.602176634E-19 * factor * netop * 1e20) * 1E-3

    # ---------------------------------
    # Store
    # ---------------------------------

    eped_results = {
        'ptop': ptop,
        'wtop': wtop,
        'Ttop': Ttop,
        'netop': netop,
        'rhotop': rhotop
    }

    # ---------------------------------
    # Modify profiles
    # ---------------------------------
    
    ratio = Ttop / np.interp(rhotop,profiles.profiles['rho(-)'],profiles.profiles['te(keV)'])
    profiles.profiles['te(keV)'] *= ratio
    
    ratio = Ttop / np.interp(rhotop,profiles.profiles['rho(-)'],profiles.profiles['ti(keV)'][:,0])
    profiles.profiles['ti(keV)'][:,0] *= ratio
    profiles.makeAllThermalIonsHaveSameTemp()

    ratio = netop*1E1 / np.interp(rhotop,profiles.profiles['rho(-)'],profiles.profiles['ne(10^19/m^3)'])
    profiles.profiles['ne(10^19/m^3)'] *= ratio
    profiles.scaleAllThermalDensities(scaleFactor=ratio)

    # ---------------------------------
    # Re-derive
    # ---------------------------------

    profiles.deriveQuantities()

    return profiles, eped_results