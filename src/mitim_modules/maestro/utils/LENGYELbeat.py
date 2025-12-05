import os
import numpy as np
import copy
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.eped_tools import EPEDtools
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from mitim_modules.powertorch.utils import CALCtools
from mitim_tools.simulation_tools.physics.LENGYELtools import Lengyel
from IPython import embed

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function
from sympy import im

def element_to_lengyel(symbol):
    
    import periodictable as pt
    e = pt.elements.symbol(symbol)                      # 'W'
    
    name = e.name                                       # 'tungsten'
    charge = e.number                                   # 74
    mass = e.mass                                       # 183.84
    
    return name[0].upper() + name[1:].lower(), charge, mass   # 'Tungsten', 74, 183.84
    
class lengyel_beat(beat):

    def __init__(self, maestro_instance, folder_name = None):
        super().__init__(maestro_instance, beat_name = 'lengyel', folder_name = folder_name)

    def prepare(self, *args, radas_dir = None, seed_impurity_species = None, fixed_impurity_species = None, rhotop=None, **kwargs):

        self.rhotop = rhotop

        if radas_dir is not None:
            radas_dir_env = radas_dir
        else:
            radas_dir_env = os.getenv("RADAS_DIR")
        
        print('\t- Using provided RADAS_DIR for Lengyel beat preparation:', radas_dir_env)
        
        # Initialize Lengyel object
        self.l = Lengyel()

        # Use seed impurity species from maestro namelist        
        seed_impurity_symbol = seed_impurity_species["name"]
        seed_impurity_ratio_sep_top = seed_impurity_species["ratio_sep_top"]
        seed_impurity_name, seed_impurity_Z, seed_impurity_A = element_to_lengyel( seed_impurity_symbol )

        # High-Z impurity search
        fixed_impurity_symbol = fixed_impurity_species
        fixed_impurity_name, fixed_impurity_Z, fixed_impurity_A = element_to_lengyel( fixed_impurity_symbol )
        
        try:
            i_W = np.where(self.profiles_current.profiles['name']==fixed_impurity_symbol)[0][0]
        except IndexError:
            raise ValueError(f"[MAESTRO][LENGYELbeat] The high-Z impurity species '{fixed_impurity_symbol}' was not found in the input.gacode profiles; please ensure it is present to keep its concentration fixed during the Lengyel beat.")
        
        fixed_impurity_weights = self.profiles_current.derived['fi_vol'][i_W]

        # Prepare Lengyel with default inputs and changes from GACODE
        self.l.prep(
            radas_dir = radas_dir_env,
            input_gacode = self.profiles_current,
            )

        # ----------------------------------------------------
        # To pass to the run
        # ----------------------------------------------------
        
        self.lengyel_args = {
            'seed_impurity_species': [ seed_impurity_name ],
            'seed_impurity_weights': [ 1.0 ],
            'fixed_impurity_species': [fixed_impurity_name],
            'fixed_impurity_weights': [fixed_impurity_weights]
        }
        
        # ----------------------------------------------------
        # Other impurity information for post-processing
        # ----------------------------------------------------
        self.seed_impurity_enrichment = seed_impurity_ratio_sep_top
        
        self.seed_impurity_symbol = seed_impurity_symbol
        self.seed_impurity_Z = seed_impurity_Z
        self.seed_impurity_A = seed_impurity_A
        
        self.fixed_impurity_symbol = fixed_impurity_symbol
        
        self._inform()

    def run(self, *args, **kwargs):
        
        # Run Lengyel standalone
        self.l.run(
            self.folder,
            cold_start=True, # It is so cheap that, if I have come to the run() command, I'll just repeat
            **self.lengyel_args
            )
        
        # Grab important parameters from the inputs
        impurity_name = self.lengyel_args['seed_impurity_species'][0] # Assume only one seed impurity in Lengyel run
        impurity_symbol = self.seed_impurity_symbol
        impurity_Z = self.seed_impurity_Z
        impurity_A = self.seed_impurity_A
        
        # Grab important parameters from the outputs
        Tesep = float(self.l.results['separatrix_electron_temp'].split()[0])*1E-3
        
        fZ_sep = self.l.results['impurity_fraction']['seed_impurity'][impurity_name] 
        fZ_top = fZ_sep / self.seed_impurity_enrichment
        
        # ------------------------------------------------
        # Modify input.gacode
        # ------------------------------------------------
        print(f'\t- Applying Lengyel outputs to profiles:')
        p = copy.deepcopy(self.profiles_current)
                   
        # Modify temperature profiles                     
        _modify_temperatures(p, Tesep, self.rhotop)
        
        # Modify impurity density profile
        
        # Find impurity index: if I have transp beat before Lengyel, I know the order of the impuritites
        if "impurity_order_transp" in self.maestro_instance.parameters_trans_beat:
            # Impurities ordered as in TRANSP
            impurities_in_transp =  list(self.maestro_instance.parameters_trans_beat['impurity_order_transp'].keys())
        
            # Do not consider the high-Z impurity, which is fixed
            impurities_in_transp.remove( self.fixed_impurity_symbol )
            
            # Get index of the FIRST (unique for now) seed impurity in TRANSP
            i_Z = self.maestro_instance.parameters_trans_beat['impurity_order_transp'][impurities_in_transp[0]]
        # Else, assume impurity is in position 3 (after D and He)
        else:
            i_Z = 3
            print(f"\t\t- No impurity order from TRANSP beat found, assuming impurity '{impurity_symbol}' is in position #{i_Z} in input.gacode", typeMsg='w')
        
        _modify_impurity_density(p, impurity_symbol, impurity_Z, impurity_A, fZ_sep, fZ_top, self.rhotop, i_Z = i_Z)
        
        # Enforce quasineutrality
        p.enforceQuasineutrality()
        
        # Check if the plasma just had too much impurity
        if p.profiles['ni(10^19/m^3)'].min() < 0:
            raise ValueError(f"[MAESTRO][LENGYELbeat] After applying Lengyel outputs, negative ions densities were found in input.gacode.lengyel; please check the impurity concentrations and/or the Lengyel settings.")
        
        # Write modified input.gacode.lengyel
        p.write_state(file=self.folder / 'input.gacode.lengyel')

    def finalize(self, *args, **kwargs):
        
        self.profiles_output = PROFILEStools.gacode_state(self.folder / 'input.gacode.lengyel')

        self.profiles_output.write_state(file=self.folder_output / 'input.gacode')

    # -----------------------------------------------------------------------------------------------------------------------
    # MAESTRO interface
    # -----------------------------------------------------------------------------------------------------------------------

    def _inform(self, *args, **kwargs):
        
        # From a previous EPED beat, grab the rhotop
        if 'rhotop' in self.maestro_instance.parameters_trans_beat:
            self.rhotop = self.maestro_instance.parameters_trans_beat['rhotop']
            print(f"\t\t- Using previous rhotop: {self.rhotop}")
            
    def _inform_save(self, *args, **kwargs):
        
        # If I have run Lengyel, I cannot reuse surrogate data #TODO: Maybe not always true?
        self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] = None 

def _modify_temperatures(p, Tesep, rhotop):
    
    print(f'\t\t* Setting electron and ion temperature at separatrix to {Tesep*1E3:.1f} eV')
    
    if rhotop is None:
        print('\t\t- No rhotop available at this beat, scaling the entire profile uniformly')
        p.profiles['te(keV)'] *= Tesep / p.profiles['te(keV)'][-1]
        p.profiles['ti(keV)'] *= Tesep / p.profiles['ti(keV)'][-1, :]
    else:
        print(f'\t\t- Using rhotop = {rhotop:.3f} to scale temperature profiles only from rhotop to the new separatrix value')
        
        _scale_quadratic(p, p.profiles['te(keV)'], rhotop, Tesep)
        for ion in range(len(p.profiles['ti(keV)'][0, :])):
            _scale_quadratic(p, p.profiles['ti(keV)'][:,ion], rhotop, Tesep)


def _modify_impurity_density(p, impurity_name, impurity_Z, impurity_A, fZ_sep, fZ_top, rhotop, i_Z):

    print(f'\t\t* Setting impurity "{impurity_name}" (Z={impurity_Z}, A={impurity_A}), at ion position #{i_Z}, density at separatrix to {fZ_top = :.1e}')
    
    p.profiles['z'][i_Z] = impurity_Z
    p.profiles['mass'][i_Z] = impurity_A
    p.profiles['name'][i_Z] = impurity_name[:2] 
    
    if rhotop is None:
        print('\t\t- No rhotop available at this beat, scaling the entire impurity density profile uniformly by the top (after applying enrichment) value, exact from ne profile')
        p.profiles['ni(10^19/m^3)'][:, i_Z] = fZ_top * p.profiles['ne(10^19/m^3)']
    
    else:
        print(f'\t\t- Using rhotop = {rhotop:.3f} to scale impurity density profiles')
        
        # First, scale impurity density profile entirely by the desired top value
        nZ_top_new = fZ_top * p.profiles['ne(10^19/m^3)'][-1]
        ix = np.argmin(np.abs(p.profiles['rho(-)'] - rhotop))
        nZ_top_old = p.profiles['ni(10^19/m^3)'][ix, i_Z]
        
        p.profiles['ni(10^19/m^3)'][:, i_Z] *= nZ_top_new / nZ_top_old
        
        # Then modify the edge such that it goes to nZ_sep (Apply quadratic scaling from rhotop to separatrix)
        nZ_sep = fZ_sep * p.profiles['ne(10^19/m^3)'][-1]
        _scale_quadratic(p, p.profiles['ni(10^19/m^3)'][:, i_Z], rhotop, nZ_sep)

def _scale_quadratic(p, var, rhotop, val_sep, plotYN=False):
    '''
    I use a quadratic scaling from rhotop to separatrix instead of a linear one, to have a smoother transition.
    '''

    var_orig = copy.deepcopy(var)

    ix = np.argmin(np.abs(p.profiles['rho(-)'] - rhotop))
    factor_array = np.ones_like( p.profiles['rho(-)'] )

    # Create a non-linear scaling that starts,io9slowly and accelerates
    n_points = len(p.profiles['rho(-)']) - ix
    t = np.linspace(0, 1, n_points)  # Normalized parameter from 0 to 1
    factor_array[ix:] = 1.0 + (val_sep / var_orig[-1] - 1.0) * t**2  # Quadratic profile
    
    var *= factor_array
    
    if plotYN:
        import matplotlib.pyplot as plt
        fig, axs= plt.subplots(nrows=2)
        ax = axs[0]
        ax.plot( p.profiles['rho(-)'], var_orig, 'o-', label='Original' )
        ax.plot( p.profiles['rho(-)'], var, 'o-', label='Modified' )
        ax = axs[1]
        ax.plot( p.profiles['rho(-)'], var/var_orig, 'o-', label='Scaling factor' )
        plt.show()
        
        embed()
        