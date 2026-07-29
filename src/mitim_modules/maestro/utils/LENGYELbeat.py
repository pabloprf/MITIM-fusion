import os
import numpy as np
import copy
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from mitim_tools.simulation_tools.physics.LENGYELtools import Lengyel
from IPython import embed
from mitim_modules.powertorch.utils import CALCtools

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

    def prepare(self, *args, lengyel_namelist_location = None, radas_dir = None, seed_impurity_species = None, fixed_impurity_species = None, rhotop=None, override_namelist_params = None, zeff_relaxation_factor = 1.0, zeff_floor = None, **kwargs):

        self.rhotop = rhotop

        # User overrides for the Lengyel namelist `input` block (keys as in input.lengyel.controls.yaml),
        # applied at run() time on top of the defaults and input.gacode-derived values
        self.override_namelist_params = override_namelist_params if override_namelist_params is not None else {}

        # Relaxation factor w in [0, 1] applied (in run(), below) to the Zeff update: the Zeff profile
        # actually realized is
        #     w * <Zeff this Lengyel beat would fully impose> + (1 - w) * <Zeff of the profiles input into
        #                                                                  this Lengyel beat>
        # w = 1.0 (default) reproduces the previous unrelaxed behavior (Zeff jumps straight to whatever the
        # divertor needs). Lower values slow down how fast Zeff/impurity content changes beat-to-beat, which
        # is useful because a higher core Zeff (e.g. 2.0-2.3) than strictly required for divertor protection
        # can be beneficial, and letting it swing every beat can make the workflow oscillate/degrade instead
        # of converging. The "profile input into this beat" is whatever input.gacode this beat received from
        # the immediately preceding beat (PORTALS, EPED, TRANSP, or a previous Lengyel beat), so this applies
        # identically on the first Lengyel beat of a sequence as on every subsequent one (see run(), below).
        if not (0.0 <= zeff_relaxation_factor <= 1.0):
            raise ValueError(f"[MAESTRO][LENGYELbeat] zeff_relaxation_factor must be in [0, 1], got {zeff_relaxation_factor}")
        self.zeff_relaxation_factor = zeff_relaxation_factor

        # Hard floor on the vol-avg Zeff after this beat (applied in run(), below, after the relaxation).
        # If the seed-impurity update (possibly relaxed) would leave the vol-avg Zeff below this value,
        # the seed-impurity density is scaled up just enough to meet the floor. null (default) means no
        # floor is applied. Set this to plasma.species.Zeff (the initialization Zeff in the MAESTRO
        # namelist) to prevent the Lengyel beat from ever dragging Zeff below the starting point.
        self.zeff_floor = zeff_floor

        if radas_dir is not None:
            radas_dir_env = radas_dir
        else:
            radas_dir_env = os.getenv("RADAS_DIR")
        
        print('\t- Using provided RADAS_DIR for Lengyel beat preparation:', radas_dir_env)
        
        # Initialize Lengyel object (documented lengyel_namelist_location honored;
        # template default otherwise — the key used to be silently swallowed by **kwargs)
        self.l = Lengyel(namelist_location = lengyel_namelist_location)

        # Use seed impurity species from maestro namelist        
        seed_impurity_symbol = seed_impurity_species["name"]
        seed_impurity_ratio_sep_top = seed_impurity_species["ratio_fZ_sep_top"]
        seed_impurity_edge_profile = seed_impurity_species["edge_profile"]
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
        
        self.seed_impurity_edge_profile = seed_impurity_edge_profile
        
        self.fixed_impurity_symbol = fixed_impurity_symbol
        
        self._inform()

    def run(self, *args, **kwargs):
        
        # Merge user-provided namelist overrides on top of the impurity args (overrides win on key collision)
        lengyel_inputs = {**self.lengyel_args, **self.override_namelist_params}

        # Run Lengyel standalone
        self.l.run(
            self.folder,
            cold_start=True, # It is so cheap that, if I have come to the run() command, I'll just repeat
            **lengyel_inputs
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
        
        print(f'\t- Enrichment factor applied: {self.seed_impurity_enrichment:.1f} (i.e. SOL concentration: {fZ_sep:.1e};  main plasma concentration: {fZ_top:.1e})')
        
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

        # Capture the FULL Zeff profile of the incoming plasma state (i.e. whatever the beat immediately
        # before this one -- PORTALS, EPED, TRANSP, or a previous Lengyel beat -- wrote to its input.gacode),
        # BEFORE this Lengyel beat touches anything. This is done at the Zeff level (not the seed-impurity
        # density level) so it works identically regardless of what species/charge previously occupied slot
        # i_Z, including on the very first Lengyel beat of a sequence (there is no special-casing needed).
        ne = p.profiles['ne(10^19/m^3)']
        Zeff_before = np.sum(p.profiles['ni(10^19/m^3)'] * p.profiles['z'] ** 2, axis=1) / ne

        # Contribution to Zeff from every species except the seed impurity; used by both the relaxation and
        # the floor below. Stays constant through _modify_impurity_density (only slot i_Z changes).
        other = np.arange(p.profiles['z'].shape[0]) != i_Z
        z2ni_other = np.sum(p.profiles['ni(10^19/m^3)'][:, other] * p.profiles['z'][other] ** 2, axis=1)

        _modify_impurity_density(p, impurity_symbol, impurity_Z, impurity_A, fZ_sep, fZ_top, self.rhotop, i_Z = i_Z, edge_profile=self.seed_impurity_edge_profile)

        # Relax the Zeff update: the Zeff profile actually realized is a weighted average of the profile
        # this Lengyel beat would fully impose (Zeff_after, computed from the just-applied unrelaxed update)
        # and the profile that was input into this beat (Zeff_before). w=1.0 (default) reproduces the
        # previous unrelaxed behavior exactly. See zeff_relaxation_factor in prepare().
        w = self.zeff_relaxation_factor
        if w < 1.0:
            Zeff_after = np.sum(p.profiles['ni(10^19/m^3)'] * p.profiles['z'] ** 2, axis=1) / ne
            Zeff_relaxed = w * Zeff_after + (1 - w) * Zeff_before

            # Solve for the seed-impurity density that reproduces Zeff_relaxed exactly, holding every other
            # species fixed at its current (pre-quasineutrality) value
            p.profiles['ni(10^19/m^3)'][:, i_Z] = (Zeff_relaxed * ne - z2ni_other) / (impurity_Z ** 2)

            fZ_top = float(p.profiles['ni(10^19/m^3)'][0, i_Z] / ne[0])
            print(f"\t\t* Applying Zeff relaxation factor {w:.2f}: Zeff profile is {w:.2f}*<full Lengyel update> + {1-w:.2f}*<profile input into this beat> (resulting core '{impurity_symbol}' concentration: {fZ_top:.1e})")


        # Apply a hard floor on the vol-avg Zeff after this beat (see zeff_floor in prepare()). If the
        # seed-impurity update (possibly relaxed) would leave the vol-avg Zeff below this value, the seed-impurity density is scaled up just enough to meet the floor.
        if self.zeff_floor is not None:
            Zeff = np.sum(p.profiles['ni(10^19/m^3)'] * p.profiles['z'] ** 2, axis=1) / p.profiles['ne(10^19/m^3)']
            Zeff_vol = CALCtools.volume_integration(Zeff, p.derived["r"], p.derived["volp_geo"])[-1] / p.derived["volume"]
            max_iter = 10
            iterations = 0
            while Zeff_vol < self.zeff_floor and iterations < max_iter:
                iterations += 1
                if Zeff_vol <= 0 or np.isclose(Zeff_vol, 0.0):
                    print(f"\t\t! Invalid vol-avg Zeff ({Zeff_vol:.3g}); aborting Zeff-floor loop")
                    break
                scale_factor = self.zeff_floor / Zeff_vol
                fZ_top = float(p.profiles['ni(10^19/m^3)'][0, i_Z] * scale_factor / ne[0])
                
                # modify impurity in-place on the profile object `p` (pass `p` as first arg)
                _modify_impurity_density(p, impurity_symbol, impurity_Z, impurity_A, fZ_sep, fZ_top, self.rhotop, i_Z = i_Z, edge_profile=self.seed_impurity_edge_profile)
                p.enforce_quasineutrality()
                print(f'Enforced Quasineutrality before recalculating Zeff_vol. Zeff_vol in mitim state object is now {p.derived["Zeff_vol"]}')
                
                # Recompute Zeff and its volume average after the change
                Zeff = np.sum(p.profiles['ni(10^19/m^3)'] * p.profiles['z'] ** 2, axis=1) / p.profiles['ne(10^19/m^3)']
                Zeff_vol = CALCtools.volume_integration(Zeff, p.derived["r"], p.derived["volp_geo"])[-1] / p.derived["volume"]
                print(f"\t\t* Applied Zeff floor, manually calculated Zeff_vol now {Zeff_vol:.2f} (iteration {iterations})")
            
            if Zeff_vol < self.zeff_floor:
                print(f"\t\t! Warning: Zeff floor not reached after {max_iter} iterations: vol-avg Zeff {Zeff_vol:.2f} < floor {self.zeff_floor:.2f}")

        # Quasineutrality
        p.enforce_quasineutrality()
        print(f'Quasineutrality enforced: Zeff_vol is now {p.derived["Zeff_vol"]}')
        
        # Check if the plasma just had too much impurity
        if p.profiles['ni(10^19/m^3)'].min() < 0:
            raise ValueError(f"[MAESTRO][LENGYELbeat] After applying Lengyel outputs, negative ions densities were found in input.gacode.lengyel; please check the impurity concentrations and/or the Lengyel settings.")
        
        # Write modified input.gacode.lengyel
        p.write_state(file=self.folder / 'input.gacode.lengyel')
        
        # For the inform later
        self.impurity_lengyel = [impurity_Z, impurity_A, fZ_top]

    def finalize(self, *args, **kwargs):

        # Persist the Lengyel namelist to beat_results (copy under keep_all_files: true;
        # move otherwise, matching the EPED/PORTALS/TRANSP beat behavior).
        src_input_namelist = self.folder / 'input.lengyel.controls.yml'
        dst_input_namelist = self.folder_output / 'input.lengyel.controls.yml'
        src_output_namelist = self.folder / 'output.lengyel.results.yml'
        dst_output_namelist = self.folder_output / 'output.lengyel.results.yml'
        if src_output_namelist.exists():
            self._persist(src_output_namelist, dst_output_namelist)
        if src_input_namelist.exists():
            self._persist(src_input_namelist, dst_input_namelist)

        # On a re-invocation after a prior keep_all_files: false cleanup wiped
        # self.folder, input.gacode.lengyel is gone and folder_output already holds
        # the finalized input.gacode from the prior run. Same guard as the
        # TRANSP/EPED/PORTALS beats.
        if not (self.folder / 'input.gacode.lengyel').exists():
            self.profiles_output = PROFILEStools.gacode_state(self.folder_output / 'input.gacode')
            return

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
        
        # Store the impurity specifications
        #self.maestro_instance.parameters_trans_beat['lowZ_impurity'] = self.impurity_lengyel

def _modify_temperatures(p, Tesep, rhotop):
    
    print(f'\t\t* Setting electron and ion temperature at separatrix to {Tesep*1E3:.1f} eV')
    
    if rhotop is None:
        print('\t\t\t- No rhotop available at this beat, shifting the entire profile by a constant offset to the new separatrix value')
        p.profiles['te(keV)'] += Tesep - p.profiles['te(keV)'][-1]
        p.profiles['ti(keV)'] += Tesep - p.profiles['ti(keV)'][-1, :]
    else:
        print(f'\t\t\t- Using rhotop = {rhotop:.3f} to blend temperature profiles only from rhotop to the new separatrix value')

        _offset_quadratic(p, p.profiles['te(keV)'], rhotop, Tesep)
        for ion in range(len(p.profiles['ti(keV)'][0, :])):
            _offset_quadratic(p, p.profiles['ti(keV)'][:,ion], rhotop, Tesep)


def _modify_impurity_density(p, impurity_name, impurity_Z, impurity_A, fZ_sep, fZ_top, rhotop, i_Z, plotYN=False, edge_profile="flat"):

    print(f'\t\t* Setting impurity "{impurity_name}" (Z={impurity_Z}, A={impurity_A}), at ion position #{i_Z}, density at separatrix to {fZ_top = :.1e}')
    
    p.profiles['z'][i_Z] = impurity_Z
    p.profiles['mass'][i_Z] = impurity_A
    p.profiles['name'][i_Z] = impurity_name[:2] 
    
    # Scale entire profile
    print(f'\t\t\t- Implementing a core concentration of {fZ_top:.1e}, applied to the electron density profile')
    p.profiles['ni(10^19/m^3)'][:, i_Z] = fZ_top * p.profiles['ne(10^19/m^3)']
    
    if edge_profile == "flat":
        print(f'\t\t\t\t- Using a flat profile, with the same concentration at the separatrix and at the core')
    elif edge_profile == "linear":
        
        if rhotop is None:
            print('\t\t\t\t- No rhotop available at this beat, entire impurity profile scaled with the same factor (i.e. no different enrichment at the core and at the separatrix)')
        else:
            print(f'\t\t\t\t- Using rhotop = {rhotop:.3f} to apply a linear ramp in impurity concentration from rhotop to the separatrix, with the value at rhotop being {fZ_top:.1e} and the value at the separatrix being {fZ_sep:.1e}')
            
            ix = np.argmin(np.abs(p.profiles['rho(-)'] - rhotop))
            
            # From rhotop to separatrix, make the concentration go linearly from the value at rhotop to fZ_sep at the separatrix
            p.profiles['ni(10^19/m^3)'][ix:, i_Z] = np.linspace(fZ_top, fZ_sep, len(p.profiles['ni(10^19/m^3)'][ix:, i_Z])) * p.profiles['ne(10^19/m^3)'][ix:]

    if plotYN:
        # Debug plot to check the impurity density modification
        import matplotlib.pyplot as plt
        fig, axs= plt.subplots(nrows=2)
        ax = axs[0]
        ax.plot( p.profiles['rho(-)'], p.profiles['ne(10^19/m^3)'], 'o-', label='ne' )
        ax.plot( p.profiles['rho(-)'], p.profiles['ni(10^19/m^3)'][:, i_Z], 'o-', label=f'{impurity_name} density' )
        
        ax.legend()
        
        ax = axs[1]
        ax.plot( p.profiles['rho(-)'], p.profiles['ni(10^19/m^3)'][:, i_Z] / p.profiles['ne(10^19/m^3)'], 'o-', label=f'{impurity_name} fraction' )
        
        ax.legend()
        
        plt.show()
        
        embed()
        
def _offset_quadratic(p, var, rhotop, val_sep, plotYN=False):
    '''
    Additive quadratic blend from rhotop to the separatrix: the value at rhotop
    (pedestal top) is held fixed and a bounded offset, growing as t^2 from 0 at
    rhotop to (val_sep - var_sep) at the separatrix, is ADDED so the foot lands on
    val_sep with a smooth (zero-slope) join at rhotop.

    Additive rather than multiplicative on purpose: the correction is bounded by the
    separatrix change |val_sep - var_sep|, not proportional to the local value, so it
    cannot lift the mid-pedestal above the pedestal top and create a spurious
    temperature bump when the Lengyel separatrix temperature greatly exceeds the
    incoming one (val_sep >> var_sep). The previous multiplicative form
    (var *= 1 + (val_sep/var_sep - 1) t^2) bumped for val_sep/var_sep >~ 2.

    The additive offset never rises above the pedestal-top value. On a perfectly
    flat top (zero incoming slope at rhotop) it could leave a sub-top wiggle, but
    realistic pedestals have finite gradient at rhotop and stay monotone.
    '''

    var_orig = copy.deepcopy(var)

    ix = np.argmin(np.abs(p.profiles['rho(-)'] - rhotop))
    offset_array = np.zeros_like( p.profiles['rho(-)'] )

    # Quadratic offset: 0 at rhotop, growing to (val_sep - var_sep) at the separatrix
    n_points = len(p.profiles['rho(-)']) - ix
    t = np.linspace(0, 1, n_points)  # Normalized parameter from 0 to 1
    offset_array[ix:] = (val_sep - var_orig[-1]) * t**2

    var += offset_array

    if plotYN:
        import matplotlib.pyplot as plt
        fig, axs= plt.subplots(nrows=2)
        ax = axs[0]
        ax.plot( p.profiles['rho(-)'], var_orig, 'o-', label='Original' )
        ax.plot( p.profiles['rho(-)'], var, 'o-', label='Modified' )
        ax.legend()
        ax = axs[1]
        ax.plot( p.profiles['rho(-)'], var - var_orig, 'o-', label='Added offset' )
        ax.legend()
        plt.show()

        embed()
        