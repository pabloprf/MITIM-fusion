import shutil
import copy
import torch
import warnings
from scipy.optimize import OptimizeWarning
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.eped_tools import EPEDtools
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from mitim_modules.powertorch.utils import CALCtools
from IPython import embed

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

class eped_beat(beat):

    def __init__(self, maestro_instance, folder_name = None):
        super().__init__(maestro_instance, beat_name = 'eped', folder_name = folder_name)

    def prepare(
            self,
            use_full_EPED = False,
            nn_location = None, 
            norm_location = None,
            transform_inputs_fun = NNtools.standard_arguments_eped,
            neped_20 = None,          # Force this pedestal density (e.g. at creator stage), otherwise from the profiles_current
            BetaN = None,             # Force this BetaN (e.g. at creator stage), otherwise from the profiles_current
            Tesep_keV = None,         # Force this Te at the separatrix, otherwise from the profiles_current
            nesep_20 = None,          # Force this ne at the separatrix, otherwise from the profiles_current
            corrections_set = None,   # Force these inputs to the NN (e.g. exact delta, Rmajor, etc)
            ptop_multiplier = 1.0,    # Multiplier for the ptop, useful for sensitivity studies
            TioverTe = 1.0,           # Ratio of Ti/Te at the top of the pedestal
            eped_params_override = None,
            **kwargs
            ):
        self.use_full_EPED = use_full_EPED
        if corrections_set is None:
            corrections_set = {}

        if not self.use_full_EPED:
            
            # If running with NN, NN location must be specified
            if nn_location is None or norm_location is None: 
                raise ValueError('nn_location and norm_location must be provided if not using full EPED')
            
            else:
                print(f'\t- Choice of EPED: NN from {IOtools.clipstr(nn_location)}', typeMsg='i')

                self.nn = NNtools.eped_nn(type='tf')
                nn_location = IOtools.expandPath(nn_location)
                norm_location = IOtools.expandPath(norm_location)

                self.nn.load(nn_location, norm=norm_location, transform_inputs_fun=transform_inputs_fun)
        else: 
            print('\t- Choice of EPED: full', typeMsg='i')

            if nn_location is None or norm_location is None:
                self.nn = None
                print('\t- No NN and/or norm location provided', typeMsg='i')
            else: 
                print('\t- NN location provided only for scans, full EPED will determine the final result', typeMsg='w')
                self.nn = NNtools.eped_nn(type='tf')
                nn_location = IOtools.expandPath(nn_location)
                norm_location = IOtools.expandPath(norm_location)
                self.nn.load(nn_location, norm=norm_location, transform_inputs_fun=transform_inputs_fun)
            

        # Parameters to run EPED with instead of those from the profiles
        self.neped_20 = neped_20
        self.BetaN = BetaN
        self.Tesep_keV = Tesep_keV
        self.nesep_20 = nesep_20 

        self.corrections_set = corrections_set
        
        self.eped_params_override = eped_params_override

        self.ptop_multiplier = ptop_multiplier
        self.TioverTe = TioverTe

        # Which equilibrium model is used by TOQ 
        if 'toq_eq_choice' in kwargs:
            print('Using the toq_eq_choice provided:', kwargs['toq_eq_choice'])
            if kwargs['toq_eq_choice'] == 'standard':  # ishape = 4 (Bondeson Dee formula)
                self.toq_eq_choice = 'standard'
            elif kwargs['toq_eq_choice'] == 'full_turnbull_miller': # ishape = 7 (Turnbull-Miller full formula, same as 4 but adding squareness)
                self.toq_eq_choice = 'full_turnbull_miller'
                print('Warning: you must have modified EPED and have TOQ set to ishape = 7 to be consistent with this choice', typeMsg='warning')
            elif kwargs['toq_eq_choice'] == 'mxh':  # ishape = 13 (MXH-based formula)
                self.toq_eq_choice = 'mxh'
                print('Warning: you must have modified EPED and have TOQ set to ishape = 13 to be consistent with this choice', typeMsg='warning')
            else:
                raise ValueError(f'Unknown toq_eq_choice: {kwargs["toq_eq_choice"]}')
        else: 
            self.toq_eq_choice = 'standard'
            print('No toq_eq_choice provided, using standard EPED default (ishape = 4)')
        self._inform()

    def run(self, **kwargs):

        # Write here
        shutil.copy2(self.initialize.folder / "input.gacode", self.folder / "input.gacode")

        # -------------------------------------------------------
        # Run EPED
        # -------------------------------------------------------

        eped_results = self._run(loopBetaN = 1, store_scan=True, nproc_per_run=kwargs.get('cpus', 16), cold_start=kwargs.get('cold_start', False))

        # -------------------------------------------------------
        # Save stuff
        # -------------------------------------------------------

        np.save(self.folder / 'eped_results.npy', eped_results)

        self.rhotop = eped_results['rhotop']

    def _run(self, loopBetaN = 1, minimum_relative_change_in_x=0.005, store_scan = False, nproc_per_run=64, cold_start=True):
        '''
            minimum_relative_change_in_x: minimum relative change in x to streach the core, otherwise it will keep the old core
        '''

        # Check to make sure using full EPED if running with squareness
        if self.toq_eq_choice != 'standard' and not self.use_full_EPED:
            print('Warning: NN-based EPED was trained using ishape = 4, ignoring requested toq_eq_choice', typeMsg='warning')
            self.toq_eq_choice = 'standard'

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
            rho_check = 0.95
            self.neped_20 = interpolation_function(rho_check,self.profiles_current.profiles['rho(-)'],self.profiles_current.profiles['ne(10^19/m^3)'])*1E-1
            print(f'\t- neped_20 not provided as part of the trans-beat information nor prepare, using ne at rho = {rho_check} -> {self.neped_20:.2f} 10^20 m^-3')

        neped_20 = self.neped_20


        kappa995 = self.profiles_current.derived['kappa995']
        delta995 = self.profiles_current.derived['delta995']
        if (self.toq_eq_choice == 'full_turnbull_miller' or self.toq_eq_choice == 'mxh'): 
            zeta995 = self.profiles_current.derived['zeta995']
        else:
            zeta995 = None
        if self.toq_eq_choice == 'mxh':
            s_three995 = 0 # TODO: pull from TRANSP (add to PROFILEtools.py)
            s_four995 = 0 # TODO: pull from TRANSP(add to PROFILEtools.py)
        else: 
            s_three995 = None
            s_four995 = None

        BetaN = self.profiles_current.derived['BetaN_engineering']
        Tesep_keV = self.profiles_current.profiles['te(keV)'][-1]
        nesep_20 = self.profiles_current.profiles['ne(10^19/m^3)'][-1]*0.1
        
        # Don't overwrite the shaping coefficients if they already were input to the simulation
        if 'kappa995' in self.__dict__ and self.kappa995 is not None:               kappa995 = self.kappa995
        if 'delta995' in self.__dict__ and self.delta995 is not None:               delta995 = self.delta995 
        if "BetaN" in self.__dict__ and self.BetaN is not None:                     BetaN = self.BetaN
        if "Tesep_keV" in self.__dict__ and self.Tesep_keV is not None:             Tesep_keV = self.Tesep_keV
        if "nesep_20" in self.__dict__ and self.nesep_20 is not None:               nesep_20 = self.nesep_20
        if self.toq_eq_choice != 'standard':
            if "zeta995" in self.__dict__ and self.zeta995 is not None:                zeta995 = self.zeta995
        if self.toq_eq_choice == 'mxh':
            if "s_three995" in self.__dict__ and self.s_three995 is not None:          s_three995 = self.s_three995
            if "s_four995" in self.__dict__ and self.s_four995 is not None:            s_four995 = self.s_four995

        nesep_ratio = nesep_20 / neped_20

        # Store evaluation
        if self.toq_eq_choice == 'standard': 
            self.current_evaluation = {
                'Ip': np.abs(Ip),
                'Bt': np.abs(Bt),
                'R': np.abs(R),
                'a': np.abs(a),
                'kappa995': np.abs(kappa995),
                'delta995': delta995,
                'neped_20': np.abs(neped_20),
                'BetaN': np.abs(BetaN),
                'zeff': np.abs(zeff),
                'Tesep_keV': np.abs(Tesep_keV),
                'nesep_ratio': np.abs(nesep_ratio),
            }
        elif self.toq_eq_choice == 'full_turnbull_miller':
            self.current_evaluation = {
                'Ip': np.abs(Ip),
                'Bt': np.abs(Bt),
                'R': np.abs(R),
                'a': np.abs(a),
                'kappa995': np.abs(kappa995),
                'delta995': np.abs(delta995),
                'neped_20': np.abs(neped_20),
                'BetaN': np.abs(BetaN),
                'zeff': np.abs(zeff),
                'Tesep_keV': np.abs(Tesep_keV),
                'nesep_ratio': np.abs(nesep_ratio), 
                'zeta': np.abs(zeta995)
            }
        elif self.toq_eq_choice == 'mxh':
            self.current_evaluation = {
                'Ip': np.abs(Ip),
                'Bt': np.abs(Bt),
                'R': np.abs(R),
                'a': np.abs(a),
                'kappa995': np.abs(kappa995),
                'delta995': np.abs(delta995),
                'neped_20': np.abs(neped_20),
                'BetaN': np.abs(BetaN),
                'zeff': np.abs(zeff),
                'Tesep_keV': np.abs(Tesep_keV),
                'nesep_ratio': np.abs(nesep_ratio), 
                'zeta': np.abs(zeta995),
                's_three': np.abs(s_three995),
                's_four': np.abs(s_four995)
            }

        # --- Sometimes we may need specific EPED inputs
        for key, value in self.corrections_set.items():
            if key not in ['ptop_kPa', 'wtop_psipol']:
                self.current_evaluation[key] = value
        # ----------------------------------------------

        print('\n\t- Running EPED with:')
        print(f'\t\t- Ip: {self.current_evaluation["Ip"]:.2f} MA')
        print(f'\t\t- Bt: {self.current_evaluation["Bt"]:.2f} T')
        print(f'\t\t- R: {self.current_evaluation["R"]:.2f} m')
        print(f'\t\t- a: {self.current_evaluation["a"]:.2f} m')
        print(f'\t\t- kappa995: {self.current_evaluation["kappa995"]:.3f}')
        print(f'\t\t- delta995: {self.current_evaluation["delta995"]:.3f}')
        print(f'\t\t- neped: {self.current_evaluation["neped_20"]:.2f} 10^20 m^-3')
        print(f'\t\t- zeff: {self.current_evaluation["zeff"]:.2f}')
        print(f'\t\t- tesep: {self.current_evaluation["Tesep_keV"]:.3f} keV')
        print(f'\t\t- nesep_ratio: {self.current_evaluation["nesep_ratio"]:.2f}')
        if (self.toq_eq_choice=='full_turnbull_miller' or self.toq_eq_choice=='mxh'):
            print(f'\t\t- zeta995: {self.current_evaluation["zeta"]:.3f}')
        if self.toq_eq_choice=='mxh':
            print(f'\t\t- s_three995: {self.current_evaluation["s_three"]:.3f}')
            print(f'\t\t- s_four995: {self.current_evaluation["s_four"]:.3f}')

        # -------------------------------------------------------
        # Run NN
        # -------------------------------------------------------

        BetaN = self.current_evaluation["BetaN"]

        BetaNs, ptop_kPas, wtop_psipols  = [], [], []
        for i in range(loopBetaN):
            print(f'\t\t- BetaN: {BetaN:.2f}')

            inputs_to_eped_NN = (
                    self.current_evaluation["Ip"],
                    self.current_evaluation["Bt"],
                    self.current_evaluation["R"],
                    self.current_evaluation["a"],
                    self.current_evaluation["kappa995"],
                    self.current_evaluation["delta995"],
                    self.current_evaluation["neped_20"]*10,
                    BetaN,
                    self.current_evaluation["zeff"],
                    self.current_evaluation["Tesep_keV"]* 1E3,
                    self.current_evaluation["nesep_ratio"]
                    )
            if self.toq_eq_choice == 'standard':
                inputs_to_eped = inputs_to_eped_NN
                
            if self.toq_eq_choice == 'full_turnbull_miller': 
                inputs_to_eped = (
                    self.current_evaluation["Ip"],
                    self.current_evaluation["Bt"],
                    self.current_evaluation["R"],
                    self.current_evaluation["a"],
                    self.current_evaluation["kappa995"],
                    self.current_evaluation["delta995"],
                    self.current_evaluation["neped_20"]*10,
                    BetaN,
                    self.current_evaluation["zeff"],
                    self.current_evaluation["Tesep_keV"]* 1E3,
                    self.current_evaluation["nesep_ratio"],
                    self.current_evaluation["zeta"]
                    )
            elif self.toq_eq_choice == 'mxh': 
                inputs_to_eped = (
                    self.current_evaluation["Ip"],
                    self.current_evaluation["Bt"],
                    self.current_evaluation["R"],
                    self.current_evaluation["a"],
                    self.current_evaluation["kappa995"],
                    self.current_evaluation["delta995"],
                    self.current_evaluation["neped_20"]*10,
                    BetaN,
                    self.current_evaluation["zeff"],
                    self.current_evaluation["Tesep_keV"]* 1E3,
                    self.current_evaluation["nesep_ratio"], 
                    self.current_evaluation["zeta"],
                    self.current_evaluation["s_three"],
                    self.current_evaluation["s_four"]
                    )

            # -------------------------------------------------------
            # Give the option to override the ptop_kPa and wtop_psipol
            if 'ptop_kPa' in self.corrections_set:
                print(f'\t\t- Overriding ptop_kPa: {self.corrections_set["ptop_kPa"]:.2f} kPa', typeMsg='w')
                ptop_kPa = self.corrections_set["ptop_kPa"]
            else:
                ptop_kPa = None
                
            if 'wtop_psipol' in self.corrections_set:
                print(f'\t\t- Overriding wtop_psipol: {self.corrections_set["wtop_psipol"]:.5f}', typeMsg='w')
                wtop_psipol = self.corrections_set["wtop_psipol"]
            else:
                wtop_psipol = None
            # -------------------------------------------------------
            
            if ptop_kPa is None or wtop_psipol is None:
                
                if not self.use_full_EPED:
                    ptop_kPa, wtop_psipol = self.nn(*inputs_to_eped)
                else:
                    ptop_kPa, wtop_psipol = self._run_full_eped(self.folder,*inputs_to_eped, eped_params_override=self.eped_params_override, nproc_per_run=nproc_per_run, cold_start=cold_start)
                    
                    if store_scan and self.nn==None:
                        store_scan = False
                        print('\t- Warning: store_scan requires a NN. Since it is unable, disabling store_scan', typeMsg='w')
            
            print('\t- Raw EPED results:')
            print(f'\t\t- ptop_kPa: {ptop_kPa:.4f}')
            print(f'\t\t- wtop_psipol: {wtop_psipol:.4f}')

            if self.ptop_multiplier != 1.0:
                print(f'\t\t- Multiplying ptop by {self.ptop_multiplier}', typeMsg='i')
                ptop_kPa *= self.ptop_multiplier

            BetaNs.append(BetaN)
            ptop_kPas.append(ptop_kPa)
            wtop_psipols.append(wtop_psipol)

            # -------------------------------------------------------
            # Produce relevant quantities
            # -------------------------------------------------------

            rhotop, netop_20, Tetop_keV, Titop_keV, rhoped = eped_postprocessing(neped_20, nesep_20, ptop_kPa, self.TioverTe, wtop_psipol, self.profiles_current)

            print('\t- Post-processed quantities:')
            print(f'\t\t- rhotop: {rhotop:.3f}')
            print(f'\t\t- netop_20: {netop_20:.3f}')
            print(f'\t\t- Tetop_keV: {Tetop_keV:.3f}')
            print(f'\t\t- Titop_keV: {Titop_keV:.3f}')
            print(f'\t\t- rhoped: {rhoped:.3f}')

            # -------------------------------------------------------
            # Put into profiles #TODO: This should be looped with the NN evaluation to find the self-consisent betaN with the current profiles
            # -------------------------------------------------------

            print('\t- Applying EPED results to profiles:')

            if 'rhotop' in self.__dict__:
                print(f'\t\t- Using previously-stored rhotop: {self.rhotop:.3f}')
                xp_old = self.rhotop
            else:
                print('\t\t- Using rhotop = 0.9 as an approximation for the stretching')
                xp_old = 0.9

            profiles_pass = copy.deepcopy(self.profiles_current)
            self.profiles_output = eped_profiler(profiles_pass, xp_old, rhotop, Tetop_keV, Titop_keV, netop_20, minimum_relative_change_in_x=minimum_relative_change_in_x)

            BetaN = self.profiles_output.derived['BetaN_engineering']

        if loopBetaN > 1:
            print('\t- Looping over BetaN:')
            print(f'\t\t* BetaN: {BetaNs}')
            print(f'\t\t* ptop_kPa: {ptop_kPas}')
            print(f'\t\t* wtop_psipol: {wtop_psipols}')

        # ---------------------------------
        # Run scans for postprocessing
        # ---------------------------------

        scan_results = None
        if store_scan:

            print('\t- Running scans of EPED inputs for postprocessing')
                
            scan_relative = {
                "Ip": 0.05,
                "Bt": 0.05,
                "R": 0.05,
                "a": 0.05,
                "kappa995": 0.05,
                "delta995": 0.05,
                "neped_20": 0.75,
                "BetaN": 0.5,
                "zeff": 0.3,
                "Tesep_keV": 0.75,
                "nesep_ratio": 0.75
            }

            scan_results = {}
            for k,key in enumerate(scan_relative):
                inputs_scan = list(copy.deepcopy(inputs_to_eped_NN))
                scan_results[key] = {'ptop_kPa': [], 'wtop_psipol': [], 'value': []}
                for m in np.linspace(1-scan_relative[key],1+scan_relative[key],15):
                    inputs_scan[k] = inputs_to_eped_NN[k]*m
                    ptop_kPa0, wtop_psipol0 = self.nn(*inputs_scan)
                    scan_results[key]['ptop_kPa'].append(ptop_kPa0)
                    scan_results[key]['wtop_psipol'].append(wtop_psipol0)
                    scan_results[key]['value'].append(inputs_scan[k])
                scan_results[key]['ptop_kPa'] = np.array(scan_results[key]['ptop_kPa'])
                scan_results[key]['wtop_psipol'] = np.array(scan_results[key]['wtop_psipol'])
                scan_results[key]['value'] = np.array(scan_results[key]['value'])

                self.nn.force_within_range = None # Do not throw warnings during the scan
                scan_results[key]['ptop_kPa_nominal'], scan_results[key]['wtop_psipol_nominal'] = self.nn(*inputs_to_eped_NN)

        # ---------------------------------
        # Store
        # ---------------------------------

        eped_results = {
            'ptop_kPa': ptop_kPa,
            'wtop_psipol': wtop_psipol,
            'Tetop_keV': Tetop_keV,
            'netop_20': netop_20,
            'neped_20': neped_20,
            'nesep_20': nesep_20,
            'rhotop': rhotop,
            'Tesep_keV': Tesep_keV,
            'inputs_to_eped': inputs_to_eped,
            'scan_results': scan_results
        }

        for key in eped_results:
            print(f'\t\t- {key}: {eped_results[key]}')

        self.profiles_output.write_state(file=self.folder / 'input.gacode.eped')

        return eped_results

    def _run_full_eped(self, folder, Ip, Bt, R, a, kappa995, delta995, neped19, BetaN, zeff, Tesep_eV, nesep_ratio, *args, eped_params_override=None, nproc_per_run=64, cold_start=True):
        '''
            Run the full EPED code with the given inputs.
            Returns ptop_kPa and wtop_psipol.
            If further shaping parameters are provided as extra arguments, use them; otherwise set them to zero.
            Note this is not the parallel to the _run method, the run method uses this where it would otherwise call the EPED NN
        '''

        # Handle further shaping parameters
        if len(args) > 0:
            zeta = args[0]
            print('Len of args > 0, using zeta =', zeta)
            if len(args) > 1: 
                s_three = args[1]
                print('Len of args > 1. Also using s_three =', s_three)
                if len(args) > 2:
                    s_four = args[2]
                    print('Len of args > 2. Also using s_four =', s_four)
                else: 
                    s_four = 0.0
                    print('Len of args <= 2. Setting s_four = 0.0')
            else: 
                s_three = 0.0
                print('Len of args <= 1. Setting s_three = 0.0')
        else:
            zeta = 0.0
            print('No zeta provided, setting zeta = 0.0')

        eped = EPEDtools.EPED(folder=folder)

        if self.toq_eq_choice == 'standard': 
            input_params = {
                'ip': Ip,
                'bt': Bt,
                'r': R,
                'a': a,
                'kappa': kappa995,
                'delta': delta995,
                'neped': neped19,
                'betan': BetaN,
                'zeffped': zeff,
                'nesep': nesep_ratio * neped19,
                'tesep': Tesep_eV
            }
            print('_run_full_eped with standard TOQ configuration. Parameters used:', input_params)
        elif self.toq_eq_choice == 'full_turnbull_miller':
            input_params = {
                'ip': Ip,
                'bt': Bt,
                'r': R,
                'a': a,
                'kappa': kappa995,
                'delta': delta995,
                'neped': neped19,
                'betan': BetaN,
                'zeffped': zeff,
                'nesep': nesep_ratio * neped19,
                'tesep': Tesep_eV,
                'zeta': zeta
            }
            print('_run_full_eped with full_turnbull_miller TOQ configuration. Parameters used:', input_params)
        elif self.toq_eq_choice == 'mxh': 
            input_params = {
                'ip': Ip,
                'bt': Bt,
                'r': R,
                'a': a,
                'kappa': kappa995,
                'delta': delta995,
                'neped': neped19,
                'betan': BetaN,
                'zeffped': zeff,
                'nesep': nesep_ratio * neped19,
                'tesep': Tesep_eV, 
                'zeta': zeta,
                's_three': s_three,
                's_four': s_four
            }
            print('_run_full_eped with mxh TOQ configuration. Parameters used:', input_params)

        eped.run(
            subfolder = 'case1',
            input_params = input_params,
            nproc_per_run = nproc_per_run,
            cold_start = cold_start,
            eped_params_override = eped_params_override,
        )

        eped.read(subfolder='case1')

        ptop_kPa = float(eped.results['case1']['run1']['ptop'])
        wtop_psipol = float(eped.results['case1']['run1']['wptop'])

        return ptop_kPa, wtop_psipol
        
    def finalize(self, **kwargs):
        
        # Remove output folders
        for item in self.folder_output.glob('*'):
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                IOtools.shutil_rmtree(item)

        # Copy eped run to outputs
        if self.use_full_EPED:
            shutil.copy2(self.folder / 'case1' / 'output_run1.nc', self.folder_output / 'output_run1.nc')
        
        # Copy results to output folder
        shutil.copy2(self.folder / 'eped_results.npy', self.folder_output / 'eped_results.npy')
        
        # Write profiles to output folder
        self.profiles_output = PROFILEStools.gacode_state(self.folder / 'input.gacode.eped')
        self.profiles_output.write_state(file=self.folder_output / 'input.gacode')

    def merge_parameters(self):
        # EPED beat does not modify the profiles grid or anything, so I can keep it fine
        pass

    def grab_output(self):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:

            loaded_results =  np.load(self.folder_output / 'eped_results.npy', allow_pickle=True).item()

            profiles = PROFILEStools.gacode_state(self.folder_output / 'input.gacode') if isitfinished else None
            
        else:

            loaded_results = None
            profiles = None

        return loaded_results, profiles

    def plot(self,  fn = None, counter = 0, full_plot = True):

        if fn is None:
            fn = GUItools.FigureNotebook("EPED")

        fig = fn.add_figure(label='EPED', tab_color=counter)
        axs = fig.subplot_mosaic(
            """
            ABCDHJ
            AEFGIK
            """
        )
        axs = [ ax for ax in axs.values() ]

        loaded_results, profiles = self.grab_output()

        profiles_current = PROFILEStools.gacode_state(self.folder / 'input.gacode')

        profiles_current.plotRelevant(axs = axs, color = 'b', label = 'orig')
        
        if loaded_results is not None:
            profiles.plotRelevant(axs = axs, color = 'r', label = 'EPED')

            axs[1].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            try:
                axs[1].axhline(loaded_results['Tetop_keV'], color='k', ls='--',lw=2)
            except:
                axs[1].axhline(loaded_results['Ttop_keV'], color='k', ls='--',lw=2)
                

            axs[2].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[2].axhline(loaded_results['netop_20'], color='k', ls='--',lw=2)

            axs[3].axvline(loaded_results['rhotop'], color='k', ls='--',lw=2)
            axs[3].axhline(loaded_results['ptop_kPa']*1E-3, color='k', ls='--',lw=2)

        GRAPHICStools.adjust_figure_layout(fig)

        # Full EPED?
        if (self.folder_output / 'output_run1.nc').exists():
            eped = EPEDtools.EPED(folder=self.folder_output)
            eped.read(subfolder='.', label='full_eped')
            eped.plot(fn=fn, labels=['full_eped'], tab_color=counter)

        # Scan results?
        if 'scan_results' in loaded_results and loaded_results['scan_results'] is not None:
            for ikey in ['ptop_kPa', 'wtop_psipol']:
                fig = fn.add_figure(label=f'EPED Scan ({ikey})', tab_color=counter)

                axs = fig.subplot_mosaic(
                    """
                    ABCD
                    EFGH
                    IJKL
                    """,
                )
                axs = [ ax for ax in axs.values() ]

                try:
                    self._plot_scan(ikey, loaded_results=loaded_results, axs=axs)
                except:
                    print(f'\t- Could not plot scan for {ikey}', typeMsg='w')

                GRAPHICStools.adjust_figure_layout(fig)

        msg = '\t\t- Plotting of EPED beat done'

        return msg
            
    def _plot_scan(self, ikey, loaded_results = None, axs = None):

        if loaded_results is None:
            loaded_results, _ = self.grab_output()

        if axs is None:
            fig = plt.figure()

            axs = fig.subplot_mosaic(
                """
                ABCD
                EFGH
                IJKL
                """,
            )
            axs = [ ax for ax in axs.values() ]

        max_val = 0
        min_val = 1E12
        for i,key in enumerate(loaded_results['scan_results']):

            axs[i].plot(loaded_results['scan_results'][key]['value'], loaded_results['scan_results'][key][ikey], 's-', color='b', markersize=3, label=f'NN scan')
            axs[i].plot([loaded_results['inputs_to_eped'][i]], [loaded_results['scan_results'][key][f'{ikey}_nominal']], 'o', color='m', label='NN nominal')
            axs[i].axhline(loaded_results['scan_results'][key][f'{ikey}_nominal'], color='m', ls='-.')

            axs[i].plot([loaded_results['inputs_to_eped'][i]], [loaded_results[ikey]], '^', color='r', label='EPED beat', markersize=8)

            axs[i].axvline(loaded_results['inputs_to_eped'][i], color='k', ls='--', label='Nominal input')
    
            max_val = np.max([max_val,np.max(loaded_results['scan_results'][key][ikey])])
            min_val = np.min([min_val,np.min(loaded_results['scan_results'][key][ikey])])
        
        for i,key in enumerate(loaded_results['scan_results']):
            axs[i].set_ylim([min_val*0.9,1.1*max_val])
            axs[i].set_xlabel(key)
            axs[i].set_ylabel(ikey)
            GRAPHICStools.addDenseAxis(axs[i])
        axs[0].legend(prop={'size': 6},loc='best')

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

        # From a geqdsk initialization
        if 'zeta995' in self.maestro_instance.parameters_trans_beat:
            self.zeta995 = self.maestro_instance.parameters_trans_beat['zeta995']
            print(f"\t\t- Using previous zeta995: {self.zeta995}")

        # From a previous EPED beat, grab the rhotop
        if 'rhotop' in self.maestro_instance.parameters_trans_beat:
            self.rhotop = self.maestro_instance.parameters_trans_beat['rhotop']
            print(f"\t\t- Using previous rhotop: {self.rhotop}")

            
    def _inform_save(self, eped_output = None):

        if eped_output is None:
            eped_output, _ = self.grab_output()

        self.maestro_instance.parameters_trans_beat['neped_20'] = eped_output['neped_20']

        self.maestro_instance.parameters_trans_beat['rhotop'] = eped_output['rhotop']

        print('\t\t- neped_20 and rhotop saved for future beats')

def scale_profile_by_stretching( x, y, xp, yp, xp_old, plotYN=False, label='', keep_aLx=True, roa = None, print_msgs = True):
    '''
    This code keeps the separatrix fixed, moves the top of the pedestal, fits pedestal and stretches the core
        xp: top of the pedestal
        roa: needed if I want to keep the aLT profile in the core-predicted region
    '''

    if print_msgs: print(f'\t\t- Scaling profile {label} by stretching')

    # Find old core
    ibc = np.argmin(np.abs(x-xp_old))
    xcore_old = x[:ibc+1]
    ycore_old = y[:ibc+1]

    if print_msgs: print(f'\t\t\t* Stretching core: [{xp_old:.3f}, {ycore_old[-1]:.3f}] -> [{xp:.3f}, {yp:.3f}]')

    # Fit new pedestal
    _, yped = FunctionalForms.pedestal_tanh(yp, y[-1], 1-xp, x=x)

    # Find extension of new core
    ibc = np.argmin(np.abs(x-xp))
    xcore = x[:ibc+1]

    # Scale core
    ycore_new = ycore_old * yped[ibc] / ycore_old[-1]

    # Stretch old core into the new extension
    if xp != xp_old:
        x_core_old_mod = xcore_old * xcore[-1] / xcore_old[-1]
        ycore_new = interpolation_function(xcore,x_core_old_mod,ycore_new)

    # Merge
    ynew = copy.deepcopy(y)
    ynew[:ibc+1] = ycore_new
    ynew[ibc+1:] = yped[ibc+1:]

    # Keep old aLT
    if keep_aLx:
        if print_msgs: print('\t\t\t* Keeping old aLT profile in the core-predicted region, using r/a for it')

        # Calculate gradient in entire region
        aLy = CALCtools.derivation_into_Lx( torch.from_numpy(roa), torch.from_numpy(y) )

        # I'm only interested in core region, plus one ghost point with the same gradient
        aLy = torch.cat( (aLy[:ibc+1], aLy[ibc].unsqueeze(0)) )

        y_mod = CALCtools.integration_Lx( torch.from_numpy(roa[:ibc+2]).unsqueeze(0), aLy.unsqueeze(0), torch.from_numpy(np.array(ynew[ibc+1])).unsqueeze(0) ).squeeze().numpy()
        ynew[:ibc+2] = y_mod


    if plotYN:
        fig, axs = plt.subplots(nrows=2, figsize=(6,10))
        ax = axs[0]
        ax.plot(x,y,'-o',color='b', label='old')
        ax.axvline(x=xp_old,color='b',ls='--')
        ax.plot(x,ynew,'-o',color='r',label='new')
        ax.axvline(x=xp,color='r',ls='--')
        ax.axhline(y=yp,color='r',ls='--')
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_xlim([0,1]); ax.set_ylim(bottom=0)
        ax.legend()

        ax = axs[1]
        aLy = CALCtools.derivation_into_Lx( torch.from_numpy(roa), torch.from_numpy(y) )
        ax.plot(x,aLy,'-o',color='b', label='old')
        ax.axvline(x=xp_old,color='b',ls='--')

        aLy = CALCtools.derivation_into_Lx( torch.from_numpy(roa), torch.from_numpy(ynew) )
        ax.plot(x,aLy,'-o',color='r', label='new')
        ax.axvline(x=xp,color='r',ls='--')

        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel('r/a'); ax.set_ylabel('aLx')
        ax.set_xlim([0,1]); ax.set_ylim(bottom=0)
        ax.legend()

    
        plt.show()
        embed()

    return ynew

# --------------------------------------------------------------------------------------------
# Additional EPED utilities
# --------------------------------------------------------------------------------------------

def eped_postprocessing(neped_20, nesep_20, ptop_kPa, TioverTe, wtop_psipol,profiles):

    # psi_pol to rhoN
    rhotop = interpolation_function(1-wtop_psipol,profiles.derived['psi_pol_n'],profiles.profiles['rho(-)'])
    rhoped = interpolation_function(1-2*wtop_psipol/3,profiles.derived['psi_pol_n'],profiles.profiles['rho(-)'])

    # Find ne at the top
    # basically, we are finding Ytop such that the functional form goes through the Yped and Ysep
    # this technically doesn't need to be done after the first time EPED is run, but I'm doing it now for completeness
    pedestal_profile = lambda x, Y: FunctionalForms.pedestal_tanh(Y, nesep_20, 1-rhotop, x=x)[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning) # Avoid annoying warning from curve_fit
        n0, _ = curve_fit(pedestal_profile, [rhoped], [neped_20])
        
    netop_20 = n0[0]

    # Find factor to account that it's not a pure plasma
    n = profiles.derived['ni_All']/profiles.profiles['ne(10^19/m^3)']
    fi = interpolation_function(rhotop, profiles.profiles['rho(-)'], n )

    e_J = 1.60218e-19
   
    if TioverTe != 1:
        print(f'\t\t\t* Scaling profiles Ti/Te={TioverTe}')
    
    # Calculate from P = (neTe + niTi), taking into account the lower ion density
    Tetop_keV = ptop_kPa * 1E3 / ( ( 1 + fi * TioverTe ) * netop_20 * 1e20) / e_J * 1E-3
    Titop_keV = Tetop_keV * TioverTe
    print(f'\t\t\t* Tetop_keV: {Tetop_keV:.3f}  Titop_keV: {Titop_keV:.3f}')
    print(f'\t\t\t* width (psipol): {wtop_psipol:.5f}  rhotop: {rhotop:.3f}  rhoped: {rhoped:.3f}')

    return rhotop, netop_20, Tetop_keV, Titop_keV, rhoped

def eped_profiler(profiles, xp_old, rhotop, Tetop_keV, Titop_keV, netop_20, minimum_relative_change_in_x=0.005,print_msgs=True):

    profiles_output = profiles
    
    # Determine coodinates
    x = profiles.profiles['rho(-)']
    xroa = profiles.derived['roa']

    if abs(rhotop-xp_old)/xp_old < minimum_relative_change_in_x:
        print(f'\t\t\t* Keeping old core position ({xp_old}) because width variation is {abs(rhotop-xp_old)/xp_old*100:.1f}% < {minimum_relative_change_in_x*100:.1f}% ({xp_old:.3f} -> {rhotop:.3f})')
        rhotop = xp_old

    # Grab quantities useful for scaling
    TiTimain_orig = profiles_output.profiles['ti(keV)'] / profiles_output.profiles['ti(keV)'][:, [0]]
    nine_orig = profiles_output.profiles['ni(10^19/m^3)'] / profiles.profiles['ne(10^19/m^3)'][:, None]

    # -------------------------------------------------------------
    # Modify profiles
    # -------------------------------------------------------------

    # Modify Te
    profiles_output.profiles['te(keV)'] = scale_profile_by_stretching(x,profiles_output.profiles['te(keV)'],rhotop,Tetop_keV,xp_old, label = 'Te', roa = xroa, print_msgs = print_msgs)

    # Modify Ti
    profiles_output.profiles['ti(keV)'][:,0] = scale_profile_by_stretching(x,profiles_output.profiles['ti(keV)'][:,0],rhotop,Titop_keV,xp_old, label = 'Ti', roa = xroa, print_msgs = print_msgs)
    
    # Modify ne
    profiles_output.profiles['ne(10^19/m^3)'] = scale_profile_by_stretching(x,profiles_output.profiles['ne(10^19/m^3)'],rhotop,netop_20*1E1,xp_old, label = 'ne', roa = xroa, print_msgs = print_msgs)
    
    # -------------------------------------------------------------
    # Postprocessing
    # -------------------------------------------------------------
    
    # Kepp the same ion concentration as before at the top
    for i in range(profiles_output.profiles['ni(10^19/m^3)'].shape[-1]):
        profiles_output.profiles['ni(10^19/m^3)'][:,i] = profiles_output.profiles['ne(10^19/m^3)'] * nine_orig[:,i]
    
    # Make all thermal ions have the same temperature profile
    profiles_output.makeAllThermalIonsHaveSameTemp()
    
    # Keep the same fast ion temperature profile, relative to the thermal ions:
    for sp in range(len(profiles_output.Species)):
        if profiles_output.Species[sp]["S"] == "fast":
            profiles_output.profiles["ti(keV)"][:, sp] = profiles_output.profiles["ti(keV)"][:, 0] * TiTimain_orig[:, sp]

    # -------------------------------------------------------------
    # Re-derive
    # -------------------------------------------------------------

    profiles_output.derive_quantities(rederiveGeometry=False)

    return profiles_output

def preprocess_run_eped(run_namelist, maestro_namelist, cpus, cold_start):
    
    cpus_eped = maestro_namelist["maestro"]["eped"]["preprocess_prepare_parameters"]["cpus"]

    run_namelist['cold_start'] = cold_start
    run_namelist['cpus'] = cpus_eped if cpus_eped is not None else cpus
    
    return run_namelist